"""Bounded candidate discovery and request-local code corpus assembly.

This module deliberately owns only filesystem-side candidate work.  It has no
ContextFS, ReadAPI, vector provider, lifecycle, Outbox, or service cache
dependency.  The caller keeps every returned object in the active request
stack and releases it when the search returns.
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import selectors
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.models import ContextNode
from core.resource_identity import canonical_node_uri
from extraction.code_processor import MAX_CODE_FILE_SIZE
from extraction.code_workspace import (
    BoundedCodeWorkspaceWalkStats,
    ValidatedCodeWorkspaceFile,
    iter_bounded_code_files,
    validate_code_candidate_path,
)
from index.index_record_builder import build_index_records
from retrieval.bm25_index import Bm25Document, Bm25Index, build_code_bm25_document
from retrieval.code_ordering import code_source_key
from service.code_resource_lifecycle import build_code_resource_nodes
from service.code_workspace_sync import ResolvedCodeWorkspace

_CANDIDATE_WALK_ENTRY_LIMIT = 100_000
_RG_OUTPUT_BYTES_LIMIT = 256 * 1024
_MAX_RG_TERMS = 16
_MAX_LITERAL_OCCURRENCES_PER_FILE = 64


@dataclass(frozen=True)
class CandidateDiscoveryBudget:
    """Frozen per-request filesystem limits for candidate-local retrieval."""

    pool_limit: int
    file_limit: int
    read_bytes: int
    deadline: float
    entry_limit: int = _CANDIDATE_WALK_ENTRY_LIMIT


@dataclass
class CandidateReadMeter:
    """Shared byte accounting for every discovery and canonical source read.

    A candidate searched by ripgrep/fallback and then canonicalized is read
    twice, so both operations must consume the request's global read budget.
    Ripgrep reservations are conservative (the child may stop before EOF),
    but they guarantee that an external scan cannot make the later canonical
    read silently exceed the cap.  A missing ripgrep executable releases only
    its never-started discovery reservations before the fallback runs.
    """

    cap: int
    used: int = 0
    exhausted: bool = False
    _discovery_reservations: dict[str, int] = field(default_factory=dict)

    def reserve(self, amount: int) -> bool:
        if amount < 0 or self.used + amount > self.cap:
            self.exhausted = True
            return False
        self.used += amount
        return True

    def release(self, amount: int) -> None:
        """Undo a conservative reservation when no external read was started."""

        if amount > 0:
            self.used = max(0, self.used - amount)
        if self.used < self.cap:
            self.exhausted = False

    def reserve_discovery_path(self, relative_path: str, amount: int) -> bool:
        """Reserve a conservative byte charge for one possible ripgrep read.

        Revalidation can observe a larger replacement file after the initial
        walker admission.  Only the additional amount is charged here; a
        smaller replacement retains the conservative earlier reservation.
        """

        existing = self._discovery_reservations.get(relative_path, 0)
        if amount <= existing:
            return True
        if not self.reserve(amount - existing):
            return False
        self._discovery_reservations[relative_path] = amount
        return True

    def reserve_path(self, relative_path: str, amount: int) -> bool:
        """Charge an actual fallback or canonical-file read.

        ``relative_path`` is accepted to keep the call sites explicit about
        which protected file is being charged.  Unlike a ripgrep reservation,
        a canonical build is a second read of the same file and must never be
        coalesced with a prior discovery charge.
        """

        del relative_path
        return self.reserve(amount)

    def release_discovery_paths(self, relative_paths: Sequence[str]) -> None:
        """Release only ripgrep reservations when its child never started."""

        released = 0
        for relative_path in relative_paths:
            released += self._discovery_reservations.pop(relative_path, 0)
        self.release(released)


@dataclass(frozen=True)
class CandidateFile:
    """Path-free candidate ranking projection."""

    relative_path: str
    path_score: int
    basename_score: int
    content_occurrences: int
    discovery_sources: tuple[str, ...]
    explicit_path_match: bool = False
    glob_anchor_match: bool = False


@dataclass
class CandidateDiscoveryOutcome:
    """Internal discovery result; paths are protected implementation state."""

    files: tuple[CandidateFile, ...]
    paths_by_relative: Mapping[str, Path]
    literal_matches: Mapping[str, tuple[int, ...]]
    read_meter: CandidateReadMeter
    diagnostics: Mapping[str, Any]
    partial_failure: bool = False


@dataclass
class CandidateCodeCorpus:
    """Request-local canonical snippets and a local field-aware BM25 index."""

    records: tuple[dict[str, Any], ...]
    nodes_by_uri: Mapping[str, ContextNode]
    bm25_index: Bm25Index
    file_ranks: Mapping[str, int]
    literal_uris: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    partial_failure: bool = False


@dataclass
class _CandidateSignals:
    path: Path
    relative_path: str
    explicit_path_match: bool = False
    glob_anchor_match: bool = False
    basename_score: int = 0
    path_score: int = 0
    content_occurrences: int = 0
    all_grep_terms_match: bool = False
    sources: set[str] = field(default_factory=set)


def candidate_discovery_budget(config: object, *, deadline: float) -> CandidateDiscoveryBudget:
    """Project the config-owned candidate limits into one request budget."""

    return CandidateDiscoveryBudget(
        pool_limit=int(getattr(config, "code_retrieval_candidate_discovery_pool_limit", 300)),
        file_limit=int(getattr(config, "code_retrieval_candidate_file_limit", 80)),
        read_bytes=int(getattr(config, "code_retrieval_candidate_read_bytes", 64 * 1024 * 1024)),
        deadline=deadline,
    )


def discover_candidate_files(
    *,
    workspace: ResolvedCodeWorkspace,
    scope: object,
    query_terms: Sequence[str],
    budget: CandidateDiscoveryBudget,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> CandidateDiscoveryOutcome:
    """Discover, pre-rank, and select bounded candidate files.

    The optional ripgrep accelerator is never trusted as an authorization
    source: each returned filename passes the same containment and language
    validation as the deterministic in-process walker.
    """

    meter = CandidateReadMeter(cap=budget.read_bytes)
    walk_stats = BoundedCodeWorkspaceWalkStats()
    terms = _literal_terms(query_terms, getattr(scope, "grep_terms", ()))
    grep_terms = tuple(str(term).lower() for term in getattr(scope, "grep_terms", ()))
    path_values = tuple(getattr(scope, "paths", ()) or ())
    glob_values = tuple(getattr(scope, "glob_patterns", ()) or ())
    signals: dict[str, _CandidateSignals] = {}
    walk_files: dict[str, Path] = {}
    content_universe: dict[str, Path] = {}

    def add_entry(entry: ValidatedCodeWorkspaceFile, *, source: str) -> _CandidateSignals:
        value = signals.get(entry.relative_path)
        if value is None:
            value = _CandidateSignals(path=entry.path, relative_path=entry.relative_path)
            signals[entry.relative_path] = value
        value.sources.add(source)
        return value

    def admit_content(entry: ValidatedCodeWorkspaceFile) -> bool:
        """Admit a bounded literal-search file without retaining its text."""

        if not terms:
            return True
        try:
            size = entry.path.stat().st_size
        except OSError:
            walk_stats.validation_failures += 1
            return True
        if size < 0 or size > MAX_CODE_FILE_SIZE:
            return True
        if not meter.reserve_discovery_path(entry.relative_path, int(size)):
            return False
        content_universe.setdefault(entry.relative_path, entry.path)
        return True

    # Explicit paths are seeded before the generic walker cap.  Otherwise a
    # lexically later requested file could be starved by an unrelated first
    # 300 files in a large repository.
    for requested_path in path_values:
        if monotonic_clock() >= budget.deadline or len(walk_files) >= budget.pool_limit:
            break
        try:
            direct = validate_code_candidate_path(
                workspace.physical_root,
                requested_path,
                language_hints=workspace.language_hints,
            )
        except (OSError, ValueError):
            direct = None
        if direct is not None:
            if not _matches_path_scope(direct.relative_path, path_values, glob_values):
                continue
            walk_files.setdefault(direct.relative_path, direct.path)
            item = add_entry(direct, source="scope_path")
            _apply_path_signals(item, path_values, glob_values, terms)
            continue
        # ``scope.paths`` also supports a logical directory prefix.  Start
        # directly inside that contained directory before generic discovery.
        try:
            scoped_root = (workspace.physical_root / requested_path).resolve(strict=True)
            scoped_root.relative_to(workspace.physical_root)
        except (OSError, ValueError):
            continue
        if not scoped_root.is_dir():
            continue
        try:
            for entry in iter_bounded_code_files(
                workspace.physical_root,
                language_hints=workspace.language_hints,
                start_relative_path=requested_path,
                entry_limit=budget.entry_limit,
                deadline=budget.deadline,
                stats=walk_stats,
                monotonic_clock=monotonic_clock,
            ):
                if not _matches_path_scope(entry.relative_path, path_values, glob_values):
                    continue
                walk_files.setdefault(entry.relative_path, entry.path)
                item = add_entry(entry, source="scope_path")
                _apply_path_signals(item, path_values, glob_values, terms)
                if len(walk_files) >= budget.pool_limit:
                    break
        except (OSError, ValueError):
            walk_stats.validation_failures += 1

    # Glob scopes are authorization predicates, not merely a weak ranking
    # signal.  Seed their matching candidates before the generic walk reaches
    # the pool cap, otherwise a lexically early but unrelated 300-file prefix
    # could make a valid late glob invisible.  The scan retains the same
    # shared entry/deadline budget as every other filesystem path.
    for start_relative_path, patterns in _glob_scan_groups(glob_values):
        if monotonic_clock() >= budget.deadline or len(walk_files) >= budget.pool_limit:
            break
        try:
            iterator = iter_bounded_code_files(
                workspace.physical_root,
                language_hints=workspace.language_hints,
                start_relative_path=start_relative_path,
                entry_limit=budget.entry_limit,
                deadline=budget.deadline,
                stats=walk_stats,
                monotonic_clock=monotonic_clock,
            )
            for entry in iterator:
                if monotonic_clock() >= budget.deadline:
                    walk_stats.deadline_reached = True
                    break
                if not any(fnmatch.fnmatchcase(entry.relative_path, pattern) for pattern in patterns):
                    continue
                if not _matches_path_scope(entry.relative_path, path_values, glob_values):
                    continue
                walk_files.setdefault(entry.relative_path, entry.path)
                item = add_entry(entry, source="scope_glob")
                _apply_path_signals(item, path_values, glob_values, terms)
                if len(walk_files) >= budget.pool_limit:
                    break
        except (OSError, ValueError):
            walk_stats.validation_failures += 1

    # Seed pre-scoped candidates into the content universe first.  They are
    # already path-validated, but literal matching must still use the same
    # bounded admission meter as later generic candidates.
    if terms:
        for relative_path, path in sorted(walk_files.items()):
            if monotonic_clock() >= budget.deadline:
                walk_stats.deadline_reached = True
                break
            if not admit_content(ValidatedCodeWorkspaceFile(relative_path, path)):
                break

    # The generic safe walk completes the one merged/deduplicated candidate
    # pool.  Literal probing is deliberately limited to that same pool: the
    # configured limit bounds every retained candidate path before pre-ranking,
    # not merely the final selected slice.  Content signals can improve the
    # ranking of an admitted path but cannot turn an unbounded secondary
    # universe into hidden candidates beyond the configured pool.
    try:
        if len(walk_files) < budget.pool_limit:
            iterator = iter_bounded_code_files(
                workspace.physical_root,
                language_hints=workspace.language_hints,
                entry_limit=budget.entry_limit,
                deadline=budget.deadline,
                stats=walk_stats,
                monotonic_clock=monotonic_clock,
            )
            for entry in iterator:
                if monotonic_clock() >= budget.deadline:
                    walk_stats.deadline_reached = True
                    break
                if len(walk_files) >= budget.pool_limit:
                    break
                if not _matches_path_scope(entry.relative_path, path_values, glob_values):
                    continue
                if terms and not admit_content(entry):
                    break
                walk_files.setdefault(entry.relative_path, entry.path)
                item = add_entry(entry, source="walker")
                _apply_path_signals(item, path_values, glob_values, terms)
    except (OSError, ValueError):
        # The root itself was strict-validated immediately before discovery.
        # A later filesystem race is a safe bounded partial outcome.
        walk_stats.validation_failures += 1

    if grep_terms:
        # Scan once for file-level AND, reserving budget for canonical reads.
        meter.release_discovery_paths(tuple(content_universe))
        rg_state, rg_matches = "rg_skipped_file_scope", {}
    else:
        rg_state, rg_matches = _ripgrep_literal_candidates(
            workspace=workspace,
            terms=terms,
            universe=content_universe if terms else walk_files,
            meter=meter,
            deadline=budget.deadline,
            monotonic_clock=monotonic_clock,
        )
    literal_matches: dict[str, tuple[int, ...]] = {}
    for relative_path, line_hits in rg_matches.items():
        matched_path = content_universe.get(relative_path) or walk_files.get(relative_path)
        if matched_path is None:
            continue
        entry = ValidatedCodeWorkspaceFile(relative_path, matched_path)
        item = add_entry(entry, source="ripgrep")
        literal_matches[relative_path] = _merge_line_hits(
            literal_matches.get(relative_path, ()),
            line_hits,
        )
        item.content_occurrences = min(
            _MAX_LITERAL_OCCURRENCES_PER_FILE,
            item.content_occurrences + len(line_hits),
        )
        _apply_path_signals(item, path_values, glob_values, terms)

    fallback_used = rg_state in {"rg_unavailable", "rg_timeout", "rg_error"}
    # Prioritize files satisfying all explicit terms before applying the cap.
    # Replace ripgrep; meter and revalidate discovery and canonical reads.
    if (fallback_used or grep_terms) and not walk_stats.deadline_reached:
        fallback_universe = content_universe if terms else walk_files
        for relative_path in sorted(fallback_universe):
            if monotonic_clock() >= budget.deadline:
                walk_stats.deadline_reached = True
                break
            # The discovery map is only a hint.  Revalidate immediately
            # before opening so a post-walk replacement cannot make the
            # in-process fallback follow an escaped symlink.
            try:
                validated = validate_code_candidate_path(
                    workspace.physical_root,
                    relative_path,
                    language_hints=workspace.language_hints,
                )
            except (OSError, ValueError):
                validated = None
            if validated is None or validated.relative_path != relative_path:
                walk_stats.validation_failures += 1
                continue
            text = _read_literal_source(
                validated.path,
                meter,
                relative_path=relative_path,
            )
            if text is None:
                if meter.exhausted:
                    break
                continue
            item = signals[relative_path]
            source_lower = text.lower()
            item.all_grep_terms_match = bool(grep_terms) and all(term in source_lower for term in grep_terms)
            line_hits = _literal_line_hits(text, terms)
            if not line_hits:
                continue
            literal_matches[relative_path] = _merge_line_hits(
                literal_matches.get(relative_path, ()),
                line_hits,
            )
            item.sources.add("literal_scan" if fallback_used else "scope_scan")
            item.content_occurrences = min(
                _MAX_LITERAL_OCCURRENCES_PER_FILE,
                len(literal_matches[relative_path]),
            )

    # Bound after all discovery sources merge so a ripgrep content hit can
    # displace a weak lexical walker candidate.  No physical path is retained
    # in the returned public-facing diagnostics.
    ranked = sorted(signals.values(), key=_candidate_rank_key)
    pool = ranked[: budget.pool_limit]
    selected_signals = pool[: budget.file_limit]
    selected = tuple(
        CandidateFile(
            relative_path=item.relative_path,
            path_score=item.path_score,
            basename_score=item.basename_score,
            content_occurrences=item.content_occurrences,
            discovery_sources=tuple(sorted(item.sources)),
            explicit_path_match=item.explicit_path_match,
            glob_anchor_match=item.glob_anchor_match,
        )
        for item in selected_signals
    )
    selected_paths = {item.relative_path: item.path for item in selected_signals}
    partial = bool(
        walk_stats.deadline_reached
        or walk_stats.entry_limit_reached
        or meter.exhausted
        or rg_state in {"rg_timeout", "rg_error"}
    )
    state = "scope_literal_scan" if grep_terms else "fallback_literal_scan" if fallback_used else rg_state
    diagnostics = {
        "state": state,
        "backend": "literal_scan" if fallback_used or grep_terms else "ripgrep",
        "fallback": fallback_used,
        "rg_state": rg_state,
        "visited_entries": walk_stats.visited_entries,
        "matched_paths": len(literal_matches),
        "content_universe_count": len(content_universe),
        "pool_count": len(pool),
        "pool_limit_reached": len(walk_files) >= budget.pool_limit,
        "file_limit_reached": len(pool) > budget.file_limit,
        "scope_path_count": len(path_values),
        "files_matching_all_grep_terms": sum(item.all_grep_terms_match for item in pool),
        "selected_files": len(selected),
        "entry_limit_reached": walk_stats.entry_limit_reached,
        "deadline_reached": walk_stats.deadline_reached,
        "read_bytes": meter.used,
        "read_limit_reached": meter.exhausted,
        "validation_failures": walk_stats.validation_failures,
    }
    return CandidateDiscoveryOutcome(
        files=selected,
        paths_by_relative=selected_paths,
        literal_matches=literal_matches,
        read_meter=meter,
        diagnostics=diagnostics,
        partial_failure=partial,
    )


def build_candidate_corpus(
    *,
    workspace: ResolvedCodeWorkspace,
    discovery: CandidateDiscoveryOutcome,
    account_id: str,
    scope: object,
    query_terms: Sequence[str],
    snippet_limit: int,
    deadline: float,
    record_matches_scope: Callable[[Mapping[str, Any], Any], bool],
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> CandidateCodeCorpus:
    """Create canonical snippets and a local BM25 corpus without persistence."""

    records: list[dict[str, Any]] = []
    nodes_by_uri: dict[str, ContextNode] = {}
    file_ranks: dict[str, int] = {}
    files_read = 0
    files_failed = 0
    files_scope_checked = 0
    files_grep_matched = 0
    empty_files = 0
    max_grep_terms_matched = 0
    grep_terms = tuple(dict.fromkeys(str(term).lower() for term in getattr(scope, "grep_terms", ())))
    scope_paths = tuple(getattr(scope, "paths", ()) or ())
    scope_globs = tuple(getattr(scope, "glob_patterns", ()) or ())

    def source_matches(source: str) -> bool:
        nonlocal files_scope_checked, files_grep_matched, empty_files, max_grep_terms_matched
        files_scope_checked += 1
        source_lower = source.lower()
        matched = sum(term in source_lower for term in grep_terms)
        max_grep_terms_matched = max(max_grep_terms_matched, matched)
        if not source.strip():
            empty_files += 1
            return False
        if matched != len(grep_terms):
            return False
        files_grep_matched += 1
        return True

    partial = discovery.partial_failure
    for file_rank, candidate in enumerate(discovery.files, start=1):
        if monotonic_clock() >= deadline:
            partial = True
            break
        # Discovery paths are only hints.  Revalidate their current
        # containment, symlink, ignore, regular-file and language state before
        # every stat/open so a replacement cannot bypass the canonical walker.
        try:
            validated = validate_code_candidate_path(
                workspace.physical_root,
                candidate.relative_path,
                language_hints=workspace.language_hints,
            )
        except (OSError, ValueError):
            validated = None
        if validated is None:
            files_failed += 1
            partial = True
            continue
        path = validated.path
        # Apply path scope before reading files or counting diagnostics.
        relative = candidate.relative_path
        if not _matches_path_scope(relative, scope_paths, scope_globs):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            files_failed += 1
            partial = True
            continue
        if (
            size < 0
            or size > MAX_CODE_FILE_SIZE
            or not discovery.read_meter.reserve_path(candidate.relative_path, int(size))
        ):
            files_failed += 1
            partial = True
            if discovery.read_meter.exhausted:
                break
            continue
        try:
            _file_key, _file_slug, nodes = build_code_resource_nodes(
                account_id=account_id,
                workspace_root=workspace.physical_root,
                file_path=path,
                resource_root_slug=workspace.root_slug,
                root_fingerprint=workspace.root_fingerprint,
                language_hint=workspace.language_hints.get(candidate.relative_path),
                source_filter=source_matches,
                retain_file_source=False,
            )
        except (OSError, UnicodeError, ValueError):
            files_failed += 1
            partial = True
            continue
        files_read += 1
        for node in nodes:
            uri = canonical_node_uri(node.uri)
            metadata = {
                **dict(node.metadata or {}),
                "uri": node.uri,
                "account_id": account_id,
                "context_type": node.context_type,
                "resource_type": "code",
                "category": "code",
                "level": node.level,
                "parent_uri": node.parent_uri,
                "abstract": node.abstract,
                "has_content": bool(node.content),
            }
            record = {
                # Keep the verified predicate request-local, outside persisted indexes.
                "_file_grep_terms": grep_terms,
                "uri": node.uri,
                "metadata": metadata,
                "abstract": node.abstract or "",
                "overview": node.overview or "",
                "content": node.content or "",
            }
            if not record_matches_scope(record, scope):
                continue
            # A node URI is deterministic for the current canonical chunk.  A
            # duplicate never needs a second request-local embedding or BM25
            # document.
            if uri in nodes_by_uri:
                continue
            nodes_by_uri[uri] = node
            file_ranks[uri] = file_rank
            records.append(record)

    # Keep discovery-confirmed literal lines ahead of ordinary relevance when
    # the canonical corpus must be clipped.  In particular, scope.grep_terms
    # need not occur in the natural-language query, yet their line-grounded
    # chunks must survive long enough for candidate_literal_fallback.
    snippet_terms = _literal_terms(query_terms, getattr(scope, "grep_terms", ()))
    literal_hit_by_uri = {
        uri: _node_contains_literal_hit(
            node,
            discovery.literal_matches.get(
                str(node.metadata.get("relative_path") or ""),
                (),
            ),
        )
        for uri, node in nodes_by_uri.items()
    }
    ranked_records = sorted(
        records,
        key=lambda record: _snippet_rank_key(
            record,
            snippet_terms,
            file_ranks.get(canonical_node_uri(str(record.get("uri") or "")), math.inf),
            literal_hit=literal_hit_by_uri.get(
                canonical_node_uri(str(record.get("uri") or "")),
                False,
            ),
        ),
    )[:snippet_limit]
    selected_uris = {canonical_node_uri(str(record.get("uri") or "")) for record in ranked_records}
    selected_nodes = {uri: node for uri, node in nodes_by_uri.items() if uri in selected_uris}
    selected_file_ranks = {uri: rank for uri, rank in file_ranks.items() if uri in selected_uris}
    documents: list[Bm25Document] = []
    for node in selected_nodes.values():
        for index_record in build_index_records(node):
            document = build_code_bm25_document(index_record)
            if document is not None:
                documents.append(document)
    bm25 = Bm25Index()
    bm25.bulk_upsert(documents)
    literal_uris = tuple(
        sorted(
            (
                uri
                for uri, node in selected_nodes.items()
                if _node_contains_literal_hit(
                    node,
                    discovery.literal_matches.get(
                        str(node.metadata.get("relative_path") or ""),
                        (),
                    ),
                )
            ),
            key=lambda uri: (
                selected_file_ranks.get(uri, math.inf),
                code_source_key(selected_nodes[uri].metadata),
            ),
        )
    )
    diagnostics = {
        "files_read": files_read,
        "files_failed": files_failed,
        "files_scope_checked": files_scope_checked,
        "files_grep_matched": files_grep_matched,
        "empty_files": empty_files,
        "grep_term_count": len(grep_terms),
        "max_grep_terms_matched": max_grep_terms_matched,
        "snippets_built": len(records),
        "snippets_selected": len(ranked_records),
        "read_bytes": discovery.read_meter.used,
        "read_limit_reached": discovery.read_meter.exhausted,
    }
    return CandidateCodeCorpus(
        records=tuple(ranked_records),
        nodes_by_uri=selected_nodes,
        bm25_index=bm25,
        file_ranks=selected_file_ranks,
        literal_uris=literal_uris,
        diagnostics=diagnostics,
        partial_failure=partial,
    )


def _literal_terms(query_terms: Sequence[str], grep_terms: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    # Explicit grep scope is a correctness filter, so it receives the scarce
    # ripgrep/fallback literal-term budget before query-only hints.
    for raw in (*grep_terms, *query_terms):
        if not isinstance(raw, str):
            continue
        term = raw.strip()
        if not term or len(term) > 256:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(term)
        if len(result) >= _MAX_RG_TERMS:
            break
    return tuple(result)


def _glob_scan_groups(globs: Sequence[str]) -> tuple[tuple[str | None, tuple[str, ...]], ...]:
    """Group glob patterns by a safe static directory prefix.

    ``iter_bounded_code_files`` can start directly below the non-glob prefix,
    which avoids repeatedly traversing unrelated siblings for patterns such as
    ``src/**/*.py``.  Patterns without a static directory component safely
    scan the resolved root.  ``CodeSearchScope`` has already normalized every
    pattern to a relative POSIX path before this internal helper is called.
    """

    grouped: dict[str | None, list[str]] = {}
    for pattern in globs:
        start_relative_path = _glob_static_directory_prefix(pattern)
        grouped.setdefault(start_relative_path, []).append(pattern)
    return tuple(
        (start_relative_path, tuple(patterns))
        for start_relative_path, patterns in grouped.items()
    )


def _glob_static_directory_prefix(pattern: str) -> str | None:
    """Return the fixed directory portion of one normalized glob pattern."""

    parts = pattern.split("/")
    directories: list[str] = []
    for part in parts[:-1]:
        if any(character in part for character in "*?["):
            break
        directories.append(part)
    return "/".join(directories) or None


def _matches_path_scope(path: str, paths: Sequence[str], globs: Sequence[str]) -> bool:
    return (
        not paths or any(path == p or path.startswith(p.rstrip("/") + "/") for p in paths)
    ) and (not globs or any(fnmatch.fnmatchcase(path, p) for p in globs))


def _apply_path_signals(
    item: _CandidateSignals,
    paths: Sequence[str],
    globs: Sequence[str],
    terms: Sequence[str],
) -> None:
    path = item.relative_path
    path_lower = path.lower()
    basename_lower = Path(path).name.lower()
    if any(path == value or path.startswith(value.rstrip("/") + "/") for value in paths):
        item.explicit_path_match = True
        item.sources.add("scope_path")
    if any(fnmatch.fnmatchcase(path, pattern) for pattern in globs):
        item.glob_anchor_match = True
        item.sources.add("scope_glob")
    basename_hits = sum(1 for term in terms if term.lower() in basename_lower)
    path_hits = sum(1 for term in terms if term.lower() in path_lower)
    if basename_hits:
        item.sources.add("basename")
    if path_hits:
        item.sources.add("path")
    item.basename_score = max(item.basename_score, basename_hits)
    item.path_score = max(item.path_score, path_hits)


def _candidate_rank_key(item: _CandidateSignals) -> tuple[int, int, int, int, int, int, str]:
    return (
        -int(item.explicit_path_match),
        -int(item.glob_anchor_match),
        -int(item.all_grep_terms_match),
        -item.basename_score,
        -item.path_score,
        -item.content_occurrences,
        item.relative_path,
    )


def _snippet_rank_key(
    record: Mapping[str, Any],
    query_terms: Sequence[str],
    file_rank: int | float,
    *,
    literal_hit: bool = False,
) -> tuple[int, float, int | float, tuple]:
    metadata = dict(record.get("metadata") or {})
    symbol = str(metadata.get("symbol") or "").lower()
    signature = str(metadata.get("signature") or "").lower()
    path = str(metadata.get("relative_path") or "").lower()
    body = str(record.get("content") or "").lower()
    score = 0.0
    for raw_term in query_terms:
        term = raw_term.lower()
        if not term:
            continue
        if symbol == term:
            score += 4.0
        elif term in symbol:
            score += 2.0
        if term in signature:
            score += 1.2
        if term in path:
            score += 0.8
        if term in body:
            score += 0.2
    return (
        -int(literal_hit),
        -score,
        file_rank,
        code_source_key(record),
    )


def _read_literal_source(
    path: Path,
    meter: CandidateReadMeter,
    *,
    relative_path: str,
) -> str | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size < 0 or size > MAX_CODE_FILE_SIZE or not meter.reserve_path(relative_path, int(size)):
        return None
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_CODE_FILE_SIZE + 1)
    except OSError:
        return None
    if len(data) > MAX_CODE_FILE_SIZE or b"\x00" in data:
        return None
    return data.decode("utf-8", errors="replace")


def _literal_line_hits(source: str, terms: Sequence[str]) -> tuple[int, ...]:
    """Return bounded one-based literal line locations without retaining text."""

    lowered_terms = tuple(term.lower() for term in terms if term)
    if not lowered_terms:
        return ()
    hits: list[int] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        lowered = line.lower()
        if any(term in lowered for term in lowered_terms):
            hits.append(line_number)
            if len(hits) >= _MAX_LITERAL_OCCURRENCES_PER_FILE:
                break
    return tuple(hits)


def _merge_line_hits(existing: Sequence[int], incoming: Sequence[int]) -> tuple[int, ...]:
    normalized: set[int] = set()
    for value in (*existing, *incoming):
        if isinstance(value, bool):
            continue
        try:
            line_number = int(value)
        except (TypeError, ValueError):
            continue
        if line_number > 0:
            normalized.add(line_number)
    return tuple(sorted(normalized))[:_MAX_LITERAL_OCCURRENCES_PER_FILE]


def _node_contains_literal_hit(node: ContextNode, line_hits: Sequence[int]) -> bool:
    start = node.metadata.get("start_line")
    end = node.metadata.get("end_line")
    if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
        return False
    return any(start <= line <= end for line in line_hits)


def _ripgrep_literal_candidates(
    *,
    workspace: ResolvedCodeWorkspace,
    terms: Sequence[str],
    universe: Mapping[str, Path],
    meter: CandidateReadMeter,
    deadline: float,
    monotonic_clock: Callable[[], float],
) -> tuple[str, Mapping[str, tuple[int, ...]]]:
    """Run a bounded literal ripgrep probe, or return a safe fallback state."""

    if not terms or not universe:
        return "rg_zero_result", {}
    # Pass only an already-validated, byte-reserved universe to ripgrep.  This
    # is deliberately stricter than letting rg recursively read an unbounded
    # root; the same shared read cap also covers canonical chunking afterwards.
    input_paths: list[str] = []
    for relative_path in sorted(universe):
        # The stored discovery path may have changed since the walker admitted
        # it.  Check the current root-contained regular file immediately
        # before handing a relative filename to an external process.
        try:
            candidate = validate_code_candidate_path(
                workspace.physical_root,
                relative_path,
                language_hints=workspace.language_hints,
            )
        except (OSError, ValueError):
            candidate = None
        if candidate is None or candidate.relative_path != relative_path:
            continue
        path = candidate.path
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if (
            size < 0
            or size > MAX_CODE_FILE_SIZE
            or not meter.reserve_discovery_path(relative_path, int(size))
        ):
            break
        input_paths.append(relative_path)
    if not input_paths:
        return "rg_zero_result", {}
    argv = [
        "rg",
        "--json",
        "--fixed-strings",
        "--no-messages",
        "--no-follow",
        "--max-count",
        "1",
    ]
    for term in terms:
        argv.extend(("-e", term))
    argv.extend(("--", *input_paths))
    state, raw_hits = _run_rg_argv(
        argv,
        cwd=workspace.physical_root,
        deadline=deadline,
        monotonic_clock=monotonic_clock,
    )
    if state == "rg_unavailable":
        # FileNotFoundError means the child never scanned the reserved input;
        # release it so the required deterministic fallback can run normally.
        meter.release_discovery_paths(input_paths)
    result: dict[str, tuple[int, ...]] = {}
    for raw_path, line_number in raw_hits:
        try:
            candidate = validate_code_candidate_path(
                workspace.physical_root,
                raw_path,
                language_hints=workspace.language_hints,
            )
        except (OSError, ValueError):
            continue
        if candidate is None or candidate.relative_path not in universe:
            continue
        result[candidate.relative_path] = _merge_line_hits(
            result.get(candidate.relative_path, ()),
            (line_number,),
        )
    return state, result


def _run_rg_argv(
    argv: Sequence[str],
    *,
    cwd: Path,
    deadline: float,
    monotonic_clock: Callable[[], float],
) -> tuple[str, tuple[tuple[str, int], ...]]:
    """Read ripgrep stdout incrementally without shell or stderr disclosure."""

    try:
        process = subprocess.Popen(  # nosec B603 - argv is fixed above
            list(argv),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return "rg_unavailable", ()
    except OSError:
        return "rg_error", ()
    if process.stdout is None:
        _terminate_process(process)
        return "rg_error", ()

    chunks = bytearray()
    try:
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - monotonic_clock()
            if remaining <= 0:
                _terminate_process(process)
                return "rg_timeout", ()
            events = selector.select(min(remaining, 0.1))
            if events:
                data = os.read(process.stdout.fileno(), 4096)
                if data:
                    chunks.extend(data)
                    if len(chunks) > _RG_OUTPUT_BYTES_LIMIT:
                        _terminate_process(process)
                        return "rg_error", ()
                    continue
            if process.poll() is not None:
                # Drain a final ready chunk after process exit.
                while True:
                    data = os.read(process.stdout.fileno(), 4096)
                    if not data:
                        break
                    chunks.extend(data)
                    if len(chunks) > _RG_OUTPUT_BYTES_LIMIT:
                        return "rg_error", ()
                break
    except (OSError, ValueError):
        _terminate_process(process)
        return "rg_error", ()
    finally:
        try:
            selector.close()
        except (UnboundLocalError, OSError):
            pass
        try:
            process.stdout.close()
        except OSError:
            pass
    return_code = process.wait()
    if return_code == 0:
        state = "rg_ok"
    elif return_code == 1:
        state = "rg_zero_result"
    else:
        return "rg_error", ()
    matches: list[tuple[str, int]] = []
    for line in bytes(chunks).splitlines():
        try:
            event = json.loads(line.decode("utf-8", errors="replace"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "rg_error", ()
        if not isinstance(event, Mapping) or event.get("type") != "match":
            continue
        event_data = event.get("data")
        if not isinstance(event_data, Mapping):
            continue
        path_data = event_data.get("path")
        line_number = event_data.get("line_number")
        raw_path = path_data.get("text") if isinstance(path_data, Mapping) else None
        if not isinstance(raw_path, str) or isinstance(line_number, bool) or not isinstance(line_number, int):
            continue
        matches.append((raw_path, line_number))
    return state, tuple(matches)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=0.2)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=0.2)
        except (OSError, subprocess.TimeoutExpired):
            pass


__all__ = [
    "CandidateCodeCorpus",
    "CandidateDiscoveryBudget",
    "CandidateDiscoveryOutcome",
    "CandidateFile",
    "CandidateReadMeter",
    "build_candidate_corpus",
    "candidate_discovery_budget",
    "discover_candidate_files",
]
