"""Account-scoped hybrid retrieval for code RESOURCE nodes."""

from __future__ import annotations

import fnmatch
import inspect
import logging
import math
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.enums import ContextType
from core.errors import ValidationError
from core.models import RequestContext
from core.resource_identity import (
    ResourceIdentityError,
    canonical_node_uri,
    normalize_posix_relative_path,
)
from index.index_record_builder import build_index_records
from retrieval.bm25_index import (
    Bm25Document,
    Bm25Index,
    build_code_bm25_document,
)
from retrieval.code_candidate_local import (
    CandidateCodeCorpus,
    build_candidate_corpus,
    candidate_discovery_budget,
    discover_candidate_files,
)
from retrieval.code_ordering import code_source_key
from service.code_workspace_sync import (
    CodeWorkspaceRegistryResolver,
    WorkspaceResolutionError,
)

logger = logging.getLogger(__name__)

_ROUTE_WEIGHTS: dict[str, float] = {
    "semantic": 0.33,
    "bm25": 0.17,
    "symbolic": 0.17,
    "graph": 0.33,
}
_RRF_CONSTANT = 60.0
_DEFAULT_TOP_K = 8
_USE_SERVICE_RELATION_STORE = object()
ACTIVE_STATUS_DISABLED = "disabled"
ACTIVE_STATUS_OK = "ok"
ACTIVE_STATUS_ZERO_RESULT = "zero_result"
ACTIVE_STATUS_PARTIAL_FAILURE = "partial_failure"
ACTIVE_STATUS_ERROR = "error"
_SAFE_ROOT_SLUG = re.compile(r"[A-Za-z0-9._-]+")
_URI_SCHEME_OR_DRIVE_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_CODE_QUERY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "when",
    "what",
    "where",
    "which",
    "how",
    "does",
    "dont",
    "doesnt",
    "handle",
    "nicely",
    "issue",
    "expected",
    "actual",
    "behavior",
    "possible",
    "file",
    "hints",
    "before",
    "after",
    "function",
    "method",
    "class",
    "module",
    "call",
    "calls",
    "using",
    "used",
}
_SYMBOL_KIND_TERMS = {
    "function": {"function", "def", "fn", "method"},
    "async_function": {"async", "function", "def", "method"},
    "type": {"class", "type", "struct", "interface", "enum"},
    "code": {"code", "snippet"},
}


class _BoundedDaemonCallRunner:
    """Bound outstanding synchronous calls after their callers time out."""

    def __init__(self, max_inflight: int = 8) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be positive")
        self._slots = threading.BoundedSemaphore(max_inflight)

    def run(self, operation, *, deadline: float):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._slots.acquire(timeout=remaining):
            raise TimeoutError("semantic route deadline exceeded")

        # Capacity may become available at the deadline boundary.  Recheck
        # before publishing any new backend work so a timed-out caller cannot
        # start a provider call merely because a semaphore waiter won a race.
        if time.monotonic() >= deadline:
            self._slots.release()
            raise TimeoutError("semantic route deadline exceeded")

        done = threading.Event()
        outcome: dict[str, object] = {}

        def worker() -> None:
            try:
                if time.monotonic() >= deadline:
                    outcome["error"] = TimeoutError("semantic route deadline exceeded")
                else:
                    outcome["value"] = operation()
            except BaseException as exc:  # surfaced in the requesting thread
                outcome["error"] = exc
            finally:
                self._slots.release()
                done.set()

        try:
            threading.Thread(
                target=worker,
                daemon=True,
                name="ogmem-code-semantic-search",
            ).start()
        except BaseException:
            self._slots.release()
            raise

        remaining = deadline - time.monotonic()
        if remaining <= 0 or not done.wait(remaining):
            raise TimeoutError("semantic route deadline exceeded")
        error = outcome.get("error")
        if isinstance(error, BaseException):
            raise error
        return outcome.get("value")


def _validation(field_name: str, reason: str) -> ValidationError:
    return ValidationError(field_name, reason)


def _coerce_string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise _validation(field_name, "must be a string or sequence of strings")
    result: list[str] = []
    for raw in values:
        if not isinstance(raw, str):
            raise _validation(field_name, "must contain only strings")
        value_text = raw.strip()
        if not value_text:
            raise _validation(field_name, "must not contain empty values")
        result.append(value_text)
    return tuple(result)


def _normalize_scope_component(value: str, *, field_name: str, allow_glob: bool) -> str:
    raw = value.replace("\\", "/").strip()
    if (
        not raw
        or "\x00" in raw
        or raw.startswith("/")
        or _URI_SCHEME_OR_DRIVE_PREFIX.match(raw)
    ):
        raise _validation(field_name, "must be a non-empty POSIX-relative value")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise _validation(field_name, "must not contain empty, '.' or '..' segments")
    if not allow_glob and any(char in raw for char in "*?[]"):
        raise _validation(field_name, "must not contain glob metacharacters")
    return "/".join(parts)


@dataclass(frozen=True)
class CodeSearchScope:
    """Shared filters; grep_terms are case-insensitive source literals ANDed per file."""

    root_slugs: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    glob_patterns: tuple[str, ...] = ()
    grep_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        roots = _coerce_string_tuple(self.root_slugs, field_name="scope.root_slugs")
        paths = _coerce_string_tuple(self.paths, field_name="scope.paths")
        globs = _coerce_string_tuple(self.glob_patterns, field_name="scope.glob_patterns")
        grep = _coerce_string_tuple(self.grep_terms, field_name="scope.grep_terms")
        normalized_roots: list[str] = []
        for root in roots:
            if not _SAFE_ROOT_SLUG.fullmatch(root) or root in {".", ".."}:
                raise _validation("scope.root_slugs", "contains an unsafe root slug")
            normalized_roots.append(root)
        normalized_paths = [
            _normalize_scope_component(path, field_name="scope.paths", allow_glob=False)
            for path in paths
        ]
        normalized_globs = [
            _normalize_scope_component(pattern, field_name="scope.glob_patterns", allow_glob=True)
            for pattern in globs
        ]
        normalized_grep: list[str] = []
        for term in grep:
            if len(term) > 256 or "\x00" in term:
                raise _validation("scope.grep_terms", "terms must be bounded literals")
            normalized_grep.append(term)
        object.__setattr__(self, "root_slugs", tuple(dict.fromkeys(normalized_roots)))
        object.__setattr__(self, "paths", tuple(dict.fromkeys(normalized_paths)))
        object.__setattr__(self, "glob_patterns", tuple(dict.fromkeys(normalized_globs)))
        object.__setattr__(self, "grep_terms", tuple(dict.fromkeys(normalized_grep)))


@dataclass(frozen=True)
class CodeSearchBudget:
    """Caller budget.  Hard caps are applied by the configured service."""

    candidate_limit: int | None = None
    result_limit: int | None = None
    snippet_chars: int | None = None
    batch_chars: int | None = None
    graph_neighbor_limit: int | None = None
    route_deadline_ms: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "candidate_limit",
            "result_limit",
            "snippet_chars",
            "batch_chars",
            "graph_neighbor_limit",
            "route_deadline_ms",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise _validation(f"budget.{name}", "must be a positive integer")


@dataclass(frozen=True)
class ActiveCodeSearchRequest:
    """Host-neutral typed request for mode-dispatched active code retrieval."""

    query: str
    scope: CodeSearchScope = field(default_factory=CodeSearchScope)
    budget: CodeSearchBudget = field(default_factory=CodeSearchBudget)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise _validation("query", "query must not be empty")
        if not isinstance(self.scope, CodeSearchScope):
            if isinstance(self.scope, Mapping):
                try:
                    object.__setattr__(self, "scope", CodeSearchScope(**dict(self.scope)))
                except TypeError as exc:
                    raise _validation("scope", "contains unsupported fields") from exc
            else:
                raise _validation("scope", "must be CodeSearchScope")
        if not isinstance(self.budget, CodeSearchBudget):
            if isinstance(self.budget, Mapping):
                try:
                    object.__setattr__(self, "budget", CodeSearchBudget(**dict(self.budget)))
                except TypeError as exc:
                    raise _validation("budget", "contains unsupported fields") from exc
            else:
                raise _validation("budget", "must be CodeSearchBudget")
        object.__setattr__(self, "query", self.query.strip())


@dataclass(frozen=True)
class CodeEvidence:
    """Stable, safe evidence returned by active retrieval."""

    uri: str
    locator: str
    relative_path: str = ""
    root_slug: str = ""
    start_line: int | None = None
    end_line: int | None = None
    symbol: str = ""
    language: str = ""
    symbol_kind: str = ""
    signature: str = ""
    excerpt: str = ""
    graph_context: tuple[dict[str, str], ...] = ()
    fusion_score: float = 0.0
    route_ranks: Mapping[str, int] = field(default_factory=dict)
    route_scores: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    identity: str = ""
    origin: str = "full"
    persisted: bool = True
    focused_read_available: bool = True
    retrieval_source: str = ""
    score_kind: str = ""
    coverage: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return self.excerpt

    @property
    def body(self) -> str:
        return self.excerpt

    @property
    def logical_uri(self) -> str:
        return self.uri

    @property
    def safe_locator(self) -> str:
        return self.locator

    def to_dict(self) -> dict[str, Any]:
        metadata = {
            "resource_type": "code",
            "resource_root_slug": self.root_slug,
            "relative_path": self.relative_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
            "language": self.language,
            "symbol_kind": self.symbol_kind,
            "signature": self.signature,
        }
        if self.retrieval_source:
            metadata["retrieval_source"] = self.retrieval_source
        if self.score_kind:
            metadata["score_kind"] = self.score_kind
        return {
            "uri": self.uri,
            "logical_uri": self.uri,
            "identity": self.identity or self.uri,
            "locator": self.locator,
            "relative_path": self.relative_path,
            "root_slug": self.root_slug,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
            "language": self.language,
            "symbol_kind": self.symbol_kind,
            "signature": self.signature,
            "excerpt": self.excerpt,
            "content": self.excerpt,
            "graph_context": [dict(value) for value in self.graph_context],
            "fusion_score": self.fusion_score,
            "route_ranks": dict(self.route_ranks),
            "route_scores": {key: dict(value) for key, value in self.route_scores.items()},
            "origin": self.origin,
            "persisted": self.persisted,
            "focused_read_available": self.focused_read_available,
            "retrieval_source": self.retrieval_source,
            "score_kind": self.score_kind,
            "metadata": metadata,
            "coverage": dict(self.coverage),
        }


@dataclass(frozen=True)
class ActiveCodeSearchDiagnostics:
    """Non-sensitive route/budget diagnostics for an active search."""

    candidate_limit: int
    result_limit: int
    snippet_chars: int
    batch_chars: int
    duration_ms: int = 0
    scope_match_count: int = 0
    snapshot_read_errors: int = 0
    hydration_errors: int = 0
    deduplicated_count: int = 0
    omitted_count: int = 0
    clipped_count: int = 0
    evidence_chars: int = 0
    line_span_count: int = 0
    budget_clamps: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    routes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    degradation: str | None = None
    mode: str = "full"
    reason: str | None = None
    workspace: Mapping[str, Any] = field(default_factory=dict)
    discovery: Mapping[str, Any] = field(default_factory=dict)
    corpus: Mapping[str, Any] = field(default_factory=dict)
    embedding: Mapping[str, Any] = field(default_factory=dict)
    fusion: Mapping[str, Any] = field(default_factory=dict)
    retention: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "candidate_limit": self.candidate_limit,
            "result_limit": self.result_limit,
            "snippet_chars": self.snippet_chars,
            "batch_chars": self.batch_chars,
            "duration_ms": max(0, int(self.duration_ms)),
            "scope_match_count": self.scope_match_count,
            "snapshot_read_errors": self.snapshot_read_errors,
            "hydration_errors": self.hydration_errors,
            "deduplicated_count": self.deduplicated_count,
            "omitted_count": self.omitted_count,
            "clipped_count": self.clipped_count,
            "budget_clamps": {key: dict(value) for key, value in self.budget_clamps.items()},
            "routes": {key: dict(value) for key, value in self.routes.items()},
            "errors": list(self.errors),
            "degradation": self.degradation,
            "mode": self.mode,
            "reason": self.reason,
            "workspace": dict(self.workspace),
            "discovery": dict(self.discovery),
            "corpus": dict(self.corpus),
            "embedding": dict(self.embedding),
            "fusion": dict(self.fusion),
            "retention": dict(self.retention),
        }
        # These two counters describe only request-local candidate evidence.
        # Keep the established full-mode diagnostics projection byte-for-byte
        # compatible rather than implying that persisted evidence has the
        # same transient-output accounting contract.
        if self.mode == "candidate_local":
            result["evidence_chars"] = max(0, int(self.evidence_chars))
            result["line_span_count"] = max(0, int(self.line_span_count))
        return result


@dataclass(frozen=True)
class ActiveCodeSearchOutcome:
    status: str
    evidence: tuple[CodeEvidence, ...] = ()
    diagnostics: ActiveCodeSearchDiagnostics | Mapping[str, Any] = field(
        default_factory=lambda: ActiveCodeSearchDiagnostics(0, 0, 0, 0)
    )
    error: str | None = None

    @property
    def results(self) -> tuple[CodeEvidence, ...]:
        return self.evidence

    @property
    def items(self) -> tuple[CodeEvidence, ...]:
        return self.evidence

    def to_dict(self) -> dict[str, Any]:
        diagnostics = (
            self.diagnostics.to_dict()
            if hasattr(self.diagnostics, "to_dict")
            else dict(self.diagnostics)
        )
        return {
            "status": self.status,
            "evidence": [item.to_dict() for item in self.evidence],
            "results": [item.to_dict() for item in self.evidence],
            "diagnostics": diagnostics,
            "error": self.error,
        }


# Stable aliases used by thin downstream adapters.
CodeSearchOutcome = ActiveCodeSearchOutcome
CodeSearchEvidence = CodeEvidence
CodeSearchRequest = ActiveCodeSearchRequest
CodeSearchDiagnostics = ActiveCodeSearchDiagnostics
CodeSearchResult = CodeEvidence


@dataclass(frozen=True)
class _CodeIndexSnapshot:
    records: tuple[dict[str, Any], ...]
    read_error_count: int = 0
    bm25_ids_by_uri: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def _route_tokens(*parts: object) -> set[str]:
    text = " ".join(str(part or "") for part in parts)
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _extract_code_query_terms(query: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", query or "")
    seen: set[str] = set()
    terms: list[str] = []
    for term in raw_terms:
        lowered = term.lower()
        if lowered in _CODE_QUERY_STOPWORDS or lowered in {"of", "in", "to", "is", "an", "as", "by", "on", "or", "at", "be", "it"} or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(term)
    return terms


def _focused_candidate_excerpt(
    content: str,
    *,
    max_chars: int,
    source_start_line: int | None,
    primary_terms: Sequence[str],
    fallback_terms: Sequence[str],
    implementation_start_line: int | None = None,
    prefer_implementation: bool = False,
) -> tuple[str, int | None, int | None]:
    """Clip candidate evidence to a bounded line window around the best match."""

    if max_chars <= 0:
        return "", None, None
    valid_start = isinstance(source_start_line, int) and not isinstance(source_start_line, bool) and source_start_line > 0
    if len(content) <= max_chars:
        line_count = max(1, len(content.splitlines()))
        if valid_start:
            return content, source_start_line, source_start_line + line_count - 1
        return content, None, None

    lines = content.splitlines(keepends=True)
    if not lines:
        return content[:max_chars].rstrip(), None, None

    normalized_groups = [
        tuple(dict.fromkeys(term for raw in group if (term := str(raw or "").strip().casefold())))
        for group in (primary_terms, fallback_terms)
    ]

    implementation_index = (
        implementation_start_line - source_start_line
        if isinstance(source_start_line, int) and isinstance(implementation_start_line, int)
        else -1
    )
    prefer_implementation = prefer_implementation and 0 <= implementation_index < len(lines)
    # Explicit literal filters still select matching documentation/declarations.
    literal_match = any(term in content.casefold() for term in normalized_groups[0])
    if prefer_implementation and not literal_match:
        body = "".join(lines[implementation_index:]).rstrip()
        if len(body) <= max_chars:
            return body, implementation_start_line, source_start_line + len(lines) - 1

    focus_index = implementation_index if prefer_implementation else 0
    matched_terms: tuple[str, ...] = ()
    for group_index, terms in enumerate(normalized_groups):
        scored_lines = []
        for index, line in enumerate(lines):
            if group_index == 1 and prefer_implementation and index < implementation_index:
                continue
            lowered = line.casefold()
            matches = tuple(term for term in terms if term in lowered)
            if matches:
                scored_lines.append(
                    (len(matches), sum(len(term) for term in matches), -index, index, matches)
                )
        if scored_lines:
            _count, _matched_chars, _order, focus_index, matched_terms = max(scored_lines)
            break

    # Prefer implementation only when no query term identifies a source line.
    # Preserve explicit docstring/declaration matches.
    if not matched_terms and 0 <= implementation_index < len(lines):
        focus_index = max(focus_index, implementation_index)
    window_start = window_end = focus_index
    focus_line = lines[focus_index]
    if len(focus_line) > max_chars:
        lowered = focus_line.casefold()
        matching_term = max(
            (term for term in matched_terms if term in lowered),
            key=len,
            default="",
        )
        match_offset = lowered.find(matching_term) if matching_term else 0
        left = max(0, match_offset - max(0, max_chars - len(matching_term)) // 2)
        left = min(left, max(0, len(focus_line) - max_chars))
        excerpt = focus_line[left : left + max_chars].rstrip()
    else:
        used = len(focus_line)
        while True:
            added = False
            if window_end + 1 < len(lines) and used + len(lines[window_end + 1]) <= max_chars:
                window_end += 1
                used += len(lines[window_end])
                added = True
            floor = implementation_index if prefer_implementation and not literal_match else 0
            if window_start > floor and used + len(lines[window_start - 1]) <= max_chars:
                window_start -= 1
                used += len(lines[window_start])
                added = True
            if not added:
                break
        excerpt = "".join(lines[window_start : window_end + 1]).rstrip()

    if valid_start:
        return (
            excerpt,
            source_start_line + window_start,
            source_start_line + window_end,
        )
    return excerpt, None, None


def _code_excerpt(
    content: str, metadata: Mapping[str, Any], max_chars: int,
    primary_terms: Sequence[str] = (), fallback_terms: Sequence[str] = (),
) -> tuple[str, int | None, int | None, dict[str, Any]]:
    """Select source once and describe its coverage identically in both modes."""
    start = metadata.get("start_line")
    implementation = metadata.get("implementation_start_line")
    terms = {str(t).casefold() for t in fallback_terms}
    graph = metadata.get("graph")
    symbol = str((graph.get("symbol") if isinstance(graph, Mapping) else None) or metadata.get("symbol") or "")
    documentation = bool(terms & {"docstring", "documentation", "docs", "example", "examples", "doctest"})
    prefer = (not terms or bool(terms & {"implementation", "logic", "body"}) or
              (symbol.rsplit(".", 1)[-1].casefold() in terms and not documentation))
    prefer = prefer and not any(str(t).casefold() in content.casefold() for t in primary_terms)
    excerpt, shown_start, shown_end = _focused_candidate_excerpt(
        content, max_chars=max_chars, source_start_line=start,
        primary_terms=primary_terms, fallback_terms=fallback_terms,
        implementation_start_line=implementation, prefer_implementation=prefer,
    )
    coverage = {"truncated": len(excerpt) < len(content.rstrip())}
    if not isinstance(start, int) or isinstance(start, bool) or start < 1:
        return excerpt, shown_start, shown_end, coverage
    lines = content.splitlines()
    source_start = metadata.get("symbol_start_line", start)
    source_end = metadata.get("symbol_end_line", start + len(lines) - 1)
    if not all(type(n) is int for n in (source_start, source_end)) or not 0 < source_start <= start <= start + len(lines) - 1 <= source_end:
        return excerpt, shown_start, shown_end, coverage
    coverage["source_lines"] = [source_start, source_end]
    whole_lines = (not metadata.get("partial_source_line") and shown_start is not None and shown_end is not None and
                   "\n".join(excerpt.splitlines()).rstrip() == "\n".join(lines[shown_start-start:shown_end-start+1]).rstrip())
    coverage["truncated"] = not (whole_lines and shown_start == source_start and shown_end == source_end)
    read_start = source_start
    if type(implementation) is int and "symbol_end_line" in metadata and source_start <= implementation <= source_end:
        coverage["implementation_lines"] = [implementation, source_end]
        coverage["implementation_complete"] = bool(whole_lines and shown_start <= implementation and shown_end >= source_end)
        if prefer:
            read_start = implementation
    missing = [[read_start, source_end]]
    if whole_lines:
        missing = [[read_start, min(source_end, shown_start-1)], [max(read_start, shown_end+1), source_end]]
    coverage["read_lines"] = [[a, b] for a, b in missing if a <= b]
    return excerpt, shown_start, shown_end, coverage


def _bm25_text(*parts: object) -> str:
    text = " ".join(str(part or "") for part in parts)
    return re.sub(r"[^A-Za-z0-9]+", " ", text).strip()


def _is_code_node(node) -> bool:
    metadata = dict(getattr(node, "metadata", None) or {})
    return bool(
        getattr(node, "context_type", "") == ContextType.RESOURCE.value
        and metadata.get("resource_type") == "code"
    )


def _candidate_canonical_uri(item: dict[str, Any]) -> str:
    metadata = dict(item.get("metadata") or {})
    return str(
        canonical_node_uri(
            str(item.get("uri") or ""),
            parent_uri=str(item.get("parent_uri") or metadata.get("parent_uri") or "") or None,
        )
    )


def weighted_rrf_fuse(
    route_hits: Mapping[str, list[dict[str, Any]]],
    *,
    limit: int = _DEFAULT_TOP_K,
    weights: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Fuse route rankings by canonical node URI with deterministic ties."""

    by_uri: dict[str, dict[str, Any]] = {}
    route_ranks: dict[str, dict[str, int]] = {}
    route_scores: dict[str, dict[str, dict[str, float]]] = {}

    effective_weights = dict(_ROUTE_WEIGHTS)
    if weights is not None:
        effective_weights.update({str(key): float(value) for key, value in weights.items()})

    for route in effective_weights:
        hits = route_hits.get(route, [])
        ranks: dict[str, int] = {}
        scores: dict[str, dict[str, float]] = {}
        for rank, item in enumerate(hits, start=1):
            canonical_uri = _candidate_canonical_uri(item)
            if not canonical_uri or canonical_uri in ranks:
                continue
            metadata = dict(item.get("metadata") or {})
            normalized = float(item.get("score") or 0.0)
            raw_value = (
                item.get("raw_score")
                if item.get("raw_score") is not None
                else metadata.get("_bm25_raw_score", normalized)
            )
            if raw_value is None:
                raw = normalized
            else:
                try:
                    raw = float(raw_value)
                except (TypeError, ValueError):
                    raw = normalized
            ranks[canonical_uri] = rank
            scores[canonical_uri] = {"raw": raw, "normalized": normalized}

            existing = by_uri.get(canonical_uri)
            if existing is None:
                existing = dict(item)
                existing["uri"] = canonical_uri
                existing["metadata"] = metadata
                existing["original_uris"] = [str(item.get("uri") or canonical_uri)]
                by_uri[canonical_uri] = existing
            else:
                existing_metadata = dict(existing.get("metadata") or {})
                existing_metadata.update(metadata)
                existing["metadata"] = existing_metadata
                original_uri = str(item.get("uri") or canonical_uri)
                originals = list(existing.get("original_uris") or [])
                if original_uri not in originals:
                    originals.append(original_uri)
                existing["original_uris"] = originals
        route_ranks[route] = ranks
        route_scores[route] = scores

    fused: list[dict[str, Any]] = []
    for uri, item in by_uri.items():
        routes = [route for route in effective_weights if uri in route_ranks[route]]
        fusion_score = sum(
            effective_weights[route] / (_RRF_CONSTANT + route_ranks[route][uri])
            for route in routes
        )
        metadata = dict(item.get("metadata") or {})
        metadata.update(
            {
                "retrieval_routes": routes,
                "route_scores": {route: route_scores[route][uri] for route in routes},
                "route_ranks": {route: route_ranks[route][uri] for route in routes},
                "fusion_score_kind": "weighted_rrf_semantic_bm25_symbolic_graph",
            }
        )
        fused_item = dict(item)
        fused_item["uri"] = uri
        fused_item["score"] = fusion_score
        fused_item["metadata"] = metadata
        fused.append(fused_item)

    fused.sort(key=lambda item: (-float(item.get("score") or 0.0), code_source_key(item)))
    return fused[: max(0, limit)]


def _weighted_sum_fuse(
    route_hits: Mapping[str, list[dict[str, Any]]],
    *,
    limit: int,
    weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Fuse normalized route scores while retaining canonical URI semantics."""

    by_uri: dict[str, dict[str, Any]] = {}
    route_ranks: dict[str, dict[str, int]] = {}
    route_scores: dict[str, dict[str, dict[str, float]]] = {}
    for route in weights:
        ranks: dict[str, int] = {}
        scores: dict[str, dict[str, float]] = {}
        hits = route_hits.get(route, [])
        raw_values = [float(item.get("score") or 0.0) for item in hits]
        max_value = max(raw_values, default=0.0)
        min_value = min(raw_values, default=0.0)
        span = max_value - min_value
        for rank, item in enumerate(hits, start=1):
            uri = _candidate_canonical_uri(item)
            if not uri or uri in ranks:
                continue
            raw_value = item.get("raw_score")
            if raw_value is None:
                raw_value = item.get("score") or 0.0
            try:
                raw = float(raw_value)
            except (TypeError, ValueError):
                raw = 0.0
            source_score = float(item.get("score") or 0.0)
            normalized = 1.0 if span <= 0.0 and source_score > 0.0 else (
                (source_score - min_value) / span if span > 0.0 else 0.0
            )
            ranks[uri] = rank
            scores[uri] = {"raw": raw, "normalized": normalized}
            existing = by_uri.get(uri)
            if existing is None:
                existing = dict(item)
                existing["uri"] = uri
                existing["metadata"] = dict(item.get("metadata") or {})
                existing["original_uris"] = [str(item.get("uri") or uri)]
                by_uri[uri] = existing
            else:
                metadata = dict(existing.get("metadata") or {})
                metadata.update(dict(item.get("metadata") or {}))
                existing["metadata"] = metadata
                original_uri = str(item.get("uri") or uri)
                if original_uri not in existing.setdefault("original_uris", []):
                    existing["original_uris"].append(original_uri)
        route_ranks[route] = ranks
        route_scores[route] = scores

    fused: list[dict[str, Any]] = []
    for uri, item in by_uri.items():
        routes = [route for route in weights if uri in route_ranks[route]]
        score = sum(weights[route] * route_scores[route][uri]["normalized"] for route in routes)
        metadata = dict(item.get("metadata") or {})
        metadata.update(
            {
                "retrieval_routes": routes,
                "route_scores": {route: route_scores[route][uri] for route in routes},
                "route_ranks": {route: route_ranks[route][uri] for route in routes},
                "fusion_score_kind": "weighted_sum",
            }
        )
        fused_item = dict(item)
        fused_item["uri"] = uri
        fused_item["score"] = score
        fused_item["metadata"] = metadata
        fused.append(fused_item)
    fused.sort(key=lambda item: (-float(item.get("score") or 0.0), code_source_key(item)))
    return fused[: max(0, limit)]


def _retain_code_matches(
    fused: list[dict[str, Any]], query: str, weights: Mapping[str, float], limit: int,
) -> list[dict[str, Any]]:
    """Reserve at most two evidence slots without changing route scores or IDs."""
    terms = _extract_code_query_terms(query)
    # Sentence capitalization alone is not an explicit code identifier.
    identifiers = [term for term in terms if (
        "_" in term.strip("_") or any(c.isupper() for c in term[1:])
        or f"`{term}`" in query or query.strip() == term
        or re.search(rf"\b{re.escape(term)}\.|\.{re.escape(term)}\b", query)
    )]

    identifiers = {re.sub(r"[^a-z0-9]", "", term.lower()) for term in identifiers}

    def matches_symbol(item):
        symbol = str(item.get("metadata", {}).get("symbol") or "")
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+", symbol)
        boundaries = {0}
        offset = 0
        for part in parts:
            offset += len(part)
            boundaries.add(offset)
        joined = "".join(parts).lower()
        return any(start + len(term) in boundaries and joined.startswith(term, start)
                   for term in identifiers for start in boundaries)

    def rank(item, route):
        if weights.get(route, 0.0) <= 0:
            return math.inf
        return item.get("metadata", {}).get("route_ranks", {}).get(route, math.inf)

    protected = []
    explicit = [item for item in fused if rank(item, "symbolic") < math.inf and matches_symbol(item)]
    if explicit:
        protected.append(min(explicit, key=lambda item: (rank(item, "symbolic"), code_source_key(item))))
    # Require lexical support for the top semantic hit.
    semantic = [item for item in fused if rank(item, "semantic") == 1
                and min(rank(item, "symbolic"), rank(item, "bm25")) <= 3
                and item.get("metadata", {}).get("route_scores", {}).get("semantic", {}).get("raw", 0.0) > 0]
    if semantic:
        protected.append(semantic[0])
    selected = []
    seen = set()
    for item in protected + fused:
        if item["uri"] not in seen:
            seen.add(item["uri"])
            selected.append(item)
    return selected[:max(0, limit)]


def fuse_route_hits(
    route_hits: Mapping[str, list[dict[str, Any]]],
    *,
    mode: str = "weighted_rrf",
    weights: Mapping[str, float] | None = None,
    limit: int = _DEFAULT_TOP_K,
    query: str = "",
) -> list[dict[str, Any]]:
    """Apply one of the supported deterministic fusion modes."""

    normalized_mode = str(mode or "weighted_rrf").strip().lower()
    effective_weights = dict(_ROUTE_WEIGHTS)
    if weights is not None:
        effective_weights.update({str(key): float(value) for key, value in weights.items()})
    fusion_limit = sum(len(hits) for hits in route_hits.values()) if query else limit
    if normalized_mode == "weighted_rrf":
        fused = weighted_rrf_fuse(route_hits, limit=fusion_limit, weights=effective_weights)
    elif normalized_mode in {"weighted_sum", "sum"}:
        fused = _weighted_sum_fuse(
            route_hits,
            limit=fusion_limit,
            weights=effective_weights,
        )
    else:
        raise ValueError(f"unsupported code retrieval fusion mode: {mode}")
    return _retain_code_matches(fused, query, effective_weights, limit) if query else fused


def _code_evidence_path(metadata: dict[str, Any]) -> str:
    """Return a root-qualified locator path without exposing filesystem paths."""

    relative_path = _safe_code_relative_path(metadata)
    if not relative_path:
        relative_path = "<unknown>"

    if "resource_root_slug" not in metadata:
        return relative_path

    root_slug = str(metadata.get("resource_root_slug") or "").strip()
    if (
        relative_path != "<unknown>"
        and _SAFE_ROOT_SLUG.fullmatch(root_slug)
        and root_slug not in {".", ".."}
    ):
        return f"{root_slug}/{relative_path}"
    return "<unknown>"


def _safe_code_relative_path(metadata: Mapping[str, Any]) -> str:
    """Return only a normalized logical path; invalid values are withheld."""

    raw_relative_path = str(metadata.get("relative_path") or "").strip()
    if not raw_relative_path or _URI_SCHEME_OR_DRIVE_PREFIX.match(raw_relative_path):
        return ""
    try:
        return str(normalize_posix_relative_path(raw_relative_path))
    except ResourceIdentityError:
        return ""


def format_code_evidence_entries(
    items: list[dict[str, Any]],
    *,
    body_budget_chars: int,
) -> list[str]:
    """Format code evidence while preserving every complete locator."""

    if not items:
        return []
    per_item_body = max(0, body_budget_chars // len(items))
    entries: list[str] = []
    for item in items:
        metadata = dict(item.get("metadata") or {})
        evidence_path = _code_evidence_path(metadata)
        start_line = metadata.get("start_line")
        end_line = metadata.get("end_line")
        if isinstance(start_line, int) and isinstance(end_line, int):
            locator = f"{evidence_path}:{start_line}-{end_line}"
        else:
            locator = evidence_path
        qualifiers = [
            str(metadata.get("symbol") or "").strip(),
            str(metadata.get("language") or "").strip(),
        ]
        qualifier_text = " | ".join(value for value in qualifiers if value)
        header = locator if not qualifier_text else f"{locator} | {qualifier_text}"
        body = str(item.get("content") or item.get("overview") or item.get("abstract") or "")
        if per_item_body and len(body) > per_item_body:
            marker = "\n... [code body clipped] ..."
            if per_item_body <= len(marker):
                body = ""
            else:
                body = body[: per_item_body - len(marker)].rstrip() + marker
        elif per_item_body == 0:
            body = ""
        entries.append(f"- [code] {header}" + (f"\n{body}" if body else ""))
    return entries


class CodeRetrievalService:
    """Run semantic, BM25, symbolic, and graph code-resource routes."""

    def __init__(
        self,
        *,
        context_fs=None,
        read_api=None,
        bm25_index: Bm25Index | None = None,
        config,
        relation_store=None,
        workspace_resolver: CodeWorkspaceRegistryResolver | None = None,
        candidate_embedder=None,
    ) -> None:
        self._fs = context_fs
        self._read_api = read_api
        self._bm25 = bm25_index
        self._cfg = config
        self._relation_store = relation_store
        self._workspace_resolver = workspace_resolver
        self._candidate_embedder = candidate_embedder
        self._cache: dict[str, _CodeIndexSnapshot] = {}
        self._cache_lock = threading.Lock()
        self._semantic_call_runner = _BoundedDaemonCallRunner()

    def invalidate(self, account_id: str | None = None) -> None:
        with self._cache_lock:
            if account_id is None:
                self._cache.clear()
            else:
                self._cache.pop(account_id, None)

    def search(
        self,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int = _DEFAULT_TOP_K,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Project the canonical active outcome for legacy Compose callers."""

        outcome = self.search_active(
            ActiveCodeSearchRequest(
                query=query,
                budget=CodeSearchBudget(
                    candidate_limit=top_k,
                    result_limit=top_k,
                ),
            ),
            ctx,
        )
        diagnostics = (
            outcome.diagnostics.to_dict()
            if hasattr(outcome.diagnostics, "to_dict")
            else dict(outcome.diagnostics)
        )
        route_diagnostics = dict(diagnostics.get("routes") or {})
        route_errors = {
            route: str(route_diagnostic.get("error") or "TimeoutError")
            for route, route_diagnostic in route_diagnostics.items()
            if route_diagnostic.get("state") in {"error", "timeout"}
        }
        items: list[dict[str, Any]] = []
        projection_hydration_errors = 0
        for evidence in outcome.evidence:
            item = evidence.to_dict()
            abstract = ""
            overview = ""
            content = evidence.excerpt
            metadata = dict(item.get("metadata") or {})
            if evidence.persisted:
                # The active outcome intentionally carries a bounded excerpt.
                # Compose historically receives the durable node body instead,
                # so rehydrate only that already-authorized exact URI here.
                # Candidate-local evidence is deliberately transient and must
                # never enter this ContextFS path.
                try:
                    node = self._fs.read_node(evidence.uri, ctx)
                except Exception:
                    projection_hydration_errors += 1
                    continue
                node_uri = str(getattr(node, "uri", ""))
                if (
                    not node_uri.startswith(f"ctx://{ctx.account_id}/")
                    or canonical_node_uri(node_uri) != canonical_node_uri(evidence.uri)
                    or not _is_code_node(node)
                ):
                    projection_hydration_errors += 1
                    continue
                abstract = str(node.abstract or "")
                overview = str(node.overview or "")
                content = str(node.content or "")
                metadata.update(dict(node.metadata or {}))
                metadata.pop("file_source", None)
            item.update(
                {
                    "abstract": abstract,
                    "overview": overview,
                    "content": content,
                    "score": evidence.fusion_score,
                    "category": "code",
                    "metadata": metadata,
                }
            )
            items.append(item)
        stats = {
            "code_retrieval_enabled": bool(getattr(self._cfg, "code_retrieval_enabled", False)),
            "code_retrieval_method": (
                "candidate_local"
                if diagnostics.get("mode") == "candidate_local"
                else "hybrid"
            ),
            "code_retrieval_routes": [
                route
                for route, route_diagnostic in route_diagnostics.items()
                if int(route_diagnostic.get("count") or 0) > 0
            ],
            "code_route_hit_counts": {
                route: int(route_diagnostic.get("count") or 0)
                for route, route_diagnostic in route_diagnostics.items()
            },
            "code_route_errors": route_errors,
            "code_hydration_errors": (
                int(diagnostics.get("hydration_errors") or 0) + projection_hydration_errors
            ),
            "code_index_read_errors": int(diagnostics.get("snapshot_read_errors") or 0),
            "code_retrieval": {
                "mode": diagnostics.get("mode", "full"),
                "status": outcome.status,
                "evidence_count": len(outcome.evidence),
                "reason": diagnostics.get("reason"),
                "diagnostics": diagnostics,
            },
        }
        return items, stats

    def search_active(
        self,
        request: ActiveCodeSearchRequest,
        ctx: RequestContext,
    ) -> ActiveCodeSearchOutcome:
        """Run the one canonical mode-dispatching code-search outcome."""

        request = self._validate_active_request(request, ctx)
        if self._code_retrieval_mode() == "candidate_local":
            return self._search_candidate_active(request, ctx)
        return self._search_full_active(request, ctx)

    def _search_full_active(
        self,
        request: ActiveCodeSearchRequest,
        ctx: RequestContext,
    ) -> ActiveCodeSearchOutcome:
        """Search imported code resources using the typed active contract.

        This method only reads ContextFS snapshots and the existing persistent
        read/vector/BM25 providers.  Scope predicates are applied to every
        route before route top-k and fusion; no workspace or lifecycle service
        is reachable from this path.
        """

        search_started = time.monotonic()
        candidate_limit, result_limit, snippet_chars, batch_chars, neighbor_limit, clamps = (
            self._active_budget(request.budget)
        )
        deadline_ms = self._effective_route_deadline_ms(request.budget, clamps)
        if not bool(getattr(self._cfg, "code_retrieval_enabled", False)):
            routes = {
                route: {"state": "disabled", "count": 0, "duration_ms": 0}
                for route in _ROUTE_WEIGHTS
            }
            return ActiveCodeSearchOutcome(
                status="disabled",
                diagnostics=ActiveCodeSearchDiagnostics(
                    candidate_limit,
                    result_limit,
                    snippet_chars,
                    batch_chars,
                    duration_ms=int(max(0.0, (time.monotonic() - search_started) * 1000)),
                    budget_clamps=clamps,
                    routes=routes,
                ),
            )

        try:
            snapshot = self._get_snapshot(ctx)
        except Exception as exc:
            logger.warning("active code snapshot failed account=%s error=%s", ctx.account_id, type(exc).__name__)
            diagnostics = ActiveCodeSearchDiagnostics(
                candidate_limit,
                result_limit,
                snippet_chars,
                batch_chars,
                duration_ms=int(max(0.0, (time.monotonic() - search_started) * 1000)),
                budget_clamps=clamps,
                routes={
                    route: {"state": "error", "count": 0, "duration_ms": 0}
                    for route in _ROUTE_WEIGHTS
                },
                errors=(type(exc).__name__,),
            )
            return ActiveCodeSearchOutcome(status="error", diagnostics=diagnostics, error=type(exc).__name__)

        scoped_records = self._scope_full_records(snapshot.records, request.scope)
        scoped_uris = {
            canonical_node_uri(
                str(record.get("uri") or ""),
                parent_uri=str(record.get("metadata", {}).get("parent_uri") or "") or None,
            )
            for record in scoped_records
        }
        scoped_record_by_uri = {
            canonical_node_uri(
                str(record.get("uri") or ""),
                parent_uri=str(record.get("metadata", {}).get("parent_uri") or "") or None,
            ): record
            for record in scoped_records
        }
        snapshot_read_errors = int(getattr(snapshot, "read_error_count", 0) or 0)
        if not scoped_records:
            routes = {
                route: {
                    "state": "zero_result" if self._route_enabled(route) else "disabled",
                    "count": 0,
                    "duration_ms": 0,
                }
                for route in _ROUTE_WEIGHTS
            }
            diagnostics = ActiveCodeSearchDiagnostics(
                candidate_limit,
                result_limit,
                snippet_chars,
                batch_chars,
                duration_ms=int(max(0.0, (time.monotonic() - search_started) * 1000)),
                scope_match_count=0,
                snapshot_read_errors=snapshot_read_errors,
                budget_clamps=clamps,
                routes=routes,
            )
            status = "partial_failure" if snapshot_read_errors else "zero_result"
            return ActiveCodeSearchOutcome(status=status, diagnostics=diagnostics)
        deadline = time.monotonic() + max(1, deadline_ms) / 1000.0
        route_hits: dict[str, list[dict[str, Any]]] = {route: [] for route in _ROUTE_WEIGHTS}
        route_diags: dict[str, dict[str, Any]] = {}
        route_errors: list[str] = []
        route_searchers = {
            "semantic": self._search_semantic,
            "bm25": self._search_bm25,
            "symbolic": self._search_symbolic,
        }
        scope_has_filters = any(
            (
                request.scope.root_slugs,
                request.scope.paths,
                request.scope.glob_patterns,
                request.scope.grep_terms,
            )
        )
        allowed_bm25_ids: set[str] | None = None
        snapshot_bm25_ids = getattr(snapshot, "bm25_ids_by_uri", None)
        if isinstance(snapshot_bm25_ids, Mapping):
            allowed_bm25_ids = {
                doc_id
                for uri in scoped_uris
                for doc_id in snapshot_bm25_ids.get(uri, ())
            }
        else:
            # A valid durable snapshot always records L0/L2 IDs.  If a
            # compatibility test double or a partially rebuilt snapshot does
            # not, fail closed instead of allowing an unscoped BM25 top-k.
            allowed_bm25_ids = set()
        for route in ("semantic", "bm25", "symbolic", "graph"):
            if not self._route_enabled(route):
                route_diags[route] = {"state": "disabled", "count": 0, "duration_ms": 0}
                continue
            started = time.monotonic()
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError("route deadline exceeded")
                if route == "graph":
                    hits = self._search_graph_resolved(
                        request.query,
                        ctx,
                        records=scoped_records,
                        top_k=candidate_limit,
                        neighbor_limit=neighbor_limit,
                        deadline=deadline,
                    )
                else:
                    extra_kwargs: dict[str, Any] = {}
                    if route == "bm25" and allowed_bm25_ids is not None:
                        extra_kwargs["allowed_ids"] = allowed_bm25_ids
                    elif route == "symbolic":
                        extra_kwargs["records"] = scoped_records
                    hits = self._invoke_route_searcher(
                        route_searchers[route],
                        request.query,
                        ctx,
                        top_k=candidate_limit,
                        deadline=deadline,
                        extra_kwargs=extra_kwargs,
                    )
                    raw_count = len(hits)
                    hits = [
                        item
                        for item in hits
                        if self._item_matches_scope(
                            item,
                            request.scope,
                            scoped_uris,
                            scoped_record_by_uri,
                        )
                    ][:candidate_limit]
                    scope_filtered_count = max(0, raw_count - len(hits))
                if route == "graph":
                    raw_count = len(hits)
                    scope_filtered_count = 0
                expected_scoped_count = min(candidate_limit, len(scoped_records))
                route_hits[route] = hits[:candidate_limit]
                route_diags[route] = {
                    "state": "ok" if hits else "zero_result",
                    "count": len(route_hits[route]),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "scope_filtered_count": scope_filtered_count,
                    # Semantic provider top-k is account/code scoped but cannot
                    # express every logical scope hint.  RETR-12 route fill is
                    # intentionally excluded; expose a bounded underfill
                    # signal instead of silently claiming complete recall.
                    "scope_underfill": bool(
                        scope_has_filters
                        and raw_count >= candidate_limit
                        and len(route_hits[route]) < expected_scoped_count
                    ),
                }
            except TimeoutError:
                route_diags[route] = {
                    "state": "timeout",
                    "count": 0,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                }
                route_errors.append(f"{route}:timeout")
            except Exception as exc:
                route_diags[route] = {
                    "state": "error",
                    "count": 0,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "error": type(exc).__name__,
                }
                route_errors.append(f"{route}:{type(exc).__name__}")
                logger.warning(
                    "active code route failed account=%s route=%s error=%s",
                    ctx.account_id,
                    route,
                    type(exc).__name__,
                )

        weights = self._code_route_weights()
        mode = str(getattr(self._cfg, "code_retrieval_fusion_mode", "weighted_rrf"))
        fused = fuse_route_hits(
            route_hits,
            query=request.query,
            mode=mode,
            weights=weights,
            limit=candidate_limit,
        )
        evidence, hydration_errors, clipped_count, omitted_count = self._hydrate_active(
            fused,
            ctx,
            result_limit=result_limit,
            snippet_chars=snippet_chars,
            batch_chars=batch_chars,
            primary_focus_terms=request.scope.grep_terms,
            fallback_focus_terms=_extract_code_query_terms(request.query),
        )
        route_count = sum(len(hits) for hits in route_hits.values())
        dedup_count = max(0, route_count - len({item.get("uri") for hits in route_hits.values() for item in hits}))
        if not evidence:
            status = (
                "partial_failure"
                if route_errors or hydration_errors or snapshot_read_errors
                else "zero_result"
            )
        else:
            status = (
                "partial_failure"
                if route_errors or hydration_errors or snapshot_read_errors
                else "ok"
            )
        diagnostics = ActiveCodeSearchDiagnostics(
            candidate_limit,
            result_limit,
            snippet_chars,
            batch_chars,
            duration_ms=int(max(0.0, (time.monotonic() - search_started) * 1000)),
            scope_match_count=len(scoped_records),
            snapshot_read_errors=snapshot_read_errors,
            hydration_errors=hydration_errors,
            deduplicated_count=dedup_count,
            omitted_count=omitted_count,
            clipped_count=clipped_count,
            budget_clamps=clamps,
            routes=route_diags,
            errors=tuple(route_errors),
            degradation=("other_persisted_routes_available" if route_errors and evidence else None),
        )
        return ActiveCodeSearchOutcome(status=status, evidence=tuple(evidence), diagnostics=diagnostics)

    def _search_candidate_active(
        self,
        request: ActiveCodeSearchRequest,
        ctx: RequestContext,
    ) -> ActiveCodeSearchOutcome:
        """Run a complete request-local candidate search without durable reads.

        Every candidate file, canonical node, local BM25 corpus, vector, and
        fusion map stays in this method's stack frame.  In particular this
        branch never calls the full snapshot, ContextFS, ReadAPI, shared BM25,
        relation store, lifecycle, or sync paths.
        """

        search_started = time.monotonic()
        candidate_limit, result_limit, snippet_chars, batch_chars, neighbor_limit, clamps = (
            self._active_budget(request.budget)
        )
        request_deadline_ms = self._candidate_request_deadline_ms(request.budget, clamps)
        request_deadline = search_started + request_deadline_ms / 1000.0
        retention = {
            "scope": "request",
            "retained_candidates": 0,
            "retained_vectors": 0,
            "cache_policy": "request_uncached",
        }

        if not bool(getattr(self._cfg, "code_retrieval_enabled", False)):
            routes = {
                route: {"state": "disabled", "count": 0, "duration_ms": 0}
                for route in _ROUTE_WEIGHTS
            }
            return ActiveCodeSearchOutcome(
                status=ACTIVE_STATUS_DISABLED,
                diagnostics=ActiveCodeSearchDiagnostics(
                    candidate_limit,
                    result_limit,
                    snippet_chars,
                    batch_chars,
                    duration_ms=int(max(0.0, (time.monotonic() - search_started) * 1000)),
                    budget_clamps=clamps,
                    routes=routes,
                    mode="candidate_local",
                    workspace={
                        "state": "not_resolved",
                        "selection": "explicit" if request.scope.root_slugs else "sole_root",
                        "available_root_count": 0,
                    },
                    retention=retention,
                ),
            )

        selection = "explicit" if request.scope.root_slugs else "sole_root"
        resolver = self._workspace_resolver
        if resolver is None:
            return self._candidate_workspace_error_outcome(
                candidate_limit=candidate_limit,
                result_limit=result_limit,
                snippet_chars=snippet_chars,
                batch_chars=batch_chars,
                clamps=clamps,
                search_started=search_started,
                selection=selection,
                reason="workspace_not_registered",
                retention=retention,
            )
        try:
            workspace = resolver.resolve(ctx=ctx, root_slugs=request.scope.root_slugs)
        except WorkspaceResolutionError as exc:
            return self._candidate_workspace_error_outcome(
                candidate_limit=candidate_limit,
                result_limit=result_limit,
                snippet_chars=snippet_chars,
                batch_chars=batch_chars,
                clamps=clamps,
                search_started=search_started,
                selection=selection,
                reason=exc.reason,
                available_root_count=exc.available_root_count,
                retention=retention,
            )
        except Exception as exc:
            logger.warning(
                "candidate workspace resolution failed account=%s error=%s",
                ctx.account_id,
                type(exc).__name__,
            )
            return self._candidate_workspace_error_outcome(
                candidate_limit=candidate_limit,
                result_limit=result_limit,
                snippet_chars=snippet_chars,
                batch_chars=batch_chars,
                clamps=clamps,
                search_started=search_started,
                selection=selection,
                reason="workspace_invalid",
                retention=retention,
            )

        workspace_diagnostics = {
            "state": "resolved",
            "selection": selection,
            "available_root_count": workspace.available_root_count,
        }
        discovery_started = time.monotonic()
        discovery_deadline = min(
            request_deadline,
            discovery_started
            + min(
                15_000,
                self._cfg_int("code_retrieval_candidate_discovery_deadline_ms", 15_000),
            )
            / 1000.0,
        )
        try:
            if time.monotonic() >= discovery_deadline:
                raise TimeoutError("candidate discovery deadline exceeded")
            discovery = discover_candidate_files(
                workspace=workspace,
                scope=request.scope,
                query_terms=_extract_code_query_terms(request.query),
                budget=candidate_discovery_budget(self._cfg, deadline=discovery_deadline),
            )
        except TimeoutError:
            return self._candidate_stage_failure_outcome(
                candidate_limit=candidate_limit,
                result_limit=result_limit,
                snippet_chars=snippet_chars,
                batch_chars=batch_chars,
                clamps=clamps,
                search_started=search_started,
                workspace=workspace_diagnostics,
                discovery={
                    "state": "timeout",
                    "duration_ms": int(max(0.0, (time.monotonic() - discovery_started) * 1000)),
                },
                retention=retention,
                error="discovery:timeout",
            )
        except Exception as exc:
            logger.warning(
                "candidate discovery failed account=%s error=%s",
                ctx.account_id,
                type(exc).__name__,
            )
            return self._candidate_stage_failure_outcome(
                candidate_limit=candidate_limit,
                result_limit=result_limit,
                snippet_chars=snippet_chars,
                batch_chars=batch_chars,
                clamps=clamps,
                search_started=search_started,
                workspace=workspace_diagnostics,
                discovery={
                    "state": "error",
                    "duration_ms": int(max(0.0, (time.monotonic() - discovery_started) * 1000)),
                },
                retention=retention,
                error="discovery:error",
            )
        discovery_diagnostics = {
            **dict(discovery.diagnostics),
            "duration_ms": int(max(0.0, (time.monotonic() - discovery_started) * 1000)),
        }

        corpus_started = time.monotonic()
        try:
            if time.monotonic() >= request_deadline:
                raise TimeoutError("candidate request deadline exceeded")
            corpus = build_candidate_corpus(
                workspace=workspace,
                discovery=discovery,
                account_id=ctx.account_id,
                scope=request.scope,
                query_terms=_extract_code_query_terms(request.query),
                snippet_limit=min(
                    500,
                    self._cfg_int("code_retrieval_candidate_snippet_limit", 500),
                ),
                deadline=request_deadline,
                record_matches_scope=self._record_matches_scope,
            )
        except TimeoutError:
            return self._candidate_stage_failure_outcome(
                candidate_limit=candidate_limit,
                result_limit=result_limit,
                snippet_chars=snippet_chars,
                batch_chars=batch_chars,
                clamps=clamps,
                search_started=search_started,
                workspace=workspace_diagnostics,
                discovery=discovery_diagnostics,
                corpus={
                    "state": "timeout",
                    "duration_ms": int(max(0.0, (time.monotonic() - corpus_started) * 1000)),
                },
                retention=retention,
                error="corpus:timeout",
            )
        except Exception as exc:
            logger.warning(
                "candidate corpus construction failed account=%s error=%s",
                ctx.account_id,
                type(exc).__name__,
            )
            return self._candidate_stage_failure_outcome(
                candidate_limit=candidate_limit,
                result_limit=result_limit,
                snippet_chars=snippet_chars,
                batch_chars=batch_chars,
                clamps=clamps,
                search_started=search_started,
                workspace=workspace_diagnostics,
                discovery=discovery_diagnostics,
                corpus={
                    "state": "error",
                    "duration_ms": int(max(0.0, (time.monotonic() - corpus_started) * 1000)),
                },
                retention=retention,
                error="corpus:error",
            )
        corpus_diagnostics = {
            **dict(corpus.diagnostics),
            "duration_ms": int(max(0.0, (time.monotonic() - corpus_started) * 1000)),
        }

        if not corpus.records:
            routes = {
                route: {
                    "state": "disabled" if not self._route_enabled(route) else "zero_result",
                    "count": 0,
                    "duration_ms": 0,
                }
                for route in _ROUTE_WEIGHTS
            }
            partial = discovery.partial_failure or corpus.partial_failure
            diagnostics = ActiveCodeSearchDiagnostics(
                candidate_limit,
                result_limit,
                snippet_chars,
                batch_chars,
                duration_ms=int(max(0.0, (time.monotonic() - search_started) * 1000)),
                scope_match_count=0,
                budget_clamps=clamps,
                routes=routes,
                errors=("candidate:truncated",) if partial else (),
                degradation="candidate_truncated" if partial else None,
                mode="candidate_local",
                workspace=workspace_diagnostics,
                discovery=discovery_diagnostics,
                corpus=corpus_diagnostics,
                embedding=self._empty_candidate_embedding_diagnostics(),
                fusion={"mode": self._candidate_fusion_mode(), "fused_count": 0, "route_fill_count": 0, "literal_fallback_count": 0},
                retention=retention,
            )
            return ActiveCodeSearchOutcome(
                status=ACTIVE_STATUS_PARTIAL_FAILURE if partial else ACTIVE_STATUS_ZERO_RESULT,
                diagnostics=diagnostics,
            )

        route_hits: dict[str, list[dict[str, Any]]] = {route: [] for route in _ROUTE_WEIGHTS}
        route_diagnostics: dict[str, dict[str, Any]] = {}
        route_errors: list[str] = []
        for route in ("bm25", "symbolic", "graph"):
            if not self._route_enabled(route):
                route_diagnostics[route] = {"state": "disabled", "count": 0, "duration_ms": 0}
                continue
            route_started = time.monotonic()
            try:
                if time.monotonic() >= request_deadline:
                    raise TimeoutError("candidate request deadline exceeded")
                if route == "bm25":
                    hits = self._search_bm25_index(
                        request.query,
                        ctx,
                        bm25_index=corpus.bm25_index,
                        top_k=candidate_limit,
                        deadline=request_deadline,
                    )
                elif route == "symbolic":
                    hits = self._search_symbolic(
                        request.query,
                        ctx,
                        records=corpus.records,
                        top_k=candidate_limit,
                        deadline=request_deadline,
                    )
                else:
                    hits = self._search_graph_resolved(
                        request.query,
                        ctx,
                        records=corpus.records,
                        top_k=candidate_limit,
                        neighbor_limit=neighbor_limit,
                        deadline=request_deadline,
                        relation_store=None,
                    )
                route_hits[route] = hits[:candidate_limit]
                route_diagnostics[route] = {
                    "state": "ok" if route_hits[route] else "zero_result",
                    "count": len(route_hits[route]),
                    "duration_ms": int(max(0.0, (time.monotonic() - route_started) * 1000)),
                }
            except TimeoutError:
                route_diagnostics[route] = {
                    "state": "timeout",
                    "count": 0,
                    "duration_ms": int(max(0.0, (time.monotonic() - route_started) * 1000)),
                }
                route_errors.append(f"{route}:timeout")
            except Exception as exc:
                logger.warning(
                    "candidate code route failed account=%s route=%s error=%s",
                    ctx.account_id,
                    route,
                    type(exc).__name__,
                )
                route_diagnostics[route] = {
                    "state": "error",
                    "count": 0,
                    "duration_ms": int(max(0.0, (time.monotonic() - route_started) * 1000)),
                    "error_kind": "route_error",
                }
                route_errors.append(f"{route}:error")

        embedding_deadline = self._candidate_embedding_deadline(
            stage_started=time.monotonic(),
            request_deadline=request_deadline,
        )
        if not self._route_enabled("semantic"):
            route_diagnostics["semantic"] = {"state": "disabled", "count": 0, "duration_ms": 0}
            embedding_diagnostics = self._empty_candidate_embedding_diagnostics(state="disabled")
        else:
            semantic_records = self._candidate_semantic_records(corpus, route_hits)
            semantic_started = time.monotonic()
            semantic_hits, embedding_diagnostics = self._search_candidate_semantic(
                query=request.query,
                ctx=ctx,
                records=semantic_records,
                embedding_deadline=embedding_deadline,
                request_deadline=request_deadline,
            )
            route_hits["semantic"] = semantic_hits[:candidate_limit]
            semantic_state = str(embedding_diagnostics.get("state") or "zero_result")
            route_diagnostics["semantic"] = {
                "state": semantic_state,
                "count": len(route_hits["semantic"]),
                "duration_ms": int(max(0.0, (time.monotonic() - semantic_started) * 1000)),
            }
            if embedding_diagnostics.get("error_kind"):
                route_diagnostics["semantic"]["error_kind"] = embedding_diagnostics["error_kind"]
            if semantic_state in {"error", "timeout", "partial_failure"}:
                route_errors.append(f"semantic:{semantic_state}")

        fusion_mode = self._candidate_fusion_mode()
        try:
            fused = fuse_route_hits(
                route_hits,
                query=request.query,
                mode=fusion_mode,
                weights=self._code_route_weights(),
                limit=candidate_limit,
            )
        except Exception:
            fused = []
            route_errors.append("fusion:error")
        candidate_items, route_fill_count = self._candidate_route_fill(
            fused,
            route_hits,
            corpus,
            result_limit=result_limit,
        )
        literal_fallback_count = 0
        if not candidate_items:
            candidate_items = self._candidate_literal_fallback(corpus, result_limit=result_limit)
            literal_fallback_count = len(candidate_items)
        evidence, hydration_errors, clipped_count, omitted_count = self._hydrate_candidate_active(
            candidate_items,
            corpus,
            ctx,
            result_limit=result_limit,
            snippet_chars=snippet_chars,
            batch_chars=batch_chars,
            primary_focus_terms=request.scope.grep_terms,
            fallback_focus_terms=_extract_code_query_terms(request.query),
        )
        if not evidence and literal_fallback_count == 0:
            literal_items = self._candidate_literal_fallback(corpus, result_limit=result_limit)
            literal_fallback_count = len(literal_items)
            fallback_evidence, fallback_hydration_errors, fallback_clipped_count, fallback_omitted_count = (
                self._hydrate_candidate_active(
                    literal_items,
                    corpus,
                    ctx,
                    result_limit=result_limit,
                    snippet_chars=snippet_chars,
                    batch_chars=batch_chars,
                    primary_focus_terms=request.scope.grep_terms,
                    fallback_focus_terms=_extract_code_query_terms(request.query),
                )
            )
            evidence = fallback_evidence
            hydration_errors += fallback_hydration_errors
            clipped_count += fallback_clipped_count
            omitted_count += fallback_omitted_count

        route_count = sum(len(hits) for hits in route_hits.values())
        deduplicated_count = max(
            0,
            route_count
            - len(
                {
                    _candidate_canonical_uri(item)
                    for hits in route_hits.values()
                    for item in hits
                }
            ),
        )
        partial = (
            discovery.partial_failure
            or corpus.partial_failure
            or bool(route_errors)
            or hydration_errors > 0
            or time.monotonic() >= request_deadline
        )
        if time.monotonic() >= request_deadline and "candidate:timeout" not in route_errors:
            route_errors.append("candidate:timeout")
        if evidence:
            status = ACTIVE_STATUS_PARTIAL_FAILURE if partial else ACTIVE_STATUS_OK
        else:
            status = ACTIVE_STATUS_PARTIAL_FAILURE if partial else ACTIVE_STATUS_ZERO_RESULT
        evidence_diagnostics = self._candidate_evidence_diagnostic_counts(evidence)
        diagnostics = ActiveCodeSearchDiagnostics(
            candidate_limit,
            result_limit,
            snippet_chars,
            batch_chars,
            duration_ms=int(max(0.0, (time.monotonic() - search_started) * 1000)),
            scope_match_count=len(corpus.records),
            hydration_errors=hydration_errors,
            deduplicated_count=deduplicated_count,
            omitted_count=omitted_count,
            clipped_count=clipped_count,
            evidence_chars=evidence_diagnostics["evidence_chars"],
            line_span_count=evidence_diagnostics["line_span_count"],
            budget_clamps=clamps,
            routes=route_diagnostics,
            errors=tuple(route_errors),
            degradation="candidate_routes_partial" if partial and evidence else None,
            mode="candidate_local",
            workspace=workspace_diagnostics,
            discovery=discovery_diagnostics,
            corpus=corpus_diagnostics,
            embedding=embedding_diagnostics,
            fusion={
                "mode": fusion_mode,
                "fused_count": len(fused),
                "route_fill_count": route_fill_count,
                "literal_fallback_count": literal_fallback_count,
            },
            retention=retention,
        )
        return ActiveCodeSearchOutcome(status=status, evidence=tuple(evidence), diagnostics=diagnostics)

    def read_focused_snippet(
        self,
        *,
        uri: str,
        ctx: RequestContext,
        max_chars: int | None = None,
    ) -> CodeEvidence:
        """Hydrate one exact, already-imported code node as bounded evidence.

        This intentionally shares the active-search snapshot and hydrator.  It
        never resolves a physical path or performs a fallback lookup.
        """

        canonical_uri = self._focused_code_uri(uri, ctx)
        if max_chars is not None and (
            isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0
        ):
            raise _validation("max_chars", "must be a positive integer")
        if not bool(getattr(self._cfg, "code_retrieval_enabled", False)):
            raise ValueError("code retrieval is disabled")
        if self._code_retrieval_mode() == "candidate_local":
            # Candidate evidence intentionally has no cross-request URI
            # lookup state.  Do not scan/rechunk the workspace or touch the
            # durable focused-read stack to reconstruct it.
            raise FileNotFoundError("code snippet not found")

        # Validate the exact logical identity before constructing or hydrating
        # the authorized persisted snapshot.  No alternate name/path lookup is
        # permitted for a focused read.
        snapshot = self._get_snapshot(ctx)
        record = next(
            (
                candidate
                for candidate in snapshot.records
                if str(candidate.get("uri") or "") == canonical_uri
                and dict(candidate.get("metadata") or {}).get("resource_type") == "code"
            ),
            None,
        )
        if record is None:
            raise FileNotFoundError("code snippet not found")

        _, _, snippet_chars, batch_chars, _, _ = self._active_budget(
            CodeSearchBudget(
                snippet_chars=max_chars,
                batch_chars=max_chars,
            )
        )
        item = self._record_item(dict(record), 0.0, "focused")
        evidence, _hydration_errors, _clipped_count, _omitted_count = self._hydrate_active(
            [item],
            ctx,
            result_limit=1,
            snippet_chars=snippet_chars,
            batch_chars=batch_chars,
        )
        if not evidence:
            # A snapshot is stale or the backing node is no longer authorized,
            # active, and code-typed.  Do not retry a less exact lookup.
            raise FileNotFoundError("code snippet not found")
        return evidence[0]

    @staticmethod
    def _focused_code_uri(uri: object, ctx: RequestContext) -> str:
        """Validate the one accepted focused-read identifier shape."""

        if not isinstance(ctx, RequestContext) or not ctx.account_id:
            raise _validation("ctx.account_id", "account_id is required")
        if not isinstance(uri, str) or not uri or uri != uri.strip():
            raise _validation("uri", "must be an exact account-local code URI")
        prefix = f"ctx://{ctx.account_id}/resources/"
        if (
            not uri.startswith(prefix)
            or uri == prefix
            or uri.endswith("/")
            or "\x00" in uri
            or "\\" in uri
            or any(character in uri for character in "*?[]#%")
            or canonical_node_uri(uri) != uri
        ):
            raise _validation("uri", "must be an exact account-local code URI")
        suffix = uri[len(prefix) :]
        if any(segment in {"", ".", ".."} for segment in suffix.split("/")):
            raise _validation("uri", "must be an exact account-local code URI")
        return uri

    def _cfg_int(self, name: str, default: int) -> int:
        value = getattr(self._cfg, name, default)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    def _code_retrieval_mode(self) -> str:
        """Return the config-owned static retrieval mode with a safe default."""

        raw = str(getattr(self._cfg, "code_retrieval_mode", "full") or "full")
        mode = raw.strip().lower()
        return mode if mode in {"full", "candidate_local"} else "full"

    @staticmethod
    def _validate_active_request(
        request: ActiveCodeSearchRequest | Mapping[str, Any],
        ctx: RequestContext,
    ) -> ActiveCodeSearchRequest:
        """Normalize and validate one public active-search request once."""

        if not isinstance(request, ActiveCodeSearchRequest):
            if isinstance(request, Mapping):
                request = ActiveCodeSearchRequest(**dict(request))
            else:
                raise _validation("request", "must be ActiveCodeSearchRequest")
        if (
            request.budget.result_limit is not None
            and request.budget.candidate_limit is not None
            and request.budget.result_limit > request.budget.candidate_limit
        ):
            raise _validation("budget.result_limit", "cannot exceed candidate_limit")
        if (
            request.budget.snippet_chars is not None
            and request.budget.batch_chars is not None
            and request.budget.snippet_chars > request.budget.batch_chars
        ):
            raise _validation("budget.snippet_chars", "cannot exceed batch_chars")
        if not isinstance(ctx, RequestContext) or not ctx.account_id:
            raise _validation("ctx.account_id", "account_id is required")
        return request

    def _effective_route_deadline_ms(
        self,
        budget: CodeSearchBudget,
        clamps: dict[str, dict[str, int]],
    ) -> int:
        configured = self._cfg_int("code_retrieval_route_deadline_ms", 500)
        requested = budget.route_deadline_ms
        effective = configured if requested is None else min(requested, configured)
        if requested is not None and requested != effective:
            clamps["route_deadline_ms"] = {
                "requested": requested,
                "effective": effective,
            }
        return effective

    def _candidate_request_deadline_ms(
        self,
        budget: CodeSearchBudget,
        clamps: dict[str, dict[str, int]],
    ) -> int:
        """Apply the candidate request deadline without inheriting full's 500ms cap."""

        configured = min(
            360_000,
            self._cfg_int("code_retrieval_candidate_request_deadline_ms", 360_000),
        )
        requested = budget.route_deadline_ms
        effective = configured if requested is None else min(requested, configured)
        if requested is not None and requested != effective:
            clamps["route_deadline_ms"] = {"requested": requested, "effective": effective}
        return effective

    def _candidate_embedding_deadline(
        self,
        *,
        stage_started: float,
        request_deadline: float,
    ) -> float:
        """Bound semantic work from its own stage start and the request cap.

        Discovery, canonicalization, and local lexical routes can legitimately
        consume time before direct embedding starts.  They must not silently
        shorten the configured embedding allowance; the enclosing request
        deadline remains the final safety boundary.
        """

        configured_ms = min(
            330_000,
            self._cfg_int("code_retrieval_candidate_embedding_deadline_ms", 330_000),
        )
        return min(request_deadline, stage_started + configured_ms / 1000.0)

    def _candidate_workspace_error_outcome(
        self,
        *,
        candidate_limit: int,
        result_limit: int,
        snippet_chars: int,
        batch_chars: int,
        clamps: Mapping[str, Mapping[str, int]],
        search_started: float,
        selection: str,
        reason: str,
        retention: Mapping[str, Any],
        available_root_count: int = 0,
    ) -> ActiveCodeSearchOutcome:
        """Render a path-free workspace business failure before filesystem work."""

        routes = {
            route: {"state": "not_run", "count": 0, "duration_ms": 0}
            for route in _ROUTE_WEIGHTS
        }
        diagnostics = ActiveCodeSearchDiagnostics(
            candidate_limit,
            result_limit,
            snippet_chars,
            batch_chars,
            duration_ms=int(max(0.0, (time.monotonic() - search_started) * 1000)),
            budget_clamps=clamps,
            routes=routes,
            errors=(reason,),
            mode="candidate_local",
            reason=reason,
            workspace={
                "state": "error",
                "selection": selection,
                "available_root_count": available_root_count,
            },
            embedding=self._empty_candidate_embedding_diagnostics(),
            fusion={"mode": self._candidate_fusion_mode(), "fused_count": 0, "route_fill_count": 0, "literal_fallback_count": 0},
            retention=retention,
        )
        return ActiveCodeSearchOutcome(status=ACTIVE_STATUS_ERROR, diagnostics=diagnostics, error=reason)

    def _candidate_stage_failure_outcome(
        self,
        *,
        candidate_limit: int,
        result_limit: int,
        snippet_chars: int,
        batch_chars: int,
        clamps: Mapping[str, Mapping[str, int]],
        search_started: float,
        workspace: Mapping[str, Any],
        discovery: Mapping[str, Any],
        retention: Mapping[str, Any],
        error: str,
        corpus: Mapping[str, Any] | None = None,
    ) -> ActiveCodeSearchOutcome:
        """Return a bounded candidate degradation without invoking full mode."""

        routes = {
            route: {"state": "not_run", "count": 0, "duration_ms": 0}
            for route in _ROUTE_WEIGHTS
        }
        diagnostics = ActiveCodeSearchDiagnostics(
            candidate_limit,
            result_limit,
            snippet_chars,
            batch_chars,
            duration_ms=int(max(0.0, (time.monotonic() - search_started) * 1000)),
            budget_clamps=clamps,
            routes=routes,
            errors=(error,),
            degradation="candidate_stage_failure",
            mode="candidate_local",
            workspace=workspace,
            discovery=discovery,
            corpus=dict(corpus or {}),
            embedding=self._empty_candidate_embedding_diagnostics(),
            fusion={"mode": self._candidate_fusion_mode(), "fused_count": 0, "route_fill_count": 0, "literal_fallback_count": 0},
            retention=retention,
        )
        return ActiveCodeSearchOutcome(status=ACTIVE_STATUS_PARTIAL_FAILURE, diagnostics=diagnostics)

    def _candidate_fusion_mode(self) -> str:
        """Return the existing config-owned fusion mode for the local corpus."""

        return str(getattr(self._cfg, "code_retrieval_fusion_mode", "weighted_rrf") or "weighted_rrf")

    def _empty_candidate_embedding_diagnostics(self, *, state: str = "zero_result") -> dict[str, Any]:
        return {
            "state": state,
            "logical_inputs": 0,
            "provider_calls": 0,
            "batches": 0,
            "estimated_tokens": 0,
            "embedded_chars": 0,
            "semantic_limit": min(
                16,
                self._cfg_int("code_retrieval_candidate_semantic_limit", 16),
            ),
            "batch_text_limit": min(
                8,
                self._cfg_int("code_retrieval_candidate_embedding_batch_texts", 8),
            ),
            "batch_token_limit": min(
                8000,
                self._cfg_int("code_retrieval_candidate_embedding_batch_tokens", 8000),
            ),
            "embedding_char_limit": min(
                4096,
                self._cfg_int("code_retrieval_candidate_embedding_chars", 4096),
            ),
            "batch_concurrency": 1,
            "cache_policy": "request_uncached",
            "deadline_ms": min(
                330_000,
                self._cfg_int("code_retrieval_candidate_embedding_deadline_ms", 330_000),
            ),
            "deadline_state": "not_started",
        }

    def _candidate_semantic_records(
        self,
        corpus: CandidateCodeCorpus,
        route_hits: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> tuple[dict[str, Any], ...]:
        """Pick the bounded local semantic union without touching full records."""

        by_uri = {
            _candidate_canonical_uri(dict(record)): dict(record)
            for record in corpus.records
        }
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()

        def append_uri(uri: str) -> None:
            if uri in seen:
                return
            record = by_uri.get(uri)
            if record is None or not str(record.get("content") or ""):
                return
            seen.add(uri)
            selected.append(record)

        # Interleave routes so BM25 cannot exhaust the semantic candidate cap.
        channels = [route_hits.get(route, ()) for route in ("symbolic", "bm25")]
        for index in range(max((len(items) for items in channels), default=0)):
            for items in channels:
                if index < len(items):
                    append_uri(_candidate_canonical_uri(dict(items[index])))
        for record in sorted(
            by_uri.values(),
            key=lambda value: (
                corpus.file_ranks.get(_candidate_canonical_uri(value), math.inf),
                code_source_key(value),
            ),
        ):
            append_uri(_candidate_canonical_uri(record))
        semantic_limit = min(
            16,
            self._cfg_int("code_retrieval_candidate_semantic_limit", 16),
        )
        return tuple(selected[:semantic_limit])

    @staticmethod
    def _candidate_estimated_tokens(text: str) -> int:
        """Use the frozen CJK/ASCII approximation without provider accounting claims."""

        if not text:
            return 0
        cjk_chars = sum(1 for character in text if "\u3400" <= character <= "\u9fff")
        other_chars = len(text) - cjk_chars
        return max(1, math.ceil(cjk_chars / 1.5 + other_chars / 4.0))

    def _candidate_embedding_copy(
        self,
        value: object,
        *,
        char_limit: int,
        token_limit: int,
    ) -> str:
        """Bound one outbound embedding copy while leaving evidence text intact."""

        text = str(value or "")[:char_limit]
        if self._candidate_estimated_tokens(text) <= token_limit:
            return text
        low = 0
        high = len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self._candidate_estimated_tokens(text[:middle]) <= token_limit:
                low = middle
            else:
                high = middle - 1
        return text[:low]

    @staticmethod
    def _candidate_valid_vector(
        value: object,
        *,
        dimension: int | None,
    ) -> list[float]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError("embedding vector is invalid")
        vector = [float(component) for component in value]
        if not vector or any(not math.isfinite(component) for component in vector):
            raise ValueError("embedding vector is invalid")
        if dimension is not None and len(vector) != dimension:
            raise ValueError("embedding vector dimension mismatch")
        return vector

    @staticmethod
    def _candidate_cosine(left: Sequence[float], right: Sequence[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        value = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
        return max(-1.0, min(1.0, value)) if math.isfinite(value) else 0.0

    @staticmethod
    def _candidate_embedding_call_timeout_seconds(
        *,
        embedding_deadline: float,
        request_deadline: float,
    ) -> float | None:
        """Return the remaining total deadline for exactly one provider call."""

        remaining = min(embedding_deadline, request_deadline) - time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0.0:
            return None
        return remaining

    def _search_candidate_semantic(
        self,
        *,
        query: str,
        ctx: RequestContext,
        records: Sequence[Mapping[str, Any]],
        embedding_deadline: float,
        request_deadline: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Embed one query and bounded local snippets with no result cache."""

        diagnostics = self._empty_candidate_embedding_diagnostics()
        char_limit = int(diagnostics["embedding_char_limit"])
        token_limit = int(diagnostics["batch_token_limit"])
        batch_text_limit = int(diagnostics["batch_text_limit"])
        effective_records = tuple(records[: int(diagnostics["semantic_limit"])])
        if not effective_records:
            return [], diagnostics
        embedder = self._candidate_embedder
        if embedder is None:
            return [], {**diagnostics, "state": "error", "error_kind": "embedder_unavailable"}
        embed_with_timeout = getattr(embedder, "embed_texts_with_timeout", None)
        if not callable(embed_with_timeout):
            return [], {
                **diagnostics,
                "state": "error",
                "error_kind": "embedder_deadline_unsupported",
            }

        query_copy = self._candidate_embedding_copy(
            query,
            char_limit=char_limit,
            token_limit=token_limit,
        )
        metrics = {
            "logical_inputs": 0,
            "provider_calls": 0,
            "batches": 0,
            "estimated_tokens": 0,
            "embedded_chars": 0,
        }
        query_timeout = self._candidate_embedding_call_timeout_seconds(
            embedding_deadline=embedding_deadline,
            request_deadline=request_deadline,
        )
        if query_timeout is None:
            return [], {
                **diagnostics,
                **metrics,
                "state": "timeout",
                "deadline_state": "timeout",
            }
        metrics.update(
            {
                "logical_inputs": 1,
                "provider_calls": 1,
                "estimated_tokens": self._candidate_estimated_tokens(query_copy),
                "embedded_chars": len(query_copy),
            }
        )
        try:
            query_vectors = embed_with_timeout(
                [query_copy],
                timeout_seconds=query_timeout,
            )
            if not isinstance(query_vectors, Sequence) or len(query_vectors) != 1:
                raise ValueError("query embedding response is invalid")
            query_vector = self._candidate_valid_vector(query_vectors[0], dimension=None)
        except TimeoutError:
            return [], {
                **diagnostics,
                **metrics,
                "state": "timeout",
                "error_kind": "embedding_timeout",
                "deadline_state": "timeout",
            }
        except Exception as exc:
            logger.warning(
                "candidate semantic query embedding failed account=%s error=%s",
                ctx.account_id,
                type(exc).__name__,
            )
            if self._candidate_embedding_call_timeout_seconds(
                embedding_deadline=embedding_deadline,
                request_deadline=request_deadline,
            ) is None:
                return [], {
                    **diagnostics,
                    **metrics,
                    "state": "timeout",
                    "error_kind": "embedding_timeout",
                    "deadline_state": "timeout",
                }
            return [], {
                **diagnostics,
                **metrics,
                "state": "error",
                "error_kind": "query_embedding_error",
            }
        if time.monotonic() >= min(embedding_deadline, request_deadline):
            return [], {
                **diagnostics,
                **metrics,
                "state": "timeout",
                "deadline_state": "timeout",
            }

        pending: list[tuple[dict[str, Any], str, int]] = []
        for raw_record in effective_records:
            record = dict(raw_record)
            copy = self._candidate_embedding_copy(
                record.get("content"),
                char_limit=char_limit,
                token_limit=token_limit,
            )
            if not copy:
                continue
            pending.append((record, copy, self._candidate_estimated_tokens(copy)))
        batches: list[list[tuple[dict[str, Any], str, int]]] = []
        current: list[tuple[dict[str, Any], str, int]] = []
        current_tokens = 0
        for item in pending:
            if current and (
                len(current) >= batch_text_limit
                or current_tokens + item[2] > token_limit
            ):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(item)
            current_tokens += item[2]
        if current:
            batches.append(current)

        document_vectors: list[tuple[dict[str, Any], list[float]]] = []
        state = "ok"
        error_kind: str | None = None
        for batch in batches:
            batch_timeout = self._candidate_embedding_call_timeout_seconds(
                embedding_deadline=embedding_deadline,
                request_deadline=request_deadline,
            )
            if batch_timeout is None:
                state = "partial_failure" if document_vectors else "timeout"
                error_kind = "embedding_timeout"
                break
            texts = [copy for _record, copy, _tokens in batch]
            metrics["logical_inputs"] += len(texts)
            metrics["provider_calls"] += 1
            metrics["batches"] += 1
            metrics["estimated_tokens"] += sum(tokens for _record, _copy, tokens in batch)
            metrics["embedded_chars"] += sum(len(copy) for _record, copy, _tokens in batch)
            try:
                response_vectors = embed_with_timeout(
                    texts,
                    timeout_seconds=batch_timeout,
                )
                if not isinstance(response_vectors, Sequence) or len(response_vectors) != len(batch):
                    raise ValueError("document embedding response is invalid")
                # A failed vector anywhere in a provider response invalidates
                # the entire in-flight batch.  Only fully validated prior
                # batches may contribute semantic hits to a partial result.
                validated_batch: list[tuple[dict[str, Any], list[float]]] = []
                for (record, _copy, _tokens), raw_vector in zip(batch, response_vectors):
                    validated_batch.append(
                        (
                            record,
                            self._candidate_valid_vector(raw_vector, dimension=len(query_vector)),
                        )
                    )
                document_vectors.extend(validated_batch)
            except TimeoutError:
                state = "partial_failure" if document_vectors else "timeout"
                error_kind = "embedding_timeout"
                break
            except Exception as exc:
                logger.warning(
                    "candidate semantic document embedding failed account=%s error=%s",
                    ctx.account_id,
                    type(exc).__name__,
                )
                if self._candidate_embedding_call_timeout_seconds(
                    embedding_deadline=embedding_deadline,
                    request_deadline=request_deadline,
                ) is None:
                    state = "partial_failure" if document_vectors else "timeout"
                    error_kind = "embedding_timeout"
                else:
                    state = "partial_failure"
                    error_kind = "document_embedding_error"
                break
            if time.monotonic() >= min(embedding_deadline, request_deadline):
                state = "partial_failure"
                error_kind = "embedding_timeout"
                break

        hits = [
            self._record_item(
                record,
                self._candidate_cosine(query_vector, vector),
                "semantic",
                raw_score=self._candidate_cosine(query_vector, vector),
            )
            for record, vector in document_vectors
        ]
        hits.sort(key=lambda item: (-float(item.get("score") or 0.0), code_source_key(item)))
        if not hits and state == "ok":
            state = "zero_result"
        result = {
            **diagnostics,
            **metrics,
            "state": state,
            "deadline_state": "timeout" if error_kind == "embedding_timeout" else "within",
        }
        if error_kind is not None:
            result["error_kind"] = error_kind
        return hits, result

    @staticmethod
    def _invoke_route_searcher(
        searcher,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int,
        deadline: float,
        extra_kwargs: Mapping[str, Any] | None = None,
    ):
        """Call route implementations while preserving older test doubles."""
        kwargs: dict[str, Any] = {"top_k": top_k}
        extras = dict(extra_kwargs or {})
        try:
            parameters: Mapping[str, inspect.Parameter] = inspect.signature(searcher).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs or "deadline" in parameters:
            kwargs["deadline"] = deadline
        for name, value in extras.items():
            if accepts_kwargs or name in parameters:
                kwargs[name] = value
        return searcher(query, ctx, **kwargs)

    def _active_budget(
        self,
        budget: CodeSearchBudget,
    ) -> tuple[int, int, int, int, int, dict[str, dict[str, int]]]:
        caps = {
            "candidate_limit": self._cfg_int("code_retrieval_candidate_limit", 50),
            "result_limit": self._cfg_int("code_retrieval_result_limit", _DEFAULT_TOP_K),
            "snippet_chars": self._cfg_int("code_retrieval_snippet_chars", 1200),
            "batch_chars": self._cfg_int("code_retrieval_batch_chars", 8000),
            "graph_neighbor_limit": self._cfg_int("code_retrieval_graph_neighbor_limit", 24),
        }
        values: dict[str, int] = {}
        clamps: dict[str, dict[str, int]] = {}
        for name, cap in caps.items():
            requested = getattr(budget, name)
            effective = cap if requested is None else min(requested, cap)
            values[name] = max(1, effective)
            if requested is not None and requested != effective:
                clamps[name] = {"requested": requested, "effective": effective}
        if values["result_limit"] > values["candidate_limit"]:
            previous = values["result_limit"]
            values["result_limit"] = values["candidate_limit"]
            clamps.setdefault(
                "result_limit",
                {"requested": previous, "effective": values["result_limit"]},
            )
        if values["batch_chars"] < values["snippet_chars"]:
            requested_snippet = values["snippet_chars"]
            values["snippet_chars"] = values["batch_chars"]
            clamps.setdefault(
                "snippet_chars",
                {"requested": requested_snippet, "effective": values["snippet_chars"]},
            )
        return (
            values["candidate_limit"],
            values["result_limit"],
            values["snippet_chars"],
            values["batch_chars"],
            values["graph_neighbor_limit"],
            clamps,
        )

    def _code_route_weights(self) -> dict[str, float]:
        names = {
            "semantic": "code_retrieval_semantic_weight",
            "bm25": "code_retrieval_bm25_weight",
            "symbolic": "code_retrieval_symbolic_weight",
            "graph": "code_retrieval_graph_weight",
        }
        weights: dict[str, float] = {}
        for route, name in names.items():
            raw = getattr(self._cfg, name, _ROUTE_WEIGHTS[route])
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = _ROUTE_WEIGHTS[route]
            weights[route] = max(0.0, value) if math.isfinite(value) else 0.0
        return weights

    @staticmethod
    def _scope_source_text(record: Mapping[str, Any]) -> str:
        # Apply literal filters to source only; metadata remains a ranking signal.
        return str(record.get("content") or "").lower()

    @classmethod
    def _scope_full_records(
        cls, records: Sequence[Mapping[str, Any]], scope: CodeSearchScope,
    ) -> tuple[Mapping[str, Any], ...]:
        if not scope.grep_terms:
            return tuple(record for record in records if cls._record_matches_scope(record, scope))
        structural_scope = CodeSearchScope(
            root_slugs=scope.root_slugs, paths=scope.paths, glob_patterns=scope.glob_patterns,
        )
        terms = tuple(term.lower() for term in scope.grep_terms)
        groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
        for record in records:
            if not cls._record_matches_scope(record, structural_scope):
                continue
            metadata = dict(record.get("metadata") or {})
            root = str(metadata.get("resource_root_slug") or "")
            path = str(metadata.get("relative_path") or "").replace("\\", "/")
            # Group only known file identities within this account snapshot.
            key = ("file", root, path, str(metadata.get("root_fingerprint") or ""),
                   str(metadata.get("file_key") or "")) if root and path else (
                       "record", str(record.get("uri") or ""), str(id(record)),
                   )
            groups.setdefault(key, []).append(record)
        selected: list[Mapping[str, Any]] = []
        for group in groups.values():
            sources = {r["_file_source"] for r in group if isinstance(r.get("_file_source"), str)}
            if not sources:
                raise _validation("scope.source_snapshot", "full source snapshot missing; refresh code resources")
            if len(sources) > 1:
                raise _validation("scope.source_snapshot_conflict", "conflicting file snapshots; refresh code resources")
            source = sources.pop().lower()
            if not all(term in source for term in terms):
                continue
            for record in group:
                text = cls._scope_source_text(record)
                if any(term in text for term in terms):
                    # Keep only source-anchored chunks; never mutate the snapshot.
                    selected.append({**record, "_file_grep_terms": terms})
        return tuple(selected)

    @staticmethod
    def _record_matches_scope(record: Mapping[str, Any], scope: CodeSearchScope) -> bool:
        metadata = dict(record.get("metadata") or {})
        if metadata.get("resource_type") != "code":
            return False
        root = str(metadata.get("resource_root_slug") or "")
        path = str(metadata.get("relative_path") or "").replace("\\", "/")
        if scope.root_slugs and root not in scope.root_slugs:
            return False
        if scope.paths and not any(path == value or path.startswith(value.rstrip("/") + "/") for value in scope.paths):
            return False
        if scope.glob_patterns and not any(fnmatch.fnmatchcase(path, pattern) for pattern in scope.glob_patterns):
            return False
        if not scope.grep_terms:
            return True
        searchable = CodeRetrievalService._scope_source_text(record)
        terms = tuple(term.lower() for term in scope.grep_terms)
        verified_file_terms = record.get("_file_grep_terms", ())
        if all(term in verified_file_terms for term in terms):
            # Require a chunk-level anchor to exclude unrelated sibling chunks.
            return any(term in searchable for term in terms)
        return all(term in searchable for term in terms)

    def _item_matches_scope(
        self,
        item: Mapping[str, Any],
        scope: CodeSearchScope,
        scoped_uris: set[str],
        scoped_records: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> bool:
        uri = _candidate_canonical_uri(dict(item))
        if uri not in scoped_uris:
            return False
        source_record = dict(scoped_records.get(uri) or {}) if scoped_records else {}
        source_record.setdefault("metadata", dict(item.get("metadata") or {}))
        source_record.setdefault("content", item.get("content"))
        return self._record_matches_scope(source_record, scope)

    def _hydrate_active(
        self,
        items: list[dict[str, Any]],
        ctx: RequestContext,
        *,
        result_limit: int,
        snippet_chars: int,
        batch_chars: int,
        primary_focus_terms: Sequence[str] = (),
        fallback_focus_terms: Sequence[str] = (),
    ) -> tuple[list[CodeEvidence], int, int, int]:
        evidence: list[CodeEvidence] = []
        hydration_errors = 0
        clipped_count = 0
        omitted_count = 0
        body_used = 0
        for item in items:
            if len(evidence) >= result_limit:
                omitted_count += 1
                continue
            uri = _candidate_canonical_uri(item)
            try:
                node = self._fs.read_node(uri, ctx)
            except Exception:
                hydration_errors += 1
                continue
            node_uri = str(getattr(node, "uri", ""))
            if (
                not node_uri.startswith(f"ctx://{ctx.account_id}/")
                or canonical_node_uri(node_uri) != uri
            ):
                hydration_errors += 1
                continue
            if not _is_code_node(node):
                hydration_errors += 1
                continue
            metadata = {**dict(item.get("metadata") or {}), **dict(node.metadata or {})}
            locator_path = _code_evidence_path(metadata)
            safe_relative_path = _safe_code_relative_path(metadata)
            safe_root_slug = str(metadata.get("resource_root_slug") or "").strip()
            if not _SAFE_ROOT_SLUG.fullmatch(safe_root_slug) or safe_root_slug in {".", ".."}:
                safe_root_slug = ""
            body = str(node.content or node.overview or node.abstract or "")
            remaining = max(0, batch_chars - body_used)
            allowed = min(snippet_chars, remaining)
            body, start_line, end_line, coverage = _code_excerpt(
                body, metadata, allowed, primary_focus_terms, fallback_focus_terms,
            )
            clipped_count += int(coverage["truncated"])
            if not body:
                omitted_count += 1
                continue
            locator = locator_path
            if start_line is not None and end_line is not None:
                locator = f"{locator_path}:{start_line}-{end_line}"
            body_used += len(body)
            graph_context = tuple(
                dict(value)
                for value in (item.get("metadata", {}).get("_graph_context") or [])
                if isinstance(value, Mapping)
            )
            evidence.append(
                CodeEvidence(
                    uri=uri,
                    identity=uri,
                    locator=locator,
                    relative_path=safe_relative_path,
                    root_slug=safe_root_slug,
                    start_line=start_line,
                    end_line=end_line,
                    symbol=str(metadata.get("symbol") or ""),
                    language=str(metadata.get("language") or ""),
                    symbol_kind=str(metadata.get("symbol_kind") or ""),
                    signature=str(metadata.get("signature") or ""),
                    excerpt=body,
                    coverage=coverage,
                    graph_context=graph_context,
                    fusion_score=float(item.get("score") or 0.0),
                    route_ranks=dict(metadata.get("route_ranks") or {}),
                    route_scores=dict(metadata.get("route_scores") or {}),
                    retrieval_source=str(metadata.get("retrieval_source") or ""),
                    score_kind=str(
                        metadata.get("score_kind")
                        or metadata.get("fusion_score_kind")
                        or ""
                    ),
                )
            )
        return evidence, hydration_errors, clipped_count, omitted_count

    def _candidate_route_fill(
        self,
        fused: Sequence[Mapping[str, Any]],
        route_hits: Mapping[str, Sequence[Mapping[str, Any]]],
        corpus: CandidateCodeCorpus,
        *,
        result_limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Fill a local fusion underflow from already-ranked local routes only."""

        items = [dict(item) for item in fused]
        seen = {_candidate_canonical_uri(item) for item in items}
        filled = 0
        if len(items) >= result_limit:
            return items, filled
        for route in ("semantic", "bm25", "symbolic", "graph"):
            for rank, raw_item in enumerate(route_hits.get(route, ()), start=1):
                if len(items) >= result_limit:
                    return items, filled
                item = dict(raw_item)
                uri = _candidate_canonical_uri(item)
                if not uri or uri in seen:
                    continue
                node = corpus.nodes_by_uri.get(uri)
                if node is None or not str(node.content or node.overview or node.abstract or ""):
                    continue
                metadata = {**dict(item.get("metadata") or {}), **dict(node.metadata or {})}
                metadata.update(
                    {
                        "retrieval_source": "candidate_local",
                        "score_kind": "candidate_route_fill",
                        "fusion_score_kind": "candidate_route_fill",
                        "route_ranks": {route: rank},
                        "route_scores": {
                            route: {
                                "raw": float(item.get("raw_score") or item.get("score") or 0.0),
                                "normalized": float(item.get("score") or 0.0),
                            }
                        },
                    }
                )
                items.append(
                    {
                        **item,
                        "uri": uri,
                        "metadata": metadata,
                    }
                )
                seen.add(uri)
                filled += 1
        return items, filled

    def _candidate_literal_fallback(
        self,
        corpus: CandidateCodeCorpus,
        *,
        result_limit: int,
    ) -> list[dict[str, Any]]:
        """Return only canonical chunks containing discovery-confirmed literal lines."""

        records_by_uri = {
            _candidate_canonical_uri(dict(record)): dict(record)
            for record in corpus.records
        }
        items: list[dict[str, Any]] = []
        for uri in corpus.literal_uris:
            if len(items) >= result_limit:
                break
            canonical_uri = canonical_node_uri(uri)
            node = corpus.nodes_by_uri.get(canonical_uri)
            record = records_by_uri.get(canonical_uri)
            if node is None or record is None or not str(node.content or node.overview or node.abstract or ""):
                continue
            metadata = {
                **dict(record.get("metadata") or {}),
                **dict(node.metadata or {}),
                "retrieval_source": "candidate_local",
                "score_kind": "candidate_literal_fallback",
                "fusion_score_kind": "candidate_literal_fallback",
                "route_ranks": {},
                "route_scores": {},
            }
            items.append(
                {
                    "uri": canonical_uri,
                    "score": 0.0,
                    "raw_score": 0.0,
                    "category": "code",
                    "metadata": metadata,
                }
            )
        return items

    def _hydrate_candidate_active(
        self,
        items: Sequence[Mapping[str, Any]],
        corpus: CandidateCodeCorpus,
        ctx: RequestContext,
        *,
        result_limit: int,
        snippet_chars: int,
        batch_chars: int,
        primary_focus_terms: Sequence[str],
        fallback_focus_terms: Sequence[str],
    ) -> tuple[list[CodeEvidence], int, int, int]:
        """Render transient evidence from request-local canonical nodes only."""

        evidence: list[CodeEvidence] = []
        hydration_errors = 0
        clipped_count = 0
        omitted_count = 0
        body_used = 0
        for raw_item in items:
            if len(evidence) >= result_limit:
                omitted_count += 1
                continue
            item = dict(raw_item)
            uri = _candidate_canonical_uri(item)
            node = corpus.nodes_by_uri.get(uri)
            node_uri = str(getattr(node, "uri", "")) if node is not None else ""
            if (
                node is None
                or not node_uri.startswith(f"ctx://{ctx.account_id}/")
                or canonical_node_uri(node_uri) != uri
                or not _is_code_node(node)
            ):
                hydration_errors += 1
                continue
            metadata = {**dict(item.get("metadata") or {}), **dict(node.metadata or {})}
            locator_path = _code_evidence_path(metadata)
            safe_relative_path = _safe_code_relative_path(metadata)
            safe_root_slug = str(metadata.get("resource_root_slug") or "").strip()
            if not _SAFE_ROOT_SLUG.fullmatch(safe_root_slug) or safe_root_slug in {".", ".."}:
                safe_root_slug = ""
            body = str(node.content or node.overview or node.abstract or "")
            remaining = max(0, batch_chars - body_used)
            allowed = min(snippet_chars, remaining)
            body, start_line, end_line, coverage = _code_excerpt(
                body, metadata, allowed, primary_focus_terms, fallback_focus_terms,
            )
            clipped_count += int(coverage["truncated"])
            if not body:
                omitted_count += 1
                continue
            locator = locator_path
            if start_line is not None and end_line is not None:
                locator = f"{locator_path}:{start_line}-{end_line}"
            body_used += len(body)
            graph_context = tuple(
                dict(value)
                for value in (metadata.get("_graph_context") or [])
                if isinstance(value, Mapping)
            )
            evidence.append(
                CodeEvidence(
                    uri=uri,
                    identity=uri,
                    locator=locator,
                    relative_path=safe_relative_path,
                    root_slug=safe_root_slug,
                    start_line=start_line,
                    end_line=end_line,
                    symbol=str(metadata.get("symbol") or ""),
                    language=str(metadata.get("language") or ""),
                    symbol_kind=str(metadata.get("symbol_kind") or ""),
                    signature=str(metadata.get("signature") or ""),
                    excerpt=body,
                    coverage=coverage,
                    graph_context=graph_context,
                    fusion_score=float(item.get("score") or 0.0),
                    route_ranks=dict(metadata.get("route_ranks") or {}),
                    route_scores=dict(metadata.get("route_scores") or {}),
                    origin="candidate_local",
                    persisted=False,
                    focused_read_available=False,
                    retrieval_source="candidate_local",
                    score_kind=str(
                        metadata.get("score_kind")
                        or metadata.get("fusion_score_kind")
                        or "candidate_fusion"
                    ),
                )
            )
        return evidence, hydration_errors, clipped_count, omitted_count

    @staticmethod
    def _candidate_evidence_diagnostic_counts(
        evidence: Sequence[CodeEvidence],
    ) -> dict[str, int]:
        """Aggregate safe output counters from the final transient evidence only."""

        evidence_chars = 0
        line_span_count = 0
        for item in evidence:
            excerpt = item.excerpt
            if isinstance(excerpt, str):
                evidence_chars += len(excerpt)
            start_line = item.start_line
            end_line = item.end_line
            if (
                isinstance(start_line, int)
                and not isinstance(start_line, bool)
                and start_line > 0
                and isinstance(end_line, int)
                and not isinstance(end_line, bool)
                and end_line >= start_line
            ):
                line_span_count += 1
        return {
            "evidence_chars": evidence_chars,
            "line_span_count": line_span_count,
        }

    def _route_enabled(self, route: str) -> bool:
        helper = getattr(self._cfg, "code_retrieval_route_enabled", None)
        if callable(helper):
            return bool(helper(route))
        return bool(
            getattr(self._cfg, "code_retrieval_enabled", False)
            and getattr(self._cfg, f"code_retrieval_{route}_enabled", True)
        )

    def _search_semantic(
        self,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int,
        deadline: float | None = None,
    ) -> list[dict[str, Any]]:
        def search_memory():
            return self._read_api.search_memory(
                query=query,
                ctx=ctx,
                top_k=min(top_k, self._cfg_int("retrieval_max_top_k", top_k)),
                fill_content_for_top_k=0,
                context_type=ContextType.RESOURCE.value,
                resource_type="code",
                mode="QUICK",
            )

        result = (
            search_memory()
            if deadline is None
            else self._semantic_call_runner.run(search_memory, deadline=deadline)
        )
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("semantic route deadline exceeded")
        items: list[dict[str, Any]] = []
        for hit in getattr(result, "hits", ()) or ():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("semantic route deadline exceeded")
            metadata = dict(getattr(hit, "metadata", None) or {})
            if metadata.get("resource_type") != "code" and getattr(hit, "category", "") != "code":
                continue
            items.append(
                {
                    "uri": hit.uri,
                    "parent_uri": metadata.get("parent_uri"),
                    "score": max(float(hit.score or 0.0), 0.0),
                    "raw_score": max(float(hit.score or 0.0), 0.0),
                    "abstract": hit.abstract or "",
                    "overview": hit.overview or "",
                    "category": "code",
                    "metadata": {**metadata, "retrieval_source": "semantic"},
                }
            )
        return items

    def _search_bm25(
        self,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int,
        deadline: float | None = None,
        allowed_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        self._get_snapshot(ctx)
        if self._bm25 is None:
            raise RuntimeError("code BM25 index is unavailable")
        return self._search_bm25_index(
            query,
            ctx,
            bm25_index=self._bm25,
            top_k=top_k,
            deadline=deadline,
            allowed_ids=allowed_ids,
        )

    def _search_bm25_index(
        self,
        query: str,
        ctx: RequestContext,
        *,
        bm25_index: Bm25Index,
        top_k: int,
        deadline: float | None = None,
        allowed_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search one explicit code BM25 corpus without snapshot ownership."""

        filters = {
            "account_id": ctx.account_id,
            "context_type": ContextType.RESOURCE.value,
            "resource_type": "code",
        }
        search_code = getattr(bm25_index, "search_code", None)
        searcher = search_code if callable(search_code) else bm25_index.search
        search_kwargs: dict[str, Any] = {
            "top_k": max(top_k, top_k * 2),
            "filters": filters,
        }
        try:
            parameters: Mapping[str, inspect.Parameter] = inspect.signature(searcher).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_kwargs or "include_nonpositive_matches" in parameters:
            search_kwargs["include_nonpositive_matches"] = True
        supports_allowed_ids = accepts_kwargs or "allowed_ids" in parameters
        if allowed_ids is not None:
            # Active scoped BM25 must be restricted before ranking.  Falling
            # back to a legacy searcher and filtering after top-k would let an
            # out-of-scope record displace an authorized candidate.
            if not supports_allowed_ids:
                return []
            search_kwargs["allowed_ids"] = allowed_ids
        hits = searcher(_bm25_text(query), **search_kwargs)
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("bm25 route deadline exceeded")
            metadata = dict(hit.metadata or {})
            if getattr(hit, "fields", None):
                metadata.setdefault("_code_fields", dict(hit.fields))
            uri = canonical_node_uri(
                str(metadata.get("uri") or hit.id),
                parent_uri=str(metadata.get("parent_uri") or "") or None,
            )
            if uri in seen:
                continue
            seen.add(uri)
            raw_score = max(float(hit.score), 0.0)
            normalized = raw_score / (raw_score + 5.0) if raw_score > 0 else 0.0
            items.append(
                {
                    "uri": uri,
                    "parent_uri": metadata.get("parent_uri"),
                    "score": normalized,
                    "raw_score": raw_score,
                    "category": "code",
                    "metadata": {
                        **metadata,
                        "_bm25_raw_score": raw_score,
                        "retrieval_source": "bm25",
                    },
                }
            )
            if len(items) >= top_k:
                break
        return items

    def _search_symbolic(
        self,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int,
        deadline: float | None = None,
        records: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        query_terms = [term.lower() for term in _extract_code_query_terms(query)]
        if not query_terms:
            return []
        query_vocab = set(query_terms)
        scored: list[tuple[float, dict[str, Any]]] = []
        source_records = self._get_snapshot(ctx).records if records is None else records
        for raw_record in source_records:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("symbolic route deadline exceeded")
            record = dict(raw_record)
            metadata = dict(record["metadata"])
            symbol = str(metadata.get("symbol") or "").strip()
            symbol_lower = symbol.lower()
            path_lower = " ".join(
                str(metadata.get(key) or "")
                for key in ("relative_path", "source_extension", "language")
            ).lower()
            signature_lower = str(metadata.get("signature") or "").lower()
            symbol_kind = str(metadata.get("symbol_kind") or "code")
            score = 0.0
            for term in query_terms:
                if term == symbol_lower or term == symbol_lower.rsplit(".", 1)[-1]:
                    score += 3.0
                elif term in symbol_lower:
                    score += 1.5
                if term in _route_tokens(signature_lower):
                    score += 3.0
                elif term in signature_lower:
                    score += 1.0
                if term in path_lower:
                    score += 0.6
            if query_vocab & _SYMBOL_KIND_TERMS.get(symbol_kind, set()):
                score += 0.8
            if score > 0.0:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], code_source_key(item[1])))
        if not scored:
            return []

        min_score = min(score for score, _ in scored)
        max_score = max(score for score, _ in scored)
        score_span = max_score - min_score
        items: list[dict[str, Any]] = []
        for raw_score, record in scored[:top_k]:
            normalized = 1.0 if score_span <= 0.0 else (raw_score - min_score) / score_span
            items.append(
                self._record_item(
                    record,
                    normalized,
                    "symbolic",
                    raw_score=raw_score,
                )
            )
        return items

    def _search_graph(
        self,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_tokens = _route_tokens(query)
        if not query_tokens:
            return []
        relation_words = {
            "call",
            "calls",
            "caller",
            "callers",
            "called",
            "use",
            "used",
            "import",
            "imports",
            "depend",
            "depends",
            "extends",
            "contains",
            "reference",
            "references",
        }
        items: list[dict[str, Any]] = []
        for record in self._get_snapshot(ctx).records:
            graph = record["metadata"].get("graph")
            if not isinstance(graph, dict):
                continue
            graph_parts: list[str] = []
            for key in ("calls", "imports", "extends", "contains"):
                values = graph.get(key)
                if isinstance(values, list):
                    graph_parts.extend(str(value) for value in values)
            relations = graph.get("relations")
            if isinstance(relations, list):
                for relation in relations:
                    if isinstance(relation, dict):
                        graph_parts.extend(
                            str(relation.get(key) or "")
                            for key in ("type", "name", "target_symbol", "target_path")
                        )
            overlap = query_tokens & _route_tokens(*graph_parts)
            if not overlap:
                continue
            score = min(1.0, 0.4 + 0.1 * len(overlap))
            if query_tokens & relation_words:
                score = min(1.0, score + 0.15)
            items.append(self._record_item(record, score, "graph"))
        items.sort(key=lambda item: (-float(item["score"]), code_source_key(item)))
        return items[:top_k]

    def _search_graph_resolved(
        self,
        query: str,
        ctx: RequestContext,
        *,
        records: Sequence[Mapping[str, Any]],
        top_k: int,
        neighbor_limit: int,
        deadline: float,
        relation_store: object = _USE_SERVICE_RELATION_STORE,
    ) -> list[dict[str, Any]]:
        """Resolve persisted graph metadata and rank a bounded one-hop graph."""

        relation_source = (
            self._relation_store
            if relation_store is _USE_SERVICE_RELATION_STORE
            else relation_store
        )

        by_uri: dict[str, dict[str, Any]] = {}
        by_path: dict[tuple[str, str], list[str]] = {}
        by_symbol: dict[tuple[str, str], list[str]] = {}
        for record in records:
            uri = _candidate_canonical_uri(dict(record))
            metadata = dict(record.get("metadata") or {})
            by_uri[uri] = dict(record)
            path = str(metadata.get("relative_path") or "").replace("\\", "/")
            root = str(metadata.get("resource_root_slug") or "")
            symbol = str(metadata.get("symbol") or "").strip().lower()
            if path:
                by_path.setdefault((root, path), []).append(uri)
            if symbol:
                by_symbol.setdefault((root, symbol), []).append(uri)
        if not by_uri:
            return []

        # Select anchors before touching optional relation storage.  The
        # snapshot may be large, but graph nodes and relation-store calls must
        # remain bounded by the active request budget.
        query_terms = {term.lower() for term in _extract_code_query_terms(query)}
        anchors: list[tuple[float, str]] = []
        for uri, record in by_uri.items():
            metadata = dict(record.get("metadata") or {})
            terms = _route_tokens(
                metadata.get("symbol"),
                metadata.get("signature"),
                metadata.get("relative_path"),
                metadata.get("language"),
            )
            overlap = query_terms & terms
            if overlap:
                anchors.append((float(len(overlap)), uri))
        def node_key(uri):
            return code_source_key(by_uri[uri])

        anchors.sort(key=lambda value: (-value[0], node_key(value[1])))
        anchor_uris = [uri for _, uri in anchors[: max(1, top_k)]]
        if not anchor_uris:
            return []

        outgoing: dict[str, set[str]] = {uri: set() for uri in anchor_uris}
        context_by_edge: dict[tuple[str, str], dict[str, str]] = {}

        def resolve_target(source_uri: str, source_metadata: Mapping[str, Any], target: object) -> str | None:
            if not isinstance(target, str) or not target.strip():
                return None
            value = target.strip()
            if value.startswith("ctx://"):
                candidate = canonical_node_uri(value)
                return candidate if candidate in by_uri and candidate.startswith(f"ctx://{ctx.account_id}/") else None
            root = str(source_metadata.get("resource_root_slug") or "")
            if (
                value.startswith("/")
                or _URI_SCHEME_OR_DRIVE_PREFIX.match(value)
                or ".." in value.replace("\\", "/").split("/")
            ):
                return None
            normalized = value.replace("\\", "/").lstrip("./")
            path_candidates = by_path.get((root, normalized), [])
            if len(path_candidates) == 1:
                return path_candidates[0]
            aliases = by_symbol.get((root, value.lower()), [])
            if len(aliases) == 1:
                return aliases[0]
            return None

        def iter_metadata_edges(
            source_uri: str,
            source_metadata: Mapping[str, Any],
        ):
            graph = source_metadata.get("graph")
            if not isinstance(graph, Mapping):
                return
            inspected = 0
            for relation_type in ("calls", "imports", "extends", "contains"):
                values = graph.get(relation_type)
                if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                    continue
                for target in values:
                    if inspected >= neighbor_limit:
                        return
                    inspected += 1
                    target_uri = resolve_target(source_uri, source_metadata, target)
                    if target_uri and target_uri != source_uri:
                        yield target_uri, {
                            "relation_type": relation_type,
                            "target_symbol": str(
                                by_uri[target_uri].get("metadata", {}).get("symbol") or ""
                            ),
                            "target_locator": _code_evidence_path(
                                dict(by_uri[target_uri].get("metadata") or {})
                            ),
                        }
            relations = graph.get("relations")
            if isinstance(relations, Sequence) and not isinstance(relations, (str, bytes)):
                for relation in relations:
                    if inspected >= neighbor_limit:
                        return
                    inspected += 1
                    if not isinstance(relation, Mapping):
                        continue
                    target = (
                        relation.get("target_uri")
                        or relation.get("target_path")
                        or relation.get("target_symbol")
                    )
                    target_uri = resolve_target(source_uri, source_metadata, target)
                    if target_uri and target_uri != source_uri:
                        yield target_uri, {
                            "relation_type": str(
                                relation.get("type") or relation.get("relation_type") or "related"
                            ),
                            "target_symbol": str(
                                by_uri[target_uri].get("metadata", {}).get("symbol") or ""
                            ),
                            "target_locator": _code_evidence_path(
                                dict(by_uri[target_uri].get("metadata") or {})
                            ),
                        }

        for source_uri in anchor_uris:
            record = by_uri[source_uri]
            if time.monotonic() >= deadline:
                raise TimeoutError("graph route deadline exceeded")
            metadata = dict(record.get("metadata") or {})
            for target_uri, edge_context in iter_metadata_edges(source_uri, metadata):
                if len(outgoing[source_uri]) >= neighbor_limit:
                    break
                outgoing[source_uri].add(target_uri)
                context_by_edge[(source_uri, target_uri)] = edge_context
            if relation_source is not None and len(outgoing[source_uri]) < neighbor_limit:
                try:
                    get_one_hop = getattr(relation_source, "get_one_hop", None)
                    if callable(get_one_hop):
                        try:
                            edges = get_one_hop(source_uri, ctx, limit=neighbor_limit)
                        except TypeError as exc:
                            if "limit" not in str(exc):
                                raise
                            edges = get_one_hop(source_uri, ctx, neighbor_limit)
                    else:
                        get_edges = getattr(relation_source, "get_edges", None)
                        if not callable(get_edges):
                            raise RuntimeError("relation store has no graph reader")
                        edges = get_edges(source_uri, ctx)
                    edges = list(edges or ())[:neighbor_limit]
                except Exception as exc:
                    # Relation storage is optional.  Surface its failure to the
                    # caller so the graph route can be diagnosed independently.
                    raise RuntimeError(f"relation store unavailable: {type(exc).__name__}") from exc
                for edge in edges or ():
                    if len(outgoing[source_uri]) >= neighbor_limit:
                        break
                    target_uri = resolve_target(source_uri, metadata, getattr(edge, "to_uri", ""))
                    if target_uri and target_uri != source_uri:
                        outgoing[source_uri].add(target_uri)
                        context_by_edge[(source_uri, target_uri)] = {
                            "relation_type": str(getattr(edge, "relation_type", "related")),
                            "target_symbol": str(by_uri[target_uri].get("metadata", {}).get("symbol") or ""),
                            "target_locator": _code_evidence_path(dict(by_uri[target_uri].get("metadata") or {})),
                        }

        # Derive bounded incoming metadata edges from the already-authorized
        # snapshot without issuing unbounded reverse relation-store queries.
        anchor_set = set(anchor_uris)
        inbound_counts = {uri: 0 for uri in anchor_uris}
        for source_uri in sorted(by_uri, key=node_key):
            if source_uri in anchor_set:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("graph route deadline exceeded")
            metadata = dict(by_uri[source_uri].get("metadata") or {})
            for target_uri, edge_context in iter_metadata_edges(source_uri, metadata):
                if target_uri not in anchor_set or inbound_counts[target_uri] >= neighbor_limit:
                    continue
                source_targets = outgoing.setdefault(source_uri, set())
                if target_uri in source_targets or len(source_targets) >= neighbor_limit:
                    continue
                source_targets.add(target_uri)
                context_by_edge[(source_uri, target_uri)] = edge_context
                inbound_counts[target_uri] += 1

        incoming: dict[str, set[str]] = {uri: set() for uri in by_uri}
        for source_uri, targets in outgoing.items():
            for target_uri in targets:
                incoming[target_uri].add(source_uri)

        bounded_nodes: set[str] = set(anchor_uris)
        for anchor in anchor_uris:
            if time.monotonic() >= deadline:
                raise TimeoutError("graph route deadline exceeded")
            neighbors = sorted(outgoing.get(anchor, set()) | incoming.get(anchor, set()), key=node_key)
            bounded_nodes.update(neighbors[:neighbor_limit])
        ordered_nodes = sorted(bounded_nodes, key=node_key)
        adjacency = {
            uri: sorted((outgoing.get(uri, set()) | incoming.get(uri, set())) & bounded_nodes, key=node_key)
            for uri in ordered_nodes
        }
        # Fixed-iteration personalized PageRank over the bounded graph.
        restart = 0.15
        scores = {uri: (1.0 / len(anchor_uris) if uri in anchor_uris else 0.0) for uri in ordered_nodes}
        for _ in range(8):
            next_scores = {uri: 0.0 for uri in ordered_nodes}
            for uri, score in scores.items():
                neighbor_targets = adjacency.get(uri, [])
                if not neighbor_targets:
                    next_scores[uri] += (1.0 - restart) * score
                    continue
                share = (1.0 - restart) * score / len(neighbor_targets)
                for target in neighbor_targets:
                    next_scores[target] += share
            for anchor in anchor_uris:
                next_scores[anchor] += restart / len(anchor_uris)
            scores = next_scores

        anchor_rank = {
            uri: rank
            for rank, uri in enumerate(anchor_uris, start=1)
        }
        walk_scores = {
            uri: sum(
                1
                for neighbor in (outgoing.get(uri, set()) | incoming.get(uri, set()))
                if neighbor in anchor_rank
            )
            for uri in ordered_nodes
        }
        walk_rank = {
            uri: rank
            for rank, uri in enumerate(
                sorted(
                    (uri for uri, value in walk_scores.items() if value > 0),
                    key=lambda value: (-walk_scores[value], node_key(value)),
                ),
                start=1,
            )
        }
        ppr_rank = {
            uri: rank
            for rank, uri in enumerate(
                sorted(bounded_nodes, key=lambda value: (-scores[value], node_key(value))),
                start=1,
            )
        }
        inner_route_ranks = {
            "anchor": anchor_rank,
            "walk": walk_rank,
            "ppr": ppr_rank,
        }
        inner_scores = {
            uri: sum(
                1.0 / (_RRF_CONSTANT + ranks[uri])
                for ranks in inner_route_ranks.values()
                if uri in ranks
            )
            for uri in ordered_nodes
        }
        ranked = sorted(bounded_nodes, key=lambda uri: (-inner_scores[uri], node_key(uri)))[:top_k]
        items: list[dict[str, Any]] = []
        max_score = max((inner_scores[uri] for uri in ranked), default=0.0)
        for uri in ranked:
            record = by_uri[uri]
            metadata = dict(record.get("metadata") or {})
            graph_context: list[dict[str, str]] = []
            for target in adjacency.get(uri, ()):
                edge = context_by_edge.get((uri, target))
                if edge is None:
                    incoming_edge = context_by_edge.get((target, uri))
                    if incoming_edge is not None:
                        source_metadata = dict(by_uri[target].get("metadata") or {})
                        edge = {
                            **incoming_edge,
                            "target_symbol": str(source_metadata.get("symbol") or ""),
                            "target_locator": _code_evidence_path(source_metadata),
                        }
                if edge:
                    graph_context.append(dict(edge))
                if len(graph_context) >= neighbor_limit:
                    break
            metadata["_graph_context"] = graph_context
            score = inner_scores[uri] / max_score if max_score > 0.0 else 0.0
            items.append(self._record_item(record, score, "graph", raw_score=inner_scores[uri]))
            items[-1]["metadata"]["_graph_context"] = graph_context
        return items

    @staticmethod
    def _record_item(
        record: dict[str, Any],
        score: float,
        source: str,
        *,
        raw_score: float | None = None,
    ) -> dict[str, Any]:
        return {
            "uri": record["uri"],
            "parent_uri": record["metadata"].get("parent_uri"),
            "score": score,
            "raw_score": score if raw_score is None else raw_score,
            "category": "code",
            "metadata": {**record["metadata"], "retrieval_source": source},
        }

    def _get_snapshot(self, ctx: RequestContext) -> _CodeIndexSnapshot:
        with self._cache_lock:
            cached = self._cache.get(ctx.account_id)
            if cached is not None:
                return cached

        root_uri = f"ctx://{ctx.account_id}/resources/"
        child_uris = sorted(set(self._fs.list_children(root_uri, ctx)))
        records: list[dict[str, Any]] = []
        bm25_documents: list[Bm25Document] = []
        bm25_ids_by_uri: dict[str, list[str]] = {}
        read_error_count = 0
        for uri in child_uris:
            # ContextFS should already enforce this boundary, but keep the
            # snapshot defensive when an adapter or test double returns an
            # unexpected child URI.  Do not read a foreign logical URI just
            # to discard it after hydration.
            if not str(uri).startswith(root_uri):
                continue
            try:
                node = self._fs.read_node(uri, ctx)
            except Exception:
                read_error_count += 1
                continue
            node_uri = str(getattr(node, "uri", ""))
            if (
                not node_uri.startswith(root_uri)
                or canonical_node_uri(node_uri) != canonical_node_uri(str(uri))
            ):
                # Defensive account boundary for test doubles and storage
                # adapters that return an unexpected child URI.
                continue
            if not _is_code_node(node):
                continue
            metadata = dict(node.metadata or {})
            file_source = metadata.pop("file_source", None)
            index_metadata = {
                **metadata,
                "uri": node.uri,
                "account_id": ctx.account_id,
                "context_type": ContextType.RESOURCE.value,
                "resource_type": "code",
                "category": "code",
                "level": node.level,
                "parent_uri": node.parent_uri,
                "abstract": node.abstract,
                "has_content": bool(node.content),
            }
            record = {
                "_file_source": file_source,
                "uri": node.uri,
                "metadata": index_metadata,
                "abstract": node.abstract or "",
                "overview": node.overview or "",
                "content": node.content or "",
            }
            records.append(record)
            for index_record in build_index_records(node):
                bm25_document = build_code_bm25_document(index_record)
                if bm25_document is not None:
                    bm25_documents.append(bm25_document)
                    bm25_ids_by_uri.setdefault(
                        canonical_node_uri(node.uri),
                        [],
                    ).append(bm25_document.id)

        replace_where = getattr(self._bm25, "replace_where", None)
        if callable(replace_where):
            replace_where(
                {
                    "account_id": ctx.account_id,
                    "context_type": ContextType.RESOURCE.value,
                    "resource_type": "code",
                },
                bm25_documents,
            )

        snapshot = _CodeIndexSnapshot(
            records=tuple(records),
            read_error_count=read_error_count,
            bm25_ids_by_uri={
                uri: tuple(ids) for uri, ids in bm25_ids_by_uri.items()
            },
        )
        with self._cache_lock:
            self._cache[ctx.account_id] = snapshot
        return snapshot
