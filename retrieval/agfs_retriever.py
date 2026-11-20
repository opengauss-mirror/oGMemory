"""Literal retrieval over published AGFS context nodes."""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections import deque
from collections.abc import Iterator
from heapq import nlargest
from urllib.parse import unquote

from core.errors import (
    AccessDeniedError,
    ConcurrentModificationError,
    NodeBrokenError,
    NodeNotFoundError,
    OperationDeadlineExceededError,
)
from core.lexical import LexicalHit, LexicalIndex
from core.memory_names import is_agfs_search_directory, memory_directory
from core.models import (
    ContextNode,
    RequestContext,
    RetrievalConfig,
    RetrievedBlock,
    RetrieverMode,
    SearchMemoryResult,
    TypedQuery,
)
from core.operation_deadline import check_deadline, operation_deadline
from extraction.agent_memory import memory_units, node_memory_fields
from fs.agfs_adapter.node_scanner import published_agfs_uri, safe_agfs_uri
from retrieval.agfs_fts import AGFSFullTextIndex
from retrieval.context_reader import canonical_agent_node, project_memory_unit
from retrieval.identity import canonical_uri, memory_unit_key
from retrieval.owner_scope import canonical_owner_space, expand_owner_space_aliases
from retrieval.query_planner import QueryPlanner, sanitize_query
from retrieval.result_ranker import is_temporal_query
from retrieval.trace import RetrievalTrace, StageTrace, TraceTimer

logger = logging.getLogger(__name__)

_MAX_NODES = 512
_MAX_DIRECTORIES = 32
_MAX_TEXT_BYTES = 8 * 1024 * 1024
_MAX_FIELD_BYTES = 16 * 1024
_MAX_EXCERPT_CHARS = 1_200
_WORDS = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*")
_HAN_RUN = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff]+)")
_SHORT_ASCII = re.compile(r"(?<![^\W_])[a-z0-9]{1,2}(?![^\W_])")
_STOP_WORDS = frozenset(
    "a an and are as at be been being by can could did do does for from had has have "
    "he her hers him his how i in is it its me my of on or our she should so than that the "
    "their them then there these they this those to up us was we were what when where which "
    "who whom whose why will with would you your".split()
)


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _native_score(raw_score: float) -> float:
    return raw_score / (1.0 + raw_score)


def _literal_fields(node: ContextNode) -> tuple[str, str, str]:
    metadata = node.metadata
    identity = _normalized(" ".join(
        str(value or "") for value in (
            unquote(node.uri.rsplit("/", 1)[-1]), metadata.get("routing_key"),
            metadata.get("title"), metadata.get("object_name"),
            metadata.get("when"), metadata.get("who"), metadata.get("where"),
        )
    ).replace("_", " "))
    return identity, f"{identity} {_normalized(node.abstract)}", _normalized(f"{node.overview} {node.content}")


def _fts_terms(literals) -> tuple[set[str], set[str]] | None:
    terms, short = set(), set()
    for literal in literals:
        for term in (*literal.anchors, literal.phrase):
            if not term:
                continue
            if "\x00" in term:
                return None
            if len(term) >= 3:
                terms.add(term)
            elif _HAN_RUN.fullmatch(term):
                short.add("h" + term)
            elif term != literal.phrase and _SHORT_ASCII.fullmatch(term):
                short.add("w" + term)
            else:
                return None
    return (terms, short) if terms or short else None


class _LiteralQuery:
    def __init__(self, query: str) -> None:
        self.phrase = _normalized(query).strip("?!。？！.")
        words = []
        for part in _HAN_RUN.split(self.phrase):
            if len(part) > 2 and _HAN_RUN.fullmatch(part):
                words.extend(part[index:index + 2] for index in range(len(part) - 1))
            else:
                words.extend(_WORDS.findall(part))
        terms = [term for term in words if term not in _STOP_WORDS and (len(term) > 1 or term.isdigit())]
        self.anchors = list(dict.fromkeys(terms or words))[:16]
        self.patterns = [
            re.compile(rf"(?<![^\W_]){re.escape(term)}(?![^\W_])" if term.isascii() else re.escape(term))
            for term in self.anchors
        ]
        self.temporal = is_temporal_query(query)

    def score(self, node: ContextNode) -> float:
        identity, header, body = _literal_fields(node)
        if self.phrase and self.phrase == identity:
            return 1.0
        phrase_hit = bool(self.phrase and (self.phrase in header or self.phrase in body))
        unmatched = [pattern for pattern in self.patterns if not pattern.search(header)]
        header_hits = len(self.patterns) - len(unmatched)
        body_hits = sum(bool(pattern.search(body)) for pattern in unmatched)
        if not (header_hits or body_hits or phrase_hit):
            return 0.0
        denominator = max(len(self.patterns), 1)
        temporal_bonus = 0.04 if self.temporal and node.metadata.get("when") else 0.0
        return min(1.0, 0.20 + 0.56 * (header_hits + body_hits) / denominator
                   + 0.12 * header_hits / denominator + 0.08 * phrase_hit + temporal_bonus)

    def excerpt(self, content: str) -> str:
        if len(content) <= _MAX_EXCERPT_CHARS:
            return content
        step = _MAX_EXCERPT_CHARS // 2

        def anchor_count(start: int) -> int:
            window = _normalized(content[start:start + _MAX_EXCERPT_CHARS])
            return sum(bool(pattern.search(window)) for pattern in self.patterns)

        start = max(range(0, len(content) - step, step), key=anchor_count)
        return content[start:start + _MAX_EXCERPT_CHARS]


def _roots(ctx: RequestContext, context_type: str) -> list[str]:
    account = f"ctx://{ctx.account_id}"
    if context_type == "RESOURCE":
        return [f"{account}/resources"]
    roots = []
    if ctx.user_id:
        roots.append(f"{account}/users/{ctx.user_id}/memories")
    if ctx.agent_id:
        roots.append(f"{account}/agents/{ctx.agent_id}/memories")
        if context_type == "SKILL":
            roots.append(f"{account}/agents/{ctx.agent_id}/skills")
    return roots


class AGFSRetrieval:
    def __init__(self, fs, config: RetrievalConfig, llm=None, *, fts_enabled: bool = True,
                 fts_ranker: str = "literal", lexical_index: LexicalIndex | None = None,
                 lexical_backfill=None) -> None:
        self._fs = fs
        self._config = config
        self._quick_planner = QueryPlanner(config)
        self._thinking_planner = QueryPlanner(config, llm=llm)
        self._fts_bm25 = fts_ranker == "sqlite_bm25"
        self._lexical_index = lexical_index
        self._lexical_backfill = lexical_backfill
        self.fts = (
            AGFSFullTextIndex(self._index_documents, bm25=self._fts_bm25)
            if fts_enabled and lexical_index is None else None
        )

    def _index_documents(self, roots: tuple[str, ...], ctx: RequestContext) -> Iterator[tuple[str, str, str, bool]]:
        for uri in self._candidates(roots, ctx, set(), None):
            try:
                node = self._fs.read_search_node(uri, ctx, _MAX_FIELD_BYTES)
            except (NodeNotFoundError, NodeBrokenError):
                continue
            if self._published_uri(node, uri) is None:
                continue
            fields = [text for unit in self._units(node) for text in _literal_fields(unit)[1:]]
            document = "\n".join(fields).replace("\x00", "\n")
            short_terms = {"w" + term for term in _SHORT_ASCII.findall(document)}
            for run in _HAN_RUN.findall(document):
                short_terms.update("h" + char for char in run)
                short_terms.update("h" + run[i:i + 2] for i in range(len(run) - 1))
            yield uri, document, " ".join(short_terms), bool(node.metadata.get("_search_truncated"))

    @staticmethod
    def _published_uri(node: ContextNode, uri: str) -> str | None:
        if node.metadata.get("node_type") == "directory":
            return None
        return published_agfs_uri(node, uri)

    @staticmethod
    def _lexical_owner_spaces(typed: TypedQuery, ctx: RequestContext) -> list[str]:
        if typed.context_type == "RESOURCE":
            spaces = [f"resource:{ctx.account_id}"]
        else:
            spaces = [space for space in (
                ctx.user_space_name() if ctx.user_id else None,
                ctx.agent_space_name() if ctx.agent_id else None,
            ) if space]
        requested = typed.owner_space
        if requested:
            if isinstance(requested, str):
                requested = [requested]
            allowed = {canonical_owner_space(space) for space in requested}
            spaces = [space for space in spaces if space in allowed]
        return expand_owner_space_aliases(spaces)

    def _native_candidates(
        self, plans: list[tuple[TypedQuery, _LiteralQuery]], ctx: RequestContext,
        stage: StageTrace,
    ) -> dict[str, list[tuple[TypedQuery, _LiteralQuery, LexicalHit]]] | None:
        index = self._lexical_index
        assert index is not None
        scoped_plans = [(typed, literal, self._lexical_owner_spaces(typed, ctx)) for typed, literal in plans]
        scopes = list(dict.fromkeys(space for _, _, spaces in scoped_plans for space in spaces))
        matches: dict[str, list[tuple[TypedQuery, _LiteralQuery, LexicalHit]]] = {}
        stage.metrics["lexical_backend"] = "opengauss"
        try:
            if scopes and self._lexical_backfill is not None and not self._lexical_backfill.ensure_ready(ctx, scopes):
                stage.warnings.append("agfs_lexical_warming")
                stage.metrics["lexical"] = "warming"
                return None
            for typed, literal, spaces in scoped_plans:
                if not spaces:
                    continue
                check_deadline()
                filters = {"account_id": ctx.account_id, "owner_space": spaces,
                           "context_type": typed.context_type, "level": [0, 1, 2]}
                if "agfs_leaves_only" in getattr(index, "supported_filters", ()):
                    filters["agfs_leaves_only"] = True
                if typed.categories:
                    filters["category"] = typed.categories
                if typed.memory_directories:
                    filters["memory_directory"] = typed.memory_directories
                top_k = min(_MAX_NODES + 1, max(typed.top_k, self._config.lexical_top_k))
                hits = index.search(typed.text, top_k, filters)
                prefixes = tuple(root + "/" for root in _roots(ctx, typed.context_type))
                for hit in hits:
                    meta = hit.metadata
                    if (hit.score <= 0 or meta.get("account_id") != ctx.account_id
                            or meta.get("owner_space") not in spaces):
                        continue
                    uri = canonical_uri(str(meta.get("uri") or ""))
                    if safe_agfs_uri(uri) and uri.startswith(prefixes) and not is_agfs_search_directory(uri):
                        matches.setdefault(uri, []).append((typed, literal, hit))
        except OperationDeadlineExceededError:
            raise
        except Exception:
            logger.warning("AGFS native lexical search unavailable", exc_info=True)
            stage.warnings.append("agfs_lexical_unavailable")
            stage.metrics["lexical"] = "unavailable"
            return None
        stage.metrics["lexical"] = "ready"
        return matches

    @staticmethod
    def _score_native_hit(node: ContextNode, literal: _LiteralQuery, hit: LexicalHit, stage: StageTrace) -> float:
        hit_unit = memory_unit_key(hit.metadata)
        unit = memory_unit_key(node.metadata)
        if hit_unit and hit_unit != unit:
            return 0.0
        if hit_unit:
            expected, observed = hit.metadata.get("unit_revision"), node.metadata.get("unit_revision")
        else:
            expected, observed = hit.metadata.get("canonical_revision"), node.metadata.get("version")
        if expected is not None and observed is not None and expected != observed:
            if "agfs_lexical_stale" not in stage.warnings:
                stage.warnings.append("agfs_lexical_stale")
            return 0.0
        if unit and not hit_unit and not literal.score(node):
            return 0.0
        return _native_score(hit.score)

    def _candidates(
        self, roots: list[str] | tuple[str, ...], ctx: RequestContext, directories: set[str],
        stage: StageTrace | None,
    ) -> Iterator[str]:
        queue = deque(iter([root]) for root in roots)
        visited = set()
        listed = 0
        while queue:
            check_deadline()
            batch = queue.popleft()
            uri = next(batch, None)
            if uri is None:
                continue
            queue.append(batch)
            if uri in visited or not safe_agfs_uri(uri):
                continue
            visited.add(uri)
            directory = memory_directory(uri)
            if directories and directory and directory not in directories:
                continue
            if not is_agfs_search_directory(uri):
                yield uri
            elif stage is None or listed < _MAX_DIRECTORIES:
                listed += 1
                queue.append(iter(self._fs.list_children_including_inactive(uri, ctx)))
            elif "agfs_directory_budget" not in stage.warnings:
                stage.warnings.append("agfs_directory_budget")

    def run(
        self,
        query: str,
        ctx: RequestContext,
        *,
        top_k: int | None = None,
        categories: list[str] | None = None,
        target_uri: str | None = None,
        session_archive: dict | None = None,
        hints: dict | None = None,
        score_threshold: float | None = None,
        mode: str = RetrieverMode.QUICK,
        fill_content_for_top_k: int = 0,
        context_type: str | None = None,
        resource_type: str | None = None,
        memory_directories: list[str] | None = None,
        collect_graph_supplement: bool = False,
        use_scene: bool | None = None,
        final_top_k: int | None = None,
    ) -> SearchMemoryResult:
        started = time.monotonic()
        trace = RetrievalTrace()
        planner = self._thinking_planner if mode == RetrieverMode.THINKING else self._quick_planner
        target = canonical_uri(target_uri) if target_uri else None
        if target:
            parts = target.removeprefix("ctx://").split("/")
            if not safe_agfs_uri(target) or (
                len(parts) >= 3 and parts[1] == "sessions" and parts[2] != ctx.session_id
            ):
                raise AccessDeniedError(target, ctx.account_id, "target URI is outside the retrieval scope")
        typed_queries = planner.plan(
            query, ctx, categories=categories, top_k=top_k, context_type=context_type,
            resource_type=resource_type, session_archive=session_archive, hints=hints,
        )
        for item in typed_queries:
            item.target_uri = target
            item.memory_directories = memory_directories
        plans = [(item, _LiteralQuery(item.text)) for item in typed_queries]
        literal_plans = [(typed, literal, None) for typed, literal in plans]
        roots = [target] if target else list(dict.fromkeys(
            root for item in typed_queries for root in _roots(ctx, item.context_type)
        ))
        threshold = self._config.default_score_threshold if score_threshold is None else score_threshold
        limit = final_top_k if final_top_k is not None else (top_k or self._config.default_top_k)
        matches: list[tuple[float, str, ContextNode, _LiteralQuery]] = []
        remaining = _MAX_TEXT_BYTES
        seen = 0
        indexed = None
        native = None
        native_bounds: dict[str, float] = {}
        native_scores: dict[tuple[str, str], float] = {}
        native_cutoff = 0.0
        with TraceTimer("agfs_scan") as stage:
            try:
                with operation_deadline(10.0 if mode == RetrieverMode.THINKING else 5.0):
                    directories = set(memory_directories or ())
                    if self._lexical_index is not None and roots and not target:
                        native = self._native_candidates(plans, ctx, stage)
                    elif self.fts is not None and roots and not target:
                        terms = _fts_terms(literal for _, literal in plans)
                        status = "unsupported_terms"
                        if terms is not None:
                            memory_roots = _roots(ctx, "SKILL")
                            index_roots = memory_roots if set(roots).issubset(memory_roots) else roots
                            lookup = self.fts.candidates(
                                tuple(index_roots), ctx, *terms, search_roots=roots,
                                directories=directories, max_candidates=_MAX_NODES,
                            )
                            indexed, status = lookup.uris, lookup.status
                            if lookup.fields_truncated:
                                stage.warnings.append("agfs_field_budget")
                            if lookup.over_budget:
                                stage.warnings.append("agfs_scan_budget")
                        stage.metrics["fts"] = status
                    candidates = indexed
                    if native is not None:
                        native_bounds = {
                            uri: max(_native_score(hit.score) for _, _, hit in records)
                            for uri, records in native.items()
                        }
                        candidates = sorted(native_bounds, key=lambda uri: (-native_bounds[uri], uri))
                        stage.metrics["native_candidate_count"] = len(candidates)
                    if candidates is None:
                        candidates = self._candidates(roots, ctx, directories, stage)
                    for uri in candidates:
                        if native is not None and native_cutoff > native_bounds[uri]:
                            break
                        check_deadline()
                        directory = memory_directory(uri)
                        if seen >= _MAX_NODES or remaining < 3:
                            if "agfs_scan_budget" not in stage.warnings:
                                stage.warnings.append("agfs_scan_budget")
                            break
                        seen += 1
                        try:
                            node = self._fs.read_search_node(uri, ctx, min(_MAX_FIELD_BYTES, remaining // 3))
                        except (NodeNotFoundError, NodeBrokenError):
                            continue
                        except ConcurrentModificationError:
                            if "agfs_node_busy" not in stage.warnings:
                                stage.warnings.append("agfs_node_busy")
                            continue
                        if node.metadata.pop("_search_truncated", False) and "agfs_field_budget" not in stage.warnings:
                            stage.warnings.append("agfs_field_budget")
                        remaining -= sum(len(text.encode("utf-8")) for text in (
                            node.abstract, node.overview, node.content,
                        ))
                        stored_uri = self._published_uri(node, uri)
                        if stored_uri is None:
                            continue
                        scoped_plans = native[uri] if native is not None else literal_plans
                        for candidate in self._units(node):
                            for typed, literal, hit in scoped_plans:
                                if not self._matches_scope(candidate, typed, directory):
                                    continue
                                if hit is not None:
                                    score = self._score_native_hit(candidate, literal, hit, stage)
                                else:
                                    score = 1.0 if target == uri else literal.score(candidate)
                                if score > 0 and score >= threshold:
                                    matches.append((score, stored_uri, candidate, literal))
                                    if native is not None:
                                        identity = (stored_uri, memory_unit_key(candidate.metadata))
                                        native_scores[identity] = max(score, native_scores.get(identity, 0.0))
                        if native is not None and 0 < limit <= len(native_scores):
                            native_cutoff = nlargest(limit, native_scores.values())[-1]
            except OperationDeadlineExceededError:
                stage.warnings.append("agfs_scan_deadline")
            stage.input_count = seen
            stage.output_count = len(matches)
            stage.metrics["text_bytes"] = _MAX_TEXT_BYTES - remaining
        trace.add_stage(stage)

        hits = []
        identities = set()
        reason = "openGauss BM25 match" if native is not None else "AGFS literal match"
        index_order = {uri: order for order, uri in enumerate(indexed or ())} if self._fts_bm25 else {}
        for score, uri, node, literal in sorted(
            matches,
            key=lambda match: (index_order.get(match[1], len(index_order)), -match[0], match[1],
                               memory_unit_key(match[2].metadata)),
        ):
            identity = (uri, memory_unit_key(node.metadata))
            if identity in identities:
                continue
            identities.add(identity)
            if len(hits) == limit:
                break
            hits.append(RetrievedBlock(
                uri=uri, context_type=node.context_type, category=node.category,
                owner_space=node.owner_space, score=score, abstract=node.abstract,
                overview=node.overview, has_overview=bool(node.overview), has_content=bool(node.content),
                content_excerpt=literal.excerpt(node.content) if len(hits) < fill_content_for_top_k else None,
                match_reason=reason, metadata=dict(node.metadata),
            ))
        trace.total_ms = (time.monotonic() - started) * 1_000
        return SearchMemoryResult(
            request_id=trace.request_id, query=sanitize_query(query), typed_queries=typed_queries,
            hits=hits, trace=trace, partial=bool(trace.warnings),
        )

    @staticmethod
    def _matches_scope(node: ContextNode, typed: TypedQuery, directory: str) -> bool:
        owners = typed.owner_space
        if isinstance(owners, str):
            owners = [owners]
        return (
            node.context_type == typed.context_type
            and (not typed.categories or node.category in typed.categories)
            and (not typed.memory_directories or directory in typed.memory_directories)
            and (not typed.resource_type or node.metadata.get("resource_type") == typed.resource_type)
            and (not owners or canonical_owner_space(node.owner_space) in {
                canonical_owner_space(owner) for owner in owners
            })
        )

    @staticmethod
    def _units(node: ContextNode) -> Iterator[ContextNode]:
        if node.metadata.get("agent_memory_version") == 1:
            units = memory_units(node.category, node_memory_fields(node))
            if units:
                for unit in units:
                    yield project_memory_unit(node, unit["key"])
                return
        yield canonical_agent_node(node)
