# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Eager prefetch for existing memories before LLM extraction."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

from core.interfaces import ContextFS, Embedder, VectorIndex
from core.models import ContextNode, RequestContext, SeedHit

if TYPE_CHECKING:
    from core.uri_resolver import URIResolver
    from extraction.schemas.registry import SchemaRegistry

logger = getLogger(__name__)

_PREFETCH_NODES = 10
_QUERY_CHARS = 1000
_MAX_QUERIES = 4
_CATALOG_CHARS = 16_000
_CATALOG_PAGE_SIZE = 100
_CATALOG_MAX_PAGES = 4
# Application character budgets, independent of provider-specific token windows.
_PREFETCH_CHARS = 32_000
_CONTEXT_CHARS = 64_000
_OUTPUT_CHARS = 16_000  # Leave room for tool schemas, JSON fallback and generated facts.
_WRAPPER_CHARS = 128


def _available_prefetch_chars(prompt_chars: int) -> int:
    return max(0, min(_PREFETCH_CHARS, _CONTEXT_CHARS - prompt_chars - _OUTPUT_CHARS - _WRAPPER_CHARS))


@dataclass
class _OccurrenceEntry:
    result: PrefetchResult
    uri: str
    version: int
    full: str
    hint: str
    related: bool
    message: str | None = None
    complete: bool = False


@dataclass
class _OccurrenceBudget:
    remaining: int
    entries: list[_OccurrenceEntry] = field(default_factory=list)

    def apply(self) -> None:
        # Preserve every minimum representation when possible. A short complete
        # body can cost less than its hint. If even these cannot all fit, spend
        # the remaining budget in priority order without impossible reservations.
        reserved = sum(min(len(entry.full), len(entry.hint)) + 1 for entry in self.entries)
        reserve_all = reserved <= self.remaining
        for entry in sorted(self.entries, key=lambda entry: not entry.related):
            reserved -= min(len(entry.full), len(entry.hint)) + 1
            available = self.remaining - reserved if reserve_all else self.remaining
            complete = len(entry.full) + 1 <= available
            message = entry.full if complete else entry.hint
            if len(message) + 1 > self.remaining:
                continue
            self.remaining -= len(message) + 1  # Include the joining newline.
            entry.message, entry.complete = message, complete
        # Budget priority does not change the categories' presentation order.
        for entry in self.entries:
            if entry.message is not None:
                entry.result.messages.append(entry.message)
                if entry.complete:
                    entry.result.read_uris.add(entry.uri)
                    entry.result.read_versions[entry.uri] = entry.version


@dataclass
class PrefetchResult:
    """Result of prefetching existing memories for a memory type."""
    messages: list[str] = field(default_factory=list)
    read_uris: set[str] = field(default_factory=set)
    listed_uris: set[str] = field(default_factory=set)
    # Lazy mode: URI hints without full content
    uri_hints: list[dict] = field(default_factory=list)  # [{"uri": str, "abstract": str, "score": float}]
    is_lazy: bool = False
    read_versions: dict[str, int] = field(default_factory=dict)
    read_nodes: dict[str, ContextNode] = field(default_factory=dict)


class MemoryPrefetcher:
    """Prefetch existing memories before LLM extraction.

    By operation_mode:
    - single-file upsert (profile): read whole file
    - multi-file upsert (preference/entity/pattern/skill/tool): vector search + read top hits
    - add_only (event/case): semantic recall with a bounded recent-entry fallback
    """

    def __init__(
        self,
        fs: ContextFS,
        vector_index: VectorIndex,
        embedder: Embedder,
        registry: SchemaRegistry,
        uri_resolver: URIResolver,
    ):
        self._fs = fs
        self._vector_index = vector_index
        self._embedder = embedder
        self._registry = registry
        self._uri_resolver = uri_resolver
    def _prefetch(
        self,
        memory_type: str,
        ctx: RequestContext,
        *,
        conversation_text: str | None = None,
        lazy: bool = False,
        query_vectors: list[list[float]] | None = None,
        occurrence_budget: _OccurrenceBudget | None = None,
    ) -> PrefetchResult:
        """Prefetch existing memories for a given memory type (internal method).

        Args:
            memory_type: Memory type identifier (e.g., "profile", "preference")
            ctx: RequestContext for multi-tenant isolation
            conversation_text: Recent conversation text for semantic query.
            lazy: If True, collect URI hints only (no content reading).

        Returns:
            PrefetchResult with messages (eager) or uri_hints (lazy)
        """
        schema = self._registry.get_compatible(memory_type)
        if schema is None:
            logger.warning(f"Cannot prefetch unknown or incompatible type: {memory_type}")
            return PrefetchResult(is_lazy=lazy)

        if schema.operation_mode == "add_only":
            return self._prefetch_add_only(memory_type, ctx, conversation_text=conversation_text,
                                           lazy=lazy, query_vectors=query_vectors,
                                           budget=occurrence_budget)
        elif schema.operation_mode == "upsert":
            if schema.is_single_file:
                return self._prefetch_single_file(memory_type, ctx, lazy=lazy)
            else:
                result = self._prefetch_multi_file(memory_type, ctx, conversation_text=conversation_text,
                                                   lazy=lazy, query_vectors=query_vectors)
                self._add_identity_catalog(memory_type, ctx, result)
                return result
        else:
            logger.warning(f"Unknown operation_mode '{schema.operation_mode}' for {memory_type}")
            return PrefetchResult(is_lazy=lazy)

    def prefetch_for_span(
        self,
        categories: list[str],
        ctx: RequestContext,
        conversation_text: str | None = None,
        lazy: bool = False,
        *,
        prompt_chars: int | None = None,
    ) -> PrefetchResult:
        """Prefetch memories for multiple categories.

        Args:
            categories: List of memory types to prefetch
            ctx: RequestContext
            conversation_text: Recent conversation text for semantic query.
            lazy: If True, collect URI hints only.
            prompt_chars: Complete extraction prompt length, before prefetch. When
                omitted, reserve space for the supplied conversation text.

        Returns:
            Combined PrefetchResult from all categories
        """
        combined = PrefetchResult(is_lazy=lazy)
        categories = list(dict.fromkeys(categories))
        vector_categories = {cat for cat in categories
                             if (schema := self._registry.get_compatible(cat))
                             and schema.is_multi_file
                             and schema.operation_mode in {"upsert", "add_only"}}
        occurrence_categories = {cat for cat in categories
                                 if (schema := self._registry.get_compatible(cat))
                                 and schema.operation_mode == "add_only"}
        query_vectors = None
        if conversation_text and vector_categories:
            try:
                query_vectors = self._embed_queries(conversation_text)
            except Exception as exc:
                logger.warning("Prefetch embedding failed: %s", exc)
                query_vectors = []
        results: dict[str, PrefetchResult] = {}
        budget = None
        # Existing upsert behavior stays intact; account for its actual context
        # before admitting the newly recalled occurrence bodies, in any input order.
        for cat in sorted(categories, key=lambda cat: not lazy and cat in occurrence_categories):
            try:
                kwargs = {"query_vectors": query_vectors} if cat in vector_categories else {}
                if not lazy and cat in occurrence_categories:
                    if budget is None:
                        used = sum(len(message) + 1 for result in results.values() for message in result.messages)
                        source_chars = len(conversation_text or "") if prompt_chars is None else prompt_chars
                        budget = _OccurrenceBudget(max(0, _available_prefetch_chars(source_chars) - used))
                    kwargs["occurrence_budget"] = budget
                results[cat] = self._prefetch(cat, ctx, conversation_text=conversation_text, lazy=lazy, **kwargs)
            except Exception as e:
                logger.error(f"Prefetch failed for {cat}: {e}")
        if budget is not None:
            budget.apply()
        for cat in categories:
            if (result := results.get(cat)) is not None:
                combined.messages.extend(result.messages)
                combined.read_uris.update(result.read_uris)
                combined.read_versions.update(result.read_versions)
                combined.read_nodes.update(result.read_nodes)
                combined.listed_uris.update(result.listed_uris)
                combined.uri_hints.extend(result.uri_hints)
        return combined

    def _embed_queries(self, text: str) -> list[list[float]]:
        """Bound query count/length while retaining windows across the whole source."""
        provider_limit = getattr(self._embedder, "max_input_chars", _QUERY_CHARS)
        limit = min(_QUERY_CHARS, provider_limit) if type(provider_limit) is int and provider_limit > 0 else _QUERY_CHARS
        count = min(_MAX_QUERIES, (len(text) + limit - 1) // limit)
        if not count:
            return []
        starts = [round(i * max(0, len(text) - limit) / (count - 1)) for i in range(count)] if count > 1 else [0]
        queries = list(dict.fromkeys(text[start:start + limit] for start in starts))
        vectors = self._embedder.embed_texts(queries)
        if len(vectors) != len(queries):
            raise ValueError("prefetch embedding requires one vector per query")
        return vectors

    def _add_identity_catalog(self, memory_type: str, ctx: RequestContext, result: PrefetchResult) -> None:
        """Supply bounded identity hints, without treating summaries as full reads."""
        if not callable(getattr(type(self._fs), "list_node_summaries_page", None)):
            return
        directory = self._uri_resolver.get_directory_uri(memory_type, ctx).rstrip("/")
        instruction = ("Read the selected URI's full body and version before updating. " if result.is_lazy
                       else "For entries without a supplied full body, emit source facts without target_uri or expected_version. ")
        lines = [f"Identity catalog: category={memory_type} directory={directory}\n"
                 "Names join to directory/name. These are identity hints, not read snapshots. "
                 + instruction + "Compare the subject and scope with the source.\n"]
        size, cursor, status = len(lines[0]), None, "partial"
        schema = self._registry.get_compatible(memory_type)
        owner = ctx.user_space_name() if schema.owner_scope == "user" else ctx.agent_space_name()
        order_key = getattr(self._fs, "list_node_order_key", lambda uri: uri)
        try:
            for _ in range(_CATALOG_MAX_PAGES):
                page = self._fs.list_node_summaries_page(directory, ctx, limit=_CATALOG_PAGE_SIZE, after_uri=cursor)
                for node in page:
                    if (node.uri in result.read_uris or node.uri.rsplit("/", 1)[0] != directory
                            or node.category != memory_type or node.owner_space != owner):
                        continue
                    fields = {"name": node.uri.rsplit("/", 1)[-1], "L0": node.abstract}
                    for key in ("scope_ref", "object_name", "who", "routing_key", "routing_key_aliases", "object_name_aliases"):
                        if node.metadata.get(key):
                            fields[key] = node.metadata[key]
                    line = json.dumps(fields, ensure_ascii=False) + "\n"
                    if size + len(line) + len("Catalog status: complete\n") > _CATALOG_CHARS:
                        break
                    lines.append(line)
                    size += len(line)
                    result.listed_uris.add(node.uri)
                else:
                    if len(page) < _CATALOG_PAGE_SIZE:
                        status = "complete"
                        break
                    cursor = order_key(page[-1].uri)
                    continue
                break
        except Exception as exc:
            logger.warning("Identity catalog partial for %s: %s", memory_type, exc)
        result.messages.append("".join(lines) + f"Catalog status: {status}\n")

    def _prefetch_single_file(self, memory_type: str, ctx: RequestContext, *, lazy: bool = False) -> PrefetchResult:
        """Prefetch for single-file schemas (e.g., profile)."""
        result = PrefetchResult(is_lazy=lazy)
        try:
            uri = self._uri_resolver.resolve(memory_type, {}, ctx)
            result.listed_uris.add(uri)

            if not self._fs.exists(uri, ctx):
                logger.debug(f"No existing {memory_type} at {uri}")
                return result

            if lazy:
                # Lazy: just get abstract (from vector index or minimal read)
                abstract = self._get_abstract_for_uri(uri, memory_type, ctx)
                result.uri_hints.append({"uri": uri, "abstract": abstract, "score": 1.0})
                logger.info(f"Lazy prefetch single-file {memory_type}: URI hint")
            else:
                # Eager: read full content
                node = self._fs.read_node(uri, ctx)
                result.read_uris.add(uri)
                result.read_versions[uri] = node.metadata.get("version", 0)
                result.read_nodes[uri] = deepcopy(node)
                result.messages.append(
                    f"== Existing {memory_type} memory ==\n"
                    f"URI: {uri}\n"
                    f"Version: {node.metadata.get('version', 0)}\n"
                    f"Routing key: {node.metadata.get('routing_key', '')}\n"
                    f"Abstract: {node.abstract}\n"
                    f"Overview: {node.overview}\n"
                    f"Content: {node.content}\n"
                )
                logger.info(f"Prefetched single-file {memory_type} from {uri}")

        except Exception as e:
            logger.error(f"Failed to prefetch single-file {memory_type}: {e}")
        return result

    def _prefetch_multi_file(
        self,
        memory_type: str,
        ctx: RequestContext,
        *,
        conversation_text: str | None = None,
        lazy: bool = False,
        query_vectors: list[list[float]] | None = None,
    ) -> PrefetchResult:
        """Prefetch for multi-file schemas using vector search."""
        result = PrefetchResult(is_lazy=lazy)
        try:
            query_text = conversation_text if conversation_text is not None else memory_type
            vectors = self._embed_queries(query_text) if query_vectors is None else query_vectors

            filters = self._build_filters(memory_type, ctx)
            unique_hits: dict[str, SeedHit] = {}
            for query_vector in vectors:
                hits = self._vector_index.search_by_vector(query_vector=query_vector, filters=filters, top_k=2 * _PREFETCH_NODES)
                for hit in hits:
                    uri = hit.uri.removesuffix("/content.md")
                    if uri not in unique_hits or hit.score > unique_hits[uri].score:
                        unique_hits[uri] = hit

            if not unique_hits:
                logger.debug(f"No vector hits for {memory_type}")
                return result

            for node_uri, hit in sorted(unique_hits.items(), key=lambda item: (-item[1].score, item[0]))[:_PREFETCH_NODES]:
                result.listed_uris.add(node_uri)

                if lazy:
                    # Lazy: use abstract from hit directly
                    result.uri_hints.append({"uri": node_uri, "abstract": hit.abstract, "score": hit.score})
                else:
                    # Eager: read full content
                    try:
                        node = self._fs.read_node(node_uri, ctx)
                        result.read_uris.add(node_uri)
                        result.read_versions[node_uri] = node.metadata.get("version", 0)
                        result.read_nodes[node_uri] = deepcopy(node)
                        result.messages.append(
                            f"[{memory_type}] URI: {node_uri}\n"
                            f"Version: {node.metadata.get('version', 0)}\n"
                            f"Routing key: {node.metadata.get('routing_key', '')}\n"
                            f"Scope: {node.metadata.get('scope_ref', '')}\n"
                            f"Object: {node.metadata.get('object_name', '')}\n"
                            f"Aliases: {json.dumps(node.metadata.get('routing_key_aliases', []), ensure_ascii=False)}\n"
                            f"Object aliases: {json.dumps(node.metadata.get('object_name_aliases', []), ensure_ascii=False)}\n"
                            f"Who: {node.metadata.get('who', '')}\n"
                            f"Abstract: {node.abstract}\n"
                            f"Overview: {node.overview}\n"
                            f"Content: {node.content}\n"
                            f"(similarity: {hit.score:.2f})"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read {hit.uri}: {e}")
                        result.messages.append(
                            f"[{memory_type}] URI: {hit.uri}\n"
                            f"Abstract: {hit.abstract}\n"
                            f"(similarity: {hit.score:.2f}, read failed)"
                        )

            count = len(result.uri_hints) if lazy else len(result.read_uris)
            logger.info(f"{'Lazy prefetch' if lazy else 'Prefetched'} {memory_type}: {count} items")

        except Exception as e:
            logger.error(f"Failed to prefetch multi-file {memory_type}: {e}")
        return result

    def _prefetch_add_only(
        self, memory_type: str, ctx: RequestContext, *,
        conversation_text: str | None = None, lazy: bool = False,
        query_vectors: list[list[float]] | None = None,
        budget: _OccurrenceBudget | None = None,
    ) -> PrefetchResult:
        """Recall related occurrences before filling remaining slots from the directory.

        Similarity selects context only; it never proves occurrence identity or
        authorizes an update. Keep full source scope and occurrence facts visible.
        """
        result = PrefetchResult(is_lazy=lazy)
        local_budget = budget is None
        if budget is None:
            budget = _OccurrenceBudget(_available_prefetch_chars(len(conversation_text or "")))
        context_note = ("Stored facts are context, not new source evidence. Similarity, shared names "
                        "or dates alone do not prove the same occurrence; preserve distinct activities.\n")
        directory = self._uri_resolver.get_directory_uri(memory_type, ctx).rstrip("/")
        hits: dict[str, SeedHit] = {}
        if conversation_text:
            try:
                vectors = (self._embed_queries(conversation_text)
                           if query_vectors is None else query_vectors)
                filters = self._build_filters(memory_type, ctx)
                for vector in vectors:
                    for hit in self._vector_index.search_by_vector(
                        query_vector=vector, filters=filters, top_k=2 * _PREFETCH_NODES,
                    ):
                        uri = hit.uri.removesuffix("/content.md")
                        if uri.rsplit("/", 1)[0] != directory or hit.category != memory_type:
                            continue
                        if uri not in hits or hit.score > hits[uri].score:
                            hits[uri] = hit
            except Exception as exc:
                logger.warning("Occurrence recall failed for %s: %s", memory_type, exc)
        uris = sorted(hits, key=lambda uri: (-hits[uri].score, uri))
        try:
            uris.extend(sorted(self._fs.list_children(directory, ctx), reverse=True))
        except Exception as exc:
            logger.warning("Occurrence listing failed for %s: %s", memory_type, exc)
        owner = ctx.agent_space_name() if self._registry.get(memory_type).owner_scope == "agent" else ctx.user_space_name()
        for uri in dict.fromkeys(uris):
            if len(result.listed_uris) >= _PREFETCH_NODES:
                break
            if uri.rsplit("/", 1)[0] != directory:
                continue
            hit = hits.get(uri)
            if lazy:
                result.listed_uris.add(uri)
                result.uri_hints.append({"uri": uri, "abstract": hit.abstract if hit else "",
                                         "score": hit.score if hit else 0.0})
                continue
            try:
                node = self._fs.read_node(uri, ctx)
                if node.category != memory_type or node.owner_space != owner:
                    continue
                metadata = {name: node.metadata.get(name) for name in
                            ("routing_key", "scope_ref", "title", "when", "who", "where")}
                result.listed_uris.add(uri)
                label = "Related" if hit else "Recent"
                full = (
                    f"[{label} {memory_type}] URI: {uri}\n"
                    f"Version: {node.metadata.get('version', 0)}\n"
                    f"Occurrence: {json.dumps(metadata, ensure_ascii=False)}\n"
                    f"Abstract: {node.abstract}\nOverview: {node.overview}\nContent: {node.content}\n"
                    + context_note
                )
                hint_metadata = "; ".join(f"{name}: {str(value or '')[:48]}" for name, value in metadata.items())
                hint_uri = uri if len(uri) <= 512 else uri[:512] + "… (URI abbreviated)"
                hint = (
                    f"[{label} {memory_type}] URI: {hint_uri}\n"
                    f"Occurrence hint (fields may be abbreviated): {hint_metadata}\n"
                    f"Abstract hint (may be abbreviated): {node.abstract[:128]}\n"
                    "Full body not supplied (context budget). Identity/summary hints are not read snapshots. "
                    "Emit source facts without target_uri or expected_version; read the full body and "
                    "version before targeting updates.\n"
                    + context_note
                )
                budget.entries.append(_OccurrenceEntry(
                    result=result, uri=uri, version=node.metadata.get("version", 0),
                    full=full, hint=hint, related=hit is not None,
                ))
            except Exception as exc:
                logger.warning("Failed to read occurrence %s: %s", uri, exc)
        if local_budget:
            budget.apply()
        return result

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _build_filters(self, memory_type: str, ctx: RequestContext) -> dict:
        """Build vector search filters for a memory type."""
        filters = {"category": memory_type, "account_id": ctx.account_id}
        schema = self._registry.get(memory_type)
        if schema and schema.owner_scope == "user":
            filters["owner_space"] = f"user:{ctx.user_id}"
        elif schema and schema.owner_scope == "agent":
            filters["owner_space"] = f"agent:{ctx.agent_id}"
        return filters

    def _get_abstract_for_uri(self, uri: str, memory_type: str, ctx: RequestContext) -> str:
        """Get abstract for a URI (for lazy single-file prefetch)."""
        # Try vector index first
        try:
            filters = self._build_filters(memory_type, ctx)
            vectors = self._embedder.embed_texts([memory_type])
            hits = self._vector_index.search_by_vector(query_vector=vectors[0], filters=filters, top_k=1)
            if hits and hits[0].uri.rstrip("/content.md") == uri:
                return hits[0].abstract
        except Exception:
            pass
        # Fallback: minimal read
        node = self._fs.read_node(uri, ctx)
        return node.abstract
