"""M3-Bench ReAct control adapter over the heterogeneous graph.

Wraps the generic :class:`core.react_loop.ReactLoop` template method
with M3-specific [Search] and [Answer] tools. The LLM sees only the
two tool definitions; the adapter handles:

* [Search] → :meth:`GraphService.search_by_embedding` against the
  EpisodicMemoryGenerator nodes (configurable via ``extractor_types``),
  with the query embedded by the configured ``Embedder``. Hits are
  returned with provenance (video_id, clip_id, description, characters,
  events) so the LLM can reason over them.
* [Answer] → marks the loop as complete and captures the answer text
  on the task context. A valid Answer completes the loop immediately,
  without an additional model call.

The control loop reuses the M3-Agent prompt structure (system
message establishing the role, then user message with the question).
Future work can extend the system prompt with per-video context,
example few-shots, or different extractor types (face_detection,
voice_segmentation) once those operators ship.
"""

from __future__ import annotations

import json
import logging
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any

from core.interfaces import Embedder
from core.models import RequestContext
from core.react_loop import (
    ReactLoop,
    ReactLoopResult,
    ReactTaskContext,
)
from core.tools.operational_tool_spec import operational_tool
from m3.context_retrieval import neighbor_plan, validate_context_options
from m3.context_budget import ContextBudget, pack_context
from m3.retrieval_core import (
    RetrievalBudget,
    fuse_evidence,
    select_evidence,
    validate_transcript_options,
)
from service.graph_service import GraphService

logger = logging.getLogger(__name__)


def _compact_cached_search_messages(messages: list[dict], start_index: int) -> None:
    """Reference duplicate evidence only when its full payload is still in history.

    The search cache, tools_used results, and search_history audit stay unchanged.
    No new or unique evidence is removed, and an absent/pruned first result
    prevents compaction. This avoids copying many kilobytes on every cache hit.
    """
    groups = ("hits", "context_hits", "boundary_hits")
    prior = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except (ValueError, TypeError):
            continue
        if not isinstance(payload, dict) or payload.get("evidence_payload_omitted"):
            continue
        values = [payload.get(group, []) for group in groups]
        if not all(isinstance(value, list) for value in values):
            continue
        hits = [hit for value in values for hit in value]
        if not hits or not all(isinstance(hit, dict) and hit.get("node_id") for hit in hits):
            continue
        reference = next((
            call_id for call_id, previous in prior
            if all(previous.get(group, []) == payload.get(group, []) for group in groups)
        ), None)
        if index >= start_index and payload.get("duplicate_query") and reference:
            compact = dict(payload)
            for group in groups:
                compact[group] = [
                    {key: hit[key] for key in ("node_id", "video_id", "clip_id") if key in hit}
                    for hit in payload.get(group, [])
                ]
            compact["evidence_payload_omitted"] = True
            compact["evidence_reference_tool_call_id"] = reference
            compact["guidance"] = (
                str(payload.get("guidance") or "")
                + f" The identical full evidence is already present in tool result {reference}. "
                "Refer to that earlier result; the references here are not new evidence."
            ).strip()
            message["content"] = json.dumps(compact, ensure_ascii=False)
        elif message.get("tool_call_id"):
            prior.append((message["tool_call_id"], payload))


# ---------------------------------------------------------------------------
# Task context
# ---------------------------------------------------------------------------

@dataclass
class M3TaskContext(ReactTaskContext):
    """Per-question state carried across the ReAct iterations.

    Holds the question, the (optional) video scope, and an
    accumulated search history that the LLM (and post-run audits) can
    inspect. ``final_answer`` is populated by the ``Answer`` tool
    once the LLM decides it has enough evidence.

    ``before_clip`` is an already-resolved inclusive clip bound; ``before_sec``
    is an absolute-video cutoff. Complete summaries crossing it are excluded.
    Annotation conversion belongs to m3.temporal, never to this adapter.
    """

    question: str
    video_id: str | None = None
    before_clip: int | None = None
    before_sec: float | None = None
    # None preserves the low-level API's already-resolved strict bounds.
    temporal_policy: str | None = None
    annotation_before_clip: Any = None
    annotation_timestamp: Any = None
    lookahead_seconds: float = 0.0
    temporal_scope: dict = field(default_factory=dict)
    search_history: list[dict] = field(default_factory=list)
    final_answer: str | None = None
    search_stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Name-stripping helpers for [Search] BM25 rerank
# ---------------------------------------------------------------------------

_NAME_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _name_tokens_from_hits(
    hits: list,
) -> set[str]:
    """Collect normalized name tokens from the ``characters`` field of each hit.

    Each ``characters`` entry may be a plain string or a dict with at
    least a ``name`` key (the shape emitted by
    :mod:`m3.episodic_memory_generator`; see
    :func:`m3.index_episodic._stringify_named_items` for the schema). Names
    are lowercased and stripped; multi-word names stay as a single phrase
    so the word-boundary regex in :func:`_strip_name_tokens_from_query`
    can match them as one unit.

    Defensive against malformed values: anything that isn't a string or
    a dict-with-name is skipped silently. A hit with no ``characters``
    field contributes nothing.
    """
    names: set[str] = set()
    for hit in hits:
        ec = getattr(getattr(hit, "node", None), "extracted_content", None) or {}
        chars = ec.get("characters") or []
        if not isinstance(chars, list):
            continue
        for c in chars:
            if isinstance(c, str):
                token = c.strip()
            elif isinstance(c, dict):
                token = str(c.get("name") or "").strip()
            else:
                continue
            if token:
                names.add(token.lower())
    return names


def _strip_name_tokens_from_query(
    query: str,
    name_tokens: set[str],
) -> str:
    """Remove ``name_tokens`` from ``query`` as whole-word, case-insensitive matches.

    Whitespace introduced by the removals is collapsed. Falls back to
    the original ``query`` when:

    * ``query`` is empty or whitespace-only, or
    * ``name_tokens`` is empty (nothing to strip), or
    * stripping would leave no alphanumeric content (an empty string
      would short-circuit the BM25 rerank and silently degrade the
      tool to vector-only ordering — which is exactly what we want to
      avoid in the small-candidate regime where BM25 is decisive).

    The patterns are cached on a module-level dict because the same set
    of names recurs across questions of the same video.
    """
    if not query or not query.strip() or not name_tokens:
        return query

    out = query
    for name in name_tokens:
        if not name:
            continue
        pattern = _NAME_PATTERN_CACHE.get(name)
        if pattern is None:
            pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
            _NAME_PATTERN_CACHE[name] = pattern
        out = pattern.sub("", out)
    out = re.sub(r"\s+", " ", out).strip()
    # Strip a bare space left in front of sentence-final punctuation so
    # the output reads naturally ("What does?" not "What does ?").
    # Cosmetic for downstream prompts; BM25 tokenization is
    # whitespace/punctuation-agnostic so this only affects display.
    out = re.sub(r"\s+([?.!,;:])", r"\1", out)
    if not out or not any(ch.isalnum() for ch in out):
        return query
    return out


# Match Title-Case word sequences (1+ letters each), optionally joined
# by a single space or hyphen ("Lewis", "Lewis Hamilton", "Anne-Marie").
# Single uppercase "I" is excluded because ``[A-Z][a-z]+`` requires at
# least one lowercase follower. All-caps abbreviations are intentionally
# excluded — M3-Bench names are always Title Case in the question text.
_QUERY_NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:[ -][A-Z][a-z]+)*\b")

# Capitalized words that are NOT names. Question-leading interrogatives
# and common sentence-initial nouns; these survive the first-word skip
# when they appear mid-sentence ("What does What look like?") and must
# not be stripped from the query. Kept short on purpose — extending it
# is safer than shrinking it (false negatives = we miss a name, false
# positives = we strip a content word).
_NAME_STOPWORDS = frozenset({
    # Question words
    "What", "Where", "When", "Who", "Why", "How",
    # Common verbs at sentence start
    "Is", "Are", "Was", "Were", "Do", "Does", "Did",
    "Can", "Could", "Would", "Should", "Will",
    "Put", "Take", "Get", "Give", "Find", "Show", "Tell",
    "Make", "Let", "Go", "Come", "Bring", "Move",
    # Determiners / pronouns
    "The", "This", "That", "These", "Those",
    "My", "Your", "His", "Her", "Our", "Their", "Its",
    # Prepositions (rare capitalized, but mid-sentence possible after "?")
    "In", "On", "At", "To", "For", "From", "By", "With", "Of",
    # Conjunctions
    "And", "Or", "But", "So", "Yet",
})

# First-word interrogatives — when the query starts with one of these,
# we skip it (it's a question word, not a name) and run the regex over
# the rest of the query. When the query starts with anything else
# (likely a name, e.g. "Lewis took what?"), we include the first word
# as a candidate.
_QUESTION_FIRST_WORDS = frozenset({
    "what", "where", "when", "who", "why", "how",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "would", "should", "will",
})


def _query_name_candidates(query: str) -> set[str]:
    """Cheap regex extractor for proper-name candidates in the query.

    Used as a SUPPLEMENT to :func:`_name_tokens_from_hits` so the
    dual-path retrieval in :func:`Search` can strip names from the
    query even when the correct clip — the one whose ``characters``
    field carries the name — never made it into the first vector
    top-k. The headline failure case is: the question says "Lewis" but
    Whisper transcribed the name as "Louis / Louise / Luis", so the
    first embedding pulls in other clips and the correct Lewis clip
    drops out of top-8. Without this regex layer, the characters from
    first hits would never include "Lewis" and we'd never strip it.

    Conservative: requires Title Case (``[A-Z][a-z]+``, one or more
    lowercase letters), filters a small stoplist of common capitalized
    non-names, and skips the first word of the query IF it is a
    question word. When the query starts with anything else (likely a
    name, e.g. "Lewis took what?"), the first word is included.

    Returns lowercased tokens so the result composes with the
    lowercased set from :func:`_name_tokens_from_hits`.
    """
    if not query or not query.strip():
        return set()

    candidates: set[str] = set()

    # Decide whether to scan the first word. Look only at the first
    # whitespace-separated token; ignore leading punctuation.
    stripped = query.lstrip()
    first_space = stripped.find(" ")
    first_word = stripped[:first_space] if first_space >= 0 else stripped

    if first_word and first_word.lower() in _QUESTION_FIRST_WORDS:
        # Skip the question word; scan the rest.
        scan = stripped[first_space + 1:] if first_space >= 0 else ""
    else:
        # First word is not a question word — scan the whole query
        # so leading names ("Lewis took what?") are captured.
        scan = stripped

    for match in _QUERY_NAME_PATTERN.finditer(scan):
        name = match.group(0)
        if name in _NAME_STOPWORDS:
            continue
        candidates.add(name.lower())
    return candidates


def _combined_name_tokens(query: str, hits: list) -> set[str]:
    """Union of characters-from-hits and regex-from-query, lowercased.

    Single entry point for both the BM25 ``lex_query`` stripping and
    the dual-path vector stripping in :func:`Search`. Combining both
    sources makes the stripping robust to the failure mode where the
    correct clip didn't make the first top-k and therefore can't
    contribute its ``characters`` field.
    """
    return _name_tokens_from_hits(hits) | _query_name_candidates(query)


def _merge_hits_by_node(*hit_lists: list) -> list:
    """Union of multiple hit lists, dedup by ``node_id``, keep the hit
    with the higher vector score for each node.

    The dual-path retrieval in :func:`Search` calls
    ``graph.search_by_embedding`` twice — once with the original
    ``query`` and once with the name-stripped ``stripped_query`` — and
    the two top-k sets can overlap on nodes that are relevant to both
    formulations. This merger keeps one row per node and preserves the
    stronger of the two cosine scores (a node that better matches
    *either* query is the better candidate).

    The union is sorted by vector score, but not truncated before reranking.
    """
    return select_evidence(
        (hit for hits in hit_lists for hit in (hits or [])
         if getattr(getattr(hit, "node", None), "node_id", None) is not None),
        identity=lambda hit: hit.node.node_id,
        score=lambda hit: hit.score,
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchConfig:
    """Immutable runtime scope; before_clip is the resolved inclusive bound."""

    video_id: str | None = None
    before_clip: int | None = None
    before_sec: float | None = None
    extractor_types: tuple[str, ...] = ("episodic_memory_generator",)
    default_top_k: int = 8
    candidate_k: int | None = None
    transcript_top_k: int = 0
    transcript_fusion: str = "rrf"
    include_neighbors: int = 1
    context_k: int = 0
    context_budget: ContextBudget | None = None


@operational_tool(name="Search", runtime_only=("search_config",))
def Search(
    query: str,
    top_k: int | None = None,
    ctx: RequestContext = None,                # type: ignore[assignment]
    graph: GraphService = None,                # type: ignore[assignment]
    embedder: Embedder = None,                # type: ignore[assignment]
    *,
    search_config: SearchConfig,
) -> dict:
    """Search the M3 graph for evidence relevant to a natural-language query.

    Embeds the query with the configured embedder, runs a top-k cosine
    similarity search over the indexed EpisractedNodes, and returns
    the hits with provenance (video_id, clip_id, description,
    characters, events, entities) so the LLM can cite each hit in
    subsequent rounds. The extractor type set and video scope come
    from adapter-level config (``extractor_types`` and
    ``default_video_id`` constructor args); they are NOT exposed to
    the LLM so the tool schema stays minimal.

    :param query: Natural-language search query (English / Chinese
        mixed scripts both work — the embedder handles tokenisation).
    :param top_k: Number of top hits to return (defaults to the run's configuration).
    """
    if not query or not query.strip():
        return {"error": "query is required"}
    effective_top_k = int(top_k) if top_k else search_config.default_top_k
    limits = search_config.context_budget
    candidate_k = search_config.candidate_k
    transcript_top_k = search_config.transcript_top_k
    if limits:
        effective_top_k = min(effective_top_k, limits.max_top_k)
        candidate_k = min(candidate_k or effective_top_k, limits.max_candidates)
        transcript_top_k = min(transcript_top_k, limits.max_candidates)
    budget = RetrievalBudget(effective_top_k, candidate_k)
    validate_transcript_options(transcript_top_k, search_config.transcript_fusion)
    if transcript_top_k:
        from core.transcript_lexical import validate_transcript_request
        validate_transcript_request(query, {
            "account_id": ctx.account_id, "video_id": search_config.video_id,
        }, transcript_top_k)
    validate_context_options(search_config.include_neighbors, search_config.context_k)
    if search_config.context_k and not search_config.video_id:
        raise ValueError("exact context requires a video_id")
    embedding = embedder.embed_texts([query])[0]

    def _build_search_kwargs(emb: list[float]) -> dict:
        """Build the kwargs for ``graph.search_by_embedding`` with the
        current per-task scope (video_id / before_clip) applied. Used by
        both the primary and the dual-path retrievals so the second
        search stays inside the same scope."""
        kw: dict = {
            "query": list(emb),
            "ctx": ctx,
            "extractor_types": search_config.extractor_types,
            "top_k": budget.fetch_k(),
        }
        if any(value is not None for value in (search_config.video_id, search_config.before_clip, search_config.before_sec)):
            from service.graph_service import ClipScope
            kw["scope"] = ClipScope(
                video_id=search_config.video_id,
                before_clip=search_config.before_clip,
                before_sec=search_config.before_sec,
            )
        return kw

    # --- Primary retrieval: embed the original query as-is -----------
    retrieval_queries = [query]
    hits = graph.search_by_embedding(**_build_search_kwargs(embedding))

    # --- Build the name set for stripping -----------------------------
    # Two sources, unioned:
    #   (a) characters from the in-hand top-k hits — catches names that
    #       the first search DID find.
    #   (b) Title-Case regex on the query itself — catches names that
    #       the first search DID NOT find (the headline failure case:
    #       "Lewis" in the question, but the correct clip — the only
    #       one whose characters field carries "Lewis" — dropped out of
    #       the first top-k because the polluted embedding dragged
    #       cosine to ~0.25). Without source (b), P1 #6 would still
    #       miss this case entirely.
    name_tokens = _combined_name_tokens(query, hits)
    stripped_query = _strip_name_tokens_from_query(query, name_tokens)

    # --- Dual-path retrieval: re-embed without the names --------------
    # Only worth the second embed + store call when stripping actually
    # changed something. Skipping the trivial case keeps latency flat
    # for questions that don't name anyone.
    if stripped_query != query and stripped_query:
        retrieval_queries.append(stripped_query)
        embedding_stripped = embedder.embed_texts([stripped_query])[0]
        hits_stripped = graph.search_by_embedding(
            **_build_search_kwargs(embedding_stripped),
        )
        hits = _merge_hits_by_node(hits, hits_stripped)

    # BM25 rerank over the union. ``lex_query`` is the stripped form
    # because names like "Lewis" are 0-hit tokens against transcripts
    # that Whisper heard as "Louis / Louise / Luis" — keeping them in
    # the BM25 query only dilutes the lexical signal that the rerank is
    # here to surface.
    from m3.transcript_rerank import rerank_with_transcript_bm25
    hits = _merge_hits_by_node(hits)
    hits = rerank_with_transcript_bm25(query, hits, lex_query=stripped_query)
    branch_scores = {}
    retrieval_branches = [{"kind": "vector", "query": q} for q in retrieval_queries]
    if transcript_top_k:
        from service.graph_service import ClipScope
        lexical_query = stripped_query or query
        lexical = graph.search_by_transcript(
            lexical_query, ctx,
            scope=ClipScope(search_config.video_id, search_config.before_clip, search_config.before_sec),
            extractor_types=search_config.extractor_types, top_k=transcript_top_k,
        )
        fused = fuse_evidence(
            hits, lexical, identity=lambda hit: hit.node.node_id,
            score=lambda hit: hit.score, method=search_config.transcript_fusion,
        )
        hits = [replace(entry.evidence, score=entry.score) for entry in fused]
        branch_scores = {entry.evidence.node.node_id: entry.source_scores for entry in fused}
        retrieval_branches.append({"kind": "transcript", "query": lexical_query})
    hits = select_evidence(
        hits, identity=lambda hit: hit.node.node_id, score=lambda hit: hit.score,
        top_k=budget.final_top_k,
    )
    boundary = []
    # Class capability check avoids inventing capabilities on dynamic mocks
    # and legacy adapters. The real GraphService provides the safe projection.
    boundary_reader = getattr(type(graph), "get_boundary_transcript", None)
    if (callable(boundary_reader) and search_config.video_id
            and search_config.before_sec is not None):
        from service.graph_service import ClipScope
        boundary = graph.get_boundary_transcript(
            ctx, scope=ClipScope(search_config.video_id, search_config.before_clip, search_config.before_sec),
            extractor_types=search_config.extractor_types,
        )
        if boundary:
            retrieval_branches.append({"kind": "boundary_transcript", "clip_ids": [h.clip_id for h in boundary]})

    main_count = len(hits)
    context_k = search_config.context_k
    radius = search_config.include_neighbors
    if limits:
        # Reserve boundary/main evidence before planning or reading neighbors.
        # Final evidence slots are separate from the candidate-clip read budget:
        # an indexed clip may have no usable evidence.
        boundary = boundary[:1]
        hits = hits[:limits.max_result_nodes - len(boundary)]
        context_k = min(context_k, limits.max_result_nodes - len(boundary) - len(hits))
    contexts = []
    if context_k and hits:
        from service.graph_service import ClipScope
        plan = neighbor_plan(
            [(h.video_id, h.clip_id, h.node.node_id) for h in hits],
            radius=radius, before_clip=search_config.before_clip,
            max_clips=limits.max_neighbor_clips if limits else None,
        )
        if plan:
            candidates = [clip for video, clip in plan if video == search_config.video_id]
            requested = []
            seen = {h.node.node_id for h in hits}
            while len(requested) < len(candidates) and len(contexts) < context_k:
                # Keep SQL parameter counts and round trips bounded. With no
                # budget retain the historical single-read behavior.
                batch_size = min(8, context_k - len(contexts)) if limits else len(candidates)
                batch = candidates[len(requested):len(requested) + batch_size]
                requested.extend(batch)
                raw_context = graph.get_evidence_by_clips(
                    batch, ctx,
                    scope=ClipScope(search_config.video_id, search_config.before_clip, search_config.before_sec),
                    extractor_types=search_config.extractor_types,
                )
                batch_keys = {(search_config.video_id, clip) for clip in batch}
                ranked_context = []
                for h in raw_context:
                    key = (h.video_id, h.clip_id)
                    if key not in batch_keys:
                        continue
                    distance, rank, parent = plan[key]
                    ranked_context.append(((distance, rank, h.clip_id, h.node.node_id), h, parent, distance))
                for _, h, parent, distance in sorted(ranked_context, key=lambda item: item[0]):
                    if h.node.node_id not in seen:
                        seen.add(h.node.node_id)
                        contexts.append((h, parent, distance))
                contexts = contexts[:context_k]
            retrieval_branches.append({"kind": "exact_neighbor", "clip_ids": requested})

    def serialize(h):
        from core.temporal import temporal_metadata
        return {
            "node_id": h.node.node_id,
            "retrieval_sources": list(branch_scores.get(h.node.node_id, {"vector": h.score})),
            "source_scores": branch_scores.get(h.node.node_id, {"vector": h.score}),
            "score": h.score,
            "video_id": h.video_id,
            "clip_id": h.clip_id,
            **temporal_metadata(h.node.extracted_content),
            "description": h.node.extracted_content.get("description", ""),
            "characters": h.node.extracted_content.get("characters", []),
            "events": h.node.extracted_content.get("events", []),
            "entities": h.node.extracted_content.get("entities", []),
            # Surface the per-clip ASR transcript so dialogue-driven
            # questions (M3-Bench "what do they plan", "what did they
            # say", "how do they feel") can be answered even when the
            # EpisodicMemoryGenerator's description didn't preserve
            # the load-bearing words verbatim.
            "speech_transcript": h.node.extracted_content.get(
                "speech_transcript", [],
            ),
        }

    retrieved_before_limit = main_count + len(contexts) + len(boundary)
    return {
        "query": query,
        "retrieval_queries": retrieval_queries,
        "candidate_k": budget.fetch_k(),
        "final_top_k": budget.final_top_k,
        "ranking_policy": (f"transcript_{search_config.transcript_fusion}"
                           if transcript_top_k else "vector_transcript_bm25"),
        "retrieval_branches": retrieval_branches,
        "transcript_top_k": transcript_top_k,
        "context_k": context_k,
        "boundary_k": 1,
        "retrieved_before_result_limit": retrieved_before_limit,
        "boundary_hits": [
            {**serialize(h), "evidence_role": "boundary_transcript",
             "retrieval_sources": ["boundary_transcript"], "source_scores": {},
             "source_start_sec": h.node.extracted_content["source_start_sec"],
             "source_end_sec": h.node.extracted_content["source_end_sec"]}
            for h in boundary
        ],
        "hits": [serialize(h) for h in hits],
        "context_hits": [
            {**serialize(h), "evidence_role": "context", "context_for": parent,
             "context_distance": distance, "retrieval_sources": ["exact_neighbor"],
             "source_scores": {}}
            for h, parent, distance in contexts
        ],
    }


@operational_tool(name="Answer")
def Answer(final_answer: str) -> dict:
    """Produce the final answer and end the loop.

    Call this when you have enough evidence to answer the question.
    The adapter captures ``final_answer`` and terminates the ReAct
    loop immediately after successful tool execution.

    :param final_answer: The answer text. Keep it concise and direct —
        the evaluator scores verbatim against the ground truth.
    """
    return {"status": "answered", "answer": final_answer}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class OGMemoryControlAdapter(ReactLoop):
    """M3 ReAct adapter — drives the [Search]/[Answer] loop.

    Usage::

        adapter = OGMemoryControlAdapter(
            llm=llm, graph=graph, embedder=embedder,
            max_iterations=5, extractor_types=("episodic_memory_generator",),
        )
        result = adapter.run(ctx, M3TaskContext(question="..."))
        print(result.output[0])  # the final answer

    Callers can retain the supplied ``M3TaskContext`` to inspect search
    history, statistics and the captured answer after :meth:`run`.
    """

    def __init__(
        self,
        llm,
        graph: GraphService,
        embedder: Embedder,
        *,
        max_iterations: int = 5,
        timeout_seconds: float = 60.0,
        extractor_types: tuple[str, ...] = ("episodic_memory_generator",),
        default_video_id: str | None = None,
        default_top_k: int = 8,
        candidate_k: int | None = None,
        transcript_top_k: int = 0,
        transcript_fusion: str = "rrf",
        include_neighbors: int = 1,
        context_k: int = 0,
        no_gain_limit: int | None = 3,
        context_budget: ContextBudget | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            operational_tools=[Search, Answer],
        )
        if no_gain_limit is not None and (
            isinstance(no_gain_limit, bool) or not isinstance(no_gain_limit, int) or no_gain_limit < 2
        ):
            raise ValueError("no_gain_limit must be an integer >= 2 or None")
        self._no_gain_limit = no_gain_limit
        self._context_budget = context_budget
        self._search_attempts = 0
        self._seen_evidence: set[str] = set()
        self._no_gain_streak = 0
        self._answer_only = False
        self._graph = graph
        self._embedder = embedder
        self._extractor_types = tuple(extractor_types)
        self._default_video_id = default_video_id
        self._default_top_k = int(default_top_k)
        RetrievalBudget(self._default_top_k, candidate_k)
        self._candidate_k = candidate_k
        validate_transcript_options(transcript_top_k, transcript_fusion)
        self._transcript_top_k = (
            min(transcript_top_k, context_budget.max_candidates) if context_budget else transcript_top_k
        )
        self._transcript_fusion = transcript_fusion
        validate_context_options(include_neighbors, context_k)
        self._include_neighbors = include_neighbors
        self._context_k = context_k
        self._search_config: SearchConfig | None = None
        # ReactLoop itself has mutable per-run state. Separate adapters may
        # run concurrently; concurrent/reentrant runs on one adapter may not.
        self._run_lock = Lock()
        # Per-run state set inside run() so the subclass can read it
        # in _execute_and_append_tool_calls / _parse_output.
        self._current_task_ctx: M3TaskContext | None = None
        self._final_answer: str | None = None
        self._search_cache: dict[tuple[str, int], dict] = {}

    # ------------------------------------------------------------------
    # ReactLoop abstract methods
    # ------------------------------------------------------------------

    def _runtime_injection_values(self, ctx: RequestContext) -> dict:
        if self._search_config is None:
            raise RuntimeError("Search configuration requires an initialized run")
        return {
            "ctx": ctx,
            "graph": self._graph,
            "embedder": self._embedder,
            "search_config": self._search_config,
        }

    def _get_custom_tools(self) -> list[dict]:
        # All tools are operational; no custom tools needed.
        return []

    def _execute_custom_tool(self, name, input, ctx):
        return {"error": f"Unknown tool: {name}"}

    def _build_messages(self, ctx, task_ctx: M3TaskContext):
        # Snapshot per-question configuration before any tools execute.
        self._current_task_ctx = task_ctx
        self._final_answer = None
        self._disable_tools_for_iteration = False
        self._search_cache.clear()
        self._seen_evidence.clear()
        self._search_attempts = 0
        self._no_gain_streak = 0
        self._answer_only = False
        task_ctx.search_history.clear()
        task_ctx.final_answer = None
        task_ctx.search_stats = {
            "search_requests": 0,
            "search_executions": 0,
            "cache_hits": 0,
            "blocked_requests": 0,
            "model_rounds": 0,
            "stop_reason": None,
        }

        scope = task_ctx.video_id or self._default_video_id
        if task_ctx.temporal_policy is not None:
            from m3.temporal import resolve_question_scope
            task_ctx.before_clip, task_ctx.before_sec, task_ctx.temporal_scope = resolve_question_scope(
                task_ctx.annotation_before_clip, task_ctx.annotation_timestamp,
                policy=task_ctx.temporal_policy, lookahead_seconds=task_ctx.lookahead_seconds,
                clip_lookup=lambda timestamp: self._graph.get_clip_at_time(
                    ctx, video_id=scope, timestamp=timestamp, extractor_types=self._extractor_types,
                ),
            )
            task_ctx.search_stats["temporal_scope"] = task_ctx.temporal_scope
        self._search_config = SearchConfig(
            video_id=scope,
            before_clip=task_ctx.before_clip,
            before_sec=task_ctx.before_sec,
            extractor_types=self._extractor_types,
            default_top_k=self._default_top_k,
            candidate_k=self._candidate_k,
            transcript_top_k=self._transcript_top_k,
            transcript_fusion=self._transcript_fusion,
            include_neighbors=self._include_neighbors,
            context_k=self._context_k,
            context_budget=self._context_budget,
        )

        system = (
            "You are an expert video analyst answering questions about a "
            "robot's point-of-view video. Use the Search function to find "
            "relevant episodic evidence in the graph memory, then call "
            "the Answer function once you have enough to respond. "
            "Invoke these functions through the provided tool-call interface; "
            "writing their names or an intended call as ordinary assistant text "
            "does not execute a search. Do not put tool instructions, XML tags, "
            "or code blocks in the final answer. Be concise — the "
            "evaluator scores verbatim against the ground truth.\n\n"
            "Search may return context_hits: exact neighboring clips linked by context_for. "
            "Read them for the preceding/following dialogue or action; adjacency alone "
            "is not proof that they answer the question.\n\n"
            "Search may also return boundary_hits: only completed speech segments "
            "from the clip crossing the question timestamp. Read these even when "
            "description is empty. Their timestamps are clip-relative; source_start_sec "
            "locates the clip in the video. No visual summary or speaker identity "
            "is available for these partial clips. Never infer later actions.\n\n"
            "Tips for searching effectively:\n"
            "- Do not repeat a query with the same top_k. Review existing "
            "hits first; if evidence is missing, search a different action, "
            "location, speaker, or synonym. Repeated queries reuse prior "
            "results and do not provide new evidence.\n"
            "- Each Search hit has TWO fields: ``description`` (a visual "
            "summary written by the LLM) and ``speech_transcript`` (a list "
            "of {start_sec, end_sec, text} dicts from the Whisper ASR). "
            "The transcript is the strongest signal for dialogue-driven "
            "questions — what people SAID, what they PLAN, how they FEEL, "
            "or any speech-anchored fact. Even when the description "
            "doesn't mention it, the transcript often does. Read BOTH "
            "fields on every hit before forming a query.\n"
            "- If the description looks thin or doesn't match your query, "
            "skim the transcript text in the same hit — the answer is "
            "often there.\n"
            "- When the description and transcript disagree (description "
            "says 'happy' but transcript says 'I'm exhausted'), the "
            "transcript wins for affect/emotion questions.\n"
            "- Only commit to an answer you can ground in a retrieved "
            "hit. If the evidence does not mention the person, object, "
            "or fact the question asks about, answer from the closest "
            "supported evidence and say so plainly rather than "
            "inventing a match.\n"
            "- For object locations, compare the observed moves in time order "
            "and use the last supported location before the question cutoff, "
            "not the earliest placement or the highest-scoring hit. Preserve "
            "the distinction between inside, on, and near. If movement is "
            "missing, search for the object being moved or put away.\n"
            "- For person-specific questions, establish who is speaking and "
            "who is addressed before resolving 'my' or 'your'. A name alone "
            "does not establish that mapping, and a speaker label from one "
            "clip does not establish identity across clips. State unresolved "
            "attribution instead of guessing a person or their gender.\n"
            "- If the first Search returns no useful hits, broaden the "
            "query with synonyms, the location, or the speaker's action "
            "before giving up."
        )
        if scope:
            system += (
                f" Restrict your evidence to video {scope!r}."
            )
        if task_ctx.temporal_scope:
            from m3.temporal import temporal_prompt
            system += "\n\n" + temporal_prompt(task_ctx.temporal_scope)
        if self._context_budget:
            system += (
                "\nSearch results are budgeted excerpts with source field/index or character offsets. "
                "evidence_refs point to evidence IDs defined in earlier tool results in THIS request. "
                "Only admitted evidence has references. Omitted facts are not absent facts, and "
                "retrieved events are not a complete count. Search a specific missing fact to "
                "select different excerpts; do not infer unseen facts. "
                "In this format, description and speech_transcript are identified by the field "
                "of each evidence[id].passages entry; read the description and adjoining "
                "speech entries from the same clip together, not as unrelated facts."
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {task_ctx.question}"},
        ]

    def _parse_output(self, content, ctx=None):  # noqa: ARG002
        from core.llm_output import final_text
        # Prefer the captured Answer text; fall back to the LLM's
        # free-form content if the model decided to answer without
        # calling [Answer] (some LLMs do this on the last round).
        if self._final_answer:
            return [self._final_answer]
        answer = final_text(content)
        if answer:
            return [answer]
        return None

    # ------------------------------------------------------------------
    # Hooks for [Answer] handling — capture the answer and end the loop
    # ------------------------------------------------------------------

    def _get_tools(self):
        tools = super()._get_tools()
        return [t for t in tools if t["name"] == "Answer"] if self._answer_only else tools

    def _allow_tool_iteration_extension(self):
        # Even a model that ignores the tool schema must have bounded work.
        return False

    def _output_after_tools(self):
        return [self._final_answer] if self._final_answer else None

    def _execute_tool(self, name, input, ctx):
        if name == "Answer":
            from core.llm_output import final_text
            if self._final_answer:
                return {"error": "An answer was already accepted"}
            answer = final_text(input.get("final_answer"))
            if not answer:
                return {"error": "final_answer must be a non-empty string; state evidence insufficiency if needed"}
            input = {**input, "final_answer": answer}
            result = super()._execute_tool(name, input, ctx)
            if isinstance(result, dict) and not result.get("error") and result.get("status") == "answered":
                self._final_answer = answer.strip()
                if self._current_task_ctx is not None:
                    self._current_task_ctx.final_answer = self._final_answer
            return result
        if name != "Search":
            return super()._execute_tool(name, input, ctx)
        if self._search_config is None:
            return {"error": "Search configuration requires an initialized run"}

        started = time.monotonic()
        params = dict(input)
        query = params.get("query", "")
        cached = None
        executed = False
        blocked = self._answer_only or bool(self._final_answer)
        if self._context_budget:
            blocked = blocked or self._search_attempts >= self._context_budget.max_search_calls
            self._search_attempts += 1
        key = None
        try:
            top_k = int(params.get("top_k") or self._search_config.default_top_k)
            if self._context_budget:
                top_k = min(top_k, self._context_budget.max_top_k)
                if isinstance(query, str) and len(query.encode("utf-8")) > self._context_budget.max_query_bytes:
                    query = ""  # reject rather than silently alter the query
            if isinstance(query, str) and query.strip() and top_k > 0:
                key = (" ".join(query.split()).casefold(), top_k)
                params["top_k"] = top_k
        except (ValueError, TypeError, OverflowError):
            top_k = None

        if blocked:
            result = {"error": "Search is closed for this run. Answer using existing evidence or explicitly state insufficiency."}
        elif key is None:
            result = {"error": "Search requires a non-empty query and positive top_k"}
        else:
            cached = self._search_cache.get(key)
            if cached is not None:
                result = deepcopy(cached)
                result["duplicate_query"] = True
            else:
                executed = True
                result = deepcopy(super()._execute_tool(name, params, ctx))
                if isinstance(result, dict) and not result.get("error"):
                    self._search_cache[key] = deepcopy(result)

        success = isinstance(result, dict) and not result.get("error")
        ids = {
            str(h["node_id"])
            for h in [*result.get("hits", []), *result.get("context_hits", []),
                      *result.get("boundary_hits", [])]
            if isinstance(h, dict) and h.get("node_id")
        } if success else set()
        new_ids = ids - self._seen_evidence
        if success:
            self._seen_evidence.update(ids)
            self._no_gain_streak = 0 if new_ids else self._no_gain_streak + 1
            if self._no_gain_limit is not None and self._no_gain_streak >= self._no_gain_limit:
                self._answer_only = True
            if self._answer_only:
                result["guidance"] = (
                    "Search stopped after consecutive results with no new evidence IDs. "
                    "Call Answer using supported evidence, or explicitly state evidence is insufficient. "
                    "No new evidence does NOT mean the answer is known."
                )
            elif not new_ids:
                result["guidance"] = (
                    "This search returned no new evidence IDs (reused results are not new evidence). "
                    "Try a different clue, or Answer and clearly state any missing evidence."
                )
        # Errors neither consume nor reset the no-gain streak: retries remain possible.
        if self._context_budget and self._search_attempts >= self._context_budget.max_search_calls:
            self._answer_only = True

        task = self._current_task_ctx
        if task is not None:
            stats = task.search_stats
            stats["search_requests"] += 1
            stats["search_executions"] += int(executed)
            stats["cache_hits"] += int(cached is not None)
            stats["blocked_requests"] += int(blocked)
            if self._answer_only:
                stats["stop_reason"] = (
                    "search_call_budget" if self._context_budget
                    and self._search_attempts >= self._context_budget.max_search_calls
                    else "consecutive_no_new_evidence"
                )
            task.search_history.append({
                "round": len(task.search_history) + 1, "query": query,
                "top_k": top_k, "stage": "requested", "cache_hit": cached is not None,
                "executed": executed, "blocked": blocked,
                "status": "success" if success else "error",
                "node_ids": sorted(ids), "new_node_ids": sorted(new_ids),
                "new_evidence_count": len(new_ids), "no_gain_streak": self._no_gain_streak,
                "duration_ms": (time.monotonic() - started) * 1000,
                "retrieval_queries": result.get("retrieval_queries", []) if success and executed else [],
                "retrieval_branches": result.get("retrieval_branches", []) if success and executed else [],
                "transcript_top_k": self._search_config.transcript_top_k,
                "context_k": self._search_config.context_k,
                "context_node_ids": [h["node_id"] for h in result.get("context_hits", [])] if success else [],
                "boundary_node_ids": [h["node_id"] for h in result.get("boundary_hits", [])] if success else [],
                "candidate_k": result.get("candidate_k") if success else None,
                "final_top_k": top_k,
                "ranking_policy": result.get("ranking_policy") if success else None,
                "temporal_scope": dict(task.temporal_scope),
                "stop_reason": stats["stop_reason"],
            })
        return result

    def _execute_and_append_tool_calls(self, messages, tool_calls, tools_used,
                                       read_uris, ctx, round_id=""):
        was_answer_only = self._answer_only
        start_index = len(messages)
        result = super()._execute_and_append_tool_calls(
            messages, tool_calls, tools_used, read_uris, ctx, round_id=round_id,
        )
        if not self._context_budget:
            _compact_cached_search_messages(messages, start_index)
        if self._answer_only and not was_answer_only and not self._final_answer:
            messages.append({"role": "user", "content": (
                "Search is now closed: no-new-evidence or search-call budget was reached. "
                "Only Answer is available. Give a supported answer or explicitly state "
                "that the evidence is insufficient; do not guess."
            )})
        return result

    def _call_llm(self, messages):
        if not self._context_budget:
            return super()._call_llm(messages)
        tools = None if self._disable_tools_for_iteration else self._get_tools()
        started = time.monotonic()
        view, stats = pack_context(
            messages, tools, self._current_task_ctx.question, self._context_budget,
        )
        stats["packing_seconds"] = time.monotonic() - started
        self._current_task_ctx.search_stats.setdefault("context_calls", []).append(stats)
        return super()._call_llm(view)

    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------

    def run(self, ctx: RequestContext, task_ctx: M3TaskContext) -> ReactLoopResult:
        """Run one question; reject same-instance overlap before mutating state."""
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError(
                "This adapter is already running; use a separate adapter "
                "for concurrent questions"
            )
        try:
            result = super().run(ctx, task_ctx)
            task_ctx.search_stats["model_rounds"] = len(result.trace.iterations)
            task_ctx.search_stats["finish_reason"] = result.finish_reason
            if task_ctx.search_stats["stop_reason"] is None:
                task_ctx.search_stats["stop_reason"] = "answer" if self._final_answer else result.finish_reason
            return result
        finally:
            self._search_config = None
            self._search_cache.clear()
            self._seen_evidence.clear()
            self._current_task_ctx = None
            self._run_lock.release()
