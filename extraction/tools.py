"""Two-phase candidate extraction.

Phase 1 — Span Identification: Cheap JSON-mode LLM call identifies which
user message ranges contain extractable information.

Phase 2 — Span Structuring: For each span, focused tool-call extraction
with +/- 3 turns context.  Parallelized across spans (max_workers=4).

Falls back to single-span mode when Phase 1 fails or returns no spans,
preserving backward compatibility with mock / test scenarios.
"""

from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from core.errors import ToolCallProtocolError
from core.interfaces import LLM
from core.models import CandidateMemory, RequestContext
from extraction.json_fallback import (
    ExtractionEnvelope,
    build_extraction_envelope_schema,
    build_extraction_fallback_prompt,
)
from extraction.prefetch import MemoryPrefetcher, PrefetchResult
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import (
    build_extraction_tools,
    build_tool_to_category,
    parse_tool_call,
)

logger = logging.getLogger(__name__)

_DEFAULT_SCHEMA_REGISTRY: SchemaRegistry | None = None
EXTRACTION_TOOLS: list[dict] = []
TOOL_TO_CATEGORY: dict[str, tuple[str, str]] = {}
_TOOL_RESULT_STUB_BLOCK = re.compile(
    r"\[ogmemory-tool-result v=2\]\r?\n.*?\r?\n\[/ogmemory-tool-result\]",
    re.DOTALL,
)
_USER_PREFERENCE_CUE = re.compile(
    r"(?:\b(?:i|i'm|i am|i'd|i would|my)\b[^.!?;\n]{0,100}"
    r"\b(?:prefer|like|love|enjoy|dislike|hate|want|choose|use|favorite|favourite)\w*\b)"
    r"|(?:\b(?:prefer|like|love|enjoy|dislike|hate|want|choose|use)\w*\b"
    r"[^.!?;\n]{0,60}\b(?:i|my)\b)",
    re.IGNORECASE,
)
_CJK_USER_PREFERENCE_CUE = re.compile(
    r"(?:我|本人|俺|咱|私).{0,40}"
    r"(?:更喜欢|不喜欢|喜欢|偏好|更爱|爱用|常用|习惯|倾向|讨厌|希望|想要|选择)",
)
_USER_PROFILE_CUE = re.compile(
    r"\b(?:i am|i'm|my name is|i work as|i work at|i work in|i live in|"
    r"i am from|i'm from|i study|i have|my (?:job|occupation|role|location|"
    r"age|background) is)\b",
    re.IGNORECASE,
)
_CJK_USER_PROFILE_CUE = re.compile(
    r"(?:我是|我叫|我住在|我来自|我有|我的(?:名字|职业|工作|住址|所在地|年龄|背景)|"
    r"我在.{0,20}(?:工作|学习))",
)
_USER_SELF_REPORT_QUESTION = re.compile(
    r"^\s*(?:do|did|can|could|would|should|is|are|am|was|were|have|has|will)\b"
    r"|\b(?:whether|if)\s+i\b",
    re.IGNORECASE,
)
_CJK_SELF_REPORT_QUESTION = re.compile(r"(?:是否|是不是|能否|吗|么)")
_USER_SENTENCE_BOUNDARY = re.compile(r"[.!?;。！？；\n]+")
_LATIN_SUPPORT_TERM = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_SUPPORT_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,}")
_GENERIC_SENSITIVE_TERMS = frozenset({
    "a", "am", "an", "and", "are", "choose", "chooses", "content",
    "dislike", "dislikes", "enjoy", "enjoys", "favorite", "favourite",
    "for", "hate", "hates", "i", "identity", "information", "is", "like",
    "likes", "love", "loves", "me", "my", "of", "person", "preference",
    "preferences", "prefer", "preferred", "prefers", "profile", "result",
    "results", "message", "messages", "assistant", "the", "to", "tool", "tools",
    "output", "outputs", "use", "uses", "user", "using", "want", "wants",
    "would",
})
_GENERIC_CJK_SENSITIVE_PHRASES = (
    "更喜欢", "不喜欢", "用户", "本人", "喜欢", "偏好", "更爱", "爱用",
    "常用", "习惯", "倾向", "讨厌", "希望", "想要", "选择", "身份", "信息",
    "画像", "个人", "我", "俺", "咱", "私",
)


def _direct_sensitive_support_terms(value: str) -> set[str]:
    """Extract topic-bearing terms while discarding generic identity wording."""
    normalized = value.casefold().replace("_", " ")
    terms = {
        term
        for term in _LATIN_SUPPORT_TERM.findall(normalized)
        if (
            len(term) >= 3
            and not term.isdigit()
            and term not in _GENERIC_SENSITIVE_TERMS
        )
    }
    cjk_text = normalized
    for phrase in _GENERIC_CJK_SENSITIVE_PHRASES:
        cjk_text = cjk_text.replace(phrase, " ")
    for run in _CJK_SUPPORT_RUN.findall(cjk_text):
        terms.update(run[index:index + 2] for index in range(len(run) - 1))
    return terms


def _has_direct_topic_overlap(
    candidate_terms: set[str],
    evidence_terms: set[str],
    *,
    strict_single_term: bool,
) -> bool:
    overlap = candidate_terms & evidence_terms
    if any(_CJK_SUPPORT_RUN.fullmatch(term) for term in overlap):
        return True
    latin_overlap = [term for term in overlap if term.isascii()]
    minimum_single_length = 6 if strict_single_term else 4
    return bool(
        any(len(term) >= minimum_single_length for term in latin_overlap)
        or len(latin_overlap) >= 2
    )


def _sensitive_candidate_has_direct_user_support(
    candidate: CandidateMemory,
    user_text: str,
    evidence_quote: str | None,
) -> bool:
    """Require a declarative first-person statement with matching topic terms."""
    candidate_terms = _direct_sensitive_support_terms("\n".join((
        candidate.routing_key,
        candidate.abstract,
        candidate.overview,
        candidate.content,
    )))
    primary_candidate_terms = _direct_sensitive_support_terms("\n".join((
        candidate.routing_key,
        candidate.abstract,
    )))
    normalized_quote = " ".join(
        str(evidence_quote or "").split()
    ).casefold().strip(" .!?;。！？；\"'“”‘’")
    if not candidate_terms or (normalized_quote and not primary_candidate_terms):
        return False
    for segment in _USER_SENTENCE_BOUNDARY.split(user_text):
        sentence = segment.strip()
        if not sentence:
            continue
        if (
            _USER_SELF_REPORT_QUESTION.search(sentence)
            or _CJK_SELF_REPORT_QUESTION.search(sentence)
        ):
            continue
        if candidate.category == "profile":
            explicit_self_report = bool(
                _USER_PROFILE_CUE.search(sentence)
                or _CJK_USER_PROFILE_CUE.search(sentence)
            )
        else:
            explicit_self_report = bool(
                _USER_PREFERENCE_CUE.search(sentence)
                or _CJK_USER_PREFERENCE_CUE.search(sentence)
            )
        if not explicit_self_report:
            continue
        normalized_sentence = " ".join(sentence.split()).casefold()
        if normalized_quote and normalized_quote not in normalized_sentence:
            continue
        sentence_terms = _direct_sensitive_support_terms(sentence)
        if not _has_direct_topic_overlap(
            candidate_terms,
            sentence_terms,
            strict_single_term=False,
        ):
            continue
        if normalized_quote and not _has_direct_topic_overlap(
            primary_candidate_terms,
            _direct_sensitive_support_terms(normalized_quote),
            strict_single_term=True,
        ):
            continue
        return True
    return False


def _get_default_schema_registry() -> SchemaRegistry:
    global _DEFAULT_SCHEMA_REGISTRY, EXTRACTION_TOOLS, TOOL_TO_CATEGORY
    if _DEFAULT_SCHEMA_REGISTRY is None:
        try:
            registry = SchemaRegistry()
            EXTRACTION_TOOLS = build_extraction_tools(registry)
            TOOL_TO_CATEGORY = build_tool_to_category(registry)
            _DEFAULT_SCHEMA_REGISTRY = registry
        except Exception:
            logger.exception("Default schema registry initialization failed")
            raise
    return _DEFAULT_SCHEMA_REGISTRY


# ---------------------------------------------------------------------------
# Internal data
# ---------------------------------------------------------------------------

@dataclass
class _Span:
    """A contiguous message range identified as containing extractable info."""
    start: int
    end: int
    reason: str = ""
    categories: list[str] = field(default_factory=list)
    message_ids: list[str] = field(default_factory=list)
    safe_user_text: str = ""
    had_externalized_stub: bool = False


@dataclass
class _PreparedSpan:
    """A span with its final Phase 2 prompt prepared for tool-calling."""
    span: _Span
    prompt: str
    prompt_sha: str


@dataclass
class _NativeSpanOutcome:
    """Results from the two native extraction attempts for one span."""
    prepared: _PreparedSpan
    candidates: list[CandidateMemory] = field(default_factory=list)
    protocol_errors: list[ToolCallProtocolError] = field(default_factory=list)


# JSON schema for Phase 1 span identification
_SPAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "reason": {"type": "string"},
                    "categories": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start", "end"],
            },
        },
    },
    "required": ["spans"],
}


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()


def _short_sha(sha: str) -> str:
    return sha[:12]


def _merge_unique_categories(primary: list[str], extra: list[str]) -> list[str]:
    """Merge categories in order so deduped prompts keep all prefetch scopes."""
    merged: list[str] = []
    seen: set[str] = set()
    for category in [*primary, *extra]:
        if not category or category in seen:
            continue
        merged.append(category)
        seen.add(category)
    return merged


# Phase 1 prompt — lightweight, no tool definitions
_SPAN_PROMPT = """\
Analyze this conversation and identify contiguous message ranges (spans) \
that contain personal information worth remembering.

Rules:
- ONLY identify spans from non-assistant messages. Ignore lines starting \
with "[N] assistant:".
- Each span should contain a coherent unit of extractable information.
- Merge adjacent user messages about the same topic into one span.
- Skip pure greetings, acknowledgments, or general knowledge.
- If a "Previously Extracted Context" section is present, DO NOT re-identify \
content already listed there.

For each span, specify:
- start: message index (the [N] prefix number)
- end: message index (inclusive)
- reason: what extractable information this span contains
- categories: likely categories from: profile, preference, entity, event, \
case, pattern, skill, tool

{session_summary_block}\
Conversation:
{conversation}

Return JSON: {{"spans": [{{"start": int, "end": int, "reason": str, \
"categories": [str]}}]}}
If nothing is extractable: {{"spans": []}}"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class Extractor:
    """Two-phase candidate memory extractor.

    Phase 1 uses a cheap JSON-mode LLM call to identify which user message
    ranges contain extractable information.  Phase 2 runs focused tool-call
    extraction per span with surrounding context, parallelized across spans.
    """

    def __init__(
        self,
        llm: LLM,
        prompt_manager: Any | None = None,
        schema_registry: SchemaRegistry | None = None,
        fs: Any | None = None,
        vector_index: Any | None = None,
        embedder: Any | None = None,
        uri_resolver: Any | None = None,
        mode: str = "eager",
        react_timeout_seconds: float = 180.0,
        internal_tool_usage_tracker: Any | None = None,
        activation_enabled: bool = False,
    ):
        """Initialize the Extractor.

        Args:
            llm: LLM instance with complete_with_tools() and complete_json().
            prompt_manager: Optional PromptManager for template-based prompts.
                          If None, auto-creates one from default templates dir.
            schema_registry: Optional SchemaRegistry override for tests or
                           custom memory types. Tool definitions are always
                           generated from SchemaRegistry; no legacy fallback
                           exists. If None, a default registry is loaded from
                           YAML definitions.
            fs: Optional ContextFS for prefetching existing memories.
            vector_index: Optional VectorIndex for prefetching existing memories.
            embedder: Optional Embedder for prefetching existing memories.
            uri_resolver: Optional URIResolver for prefetching existing memories.
            mode: Extraction mode — "eager" (prefetch + single-shot) or
                  "lazy" (ReAct loop with read tools). Defaults to "eager".
            react_timeout_seconds: Maximum seconds for ReAct loop before timeout. Defaults to 180.0.
        """
        self._llm = llm
        if prompt_manager is None:
            from extraction.prompts import PromptManager as _PM
            prompt_manager = _PM()
        self._prompt_manager = prompt_manager
        self._schema_registry = schema_registry or _get_default_schema_registry()
        self._fs = fs
        self._vector_index = vector_index
        self._embedder = embedder
        self._uri_resolver = uri_resolver
        self._mode = mode
        self._react_timeout_seconds = react_timeout_seconds
        self._internal_tool_usage_tracker = internal_tool_usage_tracker
        self._activation_enabled = activation_enabled
        self._extraction_template = "extraction_graph" if activation_enabled else "extraction"
        self._prefetcher: Optional[MemoryPrefetcher] = None
        if fs and vector_index and embedder and uri_resolver and self._schema_registry:
            self._prefetcher = MemoryPrefetcher(fs, vector_index, embedder, self._schema_registry, uri_resolver)
            logger.info(f"Extractor: prefetcher enabled ({mode} mode)")

        self._extraction_tools = (build_extraction_tools(self._schema_registry, activation_enabled=True)
                                  if activation_enabled else build_extraction_tools(self._schema_registry))
        if not self._extraction_tools:
            incompatible = [
                f"{schema.memory_type}:{schema.version}"
                for schema in self._schema_registry.list_incompatible_enabled()
            ]
            detail = f" incompatible={incompatible}" if incompatible else ""
            raise ValueError(f"No compatible extraction tools available.{detail}")
        self._tool_to_category = build_tool_to_category(self._schema_registry)
        self._parse_tool_call_fn = (
            (lambda name, inp: parse_tool_call(name, inp, self._schema_registry, activation_enabled=True))
            if activation_enabled else (lambda name, inp: parse_tool_call(name, inp, self._schema_registry)))
        logger.info(f"Extractor using SchemaRegistry with {len(self._extraction_tools)} tools")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        messages: list[dict],
        ctx: RequestContext,
        session_time: Any | None = None,
        session_summary: str = "",
        tool_stats_text: str = "",
        archive_id: str | None = None,
    ) -> list[CandidateMemory]:
        """Extract candidate memories from conversation messages.

        Args:
            messages: List of message dicts with "role" and "content"
            ctx: RequestContext for this extraction
            session_time: Optional datetime for temporal resolution
            session_summary: Optional summary of previously extracted content
            tool_stats_text: Optional tool usage statistics text
            archive_id: Optional archive_id for provenance tracking

        Returns:
            List of CandidateMemory with confidence >= 0.5
        """
        # No user content to extract from
        user_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") != "assistant"
        ]
        if not user_indices:
            return []

        # Detect language once
        language = self._detect_language(messages)

        # Phase 1: identify spans worth extracting
        spans = self._identify_spans(messages, session_time, session_summary)
        if not spans:
            user_content_len = sum(
                len(m.get("content", "") or "")
                for m in messages if m.get("role") != "assistant"
            )
            _MIN_FALLBACK_LEN = 200
            if user_content_len >= _MIN_FALLBACK_LEN:
                spans = [_Span(
                    start=0,
                    end=len(messages) - 1,
                    reason="fallback: Phase 1 returned no spans",
                    categories=[],
                )]
                logger.info(
                    "Phase 1 empty → fallback full-conversation span [0, %d] "
                    "(user_content_len=%d)",
                    len(messages) - 1, user_content_len,
                )
            else:
                logger.info(
                    "Phase 1 empty, user content too short (%d chars < %d) → skip Phase 2",
                    user_content_len, _MIN_FALLBACK_LEN,
                )
                return []

        # Phase 2: structure each span
        candidates = self._structure_spans(
            spans, messages, language, session_time, session_summary,
            tool_stats_text, ctx, archive_id,
        )

        confidence_threshold = 0.5
        filtered = [c for c in candidates if c.confidence >= confidence_threshold]
        dropped = len(candidates) - len(filtered)
        if dropped > 0:
            for c in candidates:
                if c.confidence < confidence_threshold:
                    logger.info(
                        "Dropped candidate (confidence=%.2f < %.1f): category=%s routing_key=%s abstract=%.80s",
                        c.confidence, confidence_threshold, c.category, c.routing_key, c.abstract,
                    )
            logger.info(
                "Phase 2 confidence filter: %d kept, %d dropped (< %.1f)",
                len(filtered), dropped, confidence_threshold,
            )
        return filtered

    # ------------------------------------------------------------------
    # Phase 1 — Span Identification
    # ------------------------------------------------------------------

    def _identify_spans(
        self,
        messages: list[dict],
        session_time: Any | None,
        session_summary: str,
    ) -> list[_Span]:
        """Use a cheap JSON-mode call to find extractable message ranges."""
        conversation = self._format_indexed_conversation(messages, session_time)

        summary_block = ""
        if session_summary:
            summary_block = (
                "Previously Extracted Context "
                "(DO NOT re-identify these — they are already saved):\n"
                f"{session_summary}\n\n"
            )

        prompt = _SPAN_PROMPT.format(
            conversation=conversation,
            session_summary_block=summary_block,
        )

        try:
            result = self._llm.complete_json(prompt, schema=_SPAN_SCHEMA)
            spans: list[_Span] = []
            n = len(messages)
            for s in result.get("spans", []):
                if n <= 1:
                    span_start = 0
                    span_end = 0
                else:
                    span_start = max(0, int(s.get("start", 0)))
                    span_end = min(n - 1, int(s.get("end", 0)))
                span_message_ids = [
                    m.get("id", "")
                    for m in messages[span_start:span_end + 1]
                    if m.get("id")
                ]
                span = _Span(
                    start=span_start,
                    end=span_end,
                    reason=s.get("reason", ""),
                    categories=s.get("categories", []),
                    message_ids=span_message_ids,
                )
                if span.start <= span.end:
                    spans.append(span)
            if not spans:
                logger.warning("Phase 1 returned 0 valid spans for %d messages", len(messages))
            return spans
        except Exception as exc:
            logger.error("Phase 1 span identification FAILED: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Phase 2 — Span Structuring
    # ------------------------------------------------------------------

    def _structure_spans(
        self,
        spans: list[_Span],
        messages: list[dict],
        language: str,
        session_time: Any | None,
        session_summary: str,
        tool_stats_text: str,
        ctx: RequestContext,
        archive_id: str | None = None,
    ) -> list[CandidateMemory]:
        """Run focused tool-call extraction for each span (dual-run with merge)."""
        prepared_spans = self._prepare_unique_spans(
            spans, messages, language,
            session_time, session_summary, tool_stats_text, ctx,
        )
        if not prepared_spans:
            return []

        all_candidates: list[CandidateMemory] = []

        # Single span → skip thread pool overhead
        if len(prepared_spans) == 1:
            return self._structure_prepared_span_dual(
                prepared_spans[0], ctx, archive_id
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}
            outcomes = [
                _NativeSpanOutcome(prepared=prepared)
                for prepared in prepared_spans
            ]
            for span_index, prepared in enumerate(prepared_spans):
                f1 = pool.submit(copy_context().run, self._structure_prepared_span, prepared, ctx, archive_id)
                f2 = pool.submit(copy_context().run, self._structure_prepared_span, prepared, ctx, archive_id, temperature=0)
                futures[f1] = (span_index, None)
                futures[f2] = (span_index, 0)

            for future in as_completed(futures):
                span_index, temperature = futures[future]
                outcome = outcomes[span_index]
                span = outcome.prepared.span
                try:
                    result = future.result()
                except ToolCallProtocolError as exc:
                    outcome.protocol_errors.append(exc)
                    logger.warning(
                        "Span %d-%d native extraction protocol failure: "
                        "temperature=%s error=%s",
                        span.start, span.end, temperature, exc,
                    )
                    continue
                except Exception as exc:
                    logger.error(
                        "Span %d-%d structuring FAILED: %s",
                        span.start, span.end, exc, exc_info=True,
                    )
                    raise
                outcome.candidates.extend(result)

            for outcome in outcomes:
                all_candidates.extend(
                    self._resolve_native_span_outcome(outcome, ctx, archive_id)
                )

        return self._merge_dual_run(all_candidates)

    def _structure_prepared_span_dual(
        self,
        prepared: _PreparedSpan,
        ctx: RequestContext,
        archive_id: str | None,
    ) -> list[CandidateMemory]:
        """Run the existing dual native extraction and fallback only if both fail."""
        outcome = _NativeSpanOutcome(prepared=prepared)
        for temperature in (None, 0):
            try:
                outcome.candidates.extend(
                    self._structure_prepared_span(
                        prepared, ctx, archive_id, temperature=temperature
                    )
                )
            except ToolCallProtocolError as exc:
                outcome.protocol_errors.append(exc)
                logger.warning(
                    "Span %d-%d native extraction protocol failure: "
                    "temperature=%s error=%s",
                    prepared.span.start,
                    prepared.span.end,
                    temperature,
                    exc,
                )
        return self._merge_dual_run(
            self._resolve_native_span_outcome(outcome, ctx, archive_id)
        )

    def _resolve_native_span_outcome(
        self,
        outcome: _NativeSpanOutcome,
        ctx: RequestContext,
        archive_id: str | None,
    ) -> list[CandidateMemory]:
        """Keep any native success; fallback once only after two protocol failures."""
        if outcome.candidates:
            if outcome.protocol_errors:
                logger.info(
                    "Span %d-%d using %d native candidates; JSON fallback skipped "
                    "after %d protocol failure(s)",
                    outcome.prepared.span.start,
                    outcome.prepared.span.end,
                    len(outcome.candidates),
                    len(outcome.protocol_errors),
                )
            return outcome.candidates

        if len(outcome.protocol_errors) == 2:
            logger.warning(
                "Span %d-%d both native extraction attempts failed protocol; "
                "using one JSON fallback",
                outcome.prepared.span.start,
                outcome.prepared.span.end,
            )
            return self._structure_span_json_fallback(
                outcome.prepared, ctx, archive_id
            )

        if outcome.protocol_errors:
            raise outcome.protocol_errors[0]
        return []

    @staticmethod
    def _merge_dual_run(candidates: list[CandidateMemory]) -> list[CandidateMemory]:
        """Merge dual-run results by (category, routing_key).

        For each group, keep the candidate with higher confidence and richer
        content. Merges when/who/where fields by taking non-empty values.
        """
        merged: dict[tuple[str, str], CandidateMemory] = {}
        from extraction.relation_proposals import attach, take
        proposals = take(candidates)
        for c in candidates:
            key = (c.category, c.routing_key)
            if key not in merged:
                merged[key] = c
                continue
            existing = merged[key]
            # Pick higher confidence; if equal, pick richer content
            if c.confidence > existing.confidence:
                # Keep existing content if it's richer
                if len(existing.content) > len(c.content):
                    c.content = existing.content
                if not c.when and existing.when:
                    c.when = existing.when
                if not c.who and existing.who:
                    c.who = existing.who
                if not c.where and existing.where:
                    c.where = existing.where
                merged[key] = c
            elif len(c.content) > len(existing.content) * 1.5:
                # New content is significantly richer → upgrade
                if not c.when and existing.when:
                    c.when = existing.when
                if not c.who and existing.who:
                    c.who = existing.who
                if not c.where and existing.where:
                    c.where = existing.where
                merged[key] = c
        result = list(merged.values())
        attach(result, proposals)
        return result

    def _prepare_unique_spans(
        self,
        spans: list[_Span],
        messages: list[dict],
        language: str,
        session_time: Any | None,
        session_summary: str,
        tool_stats_text: str,
        ctx: RequestContext,
    ) -> list[_PreparedSpan]:
        """Build final Phase 2 prompts and drop duplicate prompt_sha work."""
        prepared_spans: list[_PreparedSpan] = []
        seen: dict[str, _PreparedSpan] = {}
        skipped = 0
        for span in spans:
            prepared = self._prepare_span(
                span, messages, language,
                session_time, session_summary, tool_stats_text, ctx,
            )
            existing = seen.get(prepared.prompt_sha)
            if existing is not None:
                if existing.prompt == prepared.prompt:
                    original_categories = list(existing.span.categories)
                    existing.span.categories = _merge_unique_categories(
                        original_categories,
                        prepared.span.categories,
                    )
                    existing.span.message_ids = _merge_unique_categories(
                        existing.span.message_ids,
                        prepared.span.message_ids,
                    )
                    safe_user_texts = [
                        value
                        for value in (
                            existing.span.safe_user_text,
                            prepared.span.safe_user_text,
                        )
                        if value
                    ]
                    existing.span.safe_user_text = "\n".join(
                        dict.fromkeys(safe_user_texts)
                    )
                    existing.span.had_externalized_stub = bool(
                        existing.span.had_externalized_stub
                        or prepared.span.had_externalized_stub
                    )
                    skipped += 1
                    logger.info(
                        "Phase 2 duplicate prompt skipped: session=%s trace=%s "
                        "prompt_sha=%s duplicate_span=[%d-%d] original_span=[%d-%d] "
                        "duplicate_categories=%s original_categories=%s merged_categories=%s",
                        getattr(ctx, "session_id", ""),
                        getattr(ctx, "trace_id", ""),
                        _short_sha(prepared.prompt_sha),
                        prepared.span.start,
                        prepared.span.end,
                        existing.span.start,
                        existing.span.end,
                        prepared.span.categories,
                        original_categories,
                        existing.span.categories,
                    )
                    continue
                logger.warning(
                    "Phase 2 prompt_sha collision detected; keeping both spans: "
                    "session=%s trace=%s prompt_sha=%s span=[%d-%d] existing_span=[%d-%d]",
                    getattr(ctx, "session_id", ""),
                    getattr(ctx, "trace_id", ""),
                    _short_sha(prepared.prompt_sha),
                    prepared.span.start,
                    prepared.span.end,
                    existing.span.start,
                    existing.span.end,
                )

            seen[prepared.prompt_sha] = prepared
            prepared_spans.append(prepared)

        if skipped:
            logger.info(
                "Phase 2 prompt dedupe: session=%s trace=%s spans=%d unique=%d skipped=%d",
                getattr(ctx, "session_id", ""),
                getattr(ctx, "trace_id", ""),
                len(spans),
                len(prepared_spans),
                skipped,
            )
        return prepared_spans

    def _prepare_span(
        self,
        span: _Span,
        messages: list[dict],
        language: str,
        session_time: Any | None,
        session_summary: str,
        tool_stats_text: str,
        ctx: RequestContext,
    ) -> _PreparedSpan:
        """Prepare the final prompt used for one Phase 2 extraction request."""
        span_messages = messages[span.start:span.end + 1]
        span.safe_user_text = "\n".join(
            self._content_to_str(message.get("content", ""))
            for message in span_messages
            if message.get("role") == "user"
        ).strip()
        prompt = self._build_span_prompt(
            span, messages, language,
            session_time, session_summary, tool_stats_text, ctx,
        )
        if not self._use_lazy_mode():
            prompt = self._with_prefetch_context(span, prompt, ctx)
        prompt_sha = _prompt_sha(prompt)
        return _PreparedSpan(span=span, prompt=prompt, prompt_sha=prompt_sha)

    def _use_lazy_mode(self) -> bool:
        """Return whether this extractor can run the lazy ReAct path."""
        return bool(self._mode == "lazy" and self._fs and self._schema_registry and self._uri_resolver)

    def _structure_prepared_span(
        self,
        prepared: _PreparedSpan,
        ctx: RequestContext,
        archive_id: str | None = None,
        temperature: float | None = None,
    ) -> list[CandidateMemory]:
        """Extract structured candidates from an already prepared span prompt."""
        span = prepared.span
        if self._use_lazy_mode():
            return self._structure_span_lazy(span, prepared.prompt, ctx, archive_id)
        return self._structure_span_eager(span, prepared.prompt, ctx, prepared.prompt_sha, archive_id, temperature=temperature)

    def _build_span_prompt(
        self,
        span: _Span,
        messages: list[dict],
        language: str,
        session_time: Any | None,
        session_summary: str,
        tool_stats_text: str,
        ctx: RequestContext,
    ) -> str:
        """Build the base Phase 2 prompt for a span before eager prefetch."""
        # Focused context: +/- 3 turns around the span
        ctx_start = max(0, span.start - 3)
        ctx_end = min(len(messages) - 1, span.end + 3)
        focused = messages[ctx_start: ctx_end + 1]
        span.had_externalized_stub = any(
            _TOOL_RESULT_STUB_BLOCK.search(
                self._content_to_str(message.get("content", ""))
            )
            is not None
            for message in focused
        )

        conversation = self._format_conversation(
            focused, session_time,
            session_summary="",
            tool_stats_text=(
                tool_stats_text if span.end >= len(messages) - 3 else ""
            ),
        )
        return self._build_prompt(language, conversation, session_summary, ctx=ctx)

    def _structure_span_eager(
        self,
        span: _Span,
        prompt: str,
        ctx: RequestContext,
        prompt_sha: str,
        archive_id: str | None = None,
        temperature: float | None = None,
    ) -> list[CandidateMemory]:
        """Eager mode: prefetch + single-shot extraction."""
        tool_calls = self._llm.complete_with_tools(
            prompt=prompt,
            tools=self._extraction_tools,
            tool_choice="required",
            temperature=temperature,
        )

        if not tool_calls:
            logger.warning(
                "Phase 2 zero tool_calls: session=%s trace=%s span=[%d-%d] "
                "categories=%s prompt_len=%d prompt_sha=%s",
                getattr(ctx, "session_id", ""),
                getattr(ctx, "trace_id", ""),
                span.start,
                span.end,
                span.categories,
                len(prompt),
                _short_sha(prompt_sha),
            )

        candidates: list[CandidateMemory] = []
        proposals = []
        from extraction.relation_proposals import attach, parse_proposal
        for call in tool_calls:
            if self._activation_enabled and call.get("tool") == "extract_relation":
                proposal = parse_proposal(call.get("input"))
                if proposal is not None:
                    proposals.append(proposal)
                continue
            c = self._to_candidate(call)
            if c is not None:
                candidates.append(c)
        attach(candidates, proposals)
        if not candidates:
            raise ToolCallProtocolError(
                "native tool calls contained no schema-valid extraction operation",
                tool_choice="required",
            )
        self._assign_provenance(candidates, span, archive_id)
        return candidates

    def _structure_span_json_fallback(
        self,
        prepared: _PreparedSpan,
        ctx: RequestContext,
        archive_id: str | None,
    ) -> list[CandidateMemory]:
        """Run one schema-validated JSON extraction fallback for a span."""
        fallback_prompt = build_extraction_fallback_prompt(
            prepared.prompt, self._extraction_tools
        )
        raw = self._llm.complete_json(
            fallback_prompt,
            schema=build_extraction_envelope_schema(self._extraction_tools),
        )
        envelope = ExtractionEnvelope.model_validate(raw)
        candidates: list[CandidateMemory] = []
        proposals = []
        from extraction.relation_proposals import attach, parse_proposal
        for operation in envelope.operations:
            if self._activation_enabled and operation.tool == "extract_relation":
                proposal = parse_proposal(operation.input)
                if proposal is not None:
                    proposals.append(proposal)
                continue
            candidate = self._to_candidate(
                {"tool": operation.tool, "input": operation.input}
            )
            if candidate is not None:
                candidates.append(candidate)
        attach(candidates, proposals)
        if envelope.operations and not candidates:
            raise ToolCallProtocolError(
                "JSON fallback contained no schema-valid extraction operation"
            )
        self._assign_provenance(candidates, prepared.span, archive_id)
        logger.info(
            "Span %d-%d JSON extraction fallback produced %d candidates: "
            "session=%s trace=%s prompt_sha=%s",
            prepared.span.start,
            prepared.span.end,
            len(candidates),
            getattr(ctx, "session_id", ""),
            getattr(ctx, "trace_id", ""),
            _short_sha(prepared.prompt_sha),
        )
        return candidates

    def _structure_span_lazy(
        self,
        span: _Span,
        prompt: str,
        ctx: RequestContext,
        archive_id: str | None = None,
        focused_messages: list[dict] | None = None,
    ) -> list[CandidateMemory]:
        """Lazy mode: ReAct loop with read tools, fallback to eager on failure."""
        try:
            from extraction.extraction_react_loop import ExtractionReActLoop

            # Use lazy prefetch: URIs only, no content reading
            prefetch_result = None
            if self._prefetcher is not None:
                prefetch_result = self._collect_prefetch_context(
                    span.categories, ctx, focused_messages=focused_messages, lazy=True,
                )

            loop = ExtractionReActLoop(
                llm=self._llm,
                fs=self._fs,
                registry=self._schema_registry,
                uri_resolver=self._uri_resolver,
                prompt_manager=self._prompt_manager,
                prefetcher=self._prefetcher,
                timeout_seconds=self._react_timeout_seconds,
                activation_enabled=self._activation_enabled,
                internal_tool_usage_tracker=self._internal_tool_usage_tracker,
            )
            result = loop.run(prompt, ctx, prefetch_result)
            logger.info(
                f"ReAct loop completed: {len(result.candidates)} candidates, "
                f"{result.iterations} iterations, {len(result.read_uris)} reads"
            )
            candidates = result.candidates
            self._assign_provenance(candidates, span, archive_id)
            return candidates
        except Exception as e:
            logger.error(f"ReAct loop failed, falling back to eager mode: {e}")
            prompt = self._with_prefetch_context(span, prompt, ctx)
            prompt_sha = _prompt_sha(prompt)
            return self._structure_span_eager(span, prompt, ctx, prompt_sha, archive_id)

    def _with_prefetch_context(
        self,
        span: _Span,
        prompt: str,
        ctx: RequestContext,
    ) -> str:
        """Inject eager prefetch context into a Phase 2 prompt when available."""
        if self._prefetcher is None:
            return prompt
        prefetch_result = self._collect_prefetch_context(span.categories, ctx)
        if not prefetch_result or not prefetch_result.messages:
            return prompt
        prefetch_block = (
            "== Existing Memories (DO NOT re-extract these — update or extend them) ==\n"
            + "\n".join(prefetch_result.messages)
            + "\n==\n"
        )
        return prefetch_block + "\n" + prompt

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _collect_prefetch_context(
        self,
        categories: list[str],
        ctx: RequestContext,
        *,
        focused_messages: list[dict] | None = None,
        lazy: bool = False,
    ) -> PrefetchResult | None:
        """Collect prefetch context for given categories.

        Args:
            lazy: If True, collect URI hints only (no content reading).
        """
        if self._prefetcher is None:
            return None
        conv_text: str | None = None
        if focused_messages:
            parts = []
            for msg in focused_messages:
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
            if parts:
                conv_text = " ".join(parts)[:500]
        try:
            return self._prefetcher.prefetch_for_span(categories, ctx, conversation_text=conv_text, lazy=lazy)
        except Exception as e:
            logger.error(f"Prefetch failed, continuing without: {e}")
            return None

    def _build_prompt(
        self,
        language: str,
        conversation: str,
        session_summary: str = "",
        ctx: RequestContext | None = None,
    ) -> str:
        """Build extraction prompt from YAML template (Eager mode)."""
        system_prompt = self._prompt_manager.render(
            self._extraction_template, "system_prompt",
            session_summary=session_summary,
        )
        examples = self._prompt_manager.render(self._extraction_template, "examples")
        output_instruction = self._prompt_manager.render(
            self._extraction_template, "output_instruction",
            output_language=language,
        )

        # Identity anchoring from template
        identity_block = ""
        if ctx is not None:
            identity_block = self._prompt_manager.render(
                "extraction", "identity_block",
                user_id=ctx.user_id,
            )

        relation_instruction = ""
        if self._activation_enabled:
            relation_instruction = self._prompt_manager.render(
                self._extraction_template, "lazy_tools_instruction")
        # Keep graph-off input bytes unchanged; graph-on uses the archived
        # prompt contract, including examples and relationship guidance.
        relation_block = relation_instruction + "\n\n" if relation_instruction else ""
        return (
            f"{system_prompt}\n\n{identity_block}\n\n{relation_block}{examples}\n\n"
            f"{output_instruction}\n\nConversation:\n{conversation}"
        )

    @staticmethod
    def _content_to_str(content) -> str:
        """Normalize message content to plain string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return " ".join(parts).strip()
        return str(content) if content else ""

    def _detect_language(self, messages: list[dict]) -> str:
        """Detect language from user messages."""
        from core.language import detect_language

        user_text = " ".join(
            self._content_to_str(m.get("content", ""))
            for m in messages
            if m.get("role") == "user"
        )
        return detect_language(user_text)

    def _format_conversation(
        self,
        messages: list[dict],
        session_time: Any | None = None,
        session_summary: str = "",
        tool_stats_text: str = "",
    ) -> str:
        """Format messages into conversation text with session time header."""
        if session_time is None:
            session_time = datetime.now()

        time_str = session_time.strftime("%Y-%m-%d %H:%M")
        day_of_week = session_time.strftime("%A")

        lines = [f"**Session Time:** {time_str} ({day_of_week})"]
        lines.append(
            "Relative times (e.g., 'last week', 'next month') are based "
            "on Session Time, not today."
        )

        if session_summary:
            lines.append("")
            lines.append(
                "## Previously Extracted Context "
                "(DO NOT re-extract these — they are already saved)"
            )
            lines.append(session_summary)

        if tool_stats_text:
            lines.append("")
            lines.append(tool_stats_text)

        lines.append("")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = self._content_to_str(msg.get("content", ""))
            if role == "assistant":
                lines.append(
                    f"assistant [do not extract from this]: {content}"
                )
            else:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _format_indexed_conversation(
        self,
        messages: list[dict],
        session_time: Any | None = None,
    ) -> str:
        """Format messages with [N] index prefixes for span identification."""
        if session_time is None:
            session_time = datetime.now()

        time_str = session_time.strftime("%Y-%m-%d %H:%M")
        day_of_week = session_time.strftime("%A")

        lines = [f"Session Time: {time_str} ({day_of_week})"]
        lines.append("")

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = self._content_to_str(msg.get("content", ""))
            if role == "assistant":
                lines.append(f"[{i}] assistant: {content}")
            else:
                lines.append(f"[{i}] {role}: {content}")
        return "\n".join(lines)

    def _assign_provenance(
        self,
        candidates: list[CandidateMemory],
        span: _Span | None,
        archive_id: str | None,
    ) -> None:
        """Assign provenance ID to all candidates from a single span."""
        if not span:
            return
        filter_sensitive = span.had_externalized_stub
        if filter_sensitive:
            retained = []
            for candidate in candidates:
                if candidate.category not in {"profile", "preference"}:
                    retained.append(candidate)
                    continue
                evidence_quote = " ".join(
                    str(candidate.evidence_quote or "").split()
                ).casefold()
                supported_by_user = bool(
                    (evidence_quote or candidate.category == "preference")
                    and _sensitive_candidate_has_direct_user_support(
                        candidate,
                        span.safe_user_text,
                        evidence_quote or None,
                    )
                )
                if supported_by_user:
                    retained.append(candidate)
                    continue
                logger.warning(
                    "Dropped sensitive candidate without selected-span user support: "
                    "category=%s routing_key=%s",
                    candidate.category,
                    candidate.routing_key,
                )
            candidates[:] = retained
        if archive_id and span.message_ids:
            from core.provenance_resolver import ProvenanceResolver
            prov_id = ProvenanceResolver.build_id(
                "archive", archive_id, span.message_ids
            )
            for c in candidates:
                c.provenance_ids.append(prov_id)

    def _to_candidate(self, tool_call: dict) -> CandidateMemory | None:
        """Convert tool call result to CandidateMemory."""
        tool_name = tool_call.get("tool", "")
        input_data = tool_call.get("input", {})
        if "raw" in input_data and len(input_data) == 1:
            logger.warning(
                "Skipping tool call with unparsed raw arguments: tool=%s raw=%.200s",
                tool_name, input_data["raw"],
            )
            return None
        result = self._parse_tool_call_fn(tool_name, input_data)
        if result is None:
            return None
        _, _, candidate = result
        return candidate
