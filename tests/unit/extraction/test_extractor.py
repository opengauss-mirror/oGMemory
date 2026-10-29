"""Unit tests for Extractor tool-use implementation.

Tests verify:
- Empty results when no tool calls
- Single preference extraction
- Multiple types extraction
- Skill extraction with agent scope
- Pattern extraction with agent scope
- Case extraction with agent scope
- Low confidence filtering
- Tool name to category mapping
- Profile routing key defaults to category
"""

import hashlib
import logging
from unittest.mock import Mock

import pytest

from core.errors import LLMCallError, LLMOutputTruncatedError, ToolCallProtocolError
from core.models import RequestContext, CandidateMemory
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import build_extraction_tools, build_tool_to_category
from extraction.tools import Extractor, _PreparedSpan, _SPAN_SCHEMA, _Span
from providers.llm.mock_llm import MockLLM


class TestExtractorEmptyResults:
    """Tests for Extractor with no tool calls."""

    @pytest.mark.parametrize("original_error", [None, LLMCallError("output truncated")])
    def test_lazy_llm_failure_is_not_successful_empty_extraction(self, monkeypatch, original_error):
        from types import SimpleNamespace
        loop = Mock()
        loop.run.return_value = SimpleNamespace(
            finish_reason="llm_error", llm_error=original_error, candidates=[],
        )
        monkeypatch.setattr(
            "extraction.extraction_react_loop.ExtractionReActLoop",
            Mock(return_value=loop),
        )
        extractor = Extractor(MockLLM(), mode="lazy")
        extractor._structure_span_eager = Mock()
        ctx = RequestContext(account_id="a", user_id="u", agent_id="g",
                             session_id="s", trace_id="t")
        with pytest.raises(LLMCallError) as error:
            extractor._structure_span_lazy(_Span(start=0, end=0), "prompt", ctx)
        if original_error is not None:
            assert error.value is original_error
        extractor._structure_span_eager.assert_not_called()

    @pytest.mark.parametrize("message_count", [1, 3])
    def test_span_prompt_and_schema_bound_outer_message_indices(self, message_count):
        llm = MockLLM()
        llm.complete_json = Mock(return_value={"spans": []})
        extractor = Extractor(llm)
        messages = [{"role": "user", "content": "Alice: Hello\nBob: Hi\nAlice: I paint."}
                    for _ in range(message_count)]
        assert extractor._identify_spans(messages, None, "") == []
        prompt = llm.complete_json.call_args.args[0]
        schema = llm.complete_json.call_args.kwargs["schema"]
        assert f"exactly {message_count} indexed messages" in prompt
        assert "INSIDE one message are not additional messages" in prompt
        spans = schema["properties"]["spans"]
        assert spans["maxItems"] == message_count
        for key in ("start", "end"):
            assert spans["items"]["properties"][key]["minimum"] == 0
            assert spans["items"]["properties"][key]["maximum"] == message_count - 1
        assert "maxItems" not in _SPAN_SCHEMA["properties"]["spans"]
        assert "maximum" not in _SPAN_SCHEMA["properties"]["spans"]["items"]["properties"]["start"]

    def test_default_lazy_react_timeout_is_long_enough_for_slow_tool_extraction(self):
        """Default ReAct timeout gives slow tool-calling extraction room to finish."""
        extractor = Extractor(MockLLM())

        assert extractor._react_timeout_seconds == 180.0

    def test_extract_empty_when_no_tool_calls(self):
        """LLM returns no tool calls → empty result."""
        llm = MockLLM()
        llm._mock_tool_calls = []

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = extractor.extract(messages, ctx)

        assert result == []

    def test_extract_filters_low_confidence(self):
        """Candidates with confidence < 0.5 are filtered out."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_preference",
                "input": {
                    "routing_key": "coffee",
                    "abstract": "Likes coffee",
                    "overview": "Overview",
                    "content": "Content",
                    "confidence": 0.3,  # Below threshold
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "I like coffee"}]
        result = extractor.extract(messages, ctx)

        assert result == []

    def test_extract_keeps_mid_confidence_candidate(self):
        """Candidates aligned with the 0.5 prompt/config threshold are kept."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_preference",
                "input": {
                    "routing_key": "coffee",
                    "abstract": "Likes coffee",
                    "overview": "Overview",
                    "content": "Content",
                    "confidence": 0.55,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "I like coffee"}]
        result = extractor.extract(messages, ctx)

        assert result
        assert all(candidate.confidence >= 0.5 for candidate in result)


class TestExtractorPhase2PromptDedupe:
    """Tests for skipping duplicate Phase 2 prompts before tool-calling."""

    def _make_prompt_manager(self):
        prompt_manager = Mock()
        prompt_manager.render.side_effect = (
            lambda section, key, **kwargs: f"{section}:{key}:{kwargs.get('output_language', '')}"
        )
        return prompt_manager

    def test_duplicate_prompt_sha_only_calls_tools_once(self, caplog):
        """Duplicate spans that build the same prompt run one tool-calling request."""
        llm = Mock()
        span_response = {
            "spans": [
                {"start": 0, "end": 0, "reason": "event", "categories": ["event"]},
                {"start": 0, "end": 0, "reason": "profile", "categories": ["profile"]},
            ]
        }
        llm.complete_json.side_effect = lambda _prompt, schema: (
            {"operations": []}
            if "operations" in schema.get("properties", {})
            else span_response
        )
        llm.complete_with_tools.return_value = []
        extractor = Extractor(llm, prompt_manager=self._make_prompt_manager())
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        with caplog.at_level(logging.INFO):
            result = extractor.extract(
                [{"role": "user", "content": "I moved to Paris in 2024."}],
                ctx,
            )

        assert result == []
        assert llm.complete_with_tools.call_count == 2  # dual-run: 2 calls per unique span
        assert "Phase 2 profile spans coalesced" in caplog.text

    def test_duplicate_prompt_merges_categories_for_lazy_prefetch(self):
        """Deduped lazy spans keep categories from skipped duplicate prompts."""
        llm = Mock()
        extractor = Extractor(llm, prompt_manager=self._make_prompt_manager())
        extractor._mode = "lazy"
        extractor._fs = object()
        extractor._schema_registry = object()
        extractor._uri_resolver = object()
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        prepared = extractor._prepare_unique_spans(
            [
                _Span(start=0, end=0, reason="event", categories=["event"]),
                _Span(start=0, end=0, reason="profile", categories=["profile", "event"]),
            ],
            [{"role": "user", "content": "I moved to Paris in 2024."}],
            "English",
            None,
            "",
            "",
            ctx,
        )

        assert len(prepared) == 1
        assert prepared[0].span.categories == ["event", "profile"]

    def test_prepared_prompt_sha_stores_full_hash(self):
        """Prepared spans keep the full SHA-256 for dedupe; logs may truncate it."""
        llm = Mock()
        extractor = Extractor(llm, prompt_manager=self._make_prompt_manager())
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        prepared = extractor._prepare_span(
            _Span(start=0, end=0),
            [{"role": "user", "content": "I moved to Paris in 2024."}],
            "English",
            None,
            "",
            "",
            ctx,
        )

        assert prepared.prompt_sha == hashlib.sha256(
            prepared.prompt.encode("utf-8", errors="replace")
        ).hexdigest()
        assert len(prepared.prompt_sha) == 64

    def test_distinct_prompt_sha_calls_tools_for_each_span(self):
        """Different focused contexts are not dropped by prompt dedupe."""
        llm = Mock()
        span_response = {
            "spans": [
                {"start": 0, "end": 0, "reason": "first", "categories": ["event"]},
                {"start": 8, "end": 8, "reason": "second", "categories": ["event"]},
            ]
        }
        llm.complete_json.side_effect = lambda _prompt, schema: (
            {"operations": []}
            if "operations" in schema.get("properties", {})
            else span_response
        )
        llm.complete_with_tools.return_value = []
        extractor = Extractor(llm, prompt_manager=self._make_prompt_manager())
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(10)
        ]

        result = extractor.extract(messages, ctx)

        assert result == []
        assert llm.complete_with_tools.call_count == 4  # dual-run: 2 spans × 2 runs


class TestExtractorToolCallFallback:
    """Tests for native-first extraction with one JSON fallback per span."""

    @staticmethod
    def _ctx() -> RequestContext:
        return RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

    @staticmethod
    def _preference(routing_key: str = "coffee", confidence: float = 0.9) -> dict:
        return {
            "tool": "extract_preference",
            "input": {
                "routing_key": routing_key,
                "abstract": f"Likes {routing_key}",
                "overview": f"Preference for {routing_key}",
                "content": f"User likes {routing_key}",
                "confidence": confidence,
            },
        }

    @staticmethod
    def _prepared() -> _PreparedSpan:
        return _PreparedSpan(
            span=_Span(start=0, end=0, message_ids=["msg-1"]),
            prompt="Extract memory from this conversation",
            prompt_sha="a" * 64,
        )

    def test_two_native_successes_do_not_call_json_fallback(self):
        llm = Mock()
        llm.complete_with_tools.return_value = [self._preference()]
        extractor = Extractor(llm)

        result = extractor._structure_prepared_span_dual(
            self._prepared(), self._ctx(), archive_id=None
        )

        assert len(result) == 1
        assert llm.complete_with_tools.call_count == 2
        llm.complete_json.assert_not_called()

    def test_one_native_success_skips_json_fallback(self):
        llm = Mock()
        llm.complete_with_tools.side_effect = [
            [self._preference("coffee")],
            ToolCallProtocolError("empty native result"),
        ]
        llm.complete_json.return_value = {
            "operations": [self._preference("tea")]
        }
        extractor = Extractor(llm)

        result = extractor._structure_prepared_span_dual(
            self._prepared(), self._ctx(), archive_id=None
        )

        assert len(result) == 1
        assert result[0].routing_key == "coffee"
        llm.complete_json.assert_not_called()

    def test_two_protocol_failures_use_only_one_json_fallback(self):
        llm = Mock()
        llm.complete_with_tools.side_effect = ToolCallProtocolError("empty native result")
        llm.complete_json.return_value = {
            "operations": [self._preference("coffee")]
        }
        extractor = Extractor(llm)

        result = extractor._structure_prepared_span_dual(
            self._prepared(), self._ctx(), archive_id="archive-1"
        )

        assert len(result) == 1
        assert result[0].provenance_ids
        assert llm.complete_with_tools.call_count == 2
        llm.complete_json.assert_called_once()

    def test_json_fallback_allows_empty_operations(self):
        llm = Mock()
        llm.complete_with_tools.side_effect = ToolCallProtocolError("empty native result")
        llm.complete_json.return_value = {"operations": []}
        extractor = Extractor(llm)

        result = extractor._structure_prepared_span_dual(
            self._prepared(), self._ctx(), archive_id=None
        )

        assert result == []
        llm.complete_json.assert_called_once()

    @pytest.mark.parametrize(
        "operation",
        [
            {"tool": "unknown_tool", "input": {}},
            {"tool": "extract_preference", "input": {"routing_key": "coffee"}},
        ],
    )
    def test_json_fallback_rejects_invalid_operations(self, operation):
        llm = Mock()
        llm.complete_with_tools.side_effect = ToolCallProtocolError("empty native result")
        llm.complete_json.return_value = {"operations": [operation]}
        extractor = Extractor(llm)

        with pytest.raises(
            ToolCallProtocolError,
            match="JSON fallback contained no schema-valid extraction operation",
        ):
            extractor._structure_prepared_span_dual(
                self._prepared(), self._ctx(), archive_id=None
            )
        llm.complete_json.assert_called_once()

    @pytest.mark.parametrize("error_type", [ToolCallProtocolError, LLMOutputTruncatedError])
    def test_multiple_spans_keep_native_concurrency_and_fallback_once_per_failed_span(self, error_type):
        llm = Mock()

        def complete_with_tools(*, prompt, **_kwargs):
            if prompt == "failed span":
                raise error_type("incomplete native result")
            return [self._preference("tea")]

        llm.complete_with_tools.side_effect = complete_with_tools
        llm.complete_json.return_value = {
            "operations": [self._preference("coffee")]
        }
        extractor = Extractor(llm)
        prepared = [
            _PreparedSpan(
                span=_Span(start=0, end=0, message_ids=["msg-1"]),
                prompt="failed span",
                prompt_sha="a" * 64,
            ),
            _PreparedSpan(
                span=_Span(start=1, end=1, message_ids=["msg-2"]),
                prompt="successful span",
                prompt_sha="b" * 64,
            ),
        ]
        extractor._prepare_unique_spans = Mock(return_value=prepared)

        result = extractor._structure_spans(
            spans=[item.span for item in prepared],
            messages=[{"role": "user", "content": "memory"}],
            language="en",
            session_time=None,
            session_summary="",
            tool_stats_text="",
            ctx=self._ctx(),
            archive_id=None,
        )

        assert {candidate.routing_key for candidate in result} == {"coffee", "tea"}
        assert llm.complete_with_tools.call_count == 4
        llm.complete_json.assert_called_once()

    def test_real_api_errors_are_not_hidden_when_both_native_runs_fail(self):
        llm = Mock()
        llm.complete_with_tools.side_effect = RuntimeError("rate limited")
        extractor = Extractor(llm)

        with pytest.raises(RuntimeError, match="rate limited"):
            extractor._structure_prepared_span_dual(
                self._prepared(), self._ctx(), archive_id=None
            )

        llm.complete_json.assert_not_called()

    def test_fallback_candidates_still_use_public_confidence_filter(self):
        llm = Mock()
        llm.complete_json.side_effect = [
            {
                "spans": [
                    {"start": 0, "end": 0, "reason": "preference", "categories": ["preference"]}
                ]
            },
            {"operations": [self._preference("coffee", confidence=0.3)]},
        ]
        llm.complete_with_tools.side_effect = ToolCallProtocolError("empty native result")
        extractor = Extractor(llm)

        result = extractor.extract(
            [{"role": "user", "content": "I like coffee."}],
            self._ctx(),
            archive_id="archive-1",
        )

        assert result == []
        assert llm.complete_json.call_count == 2


class TestExtractorSinglePreference:
    """Tests for single preference extraction."""

    def test_extract_single_preference(self):
        """LLM returns extract_preference → CandidateMemory with category=preference."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_preference",
                "input": {
                    "routing_key": "coffee",
                    "abstract": "User's coffee preferences",
                    "overview": "Likes dark roast, drinks 2-3 cups daily",
                    "content": "Prefers dark roast coffee, drinks 2-3 cups per day",
                    "confidence": 0.85,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "I like coffee"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1  # dual-run: same candidate merged
        candidate = result[0]
        assert candidate.category == "preference"
        assert candidate.owner_scope == "user"
        assert candidate.routing_key == "coffee"
        assert candidate.confidence == 0.85


class TestExtractorMultipleTypes:
    """Tests for extracting multiple candidate types."""

    def test_extract_multiple_types(self):
        """LLM returns preference + entity → 2 candidates with correct owner_scope."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_preference",
                "input": {
                    "routing_key": "coffee",
                    "abstract": "Likes coffee",
                    "overview": "Overview",
                    "content": "Content",
                    "confidence": 0.9,
                },
            },
            {
                "tool": "extract_entity",
                "input": {
                    "routing_key": "alice",
                    "abstract": "Person named Alice",
                    "overview": "Overview",
                    "content": "Content",
                    "confidence": 0.8,
                },
            },
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "I like coffee with Alice"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 2  # dual-run merged: 2 unique tools

        # Check preference
        preference = [c for c in result if c.category == "preference"][0]
        assert preference.owner_scope == "user"
        assert preference.routing_key == "coffee"

        # Check entity
        entity = [c for c in result if c.category == "entity"][0]
        assert entity.owner_scope == "user"
        assert entity.routing_key == "alice"


class TestExtractorSkillAndPattern:
    """Tests for skill and pattern extraction (agent scope)."""

    def test_extract_skill(self):
        """extract_skill → CandidateMemory with category=skill, owner_scope=agent."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_skill",
                "input": {
                    "routing_key": "code_review_checklist",
                    "abstract": "Code review skill",
                    "overview": "Best practices for code review",
                    "content": "Full skill documentation",
                    "confidence": 0.9,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "How to do code review"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1  # dual-run merged
        candidate = result[0]
        assert candidate.category == "skill"
        assert candidate.owner_scope == "agent"
        assert candidate.routing_key == "code_review_checklist"

    def test_extract_pattern_owner_scope(self):
        """extract_pattern → CandidateMemory with owner_scope=user."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_pattern",
                "input": {
                    "routing_key": "error_handling",
                    "abstract": "Common error handling pattern",
                    "overview": "Retry with exponential backoff",
                    "content": "When errors occur, implement retry with exponential backoff",
                    "confidence": 0.85,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "I noticed a pattern"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1  # dual-run merged
        candidate = result[0]
        assert candidate.category == "pattern"
        assert candidate.owner_scope == "user"
        assert candidate.routing_key == "error_handling"


class TestExtractorCase:
    """Tests for case extraction (agent scope, problem-resolution pairs)."""

    def test_extract_case_with_agent_scope(self):
        """extract_case → CandidateMemory with category=case, owner_scope=agent."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_case",
                "input": {
                    "routing_key": "debug_api_timeout",
                    "abstract": "Fixed API timeout issue by increasing timeout and adding retry",
                    "overview": "Problem: API calls timing out. Solution: Increased timeout to 30s and added exponential backoff retry.",
                    "content": "Full case: API was timing out after 5s. Diagnosed as network latency. Increased timeout to 30s, added retry with exponential backoff.",
                    "confidence": 0.9,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "The API is timing out"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1  # dual-run merged
        candidate = result[0]
        assert candidate.category == "case"
        assert candidate.owner_scope == "agent"
        assert candidate.routing_key == "debug_api_timeout"

    def test_extract_case_requires_routing_key(self):
        """extract_case requires routing_key (no default like profile/skill)."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_case",
                "input": {
                    "routing_key": "fix_memory_leak",
                    "abstract": "Fixed memory leak in batch processing",
                    "overview": "Problem: Memory growing. Solution: Added explicit cleanup after each batch.",
                    "content": "Full case details",
                    "confidence": 0.85,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "Memory is leaking"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1  # dual-run merged
        candidate = result[0]
        assert candidate.routing_key == "fix_memory_leak"


class TestExtractorToolMapping:
    """Tests for tool name to category mapping."""

    def test_all_tool_names_mapped(self):
        """Schema extraction tools have fixed category mappings; edits use their read target."""
        registry = SchemaRegistry()
        tools = build_extraction_tools(registry)
        tool_to_category = build_tool_to_category(registry)
        tool_names = {t["name"] for t in tools if t["name"].startswith("extract_")}
        mapped_names = set(tool_to_category.keys())

        assert tool_names == mapped_names, (
            f"Mismatch between tools and mapping. "
            f"Tools: {tool_names}, Mapped: {mapped_names}"
        )

    def test_profile_routing_key_is_explicit(self):
        """extract_profile requires routing_key for field-level storage."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_profile",
                "input": {
                    "routing_key": "occupation",
                    "abstract": "User is a developer",
                    "overview": "## Occupation\n- Developer",
                    "content": "User is a developer.",
                    "confidence": 0.9,
                    "evidence_quote": "I'm a developer",
                    "attributed_speaker": "user",
                    "attribution_basis": "self_first_person",
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "I'm a developer"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1  # dual-run merged
        candidate = result[0]
        assert candidate.category == "profile"
        assert candidate.routing_key == "occupation"

    def test_skill_routing_key_is_explicit(self):
        """extract_skill requires routing_key to identify distinct skills."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "extract_skill",
                "input": {
                    "routing_key": "debug_protocol",
                    "abstract": "Debugging skill",
                    "overview": "How to debug",
                    "content": "Full debugging guide",
                    "confidence": 0.9,
                },
            }
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "How to debug"}]
        result = extractor.extract(messages, ctx)

        assert len(result) == 1  # dual-run merged
        candidate = result[0]
        assert candidate.category == "skill"
        assert candidate.routing_key == "debug_protocol"


class TestExtractorUnknownTool:
    """Tests for unknown tool handling."""

    def test_unknown_tool_is_ignored(self):
        """Tool calls with unknown names are ignored (not converted)."""
        llm = MockLLM()
        llm._mock_tool_calls = [
            {
                "tool": "unknown_tool",
                "input": {
                    "abstract": "Unknown",
                    "confidence": 0.9,
                },
            },
            {
                "tool": "extract_preference",
                "input": {
                    "routing_key": "coffee",
                    "abstract": "Likes coffee",
                    "overview": "Overview",
                    "content": "Content",
                    "confidence": 0.9,
                },
            },
        ]

        extractor = Extractor(llm)
        ctx = RequestContext(
            account_id="test-account",
            user_id="user-123",
            agent_id="agent-456",
            session_id="session-789",
            trace_id="trace-abc",
        )

        messages = [{"role": "user", "content": "Test"}]
        result = extractor.extract(messages, ctx)

        # Only extract_preference should be converted
        assert len(result) == 1  # dual-run merged
        assert result[0].category == "preference"


class TestExtractorLanguageDetection:
    """Tests for language detection in Extractor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.llm = MockLLM()
        self.llm._mock_tool_calls = []
        self.extractor = Extractor(self.llm)

    def test_detect_language_chinese(self):
        """_detect_language returns 'zh-CN' for Chinese user messages."""
        messages = [{"role": "user", "content": "你好，我是张三"}]
        assert self.extractor._detect_language(messages) == "zh-CN"

    def test_detect_language_english(self):
        """_detect_language returns 'en' for English user messages."""
        messages = [{"role": "user", "content": "Hello, I need help"}]
        assert self.extractor._detect_language(messages) == "en"

    def test_language_detection_chinese(self):
        """_detect_language returns 'zh-CN' for Chinese user messages."""
        messages = [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "你好，我是开发工程师"},
            {"role": "assistant", "content": "Hi"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "zh-CN"

    def test_language_detection_default_english(self):
        """_detect_language returns 'en' for pure English messages."""
        messages = [
            {"role": "user", "content": "Hello, I am a software engineer"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "en"

    def test_language_detection_japanese(self):
        """_detect_language returns 'ja' for Japanese user messages."""
        messages = [
            {"role": "user", "content": "こんにちは、私はエンジニアです"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "ja"

    def test_language_detection_korean(self):
        """_detect_language returns 'ko' for Korean user messages."""
        messages = [
            {"role": "user", "content": "안녕하세요, 저는 엔지니어입니다"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "ko"

    def test_language_detection_russian(self):
        """_detect_language returns 'ru' for Russian user messages."""
        messages = [
            {"role": "user", "content": "Привет, я инженер"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "ru"

    def test_language_detection_empty_messages(self):
        """_detect_language returns 'en' when no user messages."""
        messages = []

        language = self.extractor._detect_language(messages)
        assert language == "en"

    def test_language_detection_no_user_role(self):
        """_detect_language returns 'en' when no user role messages."""
        messages = [
            {"role": "assistant", "content": "Hello there"},
            {"role": "system", "content": "You are helpful"},
        ]

        language = self.extractor._detect_language(messages)
        assert language == "en"

    def test_language_detection_mixed_content_uses_first_detected(self):
        """_detect_language returns the first detected language in priority order."""
        # Japanese has higher priority now (checked before Chinese)
        # because Hiragana/Katakana are unique to Japanese
        messages = [
            {"role": "user", "content": "你好こんにちは안녕"},  # Mixed CJK
        ]

        language = self.extractor._detect_language(messages)
        assert language == "ja"  # Japanese detected first (Hiragana present)

    def test_extract_includes_language_in_prompt(self):
        """extract() includes detected language in the LLM prompt."""
        messages = [
            {"role": "user", "content": "你好，我是开发工程师"},
        ]

        # The MockLLM should receive a prompt containing the language
        result = self.extractor.extract(messages, RequestContext(
            account_id="test",
            user_id="user",
            agent_id="agent",
            session_id="session",
            trace_id="trace",
        ))

        # Verify the language was detected
        assert self.extractor._detect_language(messages) == "zh-CN"

        # Verify the prompt sent to LLM contains the language code
        assert "zh-CN" in self.llm._last_prompt
        # The prompt instructs to output fields "in zh-CN" or similar
        assert "zh-cn" in self.llm._last_prompt.lower()
