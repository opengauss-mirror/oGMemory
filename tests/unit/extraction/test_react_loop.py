"""Unit tests for ExtractionReActLoop — core loop logic only.

Tests the loop control flow, not the individual tool implementations
or JSON parsing (those belong to their own modules).
"""

import json
from unittest.mock import Mock, patch

import pytest

from core.errors import ExtractionIncompleteError
from core.models import CandidateMemory, ContextNode, RequestContext
from extraction.extraction_react_loop import (
    ExtractionReActLoop,
    ExtractionReActResult,
    ExtractionTrace,
)
from extraction.prefetch import PrefetchResult
from server.internal_tool_usage import InternalToolUsageTracker

# ---------------------------------------------------------------------------
# Patch parse_tool_call — we're testing loop logic, not schema parsing
# ---------------------------------------------------------------------------

_PARSE_PATCH = patch(
    "extraction.extraction_react_loop.parse_tool_call",
    side_effect=lambda name, inp, _reg: (
        (cat, inp.get("owner_scope", "user"), CandidateMemory(
            category=cat,
            owner_scope=inp.get("owner_scope", "user"),
            routing_key=inp.get("routing_key", "k"),
            abstract=inp.get("abstract", ""),
            overview=inp.get("overview", ""),
            content=inp.get("content", ""),
            confidence=inp.get("confidence", 0.9),
        )) if (name.startswith("extract_")) and (cat := name.replace("extract_", ""))
        else None
    ),
)


@pytest.fixture(autouse=True)
def _mock_parse():
    with _PARSE_PATCH:
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx():
    return RequestContext(account_id="acme", user_id="alice",
                         agent_id="bob", session_id="s1", trace_id="t1")


def _node(content="hello", metadata=None):
    return ContextNode(uri="ctx://acme/x", context_type="MEMORY",
                       category="profile", level=0, owner_space="user:alice",
                       abstract="a", overview="o", content=content,
                       metadata=metadata or {})


def _extract_input(routing_key="k", **kw):
    """Shorthand for extract_profile tool input."""
    return {"routing_key": routing_key, "abstract": "a", "overview": "o",
            "content": "c", "confidence": 0.9, "owner_scope": "user", **kw}


def _extract_json(**kw):
    """JSON string for a single extract_profile operation."""
    return json.dumps([{"name": "extract_profile", "input": _extract_input(**kw)}])


def _make_llm():
    llm = Mock()
    llm._queue = []
    def _call(messages, tools=None, tool_choice="auto"):
        return llm._queue.pop(0) if llm._queue else ([], "[]")
    llm.complete_with_tools_messages = _call
    return llm


def _make_fs(exists=False, content="hello"):
    fs = Mock()
    fs.read_node = Mock(return_value=_node(content=content))
    fs.list_children = Mock(return_value=[])
    fs.exists = Mock(return_value=exists)
    return fs


def _make_registry(is_add_only=False):
    schema = Mock(is_add_only=is_add_only)
    reg = Mock()
    reg.get = Mock(return_value=schema)
    reg.list_enabled = Mock(return_value=[])
    return reg


def _make_uri_resolver():
    r = Mock()
    r.resolve = Mock(return_value="ctx://acme/test/uri")
    r.validate_uri = Mock(return_value=True)
    return r


def _make_loop(llm=None, fs=None, **kw):
    from unittest.mock import Mock
    pm = kw.pop("prompt_manager", Mock())
    return ExtractionReActLoop(
        llm=llm or _make_llm(),
        fs=fs or _make_fs(),
        prompt_manager=pm,
        registry=kw.pop("registry", _make_registry()),
        uri_resolver=kw.pop("uri_resolver", _make_uri_resolver()),
        max_iterations=kw.get("max_iterations", 3),
        timeout_seconds=kw.get("timeout_seconds", 30.0),
        internal_tool_usage_tracker=kw.get("internal_tool_usage_tracker"),
    )


# ---------------------------------------------------------------------------
# Tests: Loop control flow
# ---------------------------------------------------------------------------

class TestLoopControl:
    """Core iteration, termination, and state management."""

    def test_tool_call_then_content(self):
        """LLM calls read → gets result → outputs extraction in next iteration."""
        llm = _make_llm()
        llm._queue.append(([{"name": "read", "input": {"uri": "ctx://acme/p"}, "id": "1"}], ""))
        llm._queue.append(([], _extract_json(routing_key="java-dev")))

        result = _make_loop(llm=llm).run("I switched to Java", _ctx())

        assert len(result.candidates) == 1
        assert result.candidates[0].routing_key == "java-dev"
        assert result.iterations == 2
        assert "ctx://acme/p" in result.read_uris

    def test_max_iterations_exhausted(self):
        """Repeated tool calls cannot extend extraction indefinitely."""
        llm = _make_llm()
        for _ in range(5):
            llm._queue.append(([{"name": "read", "input": {"uri": "ctx://x"}, "id": "1"}], ""))

        with pytest.raises(ExtractionIncompleteError, match="max_iterations"):
            _make_loop(llm=llm, max_iterations=3).run("conv", _ctx())
        assert len(llm._queue) == 1

    def test_state_reset_between_runs(self):
        """run() resets read_files so second run doesn't leak first run's state."""
        llm = _make_llm()
        llm._queue.append(([{"name": "read", "input": {"uri": "ctx://a"}, "id": "1"}], ""))
        llm._queue.append(([], _extract_json()))
        llm._queue.append(([], _extract_json(routing_key="k2")))

        loop = _make_loop(llm=llm)
        r1 = loop.run("first", _ctx())
        assert "ctx://a" in r1.read_uris

        r2 = loop.run("second", _ctx())
        assert "ctx://a" not in r2.read_uris

    def test_prefetch_uris_included(self):
        """PrefetchResult.read_uris are included in final read_uris."""
        llm = _make_llm()
        llm._queue.append(([], _extract_json()))
        prefetch = PrefetchResult(messages=["ctx"], read_uris={"ctx://acme/p"}, listed_uris=set())

        result = _make_loop(llm=llm).run("conv", _ctx(), prefetch)
        assert "ctx://acme/p" in result.read_uris

    def test_prefetch_tracking_is_reset_between_runs(self):
        llm = _make_llm()
        for _ in range(3):
            llm._queue.append(([], _extract_json()))
        loop = _make_loop(llm=llm)
        for uri in ("ctx://acme/first", "ctx://acme/second"):
            prefetch = PrefetchResult(
                messages=["ctx"], read_uris={uri}, listed_uris=set(),
            )
            result = loop.run("conv", _ctx(), prefetch)
            assert result.read_uris == {uri}
        assert loop.run("no prefetch", _ctx()).read_uris == set()

    def test_unread_existing_file_refetch_uris_are_deduplicated(self):
        """Multiple candidates targeting the same existing URI are refetched once."""
        fs = _make_fs(exists=True)
        loop = _make_loop(fs=fs)
        candidates = [
            CandidateMemory(
                category="profile",
                owner_scope="user",
                routing_key="same",
                abstract="a",
                overview="o",
                content="c",
                confidence=0.9,
            ),
            CandidateMemory(
                category="profile",
                owner_scope="user",
                routing_key="same",
                abstract="a2",
                overview="o2",
                content="c2",
                confidence=0.9,
            ),
        ]

        refetch_uris = loop._check_unread_existing_files(candidates, _ctx())

        assert refetch_uris == ["ctx://acme/test/uri"]
        fs.exists.assert_called_once_with("ctx://acme/test/uri", _ctx())

    def test_internal_tool_usage_records_round_and_tool_tokens(self):
        """Internal oGMem ReAct tool calls are attributed to the current session."""
        llm = _make_llm()
        llm._queue.append(([
            {
                "id": "call-read",
                "name": "read",
                "input": {"uri": "ctx://acme/p"},
                "_llm_usage": {
                    "round_id": "round-read",
                    "input_tokens": 100,
                    "output_tokens": 20,
                },
            }
        ], ""))
        llm._queue.append(([], "[]"))
        tracker = InternalToolUsageTracker()

        _make_loop(llm=llm, internal_tool_usage_tracker=tracker).run("conv", _ctx())

        stats = tracker.get_stats(session_id="s1", include_rounds=True)
        assert stats["summary"]["llm_tool_rounds"] == 1
        assert stats["summary"]["tool_calls"] == 1
        assert stats["summary"]["total_tokens"] == 120
        assert stats["tools"][0]["tool_name"] == "read"
        assert stats["tools"][0]["attributed_tokens"] == 120
        assert stats["tools"][0]["allocated_tokens"] == 120
        assert stats["rounds"][0]["round_id"] == "round-read"

    def test_content_present_but_parse_fails(self):
        """LLM returns non-JSON content → _parse_output returns None → loop ends early."""
        llm = _make_llm()

        def _track(messages, tools=None, tool_choice="auto"):
            return [], "I think the user likes Python"  # content but not JSON

        llm.complete_with_tools_messages = _track
        with pytest.raises(ExtractionIncompleteError, match="no_valid_output"):
            _make_loop(llm=llm, max_iterations=5).run("conv", _ctx())

    @pytest.mark.parametrize(
        "content",
        [
            '{"operations": [{"name": "unknown_tool", "input": {}}]}',
            '{"message": "no extraction operation"}',
        ],
    )
    def test_content_without_valid_candidates_is_invalid(self, content):
        llm = _make_llm()

        def _track(messages, tools=None, tool_choice="auto"):
            return [], content

        llm.complete_with_tools_messages = _track
        with pytest.raises(ExtractionIncompleteError, match="no_valid_output"):
            _make_loop(llm=llm, max_iterations=5).run("conv", _ctx())

    @pytest.mark.parametrize("content", ["[]", '{"operations": []}', '{"tool_calls": []}'])
    def test_explicit_empty_output_completes(self, content):
        llm = _make_llm()
        llm._queue.append(([], content))
        result = _make_loop(llm=llm).run("conv", _ctx())
        assert result.candidates == []
        assert result.iterations == 1

    def test_empty_json_fragment_does_not_hide_final_operations(self):
        llm = _make_llm()
        llm._queue.append(([], "Tool calls: []\nFinal operations: " + _extract_json()))
        result = _make_loop(llm=llm).run("conv", _ctx())
        assert len(result.candidates) == 1

    @pytest.mark.parametrize(
        "content",
        [
            "```json\n{\"operations\": [{\"name\": \"extract_profile\", \"input\": {\"routing_key\": \"tea\", \"abstract\": \"a\", \"overview\": \"o\", \"content\": \"c\", \"confidence\": 0.9}}]}\n```",
            "Result:\n{\"operations\": [{\"name\": \"extract_profile\", \"input\": {\"routing_key\": \"tea\", \"abstract\": \"a\", \"overview\": \"o\", \"content\": \"c\", \"confidence\": 0.9}}]}",
            "{\"operations\": [{\"name\": \"extract_profile\", \"input\": {\"routing_key\": \"tea\", \"abstract\": \"a\", \"overview\": \"o\", \"content\": \"c\", \"confidence\": 0.9}}],}",
            "Result:\n[{\"name\": \"extract_profile\", \"input\": {\"routing_key\": \"tea\", \"abstract\": \"a\", \"overview\": \"o\", \"content\": \"c\", \"confidence\": 0.9}}, {\"name\": \"extract_profile\", \"input\": {\"routing_key\": \"coffee\", \"abstract\": \"a\", \"overview\": \"o\", \"content\": \"c\", \"confidence\": 0.9}}]",
            "See [1]. Result: [{\"name\": \"extract_profile\", \"input\": {\"routing_key\": \"tea\", \"abstract\": \"a\", \"overview\": \"o\", \"content\": \"c\", \"confidence\": 0.9}}]",
        ],
    )
    def test_content_json_is_sanitized_before_parse(self, content, caplog):
        llm = _make_llm()

        def _track(messages, tools=None, tool_choice="auto"):
            return [], content

        llm.complete_with_tools_messages = _track
        result = _make_loop(llm=llm, max_iterations=3).run("conv", _ctx())

        expected_keys = ["tea", "coffee"] if "coffee" in content else ["tea"]
        assert [candidate.routing_key for candidate in result.candidates] == expected_keys
        assert result.candidates[0].routing_key == "tea"
        assert result.iterations == 1
        assert "could not be parsed as extraction operations" not in caplog.text


# ---------------------------------------------------------------------------
# Tests: Tool disable mechanism
# ---------------------------------------------------------------------------

class TestToolDisable:
    """_disable_tools_for_iteration flips tool_choice to 'none'."""

    def test_unknown_tool_disables_next(self):
        llm = _make_llm()
        seen = []
        n = 0

        def _track(messages, tools=None, tool_choice="auto"):
            nonlocal n
            n += 1
            seen.append(tool_choice)
            if n == 1:
                return [{"name": "bogus", "input": {}, "id": "1"}], ""
            return [], _extract_json()

        llm.complete_with_tools_messages = _track
        _make_loop(llm=llm).run("conv", _ctx())
        assert seen == ["auto", "none"]

    def test_empty_response_ends_loop(self):
        """LLM returns nothing → loop ends early instead of disabling tools."""
        llm = _make_llm()

        def _track(messages, tools=None, tool_choice="auto"):
            return ([], "")  # neither tools nor content

        llm.complete_with_tools_messages = _track
        with pytest.raises(ExtractionIncompleteError, match="no_valid_output"):
            _make_loop(llm=llm, max_iterations=5).run("conv", _ctx())


# ---------------------------------------------------------------------------
# Tests: Safety refetch
# ---------------------------------------------------------------------------

class TestSafetyRefetch:

    def test_final_iteration_refetch_retains_draft_in_model_context(self):
        llm = _make_llm()
        original = _extract_json(routing_key="before-read")
        revised = _extract_json(routing_key="after-read")
        requests = []

        def respond(messages, **kwargs):
            from copy import deepcopy
            requests.append(deepcopy(messages))
            return [], original if len(requests) == 1 else revised

        llm.complete_with_tools_messages = respond
        result = _make_loop(llm=llm, fs=_make_fs(exists=True), max_iterations=1).run("conv", _ctx())
        assert result.candidates[0].routing_key == "after-read"
        assert len(requests) == 2
        assert {"role": "assistant", "content": original} in requests[1]

    def test_timeout_discards_recorded_drafts(self):
        llm = _make_llm()
        llm._queue.append(([{"name": "extract_profile", "input": _extract_input(), "id": "1"}], ""))
        with pytest.raises(ExtractionIncompleteError, match="timeout"):
            _make_loop(llm=llm, timeout_seconds=-1).run("conv", _ctx())
    """_check_unread_existing_files + _did_safety_reread guard."""

    def test_refetch_triggered_for_unread_existing(self):
        """Candidate targets existing unread file → auto-refetch → extra iteration."""
        llm = _make_llm()
        fs = _make_fs(exists=True, content="old data")
        llm._queue.append(([], _extract_json()))          # targets unread existing
        llm._queue.append(([], _extract_json(routing_key="k2")))  # after refetch

        result = _make_loop(llm=llm, fs=fs).run("conv", _ctx())
        assert any(it.safety_check_triggered for it in result.trace.iterations)

    def test_refetch_tool_call_ids_fit_openai_limit(self):
        """Auto-refetch call IDs stay within the OpenAI 64-character limit."""
        loop = _make_loop(fs=_make_fs())
        messages = []
        long_uri = "ctx://acme/users/alice/memories/entities/" + ("very_long_name_" * 8)

        loop._add_refetch_results(messages, [long_uri], _ctx())

        assistant_msg = next(m for m in messages if m.get("tool_calls"))
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        call_id = assistant_msg["tool_calls"][0]["id"]
        assert len(call_id) <= 64
        assert tool_msg["tool_call_id"] == call_id

    def test_refetch_only_once(self):
        """_did_safety_reread prevents refetch from firing repeatedly."""
        llm = _make_llm()
        fs = _make_fs(exists=True)
        for _ in range(4):
            llm._queue.append(([], _extract_json()))

        result = _make_loop(llm=llm, fs=fs, max_iterations=5).run("conv", _ctx())
        safety_count = sum(1 for it in result.trace.iterations if it.safety_check_triggered)
        assert safety_count <= 1

    def test_add_only_schema_skips_refetch(self):
        """is_add_only=True means no conflict risk → no refetch."""
        llm = _make_llm()
        fs = _make_fs(exists=True)  # exists but shouldn't matter
        llm._queue.append(([], _extract_json()))

        result = _make_loop(llm=llm, fs=fs, registry=_make_registry(is_add_only=True)).run("conv", _ctx())
        assert all(not it.safety_check_triggered for it in result.trace.iterations)


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_unknown_tool_returns_error(self):
        """Unknown tool → error dict, loop continues."""
        llm = _make_llm()
        llm._queue.append(([{"name": "fly", "input": {}, "id": "1"}], ""))
        llm._queue.append(([], _extract_json()))

        result = _make_loop(llm=llm).run("conv", _ctx())
        assert result.tools_used[0]["result"]["error"] == "Unknown tool: fly"

    def test_tool_exception_continues_loop(self):
        """Tool throws → error dict, loop doesn't crash."""
        llm = _make_llm()
        fs = _make_fs()
        fs.read_node.side_effect = RuntimeError("boom")
        llm._queue.append(([{"name": "read", "input": {"uri": "ctx://x"}, "id": "1"}], ""))
        llm._queue.append(([], _extract_json()))

        result = _make_loop(llm=llm, fs=fs).run("conv", _ctx())
        assert "error" in result.tools_used[0]["result"]
        assert len(result.candidates) == 1

    def test_llm_exception_fails_extraction(self):
        """An API failure must not be reported as a successful empty extraction."""
        llm = _make_llm()
        llm.complete_with_tools_messages = Mock(side_effect=RuntimeError("API down"))
        with pytest.raises(ExtractionIncompleteError, match="llm_error"):
            _make_loop(llm=llm).run("conv", _ctx())
