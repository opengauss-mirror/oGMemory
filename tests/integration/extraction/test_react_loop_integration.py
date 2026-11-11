"""Integration test for ExtractionReActLoop with real SchemaRegistry and URIResolver.

Uses actual YAML schemas and parse_tool_call — only mocks LLM.
ContextFS is InMemoryContextFS (shared fixture from integration/conftest.py).
"""

import json
from unittest.mock import Mock

import pytest

from core.models import ContextNode, RequestContext
from core.uri_resolver import URIResolver
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.prompts import PromptManager
from extraction.schemas.registry import SchemaRegistry

# ---------------------------------------------------------------------------
# Real dependencies
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return SchemaRegistry()  # loads from extraction/schemas/definitions/


@pytest.fixture
def uri_resolver(registry):
    return URIResolver(registry)


@pytest.fixture
def ctx():
    return RequestContext(
        account_id="acme", user_id="alice",
        agent_id="assistant", session_id="s1", trace_id="t1",
    )


# ---------------------------------------------------------------------------
# Mock LLM only
# ---------------------------------------------------------------------------

def _make_llm():
    llm = Mock()
    llm._queue = []
    llm.complete_json.return_value = {"categories": ["preference"]}

    def _call(messages, tools=None, tool_choice="auto"):
        names = {tool["name"] for tool in tools}
        assert "extract_preference" in names
        assert "extract_profile" not in names
        return llm._queue.pop(0) if llm._queue else ([], "")

    llm.complete_with_tools_messages = _call
    return llm


def _node(uri, content="existing content", abstract="existing abstract",
          overview="existing overview", metadata=None):
    return ContextNode(
        uri=uri, context_type="MEMORY", category="preference",
        level=0, owner_space="user:alice",
        abstract=abstract, overview=overview, content=content,
        metadata=metadata or {},
    )


def _preference_input(routing_key, abstract, overview, content, confidence=0.9):
    """Valid input for extract_preference tool."""
    return {
        "routing_key": routing_key,
        "abstract": abstract,
        "overview": overview,
        "content": content,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Integration test: realistic multi-turn extraction
# ---------------------------------------------------------------------------

class TestReActLoopIntegration:
    """End-to-end ReAct loop with real schema parsing and URI resolution."""

    def test_read_existing_then_extract_preference(self, registry, uri_resolver, ctx, memory_fs):
        """LLM reads existing coding_style preference → decides to update it.

        Flow:
        1. LLM lists preferences directory
        2. LLM reads coding_style.md (existing: "likes Java")
        3. LLM extracts updated preference ("switched to Python")
        """
        llm = _make_llm()
        pref_uri = "ctx://acme/users/alice/memories/preferences/coding_style"
        memory_fs.write_node(_node(pref_uri, content="User prefers Java for backend",
                                   abstract="Java backend developer",
                                   overview="Prefers Java"), ctx)

        # Iteration 1: LLM lists the preferences directory
        llm._queue.append(([
            {"name": "list",
             "input": {"uri": "ctx://acme/users/alice/memories/preferences"},
             "id": "tc-1"},
        ], ""))

        # Iteration 2: LLM reads the existing coding_style preference
        llm._queue.append(([
            {"name": "read",
             "input": {"uri": pref_uri},
             "id": "tc-2"},
        ], ""))

        # Iteration 3: LLM returns the updated extraction
        llm._queue.append(([], json.dumps([{
            "name": "extract_preference",
            "input": _preference_input(
                routing_key="coding_style",
                abstract="Switched from Java to Python",
                overview="Now prefers Python for backend development",
                content="User switched from Java to Python for backend. Likes type hints and fastapi.",
                confidence=0.95,
            ),
        }])))

        loop = ExtractionReActLoop(
            llm=llm,
            fs=memory_fs,
            registry=registry,
            uri_resolver=uri_resolver,
            prompt_manager=PromptManager(),
            max_iterations=5,
        )
        result = loop.run(
            "User: I've been using Java for years but recently switched to Python. "
            "The type hints are great and I love fastapi.",
            ctx,
        )

        # Verify candidates
        llm.complete_json.assert_called_once()
        assert len(result.candidates) == 1
        c = result.candidates[0]
        assert c.category == "preference"
        assert c.routing_key == "coding_style"
        assert "Python" in c.content
        assert c.confidence == 0.95

        # Verify loop execution
        assert result.iterations == 3
        assert len(result.tools_used) == 2  # list + read
        assert result.tools_used[0]["tool_name"] == "list"
        assert result.tools_used[1]["tool_name"] == "read"

        # Verify read_uris tracking
        assert pref_uri in result.read_uris

        # Verify trace data
        trace = result.trace
        assert trace is not None
        assert trace.total_iterations == 3
        assert trace.final_candidate_count == 1
        assert trace.total_read_uris == 1
        assert len(trace.iterations) == 3

        # Verify iteration-level trace
        iter0 = trace.iterations[0]
        assert iter0.tool_calls_count == 1
        assert iter0.tool_call_names == ["list"]

        iter1 = trace.iterations[1]
        assert iter1.tool_call_names == ["read"]

        iter2 = trace.iterations[2]
        assert iter2.tool_calls_count == 0  # content returned, no tool calls
        assert iter2.content_length > 0

    def test_safety_refetch_with_real_schema(self, registry, uri_resolver, ctx, memory_fs):
        """LLM extracts without reading existing file → refetch triggered.

        Verifies the refetch mechanism works with real URI resolution:
        preference URI resolved via Jinja2 template from YAML schema.
        """
        llm = _make_llm()
        pref_uri = "ctx://acme/users/alice/memories/preferences/coffee"
        memory_fs.write_node(_node(pref_uri, content="User drinks black coffee",
                                   abstract="Black coffee drinker"), ctx)

        # Iteration 1: LLM extracts directly WITHOUT reading first
        llm._queue.append(([], json.dumps([{
            "name": "extract_preference",
            "input": _preference_input(
                routing_key="coffee",
                abstract="Now prefers latte",
                overview="Switched from black coffee to latte",
                content="User now prefers latte over black coffee",
                confidence=0.85,
            ),
        }])))

        # Iteration 2: After refetch, LLM returns merged result
        llm._queue.append(([], json.dumps([{
            "name": "extract_preference",
            "input": _preference_input(
                routing_key="coffee",
                abstract="Prefers latte (was black coffee)",
                overview="Evolved from black coffee to latte",
                content="User used to drink black coffee, now prefers latte",
                confidence=0.9,
            ),
        }])))

        loop = ExtractionReActLoop(
            llm=llm,
            fs=memory_fs,
            registry=registry,
            uri_resolver=uri_resolver,
            prompt_manager=PromptManager(),
            max_iterations=5,
        )
        result = loop.run("User: I switched to lattes recently", ctx)

        # Refetch should have triggered
        assert any(it.safety_check_triggered for it in result.trace.iterations)
        llm.complete_json.assert_called_once()
        assert len(result.candidates) == 1
        assert result.candidates[0].routing_key == "coffee"
        assert result.iterations == 2

        # Verify the prefetched URI is now in read_uris
        assert pref_uri in result.read_uris

    def test_extract_new_preference_no_existing(self, registry, uri_resolver, ctx, memory_fs):
        """No existing file → no refetch → direct extraction.

        Verifies URI resolution produces correct URI for new preference.
        """
        llm = _make_llm()

        # Single iteration: LLM extracts directly
        llm._queue.append(([], json.dumps([{
            "name": "extract_preference",
            "input": _preference_input(
                routing_key="music",
                abstract="Enjoys jazz music",
                overview="Likes jazz and blues",
                content="User enjoys listening to jazz and blues in their free time",
                confidence=0.8,
            ),
        }])))

        loop = ExtractionReActLoop(
            llm=llm, fs=memory_fs, registry=registry, uri_resolver=uri_resolver,
            prompt_manager=PromptManager(),
        )
        result = loop.run("User: I love jazz and blues", ctx)

        llm.complete_json.assert_called_once()
        assert len(result.candidates) == 1
        assert result.candidates[0].category == "preference"
        assert result.candidates[0].routing_key == "music"
        assert result.iterations == 1

        # No safety check should have triggered (file doesn't exist)
        assert all(not it.safety_check_triggered for it in result.trace.iterations)
