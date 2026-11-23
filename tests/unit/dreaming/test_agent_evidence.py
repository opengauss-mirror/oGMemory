from unittest.mock import Mock

from commit.dreaming_facts import assess_facts
from core.models import ContextNode, RequestContext
from extraction.agent_memory import AgentMemoryExtractor
from server.mcp_trace import capture_write_trace


def _trace(tool_id):
    return [
        {"ordinal": 1, "event": {"event_type": "tool_start", "tool_name": "catalog", "tool_use_id": tool_id,
                                 "tool_input": {"page": 1}}},
        {"ordinal": 2, "event": {"event_type": "tool_success", "tool_name": "catalog", "tool_use_id": tool_id,
                                 "tool_response": {"next_page": 2}}},
    ]


def _node(traces, category="tool"):
    grounding = []
    for uri, events in traces.items():
        _, sources, lines, _ = AgentMemoryExtractor._evidence(None, uri, events)
        citations = [{"source_id": source_id, "start_line": 1, "end_line": 1} for source_id in sources]
        grounding.extend(AgentMemoryExtractor._ground(citations, sources, lines))
    unit = {"key": "pagination", "title": "Pagination", "content": "Use the next page returned by the catalog.",
            "grounding": grounding}
    fields = {"tool_name": "catalog", "units": [unit]} if category == "tool" else unit
    return ContextNode("ctx://account/agents/agent/memories/catalog", "MEMORY", category, 0,
                       "agent:agent", "Catalog", "", unit["content"], {
                           "agent_memory_version": 1, "routing_key": "catalog",
                           "provenance_ids": list(traces), "structured_fields": fields,
                           "agent_memory_value_review": {"routing_keys": ["catalog"], "decisions": [
                               {"id": "0:pagination", "keep": True,
                                "fields": {"title": unit["title"], "content": unit["content"]}},
                           ]},
                       })


def test_tool_unit_uses_original_trace_and_counts_each_invocation_once():
    traces = {"ctx://account/traces/session/turns/one": _trace("lookup-1")}
    node = _node(traces)
    review = assess_facts(node, read_source=traces.get)
    assert review["unit_confidence"] == {"pagination": .8}
    assert review["value_reason"] == "reusable_agent_unit"
    unit = review["units"]["pagination"]
    assert len(unit["evidence"]) == 2
    assert len({proof["occurrence_id"] for proof in unit["evidence"]}) == 1

    traces["ctx://account/traces/session/turns/two"] = _trace("lookup-2")
    review = assess_facts(_node(traces), read_source=traces.get)
    assert review["unit_confidence"] == {"pagination": 64 / 65}
    assert review["recommendation"] == "recommended"


def test_experience_counts_a_captured_answer_and_its_turn_end_once():
    service = Mock()
    captured = []

    def store(params):
        ordinal = len(captured) + 1
        captured.append({"ordinal": ordinal, "event": params["event"]})
        return {"ok": True, "event_index": ordinal}

    service.add_session_trace.side_effect = store
    service.finalize_session_trace.return_value = {"ok": True}
    ctx = RequestContext("account", "user", "agent", "session", "request")
    messages = [{
        "role": "assistant", "content": "Use the next page returned by the catalog.",
        "reasoning_content": "The current response advertises another page.",
    }]
    assert capture_write_trace(service, ctx, messages, session_id="session", client_request_id="request")
    trace_uri = "ctx://account/traces/session/turns/one"
    traces = {trace_uri: captured}
    review = assess_facts(_node(traces, category="experience"), read_source=traces.get)
    assert review["unit_confidence"] == {"pagination": .8}
    assert review["recommendation"] == "observation"
    assert len(review["units"]["pagination"]["evidence"]) == 2

    captured = []
    assert capture_write_trace(service, ctx, messages, session_id="session", client_request_id="next-request")
    traces["ctx://account/traces/session/turns/two"] = captured
    review = assess_facts(_node(traces, category="experience"), read_source=traces.get)
    assert review["unit_confidence"] == {"pagination": 64 / 65}
    assert review["recommendation"] == "recommended"
