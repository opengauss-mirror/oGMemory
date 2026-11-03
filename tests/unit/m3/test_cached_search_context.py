import json
from copy import deepcopy

from m3.control_adapter import _compact_cached_search_messages


def payload(description="unique full evidence " * 500):
    return {"query": "scissors", "hits": [
        {"node_id": "n1", "video_id": "v", "clip_id": 1, "description": description,
         "speech_transcript": [{"text": "Careful with the scissors."}]}],
        "context_hits": [{"node_id": "n2", "description": "neighbor facts"}],
        "boundary_hits": []}


def tool(call_id, value):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(value)}


def test_duplicate_payload_references_existing_full_evidence():
    original = payload()
    first = tool("call_first", original)
    messages = [deepcopy(first), tool("call_repeat", {**original, "duplicate_query": True})]
    before_size = len(messages[1]["content"])
    _compact_cached_search_messages(messages, 1)
    assert messages[0] == first
    repeated = json.loads(messages[1]["content"])
    assert repeated["evidence_reference_tool_call_id"] == "call_first"
    assert repeated["hits"] == [{"node_id": "n1", "video_id": "v", "clip_id": 1}]
    assert repeated["context_hits"] == [{"node_id": "n2"}]
    assert "description" in original["hits"][0]
    assert len(messages[1]["content"]) < before_size / 5


def test_without_first_evidence_keep_full_cache_hit():
    messages = [tool("call_repeat", {**payload(), "duplicate_query": True})]
    original = deepcopy(messages)
    _compact_cached_search_messages(messages, 0)
    assert messages == original


def test_changed_evidence_is_not_compacted():
    messages = [tool("a", payload()), tool("b", {**payload("changed facts"), "duplicate_query": True})]
    original = deepcopy(messages)
    _compact_cached_search_messages(messages, 1)
    assert messages == original


def test_different_query_with_same_hits_is_not_assumed_duplicate():
    messages = [tool("a", payload()), tool("b", payload())]
    original = deepcopy(messages)
    _compact_cached_search_messages(messages, 1)
    assert messages == original


def test_multiple_repeats_reference_original_not_an_omitted_payload():
    original = payload()
    messages = [tool("a", original), tool("b", {**original, "duplicate_query": True}),
                tool("c", {**original, "duplicate_query": True})]
    _compact_cached_search_messages(messages, 1)
    for message in messages[1:]:
        assert json.loads(message["content"])["evidence_reference_tool_call_id"] == "a"


def test_opaque_tool_output_is_not_modified():
    messages = [{"role": "tool", "content": "not JSON"}, tool("x", ["not", "a", "search"])]
    original = deepcopy(messages)
    _compact_cached_search_messages(messages, 0)
    assert messages == original
