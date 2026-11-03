"""Complete span evidence survives node batching and checkpoint replay."""

import json
from dataclasses import replace
from datetime import datetime

import pytest

from service.extraction_checkpoint import CandidateCheckpoint
from tests.unit.extraction.test_node_processing import card, node_payload, operation, setup


class Journal:
    def __init__(self, ctx):
        self.ctx, self.batch = ctx, {"archive_id": "source-context", "slots": {}}

    def mutate(self, update):
        update(self.batch)
        self.batch = json.loads(json.dumps(self.batch))


def source_pipeline(mode="eager", same_target=False):
    ctx, fs, llm, api = setup(lambda _: [], mode)
    positions = (3, 15, 27, 39)
    messages = [{"role": "user", "content": f"Message {i}: " + "Source detail. " * 310}
                for i in range(48)]
    for i, position in enumerate(positions):
        messages[position]["content"] += f" MARKER_{i}: Receiver {i} operates at {37+i} MHz."
    llm.complete_json = lambda *args, **kwargs: {"spans": [
        {"start": i, "end": i, "categories": ["entity"]} for i in positions]}
    native_calls, node_calls = [], []
    target = api.write_memory(card("receiver"), ctx)["target_uri"] if same_target else ""

    def extraction(prompt):
        selected = [i for i in range(4) if f"MARKER_{i}:" in prompt]
        assert len(selected) == 1
        i = selected[0]
        native_calls.append(i)
        return {"tool": "extract_entity", "input": {
            "owner_scope": "user", "routing_key": f"receiver_{i}", "object_name": f"receiver_{i}",
            "abstract": f"Receiver {i}.", "overview": f"- It operates at {37+i} MHz.",
            "content": f"# Receiver {i}\n\n## Detail\n- It operates at {37+i} MHz.", "confidence": .99}}

    def complete(**call):
        if not call["prompt"].startswith("NODE_PROCESSING:"):
            return [extraction(call["prompt"])]
        payload = node_payload(call)
        node_calls.append(call)
        assert len(call["prompt"]) <= 110000
        for draft in payload["drafts"]:
            i = int(draft["routing_key"].rsplit("_", 1)[1])
            assert any(f"MARKER_{i}:" in source for source in payload["source"])
            assert any(messages[positions[i]-3]["content"] in source for source in payload["source"])
        assert all("2026-10-10 10:00" in source for source in payload["source"])
        operations = []
        groups = [payload["drafts"]] if target else [[d] for d in payload["drafts"]]
        for group in groups:
            op = operation([d["id"] for d in group], target=target, action="update" if target else "create")
            op.update(routing_key=group[0]["routing_key"], object_name=group[0]["object_name"])
            op["sections"][0]["facts"] = [d["content"].splitlines()[-1][2:] for d in group]
            operations.append(op)
        return [{"tool": "process_memory_nodes", "input": {"operations": operations}}]

    llm.complete_with_tools = complete

    def react(messages, tools, tool_choice):
        call = extraction("\n".join(m["content"] for m in messages))
        return [], json.dumps([{"name": call["tool"], "input": call["input"]}])

    llm.complete_with_tools_messages = react
    return ctx, fs, api, messages, native_calls, node_calls, target


@pytest.mark.parametrize("mode", ["eager", "lazy"])
@pytest.mark.parametrize("occurrence_reconciliation", [False, True])
def test_long_conversation_commits_complete_focused_sources(mode, occurrence_reconciliation):
    ctx, fs, api, messages, native, processed, _ = source_pipeline(mode)
    api._pipeline._extractors[0]._source_occurrence_reconciliation_enabled = occurrence_reconciliation
    result = api.commit_session(messages, ctx, session_time=datetime(2026, 10, 10, 10))
    assert len(json.dumps(messages)) > 110000
    assert result["writes_completed"] == len(fs.nodes) == 4
    assert sorted(native) == list(range(4))
    assert len(processed) == 2
    assert all(any(f"{37+i} MHz" in n.content for n in fs.nodes.values()) for i in range(4))


def test_cas_replays_original_spans_after_combining_one_target():
    ctx, fs, api, messages, native, processed, target = source_pipeline(same_target=True)
    checkpoint = CandidateCheckpoint(Journal(ctx), "0")
    write = api._writer.write_candidate
    versions = []

    def concurrent(candidate, ctx):
        if candidate.target_uri == target and candidate.write_operation_id and not versions:
            node = fs.read_node(target, ctx)
            api.write_memory(card("receiver", target_uri=target, expected_version=node.metadata["version"],
                                  content=node.content + "\n- Stored in the workshop."), ctx)
            versions.append(node.metadata["version"])
        return write(candidate, ctx)

    api._writer.write_candidate = concurrent
    result = api.commit_session(messages, ctx, session_time=datetime(2026, 10, 10, 10),
                                candidate_checkpoint=checkpoint)
    restored = CandidateCheckpoint(checkpoint.journal, "0")
    accepted = restored.load()[0][0]
    assert result["writes_completed"] == len(fs.nodes) == 1
    assert len(processed) == 4
    assert sorted(native) == list(range(4))
    assert accepted.node_draft_indices == list(range(4))
    assert len(accepted.source_contexts) == 4
    assert restored.record["revisions"]["0"] == 1
    saved = fs.read_node(target, ctx)
    assert "workshop" in saved.content
    assert all(f"{37+i} MHz" in saved.content for i in range(4))
    assert saved.metadata["version"] == 3


def test_saved_span_drafts_resume_node_processing():
    ctx, fs, api, messages, native, processed, _ = source_pipeline()
    drafts = api._pipeline.extract(messages, ctx, session_time=datetime(2026, 10, 10, 10))
    checkpoint = CandidateCheckpoint(Journal(ctx), "0")
    checkpoint.save_drafts([replace(c, node_draft_indices=[i]) for i, c in enumerate(drafts)], messages)
    restored = CandidateCheckpoint(checkpoint.journal, "0")
    result = api.commit_session(messages, ctx, candidate_checkpoint=restored)
    assert result["writes_completed"] == len(fs.nodes) == 4
    assert sorted(native) == list(range(4))
    assert len(processed) == 2
    assert all(len(c.source_contexts) == 1 for c in restored.load_drafts())


@pytest.mark.parametrize("occurrence_reconciliation", [False, True])
def test_read_bound_topic_spans_reach_node_processor_together(occurrence_reconciliation):
    from extraction.tools import _Span
    ctx, fs, api, messages, native, processed, target = source_pipeline(same_target=True)
    extractor = api._pipeline._extractors[0]
    extractor._source_occurrence_reconciliation_enabled = occurrence_reconciliation
    drafts = extractor.extract(messages, ctx, session_time=datetime(2026, 10, 10, 10))
    bound = [replace(c, target_uri=target, expected_version=1) for c in drafts]
    merged = extractor._merge_span_results([(_Span(i, i), [c]) for i, c in enumerate(bound)], ctx)
    result = api._node_processor().process(merged, messages, ctx)
    assert len(merged) == 4
    assert len(result) == 1
    assert len(processed) == 2
    assert len(result[0].source_contexts) == 4
    assert all(f"{37+i} MHz" in result[0].content for i in range(4))
