"""Successful owner-thread collection and parallel source-grounded model stages."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import threading

import pytest

from extraction.tools import ExtractionRuntime, _Span
from core.model_stage import stage_digest
from tests.unit.extraction.test_node_processing import card, node_payload, operation, setup
from tests.unit.extraction.test_node_read_cache import record_storage_threads


class Stages:
    def __init__(self):
        self.owner = threading.get_ident()
        self.values = {}
        self.saved = []

    def load_stage(self, name, inputs):
        assert threading.get_ident() == self.owner
        return deepcopy(self.values.get((name, stage_digest(inputs))))

    def save_stage(self, name, inputs, result):
        assert threading.get_ident() == self.owner
        stage_digest(result)
        self.values[name, stage_digest(inputs)] = deepcopy(result)
        self.saved.append(name)


def test_event_and_independent_node_groups_overlap_after_one_native_pass():
    owner = threading.get_ident()
    ctx, fs, llm, api = setup(lambda _: [])
    rendezvous = threading.Barrier(3)
    native = []
    model_threads = []

    def complete_tools(*, prompt, tools, tool_choice, temperature=None):
        if tools[0]["name"] == "process_memory_nodes":
            model_threads.append(threading.get_ident())
            rendezvous.wait(5)
            return [{"tool": "process_memory_nodes", "input": {
                "operations": [operation([0])], "related_reconciliations": []}}]
        native.append(temperature)
        topics = [{"tool": "extract_entity", "input": {
            "routing_key": "necklace", "object_name": "necklace", "scope_ref": scope,
            "abstract": "Caroline's necklace.", "overview": "Grandmother's gift.",
            "content": "Grandmother gave Caroline the necklace.", "confidence": .95}}
                  for scope in ("project_a", "project_b")]
        events = [{"tool": "extract_event", "input": {
            "routing_key": "atlas_inspection_" + day, "title": "Atlas inspection",
            "abstract": "Nora inspected Atlas on " + day, "overview": "Nora inspected Atlas on " + day,
            "content": "Nora inspected Atlas on " + day, "when": "2026-10-" + day,
            "who": "Nora", "where": "Lab A", "confidence": .95}}
                  for day in ("01", "02")]
        return topics + events

    def complete_json(prompt, schema):
        if "spans" in schema["properties"]:
            return {"spans": [{"start": 0, "end": 0, "categories": ["entity", "event"]}]}
        if "categories" in schema["properties"]:
            return {"categories": ["entity", "event"]}
        assert "groups" in schema["properties"]
        model_threads.append(threading.get_ident())
        rendezvous.wait(5)
        return {"groups": [{"members": [0]}, {"members": [1]}]}

    llm.complete_with_tools = complete_tools
    llm.complete_json = complete_json
    calls = record_storage_threads(fs)
    source = [{"id": "source", "role": "user", "content":
               "Grandmother gave Caroline necklaces for project A and project B. "
               "Nora inspected Atlas on October 1 and October 2 in Lab A."}]
    result = api.commit_session(source, ctx, session_time=datetime(2026, 10, 3))
    assert result["writes_completed"] == 4
    assert native == [None]
    assert len(set(model_threads)) == 3
    assert {thread_id for _, thread_id in calls} == {owner}
    assert {node.category for node in fs.nodes.values()} == {"event", "entity"}


def test_saved_groups_reuse_then_refresh_source_and_node_version():
    ctx, fs, llm, api = setup(lambda _: [])
    drafts = [card(scope_ref=scope) for scope in ("project_a", "project_b")]
    uris = [api.write_memory(draft, ctx)["target_uri"] for draft in drafts]
    owner = threading.get_ident()
    calls = record_storage_threads(fs)

    def decide(call):
        payload = node_payload(call)
        return [operation([0], target=payload["nodes"][0]["uri"], action="update")]

    llm.decide = decide
    stages = Stages()
    processor = api._node_processor()
    source = [{"role": "user", "content": "Each necklace has a heart charm."}]
    first = processor.process(drafts, source, ctx, checkpoint=stages)
    reused = processor.process(drafts, source, ctx, checkpoint=stages)
    assert first == reused
    assert len(llm.calls) == 2
    source = [{"role": "user", "content": "Each necklace has a silver heart charm."}]
    processor.process(drafts, source, ctx, checkpoint=stages)
    assert len(llm.calls) == 4
    updated = fs.read_node(uris[0], ctx)
    updated.content += "\n- Stored in a blue box."
    fs.write_node(updated, ctx)
    fresh = processor.process(drafts, source, ctx, checkpoint=stages)
    assert len(llm.calls) == 5
    assert next(c for c in fresh if c.target_uri == uris[0]).expected_version == 2
    assert "blue box" in next(c for c in fresh if c.target_uri == uris[0]).content
    assert {thread_id for _, thread_id in calls} == {owner}


def test_natural_input_split_preserves_pending_sequence_and_reuses_stages():
    ctx, fs, llm, api = setup(lambda _: [])
    drafts = [card(f"alias_{index}", content="# Item\n\n## Details\n- " + "Source detail. " * 2300
                   + f"\n- Fact {index}.") for index in range(4)]
    snapshots = []

    def decide(call):
        payload = node_payload(call)
        snapshots.append(payload)
        target = payload["nodes"][0]["uri"] if payload["nodes"] else ""
        op = operation(list(range(len(payload["drafts"]))), target=target,
                       action="update" if target else "create")
        op["sections"][0]["facts"] = [d["content"].splitlines()[-1][2:] for d in payload["drafts"]]
        if target:
            assert "Fact 0." in payload["nodes"][0]["content"]
            assert "Fact 1." in payload["nodes"][0]["content"]
        else:
            op.update(routing_key="alias_0", object_name="alias_0")
        return [op]

    llm.decide = decide
    stages = Stages()
    processor = api._node_processor()
    result = processor.process(drafts, [], ctx, checkpoint=stages)
    reused = processor.process(drafts, [], ctx, checkpoint=stages)
    assert result == reused
    assert len(snapshots) == 2
    assert len(result) == 1
    assert all(f"Fact {index}." in result[0].content for index in range(4))
    assert result[0].routing_key == "alias_0"


def test_native_stage_replays_complete_drafts_before_reconciliation():
    from service.extraction_retry import _format_feedback
    ctx, fs, llm, api = setup(lambda _: [])
    extractor = api._pipeline._extractors[0]
    reasons = iter(("A personal gift", "The source names a necklace"))
    categories = iter((["event", "entity", "preference", "profile", "skill"],
                       ["profile", "event", "entity", "preference", "skill"]))
    extractor._identify_spans = lambda *args: [_Span(0, 0, reason=next(reasons), categories=next(categories))]
    native = []

    def complete_tools(**kwargs):
        native.append(kwargs)
        return [{"tool": "extract_entity", "input": {
            "routing_key": "necklace", "object_name": "necklace", "abstract": "Caroline's necklace.",
            "overview": "A gift.", "content": "Grandmother gave Caroline a necklace.", "confidence": .95}}]

    llm.complete_with_tools = complete_tools
    source = [{"id": "source", "role": "user", "content": "Grandmother gave Caroline a necklace."}]
    stages = Stages()
    results = []
    for feedback in ("", "node_processing: use the complete source details"):
        runtime = ExtractionRuntime(stages)
        token = _format_feedback.set(feedback)
        try:
            with runtime.activate():
                results.append(api._pipeline.extract(source, ctx, parallel=False,
                                                     session_time=datetime(2026, 10, 3), archive_id="archive"))
        finally:
            _format_feedback.reset(token)
        assert len(runtime.entries) == 1
    assert results[0] == results[1]
    assert len(native) == 1
    assert "native_extraction" in stages.saved


def test_identification_and_native_stages_reuse_across_runtimes_then_refresh_source():
    ctx, fs, llm, api = setup(lambda _: [])
    identified, selected, native = [], [], []

    def complete_json(prompt, schema):
        if "categories" in schema["properties"]:
            selected.append((prompt, schema))
            return {"categories": ["entity"], "shared_tools": []}
        identified.append((prompt, schema))
        return {"spans": [{"start": 0, "end": 0, "reason": "A personal gift", "categories": ["entity"]}]}

    def complete_tools(**kwargs):
        native.append(kwargs)
        fact = ("The necklace has a heart charm." if "heart charm" in kwargs["prompt"]
                else "Grandmother gave Caroline a necklace.")
        return [{"tool": "extract_entity", "input": {
            "routing_key": "necklace", "object_name": "necklace", "abstract": "Caroline's necklace.",
            "overview": fact, "content": fact, "confidence": .95}}]

    llm.complete_json = complete_json
    llm.complete_with_tools = complete_tools
    first_source = [{"id": "source", "role": "user", "content": "Grandmother gave Caroline a necklace."}]
    next_source = [{**first_source[0], "id": "next-source"}]
    updated_source = [{**next_source[0], "content": "The necklace has a heart charm."}]
    stages = Stages()
    results, spans = [], []
    for source in (first_source, first_source, next_source, updated_source):
        runtime = ExtractionRuntime(stages)
        with runtime.activate():
            results.append(api._pipeline.extract(source, ctx, parallel=False,
                                                 session_time=datetime(2026, 10, 3), archive_id="archive"))
        assert len(runtime.entries) == 1
        spans.append(runtime.entries[0][1].span)
        assert spans[-1].message_ids == [source[0]["id"]]
    assert results[0] == results[1]
    assert spans[0] is not spans[1]
    assert len(identified) == len(selected) == len(native) == 3
    assert "heart charm" in results[-1][0].content
    identification_results = [value for (name, _), value in stages.values.items() if name == "span_identification"]
    assert len(identification_results) == 3
    assert all(set(value["spans"][0]) == {"start", "end", "reason", "categories"}
               for value in identification_results)


def test_same_prepared_span_reuses_native_stage_after_successful_read():
    from extraction.prefetch import PrefetchResult
    ctx, fs, llm, api = setup(lambda _: [])
    uri = api.write_memory(card(), ctx)["target_uri"]
    node = fs.read_node(uri, ctx)
    extractor = api._pipeline._extractors[0]
    source = [{"id": "source", "role": "user", "content": "The necklace has a heart charm."}]
    prepared = extractor._prepare_span(_Span(0, 0, categories=["entity"]), source,
                                       "en", datetime(2026, 10, 3), "", "", ctx)
    prepared.span.prefetch_result = PrefetchResult(
        read_uris={uri}, read_versions={uri: node.metadata["version"]}, read_nodes={uri: node})
    native = []

    def complete_tools(**kwargs):
        native.append(kwargs)
        return [{"tool": "extract_entity", "input": {
            "routing_key": "necklace", "object_name": "necklace", "abstract": "Caroline's necklace.",
            "overview": "A heart charm.", "content": "The necklace has a heart charm.", "confidence": .95}}]

    llm.complete_with_tools = complete_tools
    runtime = ExtractionRuntime(Stages())
    with runtime.activate():
        first = extractor._structure_prepared_span_dual(prepared, ctx, "archive")
        read = fs.read_node(uri, ctx)
        prepared.span.successful_read_versions.append({uri: read.metadata["version"]})
        second = extractor._structure_prepared_span_dual(prepared, ctx, "archive")
    assert first == second
    assert len(native) == 1
    assert prepared.span.prefetch_result.read_versions == {uri: 1}
    updated = fs.read_node(uri, ctx)
    updated.content += "\n- Stored in a blue box."
    fs.write_node(updated, ctx)
    latest = fs.read_node(uri, ctx)
    refreshed = extractor._prepare_span(_Span(0, 0, categories=["entity"]), source,
                                        "en", datetime(2026, 10, 3), "", "", ctx)
    refreshed.span.prefetch_result = PrefetchResult(
        read_uris={uri}, read_versions={uri: latest.metadata["version"]}, read_nodes={uri: latest})
    assert refreshed.prompt == prepared.prompt
    with ExtractionRuntime(runtime.checkpoint).activate():
        extractor._structure_prepared_span_dual(refreshed, ctx, "archive")
    assert len(native) == 2
    assert refreshed.span.prefetch_result.read_versions == {uri: 2}


@pytest.mark.parametrize("feedback", [
    "node_processing: node decisions must account for every draft exactly once",
    "node_processing: node decisions require one update per target",
])
def test_completed_node_group_reuses_base_prompt_and_new_group_receives_feedback(feedback):
    from service.extraction_retry import _format_feedback
    ctx, fs, llm, api = setup(lambda _: [operation([0])])
    stages = Stages()
    processor = api._node_processor()
    first = card(scope_ref="project_a")
    second = card(scope_ref="project_b")
    processor.process([first], [], ctx, checkpoint=stages)
    token = _format_feedback.set(feedback)
    try:
        result = processor.process([first, second], [], ctx, checkpoint=stages)
    finally:
        _format_feedback.reset(token)
    assert len(result) == 2
    assert len(llm.calls) == 2
    assert feedback in llm.calls[-1]["prompt"]
    assert node_payload(llm.calls[-1])["drafts"][0]["scope_ref"] == "project_b"


@pytest.mark.parametrize("feedback", [
    "node_processing: node decisions must account for every draft exactly once",
    "node_processing: node decisions require one update per target",
])
def test_node_feedback_combines_complete_source_facts_and_reuses_saved_stage(feedback):
    from service.extraction_retry import _format_feedback
    ctx, fs, llm, api = setup(lambda _: [])
    uri = api.write_memory(card(scope_ref="project_a"), ctx)["target_uri"]
    facts = ["The necklace has a heart charm.", "The necklace is kept in a blue box."]
    drafts = [card(scope_ref="project_a", content="# Necklace\n\n## Features\n- " + fact,
                   provenance_ids=[f"source:{index}"], source_contexts=[fact])
              for index, fact in enumerate(facts)]

    def decide(call):
        payload = node_payload(call)
        assert [draft["id"] for draft in payload["drafts"]] == [0, 1]
        op = operation([0, 1], target=uri, action="update")
        op["sections"][0]["facts"] = facts
        op["overview"] += facts
        return [op]

    llm.decide = decide
    stages = Stages()
    processor = api._node_processor()
    source = [{"role": "user", "content": " ".join(facts)}]
    token = _format_feedback.set(feedback)
    try:
        result = processor.process(drafts, source, ctx, checkpoint=stages)
    finally:
        _format_feedback.reset(token)
    reused = processor.process(drafts, source, ctx, checkpoint=stages)
    assert reused == result
    assert len(result) == len(llm.calls) == 1
    assert feedback in llm.calls[0]["prompt"]
    candidate = result[0]
    assert candidate.target_uri == uri
    assert candidate.expected_version == 1
    assert candidate.scope_ref == "project_a"
    assert "Gift from Caroline's grandma." in candidate.content
    assert all(fact in candidate.content and fact in candidate.overview for fact in facts)
    assert candidate.provenance_ids == ["source:0", "source:1"]
    assert candidate.source_contexts == facts






@pytest.mark.parametrize("actions", [("update", "noop"), ("noop", "update"), ("noop", "noop")],
                         ids=["update-then-noop", "noop-then-update", "multiple-noops"])
def test_same_target_acknowledgements_commit_complete_card_once_and_reuse_stage(actions):
    ctx, fs, llm, api = setup(lambda _: [])
    known = ["Caroline wants to work in counseling or mental health.",
             "Caroline attended an LGBTQ support group on May 7, 2023."]
    added = "Caroline started transitioning around 2020."
    original = replace(card("caroline"), object_name="Caroline", abstract=known[0],
                       overview="- " + "\n- ".join(known),
                       content="# Caroline\n\n## Background\n- " + "\n- ".join(known))
    uri = api.write_memory(original, ctx)["target_uri"]
    old = fs.read_node(uri, ctx)
    facts = [added if action == "update" else known[index] for index, action in enumerate(actions)]
    drafts = [replace(original, routing_key=f"caroline_source_{index}", content=fact,
                      provenance_ids=[f"prov:1:archive:sample0-session3:source-{index}"],
                      source_contexts=[fact], node_draft_indices=[index],
                      verified_object_aliases=[f"Caroline source {index}"])
              for index, fact in enumerate(facts)]
    source = [{"role": "user", "content": "\n".join(facts)}]

    def decide(**kwargs):
        llm.calls.append(kwargs)
        payload = node_payload(kwargs)
        assert payload["nodes"][0]["uri"] == uri
        operations = []
        for index, action in enumerate(actions):
            op = operation([index], target=uri, action=action, fact=added)
            op.update(routing_key="caroline", object_name="Caroline", abstract=known[0],
                      overview=known + ([added] if action == "update" else []),
                      sections=[{"heading": "Transition", "facts": [added]}] if action == "update" else [])
            operations.append(op)
        acknowledgement = {"target": uri, "source_draft_ids": [0, 1], "decision": "unchanged",
                           "reason": "already_current", "source_quote": next(
                               fact for fact in facts if fact in known),
                           "routing_key": "caroline", "object_name": "Caroline", "abstract": old.abstract,
                           "overview": known, "edits": [], "sections": []}
        return [{"tool": "process_memory_nodes", "input": {
            "operations": operations, "related_reconciliations": [acknowledgement]}}]

    llm.complete_with_tools = decide
    stages = Stages()
    processor = api._node_processor()
    first = processor.process(drafts, source, ctx, checkpoint=stages)
    assert processor.process(drafts, source, ctx, checkpoint=stages) == first
    assert len(llm.calls) == len(first) == 1
    assert stages.saved == ["node_processing"]
    candidate = first[0]
    assert (candidate.target_uri, candidate.expected_version) == (uri, old.metadata["version"])
    assert set(candidate.provenance_ids) == {value for draft in drafts for value in draft.provenance_ids}
    assert set(candidate.source_contexts) == set(facts)
    assert set(candidate.node_draft_indices) == {0, 1}
    assert set(candidate.verified_routing_aliases) >= {draft.routing_key for draft in drafts}
    assert set(candidate.verified_object_aliases) >= {"Caroline", "Caroline source 0", "Caroline source 1"}
    assert all(fact in candidate.content for fact in known)
    if "update" in actions:
        assert added in candidate.content and added in candidate.overview
    else:
        assert (candidate.abstract, candidate.overview, candidate.content) == (old.abstract, old.overview, old.content)

    writes = []
    write_node = fs.write_node

    def record_write(node, request):
        writes.append(node.uri)
        return write_node(node, request)

    fs.write_node = record_write
    api.write_memory(candidate, ctx)
    stored = fs.read_node(uri, ctx)
    assert writes == [uri]
    assert stored.metadata["version"] == old.metadata["version"] + 1
    assert stored.content == candidate.content
    assert set(stored.metadata["provenance_ids"]) >= set(candidate.provenance_ids)


def test_same_target_acknowledgement_stages_updated_body_for_the_next_batch():
    ctx, fs, llm, api = setup(lambda _: [])
    uri = api.write_memory(card(), ctx)["target_uri"]
    original = fs.read_node(uri, ctx)
    facts = ["It has a heart charm.", "Gift from Caroline's grandma.", "It is kept in a blue box."]
    drafts = [card(content=fact, provenance_ids=[f"source:{index}"],
                   source_contexts=[fact], node_draft_indices=[index])
              for index, fact in enumerate(facts)]
    batches = [(drafts[:2], [original]), (drafts[2:], [original])]
    source = [{"role": "user", "content": "\n".join(facts)}]

    def decide(call):
        payload = node_payload(call)
        if len(payload["drafts"]) == 2:
            update = operation([0], target=uri, action="update", fact=facts[0])
            acknowledgement = operation([1], target=uri, action="noop")
            acknowledgement["sections"] = []
            return [update, acknowledgement]
        assert facts[0] in payload["nodes"][0]["content"]
        assert facts[1] in payload["nodes"][0]["content"]
        return [operation([0], target=uri, action="update", fact=facts[2])]

    llm.decide = decide
    stages = Stages()
    processor = api._node_processor()

    def process_chain():
        generator = processor._process_chain(batches, source, ctx, stages)
        response = None
        while True:
            try:
                prompt = generator.send(response)
            except StopIteration as done:
                return done.value
            response = llm.complete_with_tools(prompt=prompt)

    first = process_chain()
    assert process_chain() == first
    assert len(first) == 1 and len(llm.calls) == 2
    assert stages.saved == ["node_processing", "node_processing"]
    candidate = first[0]
    assert all(fact in candidate.content for fact in facts)
    assert set(candidate.node_draft_indices) == {0, 1, 2}
    assert set(candidate.provenance_ids) == {"source:0", "source:1", "source:2"}
    api.write_memory(candidate, ctx)
    assert fs.read_node(uri, ctx).content == candidate.content
    assert fs.read_node(uri, ctx).metadata["version"] == original.metadata["version"] + 1
