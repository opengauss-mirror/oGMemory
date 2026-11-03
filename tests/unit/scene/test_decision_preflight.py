"""Executable preflight, bounded correction and scoped evidence contracts."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from core.provenance_resolver import ProvenanceResolver
from scene.config import SceneConfig
from scene.decision_calls import CallBudget, SceneModelCaller
from scene.decision_contracts import (
    PreflightOutput,
    QualityGateOutput,
    SceneDecisionOutput,
    SceneModelOutputError,
)
from scene.decision_evidence import SceneEvidenceBuilder
from scene.decision_preflight import ScenePreflight
from scene.decision_prompts import PREFLIGHT_SYSTEM
from scene.models import SceneSnapshot
from tests.unit.scene.test_decision_evidence import node, vectors


class JSONModel:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def complete_json(self, prompt, schema):
        self.requests.append((prompt, schema))
        return next(self.outputs)


class ToolModel(JSONModel):
    def __init__(self, response, *outputs):
        super().__init__(*outputs)
        self.response = response
        self.tool_requests = []

    def complete_with_tools_messages(self, messages, tools=None, tool_choice="auto"):
        self.tool_requests.append((messages, tools, tool_choice))
        return deepcopy(self.response)


class Reader:
    def __init__(self):
        self.calls = []

    def search_related_nodes(self, account_id, owner_space, query, limit):
        self.calls.append(("search_related_nodes", account_id, owner_space, query, limit))
        return [{"uri": f"related-{index}", "source_refs": [f"archive-{index}"]} for index in range(20)]

    def search_scene_cards(self, account_id, owner_space, query, limit):
        self.calls.append(("search_scene_cards", account_id, owner_space, query, limit))
        return [{"scene_id": f"scene-{index}"} for index in range(20)]

    def read_source_context(self, account_id, owner_space, source_refs, limit):
        self.calls.append(("read_source_context", account_id, owner_space, source_refs, limit))
        return [{"source_ref": ref, "text": "Release planning evidence"} for ref in source_refs]


def packet():
    nodes = [node(index) for index in range(3)]
    return SceneEvidenceBuilder().build(SceneSnapshot("acct", "user:alice", 0), nodes, vectors(nodes))


def test_three_stages_share_one_batch_budget():
    model = JSONModel({"status": "READY"}, {"action": "WAIT"}, {"decision": "ACCEPT"})
    caller = SceneModelCaller(model)
    budget = CallBudget(3)
    outputs = [caller.call(stage, stage, schema, budget) for stage, schema in (
        ("preflight", PreflightOutput), ("decision", SceneDecisionOutput), ("quality", QualityGateOutput),
    )]
    assert [output.model_dump()[key] for output, key in zip(outputs, ("status", "action", "decision"))] == [
        "READY", "WAIT", "ACCEPT",
    ]
    assert (budget.used, budget.remaining) == (3, 0)
    assert len(model.requests) == 3


@pytest.mark.parametrize("response", [{"status": "READY"}, '{"status":"READY"}', '```json\n{"status":"READY"}\n```'])
def test_json_provider_formats_produce_ready_with_shared_evidence(response):
    evidence = packet()
    before = deepcopy(evidence)
    model = JSONModel(response)
    preflight = ScenePreflight(SceneModelCaller(model))
    budget = CallBudget(3)
    result = preflight.enforce_gate(evidence, preflight.run(evidence, budget))
    assert result.status == "READY"
    assert budget.used == 1
    prompt, schema = model.requests[0]
    assert prompt.startswith(PREFLIGHT_SYSTEM)
    assert schema == PreflightOutput.model_json_schema()
    assert json.loads(prompt.split("\nPACKET:\n", 1)[1]) == evidence == before


@pytest.mark.parametrize("response", [([], '{"status":"READY"}'), '{"status":"READY"}', {"status": "READY"}])
def test_message_provider_formats_use_the_preflight_contract(response):
    evidence = packet()
    model = ToolModel(response)
    preflight = ScenePreflight(SceneModelCaller(model), evidence_reader=Reader())
    budget = CallBudget(3)
    assert preflight.run(evidence, budget).status == "READY"
    messages, tools, choice = model.tool_requests[0]
    assert messages[0]["content"].startswith(PREFLIGHT_SYSTEM)
    assert json.loads(messages[1]["content"]) == evidence
    assert [tool["name"] for tool in tools] == ["search_related_nodes", "search_scene_cards", "read_source_context"]
    assert choice == "auto"
    assert (budget.used, budget.remaining) == (1, 2)


def test_json_capability_fallback_completes_preflight():
    class JSONOnly(JSONModel):
        def complete_with_tools_messages(self, **kwargs):
            raise NotImplementedError

    model = JSONOnly({"status": "READY"})
    budget = CallBudget(3)
    assert ScenePreflight(SceneModelCaller(model), evidence_reader=Reader()).run(packet(), budget).status == "READY"
    assert len(model.requests) == budget.used == 1


@pytest.mark.parametrize("initial", ['{"status":"READ"}', '{"status":', '{"status":"READY","private-key":"value"}'])
def test_corrected_preflight_resumes_with_one_call_reserved(initial):
    model = ToolModel(([], initial), {"status": "READY"})
    evidence = packet()
    preflight = ScenePreflight(SceneModelCaller(model), evidence_reader=Reader())
    budget = CallBudget(3)
    assert preflight.run(evidence, budget).status == "READY"
    assert (budget.used, budget.remaining) == (2, 1)
    assert len(model.tool_requests) == len(model.requests) == 1
    prompt, schema = model.requests[0]
    correction = prompt.split("\nOUTPUT CORRECTION:\n", 1)[1].split("\n", 1)[0]
    diagnostic = json.loads(correction)
    assert diagnostic["stage"] == "preflight"
    assert diagnostic["code"] == "scene_model_output_invalid"
    assert all(set(error) == {"field", "type"} for error in diagnostic["errors"])
    assert {error["field"] for error in diagnostic["errors"]} <= {"status", "$", "[field]"}
    assert schema == PreflightOutput.model_json_schema()


def test_stage_validator_correction_retains_structured_diagnostics():
    model = JSONModel({"status": "READY"}, {"status": "READY", "anchors": ["Atlas release"]})
    caller = SceneModelCaller(model)
    budget = CallBudget(3)

    def require_anchor(output):
        if not output.anchors:
            raise SceneModelOutputError("preflight", [{"field": "anchors", "type": "anchor_required"}])

    result = caller.call("preflight", "Evidence", PreflightOutput, budget, validator=require_anchor)
    assert result.anchors == ["Atlas release"]
    assert budget.used == 2
    diagnostic = model.requests[1][0].split("\nOUTPUT CORRECTION:\n", 1)[1].split("\n", 1)[0]
    assert json.loads(diagnostic)["errors"] == [{"field": "anchors", "type": "anchor_required"}]


@pytest.mark.parametrize("tool", ["search_related_nodes", "search_scene_cards", "read_source_context"])
@pytest.mark.parametrize("response_format", ["tuple", "list"])
def test_tools_use_packet_scope_and_bound_arguments_and_results(tool, response_format):
    evidence = packet()
    reader = Reader()
    args = {"query": "Atlas " * 100, "source_refs": [f"archive-{i}" for i in range(20)], "limit": 99,
            "account_id": "supplied-account", "owner_space": "supplied-owner"}
    calls = [{"tool": tool, "input": args}] * 3
    response = (calls, "Retrieving evidence") if response_format == "tuple" else calls
    model = ToolModel(response)
    preflight = ScenePreflight(SceneModelCaller(model), evidence_reader=reader)
    budget = CallBudget(3)
    assert preflight.run(evidence, budget).status == "READY"
    assert budget.used == 1
    assert len(reader.calls) == len(evidence["tool_evidence"]) == 2
    for name, account, owner, argument, limit in reader.calls:
        assert (name, account, owner, limit) == (tool, "acct", "user:alice", 16)
        assert argument == (args["source_refs"][:16] if tool == "read_source_context" else args["query"][:500])
    assert [len(result["results"]) for result in evidence["tool_evidence"]] == [16, 16]


@pytest.mark.parametrize("owner", ["user:alice", "agent:helper"])
def test_adapter_tools_acquire_canonical_evidence_for_preflight(owner):
    from tests.unit.scene.test_adapters import adapter
    from tests.unit.scene.test_adapters import node as stored_node

    nodes = [stored_node(str(index), owner=owner) for index in range(3)]
    source = adapter(nodes)
    batch = source.load_input("acme", owner, node_uris=[item.uri for item in nodes])
    evidence = SceneEvidenceBuilder().build(batch.snapshot, list(batch.nodes), batch.embeddings)
    calls = [{"tool": "read_source_context", "input": {"source_refs": [item.uri for item in nodes]}}]
    preflight = ScenePreflight(SceneModelCaller(ToolModel((calls, ""))), evidence_reader=source)
    result = preflight.run(evidence, CallBudget(3))
    cards = evidence["tool_evidence"][0]["results"]
    assert [card["uri"] for card in cards] == [item.uri for item in nodes]
    assert all("Evidence for" in card["text"] for card in cards)
    assert preflight.enforce_gate(evidence, result).status == "READY"


def test_additional_independent_source_advances_evidence_floor_to_ready():
    evidence = packet()
    for index, card in enumerate(evidence["memories"]):
        card["source_refs"] = [ProvenanceResolver.build_id("archive", "meeting", [f"m{index}"])]
    preflight = ScenePreflight(SceneModelCaller(JSONModel()))
    proposal = PreflightOutput(status="READY")
    waiting = preflight.enforce_gate(evidence, proposal)
    assert waiting.status == "WAIT"
    assert waiting.evidence_uris == evidence["offered_member_uris"]
    evidence["tool_evidence"] = [{"results": [{
        "source_ref": ProvenanceResolver.build_id("archive", "review", ["m1"]),
    }]}]
    assert preflight.enforce_gate(evidence, proposal) == proposal


def test_model_wait_is_preserved_when_evidence_floor_is_satisfied():
    preflight = ScenePreflight(SceneModelCaller(JSONModel({"status": "WAIT", "reason": "Awaiting release date"})))
    evidence = packet()
    proposal = preflight.run(evidence, CallBudget(3))
    assert preflight.enforce_gate(evidence, proposal) == proposal
    assert proposal.reason == "Awaiting release date"


def test_configured_evidence_count_advances_after_canonical_expansion():
    config = replace(SceneConfig(), min_evidence_nodes=4)
    preflight = ScenePreflight(SceneModelCaller(JSONModel()), config)
    evidence = packet()
    proposed = PreflightOutput(status="READY")
    assert preflight.enforce_gate(evidence, proposed).status == "WAIT"
    evidence["memories"].append(SceneEvidenceBuilder().node_card(node(3)))
    assert preflight.enforce_gate(evidence, proposed).status == "READY"


@pytest.mark.parametrize("reference_kind", ["uri", "provenance"])
def test_adapter_source_gate_advances_when_second_archive_arrives(reference_kind):
    from tests.unit.scene.test_adapters import adapter
    from tests.unit.scene.test_adapters import node as stored_node

    nodes = [stored_node(str(index)) for index in range(3)]
    for index, item in enumerate(nodes):
        item.metadata["provenance_ids"] = [ProvenanceResolver.build_id("archive", "meeting", [f"m{index}"])]
    source = adapter(nodes)
    batch = source.load_input("acme", "user:alice", node_uris=[item.uri for item in nodes])
    evidence = SceneEvidenceBuilder().build(batch.snapshot, list(batch.nodes), batch.embeddings)
    refs = [item.uri if reference_kind == "uri" else ProvenanceResolver.build_id("memory", item.uri)
            for item in nodes]
    calls = [{"tool": "read_source_context", "input": {"source_refs": refs}}]
    preflight = ScenePreflight(SceneModelCaller(ToolModel((calls, ""))), evidence_reader=source)
    proposal = preflight.run(evidence, CallBudget(3))
    assert len(evidence["tool_evidence"][0]["results"]) == 3
    assert preflight.enforce_gate(evidence, proposal).status == "WAIT"

    nodes[2].metadata["provenance_ids"] = [ProvenanceResolver.build_id("archive", "review", ["m0"])]
    evidence["tool_evidence"][0]["results"] = source.read_source_context("acme", "user:alice", refs, 8)
    assert preflight.enforce_gate(evidence, proposal) == proposal


@pytest.mark.parametrize("args", [
    {"source_refs": None}, {"source_refs": 42}, {"source_refs": "archive"},
    {"source_refs": [42]}, {}, {"source_refs": ["archive"], "limit": "8"},
])
def test_tool_argument_correction_recovers_with_reserved_call(args):
    reader = Reader()
    model = ToolModel(([{"tool": "read_source_context", "input": args}], ""),
                      {"status": "READY", "reason": "Packet evidence is sufficient"})
    evidence = packet()
    preflight = ScenePreflight(SceneModelCaller(model), evidence_reader=reader)
    budget = CallBudget(3)
    proposal = preflight.run(evidence, budget)
    assert preflight.enforce_gate(evidence, proposal).reason == "Packet evidence is sufficient"
    assert (budget.used, budget.remaining) == (2, 1)
    assert reader.calls == []
    diagnostic = json.loads(model.requests[0][0].split("\nOUTPUT CORRECTION:\n", 1)[1].split("\n", 1)[0])
    assert diagnostic["stage"] == "preflight"
    assert diagnostic["code"] == "scene_model_output_invalid"
    assert diagnostic["errors"][0]["field"].startswith("read_source_context.input")


@pytest.mark.parametrize("status", ["READY", "WAIT"])
def test_configured_five_anchors_preserves_valid_model_proposal(status):
    nodes = [node(index) for index in range(5)]
    evidence = SceneEvidenceBuilder().build(SceneSnapshot("acct", "user:alice", 0), nodes, vectors(nodes))
    config = replace(SceneConfig(), max_anchor_count=5)
    model = JSONModel({"status": status, "anchors": [nodes[0].uri], "reason": "Model evidence decision"})
    preflight = ScenePreflight(SceneModelCaller(model), config)
    proposal = preflight.run(evidence, CallBudget(3))
    assert preflight.enforce_gate(evidence, proposal) == proposal


@pytest.mark.parametrize("tool", ["search_related_nodes", "search_scene_cards"])
@pytest.mark.parametrize("args", [{"query": None}, {"query": 42}, {"query": "Atlas", "limit": True}])
def test_provider_tool_batch_recovers_then_reads_validated_evidence(tool, args):
    from providers.llm.openai_llm import OpenAILLM

    valid = {"tool": "read_source_context", "input": {"source_refs": ["archive"]}}
    malformed = {"tool": tool, "input": OpenAILLM._parse_tool_args(json.dumps(args))}
    model = ToolModel(([valid, malformed], ""), {"status": "READY", "reason": "Use packet evidence"})
    reader = Reader()
    preflight = ScenePreflight(SceneModelCaller(model), evidence_reader=reader)
    budget = CallBudget(3)
    assert preflight.run(packet(), budget).reason == "Use packet evidence"
    assert (budget.used, budget.remaining, reader.calls) == (2, 1, [])

    model.response = ([valid, {"tool": tool, "input": {"query": "Atlas", "limit": 2}}], "")
    evidence = packet()
    recovered_budget = CallBudget(3)
    proposal = preflight.run(evidence, recovered_budget)
    assert preflight.enforce_gate(evidence, proposal).status == "READY"
    assert [result["tool"] for result in evidence["tool_evidence"]] == ["read_source_context", tool]
    assert [len(result["results"]) for result in evidence["tool_evidence"]] == [1, 2]
    assert len(reader.calls) == 2
    assert (recovered_budget.used, recovered_budget.remaining) == (1, 2)


@pytest.mark.parametrize("stored_format,read_format", [
    ("id", "uri"), ("id", "provenance_uri"), ("provenance_uri", "uri"),
    ("provenance_uri", "id"), ("uri", "id"), ("uri", "provenance_uri"),
    ("id", "memory_uri"), ("memory_uri", "uri"), ("uri", "memory_uri"),
])
def test_archive_reference_formats_advance_only_with_distinct_source(stored_format, read_format):
    from tests.unit.scene.test_adapters import Archives, adapter, archive
    from tests.unit.scene.test_adapters import node as stored_node

    uri = "ctx://acme/sessions/planning/history/source"
    formats = {"id": ProvenanceResolver.build_id("archive", "source", ["m1"]),
               "uri": uri, "provenance_uri": ProvenanceResolver.build_id("archive", uri, ["m2"]),
               "memory_uri": ProvenanceResolver.build_id("memory", uri)}
    nodes = [stored_node(str(index)) for index in range(3)]
    for item in nodes:
        item.metadata["provenance_ids"] = [formats[stored_format]]
    source = adapter(nodes, archive_store=Archives([archive("source"), archive("review")]))
    batch = source.load_input("acme", "user:alice", node_uris=[item.uri for item in nodes])
    evidence = SceneEvidenceBuilder().build(batch.snapshot, list(batch.nodes), batch.embeddings)
    calls = [{"tool": "read_source_context", "input": {"source_refs": [formats[read_format]]}}]
    preflight = ScenePreflight(SceneModelCaller(ToolModel((calls, ""))), evidence_reader=source)
    proposal = preflight.run(evidence, CallBudget(3))
    cards = evidence["tool_evidence"][0]["results"]
    assert cards[0]["uri"] == uri
    assert cards[0]["message_ids"]
    assert preflight.enforce_gate(evidence, proposal).status == "WAIT"

    cards.extend(source.read_source_context("acme", "user:alice",
                 [ProvenanceResolver.build_id("archive", "review", ["m1"])], 8))
    assert preflight.enforce_gate(evidence, proposal).status == "READY"


@pytest.mark.parametrize("status", ["WAIT", "READY"])
def test_packet_only_preflight_uses_structured_model_judgment(status):
    calls = [{"tool": "read_source_context", "input": {"source_refs": ["archive"]}}]
    judgment = {"status": status, "reason": "Packet evidence judgment"}
    model = ToolModel((calls, ""), judgment)
    preflight = ScenePreflight(SceneModelCaller(model))
    evidence = packet()
    budget = CallBudget(3)
    proposal = preflight.run(evidence, budget)
    assert preflight.enforce_gate(evidence, proposal) == PreflightOutput(**judgment)
    assert len(model.requests) == 1
    assert model.requests[0][1] == PreflightOutput.model_json_schema()
    assert "Evidence mode: packet-only." in model.requests[0][0]
    assert model.tool_requests == []
    assert (budget.used, budget.remaining) == (1, 2)


def test_archive_identities_retain_tenant_and_session_scope():
    uris = {"ctx://acme/sessions/planning/history/source",
            "ctx://acme/sessions/review/history/source",
            "ctx://other/sessions/planning/history/source"}
    refs = sorted(uris) + [ProvenanceResolver.build_id("archive", uri, ["m1"]) for uri in sorted(uris)]
    refs.append(ProvenanceResolver.build_id("archive", "source"))
    expected = {("archive", uri) for uri in uris}
    assert ScenePreflight.independent_sources(refs) == expected
    assert ScenePreflight.independent_sources(reversed(refs)) == expected


def test_packet_only_judgment_recovers_with_shared_budget():
    model = JSONModel('{"status":"READ"}', {"status": "WAIT", "reason": "Awaiting original evidence"})
    preflight = ScenePreflight(SceneModelCaller(model))
    budget = CallBudget(3)
    evidence = packet()
    proposal = preflight.run(evidence, budget)
    assert preflight.enforce_gate(evidence, proposal).reason == "Awaiting original evidence"
    assert proposal.status == "WAIT"
    assert (len(model.requests), budget.used, budget.remaining) == (2, 2, 1)


@pytest.mark.parametrize("owner", ["user:alice", "agent:helper"])
@pytest.mark.parametrize("stored_format", ["uri", "provenance", "mixed"])
def test_memory_reference_formats_advance_with_second_canonical_source(owner, stored_format):
    from tests.unit.scene.test_adapters import adapter
    from tests.unit.scene.test_adapters import node as stored_node

    original, second = [stored_node(name, owner=owner) for name in ("source", "review")]
    refs = [original.uri, ProvenanceResolver.build_id("memory", original.uri)]
    nodes = [stored_node(str(index), owner=owner) for index in range(3)]
    for index, item in enumerate(nodes):
        ref_index = index % 2 if stored_format == "mixed" else int(stored_format == "provenance")
        item.metadata["provenance_ids"] = [refs[ref_index]]
    source = adapter([*nodes, original, second])
    batch = source.load_input("acme", owner, node_uris=[item.uri for item in nodes])
    evidence = SceneEvidenceBuilder().build(batch.snapshot, list(batch.nodes), batch.embeddings)
    calls = [{"tool": "read_source_context", "input": {"source_refs": refs}}]
    preflight = ScenePreflight(SceneModelCaller(ToolModel((calls, ""))), evidence_reader=source)
    proposal = preflight.run(evidence, CallBudget(3))
    cards = evidence["tool_evidence"][0]["results"]
    assert [card["uri"] for card in cards] == [original.uri, original.uri]
    assert [card["text"] for card in cards] == [original.content, original.content]
    assert preflight.enforce_gate(evidence, proposal).status == "WAIT"

    cards.extend(source.read_source_context("acme", owner, [second.uri], 8))
    assert cards[-1]["uri"] == second.uri
    assert cards[-1]["text"] == second.content
    assert preflight.enforce_gate(evidence, proposal).status == "READY"


def test_memory_identities_retain_tenant_owner_and_path_scope():
    uris = {"ctx://acme/users/alice/memories/events/source",
            "ctx://other/users/alice/memories/events/source",
            "ctx://acme/users/bob/memories/events/source",
            "ctx://acme/agents/alice/memories/events/source",
            "ctx://acme/users/alice/memories/entities/source"}
    refs = sorted(uris) + [ProvenanceResolver.build_id(kind, uri)
                           for kind in ("memory", "archive") for uri in sorted(uris)]
    expected = {("memory", uri) for uri in uris}
    assert ScenePreflight.independent_sources(refs) == expected
    assert ScenePreflight.independent_sources(reversed(refs)) == expected


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("root_count", [1, 2])
def test_resolved_memory_chain_advances_with_second_archive(depth, reverse, root_count):
    from tests.unit.scene.test_adapters import Archives, adapter, archive
    from tests.unit.scene.test_adapters import node as stored_node

    chain = [stored_node(f"source-{index}") for index in range(depth)]
    reference = ProvenanceResolver.build_id("archive", "meeting", ["m1"])
    for item in chain:
        item.metadata["provenance_ids"] = [reference]
        reference = ProvenanceResolver.build_id("memory", item.uri)
    chain[0].metadata["provenance_ids"] = [ProvenanceResolver.build_id("archive", name, ["m1"])
                                          for name in ("meeting", "review")[:root_count]]
    nodes = [stored_node(str(index)) for index in range(3)]
    for index, item in enumerate(nodes):
        item.metadata["provenance_ids"] = [reference if index % 2 else chain[-1].uri]
    source = adapter([*nodes, *chain], archive_store=Archives([archive("meeting"), archive("review")]))
    batch = source.load_input("acme", "user:alice", node_uris=[item.uri for item in nodes])
    evidence = SceneEvidenceBuilder().build(batch.snapshot, list(batch.nodes), batch.embeddings)
    refs = [item.uri for item in chain]
    if reverse:
        refs.reverse()
    calls = [{"tool": "read_source_context", "input": {"source_refs": refs}}]
    preflight = ScenePreflight(SceneModelCaller(ToolModel((calls, ""))), evidence_reader=source)
    proposal = preflight.run(evidence, CallBudget(3))
    cards = evidence["tool_evidence"][0]["results"]
    assert [card["uri"] for card in cards] == refs
    assert all(card["source_refs"] for card in cards)
    assert preflight.enforce_gate(evidence, proposal).status == ("WAIT" if root_count == 1 else "READY")
    cards.extend(source.read_source_context("acme", "user:alice",
                 [ProvenanceResolver.build_id("archive", "review", ["m1"])], 8))
    assert cards[-1]["message_ids"] == ["m1"]
    assert preflight.enforce_gate(evidence, proposal).status == "READY"


@pytest.mark.parametrize("read_format", ["uri", "memory", "archive"])
def test_raw_chunk_aliases_advance_with_second_written_source(read_format):
    from providers.llm import MockLLM
    from scene.adapters import _scope_context
    from service.api import MemoryWriteAPI
    from tests.unit.scene.test_adapters import adapter
    from tests.unit.scene.test_adapters import node as stored_node

    nodes = [stored_node(str(index)) for index in range(3)]
    source = adapter(nodes)
    ctx = _scope_context("acme", "user:alice")
    api = MemoryWriteAPI(source.fs, MockLLM(), directory_summary_enabled=False)
    first = api.write_raw_chunk([{"role": "user", "content": "Release moves to Friday."}],
                                ctx, session_id="planning", chunk_index=0)["target_uri"]
    second = api.write_raw_chunk([{"role": "user", "content": "Review confirms the release date."}],
                                 ctx, session_id="planning", chunk_index=1)["target_uri"]
    for index, item in enumerate(nodes):
        item.metadata["provenance_ids"] = [ProvenanceResolver.build_id("archive", first) if index % 2 else first]
    batch = source.load_input("acme", "user:alice", node_uris=[item.uri for item in nodes])
    evidence = SceneEvidenceBuilder().build(batch.snapshot, list(batch.nodes), batch.embeddings)
    ref = first if read_format == "uri" else ProvenanceResolver.build_id(read_format, first)
    calls = [{"tool": "read_source_context", "input": {"source_refs": [ref]}}]
    preflight = ScenePreflight(SceneModelCaller(ToolModel((calls, ""))), evidence_reader=source)
    proposal = preflight.run(evidence, CallBudget(3))
    cards = evidence["tool_evidence"][0]["results"]
    assert cards[0]["uri"] == first
    assert cards[0]["category"] == "session_archive"
    assert cards[0]["text"] == "user: Release moves to Friday."
    assert preflight.enforce_gate(evidence, proposal).status == "WAIT"
    cards.extend(source.read_source_context("acme", "user:alice", [second], 8))
    assert cards[-1]["text"] == "user: Review confirms the release date."
    assert preflight.enforce_gate(evidence, proposal).status == "READY"
