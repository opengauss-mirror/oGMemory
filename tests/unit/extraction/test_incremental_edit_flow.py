"""Read-bound incremental updates preserve untouched facts and identity."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from core.errors import ConcurrentModificationError, ToolCallProtocolError
from core.models import RequestContext
from core.uri_resolver import URIResolver
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.prompts import PromptManager
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import build_extraction_tools, parse_tool_call
from extraction.tools import Extractor, _Span
from tests.unit.commit.test_memory_identity import DecisionLLM, MemoryFS, api, memory
from tests.unit.commit.test_profile_merge import profile
from tests.unit.extraction.test_memory_update_flow import ExtractionProvider, setup_flow


def edit(uri, *, version=1, **changes):
    fields = dict(target_uri=uri, expected_version=version, search="technician",
                  replacement="engineer", abstract="Nora is an engineer in Utrecht.",
                  overview="Nora lives in Utrecht and works as an engineer.", confidence=.9,
                  evidence_quote="I now work as an engineer.", attributed_speaker="user",
                  attribution_basis="self_first_person")
    fields.update(changes)
    return {"name": "edit_memory", "input": fields}


def prepare(extractor, ctx, category="profile"):
    return extractor._prepare_span(_Span(0, 0, categories=[category], message_ids=["m-new"]),
                                   [{"id": "m-new", "role": "user", "content": "I now work as an engineer."}],
                                   "English", None, "", "", ctx)


def eager_flow(operation=None, original=None, category="profile"):
    original = original or profile("identity", "Name: Nora. Residence: Utrecht. Occupation: technician.")
    ctx, fs, registry, uri, embedder, index, prefetcher = setup_flow(category, original)
    llm = ExtractionProvider(operation or edit(uri))
    extractor = Extractor(llm, mode="eager", schema_registry=registry, fs=fs,
                          vector_index=index, embedder=embedder, uri_resolver=URIResolver(registry))
    return ctx, fs, registry, uri, llm, extractor, prefetcher


def profile_read(user_id):
    ctx = RequestContext("profile-test", user_id, "assistant", "session", "trace")
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    uri = writer.write_memory(profile("identity", "Residence: Utrecht. Occupation: technician."), ctx)["target_uri"]
    return ctx, fs, writer, uri, fs.read_node(uri, ctx)


def profile_update(name, uri):
    inputs = edit(uri)["input"]
    if name == "append_memory":
        inputs.pop("search")
        inputs.pop("replacement")
        inputs.update(text="Keeps a checklist.", abstract="Profile with a checklist.",
                      overview="Lives in Utrecht, works as a technician and keeps a checklist.",
                      evidence_quote="I keep a checklist.")
    return inputs


@pytest.mark.parametrize("user_id", ["reader", "MixedCase", "Élodie", "用户A"])
@pytest.mark.parametrize("name", ["edit_memory", "append_memory"])
def test_profile_update_accepts_the_same_principals_successful_public_read(user_id, name):
    ctx, fs, writer, uri, node = profile_read(user_id)
    parsed = parse_tool_call(name, profile_update(name, uri), SchemaRegistry(),
                             read_nodes={uri: node}, ctx=ctx)
    candidate = replace(parsed[2], provenance_ids=["source:update"])
    assert (candidate.target_uri, candidate.expected_version, candidate.routing_key) == (uri, 1, "identity")
    assert writer.write_memory(candidate, ctx)["target_uri"] == uri
    stored = fs.read_node(uri, ctx)
    expected = ("Residence: Utrecht. Occupation: engineer." if name == "edit_memory"
                else node.content + "\nKeeps a checklist.")
    assert stored.content == expected
    assert (stored.owner_space, stored.category, stored.context_type) == (ctx.user_space_name(), "profile", "MEMORY")
    assert stored.metadata["version"] == 2
    assert set(stored.metadata["provenance_ids"]) == {"source:identity", "source:update"}


@pytest.mark.parametrize("name", ["edit_memory", "append_memory"])
@pytest.mark.parametrize("bad", ["owner", "category", "version", "context_type", "principal", "account", "path"])
def test_profile_update_keeps_read_namespace_and_version_checks(name, bad):
    ctx, fs, _, uri, node = profile_read("MixedCase")
    inputs = profile_update(name, uri)
    if bad == "owner":
        node.owner_space = "user:other"
    elif bad == "category":
        node.category = "entity"
    elif bad == "version":
        inputs["expected_version"] = 2
    elif bad == "context_type":
        node.context_type = "SKILL"
    elif bad == "principal":
        ctx = replace(ctx, user_id="other")
    elif bad == "account":
        ctx = replace(ctx, account_id="other")
    elif bad == "path":
        node.uri = uri + "/child"
        inputs["target_uri"] = node.uri
    with pytest.raises(ToolCallProtocolError):
        parse_tool_call(name, inputs, SchemaRegistry(), read_nodes={node.uri: node}, ctx=ctx)
    assert fs.nodes[uri].content == "Residence: Utrecht. Occupation: technician."
    assert fs.nodes[uri].metadata["version"] == 1


def test_incremental_tools_do_not_offer_identity_or_occurrence_arguments():
    _, _, registry, _, _, _, _ = eager_flow()
    tools = {tool["name"]: tool for tool in build_extraction_tools(registry)}
    assert {"edit_memory", "append_memory"} <= tools.keys()
    for name in ("edit_memory", "append_memory"):
        properties = tools[name]["input_schema"]["properties"]
        assert not ({"category", "owner_scope", "routing_key", "object_name", "scope_ref", "who"} & properties.keys())
    assert "target_uri" not in tools["extract_event"]["input_schema"]["properties"]
    assert "target_uri" not in tools["extract_case"]["input_schema"]["properties"]


def test_eager_edit_preserves_old_facts_and_public_write_provenance():
    ctx, fs, _, uri, llm, extractor, _ = eager_flow()
    fs.nodes[uri].metadata["provenance_ids"] = ["source:old"]
    candidate = extractor._structure_prepared_span(prepare(extractor, ctx), ctx, "archive-new")[0]
    assert candidate.content == "Name: Nora. Residence: Utrecht. Occupation: engineer."
    assert (candidate.category, candidate.owner_scope, candidate.routing_key) == ("profile", "user", "identity")
    assert (candidate.target_uri, candidate.expected_version) == (uri, 1)
    assert candidate.evidence_quote == "I now work as an engineer."
    written = api(fs, llm).write_memory(candidate, ctx)
    node = fs.read_node(uri, ctx)
    assert written["target_uri"] == uri and node.metadata["version"] == 2
    assert node.content == candidate.content
    assert node.metadata["provenance_ids"][0] == "source:old"
    assert len(node.metadata["provenance_ids"]) == 2
    assert llm.extractions == 1 and not llm.calls


def test_eager_edit_uses_supplied_snapshot_when_storage_changes():
    ctx, fs, _, uri, llm, extractor, _ = eager_flow()
    prepared = prepare(extractor, ctx)
    newer = fs.read_node(uri, ctx)
    newer.content = "Name: Nora. Residence: Paris. Occupation: technician."
    fs.write_node(newer, ctx)
    candidate = extractor._structure_prepared_span(prepared, ctx)[0]
    assert "Utrecht" in candidate.content and "Paris" not in candidate.content
    assert candidate.expected_version == 1
    with pytest.raises(ConcurrentModificationError):
        api(fs, llm).write_memory(candidate, ctx)
    assert fs.read_node(uri, ctx).content == newer.content


def test_json_fallback_materializes_the_same_edit():
    ctx, _, _, uri, llm, extractor, _ = eager_flow()
    llm.complete_json = lambda prompt, schema: {"operations": [{"tool": "edit_memory", "input": edit(uri)["input"]}]}
    candidate = extractor._structure_span_json_fallback(prepare(extractor, ctx), ctx, None)[0]
    assert candidate.content == "Name: Nora. Residence: Utrecht. Occupation: engineer."
    assert candidate.expected_version == 1


def test_append_keeps_entity_identity_scope_and_metadata():
    old = memory("Atlas", object_name="Atlas", scope_ref="Project A", content="Installed on rack 4.",
                 who="Panel", when="2026-01-01", where="Lab")
    ctx, fs, _, uri, llm, extractor, _ = eager_flow(original=old, category="entity")
    llm.operation = {"name": "append_memory", "input": dict(target_uri=uri, expected_version=1,
        text="Calibrated on Tuesday.", abstract="Atlas was installed and calibrated.",
        overview="Atlas in Project A was installed on rack 4 and calibrated on Tuesday.", confidence=.9)}
    candidate = extractor._structure_prepared_span(prepare(extractor, ctx, "entity"), ctx)[0]
    assert candidate.content == "Installed on rack 4.\nCalibrated on Tuesday."
    assert (candidate.object_name, candidate.scope_ref, candidate.who, candidate.when, candidate.where) == (
        "Atlas", "Project A", "Panel", "2026-01-01", "Lab")
    assert candidate.routing_key == fs.read_node(uri, ctx).metadata["routing_key"]


def test_lazy_record_retains_the_snapshot_at_execution_not_last_read():
    ctx, fs, registry, uri, llm, _, _ = eager_flow()
    calls = 0
    def respond(messages, tools, tool_choice):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{"name": "read", "input": {"uri": uri}}], ""
        if calls == 2:
            return [edit(uri), {"name": "read", "input": {"uri": uri}}], ""
        return [], "[]"
    llm.complete_with_tools_messages = respond
    reads = 0
    original_read = fs.read_node
    def advancing_read(uri, ctx):
        nonlocal reads
        reads += 1
        if reads == 2:
            newer = original_read(uri, ctx)
            newer.content = "Name: Nora. Residence: Paris. Occupation: technician."
            fs.write_node(newer, ctx)
        return original_read(uri, ctx)
    fs.read_node = advancing_read
    loop = ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PromptManager())
    result = loop.run("I now work as an engineer.", ctx)
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.expected_version == 1 and "Utrecht" in candidate.content and "Paris" not in candidate.content
    assert result.read_versions[uri] == 2


@pytest.mark.parametrize("changes", [
    {"search": "absent"}, {"search": ""}, {"replacement": 4}, {"expected_version": 2},
    {"routing_key": "other"}, {"scope_ref": "Other Project"}, {"abstract": "x" * 201},
    {"evidence_quote": ""}, {"attributed_speaker": ""},
])
def test_invalid_edit_fails_without_writing(changes):
    ctx, fs, _, uri, llm, extractor, _ = eager_flow()
    llm.operation = edit(uri, **changes)
    with pytest.raises(ToolCallProtocolError):
        extractor._structure_prepared_span(prepare(extractor, ctx), ctx)
    assert fs.read_node(uri, ctx).metadata["version"] == 1


def test_ambiguous_replace_is_rejected():
    ctx, fs, _, uri, llm, extractor, _ = eager_flow(original=profile("identity", "technician and technician"))
    with pytest.raises(ToolCallProtocolError):
        extractor._structure_prepared_span(prepare(extractor, ctx), ctx)
    assert fs.read_node(uri, ctx).content == "technician and technician"


def test_catalog_hint_does_not_authorize_edit():
    ctx, fs, _, uri, _, extractor, _ = eager_flow()
    prepared = prepare(extractor, ctx)
    prepared.span.prefetch_result.messages.clear()
    prepared.span.prefetch_result.read_uris.clear()
    prepared.span.prefetch_result.read_versions.clear()
    if hasattr(prepared.span.prefetch_result, "read_nodes"):
        prepared.span.prefetch_result.read_nodes.clear()
    with pytest.raises(ToolCallProtocolError):
        extractor._structure_prepared_span(prepared, ctx)
    assert fs.read_node(uri, ctx).metadata["version"] == 1


@pytest.mark.parametrize("category", ["event", "case", "skill", "tool"])
def test_incremental_operation_cannot_bind_excluded_categories(category):
    old = memory("inspection", category=category, content="technician inspected unit.")
    ctx, fs, _, uri, llm, extractor, _ = eager_flow(original=old, category=category)
    llm.operation = edit(uri)
    with pytest.raises(ToolCallProtocolError):
        extractor._structure_prepared_span(prepare(extractor, ctx, category), ctx)
    assert fs.read_node(uri, ctx).metadata["version"] == 1


@pytest.mark.parametrize("mode", ["eager", "json", "lazy"])
def test_two_edits_compose_locally_on_one_read_version(mode):
    ctx, _, registry, uri, llm, extractor, prefetcher = eager_flow()
    first = edit(uri)
    second = edit(uri, search="Utrecht", replacement="Paris", abstract="Nora is an engineer in Paris.",
                  overview="Nora lives in Paris and works as an engineer.")
    if mode == "eager":
        llm.complete_with_tools = lambda **kw: [{"tool": op["name"], "input": op["input"]} for op in (first, second)]
        candidates = extractor._structure_prepared_span(prepare(extractor, ctx), ctx)
    elif mode == "json":
        llm.complete_json = lambda prompt, schema: {"operations": [{"tool": op["name"], "input": op["input"]} for op in (first, second)]}
        candidates = extractor._structure_span_json_fallback(prepare(extractor, ctx), ctx, None)
    else:
        llm.complete_with_tools_messages = lambda messages, tools, tool_choice: (
            ([], "[]") if messages[-1]["role"] == "tool" or messages[-1]["role"] == "user" and "FINAL INSTRUCTION" in messages[-1].get("content", "")
            else ([first, second], ""))
        loop = ExtractionReActLoop(llm, extractor._fs, registry, URIResolver(registry), PromptManager())
        candidates = loop.run("I now work as an engineer and live in Paris.", ctx,
                              prefetcher.prefetch_for_span(["profile"], ctx)).candidates
    assert len(candidates) == 1
    assert candidates[0].content == "Name: Nora. Residence: Paris. Occupation: engineer."
    assert candidates[0].expected_version == 1


def test_append_replay_does_not_duplicate_existing_text():
    ctx, fs, registry, uri, llm, _, _ = eager_flow(original=memory("Atlas", content="Installed."), category="entity")
    inputs = dict(target_uri=uri, expected_version=1, text="Calibrated.", abstract="Installed and calibrated.",
                  overview="Installed and calibrated.", confidence=.9)
    node = fs.read_node(uri, ctx)
    first = parse_tool_call("append_memory", inputs, registry, read_nodes={uri: node}, ctx=ctx)[2]
    writer = api(fs, llm)
    writer.write_memory(first, ctx)
    current = fs.read_node(uri, ctx)
    replay = parse_tool_call("append_memory", dict(inputs, expected_version=2), registry,
                             read_nodes={uri: current}, ctx=ctx)[2]
    assert replay.content == "Installed.\nCalibrated."
    assert writer.write_memory(replay, ctx)["action"] == "skip"
    assert fs.read_node(uri, ctx).metadata["version"] == 2


def test_edit_unicode_literal_and_inputs_remain_unchanged():
    old = memory("Atlas", content="前置事实。读数：[a.*b]；后置事实。", object_name="Atlas", scope_ref="项目甲")
    ctx, fs, registry, uri, _, _, _ = eager_flow(original=old, category="entity")
    node = fs.read_node(uri, ctx)
    raw = edit(uri, search="[a.*b]", replacement="[精确值]")["input"]
    original_node, original_input = deepcopy(node), deepcopy(raw)
    candidate = parse_tool_call("edit_memory", raw, registry, read_nodes={uri: node}, ctx=ctx)[2]
    assert candidate.content == "前置事实。读数：[精确值]；后置事实。"
    assert node == original_node and raw == original_input


def test_overlapping_matches_are_ambiguous():
    ctx, _, registry, uri, _, extractor, _ = eager_flow(original=profile("identity", "abababa"))
    extractor._llm.operation = edit(uri, search="ababa", replacement="new")
    with pytest.raises(ToolCallProtocolError, match="exactly once"):
        extractor._structure_prepared_span(prepare(extractor, ctx), ctx)


@pytest.mark.parametrize("alter", [
    {"category": "event"}, {"owner_space": "user:someone-else"}, {"content": ""},
    {"metadata": {"version": 1, "routing_key": "identity", "schema_version": "99.0"}},
])
def test_only_compatible_complete_owned_snapshots_authorize_edit(alter):
    ctx, fs, registry, uri, _, _, _ = eager_flow()
    node = replace(fs.read_node(uri, ctx), **alter)
    with pytest.raises(ToolCallProtocolError):
        parse_tool_call("edit_memory", edit(uri)["input"], registry, read_nodes={uri: node}, ctx=ctx)


def test_disabled_or_append_only_schema_cannot_enable_incremental_tool():
    ctx, fs, registry, uri, _, _, _ = eager_flow()
    node = fs.read_node(uri, ctx)
    registry.register(replace(registry.get("profile"), operation_mode="add_only"))
    with pytest.raises(ToolCallProtocolError):
        parse_tool_call("edit_memory", edit(uri)["input"], registry, read_nodes={uri: node}, ctx=ctx)


def test_eager_snapshot_is_detached_even_for_fs_returning_mutable_nodes():
    ctx, fs, _, uri, _, extractor, _ = eager_flow()
    fs.read_node = lambda key, context: fs.nodes[key]
    prepared = prepare(extractor, ctx)
    fs.nodes[uri].content = "Name: Nora. Residence: Paris. Occupation: technician."
    candidate = extractor._structure_prepared_span(prepared, ctx)[0]
    assert "Utrecht" in candidate.content and "Paris" not in candidate.content


def test_incremental_and_full_snapshot_same_target_fail_closed():
    ctx, _, _, uri, llm, extractor, _ = eager_flow()
    complete = dict(edit(uri)["input"], routing_key="identity", content="Name: Nora.")
    complete.pop("search")
    complete.pop("replacement")
    llm.complete_with_tools = lambda **kw: [
        {"tool": "edit_memory", "input": edit(uri)["input"]},
        {"tool": "extract_profile", "input": complete},
    ]
    with pytest.raises(ToolCallProtocolError, match="cannot mix"):
        extractor._structure_prepared_span(prepare(extractor, ctx), ctx)


def test_lazy_safety_refetch_supplies_snapshot_for_subsequent_edit():
    ctx, fs, registry, uri, llm, _, _ = eager_flow()
    loops = 0
    def response(messages, tools, tool_choice):
        nonlocal loops
        loops += 1
        if loops == 1:
            return [{"name": "extract_profile", "input": dict(routing_key="identity", abstract="Engineer.",
                overview="Engineer.", content="Engineer.", confidence=.9, evidence_quote="I am an engineer.",
                attributed_speaker="user", attribution_basis="self_first_person")}], ""
        return [], json.dumps([edit(uri)])
    llm.complete_with_tools_messages = response
    result = ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PromptManager()).run("I now work as an engineer.", ctx)
    assert len(result.candidates) == 1
    assert "Utrecht" in result.candidates[0].content
    assert result.read_nodes[uri].metadata["version"] == 1


def test_missing_current_profile_evidence_does_not_inherit_old_quote():
    ctx, fs, registry, uri, _, _, _ = eager_flow()
    node = fs.read_node(uri, ctx)
    node.metadata.update(evidence_quote="I was a technician.", attributed_speaker="user", attribution_basis="self_first_person")
    raw = edit(uri)["input"]
    raw.pop("evidence_quote")
    with pytest.raises(ToolCallProtocolError, match="schema-valid"):
        parse_tool_call("edit_memory", raw, registry, read_nodes={uri: node}, ctx=ctx)


def test_lazy_native_then_final_json_keeps_all_incremental_changes():
    ctx, fs, registry, uri, llm, _, _ = eager_flow(original=memory("Atlas", content="Installed."), category="entity")
    def append(text):
        return {"name": "append_memory", "input": dict(target_uri=uri, expected_version=1,
            text=text, abstract="Installed, calibrated and checked.", overview="Installed, calibrated and checked.", confidence=.9)}
    queue = [([{ "name": "read", "input": {"uri": uri}}], ""), ([append("Calibrated.")], ""),
             ([], json.dumps([append("Checked.")]))]
    llm.complete_with_tools_messages = lambda **kwargs: queue.pop(0)
    candidate = ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PromptManager()).run("Calibrated and checked.", ctx).candidates[0]
    assert candidate.content == "Installed.\nCalibrated.\nChecked."


def test_boolean_stored_version_does_not_authorize_edit():
    ctx, fs, registry, uri, _, _, _ = eager_flow()
    node = fs.read_node(uri, ctx)
    node.metadata["version"] = True
    with pytest.raises(ToolCallProtocolError):
        parse_tool_call("edit_memory", edit(uri)["input"], registry, read_nodes={uri: node}, ctx=ctx)


def test_nested_leaf_outside_writer_directory_cannot_be_edited():
    ctx, fs, registry, uri, _, _, _ = eager_flow(original=memory("Atlas", content="technician"), category="entity")
    nested = uri.rsplit("/", 1)[0] + "/nested/Atlas"
    node = replace(fs.read_node(uri, ctx), uri=nested)
    with pytest.raises(ToolCallProtocolError):
        parse_tool_call("edit_memory", edit(nested)["input"], registry, read_nodes={nested: node}, ctx=ctx)


@pytest.mark.parametrize("changes", [{"target_uri": ["bad"]}, {"expected_version": [1]}, {"expected_version": True}])
def test_malformed_read_target_is_a_protocol_failure(changes):
    ctx, _, _, uri, llm, extractor, _ = eager_flow()
    llm.operation = edit(uri, **changes)
    with pytest.raises(ToolCallProtocolError):
        extractor._structure_prepared_span(prepare(extractor, ctx), ctx)


def test_parallel_attempts_do_not_share_working_incremental_content():
    from concurrent.futures import ThreadPoolExecutor

    ctx, _, _, uri, _, extractor, _ = eager_flow()
    prepared = prepare(extractor, ctx)
    calls = [edit(uri), edit(uri, search="Utrecht", replacement="Paris")]
    def parse(index):
        operation = calls[index % 2]
        return extractor._parse_candidate_calls(
            [{"tool": operation["name"], "input": operation["input"]}], prepared.span.prefetch_result, ctx,
        )[0].content
    with ThreadPoolExecutor(max_workers=2) as pool:
        contents = list(pool.map(parse, range(12)))
    assert contents[::2] == ["Name: Nora. Residence: Utrecht. Occupation: engineer."] * 6
    assert contents[1::2] == ["Name: Nora. Residence: Paris. Occupation: technician."] * 6


def test_lazy_native_edit_then_complete_snapshot_same_target_is_rejected():
    ctx, fs, registry, uri, llm, _, _ = eager_flow()
    full = dict(routing_key="identity", target_uri=uri, expected_version=1, abstract="Nora.",
                overview="Nora is an engineer.", content="Name: Nora. Occupation: engineer.", confidence=.9,
                evidence_quote="I now work as an engineer.", attributed_speaker="user", attribution_basis="self_first_person")
    queue = [([{ "name": "read", "input": {"uri": uri}}], ""), ([edit(uri)], ""),
             ([], json.dumps([{"name": "extract_profile", "input": full}]))]
    llm.complete_with_tools_messages = lambda **kwargs: queue.pop(0)
    with pytest.raises(ToolCallProtocolError, match="cannot mix"):
        ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PromptManager()).run("I now work as an engineer.", ctx)
    assert fs.read_node(uri, ctx).metadata["version"] == 1


def test_invalid_lazy_edit_cannot_return_partial_recorded_writes():
    ctx, fs, registry, uri, llm, _, _ = eager_flow()
    queue = [([{ "name": "read", "input": {"uri": uri}}], ""),
             ([edit(uri), edit(uri, search="not present")], ""), ([], "[]")]
    llm.complete_with_tools_messages = lambda **kwargs: queue.pop(0)
    with pytest.raises(ToolCallProtocolError, match="failed operations"):
        ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PromptManager()).run("I now work as an engineer.", ctx)
    assert fs.read_node(uri, ctx).metadata["version"] == 1


def test_profile_only_registry_offers_no_edits_if_profile_disabled():
    _, _, registry, _, _, _, _ = eager_flow()
    for schema in registry.list_enabled():
        registry.register(replace(schema, enabled=False))
    assert build_extraction_tools(registry) == []


def test_pattern_incremental_update_retains_skill_context_type():
    old = memory("weekly_review", category="pattern", content="Reviews notes every Friday.")
    ctx, fs, registry, uri, llm, _, _ = eager_flow(original=old, category="pattern")
    node = fs.read_node(uri, ctx)
    assert node.context_type == "SKILL"
    inputs = dict(target_uri=uri, expected_version=1, text="Keeps a review checklist.",
                  abstract="Weekly notes review with a checklist.", overview="Reviews notes every Friday and keeps a checklist.", confidence=.9)
    candidate = parse_tool_call("append_memory", inputs, registry, read_nodes={uri: node}, ctx=ctx)[2]
    assert candidate.category == "pattern" and candidate.content == "Reviews notes every Friday.\nKeeps a review checklist."
    api(fs, llm).write_memory(candidate, ctx)
    stored = fs.read_node(uri, ctx)
    assert stored.uri == uri and stored.context_type == "SKILL" and stored.metadata["version"] == 2


def composed_profile_operations(uri, *, confidence=.9, third_party=False):
    append = dict(target_uri=uri, expected_version=1, text="Possible residence: Lisbon.",
                  abstract="Nora's profile.", overview="Nora's updated profile.", confidence=confidence,
                  evidence_quote="I might move to Lisbon.", attributed_speaker="user",
                  attribution_basis="self_first_person")
    if third_party:
        append.update(text="Bob lives in Lisbon.", evidence_quote="Bob lives in Lisbon.",
                      attributed_speaker="Bob", attribution_basis="other_named")
    return [{"name": "append_memory", "input": append}, edit(uri)]


def parse_composed_profile(mode, ctx, fs, registry, uri, llm, extractor, operations):
    if mode == "lazy":
        queue = [([{"name": "read", "input": {"uri": uri}}], ""), (operations, ""), ([], "[]")]
        llm.complete_with_tools_messages = lambda **kwargs: queue.pop(0)
        return ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PromptManager()).run(
            "I now work as an engineer.", ctx).candidates
    calls = [{"tool": item["name"], "input": item["input"]} for item in operations]
    prepared = prepare(extractor, ctx)
    if mode == "json":
        llm.complete_json = lambda prompt, schema: {"operations": calls}
        return extractor._structure_span_json_fallback(prepared, ctx, None)
    llm.complete_with_tools = lambda **kwargs: calls
    return extractor._structure_prepared_span(prepared, ctx)


@pytest.mark.parametrize("mode", ["native", "json", "lazy"])
def test_composition_cannot_promote_third_party_fact_into_profile(mode):
    ctx, fs, registry, uri, llm, extractor, _ = eager_flow()
    with pytest.raises(ToolCallProtocolError):
        parse_composed_profile(mode, ctx, fs, registry, uri, llm, extractor,
                               composed_profile_operations(uri, third_party=True))
    assert fs.read_node(uri, ctx).metadata["version"] == 1
    assert "Bob" not in fs.read_node(uri, ctx).content


@pytest.mark.parametrize("mode", ["native", "json", "lazy"])
@pytest.mark.parametrize("confidence", [.1, .8])
def test_composed_confidence_preserves_every_operations_filter_threshold(mode, confidence):
    ctx, fs, registry, uri, llm, extractor, _ = eager_flow()
    candidates = parse_composed_profile(mode, ctx, fs, registry, uri, llm, extractor,
                                       composed_profile_operations(uri, confidence=confidence))
    assert len(candidates) == 1
    assert candidates[0].confidence == confidence
    assert candidates[0].content.endswith("Possible residence: Lisbon.")


@pytest.mark.parametrize("prior_native", [False, True])
def test_safety_refetch_discards_tentative_json_edits_only(prior_native):
    ctx, fs, registry, uri, llm, _, _ = eager_flow(
        original=memory("Atlas", object_name="Atlas", content="Installed."), category="entity")
    api(fs, llm).write_memory(memory("Bravo", object_name="Bravo", content="Checked."), ctx)
    def append(text):
        return {"name": "append_memory", "input": dict(target_uri=uri, expected_version=1,
            text=text, abstract="Installed and calibrated.", overview="Installed and calibrated.", confidence=.9)}
    unread = {"name": "extract_entity", "input": dict(routing_key="Bravo", object_name="Bravo", scope_ref="",
        abstract="Checked again.", overview="Checked again.", content="Checked again.", confidence=.9)}
    queue = [([{"name": "read", "input": {"uri": uri}}], "")]
    if prior_native:
        queue.append(([append("Calibration started.")], ""))
    queue += [( [], json.dumps([append("Calibration failed."), unread])),
              ( [], json.dumps([append("Calibration passed.")]))]
    llm.complete_with_tools_messages = lambda **kwargs: queue.pop(0)
    result = ExtractionReActLoop(llm, fs, registry, URIResolver(registry), PromptManager()).run(
        "Calibration passed.", ctx)
    assert any(item.safety_check_triggered for item in result.trace.iterations)
    assert len(result.candidates) == 1
    expected = "Installed.\n" + ("Calibration started.\n" if prior_native else "") + "Calibration passed."
    assert result.candidates[0].content == expected


@pytest.mark.parametrize("mode", ["native", "json", "lazy"])
@pytest.mark.parametrize("category", ["profile", "preference"])
@pytest.mark.parametrize("unsupported_position", [0, 1])
@pytest.mark.parametrize("stub", [False, True])
def test_composed_edits_require_each_steps_selected_span_support(mode, category, unsupported_position, stub):
    original = (profile("identity", "Occupation: technician.") if category == "profile"
                else memory("beverage", category="preference", content="Prefers tea.", who="user"))
    ctx, fs, registry, uri, llm, extractor, _ = eager_flow(original=original, category=category)
    supported = "I am an engineer." if category == "profile" else "I prefer coffee."
    unsupported = "I live in Lisbon." if category == "profile" else "I prefer deleting backups."
    operations = []
    for quote in (supported, unsupported) if unsupported_position else (unsupported, supported):
        inputs = dict(target_uri=uri, expected_version=1, text=quote,
                      abstract=quote, overview=quote, confidence=.9,
                      evidence_quote=quote, attributed_speaker="user", attribution_basis="self_first_person")
        operations.append({"name": "append_memory", "input": inputs})
    prepared = prepare(extractor, ctx, category)
    prepared.span.had_externalized_stub = stub
    prepared.span.safe_user_text = supported
    calls = [{"tool": item["name"], "input": item["input"]} for item in operations]
    if mode == "lazy":
        extractor._mode = "lazy"
        queue = [([{"name": "read", "input": {"uri": uri}}], ""), (operations, ""), ([], "[]")]
        llm.complete_with_tools_messages = lambda **kwargs: queue.pop(0)
        candidates = extractor._structure_prepared_span(prepared, ctx)
    elif mode == "json":
        llm.complete_json = lambda prompt, schema: {"operations": calls}
        candidates = extractor._structure_span_json_fallback(prepared, ctx, None)
    else:
        llm.complete_with_tools = lambda **kwargs: calls
        candidates = extractor._structure_prepared_span(prepared, ctx)
    if stub:
        assert candidates == []
    else:
        assert len(candidates) == 1 and all(item in candidates[0].content for item in (supported, unsupported))
    assert fs.read_node(uri, ctx).metadata["version"] == 1


@pytest.mark.parametrize("quote", [None, "I prefer coffee."])
def test_composed_source_checks_keep_each_candidates_topic_even_with_identical_quotes(quote):
    from extraction.incremental_edit import coalesce_incremental_edits

    supported = memory("beverage", category="preference",
                       content="Prefers coffee.", evidence_quote=quote,
                       target_uri="memory://beverage", expected_version=1)
    unsupported = replace(supported, abstract="Prefers deleting backups.",
                          overview="Prefers deleting backups.", content="Prefers deleting backups.")
    extractor = Extractor(None)
    candidates = coalesce_incremental_edits([(True, unsupported), (True, supported)])
    extractor._assign_provenance(candidates, _Span(0, 0, had_externalized_stub=True,
                                                 safe_user_text="I prefer coffee."), None)
    assert candidates == []


def test_source_chain_flattens_without_changing_serialized_candidate_or_single_step():
    from dataclasses import asdict

    from extraction.incremental_edit import coalesce_incremental_edits, incremental_source_steps

    first = memory("beverage", category="preference", content="Tea.",
                   target_uri="memory://beverage", expected_version=1)
    second = replace(first, content="Tea. Coffee.")
    third = replace(first, content="Tea. Coffee. Juice.")
    assert coalesce_incremental_edits([(True, first)])[0] is first
    composed = coalesce_incremental_edits([(True, first), (True, second)])[0]
    assert coalesce_incremental_edits([(True, composed)])[0] is composed
    composed = coalesce_incremental_edits([(True, composed), (True, third)])[0]
    assert incremental_source_steps(composed) == (first, second, third)
    assert asdict(composed) == asdict(third)


@pytest.mark.parametrize("different", [{"target_uri": "memory://other"}, {"expected_version": 2}])
def test_source_chains_do_not_merge_distinct_targets_or_versions(different):
    from extraction.incremental_edit import coalesce_incremental_edits

    unsupported = memory("beverage", category="preference",
                         content="Prefers deleting backups.", evidence_quote="I prefer deleting backups.",
                         target_uri="memory://beverage", expected_version=1)
    supported = replace(unsupported, abstract="Prefers coffee.", content="Prefers coffee.",
                        evidence_quote="I prefer coffee.", **different)
    candidates = coalesce_incremental_edits([(True, unsupported), (True, supported)])
    Extractor(None)._assign_provenance(candidates, _Span(0, 0, had_externalized_stub=True,
                                                       safe_user_text="I prefer coffee."), None)
    assert candidates == [supported]


@pytest.mark.parametrize("category,quote,retained", [
    ("profile", "   \t", False), ("preference", None, True), ("entity", None, True),
])
def test_source_filter_preserves_single_step_quote_and_category_rules(category, quote, retained):
    candidate = memory("coffee", category=category,
                       content="Prefers coffee.", evidence_quote=quote)
    candidates = [candidate]
    Extractor(None)._assign_provenance(candidates, _Span(0, 0, had_externalized_stub=True,
                                                       safe_user_text="I prefer coffee."), None)
    assert candidates == ([candidate] if retained else [])
