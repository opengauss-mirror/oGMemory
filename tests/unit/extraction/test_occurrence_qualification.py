"""Cross-session occurrence revisions use source evidence and a read snapshot."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Event, local

import pytest

from core.errors import ExtractionIncompleteError, ToolCallProtocolError
from core.models import RequestContext, SeedHit
from core.uri_resolver import URIResolver
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor, _NativeSpanOutcome, _Span
from tests.unit.commit.test_memory_identity import MemoryFS, api, memory
from tests.unit.extraction.test_prefetch import FakeEmbedder, FakeVectorIndex

SOURCE = "About the October 1 Atlas inspection: the gauge reading was 5 bar, not 4."
OLD = "Nora inspected Atlas at Lab A on October 1. Gauge: 4 bar. The valve was normal."
UPDATED = "Nora inspected Atlas at Lab A on October 1. Gauge: 5 bar. The valve was normal."


def snapshot(**changes):
    return {**dict(abstract="Atlas inspection: 5 bar, normal valve.", overview=UPDATED,
                   content=UPDATED, title="Corrected Atlas inspection", when="2026-10-01",
                   who="Nora", where="Lab A"), **changes}


def same(**changes):
    return {**dict(decision="same", target_index=0, identity_evidence=SOURCE,
                   snapshot=snapshot()), **changes}


class QualificationProvider:
    def __init__(self, operation, *, response=None, valid=True, fallback=False, read=True):
        self.operation = operation
        self.response = same() if response is None else response
        self.valid = valid
        self.fallback = fallback
        self.read = read
        self.calls = []
        self.uri = ""

    def complete_with_tools(self, **kwargs):
        if self.fallback:
            raise ToolCallProtocolError("native unavailable")
        return [{"tool": self.operation["name"], "input": self.operation["input"]}]

    def complete_with_tools_messages(self, messages, tools, tool_choice):
        if self.read and not any(message["role"] == "tool" for message in messages):
            return [{"name": "read", "input": {"uri": self.uri}}], ""
        return [], json.dumps([self.operation])

    def complete_json(self, prompt, schema):
        self.calls.append(prompt)
        if "operations" in schema.get("properties", {}):
            return {"operations": [{"tool": self.operation["name"],
                                     "input": self.operation["input"]}]}
        if "valid" in schema.get("properties", {}):
            return {"valid": self.valid}
        return deepcopy(self.response)


def flow(category="event", *, mode="eager", response=None, valid=True, fallback=False,
         read=True, source=SOURCE, scope="Project A"):
    ctx = RequestContext("qualification", "reader", "agent", "later-session", "trace")
    fs, registry = MemoryFS(), SchemaRegistry()
    original = replace(memory("inspection", category=category, content=OLD),
                       owner_scope="agent" if category == "case" else "user",
                       scope_ref="Project A", title="Atlas inspection", when="2026-10-01",
                       who="Nora", where="Lab A")
    uri = api(fs, None).write_memory(original, ctx)["target_uri"]
    fields = dict(routing_key="gauge_correction", abstract="Gauge correction.",
                  overview="The Atlas inspection gauge read 5 bar.",
                  content="The Atlas inspection gauge read 5 bar.", confidence=.9,
                  title="Inspection gauge correction", scope_ref=scope,
                  when="2026-10-01", who="Nora", where="Lab A")
    llm = QualificationProvider({"name": f"extract_{category}", "input": fields},
                                response=response, valid=valid, fallback=fallback, read=read)
    llm.uri = uri
    index, embedder = FakeVectorIndex(), FakeEmbedder()
    index.hits = [SeedHit(uri, category, "Atlas inspection", .95, 4)]
    extractor = Extractor(llm, fs=fs, schema_registry=registry, vector_index=index,
                          embedder=embedder, uri_resolver=URIResolver(registry), mode=mode)
    prepared = extractor._prepare_span(
        _Span(0, 0, categories=[category], message_ids=["new-message"]),
        [{"role": "user", "content": source}], "English", None, "", "", ctx,
    )
    return ctx, fs, llm, extractor, prepared, uri


@pytest.mark.parametrize("category", ["event", "case"])
@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_cross_session_correction_updates_one_complete_occurrence(category, mode):
    ctx, fs, llm, extractor, prepared, uri = flow(category, mode=mode)
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    assert (candidate.target_uri, candidate.expected_version) == (uri, 1)
    written = api(fs, llm).write_memory(candidate, ctx)
    stored = fs.read_node(uri, ctx)
    assert written["target_uri"] == uri and len(fs.nodes) == 1
    assert stored.content == UPDATED and stored.metadata["version"] == 2
    assert stored.metadata["title"] == "Corrected Atlas inspection"
    assert stored.metadata["provenance_ids"] == [
        "prov:1:archive:sample:message-one", "prov:1:archive:new-archive:new-message",
    ]
    assert len(llm.calls) == 2


def test_json_fallback_also_qualifies_the_occurrence():
    ctx, _, llm, extractor, prepared, uri = flow(fallback=True)
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    assert (candidate.target_uri, candidate.expected_version) == (uri, 1)
    assert candidate.content == UPDATED and len(llm.calls) == 3


def test_failed_identity_or_fact_verification_stops_the_revision():
    ctx, fs, _, extractor, prepared, uri = flow(valid=False)
    with pytest.raises(ExtractionIncompleteError):
        extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    assert fs.read_node(uri, ctx).content == OLD


@pytest.mark.parametrize("decision", ["new", "uncertain"])
def test_same_day_actor_and_title_do_not_force_same_occurrence(decision):
    response = dict(decision=decision, target_index=None, identity_evidence="", snapshot=None)
    ctx, fs, llm, extractor, prepared, uri = flow(response=response)
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    assert not candidate.target_uri and candidate.expected_version is None
    written = api(fs, llm).write_memory(candidate, ctx)
    assert written["target_uri"] != uri and len(fs.nodes) == 2


@pytest.mark.parametrize("changes", [{"target_index": 99}, {"identity_evidence": "invented"},
                                    {"identity_evidence": ""}, {"snapshot": None}])
def test_same_decision_requires_a_read_target_direct_user_evidence_and_complete_facts(changes):
    response = same()
    response.update(changes)
    ctx, _, _, extractor, prepared, _ = flow(response=response)
    with pytest.raises(ExtractionIncompleteError):
        extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")


def test_cross_scope_match_does_not_reach_qualification():
    ctx, _, llm, extractor, prepared, _ = flow(scope="Project B")
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    assert not candidate.target_uri
    assert not llm.calls


def test_lazy_hint_without_a_successful_read_does_not_authorize_a_revision():
    ctx, _, llm, extractor, prepared, _ = flow(mode="lazy", read=False)
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    assert not candidate.target_uri
    assert not llm.calls


def test_no_archive_source_retains_append_behavior():
    ctx, _, llm, extractor, prepared, _ = flow()
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, None)
    assert not candidate.target_uri
    assert not llm.calls


@pytest.mark.parametrize("change", ["owner", "category", "version", "partial", "provenance"])
def test_invalid_or_changed_read_snapshot_cannot_authorize_revision(change):
    ctx, fs, llm, extractor, prepared, uri = flow()
    node = fs.nodes[uri]
    if change == "owner":
        node.owner_space = "user:foreign"
    elif change == "category":
        node.category = "entity"
    elif change == "version":
        node.metadata["version"] = 2
    elif change == "partial":
        node.content = ""
    else:
        node.metadata["provenance_ids"] = []
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    assert not candidate.target_uri and not llm.calls


def test_runtime_custom_policy_guard_prevents_occurrence_binding():
    ctx, _, llm, extractor, prepared, _ = flow()
    seen = []
    extractor._occurrence_update_eligible = lambda candidate: seen.append(candidate.category) or False
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    assert seen == ["event"] and not candidate.target_uri and not llm.calls


def test_verifier_sees_all_similar_read_targets_before_accepting_unique_identity():
    ctx, fs, llm, extractor, prepared, uri = flow(valid=False)
    second = deepcopy(fs.nodes[uri])
    second.uri += "_second"
    second.content = "Nora inspected Atlas a second time on October 1. Gauge: 4 bar."
    fs.nodes[second.uri] = second
    prepared.span.prefetch_result.read_versions[second.uri] = 1
    with pytest.raises(ExtractionIncompleteError):
        extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    verification = json.loads(llm.calls[1].split("\n\n", 1)[1])
    assert {target["uri"] for target in verification["targets"]} == {uri, second.uri}


def test_assistant_context_cannot_supply_the_identity_quote():
    ctx, _, _, extractor, prepared, _ = flow(source="Thanks for checking.")
    prepared.source_context = f"assistant [do not extract from this]: {SOURCE}\nuser: Thanks for checking."
    with pytest.raises(ExtractionIncompleteError):
        extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")


def test_json_escaped_source_budget_keeps_complete_draft_as_addition():
    ctx, _, llm, extractor, prepared, _ = flow()
    prepared.source_context = '\\"' * 9_000
    prepared.span.safe_user_text = prepared.source_context
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    assert not candidate.target_uri and not llm.calls
    assert candidate.content == "The Atlas inspection gauge read 5 bar."


def test_oversized_generated_same_snapshot_cannot_bypass_the_verification_budget():
    ctx, _, llm, extractor, prepared, _ = flow(
        response=same(snapshot=snapshot(content='\\"' * 20_000)),
    )
    with pytest.raises(ExtractionIncompleteError, match="budget"):
        extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    assert len(llm.calls) == 1


def test_lazy_successful_dual_reads_must_agree_and_preserve_read_result_contract():
    ctx, _, llm, extractor, prepared, uri = flow(mode="lazy")
    attempts = 0
    original = llm.complete_with_tools_messages

    def first_reads_second_does_not(messages, tools, tool_choice):
        nonlocal attempts
        if not any(message["role"] == "tool" for message in messages):
            attempts += 1
            if attempts == 2:
                return [], json.dumps([llm.operation])
        return original(messages, tools, tool_choice)

    llm.complete_with_tools_messages = first_reads_second_does_not
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "new-archive")
    assert not candidate.target_uri and not llm.calls
    assert prepared.span.successful_read_versions == [{uri: 1}, {}]


def test_failed_lazy_attempt_does_not_clear_a_successful_read_authorization():
    ctx, _, llm, extractor, prepared, uri = flow(mode="lazy")
    # Exercise the real loop once, then finalize with the other native attempt failed.
    candidates = extractor._structure_prepared_span(prepared, ctx, "new-archive")
    outcome = _NativeSpanOutcome(prepared, candidates, [ToolCallProtocolError("second failed")])
    candidate, = extractor._finalize_span_outcome(outcome, ctx, "new-archive")
    assert (candidate.target_uri, candidate.expected_version) == (uri, 1)
    assert prepared.span.prefetch_result.read_versions == {uri: 1}
    assert prepared.span.prefetch_result.read_uris == {uri}


def qualified_span(prepared, candidate, *, start=0, evidence=SOURCE):
    span = replace(prepared.span, start=start, end=start, safe_user_text=evidence,
                   source_context=f"user: {evidence}")
    return span, [candidate]


def test_multiple_qualified_spans_preserve_context_corrections_and_all_sources():
    ctx, fs, llm, extractor, prepared, uri = flow()
    first = replace(memory("inspection", category="event", content=UPDATED), target_uri=uri,
                    expected_version=1, scope_ref="Project A", title="Corrected inspection",
                    when="2026-10-02", who="Nora", where="Lab A",
                    provenance_ids=["source:first"])
    second = replace(first, when="2026-10-01", where="Lab B",
                     provenance_ids=["source:second"])
    merged = snapshot(when="2026-10-02", where="Lab B")
    responses = iter([merged, {"valid": True}])
    llm.complete_json = lambda prompt, schema: llm.calls.append(prompt) or next(responses)
    candidate, = extractor._merge_span_results([
        qualified_span(prepared, first, evidence="Correction: that inspection took place on October 2."),
        qualified_span(prepared, second, start=1, evidence="That inspection was in Lab B."),
    ], ctx)
    assert (candidate.when, candidate.who, candidate.where) == ("2026-10-02", "Nora", "Lab B")
    assert candidate.provenance_ids == ["source:first", "source:second"]
    assert candidate.expected_version == 1 and len(llm.calls) == 2
    api(fs, llm).write_memory(candidate, ctx)
    node = fs.read_node(uri, ctx)
    assert node.metadata["when"] == "2026-10-02" and node.metadata["where"] == "Lab B"


def test_multi_span_revision_cannot_upgrade_a_stale_qualification():
    ctx, fs, llm, extractor, prepared, uri = flow()
    candidate = replace(memory("inspection", category="event", content=UPDATED),
                        target_uri=uri, expected_version=1, scope_ref="Project A")
    fs.nodes[uri].metadata["version"] = 2
    with pytest.raises(ExtractionIncompleteError, match="changed"):
        extractor._merge_span_results([qualified_span(prepared, candidate),
                                       qualified_span(prepared, candidate, start=1)], ctx)
    assert not llm.calls


def test_multi_span_complete_fact_verification_cannot_be_skipped():
    ctx, _, llm, extractor, prepared, uri = flow()
    candidate = replace(memory("inspection", category="event", content=UPDATED),
                        target_uri=uri, expected_version=1, scope_ref="Project A")
    responses = iter([snapshot(), {"valid": False}])
    llm.complete_json = lambda prompt, schema: llm.calls.append(prompt) or next(responses)
    with pytest.raises(ExtractionIncompleteError, match="verified"):
        extractor._merge_span_results([qualified_span(prepared, candidate),
                                       qualified_span(prepared, candidate, start=1)], ctx)
    assert len(llm.calls) == 2


@pytest.mark.parametrize("category", ["event", "case"])
def test_native_qualified_snapshot_explicitly_withdraws_unknown_location(category):
    source = "Correction to the October 1 Atlas inspection: Lab A was mistaken; the location is unknown."
    ctx, fs, llm, extractor, prepared, uri = flow(
        category, source=source,
        response=same(identity_evidence=source, snapshot=snapshot(where=None)),
    )
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "location-correction")
    assert candidate.complete_occurrence_context and candidate.where is None
    api(fs, llm).write_memory(candidate, ctx)
    stored = fs.read_node(uri, ctx)
    assert "where" not in stored.metadata and stored.metadata["version"] == 2


def test_single_span_native_case_aliases_reconcile_one_complete_target_before_writing():
    ctx, fs, llm, extractor, prepared, uri = flow("case")
    full = UPDATED + " The approved repair took two hours and the follow-up test passed."
    merged = snapshot(content=full, overview=full)

    def native(**kwargs):
        operation = deepcopy(llm.operation)
        operation["input"]["routing_key"] = (
            "repair_duration" if kwargs.get("temperature") == 0 else "followup_result")
        operation["input"]["content"] = (
            "The approved repair took two hours." if kwargs.get("temperature") == 0 else
            "The follow-up test passed.")
        return [{"tool": operation["name"], "input": operation["input"]}]

    def complete_json(prompt, schema):
        llm.calls.append(prompt)
        if "valid" in schema["properties"]:
            return {"valid": True}
        if "decision" in schema["properties"]:
            return same(snapshot=merged)
        return deepcopy(merged)

    prepared.span.safe_user_text = SOURCE + " The repair took two hours and the follow-up test passed."
    prepared.source_context = "user: " + prepared.span.safe_user_text
    prepared.span.source_context = prepared.source_context
    llm.complete_with_tools, llm.complete_json = native, complete_json
    candidates = extractor._structure_prepared_span_dual(prepared, ctx, "repair-followup")
    assert len(candidates) == 1
    candidate = candidates[0]
    assert (candidate.target_uri, candidate.expected_version) == (uri, 1)
    assert candidate.complete_occurrence_context and candidate.content == full
    plans = api(fs, llm)._writer.write_candidates(candidates, ctx)
    assert len(plans) == 1 and fs.read_node(uri, ctx).content == full
    assert fs.read_node(uri, ctx).metadata["version"] == 2 and len(llm.calls) == 6


def test_parallel_lazy_fallback_cannot_inherit_another_attempts_read_authorization():
    ctx, _, llm, extractor, prepared, uri = flow("case", mode="lazy")
    role = local()
    eager_entered, read_finished = Event(), Event()
    original_collect = extractor._collect_prefetch_context
    actual_fallback_reads = []

    def collect(categories, ctx, *, focused_messages=None, lazy=False, prompt_chars=None):
        result = original_collect(categories, ctx, focused_messages=focused_messages,
                                  lazy=lazy, prompt_chars=65_000 if not lazy else prompt_chars)
        if not lazy:
            actual_fallback_reads.append(dict(result.read_versions))
        return result

    def lazy(messages, tools, tool_choice):
        if role.name == "fallback":
            raise ToolCallProtocolError("native protocol unavailable")
        if not any(message["role"] == "tool" for message in messages):
            return [{"name": "read", "input": {"uri": uri}}], ""
        assert eager_entered.wait(10)
        return [], json.dumps([llm.operation])

    def eager(**kwargs):
        eager_entered.set()
        assert read_finished.wait(10)
        return [{"tool": llm.operation["name"], "input": llm.operation["input"]}]

    def attempt(name):
        role.name = name
        result = extractor._structure_prepared_span(prepared, ctx, "repair-followup")
        if name == "read":
            read_finished.set()
        return result

    extractor._collect_prefetch_context = collect
    llm.complete_with_tools_messages, llm.complete_with_tools = lazy, eager
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(attempt, "read")
        second = pool.submit(attempt, "fallback")
        first_run, second_run = first.result(timeout=15), second.result(timeout=15)
        candidates = first_run + second_run
    assert actual_fallback_reads == [{}]
    assert prepared.span.successful_read_versions == [{uri: 1}, {}]
    candidate, = extractor._finalize_span_outcome(
        _NativeSpanOutcome(prepared, candidates, native_runs={None: first_run, 0: second_run}),
        ctx, "repair-followup")
    assert not candidate.target_uri and not llm.calls


def test_existing_facts_in_native_event_revision_use_stored_evidence_before_source_only_reconciliation():
    ctx, fs, llm, extractor, prepared, uri = flow()

    def native(**kwargs):
        fields = deepcopy(llm.operation["input"])
        fields["routing_key"] = "inspection_revision" if kwargs.get("temperature") == 0 else "gauge_update"
        fields["content"] = (UPDATED if kwargs.get("temperature") == 0 else
                             "The normal valve inspection at Lab A by Nora recorded 5 bar, not 4.")
        return [{"tool": "extract_event", "input": fields}]

    def complete_json(prompt, schema):
        llm.calls.append(prompt)
        if "groups" in schema["properties"]:
            # These correct prior facts cannot be established by the new correction alone.
            raise ExtractionIncompleteError("Source-only reconciliation cannot verify retained prior facts")
        if "valid" in schema["properties"]:
            return {"valid": True}
        if "decision" in schema["properties"]:
            return same()
        return snapshot()

    llm.complete_with_tools, llm.complete_json = native, complete_json
    candidate, = extractor._structure_prepared_span_dual(prepared, ctx, "gauge-correction")
    assert candidate.target_uri == uri and candidate.content == UPDATED
    assert "The valve was normal" not in SOURCE
    assert candidate.complete_occurrence_context and len(llm.calls) == 6
    api(fs, llm).write_memory(candidate, ctx)
    assert fs.read_node(uri, ctx).content == UPDATED
