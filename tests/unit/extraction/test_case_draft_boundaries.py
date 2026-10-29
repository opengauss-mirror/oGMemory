"""Case output normalization must preserve facts and independent occurrences."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from commit.candidate_pipeline import CandidatePipeline
from commit.write_receipts import RECEIPTS_KEY
from core.candidate_dedup import case_draft_signature
from core.errors import ToolCallProtocolError
from core.memory_names import candidate_identity_key
from core.models import CandidateMemory, RequestContext
from extraction.tools import Extractor, _NativeSpanOutcome, _PreparedSpan, _Span
from service.extraction_checkpoint import ExtractionBatchJournal
from session.session_manager import SessionManager
from tests.unit.commit.test_memory_identity import MemoryFS, api


def case(**changes):
    draft = CandidateMemory(
        "case", "agent", "receiver_repair", "Receiver repaired", "Cable replacement worked.",
        "Step A: replaced the cable.", .8, when="2026-10-01", who="Nora", where="Lab A",
        scope_ref="project-a", title="Receiver repair", provenance_ids=["archive:1"],
        evidence_quote="I replaced the cable.", attributed_speaker="Nora",
        attribution_basis="other_named",
    )
    return replace(draft, **changes)


@pytest.fixture(params=[Extractor._merge_dual_run, CandidatePipeline().deduplicate],
                ids=["extraction_flat_merge", "pipeline_dedup"])
def deduplicate(request):
    return request.param


@pytest.mark.parametrize("changes", [
    {"content": "Step B: rebooted the device."},
    {"content": "Step B: rebooted the device.", "confidence": .9},
    {"overview": "Rebooting also worked."}, {"title": "Receiver follow-up"},
    {"when": "2026-10-02"}, {"who": "Ellis"}, {"where": "Lab B"},
    {"when": None}, {"scope_ref": "project-b"}, {"owner_scope": "user"},
    {"evidence_quote": "I rebooted the device."},
    {"attributed_speaker": "Ellis"}, {"attribution_basis": "self_first_person"},
])
def test_distinct_case_facts_survive_each_dedup_layer(deduplicate, changes):
    drafts = [case(), case(**changes)]
    original = deepcopy(drafts)
    assert deduplicate(drafts) == drafts
    assert drafts == original


def test_matching_abstract_prefix_does_not_drop_full_case_facts(deduplicate):
    prefix = "Receiver repair details: " + "x" * 60
    drafts = [case(abstract=prefix + "cable", content="The cable replacement worked."),
              case(abstract=prefix + "reboot", content="Rebooting the receiver worked.")]
    assert drafts[0].abstract[:80] == drafts[1].abstract[:80]
    original = deepcopy(drafts)
    assert deduplicate(drafts) == drafts
    assert drafts == original


def test_long_body_and_high_confidence_never_form_a_mixed_case(deduplicate):
    drafts = [case(content="Step A: replaced the damaged cable and tested every connector.",
                   overview="Cable replacement worked.", provenance_ids=["archive:1"]),
              case(content="Step B: rebooted it.", confidence=.95,
                   overview="Rebooting worked.", provenance_ids=["archive:2"])]
    original = deepcopy(drafts)
    result = deduplicate(drafts)
    assert drafts == original
    assert result == original


@pytest.mark.parametrize("sources", [([], []), (["archive:1"], ["archive:1"]),
                                    (["archive:1"], ["archive:2"])])
def test_exact_case_without_run_origin_stays_separate(deduplicate, sources):
    drafts = [case(provenance_ids=source) for source in sources]
    original = deepcopy(drafts)
    assert deduplicate(drafts) == drafts
    assert drafts == original


def same_span_results(first, second):
    llm = Mock()
    llm.complete_json.side_effect = AssertionError("Case normalization must not call a model")
    extractor = Extractor(llm)
    extractor._structure_prepared_span = Mock(side_effect=[first, second])
    prepared = _PreparedSpan(
        _Span(0, 0, safe_user_text="I replaced the cable. I rebooted the device."),
        "one source conversation", "a" * 64,
    )
    result = extractor._structure_prepared_span_dual(prepared, None, "case-source")
    llm.complete_json.assert_not_called()
    return result


def test_same_span_exact_native_outputs_normalize_once_with_source_union():
    first = case(confidence=.8)
    second = case(confidence=.9, provenance_ids=["archive:2", "archive:1"])
    original = deepcopy([first, second])
    result, = same_span_results([first], [second])
    assert result == replace(second, provenance_ids=["archive:1", "archive:2"])
    assert [first, second] == original
    result.provenance_ids.append("later:1")
    assert [first, second] == original


@pytest.mark.parametrize("counts", [(2, 1), (1, 2), (2, 2), (0, 2), (2, 0)])
def test_same_span_exact_case_normalization_preserves_multiplicity(counts):
    first = [case(provenance_ids=["archive:1"]) for _ in range(counts[0])]
    second = [case(provenance_ids=["archive:2"]) for _ in range(counts[1])]
    original = deepcopy([first, second])
    result = same_span_results(first, second)
    assert len(result) == max(counts)
    assert [first, second] == original
    assert [item.provenance_ids for item in result[:min(counts)]] == [
        ["archive:1", "archive:2"] for _ in range(min(counts))]


def test_same_span_different_full_case_facts_both_survive():
    drafts = [case(), case(content="Step B: rebooted the device.", confidence=.95)]
    original = deepcopy(drafts)
    assert same_span_results(drafts[:1], drafts[1:]) == drafts
    assert drafts == original


def test_independent_span_aggregation_preserves_even_exact_cases():
    extractor = object.__new__(Extractor)
    drafts = [case(), case(provenance_ids=["archive:2"])]
    spans = [(_Span(index, index), [draft]) for index, draft in enumerate(drafts)]
    assert extractor._merge_span_results(spans, None) == drafts


def test_bound_case_update_still_chooses_one_complete_snapshot(deduplicate):
    first = case(target_uri="ctx://cases/agents/agent/memories/cases/repair", expected_version=1)
    second = replace(first, confidence=.95, content="All completed repair steps.",
                     overview="Complete repair outcome.")
    original = deepcopy([first, second])
    assert deduplicate([first, second]) == [second]
    assert [first, second] == original


@pytest.mark.parametrize("changes", [
    {"abstract": "Receiver fixed"}, {"overview": "A different result."},
    {"content": "Step B: rebooted the device."}, {"title": "Receiver follow-up"},
    {"when": "2026-10-02"}, {"who": "Ellis"}, {"where": "Lab B"},
    {"scope_ref": "project-b"}, {"owner_scope": "user"},
    {"evidence_quote": "I rebooted the device."}, {"attributed_speaker": "Ellis"},
    {"attribution_basis": "self_first_person"}, {"tool_stats": {"calls": 1}},
    {"object_name": "Receiver"}, {"complete_occurrence_context": True},
])
def test_same_span_pairing_requires_equal_complete_drafts(changes):
    first, second = case(), case(**changes)
    assert case_draft_signature(first) != case_draft_signature(second)
    assert candidate_identity_key(first) != candidate_identity_key(second)
    assert same_span_results([first], [second]) == [first, second]


def test_stable_case_signature_ignores_write_operation_id():
    # The extraction signature also excludes this internal field once assigned
    # by the durable checkpoint, without depending on model-visible fields.
    accepted = case(write_operation_id="operation-a")
    second = replace(accepted, write_operation_id="operation-b", confidence=.99,
                     provenance_ids=["archive:2"])
    assert case_draft_signature(accepted) == case_draft_signature(second)


def test_flat_finalization_without_recorded_runs_preserves_exact_cases():
    llm = Mock()
    llm.complete_json.side_effect = AssertionError("Unknown origin must not call a model")
    extractor = Extractor(llm)
    prepared = _PreparedSpan(_Span(0, 0), "source", "a" * 64)
    drafts = [case(), case()]
    assert extractor._finalize_span_outcome(
        _NativeSpanOutcome(prepared, drafts), None, "source") == drafts
    llm.complete_json.assert_not_called()


@pytest.mark.parametrize("failed_run", [0, 1])
def test_single_native_success_keeps_all_exact_cases(failed_run):
    llm = Mock()
    llm.complete_json.side_effect = AssertionError("One native success must skip fallback")
    extractor = Extractor(llm)
    drafts = [case(), case()]
    outputs = [drafts, drafts]
    outputs[failed_run] = ToolCallProtocolError("native run failed")
    extractor._structure_prepared_span = Mock(side_effect=outputs)
    prepared = _PreparedSpan(_Span(0, 0), "source", "a" * 64)
    assert extractor._structure_prepared_span_dual(prepared, None, "source") == drafts
    llm.complete_json.assert_not_called()


def test_json_fallback_keeps_multiple_exact_cases_without_native_run_origin():
    llm = Mock()
    extractor = Extractor(llm)
    extractor._structure_prepared_span = Mock(side_effect=ToolCallProtocolError("failed"))
    drafts = [case(), case()]
    extractor._structure_span_json_fallback = Mock(return_value=drafts)
    prepared = _PreparedSpan(_Span(0, 0), "source", "a" * 64)
    assert extractor._structure_prepared_span_dual(prepared, None, "source") == drafts
    extractor._structure_span_json_fallback.assert_called_once()


def test_parallel_spans_normalize_each_run_pair_and_preserve_independent_cases():
    llm = Mock()
    llm.complete_json.side_effect = AssertionError("Case normalization must not call a model")
    extractor = Extractor(llm)
    prepared = [_PreparedSpan(_Span(index, index), f"source-{index}", str(index) * 64)
                for index in (0, 1)]
    extractor._prepare_unique_spans = Mock(return_value=prepared)
    drafts = [case(provenance_ids=[f"archive:{index}"]) for index in (0, 1)]
    extractor._structure_prepared_span = Mock(
        side_effect=lambda span, *args, **kwargs: [deepcopy(drafts[span.span.start])])
    result = extractor._structure_spans(
        [span.span for span in prepared], [], "English", None, "", "", None, "source")
    assert result == drafts
    assert extractor._structure_prepared_span.call_count == 4
    llm.complete_json.assert_not_called()


class CaseSourceLLM:
    """Offline provider with native outputs grounded in supplied user messages."""

    def __init__(self, runs):
        self.runs = runs
        self.native_calls = []
        self.json_calls = []

    def complete_json(self, prompt, schema):
        self.json_calls.append(prompt)
        assert set(schema["properties"]) == {"spans"}
        return {"spans": [{"start": 0, "end": 0, "categories": ["case"]}]}

    def complete_with_tools(self, *, prompt, tools, tool_choice, temperature=None):
        self.native_calls.append(temperature)
        assert "I replaced the cable." in prompt
        assert "I rebooted the device." in prompt
        return [{"tool": "extract_case", "input": {
            "routing_key": "receiver_repair", "abstract": "Receiver repaired",
            "overview": body, "content": body, "confidence": .9,
            "title": "Receiver repair", "scope_ref": "project-a",
            "when": "2026-10-01", "who": "reader", "where": "Lab A",
        }} for body in self.runs[0 if temperature is None else 1]]


@pytest.mark.parametrize("runs,expected", [
    ((["I replaced the cable."], ["I rebooted the device."]),
     ["I replaced the cable.", "I rebooted the device."]),
    ((["I replaced the cable."], ["I replaced the cable."]), ["I replaced the cable."]),
])
def test_source_controlled_extraction_pipeline_write_api_preserves_case_facts(runs, expected):
    llm = CaseSourceLLM(runs)
    fs = MemoryFS()
    writer = api(fs, llm)
    ctx = RequestContext("case-boundaries", "reader", "agent", "session", "trace")
    messages = [{"id": "message-one", "role": "user", "content":
                 "On 2026-10-01 at Lab A, I repaired the receiver in project-a. "
                 "I replaced the cable. I rebooted the device."}]
    original = deepcopy(messages)
    candidates = writer._pipeline.extract(
        messages, ctx, session_time=datetime(2026, 10, 1, tzinfo=UTC), archive_id="case-source")
    accepted = writer._pipeline.deduplicate(writer._pipeline.filter_by_confidence(candidates))
    assert [item.content for item in accepted] == expected
    assert [item.overview for item in accepted] == expected
    assert all(item.provenance_ids == ["prov:1:archive:case-source:message-one"] for item in accepted)
    plans = [writer.write_memory(item, ctx) for item in accepted]
    assert all(plan["action"] == "create" for plan in plans)
    assert len({plan["target_uri"] for plan in plans}) == len(expected)
    assert {node.content for node in fs.nodes.values()} == set(expected)
    for node in fs.nodes.values():
        assert node.overview == node.content
        assert node.metadata["provenance_ids"] == ["prov:1:archive:case-source:message-one"]
        assert node.metadata["scope_ref"] == "project-a"
    assert llm.native_calls == [None, 0]
    assert len(llm.json_calls) == 1
    assert messages == original


@pytest.mark.parametrize("initial_archive,resumed_archive", [
    ("case-source", "case-source"), ("case-source", None), (None, None),
])
def test_checkpoint_accepts_distinct_case_facts_and_resume_reuses_final_outputs(initial_archive, resumed_archive):
    fs = MemoryFS()
    llm = CaseSourceLLM((["I replaced the cable."], ["I rebooted the device."]))
    writer = api(fs, llm)
    ctx = RequestContext("case-checkpoint", "reader", "agent", "session", "trace")
    session_time = datetime(2026, 10, 1, tzinfo=UTC)
    messages = [{"id": "message-one", "role": "user", "content":
                 "On 2026-10-01 at Lab A, I repaired the receiver in project-a. "
                 "I replaced the cable. I rebooted the device."}]
    manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    manager.add_message(ctx.session_id, "user", messages[0]["content"], ctx,
                        created_at=session_time.isoformat(), message_id="message-one")
    buffer = manager.get_or_create(ctx.session_id, ctx)
    journal = ExtractionBatchJournal(
        manager, buffer, ctx, buffer.messages, archive_id="case-source", messages=messages,
        session_time=session_time, session_summary="", tool_stats_text="")
    checkpoint = journal.checkpoint("structured")
    result = writer.commit_session(
        messages, ctx, session_time=session_time, archive_id=initial_archive,
        candidate_checkpoint=checkpoint)
    accepted, extracted, filtered = checkpoint.load()
    assert (extracted, filtered, result["writes_completed"]) == (2, 0, 2)
    assert [item.content for item in accepted] == ["I replaced the cable.", "I rebooted the device."]
    assert len({item.write_operation_id for item in accepted}) == 2
    assert all(item.write_operation_id for item in accepted)
    assert all(item.provenance_ids == ["prov:1:archive:case-source:message-one"] for item in accepted)
    assert case_draft_signature(accepted[0]) == case_draft_signature(
        replace(accepted[0], write_operation_id="another-operation"))
    case_nodes = [node for node in fs.nodes.values() if node.category == "case"]
    assert len(case_nodes) == 2
    assert {node.content for node in case_nodes} == {item.content for item in accepted}
    assert all(node.overview == node.content and node.metadata["scope_ref"] == "project-a"
               and node.metadata["provenance_ids"] == ["prov:1:archive:case-source:message-one"]
               for node in case_nodes)
    assert {operation for node in case_nodes for operation in node.metadata[RECEIPTS_KEY]} == {
        item.write_operation_id for item in accepted}

    class NoModel:
        def __getattr__(self, name):
            raise AssertionError(f"Accepted case outputs must not call a model: {name}")

    stored = deepcopy(fs.nodes)
    restarted = api(fs, NoModel())
    restored_manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    restored_buffer = restored_manager.get_or_create(ctx.session_id, ctx)
    restored_journal = ExtractionBatchJournal(
        restored_manager, restored_buffer, ctx, restored_buffer.messages,
        archive_id="case-source", messages=messages, session_time=session_time,
        session_summary="", tool_stats_text="")
    restored_checkpoint = restored_journal.checkpoint("structured")
    assert restored_checkpoint.load() == (accepted, extracted, filtered)
    resumed = restarted.commit_session(
        messages, ctx, session_time=session_time, archive_id=resumed_archive,
        candidate_checkpoint=restored_checkpoint)
    assert result["archive_id"] == resumed["archive_id"] == "case-source"
    assert {key: value for key, value in resumed.items() if key != "plans"} == {
        key: value for key, value in result.items() if key != "plans"}
    assert [(plan["action"], plan["target_uri"]) for plan in resumed["plans"]] == [
        (plan["action"], plan["target_uri"]) for plan in result["plans"]]
    assert restored_checkpoint.load() == (accepted, extracted, filtered)
    assert fs.nodes == stored
    assert llm.native_calls == [None, 0] and len(llm.json_calls) == 1
