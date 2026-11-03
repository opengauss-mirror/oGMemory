"""Source-backed occurrence reconciliation runs before append-only event writes."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.errors import ExtractionIncompleteError
from core.models import RequestContext
from extraction.tools import Extractor, _PreparedSpan, _Span
from tests.unit.extraction.test_event_draft_boundaries import event


def proposal(groups):
    return {"groups": [{"members": members, "abstract": "Atlas inspected",
                        "overview": "Nora inspected Atlas. Gauge: 5 bar.",
                        "content": "Nora inspected Atlas. The gauge measured 5 bar.",
                        "title": "Atlas inspection", "when": "2026-10-01",
                        "who": "Nora", "where": "Lab A", "scope_ref": "project-a"}
                       for members in groups]}


def setup(drafts, responses):
    llm = Mock()
    llm.complete_json.side_effect = responses
    extractor = Extractor(llm)
    extractor._structure_prepared_span = Mock(side_effect=[drafts[:1], drafts[1:]])
    prepared = _PreparedSpan(_Span(0, 0, safe_user_text="Nora inspected Atlas once. Gauge: 5 bar."),
                             "source conversation", "a" * 64)
    return extractor, llm, prepared


def test_dual_run_rephrasing_keeps_one_complete_event_and_all_sources():
    drafts = [event(), event(content="Atlas was inspected by Nora. The gauge measured 5 bar.",
                              title="Nora's inspection", provenance_ids=["archive:2"])]
    original = deepcopy(drafts)
    extractor, llm, prepared = setup(drafts, [proposal([[0, 1]]), {"valid": True}])
    result = extractor._structure_prepared_span_dual(prepared, RequestContext("a", "u", "g", "s", "t"), None)
    assert len(result) == 1
    assert "5 bar" in result[0].content
    assert result[0].provenance_ids == ["archive:1", "archive:2"]
    assert drafts == original
    assert "Nora inspected Atlas once" in llm.complete_json.call_args_list[0].args[0]


@pytest.mark.parametrize("changes", [{"owner_scope": "agent"},
    {"attributed_speaker": "Ellis"}, {"target_uri": "memory://events/existing"}])
def test_incompatible_occurrences_never_reach_semantic_merge(changes):
    drafts = [event(), event(**changes)]
    extractor, llm, prepared = setup(drafts, [])
    assert extractor._structure_prepared_span_dual(prepared, None, None) == drafts
    llm.complete_json.assert_not_called()


@pytest.mark.parametrize("response", [proposal([[0]]), proposal([[0, 0, 1]]),
    proposal([[0, 1], [1]]), proposal([[0, 2]]), {"groups": []}])
def test_invalid_partition_stops_extraction_without_mutating_drafts(response):
    drafts = [event(), event(content="Gauge: 5 bar.")]
    extractor, llm, prepared = setup(drafts, [response])
    original = deepcopy(drafts)
    with pytest.raises(ExtractionIncompleteError):
        extractor._structure_prepared_span_dual(prepared, None, None)
    assert drafts == original
    assert llm.complete_json.call_count == 1


def test_failed_fact_or_occurrence_validation_stops_extraction():
    drafts = [event(), event(content="A second inspection happened later that day.")]
    extractor, _, prepared = setup(drafts, [proposal([[0, 1]]), {"valid": False}, {"repair": None}])
    with pytest.raises(ExtractionIncompleteError):
        extractor._structure_prepared_span_dual(prepared, None, None)


@pytest.mark.parametrize("layer", ["abstract", "overview", "content", "title"])
def test_oversized_generated_layers_stop_extraction(layer):
    from extraction.event_reconciliation import _LIMITS

    response = proposal([[0, 1]])
    response["groups"][0][layer] = "x" * (_LIMITS[layer] + 1)
    responses = [response, {'repairs': []}] if layer in ('abstract', 'overview') else [response]
    extractor, llm, prepared = setup([event(), event(content="Gauge: 5 bar.")], responses)
    with pytest.raises(ExtractionIncompleteError):
        extractor._structure_prepared_span_dual(prepared, None, None)
    assert llm.complete_json.call_count == (2 if layer in ('abstract', 'overview') else 1)


def test_source_evidence_is_required_for_potential_duplicate_drafts():
    extractor, llm, prepared = setup([event(), event(content="Gauge: 5 bar.")], [])
    prepared.span.safe_user_text = ""
    with pytest.raises(ExtractionIncompleteError, match="source evidence"):
        extractor._structure_prepared_span_dual(prepared, None, None)
    llm.complete_json.assert_not_called()


def test_low_confidence_draft_cannot_suppress_an_accepted_event():
    drafts = [event(), event(confidence=0.2, content="Unsupported event wording")]
    extractor, llm, prepared = setup(drafts, [])
    assert extractor._structure_prepared_span_dual(prepared, None, None) == drafts
    llm.complete_json.assert_not_called()


def test_distinct_same_day_occurrences_are_preserved():
    drafts = [event(), event(content="Nora inspected Atlas a second time in the afternoon.")]
    extractor, llm, prepared = setup(drafts, [proposal([[0], [1]])])
    assert extractor._structure_prepared_span_dual(prepared, None, None) == drafts
    assert llm.complete_json.call_count == 1


@pytest.mark.parametrize("changes", [{"when": "2026-10-02"}, {"who": "Ellis"},
                                    {"where": "Lab B"}])
def test_source_distinguishes_real_occurrence_context_changes(changes):
    drafts = [event(), event(**changes)]
    extractor, _, prepared = setup(drafts, [proposal([[0], [1]])])
    assert extractor._structure_prepared_span_dual(prepared, None, None) == drafts


def test_equivalent_date_actor_and_place_labels_can_merge_from_source():
    drafts = [event(), event(when="October 1, 2026", who="Inspector Nora", where="Laboratory A")]
    extractor, _, prepared = setup(drafts, [proposal([[0, 1]]), {"valid": True}])
    result, = extractor._structure_prepared_span_dual(prepared, None, None)
    assert (result.when, result.who, result.where) == ("2026-10-01", "Nora", "Lab A")


def test_source_distinct_scopes_remain_separate_events():
    drafts = [event(), event(scope_ref="project-b")]
    extractor, llm, prepared = setup(drafts, [proposal([[0], [1]])])
    prepared.span.safe_user_text = (
        "Nora inspected Atlas once for project-a, then separately inspected it for project-b.")
    assert extractor._structure_prepared_span_dual(prepared, None, None) == drafts
    assert llm.complete_json.call_count == 1


def test_source_grounded_scope_label_drift_merges_one_occurrence():
    drafts = [event(scope_ref="Lab A"), event(scope_ref="Atlas")]
    response = proposal([[0, 1]])
    response["groups"][0]["scope_ref"] = "Atlas"
    extractor, _, prepared = setup(drafts, [response, {"valid": True}])
    prepared.span.safe_user_text = "Nora inspected Atlas once at Lab A. This is one inspection."
    result, = extractor._structure_prepared_span_dual(prepared, None, None)
    assert result.scope_ref == "Atlas"


def test_scope_crossing_without_source_support_stops_extraction():
    drafts = [event(), event(scope_ref="project-b")]
    extractor, _, prepared = setup(drafts, [proposal([[0, 1]]), {"valid": False}, {"repair": None}])
    prepared.span.safe_user_text = "Nora inspected Atlas separately for project-a and project-b."
    with pytest.raises(ExtractionIncompleteError):
        extractor._structure_prepared_span_dual(prepared, None, None)


def test_parallel_spans_reconcile_each_span_without_combining_independent_events():
    drafts = [event(), event(content="Atlas was inspected by Nora. Gauge: 5 bar.")]
    extractor, llm, first = setup(drafts, [])
    second = replace(first, span=replace(first.span, start=1, end=1))
    extractor._prepare_unique_spans = Mock(return_value=[first, second])
    extractor._structure_prepared_span.side_effect = lambda prepared, *_args, **_kwargs: (
        drafts if prepared.span.start == 0 else
        [replace(draft, when="2026-10-02") for draft in drafts])
    second_proposal = proposal([[0, 1]])
    second_proposal["groups"][0]["when"] = "2026-10-02"
    llm.complete_json.side_effect = [proposal([[0, 1]]), {"valid": True},
                                      second_proposal, {"valid": True}]
    result = extractor._structure_spans([], [], "en", None, "", "", None)
    assert len(result) == 2
    assert {draft.when for draft in result} == {"2026-10-01", "2026-10-02"}
    assert llm.complete_json.call_count == 4


@pytest.mark.parametrize("lazy", [False, True])
def test_reconciliation_retains_focused_user_evidence_without_enriched_prompt(lazy):
    import json
    from datetime import datetime

    messages = [
        {"role": "user", "content": "I'm Nora. Yesterday I inspected Atlas in Lab A."},
        {"role": "assistant", "content": "Was the gauge reading from that inspection?"},
        {"role": "user", "content": "Yes, it read 5 bar."},
        {"role": "assistant", "content": "Perhaps the safety valve also failed."},
        {"role": "tool", "content": "unrelated tool observation"},
    ]
    drafts = [event(content="Nora inspected Atlas. The gauge measured 5 bar."),
              event(content="Atlas inspection by Nora recorded 5 bar.",
                    title="Inspection reading", provenance_ids=["archive:2"])]
    extractor, llm, _ = setup(drafts, [])
    extractor._use_lazy_mode = Mock(return_value=lazy)
    extractor._with_prefetch_context = Mock(side_effect=lambda _span, prompt, _ctx:
                                            prompt + "\nPREFETCH_ONLY_OLD_MEMORY")
    prepared = extractor._prepare_span(
        _Span(2, 2, categories=["event"]), messages, "en", datetime(2026, 10, 2, 14, 30),
        "SUMMARY_ONLY_OLD_FACT", "INTERNAL_TOOL_STATS", None,
    )
    sources = []

    def reconcile(prompt, **_kwargs):
        payload = json.loads(prompt.split("\n", 1)[1])
        source = payload["source"]
        sources.append(source)
        grounded = all(text in source for text in (
            "user: I'm Nora. Yesterday I inspected Atlas in Lab A.",
            "user: Yes, it read 5 bar.", "2026-10-02 14:30",
            "assistant [do not extract from this]: Was the gauge reading",
        ))
        if "proposed_result" in payload:
            return {"valid": grounded}
        return proposal([[0, 1]])

    llm.complete_json.side_effect = reconcile
    result, = extractor._structure_prepared_span_dual(prepared, None, None)
    assert "5 bar" in result.content
    assert result.provenance_ids == ["archive:1", "archive:2"]
    assert len(sources) == 2 and sources[0] == sources[1]
    assert "assistant [do not extract from this]: Perhaps the safety valve" in sources[0]
    assert "tool: unrelated tool observation" in sources[0]
    assert not any(text in sources[0] for text in (
        "PREFETCH_ONLY_OLD_MEMORY", "SUMMARY_ONLY_OLD_FACT", "INTERNAL_TOOL_STATS",
        "You are a memory extraction assistant",
    ))
    assert "user: Yes, it read 5 bar." in prepared.prompt


def test_reconciliation_source_stays_within_each_spans_extraction_window():
    import json
    from datetime import datetime

    drafts = [event(), event(content="Atlas was inspected by Nora. Gauge: 5 bar.")]
    extractor, llm, _ = setup(drafts, [])
    extractor._use_lazy_mode = Mock(return_value=True)
    messages = [{"role": "user", "content": f"Independent source message {i}"}
                for i in range(12)]
    prepared = [extractor._prepare_span(_Span(i, i, categories=["event"]), messages,
                                       "en", datetime(2026, 10, 2), "", "", None)
                for i in (0, 11)]
    extractor._structure_prepared_span.side_effect = lambda *_args, **_kwargs: drafts
    llm.complete_json.side_effect = [proposal([[0], [1]]), proposal([[0], [1]])]
    for span in prepared:
        assert extractor._structure_prepared_span_dual(span, None, None) == drafts
    first, second = [json.loads(call.args[0].split("\n", 1)[1])["source"]
                     for call in llm.complete_json.call_args_list]
    assert "Independent source message 0" in first
    assert "Independent source message 11" not in first
    assert "Independent source message 11" in second
    assert "Independent source message 0" not in second


def test_default_session_time_is_shared_even_across_a_date_boundary(monkeypatch):
    from datetime import datetime

    clock = Mock()
    clock.now.side_effect = [datetime(2026, 10, 2, 23, 59), datetime(2026, 10, 3, 0, 0)]
    monkeypatch.setattr("extraction.tools.datetime", clock)
    extractor = Extractor(Mock())
    extractor._use_lazy_mode = Mock(return_value=True)
    prepared = extractor._prepare_span(
        _Span(0, 0, categories=["event"]),
        [{"role": "user", "content": "Yesterday I inspected Atlas."}],
        "en", None, "", "", None,
    )
    clock.now.assert_called_once_with()
    assert "2026-10-02 23:59" in prepared.prompt
    assert "2026-10-02 23:59" in prepared.source_context
