"""Event draft deduplication must not decide occurrence identity or drop facts."""

from copy import deepcopy
from dataclasses import replace

import pytest

from commit.candidate_pipeline import CandidatePipeline
from core.models import CandidateMemory, RequestContext
from extraction.tools import Extractor, _Span
from tests.unit.commit.test_memory_identity import DecisionLLM, MemoryFS, api


@pytest.fixture(params=[Extractor._merge_dual_run, CandidatePipeline().deduplicate])
def deduplicate(request):
    return request.param


def event(**changes):
    draft = CandidateMemory(
        "event", "user", "atlas_inspection", "Atlas inspected", "Atlas inspection report",
        "Nora inspected Atlas.", 0.8, when="2026-10-01", who="Nora", where="Lab A",
        scope_ref="project-a", title="Atlas inspection", provenance_ids=["archive:1"],
        evidence_quote="I inspected Atlas.", attributed_speaker="Nora",
        attribution_basis="other_named",
    )
    return replace(draft, **changes)


@pytest.mark.parametrize("changes", [
    {"when": "2026-10-02"}, {"who": "Ellis"}, {"where": "Lab B"},
    {"when": None}, {"title": "Atlas follow-up"},
    {"content": "Nora inspected Atlas. The gauge measured 5 bar."},
    {"overview": "Atlas inspection was conditional on access approval."},
    {"abstract": "Atlas inspection cancelled"},
    {"evidence_quote": "Nora said she inspected Atlas."},
    {"attributed_speaker": "Ellis"}, {"attribution_basis": "self_first_person"},
])
def test_distinct_occurrence_or_source_facts_survive(deduplicate, changes):
    drafts = [event(), event(**changes)]
    original = deepcopy(drafts)
    assert deduplicate(drafts) == drafts
    assert drafts == original


@pytest.mark.parametrize("scores", [(0.8, 0.9), (0.9, 0.8), (0.8, 0.8)])
def test_exact_duplicates_combine_sources_without_mutating_inputs(deduplicate, scores):
    drafts = [event(confidence=scores[0]),
              event(confidence=scores[1], provenance_ids=["archive:2", "archive:1"])]
    original = deepcopy(drafts)
    result, = deduplicate(drafts)
    assert result.confidence == max(scores)
    assert result.provenance_ids == ["archive:1", "archive:2"]
    assert drafts == original
    result.provenance_ids.append("later:1")
    assert drafts == original


def test_independent_span_aggregation_keeps_all_event_facts():
    extractor = object.__new__(Extractor)
    drafts = [event(), event(when="2026-10-02"),
              event(content="Nora inspected Atlas. The gauge measured 5 bar.")]
    spans = [(_Span(index, index), [draft]) for index, draft in enumerate(drafts)]
    assert extractor._merge_span_results(spans, None) == drafts


def test_deduplication_is_idempotent_and_scope_isolated(deduplicate):
    drafts = [event(), event(provenance_ids=["archive:2"]), event(scope_ref="project-b")]
    first = deduplicate(drafts)
    assert len(first) == 2
    assert deduplicate(first) == first
    assert first[0].provenance_ids == ["archive:1", "archive:2"]


def test_distinct_dates_reach_write_api_as_two_nodes():
    ctx = RequestContext("event-boundaries", "reader", "agent", "session", "trace")
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    drafts = [event(), event(when="2026-10-02")]
    candidates = Extractor._merge_dual_run(drafts * 2)
    first = [writer.write_memory(candidate, ctx) for candidate in candidates]
    assert len(fs.nodes) == 2
    assert first[0]["target_uri"] != first[1]["target_uri"]
    assert {node.metadata["when"] for node in fs.nodes.values()} == {"2026-10-01", "2026-10-02"}
