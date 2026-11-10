"""Completed source-qualified occurrence stages reuse verified model replies."""

from dataclasses import replace

import pytest

from extraction.occurrence_qualification import qualify_occurrence_drafts, reconcile_occurrence_updates
from tests.unit.commit.test_memory_identity import memory
from tests.unit.extraction.test_occurrence_qualification import flow, SOURCE, UPDATED, snapshot
from tests.unit.extraction.test_parallel_node_stages import Stages


def test_qualified_occurrence_reuses_validated_reply_for_same_read_snapshot():
    ctx, fs, llm, extractor, prepared, uri = flow()
    candidate = replace(memory("inspection", category="event", content=UPDATED),
                        scope_ref="Project A", title="Corrected inspection",
                        when="2026-10-01", who="Nora", where="Lab A",
                        provenance_ids=["source:first"])
    cache = Stages()
    def qualify():
        return qualify_occurrence_drafts([candidate], SOURCE, SOURCE, {uri: 1}, fs,
                                        extractor._uri_resolver, extractor._schema_registry, ctx, llm,
                                        checkpoint=cache)
    first = qualify()
    assert len(llm.calls) == 2
    second = qualify()
    assert second == first
    assert first[0].content == UPDATED and first[0].target_uri == uri
    assert len(llm.calls) == 2


def test_ordered_occurrence_merge_reuses_complete_validated_snapshot():
    ctx, fs, llm, extractor, prepared, uri = flow()
    first = replace(memory("inspection", category="event", content=UPDATED), target_uri=uri,
                    expected_version=1, scope_ref="Project A", title="Corrected inspection",
                    when="2026-10-02", who="Nora", where="Lab A", provenance_ids=["source:first"])
    second = replace(first, where="Lab B", provenance_ids=["source:second"])
    replies = iter([snapshot(when="2026-10-02", where="Lab B"), {"valid": True}])
    llm.complete_json = lambda prompt, schema: llm.calls.append(prompt) or next(replies)
    cache = Stages()
    args = ([(prepared.span, first), (prepared.span, second)], fs.read_node(uri, ctx),
            ctx, extractor._uri_resolver, llm)
    result = reconcile_occurrence_updates(*args, checkpoint=cache)
    assert len(llm.calls) == 2
    replay = reconcile_occurrence_updates(*args, checkpoint=cache)
    assert replay == result and len(llm.calls) == 2
    assert result.when == "2026-10-02" and result.where == "Lab B"
    assert result.provenance_ids == ["source:first", "source:second"]


def test_independent_source_candidate_reuses_completed_identity_decision():
    ctx, fs, llm, extractor, prepared, uri = flow(valid=False)
    candidate = replace(memory("inspection", category="event", content=UPDATED),
                        scope_ref="Project A", title="Corrected inspection",
                        when="2026-10-01", who="Nora", where="Lab A",
                        provenance_ids=["source:first"])
    cache = Stages()
    args = ([candidate], SOURCE, SOURCE, {uri: 1}, fs,
            extractor._uri_resolver, extractor._schema_registry, ctx, llm)
    first = qualify_occurrence_drafts(*args, checkpoint=cache)
    replay = qualify_occurrence_drafts(*args, checkpoint=cache)
    assert first == replay == [candidate]
    assert len(llm.calls) == 2


@pytest.mark.parametrize("decision", ["new", "uncertain"])
def test_added_occurrence_preserves_source_facts_and_reuses_identity_decision(decision):
    response = {"decision": decision, "target_index": None, "identity_evidence": "", "snapshot": None}
    ctx, fs, llm, extractor, prepared, uri = flow(response=response)
    source = "Nora performed another Atlas inspection on October 2. The gauge read 6 bar."
    candidate = replace(memory("october_2_inspection", category="event", content=source),
                        scope_ref="Project A", title="October 2 Atlas inspection",
                        when="2026-10-02", who="Nora", where="Lab A",
                        provenance_ids=["source:october-2"], source_contexts=[source])
    cache = Stages()
    args = ([candidate], source, source, {uri: 1}, fs,
            extractor._uri_resolver, extractor._schema_registry, ctx, llm)
    first = qualify_occurrence_drafts(*args, checkpoint=cache)
    replay = qualify_occurrence_drafts(*args, checkpoint=cache)
    assert first == replay == [candidate]
    assert first[0].content == source
    assert first[0].source_contexts == [source]
    assert first[0].provenance_ids == ["source:october-2"]
    assert cache.saved == ["occurrence_qualification"]
    assert len(llm.calls) == 1
