from dataclasses import replace
from unittest.mock import Mock

import pytest

from core.models import CandidateMemory, ContextNode, RequestContext, WritePlan
from extraction.cross_prompt_relations import (
    generate_cross_prompt_relations, run_integrated_relations,
)
from index.activation_projection import ACTIVATION_VERSION
from index.index_record_builder import build_index_records
from index.outbox_worker import OutboxWorker
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.authorized_vector_search import AuthorizedVectorSearch
from retrieval.context_reader import ContextReader
from service.relation_candidate_service import RelationCandidateService


@pytest.fixture
def setup():
    ctx = RequestContext("acme", "alice", "agent", "session", "trace")
    old = ContextNode(uri="ctx://acme/users/alice/memories/entity/old",
        context_type="MEMORY", category="entity", level=3, owner_space="user:alice",
        abstract="A project", overview="A project", content="Project details",
        metadata={"update_activation": "The project uses a database.",
                  "update_activation_version": ACTIVATION_VERSION,
                  "provenance_ids": ["old-source"]})
    new = replace(old, uri=old.uri.replace("/old", "/new"),
                  metadata={**old.metadata, "provenance_ids": ["new-source"]})
    candidate = CandidateMemory("entity", "user", "new", new.abstract,
        new.overview, new.content, 0.9, provenance_ids=["new-source"], metadata=new.metadata)
    plan = WritePlan("create", new.uri)
    nodes = {old.uri: old, new.uri: new}
    fs = Mock()
    fs.read_node.side_effect = lambda uri, ctx: nodes[uri]
    embedder = Mock(model="test-model")
    embedder.embed_texts.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    index = InMemoryVectorIndex()
    worker = OutboxWorker(index, embedder)
    for node in nodes.values():
        records = build_index_records(node)
        index.upsert(worker._augment_with_embeddings(records, [[1.0, 0.0]] * len(records)))
    return ctx, old, new, candidate, plan, fs, embedder, index


def recommend(setup):
    ctx, old, new, candidate, plan, fs, embedder, index = setup
    return RelationCandidateService(
        AuthorizedVectorSearch(index, embedder, ContextReader(fs))
    ).recommend([candidate], [plan], ctx)[0]


def test_current_batch_excluded_and_current_historical_supplied(setup):
    selected = recommend(setup)
    assert [item.node_ref for item in selected] == [setup[1].uri]
    assert selected[0].source_ids == ("old-source",)


@pytest.mark.parametrize("change", ["content", "version", "owner", "model"])
def test_fail_closed_on_stale_or_unauthorized_projection(setup, change):
    ctx, old, new, candidate, plan, fs, embedder, index = setup
    if change == "content":
        old.content = "Changed after indexing"
    elif change == "version":
        old.metadata["update_activation_version"] = "old-standalone"
    elif change == "owner":
        old.owner_space = "user:someone-else"
    elif change == "model":
        embedder.model = "different-model"
    else:
        setup = (replace(ctx, acl_enforced=True, visible_space_ids=("private-space",)),
                 *setup[1:])
    assert recommend(setup) == []


def test_mixed_prompt_and_evidence_validation(setup):
    ctx, old, new, candidate, plan, fs, embedder, index = setup
    historical = recommend(setup)
    valid = {"source_node_ref": new.uri, "target_node_ref": old.uri,
             "relation": "extends", "evidence_ids": ["new-source", "old-source"]}
    llm = Mock()
    llm.complete_json.return_value = {"relations": [valid,
        {**valid, "evidence_ids": ["new-source"]},
        {**valid, "target_node_ref": "invented"}, valid]}
    edges, errors, count = generate_cross_prompt_relations(
        [{"id": "new-source", "role": "user", "content": "Project update"}],
        [candidate], [plan], historical, llm=llm)
    assert len(edges) == 1 and count == 4 and len(errors) == 3
    prompt = llm.complete_json.call_args.args[0]
    assert "NEWLY EXTRACTED MEMORY NODES:" in prompt
    assert "POTENTIALLY RELEVANT EXISTING MEMORY NODES:" in prompt
    assert "update_activation" not in prompt


def test_lookup_failure_still_calls_relation_once_without_activation_generation(setup):
    ctx, old, new, candidate, plan, fs, embedder, index = setup
    embedder.embed_texts.side_effect = RuntimeError("offline")
    llm = Mock()
    llm.complete_json.return_value = {"relations": []}
    edges, diagnostics = run_integrated_relations([], [candidate], [plan], ctx,
        fs=fs, vector_index=index, embedder=embedder, llm=llm)
    assert edges == [] and "historical_recommendation_failure" in diagnostics
    llm.complete_json.assert_called_once()


def test_sql_filters_and_metadata_preserve_activation_revision(setup):
    from providers.vector_index.opengauss_index import OpenGaussVectorIndex
    index = object.__new__(OpenGaussVectorIndex)
    where, params = index._build_where({"account_id": "acme", "level": 3,
        "record_kind": "update_activation", "representation_version": ACTIVATION_VERSION,
        "space_id": ["space-a", "space-b"]})
    assert "representation_version" in where and "record_kind" in where
    assert params["f_space_id"] == ["space-a", "space-b"]
    record = build_index_records(setup[1])[-1]
    hit = index._row_to_hit({"uri": record.uri, "level": 3, "metadata": record.metadata,
                             "filters": record.filters})
    assert hit.metadata["memory_revision"] == record.metadata["memory_revision"]


def test_cross_edge_cap_and_historical_only_rejection(setup):
    ctx, old, new, candidate, plan, fs, embedder, index = setup
    historical = recommend(setup)
    llm = Mock()
    llm.complete_json.return_value = {"relations": [
        {"source_node_ref": new.uri, "target_node_ref": old.uri,
         "relation": f"relationship-{i}", "evidence_ids": ["new-source", "old-source"]}
        for i in range(5)]}
    edges, errors, count = generate_cross_prompt_relations([], [candidate], [plan],
                                                          historical, llm=llm)
    assert len(edges) == 3 and count == 5
    assert all("cross_prompt_cap" in error for error in errors)


def test_wrong_account_hit_never_reaches_canonical_read(setup):
    from core.models import SeedHit
    ctx, old, new, candidate, plan, fs, embedder, index = setup
    index = Mock()
    index.search_by_vector.return_value = [SeedHit(
        uri="ctx://other/users/alice/memories/entity/secret", level=3, score=.99)]
    selected, diagnostics = RelationCandidateService(
        AuthorizedVectorSearch(index, embedder, ContextReader(fs))
    ).recommend([candidate], [plan], ctx)
    assert selected == []
    assert "wrong_account" in diagnostics.rejection_reasons
    fs.read_node.assert_not_called()


def test_unstored_or_skipped_candidate_cannot_trigger_relation_call(setup):
    ctx, old, new, candidate, plan, fs, embedder, index = setup
    fs.read_node.side_effect = KeyError("not committed")
    llm = Mock()
    for candidate_plan in [plan, replace(plan, action="skip")]:
        edges, diagnostics = run_integrated_relations([], [candidate], [candidate_plan], ctx,
            fs=fs, vector_index=index, embedder=embedder, llm=llm)
        assert edges == [] and diagnostics["status"] == "no_stored_candidates"
    llm.complete_json.assert_not_called()
    embedder.embed_texts.assert_not_called()


def test_historical_only_relation_is_rejected(setup):
    ctx, old, new, candidate, plan, fs, embedder, index = setup
    historical = recommend(setup)
    historical.append(replace(historical[0], node_ref=old.uri + "-second"))
    llm = Mock()
    llm.complete_json.return_value = {"relations": [{"source_node_ref": old.uri,
        "target_node_ref": historical[1].node_ref, "relation": "related to",
        "evidence_ids": ["old-source"]}]}
    edges, errors, count = generate_cross_prompt_relations([], [candidate], [plan],
                                                          historical, llm=llm)
    assert edges == [] and errors == ("0:historical_to_historical",)


def test_failed_write_cannot_shift_candidate_endpoint_mapping(setup):
    ctx, old, new, candidate, plan, fs, embedder, index = setup
    llm = Mock()
    edges, diagnostic = run_integrated_relations([], [candidate, candidate], [plan], ctx,
        fs=fs, embedder=embedder, vector_index=index, llm=llm)
    assert edges == [] and diagnostic["reason"] == "candidate_write_alignment_failed"
    llm.complete_json.assert_not_called()
    embedder.embed_texts.assert_not_called()
