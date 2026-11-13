"""Completed candidate projections overlap the subsequent relationship phase."""

import os
from secrets import randbits
from threading import BoundedSemaphore

from core.models import CandidateMemory, RelationEdge, WritePlan
from fs.sql_adapter.session_mutation import session_mutation_scope
from providers.llm.mock_llm import MockLLM
from service.api import MemoryWriteAPI
from tests.integration.test_graph_projection_mutation_batch import (
    _jobs, _project_independently, _projection,
)
from tests.integration.test_graph_projection_source_refresh import _update_content
from tests.integration.test_semantic_context_graph import graph_db as graph_db


def test_projection_completes_during_relationship_inference(graph_db, monkeypatch):
    ctx, fs, index, store, projections, outbox, worker, nodes, profile = graph_db
    source, target, ready = nodes
    store.upsert_edges([
        RelationEdge(source.uri, target.uri, "owns", 1., "evidence"),
        RelationEdge(source.uri, ready.uri, "supports", 1., "evidence"),
    ], ctx)
    api = MemoryWriteAPI(fs, MockLLM(), outbox, vector_index=index,
                         embedder=worker._embedder, graph_enabled=True, relation_store=store)
    candidate = CandidateMemory("entity", "user", "target", "Target", "Target details",
                                "A current observation", .9, target_uri=target.uri)
    monkeypatch.setattr(api._pipeline, "extract", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr("commit.topic_aliases.reconcile_topic_aliases", lambda candidates, *args: candidates)

    def write_candidates(candidates, context):
        for revision in (2, 3):
            _update_content(fs, context, target, f"Candidate revision {revision}")
        ready_items = _project_independently(projections, worker, profile, context.account_id)
        assert [item.target_uri for item in ready_items] == [ready.uri]
        assert _jobs(fs, context)[target.uri]["target_revision"] == 3
        return [WritePlan("update", target.uri)]

    def infer_relations(*args, **kwargs):
        accepted = _project_independently(projections, worker, profile, ctx.account_id)
        assert [(item.target_uri, item.target_revision) for item in accepted] == [(target.uri, 3)]
        assert _projection(store, ctx, source, target, profile)["target_revision"] == 3
        return [], {"projection_completed_during_inference": True}

    monkeypatch.setattr(api._writer, "write_candidates", write_candidates)
    monkeypatch.setattr("extraction.cross_prompt_relations.run_integrated_relations", infer_relations)
    with session_mutation_scope(
        os.environ["GRAPH_TEST_DSN"], ctx, randbits(63) or 1,
        BoundedSemaphore(1), timeout_seconds=5,
        dialect=os.environ.get("GRAPH_TEST_DIALECT", "postgresql"),
    ):
        result = api.commit_session([{"role": "user", "content": "A current observation"}], ctx)

    assert result["writes_completed"] == 1
    assert result["graph"]["projection_completed_during_inference"]
    assert _jobs(fs, ctx)[target.uri]["status"] == "COMPLETE"
