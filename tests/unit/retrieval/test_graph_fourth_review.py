"""PR314 fourth review: bounded valid one-hop admission and idle polling."""
import asyncio
from dataclasses import replace
from unittest.mock import Mock, MagicMock

import pytest

from core.models import OutboxEvent, RelationNeighbor, SeedHit
from index.outbox_worker import OutboxWorker
from retrieval.graph_retriever import OneHopRetriever
from service.graph_service import GraphService
from tests.unit.retrieval.test_graph_supplement import CTX, PROFILE, ROOT
from tests.unit.retrieval.test_graph_third_review import projection


class Relations:
    def __init__(self, rows):
        self.rows, self.limits = rows, []

    def get_relation_neighbors(self, uri, ctx, *, limit, min_weight):
        self.limits.append(limit)
        return [r for r in self.rows if r.weight >= min_weight][:limit]


def neighbor(name, weight=1.):
    return RelationNeighbor(uri=ROOT+name, category="entity", relation_type="supports",
        weight=weight, source_revision=1, target_revision=1, edge_projection=projection())


@pytest.mark.parametrize("invalid", ["category", "model", "revision", "similarity", "missing_projection"])
def test_fourth_valid_edge_survives_three_rejected_heavier_edges(invalid):
    rows = [neighbor(str(i), 1.-i*.05) for i in range(4)]
    for i,row in enumerate(rows[:3]):
        if invalid == "category": rows[i] = replace(row, category="preference")
        elif invalid == "model": row.edge_projection["model"] = "different-model"
        elif invalid == "revision": row.edge_projection["target_revision"] = 0
        elif invalid == "similarity": row.edge_projection["vector"] = [0., 1.]
        else: rows[i] = replace(row, edge_projection=None)
    store = Relations(rows)
    retriever = OneHopRetriever(GraphService(None, relation_reader=store), embedding_profile=PROFILE)
    hits, report = asyncio.run(retriever.search_with_report("fact", CTX,
        [SeedHit(uri=ROOT+"anchor", score=1., level=0)], top_k=15,
        query_vector=[1.,0.], categories={"entity"}))
    assert [h.uri for h in hits if h.hop_distance] == [ROOT+"3"]
    assert report.accepted_edge_count == 1
    assert len(store.limits) == 1 and 3 < store.limits[0] <= 128


def test_per_anchor_quota_applies_after_semantic_ranking_and_uri_deduplication():
    rows = [neighbor("same",1.),neighbor("same",.99),neighbor("weak",.98),
            neighbor("second",.97),neighbor("third",.96),neighbor("fourth",.95)]
    rows[2].edge_projection["vector"] = [.4, .916515138991168]
    store = Relations(rows)
    retriever = OneHopRetriever(GraphService(None, relation_reader=store), embedding_profile=PROFILE)
    hits = asyncio.run(retriever.search_with_report("fact",CTX,
        [SeedHit(uri=ROOT+"anchor",score=1.,level=0)],top_k=15,query_vector=[1.,0.]))[0]
    assert [h.uri for h in hits if h.hop_distance] == [ROOT+"same",ROOT+"second",ROOT+"third"]


def test_candidate_scan_is_bounded_even_if_all_edges_rejected():
    rows = [replace(neighbor(str(i)),edge_projection=None) for i in range(1000)]
    store = Relations(rows)
    retriever = OneHopRetriever(GraphService(None, relation_reader=store), embedding_profile=PROFILE)
    hits, report = asyncio.run(retriever.search_with_report("fact",CTX,
        [SeedHit(uri=ROOT+"anchor",score=1.,level=0)],top_k=15,query_vector=[1.,0.]))
    assert len(hits) == 1 and report.candidate_edge_count == 128
    assert store.limits == [128]


class ProjectionQueue:
    def __init__(self, due=False):
        self.due = due
        self.checks, self.drains = [], []

    def has_due_work(self, account, *, profile=None):
        self.checks.append(account)
        return self.due

    def drain_sync(self, account, embedder, profile):
        self.drains.append(account)
        self.due = False
        return 1

    def unfinished(self, account, edge):
        return self.due


def batch_queue(events=()):
    queue = Mock()
    queue.supports_batch_claim = True
    queue.claim_batch.return_value = [(e.uri,e) for e in events]
    return queue


def worker(store):
    return OutboxWorker(Mock(),Mock(),edge_projection_components=(store,PROFILE))


def test_idle_polls_do_not_claim_completed_history_or_remember_old_accounts():
    projections = ProjectionQueue()
    w = worker(projections)
    empty = batch_queue()
    for i in range(50):
        w.run_once(empty,["tenant-"+str(i)])
    checks = len(projections.checks)
    for _ in range(20): w.run_once(empty,[])
    assert not projections.drains
    assert len(projections.checks) == checks == 50


def test_many_wakeups_share_one_projection_drain_per_account_per_poll():
    projections = ProjectionQueue(due=True)
    events = [OutboxEvent(str(i),"PROJECT_GRAPH",ROOT+"a",{"edge_id":str(i)},"PROCESSING")
              for i in range(12)]
    stats = worker(projections).run_once(batch_queue(events),[CTX.account_id])
    assert stats["failed"] == 0 and stats["succeeded"] == 12
    assert projections.drains == [CTX.account_id]
    assert projections.checks == [CTX.account_id]


def test_poll_cache_expires_and_later_due_retry_is_revisited():
    projections = ProjectionQueue()
    w = worker(projections)
    empty = batch_queue()
    w.run_once(empty,[CTX.account_id])
    projections.due = True
    w.run_once(empty,[CTX.account_id])
    assert projections.drains == [CTX.account_id]
    assert projections.checks == [CTX.account_id,CTX.account_id]


def test_standalone_completed_wakeup_does_not_start_a_claim():
    projections = ProjectionQueue()
    event = OutboxEvent("done","PROJECT_GRAPH",ROOT+"a",{"edge_id":"edge"},"PENDING")
    assert worker(projections).process_event(event,CTX).success
    assert not projections.drains


def test_orphan_scrub_does_not_visit_complete_projection_history():
    from providers.relation_store.sql_projection import SQLEdgeProjectionStore
    store = object.__new__(SQLEdgeProjectionStore)
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []
    store._get_connection = Mock(return_value=conn)
    store._return_connection = Mock()
    store.bind_tenant = Mock()
    store._claimed_accounts,store._claimed_tokens = {},{}
    store._dialect = Mock(supports_batch_claim=True)
    assert store.claim_pending(CTX.account_id) == []
    cleanup = next(call.args[0] for call in cur.execute.call_args_list
                   if "DELETE FROM edge_projection_jobs" in call.args[0])
    assert "j.status IN ('PENDING', 'FAILED', 'PROCESSING')" in cleanup
