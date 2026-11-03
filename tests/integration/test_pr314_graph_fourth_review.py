"""Bounded one-hop recall and idle polling over real SQL; no paid providers."""
import asyncio
from unittest.mock import Mock

import pytest

from core.errors import AccessDeniedError
from core.models import ContextNode, RelationEdge, SeedHit
from retrieval.graph_retriever import OneHopRetriever
from service.graph_service import GraphService
from tests.integration.test_semantic_context_graph import graph_db


@pytest.mark.parametrize("rejection", ["permission", "category", "projection"])
def test_sql_onehop_recovers_fourth_edge_after_three_rejections(graph_db, monkeypatch, rejection):
    ctx,fs,_,store,_,outbox,worker,nodes,profile = graph_db
    anchor = nodes[0]
    targets = []
    for i in range(4):
        target = ContextNode(uri=anchor.uri+"-target-"+str(i),context_type="MEMORY",
            category="preference" if rejection=="category" and i<3 else "entity",level=2,
            owner_space="user:alice",abstract="Target",overview="Overview",content="Body",
            metadata={"_expected_version":0})
        fs.write_node(target,ctx)
        targets.append(target)
    store.upsert_edges([RelationEdge(anchor.uri,t.uri,"supports",1.-i*.05,"Evidence")
                        for i,t in enumerate(targets)],ctx)
    assert worker.run_once(outbox,[ctx.account_id])["failed"] == 0
    rejected = {t.uri for t in targets[:3]}
    if rejection == "permission":
        accessible = store._ensure_accessible
        def ensure(uri, *args, **kwargs):
            if uri in rejected: raise AccessDeniedError(uri,ctx.account_id,"Access revoked")
            return accessible(uri,*args,**kwargs)
        monkeypatch.setattr(store,"_ensure_accessible",ensure)
    elif rejection == "projection":
        conn = store._get_connection()
        try:
            store.bind_tenant(conn,ctx.account_id)
            with conn.cursor() as cur:
                cur.execute("UPDATE relation_edges SET edge_projection=NULL WHERE account_id=%s AND to_uri=ANY(%s)",
                            (ctx.account_id,list(rejected)))
            conn.commit()
        finally: store._return_connection(conn)
    retriever = OneHopRetriever(GraphService(None,relation_reader=store),embedding_profile=profile)
    hits, _ = asyncio.run(retriever.search_with_report("fact",ctx,
        [SeedHit(uri=anchor.uri,score=1.,level=0)],top_k=15,query_vector=[1.,0.],categories={"entity"}))
    assert [h.uri for h in hits if h.hop_distance] == [targets[3].uri]


def test_ten_thousand_completed_jobs_do_not_trigger_idle_claims(graph_db, monkeypatch):
    ctx,_,_,_,projections,outbox,worker,nodes,profile = graph_db
    assert worker.run_once(outbox,[ctx.account_id])["failed"] == 0
    conn = projections._get_connection()
    try:
        projections.bind_tenant(conn,ctx.account_id)
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO edge_projection_jobs
                (account_id,edge_id,source_uri,target_uri,relation_type,reason,target_text,
                 source_revision,target_revision,representation_version,embedding_profile,status)
                SELECT %s,'completed-'||n,%s,%s,'supports','','',1,1,%s,%s,'COMPLETE'
                FROM generate_series(1,10000) n""",
                (ctx.account_id,nodes[0].uri,nodes[1].uri,profile['representation_version'],profile['profile_id']))
        conn.commit()
    finally: projections._return_connection(conn)
    claim = Mock(wraps=projections.claim_pending)
    monkeypatch.setattr(projections,"claim_pending",claim)
    for _ in range(30):
        stats = worker.run_once(outbox,[ctx.account_id])
        assert stats["failed"] == 0
    assert claim.call_count == 0
    assert not projections.has_due_work(ctx.account_id)
    # A new genuine projection is still discovered after an idle period.
    from core.models import RelationEdge
    store = graph_db[3]
    store.upsert_edges([RelationEdge(nodes[0].uri,nodes[1].uri,"new",1.,"Evidence")],ctx)
    assert worker.run_once(outbox,[ctx.account_id])["failed"] == 0
    assert claim.call_count == 1
    assert store.get_relation_neighbors(nodes[0].uri,ctx)[0].edge_projection
