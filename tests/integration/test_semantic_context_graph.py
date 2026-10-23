"""Opt-in SQL graph integration tests; use an isolated test database only.

Run with GRAPH_TEST_DSN pointing to a disposable PostgreSQL/pgvector database.
No external LLM or embedding service is called.
"""
import os
from dataclasses import replace
from uuid import uuid4
from unittest.mock import Mock

import pytest

from core.errors import AccessDeniedError
from core.models import ContextNode, IndexRecord, LeafHit, RelationEdge, RequestContext, RetrievalConfig, SeedResult, TypedQuery
from fs.sql_adapter.sql_context_fs import SQLContextFS
from commit.sql_outbox_store import SQLOutboxStore
from index.activation_projection import ACTIVATION_VERSION
from index.index_record_builder import build_index_records
from index.outbox_worker import OutboxWorker
from providers.relation_store.sql_relation_store import SQLRelationStore
from providers.relation_store.sql_projection import SQLEdgeProjectionStore
from providers.vector_index.opengauss_index import OpenGaussVectorIndex
from retrieval.context_reader import ContextReader
from retrieval.pipeline import RetrievalPipeline
from retrieval.result_ranker import ResultRanker
from retrieval.graph_retriever import OneHopRetriever
from retrieval.graph_supplement import GraphSupplement
from service.graph_service import GraphService


@pytest.fixture
def graph_db():
    dsn = os.environ.get("GRAPH_TEST_DSN")
    if not dsn:
        pytest.skip("GRAPH_TEST_DSN must point to a disposable database")
    account = "graph-test-" + uuid4().hex[:12]
    ctx = RequestContext(account,"alice","agent","session","trace")
    fs = SQLContextFS(dsn)
    profile = dict(profile_id="test",model="test-embedding",dimension=2,representation_version="relation-target-v2")
    embedder = Mock(model="test-embedding")
    embedder.embed_texts.side_effect = lambda texts: [[1.,0.] for _ in texts]
    index = OpenGaussVectorIndex(dsn,dimension=2,table_name="graph_test_vectors")
    store = SQLRelationStore(dsn,edge_projection_profile_id="test")
    projections = SQLEdgeProjectionStore(dsn)
    outbox = SQLOutboxStore(dsn,fs=fs)
    worker = OutboxWorker(index,embedder,fs=fs,directory_summary_enabled=False,
                          edge_projection_components=(projections,profile))
    root = f"ctx://{account}/users/alice/memories/entity/"
    nodes = []
    for name in ("a","b","c"):
        node = ContextNode(uri=root+name,context_type="MEMORY",category="entity",level=2,
            owner_space="user:alice",abstract="Fact "+name,overview="Overview "+name,content="Details "+name,
            metadata={"_expected_version":0,"update_activation":"Concrete fact about "+name,
                      "update_activation_version":ACTIVATION_VERSION,"provenance_ids":[name+"-source"]})
        fs.write_node(node,ctx)
        records = build_index_records(node)
        index.upsert(worker._augment_with_embeddings(records,embedder.embed_texts([r.text for r in records])))
        nodes.append(node)
    return ctx,fs,index,store,projections,outbox,worker,nodes,profile


def test_sql_projection_outbox_and_live_retrieval(graph_db):
    ctx,fs,index,store,projections,outbox,worker,nodes,profile = graph_db
    a,b,c = nodes
    edges = [RelationEdge(a.uri,b.uri,"extends",1.,"evidence:a-source,b-source"),
             RelationEdge(b.uri,c.uri,"explains",1.,"evidence:b-source,c-source")]
    store.upsert_edges(edges,ctx)
    assert store.get_traversal_neighbors(a.uri,ctx,embedding_profile=profile) == []
    stats = worker.run_once(outbox,[ctx.account_id])
    assert stats["failed"] == 0
    neighbors = store.get_relation_neighbors(a.uri,ctx)
    assert neighbors[0].reason == edges[0].reason
    assert neighbors[0].edge_projection["vector"] == [1.,0.]
    assert "a-source" not in neighbors[0].edge_projection["text"]
    reader = ContextReader(fs,storage_backend="sql")
    graph = GraphService(None,relation_reader=store)
    lane = GraphSupplement(graph,reader,index,worker._embedder,profile)
    cfg = RetrievalConfig(enable_graph_expansion=True)
    pipe = RetrievalPipeline(Mock(),Mock(),None,ResultRanker(cfg),cfg,context_reader=reader,
        graph_retriever=OneHopRetriever(graph,embedder=worker._embedder,embedding_profile=profile),
        graph_supplement=lane)
    pipe._run_planner = Mock(return_value=[TypedQuery("fact","MEMORY",[],top_k=1)])
    pipe._run_seed = Mock(return_value=SeedResult(query_vector=[1.,0.]))
    pipe._run_expand = Mock(return_value=[LeafHit(a.uri,1.,abstract="Fact a",owner_space="user:alice")])
    result = pipe.run("fact",ctx,collect_graph_supplement=True)
    assert [h.uri for h in result.hits] == [a.uri,b.uri]
    assert result.hits[1].content_excerpt == "Details b"
    assert result.trace.graph_expansion["two_hop_supplement"]["selected"][0]["uri"] == c.uri
    records = index.get_records([IndexRecord.generate_id(a.uri,0),"missing"])
    assert records[0].metadata["_embedding"] == [1.,0.] and records[1] is None
    with pytest.raises(AccessDeniedError):
        store.get_traversal_neighbors(a.uri,replace(ctx,user_id="mallory"),embedding_profile=profile)


def test_revision_change_invalidates_and_reprojects(graph_db):
    ctx,fs,index,store,projections,outbox,worker,nodes,profile = graph_db
    a,b,_ = nodes
    edge = RelationEdge(a.uri,b.uri,"related "+"x"*80,1.,"evidence:a-source,b-source")
    store.upsert_edges([edge],ctx)
    old = projections.claim_pending(ctx.account_id)
    assert len(old) == 1
    current = fs.read_node(b.uri,ctx)
    current.content = "Changed target"
    current.metadata["_expected_version"] = current.metadata["version"]
    fs.write_node(current,ctx)
    from index.edge_projection import EdgeProjectionWorker
    EdgeProjectionWorker(worker._embedder,projections,profile).project_accepted_sync(old)
    assert store.get_traversal_neighbors(a.uri,ctx,embedding_profile=profile) == []
    projections.drain_sync(ctx.account_id,worker._embedder,profile)
    neighbors = store.get_relation_neighbors(a.uri,ctx)
    assert neighbors[0].target_revision == 2
    assert neighbors[0].edge_projection["target_revision"] == 2
    assert "Changed target" in neighbors[0].edge_projection["text"]
    store.upsert_edges([edge],ctx)
    assert projections.claim_pending(ctx.account_id) == []


def test_projection_failure_keeps_durable_wakeup_across_restart(graph_db):
    ctx,fs,index,store,projections,outbox,worker,nodes,profile = graph_db
    from index.outbox_worker import WorkerConfig
    a,b,_ = nodes
    store.upsert_edges([RelationEdge(a.uri,b.uri,"explains",1.,"evidence:a-source,b-source")],ctx)
    worker._embedder.embed_texts.side_effect = RuntimeError("embedding offline")
    worker._config = WorkerConfig(base_delay_ms=0)
    stats = worker.run_once(outbox,[ctx.account_id])
    assert stats["failed"] >= 1
    # Bring the isolated retry clock forward, without waiting a real minute.
    conn = projections._get_connection()
    try:
        projections.bind_tenant(conn,ctx.account_id)
        with conn.cursor() as cur:
            cur.execute("UPDATE edge_projection_jobs SET next_retry_at=NOW() WHERE account_id=%s",(ctx.account_id,))
        conn.commit()
    finally:
        projections._return_connection(conn)
    worker._embedder.embed_texts.side_effect = lambda texts: [[1.,0.] for _ in texts]
    restarted = OutboxWorker(index,worker._embedder,fs=fs,directory_summary_enabled=False,
        edge_projection_components=(SQLEdgeProjectionStore(os.environ["GRAPH_TEST_DSN"]),profile))
    stats = restarted.run_once(outbox,[])
    assert stats["failed"] == 0 and stats["processed"] >= 1
    assert store.get_relation_neighbors(a.uri,ctx)[0].edge_projection is not None


def test_real_extraction_writer_and_index_pipeline_keeps_descriptors(graph_db):
    """Run upstream extraction/validation/writer with fake provider responses."""
    import json
    import re
    from service.api import MemoryWriteAPI
    from retrieval.authorized_vector_search import AuthorizedVectorSearch
    from service.relation_candidate_service import RelationCandidateService
    ctx,fs,index,store,projections,outbox,worker,nodes,profile = graph_db
    llm = Mock()
    llm.complete_with_tools.return_value = [
        dict(tool="extract_entity",input=dict(routing_key=name,abstract="Project "+name,
            overview="Organization project "+name,content="Details for project "+name,
            confidence=.95,update_activation="Organization project "+name+" uses a database."))
        for name in ("projectx","projecty")]
    def relation_response(prompt,schema):
        if "spans" in schema.get("properties",{}):
            return {"spans":[dict(start=0,end=0,categories=["entity"],reason="Project facts")]}
        if "groups" in schema.get("properties",{}):
            return {"groups":[[0],[1]]}  # dev's native topic-identity review
        block = prompt.split("NEWLY EXTRACTED MEMORY NODES:",1)[1].split("POTENTIALLY RELEVANT EXISTING MEMORY NODES:",1)[0]
        new = json.loads(re.search(r"^(\[.*\])$",block,re.MULTILINE).group(1))
        return {"relations":[dict(source_node_ref=new[0]["node_ref"],target_node_ref=new[1]["node_ref"],
            relation="depends on",evidence_ids=new[0]["source_ids"]+new[1]["source_ids"])]}
    llm.complete_json.side_effect = relation_response
    api = MemoryWriteAPI(fs,llm,outbox_store=outbox,vector_index=index,embedder=worker._embedder,
        directory_summary_enabled=False,graph_enabled=True,relation_store=store,
        relation_candidate_service=RelationCandidateService(AuthorizedVectorSearch(
            index,worker._embedder,ContextReader(fs,storage_backend="sql"))))
    result = api.commit_session([dict(id="turn",role="user",content="Projectx depends on Projecty.")],
                                ctx,archive_id="graph-test-archive")
    assert result["writes_completed"] == 2
    assert result["graph"]["relation_persisted"] == 1
    # dev's span identification and topic review plus one graph relation call;
    # both native extraction passes are retained.
    assert llm.complete_with_tools.call_count == 2
    assert llm.complete_json.call_count == 3
    for plan in result["plans"]:
        node = fs.read_node(plan["target_uri"],ctx)
        assert node.metadata["update_activation_version"] == ACTIVATION_VERSION
        assert node.metadata["provenance_ids"]
    stats = worker.run_once(outbox,[ctx.account_id])
    assert stats["failed"] == 0
    source = result["plans"][0]["target_uri"]
    assert store.get_relation_neighbors(source,ctx)[0].edge_projection is not None
    descriptor = index.get_record(IndexRecord.generate_id(source,3))
    assert descriptor.level == 3 and descriptor.metadata["record_kind"] == "update_activation"


def _graph_sql(fs,ctx,query,params=()):
    conn=fs._get_connection()
    try:
        fs.bind_tenant(conn,ctx.account_id)
        with conn.cursor() as cur:
            cur.execute(query,params)
            rows=cur.fetchall() if cur.description else []
        conn.commit();return rows
    except Exception:
        conn.rollback();raise
    finally:fs._return_connection(conn)


@pytest.mark.parametrize('same_revision',[False,True])
def test_stale_worker_failure_cannot_reset_completed_replacement(graph_db,same_revision):
    from index.edge_projection import EdgeProjectionWorker
    ctx,fs,index,store,old_store,outbox,worker,nodes,profile=graph_db
    a,b,_=nodes;store.upsert_edges([RelationEdge(a.uri,b.uri,'related',1.,'evidence')],ctx)
    old=old_store.claim_pending(ctx.account_id)
    if same_revision:
        _graph_sql(fs,ctx,"UPDATE edge_projection_jobs SET updated_at=NOW()-INTERVAL '6 minutes' WHERE account_id=%s",(ctx.account_id,))
    else:
        current=fs.read_node(b.uri,ctx);current.content='New target';current.metadata['_expected_version']=current.metadata['version']
        fs.write_node(current,ctx)
    new_store=SQLEdgeProjectionStore(os.environ['GRAPH_TEST_DSN'])
    new=new_store.claim_pending(ctx.account_id)
    assert old[0].claim_token!=new[0].claim_token
    EdgeProjectionWorker(worker._embedder,new_store,profile).project_accepted_sync(new)
    old_store.mark_failed([old[0].edge_id],'late failed embedding call',claims=old)
    # A late successful result must also be fenced by the same lease.
    EdgeProjectionWorker(Mock(embed_texts=lambda texts:[[0.,1.] for _ in texts]),old_store,profile).project_accepted_sync(old)
    assert _graph_sql(fs,ctx,'SELECT status FROM edge_projection_jobs WHERE account_id=%s',(ctx.account_id,))==[('COMPLETE',)]
    assert store.get_relation_neighbors(a.uri,ctx)[0].edge_projection['vector']==[1.,0.]


def test_node_update_leaves_durable_wakeup_after_ordinary_event_ack_and_restart(graph_db):
    ctx,fs,index,store,projections,outbox,worker,nodes,profile=graph_db
    a,b,_=nodes;store.upsert_edges([RelationEdge(a.uri,b.uri,'related',1.,'evidence')],ctx)
    projections.drain_sync(ctx.account_id,worker._embedder,profile)
    _graph_sql(fs,ctx,'DELETE FROM outbox_events WHERE account_id=%s',(ctx.account_id,))
    current=fs.read_node(b.uri,ctx);current.content='Updated after indexing';current.metadata['_expected_version']=current.metadata['version']
    fs.write_node(current,ctx)
    _graph_sql(fs,ctx,"DELETE FROM outbox_events WHERE account_id=%s AND event_type <> 'PROJECT_GRAPH'",(ctx.account_id,))
    assert _graph_sql(fs,ctx,"SELECT count(*) FROM outbox_events WHERE account_id=%s AND event_type='PROJECT_GRAPH'",(ctx.account_id,))==[(1,)]
    restarted=OutboxWorker(index,worker._embedder,fs=fs,directory_summary_enabled=False,
        edge_projection_components=(SQLEdgeProjectionStore(os.environ['GRAPH_TEST_DSN']),profile))
    stats=restarted.run_once(outbox,[])
    assert stats['failed']==0
    assert store.get_relation_neighbors(a.uri,ctx)[0].edge_projection['target_revision']==2


@pytest.mark.parametrize('endpoint',['source','target'])
def test_move_rekeys_projection_jobs_rebuilds_descriptor_and_survives_next_update(graph_db,endpoint):
    from index.edge_projection import EdgeProjectionWorker
    from providers.relation_store.projection_identity import canonical_edge_id
    ctx,fs,index,store,projections,outbox,worker,nodes,profile=graph_db
    a,b,_=nodes;store.upsert_edges([RelationEdge(a.uri,b.uri,'related',1.,'evidence')],ctx)
    old=projections.claim_pending(ctx.account_id)
    moved=a if endpoint=='source' else b;destination=moved.uri+'-moved'
    fs.move_node(moved.uri,destination,ctx)
    source=destination if endpoint=='source' else a.uri;target=destination if endpoint=='target' else b.uri
    new_id=canonical_edge_id(ctx.account_id,source,target,'related')
    rows=_graph_sql(fs,ctx,'SELECT edge_id,source_uri,target_uri,status,claim_token FROM edge_projection_jobs WHERE account_id=%s',(ctx.account_id,))
    assert rows==[(new_id,source,target,'PENDING',None)]
    EdgeProjectionWorker(worker._embedder,projections,profile).project_accepted_sync(old)
    descriptor=None
    for _ in range(4):
        stats=worker.run_once(outbox,[ctx.account_id]);assert stats['failed']==0
        descriptor=index.get_record(IndexRecord.generate_id(destination,3))
        if descriptor:break
    assert descriptor and descriptor.metadata['record_kind']=='update_activation'
    assert index.get_record(IndexRecord.generate_id(moved.uri,3)) is None
    assert store.get_relation_neighbors(source,ctx)[0].edge_projection['edge_id']==new_id
    current=fs.read_node(destination,ctx);current.content='Changed moved node';current.metadata['_expected_version']=current.metadata['version']
    fs.write_node(current,ctx)
    assert _graph_sql(fs,ctx,'SELECT status FROM edge_projection_jobs WHERE account_id=%s',(ctx.account_id,))==[('PENDING',)]
    projections.drain_sync(ctx.account_id,worker._embedder,profile)
    assert store.get_relation_neighbors(source,ctx)[0].edge_projection is not None


def test_upsert_waits_for_node_without_holding_relation_lock(graph_db):
    """Deterministic old deadlock: node owner updates while edge writer waits."""
    from concurrent.futures import ThreadPoolExecutor
    import time
    ctx,fs,index,store,projections,outbox,worker,nodes,profile=graph_db
    a,b,_=nodes;edge=RelationEdge(a.uri,b.uri,'related',1.,'evidence')
    store.upsert_edges([edge],ctx)
    application='graph-review-'+uuid4().hex
    from psycopg2.extensions import make_dsn
    writer=SQLRelationStore(make_dsn(os.environ['GRAPH_TEST_DSN'], application_name=application),edge_projection_profile_id='test')
    conn=fs._get_connection()
    try:
        fs.bind_tenant(conn,ctx.account_id)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='3s'")
            cur.execute('SELECT uri FROM context_nodes WHERE uri=%s FOR UPDATE',(b.uri,))
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(writer.upsert_edges,[replace(edge,weight=.9)],ctx)
                deadline=time.monotonic()+5
                waiting=False
                while time.monotonic()<deadline:
                    rows=_graph_sql(fs,ctx,"SELECT count(*) FROM pg_stat_activity WHERE application_name=%s AND wait_event_type='Lock'",(application,))
                    if rows==[(1,)]:waiting=True;break
                    time.sleep(.02)
                try:
                    assert waiting,'Concurrent edge writer did not reach its endpoint lock'
                    cur.execute("UPDATE context_nodes SET content=content || ' changed',version=version+1 WHERE uri=%s",(b.uri,))
                    conn.commit()
                finally:conn.rollback()
                future.result(timeout=5)
    finally:conn.rollback();fs._return_connection(conn)


def test_projection_endpoint_indexes_are_installed(graph_db):
    ctx,fs,*_=graph_db
    names={r[0] for r in _graph_sql(fs,ctx,"SELECT indexname FROM pg_indexes WHERE tablename='edge_projection_jobs'")}
    assert {'idx_edge_projection_jobs_source','idx_edge_projection_jobs_target'}<=names
