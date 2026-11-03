"""PR266 review regressions: category scope and stored model identity."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.models import IndexRecord,LeafHit,RetrievalConfig,SeedResult,TypedQuery
from retrieval.pipeline import RetrievalPipeline
from retrieval.graph_retriever import GraphHit
from tests.unit.retrieval.test_graph_supplement import fixture,ROOT,CTX,PROFILE


@pytest.mark.parametrize('explicit',[True,False])
def test_pipeline_one_and_two_hop_keep_effective_category_filter(explicit):
    lane,blocks,edge=fixture()
    blocks[ROOT+'a'].category='preference'
    edge('a','b');edge('b','c')
    config=RetrievalConfig(enable_graph_expansion=True)
    graph=Mock()
    async def search(**kwargs):
        return [GraphHit(uri=ROOT+'b',score=1.,hop_distance=1)]
    graph.search_with_report=search
    pipe=RetrievalPipeline(Mock(),Mock(),None,Mock(),config,context_reader=lane.reader,
        graph_retriever=graph,graph_supplement=lane)
    pipe._run_planner=Mock(return_value=[TypedQuery('fact','MEMORY',['preference'],top_k=1)])
    pipe._run_seed=Mock(return_value=SeedResult(query_vector=[1.,0.]))
    initial=[LeafHit(ROOT+'a',1.,category='preference')]
    pipe._run_expand=Mock(return_value=initial)
    pipe._run_assembly=Mock(return_value=[])
    result=pipe.run('fact',CTX,categories=['preference'] if explicit else None,collect_graph_supplement=True)
    assert pipe._run_assembly.call_args.args[1]==initial
    supplement=result.trace.graph_expansion['two_hop_supplement']
    assert supplement['categories']==['preference'] and supplement['selected']==[]


def test_two_hop_cannot_use_excluded_category_as_bridge():
    lane,blocks,edge=fixture();edge('a','b');edge('b','c')
    for name in 'ac':blocks[ROOT+name].category='preference'
    initial=[SimpleNamespace(uri=ROOT+'a')]
    assert len(lane.select('fact',initial,initial,CTX,query_vector=[1.,0.])['selected'])==2
    assert lane.select('fact',initial,initial,CTX,query_vector=[1.,0.],categories={'preference'})['selected']==[]


@pytest.mark.parametrize('metadata',[
    {}, {'embedding_model':'other','embedding_dimension':2},
    {'embedding_model':PROFILE['model'],'embedding_dimension':3},
])
def test_incoming_vector_rejects_missing_or_different_model_identity(metadata):
    lane,blocks,edge=fixture();edge('b','a',vector=(0.,1.))
    lane.index.get_record.return_value=IndexRecord(id='id',uri=ROOT+'b',level=0,text='b',
        filters={'account_id':'acct','owner_space':'user:user'},
        metadata={'record_kind':'node_abstract','canonical_revision':1,'_embedding':[1.,0.],**metadata})
    initial=[SimpleNamespace(uri=ROOT+'a')]
    assert lane.select('fact',initial,initial,CTX,query_vector=[1.,0.])['selected']==[]


def test_final_delivery_rechecks_category_after_selection():
    from tests.unit.retrieval.test_graph_supplement import render
    lane,blocks,edge=fixture();edge('a','b')
    initial=[SimpleNamespace(uri=ROOT+'a')]
    audit=lane.select('fact',initial,initial,CTX,query_vector=[1.,0.],categories={'entity'})
    blocks[ROOT+'b'].category='preference'
    items=[dict(uri=ROOT+'a',content='Original',category='entity')]
    stats={'graph_expansion':{'two_hop_supplement':audit}}
    assert lane.deliver(items,'fact',stats,CTX,{'working_set':1000},render)==items
    assert 'category_outside_request' in stats['graph_supplement_delivery']['rejected_additions']


def test_ordinary_node_vectors_persist_real_embedding_identity():
    from index.outbox_worker import OutboxWorker
    worker=OutboxWorker(Mock(),Mock(model='new-model'))
    original=IndexRecord(id='node',uri=ROOT+'a',level=0,text='Fact',
        filters={'account_id':'acct','owner_space':'user:user'},metadata={'record_kind':'node_abstract'})
    result=worker._augment_with_embeddings([original],[[1.,0.]])[0]
    assert result.metadata['embedding_model']=='new-model'
    assert result.metadata['embedding_dimension']==2
    assert '_embedding' not in original.metadata
