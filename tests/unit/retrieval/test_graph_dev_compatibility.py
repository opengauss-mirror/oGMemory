"""Graph port preserves dev's native scopes, quotas and scene assembly."""
from types import SimpleNamespace
from unittest.mock import Mock

from core.models import LeafHit, RetrievalConfig, RetrievedBlock, TypedQuery
from retrieval.pipeline import RetrievalPipeline
from retrieval.trace import RetrievalTrace
from tests.unit.retrieval.test_graph_supplement import CTX, ROOT, fixture, render


def test_graph_evidence_is_append_only_beyond_native_fifteen_candidate_limit():
    items = [dict(uri=ROOT+str(i), category="entity", content=f"Native fact {i}.") for i in range(15)]
    extra = dict(uri=ROOT+'extra', category='entity', content='Additional graph evidence.',
                 metadata={'retrieval_origin':'graph_neighbor'})
    budget = {'working_set':10000}
    original = render(items,budget)
    enriched = render(items+[extra],budget)
    assert enriched.startswith(original)
    assert 'Additional graph evidence.' in enriched
    assert 'Native fact 14.' in enriched


def test_graph_addition_too_large_never_clips_the_native_prefix():
    items = [dict(uri=ROOT+'a',category='entity',content='Native source.')]
    extra = dict(uri=ROOT+'b',category='entity',content='Graph source. '*10000,
                 metadata={'retrieval_origin':'graph_neighbor'})
    assert render(items+[extra],{'working_set':500}) == render(items,{'working_set':500})


def test_two_hop_respects_directory_scope_without_cross_scope_bridges():
    lane,_,edge=fixture();edge('a','b');edge('b','c')
    initial=[SimpleNamespace(uri=ROOT+'a')]
    result=lane.select('memory',initial,initial,CTX,query_vector=[1.,0.],memory_directories=['preferences'])
    assert result['selected']==[]
    assert result['memory_directories']==['preferences']


def test_scene_per_query_inputs_keep_graph_additions_and_global_native_top_k():
    config=RetrievalConfig(global_result_rerank_enabled=True)
    assembly=Mock()
    assembly.assemble.side_effect=lambda tq,hits,*args,**kwargs:[
        RetrievedBlock(uri=h.uri,metadata=h.metadata,score=h.score) for h in hits]
    pipe=RetrievalPipeline(Mock(),Mock(),None,assembly,config)
    native=LeafHit(ROOT+'a',1.)
    graph=LeafHit(ROOT+'b',.5,metadata={'retrieval_origin':'graph_neighbor'})
    query=TypedQuery('fact','MEMORY',[],top_k=1)
    result=pipe._run_assembly([query],[native,graph],None,RetrievalTrace(),CTX,
                            query_results=[(query,[native],None,1)])
    assert [hit.uri for hit in result]==[native.uri,graph.uri]


def test_one_hop_hydrated_neighbor_cannot_escape_directory_filter():
    from retrieval.graph_retriever import GraphHit
    lane,_,_=fixture()
    graph=Mock()
    async def search(**kwargs):
        return [GraphHit(ROOT+'b',1.,hop_distance=1)]
    graph.search_with_report=search
    pipe=RetrievalPipeline(Mock(),Mock(),None,Mock(),RetrievalConfig(),
                           context_reader=lane.reader,graph_retriever=graph)
    initial=[LeafHit(ROOT+'a',1.)]
    assert pipe._expand_by_graph('fact',initial,CTX,RetrievalTrace(),
                                memory_directories=['preferences'])==initial
