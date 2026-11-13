"""Public memory queries cover indexed types within authenticated scopes."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.lexical import LexicalHit
from core.models import ContextNode, IndexRecord, RequestContext, RetrievalConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.context_reader import ContextReader
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from server.mcp_resources import MEMORY_TYPES
from server.mcp_tools import query_memory
from service.api import ReadAPI


QUERY = "Ben 的部署命令"


@pytest.fixture
def memory_query():
    ctx = RequestContext(
        "acct-1", "u1", "a1", "s1", "t1",
        visible_owner_spaces=("user:u1", "agent:a1", "agent:a2"),
    )
    cfg = RetrievalConfig(expand_hierarchy=False, hotness_alpha=0, owner_scope_filter_enabled=True)
    embedder = SimpleNamespace(embed_texts=Mock(return_value=[[1.0, 0.0]]))
    index = InMemoryVectorIndex(dimension=2)
    nodes, records = {}, []
    for name, spec in MEMORY_TYPES.items():
        owners = ("user:u1", "agent:a1", "agent:a2") if name == "pattern" else (
            "agent:a1" if spec.owner == "agent" else "user:u1",
        )
        for owner in owners:
            kind, owner_id = owner.split(":")
            root = f"ctx://acct-1/{kind}s/{owner_id}"
            if name == "profile":
                uri = root + "/memories/profile"
            elif name == "skill":
                uri = root + "/skills/generated/deploy"
            else:
                uri = f"{root}/memories/{spec.directory}/{name}_{owner_id}"
            context_type = "SKILL" if spec.internal_category in {"skill", "tool", "pattern"} else "MEMORY"
            nodes[uri] = ContextNode(
                uri=uri, context_type=context_type, category=spec.internal_category,
                level=2, owner_space=owner, abstract=name, overview=name, content=QUERY,
            )
            records.append(IndexRecord(
                id=uri, uri=uri + "/content.md", level=2, text=QUERY,
                filters={"account_id": ctx.account_id, "owner_space": owner},
                metadata={"context_type": context_type, "category": spec.internal_category,
                          "_embedding": [1.0, 0.0]},
            ))
    index.upsert(records)
    index.search_by_vector = Mock(wraps=index.search_by_vector)
    reader = ContextReader(SimpleNamespace(read_node=lambda uri, ctx: nodes[uri]), storage_backend="sql")
    pipeline = RetrievalPipeline(
        QueryPlanner(cfg), SeedRetriever(index, embedder, cfg), None,
        ResultRanker(cfg), cfg, context_reader=reader,
    )
    api = ReadAPI(pipeline, reader, cfg)
    api.search_memory = Mock(wraps=api.search_memory)
    service = SimpleNamespace(_cfg=SimpleNamespace(), get_read_api=lambda: api)
    return SimpleNamespace(service=service, ctx=ctx, embedder=embedder, index=index, api=api, nodes=nodes)


def test_all_memory_hint_preserves_custom_types_and_legacy_patterns(memory_query):
    h = memory_query
    uri = "ctx://acct-1/users/u1/memories/custom/atlas"
    h.nodes[uri] = ContextNode(uri=uri, context_type="MEMORY", category="custom", level=2,
                               owner_space="user:u1", abstract="Atlas", overview="Atlas",
                               content="Atlas 当前发布命令")
    h.index.upsert([IndexRecord(id=uri, uri=uri + "/content.md", level=2, text=QUERY,
        filters={"account_id": h.ctx.account_id, "owner_space": "user:u1"},
        metadata={"context_type": "MEMORY", "category": "custom", "_embedding": [1.0, 0.0]})])

    result = h.api.search_memory(QUERY, h.ctx, top_k=20, hints={"include_all_memories": True})

    assert {hit.category for hit in result.hits} == {
        spec.internal_category for spec in MEMORY_TYPES.values()
    } | {"custom"}
    assert len(result.hits) == 15
    assert {hit.owner_space for hit in result.hits if hit.category == "pattern"} == {
        "user:u1", "agent:a1", "agent:a2",
    }
    h.embedder.embed_texts.assert_called_once_with([QUERY])


@pytest.mark.parametrize("skill_sedimentation", [False, True])
def test_compose_and_prefetch_keep_current_facts_with_historical_sources(memory_query, skill_sedimentation):
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    from session.models import SessionWindowState
    from session.session_manager import SessionManager

    h = memory_query
    service = MemoryService(config=OgMemConfig(prefetch_enabled=True, topic_detection_enabled=False,
        rolling_compress_enabled=False, session_state_bridge_enabled=False))
    service._cfg.skill_sedimentation_enabled = skill_sedimentation
    service._read_api = h.api
    service._session_mgr = SessionManager(session_state_bridge_enabled=False)
    service._get_context_fs = Mock(return_value=None)
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    current_session = "当前会话：核实 Atlas 的发布安排"
    service._get_session_state = Mock(return_value=SessionWindowState(compressed_text=current_session))
    for node in h.nodes.values():
        if node.category in {"entity", "preference"}:
            node.content = "Atlas 当前：周四北京时间21:00，包管理工具uv。"
        elif node.category == "session_summary":
            node.content = "历史会话：Atlas 周四北京时间20:00发布。"
            node.metadata = {"source_session_id": "initial-session", "when": "2026-10-29"}
    params = {"_ctx": h.ctx, "sessionId": h.ctx.session_id, "query": QUERY, "prefetchTopK": 20}

    staged = service.prefetch(params)
    assert staged["prefetched"] == 14
    response = service.compose(params)

    assert response["sessionContext"] == "## Recent Session Summary\n" + current_session
    evidence = response["retrievedEvidence"]
    assert "Atlas 当前：周四北京时间21:00" in evidence
    assert "历史会话：Atlas 周四北京时间20:00" in evidence
    assert "source_session_id: initial-session" in evidence
    assert "when: 2026-10-29" in evidence
    assert evidence.count("历史会话：Atlas 周四北京时间20:00") == 1


@pytest.mark.parametrize("options", [{}, {"types": None}, {"types": []}])
def test_default_query_recalls_all_public_types_with_one_embedding(memory_query, options):
    h = memory_query
    result = query_memory(h.service, query=QUERY, top_k=20, request_context=h.ctx, **options)

    assert {hit["type"] for hit in result["hits"]} == set(MEMORY_TYPES)
    assert len(result["hits"]) == 14
    h.embedder.embed_texts.assert_called_once_with([QUERY])
    assert {call.kwargs["filters"]["account_id"] for call in h.index.search_by_vector.call_args_list} == {"acct-1"}
    assert [call.kwargs["filters"]["context_type"] for call in h.index.search_by_vector.call_args_list] == [
        "MEMORY", "SKILL", "SKILL",
    ]


@pytest.mark.parametrize("memory_type", list(MEMORY_TYPES))
def test_explicit_public_type_uses_its_directory(memory_query, memory_type):
    h = memory_query
    result = query_memory(
        h.service, query=QUERY, types=[memory_type, memory_type], top_k=20, request_context=h.ctx,
    )

    assert {hit["type"] for hit in result["hits"]} == {memory_type}
    assert len(result["hits"]) == (3 if memory_type == "pattern" else 1)
    h.embedder.embed_texts.assert_called_once_with([QUERY])
    if memory_type in {"tool_memory", "tool_evidence"}:
        assert all(call.kwargs["filters"]["memory_directory"] == [MEMORY_TYPES[memory_type].directory]
                   for call in h.index.search_by_vector.call_args_list)


@pytest.mark.parametrize("visible", [("user:u1",), ("user:u1", "agent:a1", "agent:a2"), ()])
def test_pattern_query_uses_authorized_user_and_agent_scopes(memory_query, visible):
    h = memory_query
    ctx = replace(h.ctx, visible_owner_spaces=visible)
    result = query_memory(h.service, query=QUERY, types=["pattern"], top_k=20, request_context=ctx)
    owners = visible or ("user:u1", "agent:a1")

    assert {hit["id"] for hit in result["hits"]} == {
        "pattern_" + owner.split(":")[1] for owner in owners
    }
    skill_filter = h.index.search_by_vector.call_args_list[1].kwargs["filters"]
    assert set(skill_filter["owner_space"]) == {
        alias for owner in owners for alias in (owner, owner.replace(":", "_", 1))
    }


def test_embedding_is_shared_within_each_query_request(memory_query):
    h = memory_query
    for _ in range(2):
        result = query_memory(h.service, query=QUERY, top_k=3, request_context=h.ctx)
        assert len(result["hits"]) == 3
    assert h.embedder.embed_texts.call_count == 2


def test_temporal_query_keeps_candidate_budget_and_final_limit(memory_query):
    h = memory_query
    result = query_memory(
        h.service, query="When did Ben use the deploy command?", top_k=3, request_context=h.ctx,
    )

    assert h.api.search_memory.call_args.kwargs["top_k"] == 15
    assert h.api.search_memory.call_args.kwargs["final_top_k"] == 3
    assert len(result["hits"]) == 3
    assert h.embedder.embed_texts.call_count == 1


def test_lexical_query_covers_public_types_with_zero_embedding_calls(memory_query):
    h = memory_query
    seed = h.api._pipeline.seed_retriever
    seed.config.hybrid_enabled = True
    seed.config.semantic_weight = 0

    def lexical_search(query, top_k, filters):
        return [LexicalHit(
            record.id, 1.0,
            {**record.metadata, **record.filters, "uri": record.uri, "level": record.level},
        ) for record in h.index._records.values()
            if h.index._matches_raw_filters(record, filters)][:top_k]

    seed.bm25_index = SimpleNamespace(search=Mock(side_effect=lexical_search))
    seed.lexical_backfill = SimpleNamespace(ensure_ready=lambda ctx, owners: True)

    result = query_memory(h.service, query=QUERY, top_k=20, request_context=h.ctx)

    assert {hit["type"] for hit in result["hits"]} == set(MEMORY_TYPES)
    assert len(result["hits"]) == 14
    assert seed.bm25_index.search.call_count == 3
    assert h.embedder.embed_texts.call_count == h.index.search_by_vector.call_count == 0
