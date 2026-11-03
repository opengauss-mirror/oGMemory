"""Disabling hierarchy searches L2 through every public/service query entry."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.models import ContextNode, IndexRecord, RequestContext, RetrievalConfig
from providers.embedder.mock_embedder import MockEmbedder
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from retrieval.context_reader import ContextReader
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.pipeline import RetrievalPipeline
from retrieval.query_planner import QueryPlanner
from retrieval.result_ranker import ResultRanker
from retrieval.seed_retriever import SeedRetriever
from server.mcp_server import create_mcp_server
from server.mcp_tools import query_memory, read_memory
from server.memory_service import MemoryService
from service.api import ReadAPI
from session.session_manager import SessionManager


ROOT = "ctx://acct-1/users/u1/memories/"
QUERY = "coding preferences"
LAYERS = [
    (ROOT + "preferences/abstract_only", 0, "Abstract evidence"),
    (ROOT + "preferences/overview_only", 1, "Overview evidence"),
    (ROOT + "preferences/content_only/content.md", 2, "Content evidence"),
]


class _OwnNodeFS:
    """Only the fixture's own nodes exist; field reads never discover children."""

    def __init__(self, records):
        self.nodes = {}
        self.field_reads = []
        for record in records:
            uri = record.uri.removesuffix("/content.md")
            label = "" if record.metadata["category"] == "preference" else record.metadata["category"] + " "
            self.nodes[uri] = ContextNode(
                uri=uri,
                context_type=record.metadata["context_type"],
                category=record.metadata["category"],
                level=record.level,
                owner_space=record.filters["owner_space"],
                abstract=f"Full {label}abstract evidence",
                overview=f"Full {label}overview evidence",
                content=f"Full {label}content evidence",
            )

    def read_node(self, uri, ctx):
        assert ctx.account_id == "acct-1"
        return self.nodes[uri]

    def read_node_fields(self, uri, ctx, level):
        self.field_reads.append((uri, level))
        node = self.read_node(uri, ctx)
        return ContextNode(
            uri=uri, context_type=node.context_type, category=node.category,
            level=node.level, owner_space=node.owner_space,
            abstract=node.abstract if level in {"abstract", "all"} else "",
            overview=node.overview if level in {"overview", "all"} else "",
            content=node.content if level in {"content", "all"} else "",
        )


@pytest.fixture
def harness(monkeypatch):
    # Background skill extraction needs external stores; query filtering does not.
    monkeypatch.setattr(MemoryService, "_start_skill_sedimentation_worker", lambda self: None)
    def build(layers=LAYERS, *, extra=(), skill_sedimentation=False, expand=False, query=QUERY):
        ctx = RequestContext("acct-1", "u1", "a1", "s1", "t1")
        cfg = RetrievalConfig(
            **({"expand_hierarchy": expand} if expand is not None else {}),
            hotness_alpha=0,
            default_score_threshold=-2,
        )
        embedder = MockEmbedder(dimension=8)
        # Equal positive vectors make every fixture hit deterministically visible.
        vector = embedder.embed_texts([query])[0]
        records = []
        for uri, level, text, category in [
            *((uri, level, text, "preference") for uri, level, text in layers),
            *extra,
        ]:
            context_type = "SKILL" if category in {"tool", "pattern"} else "MEMORY"
            owner_space = "agent:a1" if "/agents/" in uri else "user:u1"
            records.append(IndexRecord(
                id=f"{uri}:{level}", uri=uri, level=level, text=text,
                filters={"account_id": "acct-1", "owner_space": owner_space},
                metadata={"context_type": context_type, "category": category,
                          "parent_uri": ROOT, "_embedding": vector},
            ))
        index = InMemoryVectorIndex(dimension=8)
        index.upsert(records)
        index.search_by_vector = Mock(wraps=index.search_by_vector)
        index.search_children = Mock(wraps=index.search_children)
        searcher = HierarchicalSearcher(index, cfg)
        searcher.expand = Mock(wraps=searcher.expand)
        fs = _OwnNodeFS(records)
        reader = ContextReader(fs)
        pipeline = RetrievalPipeline(
            QueryPlanner(cfg), SeedRetriever(index, embedder, cfg),
            searcher, ResultRanker(cfg), cfg, context_reader=reader,
        )
        api = ReadAPI(pipeline, reader, cfg)
        service = MemoryService(config=OgMemConfig(
            account_id="acct-1", user_id="u1", agent_id="a1",
            prefetch_enabled=True, skill_sedimentation_enabled=skill_sedimentation,
            **({"retrieval_expand_hierarchy": expand} if expand is not None else {}),
            workers=1,
            session_state_bridge_enabled=False, rolling_compress_enabled=False,
            topic_detection_enabled=False,
        ))
        service._read_api = api
        service._session_mgr = SessionManager(session_state_bridge_enabled=False)
        monkeypatch.setattr(service, "_start_outbox_worker", lambda: None)
        monkeypatch.setattr(service, "_get_context_fs", lambda: fs)
        monkeypatch.setattr(service, "_read_profile", lambda ctx: "")
        monkeypatch.setattr(service, "_collect_archives", lambda ctx, budget: ([], []))
        monkeypatch.setattr(service, "build_context", lambda params, identity=None: ctx)
        monkeypatch.setattr(service, "get_auth_service", lambda: SimpleNamespace(
            role_control_active=lambda: False))
        monkeypatch.setattr(service, "record_retrieval_metric", lambda *args, **kwargs: None)
        return SimpleNamespace(service=service, api=api, pipeline=pipeline, ctx=ctx,
                               searcher=searcher, index=index, fs=fs)
    return build


def _query(entrypoint, h, monkeypatch, query=QUERY):
    if entrypoint == "pipeline":
        return h.pipeline.run(query, h.ctx, top_k=10).hits
    if entrypoint == "read_api":
        return h.api.search_memory(query, h.ctx, top_k=10).hits
    if entrypoint == "native_tool":
        return query_memory(h.service, query=query, top_k=10,
                            request_context=h.ctx)["hits"]
    if entrypoint == "mcp":
        pytest.importorskip("mcp", reason="MCP SDK is an optional transport dependency")
        mcp = create_mcp_server(service=h.service, config=h.service._cfg)
        tool = mcp._tool_manager.get_tool("ogm_query_memory")
        return tool.fn(query=query, top_k=10)["hits"]
    import server.app as app_module
    monkeypatch.setattr(app_module, "_service", h.service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: h.ctx)
    response = app_module.app.test_client().post(
        "/api/v1/memories/query", json={"query": query, "top_k": 10},
    )
    assert response.status_code == 200
    return response.json["hits"]


@pytest.mark.parametrize("entrypoint", ["pipeline", "read_api", "native_tool", "mcp", "http"])
@pytest.mark.parametrize("layers", [LAYERS, [], LAYERS[:1]], ids=["mixed", "empty", "l0_only"])
def test_query_entrypoints_search_l2_only(harness, monkeypatch, entrypoint, layers):
    h = harness(layers=layers)
    hits = _query(entrypoint, h, monkeypatch)
    assert h.searcher.expand.call_count == h.index.search_children.call_count == 0
    assert h.index.search_by_vector.call_args_list
    assert all(call.kwargs["filters"].get("level") == 2 for call in h.index.search_by_vector.call_args_list)
    if layers != LAYERS:
        assert hits == []
    elif entrypoint in {"pipeline", "read_api"}:
        assert [(hit.uri, hit.level_hit, hit.abstract, hit.overview, hit.content_excerpt)
                for hit in hits] == [
            (ROOT + "preferences/content_only/content.md", "L2", "Content evidence", None, None),
        ]
    else:
        assert len(hits) == 1
        assert hits[0]["uri"] == "ogmemory://memories/preference/content_only"
        assert hits[0]["abstract"] == "Content evidence"
        assert hits[0]["overview"] == "Full overview evidence"
        assert hits[0]["content_excerpt"] == "Full content evidence"
        assert "matched_levels" not in hits[0]
        detail = read_memory(h.service, uri=hits[0]["uri"], request_context=h.ctx)
        assert detail["uri"] == hits[0]["uri"]
        assert detail["content"] == "Full content evidence"


@pytest.mark.parametrize("skill_sedimentation", [False, True])
def test_compose_uses_l2_summary_and_legacy_skill(harness, skill_sedimentation):
    h = harness(layers=[], skill_sedimentation=skill_sedimentation, extra=[
        (ROOT + "session_summaries/s1/content.md", 2, "Summary evidence", "session_summary"),
        (ROOT + "skills/review/content.md", 2, "Skill evidence", "skill"),
        (ROOT + "preferences/summary_only", 0, "Summary-only evidence", "preference"),
    ])
    response = h.service.compose({"_ctx": h.ctx, "prompt": QUERY})
    assert "Full session_summary content evidence" in response["sessionContext"]
    assert "Full skill overview evidence" in response["retrievedEvidence"]
    assert "Summary-only evidence" not in response["retrievedEvidence"]
    assert h.searcher.expand.call_count == h.index.search_children.call_count == 0


@pytest.mark.parametrize("skill_sedimentation", [False, True])
@pytest.mark.parametrize("layers", [[], LAYERS[:1]], ids=["empty", "l0_only"])
def test_compose_without_l2_has_no_retrieved_evidence(harness, skill_sedimentation, layers):
    h = harness(layers=layers, skill_sedimentation=skill_sedimentation)
    response = h.service.compose({"_ctx": h.ctx, "prompt": QUERY})
    assert response["retrievedEvidence"] == ""
    assert "Session Summary" not in response["sessionContext"]
    assert h.searcher.expand.call_count == h.index.search_children.call_count == 0


@pytest.mark.parametrize("skill_sedimentation", [False, True])
@pytest.mark.parametrize("layers", [LAYERS, [], LAYERS[:1]], ids=["mixed", "empty", "l0_only"])
def test_prefetch_stages_only_l2_using_existing_staging_contract(harness, skill_sedimentation, layers):
    h = harness(layers=layers, skill_sedimentation=skill_sedimentation)
    response = h.service.prefetch({"_ctx": h.ctx, "prompt": QUERY, "prefetchTopK": 10})
    pending = h.service.get_session_manager().get_topic_buffer("s1").get_pending_injection()
    assert response == {"ok": True, "prefetched": 1 if layers == LAYERS else 0}
    assert h.searcher.expand.call_count == h.index.search_children.call_count == 0
    if layers != LAYERS:
        assert pending == []
    else:
        assert [(hit.uri, hit.level, hit.abstract, hit.metadata) for hit in pending] == [
            (ROOT + "preferences/content_only/content.md", 2, "Content evidence", {"source": "prefetch"}),
        ]


@pytest.mark.parametrize("expand", [True, None], ids=["explicit", "default"])
def test_enabled_control_really_traverses_hierarchy(harness, expand):
    h = harness(expand=expand)
    h.api.search_memory(QUERY, h.ctx)
    assert h.searcher.expand.call_count > 0
    assert h.index.search_children.call_count > 0


def test_enabled_prefetch_keeps_legacy_staging_metadata(harness):
    h = harness(expand=True)
    response = h.service.prefetch({"_ctx": h.ctx, "prompt": QUERY})
    pending = h.service.get_session_manager().get_topic_buffer("s1").get_pending_injection()
    assert response["prefetched"] > 0
    assert all(hit.level == 2 and hit.metadata == {"source": "prefetch"} for hit in pending)


@pytest.mark.parametrize("entrypoint", ["pipeline", "read_api", "native_tool", "mcp", "http"])
def test_l2_tool_and_pattern_with_skill_metadata_keep_existing_public_uris(harness, monkeypatch, entrypoint):
    query = "how to use grep"
    agent_root = "ctx://acct-1/agents/a1/memories/"
    h = harness(layers=[], query=query, extra=[
        (agent_root + "tools/grep/content.md", 2, "Tool evidence", "tool"),
        (agent_root + "patterns/search/content.md", 2, "Pattern evidence", "pattern"),
        (agent_root + "tools/", 0, "Tool directory summary", "tool"),
    ])
    hits = _query(entrypoint, h, monkeypatch, query=query)
    assert h.searcher.expand.call_count == h.index.search_children.call_count == 0
    if entrypoint in {"pipeline", "read_api"}:
        assert {(hit.uri, hit.level_hit, hit.metadata["context_type"]) for hit in hits} == {
            (agent_root + "tools/grep/content.md", "L2", "SKILL"),
            (agent_root + "patterns/search/content.md", "L2", "SKILL"),
        }
    else:
        rows = {hit["uri"]: hit for hit in hits}
        assert set(rows) == {"ogmemory://memories/tool_memory/grep", "ogmemory://memories/pattern/agent/search"}
        for uri, category in [("ogmemory://memories/tool_memory/grep", "tool"),
                              ("ogmemory://memories/pattern/agent/search", "pattern")]:
            assert rows[uri]["content_excerpt"] == f"Full {category} content evidence"
            assert "matched_levels" not in rows[uri]
            detail = read_memory(h.service, uri=uri, request_context=h.ctx)
            assert detail["uri"] == uri
            assert detail["content"] == f"Full {category} content evidence"
