from __future__ import annotations

import re
import threading
import time
from types import SimpleNamespace

import pytest

import retrieval.code_retrieval as code_retrieval_module
from core.errors import NodeNotFoundError, ValidationError
from core.models import (
    ContextNode,
    RequestContext,
    RetrievalConfig,
    RetrievedBlock,
    SearchMemoryResult,
    SeedHit,
    SeedResult,
    TypedQuery,
)
from retrieval.bm25_index import Bm25Index
from retrieval.code_retrieval import (
    ActiveCodeSearchDiagnostics,
    ActiveCodeSearchOutcome,
    ActiveCodeSearchRequest,
    CodeEvidence,
    CodeRetrievalService,
    CodeSearchBudget,
    CodeSearchScope,
    _extract_code_query_terms,
    format_code_evidence_entries,
    fuse_route_hits,
    weighted_rrf_fuse,
)
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.owner_scope import allowed_owner_space_filter
from retrieval.query_planner import QueryPlanner
from retrieval.seed_retriever import SeedRetriever
from service.api import ReadAPI


def _ctx() -> RequestContext:
    return RequestContext(
        account_id="acct",
        user_id="user",
        agent_id="agent",
        session_id="session",
        trace_id="trace",
        visible_owner_spaces=("user:user", "agent:agent"),
    )


def _code_query() -> TypedQuery:
    return TypedQuery(
        text="find parse_config",
        context_type="RESOURCE",
        categories=[],
        top_k=8,
        account_id="acct",
        owner_space=None,
        resource_type="code",
    )


def test_planner_resource_override_is_account_wide_and_code_filtered():
    planner = QueryPlanner(RetrievalConfig())

    query = planner.plan(
        "find parse_config",
        _ctx(),
        context_type="RESOURCE",
        resource_type="code",
        top_k=8,
    )[0]

    assert query.context_type == "RESOURCE"
    assert query.resource_type == "code"
    assert query.owner_space is None
    assert query.account_id == "acct"


def test_planner_rejects_resource_type_without_resource_context():
    planner = QueryPlanner(RetrievalConfig())

    with pytest.raises(ValidationError, match="resource_type requires"):
        planner.plan(
            "find parse_config",
            _ctx(),
            context_type="MEMORY",
            resource_type="code",
        )


def test_resource_owner_filter_stays_empty_when_visible_scope_enforcement_is_enabled():
    assert (
        allowed_owner_space_filter(
            _code_query(),
            _ctx(),
            enforce_visible_scope=True,
        )
        is None
    )

    ordinary_resource_query = TypedQuery(
        text="find a document",
        context_type="RESOURCE",
        categories=[],
        top_k=8,
        account_id="acct",
        owner_space=["user:user", "agent:agent"],
    )
    ordinary_scope = allowed_owner_space_filter(
        ordinary_resource_query,
        _ctx(),
        enforce_visible_scope=True,
    )
    assert set(ordinary_scope or []) == {
        "user:user",
        "user_user",
        "agent:agent",
        "agent_agent",
    }

    planned = QueryPlanner(RetrievalConfig()).plan(
        "find a document",
        _ctx(),
        context_type="RESOURCE",
    )[0]
    assert planned.owner_space == ["user:user", "agent:agent"]


def test_read_api_propagates_internal_resource_overrides():
    captured = {}

    class Pipeline:
        def run(self, query, ctx, **kwargs):
            captured.update(kwargs)
            return SearchMemoryResult(query=query)

    api = ReadAPI(Pipeline(), config=RetrievalConfig())

    api.search_memory(
        "find parse_config",
        _ctx(),
        context_type="RESOURCE",
        resource_type="code",
    )

    assert captured["context_type"] == "RESOURCE"
    assert captured["resource_type"] == "code"


def test_seed_vector_and_bm25_filters_keep_resource_type_without_owner_scope():
    captured = {"vector": None, "bm25": None}

    class Embedder:
        def embed_texts(self, texts):
            assert texts == ["find parse_config"]
            return [[0.1, 0.2]]

    class VectorIndex:
        def search_by_vector(self, *, query_vector, filters, top_k):
            del query_vector, top_k
            captured["vector"] = dict(filters)
            return []

    class Bm25:
        doc_count = 1

        def search(self, *, query, top_k, filters):
            del query, top_k
            captured["bm25"] = dict(filters)
            return []

    retriever = SeedRetriever(
        VectorIndex(),
        Embedder(),
        RetrievalConfig(owner_scope_filter_enabled=True),
        bm25_index=Bm25(),
    )

    retriever.search(_code_query(), _ctx())

    expected = {
        "level": [0, 1, 2],
        "account_id": "acct",
        "context_type": "RESOURCE",
        "resource_type": "code",
    }
    assert captured["vector"] == expected
    assert captured["bm25"] == {key: value for key, value in expected.items() if key != "level"}


def test_hierarchical_child_search_keeps_code_filter_without_owner_scope():
    captured = {}

    class VectorIndex:
        def search_children(self, *, parent_uri, query_vector, filters, top_k):
            del parent_uri, query_vector, top_k
            captured.update(filters)
            return []

    searcher = HierarchicalSearcher(
        VectorIndex(),
        RetrievalConfig(owner_scope_filter_enabled=True),
    )
    seeds = SeedResult(
        starting_points=[SeedHit(uri="ctx://acct/resources/", score=0.0, level=0)],
        query_vector=[0.1],
    )

    searcher.expand(_code_query(), seeds, _ctx(), limit=8)

    assert captured == {
        "account_id": "acct",
        "context_type": "RESOURCE",
        "resource_type": "code",
    }


def test_bm25_can_include_zero_score_code_candidates_for_rrf(monkeypatch):
    index = Bm25Index()
    index._ids = ["a", "b"]
    index._texts = ["alpha", "beta"]
    index._metas = [
        {"resource_type": "code"},
        {"resource_type": "code"},
    ]
    index._bm25 = SimpleNamespace(get_scores=lambda tokens: [0.0, -0.25])
    monkeypatch.setattr("retrieval.bm25_index._tokenize", lambda text: [text])

    default_hits = index.search("query", filters={"resource_type": "code"})
    route_hits = index.search(
        "query",
        filters={"resource_type": "code"},
        include_nonpositive_matches=True,
    )

    assert default_hits == []
    assert [hit.id for hit in route_hits] == ["a", "b"]


def test_code_bm25_uses_shared_index_and_scope_replacement(monkeypatch):
    monkeypatch.setattr(
        "retrieval.bm25_index._tokenize",
        lambda text: re.findall(r"[a-z0-9]+", text.lower()),
    )

    def code_node(slug, symbol):
        return ContextNode(
            uri=f"ctx://acct/resources/{slug}",
            context_type="RESOURCE",
            category="code",
            level=4,
            owner_space="resource:acct",
            abstract=f"function {symbol}",
            overview=f"path: src/{slug}.py\nsignature: {symbol}(payload)",
            content=f"def {symbol}(payload):\n    return payload",
            metadata={
                "resource_type": "code",
                "resource_root_slug": "repo",
                "relative_path": f"src/{slug}.py",
                "source_extension": ".py",
                "language": "python",
                "start_line": 1,
                "end_line": 2,
                "symbol": symbol,
                "symbol_kind": "function",
                "signature": f"{symbol}(payload)",
                "chunk_hash": slug,
                "chunk_key": slug * 64,
            },
        )

    nodes = {
        node.uri: node
        for node in (
            code_node("config", "parse_config"),
            code_node("loader", "load_settings"),
            code_node("writer", "write_settings"),
        )
    }

    class ContextFS:
        def list_children(self, uri, ctx):
            assert uri == "ctx://acct/resources/"
            assert ctx.account_id == "acct"
            return list(nodes)

        def read_node(self, uri, ctx):
            assert ctx.account_id == "acct"
            return nodes[uri]

    shared_bm25 = Bm25Index()
    shared_bm25.upsert(
        "other-account-code",
        "isolated marker",
        {
            "account_id": "other",
            "context_type": "RESOURCE",
            "resource_type": "code",
            "uri": "ctx://other/resources/isolated",
        },
    )
    service = CodeRetrievalService(
        context_fs=ContextFS(),
        read_api=SimpleNamespace(),
        bm25_index=shared_bm25,
        config=SimpleNamespace(code_retrieval_enabled=True),
    )

    hits = service._search_bm25("parse_config", _ctx(), top_k=8)

    assert hits[0]["uri"] == "ctx://acct/resources/config"
    assert hits[0]["raw_score"] > 0
    assert shared_bm25.doc_count == 7
    assert [
        hit.id
        for hit in shared_bm25.search(
            "isolated marker",
            filters={"account_id": "other", "resource_type": "code"},
        )
    ] == ["other-account-code"]
    assert not hasattr(service._cache["acct"], "bm25")

    del nodes["ctx://acct/resources/config"]
    nodes["ctx://acct/resources/config-v2"] = code_node(
        "config-v2",
        "parse_configuration_v2",
    )
    service.invalidate("acct")

    refreshed = service._search_bm25("parse_configuration_v2", _ctx(), top_k=8)

    assert refreshed[0]["uri"] == "ctx://acct/resources/config-v2"
    assert all(item["uri"] != "ctx://acct/resources/config" for item in refreshed)
    assert shared_bm25.doc_count == 7


def test_snapshot_skips_foreign_child_uri_before_contextfs_read():
    local_uri = "ctx://acct/resources/local"
    local_node = ContextNode(
        uri=local_uri,
        context_type="RESOURCE",
        category="code",
        level=4,
        owner_space="resource:acct",
        abstract="local",
        overview="",
        content="",
        metadata={
            "resource_type": "code",
            "resource_root_slug": "repo",
            "relative_path": "src/local.py",
            "symbol": "local",
        },
    )
    reads: list[str] = []

    class ContextFS:
        def list_children(self, uri, ctx):
            assert uri == "ctx://acct/resources/"
            assert ctx.account_id == "acct"
            return ["ctx://other/resources/foreign", local_uri]

        def read_node(self, uri, ctx):
            del ctx
            reads.append(uri)
            assert uri == local_uri
            return local_node

    service = CodeRetrievalService(
        context_fs=ContextFS(),
        read_api=SimpleNamespace(),
        bm25_index=SimpleNamespace(),
        config=SimpleNamespace(code_retrieval_enabled=True),
    )

    snapshot = service._get_snapshot(_ctx())

    assert reads == [local_uri]
    assert [record["uri"] for record in snapshot.records] == [local_uri]


def test_active_snapshot_excludes_category_code_node_without_code_resource_type():
    accepted_uri = "ctx://acct/resources/accepted"
    mislabeled_uri = "ctx://acct/resources/mislabeled"
    accepted = ContextNode(
        uri=accepted_uri,
        context_type="RESOURCE",
        category="code",
        level=4,
        owner_space="resource:acct",
        abstract="accepted",
        overview="",
        content="",
        metadata={
            "resource_type": "code",
            "resource_root_slug": "repo",
            "relative_path": "src/accepted.py",
            "symbol": "accepted",
        },
    )
    mislabeled = ContextNode(
        uri=mislabeled_uri,
        context_type="RESOURCE",
        category="code",
        level=4,
        owner_space="resource:acct",
        abstract="mislabeled",
        overview="",
        content="",
        metadata={
            "resource_type": "document",
            "resource_root_slug": "repo",
            "relative_path": "src/mislabeled.py",
            "symbol": "mislabeled",
        },
    )
    nodes = {accepted_uri: accepted, mislabeled_uri: mislabeled}

    class ContextFS:
        def list_children(self, uri, ctx):
            assert uri == "ctx://acct/resources/"
            assert ctx.account_id == "acct"
            return list(nodes)

        def read_node(self, uri, ctx):
            assert ctx.account_id == "acct"
            return nodes[uri]

    service = CodeRetrievalService(
        context_fs=ContextFS(),
        read_api=SimpleNamespace(),
        bm25_index=SimpleNamespace(),
        config=SimpleNamespace(code_retrieval_enabled=True),
    )

    snapshot = service._get_snapshot(_ctx())

    assert [record["uri"] for record in snapshot.records] == [accepted_uri]
    assert mislabeled_uri not in snapshot.bm25_ids_by_uri


def test_weighted_rrf_merges_content_and_node_uri_and_preserves_raw_scores():
    node_uri = "ctx://acct/resources/chunk"
    fused = weighted_rrf_fuse(
        {
            "semantic": [
                {
                    "uri": node_uri + "/content.md",
                    "parent_uri": node_uri,
                    "score": 0.91,
                    "raw_score": 0.91,
                    "metadata": {"resource_type": "code"},
                }
            ],
            "bm25": [
                {
                    "uri": node_uri,
                    "score": 0.4,
                    "raw_score": 7.5,
                    "metadata": {"_bm25_raw_score": 7.5},
                }
            ],
        },
        limit=8,
    )

    assert len(fused) == 1
    assert fused[0]["uri"] == node_uri
    assert set(fused[0]["original_uris"]) == {node_uri, node_uri + "/content.md"}
    assert fused[0]["score"] == pytest.approx((0.33 / 61) + (0.17 / 61))
    assert fused[0]["metadata"]["route_scores"] == {
        "semantic": {"raw": 0.91, "normalized": 0.91},
        "bm25": {"raw": 7.5, "normalized": 0.4},
    }


def test_weighted_rrf_uses_canonical_uri_as_deterministic_tie_break():
    fused = weighted_rrf_fuse(
        {
            "semantic": [
                {"uri": "ctx://acct/resources/b", "score": 0.8},
                {"uri": "ctx://acct/resources/a", "score": 0.8},
            ],
            "graph": [
                {"uri": "ctx://acct/resources/a", "score": 0.2},
                {"uri": "ctx://acct/resources/b", "score": 0.2},
            ],
        },
        limit=8,
    )

    assert fused[0]["score"] == fused[1]["score"]
    assert [item["uri"] for item in fused] == [
        "ctx://acct/resources/a",
        "ctx://acct/resources/b",
    ]


def test_evidence_locators_distinguish_duplicate_paths_across_roots():
    entries = format_code_evidence_entries(
        [
            {
                "metadata": {
                    "resource_root_slug": "alpha-root",
                    "relative_path": "src/foo.py",
                    "start_line": 1,
                    "end_line": 3,
                },
            },
            {
                "metadata": {
                    "resource_root_slug": "beta-root",
                    "relative_path": "src/foo.py",
                    "start_line": 7,
                    "end_line": 9,
                },
            },
        ],
        body_budget_chars=0,
    )

    assert entries == [
        "- [code] alpha-root/src/foo.py:1-3",
        "- [code] beta-root/src/foo.py:7-9",
    ]


def test_evidence_budget_never_truncates_root_qualified_code_locator():
    entries = format_code_evidence_entries(
        [
            {
                "content": "body" * 100,
                "metadata": {
                    "resource_root_slug": "alpha-root",
                    "relative_path": "src/package/really_long_module.py",
                    "start_line": 101,
                    "end_line": 144,
                    "symbol": "parse_configuration",
                    "language": "python",
                },
            }
        ],
        body_budget_chars=1,
    )

    assert entries == [
        "- [code] alpha-root/src/package/really_long_module.py:101-144 | "
        "parse_configuration | python"
    ]


def test_evidence_legacy_item_uses_safe_relative_path_fallback():
    entries = format_code_evidence_entries(
        [
            {
                "metadata": {
                    "relative_path": "legacy/module.py",
                    "start_line": 4,
                    "end_line": 6,
                }
            }
        ],
        body_budget_chars=0,
    )

    assert entries == ["- [code] legacy/module.py:4-6"]


@pytest.mark.parametrize(
    "unsafe_root_slug",
    [
        "",
        ".",
        "..",
        "../alpha-root",
        "/srv/alpha-root",
        r"C:\\private\\alpha-root",
        "file://alpha-root",
        "alpha/root",
    ],
)
def test_evidence_item_with_invalid_root_slug_fails_closed(unsafe_root_slug):
    entries = format_code_evidence_entries(
        [
            {
                "metadata": {
                    "resource_root_slug": unsafe_root_slug,
                    "relative_path": "legacy/module.py",
                    "start_line": 4,
                    "end_line": 6,
                }
            }
        ],
        body_budget_chars=0,
    )

    assert entries == ["- [code] <unknown>:4-6"]
    if unsafe_root_slug:
        assert unsafe_root_slug not in entries[0]


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/srv/private/repo.py",
        r"C:\\private\\repo.py",
        r"C:private\\repo.py",
        "file:///srv/private/repo.py",
        "https://internal.example/private/repo.py",
        "../private/repo.py",
    ],
)
def test_evidence_legacy_item_never_exposes_unsafe_path(unsafe_path):
    entries = format_code_evidence_entries(
        [
            {
                "metadata": {
                    "relative_path": unsafe_path,
                    "start_line": 4,
                    "end_line": 6,
                }
            }
        ],
        body_budget_chars=0,
    )

    assert entries == ["- [code] <unknown>:4-6"]
    assert unsafe_path not in entries[0]


def test_route_failure_degrades_only_that_route_and_hydration_uses_contextfs(monkeypatch):
    node_uri = "ctx://acct/resources/chunk"
    node = ContextNode(
        uri=node_uri,
        context_type="RESOURCE",
        category="code",
        level=2,
        owner_space="resource:acct",
        abstract="parse config",
        overview="parser overview",
        content="def parse_config():\n    return True",
        metadata={
            "resource_type": "code",
            "relative_path": "src/config.py",
            "start_line": 1,
            "end_line": 2,
            "symbol": "parse_config",
            "language": "python",
        },
    )

    class ContextFS:
        def read_node(self, uri, ctx):
            del ctx
            assert uri == node_uri
            return node

    class ReadAPI:
        def search_memory(self, **kwargs):
            del kwargs
            raise RuntimeError("semantic unavailable")

    config = SimpleNamespace(
        code_retrieval_enabled=True,
        code_retrieval_route_enabled=lambda route: route in {"semantic", "symbolic"},
    )
    service = CodeRetrievalService(
        context_fs=ContextFS(),
        read_api=ReadAPI(),
        bm25_index=Bm25Index(),
        config=config,
    )
    monkeypatch.setattr(
        service,
        "_get_snapshot",
        lambda ctx: SimpleNamespace(
            records=(
                {
                    "uri": node_uri,
                    "metadata": dict(node.metadata),
                },
            )
        ),
    )
    language_hits = service._search_symbolic("python", _ctx(), top_k=8)
    assert [item["uri"] for item in language_hits] == [node_uri]

    monkeypatch.setattr(
        service,
        "_search_symbolic",
        lambda query, ctx, top_k: [
            {
                "uri": node_uri,
                "score": 1.0,
                "raw_score": 1.0,
                "metadata": {"resource_type": "code"},
            }
        ],
    )

    items, stats = service.search("find parse_config", _ctx())

    assert len(items) == 1
    assert items[0]["content"].startswith("def parse_config")
    assert items[0]["metadata"]["relative_path"] == "src/config.py"
    assert stats["code_route_errors"] == {"semantic": "RuntimeError"}
    assert stats["code_retrieval_routes"] == ["symbolic"]


def test_legacy_search_projection_restores_full_persisted_body_without_rerunning_routes(monkeypatch):
    uri = "ctx://acct/resources/full-body"
    full_body = "x" * 1600
    node = ContextNode(
        uri=uri,
        context_type="RESOURCE",
        category="code",
        level=4,
        owner_space="resource:acct",
        abstract="full abstract",
        overview="full overview",
        content=full_body,
        metadata={
            "resource_type": "code",
            "resource_root_slug": "repo",
            "relative_path": "src/full.py",
        },
    )
    active_calls = []

    class ContextFS:
        def read_node(self, requested_uri, ctx):
            assert ctx.account_id == "acct"
            assert requested_uri == uri
            return node

    service = CodeRetrievalService(
        context_fs=ContextFS(),
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(code_retrieval_enabled=True),
    )
    outcome = ActiveCodeSearchOutcome(
        status="ok",
        evidence=(
            CodeEvidence(
                uri=uri,
                identity=uri,
                locator="repo/src/full.py",
                excerpt=full_body[:1200],
                persisted=True,
            ),
        ),
        diagnostics=ActiveCodeSearchDiagnostics(8, 8, 1200, 8000),
    )
    monkeypatch.setattr(
        service,
        "search_active",
        lambda request, ctx: (active_calls.append((request, ctx)) or outcome),
    )

    items, stats = service.search("full body", _ctx())

    assert len(active_calls) == 1
    assert items[0]["content"] == full_body
    assert items[0]["abstract"] == "full abstract"
    assert items[0]["overview"] == "full overview"
    assert stats["code_hydration_errors"] == 0


def test_legacy_search_projection_drops_unreadable_persisted_evidence(monkeypatch):
    uri = "ctx://acct/resources/unreadable"

    class ContextFS:
        def read_node(self, requested_uri, ctx):
            del requested_uri, ctx
            raise NodeNotFoundError("missing")

    service = CodeRetrievalService(
        context_fs=ContextFS(),
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(code_retrieval_enabled=True),
    )
    outcome = ActiveCodeSearchOutcome(
        status="ok",
        evidence=(
            CodeEvidence(
                uri=uri,
                identity=uri,
                locator="repo/src/missing.py",
                excerpt="bounded active excerpt",
                persisted=True,
            ),
        ),
        diagnostics=ActiveCodeSearchDiagnostics(8, 8, 1200, 8000),
    )
    monkeypatch.setattr(service, "search_active", lambda request, ctx: outcome)

    items, stats = service.search("missing", _ctx())

    assert items == []
    assert stats["code_hydration_errors"] == 1


def test_symbolic_query_terms_apply_length_stopword_and_dedup_rules():
    assert _extract_code_query_terms(
        "x xy Abc abc ABC _id _ID method FILE callable max_length"
    ) == ["xy", "Abc", "_id", "callable", "max_length"]


def test_symbolic_ranking_prioritizes_exact_identifier_and_survives_route_top_k(monkeypatch):
    def record(symbol, *, relative_path, signature):
        return {
            "uri": f"ctx://acct/resources/{symbol.lower()}",
            "metadata": {
                "resource_type": "code",
                "relative_path": relative_path,
                "source_extension": ".py",
                "language": "python",
                "symbol": symbol,
                "symbol_kind": "type",
                "signature": signature,
            },
        }

    records = (
        record(
            "In",
            relative_path="django/db/models/lookups.py",
            signature="class In",
        ),
        record(
            "Mod",
            relative_path="django/db/models/functions/math.py",
            signature="class Mod",
        ),
        record(
            "Length",
            relative_path="django/db/models/functions/text.py",
            signature="class Length",
        ),
        record(
            "Migration",
            relative_path="django/db/migrations/migration.py",
            signature="class Migration",
        ),
        record(
            "FilePathField",
            relative_path="django/db/models/fields/__init__.py",
            signature="class FilePathField",
        ),
        record(
            "Callable",
            relative_path="django/utils/callable.py",
            signature="class Callable",
        ),
        record(
            "Settings",
            relative_path="django/conf/settings.py",
            signature="class Settings",
        ),
        record(
            "Production",
            relative_path="deploy/production.py",
            signature="class Production",
        ),
        record(
            "Different",
            relative_path="docs/different.py",
            signature="class Different",
        ),
        record(
            "Machine",
            relative_path="runtime/machine.py",
            signature="class Machine",
        ),
        record(
            "Server",
            relative_path="runtime/server.py",
            signature="class Server",
        ),
        record(
            "Running",
            relative_path="commands/running.py",
            signature="class Running",
        ),
        record(
            "Resolve",
            relative_path="paths/resolve.py",
            signature="class Resolve",
        ),
        record(
            "Local",
            relative_path="files/local.py",
            signature="class Local",
        ),
    )
    service = CodeRetrievalService(
        context_fs=SimpleNamespace(),
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(code_retrieval_enabled=True),
    )
    monkeypatch.setattr(
        service,
        "_get_snapshot",
        lambda ctx: SimpleNamespace(records=records),
    )

    query = (
        "Allow FilePathField path to accept a callable. I have a special case where I want "
        "to create a model containing the path to some local files on the server machine. "
        "Seeing as the place where these files are stored is different on different machines, "
        "running makemigrations resolves the path on the local machine instead of production."
    )
    all_hits = service._search_symbolic(
        query,
        _ctx(),
        top_k=len(records),
    )
    top_hits = service._search_symbolic(query, _ctx(), top_k=8)

    raw_scores = {item["metadata"]["symbol"]: item["raw_score"] for item in all_hits}
    exact_hit = all_hits[0]
    assert len(all_hits) > 8
    assert len(top_hits) == 8
    assert exact_hit["metadata"]["symbol"] == "FilePathField"
    assert top_hits[0]["metadata"]["symbol"] == "FilePathField"
    assert all(
        raw_scores["FilePathField"] > raw_scores.get(symbol, 0.0)
        for symbol in ("In", "Mod", "Length", "Migration")
    )
    assert exact_hit["raw_score"] > 1.0
    assert exact_hit["score"] == 1.0
    assert exact_hit["score"] != exact_hit["raw_score"]

    fused = weighted_rrf_fuse({"symbolic": top_hits}, limit=8)
    assert fused[0]["metadata"]["route_scores"]["symbolic"] == {
        "raw": exact_hit["raw_score"],
        "normalized": 1.0,
    }


def test_hydration_failure_drops_candidate_instead_of_returning_empty_evidence(monkeypatch):
    class ContextFS:
        def list_children(self, uri, ctx):
            del uri, ctx
            return ["ctx://acct/resources/missing"]

        def read_node(self, uri, ctx):
            del uri, ctx
            raise NodeNotFoundError("missing")

    class ReadAPI:
        def search_memory(self, **kwargs):
            del kwargs
            return SearchMemoryResult(
                hits=[
                    RetrievedBlock(
                        uri="ctx://acct/resources/missing/content.md",
                        score=0.9,
                        category="code",
                        metadata={
                            "resource_type": "code",
                            "parent_uri": "ctx://acct/resources/missing",
                        },
                    )
                ]
            )

    config = SimpleNamespace(
        code_retrieval_enabled=True,
        code_retrieval_route_enabled=lambda route: route == "semantic",
    )
    service = CodeRetrievalService(
        context_fs=ContextFS(),
        read_api=ReadAPI(),
        bm25_index=Bm25Index(),
        config=config,
    )
    monkeypatch.setattr(
        service,
        "_get_snapshot",
        lambda ctx: SimpleNamespace(
            records=(
                {
                    "uri": "ctx://acct/resources/missing",
                    "metadata": {"resource_type": "code"},
                },
            ),
            read_error_count=0,
            bm25_ids_by_uri={},
        ),
    )

    items, stats = service.search("find missing", _ctx())

    assert items == []
    assert stats["code_hydration_errors"] == 1


def test_active_request_normalizes_and_rejects_unsafe_scope_values():
    request = ActiveCodeSearchRequest(
        "  parse_config  ",
        scope=CodeSearchScope(
            root_slugs=("repo",),
            paths=("src",),
            glob_patterns=("src/*.py",),
            grep_terms=("payload",),
        ),
        budget=CodeSearchBudget(
            candidate_limit=4,
            result_limit=2,
            snippet_chars=20,
            batch_chars=40,
        ),
    )

    assert request.query == "parse_config"
    assert request.scope.paths == ("src",)

    with pytest.raises(ValidationError):
        CodeSearchScope(paths=("../secret",))
    with pytest.raises(ValidationError):
        CodeSearchScope(paths=("",))
    with pytest.raises(ValidationError):
        ActiveCodeSearchRequest("query", scope={"unknown": "value"})


def test_active_fusion_modes_dedupe_chunk_uri_and_tie_break():
    route_hits = {
        "semantic": [
            {"uri": "ctx://acct/resources/node", "score": 1.0},
            {"uri": "ctx://acct/resources/other", "score": 1.0},
        ],
        "bm25": [
            {"uri": "ctx://acct/resources/node/content.md", "score": 0.5},
        ],
        "symbolic": [],
        "graph": [],
    }

    for mode in ("weighted_rrf", "weighted_sum", "sum"):
        fused = fuse_route_hits(route_hits, mode=mode, limit=4)
        assert [item["uri"] for item in fused] == [
            "ctx://acct/resources/node",
            "ctx://acct/resources/other",
        ]


def test_active_search_returns_bounded_safe_evidence_and_uses_persisted_scope(monkeypatch):
    def node(slug: str, *, root: str = "repo", path: str = "src/config.py") -> ContextNode:
        return ContextNode(
            uri=f"ctx://acct/resources/{slug}",
            context_type="RESOURCE",
            category="code",
            level=4,
            owner_space="resource:acct",
            abstract="parse_config function",
            overview="path: " + path,
            content="def parse_config(payload):\n    return payload\n" * 4,
            metadata={
                "resource_type": "code",
                "resource_root_slug": root,
                "relative_path": path,
                "start_line": 10,
                "end_line": 14,
                "symbol": "parse_config",
                "symbol_kind": "function",
                "signature": "parse_config(payload)",
                "language": "python",
            },
        )

    selected = node("selected")
    excluded = node("excluded", root="other-root", path="lib/other.py")
    nodes = {selected.uri: selected, excluded.uri: excluded}

    class ContextFS:
        def list_children(self, uri, ctx):
            raise AssertionError("active test uses a persisted snapshot")

        def read_node(self, uri, ctx):
            assert ctx.account_id == "acct"
            return nodes[uri]

    config = SimpleNamespace(
        code_retrieval_enabled=True,
        code_retrieval_route_enabled=lambda route: route == "symbolic",
        code_retrieval_candidate_limit=4,
        code_retrieval_result_limit=2,
        code_retrieval_snippet_chars=24,
        code_retrieval_batch_chars=24,
        code_retrieval_route_deadline_ms=500,
    )
    service = CodeRetrievalService(
        context_fs=ContextFS(),
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=config,
    )
    monkeypatch.setattr(
        service,
        "_get_snapshot",
        lambda ctx: SimpleNamespace(
            records=tuple(
                {
                    "uri": item.uri,
                    "metadata": dict(item.metadata),
                    "content": item.content,
                }
                for item in nodes.values()
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "_search_symbolic",
        lambda query, ctx, top_k: [
            {
                "uri": excluded.uri,
                "score": 1.0,
                "raw_score": 1.0,
                "metadata": dict(excluded.metadata),
            },
            {
                "uri": selected.uri,
                "score": 0.8,
                "raw_score": 0.8,
                "metadata": dict(selected.metadata),
            },
        ],
    )

    outcome = service.search_active(
        ActiveCodeSearchRequest(
            "parse_config",
            scope=CodeSearchScope(root_slugs=("repo",), paths=("src/config.py",)),
            budget=CodeSearchBudget(result_limit=1, snippet_chars=12, batch_chars=12),
        ),
        _ctx(),
    )

    assert outcome.status == "ok"
    assert len(outcome.evidence) == 1
    assert outcome.evidence[0].uri == selected.uri
    assert outcome.evidence[0].locator == "repo/src/config.py:10-10"
    assert outcome.evidence[0].coverage["truncated"] is True
    assert len(outcome.evidence[0].excerpt) <= 12
    assert outcome.diagnostics.scope_match_count == 1
    assert outcome.diagnostics.clipped_count == 1
    assert outcome.diagnostics.duration_ms >= 0
    assert outcome.to_dict()["diagnostics"]["duration_ms"] == outcome.diagnostics.duration_ms


def test_active_symbolic_scope_is_applied_before_route_top_k(monkeypatch):
    def record(uri: str, root: str) -> dict:
        return {
            "uri": uri,
            "metadata": {
                "resource_type": "code",
                "resource_root_slug": root,
                "relative_path": "src/target.py",
                "symbol": "target",
                "symbol_kind": "function",
                "signature": "target()",
                "language": "python",
            },
            "content": "def target():\n    return 1",
        }

    unscoped = record("ctx://acct/resources/a", "other")
    scoped = record("ctx://acct/resources/z", "repo")
    nodes = {unscoped["uri"]: unscoped, scoped["uri"]: scoped}

    class ContextFS:
        def read_node(self, uri, ctx):
            assert ctx.account_id == "acct"
            return ContextNode(
                uri=uri,
                context_type="RESOURCE",
                category="code",
                level=4,
                owner_space="resource:acct",
                abstract="target",
                overview="",
                content=nodes[uri]["content"],
                metadata=nodes[uri]["metadata"],
            )

    service = CodeRetrievalService(
        context_fs=ContextFS(),
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_route_enabled=lambda route: route == "symbolic",
            code_retrieval_candidate_limit=1,
            code_retrieval_result_limit=1,
            code_retrieval_snippet_chars=100,
            code_retrieval_batch_chars=100,
            code_retrieval_route_deadline_ms=500,
        ),
    )
    monkeypatch.setattr(
        service,
        "_get_snapshot",
        lambda ctx: SimpleNamespace(records=(unscoped, scoped)),
    )

    outcome = service.search_active(
        ActiveCodeSearchRequest("target", scope=CodeSearchScope(root_slugs=("repo",))),
        _ctx(),
    )

    assert outcome.status == "ok"
    assert [item.uri for item in outcome.evidence] == [scoped["uri"]]
    assert outcome.diagnostics.routes["symbolic"]["scope_filtered_count"] == 0


def test_active_semantic_scope_underfill_is_diagnosed_without_route_fill(monkeypatch):
    def hit(uri: str, root: str) -> RetrievedBlock:
        return RetrievedBlock(
            uri=uri,
            score=1.0,
            category="code",
            metadata={
                "resource_type": "code",
                "resource_root_slug": root,
                "relative_path": "src/target.py",
                "symbol": "target",
                "language": "python",
            },
        )

    unscoped_uri = "ctx://acct/resources/a"
    scoped_uri = "ctx://acct/resources/z"
    records = (
        {
            "uri": unscoped_uri,
            "metadata": dict(hit(unscoped_uri, "other").metadata),
            "content": "unscoped",
        },
        {
            "uri": scoped_uri,
            "metadata": dict(hit(scoped_uri, "repo").metadata),
            "content": "scoped",
        },
    )

    class ReadAPI:
        def search_memory(self, **kwargs):
            return SearchMemoryResult(
                hits=[hit(unscoped_uri, "other"), hit(scoped_uri, "repo")][: kwargs["top_k"]]
            )

    service = CodeRetrievalService(
        context_fs=SimpleNamespace(),
        read_api=ReadAPI(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_route_enabled=lambda route: route == "semantic",
            code_retrieval_candidate_limit=1,
            code_retrieval_result_limit=1,
            code_retrieval_snippet_chars=100,
            code_retrieval_batch_chars=100,
            code_retrieval_route_deadline_ms=500,
        ),
    )
    monkeypatch.setattr(service, "_get_snapshot", lambda ctx: SimpleNamespace(records=records))

    outcome = service.search_active(
        ActiveCodeSearchRequest("target", scope=CodeSearchScope(root_slugs=("repo",))),
        _ctx(),
    )

    assert outcome.status == "zero_result"
    semantic_diag = outcome.diagnostics.routes["semantic"]
    assert semantic_diag["scope_filtered_count"] == 1
    assert semantic_diag["scope_underfill"] is True
    assert outcome.diagnostics.duration_ms >= 0


def test_active_search_disabled_is_explicit_and_does_not_read_persisted_state():
    class ContextFS:
        def list_children(self, uri, ctx):
            raise AssertionError("disabled active retrieval must not read ContextFS")

    service = CodeRetrievalService(
        context_fs=ContextFS(),
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(code_retrieval_enabled=False),
    )

    outcome = service.search_active(
        ActiveCodeSearchRequest("query", budget=CodeSearchBudget(candidate_limit=9999)),
        _ctx(),
    )

    assert outcome.status == "disabled"
    assert outcome.evidence == ()
    assert outcome.diagnostics.budget_clamps["candidate_limit"]["effective"] == 50
    assert all(value["state"] == "disabled" for value in outcome.diagnostics.routes.values())
    assert outcome.diagnostics.duration_ms >= 0


def test_active_search_hydration_failure_keeps_later_persisted_evidence(monkeypatch):
    def record(slug: str) -> dict:
        return {
            "uri": f"ctx://acct/resources/{slug}",
            "metadata": {
                "resource_type": "code",
                "resource_root_slug": "repo",
                "relative_path": f"src/{slug}.py",
                "symbol": slug,
                "language": "python",
            },
            "content": "persisted",
        }

    missing = record("missing")
    selected = record("selected")
    selected_node = ContextNode(
        uri=selected["uri"],
        context_type="RESOURCE",
        category="code",
        level=4,
        owner_space="resource:acct",
        abstract="selected",
        overview="",
        content="selected body",
        metadata=selected["metadata"],
    )

    class ContextFS:
        def read_node(self, uri, ctx):
            if uri == missing["uri"]:
                raise NodeNotFoundError(uri)
            return selected_node

    service = CodeRetrievalService(
        context_fs=ContextFS(),
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_route_enabled=lambda route: route == "symbolic",
        ),
    )
    monkeypatch.setattr(
        service,
        "_get_snapshot",
        lambda ctx: SimpleNamespace(records=(missing, selected)),
    )
    monkeypatch.setattr(
        service,
        "_search_symbolic",
        lambda query, ctx, top_k: [
            {"uri": missing["uri"], "score": 1.0, "metadata": missing["metadata"]},
            {"uri": selected["uri"], "score": 0.9, "metadata": selected["metadata"]},
        ],
    )

    outcome = service.search_active(ActiveCodeSearchRequest("selected"), _ctx())

    assert outcome.status == "partial_failure"
    assert [item.uri for item in outcome.evidence] == [selected["uri"]]
    assert outcome.diagnostics.hydration_errors == 1
    assert outcome.diagnostics.duration_ms >= 0


def test_active_search_snapshot_and_route_failures_keep_safe_timing_diagnostics(monkeypatch):
    config = SimpleNamespace(
        code_retrieval_enabled=True,
        code_retrieval_route_enabled=lambda route: route == "symbolic",
        code_retrieval_candidate_limit=2,
        code_retrieval_result_limit=1,
        code_retrieval_snippet_chars=100,
        code_retrieval_batch_chars=100,
        code_retrieval_route_deadline_ms=500,
    )
    service = CodeRetrievalService(
        context_fs=SimpleNamespace(),
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=config,
    )
    monkeypatch.setattr(
        service,
        "_get_snapshot",
        lambda ctx: (_ for _ in ()).throw(RuntimeError("private snapshot failure")),
    )

    snapshot_failure = service.search_active(ActiveCodeSearchRequest("query"), _ctx())

    assert snapshot_failure.status == "error"
    assert snapshot_failure.diagnostics.duration_ms >= 0
    assert all(
        route["state"] == "error" and route["duration_ms"] == 0
        for route in snapshot_failure.diagnostics.routes.values()
    )
    assert "private snapshot failure" not in str(snapshot_failure.to_dict()["diagnostics"])

    record = {
        "uri": "ctx://acct/resources/target",
        "metadata": {
            "resource_type": "code",
            "resource_root_slug": "repo",
            "relative_path": "src/target.py",
            "symbol": "target",
            "language": "python",
        },
        "content": "persisted body",
    }
    monkeypatch.setattr(service, "_get_snapshot", lambda ctx: SimpleNamespace(records=(record,)))
    monkeypatch.setattr(
        service,
        "_search_symbolic",
        lambda query, ctx, top_k: (_ for _ in ()).throw(TimeoutError("route deadline exceeded")),
    )

    timeout = service.search_active(ActiveCodeSearchRequest("target"), _ctx())

    assert timeout.status == "partial_failure"
    assert timeout.diagnostics.duration_ms >= 0
    assert timeout.diagnostics.routes["symbolic"]["state"] == "timeout"
    assert timeout.diagnostics.routes["symbolic"]["duration_ms"] >= 0
    assert timeout.diagnostics.errors == ("symbolic:timeout",)


def test_active_semantic_deadline_bounds_blocking_empty_backend(monkeypatch):
    backend_started = threading.Event()
    release_backend = threading.Event()
    backend_finished = threading.Event()

    class ReadAPI:
        def search_memory(self, **kwargs):
            del kwargs
            backend_started.set()
            release_backend.wait(timeout=1.0)
            backend_finished.set()
            return SearchMemoryResult()

    record = {
        "uri": "ctx://acct/resources/target",
        "metadata": {
            "resource_type": "code",
            "resource_root_slug": "repo",
            "relative_path": "src/target.py",
            "symbol": "target",
            "language": "python",
        },
        "content": "def target():\n    return True",
    }
    service = CodeRetrievalService(
        context_fs=SimpleNamespace(),
        read_api=ReadAPI(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_route_enabled=lambda route: route == "semantic",
            code_retrieval_route_deadline_ms=25,
        ),
    )
    monkeypatch.setattr(service, "_get_snapshot", lambda ctx: SimpleNamespace(records=(record,)))

    started = time.monotonic()
    try:
        outcome = service.search_active(ActiveCodeSearchRequest("target"), _ctx())
        elapsed = time.monotonic() - started

        assert backend_started.wait(timeout=0.1)
        assert elapsed < 0.25
        assert not backend_finished.is_set()
        assert outcome.status == "partial_failure"
        assert outcome.diagnostics.routes["semantic"]["state"] == "timeout"
        assert outcome.diagnostics.errors == ("semantic:timeout",)
    finally:
        release_backend.set()
        assert backend_finished.wait(timeout=1.0)


def test_semantic_runner_does_not_start_call_after_capacity_wait_expires(monkeypatch):
    class CapacitySlot:
        def __init__(self):
            self.release_calls = 0

        def acquire(self, *, timeout):
            assert timeout == pytest.approx(0.1)
            return True

        def release(self):
            self.release_calls += 1

    class UnexpectedThread:
        def __init__(self, **kwargs):
            del kwargs
            raise AssertionError("expired call must not create a backend thread")

    ticks = iter((10.0, 10.2))
    slot = CapacitySlot()
    runner = code_retrieval_module._BoundedDaemonCallRunner(max_inflight=1)
    runner._slots = slot
    monkeypatch.setattr(
        code_retrieval_module,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    monkeypatch.setattr(
        code_retrieval_module,
        "threading",
        SimpleNamespace(Event=threading.Event, Thread=UnexpectedThread),
    )
    provider_called = False

    def operation():
        nonlocal provider_called
        provider_called = True

    with pytest.raises(TimeoutError, match="semantic route deadline exceeded"):
        runner.run(operation, deadline=10.1)

    assert provider_called is False
    assert slot.release_calls == 1


def test_focused_snippet_requires_exact_account_code_uri_and_reuses_hydrator(monkeypatch):
    """Unit-only: a ContextFS double proves local URI and clipping behavior."""

    uri = "ctx://acct/resources/focused"
    node = ContextNode(
        uri=uri,
        context_type="RESOURCE",
        category="code",
        level=4,
        owner_space="resource:acct",
        abstract="focused",
        overview="",
        content="def focused(value):\n    return value\n" * 3,
        metadata={
            "resource_type": "code",
            "resource_root_slug": "repo",
            "relative_path": "src/focused.py",
            "start_line": 4,
            "end_line": 6,
            "symbol": "focused",
            "symbol_kind": "function",
            "signature": "focused(value)",
            "language": "python",
        },
    )

    class ContextFS:
        def __init__(self):
            self.read_calls = []

        def read_node(self, requested_uri, ctx):
            assert ctx.account_id == "acct"
            self.read_calls.append(requested_uri)
            if requested_uri != uri:
                raise NodeNotFoundError(requested_uri)
            return node

    fs = ContextFS()
    service = CodeRetrievalService(
        context_fs=fs,
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(
            code_retrieval_enabled=True,
            code_retrieval_snippet_chars=12,
            code_retrieval_batch_chars=20,
        ),
    )
    snapshot = SimpleNamespace(
        records=(
            {
                "uri": uri,
                "metadata": dict(node.metadata),
                "content": node.content,
            },
        )
    )
    monkeypatch.setattr(service, "_get_snapshot", lambda ctx: snapshot)

    evidence = service.read_focused_snippet(uri=uri, ctx=_ctx(), max_chars=99)

    assert evidence.uri == uri
    assert evidence.locator == "repo/src/focused.py:4-4"
    assert evidence.coverage["truncated"] is True
    assert len(evidence.excerpt) <= 12
    assert fs.read_calls == [uri]

    monkeypatch.setattr(
        service,
        "_get_snapshot",
        lambda ctx: pytest.fail("invalid focused URI must not hydrate a snapshot"),
    )
    for invalid_uri in (
        "focused.py",
        "/srv/repo/src/focused.py",
        "ctx://acct/resources/*",
        "ctx://other/resources/focused",
        "ctx://acct/resources/focused/content.md",
    ):
        with pytest.raises(ValidationError, match="exact account-local code URI"):
            service.read_focused_snippet(uri=invalid_uri, ctx=_ctx())


def test_focused_snippet_rejects_non_code_or_missing_snapshot_identity(monkeypatch):
    """Unit-only: the focused API has no basename or fallback retrieval path."""

    service = CodeRetrievalService(
        context_fs=SimpleNamespace(),
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(code_retrieval_enabled=True),
    )
    uri = "ctx://acct/resources/not-code"
    monkeypatch.setattr(
        service,
        "_get_snapshot",
        lambda ctx: SimpleNamespace(
            records=(
                {
                    "uri": uri,
                    "metadata": {"resource_type": "memory"},
                    "content": "not code",
                },
            )
        ),
    )

    with pytest.raises(FileNotFoundError, match="code snippet not found"):
        service.read_focused_snippet(uri=uri, ctx=_ctx())


def test_resolved_graph_uses_bounded_relation_store_lookup():
    source_uri = "ctx://acct/resources/source"
    target_uri = "ctx://acct/resources/target"
    records = (
        {
            "uri": source_uri,
            "metadata": {
                "resource_type": "code",
                "resource_root_slug": "repo",
                "relative_path": "src/source.py",
                "symbol": "source",
                "language": "python",
            },
        },
        {
            "uri": target_uri,
            "metadata": {
                "resource_type": "code",
                "resource_root_slug": "repo",
                "relative_path": "src/target.py",
                "symbol": "target",
                "language": "python",
            },
        },
    )

    class RelationStore:
        def __init__(self):
            self.calls = []

        def get_one_hop(self, uri, ctx, limit=3):
            self.calls.append((uri, ctx.account_id, limit))
            if uri == source_uri:
                return [SimpleNamespace(to_uri=target_uri, relation_type="calls")]
            return []

    relation_store = RelationStore()
    service = CodeRetrievalService(
        context_fs=SimpleNamespace(),
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(code_retrieval_enabled=True),
        relation_store=relation_store,
    )

    hits = service._search_graph_resolved(
        "source",
        _ctx(),
        records=records,
        top_k=2,
        neighbor_limit=1,
        deadline=time.monotonic() + 1.0,
    )

    assert relation_store.calls
    assert all(limit == 1 for _, _, limit in relation_store.calls)
    assert any(
        context.get("target_locator") == "repo/src/target.py"
        for item in hits
        for context in item.get("metadata", {}).get("_graph_context", ())
    )


def test_resolved_graph_queries_relation_store_only_for_selected_anchors():
    records = tuple(
        {
            "uri": f"ctx://acct/resources/anchor-{index:02d}",
            "metadata": {
                "resource_type": "code",
                "resource_root_slug": "repo",
                "relative_path": f"src/anchor_{index:02d}.py",
                "symbol": "needle",
                "language": "python",
            },
        }
        for index in range(40)
    )

    class RelationStore:
        def __init__(self):
            self.calls = []

        def get_one_hop(self, uri, ctx, limit=3):
            self.calls.append((uri, ctx.account_id, limit))
            return []

    relation_store = RelationStore()
    service = CodeRetrievalService(
        context_fs=SimpleNamespace(),
        read_api=SimpleNamespace(),
        bm25_index=Bm25Index(),
        config=SimpleNamespace(code_retrieval_enabled=True),
        relation_store=relation_store,
    )

    hits = service._search_graph_resolved(
        "needle",
        _ctx(),
        records=records,
        top_k=2,
        neighbor_limit=1,
        deadline=time.monotonic() + 1.0,
    )

    assert hits
    assert relation_store.calls == [
        ("ctx://acct/resources/anchor-00", "acct", 1),
        ("ctx://acct/resources/anchor-01", "acct", 1),
    ]
