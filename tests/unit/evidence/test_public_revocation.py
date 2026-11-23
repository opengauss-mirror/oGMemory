"""Public memory readers must honor revocation before physical cleanup succeeds."""

import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flask import Flask

from core.errors import AccessDeniedError, NodeNotFoundError
from core.models import ContextNode, RetrievalConfig, Role
from fs.sql_adapter.session_mutation import SQLSessionGuard
from fs.sql_adapter.sql_context_fs import SQLContextFS
from retrieval.agfs_retriever import AGFSRetrieval
from retrieval.context_reader import ContextReader
from server.mcp_resources import internal_to_external_uri, render_resource, visible_memory_nodes
from server.mcp_security import MCPVisibilityError
from server.mcp_tools import list_memories, read_memory
from server.memory_viewer import create_memory_viewer_blueprint
from service.evidence_lifecycle import EvidenceLifecycle, ensure_sources_active
from tests.unit.evidence.test_directory_revocation import DIR, write_fact
from tests.unit.evidence.test_ingest import configured_service
from tests.unit.evidence.test_sources import CTX, message
from tests.unit.evidence.test_writer import write_candidates


@pytest.mark.parametrize("category", ["fact", "profile"])
def test_public_reads_and_lists_hide_revoked_sources_before_cleanup(setup, monkeypatch, category):
    import server.app as http

    fs, _, store, _ = setup
    plan, = write_candidates(setup, [message()], categories={"fact"})
    node = fs.read_node(plan.target_uri, CTX)
    if category == "profile":
        node = replace(node, category="profile", uri=plan.target_uri.rsplit("/", 2)[0] + "/profile",
                       metadata={**node.metadata, "_expected_version": 0})
        fs.write_node(node, CTX)
    uri = internal_to_external_uri(node.uri, category=node.category)
    service, _, _ = configured_service(fs)
    monkeypatch.setattr("server.mcp_tools.build_mcp_context", lambda *a, **kw: CTX)
    monkeypatch.setattr(http, "_service", service)
    monkeypatch.setattr(http, "_build_authenticated_context", lambda _: CTX)
    assert read_memory(service, uri=uri, request_context=CTX)["content"]
    assert render_resource(service, uri, CTX)["content"]
    store.revoke_session(CTX)
    assert fs.exists(node.uri, CTX)  # Deliberately leave canonical cleanup pending.
    for level in ["abstract", "overview", "content", "all"]:
        with pytest.raises(MCPVisibilityError):
            read_memory(service, uri=uri, level=level, request_context=CTX)
        response = http.app.test_client().post("/api/v1/memories/read", json={"uri": uri, "level": level})
        assert response.status_code == 403
    with pytest.raises(MCPVisibilityError):
        render_resource(service, uri, CTX)
    assert render_resource(service, f"ogmemory://memories/{category}", CTX)["memories"] == []
    assert list_memories(service)["memories"] == []


@pytest.mark.parametrize("summary_only", [False, True])
def test_filtered_pages_continue_to_visible_ordinary_memories(setup, summary_only):
    fs, _, store, _ = setup
    plan, = write_candidates(setup, [message()], categories={"fact"})
    node = fs.read_node(plan.target_uri, CTX)
    base = node.uri.rsplit("/", 1)[0]
    for key, metadata in [("a", node.metadata), ("b", node.metadata), ("z", {})]:
        fs.write_node(replace(node, uri=base + "/" + key, metadata={**metadata, "_expected_version": 0}), CTX)
    store.revoke_session(CTX)
    if summary_only:
        class SummaryOnly:
            def list_node_summaries(self, uri, ctx, limit):
                return fs.list_node_summaries(uri, ctx, limit)

            read_node = fs.read_node
        reader = SummaryOnly()
    else:
        reader = fs
    assert [n.uri for n in visible_memory_nodes(reader, base, CTX, 1)] == [base + "/z"]
    assert [n.uri for n in visible_memory_nodes(reader, base, CTX, 1, after_uri=base + "/b")] == [base + "/z"]
    assert visible_memory_nodes(reader, base, CTX, 1, after_uri=base + "/z") == []


# Public reads preserve source revocation and the caller's authorization scope.


@pytest.mark.parametrize("admin", [False, True])
@pytest.mark.parametrize("profile", [False, True])
def test_viewer_hides_revoked_sources_when_cleanup_fails(setup, monkeypatch, admin, profile):
    fs, _, _, _ = setup
    uri = write_fact(setup)
    if profile:
        node = fs.read_node(uri, CTX)
        uri = DIR.rsplit("facts/", 1)[0] + "profile"
        fs.write_node(replace(node, uri=uri, category="profile"), CTX)
    else:
        for name in ("zz-one", "zz-two"):
            fs.write_node(ContextNode(DIR + name, "MEMORY", "fact", 2, CTX.user_space_name(),
                                      name, "", "ordinary memory", {}), CTX)
    context = replace(CTX, agent_id="", role=Role.ADMIN)
    service = SimpleNamespace(_get_context_fs=lambda: fs)
    app = Flask(__name__)
    app.register_blueprint(create_memory_viewer_blueprint(
        lambda: service, lambda params: context,
        lambda exc: ({"error": str(exc)}, 404 if isinstance(exc, NodeNotFoundError) else 500)))
    # Authentication/target selection is covered by the Viewer HTTP suite.
    monkeypatch.setattr("server.memory_viewer._require_admin", lambda *args: None)
    monkeypatch.setattr("server.memory_viewer._admin_target_context", lambda *args: context)
    client = app.test_client()
    list_url = "/api/v1/memory-viewer/admin/memories" if admin else "/api/v1/memories"
    detail_url = "/api/v1/memory-viewer/admin/detail" if admin else "/api/v1/memories/detail"
    params = {"type": "profile" if profile else "fact", "limit": 1}
    external = internal_to_external_uri(uri, category=params["type"])
    assert client.get(detail_url, query_string={"uri": external}).status_code == 200
    assert client.get(list_url, query_string=params).get_json()["count"] == 1
    outbox = Mock()
    outbox.register_once.side_effect = ConnectionError("outbox unavailable")
    with pytest.raises(ConnectionError, match="outbox"):
        EvidenceLifecycle(fs, outbox).delete_session(CTX)
    assert fs.read_node(uri, CTX).content  # Canonical cleanup has not hidden it.
    assert client.get(detail_url, query_string={"uri": external}).status_code == 404
    page = client.get(list_url, query_string=params).get_json()
    if profile:
        assert page["memories"] == []
    else:
        assert [item["context_uri"] for item in page["memories"]] == [DIR + "zz-one"]
        assert page["truncated"] and page["next_cursor"]
        page = client.get(list_url, query_string={**params, "cursor": page["next_cursor"]}).get_json()
        assert [item["context_uri"] for item in page["memories"]] == [DIR + "zz-two"]
        assert not page["truncated"]


@pytest.mark.parametrize("projection", ["fact", "directory", "root"])
@pytest.mark.parametrize("method", ["read", "read_many", "read_authorized"])
def test_cross_user_sources_keep_authorization_and_session_identity(setup, monkeypatch, projection, method):
    fs, _, store, _ = setup
    uri = write_fact(setup)
    if projection != "fact":
        node = fs.read_node(uri, CTX)
        uri = DIR if projection == "directory" else DIR.removesuffix("/facts/")
        fs.write_node(replace(node, uri=uri, metadata={"node_type": "directory",
            "evidence_source_refs": node.metadata["deterministic"]["source_refs"]}), CTX)
    request = replace(CTX, user_id="admin", session_id="http", role=Role.ADMIN,
                      visible_owner_spaces=("user:alice", "user:bob"))
    guard = SQLSessionGuard(Mock(closed=False), "dbname=test", request, 1)
    reads = []
    read = fs.read_node

    def guarded_read(uri, ctx):
        assert ctx is request
        reads.append(uri)
        return read(uri, ctx)

    def retrieve():
        with monkeypatch.context() as patch:
            # Use the actual SQL ACL and its session guard on the in-memory store.
            patch.setattr(fs, "_ensure_accessible", SQLContextFS._ensure_accessible)
            patch.setattr("fs.sql_adapter.session_mutation.active_session_guard", lambda: guard)
            patch.setattr(fs, "read_node", guarded_read)
            reader = ContextReader(fs, storage_backend="sql")
            if method == "read_many":
                result = reader.read_many([(uri, None)], request)[uri, None]
                if isinstance(result, Exception):
                    raise result
                return result
            return getattr(reader, method)(uri, request)

    assert retrieve().abstract
    assert any("/users/alice/evidence/sources/" in path for path in reads)
    assert any("/users/alice/evidence/sessions/" in path for path in reads)
    assert guard.poison_reason is None
    store.revoke_session(CTX)
    with pytest.raises(NodeNotFoundError):
        retrieve()


@pytest.mark.parametrize("account,spaces", [("account", ("user:bob",)), ("other", ("user:alice",))])
def test_source_location_does_not_grant_access(setup, monkeypatch, account, spaces):
    fs, _, _, _ = setup
    node = fs.read_node(write_fact(setup), CTX)
    request = replace(CTX, account_id=account, user_id="bob", visible_owner_spaces=spaces)
    monkeypatch.setattr(fs, "_ensure_accessible", SQLContextFS._ensure_accessible)
    with pytest.raises((AccessDeniedError, NodeNotFoundError)):
        ensure_sources_active(node, fs, request)


def test_source_owner_must_match_canonical_memory_uri(setup):
    fs, _, _, _ = setup
    node = fs.read_node(write_fact(setup), CTX)
    with pytest.raises(NodeNotFoundError):
        ensure_sources_active(replace(node, owner_space="user:bob"), fs, CTX)


# AGFS scan, cached FTS and direct targets retain source revocation checks.


@pytest.mark.parametrize("mode", ["scan", "literal", "sqlite_bm25", "sqlite_index", "target"])
@pytest.mark.parametrize("derived", [False, True])
def test_revoked_sources_are_filtered_before_agfs_top_k(setup, monkeypatch, tmp_path, mode, derived):
    fs, _, store, _ = setup
    uri = write_fact(setup)
    node = fs.read_node(uri, CTX)
    if derived:
        fs.write_node(replace(node, metadata={
            "evidence_source_refs": node.metadata["deterministic"]["source_refs"],
            "_expected_version": node.metadata["version"],
        }), CTX)
    ordinary = uri.rsplit("/", 1)[0] + "/ordinary"
    fs.write_node(ContextNode(ordinary, "MEMORY", "fact", 2, CTX.user_space_name(),
                              "Paris", "Paris", "An ordinary Paris memory.", {}), CTX)
    read = fs._client.read
    monkeypatch.setattr(fs._client, "read", lambda path, size=None: read(path)[:size])
    lexical = None
    if mode == "sqlite_index":
        from core.lexical import lexical_record_from_index_record
        from index.index_record_builder import build_agfs_lexical_records
        from providers.lexical_index.sqlite_index import SQLiteLexicalIndex

        lexical = SQLiteLexicalIndex(str(tmp_path / "evidence-test.db"))
        lexical.upsert_many([lexical_record_from_index_record(record) for target in (uri, ordinary)
                            for record in build_agfs_lexical_records(fs.read_node(target, CTX))])
    pipeline = AGFSRetrieval(fs, RetrievalConfig(), fts_enabled=mode in {"literal", "sqlite_bm25"},
                             fts_ranker="sqlite_bm25" if mode == "sqlite_bm25" else "literal",
                             lexical_index=lexical)
    options = {"context_type": "MEMORY", "fill_content_for_top_k": 1}
    if mode == "target":
        options["target_uri"] = uri
    try:
        result = pipeline.run("Paris", CTX, **options)
        if pipeline.fts is not None:
            deadline = time.monotonic() + 5
            while result.trace.stages[0].metrics.get("fts") != "ready":
                assert time.monotonic() < deadline
                time.sleep(.01)
                result = pipeline.run("Paris", CTX, **options)
        assert uri in {hit.uri for hit in result.hits}
        if lexical is not None:
            assert result.trace.stages[0].metrics["lexical"] == "ready"
        store.revoke_session(CTX)
        # Retain the stale canonical card and FTS snapshot, as on interrupted cleanup.
        assert fs.read_node(uri, CTX)
        result = pipeline.run("Paris", CTX, top_k=1, **options)
        assert [hit.uri for hit in result.hits] == ([] if mode == "target" else [ordinary])
        documents = list(pipeline._index_documents((uri.rsplit("/", 1)[0],), CTX))
        assert [item[0] for item in documents] == [ordinary]
    finally:
        if pipeline.fts is not None:
            pipeline.fts.close(5)
        if lexical is not None:
            lexical.close()
