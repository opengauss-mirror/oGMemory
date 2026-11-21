"""AGFS/SQL content transitions and destination representation boundaries."""
import json
import os
import subprocess
import sys
import uuid
from contextlib import ExitStack, closing
from dataclasses import fields

import pytest

from providers.unified_config import OgMemConfig
from service.migration.storage import storage
from tests.migration_support import MigrationHarness as MigrationService
from tests.integration import test_memory_migration as common
from tests.integration.test_memory_migration import (
    new_config, fixtures, assert_merged_package, clean_isolated_databases,
)

pytestmark = common.pytestmark


@pytest.mark.parametrize("weight", [0, 1])
@pytest.mark.parametrize("legacy_package", [False, True])
def test_integer_relation_weight_restores_to_sql(tmp_path, weight, legacy_package):
    from service.migration.package import write_package

    source, target = new_config("agfs", "integer-source"), new_config("sql", "integer-target")
    data = fixtures()
    data["relations"][0]["weight"] = weight
    package = tmp_path / "integer.tar.gz"
    if legacy_package:
        # Older packages may retain integer JSON numbers and omit descriptions.
        write_package(package, data, source_backend="agfs", source_version="legacy")
    else:
        with storage(source) as adapter:
            for kind, rows in data.items():
                adapter.restore(kind, rows)
        MigrationService(source).export(package)
    preview = MigrationService(target).preview(package)
    result = MigrationService(target).import_package(package)
    assert_merged_package(package, target, result, preview["data"])
    assert result["index_records"] == 13
    with storage(target) as adapter:
        edges = list(adapter.records("relations"))
    assert len(edges) == 1 and edges[0]["weight"] == float(weight)


@pytest.mark.parametrize("archive_endpoint", ["from_uri", "to_uri"])
def test_archive_relation_requires_node_endpoints_only_for_sql_graph(tmp_path, archive_endpoint):
    from dataclasses import replace

    from pyagfs import AGFSClient

    from core.models import RelationEdge, RequestContext
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from providers.relation_store.agfs_relation_store import AGFSRelationStore
    from service.api import MemoryWriteAPI
    from service.migration.package import PackageError
    from session.archive_store import SessionArchiveStore
    from tests.unit.commit.test_memory_identity import memory
    from tests.unit.server.test_extraction_batch_recovery import NoModel

    source, target = (
        new_config("agfs", "archive-edge-source"),
        new_config("sql", "archive-edge-target"),
    )
    target.graph_backend = "sql"
    client = AGFSClient(api_base_url=source.agfs_base_url)
    fs = AGFSContextFS(client, source.agfs_mount_prefix)
    ctx = RequestContext("acme", "alice", "main", "edge-session", "edge-test")
    archived = SessionArchiveStore(fs).write_archive(
        ctx.session_id, "history", "history", [{"role": "user", "content": "theme"}], ctx
    )
    assert archived.success, archived.error
    uri = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False).write_memory(
        replace(memory("theme", category="preference"), provenance_ids=[]), ctx
    )["target_uri"]
    edge = dict(
        from_uri=uri,
        to_uri=uri,
        relation_type="derived_from",
        weight=1.0,
        reason="history evidence",
    )
    edge[archive_endpoint] = archived.uri
    AGFSRelationStore(client, source.agfs_mount_prefix).upsert_edges([RelationEdge(**edge)], ctx)
    package = tmp_path / "archive-edge.tar.gz"
    MigrationService(source).export(package)
    importer = MigrationService(target)
    for operation in (importer.preview, importer.import_package):
        with pytest.raises(PackageError, match="SQL graph projection requires node endpoints"):
            operation(package)
        with storage(target) as adapter:
            adapter.check(empty=True)
            assert all(
                not list(adapter.records(kind)) for kind in ("nodes", "archives", "relations")
            )
        index = target.to_provider_config().create_vector_index()
        try:
            assert index.count() == 0
        finally:
            index.close()
    target.graph_backend = "none"
    importer.preview(package)
    importer.import_package(package)
    with storage(target) as adapter:
        assert len(list(adapter.records("nodes"))) == 1
        assert len(list(adapter.records("archives"))) == 1
        assert list(adapter.records("relations")) == [edge]


@pytest.mark.parametrize("graph", ["none", "sql"])
@pytest.mark.parametrize("alias", [True, False])
def test_directory_relation_uri_matches_sql_identity(tmp_path, graph, alias):
    from dataclasses import replace

    from pyagfs import AGFSClient

    from core.models import ContextNode, RelationEdge, RequestContext
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from providers.relation_store.agfs_relation_store import AGFSRelationStore
    from service.api import MemoryWriteAPI
    from service.migration.package import PackageError
    from tests.unit.commit.test_memory_identity import memory
    from tests.unit.server.test_extraction_batch_recovery import NoModel

    source, target = new_config("agfs", "directory-source"), new_config("sql", "directory-target")
    target.graph_backend = graph
    client = AGFSClient(api_base_url=source.agfs_base_url)
    fs = AGFSContextFS(client, source.agfs_mount_prefix)
    ctx = RequestContext("acme", "alice", "main", "s", "test")
    uri = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False).write_memory(
        replace(memory("editor", category="preference"), provenance_ids=[]), ctx
    )["target_uri"]
    directory = uri.rsplit("/", 1)[0] + "/"
    fs.write_node(
        ContextNode(
            uri=directory,
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space="user:alice",
            abstract="preferences",
            overview="preferences",
            content="",
            metadata={},
        ),
        ctx,
    )
    edge = RelationEdge(
        from_uri=uri,
        to_uri=directory.rstrip("/") if alias else directory,
        relation_type="related_to",
        weight=1,
        reason="directory",
    )
    AGFSRelationStore(client, source.agfs_mount_prefix).upsert_edges([edge], ctx)
    package = tmp_path / "directory.tar.gz"
    MigrationService(source).export(package)
    importer = MigrationService(target)
    if alias:
        for operation in (importer.preview, importer.import_package):
            with pytest.raises(PackageError, match="exactly match stored URI"):
                operation(package)
        with storage(target) as adapter:
            adapter.check(empty=True)
        index = target.to_provider_config().create_vector_index()
        try:
            assert index.count() == 0
        finally:
            index.close()
    else:
        importer.preview(package)
        importer.import_package(package)
        with storage(target) as adapter:
            rows = list(adapter.records("relations"))
            nodes = {r["uri"] for r in adapter.records("nodes")}
        assert rows and all(r["from_uri"] in nodes and r["to_uri"] in nodes for r in rows)


@pytest.mark.parametrize("target_backend", ["sql", "agfs"])
@pytest.mark.parametrize("content", ["a\x00b", r"a\u0000b"])
def test_archive_nul_sql_preflight_before_writes(tmp_path, target_backend, content):
    from pyagfs import AGFSClient

    from core.models import RequestContext
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from service.migration.package import PackageError
    from session.archive_store import SessionArchiveStore
    from tests.unit.service.migration.test_engine import node

    source, target = new_config("agfs", "nul-source"), new_config(target_backend, "nul-target")
    with storage(source) as adapter:
        adapter.restore("nodes", [node()])
    fs = AGFSContextFS(AGFSClient(api_base_url=source.agfs_base_url), source.agfs_mount_prefix)
    ctx = RequestContext("acme", "alice", "main", "nul-session", "nul-test")
    messages = [{"role": "user", "content": content}]
    written = SessionArchiveStore(fs).write_archive(ctx.session_id, "overview", "summary", messages, ctx)
    assert written.success, written.error
    path = tmp_path / "nul.tar.gz"
    expected = MigrationService(source).export(path)["data"]
    importer = MigrationService(target)
    if target_backend == "sql" and "\x00" in content:
        for operation in (importer.preview, importer.import_package):
            with pytest.raises(PackageError, match="SQL target cannot store NUL"):
                operation(path)
            with storage(target) as adapter:
                adapter.check(empty=True)
    else:
        importer.preview(path)
        assert importer.import_package(path)["data"] == expected
        with storage(target) as adapter:
            assert list(adapter.records("archives"))[0]["messages"] == messages


def test_sql_relation_group_mirrors_and_consumer(tmp_path):
    from dataclasses import asdict

    from core.models import RequestContext
    from fs.sql_adapter.pool import SharedConnectionPool
    from providers.relation_store.sql_relation_store import SQLRelationStore
    from tests.unit.service.migration.test_engine import node

    source, target = new_config("agfs", "edge-source"), new_config("sql", "edge-target")
    nodes, edges = [], []
    for account, slug, count in [("a", "first", 256), ("a", "second", 257), ("b", "third", 1000)]:
        row = node()
        row["account_id"] = account
        row["uri"] = f"ctx://{account}/users/alice/memories/preferences/{slug}"
        nodes.append(row)
        edges.extend(dict(from_uri=row["uri"], to_uri=row["uri"], relation_type=f"rel{i:04}",
                          weight=1.0, reason="reason") for i in range(count))
    with storage(source) as adapter:
        adapter.restore("nodes", nodes)
        adapter.restore("relations", edges)
    path = tmp_path / "edges.tar.gz"
    expected = MigrationService(source).export(path)["data"]
    assert MigrationService(target).import_package(path)["data"] == expected
    with storage(target) as adapter:
        with adapter.conn.cursor() as cur:
            cur.execute("SELECT uri,relations FROM context_nodes ORDER BY uri")
            mirrors = dict(cur.fetchall())
    pool = SharedConnectionPool(target.sql_connection_string, dialect=target.sql_dialect, pool_size=2)
    try:
        consumer = SQLRelationStore(pool=pool)
        for row in nodes:
            expected_edges = [e for e in edges if e["from_uri"] == row["uri"]]
            assert sorted(mirrors[row["uri"]], key=lambda e: e["relation_type"]) == expected_edges
            ctx = RequestContext(row["account_id"], "alice", "main", "", "test")
            actual = [asdict(e) for e in consumer.get_edges(row["uri"], ctx)]
            assert sorted(actual, key=lambda e: e["relation_type"]) == expected_edges
    finally:
        pool.close_all()


@pytest.mark.parametrize('source,target', [('agfs', 'sql'), ('sql', 'agfs')])
def test_cross_content_migration(tmp_path, request, source, target):
    common.exercise_migration(tmp_path, source, target, 'opengauss', request)


@pytest.mark.parametrize('source,target', [('agfs', 'sql'), ('sql', 'agfs')])
def test_cross_content_write_receipts(tmp_path, source, target):
    common.test_completed_write_receipts_are_excluded(tmp_path, source, target)


@pytest.mark.parametrize('source,target', [('agfs', 'sql'), ('sql', 'agfs')])
def test_cross_content_archive_provenance(tmp_path, source, target):
    common.test_archive_producer_to_provenance_consumer(tmp_path, source, target)


@pytest.mark.parametrize('identity_case', ['account', 'user', 'agent'])
def test_cross_content_sessions_identifier(tmp_path, identity_case):
    common.test_sessions_identifier_survives_sql_migration(tmp_path, 'agfs', identity_case)


def test_cross_content_directory_alias(tmp_path):
    common.test_agfs_directory_source_alias_preserves_backend_semantics(tmp_path, 'sql')
