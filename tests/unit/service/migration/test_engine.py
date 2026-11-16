import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from service.migration.engine import MigrationService
from service.migration.package import PackageError, open_package, write_package
from service.migration.storage import AGFSStorage


class Files:
    def __init__(self):
        self.files = {}
        self.directories = set()

    def mkdir(self, path):
        self.directories.add(path.rstrip("/"))

    def write(self, path, value):
        self.files[path] = value

    def read(self, path):
        return self.files[path]

    def ls(self, path):
        path = path.rstrip("/") + "/"
        entries = {}
        for name in [*self.directories, *self.files]:
            if name.startswith(path):
                suffix = name[len(path) :]
                if suffix:
                    base = suffix.split("/")[0]
                    entries[base] = {
                        "name": base,
                        "isDir": "/" in suffix or name in self.directories,
                    }
        return list(entries.values())


def config():
    return SimpleNamespace(
        storage_backend="agfs",
        agfs_base_url="",
        agfs_mount_prefix="/local",
        vector_db_type="chroma",
        chroma_persist_directory="/isolated",
        scene_enabled=False,
    )


def node(status="ACTIVE"):
    return dict(
        uri="ctx://acme/users/alice/memories/preferences/editor",
        account_id="acme",
        owner_space="user:alice",
        category="preference",
        context_type="MEMORY",
        level=0,
        status=status,
        content="使用编辑器",
        abstract="偏好",
        overview="编辑器偏好",
        metadata={"custom": True},
        version=9,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-02-01T00:00:00+00:00",
    )


def archive():
    return dict(
        archive_id="archive1",
        session_id="session1",
        account_id="acme",
        owner_user_id="alice",
        owner_agent_id="main",
        abstract="历史",
        overview="对话",
        messages=[{"content": "你好"}],
        metadata={"archive_status": "MERGED"},
        created_at="2026-01-01T00:00:00+00:00",
    )


class Indexes:
    def __init__(self, cfg):
        self.count = 0

    def check_empty(self):
        pass

    def validate_records(self, inv):
        pass

    def probe(self):
        pass

    def rebuild(self, inv, progress):
        from service.migration.rebuild import records_for_node

        self.count = sum(len(records_for_node(r)) for r in inv.records("nodes"))
        return self.count

    def verify(self, inv):
        pass

    def close(self):
        pass


def service(files):
    @contextmanager
    def factory(cfg):
        yield AGFSStorage(cfg, client=files)

    return MigrationService(config(), adapter_factory=factory, index_factory=Indexes)


def test_agfs_export_restore_keeps_archive_node_metadata_and_relations(tmp_path):
    source = Files()
    adapter = AGFSStorage(config(), source)
    r = node("ARCHIVED")
    r["metadata"]["provenance_ids"] = ["prov:1:archive:archive1:"]
    edge = dict(
        from_uri=r["uri"], to_uri=r["uri"], relation_type="related_to", weight=0.8, reason="关联"
    )
    adapter.restore("nodes", [r])
    adapter.restore("archives", [archive()])
    adapter.restore("relations", [edge])
    path = tmp_path / "source.tar.gz"
    expected = service(source).export(path)["data"]
    target = Files()
    result = service(target).restore(path)
    assert result["data"] == expected
    assert result["index_records"] == 0
    restored = AGFSStorage(config(), target)
    assert list(restored.records("nodes")) == [r]
    assert list(restored.records("archives")) == [archive()]
    assert list(restored.records("relations")) == [edge]


def test_missing_reference_rejected_before_target_writes(tmp_path):
    r = node()
    r["metadata"]["provenance_ids"] = ["prov:1:archive:missing:"]
    path = tmp_path / "bad.tar.gz"
    write_package(
        path,
        {"nodes": [r], "relations": [], "archives": []},
        source_backend="sql",
        source_version="test",
    )
    target = Files()
    with pytest.raises(PackageError, match="reference"):
        service(target).restore(path)
    assert not target.files


def test_nonempty_target_and_memory_index_rejected(tmp_path):
    source = Files()
    AGFSStorage(config(), source).restore("nodes", [node()])
    path = tmp_path / "source.tar.gz"
    service(source).export(path)
    with pytest.raises(PackageError, match="not empty"):
        service(source).restore(path)
    target = service(Files())
    target.cfg.vector_db_type = "memory"
    from providers.unified_config import OgMemConfig
    from service.migration.rebuild import Rebuilder

    target.cfg.to_provider_config = lambda: OgMemConfig(
        vector_db_type="memory"
    ).to_provider_config()
    target.index_factory = Rebuilder
    target.cfg.retrieval_hybrid_enabled = False
    with pytest.raises(PackageError, match="persistent"):
        target.preflight(path)


def test_unsupported_files_fail_export_without_publishing(tmp_path):
    source = Files()
    AGFSStorage(config(), source).restore("nodes", [node()])
    source.files["/local/accounts/acme/users/alice/memories/preferences/editor/image.png"] = (
        b"image"
    )
    with pytest.raises(PackageError, match="payload files"):
        service(source).export(tmp_path / "bad.tar.gz")
    assert not (tmp_path / "bad.tar.gz").exists()


def test_runtime_session_windows_not_exported(tmp_path):
    source = Files()
    path = "/local/accounts/acme/sessions/s/state"
    source.files[path + "/.meta.json"] = json.dumps(
        {"uri": "ctx://acme/sessions/s/state", "category": "state"}
    ).encode()
    path = tmp_path / "empty.tar.gz"
    service(source).export(path)
    with open_package(path) as package:
        assert list(package.records("nodes")) == []


def test_memory_name_containing_lock_is_not_a_mutation_lock(tmp_path):
    source = Files()
    r = node()
    r["uri"] = r["uri"].replace("editor", "blockchain")
    AGFSStorage(config(), source).restore("nodes", [r])
    service(source).export(tmp_path / "valid.tar.gz")


def test_existing_runtime_state_also_makes_target_nonempty():
    target = Files()
    path = "/local/accounts/acme/sessions/s/state"
    target.files[path + "/.meta.json"] = json.dumps(
        {"uri": "ctx://acme/sessions/s/state", "category": "state"}
    ).encode()
    with pytest.raises(PackageError, match="not empty"):
        AGFSStorage(config(), target).check(empty=True)


def test_legacy_merged_archive_stays_merged():
    from service.migration.records import normalize_archive

    r = archive()
    r["metadata"] = {"status": "MERGED", "merged_into": "archive2"}
    normalized = normalize_archive(r)
    assert normalized["metadata"] == {"archive_status": "MERGED", "merged_into": "archive2"}


def test_cross_account_evidence_rejected_before_writes(tmp_path):
    from urllib.parse import quote

    r = node()
    other = {**node(), "account_id": "other", "uri": node()["uri"].replace("acme", "other")}
    r["metadata"] = {"provenance_ids": ["prov:1:memory:" + quote(other["uri"], safe="") + ":"]}
    path = tmp_path / "bad-evidence.tar.gz"
    write_package(
        path,
        {"nodes": [r, other], "relations": [], "archives": []},
        source_backend="sql",
        source_version="test",
    )
    target = Files()
    with pytest.raises(PackageError, match="cross-account"):
        service(target).restore(path)
    assert not target.files


def test_uri_aliases_cannot_overwrite_the_same_agfs_path(tmp_path):
    r = {
        **node(),
        "context_type": "SKILL",
        "category": "skill",
        "owner_space": "agent:main",
        "uri": "ctx://acme/agents/main/skills/generated/test%20skill",
    }
    alias = {**r, "uri": r["uri"].replace("%20", " ")}
    path = tmp_path / "aliases.tar.gz"
    write_package(
        path,
        {"nodes": [r, alias], "relations": [], "archives": []},
        source_backend="sql",
        source_version="test",
    )
    target = Files()
    with pytest.raises(PackageError, match="same storage path"):
        service(target).restore(path)
    assert not target.files


def test_unrepresentable_sql_identity_is_rejected_during_preflight(tmp_path):
    r = {**node(), "category": "x" * 65}
    path = tmp_path / "long.tar.gz"
    write_package(
        path,
        {"nodes": [r], "relations": [], "archives": []},
        source_backend="agfs",
        source_version="test",
    )
    with pytest.raises(PackageError, match="storage limit"):
        service(Files()).preflight(path)


def test_node_attachment_roundtrip_and_reexport(tmp_path):
    from pathlib import Path

    source_assets = tmp_path / "source-assets"
    source_assets.mkdir()
    blob = source_assets / "evidence.bin"
    blob.write_bytes(b"\x00\xff\x12" * 1024)
    source = Files()
    r = node()
    r["metadata"]["source_path"] = str(blob)
    AGFSStorage(config(), source).restore("nodes", [r])
    exporter = service(source)
    exporter.asset_root = source_assets
    package = tmp_path / "assets.tar.gz"
    original = exporter.export(package)
    target = Files()
    importer = service(target)
    importer.asset_root = tmp_path / "target-assets"
    result = importer.restore(package)
    assert result["data"] == original["data"]
    restored = list(AGFSStorage(config(), target).records("nodes"))[0]
    target_file = Path(restored["metadata"]["source_path"])
    assert target_file.parent == importer.asset_root
    assert target_file.read_bytes() == blob.read_bytes()
    assert importer.export(tmp_path / "again.tar.gz")["data"] == original["data"]


def test_source_attachment_must_be_inside_explicit_root(tmp_path):
    source = Files()
    r = node()
    r["metadata"]["source_path"] = str(tmp_path / "outside.bin")
    (tmp_path / "outside.bin").write_bytes(b"private")
    AGFSStorage(config(), source).restore("nodes", [r])
    exporter = service(source)
    exporter.asset_root = tmp_path / "allowed"
    with pytest.raises(PackageError, match="outside"):
        exporter.export(tmp_path / "rejected.tar.gz")
    assert not (tmp_path / "rejected.tar.gz").exists()


def test_pending_directory_summary_blocks_export(tmp_path):
    source = Files()
    AGFSStorage(config(), source).restore("nodes", [node()])
    path = "/local/accounts/acme/users/alice/memories/preferences/editor/.outbox/event.json"
    source.files[path] = json.dumps(
        {"event_type": "UPSERT_DIRECTORY", "status": "PENDING"}
    ).encode()
    with pytest.raises(PackageError, match="unfinished business operation"):
        service(source).export(tmp_path / "pending.tar.gz")
    assert not (tmp_path / "pending.tar.gz").exists()


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "layout", ["candidates", "example_evidence", "state.json", "state.json.bak"]
)
def test_agfs_skill_data_rejected_for_source_and_target(empty, layout):
    files = Files()
    root = "/local/.ogmemory/trace-store/accounts/account/"
    if layout.endswith((".json", ".bak")):
        files.files[root + layout] = json.dumps(
            {"candidates": {"id": {"state": "pending"}}}
        ).encode()
    else:
        files.files[root + layout + "/record.json"] = b"{}"
    with pytest.raises(PackageError, match="unsupported AGFS"):
        AGFSStorage(config(), files).check(empty=empty)


def test_agfs_trace_without_skill_data_is_not_rejected():
    files = Files()
    root = "/local/.ogmemory/trace-store/accounts/account/"
    files.files[root + "state.json"] = b'{"schema_version":5}'
    files.files[root + "events/session/event.json"] = b"{}"
    AGFSStorage(config(), files).check()


@pytest.mark.parametrize("weight", [0, 1, -0.0])
def test_relation_weight_uses_one_numeric_representation(tmp_path, weight):
    from service.migration.engine import Inventory

    edge = dict(
        from_uri=node()["uri"],
        to_uri=node()["uri"],
        relation_type="related_to",
        weight=weight,
        reason="same relation",
    )
    inv = Inventory(tmp_path / "numbers.sqlite")
    try:
        inv.add("relations", edge)
        original = inv.summary()
        inv.add("relations", {**edge, "weight": float(weight)})
        inv.add("relations", {**edge, "weight": int(weight)})
        assert inv.summary() == original
        assert len(list(inv.records("relations"))) == 1
        assert type(list(inv.records("relations"))[0]["weight"]) is float
        assert edge["weight"] is weight
    finally:
        inv.close()


@pytest.mark.parametrize(
    "actual,configured,rejected", [(384, 1024, True), (384, 384, False), (None, 1024, False)]
)
def test_preflight_checks_target_model_dimension_before_opening_index(
    tmp_path, actual, configured, rejected
):
    source = Files()
    AGFSStorage(config(), source).restore("nodes", [node()])
    package = tmp_path / "source.tar.gz"
    service(source).export(package)
    target = Files()
    importer = service(target)
    importer.cfg.embedding_provider = "st"
    importer.cfg.st_model = "local-model"
    importer.cfg.opengauss_dimension = configured
    importer.cfg.to_provider_config = lambda: SimpleNamespace(
        create_embedder=lambda: SimpleNamespace(dimension=actual), embedding_dimension=configured
    )
    opened = []

    def indexes(cfg):
        opened.append(True)
        return Indexes(cfg)

    importer.index_factory = indexes
    if rejected:
        with pytest.raises(PackageError, match="dimension 384.*dimension 1024"):
            importer.preflight(package)
        assert not opened
    else:
        report = importer.preflight(package)
        assert opened
        if actual is None:
            assert any("目标模型维度尚未验证" in n for n in report["compatibility"]["notices"])
    assert not target.files
    assert importer.cfg.opengauss_dimension == configured


def test_unknown_dimension_still_requires_probe_before_restore(tmp_path):
    source = Files()
    AGFSStorage(config(), source).restore("nodes", [node()])
    package = tmp_path / "source.tar.gz"
    service(source).export(package)
    target = Files()
    importer = service(target)

    class FailingProbe(Indexes):
        def probe(self):
            raise PackageError("embedding output count/dimension is invalid")

    importer.index_factory = FailingProbe
    importer.preflight(package)
    with pytest.raises(PackageError, match="embedding output count/dimension"):
        importer.restore(package)
    assert not target.files


@pytest.mark.parametrize("owner", ["users/alice", "agents/main"])
def test_write_receipt_namespace_is_runtime_only(tmp_path, owner):
    from service.migration.records import runtime_node

    uri = f"ctx://acme/{owner}/memories/.write_operations/" + "a" * 64
    assert runtime_node({"uri": uri})
    assert not runtime_node({"uri": f"ctx://acme/{owner}/memories/preferences/write_operations"})
    files = Files()
    path = "/local/accounts/acme/" + owner + "/memories/.write_operations/" + "a" * 64
    files.files[path + "/.meta.json"] = json.dumps(
        {"uri": uri, "category": "write_operation"}
    ).encode()
    exported = service(files).export(tmp_path / "receipt-only.tar.gz")
    assert exported["data"]["nodes"]["count"] == 0
    # Excluding source runtime state must not weaken the empty-target check.
    with pytest.raises(PackageError, match="not empty"):
        AGFSStorage(config(), files).check(empty=True)


def test_package_with_write_receipt_node_rejected_before_restore(tmp_path):
    receipt = {
        **node(),
        "uri": "ctx://acme/users/alice/memories/.write_operations/" + "a" * 64,
        "category": "write_operation",
        "context_type": "RESOURCE",
        "metadata": {"indexing_disabled": True},
    }
    package = tmp_path / "legacy-receipt.tar.gz"
    write_package(
        package,
        {"nodes": [receipt], "relations": [], "archives": []},
        source_backend="sql",
        source_version="legacy",
    )
    target = Files()
    with pytest.raises(PackageError, match="runtime-only"):
        service(target).restore(package)
    assert not target.files


def test_receipt_metadata_is_removed_without_losing_business_metadata(tmp_path):
    r = node()
    r["metadata"].update(
        memory_write_receipts={"old": "receipt"}, memory_write_projection_receipts={"old": "index"}
    )
    package = tmp_path / "legacy-metadata.tar.gz"
    write_package(
        package,
        {"nodes": [r], "relations": [], "archives": []},
        source_backend="sql",
        source_version="legacy",
    )
    target = Files()
    service(target).restore(package)
    restored = list(AGFSStorage(config(), target).records("nodes"))[0]
    assert restored["content"] == r["content"]
    assert restored["metadata"] == {"custom": True}


@pytest.mark.parametrize("slug", ["stage_results", "_system"])
def test_runtime_names_in_business_paths_are_preserved(slug):
    files = Files()
    adapter = AGFSStorage(config(), files)
    record = node()
    record["uri"] = record["uri"].replace("/editor", "/" + slug)
    adapter.restore("nodes", [record])
    assert list(adapter.records("nodes")) == [record]
    with pytest.raises(PackageError, match="not empty"):
        adapter.check(empty=True)


@pytest.mark.parametrize("field", ["provenance_ids", "grounding"])
def test_actual_trace_namespace_rejected_before_import_writes(tmp_path, field):
    r = node()
    reference = "ctx://acme/traces/session/turns/turn#event-1"
    r["metadata"][field] = [reference] if field == "provenance_ids" else [{"uri": reference}]
    path = tmp_path / "trace.tar.gz"
    write_package(
        path,
        {"nodes": [r], "relations": [], "archives": []},
        source_backend="agfs",
        source_version="test",
    )
    target = Files()
    with pytest.raises(PackageError, match="trace/tool evidence"):
        service(target).restore(path)
    assert not target.files


def test_shared_attachments_count_once_against_byte_budget(tmp_path, monkeypatch):
    import service.migration.assets as module

    monkeypatch.setattr(module, "MAX_ASSET_BYTES", 32)
    stage = tmp_path / "stage"
    stage.mkdir()
    first, duplicate, other = (tmp_path / name for name in ("first", "duplicate", "other"))
    first.write_bytes(b"x" * 20)
    duplicate.write_bytes(first.read_bytes())
    other.write_bytes(b"y" * 20)
    assets = module.Assets(tmp_path, stage)
    references = [
        assets.capture({"metadata": {"source_path": str(path)}})
        for path in (first, first, duplicate)
    ]
    assert references[0] == references[1] == references[2]
    assert assets.size == 20 and len(assets.files) == 1
    assert next(iter(assets.files.values())).read_bytes() == first.read_bytes()
    with pytest.raises(PackageError, match="byte limit"):
        assets.capture({"metadata": {"source_path": str(other)}})


def test_restore_reuses_one_validated_package_and_one_target_index(tmp_path):
    from unittest.mock import patch

    source = Files()
    AGFSStorage(config(), source).restore("nodes", [node()])
    package = tmp_path / "source.tar.gz"
    expected = service(source).export(package)["data"]
    target = Files()
    importer = service(target)
    opened = []

    def create_indexes(cfg):
        indexes = Indexes(cfg)
        opened.append(indexes)
        return indexes

    importer.index_factory = create_indexes
    with patch("service.migration.engine.open_package", wraps=open_package) as unpack:
        result = importer.restore(package)
    assert unpack.call_count == 1
    assert len(opened) == 1
    assert result["data"] == expected
    assert list(AGFSStorage(config(), target).records("nodes")) == [node()]


@pytest.mark.parametrize("changed", ["package", "target"])
def test_restore_rechecks_changes_after_preview(tmp_path, changed):
    source = Files()
    AGFSStorage(config(), source).restore("nodes", [node()])
    package = tmp_path / "source.tar.gz"
    service(source).export(package)
    target = Files()
    importer = service(target)
    importer.preflight(package)
    if changed == "package":
        package.write_bytes(package.read_bytes()[:-8])
    else:
        AGFSStorage(config(), target).restore("nodes", [node()])
    before = dict(target.files)
    with pytest.raises(PackageError):
        importer.restore(package)
    assert target.files == before


def test_restore_rechecks_model_configuration_after_preview(tmp_path):
    source = Files()
    AGFSStorage(config(), source).restore("nodes", [node()])
    package = tmp_path / "source.tar.gz"
    service(source).export(package)
    target = Files()
    importer = service(target)
    importer.cfg.embedding_provider = "st"
    importer.cfg.st_model = "local-model"
    importer.cfg.opengauss_dimension = 384
    importer.cfg.to_provider_config = lambda: SimpleNamespace(
        create_embedder=lambda: SimpleNamespace(dimension=384),
        embedding_dimension=importer.cfg.opengauss_dimension,
    )
    importer.preflight(package)
    importer.cfg.opengauss_dimension = 1024

    def unexpected_index(cfg):
        pytest.fail("invalid configuration reached index creation")

    importer.index_factory = unexpected_index
    with pytest.raises(PackageError, match="dimension 384.*dimension 1024"):
        importer.restore(package)
    assert not target.files


@pytest.mark.parametrize("endpoint", ["from_uri", "to_uri"])
@pytest.mark.parametrize("operation", ["preflight", "restore"])
def test_sql_graph_archive_endpoint_rejected_before_target_access(tmp_path, endpoint, operation):
    from service.migration.records import archive_uri

    edge = dict(
        from_uri=node()["uri"],
        to_uri=node()["uri"],
        relation_type="derived_from",
        weight=1.0,
        reason="history evidence",
    )
    edge[endpoint] = archive_uri(archive())
    package = tmp_path / "archive-edge.tar.gz"
    write_package(
        package,
        {"nodes": [node()], "archives": [archive()], "relations": [edge]},
        source_backend="agfs",
        source_version="test",
    )
    target = Files()
    importer = service(target)
    importer.cfg.graph_backend = "sql"

    def forbidden(cfg):
        pytest.fail("invalid graph endpoint reached target storage")

    importer.adapter_factory = forbidden
    with pytest.raises(PackageError, match="SQL graph projection requires node endpoints"):
        getattr(importer, operation)(package)
    assert not target.files


@pytest.mark.parametrize("graph", ["none", "sql"])
@pytest.mark.parametrize("endpoint", ["from_uri", "to_uri"])
def test_sql_relation_alias_rejected_before_target_access(tmp_path, graph, endpoint):
    directory = {**node(), "uri": node()["uri"].rsplit("/", 1)[0] + "/", "content": ""}
    edge = dict(
        from_uri=node()["uri"],
        to_uri=node()["uri"],
        relation_type="related_to",
        weight=1.0,
        reason="directory",
    )
    edge[endpoint] = directory["uri"].rstrip("/")
    package = tmp_path / "alias.tar.gz"
    write_package(
        package,
        {"nodes": [node(), directory], "relations": [edge], "archives": []},
        source_backend="agfs",
        source_version="test",
    )
    importer = service(Files())
    importer.cfg.storage_backend = "sql"
    importer.cfg.graph_backend = graph

    def forbidden(cfg):
        pytest.fail("alias reached target adapter")

    importer.adapter_factory = forbidden
    for operation in (importer.preflight, importer.restore):
        with pytest.raises(PackageError, match="exactly match stored URI"):
            operation(package)


@pytest.mark.parametrize("count", [8, 16])
def test_scoped_index_reads_decode_each_node_once(tmp_path, count):
    from unittest.mock import patch

    from service.migration.engine import Inventory
    from service.migration.rebuild import Rebuilder

    inv = Inventory(tmp_path / "inventory.sqlite")
    try:
        scopes = []
        for i in range(count):
            account, owner = f"account{i % 2}", f"user:u{i}"
            inv.add(
                "nodes",
                {
                    **node(),
                    "account_id": account,
                    "owner_space": owner,
                    "uri": f"ctx://{account}/users/u{i}/memories/preferences/editor",
                },
            )
            scopes.append((account, owner))
        inv.finish()
        rebuilder = Rebuilder.__new__(Rebuilder)
        with patch("service.migration.engine.json.loads", wraps=json.loads) as decode:
            for scope in scopes:
                records = list(rebuilder._records(inv, scope))
                assert records
                assert all(
                    (r.filters["account_id"], r.filters["owner_space"]) == scope for r in records
                )
            assert decode.call_count == count
        plan = inv.db.execute(
            "EXPLAIN QUERY PLAN SELECT body FROM records WHERE kind=? "
            "AND json_extract(body,'$.account_id')=? AND json_extract(body,'$.owner_space')=? ORDER BY key",
            ("nodes", *scopes[0]),
        ).fetchall()
        assert any("record_scope" in row[-1] for row in plan)
    finally:
        inv.close()


@pytest.mark.parametrize("count", [8, 16])
def test_agfs_edges_are_written_once_per_source(count):
    from unittest.mock import patch

    files = Files()
    adapter = AGFSStorage(config(), files)
    adapter.restore("nodes", [node()])
    edges = [
        dict(
            from_uri=node()["uri"],
            to_uri=node()["uri"],
            relation_type=f"related_{i}",
            weight=1.0,
            reason="edge",
        )
        for i in range(count)
    ]
    with (
        patch.object(files, "read", wraps=files.read) as read,
        patch.object(files, "write", wraps=files.write) as write,
    ):
        adapter.restore("relations", iter(edges))
        assert read.call_count == write.call_count == 1
    assert list(adapter.records("relations")) == edges


@pytest.mark.parametrize("source", ["alias", "other_node", "other_account"])
def test_agfs_relation_source_is_checked_by_physical_path(tmp_path, source):
    from fs.agfs_adapter.agfs_context_fs import uri_to_path

    files = Files()
    adapter = AGFSStorage(config(), files)
    directory = {**node(), "uri": node()["uri"].rsplit("/", 1)[0] + "/", "content": ""}
    adapter.restore("nodes", [directory, node()])
    from_uri = directory["uri"].rstrip("/")
    if source == "other_node":
        from_uri = node()["uri"]
    elif source == "other_account":
        from_uri = from_uri.replace("ctx://acme/", "ctx://other/")
    edge = dict(
        from_uri=from_uri,
        to_uri=node()["uri"],
        relation_type="related_to",
        weight=1.0,
        reason="directory",
    )
    path = config().agfs_mount_prefix + uri_to_path(directory["uri"]) + ".relations.json"
    files.write(path, json.dumps([edge]).encode())
    package = tmp_path / "alias.tar.gz"
    if source == "alias":
        service(files).export(package)
        target = Files()
        service(target).restore(package)
        assert list(AGFSStorage(config(), target).records("relations")) == [edge]
    else:
        with pytest.raises(PackageError, match="relation source differs from owning node"):
            service(files).export(package)
        assert not package.exists()


@pytest.mark.parametrize('oversize', ['text', 'metadata'])
def test_milvus_payload_rejected_before_content_restore(tmp_path, oversize):
    from providers.vector_index.milvus_index import MilvusVectorIndex
    from service.migration.rebuild import Rebuilder

    source, target = Files(), Files()
    row = node()
    if oversize == 'text':
        row['content'] = '🙂' * 100000
    else:
        # A long model identity exercises metadata added by the rebuild step.
        row['content'] = 'small'
    AGFSStorage(config(), source).restore('nodes', [row])
    package = tmp_path / 'source.tar.gz'
    service(source).export(package)
    vector = object.__new__(MilvusVectorIndex)
    vector._dimension = 8

    class CheckedIndexes(Indexes):
        validate_records = Rebuilder.validate_records
        _records = Rebuilder._records

        def __init__(self, cfg):
            self.vector = vector
            self.embedder = SimpleNamespace(model='m' * (65536 if oversize == 'metadata' else 8))
            self.provider = SimpleNamespace(embedding_dimension=8)

        def rebuild(self, inv, progress):
            pytest.fail('payload must be rejected before rebuilding')

    migration = service(target)
    migration.cfg.vector_db_type = 'milvus'
    migration.index_factory = CheckedIndexes
    for operation in (migration.preflight, migration.restore):
        with pytest.raises(PackageError, match=f'Milvus {oversize} exceeds'):
            operation(package)
        assert not target.files


def test_oversize_attachment_rejected_before_hashing(tmp_path, monkeypatch):
    from service.migration.assets import Assets

    blob = tmp_path / 'large.bin'
    with blob.open('wb') as stream:
        stream.truncate(9 * 1024**3)
    row = node()
    row['metadata']['source_path'] = str(blob)
    monkeypatch.setattr('hashlib.file_digest', lambda *a: pytest.fail('must reject before hashing'))
    with pytest.raises(PackageError, match='USTAR'):
        Assets(tmp_path, tmp_path).capture(row)


@pytest.mark.parametrize('kind', ['owner', 'session', 'slug', 'encoded_slug'])
@pytest.mark.parametrize('component_bytes', [255, 256])
def test_agfs_target_path_byte_capacity_before_restore(tmp_path, kind, component_bytes):
    from urllib.parse import quote

    component = '用' * 85 + ('x' if component_bytes == 256 else '')
    records = [node()]
    archives = []
    if kind == 'session':
        item = archive()
        item['session_id'] = component
        archives.append(item)
    else:
        item = node()
        if kind == 'owner':
            item['uri'] = item['uri'].replace('/alice/', '/' + component + '/')
            item['owner_space'] = 'user:' + component
        else:
            # Keep the encoded URI below the independent 512-character limit.
            slug = quote('用') + component[1:] if kind == 'encoded_slug' else component
            item['uri'] = item['uri'].rsplit('/', 1)[0] + '/' + slug
        records.append(item)
    package = tmp_path / 'sql-source.tar.gz'
    write_package(package, {'nodes': records, 'archives': archives, 'relations': []},
                  source_backend='sql', source_version='test')
    target = Files()
    migration = service(target)
    if component_bytes == 256:
        for operation in (migration.preflight, migration.restore):
            with pytest.raises(PackageError, match='AGFS target path component'):
                operation(package)
            assert not target.files
            assert not target.directories
    else:
        migration.preflight(package)
        migration.restore(package)
        assert len(list(AGFSStorage(config(), target).records('nodes'))) == len(records)
        assert len(list(AGFSStorage(config(), target).records('archives'))) == len(archives)


def test_sql_target_does_not_apply_agfs_path_byte_limit(tmp_path):
    from service.migration.engine import inventory
    from service.migration.storage import SQLStorage

    row = node()
    owner = '用' * 100
    row['uri'] = row['uri'].replace('/alice/', '/' + owner + '/')
    row['owner_space'] = 'user:' + owner
    with inventory(tmp_path / 'inventory.sqlite',
                   lambda kind: [row] if kind == 'nodes' else []) as inv:
        SQLStorage.__new__(SQLStorage).validate_restore(inv)
