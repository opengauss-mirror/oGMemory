"""Failure-path coverage for deployment preflight and cached-client readiness."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from providers.unified_config import OgMemConfig
from scripts import check_milvus
from server.memory_service import MemoryService


def probe_config(monkeypatch):
    index = Mock()
    records = []
    stored = {}

    def upsert(value):
        records.extend(value)
        stored.update((record.id, record) for record in value)

    index.upsert.side_effect = upsert
    index.get_record.side_effect = stored.get
    index.delete.side_effect = lambda ids: [stored.pop(key, None) for key in ids]
    index.search_by_vector.side_effect = lambda *a: [SimpleNamespace(uri=records[0].uri)]
    config = SimpleNamespace(embedding_dimension=3)
    monkeypatch.setattr(check_milvus, "create_vector_index", Mock(return_value=index))
    return config, index, records


def test_preflight_checks_read_search_delete_and_removes_only_its_own_record(monkeypatch):
    config, index, records = probe_config(monkeypatch)
    check_milvus.check_milvus(config)
    assert len(records) == 1
    check_milvus.create_vector_index.assert_called_once_with(config)
    record = records[0]
    assert record.filters["account_id"].startswith("ogmemory-preflight-")
    assert record.uri.startswith(f"ctx://{record.filters['account_id']}/")
    index.search_by_vector.assert_called_once_with(
        [1., 0., 0.], {"account_id": record.filters["account_id"]}, 1,
    )
    assert index.delete.call_count == 2
    assert all(call.args == ([record.id],) for call in index.delete.call_args_list)
    assert index.get_record.call_count == 2
    index.close.assert_called_once_with()


@pytest.mark.parametrize("operation", ["upsert", "read", "search", "delete", "cleanup"])
def test_preflight_fails_closed_and_closes_client(operation, monkeypatch):
    config, index, records = probe_config(monkeypatch)
    if operation == "upsert":
        index.upsert.side_effect = PermissionError("write denied")
    elif operation == "read":
        index.get_record.side_effect = [None, None]
    elif operation == "search":
        index.search_by_vector.side_effect = PermissionError("search denied")
    elif operation == "delete":
        index.delete.side_effect = PermissionError("delete denied")
    else:
        index.get_record.side_effect = lambda _: records[0]
    with pytest.raises((PermissionError, RuntimeError)):
        check_milvus.check_milvus(config)
    assert index.delete.call_count == (1 if operation == "delete" else 2)
    if operation == "delete":
        index.upsert.assert_not_called()
    index.close.assert_called_once_with()


def test_preflight_reports_orphan_if_delete_permission_is_revoked(monkeypatch, capsys):
    config, index, records = probe_config(monkeypatch)
    index.delete.side_effect = [None, PermissionError("private-token")]
    monkeypatch.setattr(check_milvus.logging, "disable", lambda *a: None)
    monkeypatch.setenv("VECTOR_DB_DIMENSION", "3")
    monkeypatch.setenv("MILVUS_URI", "http://user:private-token@localhost:19530")
    monkeypatch.setenv("MILVUS_COLLECTION", "test")
    monkeypatch.setenv("MILVUS_DB_NAME", "default")
    monkeypatch.setenv("MILVUS_TIMEOUT", "5")
    assert check_milvus.main() == 1
    error = capsys.readouterr().err
    assert "stage=cleanup" in error
    assert f"record_id={records[0].id}" in error
    assert f"account_id={records[0].filters['account_id']}" in error
    assert "private-token" not in error
    assert "localhost" not in error
    index.close.assert_called_once_with()


def test_preflight_cli_uses_actual_environment_and_does_not_echo_credentials(monkeypatch, capsys):
    values = {"VECTOR_DB_DIMENSION": "1024", "MILVUS_URI": "https://selected.example:19530",
              "MILVUS_TOKEN": "private-token", "MILVUS_COLLECTION": "selected",
              "MILVUS_DB_NAME": "selected_db", "MILVUS_TIMEOUT": "7.5"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(check_milvus.logging, "disable", lambda *a: None)
    probe = Mock(side_effect=PermissionError("private-token"))
    monkeypatch.setattr(check_milvus, "check_milvus", probe)
    assert check_milvus.main() == 1
    config = probe.call_args.args[0]
    assert (config.milvus_uri, config.milvus_token, config.milvus_collection_name,
            config.milvus_db_name, config.milvus_timeout, config.embedding_dimension) == (
                values["MILVUS_URI"], "private-token", "selected", "selected_db", 7.5, 1024,
            )
    assert "private-token" not in capsys.readouterr().err


@pytest.mark.parametrize("storage", ["agfs", "sql"])
def test_http_readiness_rechecks_cached_milvus_and_recovers(monkeypatch, storage):
    cfg = OgMemConfig(provider="mock", vector_db_type="milvus", storage_backend=storage,
                      sql_connection_string="host=unused", account_id="selected-account")
    service = MemoryService(config=cfg)
    monkeypatch.setattr("server.memory_service._HAS_AGFS", True)
    monkeypatch.setattr("server.memory_service.AGFSClient", Mock())
    monkeypatch.setattr(service, "_get_shared_sql_pool", Mock(return_value=MagicMock()))
    index = Mock()
    index.count.side_effect = [0, ConnectionError("private-token"), 0]
    monkeypatch.setattr(service, "_get_shared_vector_index", Mock(return_value=index))
    app = import_module("server.app")
    monkeypatch.setattr(app, "_get_service", lambda: service)
    with app.app.test_client() as client:
        for status, available in [(200, True), (503, False), (200, True)]:
            response = client.get("/api/v1/health")
            assert response.status_code == status
            assert "private-token" not in response.get_data(as_text=True)
            assert response.json["vector_db"]["available"] == available
    assert index.count.call_count == 3
    assert all(call.args == ({"account_id": "selected-account"},)
               for call in index.count.call_args_list)


def test_health_rejects_initial_milvus_connection_failure(monkeypatch):
    service = MemoryService(config=OgMemConfig(provider="mock", vector_db_type="milvus"))
    monkeypatch.setattr("server.memory_service._HAS_AGFS", True)
    monkeypatch.setattr("server.memory_service.AGFSClient", Mock())
    monkeypatch.setattr(service, "_get_shared_vector_index", Mock(side_effect=PermissionError("denied")))
    result = service.health()
    assert result["status"] == "error"
    assert result["vector_db"]["available"] is False


def test_preflight_import_does_not_read_unexpanded_application_yaml(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    config = tmp_path / "template.yaml"
    config.write_text("service:\n  http_port: ${OGMEM_HTTP_PORT}\n")
    completed = subprocess.run(
        [sys.executable, "-c", "import scripts.check_milvus"],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "OGMEM_CONFIG": str(config)},
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
