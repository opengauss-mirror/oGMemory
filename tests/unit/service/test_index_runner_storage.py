from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from providers.unified_config import OgMemConfig
from scripts.run_index_service import _create_storage


def test_sql_runner_uses_sql_outbox_and_shared_pool():
    cfg = SimpleNamespace(
        storage_backend="sql", sql_connection_string="test-dsn",
        sql_pool_size=3, sql_dialect="postgresql",
    )
    with (
        patch("fs.sql_adapter.pool.SharedConnectionPool") as pool,
        patch("fs.sql_adapter.SQLContextFS") as fs,
        patch("commit.sql_outbox_store.SQLOutboxStore") as outbox,
    ):
        result = _create_storage(cfg)
    pool.assert_called_once_with("test-dsn", pool_size=3, dialect="postgresql")
    fs.assert_called_once_with(pool=pool.return_value)
    outbox.assert_called_once_with(pool=pool.return_value, fs=fs.return_value)
    assert result == (fs.return_value, outbox.return_value, pool.return_value)


def test_sql_runner_rejects_missing_dsn_instead_of_using_agfs():
    with pytest.raises(ValueError, match="storage.connection_string"):
        _create_storage(SimpleNamespace(storage_backend="sql", sql_connection_string=""))


@pytest.mark.parametrize("transport", ["http", "native"])
@pytest.mark.parametrize("mode,provider,hybrid", [
    ("default", "none", False),
    ("default", "opengauss_native", True),
    ("vector_free", "opengauss_native", False),
])
def test_agfs_runner_keeps_configured_mount_and_outbox(transport, mode, provider, hybrid):
    cfg = OgMemConfig(
        storage_backend="agfs", agfs_base_url="http://agfs:8080",
        agfs_mount_prefix="/memory", agfs_transport=transport,
        retrieval_mode=mode, retrieval_lexical_provider=provider,
        retrieval_hybrid_enabled=hybrid,
    )
    invalidate = Mock() if cfg.retrieval_uses_lexical_projection else None
    with (
        patch("fs.agfs_adapter.client.create_agfs_client") as client,
        patch("fs.agfs_adapter.agfs_context_fs.AGFSContextFS") as fs,
        patch("commit.outbox_store.OutboxStore") as outbox,
    ):
        result = _create_storage(cfg, invalidate_lexical_scopes=invalidate)
    client.assert_called_once_with(cfg)
    fs.assert_called_once_with(client=client.return_value, mount_prefix="/memory")
    outbox.assert_called_once_with(
        client=client.return_value, fs=fs.return_value, mount_prefix="/memory",
        invalidate_lexical_scopes=invalidate,
    )
    assert result == (fs.return_value, outbox.return_value, None)


@pytest.mark.parametrize("close_error", [False, True])
def test_shared_runner_closes_client_if_bridge_initialization_fails(monkeypatch, close_error):
    from scripts.run_index_service import _create_storage
    pool, client = MagicMock(), MagicMock()
    if close_error:
        client.close.side_effect = RuntimeError("client close failed")
    monkeypatch.setattr('fs.sql_adapter.pool.SharedConnectionPool', lambda *a, **kw: pool)
    monkeypatch.setattr('fs.agfs_adapter.client.create_agfs_client', lambda cfg: client)
    def fail(*a, **kw):
        raise RuntimeError('bridge schema/init failed')
    monkeypatch.setattr('fs.agfs_adapter.space_bridge.AGFSSpaceBridge', fail)
    cfg = SimpleNamespace(storage_backend='agfs', shared_spaces_enabled=True,
                          sql_connection_string='test', sql_pool_size=1,
                          sql_dialect='opengauss', agfs_mount_prefix='/local')
    with pytest.raises(RuntimeError):
        _create_storage(cfg)
    pool.close_all.assert_called_once()
    client.close.assert_called_once()



def test_shared_runner_registers_bridge_as_only_owner_of_stage_pool(monkeypatch):
    pool, client, bridge = Mock(), Mock(), Mock()
    monkeypatch.setattr('fs.sql_adapter.pool.SharedConnectionPool', lambda *a, **kw: pool)
    monkeypatch.setattr('fs.agfs_adapter.client.create_agfs_client', lambda cfg: client)
    monkeypatch.setattr('fs.agfs_adapter.space_bridge.AGFSSpaceBridge', lambda *a: bridge)
    monkeypatch.setattr('commit.sql_outbox_store.SQLOutboxStore', lambda **kw: object())
    cfg = SimpleNamespace(storage_backend='agfs', shared_spaces_enabled=True,
                          sql_connection_string='test', sql_pool_size=1,
                          sql_dialect='opengauss', agfs_mount_prefix='/local')
    registered = []
    guard = Mock()

    _create_storage(cfg, register_cleanup=registered.append, guard=guard)

    assert registered == [pool, client, bridge]
    guard.assert_called_once_with(client)
