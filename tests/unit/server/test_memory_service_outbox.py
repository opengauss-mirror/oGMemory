"""Unit tests for MemoryService outbox worker wiring."""

from unittest.mock import Mock, patch

import pytest

from core.lexical import HybridLexicalIndex
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def test_start_outbox_worker_uses_sql_notify_listener(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=localhost dbname=test",
    )
    service = MemoryService(config=cfg)

    fake_fs = Mock()
    fake_outbox = Mock()
    fake_outbox.supports_batch_claim = True

    monkeypatch.setattr(service, "_get_context_fs", lambda: fake_fs)
    monkeypatch.setattr(service, "_get_outbox_store", lambda fs: fake_outbox)

    fake_listener = Mock()
    fake_listener_cls = Mock(return_value=fake_listener)
    monkeypatch.setattr(
        "index.sql_notify_listener.SQLNotifyListener",
        fake_listener_cls,
    )
    projection = Mock()
    monkeypatch.setattr(
        service, "_get_shared_projection_index", Mock(return_value=projection)
    )

    thread_state = {}

    class FakeThread:
        def __init__(self, target, daemon, name):
            thread_state["target"] = target
            thread_state["daemon"] = daemon
            thread_state["name"] = name

        def start(self):
            thread_state["started"] = True

    monkeypatch.setattr("server.memory_service.threading.Thread", FakeThread)

    service._start_outbox_worker()

    fake_listener_cls.assert_called_once()
    worker = fake_listener_cls.call_args.kwargs["worker"]
    assert worker._vector_index is projection
    assert service._outbox_listener is fake_listener
    assert thread_state["target"] == fake_listener.run_forever
    assert thread_state["daemon"] is True
    assert thread_state["name"] == "outbox-listener"
    assert thread_state["started"] is True


def test_shared_sql_pool_receives_configured_dialect(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_dialect="opengauss",
        sql_connection_string="host=storage dbname=test",
    )
    service = MemoryService(config=cfg)
    pool = Mock()
    pool_class = Mock(return_value=pool)
    monkeypatch.setattr("fs.sql_adapter.pool.SharedConnectionPool", pool_class)

    assert service._get_shared_sql_pool() is pool
    pool_class.assert_called_once_with(
        connection_string="host=storage dbname=test",
        pool_size=5,
        dialect="opengauss",
        deadline=None,
    )


@pytest.mark.parametrize("dialect", ["opengauss", "postgresql"])
def test_vector_index_shares_configured_sql_pool(monkeypatch, dialect):
    dsn = "host=storage dbname=memory"
    service = MemoryService(config=OgMemConfig(
        provider="mock", vector_db_type="opengauss", storage_backend="sql",
        sql_dialect=dialect, sql_connection_string=dsn,
        opengauss_connection_string=dsn,
    ))
    pool = Mock()
    index = Mock()
    monkeypatch.setattr(service, "_get_shared_sql_pool", lambda: pool)
    factory = Mock(return_value=index)
    monkeypatch.setattr(service._provider_cfg, "create_vector_index", factory)

    assert service._get_shared_vector_index() is index
    assert service._get_shared_vector_index() is index
    factory.assert_called_once_with(pool=pool)


def test_instance_can_disable_class_batch_claim_capability():
    class Store:
        supports_batch_claim = True

        def __init__(self):
            self.supports_batch_claim = False

    assert MemoryService._supports_batch_claim(Store()) is False


def test_shared_providers_use_instance_config(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=localhost dbname=test",
    )
    service = MemoryService(config=cfg)
    usage_store = object()
    service._model_usage_store = usage_store

    fake_index = Mock()
    fake_embedder = Mock()
    create_vector_index = Mock(return_value=fake_index)
    create_embedder = Mock(return_value=fake_embedder)
    monkeypatch.setattr(service._provider_cfg, "create_vector_index", create_vector_index)
    monkeypatch.setattr(service._provider_cfg, "create_embedder", create_embedder)

    with patch(
        "server.memory_service.ProviderConfig.from_env",
        side_effect=AssertionError("should not consult global config"),
    ):
        assert service._get_shared_vector_index() is fake_index
        assert service._get_shared_embedder() is fake_embedder

    create_vector_index.assert_called_once_with()
    create_embedder.assert_called_once_with()
    fake_embedder.token_tracker.set_usage_store.assert_called_once_with(usage_store)


def test_hybrid_initialization_publishes_one_shared_projection(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=storage dbname=test",
        retrieval_hybrid_enabled=True,
        opengauss_connection_string="host=gauss dbname=test",
    )
    service = MemoryService(config=cfg)
    vector = Mock()
    lexical = Mock(spec=HybridLexicalIndex)
    monkeypatch.setattr(service._provider_cfg, "create_vector_index", Mock(return_value=vector))
    monkeypatch.setattr(
        "providers.lexical_index.create_lexical_index", Mock(return_value=lexical)
    )

    assert service._get_shared_vector_index() is vector
    projection = service._get_shared_projection_index()

    assert projection.vector_index is vector
    assert projection.lexical_index is lexical
    assert service._lexical_index is lexical


def test_hybrid_backfill_coordinator_is_created_once(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=storage dbname=test",
        retrieval_hybrid_enabled=True,
    )
    service = MemoryService(config=cfg)
    service._vector_index = Mock()
    service._lexical_index = Mock()
    pool = Mock()
    scanner = Mock()
    backfill = Mock()
    monkeypatch.setattr(service, "_get_shared_sql_pool", Mock(return_value=pool))

    with (
        patch(
            "fs.sql_adapter.node_scanner.SQLCanonicalNodeScanner",
            Mock(return_value=scanner),
        ) as scanner_factory,
        patch(
            "providers.lexical_backfill.LexicalBackfill",
            Mock(return_value=backfill),
        ) as backfill_factory,
    ):
        assert service._get_lexical_backfill() is backfill
        assert service._get_lexical_backfill() is backfill

    scanner_factory.assert_called_once_with(pool)
    backfill_factory.assert_called_once_with(
        scanner,
        service._lexical_index,
        page_size=500,
    )


def test_required_lexical_init_failure_is_retryable(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=storage dbname=test",
        retrieval_hybrid_enabled=True,
        opengauss_connection_string="host=gauss dbname=test",
    )
    service = MemoryService(config=cfg)
    first_vector, second_vector = Mock(), Mock()
    lexical = Mock(spec=HybridLexicalIndex)
    monkeypatch.setattr(
        service._provider_cfg,
        "create_vector_index",
        Mock(side_effect=[first_vector, second_vector]),
    )
    lexical_factory = Mock(side_effect=[RuntimeError("temporary"), lexical])
    monkeypatch.setattr("providers.lexical_index.create_lexical_index", lexical_factory)

    with pytest.raises(RuntimeError, match="temporary"):
        service._get_shared_vector_index()

    assert service._vector_index is None
    first_vector.close.assert_called_once_with()
    assert service._get_shared_vector_index() is second_vector
    assert service._lexical_index is lexical
    assert lexical_factory.call_count == 2


def test_lexical_init_error_is_not_masked_by_cleanup_failure(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=storage dbname=test",
        retrieval_hybrid_enabled=True,
        opengauss_connection_string="host=gauss dbname=test",
    )
    service = MemoryService(config=cfg)
    vector = Mock()
    vector.close.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(service._provider_cfg, "create_vector_index", Mock(return_value=vector))
    monkeypatch.setattr(
        "providers.lexical_index.create_lexical_index",
        Mock(side_effect=RuntimeError("lexical unavailable")),
    )

    with pytest.raises(RuntimeError, match="lexical unavailable"):
        service._get_shared_vector_index()


def test_projection_failure_marks_affected_lexical_scope_dirty(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=storage dbname=test",
        retrieval_hybrid_enabled=True,
        opengauss_connection_string="host=gauss dbname=test",
    )
    service = MemoryService(config=cfg)
    vector = Mock()
    lexical = Mock(spec=HybridLexicalIndex)
    lexical.upsert_many.side_effect = RuntimeError("write failed")
    monkeypatch.setattr(service._provider_cfg, "create_vector_index", Mock(return_value=vector))
    monkeypatch.setattr(
        "providers.lexical_index.create_lexical_index", Mock(return_value=lexical)
    )
    projection = service._get_shared_projection_index()
    record = Mock(
        id="r1",
        uri="ctx://acct/users/u1/memories/profile",
        level=2,
        text="text",
        metadata={},
        filters={"account_id": "acct", "owner_space": "user_u1"},
    )

    with pytest.raises(RuntimeError, match="write failed"):
        projection.upsert([record])

    lexical.mark_scopes_dirty.assert_called_once_with((("acct", "user_u1"),))


def test_disabled_hybrid_does_not_initialize_lexical_provider(monkeypatch):
    service = MemoryService(config=OgMemConfig(provider="mock", vector_db_type="memory"))
    lexical_factory = Mock(side_effect=AssertionError("lexical provider must stay disabled"))
    monkeypatch.setattr("providers.lexical_index.create_lexical_index", lexical_factory)

    service._get_shared_vector_index()

    lexical_factory.assert_not_called()


def test_shutdown_stops_listener_and_joins_thread():
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=localhost dbname=test",
    )
    service = MemoryService(config=cfg)

    listener = Mock()
    thread = Mock()
    thread.is_alive.side_effect = [True, False]

    service._outbox_listener = listener
    service._outbox_thread = thread

    service.shutdown(join_timeout=0.25)

    listener.stop.assert_called_once_with()
    thread.join.assert_called_once_with(timeout=0.25)
    assert service._outbox_listener is None
    assert service._outbox_thread is None


def test_shutdown_retains_resources_when_outbox_thread_does_not_stop():
    service = MemoryService(config=OgMemConfig(provider="mock", vector_db_type="memory"))
    listener = Mock()
    thread = Mock()
    thread.is_alive.return_value = True
    projection = Mock()
    service._outbox_listener = listener
    service._outbox_thread = thread
    service._projection_index = projection
    service._lexical_index = Mock()
    service._vector_index = Mock()

    service.shutdown(join_timeout=0.01)

    listener.stop.assert_called_once_with()
    thread.join.assert_called_once_with(timeout=0.01)
    projection.close.assert_not_called()
    assert service._outbox_listener is listener
    assert service._outbox_thread is thread
    assert service._projection_index is projection


def test_shutdown_retains_hybrid_resources_when_backfill_does_not_stop():
    service = MemoryService(
        config=OgMemConfig(
            provider="mock",
            vector_db_type="memory",
            storage_backend="sql",
            sql_connection_string="host=localhost dbname=test",
        )
    )
    backfill = Mock()
    backfill.close.return_value = False
    projection = Mock()
    lexical = Mock()
    vector = Mock()
    service._lexical_backfill = backfill
    service._projection_index = projection
    service._lexical_index = lexical
    service._vector_index = vector

    service.shutdown(join_timeout=0.01)

    backfill.close.assert_called_once_with(timeout=0.01)
    projection.close.assert_not_called()
    assert service._lexical_backfill is backfill
    assert service._projection_index is projection


def test_successful_hybrid_shutdown_stops_backfill_before_projection():
    service = MemoryService(config=OgMemConfig(provider="mock", vector_db_type="memory"))
    calls = []
    backfill = Mock()
    backfill.close.side_effect = lambda timeout: calls.append(("backfill", timeout)) or True
    projection = Mock()
    projection.close.side_effect = lambda timeout: calls.append(("projection", timeout)) or True
    service._lexical_backfill = backfill
    service._projection_index = projection
    service._lexical_index = Mock()
    service._vector_index = Mock()

    service.shutdown(join_timeout=0.25)

    assert calls == [("backfill", 0.25), ("projection", 0.25)]
    assert service._lexical_backfill is None
    assert service._projection_index is None
    assert service._lexical_index is None
    assert service._vector_index is None


def test_health_skips_agfs_probe_in_sql_mode(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=localhost dbname=test",
    )
    service = MemoryService(config=cfg)

    agfs_client = Mock(side_effect=AssertionError("AGFS probe should be skipped"))
    monkeypatch.setattr("server.memory_service.AGFSClient", agfs_client)

    # Mock the SQL pool to avoid actual connection attempts
    mock_pool = Mock()
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_pool.get_connection.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
    mock_conn.rollback = Mock()
    monkeypatch.setattr(service, "_get_shared_sql_pool", Mock(return_value=mock_pool))

    health = service.health()

    assert health["status"] == "ok"
    assert health["storage_backend"] == "sql"
    assert health["agfs"] is False
    assert health["sql"] == "connected"
    assert health["sql_dialect"] == "postgresql"
    agfs_client.assert_not_called()


def test_hybrid_health_initializes_backfill_before_first_read(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="sql",
        sql_connection_string="host=localhost dbname=test",
        retrieval_hybrid_enabled=True,
    )
    service = MemoryService(config=cfg)
    lexical = Mock()
    lexical.health.return_value = {"available": True}
    service._lexical_index = lexical
    backfill = Mock()
    backfill.health.return_value = {
        "queued": 0,
        "running": False,
        "at_capacity": False,
        "cooldown_failures": 0,
        "last_errors": [],
        "stopping": False,
    }
    monkeypatch.setattr(service, "_get_shared_vector_index", Mock(return_value=Mock()))
    backfill_factory = Mock(return_value=backfill)
    monkeypatch.setattr(service, "_get_lexical_backfill", backfill_factory)
    pool = Mock()
    connection = Mock()
    cursor = Mock()
    connection.cursor.return_value.__enter__ = Mock(return_value=cursor)
    connection.cursor.return_value.__exit__ = Mock(return_value=False)
    pool.get_connection.return_value = connection
    monkeypatch.setattr(service, "_get_shared_sql_pool", Mock(return_value=pool))

    health = service.health()

    assert health["status"] == "ok"
    assert health["lexical_backfill"] == backfill.health.return_value
    backfill_factory.assert_called_once_with()

    backfill.health.return_value = {
        **backfill.health.return_value,
        "stopping": True,
    }

    stopping_health = service.health()

    assert stopping_health["status"] == "degraded"
    assert stopping_health["error"] == "lexical historical projection is not ready"
    assert backfill_factory.call_count == 2


def test_health_probes_agfs_in_agfs_mode(monkeypatch):
    cfg = OgMemConfig(
        provider="mock",
        vector_db_type="memory",
        storage_backend="agfs",
        agfs_base_url="http://localhost:1833",
    )
    service = MemoryService(config=cfg)

    client = Mock()
    client.ls.return_value = []
    agfs_client = Mock(return_value=client)
    monkeypatch.setattr("server.memory_service.AGFSClient", agfs_client)

    health = service.health()

    assert health["status"] == "ok"
    assert health["storage_backend"] == "agfs"
    assert health["agfs"] is True
    agfs_client.assert_called_once_with(api_base_url="http://localhost:1833")
    client.ls.assert_called_once_with("/")


@pytest.mark.parametrize('counts,visible', [
    ({'total': 0, 'failed': 1}, False),
    ({'total': 0, 'failed': 0}, True),
    ({'total': 1, 'failed': 0}, False),
    ({'total': 0}, True),  # Legacy AGFS counter shape.
    ({'failed': 0}, False),
])
def test_code_index_visibility_rejects_dead_letters(monkeypatch, counts, visible):
    from core.models import RequestContext

    service = MemoryService(config=OgMemConfig(provider='mock', vector_db_type='memory'))
    fs, outbox = Mock(), Mock()
    outbox.count_unfinished.return_value = counts
    monkeypatch.setattr(service, '_get_context_fs', lambda: fs)
    monkeypatch.setattr(service, '_get_outbox_store', lambda _: outbox)
    ctx = RequestContext(account_id='acme', user_id='u', agent_id='a', session_id='s', trace_id='t')
    assert service._code_workspace_index_visible(ctx) is visible
    outbox.count_unfinished.assert_called_once_with('acme')
