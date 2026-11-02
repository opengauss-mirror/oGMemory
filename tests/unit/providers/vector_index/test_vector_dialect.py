"""Vector constructor and provider factory dialect selection."""
from unittest.mock import MagicMock

import pytest

from fs.sql_adapter.pool import SharedConnectionPool
from providers.config import ProviderConfig
from providers.vector_index.opengauss_index import OpenGaussVectorIndex


@pytest.mark.parametrize("dialect", ["opengauss", "postgresql"])
def test_standalone_bootstrap_uses_selected_dialect(monkeypatch, dialect):
    connection = MagicMock()
    monkeypatch.setattr(OpenGaussVectorIndex, "_get_connection", lambda self: connection)
    monkeypatch.setattr(OpenGaussVectorIndex, "_return_connection", lambda self, conn: None)
    kwargs = {} if dialect == "opengauss" else {"dialect": dialect}
    index = OpenGaussVectorIndex("host=storage dbname=memory", dimension=3, **kwargs)
    statements = connection.cursor.return_value.__enter__.return_value.execute.call_args_list
    first = statements[0].args[0].strip()
    assert index.sql_dialect.name == dialect
    expected = "CREATE TABLE" if dialect == "opengauss" else "CREATE EXTENSION"
    assert first.startswith(expected)


def test_provider_factory_defaults_to_opengauss(monkeypatch):
    monkeypatch.setattr(OpenGaussVectorIndex, "_ensure_table", lambda self: None)
    config = ProviderConfig(vector_db_type="opengauss", opengauss_connection_string="host=storage dbname=memory")
    index = config.create_vector_index()
    assert index.sql_dialect.name == "opengauss"


@pytest.mark.parametrize("dialect", ["opengauss", "postgresql"])
def test_provider_uses_explicit_shared_pool_dialect(monkeypatch, dialect):
    monkeypatch.setattr(OpenGaussVectorIndex, "_ensure_table", lambda self: None)
    pool = SharedConnectionPool("host=storage dbname=memory", dialect=dialect)
    index = ProviderConfig(vector_db_type="opengauss").create_vector_index(pool=pool)
    assert index.sql_dialect is pool.sql_dialect
    assert index._shared_pool is pool


@pytest.mark.parametrize("backend,dialect", [("opengauss", "opengauss"), ("pgvector", "postgresql")])
def test_m3_backend_selects_sql_dialect(monkeypatch, backend, dialect):
    from types import SimpleNamespace
    from m3.vector_index import build_m3_vector_index

    monkeypatch.setattr(OpenGaussVectorIndex, "_ensure_table", lambda self: None)
    runtime = SimpleNamespace(vector_index_backend=backend,
                              vector_index_connection_string="host=storage dbname=memory",
                              vector_index_table_name="vectors", vector_index_pool_size=2,
                              embedding_dimension=3)
    assert build_m3_vector_index(runtime).sql_dialect.name == dialect


@pytest.mark.parametrize("source", ["constructor", "yaml", "env", "vector_dsn_fallback"])
@pytest.mark.parametrize("engine", ["opengauss", "postgresql"])
def test_omitted_storage_dialect_reaches_shared_engine(monkeypatch, source, engine):
    from threading import RLock
    from providers import unified_config as uc
    from server.memory_service import MemoryService

    dsn = "host=127.0.0.1 dbname=og"
    for key in ("SQL_DIALECT", "VECTOR_DB_TYPE", "VECTOR_DB_CONNECTION_STRING",
                "OPENGAUSS_CONNECTION_STRING", "STORAGE_DB_CONNECTION_STRING", "SQL_CONNECTION_STRING"):
        monkeypatch.delenv(key, raising=False)
    data = {"storage": {"backend": "sql", "connection_string": dsn},
            "vector_db": {"type": "opengauss", "connection_string": dsn}}
    if source == "constructor":
        config = uc.OgMemConfig(storage_backend="sql", vector_db_type="opengauss",
                               sql_connection_string=dsn, opengauss_connection_string=dsn)
    else:
        if source == "env":
            data = {}
            monkeypatch.setenv("STORAGE_BACKEND", "sql")
            monkeypatch.setenv("VECTOR_DB_TYPE", "opengauss")
            monkeypatch.setenv("VECTOR_DB_CONNECTION_STRING", dsn)
            monkeypatch.setenv("STORAGE_DB_CONNECTION_STRING", dsn)
        elif source == "vector_dsn_fallback":
            data["storage"].pop("connection_string")
        monkeypatch.setattr(uc, "_load_yaml", lambda path: data)
        config = uc.OgMemConfig.load("unused.yaml")
    service = MemoryService.__new__(MemoryService)
    service._cfg, service._sql_pool, service._init_lock = config, None, RLock()
    connection = MagicMock()
    connection.poll.return_value = 0
    connection.cursor.return_value.__enter__.return_value.fetchone.return_value = (engine + " 16.0",)
    monkeypatch.setattr("fs.sql_adapter.pool.psycopg2.connect", lambda *a, **kw: connection)
    pool = service._get_shared_sql_pool()
    monkeypatch.setattr(OpenGaussVectorIndex, "_ensure_table", lambda self: None)
    index = ProviderConfig.from_ogmem_config(config).create_vector_index(pool=pool)
    assert config.sql_dialect == "auto"
    assert pool.sql_dialect.name == index.sql_dialect.name == engine
    connection.close.assert_called_once()
    assert index._shared_pool is pool


@pytest.mark.parametrize("storage_dsn,explicit,expected", [
    ("host=shared dbname=og", "postgresql", "postgresql"),
    ("host=shared dbname=og", "opengauss", "opengauss"),
    ("host=separate dbname=pg", None, "postgresql"),
])
def test_storage_dialect_resolution_preserves_explicit_and_separate_backends(storage_dsn, explicit, expected):
    from providers.unified_config import OgMemConfig

    config = OgMemConfig(vector_db_type="opengauss", opengauss_connection_string="host=shared dbname=og",
                         sql_connection_string=storage_dsn, sql_dialect=explicit)
    assert config.sql_dialect == expected


@pytest.mark.parametrize("engine", ["opengauss", "postgresql"])
@pytest.mark.parametrize("dialect", [None, "auto", "AUTO", " Auto "])
def test_constructor_inherits_shared_pool_dialect(monkeypatch, engine, dialect):
    monkeypatch.setattr(OpenGaussVectorIndex, "_ensure_table", lambda self: None)
    pool = SharedConnectionPool("dbname=memory", dialect=engine)
    index = OpenGaussVectorIndex(pool=pool, dialect=dialect)
    assert index.sql_dialect is pool.sql_dialect


@pytest.mark.parametrize("source", ["yaml", "env", "storage"])
def test_standalone_factory_keeps_explicit_postgresql(monkeypatch, source):
    from providers import unified_config as uc

    data = {"vector_db": {"type": "opengauss", "connection_string": "dbname=memory"}}
    if source == "storage":
        data["storage"] = {"dialect": "postgresql"}
    elif source == "env":
        monkeypatch.setenv("VECTOR_DB_DIALECT", "postgresql")
    else:
        data["vector_db"]["dialect"] = "postgresql"
    monkeypatch.setattr(uc, "_load_yaml", lambda path: data)
    monkeypatch.setattr(OpenGaussVectorIndex, "_ensure_table", lambda self: None)
    config = uc.OgMemConfig.load("unused.yaml")
    provider = ProviderConfig.from_ogmem_config(config)
    assert provider.create_vector_index().sql_dialect.name == "postgresql"
