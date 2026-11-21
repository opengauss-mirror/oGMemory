from types import SimpleNamespace

import pytest

from providers.lexical_index import factory, register_lexical_provider


def _config(provider="opengauss_native", options=None):
    return SimpleNamespace(
        retrieval_lexical_provider=provider,
        retrieval_lexical_options=options or {},
        sql_connection_string="host=sql dbname=ogmemory",
        opengauss_connection_string="host=gauss dbname=ogmemory",
    )


def test_factory_routes_open_gauss_connection_and_options(monkeypatch):
    constructed = {}

    class FakeOpenGaussLexicalIndex:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    import providers.lexical_index.native_sql_index as native

    monkeypatch.setattr(native, "OpenGaussLexicalIndex", FakeOpenGaussLexicalIndex)
    factory.create_lexical_index(
        _config(
            options={
                "table_name": "lexical_custom",
                "bm25_topk": 25,
                "max_query_chars": 4096,
            }
        )
    )

    assert constructed["connection_string"] == "host=gauss dbname=ogmemory"
    assert constructed["table_name"] == "lexical_custom"
    assert constructed["bm25_topk"] == 25
    assert constructed["max_query_chars"] == 4096


def test_factory_allows_explicit_lexical_connection(monkeypatch):
    constructed = {}

    class FakeOpenGaussLexicalIndex:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    import providers.lexical_index.native_sql_index as native

    monkeypatch.setattr(native, "OpenGaussLexicalIndex", FakeOpenGaussLexicalIndex)
    factory.create_lexical_index(
        _config(options={"connection_string": "host=override dbname=ogmemory"})
    )

    assert constructed["connection_string"] == "host=override dbname=ogmemory"


def test_factory_falls_back_to_sql_connection(monkeypatch):
    constructed = {}

    class FakeOpenGaussLexicalIndex:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    import providers.lexical_index.native_sql_index as native

    monkeypatch.setattr(native, "OpenGaussLexicalIndex", FakeOpenGaussLexicalIndex)
    config = _config()
    config.opengauss_connection_string = None
    factory.create_lexical_index(config)

    assert constructed["connection_string"] == "host=sql dbname=ogmemory"


def test_factory_rejects_unrelated_provider():
    with pytest.raises(
        ValueError,
        match="unsupported lexical provider 'memory'; available providers: opengauss_native, sqlite_fts",
    ):
        factory.create_lexical_index(_config(provider="memory"))


def test_factory_creates_shared_sqlite_projection(tmp_path):
    config = _config(provider="sqlite_fts", options={"path": str(tmp_path / "index.db")})
    projection = factory.create_lexical_index(config)
    assert projection.health() == {"provider": "sqlite_fts", "available": True, "records": 0}
    assert projection.close()


def test_factory_rejects_unknown_options():
    with pytest.raises(ValueError, match="unknown openGauss lexical provider options"):
        factory.create_lexical_index(_config(options={"typo": True}))


def test_factory_normalizes_provider_name(monkeypatch):
    constructed = {}

    class FakeOpenGaussLexicalIndex:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    import providers.lexical_index.native_sql_index as native

    monkeypatch.setattr(native, "OpenGaussLexicalIndex", FakeOpenGaussLexicalIndex)
    factory.create_lexical_index(_config(provider=" OpenGauss_Native "))

    assert constructed["provider_name"] == "opengauss_native"


def test_register_lexical_provider_routes_config(monkeypatch):
    monkeypatch.setattr(factory, "_PROVIDERS", dict(factory._PROVIDERS))
    received = []
    sentinel = object()

    def create_custom(config):
        received.append(config)
        return sentinel

    config = _config(provider=" Custom ")
    register_lexical_provider("custom", create_custom)

    assert factory.create_lexical_index(config) is sentinel
    assert received == [config]


def test_register_lexical_provider_rejects_duplicate(monkeypatch):
    monkeypatch.setattr(factory, "_PROVIDERS", dict(factory._PROVIDERS))

    with pytest.raises(ValueError, match="already registered"):
        factory.register_lexical_provider(" OPENGAUSS_NATIVE ", lambda config: None)


def test_register_lexical_provider_rejects_empty_name():
    with pytest.raises(ValueError, match="name must not be empty"):
        factory.register_lexical_provider("  ", lambda config: None)


def test_register_lexical_provider_rejects_non_callable():
    with pytest.raises(TypeError, match="factory must be callable"):
        factory.register_lexical_provider("broken", None)
