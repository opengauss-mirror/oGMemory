from types import SimpleNamespace

import pytest

from providers.vector_index import factory, register_vector_provider


def _config(provider="memory"):
    return SimpleNamespace(
        vector_db_type=provider,
        opengauss_connection_string="postgres://db",
        opengauss_dimension=16,
        opengauss_table_name="vectors",
        opengauss_pool_size=9,
        chroma_collection_name="memories",
        chroma_persist_directory=".vectors",
        embedding_dimension=32,
    )


def test_factory_builds_memory_provider():
    index = factory.create_vector_index(_config())

    assert index._dimension == 16


def test_factory_builds_opengauss_provider(monkeypatch):
    constructed = {}

    class FakeOpenGaussVectorIndex:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    import providers.vector_index.opengauss_index as opengauss

    monkeypatch.setattr(opengauss, "OpenGaussVectorIndex", FakeOpenGaussVectorIndex)
    factory.create_vector_index(_config("opengauss"))

    assert constructed == {
        "connection_string": "postgres://db",
        "dimension": 16,
        "table_name": "vectors",
        "pool_size": 9,
        "dialect": None,
    }


def test_factory_requires_opengauss_connection():
    config = _config("opengauss")
    config.opengauss_connection_string = None

    with pytest.raises(ValueError, match="OPENGAUSS_CONNECTION_STRING"):
        factory.create_vector_index(config)


def test_factory_builds_chroma_provider(monkeypatch):
    constructed = {}

    class FakeChromaVectorIndex:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

    import providers.vector_index.chroma_index as chroma

    monkeypatch.setattr(chroma, "ChromaVectorIndex", FakeChromaVectorIndex)
    factory.create_vector_index(_config(" ChRoMa "))

    assert constructed == {
        "collection_name": "memories",
        "persist_directory": ".vectors",
        "dimension": 32,
    }


def test_register_vector_provider_routes_config(monkeypatch):
    monkeypatch.setattr(factory, "_PROVIDERS", dict(factory._PROVIDERS))
    received = []
    sentinel = object()

    def create_custom(config):
        received.append(config)
        return sentinel

    config = _config(" Custom ")
    register_vector_provider("custom", create_custom)

    assert factory.create_vector_index(config) is sentinel
    assert received == [config]


def test_register_vector_provider_rejects_duplicate(monkeypatch):
    monkeypatch.setattr(factory, "_PROVIDERS", dict(factory._PROVIDERS))

    with pytest.raises(ValueError, match="already registered"):
        register_vector_provider(" MEMORY ", lambda config: None)


def test_register_vector_provider_rejects_empty_name():
    with pytest.raises(ValueError, match="name must not be empty"):
        register_vector_provider("  ", lambda config: None)


def test_register_vector_provider_rejects_non_callable():
    with pytest.raises(TypeError, match="factory must be callable"):
        register_vector_provider("broken", None)


def test_factory_rejects_unknown_provider():
    with pytest.raises(
        ValueError,
        match=(
            "Unknown vector_db_type: 'missing'; available providers: "
            "chroma, memory, opengauss"
        ),
    ):
        factory.create_vector_index(_config("missing"))
