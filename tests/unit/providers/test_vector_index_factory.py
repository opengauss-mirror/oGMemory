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
            "chroma, memory, milvus, opengauss"
        ),
    ):
        factory.create_vector_index(_config("missing"))


@pytest.mark.parametrize(
    "provider,class_name",
    [
        ("chroma", "ChromaVectorIndex"),
        ("milvus", "MilvusVectorIndex"),
    ],
)
def test_yaml_type_selects_backend_through_shared_factory(
    tmp_path, monkeypatch, provider, class_name
):
    from unittest.mock import MagicMock

    from providers.unified_config import OgMemConfig

    monkeypatch.setenv("VECTOR_DB_TYPE", "opengauss")
    path = tmp_path / "ogmem.yaml"
    path.write_text(
        f"vector_db:\n  type: {provider}\n  dimension: 3\n"
        "  milvus_uri: http://vectors:19530\n  milvus_collection: memories\n"
    )
    constructor = MagicMock()
    monkeypatch.setattr(f"providers.vector_index.{provider}_index.{class_name}", constructor)
    config = OgMemConfig.load(str(path))
    assert config.vector_db_type == provider  # Explicit YAML wins over environment.
    assert config.to_provider_config().create_vector_index() is constructor.return_value
    assert constructor.call_args.kwargs["dimension"] == 3
    if provider == "milvus":
        assert constructor.call_args.kwargs["uri"] == "http://vectors:19530"
        assert constructor.call_args.kwargs["collection_name"] == "memories"
