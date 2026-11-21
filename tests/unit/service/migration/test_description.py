"""Source descriptions are informational and exclude connection credentials."""

import copy
from types import SimpleNamespace

import pytest

from service.migration.description import compatibility, describe
from service.migration.package import DATASETS, PackageError, open_package, write_package


def environment():
    cfg = SimpleNamespace(
        provider="openai",
        embedding_provider=None,
        openai_embedding_model="model-a",
        opengauss_dimension=8,
        vector_db_type="chroma",
        openai_api_key="secret-key",
        sql_connection_string="host=private password=secret-password",
    )
    adapter = SimpleNamespace(
        backend="agfs",
        description=lambda: {
            "backend": "agfs",
            "database_type": None,
            "database_version": None,
        },
    )
    return cfg, describe(cfg, adapter)


def test_description_roundtrip_and_changes_do_not_apply_source_config(tmp_path):
    cfg, source = environment()
    source["embedding"]["model_version"] = "revision-1"
    package = tmp_path / "description.tar.gz"
    write_package(
        package,
        {kind: [] for kind in DATASETS},
        source_backend="agfs",
        source_version="test",
        source_environment=source,
    )
    with open_package(package) as loaded:
        assert loaded.manifest["source_environment"] == source
        assert "secret" not in str(loaded.manifest) and "private" not in str(loaded.manifest)
    target = copy.deepcopy(source)
    target["embedding"].update(model="model-b", dimension=16)
    original = copy.deepcopy(target)
    report = compatibility(source, target)
    assert target == original and cfg.openai_embedding_model == "model-a"
    assert set(report["changes"]) >= {
        "embedding.model",
        "embedding.dimension",
    }
    assert any("检索结果可能变化" in n for n in report["notices"])
    assert "始终使用目标模型重建" in compatibility(source, source)["notices"][0]


def test_old_package_has_explicit_unknown_description(tmp_path):
    _, target = environment()
    package = tmp_path / "legacy.tar.gz"
    write_package(
        package, {kind: [] for kind in DATASETS}, source_backend="agfs", source_version="test"
    )
    with open_package(package) as loaded:
        report = compatibility(loaded.manifest.get("source_environment"), target)
    assert report["source"] is None
    assert any("缺少来源环境描述" in n for n in report["notices"])


@pytest.mark.parametrize(
    "field,value", [("api_key", "secret"), ("dimension", True), ("dimension", 0)]
)
def test_description_rejects_unexpected_or_invalid_fields(tmp_path, field, value):
    _, source = environment()
    source["embedding"][field] = value
    with pytest.raises(PackageError):
        write_package(
            tmp_path / "invalid.tar.gz",
            {kind: [] for kind in DATASETS},
            source_backend="agfs",
            source_version="test",
            source_environment=source,
        )


def test_sentence_transformer_dimension_comes_from_model():
    cfg, _ = environment()
    cfg.embedding_provider = "st"
    cfg.st_model = "test-model"
    cfg.opengauss_dimension = 1024
    cfg.to_provider_config = lambda: SimpleNamespace(
        create_embedder=lambda: SimpleNamespace(dimension=384)
    )
    adapter = SimpleNamespace(
        backend="agfs",
        description=lambda: {
            "backend": "agfs",
            "database_type": None,
            "database_version": None,
        },
    )
    assert describe(cfg, adapter)["embedding"]["dimension"] == 384
    assert cfg.opengauss_dimension == 1024


def test_unavailable_sentence_transformer_dimension_is_explicitly_unknown():
    cfg, source = environment()
    cfg.embedding_provider = "st"
    cfg.st_model = "unavailable"

    def unavailable():
        raise ImportError("optional model unavailable")

    cfg.to_provider_config = lambda: SimpleNamespace(create_embedder=unavailable)
    adapter = SimpleNamespace(backend="agfs", description=lambda: source["content_storage"])
    target = describe(cfg, adapter)
    assert target["embedding"]["dimension"] is None
    assert any("维度无法获取" in n for n in compatibility(source, target)["notices"])
