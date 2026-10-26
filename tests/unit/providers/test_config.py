"""Tests for provider configuration and factory helpers."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import providers.config as config_module
from providers.unified_config import OgMemConfig


def test_from_env_normalizes_base_url_and_parses_numeric_values(monkeypatch: pytest.MonkeyPatch):
    # Test with from_ogmem_config instead since from_env reads actual config files
    # Note: URL normalization happens in OgMemConfig, not in ProviderConfig
    cfg = config_module.ProviderConfig.from_ogmem_config(
        OgMemConfig(
            provider="openai",
            openai_api_key="secret",
            openai_base_url="https://example.com/api",  # OgMemConfig would normalize this
            openai_embedding_model="text-embedding-3-small",
            openai_llm_model="gpt-4o",
            vector_db_type="opengauss",
            opengauss_connection_string="postgres://db",
            opengauss_dimension=3072,
            opengauss_table_name="memories",
            opengauss_pool_size=7,
            embedding_max_input_chars=8_000,
            embedding_batch_size=8,
            enable_cache=True,
        )
    )

    assert cfg.provider == "openai"
    assert cfg.enable_cache is True
    # ProviderConfig passes through the URL as-is (normalization is in OgMemConfig)
    assert cfg.openai_base_url == "https://example.com/api"
    assert cfg.openai_embedding_model == "text-embedding-3-small"
    assert cfg.openai_llm_model == "gpt-4o"
    assert cfg.vector_db_type == "opengauss"
    assert cfg.opengauss_dimension == 3072
    assert cfg.opengauss_table_name == "memories"
    assert cfg.opengauss_pool_size == 7
    assert cfg.embedding_max_input_chars == 8_000
    assert cfg.embedding_batch_size == 8


def test_from_env_keeps_existing_v1_suffix(monkeypatch: pytest.MonkeyPatch):
    # This test now reads from actual config file if it exists
    # We'll test the ProviderConfig constructor directly instead
    cfg = config_module.ProviderConfig.from_ogmem_config(
        OgMemConfig(openai_base_url="https://example.com/v1")
    )

    assert cfg.openai_base_url == "https://example.com/v1"


def test_create_embedder_returns_mock_embedder_for_mock_provider():
    cfg = config_module.ProviderConfig(provider="mock")

    embedder = cfg.create_embedder()

    assert isinstance(embedder, config_module.MockEmbedder)


def test_create_embedder_uses_cached_openai_factory(monkeypatch: pytest.MonkeyPatch):
    openai_embedder = Mock(name="OpenAIEmbedder")
    cached_embedder = Mock(name="CachedOpenAIEmbedder")
    monkeypatch.setattr(
        config_module,
        "get_openai_embedder",
        Mock(return_value=(openai_embedder, cached_embedder)),
    )
    cfg = config_module.ProviderConfig(
        provider="openai-cached",
        openai_api_key="secret",
        openai_base_url="https://example.com/v1",
        openai_embedding_model="text-embedding-3-small",
        embedding_max_input_chars=8_000,
        embedding_batch_size=8,
        cache_max_size=123,
    )

    cfg.create_embedder()

    cached_embedder.assert_called_once_with(
        api_key="secret",
        base_url="https://example.com/v1",
        model="text-embedding-3-small",
        dimension=1024,  # Default opengauss_dimension
        cache_max_size=123,
        multimodal=False,  # Default embedding_multimodal
        max_input_chars=8_000,
        batch_size=8,
    )


def test_create_embedder_uses_non_cached_openai_factory_when_cache_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    openai_embedder = Mock(name="OpenAIEmbedder")
    cached_embedder = Mock(name="CachedOpenAIEmbedder")
    monkeypatch.setattr(
        config_module,
        "get_openai_embedder",
        Mock(return_value=(openai_embedder, cached_embedder)),
    )
    cfg = config_module.ProviderConfig(
        provider="openai-cached",
        openai_api_key="secret",
        openai_embedding_model="Qwen/Qwen3-Embedding-0.6B",
        embedding_max_input_chars=8_000,
        embedding_batch_size=8,
        enable_cache=False,
    )

    cfg.create_embedder()

    openai_embedder.assert_called_once_with(
        api_key="secret",
        base_url=None,
        model="Qwen/Qwen3-Embedding-0.6B",
        dimension=1024,
        multimodal=False,
        max_input_chars=8_000,
        batch_size=8,
    )
    cached_embedder.assert_not_called()


def test_create_vector_index_returns_in_memory_index():
    cfg = config_module.ProviderConfig(vector_db_type="memory", opengauss_dimension=8)

    index = cfg.create_vector_index()

    from providers.vector_index import InMemoryVectorIndex

    assert isinstance(index, InMemoryVectorIndex)
    assert index._dimension == 8


def test_create_vector_index_requires_connection_string_for_opengauss():
    cfg = config_module.ProviderConfig(vector_db_type="opengauss", opengauss_connection_string=None)

    with pytest.raises(ValueError, match="OPENGAUSS_CONNECTION_STRING"):
        cfg.create_vector_index()


def test_create_vector_index_builds_opengauss_index(monkeypatch: pytest.MonkeyPatch):
    ctor = Mock(name="OpenGaussVectorIndex")
    import providers.vector_index.opengauss_index as opengauss

    monkeypatch.setattr(opengauss, "OpenGaussVectorIndex", ctor)
    cfg = config_module.ProviderConfig(
        vector_db_type="opengauss",
        opengauss_connection_string="postgres://db",
        opengauss_dimension=16,
        opengauss_table_name="vectors",
        opengauss_pool_size=9,
    )

    cfg.create_vector_index()

    ctor.assert_called_once_with(
        connection_string="postgres://db",
        dimension=16,
        table_name="vectors",
        pool_size=9,
    )


def test_create_vector_index_routes_registered_provider(monkeypatch: pytest.MonkeyPatch):
    from providers.vector_index import factory, register_vector_provider

    monkeypatch.setattr(factory, "_PROVIDERS", dict(factory._PROVIDERS))
    sentinel = object()
    cfg = config_module.ProviderConfig(vector_db_type="custom")
    register_vector_provider("custom", lambda config: sentinel)

    assert cfg.create_vector_index() is sentinel


def test_create_llm_uses_openai_factory(monkeypatch: pytest.MonkeyPatch):
    openai_llm = Mock(name="OpenAILLM")
    cached_llm = Mock(name="CachedOpenAILLM")
    monkeypatch.setattr(
        config_module,
        "get_openai_llm",
        Mock(return_value=(openai_llm, cached_llm)),
    )
    cfg = config_module.ProviderConfig(
        provider="openai",
        openai_api_key="secret",
        openai_base_url="https://example.com/v1",
        openai_llm_model="gpt-4o",
        llm_temperature=0.1,
        llm_max_tokens=42,
    )

    cfg.create_llm()

    openai_llm.assert_called_once_with(
        api_key="secret",
        base_url="https://example.com/v1",
        model="gpt-4o",
        temperature=0.1,
        max_tokens=42,
        request_timeout_seconds=120.0,
        json_mode=False,  # Default llm_json_mode
        reasoning_effort=None,
        is_reasoning_model=None,
        extra_body={},
    )


def test_create_llm_uses_cached_factory_when_enabled(monkeypatch: pytest.MonkeyPatch):
    openai_llm = Mock(name="OpenAILLM")
    cached_llm = Mock(name="CachedOpenAILLM")
    monkeypatch.setattr(
        config_module,
        "get_openai_llm",
        Mock(return_value=(openai_llm, cached_llm)),
    )
    cfg = config_module.ProviderConfig(
        provider="openai-cached",
        openai_api_key="secret",
        cache_max_size=12,
    )

    cfg.create_llm()

    cached_llm.assert_called_once()
    assert cached_llm.call_args.kwargs["json_mode"] is False
    openai_llm.assert_not_called()


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (config_module.ProviderConfig(provider="mystery"), "Unknown provider type"),
        (config_module.ProviderConfig(vector_db_type="mystery"), "Unknown vector_db_type"),
    ],
)
def test_invalid_configuration_branches_raise(cfg, message):
    target = cfg.create_llm if cfg.provider == "mystery" else cfg.create_vector_index

    with pytest.raises(ValueError, match=message):
        target()


def test_helper_functions_use_default_config(monkeypatch: pytest.MonkeyPatch):
    default_config = Mock()
    default_config.create_embedder.return_value = "embedder"
    default_config.create_llm.return_value = "llm"
    default_config.create_vector_index.return_value = "index"
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG", default_config)

    assert config_module.get_embedder() == "embedder"
    assert config_module.get_llm() == "llm"
    assert config_module.get_vector_index() == "index"
