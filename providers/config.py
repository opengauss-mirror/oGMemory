"""Provider configuration for ContextEngine.

Centralized configuration for LLM and Embedder providers.
Now delegates to ``OgMemConfig`` for unified YAML + env loading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from core.interfaces import VectorIndex
from providers.embedder import Embedder, MockEmbedder, get_openai_embedder
from providers.llm import LLM, MockLLM, get_openai_llm

if TYPE_CHECKING:
    from providers.unified_config import OgMemConfig, SecretCommandSpec


ProviderType = Literal["mock", "openai", "openai-cached", "st", "volcengine"]
# Open string type allows names supplied through the vector provider registry.
VectorDbType = str


@dataclass
class ProviderConfig:
    """Configuration for LLM and Embedder providers."""

    # Provider type
    provider: ProviderType = "mock"

    # OpenAI configuration
    openai_api_key: str | None = field(default=None, repr=False)
    openai_api_key_command: SecretCommandSpec | None = field(default=None, repr=False)
    openai_base_url: str | None = None
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_llm_model: str = "gpt-4o-mini"

    # Separate URL/key overrides for embedding (if different from LLM)
    openai_embedding_base_url: str | None = None
    openai_embedding_api_key: str | None = field(default=None, repr=False)
    openai_embedding_api_key_command: SecretCommandSpec | None = field(default=None, repr=False)

    # Cache configuration (for cached providers)
    enable_cache: bool = True
    cache_max_size: int = 1000

    # Model parameters
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_request_timeout_seconds: float = 120.0
    llm_json_mode: bool = False
    llm_reasoning_effort: str | None = None
    llm_is_reasoning_model: bool | None = None
    llm_extra_body: dict[str, Any] = field(default_factory=dict)

    # Vector database configuration
    vector_db_type: VectorDbType = "chroma"
    opengauss_connection_string: str | None = None
    opengauss_dimension: int = 1024
    opengauss_table_name: str = "vector_index"
    opengauss_pool_size: int = 5

    # ChromaDB configuration
    chroma_persist_directory: str = ".chroma_data"
    chroma_collection_name: str = "contextengine"

    # Separate embedding provider (overrides main provider for embedding only)
    embedding_provider: str | None = None

    # ST (sentence-transformers) model for local embedding
    st_model: str = "BAAI/bge-m3"

    # Embedding output dimension. In the current design this must stay aligned
    # with the vector-index schema dimension configured for persistence.
    embedding_dimension: int = 1024

    # Embedding multimodal flag
    embedding_multimodal: bool = False

    # Maximum characters sent for each OpenAI-compatible embedding input.
    embedding_max_input_chars: int = 30_000
    embedding_batch_size: int = 2048

    # Lazy secrets cache
    _cached_openai_api_key: str | None = field(default=None, init=False, repr=False)
    _cached_embedding_api_key: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_env(cls) -> ProviderConfig:
        """Create configuration from the unified config loader.

        Environment variables:
            CONTEXTENGINE_PROVIDER: Provider type (mock/openai/openai-cached)
            OGMEM_API_KEY: OpenAI API key
            OGMEM_BASE_URL: Custom base URL for OpenAI-compatible API (auto-appends /v1 if needed)
            OGMEM_EMBEDDING_MODEL: Embedding model name
            OGMEM_LLM_MODEL: LLM model name
            VECTOR_DB_TYPE: Vector database type (memory/opengauss)
            VECTOR_DB_CONNECTION_STRING: PostgreSQL/openGauss connection string
                (legacy alias: OPENGAUSS_CONNECTION_STRING)
            VECTOR_DB_DIMENSION: Embedding dimension (default 1024)
            VECTOR_DB_TABLE_NAME: Table name for vector storage
            VECTOR_DB_POOL_SIZE: Connection pool size
        """
        from providers.unified_config import get_config
        return get_config().to_provider_config()

    @classmethod
    def from_ogmem_config(cls, cfg: OgMemConfig) -> ProviderConfig:
        """Create a provider-focused projection from an OgMemConfig instance."""
        return cls(
            provider=cfg.provider,  # type: ignore
            openai_api_key=cfg.openai_api_key,
            openai_api_key_command=cfg.openai_api_key_command,
            openai_base_url=cfg.openai_base_url,
            openai_embedding_model=cfg.openai_embedding_model,
            openai_llm_model=cfg.openai_llm_model,
            enable_cache=cfg.enable_cache,
            cache_max_size=cfg.cache_max_size,
            llm_temperature=cfg.llm_temperature,
            llm_max_tokens=cfg.llm_max_tokens,
            llm_request_timeout_seconds=cfg.llm_request_timeout_seconds,
            llm_json_mode=getattr(cfg, 'llm_json_mode', False),
            llm_reasoning_effort=cfg.llm_reasoning_effort,
            llm_is_reasoning_model=cfg.llm_is_reasoning_model,
            llm_extra_body=dict(cfg.llm_extra_body),
            embedding_provider=cfg.embedding_provider,
            st_model=cfg.st_model,
            vector_db_type=cfg.vector_db_type,
            opengauss_connection_string=cfg.opengauss_connection_string,
            opengauss_dimension=cfg.opengauss_dimension,
            opengauss_table_name=cfg.opengauss_table_name,
            opengauss_pool_size=cfg.opengauss_pool_size,
            chroma_persist_directory=cfg.chroma_persist_directory,
            chroma_collection_name=cfg.chroma_collection_name,
            openai_embedding_base_url=cfg.openai_embedding_base_url,
            openai_embedding_api_key=cfg.openai_embedding_api_key,
            openai_embedding_api_key_command=cfg.openai_embedding_api_key_command,
            embedding_multimodal=cfg.embedding_multimodal,
            embedding_max_input_chars=cfg.embedding_max_input_chars,
            embedding_batch_size=cfg.embedding_batch_size,
            embedding_dimension=cfg.opengauss_dimension,
        )

    def effective_openai_api_key(self) -> str | None:
        if self.openai_api_key:
            return self.openai_api_key
        if self._cached_openai_api_key is not None:
            return self._cached_openai_api_key
        if self.openai_api_key_command:
            from providers.unified_config import _run_secret_command
            self._cached_openai_api_key = _run_secret_command(
                self.openai_api_key_command, label="llm.api_key"
            )
            return self._cached_openai_api_key
        return None

    def effective_embedding_api_key(self) -> str | None:
        if self.openai_embedding_api_key:
            return self.openai_embedding_api_key
        if self._cached_embedding_api_key is not None:
            return self._cached_embedding_api_key
        if self.openai_embedding_api_key_command:
            from providers.unified_config import _run_secret_command
            self._cached_embedding_api_key = _run_secret_command(
                self.openai_embedding_api_key_command, label="embedding.api_key"
            )
            return self._cached_embedding_api_key
        return self.effective_openai_api_key()

    def create_embedder(self) -> Embedder:
        """Create an Embedder instance based on configuration.

        If embedding_provider is set, it overrides the main provider for embedding only.
        """
        ep = self.embedding_provider or self.provider

        if ep == "mock":
            return MockEmbedder(dimension=self.opengauss_dimension)

        if ep == "st":
            from providers.embedder.st_embedder import SentenceTransformerEmbedder
            return SentenceTransformerEmbedder(model=self.st_model)

        if ep == "volcengine":
            from providers.embedder.volcengine_embedder import VolcengineEmbedder
            api_key = self.effective_embedding_api_key() or os.environ.get("VOLC_API_KEY", "")
            base_url = (self.openai_embedding_base_url
                        or os.environ.get("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3"))
            model = os.environ.get("VOLC_EMBEDDING_MODEL",
                                   self.openai_embedding_model or "doubao-embedding-vision-250615")
            return VolcengineEmbedder(
                api_key=api_key,
                model=model,
                base_url=base_url,
                dimension=self.opengauss_dimension or 1024,
            )

        if ep in ("openai", "openai-cached"):
            emb_url = self.openai_embedding_base_url or self.openai_base_url
            emb_key = self.effective_embedding_api_key()
            OpenAIEmbedder, CachedOpenAIEmbedder = get_openai_embedder()
            if ep == "openai-cached" and self.enable_cache:
                return CachedOpenAIEmbedder(
                    api_key=emb_key,
                    base_url=emb_url,
                    model=self.openai_embedding_model,
                    dimension=self.opengauss_dimension,
                    cache_max_size=self.cache_max_size,
                    multimodal=self.embedding_multimodal,
                    max_input_chars=self.embedding_max_input_chars,
                    batch_size=self.embedding_batch_size,
                )
            return OpenAIEmbedder(
                api_key=emb_key,
                base_url=emb_url,
                model=self.openai_embedding_model,
                dimension=self.opengauss_dimension,
                multimodal=self.embedding_multimodal,
                max_input_chars=self.embedding_max_input_chars,
                batch_size=self.embedding_batch_size,
            )

        raise ValueError(f"Unknown embedding provider: {ep}")

    def create_vector_index(self, pool=None) -> VectorIndex:
        """Create a VectorIndex instance based on configuration.

        Args:
            pool: Optional SharedConnectionPool to share with other SQL stores.
                  When provided, the vector index uses the same pool for
                  unified connection management and cross-store atomicity.
        """
        from providers.vector_index.factory import create_vector_index

        if pool is None or str(self.vector_db_type).strip().casefold() != "opengauss":
            return create_vector_index(self)

        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        return OpenGaussVectorIndex(
            connection_string=self.opengauss_connection_string,
            dimension=self.opengauss_dimension,
            table_name=self.opengauss_table_name,
            pool_size=self.opengauss_pool_size,
            pool=pool,
        )

    def create_llm(self) -> LLM:
        """Create an LLM instance based on configuration."""
        if self.provider == "mock":
            return MockLLM()

        if self.provider in ("openai", "openai-cached"):
            OpenAILLM, CachedOpenAILLM = get_openai_llm()
            if self.provider == "openai-cached" and self.enable_cache:
                return CachedOpenAILLM(
                    api_key=self.effective_openai_api_key(),
                    base_url=self.openai_base_url,
                    model=self.openai_llm_model,
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens,
                    request_timeout_seconds=self.llm_request_timeout_seconds,
                    json_mode=self.llm_json_mode,
                    reasoning_effort=self.llm_reasoning_effort,
                    is_reasoning_model=self.llm_is_reasoning_model,
                    extra_body=self.llm_extra_body,
                    cache_max_size=self.cache_max_size,
                )
            return OpenAILLM(
                api_key=self.effective_openai_api_key(),
                base_url=self.openai_base_url,
                model=self.openai_llm_model,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                request_timeout_seconds=self.llm_request_timeout_seconds,
                json_mode=self.llm_json_mode,
                reasoning_effort=self.llm_reasoning_effort,
                is_reasoning_model=self.llm_is_reasoning_model,
                extra_body=self.llm_extra_body,
            )

        raise ValueError(f"Unknown provider type: {self.provider}")

    # Helper: convert the provider-facing subset back into an OgMemConfig.
    # This is intentionally not a full-fidelity round trip for service/auth/runtime fields.
    def to_ogmem_config(self) -> OgMemConfig:
        from providers.unified_config import OgMemConfig
        return OgMemConfig(
            provider=self.provider,
            openai_api_key=self.openai_api_key,
            openai_api_key_command=self.openai_api_key_command,
            openai_base_url=self.openai_base_url,
            openai_llm_model=self.openai_llm_model,
            llm_temperature=self.llm_temperature,
            llm_max_tokens=self.llm_max_tokens,
            llm_request_timeout_seconds=self.llm_request_timeout_seconds,
            llm_json_mode=self.llm_json_mode,
            llm_reasoning_effort=self.llm_reasoning_effort,
            llm_is_reasoning_model=self.llm_is_reasoning_model,
            llm_extra_body=dict(self.llm_extra_body),
            openai_embedding_model=self.openai_embedding_model,
            openai_embedding_base_url=self.openai_embedding_base_url,
            openai_embedding_api_key=self.openai_embedding_api_key,
            openai_embedding_api_key_command=self.openai_embedding_api_key_command,
            embedding_multimodal=self.embedding_multimodal,
            embedding_max_input_chars=self.embedding_max_input_chars,
            embedding_batch_size=self.embedding_batch_size,
            vector_db_type=self.vector_db_type,
            opengauss_connection_string=self.opengauss_connection_string,
            opengauss_dimension=self.opengauss_dimension,
            opengauss_table_name=self.opengauss_table_name,
            opengauss_pool_size=self.opengauss_pool_size,
            enable_cache=self.enable_cache,
            cache_max_size=self.cache_max_size,
        )

# Default configuration — loaded lazily via the unified config system
DEFAULT_CONFIG = ProviderConfig.from_env()


def get_embedder(config: ProviderConfig | None = None) -> Embedder:
    """Get configured Embedder instance."""
    config = config or DEFAULT_CONFIG
    return config.create_embedder()


def get_llm(config: ProviderConfig | None = None) -> LLM:
    """Get configured LLM instance."""
    config = config or DEFAULT_CONFIG
    return config.create_llm()


def get_vector_index(config: ProviderConfig | None = None) -> VectorIndex:
    """Get configured VectorIndex instance."""
    config = config or DEFAULT_CONFIG
    return config.create_vector_index()
