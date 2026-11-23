# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Construction and registration for lexical providers."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

from core.lexical import DEFAULT_MAX_QUERY_TERMS, HybridLexicalIndex

LexicalFactory = Callable[[Any], HybridLexicalIndex]

_PROVIDERS: dict[str, LexicalFactory] = {}
_PROVIDER_LOCK = RLock()

_OPENGAUSS_NATIVE_OPTIONS = frozenset(
    {
        "connection_string",
        "table_name",
        "pool_size",
        "max_write_batch_size",
        "max_query_terms",
        "max_query_chars",
        "rebuild_lease_seconds",
        "max_document_chars",
        "bm25_topk",
        "bm25_k1",
        "bm25_b",
        "bm25_taat",
        "query_timeout_ms",
        "connect_timeout_seconds",
        "operation_timeout_ms",
    }
)


def _validated_options(config) -> dict[str, Any]:
    options = dict(getattr(config, "retrieval_lexical_options", {}) or {})
    unknown = set(options) - _OPENGAUSS_NATIVE_OPTIONS
    if unknown:
        raise ValueError(f"unknown openGauss lexical provider options: {sorted(unknown)}")
    return options


def _normalize_provider_name(name: str) -> str:
    key = name.strip().casefold()
    if not key:
        raise ValueError("lexical provider name must not be empty")
    return key


def register_lexical_provider(name: str, provider_factory: LexicalFactory) -> None:
    """Register a lexical provider factory."""

    key = _normalize_provider_name(name)
    if not callable(provider_factory):
        raise TypeError("lexical provider factory must be callable")
    with _PROVIDER_LOCK:
        if key in _PROVIDERS:
            raise ValueError(f"lexical provider {key!r} is already registered")
        _PROVIDERS[key] = provider_factory


def _create_opengauss_native(config) -> HybridLexicalIndex:
    """Create the built-in database-native openGauss lexical index."""

    options = _validated_options(config)
    options["connection_string"] = (
        options.get("connection_string")
        or getattr(config, "opengauss_connection_string", None)
        or getattr(config, "sql_connection_string", None)
        or ""
    )
    if not options["connection_string"]:
        raise ValueError("openGauss lexical retrieval requires an openGauss connection")

    from providers.lexical_index.native_sql_index import (
        DEFAULT_MAX_QUERY_CHARS,
        OpenGaussLexicalIndex,
    )
    from providers.lexical_index.sql_index import (
        DEFAULT_MAX_DOCUMENT_CHARS,
        DEFAULT_MAX_WRITE_BATCH_SIZE,
        DEFAULT_POOL_SIZE,
        DEFAULT_REBUILD_LEASE_SECONDS,
    )

    return OpenGaussLexicalIndex(
        connection_string=str(options.get("connection_string") or ""),
        table_name=str(options.get("table_name", "lexical_index_opengauss_native")),
        provider_name="opengauss_native",
        pool_size=int(options.get("pool_size", DEFAULT_POOL_SIZE)),
        max_write_batch_size=int(
            options.get("max_write_batch_size", DEFAULT_MAX_WRITE_BATCH_SIZE)
        ),
        max_query_terms=int(options.get("max_query_terms", DEFAULT_MAX_QUERY_TERMS)),
        max_query_chars=int(options.get("max_query_chars", DEFAULT_MAX_QUERY_CHARS)),
        rebuild_lease_seconds=int(
            options.get("rebuild_lease_seconds", DEFAULT_REBUILD_LEASE_SECONDS)
        ),
        max_document_chars=int(
            options.get("max_document_chars", DEFAULT_MAX_DOCUMENT_CHARS)
        ),
        query_timeout_ms=int(options.get("query_timeout_ms", 5000)),
        connect_timeout_seconds=int(options.get("connect_timeout_seconds", 10)),
        operation_timeout_ms=int(options.get("operation_timeout_ms", 300000)),
        bm25_topk=int(options["bm25_topk"]) if "bm25_topk" in options else None,
        bm25_k1=float(options["bm25_k1"]) if "bm25_k1" in options else None,
        bm25_b=float(options["bm25_b"]) if "bm25_b" in options else None,
        bm25_taat=options.get("bm25_taat"),
    )


register_lexical_provider("opengauss_native", _create_opengauss_native)


def create_lexical_index(config) -> HybridLexicalIndex:
    """Create the configured lexical index."""

    name = _normalize_provider_name(str(config.retrieval_lexical_provider))
    with _PROVIDER_LOCK:
        provider_factory = _PROVIDERS.get(name)
        if provider_factory is None:
            available = ", ".join(sorted(_PROVIDERS))
            raise ValueError(
                f"unsupported lexical provider {name!r}; available providers: {available}"
            )
    return provider_factory(config)
