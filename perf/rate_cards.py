# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Rate card loader and per-span $-cost calculator.

Rates are expressed in USD per 1M tokens. The default card lives next to
this file (``rate_cards.yaml``); override via ``OGMEM_PERF_RATE_CARD=/path``.

Unknown models fall through to the ``default`` entry within each family.
If neither the model nor a ``default`` entry exists, a ``KeyError`` is
raised — no silent fallback (per CLAUDE.md §工程原则).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger("ogmem.perf.rate_cards")

_DEFAULT_CARD_PATH = Path(__file__).parent / "rate_cards.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for perf.rate_cards; install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=4)
def _load_cached(resolved_path: str) -> dict[str, Any]:
    """Load and cache a rate card from an already-resolved path string."""
    p = Path(resolved_path)
    if not p.exists():
        raise FileNotFoundError(f"rate card not found: {p}")
    data = _load_yaml(p)
    if "llm" not in data and "embedding" not in data:
        raise ValueError(f"invalid rate card {p}: missing llm/embedding families")
    return data


def load_rate_card(path: str | None = None) -> dict[str, Any]:
    """Load a rate card from YAML.

    Args:
        path: Explicit path, or ``None`` to honour ``OGMEM_PERF_RATE_CARD``
              and finally the packaged default.

    Returns:
        Parsed dict with ``llm`` and ``embedding`` families.
    """
    resolved = str(Path(path or os.environ.get("OGMEM_PERF_RATE_CARD") or _DEFAULT_CARD_PATH))
    return _load_cached(resolved)


def _lookup(card: dict[str, Any], family: str, model: str | None) -> dict[str, float]:
    entries = card.get(family, {}) or {}
    if model and model in entries:
        return entries[model]
    # Match by provider-prefixed cache key (e.g. "openai/gpt-4o-mini")
    if model and "/" in model:
        suffix = model.split("/", 1)[1]
        if suffix in entries:
            return entries[suffix]
    if "default" in entries:
        return entries["default"]
    raise KeyError(
        f"rate card family={family!r} has no entry for model={model!r} and no 'default'"
    )


def _per_million(rate: float, tokens: int) -> float:
    if not tokens:
        return 0.0
    if tokens < 0:
        raise ValueError(f"negative token count: {tokens}")
    return float(tokens) * float(rate) / 1_000_000.0


def compute_cost(
    llm_tokens: dict[str, int],
    embed_tokens: dict[str, int],
    llm_model: str | None,
    embed_model: str | None,
    card: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute $-cost breakdown for a span's token deltas.

    Args:
        llm_tokens: dict with ``input_tokens`` / ``output_tokens`` /
                    ``cache_read`` / ``cache_write``.
        embed_tokens: dict with ``embed_tokens``.
        llm_model: resolved LLM model name or ``None``.
        embed_model: resolved embedding model name or ``None``.
        card: pre-loaded rate card; ``None`` loads the default.

    Returns:
        ``{"llm": float, "embedding": float, "total": float}`` in USD.
    """
    card = card or load_rate_card()

    llm_cost = 0.0
    if llm_tokens:
        rates = _lookup(card, "llm", llm_model)
        llm_cost += _per_million(rates.get("input", 0.0), llm_tokens.get("input_tokens", 0))
        llm_cost += _per_million(rates.get("output", 0.0), llm_tokens.get("output_tokens", 0))
        # Cached reads are cheaper but not free
        if "cache_read" in rates:
            llm_cost += _per_million(rates["cache_read"], llm_tokens.get("cache_read", 0))
        # Cache writes (creating cache) have cost if rate is defined
        cache_write_tokens = llm_tokens.get("cache_write", 0)
        if cache_write_tokens > 0 and "cache_write" in rates:
            llm_cost += _per_million(rates["cache_write"], cache_write_tokens)

    embed_cost = 0.0
    if embed_tokens:
        rates = _lookup(card, "embedding", embed_model)
        embed_cost += _per_million(rates.get("input", 0.0), embed_tokens.get("embed_tokens", 0))

    return {
        "llm": round(llm_cost, 8),
        "embedding": round(embed_cost, 8),
        "total": round(llm_cost + embed_cost, 8),
    }
