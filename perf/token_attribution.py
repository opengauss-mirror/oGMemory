# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Per-context token attribution for perf spans.

Historical design (deprecated)
------------------------------
``snapshot_service`` used to read ``service._llm.token_tracker`` and
``service._embedder.token_tracker`` directly via reflection, then
:func:`diff` subtracted two snapshots to attribute tokens to a span.
That approach broke under concurrency: ``token_tracker`` is a process-wide
cumulative counter, so any thread that completed *after* siblings would
see those siblings' tokens included in its delta.

Current design (bucket-based)
-----------------------------
:func:`snapshot_service` now pushes a fresh
:class:`providers.token_tracker.TokenBucket` onto a ``ContextVar`` stack
maintained by ``providers.token_tracker``. Every provider call records
into the global counter *and* into every bucket on the active context's
stack. Each thread / asyncio task therefore sees only the LLM and
embedding work performed inside its own perf span, regardless of how
many concurrent extractions or compose calls run alongside it.

The function returns an opaque dict that contains the bucket reference
and the ``ContextVar`` reset token. :func:`diff` consumes that dict to
pop the bucket and render the accumulated totals in the same shape used
by report renderers, rate cards, and downstream consumers.

The caller surface is unchanged on purpose: ``Recorder.snapshot_tokens``
calls this module's :func:`snapshot_service`, and
``Recorder.finalize_tokens`` calls :func:`diff(after, before)`. Existing
HTTP handlers and the ``_background_extract_write`` instrumentation
continue to work without modification.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

from providers.token_tracker import TokenBucket, current_buckets, pop_bucket, push_bucket

logger = logging.getLogger("ogmem.perf.attribution")


# Sentinel keys used inside the dicts returned by :func:`snapshot_service`.
_BUCKET = "_bucket"
_RESET_TOKEN = "_reset_token"


def _model_of(obj: Any) -> str | None:
    """Best-effort model identifier from a provider object."""
    if obj is None:
        return None
    model = getattr(obj, "model", None)
    if isinstance(model, str) and model:
        return model
    return obj.__class__.__name__


def snapshot_service(service: Any) -> dict[str, Any]:
    """Open a new attribution scope.

    Pushes a fresh :class:`TokenBucket` onto the active context's bucket
    stack. The returned dict is opaque to callers — pass it as the
    ``before`` argument to :func:`diff` to close the scope and read the
    accumulated totals.

    The ``service`` argument is retained for API compatibility and used
    only as a fallback for model-name attribution; the per-call model
    identifier flows through :class:`providers.token_tracker.TokenTracker`
    so providers that mix model versions remain accurate.
    """
    bucket, token = push_bucket()
    fallback_llm = _model_of(getattr(service, "_llm", None))
    fallback_embed = _model_of(getattr(service, "_embedder", None))
    return {
        _BUCKET: bucket,
        _RESET_TOKEN: token,
        "llm_model_fallback": fallback_llm,
        "embed_model_fallback": fallback_embed,
    }


def _close(scope: dict[str, Any] | None) -> None:
    """Pop the bucket associated with ``scope`` if present."""
    if not scope:
        return
    token = scope.get(_RESET_TOKEN)
    if token is None:
        return
    try:
        pop_bucket(token)
    except (LookupError, ValueError) as exc:
        # ContextVar.reset raises if the token is from a different context
        # (e.g. the scope was opened in a different thread). The bucket
        # will be cleaned up automatically when the originating context
        # exits, so we just log and move on.
        logger.debug("pop_bucket skipped (token from different context): %s", exc)


def diff(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    """Close an attribution scope and return its accumulated totals.

    The legacy signature is preserved: ``before`` is the dict returned
    by :func:`snapshot_service`, ``after`` is whatever the caller passed
    (it may be a redundant snapshot opened just before this call). Both
    dicts have their buckets popped to keep the stack clean.

    Returns a dict shaped like::

        {
            "llm":   {... per-bucket llm counters ...},
            "embed": {... per-bucket embed counters ...},
            "llm_model":   "gpt-4o-mini",
            "embed_model": "text-embedding-3-small",
        }
    """
    bucket: TokenBucket | None = None
    if before:
        bucket = before.get(_BUCKET)

    # Always pop both — diff is the canonical close point. If the caller
    # also passed an ``after`` snapshot, its bucket is empty (no LLM
    # calls happen between snapshot and diff at the same call site) and
    # we just discard it.
    _close(after)
    _close(before)

    if bucket is None:
        return {
            "llm": {},
            "embed": {},
            "llm_model": None,
            "embed_model": None,
        }

    rendered = bucket.to_attribution_dict()
    if not rendered.get("llm_model"):
        rendered["llm_model"] = (before or {}).get("llm_model_fallback")
    if not rendered.get("embed_model"):
        rendered["embed_model"] = (before or {}).get("embed_model_fallback")
    return rendered


def token_source(snapshot: dict[str, Any]) -> str:
    """Classify where the token attribution came from.

    Returns ``"tracker"`` when at least one provider call was recorded
    into the bucket, ``"missing"`` otherwise. The name is preserved for
    backward compatibility with consumers that branch on this value.
    """
    llm = snapshot.get("llm") or {}
    embed = snapshot.get("embed") or {}
    has_llm = bool(llm.get("llm_calls") or llm.get("input_tokens") or llm.get("output_tokens"))
    has_embed = bool(embed.get("embed_calls") or embed.get("embed_tokens"))
    return "tracker" if (has_llm or has_embed) else "missing"


def detect_counter_reset(after: dict[str, Any], before: dict[str, Any], field: str = "input_tokens") -> bool:
    """Backward-compatible no-op.

    The bucket model has no notion of a "reset" — each bucket starts
    empty by construction. This function previously detected upstream
    token-counter rollovers; it now always returns ``False`` and exists
    only so callers that imported it continue to work.
    """
    return False


__all__ = [
    "snapshot_service",
    "diff",
    "token_source",
    "detect_counter_reset",
]
