# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Lifecycle Recorder — central coordinator for perf events.

The recorder is a process-wide singleton retrieved via ``get_recorder()``.
It is a no-op (zero overhead) until ``enable()`` is called (either
explicitly from a driver or implicitly when ``OGMEM_PERF_ENABLED=1`` is
set in the environment at import time of any decorated stage).

Responsibilities:
    * hold the attached ``MemoryService`` so ``record_stage`` / ``span``
      can diff token trackers;
    * own the sink (``JsonlSink`` by default);
    * mint ``SpanEvent`` records and forward them.

Threading: ``enable()`` / ``attach()`` are idempotent and guarded by a
lock. Spans themselves rely on ``perf.span_stack`` (ContextVar) for
per-task isolation.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from perf.rate_cards import compute_cost, load_rate_card
from perf.sinks import HttpSink, JsonlSink, MemorySink, Sink
from perf.token_attribution import diff as token_diff
from perf.token_attribution import snapshot_service, token_source

logger = logging.getLogger("ogmem.perf.recorder")


@dataclass
class SpanEvent:
    """A single completed span (stage root or sub-span)."""

    run_id: str
    session_id: str | None
    trace_id: str | None
    stage: str
    span: str                        # "" for the stage root
    parent_span: str | None
    started_at: float
    wall_ms: float
    cpu_ms: float
    ok: bool
    error: str | None
    llm_model: str | None
    embed_model: str | None
    tokens: dict[str, Any] = field(default_factory=dict)
    cost_usd: dict[str, float] = field(default_factory=dict)
    token_source: str = "tracker"    # "tracker" | "missing"
    meta: dict[str, Any] = field(default_factory=dict)


class Recorder:
    """Process singleton. See module docstring."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False
        self._service: Any | None = None
        self._sink: Sink | None = None
        self._run_id: str = ""
        self._rate_card: dict[str, Any] | None = None
        self._record_tokens = True

    # -- configuration ----------------------------------------------------

    def enable(
        self,
        *,
        sink: Sink | None = None,
        run_id: str | None = None,
        rate_card_path: str | None = None,
        record_tokens: bool = True,
    ) -> "Recorder":
        """Turn recording on. Idempotent — later calls update options."""
        with self._lock:
            self._enabled = True
            self._run_id = run_id or self._run_id or str(uuid4())
            self._record_tokens = record_tokens
            if sink is not None:
                # Close any previously-attached sink before replacing
                if self._sink is not None and self._sink is not sink:
                    try:
                        self._sink.close()
                    except Exception as exc:  # pragma: no cover
                        logger.warning("prior sink close failed: %s", exc)
                self._sink = sink
            if self._sink is None:
                # Default: packaged JSONL file under ./perf_logs/
                default_path = os.path.join(
                    "perf_logs", f"{self._run_id}.jsonl"
                )
                self._sink = JsonlSink(default_path)
            if rate_card_path is not None:
                self._rate_card = load_rate_card(rate_card_path)
        logger.info(
            "perf recorder enabled run_id=%s sink=%s",
            self._run_id,
            type(self._sink).__name__,
        )
        return self

    def attach(self, service: Any) -> None:
        """Attach a ``MemoryService`` for token-tracker snapshots."""
        with self._lock:
            self._service = service

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            if self._sink is not None:
                try:
                    self._sink.close()
                except Exception as exc:  # pragma: no cover
                    logger.warning("sink close failed: %s", exc)
            self._sink = None

    # -- introspection ----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def sink(self) -> Sink | None:
        return self._sink

    @property
    def service(self) -> Any | None:
        return self._service

    # -- hot path --------------------------------------------------------

    def snapshot_tokens(self) -> dict[str, Any]:
        """Open a per-context token attribution scope.

        Pushes a fresh :class:`providers.token_tracker.TokenBucket` onto
        the active context's bucket stack. The returned handle must be
        passed back to :meth:`finalize_tokens` (as the ``before`` arg)
        to close the scope and read the accumulated totals.

        Returns ``{}`` when recording is disabled — in that case no
        bucket is pushed and ``finalize_tokens`` is a no-op.
        """
        if not (self._enabled and self._record_tokens):
            return {}
        try:
            return snapshot_service(self._service)
        except Exception as exc:
            logger.warning("token snapshot failed: %s", exc, exc_info=True)
            return {}

    def finalize_tokens(
        self,
        after: dict[str, Any],
        before: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float], str]:
        """Close an attribution scope opened by :meth:`snapshot_tokens`.

        ``before`` is the handle returned by ``snapshot_tokens`` at the
        start of the span; ``after`` is an optional second handle (some
        legacy call sites snapshot again before finalising — both are
        accepted and both buckets are popped to keep the stack clean).

        Returns ``(tokens, cost_usd, source)`` where ``tokens`` matches
        the legacy snapshot-diff layout so report rendering needs no
        changes.
        """
        if not after and not before:
            return {}, {"llm": 0.0, "embedding": 0.0, "total": 0.0}, "missing"
        delta = token_diff(after, before)
        source = token_source(delta)
        cost = {"llm": 0.0, "embedding": 0.0, "total": 0.0}
        try:
            cost = compute_cost(
                llm_tokens=delta.get("llm", {}),
                embed_tokens=delta.get("embed", {}),
                llm_model=delta.get("llm_model"),
                embed_model=delta.get("embed_model"),
                card=self._rate_card,
            )
        except KeyError:
            # No silent fallback — rate card design requires explicit model entry.
            # Re-raise so the caller sees the failure rather than silently reporting $0.
            raise
        except Exception as exc:
            logger.warning("cost computation failed: %s", exc, exc_info=True)
        return delta, cost, source

    def emit(self, event: SpanEvent) -> None:
        if not self._enabled or self._sink is None:
            return
        try:
            self._sink.emit(event)
        except Exception as exc:
            # Do NOT swallow — per CLAUDE.md §工程原则, sink failures surface.
            logger.error("sink emit failed: %s", exc, exc_info=True)
            raise

    def get_all_events(self) -> list[dict]:
        """Return all events from a MemorySink as dicts, or [] if unavailable."""
        if self._sink is None:
            return []
        if isinstance(self._sink, MemorySink):
            with self._lock:
                return list(self._sink.get_events())
        return []


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton: Recorder | None = None


def get_recorder() -> Recorder:
    """Return the process-wide ``Recorder``. Creates it on first call.

    If ``OGMEM_PERF_ENABLED=1`` is set we auto-enable with default options
    (JSONL sink under ``perf_logs/``). Callers can override later via
    ``Recorder.enable(...)``.
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            rec = Recorder()
            if os.environ.get("OGMEM_PERF_ENABLED", "").lower() in ("1", "true", "yes"):
                run_id = os.environ.get("OGMEM_PERF_RUN_ID")
                path = os.environ.get("OGMEM_PERF_OUT")
                http_url = os.environ.get("OGMEM_PERF_HTTP_URL")
                sink: Sink | None = None
                if http_url:
                    sink = HttpSink(http_url)
                elif path:
                    # Prefix per-worker JSONL path with worker_id so multi-worker
                    # deployments write distinct files that perf.report can later
                    # aggregate (方案B).
                    worker_id = os.environ.get("OGMEM_WORKER_ID", "")
                    if worker_id:
                        base, ext = re.sub(r'(\.jsonl)?$', '', path), '.jsonl'
                        path = f"{base}.w{worker_id}{ext}"
                    sink = JsonlSink(path)
                rate_card = os.environ.get("OGMEM_PERF_RATE_CARD")
                rec.enable(sink=sink, run_id=run_id, rate_card_path=rate_card)
            _singleton = rec
    return _singleton


def is_enabled() -> bool:
    """Cheap check used by decorators to short-circuit when disabled."""
    return get_recorder().enabled


# Timing primitives we expose so decorators / span() can share implementation.
def clocks() -> tuple[float, float, float]:
    """Return ``(epoch_now, perf_counter, process_time)``."""
    return time.time(), time.perf_counter(), time.process_time()
