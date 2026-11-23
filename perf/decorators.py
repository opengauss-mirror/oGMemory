# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Public API: ``@record_stage`` decorator and ``span`` context manager."""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator
from uuid import uuid4

from perf import span_stack
from perf.recorder import SpanEvent, clocks, get_recorder

logger = logging.getLogger("ogmem.perf.decorators")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ids_from_params(params: Any) -> tuple[str | None, str | None]:
    """Pull ``session_id`` / ``trace_id`` out of a bridge ``params`` dict."""
    if not isinstance(params, dict):
        return None, None
    sid = params.get("sessionId") or params.get("session_id")
    tid = params.get("traceId") or params.get("trace_id")
    return (sid or None), (tid or None)


def _ids_from_kwargs(args: tuple, kwargs: dict) -> tuple[str | None, str | None]:
    """Recover IDs from a call's args/kwargs. ``args[0]`` is ``self`` on methods.

    The bridge convention is ``method(self, params)`` so we check ``args[1]``.
    """
    for candidate in list(args[1:]) + list(kwargs.values()):
        if isinstance(candidate, dict):
            sid, tid = _ids_from_params(candidate)
            if sid or tid:
                return sid, tid
    return None, None


# ---------------------------------------------------------------------------
# @record_stage
# ---------------------------------------------------------------------------

def record_stage(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a ``MemoryService`` lifecycle method.

    Emits a single ``SpanEvent`` (the stage root) summarising wall time,
    CPU time, token deltas, $-cost, and any sub-span metadata accumulated
    during execution.

    Errors are recorded but **re-raised** — per CLAUDE.md §工程原则
    "禁止 fallback 路径".
    """

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            rec = get_recorder()
            if not rec.enabled:
                # Fast path: recorder disabled → no overhead beyond one dict lookup.
                return fn(*args, **kwargs)

            started_at, wall_start, cpu_start = clocks()
            sid, tid = _ids_from_kwargs(args, kwargs)
            # Stage root has parent_span == None and span == ""
            frame = span_stack.SpanFrame(
                stage=stage,
                span="",
                parent_span=None,
                started_at=started_at,
                wall_start=wall_start,
                cpu_start=cpu_start,
                session_id=sid,
                trace_id=tid,
                run_id=rec.run_id,
                token_snapshot=rec.snapshot_tokens(),
            )
            span_stack.push(frame)
            ok = True
            error: str | None = None
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                ok = False
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                popped = span_stack.pop()
                after = rec.snapshot_tokens()
                tokens, cost, source = rec.finalize_tokens(
                    after, popped.token_snapshot if popped else {}
                )
                wall_ms = (time.perf_counter() - (popped.wall_start if popped else wall_start)) * 1000.0
                cpu_ms = (time.process_time() - (popped.cpu_start if popped else cpu_start)) * 1000.0
                event = SpanEvent(
                    run_id=rec.run_id,
                    session_id=(popped.session_id if popped else sid),
                    trace_id=(popped.trace_id if popped else tid),
                    stage=stage,
                    span="",
                    parent_span=None,
                    started_at=popped.started_at if popped else started_at,
                    wall_ms=round(wall_ms, 3),
                    cpu_ms=round(cpu_ms, 3),
                    ok=ok,
                    error=error,
                    llm_model=tokens.get("llm_model"),
                    embed_model=tokens.get("embed_model"),
                    tokens={
                        "llm": tokens.get("llm", {}),
                        "embed": tokens.get("embed", {}),
                    },
                    cost_usd=cost,
                    token_source=source,
                    meta=dict(popped.meta) if popped else {},
                )
                try:
                    rec.emit(event)
                except Exception as exc:  # pragma: no cover - surface but never mask primary error
                    logger.error("emit failed: %s", exc, exc_info=True)
                    if ok:
                        raise

        return _wrapper

    return _decorator


# ---------------------------------------------------------------------------
# span(...) context manager
# ---------------------------------------------------------------------------

@contextmanager
def span(name: str, **meta: Any) -> Iterator[span_stack.SpanFrame]:
    """Context manager for a named sub-span inside a decorated stage.

    Usage:
        with span("extract_llm", num_messages=len(msgs)) as s:
            candidates = extractor.extract(...)
            s.meta["num_candidates"] = len(candidates)

    If the recorder is disabled, this is a zero-cost no-op.
    """
    rec = get_recorder()
    if not rec.enabled:
        # Provide a throw-away frame so callers can still write to ``.meta``
        # without branching on ``is_enabled()``.
        _dummy = span_stack.SpanFrame(
            stage="(disabled)", span=name, parent_span=None,
            started_at=0.0, wall_start=0.0, cpu_start=0.0,
        )
        _dummy.meta.update(meta)
        yield _dummy
        return

    parent = span_stack.top()
    stage_frame = span_stack.stage_root()
    stage_name = stage_frame.stage if stage_frame else "(orphan)"

    started_at, wall_start, cpu_start = clocks()
    frame = span_stack.SpanFrame(
        stage=stage_name,
        span=name,
        parent_span=(parent.span if parent else None) or "",
        started_at=started_at,
        wall_start=wall_start,
        cpu_start=cpu_start,
        session_id=parent.session_id if parent else None,
        trace_id=parent.trace_id if parent else None,
        run_id=rec.run_id,
        token_snapshot=rec.snapshot_tokens(),
    )
    frame.meta.update(meta)
    span_stack.push(frame)
    ok = True
    error: str | None = None
    try:
        yield frame
    except Exception as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        popped = span_stack.pop()
        after = rec.snapshot_tokens()
        tokens, cost, source = rec.finalize_tokens(
            after, popped.token_snapshot if popped else {}
        )
        wall_ms = (time.perf_counter() - (popped.wall_start if popped else wall_start)) * 1000.0
        cpu_ms = (time.process_time() - (popped.cpu_start if popped else cpu_start)) * 1000.0
        event = SpanEvent(
            run_id=rec.run_id,
            session_id=(popped.session_id if popped else None),
            trace_id=(popped.trace_id if popped else None),
            stage=stage_name,
            span=name,
            parent_span=(popped.parent_span if popped else None) or None,
            started_at=popped.started_at if popped else started_at,
            wall_ms=round(wall_ms, 3),
            cpu_ms=round(cpu_ms, 3),
            ok=ok,
            error=error,
            llm_model=tokens.get("llm_model"),
            embed_model=tokens.get("embed_model"),
            tokens={
                "llm": tokens.get("llm", {}),
                "embed": tokens.get("embed", {}),
            },
            cost_usd=cost,
            token_source=source,
            meta=dict(popped.meta) if popped else dict(meta),
        )
        try:
            rec.emit(event)
        except Exception as exc:  # pragma: no cover
            logger.error("span emit failed: %s", exc, exc_info=True)
            if ok:
                raise
