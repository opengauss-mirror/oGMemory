# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Async-safe + thread-safe span stack using ``contextvars``.

Each running task/thread keeps its own ``SpanFrame`` stack so nested
``span()`` blocks inside decorated stages can attribute time + tokens to
the right parent.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpanFrame:
    """A live span pushed onto the stack while its block executes."""

    stage: str                    # top-level lifecycle stage (bootstrap/compose/...)
    span: str                     # "" for the stage root, else a sub-span name
    parent_span: str | None
    started_at: float             # epoch seconds (time.time())
    wall_start: float             # time.perf_counter() snapshot
    cpu_start: float              # time.process_time() snapshot
    session_id: str | None = None
    trace_id: str | None = None
    run_id: str | None = None
    # Token snapshot captured at span entry for diffing at exit
    token_snapshot: dict[str, Any] = field(default_factory=dict)
    # Free-form counters the caller can fill in
    meta: dict[str, Any] = field(default_factory=dict)


_current_stack: ContextVar[list[SpanFrame]] = ContextVar(
    "ogmem_perf_span_stack", default=[]
)


def current_stack() -> list[SpanFrame]:
    """Return the current (copy-on-write) span stack for this task."""
    return list(_current_stack.get())


def push(frame: SpanFrame) -> None:
    """Push a new frame onto the stack."""
    stack = list(_current_stack.get())
    stack.append(frame)
    _current_stack.set(stack)


def pop() -> SpanFrame | None:
    """Pop the top frame. Returns ``None`` if the stack is empty."""
    stack = list(_current_stack.get())
    if not stack:
        return None
    top = stack.pop()
    _current_stack.set(stack)
    return top


def top() -> SpanFrame | None:
    """Peek at the top frame without popping."""
    stack = _current_stack.get()
    if not stack:
        return None
    return stack[-1]


def stage_root() -> SpanFrame | None:
    """Return the bottom-most stage frame (the lifecycle entry)."""
    stack = _current_stack.get()
    if not stack:
        return None
    return stack[0]


def reset() -> None:
    """Clear the stack. Used by tests and the dispose path."""
    _current_stack.set([])
