"""Optional monotonic deadlines for caller-owned synchronous work.

Deadlines are cooperative: an in-flight storage operation retains its ordinary
receipt/outcome semantics. Cleanup and recording a completed response may finish
after expiry, but callers must check before starting new work.
"""

import math
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

from core.errors import OperationDeadlineExceededError

_deadline = ContextVar("operation_deadline", default=None)


def deadline_expired(deadline: float | None) -> bool:
    """Check an optional absolute deadline on the monotonic clock."""
    return deadline is not None and monotonic() >= deadline


def current_deadline() -> float | None:
    return _deadline.get()


def _bounded_deadline(deadline):
    inherited = current_deadline()
    if inherited is None:
        return deadline
    return inherited if deadline is None else min(inherited, deadline)


def remaining_seconds():
    end = current_deadline()
    if end is None:
        return None
    remaining = end - monotonic()
    if remaining <= 0:
        raise OperationDeadlineExceededError("Operation deadline exceeded")
    return remaining


def check_deadline(deadline=None):
    if deadline_expired(_bounded_deadline(deadline)):
        raise OperationDeadlineExceededError("Operation deadline exceeded")


@contextmanager
def operation_deadline_at(deadline: float | None):
    """Expose an absolute budget to nested work, including copied contexts."""
    end = _bounded_deadline(deadline)
    token = _deadline.set(end)
    try:
        yield end
    finally:
        _deadline.reset(token)


@contextmanager
def write_outcome_confirmation():
    """Give an already-started write ten seconds to confirm its durable receipt."""
    token = _deadline.set(monotonic() + 10.0)
    try:
        yield
    finally:
        _deadline.reset(token)


@contextmanager
def operation_deadline(timeout_seconds=None):
    """Inherit the current deadline, optionally shortening it with a new budget."""
    proposed = None
    if timeout_seconds is not None:
        try:
            valid = (type(timeout_seconds) in (int, float)
                     and math.isfinite(timeout_seconds) and timeout_seconds > 0)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("timeoutSeconds must be a positive finite number")
        proposed = monotonic() + timeout_seconds
    with operation_deadline_at(proposed):
        check_deadline()
        yield
