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


def remaining_seconds():
    end = _deadline.get()
    if end is None:
        return None
    remaining = end - monotonic()
    if remaining <= 0:
        raise OperationDeadlineExceededError("Operation deadline exceeded")
    return remaining


def check_deadline():
    remaining_seconds()


@contextmanager
def operation_deadline(timeout_seconds=None):
    """Inherit the current deadline, optionally shortening it with a new budget."""
    end = _deadline.get()
    if timeout_seconds is not None:
        try:
            valid = (type(timeout_seconds) in (int, float)
                     and math.isfinite(timeout_seconds) and timeout_seconds > 0)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("timeoutSeconds must be a positive finite number")
        proposed = monotonic() + timeout_seconds
        end = proposed if end is None else min(end, proposed)
    token = _deadline.set(end)
    try:
        check_deadline()
        yield
    finally:
        _deadline.reset(token)
