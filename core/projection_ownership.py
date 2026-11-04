"""Carry the active outbox claim to the index mutation boundary."""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

_owner: ContextVar[Callable[[], None] | None] = ContextVar("projection_owner", default=None)


@contextmanager
def projection_owner(check):
    token = _owner.set(check)
    try:
        yield
    finally:
        _owner.reset(token)


def current_projection_owner():
    return _owner.get()


def check_projection_owner():
    check = current_projection_owner()
    if check is not None:
        check()
