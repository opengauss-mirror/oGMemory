"""Coalesce graph jobs written by one extraction chunk on its owning session."""

from contextlib import contextmanager
from contextvars import ContextVar
from secrets import randbits

from fs.sql_adapter.session_mutation import SessionMutationBusyError


BATCH_SETTING = "app.graph_projection_batch_key"
_ACTIVE_BATCH = ContextVar("graph_projection_mutation_batch", default=None)


@contextmanager
def graph_projection_batch(guard):
    """Publish a unique batch lease until this chunk's canonical writes finish."""
    if guard is None:
        yield
        return
    active = _ACTIVE_BATCH.get()
    if active is not None and active[0] is guard:
        guard.check_active()
        yield
        return

    guard.ensure_owned()
    key = randbits(63) or 1
    acquired = False
    previous = ""
    context_token = None
    try:
        with guard.raw.cursor() as cursor:
            guard.call(cursor.execute, "SELECT ogmem_graph_projection_batch_key()")
            saved_key = cursor.fetchone()[0]
            previous = str(saved_key) if saved_key is not None else ""
            guard.call(cursor.execute, "SELECT pg_try_advisory_lock(%s)", (key,))
            acquired = bool(cursor.fetchone()[0])
            if not acquired:
                raise SessionMutationBusyError("graph projection batch identity is busy")
            guard.call(cursor.execute, "SELECT set_config(%s, %s, false)", (BATCH_SETTING, str(key)))
        guard.call(guard.raw.commit)
        context_token = _ACTIVE_BATCH.set((guard, key))
        yield
    finally:
        if context_token is not None:
            _ACTIVE_BATCH.reset(context_token)
        if acquired and not guard.poison_reason and not guard.raw.closed:
            if guard.borrowed:
                guard.poison("graph batch ended with an active store transaction")
                guard.check_active()
            guard.call(guard.raw.rollback)
            with guard.raw.cursor() as cursor:
                guard.call(cursor.execute, "SELECT set_config(%s, %s, false)", (BATCH_SETTING, previous))
                guard.call(cursor.execute, "SELECT pg_advisory_unlock(%s)", (key,))
            guard.call(guard.raw.commit)
