"""Track durable Outbox references for one write invocation."""

from contextvars import ContextVar
from functools import wraps

_events = ContextVar("request_outbox_events", default=None)
_untracked_accounts = ContextVar("request_outbox_untracked_accounts", default=None)


def is_capturing_outbox_events():
    return _events.get() is not None


def record_persisted_event(node_uri, event_id):
    events = _events.get()
    if events is not None:
        events[(node_uri, event_id)] = None


def capture_outbox_events(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _events.get() is not None:
            return function(*args, **kwargs)
        token = _events.set({})
        untracked_token = _untracked_accounts.set(set())
        try:
            return function(*args, **kwargs)
        finally:
            _untracked_accounts.reset(untracked_token)
            _events.reset(token)
    return wrapped


def remember_persisted_event(function):
    @wraps(function)
    def wrapped(self, event, node_uri):
        result = function(self, event, node_uri)
        record_persisted_event(node_uri, event.event_id)
        return result
    return wrapped


def current_outbox_events(account_id=None):
    events = _events.get()
    if events is None:
        return None
    untracked_accounts = _untracked_accounts.get() or set()
    if untracked_accounts and (account_id is None or account_id in untracked_accounts):
        return None
    prefix = f"ctx://{account_id}/" if account_id else "ctx://"
    return [(uri, event_id) for uri, event_id in events if uri.startswith(prefix)]


def restore_outbox_events(batch, account_id):
    """Include durable references when continuing an accepted source batch."""
    if _events.get() is None:
        return
    uncertain_write = any(operation.get("status") == "writing"
                          for slot in batch.get("slots", {}).values()
                          for operation in slot.get("operations", {}).values())
    uncertain_archive = batch.get("lifecycle", {}).get("archive", {}).get("status") == "writing"
    if batch.get("outbox_tracking_version") != 1 or uncertain_write or uncertain_archive:
        _untracked_accounts.get().add(account_id)
    for uri, event_id in batch.get("outbox_events", []):
        if uri.startswith(f"ctx://{account_id}/"):
            record_persisted_event(uri, event_id)
