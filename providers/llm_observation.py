"""Opt-in, call-local experiment evidence; never used as recovery authority.

Collectors own retention and access to captured content. Default capture contains
only stage, correlation and usage; content requires an explicit private collector.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from threading import Lock
from uuid import uuid4

_capture = ContextVar('llm_experiment_capture', default=None)
_request = ContextVar('llm_observed_request', default=None)


@contextmanager
def capture_llm_events(*, include_content=False):
    events = []
    token = _capture.set((events, Lock(), include_content))
    try:
        yield events
    finally:
        _capture.reset(token)


def _emit(event, **fields):
    capture, request = _capture.get(), _request.get()
    if capture is None or request is None:
        return
    events, lock, _ = capture
    with lock:
        events.append({**request, 'event': event, **fields})


def _copy_content(**fields):
    try:
        return deepcopy(fields)
    except Exception:
        return {'content_unavailable': True}


def observe_call(stage, invoke, *args, correlation=None, **kwargs):
    """Preserve provider behavior while correlating actual requests and usage."""
    capture = _capture.get()
    if capture is None:
        return invoke(*args, **kwargs)
    parent = _request.get() or {}
    request = {**parent, **(correlation or {}), 'stage': stage,
               'request_id': uuid4().hex}
    token = _request.set(request)
    try:
        content = _copy_content(args=args, kwargs=kwargs) if capture[2] else {}
        _emit('request', **content)
        result = invoke(*args, **kwargs)
        with capture[1]:
            usage_observed = any(e.get('request_id') == request['request_id']
                                 and e['event'] == 'usage' for e in capture[0])
        _emit('result', usage_observed=usage_observed,
              **(_copy_content(result=result) if capture[2] else {}))
        return result
    except Exception as exc:
        _emit('error', error_type=type(exc).__name__)
        raise
    finally:
        _request.reset(token)


def record_observed_usage(*, input_tokens=None, output_tokens=None, model=None,
                          failed=False, usage_known=True, event_id=None):
    """One event per physical provider response/failure, independent of tools."""
    _emit('usage', provider_event_id=event_id or uuid4().hex, model=model,
          input_tokens=input_tokens if usage_known else None,
          output_tokens=output_tokens if usage_known else None,
          total_tokens=(input_tokens + output_tokens) if usage_known else None,
          failed=failed, usage_known=usage_known)


def correlate_extraction_calls(invoke):
    """Carry existing span/session identity through native worker contexts."""
    from functools import wraps
    from inspect import signature
    parameters = signature(invoke)

    @wraps(invoke)
    def wrapped(*args, **kwargs):
        if _capture.get() is None:
            return invoke(*args, **kwargs)
        bound = parameters.bind(*args, **kwargs).arguments
        ctx = bound.get('ctx')
        prepared = bound.get('prepared')
        span = bound.get('span') or getattr(prepared, 'span', None)
        metadata = dict(_request.get() or {})
        metadata.setdefault('operation_id', uuid4().hex)
        metadata.setdefault('request_id', uuid4().hex)
        if ctx is not None:
            metadata.update({name: getattr(ctx, name, '') for name in
                             ('account_id', 'session_id', 'user_id', 'trace_id')})
        if span is not None:
            metadata['span_id'] = f'{span.start}:{span.end}'
        if bound.get('archive_id') is not None:
            metadata['archive_id'] = bound['archive_id']
        token = _request.set(metadata)
        try:
            return invoke(*args, **kwargs)
        finally:
            _request.reset(token)
    return wrapped


def record_assembly_event(*, selection=None, decision=None, attempt=None):
    """Capture bounded assembly metadata, separate from physical model usage."""
    _emit('assembly', **_copy_content(selection=selection, decision=decision, attempt=attempt))
