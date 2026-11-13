"""Stable model identities and owner-thread collection of prepared stages."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from contextvars import ContextVar, copy_context


_collectors = ContextVar("model_stage_collectors", default=())
_collecting = ContextVar("model_stage_collecting", default=False)


def stage_digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_signature(llm, **overrides):
    """Identify the provider and effective generation configuration."""
    names = ("_model", "model", "_temperature", "temperature", "_max_tokens", "max_tokens",
             "_json_mode", "_reasoning_effort", "_is_reasoning_model_override",
             "_extra_body", "_base_url", "base_url")
    fields = vars(llm) if hasattr(llm, "__dict__") else {}
    config = {name: fields[name] for name in names if name in fields}
    config.update(overrides)
    return stage_digest({"class": f"{type(llm).__module__}.{type(llm).__qualname__}",
                         "configuration": config})


@contextmanager
def stage_collector_scope(callback):
    """Register a collector on the thread that owns its persistence context."""
    token = _collectors.set((*_collectors.get(), (threading.get_ident(), callback)))
    try:
        yield
    finally:
        _collectors.reset(token)


def collect_stage_results():
    if _collecting.get():
        return
    token = _collecting.set(True)
    try:
        owner = threading.get_ident()
        for thread_id, callback in _collectors.get():
            if thread_id == owner:
                callback()
    finally:
        _collecting.reset(token)


def run_model_call(call, *args, **kwargs):
    """Compute a model reply while its owner persists completed sibling stages."""
    owner = threading.get_ident()
    if not any(thread_id == owner for thread_id, _ in _collectors.get()):
        return call(*args, **kwargs)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-stage") as pool:
        future = pool.submit(copy_context().run, call, *args, **kwargs)
        return wait_model_result(future)


def wait_model_result(future):
    """Join a pure model computation while the owner collects sibling results."""
    while True:
        collect_stage_results()
        try:
            result = future.result(timeout=0.1)
        except TimeoutError:
            if future.done():
                return future.result()
            continue
        collect_stage_results()
        return result
