"""Operation-local evidence timings; logging never changes API results."""

import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from uuid import uuid4

logger = logging.getLogger("ogmem.evidence.timing")
_current = ContextVar("evidence_timing", default=None)


@contextmanager
def stage(name):
    timing = _current.get()
    if timing is None:
        yield
        return
    frame = [time.perf_counter(), 0.0]
    timing["stack"].append(frame)
    try:
        yield
    except BaseException as exc:
        timing["errors"].append({"stage": name, "error": type(exc).__name__})
        raise
    finally:
        duration = time.perf_counter() - frame[0]
        timing["stack"].pop()
        timing["exclusive"][name] += duration - frame[1]
        timing["inclusive"][name] += duration
        timing["calls"][name] += 1
        if timing["stack"]:
            timing["stack"][-1][1] += duration


def timed(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with stage(name):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


def request_id():
    timing = _current.get()
    return timing["request_id"] if timing else uuid4().hex


@contextmanager
def operation(name, correlation_id=""):
    if not logger.isEnabledFor(logging.DEBUG):
        yield
        return
    timing = {"request_id": correlation_id or uuid4().hex, "stack": [],
              "exclusive": defaultdict(float), "inclusive": defaultdict(float), "calls": defaultdict(int),
              "errors": []}
    token = _current.set(timing)
    started = time.perf_counter()
    error = None
    try:
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        total = time.perf_counter() - started
        _current.reset(token)
        try:
            payload = {"ts": datetime.now(UTC).isoformat(),
                       "request_id": timing["request_id"], "operation": name, "error": error,
                       "stage_errors": timing["errors"],
                       "total_ms": round(total * 1000, 3),
                       "stages_ms": {key: round(value * 1000, 3) for key, value in timing["exclusive"].items()},
                       "inclusive_ms": {key: round(value * 1000, 3) for key, value in timing["inclusive"].items()},
                       "calls": dict(timing["calls"]),
                       "unattributed_ms": round(max(0, total - sum(timing["exclusive"].values())) * 1000, 3)}
            logger.debug("EVIDENCE_TIMING %s", json.dumps(payload, sort_keys=True))
        except Exception:
            pass  # Observability must not mask a result or the original error.
