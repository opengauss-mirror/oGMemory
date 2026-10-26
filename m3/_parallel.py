"""Shared parallel-execution primitive for M3 pipeline drivers.

Every per-clip / per-question loop in ``m3/`` funnels through
:func:`parallel_clips`, a bounded thread pool that preserves the input
order in the results and never lets one failing unit abort the batch.
Drivers keep their "collect failures, keep going" semantics: a unit that
raises is returned as a :class:`UnitError` which the main thread logs
with its original traceback and counts as a failure.

Thread-safety contract (callers rely on this):

* SQL graph stores and vector indexes — **safe to share** across worker
  threads. They use bounded connection pools (see
  ``fs/sql_adapter/pool.py``); raise the pool size to at least the
  worker count (``OGMEM_GRAPH_POOL_SIZE`` / vector pool) or threads will
  queue on a pool slot.
* OpenAI-compatible LLM clients — **safe to share** (the SDK client and
  the token tracker are thread-safe).
* Local model backends (faster-whisper CTranslate2, torch / pyannote
  models) and ``OGMemoryControlAdapter`` (its ``run()`` rejects
  concurrent questions) — **NOT safe to share**. Build one per worker
  thread with :class:`ThreadLocalModels`.
* ``RequestContext`` is a frozen dataclass; worker units should pass a
  per-clip copy with a unique ``trace_id`` (``dataclasses.replace(ctx,
  trace_id=...)``) so log lines correlate with the originating clip.
"""

from __future__ import annotations

import logging
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


def env_max_workers(*, default: int = 8, env: str = "OGMEM_M3_MAX_WORKERS") -> int:
    """Read a worker-count env var, clamped to ``[1, 128]``.

    Garbage values fall back to ``default`` with a warning.
    """
    raw = os.environ.get(env)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r — using default %d", env, raw, default)
        return default
    if value < 1:
        return 1
    if value > 128:
        return 128
    return value


def face_max_workers() -> int:
    """Per-frame face-detection fan-out cap.

    Face detection is an HTTP fan-out against a single ``cv_service``
    whose ONNX-runtime session serialises ``detect()``; a small cap
    avoids request pile-up.
    """
    return env_max_workers(default=4, env="OGMEM_M3_FACE_WORKERS")


@dataclass(frozen=True)
class UnitError:
    """A failed unit, carrying the original exception and traceback.

    Callers re-emit faithful ``logger.error(..., exc_info=...)`` lines in
    the main thread. ``__str__`` returns the exception message so
    existing ``"...: %s"`` log formatting keeps working.
    """

    exception: BaseException
    tb: traceback.TracebackType | str | None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.exception)


def parallel_clips(
    clips: list[Any],
    unit: Callable[[Any], Any],
    max_workers: int | None = None,
    *,
    label: str = "clip(s)",
) -> list[tuple[Any, Any]]:
    """Run ``unit`` for every entry of ``clips`` on a bounded pool.

    Returns ``[(clip0, result0), ...]`` in **input order**. A unit that
    raises produces a :class:`UnitError` result instead of aborting the
    batch. ``max_workers=None`` resolves to ``env_max_workers()`` and is
    capped at ``len(clips)``; ``workers == 1`` runs synchronously with
    identical semantics (no pool).
    """
    if not clips:
        return []
    workers = env_max_workers() if max_workers is None else max(1, max_workers)
    workers = min(workers, len(clips))
    if workers == 1:
        results = []
        for clip in clips:
            try:
                results.append((clip, unit(clip)))
            except Exception as exc:  # noqa: BLE001 - collected, not raised
                results.append((clip, UnitError(exc, traceback.format_exc())))
        return results

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="m3clip") as executor:
        futures = {index: executor.submit(unit, clip) for index, clip in enumerate(clips)}
        results: list[tuple[Any, Any] | None] = [None] * len(clips)
        for index, future in futures.items():
            try:
                results[index] = (clips[index], future.result())
            except Exception as exc:  # noqa: BLE001 - collected, not raised
                results[index] = (clips[index], UnitError(exc, exc.__traceback__))
    return [entry for entry in results if entry is not None]


class ThreadLocalModels:
    """One model-backed object per worker thread, built lazily.

    For backends that are not safe to share across threads
    (faster-whisper CTranslate2 models, torch models, and
    ``OGMemoryControlAdapter``), each worker thread gets its own instance
    from ``factory`` on first :meth:`get` in that thread.
    """

    def __init__(self, factory: Callable[[], Any], *, name: str = "model"):
        self._factory = factory
        self._name = name
        self._local = threading.local()

    def get(self) -> Any:
        instance = getattr(self._local, "instance", None)
        if instance is None:
            instance = self._factory()
            self._local.instance = instance
            logger.debug("ThreadLocalModels: built %s for thread %s", self._name, threading.current_thread().name)
        return instance
