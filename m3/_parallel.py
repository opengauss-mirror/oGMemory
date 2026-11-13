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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _resolve_workers(clips: list[Any], max_workers: int | None) -> int:
    workers = env_max_workers() if max_workers is None else max(1, max_workers)
    return min(workers, max(1, len(clips)))


def iter_parallel_clips(
    clips: list[Any],
    unit: Callable[[Any], Any],
    max_workers: int | None = None,
    *,
    label: str = "clip(s)",
):
    """Run ``unit`` for every entry of ``clips`` on a bounded pool, yielding
    ``(clip, result)`` **incrementally, in input order**.

    A result is yielded as soon as every earlier clip has also finished
    (a reorder buffer holds completed out-of-order results until the
    prefix is complete). Incremental consumption lets QA/eval writers
    append + flush each completed answer/judgment while later questions
    are still running, so an interruption no longer discards the batch of
    paid results — everything up to the last unfinished question is
    already on disk and resume finds those rows (review item 8). Output
    order stays deterministic and identical to the input order, which the
    eval results contract requires.

    Tradeoff: a clip that blocks the head of the queue withholds its own
    result and everything after it until it resolves (in practice the
    per-question timeout bounds this — the timed-out error row unblocks
    the stream). A unit that raises yields a :class:`UnitError` result
    instead of aborting the batch. ``max_workers=None`` resolves to
    ``env_max_workers()``; ``workers == 1`` runs synchronously with
    identical semantics (no pool). Abandoning the generator cancels
    queued units; in-flight units are left to finish on their own.
    """
    if not clips:
        return
    for _index, clip, result in _iter_indexed(clips, unit, max_workers):
        yield clip, result


def _iter_indexed(
    clips: list[Any],
    unit: Callable[[Any], Any],
    max_workers: int | None,
):
    """In-order incremental core yielding ``(index, clip, result)``.

    Completion events are buffered; the next expected index is yielded
    as soon as it is available, so the consumer sees a continuous,
    input-ordered stream instead of one final list.
    """
    workers = _resolve_workers(clips, max_workers)
    if workers == 1:
        for index, clip in enumerate(clips):
            try:
                yield index, clip, unit(clip)
            except Exception as exc:  # noqa: BLE001 - collected, not raised
                yield index, clip, UnitError(exc, traceback.format_exc())
        return

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="m3clip")
    try:
        future_to_index = {
            executor.submit(unit, clip): index
            for index, clip in enumerate(clips)
        }
        buffered: dict[int, tuple[Any, Any]] = {}
        next_index = 0

        def _drain():
            nonlocal next_index
            while next_index in buffered:
                clip, result = buffered.pop(next_index)
                yield next_index, clip, result
                next_index += 1

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            clip = clips[index]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - collected, not raised
                result = UnitError(exc, exc.__traceback__)
            buffered[index] = (clip, result)
            yield from _drain()
    finally:
        # Normal exhaustion: every future is done, this is a no-op.
        # Early abandonment (GeneratorExit / exception in the consumer):
        # cancel the queued units instead of blocking on them.
        executor.shutdown(wait=False, cancel_futures=True)


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

    Implemented on top of :func:`iter_parallel_clips` (which yields in
    completion order); consumers that checkpoint incrementally should use
    the iterator directly.
    """
    if not clips:
        return []
    results: list[tuple[Any, Any] | None] = [None] * len(clips)
    for index, clip, result in _iter_indexed(clips, unit, max_workers):
        results[index] = (clip, result)
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
