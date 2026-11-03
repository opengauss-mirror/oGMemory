"""Unit tests for the shared M3 parallel primitive (m3/_parallel.py)."""

import threading

import pytest

from m3 import _parallel
from m3._parallel import ThreadLocalModels, UnitError, env_max_workers, parallel_clips


def test_empty_input_returns_empty():
    assert parallel_clips([], lambda c: c) == []


def test_results_preserve_input_order():
    clips = list(range(20))
    results = parallel_clips(clips, lambda c: c * 10, max_workers=8)
    assert results == [(c, c * 10) for c in clips]


def test_unit_error_does_not_abort_batch():
    def unit(c):
        if c % 3 == 0:
            raise ValueError(f"boom {c}")
        return c

    results = parallel_clips(list(range(6)), unit, max_workers=4)
    assert len(results) == 6
    for clip, result in results:
        if clip % 3 == 0:
            assert isinstance(result, UnitError)
            assert str(result) == f"boom {clip}"
            assert result.tb  # traceback carried for re-emission
        else:
            assert result == clip


def test_single_worker_is_synchronous():
    seen_threads = set()

    def unit(c):
        seen_threads.add(threading.current_thread().name)
        return c

    results = parallel_clips([1, 2, 3], unit, max_workers=1)
    assert results == [(1, 1), (2, 2), (3, 3)]
    assert seen_threads == {threading.current_thread().name}


def test_env_max_workers_parsing(monkeypatch):
    monkeypatch.setenv("OGMEM_M3_MAX_WORKERS", "12")
    assert env_max_workers() == 12
    monkeypatch.setenv("OGMEM_M3_MAX_WORKERS", "0")
    assert env_max_workers() == 1
    monkeypatch.setenv("OGMEM_M3_MAX_WORKERS", "999")
    assert env_max_workers() == 128
    monkeypatch.setenv("OGMEM_M3_MAX_WORKERS", "garbage")
    assert env_max_workers() == 8
    monkeypatch.delenv("OGMEM_M3_MAX_WORKERS")
    assert env_max_workers() == 8


def test_face_max_workers_default_and_env(monkeypatch):
    monkeypatch.delenv("OGMEM_M3_FACE_WORKERS", raising=False)
    assert _parallel.face_max_workers() == 4
    monkeypatch.setenv("OGMEM_M3_FACE_WORKERS", "2")
    assert _parallel.face_max_workers() == 2


def test_parallel_clips_env_default_workers(monkeypatch):
    monkeypatch.setenv("OGMEM_M3_MAX_WORKERS", "3")
    results = parallel_clips([1, 2, 3, 4, 5], lambda c: c)
    assert [r for _, r in results] == [1, 2, 3, 4, 5]


def test_thread_local_models_one_instance_per_thread():
    built = []

    def factory():
        obj = object()
        built.append(obj)
        return obj

    models = ThreadLocalModels(factory, name="test-model")
    main_obj = models.get()
    assert models.get() is main_obj  # same thread -> same instance
    assert len(built) == 1

    other = []
    thread = threading.Thread(target=lambda: other.append(models.get()))
    thread.start()
    thread.join()
    assert other[0] is not main_obj
    assert len(built) == 2


class TestIterParallelClips:
    """Review item 8: iter_parallel_clips yields results as units COMPLETE,
    so QA/eval writers can checkpoint finished answers while stragglers run."""

    def test_yields_incrementally_in_input_order(self):
        import threading
        from m3._parallel import iter_parallel_clips
        release = threading.Event()

        def unit(x):
            if x == 3:
                release.wait(timeout=10)  # straggler at the TAIL
            return x * 10

        # Clips 0..2 complete while the tail straggler runs: each is
        # yielded as soon as its prefix is complete — no waiting for the
        # whole batch (review item 8).
        it = iter_parallel_clips([0, 1, 2, 3], unit, 2)
        assert next(it) == (0, 0)
        assert next(it) == (1, 10)
        assert next(it) == (2, 20)
        release.set()
        assert next(it) == (3, 30)
        with pytest.raises(StopIteration):
            next(it)

    def test_head_of_queue_straggler_withholds_only_its_suffix(self):
        import threading
        from m3._parallel import iter_parallel_clips
        release = threading.Event()

        def unit(x):
            if x == 0:
                release.wait(timeout=10)  # head-of-queue straggler
            return x * 10

        it = iter_parallel_clips([0, 1, 2, 3], unit, 2)
        release.set()
        # In-order streaming: the head blocks its suffix (documented
        # tradeoff; per-question timeouts bound it in practice), and the
        # whole stream then flows in input order.
        assert [next(it) for _ in range(4)] == [(0, 0), (1, 10), (2, 20), (3, 30)]
        with pytest.raises(StopIteration):
            next(it)

    def test_abandoned_generator_cancels_queued_units(self):
        import threading
        from m3._parallel import iter_parallel_clips
        started = threading.Event()
        release = threading.Event()
        calls = []

        def unit(x):
            calls.append(x)
            if x == 0:
                started.set()
                release.wait(timeout=10)
            return x

        it = iter_parallel_clips(list(range(20)), unit, 2)
        started.wait(timeout=10)     # first unit occupies a worker
        it.close()                   # abandon: queued units must be cancelled
        release.set()
        # Only the in-flight units ran (at most 2), not the whole queue.
        assert len(calls) <= 2

    def test_unit_errors_are_yielded_not_raised(self):
        from m3._parallel import iter_parallel_clips, UnitError
        results = dict(iter_parallel_clips(
            ["a", "b"], lambda x: (_ for _ in ()).throw(RuntimeError("boom")), 2))
        assert isinstance(results["a"], UnitError)
        assert isinstance(results["b"], UnitError)

    def test_parallel_clips_wrapper_preserves_input_order(self):
        import time
        from m3._parallel import parallel_clips

        def unit(x):
            time.sleep((4 - x) * 0.01)  # reverse completion order
            return x

        assert parallel_clips([0, 1, 2, 3], unit, 4) == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_parallel_clips_wrapper_handles_duplicate_clips(self):
        from m3._parallel import parallel_clips
        shared = {"k": 1}
        clips = [shared, shared, {"k": 2}]
        results = parallel_clips(clips, lambda c: c["k"], 3)
        assert results == [(shared, 1), (shared, 1), ({"k": 2}, 2)]
