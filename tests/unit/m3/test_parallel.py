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


def test_thread_local_models_warmup_builds_in_parallel():
    built = []

    def factory():
        obj = object()
        built.append(obj)
        return obj

    models = ThreadLocalModels(factory, name="test-model")
    models.warmup(4)
    assert len(built) == 4


@pytest.mark.parametrize("count", [0, -1])
def test_thread_local_models_warmup_noop(count):
    models = ThreadLocalModels(object, name="test-model")
    models.warmup(count)  # must not raise
