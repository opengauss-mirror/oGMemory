"""Owner-scoped vector components shared by graph projection workers."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingComponentKey:
    owner: str
    profile_id: str
    model: str
    dimension: int
    representation: str
    kind: str
    text: str
    preprocessing: tuple[tuple[str, str], ...] = ()


class EmbeddingComponentCache:
    """Bounded LRU with one computation per concurrent component key."""

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries < 1:
            raise ValueError("component cache capacity must be positive")
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._vectors: OrderedDict[EmbeddingComponentKey, tuple[float, ...]] = OrderedDict()
        self._inflight: dict[EmbeddingComponentKey, Future] = {}
        self._hits = 0
        self._misses = 0
        self._waits = 0

    def snapshot(self) -> dict[str, int]:
        """Return instance totals for unique component lookups per resolve call."""
        with self._lock:
            return dict(hits=self._hits, misses=self._misses, waits=self._waits,
                        entries=len(self._vectors), inflight=len(self._inflight), capacity=self._max_entries)

    def resolve(
        self,
        keys: Sequence[EmbeddingComponentKey],
        compute_many: Callable[[list[EmbeddingComponentKey]], Sequence[Sequence[float]]],
    ) -> dict[EmbeddingComponentKey, tuple[float, ...]]:
        values: dict[EmbeddingComponentKey, tuple[float, ...]] = {}
        owned: dict[EmbeddingComponentKey, Future] = {}
        waiting: dict[EmbeddingComponentKey, Future] = {}
        with self._lock:
            for key in dict.fromkeys(keys):
                if key in self._vectors:
                    self._hits += 1
                    values[key] = self._vectors[key]
                    self._vectors.move_to_end(key)
                elif key in self._inflight:
                    self._waits += 1
                    waiting[key] = self._inflight[key]
                else:
                    self._misses += 1
                    owned[key] = self._inflight[key] = Future()
        if owned:
            try:
                vectors = list(compute_many(list(owned)))
                if len(vectors) != len(owned):
                    raise ValueError("component vector count must match input count")
                computed = {key: tuple(float(value) for value in vector)
                            for key, vector in zip(owned, vectors, strict=True)}
            except BaseException as exc:
                with self._lock:
                    for key, future in owned.items():
                        self._inflight.pop(key, None)
                        future.set_exception(exc)
                raise
            with self._lock:
                for key, vector in computed.items():
                    self._vectors[key] = vector
                    self._vectors.move_to_end(key)
                    self._inflight.pop(key, None)
                    owned[key].set_result(vector)
                while len(self._vectors) > self._max_entries:
                    self._vectors.popitem(last=False)
            values.update(computed)
        # Publish owned work before joining another worker's component futures.
        for key, future in waiting.items():
            values[key] = future.result()
        return values
