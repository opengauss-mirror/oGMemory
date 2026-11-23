# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Event sinks — where ``SpanEvent`` records get persisted."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("ogmem.perf.sinks")


class Sink(Protocol):
    """Minimal sink interface. Implementations must be thread-safe."""

    def emit(self, event: Any) -> None: ...
    def close(self) -> None: ...


def _to_jsonable(event: Any) -> dict:
    if is_dataclass(event):
        return asdict(event)
    if isinstance(event, dict):
        return dict(event)
    raise TypeError(f"unsupported event type for sink: {type(event)!r}")


class MemorySink:
    """In-process sink for tests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[dict] = []

    def emit(self, event: Any) -> None:
        with self._lock:
            self.events.append(_to_jsonable(event))

    def close(self) -> None:  # no-op
        return None

    def get_events(self) -> list[dict]:
        """Return a copy of all accumulated events (thread-safe)."""
        with self._lock:
            return list(self.events)


class JsonlSink:
    """Append-only JSON-lines sink.

    Writes are synchronous and flushed per event so a crashed process
    still leaves a usable trace file. A single ``threading.Lock`` guards
    the file handle so emits from multiple threads stay well-formed.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered so each ``write`` call hits disk promptly.
        self._fh = self._path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, event: Any) -> None:
        line = json.dumps(_to_jsonable(event), default=str, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()


class CompositeSink:
    """Fan-out to multiple sinks."""

    def __init__(self, sinks: list[Sink]):
        self._sinks = list(sinks)

    def emit(self, event: Any) -> None:
        for s in self._sinks:
            s.emit(event)

    def close(self) -> None:
        for s in self._sinks:
            s.close()


class HttpSink:
    """Streaming HTTP sink — POSTs each event as JSON to a remote endpoint.

    Uses a connection pool with a fixed size so concurrent emit calls are
    safe.  Events are fire-and-forget (non-blocking per call); network errors
    are logged but never re-raised so they cannot crash the instrumented
    code path.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 10.0,
        max_pool: int = 4,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._url = url
        self._timeout = timeout
        try:
            import urllib.request
            self._pool: list[urllib.request.OpenerDirector] = []
            for _ in range(max_pool):
                opener = urllib.request.OpenerDirector()
                opener.add_handler(urllib.request.HTTPHandler())
                opener.add_handler(urllib.request.HTTPSHandler())
                self._pool.append(opener)
            self._lock = threading.Lock()
            self._pool_index = 0
        except Exception as exc:
            logger.warning("HttpSink pool init failed, falling back to default opener: %s", exc)
            import urllib.request
            self._pool = [urllib.request.OpenerDirector()]  # type: ignore[attr-defined]
            self._lock = threading.Lock()
            self._pool_index = 0

        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **(headers or {}),
        }

    def _acquire_opener(self):
        with self._lock:
            opener = self._pool[self._pool_index % len(self._pool)]
            self._pool_index += 1
            return opener

    def emit(self, event: Any) -> None:
        import urllib.request

        data = _to_jsonable(event)
        payload = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        opener = self._acquire_opener()
        req = urllib.request.Request(
            self._url,
            data=payload,
            headers=self._headers,
            method="POST",
        )
        try:
            opener.open(req, timeout=self._timeout)
        except Exception as exc:
            logger.warning("HttpSink POST failed to %s: %s", self._url, exc)

    def close(self) -> None:
        pass
