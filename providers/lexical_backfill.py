"""Lifecycle-owned canonical-data backfill for lexical projections."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence

from core.canonical_scanner import CanonicalNodeScanner
from core.lexical import lexical_record_from_index_record
from index.index_record_builder import build_index_records

logger = logging.getLogger(__name__)
BackfillKey = tuple[str, str]


class LexicalBackfill:
    """Run bounded rebuild jobs on one scheduler with autonomous retries."""

    def __init__(
        self,
        scanner: CanonicalNodeScanner,
        lexical_index,
        *,
        page_size: int = 500,
        max_pending: int = 16,
        retry_base_seconds: float = 5.0,
        max_attempts: int = 5,
        circuit_cooldown_seconds: float = 300.0,
    ) -> None:
        if page_size < 1 or max_pending < 1:
            raise ValueError("backfill bounds must be positive")
        if (
            retry_base_seconds < 0.1
            or max_attempts < 1
            or circuit_cooldown_seconds < 0.1
        ):
            raise ValueError("backfill retry bounds are invalid")
        self._scanner = scanner
        self._lexical = lexical_index
        self._page_size = page_size
        self._max_pending = max_pending
        self._retry_base_seconds = retry_base_seconds
        self._max_attempts = max_attempts
        self._circuit_cooldown_seconds = circuit_cooldown_seconds
        self._condition = threading.Condition()
        self._jobs: dict[BackfillKey, float] = {}
        self._failure_count: dict[BackfillKey, int] = {}
        self._last_error: OrderedDict[BackfillKey, str] = OrderedDict()
        self._running: BackfillKey | None = None
        self._worker: threading.Thread | None = None
        self._stopping = False
        self._saturated = 0
        self._cooldowns: dict[BackfillKey, float] = {}
        self._terminal_failures = 0

    def ensure_ready(self, ctx, owner_spaces: Sequence[str]) -> bool:
        scopes = tuple(dict.fromkeys(str(value) for value in owner_spaces if value))
        if not scopes:
            return False
        if not self._scanner.projection_is_quiescent(account_id=ctx.account_id):
            return False
        ready = self._lexical.ready_scopes(ctx.account_id, scopes)
        if ready == frozenset(scopes):
            return True
        with self._condition:
            if self._stopping:
                return False
            now = time.monotonic()
            for key, expires_at in tuple(self._cooldowns.items()):
                if expires_at <= now:
                    self._cooldowns.pop(key, None)
            scheduled: list[BackfillKey] = []
            for scope in (scope for scope in scopes if scope not in ready):
                key = (ctx.account_id, scope)
                if key in self._cooldowns or key in self._jobs or key == self._running:
                    continue
                if self._active_count_locked() >= self._max_pending:
                    self._saturated += 1
                    break
                self._jobs[key] = time.monotonic()
                scheduled.append(key)
            try:
                self._ensure_worker_locked()
            except Exception as exc:
                for key in scheduled:
                    self._jobs.pop(key, None)
                    self._record_error_locked(key, exc)
                return False
            self._condition.notify_all()
        return False

    def _ensure_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        worker = threading.Thread(
            target=self._run_scheduler,
            daemon=True,
            name="lexical-backfill",
        )
        worker.start()
        self._worker = worker

    def _active_count_locked(self) -> int:
        return len(self._jobs) + len(self._cooldowns) + int(self._running is not None)

    def _run_scheduler(self) -> None:
        while True:
            with self._condition:
                while not self._stopping and not self._jobs:
                    self._condition.wait()
                if self._stopping:
                    return
                key, due_at = min(self._jobs.items(), key=lambda item: item[1])
                remaining = due_at - time.monotonic()
                if remaining > 0:
                    self._condition.wait(remaining)
                    continue
                self._jobs.pop(key, None)
                self._running = key
            succeeded = self._run_once(key)
            with self._condition:
                self._running = None
                if not succeeded and not self._stopping:
                    failures = self._failure_count.get(key, 1)
                    if failures >= self._max_attempts:
                        self._terminal_failures += 1
                        self._cooldowns[key] = (
                            time.monotonic() + self._circuit_cooldown_seconds
                        )
                        self._failure_count.pop(key, None)
                    else:
                        delay = min(
                            300.0,
                            self._retry_base_seconds * (2 ** min(failures - 1, 16)),
                        )
                        self._jobs[key] = time.monotonic() + delay
                self._condition.notify_all()

    def _run_once(self, key: BackfillKey) -> bool:
        account_id, owner_space = key
        scopes = (owner_space,)
        rebuild_id = None
        try:
            rebuild_id = self._lexical.begin_rebuild()
            nodes = self._scanner.scan_active_nodes(
                account_id=account_id,
                owner_spaces=scopes,
                page_size=self._page_size,
            )
            records = (
                lexical_record_from_index_record(record)
                for node in nodes
                for record in build_index_records(node)
                if record.level in (0, 1, 2)
            )
            if not self._lexical.replace_scopes(
                rebuild_id=rebuild_id,
                account_id=account_id,
                owner_spaces=scopes,
                records=records,
            ):
                raise RuntimeError("lexical rebuild was superseded by a newer write")
            with self._condition:
                self._failure_count.pop(key, None)
                self._last_error.pop(key, None)
            return True
        except Exception as exc:
            if rebuild_id is not None:
                try:
                    self._lexical.abort_rebuild(rebuild_id)
                except Exception:
                    logger.warning("Unable to abort lexical rebuild", exc_info=True)
            with self._condition:
                self._failure_count[key] = self._failure_count.get(key, 0) + 1
                self._record_error_locked(key, exc)
            logger.warning("Lexical backfill failed for %s %s", account_id, scopes, exc_info=True)
            return False

    def _record_error_locked(self, key: BackfillKey, exc: Exception) -> None:
        self._last_error.pop(key, None)
        self._last_error[key] = f"{type(exc).__name__}: {exc}"
        while len(self._last_error) > self._max_pending:
            self._last_error.popitem(last=False)

    def health(self) -> dict:
        with self._condition:
            return {
                "queued": len(self._jobs),
                "running": self._running is not None,
                "capacity": self._max_pending,
                "saturation_events": self._saturated,
                "at_capacity": self._active_count_locked() >= self._max_pending,
                "terminal_failures": self._terminal_failures,
                "cooldown_failures": len(self._cooldowns),
                "stopping": self._stopping,
                "last_errors": list(self._last_error.values()),
            }

    def close(self, timeout: float = 5.0) -> bool:
        with self._condition:
            self._stopping = True
            self._jobs.clear()
            self._condition.notify_all()
            worker = self._worker
        if worker is None or worker is threading.current_thread():
            return True
        worker.join(timeout=max(0.0, timeout))
        return not worker.is_alive()
