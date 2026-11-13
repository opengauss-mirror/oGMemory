"""LISTEN/NOTIFY-driven SQL outbox listener."""

from __future__ import annotations

import logging
import select
import threading
import time


logger = logging.getLogger(__name__)


class SQLNotifyListener:
    """Wake the OutboxWorker from PostgreSQL notifications.

    Notifications are only hints. The durable source of truth remains the
    ``outbox_events`` table, so every wake-up drains all ready events until the
    queue is empty.
    """

    def __init__(
        self,
        outbox_store,
        worker,
        worker_id: str = "sql-listener",
        idle_sweep_seconds: float = 30.0,
        reconnect_delay_seconds: float = 1.0,
    ) -> None:
        self._outbox_store = outbox_store
        self._worker = worker
        self._worker_id = worker_id
        self._idle_sweep_seconds = idle_sweep_seconds
        self._reconnect_delay_seconds = reconnect_delay_seconds
        self._stop_event = threading.Event()
        self._listener_conn = None

    @staticmethod
    def _empty_stats() -> dict[str, int]:
        return {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "moved_to_dlq": 0,
            "skipped": 0,
        }

    @staticmethod
    def _merge_stats(target: dict[str, int], update: dict) -> None:
        for key in target:
            target[key] += int(update.get(key, 0))

    def stop(self) -> None:
        """Request shutdown and close any live LISTEN connection."""
        self._stop_event.set()
        if self._listener_conn is not None:
            try:
                self._listener_conn.close()
            except Exception:
                pass

    def _drain_ready_events(self) -> dict[str, int]:
        """Process ready events until no more claimed work remains."""
        totals = self._empty_stats()

        while not self._stop_event.is_set():
            stats = self._worker.run_once(
                outbox_store=self._outbox_store,
                account_ids=[],
                worker_id=self._worker_id,
            )
            self._merge_stats(totals, stats)
            if stats.get("processed", 0) == 0:
                break

        return totals

    def run_cycle(self, conn) -> dict[str, int]:
        """Wait for one signal window, then drain all ready events."""
        if self._stop_event.is_set():
            return self._empty_stats()

        delay = self._outbox_store.future_pending_delay(
            None, exclude_event_types=self._worker.excluded_event_types(),
        )
        timeout = self._idle_sweep_seconds
        if delay is not None:
            # An overdue retry may still be locked by another worker.
            timeout = min(timeout, max(0.1, delay))
        ready, _, _ = select.select([conn], [], [], timeout)
        if ready:
            conn.poll()
            conn.notifies.clear()

        return self._drain_ready_events()

    def run_forever(self) -> None:
        """Run the listener loop with reconnect on transient failures."""
        logger.info(
            "SQLNotifyListener started on channel %s",
            getattr(self._outbox_store, "notify_channel", "ogmem_outbox"),
        )

        while not self._stop_event.is_set():
            conn = None
            try:
                conn = self._outbox_store.open_listener_connection()
                self._listener_conn = conn
                self._drain_ready_events()

                while not self._stop_event.is_set():
                    self.run_cycle(conn)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    "SQLNotifyListener loop error: %s",
                    exc,
                    exc_info=True,
                )
                time.sleep(self._reconnect_delay_seconds)
            finally:
                self._listener_conn = None
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
