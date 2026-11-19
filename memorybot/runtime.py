"""Worker polling, leases and task outcome handling."""

from __future__ import annotations

import logging
import threading
from datetime import timedelta

from memorybot.config import BotConfig
from memorybot.conversation import ChatRun
from memorybot.execution import Execution, _CancelledError
from memorybot.gateway import OutcomeUnknownError, WorkPendingError
from memorybot.models import BotError, new_id
from memorybot.store import LeaseLost
from memorybot.tasks import delivery, ingestion, memory_write

logger = logging.getLogger(__name__)


def _chat(execution):
    ChatRun(execution).run()


TASK_HANDLERS = {
    "chat": _chat,
    "ingest": ingestion.run,
    "memory_write": memory_write.run,
    "delivery": delivery.run,
}


class BotWorker:
    def __init__(self, store, gateway, files, config=None, *, templates=None):
        self.store, self.gateway, self.files = store, gateway, files
        self.config = config or BotConfig()
        self.templates = templates
        self.owner_id = new_id("worker")

    def run_forever(self, stop_event=None):
        stop_event = stop_event or threading.Event()

        def loop():
            owner = new_id("worker")
            while not stop_event.is_set():
                try:
                    busy = self.run_once(owner)
                except Exception:
                    logger.exception("memoryBot worker polling failed")
                    busy = False
                if not busy:
                    stop_event.wait(self.config.poll_seconds)

        threads = [
            threading.Thread(target=loop, name=f"memorybot-slot-{index}")
            for index in range(self.config.worker_concurrency)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def run_once(self, owner_id=None):
        self.store.trigger_due(
            max_pending=self.config.max_pending, max_user_pending=self.config.max_user_pending
        )
        task = self.store.claim(
            owner_id or self.owner_id,
            lease_seconds=self.config.lease_seconds,
            cluster_limit=self.config.cluster_concurrency,
            tenant_limit=self.config.tenant_concurrency,
        )
        if task is None:
            return False
        execution = Execution(self, task)
        renewal_stop = threading.Event()

        # Keep the lease valid while the worker waits for external calls.
        def renew():
            while not renewal_stop.wait(self.config.renew_seconds):
                try:
                    self.store.renew(*execution.identity, lease_seconds=self.config.lease_seconds)
                except Exception:
                    execution.lost.set()
                    return

        renewal_thread = threading.Thread(target=renew, daemon=True, name="memorybot-lease")
        renewal_thread.start()
        try:
            execution.run(TASK_HANDLERS)
        except LeaseLost:
            logger.info("memoryBot lease transferred task=%s", task["task_id"])
        except _CancelledError:
            self.store.finish(
                *execution.identity,
                result=execution.partial,
                state="cancelled",
                mutate=execution.cancel_related,
            )
        except OutcomeUnknownError as exc:
            self.store.wait_confirmation(
                *execution.identity,
                error={
                    "code": "OUTCOME_UNKNOWN",
                    "message": str(exc)[:1024],
                    "details": {
                        "operation_id": execution.operation_id,
                        "reason": str(exc)[:512],
                        "actions": ["recheck", "cancel"],
                    },
                },
            )
        except WorkPendingError:
            # Let other tasks run while a dependency finishes.
            with self.store.execution(*execution.identity) as tx:
                tx.update(
                    "bot_task",
                    {"task_id": task["task_id"]},
                    {
                        "state": "retry_wait",
                        "owner_id": None,
                        "lease_until": None,
                        "next_attempt_at": tx.now() + timedelta(seconds=2),
                    },
                )
                self.store.append_event(
                    tx,
                    task["task_id"],
                    "task.updated",
                    {"state": "retry_wait", "confirmation": None},
                )
        except Exception as exc:
            error = (
                exc.payload()
                if isinstance(exc, BotError)
                else {
                    "code": (
                        "PERMISSION_DENIED" if isinstance(exc, PermissionError) else "TASK_ERROR"
                    ),
                    "message": str(exc)[:1024],
                    "details": {"operation_id": execution.operation_id},
                }
            )
            if (
                isinstance(exc, (BotError, ValueError, PermissionError))
                or task.get("retry_count", 0) >= 3
            ):
                self.store.finish(
                    *execution.identity,
                    result=execution.partial,
                    state="failed",
                    error=error,
                    mutate=execution.fail_related,
                )
            else:
                self.store.retry(*execution.identity, error=error)
        finally:
            renewal_stop.set()
            renewal_thread.join(timeout=1)
        return True
