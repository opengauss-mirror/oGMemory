# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Outbox scheduler for periodic polling.

This module provides the OutboxScheduler class that wraps APScheduler
to periodically trigger OutboxWorker.run_once() for async index sync.
"""

import logging
import signal
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED, JobEvent
from apscheduler.schedulers.background import BackgroundScheduler

from commit.outbox_store import OutboxStore
from index.outbox_worker import OutboxWorker

logger = logging.getLogger(__name__)


@dataclass
class SchedulerMetrics:
    """Scheduler runtime metrics."""
    worker_id: str
    last_run_time: datetime | None = None
    last_run_stats: dict | None = None
    total_processed: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    total_dlq: int = 0
    consecutive_errors: int = 0
    is_running: bool = False


class OutboxScheduler:
    """Scheduler that periodically triggers OutboxWorker.run_once().
    
    Usage:
        scheduler = OutboxScheduler(
            worker=worker,
            outbox_store=store,
            get_account_ids=get_active_account_ids,
            worker_id="worker-1",
            interval_seconds=30,
        )
        scheduler.start()
        
        # Later...
        scheduler.stop(wait=True)
    """
    
    def __init__(
        self,
        worker: OutboxWorker,
        outbox_store: OutboxStore,
        get_account_ids: Callable[[], list[str]],
        worker_id: str | None = None,
        interval_seconds: int = 30,
        misfire_grace_time: int = 10,
    ):
        """Initialize the scheduler.
        
        Args:
            worker: OutboxWorker instance for processing events
            outbox_store: OutboxStore instance for event persistence
            get_account_ids: Callable that returns list of active account IDs
            worker_id: Unique worker identifier (auto-generated if None)
            interval_seconds: Polling interval in seconds
            misfire_grace_time: Grace period for missed executions in seconds
        """
        self._worker = worker
        self._store = outbox_store
        self._get_account_ids = get_account_ids
        self._worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self._interval = interval_seconds
        self._misfire_grace_time = misfire_grace_time
        self._scheduler = BackgroundScheduler()
        self._running = False
        self._lock = threading.Lock()
        self._metrics = SchedulerMetrics(worker_id=self._worker_id)
    
    @property
    def worker_id(self) -> str:
        """Return the worker ID."""
        return self._worker_id
    
    @property
    def metrics(self) -> SchedulerMetrics:
        """Return current metrics."""
        return self._metrics
    
    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running
    
    def start(self) -> None:
        """Start the scheduler."""
        with self._lock:
            if self._running:
                logger.warning(f"Scheduler already running, worker_id={self._worker_id}")
                return
            
            self._scheduler.add_job(
                func=self._run_once_wrapped,
                trigger="interval",
                seconds=self._interval,
                max_instances=1,
                misfire_grace_time=self._misfire_grace_time,
                coalesce=True,
                id="outbox_worker",
                name=f"outbox_worker_{self._worker_id}",
            )
            
            self._scheduler.add_listener(
                self._on_job_error,
                EVENT_JOB_ERROR | EVENT_JOB_MISSED
            )
            
            self._scheduler.start()
            self._running = True
            self._metrics.is_running = True
            logger.info(f"Scheduler started, worker_id={self._worker_id}, interval={self._interval}s")
    
    def stop(self, wait: bool = True) -> None:
        """Stop the scheduler.
        
        Args:
            wait: Whether to wait for current job to complete
        """
        with self._lock:
            if not self._running:
                return
            
            self._scheduler.shutdown(wait=wait)
            self._running = False
            self._metrics.is_running = False
            logger.info(f"Scheduler stopped, worker_id={self._worker_id}")
    
    def _run_once_wrapped(self) -> dict:
        """Wrap run_once with logging and metrics tracking."""
        self._metrics.last_run_time = datetime.now()
        
        try:
            account_ids = self._get_account_ids()
            stats = self._worker.run_once(
                outbox_store=self._store,
                account_ids=account_ids,
                worker_id=self._worker_id,
            )
            
            self._metrics.last_run_stats = stats
            self._metrics.total_processed += stats.get("processed", 0)
            self._metrics.total_succeeded += stats.get("succeeded", 0)
            self._metrics.total_failed += stats.get("failed", 0)
            self._metrics.total_dlq += stats.get("moved_to_dlq", 0)
            self._metrics.consecutive_errors = 0
            
            # Empty polls are expected while the outbox is drained.  Keep
            # those details available at DEBUG, but make actual work and
            # queue failures visible at the normal service log level.
            has_failure = bool(stats.get("failed", 0) or stats.get("moved_to_dlq", 0))
            has_activity = any(
                stats.get(key, 0)
                for key in ("processed", "succeeded", "skipped")
            )
            if has_failure:
                logger.warning("run_once completed with failures: %s, worker_id=%s", stats, self._worker_id)
            elif has_activity:
                logger.info("run_once completed: %s, worker_id=%s", stats, self._worker_id)
            else:
                logger.debug("run_once completed (empty poll): %s, worker_id=%s", stats, self._worker_id)
            return stats
            
        except Exception as e:
            self._metrics.consecutive_errors += 1
            logger.error(f"run_once failed: {e}, worker_id={self._worker_id}", exc_info=True)
            raise
    
    def _on_job_error(self, event: JobEvent) -> None:
        """Handle job execution errors."""
        if event.exception:
            logger.error(
                f"Job {event.job_id} failed: {event.exception}",
                exc_info=event.exception
            )
        elif event.code == EVENT_JOB_MISSED:
            logger.warning(f"Job {event.job_id} missed, worker_id={self._worker_id}")


def setup_signal_handlers(schedulers: list[OutboxScheduler]) -> None:
    """Setup signal handlers for graceful shutdown.
    
    Args:
        schedulers: List of schedulers to stop on signal
    """
    def signal_handler(signum: int, frame) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        for scheduler in schedulers:
            scheduler.stop(wait=True)
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
