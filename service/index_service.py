"""Index service for async index synchronization.

Provides the IndexService class that manages OutboxScheduler lifecycle
and coordinates async index updates for the ContextEngine.
"""

import logging
import signal
import sys
import threading
import uuid
from typing import Callable, Optional

from commit.outbox_store import OutboxStore
from index.outbox_worker import OutboxWorker
from index.scheduler import OutboxScheduler, SchedulerMetrics

logger = logging.getLogger(__name__)


class IndexService:
    """Service for managing async index synchronization.

    This service wraps OutboxScheduler and provides a simple interface
    for starting/stopping the index sync process.

    Usage:
        # Initialize
        index_service = IndexService(
            outbox_store=store,
            embedder=embedder,
            vector_index=vector_index,
            get_account_ids=get_active_account_ids,
            fs=agfs_context_fs,  # Required for UPSERT_DIRECTORY events
            llm=llm,             # Required for UPSERT_DIRECTORY events
        )

        # Start with multiple workers
        index_service.start(worker_count=3)

        # Later...
        index_service.stop()

    Or use the singleton pattern:
        from service.index_service import init_index_service, get_index_service

        init_index_service(outbox_store, embedder, vector_index, get_account_ids, fs=fs, llm=llm)
        get_index_service().start()
    """
    
    def __init__(
        self,
        outbox_store: OutboxStore,
        embedder,
        vector_index,
        get_account_ids: Callable[[], list[str]],
        interval_seconds: int = 30,
        worker_count: int = 3,
        fs=None,
        llm=None,
        directory_summary_enabled: bool = False,
        admission=None,
    ):
        """Initialize the index service.

        Args:
            outbox_store: OutboxStore for event persistence
            embedder: Embedder for vector generation
            vector_index: VectorIndex for storing embeddings
            get_account_ids: Callable returning list of active account IDs
            interval_seconds: Polling interval in seconds
            worker_count: Number of worker threads
            fs: ContextFS for UPSERT_DIRECTORY events (optional)
            llm: LLM for directory summary generation (optional)
            directory_summary_enabled: Whether to enable directory summary generation
        """
        self._outbox_store = outbox_store
        self._embedder = embedder
        self._vector_index = vector_index
        self._get_account_ids = get_account_ids
        self._interval_seconds = interval_seconds
        self._worker_count = worker_count
        self._fs = fs
        self._llm = llm
        self._directory_summary_enabled = directory_summary_enabled
        self._admission = admission
        
        self._worker: Optional[OutboxWorker] = None
        self._schedulers: list[OutboxScheduler] = []
        self._threads: list[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()
    
    @property
    def is_running(self) -> bool:
        """Check if the service is running."""
        return self._running
    
    @property
    def schedulers(self) -> list[OutboxScheduler]:
        """Get the list of schedulers."""
        return self._schedulers
    
    def start(self, worker_count: Optional[int] = None) -> None:
        """Start the index service with multiple worker threads.

        Args:
            worker_count: Number of workers (uses constructor value if None)
        """
        with self._lock:
            if self._running:
                logger.warning("IndexService already running")
                return

            worker_count = worker_count or self._worker_count

            self._worker = OutboxWorker(
                vector_index=self._vector_index,
                embedder=self._embedder,
                fs=self._fs,
                llm=self._llm,
                directory_summary_enabled=self._directory_summary_enabled,
            )
            
            if self._admission is not None:
                self._worker.run_once = self._admission.wrap(self._worker.run_once, write=True)

            # A partial startup must remain stoppable if a later thread fails.
            self._running = True
            for i in range(worker_count):
                worker_id = f"worker-{uuid.uuid4().hex[:8]}"
                
                scheduler = OutboxScheduler(
                    worker=self._worker,
                    outbox_store=self._outbox_store,
                    get_account_ids=self._get_account_ids,
                    worker_id=worker_id,
                    interval_seconds=self._interval_seconds,
                )
                
                def make_scheduler_starter(scheduler: OutboxScheduler):
                    def starter():
                        scheduler.start()
                        self._schedulers.append(scheduler)
                    return starter

                t = threading.Thread(
                    target=make_scheduler_starter(scheduler),
                    name=f"index-worker-{i+1}",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
            
            self._running = True
            logger.info(f"IndexService started with {worker_count} workers")
    
    def stop(self, wait: bool = True) -> None:
        """Stop the index service.
        
        Args:
            wait: Whether to wait for current jobs to complete
        """
        with self._lock:
            if not self._running:
                return
            
            # Join startup threads before enumerating schedulers: otherwise a
            # scheduler may publish itself after maintenance has stopped them.
            for thread in self._threads:
                thread.join()
            for scheduler in self._schedulers:
                scheduler.stop(wait=wait)
            
            self._schedulers.clear()
            self._threads.clear()
            self._running = False
            logger.info("IndexService stopped")
    
    def get_metrics(self) -> list[SchedulerMetrics]:
        """Get metrics from all schedulers.
        
        Returns:
            List of SchedulerMetrics from each worker
        """
        return [s.metrics for s in self._schedulers]
    
    def get_aggregated_stats(self) -> dict:
        """Get aggregated statistics from all workers.
        
        Returns:
            Dict with aggregated stats:
            {
                "worker_count": int,
                "total_processed": int,
                "total_succeeded": int,
                "total_failed": int,
                "total_dlq": int,
                "is_running": bool,
            }
        """
        metrics = self.get_metrics()
        return {
            "worker_count": len(metrics),
            "total_processed": sum(m.total_processed for m in metrics),
            "total_succeeded": sum(m.total_succeeded for m in metrics),
            "total_failed": sum(m.total_failed for m in metrics),
            "total_dlq": sum(m.total_dlq for m in metrics),
            "is_running": self._running,
        }


def setup_signal_handlers(service: IndexService) -> None:
    """Setup signal handlers for graceful shutdown.
    
    Args:
        service: IndexService to stop on signal
    """
    def signal_handler(signum: int, frame) -> None:
        logger.info(f"Received signal {signum}, shutting down index service...")
        service.stop(wait=True)
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


# Singleton instance
_default_index_service: Optional[IndexService] = None


def init_index_service(
    outbox_store: OutboxStore,
    embedder,
    vector_index,
    get_account_ids: Callable[[], list[str]],
    interval_seconds: int = 30,
    worker_count: int = 3,
    fs=None,
    llm=None,
    directory_summary_enabled: bool = False,
) -> IndexService:
    """Initialize the global index service instance.

    Args:
        outbox_store: OutboxStore for event persistence
        embedder: Embedder for vector generation
        vector_index: VectorIndex for storing embeddings
        get_account_ids: Callable returning list of active account IDs
        interval_seconds: Polling interval in seconds
        worker_count: Number of worker threads
        fs: ContextFS for UPSERT_DIRECTORY events (optional)
        llm: LLM for directory summary generation (optional)
        directory_summary_enabled: Whether to enable directory summary generation
    
    Returns:
        Configured IndexService instance
    """
    global _default_index_service
    _default_index_service = IndexService(
        outbox_store=outbox_store,
        embedder=embedder,
        vector_index=vector_index,
        get_account_ids=get_account_ids,
        interval_seconds=interval_seconds,
        worker_count=worker_count,
        fs=fs,
        llm=llm,
        directory_summary_enabled=directory_summary_enabled,
    )
    return _default_index_service


def get_index_service() -> Optional[IndexService]:
    """Get the global index service instance.
    
    Returns:
        IndexService if initialized, None otherwise
    """
    return _default_index_service
