#!/usr/bin/env python3
"""Standalone IndexService runner.

Starts the OutboxScheduler as a long-running process that periodically
scans the configured SQL or AGFS outbox events and processes their projections.

Usage:
    python scripts/run_index_service.py                    # uses config.yaml or env vars
    OGMEM_CONFIG=/path/to/config.yaml python scripts/run_index_service.py
"""

import logging
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stderr,
)
from core.logging_config import configure_third_party_loggers

# APScheduler emits a start/finish pair for every interval job at INFO.  The
# index scheduler reports the useful outcome itself, so keep the framework's
# normal execution chatter out of the service log while preserving warnings
# and errors (missed/failed jobs).
logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)
configure_third_party_loggers()
logger = logging.getLogger("index-service")


def _create_storage(cfg, *, invalidate_lexical_scopes=None, remember=None, guard=None):
    """Use the same storage backend as the API, not an unrelated AGFS outbox."""
    if cfg.storage_backend == "sql" or cfg.shared_spaces_enabled:
        from commit.sql_outbox_store import SQLOutboxStore
        from fs.sql_adapter import SQLContextFS
        from fs.sql_adapter.pool import SharedConnectionPool

        if not cfg.sql_connection_string:
            raise ValueError("SQL index service requires storage.connection_string")
        pool = SharedConnectionPool(
            cfg.sql_connection_string, pool_size=cfg.sql_pool_size,
            dialect=cfg.sql_dialect,
        )
        client = fs = None
        try:
            if cfg.storage_backend == "agfs":
                from fs.agfs_adapter.client import create_agfs_client
                from fs.agfs_adapter.space_bridge import AGFSSpaceBridge

                client = create_agfs_client(cfg)
                fs = AGFSSpaceBridge(pool, client, cfg.agfs_mount_prefix)
            else:
                fs = SQLContextFS(pool=pool)
            outbox = SQLOutboxStore(pool=pool, fs=fs)
        except BaseException:
            try:
                close = getattr(fs, 'close', None)
                if callable(close) and close() is False:
                    raise RuntimeError('shared content reader has not stopped')
            finally:
                try:
                    close = getattr(client, 'close', None)
                    if callable(close):
                        close()
                finally:
                    pool.close_all()
            raise
        if remember is not None:
            remember(pool)
            if client is not None:
                remember(client)
                remember(fs)
        if guard is not None and client is not None:
            guard(client)
        return fs, outbox, pool

    from commit.outbox_store import OutboxStore
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
    from fs.agfs_adapter.client import create_agfs_client

    client = create_agfs_client(cfg)
    try:
        fs = AGFSContextFS(client=client, mount_prefix=cfg.agfs_mount_prefix)
        outbox = OutboxStore(client=client, fs=fs, mount_prefix=cfg.agfs_mount_prefix,
                             invalidate_lexical_scopes=invalidate_lexical_scopes)
    except BaseException:
        close = getattr(client, 'close', None)
        if callable(close):
            close()
        raise
    if remember is not None:
        remember(client)
    if guard is not None:
        guard(client)
    return fs, outbox, None


def _build_service(cfg, admission):
    """Construct only index resources, while holding the process write permit."""
    from providers.config import ProviderConfig
    from service.index_service import IndexService

    closers = []
    def remember(obj, *, migration=False):
        close = (getattr(obj, 'release_for_migration', None) if migration else None)
        close = close or getattr(obj, 'close_all', None) or getattr(obj, 'close', None)
        if callable(close):
            closers.append(close)
        return obj
    def cleanup():
        # Preserve a failed close for explicit lifecycle retry.
        while closers:
            if closers[-1]() is False:
                raise RuntimeError("index resource has not finished closing")
            closers.pop()
    try:
        provider_cfg = ProviderConfig.from_ogmem_config(cfg)
        embedder = vector_index = None
        vector_cleanup = len(closers)
        if not cfg.agfs_retrieval_enabled:
            embedder = remember(provider_cfg.create_embedder())
            vector_cleanup = len(closers)
            vector_index = admission.guard(remember(provider_cfg.create_vector_index(), migration=True))
        from providers.composite_index import create_projection_writer
        projection = create_projection_writer(cfg, vector_index)
        if projection is not vector_index:
            if callable(getattr(projection, "release_for_migration", None)):
                del closers[vector_cleanup:]  # The composite owns the vector client.
            remember(projection, migration=True)
            admission.guard(projection.lexical_index)
            admission.guard(projection)
        lexical = getattr(projection, 'lexical_index', None)
        fs, outbox_store, storage_pool = _create_storage(
            cfg, remember=remember, guard=admission.guard,
            invalidate_lexical_scopes=lexical.mark_scopes_dirty if lexical is not None else None)
        fs = admission.guard(fs)
        outbox_store = admission.guard(outbox_store)
        llm = remember(provider_cfg.create_llm())
        usage_pool = storage_pool
        if cfg.sql_connection_string:
            from fs.sql_adapter.model_usage_store import SQLModelUsageStore
            from fs.sql_adapter.pool import SharedConnectionPool
            if usage_pool is None:
                usage_pool = remember(SharedConnectionPool(
                    cfg.sql_connection_string, pool_size=cfg.sql_pool_size, dialect=cfg.sql_dialect))
            usage_store = SQLModelUsageStore(usage_pool)
            for provider in (llm, embedder):
                tracker = getattr(provider, 'token_tracker', None)
                if tracker is not None:
                    tracker.set_usage_store(usage_store)
        service = IndexService(
            outbox_store=outbox_store, embedder=embedder, vector_index=projection,
            get_account_ids=lambda: [cfg.account_id], interval_seconds=cfg.index_interval,
            worker_count=cfg.index_workers, fs=fs, llm=llm,
            directory_summary_enabled=getattr(cfg, 'directory_summary_enabled', False),
            admission=admission)
        return service, cleanup
    except BaseException:
        cleanup()
        raise


class CoordinatedIndexRunner:
    """Register the official standalone producer in the instance migration gate."""
    def __init__(self, cfg, *, build_service=_build_service):
        from pathlib import Path

        from service.migration.maintenance import MaintenanceStore
        from service.migration.participants import Participant

        self.cfg, self.build_service = cfg, build_service
        self.service = self.cleanup = None
        workspace = Path(cfg.migration_workspace).expanduser().resolve()
        self.store = MaintenanceStore(workspace / 'maintenance.sqlite', str(workspace))
        self.participant = Participant(self.store, pause=self.pause, refresh=self.refresh, resume=self.resume)
        self.admission = self.participant.admission
        try:
            self.participant.start_service()
        except BaseException:
            self.participant.close()
            raise

    def resume(self):
        if self.service is not None:
            return
        with self.admission.backend(write=True):
            self.service, self.cleanup = self.build_service(self.cfg, self.admission)
            try:
                self.service.start()
            except BaseException:
                self.pause()
                raise

    def pause(self):
        if self.service is not None:
            self.service.stop(wait=True)
        if self.cleanup is not None:
            self.cleanup()
        self.service = self.cleanup = None
        return True

    def refresh(self):
        # pause already discarded all process-local index/storage clients.
        if self.service is not None:
            raise RuntimeError('index service must stop before refresh')

    def get_aggregated_stats(self):
        service = self.service
        return service.get_aggregated_stats() if service is not None else {}

    def close(self, timeout=30):
        with self.participant.shutdown_scope(timeout):
            self.pause()
        self.participant.close()


def main():
    from providers.unified_config import get_config
    cfg = get_config()
    service = CoordinatedIndexRunner(cfg)
    logger.info("Params: account=%s, interval=%ds, workers=%d",
                cfg.account_id, cfg.index_interval, cfg.index_workers)

    def shutdown(signum, frame):
        logger.info("Received signal %d, stopping...", signum)
        service.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    logger.info("IndexService running (Ctrl+C to stop)")

    last_stats = None
    while True:
        time.sleep(60)
        stats = service.get_aggregated_stats()
        if last_stats is not None and stats != last_stats:
            # Report only meaningful changes at INFO (new work, failures, or
            # worker lifecycle changes).  The full idle heartbeat remains
            # available at DEBUG for troubleshooting.
            logger.info("IndexService stats changed: %s", stats)
        else:
            logger.debug("IndexService heartbeat: %s", stats)
        last_stats = stats


if __name__ == "__main__":
    main()
