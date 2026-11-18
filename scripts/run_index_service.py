#!/usr/bin/env python3
"""Standalone IndexService runner.

Starts the OutboxScheduler as a long-running process that periodically
scans the configured SQL or AGFS outbox events, generates embeddings, and upserts into openGauss.

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


def _create_storage(cfg):
    """Use the same storage backend as the API, not an unrelated AGFS outbox."""
    if cfg.storage_backend == "sql":
        from commit.sql_outbox_store import SQLOutboxStore
        from fs.sql_adapter import SQLContextFS
        from fs.sql_adapter.pool import SharedConnectionPool

        if not cfg.sql_connection_string:
            raise ValueError("SQL index service requires storage.connection_string")
        pool = SharedConnectionPool(
            cfg.sql_connection_string, pool_size=cfg.sql_pool_size,
            dialect=cfg.sql_dialect,
        )
        fs = SQLContextFS(pool=pool)
        return fs, SQLOutboxStore(pool=pool, fs=fs), pool

    from fs.agfs_adapter.client import create_agfs_client
    from commit.outbox_store import OutboxStore
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS

    client = create_agfs_client(cfg)
    fs = AGFSContextFS(client=client, mount_prefix=cfg.agfs_mount_prefix)
    return fs, OutboxStore(client=client, fs=fs, mount_prefix=cfg.agfs_mount_prefix), None


def main():
    from providers.config import ProviderConfig
    from providers.unified_config import get_config
    from service.index_service import init_index_service

    cfg = get_config()
    if cfg.retrieval_hybrid_enabled:
        raise RuntimeError(
            "hybrid retrieval is unsupported by the standalone index service; "
            "run the SQL-backed MemoryService so every outbox event dual-writes"
        )
    provider_cfg = ProviderConfig.from_ogmem_config(cfg)

    embedder = provider_cfg.create_embedder()
    vector_index = provider_cfg.create_vector_index()

    fs, outbox_store, storage_pool = _create_storage(cfg)

    logger.info("Config: provider=%s, embedding=%s, db=%s, storage=%s",
                cfg.provider, cfg.openai_embedding_model,
                cfg.vector_db_type, cfg.storage_backend)
    logger.info("Params: account=%s, interval=%ds, workers=%d",
                cfg.account_id, cfg.index_interval, cfg.index_workers)

    llm = provider_cfg.create_llm()

    usage_pool = storage_pool
    if cfg.sql_connection_string:
        from fs.sql_adapter.model_usage_store import SQLModelUsageStore
        from fs.sql_adapter.pool import SharedConnectionPool

        if usage_pool is None:
            usage_pool = SharedConnectionPool(
                cfg.sql_connection_string, pool_size=cfg.sql_pool_size, dialect=cfg.sql_dialect,
            )
        usage_store = SQLModelUsageStore(usage_pool)
        for provider in (llm, embedder):
            tracker = getattr(provider, 'token_tracker', None)
            if tracker is not None:
                tracker.set_usage_store(usage_store)

    service = init_index_service(
        outbox_store=outbox_store,
        embedder=embedder,
        vector_index=vector_index,
        get_account_ids=lambda: [cfg.account_id],
        interval_seconds=cfg.index_interval,
        worker_count=cfg.index_workers,
        fs=fs,
        llm=llm,
        directory_summary_enabled=getattr(cfg, 'directory_summary_enabled', False),
    )

    def shutdown(signum, frame):
        logger.info("Received signal %d, stopping...", signum)
        service.stop(wait=True)
        if usage_pool is not None:
            usage_pool.close_all()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    service.start()
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
