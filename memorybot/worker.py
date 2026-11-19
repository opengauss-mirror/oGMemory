"""Composition root and separately deployable task worker."""

from __future__ import annotations

import logging
import signal
import threading

from memorybot.config import BotConfig
from memorybot.models import BotError


def create_service(memory_service, config=None):
    from memorybot.files import AGFSFileStore
    from memorybot.service import BotService
    from memorybot.store import BotStore

    config = config or BotConfig.load(ogmem=memory_service._cfg)
    if not config.enabled:
        raise BotError("DEPENDENCY_UNAVAILABLE", "请配置 bot.enabled=true", 503)
    if not memory_service.get_auth_service().role_control_active():
        raise BotError("DEPENDENCY_UNAVAILABLE", "memoryBot 需要启用用户 API Key 鉴权", 503)
    store = BotStore(
        config.connection_string,
        pool_size=config.pool_size,
        dialect=memory_service._cfg.sql_dialect,
    )
    files = AGFSFileStore(
        config.files_base_url,
        config.files_mount_prefix,
        timeout=config.http_timeout_seconds,
        allowed_hosts=config.allowed_web_hosts,
    )
    service = BotService(store, memory_service, files, config)
    return service


def run_worker(config_path=None):
    from memorybot.runtime import BotWorker
    from providers.unified_config import get_config
    from server.memory_service import MemoryService

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = get_config(config_path)
    bot_config = BotConfig.load(config_path, config)
    memory = MemoryService(config=config)
    service = create_service(memory, bot_config)
    stop = threading.Event()

    def shutdown(signum, frame):
        stop.set()

    for name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), shutdown)
    worker = BotWorker(service.store, service.gateway, service.files, bot_config)
    try:
        worker.run_forever(stop)
    finally:
        service.store.close()
        memory.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the memoryBot task worker")
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    run_worker(args.config)
