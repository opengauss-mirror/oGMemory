# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Example usage of IndexService for async index synchronization."""

import os
import signal

from pyagfs import AGFSClient

from commit.outbox_store import OutboxStore
from fs.agfs_adapter import AGFSContextFS
from providers.embedder import MockEmbedder, get_openai_embedder
from providers.vector_index import InMemoryVectorIndex
from service import (
    IndexService,
    setup_signal_handlers,
)


def get_active_account_ids_from_agfs(client: AGFSClient, mount_prefix: str = "") -> list[str]:
    """Get active account IDs by listing AGFS accounts directory.

    Args:
        client: AGFS client instance
        mount_prefix: Optional mount prefix path

    Returns:
        List of active account IDs
    """
    normalized_prefix = mount_prefix.rstrip("/")
    accounts_dir = f"{normalized_prefix}/accounts" if normalized_prefix else "/accounts"
    try:
        entries = client.ls(accounts_dir)
        return [
            entry["name"]
            for entry in entries
            if entry.get("is_dir", entry.get("isDir", False))
        ]
    except Exception as e:
        if "no such directory" in str(e).lower():
            return []
        print(f"Failed to list accounts: {e}")
        return []


def create_index_service_with_mock(
    agfs_client: AGFSClient,
    fs: AGFSContextFS,
    mount_prefix: str = "/local",
    worker_count: int = 3,
    interval_seconds: int = 30,
) -> IndexService:
    """Create IndexService with mock embedder for testing.

    Args:
        agfs_client: AGFS client instance
        fs: Context filesystem adapter used by OutboxStore
        mount_prefix: AGFS mount prefix used by ContextFS and OutboxStore
        worker_count: Number of worker threads
        interval_seconds: Polling interval

    Returns:
        Configured IndexService instance
    """
    outbox_store = OutboxStore(agfs_client, fs=fs, mount_prefix=mount_prefix)
    embedder = MockEmbedder()
    vector_index = InMemoryVectorIndex()

    def get_account_ids() -> list[str]:
        return get_active_account_ids_from_agfs(agfs_client, mount_prefix=mount_prefix)

    return IndexService(
        outbox_store=outbox_store,
        embedder=embedder,
        vector_index=vector_index,
        get_account_ids=get_account_ids,
        interval_seconds=interval_seconds,
        worker_count=worker_count,
    )


def create_index_service_with_openai(
    agfs_client: AGFSClient,
    fs: AGFSContextFS,
    api_key: str | None = None,
    base_url: str | None = None,
    mount_prefix: str = "/local",
    worker_count: int = 3,
    interval_seconds: int = 30,
) -> IndexService:
    """Create IndexService with OpenAI embedder for production.

    Args:
        agfs_client: AGFS client instance
        fs: Context filesystem adapter used by OutboxStore
        api_key: OpenAI API key (uses env var if None)
        base_url: Custom base URL (optional)
        mount_prefix: AGFS mount prefix used by ContextFS and OutboxStore
        worker_count: Number of worker threads
        interval_seconds: Polling interval

    Returns:
        Configured IndexService instance
    """
    openai_embedder_cls, _ = get_openai_embedder()

    outbox_store = OutboxStore(agfs_client, fs=fs, mount_prefix=mount_prefix)
    embedder = openai_embedder_cls(
        api_key=api_key or os.environ.get("OGMEM_API_KEY"),
        base_url=base_url,
    )
    vector_index = InMemoryVectorIndex()

    def get_account_ids() -> list[str]:
        return get_active_account_ids_from_agfs(agfs_client, mount_prefix=mount_prefix)

    return IndexService(
        outbox_store=outbox_store,
        embedder=embedder,
        vector_index=vector_index,
        get_account_ids=get_account_ids,
        interval_seconds=interval_seconds,
        worker_count=worker_count,
    )


def main():
    """Main entry point for running the index service as a standalone process."""
    import argparse

    parser = argparse.ArgumentParser(description="Run async index sync service")
    parser.add_argument("--workers", type=int, default=3, help="Number of worker threads")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--mock", action="store_true", help="Use mock embedder for testing")
    parser.add_argument(
        "--agfs-url",
        type=str,
        default=os.environ.get("AGFS_BASE_URL", "http://localhost:8080"),
        help="AGFS API base URL",
    )
    parser.add_argument(
        "--agfs-mount",
        type=str,
        default=os.environ.get("AGFS_MOUNT_PREFIX", "/local"),
        help="AGFS mount prefix used by ContextFS and OutboxStore",
    )
    args = parser.parse_args()

    agfs_client = AGFSClient(api_base_url=args.agfs_url)
    fs = AGFSContextFS(client=agfs_client, mount_prefix=args.agfs_mount)

    if args.mock:
        service = create_index_service_with_mock(
            agfs_client=agfs_client,
            fs=fs,
            mount_prefix=args.agfs_mount,
            worker_count=args.workers,
            interval_seconds=args.interval,
        )
    else:
        service = create_index_service_with_openai(
            agfs_client=agfs_client,
            fs=fs,
            mount_prefix=args.agfs_mount,
            worker_count=args.workers,
            interval_seconds=args.interval,
        )

    setup_signal_handlers(service)

    service.start()
    print(
        f"IndexService started with {args.workers} workers, "
        f"interval={args.interval}s, agfs_url={args.agfs_url}, mount={args.agfs_mount}"
    )
    print("Press Ctrl+C to stop")

    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        print("\nShutting down...")
        service.stop(wait=True)
        print("IndexService stopped")


if __name__ == "__main__":
    main()
