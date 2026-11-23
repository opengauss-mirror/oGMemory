# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from __future__ import annotations

"""Access tracking for memory nodes."""

import time
from collections import defaultdict
from logging import getLogger

logger = getLogger(__name__)


class AccessTracker:
    """Tracks and aggregates memory access statistics."""

    def __init__(self, window_days: int = 30):
        """Initialize the AccessTracker.

        Args:
            window_days: Number of days to track hits for (default 30)
        """
        self._hit_buffer: dict[str, list[float]] = defaultdict(list)
        self._window_seconds = window_days * 24 * 3600

    def record_hits(self, uris: list[str]) -> None:
        """Record access hits for URIs.

        Args:
            uris: List of URIs that were accessed
        """
        now = time.time()
        for uri in uris:
            self._hit_buffer[uri].append(now)

    def flush_to_metadata(self, fs, ctx) -> int:
        """Flush buffered hits to node metadata.

        Args:
            fs: ContextFS instance
            ctx: RequestContext

        Returns:
            Number of nodes updated
        """
        if not self._hit_buffer:
            return 0

        updated = 0
        now = time.time()
        cutoff = now - self._window_seconds

        for uri in list(self._hit_buffer.keys()):
            hits = self._hit_buffer[uri]
            recent = [t for t in hits if t > cutoff]

            if not recent:
                del self._hit_buffer[uri]
                continue

            try:
                node = fs.read_node(uri, ctx)
                node.metadata["last_accessed_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(recent))
                )
                existing = node.metadata.get("hit_count_30d", 0)
                node.metadata["hit_count_30d"] = existing + len(recent)
                fs.write_node(node, ctx)
                updated += 1
                del self._hit_buffer[uri]
            except Exception as e:
                logger.warning(f"Failed to flush stats for {uri}: {e}")

        logger.info(f"Flushed access stats for {updated} nodes")
        return updated

    def get_stats(self, uri: str) -> dict:
        """Get buffered stats for a URI.

        Args:
            uri: URI to get stats for

        Returns:
            Dict with uri and buffered_hits count
        """
        hits = self._hit_buffer.get(uri, [])
        now = time.time()
        cutoff = now - self._window_seconds
        recent = [t for t in hits if t > cutoff]
        return {"uri": uri, "buffered_hits": len(recent)}
