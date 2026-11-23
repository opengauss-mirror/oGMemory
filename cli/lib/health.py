# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""HTTP health check helpers using stdlib http.client."""

from __future__ import annotations

import http.client
import json
import logging
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 3.0


def port_alive(host: str, port: int, timeout: float = DEFAULT_TIMEOUT) -> bool:
    """Return True if something is listening on host:port."""
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/")
        conn.close()
        return True
    except (ConnectionRefusedError, OSError, TimeoutError):
        return False


def _http_get_json(
    host: str, port: int, path: str, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, Any] | None:
    """GET a JSON endpoint, return parsed dict or None on failure."""
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", path)
        resp = conn.getresponse()
        if resp.status < 400:
            body = resp.read().decode()
            conn.close()
            return json.loads(body)
        conn.close()
    except (ConnectionRefusedError, OSError, TimeoutError, json.JSONDecodeError):
        pass
    return None


def check_agfs(host: str = "127.0.0.1", port: int = 1833) -> dict[str, Any] | None:
    """Check AGFS health at /api/v1/health. Returns health dict or None."""
    return _http_get_json(host, port, "/api/v1/health")


def check_ogmem(host: str = "127.0.0.1", port: int = 8090) -> dict[str, Any] | None:
    """Check oG-Memory health at /api/v1/health. Returns health dict or None."""
    return _http_get_json(host, port, "/api/v1/health")


def wait_for_healthy(
    host: str,
    port: int,
    kind: str,
    timeout: int = 30,
    interval: float = 1.0,
) -> bool:
    """Poll until health endpoint responds. Returns True if healthy within timeout."""
    check_fn = check_agfs if kind == "agfs" else check_ogmem
    elapsed = 0.0
    while elapsed < timeout:
        result = check_fn(host, port)
        if result is not None:
            return True
        import time
        time.sleep(interval)
        elapsed += interval
    return False
