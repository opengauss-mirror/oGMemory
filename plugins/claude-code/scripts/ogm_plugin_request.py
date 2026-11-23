# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""HTTP base URL, JSON headers, and identity for oG-Memory Claude Code plugin scripts.

Uses stdlib and optional :class:`OgMemConfig` when the monorepo is on ``sys.path``;
otherwise environment variables (see :func:`_identity_and_url`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# When hooks run as ``plugins/claude-code/scripts/<script>.py``, four parents = repo root.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _try_ogmem_config():
    try:
        from providers.unified_config import OgMemConfig

        return OgMemConfig.load()
    except Exception:  # noqa: BLE001
        return None


def _default_api_url_from_cfg(cfg) -> str:
    explicit = (os.environ.get("OG_MEMORY_URL") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    return f"http://127.0.0.1:{getattr(cfg, 'http_port', 8090)}"


def _identity_and_url() -> tuple[str, str, str, str]:
    """Return ``(account_id, user_id, agent_id, api_url)`` for this process."""
    cfg = _try_ogmem_config()
    if cfg is not None:
        api = _default_api_url_from_cfg(cfg)
        return (
            os.environ.get("OG_MEMORY_ACCOUNT_ID") or os.environ.get("OG_ACCOUNT_ID") or str(cfg.account_id),
            os.environ.get("OG_MEMORY_USER_ID") or os.environ.get("OG_USER_ID") or str(cfg.user_id),
            os.environ.get("OG_MEMORY_AGENT_ID") or os.environ.get("OG_AGENT_ID") or str(cfg.agent_id),
            str(api).rstrip("/"),
        )
    return (
        os.environ.get("OG_MEMORY_ACCOUNT_ID") or os.environ.get("OG_ACCOUNT_ID", "acct-demo"),
        os.environ.get("OG_MEMORY_USER_ID") or os.environ.get("OG_USER_ID", "u-claude"),
        os.environ.get("OG_MEMORY_AGENT_ID") or os.environ.get("OG_AGENT_ID", "claude-code"),
        (os.environ.get("OG_MEMORY_URL") or "http://localhost:8090").rstrip("/"),
    )


def _plugin_api_key() -> str | None:
    """Same key semantics as OpenClaw plugin: ``OG_AUTH_API_KEY``"""
    v = (os.environ.get("OG_AUTH_API_KEY") or "").strip()
    if v:
        return v
    return None


def base_api_url() -> str:
    """Base URL for oG-Memory API (no trailing slash)."""
    *_, u = _identity_and_url()
    if u:
        return u.rstrip("/")
    return (os.environ.get("OG_MEMORY_URL") or "http://localhost:8090").rstrip("/")


def http_plugin_headers() -> dict[str, str]:
    """JSON headers. If ``_plugin_api_key()`` is set, add
    ``X-API-Key`` / ``X-Account-ID`` / ``X-User-ID`` (same as OpenClaw ``authApiKey`` path).
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    key = _plugin_api_key()
    if not key:
        return headers
    acc, uid, _, _ = _identity_and_url()
    headers["X-API-Key"] = key
    headers["X-Account-ID"] = acc
    headers["X-User-ID"] = uid
    return headers


def base_ctx(session_id: str) -> dict:
    """``accountId``, ``userId``, ``agentId``, ``sessionId`` for request bodies."""
    acc, uid, agid, _ = _identity_and_url()
    return {
        "accountId": acc,
        "userId": uid,
        "agentId": agid,
        "sessionId": session_id,
    }


_TRUE_VALUES = {"1", "true", "yes"}


def _env_bool(name: str, default: bool) -> bool:
    """Parse the plugin's boolean environment values conservatively."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUE_VALUES


def auto_capture_enabled() -> bool:
    """Allow automatic transcript and tool-message writes (enabled by default)."""
    return _env_bool("OGMEM_AUTO_CAPTURE", True)


def auto_recall_enabled() -> bool:
    """Allow automatic prompt recall, independently of capture."""
    return _env_bool("OGMEM_AUTO_RECALL", True)


def mcp_recovery_enabled() -> bool:
    """Return whether Claude has an explicitly configured MCP recovery path.

    The Hook can reach the oGMemory HTTP API without a Claude MCP connection,
    so the server-side ``mcp_enabled`` setting is not sufficient evidence here.
    The installation flow sets this opt-in only after ``ogm_read_memory`` is
    visible to the Claude client.
    """

    return _env_bool("OGMEM_CLAUDE_MCP_RECOVERY_ENABLED", False)
