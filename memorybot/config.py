"""MemoryBot deployment settings, independent of memory storage selection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class BotConfig:
    enabled: bool = False
    connection_string: str = ""
    pool_size: int = 10
    files_backend: str = "agfs"
    files_base_url: str = "http://127.0.0.1:1833"
    files_mount_prefix: str = "/memorybot"
    lease_seconds: int = 30
    renew_seconds: int = 10
    poll_seconds: float = 1.0
    worker_concurrency: int = 4
    cluster_concurrency: int = 16
    tenant_concurrency: int = 8
    max_pending: int = 10000
    max_user_pending: int = 200
    max_model_calls: int = 8
    handbook_max_model_calls_per_file: int = 8
    max_tool_calls: int = 12
    max_input_tokens: int = 60000
    max_output_tokens: int = 8000
    task_timeout_seconds: int = 120
    file_timeout_seconds: int = 900
    max_file_bytes: int = 50 * 1024 * 1024
    event_retention_days: int = 7
    retention_days: int = 90
    http_timeout_seconds: int = 30
    allowed_web_hosts: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, config_path: str | None = None, ogmem=None) -> BotConfig:
        from providers.unified_config import DEFAULT_CONFIG_PATH, get_config

        ogmem = ogmem or get_config(config_path)
        path = Path(config_path or os.environ.get("OGMEM_CONFIG") or DEFAULT_CONFIG_PATH)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        settings = (raw or {}).get("bot") or {}
        values = {f.name: settings[f.name] for f in fields(cls) if f.name in settings}
        values["connection_string"] = getattr(ogmem, "sql_connection_string", "") or ""
        files_config = settings.get("files") or {}
        for name in ("backend", "base_url", "mount_prefix"):
            if name in files_config:
                values["files_" + name] = files_config[name]
        if "OGMEM_BOT_ENABLED" in os.environ:
            values["enabled"] = os.environ["OGMEM_BOT_ENABLED"].lower() in {"true", "1", "yes"}
        config = cls(**values)
        config.validate()
        if config.enabled and getattr(ogmem, "storage_backend", None) != "sql":
            raise ValueError("memoryBot requires storage.backend=sql")
        return config

    def validate(self) -> None:
        if self.enabled and not self.connection_string:
            raise ValueError("memoryBot requires storage.connection_string")
        if self.files_backend != "agfs":
            raise ValueError("bot.files.backend must be agfs")
        if not self.files_mount_prefix.startswith("/") or ".." in self.files_mount_prefix.split(
            "/"
        ):
            raise ValueError("bot.files.mount_prefix must be an absolute mount path")
        for name in (
            "lease_seconds",
            "renew_seconds",
            "poll_seconds",
            "worker_concurrency",
            "cluster_concurrency",
            "tenant_concurrency",
            "pool_size",
            "max_pending",
            "max_user_pending",
            "max_model_calls",
            "handbook_max_model_calls_per_file",
            "max_tool_calls",
            "max_input_tokens",
            "max_output_tokens",
            "task_timeout_seconds",
            "file_timeout_seconds",
            "max_file_bytes",
            "event_retention_days",
            "retention_days",
            "http_timeout_seconds",
        ):
            if not isinstance(getattr(self, name), (int, float)) or getattr(self, name) <= 0:
                raise ValueError(f"bot.{name} must be positive")
        if self.renew_seconds >= self.lease_seconds:
            raise ValueError("bot.renew_seconds must be less than lease_seconds")
        if self.max_file_bytes > 50 * 1024 * 1024:
            raise ValueError("bot.max_file_bytes must be at most 50 MiB")
