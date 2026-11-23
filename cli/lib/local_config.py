# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Resolve the configuration used by local service lifecycle commands."""

from __future__ import annotations

import os
from pathlib import Path

SYSTEM_CONFIG_PATH = Path("/etc/ogmem/config.yaml")


def load_env(root: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    path = root / "config" / ".env"
    if not path.is_file():
        return {}
    return {key: value for key, value in dotenv_values(path).items() if value is not None}


def resolve_local_config_path(root: Path, explicit_path: str | None = None) -> Path:
    """Resolve the same target for onboarding and startup, relative to caller CWD."""
    path = explicit_path or os.environ.get("OGMEM_CONFIG") or load_env(root).get("OGMEM_CONFIG")
    if path:
        # Accept the usual quoted dotenv spelling without retaining the quotes.
        if len(path) >= 2 and path[0] == path[-1] and path[0] in {"'", '"'}:
            path = path[1:-1]
        return Path(path).expanduser().resolve()
    project_path = root / "config" / "ogmem.yaml"
    if not project_path.exists() and SYSTEM_CONFIG_PATH.is_file():
        return SYSTEM_CONFIG_PATH
    return project_path


def load_local_config(root: Path, explicit_path: str | None = None):
    path = resolve_local_config_path(root, explicit_path)
    selected = bool(explicit_path or os.environ.get("OGMEM_CONFIG") or load_env(root).get("OGMEM_CONFIG"))
    for key, value in load_env(root).items():
        os.environ.setdefault(key, value)
    if selected and not path.is_file():
        raise ValueError("The selected configuration file does not exist")
    # The child server must load exactly the file selected by its launcher.
    os.environ["OGMEM_CONFIG"] = str(path)
    from providers.unified_config import OgMemConfig

    config = OgMemConfig.load(str(path))
    if config.storage_backend not in {"agfs", "sql"}:
        raise ValueError("Storage backend must be agfs or sql")
    return config
