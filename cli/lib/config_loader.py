# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unified config path resolution for oG-Memory templates.

Provides fallback logic: new config/ paths first, old paths for backward compat.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent  # ContextEngine/
_CONFIG_DIR = _ROOT / "config"


def get_project_root() -> Path:
    return _ROOT


def get_config_dir() -> Path:
    return _CONFIG_DIR


def get_reference_yaml() -> Path:
    """Return the unified reference YAML template."""
    p = _CONFIG_DIR / "ogmem.reference.yaml"
    if p.exists():
        return p
    raise FileNotFoundError("No reference YAML found (expected config/ogmem.reference.yaml)")


def get_deploy_env_template() -> Path:
    """Return the deploy.env reference template."""
    p = _CONFIG_DIR / "deploy.env.reference"
    if p.exists():
        return p
    raise FileNotFoundError("No deploy.env template found (expected config/deploy.env.reference)")


def get_agfs_config_template() -> Path:
    """Return the AGFS config reference template."""
    p = _CONFIG_DIR / "agfs" / "config.reference.yaml"
    if p.exists():
        return p
    raise FileNotFoundError("No AGFS config template found (expected config/agfs/config.reference.yaml)")


def get_openclaw_template() -> Path:
    """Return the OpenClaw template JSON."""
    p = _CONFIG_DIR / "openclaw" / "ogmem.template.json"
    if p.exists():
        return p
    raise FileNotFoundError("No OpenClaw template found (expected config/openclaw/ogmem.template.json)")


def load_template(path: Path) -> str:
    """Read a template file, returning its contents."""
    return path.read_text(encoding="utf-8")
