# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Environment diagnostics for oG-Memory."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys

from cli.lib.colors import fail, info, ok, warn
from cli.lib.health import check_agfs


def run(args) -> int:
    """Run environment diagnostics. Returns 0 if all required checks pass."""
    passed = failed = warnings = 0

    def check(condition, msg_pass, msg_fail, is_warn=False):
        nonlocal passed, failed, warnings
        if condition:
            print(ok(msg_pass))
            passed += 1
        elif is_warn:
            print(warn(msg_fail))
            warnings += 1
        else:
            print(fail(msg_fail))
            failed += 1

    def get_version(cmd, flag="--version"):
        try:
            r = subprocess.run([cmd, flag], capture_output=True, text=True, timeout=2)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    print("\n" + "=" * 80 + "\n                    oG-Memory Environment Check\n" + "=" * 80)

    # Section 1: Basic dependencies
    print("\n" + "-" * 40 + "\n  Section 1: Basic Dependencies\n" + "-" * 40)
    check(sys.version_info >= (3, 11), f"Python {sys.version_info.major}.{sys.version_info.minor}",
          f"Python {sys.version_info.major}.{sys.version_info.minor} (requires >= 3.11)")

    for cmd, name, opt in [("go", "Go", True), ("docker", "Docker", True),
                            ("node", "Node.js", True), ("openclaw", "OpenClaw", True)]:
        v = get_version(cmd) if shutil.which(cmd) else None
        check(v, f"{name}: {v}", f"{name} not installed", is_warn=opt)

    # Onboard knows its selected backend; SQL local and Docker do not need host AGFS.
    mode = getattr(args, "mode", None)
    backend = getattr(args, "storage_backend", None) or "sql"
    needs_host_agfs = mode not in {"docker", "local", "headless"} or (
        mode in {"local", "headless"} and backend == "agfs"
    )
    if needs_host_agfs:
        # Section 2: AGFS service
        print("\n" + "-" * 40 + "\n  Section 2: AGFS Service\n" + "-" * 40)
        check(shutil.which("agfs-server"), "agfs-server found", "agfs-server not found")
        agfs_cfg = os.path.expanduser("~/.agfs-config.yaml")
        check(os.path.exists(agfs_cfg), "~/.agfs-config.yaml exists", "~/.agfs-config.yaml not found")

        health = check_agfs()
        check(health, f"AGFS running (port 1833)", "AGFS not responding on port 1833")

        agfs_data = os.environ.get("AGFS_DATA_DIR", "/tmp/agfs-data")
        check(os.path.isdir(agfs_data), f"AGFS data directory: {agfs_data}",
              f"AGFS data directory not found: {agfs_data}")

    else:
        print(info("Host AGFS checks skipped for the selected deployment/storage mode"))

    # Section 3: Python packages
    print("\n" + "-" * 40 + "\n  Section 3: Python Packages\n" + "-" * 40)
    for mod, pip in [("pyagfs", "pyagfs"), ("openai", "openai"),
                     ("apscheduler", "apscheduler"), ("jinja2", "jinja2"),
                     ("yaml", "pyyaml")]:
        try:
            importlib.import_module(mod)
            print(ok(f"{mod} installed"))
            passed += 1
        except ImportError:
            print(fail(f"{mod} not installed (pip install {pip})"))
            failed += 1

    # Section 4: OpenClaw plugin
    print("\n" + "-" * 40 + "\n  Section 4: OpenClaw Plugin\n" + "-" * 40)
    openclaw_dir = os.path.expanduser("~/.openclaw")
    if os.path.exists(openclaw_dir):
        print(ok("~/.openclaw exists"))
        passed += 1
        cfg = os.path.join(openclaw_dir, "openclaw.json")
        if os.path.exists(cfg):
            try:
                import json
                with open(cfg) as f:
                    has_plugin = "og-memory-context-engine" in json.load(f).get("plugins", {}).get("entries", {})
                    check(has_plugin, "og-memory-context-engine registered",
                          "og-memory-context-engine not registered", is_warn=True)
            except Exception:
                print(warn("Could not parse openclaw.json"))
                warnings += 1
        else:
            print(warn("openclaw.json not found"))
            warnings += 1
    else:
        print(warn("~/.openclaw not found"))
        warnings += 1

    # Section 5: Environment variables
    print("\n" + "-" * 40 + "\n  Section 5: Environment Variables\n" + "-" * 40)
    api_key = os.environ.get("OGMEM_API_KEY") or os.environ.get("LLM_API_KEY")
    check(api_key, "OGMEM_API_KEY or LLM_API_KEY set",
          "OGMEM_API_KEY and LLM_API_KEY not set (mock mode)", is_warn=True)

    print(info(f"AGFS_BASE_URL: {os.environ.get('AGFS_BASE_URL', 'http://localhost:1833')}"))
    vdb = os.environ.get("VECTOR_DB_TYPE")
    check(vdb, f"VECTOR_DB_TYPE: {vdb}", "VECTOR_DB_TYPE not set (will use default)", is_warn=True)

    # Summary
    print("\n" + "-" * 40 + "\n  Summary\n" + "-" * 40 + f"\n  {passed} passed, {failed} failed, {warnings} warnings\n")
    return 0 if failed == 0 else 1
