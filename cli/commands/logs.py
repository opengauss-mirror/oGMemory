# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""ogmem logs command - show service logs."""

from __future__ import annotations

import subprocess
from pathlib import Path

from cli.lib.colors import fail, ok
from cli.lib.process import find_project_root


def run(args) -> int:
    """Show/tail service logs."""
    follow = getattr(args, "follow", False)
    svc = getattr(args, "service", None)
    root = find_project_root()
    logs_dir = root / ".ogmem_data" / "logs"

    # Try Docker first
    try:
        r = subprocess.run(["docker", "ps", "--filter", "name=ogmem", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=5)
        if r.stdout.strip():
            containers = [c for c in r.stdout.strip().split("\n") if c]
            if svc:
                containers = [c for c in containers if svc.lower() in c.lower()]
            if not containers:
                print(fail("No ogmem containers running"))
                return 1
            ok(f"Showing {len(containers)} container(s)\n")
            subprocess.run(["docker", "logs"] + (["-f"] if follow else []) + containers)
            return 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Local mode
    if not logs_dir.exists():
        print(fail(f"Logs dir not found: {logs_dir}"))
        return 1
    logs = [f for f in logs_dir.glob("*.log") if not svc or svc.lower() in f.name.lower()]
    if not logs:
        print(fail("No log files found"))
        return 1
    cmd = ["tail"] + (["-f"] if follow else []) + [str(f) for f in logs]
    ok(f"Showing {len(logs)} file(s)\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
    return 0
