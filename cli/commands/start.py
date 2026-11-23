# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""ogmem start command - launch services."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from cli.lib.colors import CYAN, NC, YELLOW, fail, ok
from cli.lib.health import check_agfs
from cli.lib.local_config import load_local_config
from cli.lib.process import ensure_data_dirs, find_project_root, spawn_process, start_and_wait, write_pid


def start_headless(args) -> int:
    root = find_project_root()
    from cli.lib.local_runtime import require_stopped
    if not getattr(args, "dry_run", False):
        try:
            require_stopped(root)
        except (RuntimeError, OSError, ValueError) as exc:
            print(fail(str(exc)))
            return 1
    try:
        cfg = load_local_config(root, getattr(args, "config", None))
    except Exception as exc:
        print(fail(f"Cannot load local configuration ({type(exc).__name__}); check --config / OGMEM_CONFIG"))
        return 1
    ensure_data_dirs(root)
    env, d = os.environ, root / ".ogmem_data"
    ce_port = cfg.http_port
    uses_agfs = cfg.storage_backend == "agfs"
    dry, daemon = getattr(args, "dry_run", False), getattr(args, "daemon", False)
    def start(name, cmd, pidf, logf, port, kind, cwd):
        if dry:
            print(f"Would start: {' '.join(cmd)}")
            return True
        try:
            pid = start_and_wait(cmd, pidf, logf, "127.0.0.1", port, kind, cwd=cwd)
            print(ok(f"{name} up (PID {pid})"))
            return True
        except RuntimeError as e:
            print(fail(str(e)))
            return False
    if uses_agfs:
        agfs_dir = Path(env.get("AGFS_SERVER_DIR", str(root / "agfs")))
        agfs_bin = Path(getattr(args, "agfs_bin", None) or env.get("AGFS_SERVER_BIN", str(agfs_dir / "build" / "agfs-server")))
        agfs_cfg = Path(env.get("AGFS_SERVER_CONFIG", str(agfs_dir / "config.yaml")))
        url = env.get("AGFS_BASE_URL", "http://127.0.0.1:1833")
        agfs_port = int(env.get("AGFS_PORT", url.rsplit(":", 1)[-1].rstrip("/")))
        if dry:
            print(f"Would start AGFS if needed: {agfs_bin} -c {agfs_cfg} -addr :{agfs_port}")
        elif check_agfs(port=agfs_port):
            print(ok(f"AGFS already running on :{agfs_port}"))
        elif not (agfs_bin.exists() and os.access(agfs_bin, os.X_OK)):
            print(fail(f"AGFS binary not found: {agfs_bin}"))
            print("  Set AGFS_SERVER_BIN to override, or install AGFS first")
            return 1
        elif not start("AGFS", [str(agfs_bin), "-c", str(agfs_cfg), "-addr", f":{agfs_port}"], d / "agfs.pid", d / "logs" / "agfs.log", agfs_port, "agfs", agfs_dir):
            return 1
    else:
        from cli.lib.db_setup import check_postgresql

        if not cfg.sql_connection_string:
            print(fail("SQL storage requires storage.connection_string or SQL_CONNECTION_STRING"))
            return 1
        if dry:
            print("Would check SQL database connectivity")
        elif not check_postgresql(cfg.sql_connection_string):
            print(fail("SQL database is not reachable; start the database and check the connection settings"))
            return 1
        else:
            print(ok("SQL database reachable"))
    if dry:
        print(f"Would start: {sys.executable} server/app.py")
    else:
        from cli.lib.local_runtime import ensure_service
        try:
            pid = ensure_service(root, cfg, timeout=getattr(args, "health_timeout", 30))
        except (RuntimeError, OSError, ValueError) as exc:
            print(fail(str(exc)))
            return 1
        print(ok(f"ContextEngine started (PID {pid})"))
    status = "PLANNED" if dry else "UP"
    if uses_agfs:
        print(f"\n  AGFS            http://127.0.0.1:{agfs_port}  {status}")
    print(f"\n  ContextEngine   http://127.0.0.1:{ce_port}  {status}\n")
    if not dry:
        agfs_log = f"{CYAN}.ogmem_data/logs/agfs.log{NC}  " if uses_agfs else ""
        print(f"  Logs: {agfs_log}{CYAN}.ogmem_data/logs/context_engine.log{NC}\n  Stop: {YELLOW}ogmem stop local{NC}")
    if daemon or dry:
        print(ok("Started in daemon mode" if daemon else "Dry run complete"))
        return 0
    print(f"\n  {YELLOW}Ctrl+C to stop{NC}")
    def shutdown(s, f):
        print("\nShutting down...")
        from cli.lib.process import stop_service
        if uses_agfs:
            stop_service(d / "agfs.pid", agfs_port)
        from cli.lib.local_runtime import stop_recorded_service
        stop_recorded_service(root)
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)
    return 0


def start_index(args) -> int:
    root = find_project_root()
    ensure_data_dirs(root)
    dry, daemon = getattr(args, "dry_run", False), getattr(args, "daemon", False)
    idx_script = root / "scripts" / "run_index_service.py"
    if not idx_script.exists():
        print(fail(f"Index script not found: {idx_script}"))
        return 1
    if dry:
        print(f"Would start: {sys.executable} {idx_script}")
        return 0
    log_file, pid_file = root / ".ogmem_data" / "logs" / "index.log", root / ".ogmem_data" / "index.pid"
    proc = spawn_process([sys.executable, str(idx_script)], log_file=log_file, cwd=root)
    write_pid(pid_file, proc.pid)
    print(ok(f"Index worker started (PID {proc.pid})"))
    print(f"  Log: {CYAN}{log_file}{NC}")
    if daemon:
        return 0
    print(f"\n  {YELLOW}Ctrl+C to stop{NC}")
    signal.signal(signal.SIGINT, lambda s, f: os.kill(proc.pid, signal.SIGTERM))
    signal.signal(signal.SIGTERM, lambda s, f: os.kill(proc.pid, signal.SIGTERM))
    proc.wait()
    return 0


def run(args) -> int:
    mode = getattr(args, "mode", None) or "headless"
    f = {"headless": start_headless, "local": start_headless, "index": start_index}.get(mode)
    if f:
        return f(args)
    print(fail(f"Unknown mode: {mode}"))
    return 1
