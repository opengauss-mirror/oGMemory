# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Process management: PID files, spawn, terminate, port-based kill."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Walk up from CWD to find project root (has pyproject.toml or server/)."""
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / "pyproject.toml").exists() or (p / "server").is_dir():
            return p
    return cwd


def ensure_data_dirs(root: Path) -> None:
    """Create .ogmem_data subdirectories if they don't exist."""
    data = root / ".ogmem_data"
    for subdir in ("", "agfs", "chroma", "control", "logs"):
        d = data / subdir
        d.mkdir(parents=True, exist_ok=True)


# -- PID file management ---------------------------------------------------

def read_pid(pid_file: Path) -> int | None:
    """Read PID from file, return None if missing or invalid."""
    try:
        return int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def write_pid(pid_file: Path, pid: int) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))


def remove_pid(pid_file: Path) -> None:
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass


def is_process_alive(pid: int) -> bool:
    """Check if a process with given PID exists (signal 0)."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# -- Process lifecycle ------------------------------------------------------

def spawn_process(
    args: list[str],
    log_file: Path | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Spawn a background process, optionally redirecting output to log_file."""
    proc_env = {**os.environ, **(env or {})}
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_file, "a")
        return subprocess.Popen(
            args,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=proc_env,
            start_new_session=True,
        )
    return subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=cwd,
        env=proc_env,
        start_new_session=True,
    )


def terminate_process(pid: int, timeout: int = 10) -> bool:
    """Send SIGTERM, wait up to timeout seconds. Returns True if process died."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_process_alive(pid):
            return True
        time.sleep(0.2)
    # Force kill
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return True


def kill_by_port(port: int) -> bool:
    """Kill processes listening on the given port (uses lsof on macOS/Linux)."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = result.stdout.strip().split("\n")
        for pid_str in pids:
            pid_str = pid_str.strip()
            if pid_str.isdigit():
                try:
                    os.kill(int(pid_str), signal.SIGTERM)
                    log.info(f"Killed PID {pid_str} on port {port}")
                except ProcessLookupError:
                    pass
        return bool(pids and pids[0])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def stop_service(pid_file: Path, port: int) -> None:
    """Stop a service by PID file, falling back to port-based kill."""
    pid = read_pid(pid_file)
    if pid and is_process_alive(pid):
        log.info(f"Stopping PID {pid}")
        terminate_process(pid)
    remove_pid(pid_file)
    kill_by_port(port)


def start_and_wait(
    args: list[str],
    pid_file: Path,
    log_file: Path,
    host: str,
    port: int,
    kind: str,
    timeout: int = 30,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Spawn a process, record PID, and wait for health check.

    Returns the PID. Raises RuntimeError if health check fails.
    """
    from cli.lib.health import wait_for_healthy

    proc = spawn_process(args, log_file=log_file, cwd=cwd, env=env)
    pid = proc.pid
    write_pid(pid_file, pid)

    if not wait_for_healthy(host, port, kind, timeout=timeout):
        # Check if process died
        if not is_process_alive(pid):
            raise RuntimeError(
                f"Process died. Check log: {log_file}"
            )
        raise RuntimeError(
            f"Service did not become healthy within {timeout}s. "
            f"Check log: {log_file}"
        )
    return pid
