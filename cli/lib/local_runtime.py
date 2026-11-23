# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Start and stop one verified local HTTP process per project."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

from cli.lib.health import check_ogmem


def private_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def runtime_lock(root: Path):
    import fcntl
    path = root / ".ogmem_data/local.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def process_identity(pid: int, root: Path) -> str | None:
    """Only recognize this project's server process; also detect PID reuse."""
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z" or (proc / "cwd").resolve() != root.resolve():
            return None
        argv = (proc / "cmdline").read_bytes().split(b"\0")
        if not any(arg in {b"server/app.py", str(root / "server/app.py").encode()} for arg in argv):
            return None
        return stat[19]
    except (OSError, IndexError):
        if Path("/proc").exists():
            return None
    # macOS: new launches use an absolute script path; verify it and start time.
    result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=,command="], capture_output=True, text=True)
    if result.returncode == 0 and str(root / "server/app.py") in result.stdout:
        return result.stdout[:24].strip()
    return None


def running_state(root: Path) -> dict | None:
    path = root / ".ogmem_data/local-runtime.json"
    if path.exists():
        state = json.loads(path.read_text())
        identity = process_identity(state["pid"], root)
        if identity and identity == state["identity"]:
            return state
        # A live PID that cannot be verified must never be stopped or overwritten.
        try:
            os.kill(state["pid"], 0)
        except ProcessLookupError:
            return None
        if Path(f'/proc/{state["pid"]}/stat').exists():
            if Path(f'/proc/{state["pid"]}/stat').read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return None
        raise RuntimeError("Recorded service PID cannot be verified; inspect local-runtime.json before proceeding")
    pidfile = root / ".ogmem_data/ce.pid"
    if pidfile.exists():
        pid = int(pidfile.read_text().strip())
        identity = process_identity(pid, root)
        if identity:
            return {"pid": pid, "identity": identity, "legacy": True}
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return None
        raise RuntimeError("Existing ce.pid does not identify this project's server; refusing to replace it")
    return None


def port_available(port: int) -> bool:
    try:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False


def stop_verified(root: Path, state: dict) -> None:
    pid = state["pid"]
    if process_identity(pid, root) != state["identity"]:
        raise RuntimeError("Service identity changed; refusing to signal PID")
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if process_identity(pid, root) is None:
            return
        time.sleep(.1)
    if process_identity(pid, root) == state["identity"]:
        os.kill(pid, signal.SIGKILL)
    for _ in range(20):
        if process_identity(pid, root) is None:
            return
        time.sleep(.1)
    raise RuntimeError("Old service did not stop; new service was not started")


def require_stopped(root: Path) -> None:
    state = running_state(root)
    if state:
        raise RuntimeError(f"Local service is running (PID {state['pid']}). Run 'ogmem stop local' before configuring or starting it again.")


def ensure_service(root: Path, cfg, *, timeout: int = 30) -> int:
    """Refuse a second launch; never claim saved changes are already active."""
    with runtime_lock(root):
        require_stopped(root)
        if not port_available(cfg.http_port):
            raise RuntimeError(f"Port {cfg.http_port} is occupied; no service was stopped")
        data = root / ".ogmem_data"
        log = data / "logs/context_engine.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        proc = None
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def cancel_startup(signum, _frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, cancel_startup)
        try:
            with log.open("a") as stream:
                proc = subprocess.Popen([sys.executable, str(root / "server/app.py")], cwd=root,
                                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            identity = process_identity(proc.pid, root)
            if not identity:
                raise RuntimeError("Cannot verify started service identity")
            # Record the detached process before waiting so stop/status can always find it.
            state = {"pid": proc.pid, "identity": identity, "port": cfg.http_port}
            private_write(data / "local-runtime.json", json.dumps(state))
            private_write(data / "ce.pid", str(proc.pid))
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"Service exited during startup; see {log}")
                response = check_ogmem(port=cfg.http_port)
                if response and response.get("status") == "ok" and response.get("backend") == "og-memory":
                    return proc.pid
                time.sleep(.2)
            raise RuntimeError(f"Service did not become healthy; see {log}")
        except BaseException:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            # The old service was already stopped explicitly. Do not leave a failed PID.
            for name in ("ce.pid", "local-runtime.json"):
                (data / name).unlink(missing_ok=True)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)


def stop_recorded_service(root: Path) -> bool:
    with runtime_lock(root):
        state = running_state(root)
        if state:
            stop_verified(root, state)
        for name in ("ce.pid", "local-runtime.json"):
            (root / ".ogmem_data" / name).unlink(missing_ok=True)
        return state is not None
