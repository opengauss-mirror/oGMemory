"""Run the real shell entrypoint with lightweight service processes."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX shell and signals required")


def wait_for(predicate, process):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if predicate():
            return
        if process.poll() is not None:
            pytest.fail(f"entrypoint exited early: {process.communicate()}")
        time.sleep(0.02)
    pytest.fail("service processes did not become ready")


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT, None])
@pytest.mark.parametrize("bundled,mcp", [(False, False), (False, True), (True, False), (True, True)])
def test_shutdown_drains_children_and_preserves_api_failure(tmp_path, stop_signal, bundled, mcp):
    events = tmp_path / "events"
    events.touch()
    stub = tmp_path / "python3"
    stub.write_text(f"#!{sys.executable}\n" + '''
import os, signal, sys, time
from pathlib import Path
args = sys.argv[1:]
if not args and Path(sys.argv[0]).name != 'agfs-server':
    sys.stdin.read()
    sys.exit(0)
if args and args[0].endswith('yaml_to_env.py'):
    print('export OGMEM_START_BUNDLED_AGFS=' + os.environ['BUNDLED'])
    print('export OGMEM_MCP_ENABLED=' + os.environ['MCP'])
    sys.exit(0)
role = ('agfs' if Path(sys.argv[0]).name == 'agfs-server' else
        'api' if 'gunicorn' in args else 'mcp' if 'cli.main' in args else 'index')
def event(action):
    with open(os.environ['EVENTS'], 'a') as f:
        f.write(role + ':' + action + '\\n')
def stop(*_):
    event('stop')
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
event('ready')
while True:
    if role == 'api' and Path(os.environ['EXIT_API']).exists():
        event('failed')
        sys.exit(7)
    time.sleep(0.01)
''')
    stub.chmod(0o755)
    agfs = tmp_path / "agfs-server"
    agfs.symlink_to(stub)
    curl = tmp_path / "curl"
    curl.write_text("#!/bin/sh\nexit 0\n")
    curl.chmod(0o755)
    script = (ROOT / "docker/entrypoint-standalone.sh").read_text()
    script = script.replace("/opt/ogmem", str(tmp_path))
    script = script.replace("/opt/agfs/agfs-server", str(agfs))
    script = script.replace("/data/agfs", str(tmp_path / "data"))
    entrypoint = tmp_path / "entrypoint.sh"
    entrypoint.write_text(script)
    process = subprocess.Popen(
        ["/bin/sh", str(entrypoint)],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "EVENTS": str(events), "EXIT_API": str(tmp_path / "exit-api"),
             "BUNDLED": str(bundled).lower(), "MCP": str(mcp).lower()},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    roles = {"api", "index"} | ({"mcp"} if mcp else set()) | ({"agfs"} if bundled else set())
    try:
        wait_for(lambda: all(f"{role}:ready" in events.read_text() for role in roles), process)
        if stop_signal is None:
            (tmp_path / "exit-api").touch()
        else:
            process.send_signal(stop_signal)
        output = process.communicate(timeout=8)
        assert process.returncode == (7 if stop_signal is None else 0), output
        lines = events.read_text().splitlines()
        assert lines.count("api:failed" if stop_signal is None else "api:stop") == 1
        for role in roles - {"api"}:
            assert lines.count(f"{role}:stop") == 1, lines
        if bundled:
            assert lines[-1] == "agfs:stop", lines
        assert output[0].count("Shutting down...") == 1
    finally:
        # Ensure a failing regression never leaves the service stubs running.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)
