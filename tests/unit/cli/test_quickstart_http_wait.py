"""Run real shell helpers against stalled HTTP connections and URI variants."""

import os
import re
import shutil
import signal
import socket
import subprocess
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HAS_BASH4 = (os.name != "nt" and shutil.which("bash") is not None and
             subprocess.run(["bash", "-c", "((BASH_VERSINFO[0] >= 4))"], check=False).returncode == 0)
pytestmark = pytest.mark.skipif(not HAS_BASH4, reason="Quickstart requires Bash 4 or newer")


def function(name):
    source = (ROOT / "deploy/user-quickstart/deploy.sh").read_text()
    return re.search(rf"^{name}\(\) \{{.*?^\}}", source, re.M | re.S).group()


@pytest.mark.parametrize("response,status", [("ok", 0), ("silent", 1), ("unready", 1)])
def test_http_wait_has_request_and_overall_deadlines(tmp_path, response, status):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    stopped = threading.Event()

    def serve():
        try:
            listener.settimeout(8)
            conn, _ = listener.accept()
            with conn:
                if response == "silent":
                    stopped.wait(10)
                else:
                    code = "200 OK" if response == "ok" else "503 Unavailable"
                    conn.sendall(f"HTTP/1.1 {code}\r\nContent-Length: 0\r\n\r\n".encode())
        except (OSError, TimeoutError):
            pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nexit 0\n")
    docker.chmod(0o700)
    process = subprocess.Popen(
        ["bash", "-c", function("fail") + "\n" + function("wait_http") +
         '\nwait_http test "$1" isolated', "check", f"http://127.0.0.1:{port}"],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
             "OGMEM_DEPLOY_WAIT_SECONDS": "2", "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == status, (stdout, stderr)
        assert ("启动超时" in stderr) == (status == 1)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)
        stopped.set()
        listener.close()
        thread.join(timeout=2)


@pytest.mark.parametrize("uri,host", [
    ("http://milvus:19530", "milvus"),
    ("https://Milvus.Example:443/path", "milvus.example"),
    ("http://user:password@milvus.example:19530", "milvus.example"),
    ("http://[2001:DB8::1]:19530", "2001:db8::1"),
    ("https://milvus.example?option=value", "milvus.example"),
])
def test_proxy_bypass_uses_hostname_not_port_userinfo_or_uri_suffix(uri, host):
    result = subprocess.check_output(
        ["bash", "-c", function("milvus_proxy_host") + '\nmilvus_proxy_host "$1"', "check", uri],
        text=True,
    )
    assert result == host
