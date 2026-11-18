from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from cli.commands import start, status, stop
from cli.lib.local_config import load_local_config, resolve_local_config_path
from cli.main import build_parser
from cli.lib import local_runtime


def forbid(*args, **kwargs):
    raise AssertionError("Unexpected service operation")


@pytest.fixture
def setup_local(tmp_path, monkeypatch):
    for name in ("OGMEM_CONFIG", "SQL_CONNECTION_STRING", "STORAGE_BACKEND", "AGFS_PORT", "AGFS_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(start, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(stop, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(start, "check_agfs", forbid)
    monkeypatch.setattr(start, "start_and_wait", forbid)
    monkeypatch.setattr(local_runtime, "ensure_service", forbid)
    monkeypatch.setattr(local_runtime, "stop_recorded_service", lambda root: False)
    monkeypatch.setattr(stop, "stop_service", forbid)
    monkeypatch.setattr("cli.lib.db_setup.check_postgresql", forbid)

    def config(backend="sql", *, connection="host=db dbname=test password=PRIVATE_TEST_VALUE", port=28090):
        path = tmp_path / "config" / "ogmem.yaml"
        path.parent.mkdir(exist_ok=True)
        value = {"storage": {"backend": backend}, "service": {"http_port": port}}
        if connection is not None:
            value["storage"]["connection_string"] = connection
        path.write_text(json.dumps(value))
        return path
    return tmp_path, config


def arguments(**kwargs):
    return SimpleNamespace(mode="local", config=None, daemon=True, dry_run=False, **kwargs)


def test_export_prefixed_dotenv_selects_custom_config(tmp_path, monkeypatch):
    monkeypatch.delenv("OGMEM_CONFIG", raising=False)
    custom = tmp_path / "custom config.yaml"
    custom.write_text("storage: {backend: sql}")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text(f"export OGMEM_CONFIG='{custom}'\n")

    assert resolve_local_config_path(tmp_path) == custom.resolve()


@pytest.mark.skipif(not Path("/proc").exists(), reason="Linux process identity required")
def test_sigterm_during_startup_cleans_process_and_records(tmp_path):
    project = tmp_path / "project"
    server = project / "server"
    server.mkdir(parents=True)
    (server / "app.py").write_text(
        "import time\nwhile True:\n    time.sleep(1)\n"
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    repository = Path(__file__).resolve().parents[3]
    code = (
        "from pathlib import Path; from types import SimpleNamespace; import sys; "
        "from cli.lib import local_runtime; "
        "local_runtime.check_ogmem=lambda **kwargs: None; "
        "local_runtime.ensure_service(Path(sys.argv[1]), SimpleNamespace(http_port=int(sys.argv[2])), timeout=30)"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repository) + os.pathsep + env.get("PYTHONPATH", "")
    parent = subprocess.Popen(
        [sys.executable, "-c", code, str(project), str(port)],
        cwd=repository, env=env,
    )
    child_pid = None
    try:
        state_path = project / ".ogmem_data/local-runtime.json"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not state_path.exists():
            if parent.poll() is not None:
                pytest.fail("launcher exited before recording the child process")
            time.sleep(.05)
        assert state_path.exists()
        child_pid = json.loads(state_path.read_text())["pid"]

        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=5) == 128 + signal.SIGTERM
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and Path(f"/proc/{child_pid}").exists():
            time.sleep(.05)
        assert not Path(f"/proc/{child_pid}").exists()
        assert not state_path.exists()
        assert not (project / ".ogmem_data/ce.pid").exists()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if child_pid and Path(f"/proc/{child_pid}").exists():
            os.kill(child_pid, signal.SIGKILL)


def test_sql_starts_ce_without_consulting_agfs(setup_local, monkeypatch, capsys):
    root, write_config = setup_local
    path = write_config()
    monkeypatch.setenv("AGFS_SERVER_BIN", str(root / "does-not-exist"))
    monkeypatch.setenv("AGFS_PORT", "not-a-port-for-this-backend")
    connections, launches = [], []
    monkeypatch.setattr("cli.lib.db_setup.check_postgresql", lambda conn: connections.append(conn) or True)
    monkeypatch.setattr(local_runtime, "ensure_service", lambda *args, **kwargs: launches.append((args, kwargs)) or 12345)
    assert start.run(arguments()) == 0
    assert len(connections) == 1
    assert len(launches) == 1
    assert launches[0][0][0] == root
    assert launches[0][0][1].http_port == 28090
    import os
    assert os.environ["OGMEM_CONFIG"] == str(path.resolve())
    output = capsys.readouterr().out
    assert "AGFS" not in output and "PRIVATE_TEST_VALUE" not in output
    assert "ogmem stop local" in output
    assert "--config" not in output


@pytest.mark.parametrize("connection", [None, "host=unreachable"])
def test_sql_bad_connection_blocks_service_launch(setup_local, monkeypatch, capsys, connection):
    _, write_config = setup_local
    write_config(connection=connection)
    checked = []
    monkeypatch.setattr("cli.lib.db_setup.check_postgresql", lambda conn: checked.append(conn) or False)
    assert start.run(arguments()) == 1
    assert checked == ([] if connection is None else [connection])
    assert "SQL" in capsys.readouterr().out


def test_sql_dry_run_never_probes_or_claims_service_is_up(setup_local, monkeypatch, capsys):
    _, write_config = setup_local
    write_config()
    args = arguments()
    args.dry_run = True
    assert start.run(args) == 0
    output = capsys.readouterr().out
    assert "Would check SQL database" in output
    assert "PLANNED" in output and "AGFS" not in output
    assert " UP" not in output and "PRIVATE_TEST_VALUE" not in output


def test_sql_stop_only_stops_ce_and_index(setup_local, monkeypatch):
    _, write_config = setup_local
    write_config(port=28093)
    calls = []
    monkeypatch.setattr(local_runtime, "stop_recorded_service", lambda root: calls.append(("tracked-ce", None)))
    monkeypatch.setenv("AGFS_PORT", "unused-invalid-port")
    monkeypatch.setattr(stop, "stop_service", lambda path, port: calls.append((path.name, port)))
    args = build_parser().parse_args(["stop", "local"])
    assert stop.run(args) == 0
    assert calls == [("tracked-ce", None), ("index.pid", 0)]


def test_sql_foreground_shutdown_does_not_stop_agfs(setup_local, monkeypatch):
    _, write_config = setup_local
    write_config()
    monkeypatch.setattr("cli.lib.db_setup.check_postgresql", lambda conn: True)
    monkeypatch.setattr(local_runtime, "ensure_service", lambda *args, **kwargs: 12345)
    monkeypatch.setattr(start.signal, "signal", lambda *args: None)
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(start.time, "sleep", interrupt)
    calls = []
    monkeypatch.setattr("cli.lib.process.stop_service", lambda path, port: calls.append((path.name, port)))
    args = arguments()
    args.daemon = False
    with pytest.raises(SystemExit) as exc:
        start.run(args)
    assert exc.value.code == 0
    assert calls == []


def test_native_agfs_start_and_status_use_service_health(setup_local, monkeypatch):
    root, write_config = setup_local
    write_config("agfs")
    monkeypatch.setenv("AGFS_TRANSPORT", "native")
    launches = []
    monkeypatch.setattr(local_runtime, "ensure_service", lambda project, cfg, **kw: launches.append(cfg.agfs_transport) or 123)
    assert start.run(arguments()) == 0
    assert launches == ["native"]

    health = {"backend": "og-memory", "status": "ok", "storage_backend": "agfs",
              "agfs_transport": "native", "agfs": True, "agfs_url": "agfs-native:///opt/agfs/libagfsbinding.so"}
    monkeypatch.setattr(status, "running_state", lambda project: {"port": 28090})
    monkeypatch.setattr(status, "check_ogmem", lambda **kw: health)
    state = status._check_local_mode(root)
    assert state["healthy"] and state["agfs"]
    assert state["agfs_url"] == health["agfs_url"]


def test_agfs_still_starts_and_stops_both_services(setup_local, monkeypatch):
    root, write_config = setup_local
    write_config("agfs")
    binary = root / "agfs-server"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    monkeypatch.setenv("AGFS_SERVER_BIN", str(binary))
    monkeypatch.setenv("AGFS_PORT", "21833")
    monkeypatch.setattr(start, "check_agfs", lambda **kwargs: None)
    launches = []
    monkeypatch.setattr(start, "start_and_wait", lambda *args, **kwargs: launches.append(args[5]) or 123)
    monkeypatch.setattr(local_runtime, "ensure_service", lambda *args, **kwargs: launches.append("ogmem") or 124)
    assert start.run(arguments()) == 0
    assert launches == ["agfs", "ogmem"]
    (root / ".ogmem_data/agfs.pid").write_text("123")
    stops = []
    monkeypatch.setattr(stop, "stop_service", lambda path, port: stops.append((path.name, port)))
    assert stop.run(arguments()) == 0
    assert stops == [("agfs.pid", 0), ("index.pid", 0)]


def test_agfs_missing_binary_still_blocks_start(setup_local, monkeypatch, capsys):
    root, write_config = setup_local
    write_config("agfs")
    monkeypatch.setenv("AGFS_SERVER_BIN", str(root / "missing"))
    monkeypatch.setattr(start, "check_agfs", lambda **kwargs: None)
    assert start.run(arguments()) == 1
    assert "AGFS binary not found" in capsys.readouterr().out


@pytest.mark.parametrize("command", [start, stop])
def test_invalid_config_cannot_fall_back_to_agfs(setup_local, command, capsys, monkeypatch):
    root, write_config = setup_local
    path = write_config()
    path.write_text("storage: [invalid\n")
    if command is stop:
        calls = []
        monkeypatch.setattr(local_runtime, "stop_recorded_service", lambda root: calls.append("tracked-ce"))
        monkeypatch.setattr(stop, "stop_service", lambda path, port: calls.append(path.name))
        assert command.run(arguments()) == 0
        assert calls == ["tracked-ce", "index.pid"]
    else:
        assert command.run(arguments()) == 1
        assert "Cannot load local configuration" in capsys.readouterr().out


def test_loader_honors_explicit_config_and_ignores_stale_singleton(setup_local, monkeypatch):
    root, write_config = setup_local
    path = write_config()
    stale = SimpleNamespace(storage_backend="agfs", http_port=8090)
    monkeypatch.setattr("providers.unified_config._global_config", stale)
    monkeypatch.setenv("OGMEM_CONFIG", str(root / "wrong-config.yaml"))
    config = load_local_config(root, str(path))
    assert config.storage_backend == "sql" and config.http_port == 28090
    write_config(port=28094)
    assert load_local_config(root, str(path)).http_port == 28094


def test_loader_preserves_explicit_process_environment(setup_local, monkeypatch):
    root, write_config = setup_local
    path = write_config()
    (root / "config" / ".env").write_text(
        "SQL_CONNECTION_STRING=from-dotenv\nLLM_API_KEY=from-dotenv\n"
    )
    monkeypatch.setenv("SQL_CONNECTION_STRING", "from-process")
    monkeypatch.setenv("LLM_API_KEY", "from-process")
    load_local_config(root, str(path))
    assert os.environ["SQL_CONNECTION_STRING"] == "from-process"
    assert os.environ["LLM_API_KEY"] == "from-process"


def test_stale_local_record_does_not_hide_docker_status(tmp_path, monkeypatch, capsys):
    data = tmp_path / ".ogmem_data"
    docker = data / "onboard-docker"
    docker.mkdir(parents=True)
    (data / "ce.pid").write_text("99999999")
    (docker / "ogmemory.env").write_text("export INSTANCE_NAME=status-test\n")
    monkeypatch.setattr(status, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(status, "running_state", lambda root: None)
    monkeypatch.setattr(status, "_check_local_mode", forbid)
    calls = []
    monkeypatch.setattr(status, "_check_docker_mode", lambda root: calls.append(root) or {
        "agfs": False, "agfs_health": None, "agfs_required": False,
        "ogmem": True, "ogmem_health": {"status": "ok"},
        "ogmem_port": 18090, "backend": "sql", "healthy": True, "docker": True,
    })

    assert status.run(SimpleNamespace()) == 0
    assert calls == [tmp_path]
    assert "Docker" in capsys.readouterr().out


@pytest.mark.parametrize("health,expected", [
    ({"backend": "og-memory", "status": "ok", "storage_backend": "agfs"}, 0),
    (None, 1),
])
def test_docker_status_uses_exact_instance_binding_and_health(
    tmp_path, monkeypatch, capsys, health, expected,
):
    envfile = tmp_path / ".ogmem_data/onboard-docker/ogmemory.env"
    envfile.parent.mkdir(parents=True)
    envfile.write_text("export INSTANCE_NAME=status-test\nexport OGMEM_HTTP_HOST_PORT=18090\n")
    monkeypatch.setattr(status, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(status, "check_agfs", forbid)
    checked_ports = []
    monkeypatch.setattr(
        status, "check_ogmem", lambda *, port: checked_ports.append(port) or health,
    )

    def docker(command, **kwargs):
        assert command[0] == "docker" and "status-test_ogmemory" in command
        if command[1] == "inspect":
            return SimpleNamespace(returncode=0, stdout="true\n")
        assert command[1:4] == ["port", "status-test_ogmemory", "8090/tcp"]
        return SimpleNamespace(returncode=0, stdout="0.0.0.0:28090\n")

    monkeypatch.setattr(status.subprocess, "run", docker)
    assert status.run(SimpleNamespace()) == expected
    output = capsys.readouterr().out
    assert checked_ports == [28090]
    assert "http://127.0.0.1:28090" in output
    assert "AGFS            " not in output
    assert (" UP" in output) == (expected == 0)
