"""Exercise configuration and real child-process lifecycle failures."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def runtime():
    # Import inside the test so a missing runtime is reported as the missing feature.
    path = ROOT / 'scripts' / 'standalone_runtime.py'
    assert path.is_file(), 'standalone runtime must supervise children and resolve config safely'
    import importlib.util
    spec = importlib.util.spec_from_file_location('standalone_runtime', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_env_values_remain_yaml_strings_and_dsn_round_trips(tmp_path, monkeypatch):
    m = runtime()
    secret = "p'ass\\word\n: [value] # still secret"
    monkeypatch.setenv('TEST_SECRET', secret)
    monkeypatch.setenv('OPENGAUSS_PASSWORD', secret)
    source = tmp_path / 'source.yaml'
    source.write_text('llm:\n  api_key: "${TEST_SECRET}"\nvector_db:\n  connection_string: old\n')
    target = tmp_path / 'resolved.yaml'
    m.resolve_config(source, target)
    result = yaml.safe_load(target.read_text())
    assert result['llm']['api_key'] == secret
    from psycopg2.extensions import parse_dsn
    assert parse_dsn(result['vector_db']['connection_string'])['password'] == secret
    assert target.stat().st_mode & 0o777 == 0o600


def test_rejects_non_mapping_config(tmp_path):
    m = runtime()
    source = tmp_path / 'bad.yaml'
    source.write_text('- not a config mapping\n')
    with pytest.raises(ValueError, match='mapping'):
        m.resolve_config(source, tmp_path / 'out.yaml')


def test_child_failure_stops_other_children(tmp_path):
    m = runtime()
    marker = tmp_path / 'stopped'
    child = "import signal,time,pathlib,sys; signal.signal(signal.SIGTERM, lambda *_: (pathlib.Path(sys.argv[1]).write_text('stopped'),sys.exit(0))); print('ready',flush=True); time.sleep(60)"
    supervisor = m.Supervisor(shutdown_timeout=2)
    survivor = supervisor.start('survivor', [sys.executable, '-u', '-c', child, str(marker)])
    time.sleep(.2)
    supervisor.start('failed', [sys.executable, '-c', 'raise SystemExit(7)'])
    assert supervisor.monitor() == 1
    assert survivor.poll() is not None
    assert marker.read_text() == 'stopped'


def test_agfs_readiness_timeout_is_fatal_and_cleans_up():
    m = runtime()
    supervisor = m.Supervisor(shutdown_timeout=1)
    child = supervisor.start('agfs', [sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        with pytest.raises(RuntimeError, match='ready'):
            supervisor.wait_ready('http://127.0.0.1:1/not-running', timeout=.1)
    finally:
        supervisor.stop()
    assert child.poll() is not None


@pytest.mark.parametrize('business_name', ['HTTP', 'MCP', 'IndexService'])
def test_shutdown_keeps_agfs_available_until_business_drains(tmp_path, business_name):
    m = runtime()
    available = tmp_path / 'agfs-available'
    ready = tmp_path / 'business-ready'
    result = tmp_path / 'drain-result'
    agfs_code = """
import pathlib, signal, sys, time
available = pathlib.Path(sys.argv[1])
def stop(*_):
    available.unlink()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
available.touch()
while True: time.sleep(.01)
"""
    business_code = """
import pathlib, signal, sys, time
available, ready, result = map(pathlib.Path, sys.argv[1:])
def stop(*_):
    time.sleep(.1)
    result.write_text('drained' if available.exists() else 'storage unavailable')
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
ready.touch()
while True: time.sleep(.01)
"""
    supervisor = m.Supervisor(shutdown_timeout=2)
    agfs = supervisor.start('AGFS', [sys.executable, '-c', agfs_code, str(available)])
    business = supervisor.start(business_name, [sys.executable, '-c', business_code,
                                               str(available), str(ready), str(result)])
    try:
        deadline = time.monotonic() + 5
        while not (available.exists() and ready.exists()):
            assert time.monotonic() < deadline, 'children did not become ready'
            time.sleep(.01)
        supervisor.stop()
        assert result.read_text() == 'drained'
        assert agfs.poll() is not None
        assert business.poll() is not None
    finally:
        for child in (agfs, business):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def test_shutdown_kills_unresponsive_children_within_timeout(tmp_path):
    m = runtime()
    ready = tmp_path / 'ready'
    code = "import signal,time,pathlib,sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
    supervisor = m.Supervisor(shutdown_timeout=.3)
    child = supervisor.start('HTTP', [sys.executable, '-c', code, str(ready)])
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        started = time.monotonic()
        supervisor.stop()
        assert time.monotonic() - started < 1.5
        assert child.returncode == -signal.SIGKILL
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()


@pytest.mark.parametrize(('extra_body', 'expected'), [
    (None, {}),
    ('', {}),
    ('{}', {}),
    ('{"thinking":{"type":"disabled"}}', {'thinking': {'type': 'disabled'}}),
])
def test_compose_optional_extra_body_loads_configuration(tmp_path, monkeypatch, extra_body, expected):
    import json
    import re
    import shutil
    from providers.unified_config import OgMemConfig

    if shutil.which('docker') is None:
        pytest.skip('Docker Compose CLI required; no daemon needed')
    compose_version = subprocess.run(['docker', 'compose', 'version'],
                                     capture_output=True, text=True, timeout=10)
    if compose_version.returncode != 0:
        pytest.skip('Docker Compose plugin unavailable; no daemon needed')
    monkeypatch.delenv('LLM_EXTRA_BODY', raising=False)
    monkeypatch.delenv('OGMEM_MODEL_CONFIG_OVERRIDES', raising=False)
    template = (ROOT / '.env.example').read_text()
    if extra_body is None:
        template = re.sub(r'^LLM_EXTRA_BODY=.*\n', '', template, flags=re.MULTILINE)
    else:
        template = re.sub(r'^LLM_EXTRA_BODY=.*$', 'LLM_EXTRA_BODY=' + extra_body,
                          template, flags=re.MULTILINE)
    env_file = tmp_path / '.env'
    env_file.write_text(template)
    for key in re.findall(r'\$\{(\w+):\?', (ROOT / 'compose.yaml').read_text()):
        monkeypatch.setenv(key, '1024' if key == 'EMBEDDING_DIMENSION' else 'test-only')
    result = subprocess.run(['docker', 'compose', '--env-file', str(env_file),
                             '-f', str(ROOT / 'compose.yaml'), 'config', '--format', 'json'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    environment = json.loads(result.stdout)['services']['ogmemory']['environment']
    for key, value in environment.items():
        monkeypatch.setenv(key, str(value))
    # Image deployment stores model configuration in a generated YAML, not service env.
    # Keep testing the source runtime's optional JSON override independently.
    for key in re.findall(r'\$\{(\w+)\}', (ROOT / 'deploy/compose/ogmemory.yaml').read_text()):
        if key not in environment:
            monkeypatch.setenv(key, '1024' if key == 'EMBEDDING_DIMENSION' else
                               '1' if key == 'OGMEM_AFTER_TURN_THRESHOLD' else 'test-only')
    if extra_body is not None:
        monkeypatch.setenv('LLM_EXTRA_BODY', extra_body or '{}')
    resolved = tmp_path / 'resolved.yaml'
    runtime().resolve_config(ROOT / 'deploy/compose/ogmemory.yaml', resolved)
    assert OgMemConfig.load(str(resolved)).llm_extra_body == expected


@pytest.mark.parametrize('availability', ['no-docker', 'no-compose', 'bad-config'])
def test_compose_dependency_gate_skips_only_missing_tools(tmp_path, monkeypatch, availability):
    tool_dir = tmp_path / 'tools'
    tool_dir.mkdir()
    monkeypatch.setenv('PATH', str(tool_dir))
    if availability != 'no-docker':
        fake_docker = tool_dir / 'docker'
        fake_docker.write_text(
            f'#!{sys.executable}\n'
            'import sys\n'
            + ('print("Compose plugin unavailable"); sys.exit(1)\n'
               if availability == 'no-compose' else
               'if sys.argv[1:] == ["compose", "version"]: sys.exit(0)\n'
               'print("Invalid Compose configuration", file=sys.stderr); sys.exit(9)\n')
        )
        fake_docker.chmod(0o755)
    expected_exception = AssertionError if availability == 'bad-config' else pytest.skip.Exception
    with pytest.raises(expected_exception):
        test_compose_optional_extra_body_loads_configuration(tmp_path, monkeypatch, '', {})
