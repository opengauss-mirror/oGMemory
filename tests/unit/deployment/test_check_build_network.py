import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


def probe():
    path = ROOT / 'scripts/check_build_network.py'
    assert path.is_file(), 'build network preflight is required'
    spec = importlib.util.spec_from_file_location('build_network_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_uses_compose_selected_sources_and_platform():
    m = probe()
    config = {'services': {'ogmemory': {'platform': 'linux/arm64', 'build': {
        'context': str(ROOT), 'dockerfile': 'docker/Dockerfile.compose',
        'args': {'GOPROXY': 'https://mirror.example.test,direct',
                 'GOSUMDB': 'sum.golang.org', 'PRIVATE_KEY': 'must-not-pass'},
    }, 'environment': {'LLM_API_KEY': 'must-not-pass'}}}}
    spec = m.build_spec(config)
    service = spec['services']['source-check']
    assert service['platform'] == 'linux/arm64'
    assert service['build']['target'] == 'agfs-deps'
    assert service['build']['args'] == {
        'GOPROXY': 'https://mirror.example.test,direct', 'GOSUMDB': 'sum.golang.org'}
    assert 'must-not-pass' not in str(spec)


def test_invalid_compose_reports_no_configuration_or_credentials(monkeypatch, capsys):
    m = probe()
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 1, '', 'LLM_API_KEY=secret'))
    assert m.main() == 1
    assert 'secret' not in capsys.readouterr().out


def test_preflight_failure_is_returned(monkeypatch, capsys):
    m = probe()
    monkeypatch.setattr(m, 'load_config', lambda: {})
    monkeypatch.setattr(m, 'check_images', lambda _: 0)
    monkeypatch.setattr(m, 'build_spec', lambda _, target: {})
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 7))
    assert m.main() == 7
    assert 'failed' in capsys.readouterr().out


def test_missing_image_pull_failure_blocks_dependency_build(monkeypatch, capsys):
    m = probe()
    config = {'services': {'ogmemory': {'build': {'args': {
        'GO_BASE_IMAGE': 'mirror.test/go:1.25', 'PYTHON_BASE_IMAGE': 'mirror.test/python:3.11',
    }}}, 'opengauss': {'image': 'mirror.test/opengauss:latest'}}}
    monkeypatch.setattr(m, 'load_config', lambda: config)
    def run(command, **kwargs):
        if command[:3] == ['docker', 'image', 'inspect']:
            return subprocess.CompletedProcess(command, 1, '', '')
        assert command[:4] == ['docker', 'pull', '--platform', 'linux/arm64']
        return subprocess.CompletedProcess(command, 9)
    monkeypatch.setattr(m.subprocess, 'run', run)
    assert m.main() == 9
    assert 'Image check failed' in capsys.readouterr().out


def test_python_preflight_uses_selected_package_sources():
    m = probe()
    config = {'services': {'ogmemory': {'platform': 'linux/arm64', 'build': {
        'context': str(ROOT), 'dockerfile': 'docker/Dockerfile.compose', 'args': {
            'DEBIAN_MIRROR': 'https://apt.example.test/debian',
            'DEBIAN_SECURITY_MIRROR': 'https://apt.example.test/debian-security',
            'PIP_INDEX_URL': 'https://pip.example.test/simple',
        }}}}}
    spec = m.build_spec(config, 'python-deps')
    build = spec['services']['source-check']['build']
    assert build['target'] == 'python-deps'
    assert build['args'] == config['services']['ogmemory']['build']['args']


@pytest.mark.parametrize('target', ['agfs-deps', 'python-deps'])
def test_preflight_preserves_build_host_gateway(target):
    m = probe()
    config = {'services': {'ogmemory': {'platform': 'linux/arm64', 'build': {
        'context': str(ROOT), 'dockerfile': 'docker/Dockerfile.compose',
        'extra_hosts': {'host.docker.internal': 'host-gateway'},
        'args': {'HTTP_PROXY': 'http://host.docker.internal:7890'},
    }}}}
    build = m.build_spec(config, target)['services']['source-check']['build']
    assert build['extra_hosts'] == {'host.docker.internal': 'host-gateway'}


def test_image_deployment_can_reach_host_models():
    import yaml
    config = yaml.safe_load((ROOT / 'compose.yaml').read_text())
    assert 'host.docker.internal:host-gateway' in config['services']['ogmemory']['extra_hosts']
