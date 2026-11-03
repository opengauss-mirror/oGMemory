import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def probe():
    path = ROOT / 'scripts/check_deployment.py'
    assert path.is_file(), 'deployment probe is required'
    spec = importlib.util.spec_from_file_location('deployment_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_reports_stage_without_credentials():
    m = probe()
    def fail():
        raise RuntimeError('password=do-not-log https://user:secret@example.test')
    result = m.run_checks({'database': fail, 'agfs': lambda: None})
    assert result == {'database': {'ok': False, 'error_type': 'RuntimeError'}, 'agfs': {'ok': True}}


def test_wrong_embedding_dimension_is_rejected():
    m = probe()
    with pytest.raises(ValueError, match='dimension'):
        m.validate_embedding([[1.0, 2.0]], 3)


@pytest.mark.parametrize('vectors', [[], [[float('nan'), 0]], [[0, float('inf')]], [[0, 0]]])
def test_unusable_embedding_is_rejected(vectors):
    m = probe()
    with pytest.raises(ValueError):
        m.validate_embedding(vectors, 2)


def test_container_exec_uses_resolved_config(tmp_path, monkeypatch):
    m = probe()
    source = tmp_path / 'source.yaml'
    source.write_text('vector_db: {}')
    resolved = tmp_path / 'resolved.yaml'
    resolved.write_text('vector_db: {connection_string: resolved}')
    monkeypatch.setenv('OGMEM_CONFIG', str(source))
    assert m.config_path(resolved) == str(resolved)
    resolved.unlink()
    assert m.config_path(resolved) == str(source)


def test_nested_provider_config_uses_same_resolved_file(tmp_path, monkeypatch):
    import sys
    from providers.unified_config import get_config, reset_config
    m = probe()
    source = tmp_path / 'source.yaml'
    source.write_text('vector_db: {dimension: "${MISSING_DIMENSION}"}')
    resolved = tmp_path / 'resolved.yaml'
    resolved.write_text('vector_db: {dimension: 3}')
    monkeypatch.setenv('OGMEM_CONFIG', str(source))
    monkeypatch.setattr(m, 'config_path', lambda: str(resolved))
    monkeypatch.setattr(sys, 'argv', ['check_deployment'])
    def check_nested(_checks):
        assert get_config().opengauss_dimension == 3
        return {'nested_config': {'ok': True}}
    monkeypatch.setattr(m, 'run_checks', check_nested)
    import logging
    previous_disable = logging.root.manager.disable
    reset_config()
    try:
        assert m.main() == 0
    finally:
        reset_config()
        logging.disable(previous_disable)
