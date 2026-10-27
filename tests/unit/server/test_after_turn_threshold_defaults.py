"""Buffer defaults and explicit overrides for threshold comparisons."""

import pytest

from providers.unified_config import OgMemConfig


def test_default_buffer_is_400_tokens(tmp_path, monkeypatch):
    monkeypatch.delenv("OGMEM_AFTER_TURN_THRESHOLD", raising=False)
    assert OgMemConfig().after_turn_threshold == 400
    assert OgMemConfig.load(str(tmp_path / "missing.yaml")).after_turn_threshold == 400


@pytest.mark.parametrize("threshold", [200, 400])
def test_explicit_yaml_threshold_is_preserved(tmp_path, threshold, monkeypatch):
    monkeypatch.delenv("OGMEM_AFTER_TURN_THRESHOLD", raising=False)
    path = tmp_path / "ogmemory.yaml"
    path.write_text(f"memory:\n  after_turn_threshold: {threshold}\n")
    assert OgMemConfig.load(str(path)).after_turn_threshold == threshold


@pytest.mark.parametrize("threshold", [200, 400])
def test_explicit_env_threshold_is_preserved(tmp_path, threshold, monkeypatch):
    monkeypatch.setenv("OGMEM_AFTER_TURN_THRESHOLD", str(threshold))
    assert OgMemConfig.load(str(tmp_path / "missing.yaml")).after_turn_threshold == threshold
