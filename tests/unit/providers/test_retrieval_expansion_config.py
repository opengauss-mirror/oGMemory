"""Service-level hierarchy expansion configuration and validation."""

from __future__ import annotations

import pytest

from providers.unified_config import OgMemConfig

ENV_NAME = "OGMEM_RETRIEVAL_EXPAND_HIERARCHY"


@pytest.fixture(autouse=True)
def clean_retrieval_expansion_env(monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)


def write_config(tmp_path, value: str | None = None):
    path = tmp_path / "ogmem.yaml"
    path.write_text(
        "{}\n" if value is None else f"retrieval:\n  expand_hierarchy: {value}\n",
        encoding="utf-8",
    )
    return str(path)


def test_hierarchy_expansion_defaults_to_enabled(tmp_path):
    assert OgMemConfig().retrieval_expand_hierarchy is True
    assert OgMemConfig.load(write_config(tmp_path)).retrieval_expand_hierarchy is True


@pytest.mark.parametrize(
    ("yaml_value", "expected"),
    [
        ("true", True),
        ("false", False),
        ("'  TrUe  '", True),
        ("'NO'", False),
        ("1", True),
        ("0", False),
    ],
)
def test_load_hierarchy_expansion_from_yaml(tmp_path, yaml_value, expected):
    cfg = OgMemConfig.load(write_config(tmp_path, yaml_value))
    assert cfg.retrieval_expand_hierarchy is expected


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("true", True), ("false", False), ("  YES  ", True), ("NO", False), ("1", True), ("0", False)],
)
def test_load_hierarchy_expansion_from_env(tmp_path, monkeypatch, env_value, expected):
    monkeypatch.setenv(ENV_NAME, env_value)
    cfg = OgMemConfig.load(write_config(tmp_path))
    assert cfg.retrieval_expand_hierarchy is expected


@pytest.mark.parametrize(("yaml_value", "expected"), [("true", True), ("false", False)])
def test_yaml_hierarchy_expansion_overrides_env(tmp_path, monkeypatch, yaml_value, expected):
    monkeypatch.setenv(ENV_NAME, "false" if expected else "true")
    cfg = OgMemConfig.load(write_config(tmp_path, yaml_value))
    assert cfg.retrieval_expand_hierarchy is expected


def test_yaml_hierarchy_expansion_ignores_invalid_lower_priority_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_NAME, "invalid")
    cfg = OgMemConfig.load(write_config(tmp_path, "false"))
    assert cfg.retrieval_expand_hierarchy is False


@pytest.mark.parametrize("yaml_value", ["'fals'", "''", "2", "1.0", "[]", "{}"])
def test_invalid_yaml_hierarchy_expansion_is_rejected(tmp_path, monkeypatch, yaml_value):
    monkeypatch.setenv(ENV_NAME, "true")
    with pytest.raises(ValueError, match=r"retrieval\.expand_hierarchy.*boolean"):
        OgMemConfig.load(write_config(tmp_path, yaml_value))


@pytest.mark.parametrize("env_value", ["fals", "", "2", "1.0", "[]"])
def test_invalid_env_hierarchy_expansion_is_rejected(tmp_path, monkeypatch, env_value):
    monkeypatch.setenv(ENV_NAME, env_value)
    with pytest.raises(ValueError, match=f"{ENV_NAME}.*boolean"):
        OgMemConfig.load(write_config(tmp_path))


@pytest.mark.parametrize("value", ["false", 0, 1, [], None])
def test_direct_config_requires_a_boolean_hierarchy_expansion(value):
    with pytest.raises(ValueError, match="retrieval_expand_hierarchy.*boolean"):
        OgMemConfig(retrieval_expand_hierarchy=value)


def test_new_flag_does_not_change_other_boolean_parsing(tmp_path):
    path = tmp_path / "ogmem.yaml"
    path.write_text(
        "retrieval:\n  expand_hierarchy: false\n  candidate_pool_enabled: invalid\n",
        encoding="utf-8",
    )
    cfg = OgMemConfig.load(str(path))
    assert cfg.retrieval_expand_hierarchy is False
    assert cfg.retrieval_candidate_pool_enabled is False
