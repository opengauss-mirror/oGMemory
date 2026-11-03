"""Local embedding endpoint batch limits must reach the M3 provider."""

from unittest.mock import Mock

import pytest

from m3.config import load_m3_runtime_config
from m3.index_episodic import build_embedder


@pytest.mark.parametrize("override,expected", [("", 8), ("  embedding:\n    batch_size: 4\n", 4)])
def test_m3_batch_size_fallback_and_override(tmp_path, monkeypatch, override, expected):
    path = tmp_path / "config.yaml"
    path.write_text(
        "embedding:\n  provider: openai\n  api_key: test\n  batch_size: 8\n"
        "m3:\n  embedding_unused: null\n" + override
    )
    assert load_m3_runtime_config(str(path)).embedding_batch_size == expected
    ctor = Mock()
    monkeypatch.setattr("providers.embedder.get_openai_embedder", lambda: (ctor, ctor))
    build_embedder(None, config_path=str(path))
    assert ctor.call_args.kwargs["batch_size"] == expected


@pytest.mark.parametrize("value", ["0", "-1", "true", "1.5", "2049"])
def test_m3_rejects_invalid_batch_size(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(f"embedding:\n  batch_size: {value}\n")
    with pytest.raises(ValueError, match="embedding_batch_size"):
        load_m3_runtime_config(str(path))
