"""M3 must not silently replace a configured model timeout with SDK defaults."""
from unittest.mock import Mock

import pytest
import yaml

from m3.config import load_m3_runtime_config
from m3.llm_factory import build_m3_llm


def write_config(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_timeout_inherits_global_and_allows_role_override(tmp_path):
    config = write_config(tmp_path, {
        "llm": {"request_timeout_seconds": 300},
        "m3": {"episodic_llm": {"request_timeout_seconds": 480}},
    })
    runtime = load_m3_runtime_config(config)
    assert runtime.episodic_llm.request_timeout_seconds == 480
    assert runtime.qa_llm.request_timeout_seconds == 300
    assert runtime.judge_llm.request_timeout_seconds == 300


def test_timeout_preserves_existing_default(tmp_path):
    runtime = load_m3_runtime_config(write_config(tmp_path, {}))
    for role in ("episodic_llm", "qa_llm", "judge_llm"):
        assert getattr(runtime, role).request_timeout_seconds == 120


@pytest.mark.parametrize("value", [0, -1, True, False, "oops", float("inf"), float("nan")])
def test_invalid_timeout_rejected(tmp_path, value):
    config = write_config(tmp_path, {"llm": {"request_timeout_seconds": value}})
    with pytest.raises(ValueError, match="positive finite number"):
        load_m3_runtime_config(config)


def test_timeout_expands_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("M3_TEST_TIMEOUT", "360.5")
    config = write_config(tmp_path, {"llm": {"request_timeout_seconds": "${M3_TEST_TIMEOUT}"}})
    assert load_m3_runtime_config(config).qa_llm.request_timeout_seconds == 360.5


@pytest.mark.parametrize("role", ["episodic_llm", "qa_llm", "judge_llm"])
@pytest.mark.parametrize("provider", ["openai", "vllm", "minimax"])
def test_factory_forwards_role_timeout(tmp_path, monkeypatch, role, provider):
    constructor = Mock()
    monkeypatch.setattr("providers.llm.get_openai_llm", lambda: (constructor, None))
    monkeypatch.setattr("providers.llm.get_minimax_llm", lambda: constructor)
    config = write_config(tmp_path, {
        "llm": {
            "provider": provider, "api_key": "unit-test-only",
            "model": "local-model", "base_url": "http://localhost.invalid/v1",
            "request_timeout_seconds": 300,
        },
        "m3": {role: {"request_timeout_seconds": 480}},
    })
    build_m3_llm(config_path=config, config_role=role)
    assert constructor.call_args.kwargs["request_timeout_seconds"] == 480


def test_minimax_passes_timeout_to_openai_base(monkeypatch):
    from providers.llm.minimax_llm import MiniMaxLLM
    init = Mock(return_value=None)
    monkeypatch.setattr("providers.llm.openai_llm.OpenAILLM.__init__", init)
    MiniMaxLLM(api_key="unit-test-only", request_timeout_seconds=360)
    assert init.call_args.kwargs["request_timeout_seconds"] == 360
