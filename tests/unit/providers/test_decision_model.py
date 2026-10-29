import json
import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from providers.decision import DecisionConfig, DecisionError, JevDecisionClient
from providers.unified_config import OgMemConfig

CONFIG = {"base_url": "http://localhost:56203", "model": "multilingual"}
NOUL = {"q": {"type": "noul", "instructions": "Relevant?"}}
CHOICE = {"q": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}}
ANSWER = {"type": "choice", "choice": "a", "probabilities": {"a": 0.9, "b": 0.1}}


@pytest.fixture
def load_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OGMEM_DECISION_API_KEY", "test-key")

    def load(value):
        path = tmp_path / "config.yaml"
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        return OgMemConfig.load(str(path))

    return load


def test_explicit_opt_in(load_config):
    for value in ({}, {"decision": None}):
        assert load_config(value).decision is None
    with pytest.raises(ValueError):
        load_config({"decision": {}})


@pytest.mark.parametrize("mode", ["eager", "lazy"])
def test_config_is_independent_and_redacts_key(load_config, mode):
    cfg = load_config({"decision": CONFIG, "memory": {"extraction": {"mode": mode}}})
    assert cfg.extraction_mode == mode and cfg.decision.api_key == "test-key"
    assert "test-key" not in repr(cfg)


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "openai"},
        {"base_url": "file:///tmp/x"},
        {"model": ""},
        {"timeout_seconds": True},
        {"timeout_seconds": 0},
        {"api_key": 123},
        {"unknown": 1},
    ],
)
def test_invalid_config(change):
    with pytest.raises(ValueError):
        DecisionConfig.from_mapping({**CONFIG, **change})


@pytest.mark.parametrize("document", ["guide", "reference"])
def test_published_examples_read_the_environment_key(load_config, document):
    root = Path(__file__).resolve().parents[3]
    if document == "guide":
        text = (root / "docs/reference/decision-models.md").read_text()
        example = re.search(r"```yaml\n(.*?)\n```", text, re.S).group(1)
    else:
        text = (root / "config/ogmem.reference.yaml").read_text()
        block = text.split("# decision:\n", 1)[1].split("\n\n", 1)[0]
        example = "decision:\n" + "\n".join(line.removeprefix("# ") for line in block.splitlines())
    assert load_config(example).decision.api_key == "test-key"


@pytest.mark.parametrize(
    "kind,model,answer",
    [
        ("noul", "typesafe-ai/jev", {"type": "noul", "noul": 0.8}),
        ("choice", "laya-rl-agent", ANSWER),
    ],
)
def test_standard_request_and_compatible_response(kind, model, answer):
    transport = Mock(return_value={"model": model, "answers": {"q": answer}})
    client = JevDecisionClient(DecisionConfig(**CONFIG), transport=transport)
    result = client.evaluate({}, NOUL) if kind == "noul" else client.evaluate_choice({}, CHOICE)
    assert transport.call_args.args == (
        {"model": "multilingual", "state": {}, "questions": NOUL if kind == "noul" else CHOICE},
    )
    assert result.answers["q"] == (0.8 if kind == "noul" else ANSWER["probabilities"])
    assert result.choices == ({} if kind == "noul" else {"q": "a"})
    assert client.endpoint == "http://localhost:56203/v1/systemone"


@pytest.mark.parametrize(
    "answers,usage",
    [
        ({}, {}),
        ({"q": {"type": "noul", "noul": float("nan")}}, {}),
        ({"q": {"type": "noul", "noul": 0.8}}, {"truncated": True}),
        ({"q": {"type": "noul", "noul": 0.8}}, {"state_tokens_dropped": 1}),
        ({"q": {"type": "noul", "noul": 0.8}}, {"truncated_questions": ["q"]}),
    ],
)
def test_invalid_or_truncated_response(answers, usage):
    client = JevDecisionClient(
        DecisionConfig(**CONFIG), transport=Mock(return_value={"answers": answers, "usage": usage})
    )
    with pytest.raises(DecisionError):
        client.evaluate({}, NOUL)


@pytest.mark.parametrize("choice", [[], {}, None, 1, "missing", "b"])
def test_invalid_choice(choice):
    response = {"answers": {"q": {**ANSWER, "choice": choice}}}
    client = JevDecisionClient(DecisionConfig(**CONFIG), transport=Mock(return_value=response))
    with pytest.raises(DecisionError):
        client.evaluate_choice({}, CHOICE)


def test_transport_failure_is_safe_and_not_retried():
    transport = Mock(side_effect=TimeoutError("secret upstream data"))
    client = JevDecisionClient(DecisionConfig(**CONFIG), transport=transport)
    with pytest.raises(DecisionError) as caught:
        client.evaluate({}, NOUL)
    transport.assert_called_once()
    assert "secret" not in str(caught.value) and caught.value.error_type == "TimeoutError"
