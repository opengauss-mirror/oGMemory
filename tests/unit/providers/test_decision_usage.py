from unittest.mock import Mock

import pytest

from core.models import RequestContext
from providers.decision import DecisionConfig, DecisionError, JevDecisionClient
from providers.token_tracker import request_usage_scope


@pytest.mark.parametrize(
    "usage,expected,missing",
    [
        ({"input_tokens": 320, "output_tokens": 20}, (320, 20), 0),
        (None, (0, 0), 1),
        ({"input_tokens": -1, "output_tokens": True}, (0, 0), 1),
        ({"input_tokens": 5}, (5, 0), 1),
    ],
)
def test_decision_usage_is_scoped_and_persisted_once(usage, expected, missing):
    raw = {"answers": {"q": {"type": "noul", "noul": 0.8}}}
    if usage is not None:
        raw["usage"] = usage
    client = JevDecisionClient(
        DecisionConfig(base_url="http://localhost", model="jev"), transport=Mock(return_value=raw)
    )
    store = Mock()
    client.token_tracker.set_usage_store(store)
    ctx = RequestContext("a", "u", "agent", "s", "t")
    with request_usage_scope(ctx):
        client.evaluate({}, {"q": {"type": "noul", "instructions": "Relevant?"}})
    snapshot = client.token_tracker.scoped_snapshot("a", (ctx.agent_space_name(),))
    assert (snapshot.input_tokens, snapshot.output_tokens) == expected
    assert snapshot.llm_calls == 1
    assert client.token_tracker.scoped_snapshot("other").llm_calls == 0
    assert client.token_tracker.decision_status("a", (ctx.agent_space_name(),)) == {
        "failed_calls": 0,
        "missing_usage_calls": missing,
    }
    store.record.assert_called_once()
    event = store.record.call_args.args[0]
    assert event["kind"] == "decision" and event["model"] == "jev"


def test_failure_and_no_request_accounting():
    client = JevDecisionClient(
        DecisionConfig(base_url="http://localhost", model="jev"),
        transport=Mock(side_effect=TimeoutError()),
    )
    client.evaluate({}, {})
    assert client.token_tracker.snapshot().llm_calls == 0
    with pytest.raises(DecisionError):
        client.evaluate({}, {"q": {"type": "noul", "instructions": "Relevant?"}})
    assert client.token_tracker.snapshot().llm_calls == 1
    assert client.token_tracker.decision_status() == {"failed_calls": 1, "missing_usage_calls": 1}


def test_protocol_failure_preserves_reported_tokens():
    client = JevDecisionClient(
        DecisionConfig(base_url="http://localhost", model="jev"),
        transport=Mock(
            return_value={"answers": {}, "usage": {"input_tokens": 9, "output_tokens": 1}}
        ),
    )
    with pytest.raises(DecisionError):
        client.evaluate({}, {"q": {"type": "noul", "instructions": "Relevant?"}})
    assert client.token_tracker.snapshot().input_tokens == 9
    assert client.token_tracker.decision_status() == {"failed_calls": 1, "missing_usage_calls": 0}
