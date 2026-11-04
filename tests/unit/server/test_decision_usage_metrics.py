from types import SimpleNamespace
from unittest.mock import Mock

from providers.decision import DecisionConfig
from providers.decision.usage import DecisionTokenTracker
from providers.token_tracker import TokenTracker, usage_scope
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService


def test_decision_usage_is_separate_scoped_and_included_once():
    svc = MemoryService(OgMemConfig())
    decision = DecisionTokenTracker("jev")
    llm = TokenTracker("generator")
    embedding = TokenTracker("embedder")
    with usage_scope("a", "user:alice"):
        decision.record_response({"usage": {"input_tokens": 320, "output_tokens": 20}})
        decision.record_response({}, failed=True)
        llm.record_llm(10, 5)
        embedding.record_embed(7)
    with usage_scope("b", "user:bob"):
        decision.record_response({"usage": {"input_tokens": 900, "output_tokens": 100}})
    svc._decision_client = SimpleNamespace(token_tracker=decision)
    svc._llm = SimpleNamespace(token_tracker=llm)
    svc._embedder = SimpleNamespace(token_tracker=embedding)
    metrics = svc.get_model_usage_metrics(days=1, account_id="a", owner_spaces=("user:alice",))
    assert metrics["models"]["decision"] == "jev"
    assert metrics["cumulative"]["decision"] == {
        "input_tokens": 320,
        "output_tokens": 20,
        "total_tokens": 340,
        "calls": 2,
        "failed_calls": 1,
        "missing_usage_calls": 1,
    }
    assert metrics["cumulative"]["llm"]["total_tokens"] == 15
    assert metrics["cumulative"]["total_tokens"] == 362
    assert metrics["today"]["decision_input"] == 320
    assert metrics["today"]["decision_output"] == 20
    assert metrics["today"]["total"] == 362


def test_decision_tracker_is_bound_to_existing_store_once():
    config = OgMemConfig(decision=DecisionConfig(base_url="http://localhost:1", model="jev"))
    svc = MemoryService(config)
    store = Mock()
    svc._get_model_usage_store = Mock(return_value=store)
    client = svc.get_decision_client()
    assert svc.get_decision_client() is client
    with usage_scope("a", "user:alice"):
        client.token_tracker.record_response({"usage": {"input_tokens": 1, "output_tokens": 2}})
    store.record.assert_called_once()
    event = store.record.call_args.args[0]
    assert event["kind"] == "decision"
    assert event["allocations"][("a", "user:alice")]["input_tokens"] == 1


def test_empty_sql_decision_model_uses_configuration():
    config = OgMemConfig(
        decision=DecisionConfig(base_url="http://localhost:1", model="configured-jev")
    )
    svc = MemoryService(config)
    store = Mock()
    store.metrics.return_value = {
        "models": {"llm": "generator", "embedding": "embedder", "decision": ""}
    }
    svc._get_model_usage_store = Mock(return_value=store)
    assert svc.get_model_usage_metrics(account_id="a")["models"]["decision"] == "configured-jev"
