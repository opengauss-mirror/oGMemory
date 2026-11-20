import threading
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


def test_first_lazy_write_and_retrieval_share_decision_client_without_deadlock(monkeypatch):
    svc = MemoryService(OgMemConfig(
        provider="mock", extraction_mode="lazy", retrieval_mode="agfs",
        decision=DecisionConfig(base_url="http://localhost:1", model="jev"),
        retrieval_decision_rerank_enabled=True,
    ))
    write_started, retrieval_waiting = threading.Event(), threading.Event()
    init_lock = svc._init_lock

    class ObservedInitLock:
        def __enter__(self):
            if threading.current_thread().name == "first-decision":
                retrieval_waiting.set()
            return init_lock.__enter__()

        def __exit__(self, *args):
            return init_lock.__exit__(*args)

    def context_fs():
        write_started.set()
        assert retrieval_waiting.wait(3)
        return Mock()

    monkeypatch.setattr(svc, "_init_lock", ObservedInitLock())
    monkeypatch.setattr(svc, "_get_context_fs", context_fs)
    monkeypatch.setattr(svc, "_get_outbox_store", lambda fs: Mock())
    monkeypatch.setattr(svc, "_start_outbox_worker", lambda: None)
    results, errors = {}, []

    def invoke(key, callback):
        try:
            results[key] = callback()
        except Exception as exc:
            errors.append(exc)

    writer = threading.Thread(target=invoke, args=("write", svc.get_write_api), daemon=True)
    reader = threading.Thread(target=invoke, args=("decision", svc.get_decision_client),
                              name="first-decision", daemon=True)
    writer.start()
    assert write_started.wait(3)
    reader.start()
    writer.join(3)
    reader.join(3)
    assert not writer.is_alive() and not reader.is_alive(), "first initialization deadlocked"
    assert not errors
    assert results["write"]._decision_client is results["decision"]
    assert svc._get_model_usage_store() is None
    with usage_scope("a", "user:alice"):
        results["decision"].token_tracker.record_response({"usage": {"input_tokens": 1, "output_tokens": 2}})
    assert results["decision"].token_tracker.snapshot().input_tokens == 1
