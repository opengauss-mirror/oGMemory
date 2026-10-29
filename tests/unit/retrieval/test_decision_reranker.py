from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from providers.decision import DecisionConfig, DecisionError
from providers.unified_config import OgMemConfig
from retrieval.decision_reranker import rerank_evidence
from server.memory_service import MemoryService


def items():
    return [
        {
            "uri": "ctx://a/one",
            "category": "entity",
            "content": "one",
            "score": 0.9,
            "metadata": {"unit_key": "one"},
        },
        {"uri": "ctx://a/code", "category": "code", "content": "code", "score": 0.99},
        {"uri": "ctx://a/two", "category": "event", "content": "two", "score": 0.7},
    ]


def client(scores):
    obj = Mock()
    obj.evaluate.return_value = SimpleNamespace(
        answers={f"candidate_{i}": p for i, p in enumerate(scores)}
    )
    return obj


def test_rerank_only_changes_ordinary_order_not_scores_or_code():
    original = items()
    obj = client([0.2, 0.8])
    ranked, stats = rerank_evidence("query", original, obj)
    assert ranked == [original[2], original[1], original[0]]
    assert ranked[0] is original[2] and original[0]["score"] == 0.9
    assert stats["fallback"] is False and stats["candidate_count"] == 2
    state, questions = obj.evaluate.call_args.args
    assert len(questions) == 2 and all(q["type"] == "noul" for q in questions.values())
    assert "code" not in str(state)


@pytest.mark.parametrize("scores", [[0.5, 0.5], [float("nan"), 0.5], [0.8]])
def test_stable_ties_and_bad_scores_preserve_native_order(scores):
    original = items()
    ranked, stats = rerank_evidence("query", original, client(scores))
    assert ranked == original
    assert stats["fallback"] is (scores != [0.5, 0.5])


def test_error_preserves_native_order_without_exposing_exception():
    obj = Mock()
    error = DecisionError("secret source")
    error.status_code = 400
    obj.evaluate.side_effect = error
    original = items()
    ranked, stats = rerank_evidence("query", original, obj)
    assert ranked == original and stats["fallback"] is True
    assert "secret" not in str(stats) and stats["status_code"] == 400


def test_empty_and_code_only_have_no_model_calls():
    obj = Mock()
    for original in ([], [items()[1]]):
        ranked, stats = rerank_evidence("q", original, obj)
        assert ranked == original
    obj.evaluate.assert_not_called()


def test_configured_model_is_shared_and_eager_can_rerank():
    cfg = OgMemConfig(
        decision=DecisionConfig(base_url="http://localhost:1", model="multilingual"),
        extraction_mode="eager",
        retrieval_decision_rerank_enabled=True,
    )
    svc = MemoryService(cfg)
    assert svc.get_decision_client() is svc.get_decision_client()
    svc._decision_client = client([0.2, 0.8])
    ranked, stats = svc._rerank_decision_evidence("q", items())
    assert ranked[0]["uri"] == "ctx://a/two"


def test_unconfigured_decision_does_not_construct_client():
    svc = MemoryService(OgMemConfig())
    assert svc.get_decision_client() is None
    original = items()
    ranked, stats = svc._rerank_decision_evidence("q", original)
    assert ranked is original and stats is None


def test_compose_scores_only_successfully_hydrated_acl_checked_memories():
    from core.models import RequestContext
    from session.models import SessionWindowState

    cfg = OgMemConfig(
        decision=DecisionConfig(base_url="http://localhost:1", model="multilingual"),
        retrieval_decision_rerank_enabled=True,
    )
    svc = MemoryService(cfg)
    svc._get_context_fs = Mock(return_value=None)
    svc._read_profile = Mock(return_value="")
    svc._collect_archives = Mock(return_value=([], []))
    svc._get_session_state = Mock(return_value=SessionWindowState())
    original = [i for i in items() if i["category"] != "code"]
    denied = {"uri": "ctx://a/denied", "category": "entity", "abstract": "SECRET", "score": 0.8}
    svc._search_working_set = Mock(return_value=(original + [denied], [], {}))

    def read(uri, ctx, **kwargs):
        if uri == denied["uri"]:
            raise PermissionError("denied")
        return SimpleNamespace(
            abstract="hydrated", overview="", content_excerpt=uri, category="entity", metadata={}
        )

    reader = Mock()
    reader.read_memory.side_effect = read
    svc.get_read_api = Mock(return_value=reader)
    svc._decision_client = client([0.2, 0.8])
    ctx = RequestContext("a", "u", "agent", "session", "trace")
    result = svc.compose(
        {"query": "Which memory?", "sessionId": "session", "_ctx": ctx, "messages": []}
    )
    state, _ = svc._decision_client.evaluate.call_args.args
    assert "SECRET" not in str(state) and "denied" not in str(state)
    assert result["stats"]["decision_rerank"]["candidate_count"] == 2
    assert reader.read_memory.call_count == 3


def test_expired_deadline_never_starts_rerank():
    from core.errors import OperationDeadlineExceededError
    from core.operation_deadline import _deadline

    obj = Mock()
    token = _deadline.set(0)
    try:
        with pytest.raises(OperationDeadlineExceededError):
            rerank_evidence("query", items(), obj)
    finally:
        _deadline.reset(token)
    obj.evaluate.assert_not_called()


def test_configured_model_does_not_enable_rerank_by_itself():
    svc = MemoryService(
        OgMemConfig(decision=DecisionConfig(base_url="http://localhost:1", model="test"))
    )
    svc.get_decision_client = Mock(side_effect=AssertionError("disabled rerank constructed client"))
    original = items()
    ranked, stats = svc._rerank_decision_evidence("q", original)
    assert ranked is original and stats is None


def test_rerank_requires_model():
    with pytest.raises(ValueError, match="decision"):
        OgMemConfig(retrieval_decision_rerank_enabled=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_rerank_yaml_switch(tmp_path, enabled):
    import json

    path = tmp_path / "ogmem.yaml"
    path.write_text(
        json.dumps(
            {
                "decision": {"base_url": "http://localhost:1", "model": "test"},
                "retrieval": {"decision_rerank_enabled": enabled},
            }
        )
    )
    assert OgMemConfig.load(str(path)).retrieval_decision_rerank_enabled is enabled


def test_complete_reference_template_loads():
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[3] / "config/ogmem.reference.yaml"
    raw = yaml.safe_load(path.read_text())
    assert raw["retrieval"]["decision_rerank_enabled"] is False
    assert "decision_rerank_enabled" not in raw["features"]
    config = OgMemConfig.load(str(path))
    assert config.retrieval_decision_rerank_enabled is False
    assert config.decision is None


def test_usage_persistence_failure_is_not_provider_fallback():
    from providers.decision import JevDecisionClient
    from providers.token_tracker import UsagePersistenceError

    obj = JevDecisionClient(
        DecisionConfig(base_url="http://example.invalid", model="test"),
        transport=Mock(
            return_value={
                "answers": {
                    "candidate_0": {"type": "noul", "noul": 0.5},
                    "candidate_1": {"type": "noul", "noul": 0.5},
                },
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        ),
    )
    store = Mock()
    store.record.side_effect = RuntimeError("synthetic ledger failure")
    obj.token_tracker.set_usage_store(store)
    with pytest.raises(UsagePersistenceError):
        rerank_evidence("q", items(), obj)
    store.record.assert_called_once()
