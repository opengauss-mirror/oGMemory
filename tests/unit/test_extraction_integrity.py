"""Extraction must fail instead of committing incomplete model results."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from commit.candidate_pipeline import CandidatePipeline
from core.errors import ValidationError
from core.models import CandidateMemory, RequestContext
from providers.llm.openai_llm import OpenAILLM


@pytest.mark.parametrize("parallel", [False, True])
def test_extractor_failure_does_not_return_partial_success(parallel):
    healthy, failed = Mock(), Mock()
    healthy.extract.return_value = []
    failed.extract.side_effect = RuntimeError("extraction unavailable")
    ctx = RequestContext(account_id="test", user_id="user", agent_id="agent",
                         session_id="session", trace_id="trace")
    with pytest.raises(RuntimeError, match="extraction unavailable"):
        CandidatePipeline([healthy, failed]).extract([], ctx, parallel=parallel)


@pytest.mark.parametrize("parallel", [False, True])
def test_invalid_candidate_does_not_return_valid_subset(parallel):
    extractor = Mock()
    valid = CandidateMemory(category="preference", owner_scope="user", routing_key="tea",
                            abstract="Prefers tea", overview="", content="", confidence=0.9)
    invalid = CandidateMemory(category="profile", owner_scope="user", routing_key="profile",
                              abstract="Invalid identity", overview="", content="", confidence=0.9)
    extractor.extract.return_value = [valid, invalid]
    ctx = RequestContext("test", "user", "agent", "session", "trace")
    with pytest.raises(ValidationError):
        CandidatePipeline([extractor]).extract([], ctx, parallel=parallel)


@pytest.mark.parametrize("method", ["json", "structured_json", "tools", "messages"])
def test_truncation_is_rejected_before_parsing_and_usage_is_retained(monkeypatch, method):
    client = Mock()
    monkeypatch.setattr("providers.llm.openai_llm.OpenAI", Mock(return_value=client))
    llm = OpenAILLM(api_key="test", json_mode=method == "structured_json")
    llm.token_tracker = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content='{"operations": []}', tool_calls=[]),
        )],
        model="test-model",
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
    )
    with pytest.raises(RuntimeError, match="finish_reason=length"):
        if method in {"json", "structured_json"}:
            llm.complete_json("extract", {"type": "object"})
        elif method == "tools":
            llm.complete_with_tools("extract", [], tool_choice="auto")
        else:
            llm.complete_with_tools_messages([{"role": "user", "content": "extract"}])
    assert client.chat.completions.create.call_count == 1
    llm.token_tracker.record_llm.assert_called_once_with(
        20, 10, 0, 0, usage_known=True, model="test-model",
    )
