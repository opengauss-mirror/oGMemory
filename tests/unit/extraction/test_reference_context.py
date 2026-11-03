"""Incremental extraction retains topic context without turning it into evidence."""

from datetime import datetime

from core.models import RequestContext
from extraction.tools import Extractor, _Span


def test_incremental_project_context_reaches_native_and_fallback_prompts():
    extractor = Extractor(object())
    ctx = RequestContext(account_id="local", user_id="admin", agent_id="test",
                         session_id="reference-test", trace_id="reference-trace")
    messages = [{"id": "requirement", "role": "user", "content": "数据库要使用openGauss"}]
    summary = "用户正在设计 OGMemory Gateway；OpenViking 是参考产品。"
    span = _Span(0, 0, categories=["preference"])
    prompt = extractor._build_span_prompt(
        span, messages, "zh", datetime(2026, 10, 19), summary, "", ctx,
    )
    for value in (prompt, span.fallback_source_prompt):
        assert summary in value
        assert "reference resolution only" in value
        assert "数据库要使用openGauss" in value
    # Stored evidence still consists of actual turns, not a saved/model summary.
    assert span.focused_messages == messages
    assert summary not in extractor._format_conversation(span.focused_messages)


def test_reference_context_is_bounded_and_does_not_truncate_current_source():
    extractor = Extractor(object())
    source = "user: 当前系统客户端只改模型地址就能接入。"
    prompt = extractor._build_prompt("zh", source, "A" * 8000 + "OLD_TAIL")
    assert "OLD_TAIL" not in prompt
    assert "Prior session context truncated" in prompt
    assert prompt.endswith(source)


def test_empty_prior_context_does_not_add_reference_block():
    prompt = Extractor(object())._build_prompt("en", "user: I prefer tea.")
    assert "## Prior session context" not in prompt
