"""Expiry terminates extraction instead of creating more recovery work."""

from unittest.mock import Mock

import pytest

import core.operation_deadline as deadline
from core.errors import OperationDeadlineExceededError, ToolCallProtocolError
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.tools import _PreparedSpan, _Span
from tests.unit.extraction.test_lazy_fallback_budget import setup
from tests.unit.extraction.test_span_amplification import CTX, MESSAGES, TIME, make_extractor


def test_native_expiry_stops_second_attempt_and_json_fallback(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(deadline, "monotonic", lambda: clock[0])
    extractor, llm = make_extractor()

    def late_native(**kwargs):
        clock[0] += 2
        raise ToolCallProtocolError("invalid late output")

    llm.complete_with_tools.side_effect = late_native
    with deadline.operation_deadline(1), pytest.raises(OperationDeadlineExceededError):
        extractor._structure_prepared_span_dual(_PreparedSpan(_Span(0, 0), "source", "hash"), CTX, None)
    assert llm.complete_with_tools.call_count == 1
    llm.complete_json.assert_not_called()


def test_prefetch_deadline_is_not_swallowed():
    extractor, llm = make_extractor()
    extractor._prefetcher.prefetch_for_span.side_effect = OperationDeadlineExceededError("expired")
    with pytest.raises(OperationDeadlineExceededError):
        extractor._structure_spans([_Span(0, 0, categories=["event"])], MESSAGES, "en", TIME, "", "", CTX)
    llm.complete_with_tools.assert_not_called()


def test_lazy_deadline_does_not_start_eager_fallback(monkeypatch):
    extractor, llm, span, ctx, _ = setup(monkeypatch, large=False)
    monkeypatch.setattr(ExtractionReActLoop, "run", Mock(side_effect=OperationDeadlineExceededError("expired")))
    with pytest.raises(OperationDeadlineExceededError):
        extractor._structure_span_lazy(span, "source", ctx)
    assert llm.calls == []


def test_lazy_tool_deadline_is_not_converted_to_tool_error():
    loop = object.__new__(ExtractionReActLoop)
    loop._fs = Mock()
    loop._fs.read_node.side_effect = OperationDeadlineExceededError("expired")
    with pytest.raises(OperationDeadlineExceededError):
        loop._execute_tool("read", {"uri": "memory://source"}, CTX)
