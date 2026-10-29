"""Repeated Phase 1 ranges must not multiply eager extraction or lose context."""

from datetime import datetime
from threading import get_ident
from unittest.mock import Mock

import pytest

from core.models import RequestContext
from extraction.prefetch import PrefetchResult
from extraction.tools import Extractor, _Span

CTX = RequestContext(account_id="account", user_id="user", agent_id="agent", session_id="session", trace_id="trace")
TIME = datetime(2026, 1, 1)
MESSAGES = [{"id": "source-1", "role": "user", "content": "I like coffee and finished a workshop."}]


def make_extractor(*, reconcile=False):
    llm = Mock()
    llm.complete_with_tools.return_value = []
    llm.complete_json.return_value = {"operations": []}
    extractor = Extractor(llm, source_occurrence_reconciliation_enabled=reconcile)
    extractor._prefetcher = Mock()
    extractor._prefetcher.prefetch_for_span.side_effect = lambda categories, ctx, **kwargs: PrefetchResult(
        messages=[f"Existing {category} context" for category in categories],
    )
    return extractor, llm


@pytest.mark.parametrize("reconcile", [False, True])
def test_identical_ranges_prefetch_union_once_before_native_attempts(reconcile):
    extractor, llm = make_extractor(reconcile=reconcile)
    spans = [
        _Span(0, 0, "preference", ["preferences"], ["source-1"]),
        _Span(0, 0, "workshop", ["event"], ["source-1"]),
        _Span(0, 0, "both", ["event", "preferences"], ["source-1"]),
    ]
    result = extractor._structure_spans(spans, MESSAGES, "English", TIME, "", "", CTX)
    assert result == []
    assert extractor._prefetcher.prefetch_for_span.call_count == 1
    assert extractor._prefetcher.prefetch_for_span.call_args.args == (["preferences", "event"], CTX)
    assert llm.complete_with_tools.call_count == 2
    assert [call.kwargs["temperature"] for call in llm.complete_with_tools.call_args_list] == [None, 0]
    for call in llm.complete_with_tools.call_args_list:
        assert "Existing preferences context" in call.kwargs["prompt"]
        assert "Existing event context" in call.kwargs["prompt"]
        assert MESSAGES[0]["content"] in call.kwargs["prompt"]
    assert llm.complete_json.call_count == 1  # One shared JSON fallback after both failures.


def test_range_union_retains_source_ids_and_custom_template():
    extractor, _ = make_extractor()
    messages = [*MESSAGES, {"id": "source-2", "role": "user", "content": "The workshop was useful."}]
    extractor._prompt_manager.render = Mock(side_effect=lambda section, key, **kwargs: f"CUSTOM {section} {key}")
    prepared = extractor._prepare_unique_spans([
        _Span(0, 1, "first", ["preferences"], ["source-1"]),
        _Span(0, 1, "second", ["event"], ["source-1", "source-2"]),
    ], messages, "English", TIME, "summary", "stats", CTX)
    assert len(prepared) == 1
    span = prepared[0].span
    assert span.categories == ["preferences", "event"]
    assert span.message_ids == ["source-1", "source-2"]
    assert "first" in span.reason and "second" in span.reason
    assert span.focused_messages == messages
    assert span.safe_user_text == "\n".join(message["content"] for message in messages)
    assert "CUSTOM extraction" in prepared[0].prompt
    assert MESSAGES[0]["content"] in prepared[0].source_context


def test_distinct_overlapping_ranges_keep_their_own_provenance():
    extractor, _ = make_extractor(reconcile=True)
    messages = [*MESSAGES, {"id": "source-2", "role": "user", "content": "I finished another workshop."}]
    prepared = extractor._prepare_unique_spans([
        _Span(0, 0, categories=["event"]),
        _Span(0, 1, categories=["event"]),
    ], messages, "English", TIME, "", "", CTX)
    assert len(prepared) == 2
    assert [item.span.message_ids for item in prepared] == [["source-1"], ["source-1", "source-2"]]
    assert [(item.span.start, item.span.end) for item in prepared] == [(0, 0), (0, 1)]


@pytest.mark.parametrize("metadata", [
    {"message_ids": ["another-source"]},
    {"prefetch_result": PrefetchResult(read_versions={"memory://other": 7})},
    {"successful_read_versions": [{"memory://other": 7}]},
    {"occurrence_source": {"known_origin": True}},
    {"focused_messages": [{"role": "user", "content": "another source"}]},
])
def test_inconsistent_source_or_prepared_authority_is_not_coalesced(metadata):
    extractor, _ = make_extractor()
    prepared = extractor._prepare_unique_spans([
        _Span(0, 0, categories=["preferences"], message_ids=["source-1"]),
        _Span(0, 0, categories=["event"], **metadata),
    ], MESSAGES, "English", TIME, "", "", CTX)
    assert len(prepared) == 2
    assert extractor._prefetcher.prefetch_for_span.call_count == 2


def test_thread_bound_storage_stays_on_owner_for_all_distinct_spans():
    extractor, llm = make_extractor(reconcile=True)
    owner = get_ident()

    class ThreadBoundStorage:
        def requires_thread_bound_access(self):
            return True

    extractor._fs = ThreadBoundStorage()

    def assert_owner(**kwargs):
        assert get_ident() == owner
        return []

    llm.complete_with_tools.side_effect = assert_owner
    messages = [*MESSAGES, {"id": "source-2", "role": "user", "content": "I finished another workshop."}]
    assert extractor._structure_spans([
        _Span(0, 0, categories=["event"]),
        _Span(1, 1, categories=["event"]),
    ], messages, "English", TIME, "", "", CTX) == []
    assert llm.complete_with_tools.call_count == 4
