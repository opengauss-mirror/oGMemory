from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from core.models import RequestContext
from extraction.tools import Extractor, _PreparedSpan, _Span
from providers.token_tracker import TokenTracker, request_usage_scope
from server.memory_service import _run_with_usage_scope
from skills.sedimentation import _attribute_skill_usage


def test_extraction_pool_preserves_each_callers_scope():
    events = []
    tracker = TokenTracker("test-llm")
    tracker.set_usage_store(SimpleNamespace(record=events.append))

    def extract(account):
        ctx = RequestContext(
            account_id=account, user_id="alice", agent_id="", session_id="s", trace_id="t"
        )
        extractor = Extractor(Mock())
        spans = [_Span(start=i, end=i) for i in range(2)]
        extractor._prepare_unique_spans = Mock(
            return_value=[_PreparedSpan(span=s, prompt="", prompt_sha="") for s in spans]
        )

        def structure(*args, **kwargs):
            tracker.record_llm(10, 3)
            return []

        extractor._structure_prepared_span = structure
        with request_usage_scope(ctx):
            extractor._structure_spans(spans, [], "en", None, "", "", ctx)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(extract, ["a", "b"]))
    scopes = [next(iter(event["allocations"])) for event in events]
    assert scopes.count(("a", "user:alice")) == 4
    assert scopes.count(("b", "user:alice")) == 4


def test_background_and_skill_threads_attribute_agent_owner():
    events = []
    tracker = TokenTracker("test-llm")
    tracker.set_usage_store(SimpleNamespace(record=events.append))
    ctx = RequestContext(
        account_id="a", user_id="alice", agent_id="helper", session_id="s", trace_id="t"
    )

    @_attribute_skill_usage
    def skill(self, turn):
        tracker.record_llm(7, 2)

    with ThreadPoolExecutor(1) as pool:
        pool.submit(_run_with_usage_scope, ctx, tracker.record_llm, 10, 3).result()
        pool.submit(skill, None, {"account_id": "b", "agent_id": "worker"}).result()
    assert [next(iter(e["allocations"])) for e in events] == [
        ("a", "agent:helper"),
        ("b", "agent:worker"),
    ]
