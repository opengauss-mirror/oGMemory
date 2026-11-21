"""Physical model events must correlate with their existing usage ledger rows."""

import pytest

from providers.decision.usage import DecisionTokenTracker
from providers.llm_observation import capture_llm_events, observe_call
from providers.token_tracker import TokenTracker, usage_scope


class RecordingStore:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


@pytest.mark.parametrize("kind", ["decision", "llm", "failure"])
def test_usage_event_identity_matches_the_ledger_for_success_and_failure(kind):
    tracker = DecisionTokenTracker("jev") if kind == "decision" else TokenTracker("lm")
    store = RecordingStore()
    tracker.set_usage_store(store)

    def call():
        if kind == "decision":
            tracker.record_response({"usage": {"input_tokens": 10, "output_tokens": 2}})
        elif kind == "failure":
            tracker.record_failure("llm")
        else:
            tracker.record_llm(10, 2)

    with usage_scope("acme", "user:alice"), capture_llm_events() as events:
        observe_call("extraction", call)
    physical = [e for e in events if e["event"] == "usage"]
    assert len(physical) == len(store.events) == 1
    assert physical[0]["provider_event_id"] == store.events[0]["id"]
    assert store.events[0]["allocations"].keys() == {("acme", "user:alice")}


def test_decision_missing_usage_is_observed_as_unknown_not_zero_cost():
    tracker = DecisionTokenTracker("jev")
    with capture_llm_events() as events:
        observe_call("extraction", lambda: tracker.record_response(None, failed=True))
    usage = [e for e in events if e["event"] == "usage"]
    assert len(usage) == 1
    assert usage[0]["usage_known"] is False and usage[0]["total_tokens"] is None
    assert usage[0]["failed"] is True
