import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.models import RequestContext
from providers.token_tracker import TokenTracker
from core.model_stage import run_model_call
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.server.test_extraction_batch_recovery import candidate
from tests.unit.server.test_lifecycle_batch_recovery import real_archives, seed


@pytest.mark.parametrize("entry", ["prepare_compaction", "compact"])
@pytest.mark.parametrize("agent_id", ["", "main"])
def test_compaction_attributes_extraction_and_parallel_summary(entry, agent_id):
    ctx = RequestContext("compact-account", "reader", agent_id, "compact-session", "usage-proof")
    events, calls = [], []
    tracker = TokenTracker("normal-test-model")
    tracker.set_usage_store(SimpleNamespace(record=events.append))

    class SummaryModel:
        def complete_json(self, prompt, schema):
            calls.append(("summary", threading.current_thread().name))
            tracker.record_llm(21, 7)
            return {"facts_and_details": "Nora repaired the receiver."}

    def extract(**kwargs):
        def complete():
            calls.append(("extraction", threading.current_thread().name))
            tracker.record_llm(31, 11)
            return [replace(candidate(), provenance_ids=[
                f"prov:1:archive:{kwargs['archive_id']}:msg-one"])]
        return run_model_call(complete)

    service, _, manager = real_archives(MemoryFS(), ctx, extract=extract)
    del service._generate_summary_candidate
    service.get_llm = lambda: SummaryModel()
    seed(manager, ctx)
    result = getattr(service, entry)({"sessionId": ctx.session_id, "_ctx": ctx})

    assert result.get("ok", True) is True
    assert result.get("prepareToken") if entry == "prepare_compaction" else result["compacted"]
    expected_owner = "agent:main" if agent_id else "user:reader"
    assert len(events) == 2
    assert all(list(event["allocations"]) == [(ctx.account_id, expected_owner)] for event in events)
    assert sorted((event["allocations"][(ctx.account_id, expected_owner)]["input_tokens"],
                   event["allocations"][(ctx.account_id, expected_owner)]["output_tokens"])
                  for event in events) == [(21, 7), (31, 11)]
    assert any(kind == "summary" and name.startswith("session-summary") for kind, name in calls)
    assert any(kind == "extraction" and name.startswith("model-stage") for kind, name in calls)
