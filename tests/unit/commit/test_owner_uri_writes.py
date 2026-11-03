"""Exact principal namespaces through public memory writes and durable retries."""

from copy import deepcopy
from dataclasses import replace

import pytest

from commit.write_receipts import RECEIPTS_KEY, WriteNotStartedError, operation_digest
from core.errors import ValidationError
from core.models import RequestContext
from tests.unit.commit.test_memory_identity import MemoryFS, api, memory
from tests.unit.commit.test_occurrence_replay import occurrence
from tests.unit.commit.test_profile_merge import profile
from tests.unit.server.test_extraction_batch_recovery import NoModel


@pytest.mark.parametrize("field,category", [
    ("account_id", "event"), ("account_id", "case"), ("user_id", "event"), ("agent_id", "case"),
])
def test_case_distinct_owners_have_separate_draft_replays(category, field):
    fs = MemoryFS()
    upper = RequestContext("Acme", "Nora", "Agent", "session", "trace")
    lower = replace(upper, **{field: getattr(upper, field).lower()})
    candidate = occurrence(category)
    targets = []
    for ctx in (upper, lower):
        created = api(fs, NoModel()).write_memory(candidate, ctx)
        replay = api(fs, NoModel()).write_memory(candidate, replace(ctx, session_id="retry"))
        assert created["action"] == "create" and replay["action"] == "skip"
        assert created["target_uri"] == replay["target_uri"]
        targets.append(created["target_uri"])
    assert targets[0] != targets[1] and len(fs.nodes) == 2
    assert operation_digest(candidate, upper) != operation_digest(candidate, lower)


@pytest.mark.parametrize("category", ["profile", "entity", "preference", "pattern", "event", "case", "skill", "tool"])
def test_original_case_durable_receipt_survives_new_writer_and_hard_cas(category):
    fs = MemoryFS()
    ctx = RequestContext("Acme", "Nora", "Agent", "session", "trace")
    first = profile("identity", "Initial supported fact.") if category == "profile" else memory(
        "supported_topic", category=category, content="Initial supported fact.")
    first = replace(first, owner_scope="agent" if category in {"case", "skill", "tool"} else "user")
    uri = api(fs, NoModel()).write_memory(first, ctx)["target_uri"]
    accepted = replace(first, target_uri=uri, expected_version=1,
                       content="Initial supported fact. Added fact.", write_operation_id="accepted-operation")
    assert api(fs, NoModel()).write_memory(accepted, ctx)["action"] == "merge"
    later = replace(accepted, expected_version=2, write_operation_id="later-operation",
                    content="Initial supported fact. Added fact. Later fact.")
    assert api(fs, NoModel()).write_memory(later, ctx)["action"] == "merge"
    before = deepcopy(fs.nodes)
    assert api(fs, NoModel()).write_memory(accepted, replace(ctx, trace_id="retry"))["action"] == "skip"
    with pytest.raises(WriteNotStartedError):
        api(fs, NoModel()).write_memory(replace(accepted, write_operation_id="stale-new-operation"), ctx)
    assert fs.nodes == before
    assert fs.read_node(uri, ctx).metadata["version"] == 3
    assert set(fs.read_node(uri, ctx).metadata[RECEIPTS_KEY]) == {"accepted-operation", "later-operation"}


@pytest.mark.parametrize("field,category", [("account_id", "event"), ("user_id", "event"), ("agent_id", "case")])
@pytest.mark.parametrize("unsafe", ["../other", "other\\owner", "other%2Fowner", "other\0owner"])
def test_unsupported_owner_is_rejected_before_public_write_mutates(field, category, unsafe):
    fs = MemoryFS()
    ctx = replace(RequestContext("acme", "nora", "agent", "session", "trace"), **{field: unsafe})
    candidate = occurrence(category)
    for operation in ("", "accepted-operation"):
        with pytest.raises((ValueError, ValidationError, WriteNotStartedError)):
            api(fs, NoModel()).write_memory(replace(candidate, write_operation_id=operation), ctx)
        assert fs.nodes == {}
