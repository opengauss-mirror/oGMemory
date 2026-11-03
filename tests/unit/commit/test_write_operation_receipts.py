"""An exact committed operation can recover; a new stale update still cannot."""

from dataclasses import replace

import pytest

from commit.write_receipts import RECEIPTS_KEY, WriteNotStartedError
from core.models import CandidateMemory, RequestContext
from service.api import MemoryWriteAPI
from tests.unit.commit.test_memory_identity import MemoryFS, memory
from tests.unit.commit.test_profile_merge import profile
from tests.unit.server.test_extraction_batch_recovery import NoModel


@pytest.fixture
def ctx():
    return RequestContext(account_id="receipt-proof", user_id="reader", agent_id="agent",
                          session_id="source-session", trace_id="trace")


@pytest.mark.parametrize("category", ["skill", "tool", "case"])
@pytest.mark.parametrize("operation_id", ["", "accepted-operation"])
@pytest.mark.parametrize("bound", [False, True])
def test_missing_agent_skips_without_storage_access(ctx, category, operation_id, bound):
    from unittest.mock import Mock

    fs = Mock(spec=MemoryFS)
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    accepted = CandidateMemory(
        category=category, owner_scope="agent", routing_key="topic",
        content="Supported fact.", abstract="Supported fact.", overview="Supported fact.",
        confidence=.9, write_operation_id=operation_id,
        target_uri=f"ctx://receipt-proof/agents/agent/memories/{category}s/topic" if bound else "",
        expected_version=1 if bound else None,
    )

    result = writer.write_memory(accepted, replace(ctx, agent_id=""))

    assert result["action"] == "skip"
    assert result["target_uri"] == ""
    assert not fs.mock_calls


@pytest.mark.parametrize("category", ["profile", "entity", "preference", "event", "case"])
def test_old_version_skips_only_when_exact_operation_was_committed(ctx, category):
    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    if category == "profile":
        original = profile("identity", "Initial supported fact.")
    else:
        original = memory("topic", category=category, content="Initial supported fact.")
        if category == "case":
            original = replace(original, owner_scope="agent")
    uri = writer.write_memory(original, ctx)["target_uri"]
    accepted = replace(original, target_uri=uri, expected_version=1,
                       abstract="Initial supported fact. Added fact.",
                       overview="Initial supported fact. Added fact.",
                       content="Initial supported fact. Added fact.", write_operation_id="accepted-operation")
    assert writer.write_memory(accepted, ctx)["action"] == "merge"
    assert fs.read_node(uri, ctx).metadata["version"] == 2
    later = replace(accepted, expected_version=2, write_operation_id="", content="Initial fact. Added fact. Later fact.")
    writer.write_memory(later, ctx)
    current = fs.read_node(uri, ctx)
    assert current.metadata["version"] == 3 and "accepted-operation" in current.metadata[RECEIPTS_KEY]

    restarted = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    proof = restarted.write_memory(accepted, replace(ctx, trace_id="different-retry-trace"))
    assert proof["action"] == "skip" and proof["merged_fields"]["status"] == "operation_committed"
    assert fs.read_node(uri, ctx) == current
    with pytest.raises(WriteNotStartedError):
        restarted.write_memory(replace(accepted, write_operation_id="new-operation"), ctx)
    assert fs.read_node(uri, ctx) == current


def test_operation_payload_or_session_mismatch_cannot_reuse_proof(ctx):
    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    accepted = replace(memory("Atlas", object_name="Atlas", content="Initial fact."),
                       write_operation_id="accepted-operation")
    uri = writer.write_memory(accepted, ctx)["target_uri"]
    current = fs.read_node(uri, ctx)
    for changed, request in [(replace(accepted, content="Different payload."), ctx),
                             (accepted, replace(ctx, session_id="another-session"))]:
        with pytest.raises(WriteNotStartedError, match="does not match"):
            writer.write_memory(changed, request)
        assert fs.read_node(uri, ctx) == current


def test_operation_target_mismatch_does_not_create_a_second_receipt(ctx):
    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    first = memory("Atlas", object_name="Atlas", content="Atlas fact.")
    other = memory("Bravo", object_name="Bravo", content="Bravo fact.")
    first_uri = writer.write_memory(first, ctx)["target_uri"]
    other_uri = writer.write_memory(other, ctx)["target_uri"]
    accepted = replace(first, target_uri=first_uri, expected_version=1,
                       content="Atlas updated.", write_operation_id="accepted-operation")
    writer.write_memory(accepted, ctx)
    old_other = fs.read_node(other_uri, ctx)
    with pytest.raises(WriteNotStartedError, match="does not match"):
        writer.write_memory(replace(accepted, target_uri=other_uri), ctx)
    assert fs.read_node(other_uri, ctx) == old_other


def test_preference_role_normalization_keeps_operation_receipt_stable(ctx):
    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    accepted = replace(memory("pets", category="preference", who="user", content="Likes cats."),
                       write_operation_id="self-preference-operation")
    uri = writer.write_memory(accepted, ctx)["target_uri"]
    assert writer.write_memory(accepted, ctx)["action"] == "skip"
    assert fs.read_node(uri, ctx).metadata["version"] == 1


@pytest.mark.parametrize("category", ["skill", "tool"])
def test_durable_nonmanaged_operations_do_not_use_content_only_cache(ctx, category):
    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    writer._writer._archive_builder._llm_merge_content = lambda existing, new: existing + "\n\n" + new
    first = CandidateMemory(category=category, owner_scope="agent", routing_key="topic",
                            content="Initial fact.", abstract="Initial fact.", overview="Initial fact.",
                            confidence=.9, write_operation_id="create-operation")
    uri = writer.write_memory(first, ctx)["target_uri"]
    update = replace(first, content="Same update content.", abstract="First abstract.",
                     write_operation_id="first-update")
    writer.write_memory(update, ctx)
    second = replace(update, abstract="New supported abstract.", provenance_ids=["new-source"],
                     write_operation_id="second-update")
    assert writer.write_memory(second, ctx)["action"] == "merge"
    current = fs.read_node(uri, ctx)
    assert current.metadata["version"] == 3 and current.abstract == second.abstract
    assert set(current.metadata[RECEIPTS_KEY]) == {"create-operation", "first-update", "second-update"}
    assert writer.write_memory(second, ctx)["merged_fields"]["status"] == "operation_committed"


@pytest.mark.parametrize("category", ["skill", "tool", "custom_entity"])
def test_bound_durable_policy_cannot_change_target_uri(ctx, category):
    from commit.merge_policies import AggregateTopicPolicy

    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    writer._writer._archive_builder._llm_merge_content = lambda existing, new: existing + "\n\n" + new
    if category == "custom_entity":
        category = "entity"
        writer._writer._policy_router.register_policy(category, AggregateTopicPolicy(
            fs, writer._writer._identity_resolver.uris))
        first = memory("read_target", content="A original.")
        other = memory("other_target", content="B original.")
    else:
        first = CandidateMemory(category=category, owner_scope="agent", routing_key="read_target",
                                content="A original.", abstract="A original.", overview="A original.", confidence=.9)
        other = replace(first, routing_key="other_target", content="B original.")
    first_uri = writer.write_memory(first, ctx)["target_uri"]
    other_uri = writer.write_memory(other, ctx)["target_uri"]
    bound = replace(other, target_uri=first_uri, expected_version=1,
                    content="Update intended for A.", write_operation_id="bound-operation")
    with pytest.raises(WriteNotStartedError, match="changed.*target"):
        writer.write_memory(bound, ctx)
    assert fs.read_node(first_uri, ctx).metadata["version"] == 1
    assert fs.read_node(other_uri, ctx).metadata["version"] == 1


def test_custom_policy_durable_target_cannot_escape_category_receipt_scope(ctx):
    from unittest.mock import Mock

    from core.models import WritePlan

    fs = MemoryFS()
    writer = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)
    resolver = writer._writer._identity_resolver
    uri = resolver.uris.resolve("preference", {"routing_key": "topic"}, ctx)
    policy = Mock()
    policy.plan.return_value = WritePlan(action="create", target_uri=uri, merged_fields={})
    writer._writer._policy_router.register_policy("entity", policy)
    with pytest.raises(WriteNotStartedError, match="schema directory"):
        writer.write_memory(replace(memory("topic"), write_operation_id="custom-operation"), ctx)
    assert not fs.nodes
