from dataclasses import replace
from unittest.mock import Mock

import pytest

from commit.archive_builder import ArchiveBuilder
from commit.sql_outbox_store import SQLOutboxStore
from core.models import CandidateMemory, RequestContext, WritePlan
from index.activation_projection import ACTIVATION_VERSION, is_current_activation
from index.index_record_builder import build_index_records
from index.outbox_worker import OutboxWorker


def candidate():
    return CandidateMemory(
        category="profile", owner_scope="user", routing_key="hobby",
        abstract="Enjoys pottery", overview="Pottery hobby", content="Enjoys pottery.",
        confidence=0.95, metadata={"update_activation": "A later message discusses pottery.",
                                   "update_activation_version": ACTIVATION_VERSION})


def node():
    return ArchiveBuilder(Mock()).build(
        candidate(), WritePlan("create", "ctx://acme/users/alice/memories/profile/hobby"),
        RequestContext("acme", "alice", "agent", "session", "trace"))


def test_candidate_persistence_and_single_embedding_batch():
    stored = node()
    assert stored.metadata["update_activation"] == candidate().metadata["update_activation"]
    event = SQLOutboxStore.build_write_event(Mock(), stored)
    embedder = Mock(model="test-model")
    embedder.embed_texts.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    index = Mock()
    result = OutboxWorker(index, embedder).process_event(event)
    assert result.success
    embedder.embed_texts.assert_called_once()
    records = index.upsert.call_args.args[0]
    assert [r.level for r in records] == [0, 2, 3]
    activation = records[-1]
    assert activation.filters["account_id"] == "acme"
    assert activation.filters["owner_space"] == "user:alice"
    assert activation.metadata["embedding_model"] == "test-model"
    assert activation.metadata["embedding_dimension"] == 2
    assert is_current_activation(activation, stored)


@pytest.mark.parametrize("value", [None, "", "\n", "first\nsecond", 12])
def test_malformed_activation_has_no_projection(value):
    stored = node()
    stored.metadata["update_activation"] = value
    assert [r.level for r in build_index_records(stored)] == [0, 2]


def test_legacy_and_obsolete_version_not_indexed():
    stored = node()
    stored.metadata["update_activation_version"] = "old-standalone-version"
    assert len(build_index_records(stored)) == 2
    stored.metadata = {}
    assert len(build_index_records(stored)) == 2


def test_revision_rejects_stale_memory_and_activation():
    stored = node()
    projection = build_index_records(stored)[-1]
    assert not is_current_activation(projection, replace(stored, content="Changed memory"))
    stored.metadata["update_activation"] = "A different future message."
    updated = build_index_records(stored)[-1]
    assert updated.id == projection.id
    assert not is_current_activation(projection, stored)
    assert is_current_activation(updated, stored)


@pytest.mark.parametrize("event_type", ["DELETE_CONTEXT", "ARCHIVE_CONTEXT"])
def test_removal_cleans_activation(event_type):
    stored = node()
    event = SQLOutboxStore.build_write_event(Mock(), stored)
    event.event_type = event_type
    event.payload = {"ids_to_delete": ["legacy-record"]}
    index = Mock()
    assert OutboxWorker(index, Mock()).process_event(event).success
    assert build_index_records(stored)[-1].id in index.delete.call_args.args[0]
