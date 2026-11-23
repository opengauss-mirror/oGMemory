"""Deterministic cards use normal ContextWriter persistence, CAS and index events."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest

from commit.deterministic import deterministic_batch, validate_payload
from core.errors import OperationDeadlineExceededError, ValidationError
from core.evidence import digest
from core.operation_deadline import operation_deadline
from core.validation import MAX_CONTENT_LEN, validate_candidate
from extraction.deterministic.verification import verify_record
from service.evidence_store import EvidenceStore
from tests.unit.evidence.test_extractor import extractor
from tests.unit.evidence.test_sources import CTX, message


def frozen_candidates(store, messages, ctx=CTX, *, batch_id="batch"):
    store.register(messages, ctx, batch_id=batch_id)
    return extractor().extract(messages, ctx)


def write_candidates(setup, messages, *, categories=None):
    _, api, store, _ = setup
    candidates = frozen_candidates(store, messages)
    with deterministic_batch():
        return [api._writer.write_candidate(candidate, CTX) for candidate in candidates
                if categories is None or candidate.category in categories]


def test_deterministic_writes_are_source_verified_and_replayable(setup):
    fs, api, store, llm = setup
    candidates = frozen_candidates(store, [message()])
    with deterministic_batch():
        plans = [api._writer.write_candidate(c, CTX) for c in candidates]
        replay = [api._writer.write_candidate(c, CTX) for c in candidates]
    assert {plan.action for plan in plans} == {"create"}
    assert {plan.action for plan in replay} == {"skip"}
    for plan in plans:
        node = fs.read_node(plan.target_uri, CTX)
        assert node.metadata["deterministic"]["fact_records"]
        assert node.metadata["version"] == 1
    llm.complete_json.assert_not_called()


def test_client_metadata_cannot_bypass_normal_write_processing(setup):
    _, api, store, _ = setup
    candidate = frozen_candidates(store, [message()])[0]
    with pytest.raises(ValidationError, match="reserved"):
        api._writer.write_candidate(candidate, CTX)
    with deterministic_batch():
        candidate.metadata["deterministic"]["source_refs"][0]["quote"] = "invented"
        with pytest.raises(ValidationError, match="unverified"):
            api._writer.write_candidate(candidate, CTX)


def test_llm_cannot_overwrite_deterministic_entity_by_explicit_target(setup):
    fs, api, store, _ = setup
    messages = [message("Caroline visited Paris."), message("Caroline completed 7 essays.", id="two", sequence=4)]
    candidate = next(c for c in frozen_candidates(store, messages) if c.category == "entity")
    with deterministic_batch():
        plan = api._writer.write_candidate(candidate, CTX)
    node = fs.read_node(plan.target_uri, CTX)
    llm_candidate = replace(candidate, metadata={}, target_uri=plan.target_uri,
                            expected_version=node.metadata["version"], content="overwrite")
    with pytest.raises((ValueError, ValidationError)):
        api._writer.write_candidate(llm_candidate, CTX)
    assert fs.read_node(plan.target_uri, CTX).content == node.content


@pytest.mark.parametrize("keep_other_session", [False, True])
def test_revocation_survives_failed_cleanup_and_reuses_event_ids(setup, keep_other_session):
    from uuid import UUID

    from commit.outbox_store import OutboxStore
    from core.errors import NodeNotFoundError
    from retrieval.context_reader import ContextReader
    from service.evidence_lifecycle import EvidenceLifecycle

    fs, _, store, _ = setup
    plan, = write_candidates(setup, [message()], categories={"fact"})
    if keep_other_session:
        other = replace(CTX, session_id="other")
        extra = next(c for c in frozen_candidates(store, [message("I visited Paris.")], other)
                     if c.category == "fact").metadata["deterministic"]["fact_records"][0]
        node = fs.read_node(plan.target_uri, CTX)
        data = node.metadata["deterministic"]
        data["fact_records"].append(extra)
        data["source_refs"].extend(extra["source_refs"])
        store.add_support(node.uri, extra["source_refs"], other)
        fs.write_node(replace(node, metadata={**node.metadata, "_expected_version": node.metadata["version"]}), CTX)
    reader = ContextReader(fs)
    assert reader.read(plan.target_uri, CTX).abstract
    events = []
    outbox = Mock()
    outbox.build_write_event.side_effect = OutboxStore(fs._client, fs).build_write_event

    def register(event, ctx):
        assert len(event.event_id) == 36
        UUID(event.event_id)
        events.append(event)
        if len(events) == 1:
            raise ConnectionError("retry cleanup")

    outbox.register_once.side_effect = register
    lifecycle = EvidenceLifecycle(fs, outbox)
    with pytest.raises(ConnectionError, match="retry cleanup"):
        lifecycle.delete_session(CTX)
    assert fs.exists(plan.target_uri, CTX)
    if not keep_other_session:
        for read in (reader.read, reader.read_authorized):
            with pytest.raises(NodeNotFoundError):
                read(plan.target_uri, CTX)
    assert lifecycle.delete_session(CTX)["memories_cleaned"] == 1
    assert fs.exists(plan.target_uri, CTX) is keep_other_session
    assert lifecycle.delete_session(CTX)["memories_cleaned"] == 0
    assert events[0].event_id == events[1].event_id
    assert events[0].event_type == ("UPSERT_CONTEXT" if keep_other_session else "ARCHIVE_CONTEXT")


# Large entity bodies must not reject otherwise valid extraction batches.


def messages(start, count):
    # Individual sentences are below the 4000-character extraction cap; both
    # source messages are legal, while their shared entity exceeds 100k.
    text = " ".join(f"Caroline visited place{index} " + "x" * 3500 + "." for index in range(start, start + count))
    return [message(text, id=str(start), sequence=start),
            message(f"Caroline completed task{start}.", id=str(start) + "-other", sequence=start + 1)]


def test_accumulated_entity_merge_is_bounded_without_dropping_records_or_replay_identity(setup):
    fs, api, store, _ = setup
    all_records = {}
    with deterministic_batch():
        for start in (0, 20, 40):
            batch = messages(start, 16)
            candidate = next(c for c in frozen_candidates(store, batch, batch_id=str(start)) if c.category == "entity")
            all_records.update((record["fact_id"], record) for record in candidate.metadata["deterministic"]["fact_records"])
            plan = api._writer.write_candidate(candidate, CTX)
            node = fs.read_node(plan.target_uri, CTX)
            assert len(node.content) <= MAX_CONTENT_LEN
            assert {r["fact_id"]: r for r in node.metadata["deterministic"]["fact_records"]} == all_records
            assert api._writer.write_candidate(candidate, CTX).action == "skip"
    assert len(all_records) == 51
    assert len({ref["message_id"] for ref in node.metadata["deterministic"]["source_refs"]}) == 6
    omitted = next(record for record in all_records.values()
                   if len(record["quote"]) < 100 and record["quote"] not in node.content)
    assert verify_record(omitted, EvidenceStore(fs), CTX)


# Trusted batch checks hash each candidate once while retaining source checks.


@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize("verify_structure", [False, True])
def test_payload_hash_count_is_independent_of_record_count(setup, monkeypatch, trusted, verify_structure):
    _, _, store, _ = setup
    batch = messages(0, 32)
    candidate = next(c for c in frozen_candidates(store, batch) if c.category == "entity")
    read = Mock(wraps=store.read_source)
    monkeypatch.setattr(store, "read_source", read)
    verify = Mock(return_value=True)
    monkeypatch.setattr("extraction.deterministic.verification.verify_record", verify)
    with deterministic_batch([candidate] if trusted else []):
        hashed = Mock(wraps=digest)
        monkeypatch.setattr("commit.deterministic.digest", hashed)
        assert validate_payload(candidate, store, CTX, verify_structure=verify_structure) == candidate.metadata["deterministic"]
    assert hashed.call_count == (1 if verify_structure else 0)
    count = len(candidate.metadata["deterministic"]["fact_records"])
    assert read.call_count == len(batch)
    assert verify.call_count == (count if verify_structure and not trusted else 0)


def test_changed_payload_and_revoked_source_cannot_reuse_trusted_batch(setup):
    _, api, store, _ = setup
    candidate = next(c for c in frozen_candidates(store, messages(0, 2)) if c.category == "entity")
    changed = deepcopy(candidate)
    with deterministic_batch([candidate]):
        changed.metadata["deterministic"]["fact_records"][-1]["subject"] = "invented"
        with pytest.raises(ValidationError, match="structured assertion"):
            api._writer.write_candidate(changed, CTX)
        store.revoke_session(CTX)
        with pytest.raises(ValidationError, match="revoked"):
            validate_payload(candidate, store, CTX)


def test_large_entity_keeps_lineage_and_reads_each_source_once_per_pass(setup, monkeypatch):
    from collections import Counter

    import extraction.deterministic.extractor as rules

    _, api, store, _ = setup
    candidates = frozen_candidates(store, messages(0, 100))
    assert {c.category for c in candidates} == {"fact", "event", "entity"}
    for candidate in candidates:
        validate_candidate(candidate)
    entity = next(c for c in candidates if c.category == "entity")
    records = entity.metadata["deterministic"]["fact_records"]
    assert len(records) == 101 and len(entity.content) <= MAX_CONTENT_LEN
    assert entity.content.rstrip().endswith(".")
    assert all(record["quote"] == record["source_refs"][0]["quote"] for record in records)
    counts = Counter()
    load = EvidenceStore.read_source
    count_quotes = Mock(wraps=rules._QUOTED.finditer)

    def read(self, key, ctx, **kwargs):
        counts[key] += 1
        return load(self, key, ctx, **kwargs)

    monkeypatch.setattr(EvidenceStore, "read_source", read)
    monkeypatch.setattr(rules, "_QUOTED", Mock(finditer=count_quotes))
    with deterministic_batch():
        assert api._writer.write_candidate(entity, CTX).action == "create"
    assert sorted(counts.values()) == [3, 3]  # Preflight, support registration and locked publication.
    assert count_quotes.call_count == 2


@pytest.mark.parametrize("revoke_after_registration", [False, True])
def test_writer_rejects_revocation_before_and_during_publication(setup, monkeypatch, revoke_after_registration):
    fs, api, store, _ = setup
    candidate = next(c for c in frozen_candidates(store, messages(0, 2)) if c.category == "entity")
    add_support = EvidenceStore.add_support

    def revoke_after_support(self, uri, refs, ctx):
        add_support(self, uri, refs, ctx)
        self.revoke_session(ctx)
        assert not fs.exists(uri, ctx)

    if revoke_after_registration:
        monkeypatch.setattr(EvidenceStore, "add_support", revoke_after_support)
    else:
        store.revoke_session(CTX)
    with deterministic_batch([candidate]), pytest.raises(ValidationError, match="revoked"):
        api._writer.write_candidate(candidate, CTX)


def test_validation_observes_deadline_between_cached_references(setup, monkeypatch):
    from core import operation_deadline as deadline

    _, _, store, _ = setup
    candidate = next(c for c in frozen_candidates(store, messages(0, 2)) if c.category == "entity")
    clock = [100.]
    read = store.read_source

    def slow_read(key, ctx):
        result = read(key, ctx)
        clock[0] += 2
        return result

    monkeypatch.setattr(deadline, "monotonic", lambda: clock[0])
    monkeypatch.setattr(store, "read_source", Mock(side_effect=slow_read))
    with operation_deadline(1), pytest.raises(OperationDeadlineExceededError):
        validate_payload(candidate, store, CTX, verify_structure=False)
    assert store.read_source.call_count == 1
