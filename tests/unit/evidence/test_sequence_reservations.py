"""Freezing and publication share one durable message order."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from core.errors import ValidationError
from service.evidence_store import EvidenceStore, snapshot_messages
from tests.unit.evidence.test_sources import CTX, message

def freeze(store, identity="live"):
    return snapshot_messages([SimpleNamespace(id=identity, role="user", content="Did I visit Paris?",
                                             created_at=None)], CTX, store.fs)


def test_reservations_survive_restart_and_failed_checkpoint(store):
    frozen = freeze(store)
    restarted = EvidenceStore(store.fs)
    assert freeze(restarted) == frozen
    assert freeze(restarted, "next")[0]["sequence"] == 1
    store.register(frozen, CTX, batch_id="batch")
    assert freeze(restarted) == frozen


@pytest.mark.parametrize("published", [False, True])
def test_publication_rejects_duplicate_slots_before_writing_sources(store, published):
    frozen = freeze(store)
    if published:
        store.register(frozen, CTX, batch_id="live")
    with pytest.raises(ValidationError, match="sequence"):
        store.register([message(id="intruder", sequence=0)], CTX, batch_id="other")
    with pytest.raises(ValidationError, match="sequence"):
        store.register([{**frozen[0], "sequence": 7}], CTX, batch_id="changed")
    assert len(store.fs.list_children("ctx://account/users/alice/evidence/sources", CTX)) == int(published)


def test_one_batch_cannot_claim_the_same_slot_twice(store):
    with pytest.raises(ValidationError, match="sequence"):
        store.register([message(id="a", sequence=0), message(id="b", sequence=0)], CTX, batch_id="bad")
    assert store.manifest(CTX)[0] is None


def test_concurrent_freezes_reserve_distinct_positions(store):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(freeze, store, identity) for identity in ("a", "b")]
        positions = [future.result(timeout=10)[0]["sequence"] for future in futures]
    assert sorted(positions) == [0, 1]


def test_revoke_before_freeze_rejects_the_reservation(store):
    store.revoke_session(CTX)
    with pytest.raises(ValidationError, match="revoked"):
        freeze(store)
