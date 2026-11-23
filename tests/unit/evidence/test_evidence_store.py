"""Exercise publication, restart and revocation through the real AGFS adapter."""

from dataclasses import replace

import pytest

from core.errors import AccessDeniedError, ValidationError
from core.evidence import evidence_uri
from fs.agfs_adapter import AGFSContextFS
from index.index_record_builder import build_index_records
from service.evidence_store import EvidenceStore
from tests.unit.evidence.test_sources import CTX, message


def test_publish_replay_and_restart(store):
    sources = store.register([message(), message("I played music.", id="two", sequence=4)], CTX,
                             batch_id="batch")
    versions = [store.read_source(s["source_key"], CTX)["source_node_version"] for s in sources]
    assert store.register([message(), message("I played music.", id="two", sequence=4)], CTX,
                          batch_id="batch") == sources
    restarted = EvidenceStore(AGFSContextFS(store.fs._client, mount_prefix="/local"))
    assert [restarted.read_source(s["source_key"], CTX)["source_node_version"] for s in sources] == versions
    manifest, _ = restarted.manifest(CTX)
    assert len(manifest["sources"]) == 2 and len(manifest["batches"]) == 1
    for uri in [restarted.session_uri(CTX), evidence_uri(CTX, "sources", sources[0]["source_key"])]:
        assert build_index_records(store.fs.read_node(uri, CTX)) == []


def test_partial_batch_is_invisible_and_resumes_without_duplicate_sources(store, monkeypatch):
    write = store._write

    def fail_manifest(uri, *args):
        if "/evidence/sessions/" in uri:
            raise OSError("interrupted publication")
        return write(uri, *args)

    monkeypatch.setattr(store, "_write", fail_manifest)
    with pytest.raises(OSError, match="publication"):
        store.register([message()], CTX, batch_id="batch")
    uris = store.fs.list_children(evidence_uri(CTX, "sources"), CTX)
    assert len(uris) == 1
    key = uris[0].rsplit("/", 1)[1]
    assert store.read_source(key, CTX) is None
    monkeypatch.setattr(store, "_write", write)
    store.register([message()], CTX, batch_id="batch")
    assert store.read_source(key, CTX)["source_node_version"] == 1


def test_changed_message_id_content_and_changed_batch_are_rejected(store):
    source = store.register([message()], CTX, batch_id="batch")[0]
    with pytest.raises(ValidationError, match="identity conflict"):
        store.register([message("I completed 4 essays.")], CTX, batch_id="next")
    with pytest.raises(ValidationError, match="batch changed"):
        store.register([message(id="new")], CTX, batch_id="batch")
    assert store.read_source(source["source_key"], CTX)["text"] == message()["content"]


def test_revocation_survives_restart_and_blocks_replay_and_support(store):
    source = store.register([message()], CTX, batch_id="batch")[0]
    ref = {"source_key": source["source_key"], "content_hash": source["content_hash"],
           "evidence_unit_id": source["units"][0]["evidence_unit_id"],
           "span": [0, len(source["text"])], "quote": source["text"]}
    assert store.verify_ref(ref, CTX)
    assert not store.verify_ref({**ref, "quote": "invented fact"}, CTX)
    uri = "ctx://account/users/alice/memories/facts/one"
    store.add_support(uri, [ref], CTX)
    assert store.revoke_session(CTX)["supports"] == {uri: [source["source_key"]]}
    restarted = EvidenceStore(AGFSContextFS(store.fs._client, mount_prefix="/local"))
    assert restarted.read_source(source["source_key"], CTX) is None
    assert not restarted.verify_ref(ref, CTX)
    with pytest.raises(ValidationError, match="revoked"):
        restarted.register([message()], CTX, batch_id="new-batch")
    with pytest.raises(ValidationError, match="unavailable"):
        restarted.add_support(uri, [ref], CTX)
    assert restarted.revoke_session(CTX)["status"] == "revoked"


def test_source_identity_isolated_even_with_same_session_and_message(store):
    source = store.register([message()], CTX, batch_id="batch")[0]
    other = replace(CTX, user_id="bob")
    assert store.read_source(source["source_key"], other) is None
    with pytest.raises(AccessDeniedError):
        store.fs.read_node(evidence_uri(CTX, "sources", source["source_key"]), other)
    assert store.register([message()], other, batch_id="batch")[0]["source_key"] != source["source_key"]


def test_invalid_timezone_does_not_persist_any_part_of_a_batch(store):
    with pytest.raises(ValidationError, match="timezone"):
        store.register([message(), message(id="bad", sequence=4, timezone="Invalid/Zone")], CTX, batch_id="bad")
    assert store.manifest(CTX)[0] is None
    assert store.fs.list_children(evidence_uri(CTX, "sources"), CTX) == []
    store.register([message(timezone="UTC")], CTX, batch_id="valid")
