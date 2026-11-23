"""Delete source copies left behind before manifest publication."""

from dataclasses import replace

import pytest

from core.errors import ValidationError
from core.evidence import evidence_uri
from service.evidence_lifecycle import EvidenceLifecycle
from tests.unit.evidence.test_sources import CTX, message

@pytest.mark.parametrize("existing_manifest", [False, True])
@pytest.mark.parametrize("failure", ["source", "window", "manifest"])
def test_delete_discovers_unpublished_sources_and_windows(store, monkeypatch, existing_manifest, failure):
    other = replace(CTX, session_id="other")
    other_source = store.register([message()], other, batch_id="other")[0]
    if existing_manifest:
        store.register([message(id="old", sequence=2)], CTX, batch_id="old")
    write = store._write
    source_writes = []

    def interrupt(uri, *args):
        if "/sources/" in uri:
            source_writes.append(uri)
            if failure == "source" and len(source_writes) == 2:
                raise OSError("interrupted source publication")
        if failure == "window" and "/windows/" in uri or failure == "manifest" and "/sessions/" in uri:
            raise OSError("interrupted publication")
        return write(uri, *args)

    monkeypatch.setattr(store, "_write", interrupt)
    with pytest.raises(OSError):
        store.register([message(id="one"), message(id="two", sequence=4)], CTX, batch_id="partial")
    monkeypatch.setattr(store, "_write", write)
    result = EvidenceLifecycle(store.fs).delete_session(CTX)
    assert result["sources_revoked"] == (1 if failure == "source" else 2) + int(existing_manifest)
    for uri in store.fs.list_children(evidence_uri(CTX, "sources"), CTX):
        value, _ = store._read(uri, CTX)
        if uri.endswith(other_source["source_key"]):
            assert value["source"]["text"] == message()["content"]
        else:
            assert value["status"] == "revoked" and "source" not in value
    for uri in store.fs.list_children(evidence_uri(CTX, "windows"), CTX):
        value, _ = store._read(uri, CTX)
        assert value["status"] == ("active" if value["session_id"] == "other" else "revoked")
    with pytest.raises(ValidationError, match="revoked"):
        store.register([message(id="retry")], CTX, batch_id="retry")
    assert EvidenceLifecycle(store.fs).delete_session(CTX)["memories_cleaned"] == 0


def test_orphan_discovery_pages_and_resumes_after_interruption(store, monkeypatch):
    write = store._write
    monkeypatch.setattr(store, "_write", lambda uri, *args: (_ for _ in ()).throw(OSError("publication"))
                        if "/sessions/" in uri else write(uri, *args))
    with pytest.raises(OSError):
        store.register([message(id=str(i), sequence=i) for i in range(105)], CTX, batch_id="partial")
    lifecycle = EvidenceLifecycle(store.fs)
    save = lifecycle.store._write
    checkpoints = []

    def fail_second_checkpoint(uri, value, *args):
        if "/sessions/" in uri and value.get("source_discovery_cursor"):
            checkpoints.append(value["source_discovery_cursor"])
            if len(checkpoints) == 2:
                raise OSError("checkpoint unavailable")
        return save(uri, value, *args)

    monkeypatch.setattr(lifecycle.store, "_write", fail_second_checkpoint)
    with pytest.raises(OSError, match="checkpoint"):
        lifecycle.delete_session(CTX)
    manifest, _ = store.manifest(CTX)
    assert manifest["status"] == "revoked" and len(manifest["sources"]) == 100
    assert EvidenceLifecycle(store.fs).delete_session(CTX)["sources_revoked"] == 105
    assert all(store._read(uri, CTX)[0]["status"] == "revoked"
               for uri in store.fs.list_children(evidence_uri(CTX, "sources"), CTX))


def test_no_evidence_deletion_persists_revocation_before_any_publication(store):
    assert EvidenceLifecycle(store.fs).delete_session(CTX) == {"sources_revoked": 0, "memories_cleaned": 0}
    manifest, version = store.manifest(CTX)
    assert manifest["status"] == "revoked" and manifest["sources"] == []
    with pytest.raises(ValidationError, match="revoked"):
        store.register([message()], CTX, batch_id="late-backfill")
    EvidenceLifecycle(store.fs).delete_session(CTX)
    assert store.manifest(CTX) == (manifest, version)
