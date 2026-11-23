"""Keep deterministic source and lease contracts in the asynchronous Agent path."""

from contextlib import contextmanager, nullcontext
from unittest.mock import Mock

import pytest

from fs.agfs_adapter import AGFSContextFS
from service.evidence_store import EvidenceStore
from service.extraction_checkpoint import EXTENSION_KEY, ExtractionBatchJournal
from service.extraction_recovery_discovery import AGENT_TURN_TASK
from tests.unit.evidence.test_ingest import configured_service
from tests.unit.evidence.test_ingest import missing_nlp as missing_nlp
from tests.unit.evidence.test_sources import CTX
from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient
from tests.unit.server.test_extraction_batch_recovery import params

pytestmark = pytest.mark.usefixtures("missing_nlp")


def async_service(monkeypatch, *, role="user"):
    client = _AtomicMemoryAGFSClient()
    client.api_base = "http://test-agfs"
    fs = AGFSContextFS(client, mount_prefix="/local")
    service, api, manager = configured_service(fs)
    monkeypatch.setattr(service._extraction_retry, "wake", Mock())
    monkeypatch.setattr(manager, "parallel_archive_compression", lambda *a, **kw: nullcontext(None))
    content = "I recommend visiting Paris." if role == "assistant" else "I completed 7 essays."
    request = {**params(CTX, content=content), "wait": False, "forceExtract": True}
    request["messages"][0]["role"] = role
    result = service.after_turn(request)
    assert result["accepted"], result
    api._pipeline.extract.assert_not_called()
    ready_at_ms = service._extraction_retry.wake.call_args.args[1]["ready_at_ms"]
    with service._session_mutation_scope(CTX):
        journal, delay = service._agent_turn_journal(CTX)
    assert delay is None
    assert journal.batch["agent_async"]
    assert isinstance(ready_at_ms, int) and journal.batch["ready_at_ms"] == ready_at_ms
    assert journal.batch["extraction_config"]["engine"] == "deterministic"
    assert journal.batch["messages"][0]["role"] == role
    return service, api, manager, fs, request


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_async_turn_keeps_frozen_engine_sources_and_publication_fences(monkeypatch, role):
    service, api, manager, fs, request = async_service(monkeypatch, role=role)
    # A queued batch keeps its original engine even when the active config changes.
    api._extraction_engine = service._cfg.extraction_engine = "llm"
    original_process = service._process_agent_turn
    original_register = EvidenceStore.register
    original_write = api._writer.write_candidate
    depth, writes = 0, []

    def process(ctx, journal, publication_scope):
        @contextmanager
        def fenced():
            nonlocal depth
            with publication_scope():
                depth += 1
                try:
                    yield
                finally:
                    depth -= 1
        return original_process(ctx, journal, fenced)

    def register(store, *args, **kwargs):
        assert depth, "evidence source publication escaped the lease"
        return original_register(store, *args, **kwargs)

    def write(candidate, ctx, **kwargs):
        if candidate.metadata.get("deterministic"):
            assert depth, "memory publication escaped the lease"
            writes.append(candidate)
        return original_write(candidate, ctx, **kwargs)

    extractor = next(iter(api._deterministic_extractors.values()))
    original_extract = extractor.extract

    def extract(*args, **kwargs):
        assert depth == 0, "NLP must not hold the publication scope"
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(service, "_process_agent_turn", process)
    monkeypatch.setattr(EvidenceStore, "register", register)
    monkeypatch.setattr(api._writer, "write_candidate", write)
    monkeypatch.setattr(extractor, "extract", extract)
    assert service._extraction_retry.run_once(CTX, AGENT_TURN_TASK) == 0
    api._pipeline.extract.assert_not_called()
    assert writes
    store = EvidenceStore(fs)
    manifest, _ = store.manifest(CTX)
    source = store.read_source(manifest["sources"][0], CTX)
    assert source["role"] == role and source["sequence"] == 0
    assert source["text"] == request["messages"][0]["content"]
    assert not manager.get_or_create(CTX.session_id).runtime_extensions.get(EXTENSION_KEY)
    assert service.after_turn(request)["status"] == "duplicate"


def test_async_deterministic_turn_cannot_publish_after_losing_lease(monkeypatch):
    service, api, manager, fs, _ = async_service(monkeypatch)
    buffer = manager.get_or_create(CTX.session_id)
    extractor = next(iter(api._deterministic_extractors.values()))
    original = extractor.extract

    def lose_lease(*args, **kwargs):
        result = original(*args, **kwargs)
        with service._session_mutation_scope(CTX):
            journal = ExtractionBatchJournal.resume(manager, buffer, CTX)
            journal.mutate(lambda batch: batch["retry_task"].update(lease_token="new-owner"))
        return result

    monkeypatch.setattr(extractor, "extract", lose_lease)
    assert service._extraction_retry.run_once(CTX, AGENT_TURN_TASK) == 1
    assert not fs.list_children("ctx://account/users/alice/memories/facts", CTX)
    assert buffer.runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]["lease_token"] == "new-owner"
    api._pipeline.extract.assert_not_called()
