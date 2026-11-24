"""Public ingestion binds deterministic extraction to the durable source batch."""

from unittest.mock import Mock

import pytest

from extraction.deterministic.nlp import DeterministicConfig
from extraction.summary_generator import SessionSummaryGenerator
from fs.agfs_adapter import AGFSContextFS
from providers.unified_config import OgMemConfig
from service.api import MemoryWriteAPI
from service.evidence_store import EvidenceStore
from tests.unit.evidence.test_sources import CTX
from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient
from tests.unit.server.test_extraction_batch_recovery import params, setup_service


@pytest.fixture(autouse=True)
def missing_nlp(monkeypatch):
    monkeypatch.setattr("extraction.deterministic.nlp.load_model", Mock(side_effect=OSError("absent")))


def configured_service(fs, *, outbox=None):
    service, api, manager = setup_service(fs, CTX, outbox=outbox)
    api._extraction_engine = "deterministic"
    api._deterministic_config = DeterministicConfig(on_model_missing="rules_only")
    service._cfg.extraction_engine = "deterministic"
    service._get_context_fs = lambda: fs
    return service, api, manager


def test_engine_is_only_evidence_switch_and_yaml_priority(monkeypatch):
    assert not OgMemConfig().evidence_enabled
    assert OgMemConfig(extraction_engine="deterministic").evidence_enabled
    monkeypatch.setattr("providers.unified_config._load_yaml", lambda _: {
        "memory": {"extraction": {"engine": "deterministic", "deterministic": {"on_model_missing": "rules_only"}}}})
    monkeypatch.setenv("OGMEM_EXTRACTION_ENGINE", "llm")
    assert OgMemConfig.load("ignored").evidence_enabled
    monkeypatch.setattr("providers.unified_config._load_yaml", lambda _: {"memory": {"evidence": {"enabled": True}}})
    with pytest.raises(ValueError, match="unsupported"):
        OgMemConfig.load("ignored")


def test_after_turn_preserves_assistant_sources_and_original_llm_path_is_unused():
    client = _AtomicMemoryAGFSClient()
    client.api_base = "http://test-agfs"
    fs = AGFSContextFS(client, mount_prefix="/local")
    service, api, _ = configured_service(fs)
    request = params(CTX, content="I completed 7 essays.")
    request["messages"].append({"role": "assistant", "content": "I recommend visiting Paris."})
    result = service.after_turn(request)
    assert result["status"] == "completed", result
    api._pipeline.extract.assert_not_called()
    manifest, _ = EvidenceStore(fs).manifest(CTX)
    sources = [EvidenceStore(fs).read_source(key, CTX) for key in manifest["sources"]]
    assert {source["role"] for source in sources} == {"user", "assistant"}
    assert sorted(source["sequence"] for source in sources) == [0, 1]
    facts = fs.list_children("ctx://account/users/alice/memories/facts", CTX)
    assert len(facts) == 2
    assert service.after_turn(request)["status"] == "duplicate"
    next_result = service.after_turn(params(CTX, request="next", content="I visited Paris."))
    assert next_result["status"] == "completed", next_result
    manifest, _ = EvidenceStore(fs).manifest(CTX)
    assert sorted(EvidenceStore(fs).read_source(key, CTX)["sequence"] for key in manifest["sources"]) == [0, 1, 2]


@pytest.mark.parametrize("text", ["I recommend visiting Paris.", "Okay."])
def test_assistant_only_turn_registers_evidence_once(text):
    client = _AtomicMemoryAGFSClient()
    client.api_base = "http://test-agfs"
    fs = AGFSContextFS(client, mount_prefix="/local")
    outbox = Mock()
    service, api, manager = configured_service(fs, outbox=outbox)
    request = params(CTX)
    request["messages"] = [{"id": "assistant-only", "role": "assistant", "content": text,
                            "created_at": "2026-11-14T01:02:03+00:00"}]
    result = service.after_turn(request)
    assert result["status"] == "completed", result
    api._pipeline.extract.assert_not_called()
    manifest, _ = EvidenceStore(fs).manifest(CTX)
    assert len(manifest["sources"]) == 1
    source = EvidenceStore(fs).read_source(manifest["sources"][0], CTX)
    assert source["message_id"] == "assistant-only"
    assert source["role"] == "assistant" and source["text"] == text
    assert source["recorded_at"] == "2026-11-14T01:02:03+00:00"
    assert service.after_turn(request)["status"] == "duplicate"
    assert len(EvidenceStore(fs).manifest(CTX)[0]["sources"]) == 1
    assert not manager.get_or_create(CTX.session_id).pending_extraction_ids()


def test_llm_assistant_only_turn_keeps_archive_only_behavior():
    client = _AtomicMemoryAGFSClient()
    client.api_base = "http://test-agfs"
    fs = AGFSContextFS(client, mount_prefix="/local")
    service, api, manager = setup_service(fs, CTX)
    request = params(CTX)
    request["messages"] = [{"role": "assistant", "content": "I recommend visiting Paris."}]
    result = service.after_turn(request)
    assert result["status"] == "no_extractable_messages", result
    api._pipeline.extract.assert_not_called()
    manager.commit_snapshot.assert_called_once()
    assert EvidenceStore(fs).manifest(CTX)[0] is None


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("mode", ["eager", "fast"])
def test_accepted_batch_recovers_after_engine_switch_without_nlp(monkeypatch, role, mode):
    from service.extraction_checkpoint import EXTENSION_KEY

    client = _AtomicMemoryAGFSClient()
    client.api_base = "http://test-agfs"
    fs = AGFSContextFS(client, mount_prefix="/local")
    service, _, manager = configured_service(fs)
    save = manager.save_session_state
    failed = []

    def fail_completion(session_id, ctx):
        batch = manager.get_or_create(session_id).runtime_extensions.get(EXTENSION_KEY, {}).get("batch", {})
        if not failed and any(op.get("status") == "complete" for slot in batch.get("slots", {}).values()
                              for op in slot["operations"].values()
                              if slot["candidates"] and slot["candidates"][0].get("metadata", {}).get("deterministic")):
            failed.append(True)
            return False
        return save(session_id, ctx)

    manager.save_session_state = fail_completion
    request = params(CTX, content="I recommend visiting Paris." if role == "assistant" else "I completed 7 essays.")
    request["messages"][0]["role"] = role
    assert service.after_turn(request)["status"] == "failed"
    assert failed
    restarted, writer, _ = setup_service(fs, CTX)
    if mode == "fast":
        writer = MemoryWriteAPI(fs, restarted.get_llm(), extraction_mode=mode, directory_summary_enabled=False)
        writer._pipeline.extract = Mock()
        restarted._write_api = writer
        restarted._cfg.extraction_mode = mode
    restarted._get_context_fs = lambda: fs
    monkeypatch.setattr("extraction.deterministic.nlp.load_model", Mock(side_effect=AssertionError("NLP recovery")))
    monkeypatch.setattr("service.deterministic_ingest.get_extractor", Mock(side_effect=AssertionError("re-extraction")))
    result = restarted.after_turn(request)
    assert result["status"] == "completed", result
    writer._pipeline.extract.assert_not_called()
    facts = fs.list_children("ctx://account/users/alice/memories/facts", CTX)
    assert len(facts) == 1
    assert fs.read_node(facts[0], CTX).metadata["version"] == 1


def test_accepted_llm_batch_keeps_old_engine_after_switch(monkeypatch):
    from commit.write_receipts import WriteNotStartedError

    client = _AtomicMemoryAGFSClient()
    client.api_base = "http://test-agfs"
    fs = AGFSContextFS(client, mount_prefix="/local")
    service, writer, _ = setup_service(fs, CTX)
    service._get_context_fs = lambda: fs
    original_write = writer._writer.write_candidate

    def fail_memory(candidate, ctx):
        if candidate.category != "session_archive":
            raise WriteNotStartedError("retry later")
        return original_write(candidate, ctx)

    writer._writer.write_candidate = fail_memory
    request = params(CTX)
    assert service.after_turn(request)["status"] == "failed"
    restarted, writer, _ = configured_service(fs)
    monkeypatch.setattr("service.deterministic_ingest.get_extractor", Mock(side_effect=AssertionError("engine changed")))
    result = restarted.after_turn(request)
    assert result["status"] == "completed", result
    writer._pipeline.extract.assert_not_called()
    assert EvidenceStore(fs).manifest(CTX)[0] is None


def test_fast_restores_missing_summary_of_a_deterministic_batch():
    client = _AtomicMemoryAGFSClient()
    client.api_base = "http://test-agfs"
    fs = AGFSContextFS(client, mount_prefix="/local")
    service, api, _ = configured_service(fs)
    api.write_raw_chunk = Mock(side_effect=RuntimeError("interrupted before summary creation"))
    text = "I completed 7 essays."
    request = params(CTX, content=text)
    pending = service.after_turn(request)
    assert pending["request_status"] == "RECOVERING"

    restarted, _, _ = setup_service(fs, CTX)
    restarted._cfg.extraction_mode = "fast"
    restarted._get_context_fs = lambda: fs
    api = MemoryWriteAPI(fs, restarted.get_llm(), extraction_mode="fast", directory_summary_enabled=False)
    restarted._write_api = api
    summary = SessionSummaryGenerator.build_candidate({"facts_and_details": text}, text, session_id=CTX.session_id)
    restarted._generate_summary_candidate = Mock(return_value=summary)

    result = restarted.after_turn(request)

    assert result["request_status"] == "SUCCEEDED"
    restarted._generate_summary_candidate.assert_called_once()
    summaries = api._writer._identity_resolver.uris.get_directory_uri("session_summary", CTX)
    assert [fs.read_node(uri, CTX).content for uri in fs.list_children(summaries, CTX)] == [text]


@pytest.mark.parametrize("entry", ["compaction", "dispose"])
def test_deterministic_lifecycle_entries_preserve_full_source_view(entry, monkeypatch):
    from threading import Thread

    from tests.unit.server.test_lifecycle_batch_recovery import lifecycle_service

    client = _AtomicMemoryAGFSClient()
    client.api_base = "http://test-agfs"
    fs = AGFSContextFS(client, mount_prefix="/local")
    service, writer, manager = lifecycle_service(fs, CTX)
    writer._extraction_engine = "deterministic"
    writer._deterministic_config = DeterministicConfig(on_model_missing="rules_only")
    service._cfg.extraction_engine = "deterministic"
    manager.add_message(CTX.session_id, "user", "I completed 7 essays. " * 40, CTX, message_id="original-user")
    manager.add_message(CTX.session_id, "assistant", "I recommend visiting Paris.", CTX, message_id="original-ai")
    assert manager.save_session_state(CTX.session_id, CTX)
    if entry == "compaction":
        result = service.prepare_compaction({"sessionId": CTX.session_id, "_ctx": CTX})
    else:
        workers = []

        def tracked_thread(*args, **kwargs):
            worker = Thread(*args, **kwargs)
            workers.append(worker)
            return worker

        monkeypatch.setattr("server.memory_service.threading.Thread", tracked_thread)
        result = service.dispose({"sessionId": CTX.session_id, "_ctx": CTX})
        for worker in workers:
            worker.join(timeout=10)
            assert not worker.is_alive()
    writer._pipeline.extract.assert_not_called()
    store = EvidenceStore(fs)
    manifest, _ = store.manifest(CTX)
    assert manifest is not None, result
    sources = [store.read_source(key, CTX) for key in manifest["sources"]]
    assert {source["message_id"] for source in sources} == {"original-user", "original-ai"}
    assert {source["role"] for source in sources} == {"user", "assistant"}
    assert fs.list_children("ctx://account/users/alice/memories/facts", CTX)
