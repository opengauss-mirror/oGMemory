"""Prepared model stages persist independently and resume from matching inputs."""

import json
import subprocess
import sys
import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from commit.write_receipts import RECEIPTS_KEY
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import ExtractionBatchJournal
from core.model_stage import (collect_stage_results, model_signature, run_model_call,
                                 stage_collector_scope, stage_digest)
from session.session_manager import SessionManager
from tests.unit.commit.test_memory_identity import memory
from tests.unit.commit.test_request_outbox_events import make_runtime, request_context
from tests.unit.server.test_extraction_batch_recovery import NoModel


def test_model_stage_dependencies_load_from_fresh_processes():
    for module in ("commit", "fs.sql_adapter", "extraction", "service"):
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}; from core.model_stage import run_model_call; "
             "print(run_model_call(lambda: 'ready'))"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ready"


def journal_for(fs, ctx):
    manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    manager.add_message(ctx.session_id, "user", "Melanie painted a lake sunrise last year.", ctx,
                        message_id="source-1", created_at="2023-05-08T13:56:00+00:00")
    buffer = manager.get_or_create(ctx.session_id, ctx)
    journal = ExtractionBatchJournal(manager, buffer, ctx, list(buffer.messages),
                                     archive_id="sample0-session1", messages=[{
                                         "role": "user", "content": buffer.messages[0].content}],
                                     session_time="2023-05-08T13:56:00+00:00", session_summary="",
                                     tool_stats_text="", client_request_id="source-request")
    return manager, journal


def restore(fs, ctx):
    manager = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    buffer = manager.get_or_create(ctx.session_id, ctx)
    manager.enable_extraction_recovery(ctx.session_id, ctx, fs)
    return ExtractionBatchJournal.resume(manager, buffer, ctx)


def test_completed_stage_restores_with_compact_checkpoint_and_original_result():
    _, fs, _ = make_runtime()
    ctx = request_context("stage-results")
    manager, journal = journal_for(fs, ctx)
    inputs = {"prompt": "Melanie painted a lake sunrise last year.", "read_versions": {"painting": 7}}
    result = {"candidate": {"content": "Melanie painted a lake sunrise in 2022."}}
    checkpoint = journal.checkpoint("chunk:0")
    checkpoint.save_stage("entity", inputs, result)
    record = journal.batch["stage_results"]["slot:chunk:0/entity"][stage_digest(inputs)]
    state = fs.read_node(manager.get_session_state().state_uri(ctx.account_id, ctx.session_id), ctx)
    saved = json.loads(state.content)["runtime_extensions"]["memory_extraction_recovery"]["batch"]
    assert set(saved["stage_results"]["slot:chunk:0/entity"][stage_digest(inputs)]) == {
        "version", "status", "input_sha256", "output_sha256", "uri"}
    artifact = json.loads(fs.read_node(record["uri"], ctx).content)
    assert artifact["result"] == result
    restored = restore(fs, ctx)
    assert restored.checkpoint("chunk:0").load_stage("entity", inputs) == result
    restored.finish()
    assert fs.exists(record["uri"], ctx) is False
    final = SessionManager(get_context_fs=lambda: fs, session_state_bridge_enabled=False)
    buffer = final.get_or_create(ctx.session_id, ctx)
    final.enable_extraction_recovery(ctx.session_id, ctx, fs)
    assert buffer.request_completed("source-request")
    assert buffer.messages == []
    assert "memory_extraction_recovery" not in buffer.runtime_extensions


def test_refreshed_inputs_save_their_own_complete_result_and_release_all_artifacts():
    _, fs, _ = make_runtime()
    ctx = request_context("stage-refreshed")
    _, journal = journal_for(fs, ctx)
    for version in (7, 8):
        inputs = {"prompt": "Painting preferences", "read_versions": {"painting": version}}
        result = {"overview": f"Version {version} with the complete painting facts"}
        journal.save_stage("summary", inputs, result)
        resumed = restore(fs, ctx)
        assert resumed.load_stage("summary", inputs) == result
    uris = list(journal.batch["stage_result_uris"])
    assert len(uris) == 2
    for version in (7, 8):
        inputs = {"prompt": "Painting preferences", "read_versions": {"painting": version}}
        assert resumed.load_stage("summary", inputs) == {
            "overview": f"Version {version} with the complete painting facts"}
    assert resumed.has_stage("summary") is True
    journal.finish()
    assert [fs.exists(uri, ctx) for uri in uris] == [False, False]


def test_identity_repreparation_preserves_source_peer_slots_and_committed_receipts():
    _, fs, _ = make_runtime()
    ctx = request_context("stage-identity-repreparation")
    _, journal = journal_for(fs, ctx)
    source = {key: deepcopy(journal.batch[key]) for key in
              ("owner", "archive_id", "snapshot", "messages", "message_ids", "request_source_ids")}
    candidate = memory("lake_sunrise", object_name="Lake sunrise painting",
                       content="Melanie painted a lake sunrise in 2022.")
    inputs = {"prompt": source["messages"][0]["content"], "read_versions": {}}
    result = {"object_name": candidate.object_name, "content": candidate.content}
    current = journal.checkpoint("chunk:1")
    peer = journal.checkpoint("chunk:10")
    accepted = journal.checkpoint("chunk:0")
    for checkpoint in (current, peer, accepted):
        checkpoint.save_stage("native_extraction", inputs, result)
        checkpoint.save_drafts([candidate], source["messages"])
    current.save_stage("span_identification", inputs, {"spans": [{"start": 0, "end": 0}]})
    current.save_stage("node_processing", inputs, {"action": "create", "object_name": candidate.object_name})
    committed = accepted.accept([candidate], 1, 0)
    plans = MemoryWriteAPI(fs, NoModel(), directory_summary_enabled=False)._write_checkpointed_candidates(
        committed, ctx, accepted)
    committed_record = deepcopy(accepted.record)
    receipts = deepcopy(fs.read_node(plans[0].target_uri, ctx).metadata[RECEIPTS_KEY])
    old_uris = list(journal.batch["stage_result_uris"])

    current.invalidate_unaccepted_preparation()
    resumed = restore(fs, ctx)
    refreshed = resumed.checkpoint("chunk:1")
    assert [refreshed.has_stage(name) for name in
            ("span_identification", "native_extraction", "node_processing")] == [False, False, False]
    assert refreshed.load_drafts() is None
    assert {key: resumed.batch[key] for key in source} == source
    assert resumed.checkpoint("chunk:10").load_stage("native_extraction", inputs) == result
    assert resumed.checkpoint("chunk:10").load_drafts() == [candidate]
    assert resumed.checkpoint("chunk:0").record == committed_record
    assert resumed.checkpoint("chunk:0").load() == (committed, 1, 0)
    assert resumed.checkpoint("chunk:0").load_stage("native_extraction", inputs) == result
    assert fs.read_node(plans[0].target_uri, ctx).metadata[RECEIPTS_KEY] == receipts
    assert resumed.batch["stage_result_uris"] == old_uris
    assert all(fs.exists(uri, ctx) for uri in old_uris)

    current.save_stage("native_extraction", inputs, result)
    current.save_drafts([candidate], source["messages"])
    prepared = restore(fs, ctx).checkpoint("chunk:1")
    assert prepared.load_stage("native_extraction", inputs) == result
    assert prepared.load_drafts() == [candidate]
    assert prepared.journal.batch["stage_result_uris"] == old_uris
    prepared.journal.finish()
    assert [fs.exists(uri, ctx) for uri in old_uris] == [False] * len(old_uris)


def test_nested_collectors_persist_on_owner_while_model_runs():
    owner = threading.get_ident()
    model_started = threading.Event()
    sibling_saved = threading.Event()
    observed = []

    def archive_ready():
        if model_started.is_set():
            observed.append(("archive", threading.get_ident()))
            sibling_saved.set()

    def summary_ready():
        if model_started.is_set():
            observed.append(("summary", threading.get_ident()))

    def model():
        model_started.set()
        collect_stage_results()
        assert sibling_saved.wait(3)
        return {"thread": threading.get_ident(), "complete": True}

    with stage_collector_scope(archive_ready), stage_collector_scope(summary_ready):
        result = run_model_call(model)
    assert result["complete"] is True
    assert result["thread"] != owner
    assert {name for name, ident in observed} == {"archive", "summary"}
    assert all(ident == owner for _, ident in observed)


def test_model_configuration_signature_preserves_equivalent_provider_identity():
    class Model:
        def __init__(self, version):
            self._model = "same-model"
            self._base_url = "https://provider.example/v1"
            self._max_tokens = version
            self._extra_body = {"thinking": {"type": "disabled"}}

    assert model_signature(Model(6144)) == model_signature(Model(6144))
    assert len({model_signature(Model(value)) for value in (6144, 8192)}) == 2


def test_worker_context_returns_pure_model_result_and_owner_collects_it():
    owner = threading.get_ident()
    calls = []
    with stage_collector_scope(lambda: calls.append(threading.get_ident())):
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(copy_context().run, run_model_call,
                                 lambda: {"thread": threading.get_ident()}).result()
        collect_stage_results()
    assert result["thread"] != owner
    assert calls == [owner]
