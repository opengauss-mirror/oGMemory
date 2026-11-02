"""Offline safety checks: no Docker daemon, provider calls or production DB."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

from scripts import beam_eval_safety as safety
from scripts.beam_capture import sitecustomize as capture
from scripts import experiment_compose_capture as compose_capture


@pytest.mark.parametrize("dual_pass,node_processing,expected", [
    (True, False, [None, 0]), (False, False, [None]), (True, True, [None])])
def test_native_policy_receipt_respects_actual_extraction_mode_without_provider_construction(
        dual_pass, node_processing, expected):
    from providers.unified_config import OgMemConfig
    policy = safety.native_policy_manifest(OgMemConfig(
        dual_pass_enabled=dual_pass, node_processing_enabled=node_processing, llm_temperature=0.7))
    assert policy["extraction_temperature_arguments"] == expected
    assert policy["extraction_effective_temperatures"] == [0.7 if t is None else t for t in expected]
    assert policy["after_turn_threshold"] == 400 and policy["recall_score_gap_ratio"] == .10


@pytest.mark.parametrize("database", ["postgres", "production", "beam_eval_x", "beam_eval_valid_name;DROP"])
def test_non_disposable_database_rejected_before_docker(monkeypatch, database):
    monkeypatch.setattr(safety, "inspect_container", lambda *_: pytest.fail("Docker was called"))
    with pytest.raises(RuntimeError, match="Refusing non-disposable"):
        safety.SnapshotDB("postgres", database)


def test_container_labels_must_match_run_and_role(monkeypatch):
    monkeypatch.setattr(safety, "inspect_container", lambda *_: {
        "Config": {"Labels": {"anttrail.eval.run": "other", "anttrail.eval.role": "memory"}}})
    with pytest.raises(RuntimeError, match="not labelled"):
        safety.require_eval_container("test-memory", "beam-run", "memory")


@pytest.mark.parametrize("mount", ["/tmp/beam/answers", "/tmp/beam", "/tmp/beam/answers/q1"])
def test_agent_cannot_mount_answers_or_any_parent_child(mount):
    info = {"Mounts": [{"Type": "bind", "Source": mount}]}
    with pytest.raises(RuntimeError, match="Answering container can access"):
        safety.validate_agent_mounts(info, [Path("/tmp/beam/answers")])


def test_agent_can_mount_its_own_isolated_config():
    safety.validate_agent_mounts({"Mounts": [{"Type": "bind", "Source": "/tmp/agent-config"}]},
                                 [Path("/tmp/beam/answers")])


def test_snapshot_dump_tampering_rejected_before_drop(tmp_path):
    db = object.__new__(safety.SnapshotDB)
    db.database, db.container_id = "beam_eval_test_valid", "postgres-id"
    (tmp_path / "snapshot.dump").write_bytes(b"changed dump")
    (tmp_path / "snapshot.json").write_text(json.dumps({
        "protocol": safety.PROTOCOL, "database": db.database,
        "postgres_container_id": db.container_id, "dump_sha256": "old-hash"}))
    db.sql = lambda *_args, **_kwargs: pytest.fail("SQL was called")
    with pytest.raises(RuntimeError, match="Snapshot dump changed"):
        db.restore(tmp_path)


def test_private_first_compose_is_preserved_without_changing_object(tmp_path):
    identity = str(uuid.uuid4())
    value = {"retrievedEvidence": "original memory", "stats": {"total": 1}}
    capture.capture_first_compose({"sessionId": identity}, value, tmp_path)
    path = tmp_path / (identity + ".json")
    first = path.read_bytes()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    capture.capture_first_compose({"sessionId": identity}, {"other": "later"}, tmp_path)
    assert path.read_bytes() == first
    assert json.loads(first) == value


@pytest.mark.parametrize("identity", ["../escape", "not-a-uuid", None, "-" * 36])
def test_non_qa_session_ids_are_not_written(tmp_path, identity):
    capture.capture_first_compose({"sessionId": identity}, {}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_passive_hook_calls_original_once_and_returns_exact_object(monkeypatch):
    calls, writes = [], []
    result = {"retrievedEvidence": "same object", "stats": {}}

    class Service:
        def compose(self, params):
            calls.append(params)
            return result

    monkeypatch.setitem(sys.modules, "server.memory_service", SimpleNamespace(MemoryService=Service))
    monkeypatch.setattr(capture, "capture_first_compose", lambda p, r: writes.append((p, r)))
    capture.install()
    first_wrapper = Service.compose
    capture.install()
    assert Service.compose is first_wrapper
    params = {"sessionId": str(uuid.uuid4())}
    assert Service().compose(params) is result
    assert calls == [params]
    assert writes == [(params, result)]


def test_capture_bootstraps_in_a_cold_python_process():
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, "-c",
        "from server.memory_service import MemoryService; "
        "assert getattr(MemoryService.compose, '_beam_passive_capture', False); print('capture-ready')"],
        cwd=root, env={**os.environ, "BEAM_COMPOSE_CAPTURE": "1",
            "PYTHONPATH": str(root / "scripts/beam_capture") + os.pathsep + str(root)},
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "Error in sitecustomize" not in result.stderr
    assert "capture-ready" in result.stdout


@pytest.mark.parametrize("tool_message", [
    {"role": "tool", "content": "tool result"},
    {"role": "assistant", "tool_calls": [{"name": "read_file"}]},
    {"role": "assistant", "content": [{"type": "toolCall", "name": "read_file"}]},
])
def test_structured_tools_invalidate_answer(tool_message):
    messages = [{"role": "user", "content": "question?"},
                {"role": "assistant", "content": "answer"}, tool_message]
    session = "\n".join(json.dumps({"type": "message", "message": m}) for m in messages)
    with pytest.raises(RuntimeError, match="structured tool calls"):
        safety.audit_session(session, "question?")


def test_metadata_is_not_proof_that_question_was_asked():
    session = json.dumps({"type": "message", "question": "question?", "message": {
        "role": "assistant", "content": "answer"}})
    with pytest.raises(RuntimeError, match="lacks the expected question"):
        safety.audit_session(session, "question?")


def test_capture_unsafe_session_never_invokes_docker(monkeypatch):
    monkeypatch.setattr(compose_capture.subprocess, "run", lambda *_a, **_k: pytest.fail("Docker"))
    with pytest.raises(ValueError, match="Unsafe answering session"):
        compose_capture.capture_compose("agent", "memory", "../answer")


def test_runtime_supports_current_startup_logs_without_legacy_patch_lines(monkeypatch):
    config = {"account_id": "a", "user_id": "u", "opengauss_table_name": "vector_index", "enable_cache": False,
              "storage_backend": "sql"}
    actual = {"targets": [["beam_eval_example_valid", "cluster"], ["beam_eval_example_valid", "cluster"]],
              "config": config, "source_hashes": {"server/example.py": "hash"}}
    monkeypatch.setattr(safety, "docker_text", lambda *_: json.dumps(actual))
    monkeypatch.setattr(safety, "source_hashes", lambda *_: actual["source_hashes"])
    monkeypatch.setattr(safety, "inspect_container", lambda *_: {"State": {"StartedAt": "2026-10-28T00:00:00Z"}})
    monkeypatch.setattr(safety.subprocess, "run", lambda *_a, **_k: SimpleNamespace(
        stdout="[entrypoint] Starting HTTP\n", stderr=""))
    result = safety.runtime_preflight("memory", Path("."), "beam_eval_example_valid", "a", "u")
    assert result["startup_db_targets"] == {}


def test_mismatching_live_storage_and_vector_databases_are_rejected(monkeypatch):
    monkeypatch.setattr(safety, "docker_text", lambda *_: json.dumps({
        "targets": [["beam_eval_example_valid", "cluster"], ["wrong", "cluster"]]}))
    with pytest.raises(RuntimeError, match="Storage and vector SQL targets differ"):
        safety.runtime_preflight("memory", Path("."), "beam_eval_example_valid", "a", "u")
