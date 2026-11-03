"""Portable native runner contract checks; provider and Docker calls are fake."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import requests

from scripts import run_stage8r_beam_clean_eval as runner
from scripts import experiment_compose_capture


def arguments(tmp_path, command="run", execute=False):
    values = ["beam", command, "--database", "beam_eval_test_fixture", "--memory-container", "memory",
              "--postgres-container", "postgres", "--run-id", "beam-test", "--account", "a", "--user", "u",
              "--memory-url", "http://127.0.0.1:8090", "--source-root", str(tmp_path / "source"),
              "--output-dir", str(tmp_path / "qa")]
    return values + (["--execute"] if execute else [])


@pytest.mark.parametrize("command", ["preflight", "freeze", "run"])
def test_dry_run_does_not_invoke_docker_or_create_output(tmp_path, monkeypatch, capsys, command):
    monkeypatch.setattr(sys, "argv", arguments(tmp_path, command))
    monkeypatch.setattr(runner, "SnapshotDB", lambda *_: pytest.fail("Docker was used"))
    monkeypatch.setattr(requests, "post", lambda *_a, **_k: pytest.fail("Provider was used"))
    runner.main()
    assert json.loads(capsys.readouterr().out)["status"] == "prepared_offline"
    assert not (tmp_path / "qa").exists()


def make_ingestion(tmp_path, monkeypatch):
    dataset, preflight = tmp_path / "dataset.parquet", tmp_path / "preflight.json"
    dataset.write_bytes(b"fixture dataset")
    preflight.write_text("{}")
    pairs = [[{"role": "user", "content": "source"}, {"role": "assistant", "content": "reply"}]]
    monkeypatch.setattr(runner, "load_source_pairs", lambda *_: ("7", pairs))
    report = {"status": "ingested_unfrozen", "run_id": "beam-test", "database": "beam_eval_test_fixture",
              "conversation_id": "7", "conversation_index": 6, "source_pairs": 1, "completed_pairs": 1,
              "source_sha256": hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest(),
              "parquet_sha256": runner.digest(dataset), "preflight_sha256": runner.digest(preflight)}
    manifest = tmp_path / "ingestion.json"
    manifest.write_text(json.dumps(report))
    args = SimpleNamespace(ingestion_manifest=manifest, parquet=dataset, preflight=preflight,
                           conversation_index=6, run_id="beam-test", database="beam_eval_test_fixture")
    return args, report


def test_freeze_requires_every_source_pair_and_exact_dataset(tmp_path, monkeypatch):
    args, _ = make_ingestion(tmp_path, monkeypatch)
    assert runner.validate_ingestion_manifest(args)["source_pairs"] == 1
    args.parquet.write_bytes(b"different dataset")
    with pytest.raises(RuntimeError, match="source/dataset/preflight binding changed"):
        runner.validate_ingestion_manifest(args)


@pytest.mark.parametrize("changed", ["dataset", "index", "conversation", "missing"])
def test_qa_rejects_mismatched_snapshot_without_restoring(tmp_path, monkeypatch, changed):
    args, _ = make_ingestion(tmp_path, monkeypatch)
    binding = runner.validate_ingestion_manifest(args)
    snapshot = {"ingestion": binding}
    runner.validate_snapshot_dataset(snapshot, args, "7")
    if changed == "dataset":
        args.parquet.write_bytes(b"changed dataset")
    elif changed == "index":
        args.conversation_index = 7
    elif changed == "conversation":
        snapshot["ingestion"]["conversation_id"] = "other"
    else:
        del snapshot["ingestion"]
    with pytest.raises(RuntimeError):
        runner.validate_snapshot_dataset(snapshot, args, "7")


@pytest.mark.parametrize("field,value", [("status", "INVALID"), ("completed_pairs", 0),
                                        ("run_id", "other-run"), ("conversation_index", 1)])
def test_freeze_rejects_partial_or_misbound_ingestion(tmp_path, monkeypatch, field, value):
    args, report = make_ingestion(tmp_path, monkeypatch)
    report[field] = value
    args.ingestion_manifest.write_text(json.dumps(report))
    with pytest.raises(RuntimeError):
        runner.validate_ingestion_manifest(args)


def test_native_qa_restores_each_question_and_preserves_original_prefix(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    runtime, inventory = {"config": {"enable_cache": False}}, {"tables": {}}
    dataset = tmp_path / "dataset.parquet"
    dataset.write_bytes(b"fixture")
    (snapshot / "snapshot.json").write_text(json.dumps({
        "run_id": "beam-test", "database": "beam_eval_test_fixture", "runtime": runtime, "inventory": inventory,
        "ingestion": {"parquet_sha256": runner.digest(dataset), "conversation_index": 0,
                      "conversation_id": "7", "source_pairs": 94}}))
    probes = {category: [{"question": f"{category} question {i}", "rubric": ["original rubric"]}
                         for i in (1, 2)] for category in runner.CATEGORIES}
    monkeypatch.setattr(runner, "load_questions", lambda *_: ("7", probes))
    restores, starts, queries, stopped = [], [], [], []

    class FakeDB:
        def __init__(self, *_):
            pass

        def restore(self, path):
            restores.append(path)
            return {"inventory_sha256": "verified", "dump_sha256": "same"}

        def fingerprint(self):
            return inventory

    monkeypatch.setattr(runner, "SnapshotDB", FakeDB)
    monkeypatch.setattr(runner, "require_eval_container", lambda *_: {"HostConfig": {"NetworkMode": "host"}})
    monkeypatch.setattr(runner, "validate_agent_mounts", lambda *_: None)
    monkeypatch.setattr(runner, "agent_preflight", lambda *_a, **_k: ({"model": "unchanged"}, {}))
    monkeypatch.setattr(runner, "docker_text", lambda *_: json.dumps({"gateway": {"port": 18789}}))
    monkeypatch.setattr(runner, "stop_agent", lambda *_: stopped.append("agent"))
    monkeypatch.setattr(runner, "stop_memory", lambda *_: stopped.append("memory"))
    monkeypatch.setattr(runner, "check_runtime", lambda *_: runtime)
    monkeypatch.setattr(runner, "wait_health", lambda *_: None)
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **_: starts.append(command))
    monkeypatch.setattr(experiment_compose_capture, "capture_compose", lambda *_: json.dumps({
        "stats": {}, "retrievedEvidence": "first actual evidence"}))

    def reply(url, **kwargs):
        queries.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "first answer"}]}]})

    monkeypatch.setattr(requests, "post", reply)
    monkeypatch.setattr(runner, "collect_session", lambda _c, _s, question: "\n".join(
        json.dumps({"type": "message", "message": m}) for m in [
            {"role": "user", "content": question}, {"role": "assistant", "content": "first answer"}]))
    monkeypatch.setenv("BEAM_AGENT_TOKEN", "not-a-real-token")
    argv = arguments(tmp_path, execute=True) + ["--snapshot", str(snapshot), "--parquet", str(dataset),
            "--agent-container", "agent", "--base-url", "http://127.0.0.1:18789", "--variant", "graph"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main()
    result = json.loads((tmp_path / "qa/run.json").read_text())
    assert result["status"] == "completed_unjudged"
    assert result["variant"] == "graph"
    assert len(restores) == 21  # One complete restore per question plus final cleanup.
    assert len(queries) == 20
    expected = [f"Answer the question directly: {p['question']}"
                for category in runner.CATEGORIES for p in probes[category]]
    assert [kwargs["json"]["input"] for _, kwargs in queries] == expected
    assert all(url.endswith("/v1/responses") for url, _ in queries)
    answers = json.loads((tmp_path / "qa/answers.json").read_text())
    assert all(q["response"] == "first answer" for rows in answers.values() for q in rows)
    assert all(q["audit"]["structured_tool_calls"] == 0 for q in result["questions"])
    assert stopped[-2:] == ["agent", "memory"]
