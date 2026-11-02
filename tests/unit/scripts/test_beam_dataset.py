"""Portable BEAM data/ingestion contracts; no Docker or provider access."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import beam_dataset as data
from scripts import ingest_beam_fresh as ingest


def pairs():
    return [[{"id": "u1", "role": "user", "content": "Weather API"},
             {"id": "a1", "role": "assistant", "content": "API v2"}]]


def questions():
    return {category: [{"question": f"{category} {i}", "rubric": ["fact one"]}
                       for i in range(2)] for category in data.CATEGORIES}


def test_source_reader_never_requests_held_out_columns(monkeypatch):
    reader = Mock(return_value={"conversation_id": "one", "chat": pairs()})
    monkeypatch.setattr(data, "_row", reader)
    assert data.load_source_pairs(Path("input.parquet"), 5) == ("one", pairs())
    reader.assert_called_once_with(Path("input.parquet"), 5, ["conversation_id", "chat"])


@pytest.mark.parametrize("serialized", [False, True])
def test_all_official_categories_and_dynamic_rubrics(monkeypatch, serialized):
    probes = questions()
    probes[data.CATEGORIES[0]][0]["rubric"].append("fact two")
    monkeypatch.setattr(data, "_row", lambda *_: {
        "conversation_id": 8, "probing_questions": repr(probes) if serialized else probes})
    identity, actual = data.load_questions(Path("input"), 7)
    assert identity == 8
    assert list(actual) == list(data.CATEGORIES)
    assert sum(len(q["rubric"]) for group in actual.values() for q in group) == 21


@pytest.mark.parametrize("chat", [[], [[{"role": "user"}]],
    [[{"id": "x", "role": "assistant", "content": "a"}, {"id": "y", "role": "user", "content": "u"}]],
    [[{"id": "x", "role": "user", "content": ""}, {"id": "y", "role": "assistant", "content": "a"}]]])
def test_invalid_source_fails_before_runtime(monkeypatch, chat):
    monkeypatch.setattr(data, "_row", lambda *_: {"conversation_id": 1, "chat": chat})
    with pytest.raises(ValueError):
        data.load_source_pairs(Path("input"))


def test_missing_probe_category_is_rejected(monkeypatch):
    probes = questions()
    probes.pop(data.CATEGORIES[0])
    monkeypatch.setattr(data, "_row", lambda *_: {"conversation_id": 1, "probing_questions": probes})
    with pytest.raises(ValueError, match="ten official"):
        data.load_questions(Path("input"))


def test_real_parquet_column_projection_and_any_conversation(tmp_path):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "sources.parquet"
    parquet.write_table(arrow.Table.from_pylist([
        {"conversation_id": "1", "chat": pairs()},
        {"conversation_id": "2", "chat": pairs() + pairs()},
    ]), path)
    # File deliberately has no held-out questions column at all.
    assert len(data.load_source_pairs(path, 1)[1]) == 2
    with pytest.raises(ValueError, match="missing"):
        data.load_source_pairs(path, 2)


@pytest.mark.parametrize("index", [-1, True, "0"])
def test_invalid_indices_do_not_open_parquet(index):
    with pytest.raises(ValueError, match="non-negative"):
        data._row(Path("missing"), index, ["chat"])


def test_dry_run_never_calls_docker_http_or_reads_probes(monkeypatch, tmp_path):
    source = tmp_path / "input.parquet"
    source.write_bytes(b"not accessed beyond hash: mocked projected reader")
    monkeypatch.setattr(ingest, "source_pairs", lambda *_: (3, pairs()))
    operation = Mock(side_effect=AssertionError("live action"))
    monkeypatch.setattr(ingest.subprocess, "run", operation)
    monkeypatch.setattr(ingest, "post", operation)
    args = SimpleNamespace(run_id="beam-smoke-1234", database="beam_eval_smoke_1234",
                           parquet=source, conversation_index=2, source_root=tmp_path,
                           execute=False)
    report = ingest.ingest(args)
    assert report["status"] == "planned"
    assert report["source_pairs"] == 1
    assert report["completed_pairs"] == 0
    assert report["read_columns"] == ["conversation_id", "chat"]
    assert not operation.called


def test_http_headers_use_env_not_saved_credentials(monkeypatch):
    monkeypatch.setenv("BEAM_TEST_KEY", "private-token")
    args = SimpleNamespace(account="a", user="u", memory_api_key_env="BEAM_TEST_KEY")
    assert ingest.request_headers(args) == {
        "X-API-Key": "private-token", "X-Account-ID": "a", "X-User-ID": "u", "X-Agent-ID": "main"}


def test_inventory_tolerates_graph_disabled_and_checks_failed_jobs():
    db = Mock()
    db.sql.side_effect = ["f", json.dumps({"nodes": 1, "edges": 0})]
    assert ingest.inventory(db) == {"nodes": 1, "edges": 0}
    query = db.sql.call_args_list[-1].args[0]
    assert "upper(status)='FAILED'" in query
    assert "FROM edge_projection_jobs WHERE" not in query


@pytest.mark.parametrize("body", [{"ok": False}, {"status": "needs_confirmation"},
    {"status": "failed"}, {"error": "private error"}, {"retryable": False}])
def test_http_200_does_not_hide_lifecycle_failure(monkeypatch, body):
    requests = pytest.importorskip("requests")
    response = Mock()
    response.json.return_value = body
    monkeypatch.setattr(requests, "post", Mock(return_value=response))
    args = SimpleNamespace(account="a", user="u", memory_api_key_env="MISSING_BEAM_KEY",
                           memory_url="http://127.0.0.1:8090", timeout=1)
    with pytest.raises(RuntimeError, match="reported failure") as exc:
        ingest.post(args, "after_turn", {})
    assert "private error" not in str(exc.value)


def test_complete_ingestion_preserves_source_payload_and_freeze_manifest(monkeypatch, tmp_path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"mocked parquet")
    preflight = tmp_path / "preflight.json"
    fingerprint, runtime = {"tables": {}}, {"config": {"enable_cache": False}}
    preflight.write_text(json.dumps({"corpus_origin": "fresh", "run_id": "beam-test-1234",
        "database": "beam_eval_fixture_1234", "inventory": fingerprint, "runtime": runtime}))
    monkeypatch.setattr(ingest, "source_pairs", lambda *_: (7, pairs()))
    monkeypatch.setattr(ingest, "require_eval_container", lambda *_: {"State": {"Running": False}})
    db = Mock(spec=["assert_idle", "fingerprint"])
    db.fingerprint.return_value = fingerprint
    monkeypatch.setattr(ingest, "SnapshotDB", lambda *_: db)
    monkeypatch.setattr(ingest, "check_runtime", lambda *_: runtime)
    monkeypatch.setattr(ingest, "wait_health", lambda *_: None)
    monkeypatch.setattr(ingest.time, "sleep", lambda *_: None)
    monkeypatch.setattr(ingest.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout="", stderr=""))
    captured = []

    def fake_post(_args, method, payload):
        captured.append((method, payload))
        return ({"idle": True} if method == "wait_until_idle" else
                {"disposed": True, "reason": "no_pending_messages"} if method == "dispose" else
                {"ok": True, "status": "completed"})

    monkeypatch.setattr(ingest, "post", fake_post)
    counts = {"nodes": 2, "edges": 1, "outbox_failed": 0, "outbox": 0, "projection_pending": 0}
    monkeypatch.setattr(ingest, "inventory", lambda *_: counts)
    args = SimpleNamespace(run_id="beam-test-1234", database="beam_eval_fixture_1234", parquet=source,
                           conversation_index=6, source_root=tmp_path, execute=True, preflight=preflight,
                           memory_container="memory", postgres_container="postgres", memory_url="http://127.0.0.1:8090",
                           account="a", user="u", output_dir=tmp_path / "ingest", timeout=5)
    result = ingest.ingest(args)
    assert result["status"] == "ingested_unfrozen"
    assert result["source_pairs"] == result["completed_pairs"] == 1
    assert result["parquet_sha256"] == ingest.digest(source)
    assert result["preflight_sha256"] == ingest.digest(preflight)
    assert json.loads((args.output_dir / "ingestion.json").read_text()) == result
    after_turn = next(payload for method, payload in captured if method == "after_turn")
    assert after_turn["wait"] is True
    assert after_turn["messages"] == [
        {"role": m["role"], "content": m["content"], "source_chat_id": m["id"]} for m in pairs()[0]]
    assert [method for method, _ in captured] == ["after_turn", "wait_until_idle", "dispose", "wait_until_idle"]
    assert (args.output_dir / "turn-001.completed.json").is_file()
