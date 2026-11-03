"""Clean Dev has no graph configuration or projection queue: do not require one."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import beam_eval_safety as safety
from scripts import run_stage8r_beam_clean_eval as runner
from scripts import run_beam_eval as launcher
from tests.unit.scripts.test_beam_clean_eval import arguments


BASE_TABLES = {"context_nodes", "vector_index", "relation_edges", "session_archives", "outbox_events"}


def inventory(projector=False):
    names = BASE_TABLES | ({"edge_projection_jobs"} if projector else set())
    return {"tables": {"public." + name: {"count": 0, "rows_md5": "empty"} for name in names},
            "sequences": {"sequence": "1|f"}, "schema_sha256": "strict-original-schema"}


@pytest.mark.parametrize("graph", [None, "none", "sql"])
def test_native_preflight_accepts_only_the_schema_needed_by_its_real_configuration(
        tmp_path, monkeypatch, graph):
    config = {"graph_backend": graph}
    corpus = inventory(projector=graph == "sql")
    monkeypatch.setattr(runner, "SnapshotDB", lambda *_: SimpleNamespace(
        assert_idle=lambda: None, fingerprint=lambda: deepcopy(corpus)))
    monkeypatch.setattr(runner, "check_runtime", lambda *_: {"config": config})
    monkeypatch.setattr(runner, "stop_memory", lambda *_: None)
    monkeypatch.setattr("sys.argv", arguments(tmp_path, command="preflight", execute=True))
    runner.main()
    report = json.loads((tmp_path / "qa/preflight.json").read_text())
    assert report["inventory"] == corpus  # No table/schema/sequence filtering.


@pytest.mark.parametrize("graph", [None, "none", "sql"])
def test_freeze_keeps_complete_inventory_without_requiring_absent_graph_features(
        tmp_path, monkeypatch, graph):
    db = object.__new__(safety.SnapshotDB)
    db.database, db.container_id, db.container, db.user = "beam_eval_native_dev", "pg-id", "pg", "postgres"
    corpus = inventory(projector=graph == "sql")
    corpus["tables"]["public.unrelated_native_table"] = {"count": 5, "rows_md5": "all-five-rows"}
    db.assert_idle = lambda: None
    db.fingerprint = lambda: deepcopy(corpus)
    db.sql = lambda query, **_: json.dumps({"provider": "c", "encoding": "UTF8", "collate": "C", "ctype": "C"}) if "json_build_object" in query else "0"

    def dump(command, **kwargs):
        assert command[:4] == ["docker", "exec", "pg", "pg_dump"]
        kwargs["stdout"].write(b"exact dump")
    monkeypatch.setattr(safety.subprocess, "run", dump)
    report = db.freeze(tmp_path / "freeze", {"runtime": {"config": {"graph_backend": graph}}})
    assert report["inventory"] == corpus


@pytest.mark.parametrize("missing", sorted(BASE_TABLES))
def test_missing_core_table_is_never_excused_for_clean_dev(tmp_path, monkeypatch, missing):
    corpus = inventory()
    del corpus["tables"]["public." + missing]
    monkeypatch.setattr(runner, "SnapshotDB", lambda *_: SimpleNamespace(
        assert_idle=lambda: None, fingerprint=lambda: corpus))
    monkeypatch.setattr(runner, "check_runtime", lambda *_: {"config": {"graph_backend": None}})
    monkeypatch.setattr(runner, "stop_memory", lambda *_: None)
    monkeypatch.setattr("sys.argv", arguments(tmp_path, command="preflight", execute=True))
    with pytest.raises(RuntimeError, match="Fresh corpus"):
        runner.main()


def test_graph_enabled_missing_projection_table_still_fails_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SnapshotDB", lambda *_: SimpleNamespace(
        assert_idle=lambda: None, fingerprint=lambda: inventory()))
    monkeypatch.setattr(runner, "check_runtime", lambda *_: {"config": {"graph_backend": "sql"}})
    monkeypatch.setattr(runner, "stop_memory", lambda *_: None)
    monkeypatch.setattr("sys.argv", arguments(tmp_path, command="preflight", execute=True))
    with pytest.raises(RuntimeError, match="Fresh corpus"):
        runner.main()


def test_clean_dev_defaults_to_baseline_label_not_a_graph_claim(tmp_path):
    from tests.unit.scripts.test_run_beam_eval import config
    row = config(tmp_path)
    row["source_root"] = str(launcher.ROOT)
    row["conversation_index"] = 0
    command = launcher.stage_command("qa", row)
    assert command[command.index("--variant") + 1] == "baseline"


def test_unknown_backend_is_not_silently_treated_as_graph_off():
    with pytest.raises(RuntimeError, match="Unsupported graph backend"):
        safety.required_corpus_tables({"graph_backend": "unrecognized"})


def test_optional_projection_table_remains_in_inventory_and_must_be_settled(tmp_path):
    db = object.__new__(safety.SnapshotDB)
    db.database, db.container_id = "beam_eval_native_dev", "pg-id"
    db.assert_idle = lambda: None
    db.fingerprint = lambda: inventory(projector=True)
    db.sql = lambda *a, **kw: "1"
    with pytest.raises(RuntimeError, match="Projection jobs have not settled"):
        db.freeze(tmp_path / "freeze", {"runtime": {"config": {"graph_backend": None}}})
