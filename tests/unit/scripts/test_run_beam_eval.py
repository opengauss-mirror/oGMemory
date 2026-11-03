from pathlib import Path
from unittest.mock import Mock
import json

import pytest

from scripts import run_beam_eval as launcher


def config(tmp_path):
    return dict(parquet=str(tmp_path / "input.parquet"), output_root=str(tmp_path / "run"),
                database="beam_eval_t1_unique_1234", memory_container="memory", postgres_container="pg",
                agent_container="agent", run_id="beam-t1-unique-1234", account="eval", user="u",
                memory_url="http://127.0.0.1:18800", base_url="http://127.0.0.1:20800")


def test_setup_does_not_depend_on_any_other_checkout(tmp_path):
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(config(tmp_path)))
    actual = launcher.load_setup(path)
    assert Path(actual["source_root"]) == launcher.ROOT
    assert actual["conversation_index"] == 0


@pytest.mark.parametrize("field,value", [
    ("source_root", "/some/other/checkout"), ("parquet", "relative.parquet"),
    ("output_root", str(launcher.ROOT / "results")), ("conversation_index", True),
    ("variant", "unapproved"), ("memory_container", ""),
    ("variant", "graph-conflict"),
    ("judge_base_url", "https://key:secret@example.com"),
])
def test_bad_setup_fails_closed(tmp_path, field, value):
    row = config(tmp_path)
    row[field] = value
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        launcher.load_setup(path)


def test_plan_default_does_not_spawn_process(tmp_path, monkeypatch, capsys):
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(config(tmp_path)))
    invoke = Mock(side_effect=AssertionError("live execution"))
    monkeypatch.setattr(launcher.subprocess, "run", invoke)
    launcher.main(["qa", "--setup", str(path)])
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "planned"
    assert "--execute" not in plan["command"]
    assert not invoke.called
    assert not (tmp_path / "run").exists()


def test_execute_routes_to_packaged_module_and_freeze_completion(tmp_path):
    row = config(tmp_path)
    row.update(source_root=str(launcher.ROOT), conversation_index=1)
    command = launcher.stage_command("freeze", row, execute=True)
    assert command[1:4] == ["-m", "scripts.run_stage8r_beam_clean_eval", "freeze"]
    assert command[-1] == "--execute"
    assert command[command.index("--ingestion-manifest") + 1] == str(tmp_path / "run/ingest/ingestion.json")


def test_qa_uses_same_frozen_corpus_but_never_forces_sampling(tmp_path):
    row = config(tmp_path)
    row.update(source_root=str(launcher.ROOT), conversation_index=0, variant="baseline")
    command = launcher.stage_command("qa", row)
    assert command[command.index("--snapshot") + 1] == str(tmp_path / "run/freeze")
    assert command[command.index("--variant") + 1] == "baseline"
    assert "--temperature" not in command


def test_prepare_agent_routes_only_explicit_identity_and_endpoints(tmp_path):
    row = config(tmp_path)
    command = launcher.stage_command("prepare-agent", row, execute=True)
    assert command[1:3] == ["-m", "scripts.prepare_beam_agent"]
    assert command[-1] == "--execute" and "--temperature" not in command
    assert "--database" not in command


def test_qa_requires_prepared_agent_receipt_before_any_docker_start(tmp_path, monkeypatch):
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(config(tmp_path)))
    invoke = Mock(side_effect=AssertionError("unexpected process"))
    monkeypatch.setattr(launcher.subprocess, "run", invoke)
    with pytest.raises(RuntimeError, match="prepare-agent"):
        launcher.main(["qa", "--setup", str(path), "--execute"])
    invoke.assert_not_called()


@pytest.mark.parametrize("changed", [False, True])
def test_qa_startup_must_preserve_prepared_hash(tmp_path, monkeypatch, changed):
    from scripts import beam_eval_safety as safety
    from scripts import run_stage8r_beam_clean_eval as native
    row = config(tmp_path)
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(row))
    receipt = tmp_path / "run/prepared-agent/preparation.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "prepared", "run_id": row["run_id"],
        "container_id": "agent-id", "prepared_config_sha256": "expected"}))
    monkeypatch.setattr(safety, "require_eval_container", lambda *_: {"Id": "agent-id"})
    monkeypatch.setattr(safety, "agent_preflight", lambda *_a, **_k: (
        {"config_sha256": "different" if changed else "expected"}, {}))
    monkeypatch.setattr(native, "wait_health", lambda *_: None)
    invoke = Mock()
    monkeypatch.setattr(launcher.subprocess, "run", invoke)
    if changed:
        with pytest.raises(RuntimeError, match="startup replaced"):
            launcher.main(["qa", "--setup", str(path), "--execute"])
        assert invoke.call_args_list[-1].args[0] == ["docker", "stop", "agent"]
        assert not any("scripts.run_stage8r_beam_clean_eval" in c.args[0] for c in invoke.call_args_list)
    else:
        launcher.main(["qa", "--setup", str(path), "--execute"])
        assert "scripts.run_stage8r_beam_clean_eval" in invoke.call_args_list[-1].args[0]


def test_invalid_offline_execution_combination_before_spawn(tmp_path):
    with pytest.raises(SystemExit):
        launcher.main(["grade", "--setup", str(tmp_path / "missing"),
                       "--execute", "--resume", "--verify-offline"])
