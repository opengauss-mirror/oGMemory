"""Offline regressions for benchmark runtime settings; no model or DB calls."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from m3.pipeline.spec import StageSpec
from m3.pipeline.stages import qa_stage
from m3.qa import parse_react_qa_args, run_react_qa


@pytest.mark.parametrize("key,value", [
    ("timeout_seconds", 300.0),
    ("no_gain_limit", 0),
    ("llm_model", "local-served-model"),
    ("llm_base_url", "http://local-model.invalid/v1"),
    ("llm_api_key", "test-only"),
    ("embedding_model", "local-embedding"),
    ("embedding_base_url", "http://local-embedding.invalid/v1"),
    ("embedding_api_key", "test-only"),
    ("temporal_policy", "strict"),
    ("lookahead_seconds", 3.0),
])
def test_qa_stage_forwards_runtime_overrides(tmp_path, monkeypatch, key, value):
    runner = Mock(return_value=0)
    monkeypatch.setattr("m3.qa.run_react_qa", runner)
    context = SimpleNamespace(
        work_dir=tmp_path, graph_store="memory", config_path=tmp_path / "config.yaml",
        connection_string=None, graph=Mock(), account_id="custom-account",
        artifact=lambda name: tmp_path / name,
    )
    qa_stage(context, StageSpec(id="qa", type="qa", params={key: value}))
    args = runner.call_args.args[0]
    assert getattr(args, key) == value
    assert args.account_id == context.account_id


def test_react_qa_uses_requested_account(tmp_path, monkeypatch):
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"v": {"qa_list": [
        {"question_id": "q", "question": "Where?", "answer": "Here"},
    ]}}))
    args = parse_react_qa_args([
        "--annotations", str(annotations),
        "--output-jsonl", str(tmp_path / "qa.jsonl"),
        "--account-id", "custom-account",
    ])
    adapter = Mock()
    adapter.run.return_value = SimpleNamespace(
        output=["Here"], completed=True, iterations=1, finish_reason="done",
    )
    monkeypatch.setattr("m3.control_adapter.OGMemoryControlAdapter", Mock(return_value=adapter))
    assert run_react_qa(args, graph=Mock(), embedder=Mock(), llm=Mock()) == 0
    assert adapter.run.call_args.args[0].account_id == "custom-account"


def test_partial_qa_failure_returns_nonzero_and_keeps_both_records(tmp_path, monkeypatch):
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"v": {"qa_list": [
        {"question_id": "q1", "question": "Where?", "answer": "Here"},
        {"question_id": "q2", "question": "When?", "answer": "Now"},
    ]}}))
    output = tmp_path / "qa.jsonl"
    args = parse_react_qa_args([
        "--annotations", str(annotations), "--output-jsonl", str(output),
    ])
    adapter = Mock()
    adapter.run.side_effect = [
        SimpleNamespace(output=["Here"], completed=True, iterations=1, finish_reason="done"),
        SimpleNamespace(output=[], completed=False, iterations=3,
                        finish_reason="llm_error", llm_error=RuntimeError("GPU out of memory")),
    ]
    monkeypatch.setattr("m3.control_adapter.OGMemoryControlAdapter", Mock(return_value=adapter))
    assert run_react_qa(args, graph=Mock(), embedder=Mock(), llm=Mock()) == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["predicted_answer"] == "Here" and not rows[0]["error"]
    assert rows[1]["predicted_answer"] == ""
    assert "out of memory" in rows[1]["error"]


def test_qa_only_resume_preserves_account(tmp_path, monkeypatch):
    from m3 import resume

    runner = Mock(return_value=0)
    monkeypatch.setattr("m3.qa.run_react_qa", runner)
    monkeypatch.setattr(resume, "build_graph_service", lambda *args: (Mock(), "memory"))
    resume.run_qa_only(
        graph_store="memory", connection_string=None, account_id="custom-account",
        config_path=tmp_path / "config.yaml", annotations_path=tmp_path / "ann.json",
        video_id="v", work_dir=tmp_path,
    )
    assert runner.call_args.args[0].account_id == "custom-account"
    assert runner.call_args.args[0].temporal_policy == "current_clip"
    assert "graph" not in runner.call_args.kwargs


def test_builtin_judge_keeps_config_and_does_not_limit_questions(tmp_path, monkeypatch):
    from scripts import run_m3_bench_subset as driver

    args = driver.parse_args([
        "--config", str(tmp_path / "custom.yaml"), "--limit", "1",
    ])
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(driver.subprocess, "run", run)
    monkeypatch.setattr(driver, "resolve_subprocess_env", lambda args: {})
    driver.run_judge(
        annotations=tmp_path / "annotations.json", merged_jsonl=tmp_path / "qa.jsonl",
        output_dir=tmp_path / "judge", args=args, subset=["v"],
    )
    cmd = run.call_args.args[0]
    assert "--config" in cmd
    assert cmd[cmd.index("--config") + 1] == str(tmp_path / "custom.yaml")
    assert "--limit" not in cmd
    assert cmd[cmd.index("--video") + 1] == "v"
