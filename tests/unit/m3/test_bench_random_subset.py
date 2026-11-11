"""Reproducible sampling must not launch paid jobs during cohort selection."""
import json
import random

import pytest

from scripts import run_m3_bench_subset as driver


def annotations(n=20):
    return {
        f"v{i:02d}": {"video_path": f"v{i:02d}.mp4",
                     "qa_list": [{"question_id": f"q{i}", "timestamp": "01:00"}]}
        for i in range(n)
    }


def sample(data, **kwargs):
    return driver.select_subset(data, [], kwargs.pop("exclude", []), None,
                                sample_size=kwargs.pop("size", 10), **kwargs)


def test_seed_is_reproducible_order_independent_and_rng_is_local():
    data = annotations()
    state = random.getstate()
    first = sample(data, seed=42)
    assert len(first) == len(set(first)) == 10
    assert first == sample(dict(reversed(list(data.items()))), seed=42)
    assert first != sample(data, seed=43)
    assert random.getstate() == state


def test_exclusions_and_empty_qa_do_not_enter_population():
    data = annotations(12)
    data["v00"]["qa_list"] = []
    assert set(sample(data, exclude=["v01"])) == set(data) - {"v00", "v01"}


@pytest.mark.parametrize("kwargs", [
    {"size": 0}, {"size": -1}, {"size": 21}, {"exclude": ["typo"]},
])
def test_invalid_samples_fail(kwargs):
    with pytest.raises(ValueError):
        sample(annotations(), **kwargs)


@pytest.mark.parametrize("argv", [
    ["--selection-only"],
    ["--sample-size", "10", "--videos", "v00"],
    ["--sample-size", "10", "--limit", "10"],
])
def test_incompatible_cli_options_fail(argv):
    with pytest.raises(SystemExit):
        driver.parse_args(argv)


def test_legacy_selection_is_unchanged():
    data = annotations()
    assert driver.select_subset(data, [], ["v00"], 2) == ["v01", "v02"]
    assert driver.select_subset(data, ["v03", "v01"], [], None) == ["v03", "v01"]


def test_manifest_is_pinned_and_cannot_be_overwritten(tmp_path):
    path = tmp_path / "sample_manifest.json"
    data = annotations()
    selected = sample(data)
    original = driver.save_sample_manifest(path, data, selected, seed=42, exclude=[])
    assert original["total_questions"] == 10
    assert driver.save_sample_manifest(path, data, selected, seed=42, exclude=[]) == original
    for changed_data, changed_seed in [(data, 43), ({**data, "new": data["v00"]}, 42)]:
        with pytest.raises(ValueError, match="new --work-dir-root"):
            driver.save_sample_manifest(
                path, changed_data, selected, seed=changed_seed, exclude=[],
            )
    assert json.loads(path.read_text()) == original


def setup_main(tmp_path, monkeypatch):
    data = annotations()
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(driver, "run_one_video", lambda **kw: pytest.fail("pipeline ran"))
    monkeypatch.setattr(driver, "run_judge", lambda **kw: pytest.fail("judge ran"))
    monkeypatch.setattr(driver, "_build_graph_service", lambda *a: pytest.fail("DB accessed"))
    monkeypatch.setattr(driver, "resolve_video_path", lambda *a, **kw: tmp_path / "video.mp4")
    return ["--annotations", str(path), "--sample-size", "10", "--seconds", "0",
            "--work-dir-root", str(tmp_path / "run"), "--judge",
            "--config", str(tmp_path / "nonexistent-config")]


def test_selection_only_needs_no_config_or_services(tmp_path, monkeypatch):
    argv = setup_main(tmp_path, monkeypatch)
    assert driver.main(argv + ["--selection-only"]) == 0
    manifest = json.loads((tmp_path / "run/sample_manifest.json").read_text())
    assert len(manifest["selected_video_ids"]) == 10
    assert not (tmp_path / "run/_aggregate").exists()


def test_missing_video_aborts_without_resampling_or_running(tmp_path, monkeypatch):
    argv = setup_main(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, "resolve_video_path", lambda *a, **kw: None)
    assert driver.main(argv + ["--selection-only"]) == 1
    assert not (tmp_path / "run/sample_manifest.json").exists()


def test_dry_run_preserves_existing_aggregate(tmp_path, monkeypatch):
    argv = setup_main(tmp_path, monkeypatch)
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    monkeypatch.setattr(driver, "run_one_video", lambda **kw: {"status": "dry_run"})
    aggregate = tmp_path / "run/_aggregate/all_m3_qa_results.jsonl"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text("existing results\n")
    assert driver.main(argv + ["--config", str(config), "--dry-run"]) == 0
    assert aggregate.read_text() == "existing results\n"


def test_dry_run_with_graph_reuse_does_not_access_services(tmp_path, monkeypatch):
    args = driver.parse_args(["--dry-run", "--reuse-graph", "--graph-store", "sql"])
    args.work_dir_root = tmp_path
    monkeypatch.setattr(
        driver, "_build_graph_service", lambda *a: pytest.fail("DB accessed"),
    )
    monkeypatch.setattr(
        driver.subprocess, "run", lambda *a, **kw: pytest.fail("pipeline ran"),
    )
    result = driver.run_one_video(
        video_id="v00", video_path=tmp_path / "v00.mp4", args=args,
        annotations=annotations(), expected_qids={"q0"},
    )
    assert result["status"] == "dry_run"
    assert not (tmp_path / "v00/pipeline.log").exists()


def test_empty_predictions_skip_judge_and_fail(tmp_path, monkeypatch):
    argv = setup_main(tmp_path, monkeypatch)
    config = tmp_path / 'config.yaml'
    config.write_text('{}')
    monkeypatch.setattr(driver, 'run_one_video', lambda **kw: {'status': 'failed'})
    assert driver.main(argv + ['--config', str(config)]) == 2
    report = json.loads((tmp_path / 'run/_aggregate/run_report.json').read_text())
    assert report['status'] == 'failed'
    assert report['judge'] == {'status': 'skipped', 'reason': 'no_qa_results'}


def test_judge_failure_propagates_to_run_status(tmp_path, monkeypatch):
    argv = setup_main(tmp_path, monkeypatch)
    config = tmp_path / 'config.yaml'
    config.write_text('{}')
    work_root = tmp_path / "run"

    # WS 2.5: judging is live and per-video — a per-video judge failure
    # (returncode != 0) must still propagate to the overall run status.
    def fake_run_one_video(**kw):
        vid = kw["video_id"]
        work = work_root / vid
        work.mkdir(parents=True, exist_ok=True)
        (work / "m3_qa_results.jsonl").write_text(json.dumps({
            "video_id": vid, "question_id": "q", "predicted_answer": "a",
        }) + "\n")
        return {"video_id": vid, "status": "ok", "returncode": 0}

    monkeypatch.setattr(driver, 'run_one_video', fake_run_one_video)
    monkeypatch.setattr(driver, 'aggregate_jsonl', lambda *a: {
        'total_records': 1, 'answered': 1, 'failed': 0,
    })
    monkeypatch.setattr(driver, 'run_judge', lambda **kw: {'returncode': 2})
    assert driver.main(argv + ['--config', str(config)]) == 2
    report = json.loads((tmp_path / 'run/_aggregate/run_report.json').read_text())
    assert report['status'] == 'failed'
