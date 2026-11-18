"""WS 2.5 — video-parallel bench driver: threads, live per-video judge, merge.

Covers ``scripts/run_m3_bench_subset.py::run_videos`` /
``live_judge_video`` / ``merge_per_video_judges`` with a fake
``run_one_video`` writing per-video JSONL and a fake judge, DB-free.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

from scripts import run_m3_bench_subset as driver


def _annotations(videos: list[str]) -> dict:
    return {
        vid: {
            "video_path": vid,
            "qa_list": [
                {"question_id": f"{vid}_Q{i}", "question": f"q{i}",
                 "answer": f"a{i}", "timestamp": "00:10", "before_clip": 1}
                for i in range(1, 3)
            ],
        }
        for vid in videos
    }


def _args(tmp_path: Path, **overrides) -> SimpleNamespace:
    base = dict(
        video_workers=1,
        judge=False,
        force=False,
        dry_run=False,
        work_dir_root=str(tmp_path),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _question_row(video_id: str, qid: str, **overrides) -> dict:
    row = {
        "video_id": video_id,
        "question_id": qid,
        "question_index": 0,
        "question": f"q for {qid}",
        "expected_answer": "gold",
        "predicted_answer": "pred",
        "judge_score": 1.0,
        "judge_error": False,
        "error": "",
        "iterations": 2,
        "latency_seconds": 0.5,
    }
    row.update(overrides)
    return row


def _install_fake_runner(monkeypatch, tmp_path: Path, on_run=None):
    def fake_run_one_video(*, video_id, video_path, args, annotations,
                           expected_qids):
        if on_run is not None:
            on_run(video_id)
        work = Path(args.work_dir_root) / video_id
        work.mkdir(parents=True, exist_ok=True)
        jsonl = work / "m3_qa_results.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            for q in annotations[video_id]["qa_list"]:
                f.write(json.dumps({
                    "video_id": video_id,
                    "question_id": q["question_id"],
                    "question": q["question"],
                    "expected_answer": q["answer"],
                    "predicted_answer": "pred",
                }) + "\n")
        return {
            "video_id": video_id,
            "status": "ok",
            "duration_sec": 0.1,
            "jsonl": str(jsonl),
            "returncode": 0,
        }

    monkeypatch.setattr(driver, "run_one_video", fake_run_run := fake_run_one_video)

    def fake_resolve_video_path(declared, *, project_root, videos_root):
        return tmp_path / f"{declared}.mp4"

    monkeypatch.setattr(driver, "resolve_video_path", fake_resolve_video_path)
    return fake_run_run


def test_video_workers_run_videos_concurrently(monkeypatch, tmp_path):
    """With --video-workers N, N videos overlap; summaries stay in subset order."""
    videos = ["v1", "v2", "v3"]
    barrier = threading.Barrier(len(videos), timeout=15)
    threads_seen: set[str] = set()

    def on_run(video_id):
        threads_seen.add(threading.current_thread().name)
        # Passes only if all N workers are inside run_one_video at once —
        # a serial loop would deadlock on the barrier and time out.
        barrier.wait()

    _install_fake_runner(monkeypatch, tmp_path, on_run=on_run)
    result = driver.run_videos(
        subset=videos,
        args=_args(tmp_path, video_workers=3),
        annotations=_annotations(videos),
        annotations_path=tmp_path / "robot.json",
        project_root=tmp_path,
        videos_root=tmp_path,
    )

    assert [s["video_id"] for s in result] == videos  # deterministic order
    assert all(s["status"] == "ok" for s in result)
    assert len(threads_seen) >= 2  # genuinely concurrent, not one loop thread


def test_serial_default_single_threaded_and_ordered(monkeypatch, tmp_path):
    """Wv=1 (default) behaves like the old serial loop: main-thread, subset order."""
    videos = ["v1", "v2"]
    run_order: list[str] = []
    threads_seen: set[str] = set()

    def on_run(video_id):
        threads_seen.add(threading.current_thread().name)
        run_order.append(video_id)

    _install_fake_runner(monkeypatch, tmp_path, on_run=on_run)
    result = driver.run_videos(
        subset=videos,
        args=_args(tmp_path),  # video_workers=1
        annotations=_annotations(videos),
        annotations_path=tmp_path / "robot.json",
        project_root=tmp_path,
        videos_root=tmp_path,
    )

    assert run_order == videos  # strict serial, no scheduling reorder
    assert threading.main_thread().name in threads_seen
    assert [s["video_id"] for s in result] == videos


class _JudgeCalls(list):
    """Call list that also records the ``resume`` kwarg per call."""


def _install_fake_judge(monkeypatch):
    calls: _JudgeCalls = _JudgeCalls()
    resumes: list[bool | None] = []

    def fake_run_judge(*, annotations, merged_jsonl, output_dir, args, subset,
                       resume=None):
        vid = subset[0]
        calls.append(vid)
        resumes.append(resume)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            _question_row(vid, f"{vid}_Q1"),
            _question_row(vid, f"{vid}_Q2", judge_score=0.0),
        ]
        (output_dir / "m3_eval_results.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8",
        )
        summary = {
            "judge_provider": "fake",
            "judge_model": "fake-model",
            "overall_mean_score": 0.5,
            "total_questions": 2,
        }
        (output_dir / "m3_eval_summary.json").write_text(
            json.dumps(summary), encoding="utf-8",
        )
        return {
            "returncode": 0,
            "summary_path": str(output_dir / "m3_eval_summary.json"),
            "summary": summary,
        }

    monkeypatch.setattr(driver, "run_judge", fake_run_judge)
    calls.resumes = resumes
    return calls


def test_live_per_video_judge_and_merged_aggregate(monkeypatch, tmp_path):
    """--judge scores each video immediately; _aggregate is a merge, not a re-judge."""
    videos = ["v1", "v2"]
    _install_fake_runner(monkeypatch, tmp_path)
    judge_calls = _install_fake_judge(monkeypatch)

    summaries = driver.run_videos(
        subset=videos,
        args=_args(tmp_path, video_workers=2, judge=True),
        annotations=_annotations(videos),
        annotations_path=tmp_path / "robot.json",
        project_root=tmp_path,
        videos_root=tmp_path,
    )

    # Live per-video judges ran inside the run, one per video.
    assert sorted(judge_calls) == videos
    for summary in summaries:
        assert summary["judge"]["returncode"] == 0
        per_video_summary = (
            tmp_path / summary["video_id"] / "m3_eval_summary.json"
        )
        assert per_video_summary.exists()

    # The merged _aggregate summary recomputes from per-video rows — no
    # second judge pass, no extra run_judge invocation.
    output_dir = tmp_path / "_aggregate"
    merged = driver.merge_per_video_judges(videos, tmp_path, output_dir)
    assert merged is not None
    assert merged["total_questions"] == 4
    assert merged["overall_mean_score"] == 0.5  # two 1.0s + two 0.0s
    assert merged["judge_provider"] == "fake"
    assert len(judge_calls) == 2  # unchanged — merge does not re-judge
    assert (output_dir / "m3_eval_summary.json").exists()
    assert (output_dir / "m3_eval_results.jsonl").exists()


def test_live_judge_reuses_existing_per_video_summary(monkeypatch, tmp_path):
    """Resume: an existing per-video m3_eval_summary.json is reused, not re-judged."""
    videos = ["v1"]
    _install_fake_runner(monkeypatch, tmp_path)
    judge_calls = _install_fake_judge(monkeypatch)

    driver.run_videos(
        subset=videos,
        args=_args(tmp_path, judge=True),
        annotations=_annotations(videos),
        annotations_path=tmp_path / "robot.json",
        project_root=tmp_path,
        videos_root=tmp_path,
    )
    assert judge_calls == ["v1"]

    second = driver.run_videos(
        subset=videos,
        args=_args(tmp_path, judge=True),  # force=False
        annotations=_annotations(videos),
        annotations_path=tmp_path / "robot.json",
        project_root=tmp_path,
        videos_root=tmp_path,
    )
    assert judge_calls == ["v1"]  # reused — judge did not run again
    assert second[0]["judge"]["status"] == "reused"


def test_on_video_done_called_once_per_video(monkeypatch, tmp_path):
    """The incremental run-report hook fires once per completed video."""
    videos = ["v1", "v2", "v3"]
    _install_fake_runner(monkeypatch, tmp_path)
    seen: list[str] = []

    def on_video_done(summary):
        seen.append(summary["video_id"])
        # The callback runs under the shared lock while earlier videos may
        # still be running — the refresh must be safe to call concurrently.
        (tmp_path / f"report_{summary['video_id']}.json").write_text(
            json.dumps(summary), encoding="utf-8",
        )

    driver.run_videos(
        subset=videos,
        args=_args(tmp_path, video_workers=3),
        annotations=_annotations(videos),
        annotations_path=tmp_path / "robot.json",
        project_root=tmp_path,
        videos_root=tmp_path,
        on_video_done=on_video_done,
    )

    assert sorted(seen) == videos
    for vid in videos:
        assert (tmp_path / f"report_{vid}.json").exists()


def test_failed_video_is_skipped_not_fatal(monkeypatch, tmp_path):
    """One video's driver exception produces a failed summary; siblings run."""
    videos = ["bad", "good"]
    annotations = _annotations(videos)

    def fake_run_one_video(*, video_id, video_path, args, annotations,
                           expected_qids):
        if video_id == "bad":
            raise RuntimeError("boom")
        return {"video_id": video_id, "status": "ok", "returncode": 0}

    monkeypatch.setattr(driver, "run_one_video", fake_run_one_video)

    def fake_resolve_video_path(declared, *, project_root, videos_root):
        return tmp_path / f"{declared}.mp4"

    monkeypatch.setattr(driver, "resolve_video_path", fake_resolve_video_path)

    result = driver.run_videos(
        subset=videos,
        args=_args(tmp_path, video_workers=2),
        annotations=annotations,
        annotations_path=tmp_path / "robot.json",
        project_root=tmp_path,
        videos_root=tmp_path,
    )

    by_id = {s["video_id"]: s for s in result}
    assert by_id["bad"]["status"] == "failed"
    assert by_id["good"]["status"] == "ok"


def test_dry_run_skips_live_judge(monkeypatch, tmp_path):
    videos = ["v1"]
    _install_fake_runner(monkeypatch, tmp_path)
    judge_calls = _install_fake_judge(monkeypatch)

    summaries = driver.run_videos(
        subset=videos,
        args=_args(tmp_path, judge=True, dry_run=True),
        annotations=_annotations(videos),
        annotations_path=tmp_path / "robot.json",
        project_root=tmp_path,
        videos_root=tmp_path,
    )

    assert judge_calls == []
    assert "judge" not in summaries[0]


def _install_resume_runner(monkeypatch, tmp_path):
    """A runner that leaves the existing JSONL untouched (resume semantics:
    externally repaired QA rows must survive the re-run)."""

    def fake_run_one_video(*, video_id, video_path, args, annotations,
                           expected_qids):
        return {
            "video_id": video_id,
            "status": "ok",
            "duration_sec": 0.1,
            "jsonl": str(Path(args.work_dir_root) / video_id / "m3_qa_results.jsonl"),
            "returncode": 0,
        }

    monkeypatch.setattr(driver, "run_one_video", fake_run_one_video)

    def fake_resolve_video_path(declared, *, project_root, videos_root):
        return tmp_path / f"{declared}.mp4"

    monkeypatch.setattr(driver, "resolve_video_path", fake_resolve_video_path)


def test_live_judge_rejudges_when_predictions_change(monkeypatch, tmp_path):
    """Review item 9: a cached judgment is bound to the digest of its
    inputs — repaired predictions must re-judge, not reuse stale scores."""
    videos = ["v1"]
    _install_fake_runner(monkeypatch, tmp_path)
    judge_calls = _install_fake_judge(monkeypatch)
    annotations_path = tmp_path / "robot.json"

    driver.run_videos(
        subset=videos, args=_args(tmp_path, judge=True),
        annotations=_annotations(videos), annotations_path=annotations_path,
        project_root=tmp_path, videos_root=tmp_path,
    )
    assert judge_calls == ["v1"]

    # Repair the predictions (as a normal resume does for failed/partial
    # rows) — the recorded digest no longer matches.
    jsonl = tmp_path / "v1" / "m3_qa_results.jsonl"
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l]
    rows[0]["predicted_answer"] = "repaired answer"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    _install_resume_runner(monkeypatch, tmp_path)  # resume: JSONL untouched
    second = driver.run_videos(
        subset=videos, args=_args(tmp_path, judge=True),  # force=False
        annotations=_annotations(videos), annotations_path=annotations_path,
        project_root=tmp_path, videos_root=tmp_path,
    )
    assert judge_calls == ["v1", "v1"]  # re-judged, not reused
    assert second[0]["judge"].get("status") != "reused"


def test_live_judge_rejudges_when_judge_model_changes(monkeypatch, tmp_path):
    """Switching the judge model invalidates cached judgments."""
    videos = ["v1"]
    _install_fake_runner(monkeypatch, tmp_path)
    judge_calls = _install_fake_judge(monkeypatch)
    annotations_path = tmp_path / "robot.json"

    driver.run_videos(
        subset=videos, args=_args(tmp_path, judge=True),
        annotations=_annotations(videos), annotations_path=annotations_path,
        project_root=tmp_path, videos_root=tmp_path,
    )
    assert judge_calls == ["v1"]

    _install_resume_runner(monkeypatch, tmp_path)
    second = driver.run_videos(
        subset=videos,
        args=_args(tmp_path, judge=True, judge_model="other-model"),
        annotations=_annotations(videos), annotations_path=annotations_path,
        project_root=tmp_path, videos_root=tmp_path,
    )
    assert judge_calls == ["v1", "v1"]
    assert second[0]["judge"].get("status") != "reused"


def test_aggregate_jsonl_dedupes_retry_records(tmp_path):
    """QA resume *appends* retry attempts; the merge must collapse them.

    A question that failed once and was answered on retry produces two
    rows in the per-video jsonl. The merged file must contain one row
    (the latest, last-wins like the judge) and count it as answered —
    not as one answered + one failed attempt.
    """
    work = tmp_path / "work"
    vid_dir = work / "v1"
    vid_dir.mkdir(parents=True)
    rows = [
        {"video_id": "v1", "question_id": "v1_Q1", "predicted_answer": "ok", "error": None},
        {"video_id": "v1", "question_id": "v1_Q2", "predicted_answer": "", "error": "llm_error"},
        {"video_id": "v1", "question_id": "v1_Q2", "predicted_answer": "recovered", "error": None},
    ]
    with (vid_dir / "m3_qa_results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    out = tmp_path / "agg" / "all.jsonl"
    result = driver.aggregate_jsonl(["v1"], work, out)

    merged = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(merged) == 2
    assert merged[-1]["predicted_answer"] == "recovered"
    assert result["total_records"] == 2
    assert result["answered"] == 2
    assert result["failed"] == 0
    assert result["per_video"] == {"v1": 3}  # raw rows read, attempts included


def test_aggregate_jsonl_malformed_and_noid_rows_pass_through(tmp_path):
    work = tmp_path / "work"
    vid_dir = work / "v1"
    vid_dir.mkdir(parents=True)
    with (vid_dir / "m3_qa_results.jsonl").open("w") as f:
        f.write(json.dumps({"video_id": "v1", "question_id": "v1_Q1",
                            "predicted_answer": "ok", "error": None}) + "\n")
        f.write("{not json\n")
        f.write(json.dumps({"video_id": "v1", "predicted_answer": "x"}) + "\n")

    out = tmp_path / "agg" / "all.jsonl"
    result = driver.aggregate_jsonl(["v1"], work, out)

    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 3  # nothing silently dropped
    assert result["total_records"] == 3
    # The malformed line counts as failed; the no-question-id record has no
    # error field and keeps the legacy "answered" classification.
    assert result["answered"] == 2
    assert result["failed"] == 1


def test_live_judge_rejudge_disables_resume_on_digest_mismatch(monkeypatch, tmp_path):
    """A re-judge after a digest mismatch must not forward --resume:
    m3.eval --resume skips by question id without re-checking whether
    the prediction changed, so stale per-question results would shadow
    the re-judge."""
    videos = ["v1"]
    _install_fake_runner(monkeypatch, tmp_path)
    judge_calls = _install_fake_judge(monkeypatch)
    annotations_path = tmp_path / "robot.json"

    args = _args(tmp_path, judge=True)
    args.resume = True
    driver.run_videos(
        subset=videos, args=args,
        annotations=_annotations(videos), annotations_path=annotations_path,
        project_root=tmp_path, videos_root=tmp_path,
    )
    assert judge_calls == ["v1"]
    # First judge: no summary yet, but resume is still disabled — the
    # per-question results (if any) are not digest-verified.
    assert judge_calls.resumes == [False]

    # Change predictions -> digest mismatch -> re-judge without resume.
    jsonl = tmp_path / "v1" / "m3_qa_results.jsonl"
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l]
    rows[0]["predicted_answer"] = "repaired answer"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    _install_resume_runner(monkeypatch, tmp_path)
    driver.run_videos(
        subset=videos, args=args,
        annotations=_annotations(videos), annotations_path=annotations_path,
        project_root=tmp_path, videos_root=tmp_path,
    )
    assert judge_calls == ["v1", "v1"]
    assert judge_calls.resumes == [False, False]
