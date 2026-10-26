"""Resume must validate actual successful output and the effective input."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from m3.pipeline.context import PipelineContext
from m3.pipeline.spec import StageSpec
from m3.pipeline import stages
from m3.qa_results import completed_keys


def context(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source-video")
    return PipelineContext(video=video, video_id="v", seconds=120,
        work_dir=tmp_path, config_path=tmp_path / "config.yaml", graph_store="memory",
        connection_string=None, account_id="a", variables={}, artifacts={
            "selected_video": tmp_path / "selected.mp4", "clips_root": tmp_path / "clips",
            "annotations": tmp_path / "annotations.json"})


def annotation(path, ids):
    path.write_text(json.dumps({"v": {"qa_list": [
        {"question_id": qid, "question": "Where?", "answer": "Here"} for qid in ids]}}))


def row(qid="q1", **kw):
    return dict(video_id="v", question_id=qid, predicted_answer="Here", error="", **kw)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_qa_probe_checks_current_questions_failures_and_video(tmp_path):
    ctx = context(tmp_path)
    annotation(ctx.artifact("annotations"), ["q1", "q2"])
    output = tmp_path / "m3_qa_results.jsonl"
    stage = StageSpec("qa", "m3.qa", params={"video": "v", "resume": True})
    bad = row("q2"); bad["error"] = "temporary failure"
    write_rows(output, [row(), bad])
    assert not stages._probe_qa(ctx, stage, None).complete
    with output.open("a") as f:
        f.write(json.dumps(row("q2")) + "\n")
    assert stages._probe_qa(ctx, stage, None).complete
    annotation(ctx.artifact("annotations"), ["q1", "q2", "q3"])
    assert not stages._probe_qa(ctx, stage, None).complete
    stage.params["limit"] = 2
    assert stages._probe_qa(ctx, stage, None).complete
    other = row("q2"); other["video_id"] = "other"
    write_rows(output, [row(), other])
    assert not stages._probe_qa(ctx, stage, None).complete


@pytest.mark.parametrize("react", [True, False])
def test_resume_retries_failure_and_truncated_line_but_not_success(tmp_path, monkeypatch, react):
    import m3.qa as qa
    ann, output = tmp_path / "ann.json", tmp_path / "out.jsonl"
    annotation(ann, ["q1", "q2", "q3"])
    failed = row("q2"); failed["error"] = "temporary failure"
    write_rows(output, [row(), failed])
    with output.open("a") as f:
        f.write('{"video_id":')
    argv = ["--annotations", str(ann), "--output-jsonl", str(output), "--resume"]
    if react:
        adapter = Mock()
        adapter.run.return_value = SimpleNamespace(output=["[Answer] Here"], completed=True,
                                                   iterations=1, finish_reason="done")
        monkeypatch.setattr("m3.control_adapter.OGMemoryControlAdapter", Mock(return_value=adapter))
        assert qa.run_react_qa(qa.parse_react_qa_args(argv), graph=Mock(), embedder=Mock(), llm=Mock()) == 0
        assert adapter.run.call_count == 2
        rows = [json.loads(line) for line in output.read_text().splitlines() if '"warning"' in line]
        assert all(r["warning"] and not r["error"] for r in rows)
    else:
        monkeypatch.setattr(qa, "build_embedder", Mock())
        monkeypatch.setattr(qa, "build_llm", Mock())
        monkeypatch.setattr(qa.EpisodicSearcher, "from_records", Mock(return_value=Mock(search=Mock(return_value=[]))))
        answer = Mock(return_value={"answer": "Here", "reasoning": "", "confidence": 1})
        monkeypatch.setattr(qa, "answer_question", answer)
        assert qa.run_qa(qa.parse_args(argv + ["--skip-vector-index"])) == 0
        assert answer.call_count == 2
    assert completed_keys(output) == {("v", "q1"), ("v", "q2"), ("v", "q3")}


def test_audio_cli_preserves_profile_until_explicit_override():
    from m3.run_video_pipeline import parse_args, _legacy_set_overrides
    for flags, expected in [([], None), (["--with-audio"], "false"), (["--no-audio"], "true")]:
        overrides = _legacy_set_overrides(parse_args(["--video", "v.mp4"] + flags))
        audio = [v for v in overrides if ".no_audio=" in v]
        assert audio == ([] if expected is None else [f"episodic_memory.params.no_audio={expected}"])


def test_index_probe_uses_tenant_scoped_vector_not_graph_embedding(tmp_path):
    from m3.resume import video_fully_indexed
    from m3.episodic_memory import extracted_node_id
    from m3.index_episodic import _upsert_to_vector_index
    from providers.vector_index.in_memory_index import InMemoryVectorIndex
    ctx = context(tmp_path)
    ext_id = extracted_node_id("episodic_memory_generator", "raw")
    raw = SimpleNamespace(node_id="raw", metadata={"video_id": "v", "clip_id": 0})
    ext = SimpleNamespace(node_id=ext_id, extraction_status="completed", metadata={})
    ctx.graph = Mock(list_raw_nodes=Mock(return_value=[raw]), get_extracted_node=Mock(return_value=ext))
    ctx.vector_index = InMemoryVectorIndex(2)
    record = {"video_id": "v", "clip_id": 0, "source_extracted_node_id": ext_id,
              "embedding": [1., 0.], "text": "Here"}
    _upsert_to_vector_index([record], ctx.vector_index, SimpleNamespace(account_id="other"))
    assert not stages._probe_index_episodic(ctx, StageSpec("index", "m3.index_episodic"), ctx.graph).complete
    _upsert_to_vector_index([record], ctx.vector_index, SimpleNamespace(account_id="a"))
    assert video_fully_indexed(ctx.graph, "a", "v", vector_index=ctx.vector_index)["complete"]
    assert stages._probe_index_episodic(ctx, StageSpec("index", "m3.index_episodic"), ctx.graph).complete
    ext.extraction_status = "pending"
    assert not video_fully_indexed(ctx.graph, "a", "v", vector_index=ctx.vector_index)["complete"]


@pytest.mark.parametrize("change", ["source", "seconds", "clip_seconds"])
def test_changed_media_input_refuses_stale_graph(tmp_path, change):
    from m3.pipeline.provenance import validate_input
    ctx = context(tmp_path)
    stage = StageSpec("preprocess", "m3.preprocess", params={"clip_seconds": 30})
    spec = SimpleNamespace(stages=[stage])
    validate_input(ctx, spec)
    validate_input(ctx, spec)
    if change == "source":
        ctx.video.write_bytes(b"different-source")
    elif change == "seconds":
        ctx.seconds = 60
    else:
        stage.params["clip_seconds"] = 60
    with pytest.raises(RuntimeError, match="new --work-dir AND"):
        validate_input(ctx, spec)


@pytest.mark.parametrize("kind", ["trim", "preprocess"])
def test_media_probe_requires_provenance_and_complete_outputs(tmp_path, kind):
    from m3.pipeline.provenance import record_stage
    ctx = context(tmp_path)
    ctx.artifact("selected_video").write_bytes(b"selected")
    ctx.artifact("clips_root").mkdir()
    clip = ctx.artifact("clips_root") / "0.mp4"
    clip.write_bytes(b"clip")
    stage = StageSpec(kind, "video.trim" if kind == "trim" else "m3.preprocess")
    probe = stages._probe_video_trim if kind == "trim" else stages._probe_preprocess
    assert not probe(ctx, stage, None).complete
    record_stage(ctx, stage)
    assert probe(ctx, stage, None).complete
    target = ctx.artifact("selected_video") if kind == "trim" else clip
    target.write_bytes(b"")
    assert not probe(ctx, stage, None).complete


@pytest.mark.parametrize("disabled", [
    ["trim"], ["preprocess"], ["load_clips"], ["trim", "preprocess", "load_clips"],
])
def test_targeted_resume_keeps_original_media_identity(tmp_path, disabled):
    from m3.pipeline.provenance import validate_input
    ctx = context(tmp_path)
    spec = SimpleNamespace(stages=[
        StageSpec("trim", "video.trim", params={"seconds": "{seconds}"}),
        StageSpec("preprocess", "m3.preprocess", params={"clip_seconds": 30}),
        StageSpec("load_clips", "m3.load_clips", params={"clip_seconds": 30}),
    ])
    ctx.variables["seconds"] = ctx.seconds
    validate_input(ctx, spec)
    path = tmp_path / "input_provenance.json"
    original = path.read_bytes()
    for stage in spec.stages:
        stage.enabled = stage.id not in disabled
    validate_input(ctx, spec)
    assert path.read_bytes() == original


@pytest.mark.parametrize("change", [
    "source", "seconds", "account_id", "video_id", "trim", "preprocess", "load_clips",
    "removed_stage",
])
def test_targeted_resume_still_rejects_changed_media_identity(tmp_path, change):
    from m3.pipeline.provenance import validate_input
    ctx = context(tmp_path)
    spec = SimpleNamespace(stages=[
        StageSpec("trim", "video.trim", params={"seconds": 120}),
        StageSpec("preprocess", "m3.preprocess", params={"clip_seconds": 30}),
        StageSpec("load_clips", "m3.load_clips", params={"clip_seconds": 30}),
    ])
    validate_input(ctx, spec)
    path = tmp_path / "input_provenance.json"
    original = path.read_bytes()
    for stage in spec.stages:
        stage.enabled = False
    if change == "source":
        ctx.video.write_bytes(b"changed")
    elif change == "seconds":
        ctx.seconds = 60
    elif change in ("account_id", "video_id"):
        setattr(ctx, change, "different")
    elif change == "removed_stage":
        spec.stages.pop()
    else:
        stage = next(stage for stage in spec.stages if stage.id == change)
        stage.params["seconds" if change == "trim" else "clip_seconds"] = 60
    with pytest.raises(RuntimeError, match="new --work-dir AND"):
        validate_input(ctx, spec)
    assert path.read_bytes() == original


def test_resume_cli_can_disable_completed_stages_without_changing_provenance(tmp_path, monkeypatch):
    from m3.run_video_pipeline import parse_args, run_pipeline
    ctx = context(tmp_path)
    run = Mock(return_value={})
    monkeypatch.setattr("m3.pipeline.runner.PipelineRunner.run", run)
    argv = [
        "--video", str(ctx.video), "--video-id", "v", "--seconds", "0",
        "--work-dir", str(tmp_path), "--config", str(ctx.config_path),
        "--pipeline", "config/m3/pipelines/video_episodic_full.yaml",
        "--account-id", "m3bench", "--no-smart-resume",
    ]
    run_pipeline(parse_args(argv))
    path = tmp_path / "input_provenance.json"
    original = path.read_bytes()
    disabled = [
        "trim", "annotate", "preprocess", "load_clips", "extract_audio",
        "extract_frames", "detect_faces", "voice_asr", "speaker_diarization",
        "speaker_embedding", "episodic_memory", "create_memory_nodes",
    ]
    for name in disabled:
        argv.extend(["--disable-stage", name])
    run_pipeline(parse_args(argv))
    assert run.call_count == 2
    spec = run.call_args.args[0]
    assert [stage.id for stage in spec.stages if stage.enabled] == [
        "build_index", "face_clustering", "voice_segmentation",
        "voice_clustering", "semantic_memory_consolidator", "qa",
    ]
    assert path.read_bytes() == original


def test_reuse_graph_resolves_yaml_dsn_before_probe(tmp_path, monkeypatch):
    from m3 import resume
    from m3.run_video_pipeline import parse_args, run_pipeline
    ctx = context(tmp_path)
    ctx.config_path.write_text("m3:\n  graph_store:\n    connection_string: postgresql://yaml-only/db\n")
    graph_builder = Mock(return_value=(Mock(), "sql"))
    monkeypatch.setattr(resume, "build_graph_service", graph_builder)
    monkeypatch.setattr("m3.pipeline.runner._build_vector_index", Mock(return_value=Mock()))
    monkeypatch.setattr(resume, "video_fully_indexed", Mock(return_value={
        "complete": True, "video_clip_count": 1, "missing_extracted": [],
        "unfinished_extracted": [], "missing_embedding": []}))
    qa = Mock(return_value={"jsonl": str(tmp_path / "qa.jsonl"), "duration_sec": 0,
                            "returncode": 0, "status": "ok"})
    monkeypatch.setattr(resume, "run_qa_only", qa)
    args = parse_args(["--video", str(ctx.video), "--work-dir", str(tmp_path),
                       "--config", str(ctx.config_path), "--graph-store", "sql", "--reuse-graph"])
    run_pipeline(args)
    assert graph_builder.call_args.args == ("sql", "postgresql://yaml-only/db")
    assert qa.call_args.kwargs["connection_string"] == "postgresql://yaml-only/db"


def test_qa_only_successful_retry_supersedes_old_error(tmp_path, monkeypatch):
    from m3.resume import run_qa_only
    ann = tmp_path / "ann.json"
    annotation(ann, ["q1"])
    failed = row(); failed["error"] = "transient"
    write_rows(tmp_path / "m3_qa_results.jsonl", [failed, row()])
    monkeypatch.setattr("m3.qa.run_react_qa", Mock(return_value=0))
    result = run_qa_only(graph_store="memory", connection_string=None, account_id="a",
                         config_path=tmp_path / "config.yaml", annotations_path=ann,
                         video_id="v", work_dir=tmp_path, resume=True)
    assert result["status"] == "ok"
    assert result["qa_only_missing_qids"] == []


def test_old_media_without_provenance_is_not_silently_trusted(tmp_path):
    from m3.pipeline.provenance import validate_input
    ctx = context(tmp_path)
    ctx.artifact("selected_video").write_bytes(b"old-input")
    with pytest.raises(RuntimeError, match="no input provenance"):
        validate_input(ctx, SimpleNamespace(stages=[]))


def test_changed_index_text_is_not_skipped(tmp_path):
    from m3.index_status import current_text_is_indexed, episodic_uri
    from m3.index_episodic import build_retrieval_text
    from providers.vector_index.in_memory_index import InMemoryVectorIndex
    from core.models import IndexRecord
    ctx = context(tmp_path)
    path = tmp_path / "episodic.jsonl"
    ctx.artifacts["episodic_jsonl"] = path
    rec = {"video_id": "v", "clip_id": 0, "description": "Original scene"}
    write_rows(path, [rec])
    uri = episodic_uri("a", "v", 0)
    ctx.vector_index = InMemoryVectorIndex(2)
    ctx.vector_index.upsert([IndexRecord(id=IndexRecord.generate_id(uri, level=2),
        uri=uri, level=2, text=build_retrieval_text(rec), filters={"account_id": "a"},
        metadata={"_embedding": [1., 0.]})])
    stage = StageSpec("index", "m3.index_episodic")
    assert current_text_is_indexed(ctx, stage)
    rec["description"] = "Updated scene"
    write_rows(path, [rec])
    assert not current_text_is_indexed(ctx, stage)
