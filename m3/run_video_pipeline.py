"""Run a declarative M3 video pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

from m3.config import M3BENCH_ACCOUNT_ID
from m3.pipeline import PipelineRunner, load_pipeline_spec

logger = logging.getLogger(__name__)

DEFAULT_PIPELINE = "config/m3/pipelines/video_episodic.yaml"


def _legacy_set_overrides(args: argparse.Namespace) -> list[str]:
    """Map legacy convenience flags onto declarative stage param overrides."""
    out: list[str] = []
    out.extend([
        f"preprocess.params.clip_seconds={args.clip_seconds}",
        f"load_clips.params.clip_seconds={args.clip_seconds}",
        f"extract_frames.params.fps={args.fps}",
        f"extract_frames.params.max_frames={args.max_frames}",
        f"extract_frames.params.resize={args.resize}",
        f"extract_frames.params.jpeg_quality={args.jpeg_quality}",
        f"extract_frames.params.keyframes_only={str(args.keyframes_only).lower()}",
        f"episodic_memory.params.max_frames={args.episodic_max_frames}",
    ])
    if args.with_audio is not None:
        out.append(f"episodic_memory.params.no_audio={str(not args.with_audio).lower()}")
    if args.llm:
        out.append(f"episodic_memory.params.llm={args.llm}")
    if args.face_backend:
        out.append(f"detect_faces.params.backend={args.face_backend}")
    if args.cv_service_url:
        out.append(f"detect_faces.params.cv_service_url={args.cv_service_url}")
    if args.cv_timeout is not None:
        out.append(f"detect_faces.params.cv_timeout={args.cv_timeout}")
    out.extend([
        f"detect_faces.params.min_confidence={args.face_min_confidence}",
        f"detect_faces.params.max_faces={args.face_max_faces}",
    ])
    if args.memory_topic:
        out.append(f"create_memory_nodes.params.memory_topic={args.memory_topic}")
    if args.embedder:
        out.append(f"build_index.params.embedder={args.embedder}")
    if args.embedding_dimension is not None:
        out.append(f"build_index.params.embedding_dimension={args.embedding_dimension}")
    if args.ffmpeg_bin:
        out.append(f"trim.params.ffmpeg_bin={args.ffmpeg_bin}")
        out.append(f"preprocess.params.ffmpeg_bin={args.ffmpeg_bin}")
    if args.ffprobe_bin:
        out.append(f"preprocess.params.ffprobe_bin={args.ffprobe_bin}")
        out.append(f"load_clips.params.ffprobe_bin={args.ffprobe_bin}")
    if args.ffmpeg_log_level:
        out.append(f"trim.params.ffmpeg_log_level={args.ffmpeg_log_level}")
    # Forward --video-id to the QA stage's --video filter so the
    # adapter answers only this video's questions (instead of every
    # video in the annotations file). When --video-id is omitted the
    # driver falls back to the file stem (set in run_pipeline()).
    if getattr(args, "video_id", None):
        out.append(f"qa.params.video={args.video_id}")
    return out


def run_pipeline(args: argparse.Namespace) -> dict:
    """Load the requested YAML spec and execute it.

    With ``--reuse-graph`` and ``--graph-store sql``, the graph is
    probed for completeness before the runner is invoked. If every
    VIDEO clip of this video already has a completed
    ``episodic_memory_generator`` ExtractedNode with a persisted vector, the
    pipeline stages are skipped and :func:`m3.resume.run_qa_only`
    drives ``OGMemoryControlAdapter`` directly against the pre-loaded
    graph. A minimal ``pipeline_summary.json`` is written so downstream
    tooling (``scripts/m3_eval.py`` etc.) still discovers the run.
    """
    video = Path(args.video).resolve()
    if not video.exists():
        raise FileNotFoundError(f"video not found: {video}")

    video_id = args.video_id or video.stem
    work_dir = Path(args.work_dir).resolve()
    config_path = Path(args.config).resolve()

    spec = load_pipeline_spec(args.pipeline)
    # Smart-resume is opt-out: the probe layer short-circuits stages
    # whose work is already done in the graph + artefact state (Layer 2
    # of the resume strategy). Each driver additionally has Layer 1
    # per-source-node dedup (``--force`` overrides the driver layer).
    # Pass ``--no-smart-resume`` to disable the probe layer and always
    # run every stage from scratch.
    runner = PipelineRunner(smart_resume=not args.no_smart_resume)

    enable = list(args.enable_stage or [])
    disable = list(args.disable_stage or [])
    if args.detect_faces:
        enable.append("detect_faces")
    if args.build_index:
        enable.append("build_index")
    stage_ids = {stage.id for stage in spec.stages}
    set_values = [value for value in _legacy_set_overrides(args)
                  if value.split(".", 1)[0] in stage_ids] + list(args.set or [])
    runner.apply_overrides(
        spec,
        enable_stages=enable,
        disable_stages=disable,
        set_values=set_values,
    )

    ctx = runner.build_context(
        spec,
        video=video,
        video_id=video_id,
        seconds=args.seconds,
        work_dir=work_dir,
        config_path=config_path,
        graph_store=args.graph_store,
        connection_string=args.connection_string,
        account_id=args.account_id,
        variables={
            "force": args.force,
            "ffmpeg_bin": args.ffmpeg_bin,
            "ffprobe_bin": args.ffprobe_bin,
            "ffmpeg_log_level": args.ffmpeg_log_level,
        },
    )

    from m3.pipeline.provenance import validate_input
    validate_input(ctx, spec)

    # ----- Graph-aware resume (--reuse-graph) -----
    if args.reuse_graph and not args.force:
        if args.graph_store != "sql":
            logger.warning(
                "--reuse-graph requires --graph-store sql (current %s); "
                "ignoring --reuse-graph",
                args.graph_store,
            )
        else:
            try:
                from m3.resume import (
                    build_graph_service,
                    run_qa_only,
                    video_fully_indexed,
                    write_resume_summary,
                )

                graph, store_kind = build_graph_service(
                    args.graph_store, ctx.connection_string
                )
                from m3.pipeline.runner import _build_vector_index
                ctx.vector_index = _build_vector_index(ctx)
                probe = video_fully_indexed(
                    graph, args.account_id, video_id, vector_index=ctx.vector_index
                )
                logger.info(
                    "[%s] graph probe: %d clips, complete=%s "
                    "(missing_extracted=%d, unfinished=%d, missing_embedding=%d)",
                    video_id,
                    probe["video_clip_count"],
                    probe["complete"],
                    len(probe["missing_extracted"]),
                    len(probe["unfinished_extracted"]),
                    len(probe["missing_embedding"]),
                )
                if probe["complete"]:
                    annotations_path = Path(args.annotations).resolve() \
                        if getattr(args, "annotations", None) \
                        else (work_dir / "annotations" / "one_video.json")
                    summary_qa = run_qa_only(
                        graph_store=args.graph_store,
                        connection_string=ctx.connection_string,
                        account_id=args.account_id,
                        config_path=config_path,
                        annotations_path=annotations_path,
                        video_id=video_id,
                        work_dir=work_dir,
                        max_iterations=getattr(args, "max_iterations", 6),
                        timeout_seconds=int(getattr(args, "timeout_seconds", 90)),
                        top_k=getattr(args, "top_k", None),
                        resume=getattr(args, "resume", False),
                        verbose=args.verbose,
                    )
                    jsonl_path = Path(summary_qa["jsonl"])
                    summary_path = write_resume_summary(
                        work_dir=work_dir,
                        video_id=video_id,
                        pipeline_name=spec.name,
                        pipeline_version=spec.version,
                        jsonl_path=jsonl_path,
                        duration_sec=summary_qa["duration_sec"],
                        rc=(summary_qa["returncode"] or
                            (2 if summary_qa.get("status") == "failed" else 0)),
                    )
                    full_summary = {
                        **summary_qa,
                        "summary_path": str(summary_path),
                        "pipeline": {
                            "name": spec.name,
                            "version": spec.version,
                            "description": spec.description,
                        },
                    }
                    full_summary["summary_path"] = str(summary_path)
                    return full_summary
            except Exception as exc:
                raise RuntimeError(f"[{video_id}] reuse-graph failed; refusing expensive fallback") from exc

    return runner.run(spec, ctx)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one-video M3 analysis and create MemoryNodes.")
    p.add_argument("--pipeline", default=DEFAULT_PIPELINE,
                   help="YAML pipeline spec (default: %(default)s)")
    p.add_argument("--enable-stage", action="append", default=[])
    p.add_argument("--disable-stage", action="append", default=[])
    p.add_argument("--set", action="append", default=[],
                   help="Override a stage value, e.g. extract_frames.params.max_frames=16")
    p.add_argument("--video", required=True)
    p.add_argument("--video-id", default=None)
    p.add_argument("--seconds", type=int, default=120,
                   help="Analyze only the first N seconds; <=0 means full video (default: %(default)s)")
    p.add_argument("--work-dir", default="data/m3_runs/one_video")
    p.add_argument("--config", default="deploy/ogmemory.yaml")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None)
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID)
    p.add_argument("--reuse-graph", action="store_true",
                   help="Probe the SQL graph for completeness before "
                        "running the pipeline. If every VIDEO clip of "
                        "this video already has a completed "
                        "episodic_memory_generator ExtractedNode and a "
                        "persisted vector, skip the heavy stages and run "
                        "QA-only against the existing graph. Requires "
                        "--graph-store sql.")
    p.add_argument("--annotations", default=None,
                   help="Path to the M3-Bench annotations JSON used "
                        "by the QA-only resume path (defaults to "
                        "<work-dir>/annotations/one_video.json). "
                        "Ignored when --reuse-graph does not trigger.")
    p.add_argument("--max-iterations", type=int, default=6,
                   help="QA adapter max iterations when --reuse-graph "
                        "triggers the QA-only path.")
    p.add_argument("--timeout-seconds", type=int, default=90,
                   help="QA adapter per-call timeout for the QA-only "
                        "resume path.")
    p.add_argument("--top-k", type=int, default=None,
                   help="[Search] top-k for the QA-only resume path.")
    p.add_argument("--resume", action="store_true",
                   help="Forward --resume to m3.qa.run_react_qa when "
                        "--reuse-graph triggers the QA-only path "
                        "(appends to an existing JSONL instead of "
                        "overwriting).")

    # Backward-compatible convenience overrides for the default spec.
    p.add_argument("--llm", choices=("minimax", "openai", "mock"), default=None,
                   help="Override m3.episodic_llm.provider from config")
    audio = p.add_mutually_exclusive_group()
    audio.add_argument("--with-audio", dest="with_audio", action="store_true", default=None,
                       help="Enable audio (default: preserve pipeline YAML)")
    audio.add_argument("--no-audio", dest="with_audio", action="store_false")
    p.add_argument("--clip-seconds", type=int, default=30)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=16,
                   help="Max frames to extract per clip")
    p.add_argument("--episodic-max-frames", type=int, default=16,
                   help="Max extracted frames to send to the episodic LLM per clip")
    p.add_argument("--resize", type=int, default=512)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--keyframes-only", action="store_true")
    p.add_argument("--detect-faces", action="store_true",
                   help="Optional stage: run m3.detect_faces after frame extraction")
    p.add_argument("--face-backend", choices=("http", "mock"), default="http")
    p.add_argument("--cv-service-url", default=None)
    p.add_argument("--cv-timeout", type=float, default=None,
                   help="Override cv_service.timeout_seconds from config")
    p.add_argument("--face-min-confidence", type=float, default=0.5)
    p.add_argument("--face-max-faces", type=int, default=20)
    p.add_argument("--memory-topic", default=None)
    p.add_argument("--build-index", action="store_true")
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None,
                   help="Override m3.embedding.provider when --build-index is set")
    p.add_argument("--embedding-dimension", type=int, default=None)
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    p.add_argument("--ffprobe-bin", default="ffprobe")
    p.add_argument("--ffmpeg-log-level", default="error")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-smart-resume", action="store_true",
                   help="Disable the per-stage probe layer (Layer 2) "
                        "and always run every stage from scratch. The "
                        "per-source-node skip inside drivers (Layer 1, "
                        "e.g. voice_asr) is unaffected; use --force on "
                        "the individual driver to override that.")
    p.add_argument("--summary-format", choices=("json", "pretty"), default="json",
                   help="How to render the final pipeline summary "
                        "(default: %(default)s)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.resize == 0:
        args.resize = None
    # Default --video-id to the source file stem so the QA stage's
    # --video filter picks up the right video_id even when the caller
    # only passes --video. Without this, qa.params.video would be unset
    # and the QA stage would answer every video's questions.
    if not args.video_id:
        args.video_id = Path(args.video).stem
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = run_pipeline(args)
    except Exception as exc:
        logger.exception("M3 one-video pipeline failed: %s", exc)
        return 1
    if args.summary_format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.summary_format == "pretty":
        print(_format_summary_pretty(summary))
    else:
        # Should be unreachable thanks to argparse ``choices``.
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if summary.get("status") == "failed" or any(
        not result.get("ok", True)
        for result in (summary.get("stages") or {}).values()
    ) else 0


def _format_summary_pretty(summary: dict) -> str:
    """Render the runner summary as a human-friendly block.

    JSON remains the canonical machine-readable form (``--summary-format json``,
    also the default for backwards compatibility). ``pretty`` keeps the same
    data but lays it out for terminals and grep-friendly logs.
    """
    pipeline = summary.get("pipeline") or {}
    lines = [
        f"Pipeline: {pipeline.get('name', '?')} v{pipeline.get('version', '?')}",
        f"Video:    {summary.get('video_id', '?')}",
        f"Work dir: {summary.get('artifacts', {}).get('summary_json', '?')}",
        "",
        "Stages:",
    ]
    for stage_id, info in (summary.get("stages") or {}).items():
        status = "ok" if info.get("ok") else "FAILED"
        counts = info.get("counts") or {}
        counts_str = ", ".join(f"{k}={v}" for k, v in counts.items()) or "-"
        lines.append(f"  - {stage_id:<24} [{status}] {counts_str}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
