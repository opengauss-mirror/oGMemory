"""Phase 1.5 driver: extract frames from every VIDEO RawNode and persist them.

For each VIDEO RawNode in the graph, this script invokes the
:class:`VideoFrameExtractor` operator via :class:`OperatorService`, which
materializes every sampled frame as its own ``IMAGE`` ``RawNode`` (with a
``REFERENCES`` edge back to the VIDEO) and writes a lightweight
``ExtractedNode`` summary listing the child IDs.

Heavy base64 JPEG payloads are NOT kept in the graph: the
``OperatorService`` decoder hands the bytes to the ``bytes_writer``
closure below, which writes each frame to disk at
``frames_root/<video_id>/<clip_id>/<frame_idx>.jpg`` and records the path
on the child ``RawNode.metadata.source_path``.

Idempotency: deterministic child node IDs derived from
``sha1("frame/{source_raw_id}/{frame_idx}")`` plus
``OperatorService``'s ``skip_existing=True`` make re-runs no-ops.

Usage::

    python -m m3.load_clips --annotations data/annotations/robot.json \\
        --clips-root data/clips --videos-root data/videos
    python -m m3.extract_frames \\
        --manifest data/data.jsonl \\
        --frames-root data/frames
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

from m3._parallel import UnitError, parallel_clips
from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID
from m3.resume_status import is_completed_extraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic IDs (kept here because the M3-Bench manifest format pins
# them — the operator layer uses its own derivation that's equivalent for
# new data, but existing manifests and downstream tooling rely on these
# specific formulas).
# ---------------------------------------------------------------------------

def extracted_node_id(extractor_type: str, source_raw_id: str) -> str:
    """Deterministic id for the parent ``ExtractedNode``.

    Reused from the pre-refactor script so existing manifests stay
    readable. ``OperatorService.run_operator`` writes the ExtractedNode
    with this id (the m3 driver calls it twice: once for the idempotency
    pre-check, once as the desired node_id via ``desired_node_id``).
    """
    seed = f"extracted/{extractor_type}/{source_raw_id}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


def frame_node_id(video_id: str, clip_id: int, frame_idx: int) -> str:
    """Deterministic id for a frame ``IMAGE`` ``RawNode`` (legacy formula).

    Kept for backward compatibility with existing manifests. The operator
    layer's :data:`VIDEO_FRAME_MATERIALIZATION` derives a different but
    equivalent id from ``source_raw_id + frame_idx``; both are unique
    per (video, clip, frame_idx). New code can use either.
    """
    seed = f"frame/{video_id}/{clip_id}/{frame_idx}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Frame persistence
# ---------------------------------------------------------------------------

def _write_frame_jpeg(
    frames_root: Path,
    video_id: str,
    clip_id: int,
    frame_idx: int,
    jpeg_bytes: bytes,
) -> Path:
    out_dir = frames_root / video_id / str(clip_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{frame_idx}.jpg"
    out_path.write_bytes(jpeg_bytes)
    return out_path


def _build_frame_writer(
    frames_root: Path,
    video_id: str,
    clip_id: int,
):
    """Build a ``bytes_writer`` closure for a single (video, clip) pair.

    The closure captures the filesystem layout for this clip so
    :func:`extraction.operators.materialize.materialize_children` can
    call it with ``(item, raw_bytes)`` and get back a string path to
    store under ``RawNode.metadata.source_path``.
    """
    def _writer(item: dict, raw_bytes: bytes) -> str:
        out_path = _write_frame_jpeg(
            frames_root, video_id, clip_id, int(item["frame_idx"]), raw_bytes,
        )
        return str(out_path)
    return _writer


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def extract_frames_for_clip(
    graph,
    op_service,
    ctx,
    *,
    video_raw_node,
    video_id: str,
    clip_id: int,
    frames_root: Path,
    fps: int = 5,
    max_frames: int | None = None,
    keyframe_threshold: float = 0.1,
    resize: int | None = 720,
    jpeg_quality: int = 85,
    keyframes_only: bool = False,
) -> dict:
    """Run :class:`VideoFrameExtractor` on one VIDEO RawNode via ``op_service``.

    Returns a dict suitable for the manifest::

        {
          "video_id": "...",
          "clip_id": 7,
          "video_raw_node_id": "...",
          "extracted_node_id": "...",
          "child_node_ids": [...],   # IMAGE RawNode IDs
          "child_count": 12,
          "resumed": True/False,
        }

    ``op_service`` owns all graph-side persistence (operator extraction,
    child materialization, ExtractedNode summary); this function only
    decides the bytes-writer closure and tracks the manifest-shaped
    summary that m3 tooling downstream relies on.
    """
    ext_id = extracted_node_id("video_frame_extractor", video_raw_node.node_id)
    # Idempotency pre-check: if the parent ExtractedNode already exists
    # with a successful result, skip work entirely and return the
    # lightweight summary. Pending/failed nodes (e.g. from a previously
    # failed run) are intentionally NOT reused, so the next run actually
    # re-extracts frames and refreshes the graph.
    existing = graph.get_extracted_node(ext_id, ctx)
    if existing is not None and is_completed_extraction(existing.extraction_status):
        content = existing.extracted_content or {}
        child_ids = content.get("child_node_ids") or []
        return {
            "video_id": video_id,
            "clip_id": clip_id,
            "video_raw_node_id": video_raw_node.node_id,
            "extracted_node_id": ext_id,
            "child_node_ids": list(child_ids),
            "child_count": len(child_ids),
            "resumed": True,
        }

    bytes_writer = _build_frame_writer(frames_root, video_id, clip_id)

    # Instantiate the operator directly so per-clip config (fps, resize,
    # keyframes_only, ...) actually reconfigures it — :meth:`OperatorService
    # .run_operator` looks up a registry-cached instance and ignores
    # caller-supplied kwargs. We hand the resulting ExtractionResult to
    # :meth:`persist_extraction` so the service still owns all graph
    # side effects (materialization, ExtractedNode write, lightweight
    # summary).
    from extraction.operators.builtin import VideoFrameExtractor
    extractor = VideoFrameExtractor(
        fps=fps,
        max_frames=max_frames,
        keyframe_threshold=keyframe_threshold,
        resize=resize,
        jpeg_quality=jpeg_quality,
        keyframes_only=keyframes_only,
    )
    extraction = extractor.extract(video_raw_node, ctx)
    result = op_service.persist_extraction(
        raw_node=video_raw_node,
        operator=extractor,
        extraction=extraction,
        ctx=ctx,
        bytes_writer=bytes_writer,
        desired_node_id=ext_id,
    )
    if result.get("error"):
        raise RuntimeError(
            f"video_frame_extractor failed for {video_raw_node.node_id}: "
            f"{result['error']}"
        )

    materialized = result.get("materialized_children") or []
    child_ids = [m["child_node_id"] for m in materialized]
    if not child_ids and result.get("extracted_content", {}).get("child_count"):
        # Opened successfully but produced zero frames (e.g. very short
        # video or keyframes_only with no detected keyframes). Keep the
        # empty ExtractedNode so future runs don't re-process.
        pass

    # After-materialization sanity: the operator's idempotency check is
    # per-child, so the parent ExtractedNode id is whatever OperatorService
    # assigned. Surface that to the caller under our legacy key.
    actual_ext_id = result.get("extracted_node_id") or ext_id
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "video_raw_node_id": video_raw_node.node_id,
        "extracted_node_id": actual_ext_id,
        "child_node_ids": child_ids,
        "child_count": len(child_ids),
        "resumed": False,
    }


# ---------------------------------------------------------------------------
# Driver entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace, graph=None, op_service=None) -> int:
    """CLI entry point. Tests pass ``graph``/``op_service`` to share state
    with a fixture; production callers leave them as ``None`` and the
    driver builds fresh instances from ``--graph-store``."""
    from core.models import RequestContext
    from m3.load_clips import build_graph_service

    if graph is None:
        graph, store_kind = build_graph_service(args.graph_store, args.connection_string)
    else:
        # Caller (typically a test) supplied the graph; report based on
        # the store kind we can introspect.
        store_kind = type(graph._store).__name__ if hasattr(graph, "_store") else "injected"

    if op_service is None:
        from extraction.operators import OperatorRegistry, default_registry
        from extraction.operators.builtin import register_all_defaults
        registry: OperatorRegistry = default_registry()
        register_all_defaults(registry)

        from service.operator_service import OperatorService
        op_service = OperatorService(graph, registry=registry)

    ctx = RequestContext(
        account_id=args.account_id,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-extract-frames",
        trace_id=f"m3-frames-{uuid.uuid4().hex}",
    )
    logger.info("Using graph store: %s, account_id=%s", store_kind, ctx.account_id)

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        logger.error("Manifest not found: %s (run m3.load_clips first)", manifest_path)
        return 1
    manifest_lines = manifest_path.read_text(encoding="utf-8").strip().split("\n")
    manifest_records = [json.loads(line) for line in manifest_lines if line]

    frames_root = Path(args.frames_root).resolve()
    frames_root.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    output_lines: list[str] = []
    total_children = 0
    total_extractions = 0
    failed: list[dict] = []

    only_videos = set(args.video.split(",")) if args.video else None

    # Flatten the (video, clip) pairs into per-clip descriptors; each
    # worker opens its own cv2.VideoCapture, so decode concurrency is
    # limited only by cores. The main thread serialises output writes
    # in input order.
    clips: list[dict] = []
    for video_record in manifest_records:
        video_id = video_record["id"]
        if only_videos and video_id not in only_videos:
            continue
        for raw_record in video_record.get("raw_nodes", []):
            clip_id = raw_record["clip_id"]
            video_node = graph.get_raw_node(raw_record["raw_node_id"], ctx)
            if video_node is None:
                logger.warning(
                    "VIDEO RawNode %s missing for %s clip %d",
                    raw_record["raw_node_id"], video_id, clip_id,
                )
                continue
            clips.append({
                "video_id": video_id,
                "clip_id": clip_id,
                "video_node": video_node,
            })

    def _unit(clip: dict) -> dict:
        video_id = clip["video_id"]
        clip_id = clip["clip_id"]
        clip_ctx = dataclasses.replace(
            ctx, trace_id=f"{ctx.trace_id}-{video_id}-{clip_id}"
        )
        return extract_frames_for_clip(
            op_service=op_service, graph=graph, ctx=clip_ctx,
            video_raw_node=clip["video_node"],
            video_id=video_id,
            clip_id=clip_id,
            frames_root=frames_root,
            fps=args.fps,
            max_frames=args.max_frames,
            keyframe_threshold=args.keyframe_threshold,
            resize=args.resize,
            jpeg_quality=args.jpeg_quality,
            keyframes_only=args.keyframes_only,
        )

    for clip, result in parallel_clips(clips, _unit):
        if isinstance(result, UnitError):
            logger.error(
                "Frame extraction failed for %s clip %s: %s",
                clip["video_id"], clip["clip_id"], result,
                exc_info=result.exception,
            )
            failed.append({
                "video_id": clip["video_id"],
                "clip_id": clip["clip_id"],
                "error": str(result),
            })
            continue
        output_lines.append(json.dumps({
            "video_id": result["video_id"],
            "clip_id": result["clip_id"],
            "video_raw_node_id": result["video_raw_node_id"],
            "extracted_node_id": result["extracted_node_id"],
            "child_node_ids": result["child_node_ids"],
            "child_count": result["child_count"],
            "resumed": result["resumed"],
        }, ensure_ascii=False))
        total_children += result["child_count"]
        if not result["resumed"]:
            total_extractions += 1

    with output_jsonl.open("w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    logger.info(
        "Wrote %d extraction records (%d new, %d frames, %d failures) -> %s",
        len(output_lines), total_extractions, total_children, len(failed), output_jsonl,
    )
    if failed:
        failed_path = output_jsonl.parent / "extract_frames_failures.json"
        with failed_path.open("w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        return 2 if not output_lines else 0
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run video_frame_extractor on every VIDEO RawNode and persist frames as IMAGE nodes."
    )
    p.add_argument("--manifest", required=True,
                   help="Path to data.jsonl produced by m3.load_clips")
    p.add_argument("--frames-root", default="data/frames",
                   help="Directory for JPEG byte payloads (default: %(default)s)")
    p.add_argument("--output-jsonl", default="data/frames.jsonl",
                   help="Output manifest path (default: %(default)s)")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN; falls back to OGMEM_GRAPH_CONNECTION_STRING")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID)
    p.add_argument("--video", default=None,
                   help="Comma-separated list of video_ids to process (default: all)")
    # Operator config — forwarded to VideoFrameExtractor.
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--keyframe-threshold", type=float, default=0.1)
    p.add_argument("--resize", type=int, default=720)
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--keyframes-only", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
