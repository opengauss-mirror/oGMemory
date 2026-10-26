"""Phase 1.7 driver: detect faces in every IMAGE RawNode (frame).

For each IMAGE RawNode in the graph (produced by ``m3.extract_frames``),
this script invokes :class:`FaceDetectionExtractor` via
:class:`OperatorService`, which materializes every detected face as its
own ``ExtractedNode`` carrying bbox / confidence / embedding / detector
metadata. The auto-wired ``EXTRACTED_FROM`` edge points from the source
frame back to each face.

Unlike ``m3.extract_frames``, no per-call ``bytes_writer`` is needed:
face embeddings are 512 floats (~2 KB) and live directly on the
``ExtractedNode.extracted_content``. The CV inference service
(``scripts/cv_service.py``) does the actual detection — this driver
just configures the operator and persists the structured output.

Backends:
  - Default ``HTTPBackend`` pointing at a running ``cv_service``
    (recommended: real InsightFace on the CPU box via the proxy).
  - For tests and offline smoke runs, ``--backend mock`` swaps in the
    in-process ``MockBackend`` and the driver works without any service.

Idempotency: deterministic face IDs derived from
``sha1("face/{source_frame_id}/{face_id}")`` plus
``OperatorService``'s ``skip_existing=True`` make re-runs no-ops.

Usage::

    # After extract_frames
    python -m m3.extract_frames --manifest data/data.jsonl \\
        --frames-root data/frames

    # Run face detection against a running cv_service
    OGMEM_CV_SERVICE_URL=http://127.0.0.1:8081 \\
    python -m m3.detect_faces --frames-manifest data/frames.jsonl

    # Or with the in-process mock for tests / offline runs
    python -m m3.detect_faces --frames-manifest data/frames.jsonl \\
        --backend mock --output-jsonl data/faces.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID
from m3.resume_status import is_completed_extraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic IDs (manifest format pins these — keep formulas stable
# across re-runs so downstream tooling can look faces up by id).
# ---------------------------------------------------------------------------

def extracted_node_id(extractor_type: str, source_raw_id: str) -> str:
    """Deterministic id for the parent face_detection ``ExtractedNode``."""
    seed = f"extracted/{extractor_type}/{source_raw_id}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def detect_faces_for_frame(
    graph,
    op_service,
    ctx,
    *,
    frame_raw_node,
    backend,
    min_confidence: float = 0.5,
    max_faces: int = 20,
    force: bool = False,
) -> dict:
    """Run :class:`FaceDetectionExtractor` on one IMAGE RawNode via ``op_service``.

    Returns a manifest-shaped dict::

        {
          "frame_node_id": "...",
          "extracted_node_id": "...",
          "child_node_ids": [...],   # face ExtractedNode ids
          "face_count": int,
          "resumed": bool,
        }

    The parent ``ExtractedNode`` carries the lightweight summary
    (``face_count`` + ``child_node_ids``); each child ExtractedNode
    holds the per-detection bbox / confidence / embedding.
    """
    ext_id = extracted_node_id("face_detection", frame_raw_node.node_id)

    # Idempotency pre-check: skip work when the parent ExtractedNode
    # already exists with a successful result. Re-runs return the
    # cached child_node_ids and mark the record as resumed so the
    # manifest reflects stability.
    #
    # Crucially, this is *not* a free pass: nodes left over from a
    # failed or partial previous run (extraction_status pending/failed
    # or no child_node_ids recorded) must be re-processed. Otherwise
    # the cv_service is silently skipped on every rerun and stale
    # 0-faces records are kept forever.
    existing = graph.get_extracted_node(ext_id, ctx)
    # Layer-1 resume: the parent face_detection ExtractedNode's
    # extraction_status reflects whether the CV call completed; if so,
    # reuse the cached child_node_ids without re-firing the HTTP
    # request. ``--force`` (forwarded as the ``force`` kwarg) bypasses
    # this so users can rerun after changing the CV backend or
    # min_confidence threshold. Without this check, every retry
    # re-fires the HTTP call for every frame (~1-5 s per frame on the
    # cv_service proxy) — the dominant cost on long videos.
    if existing is not None and is_completed_extraction(
        existing.extraction_status
    ) and not force:
        content = existing.extracted_content or {}
        child_ids = content.get("child_node_ids") or []
        return {
            "frame_node_id": frame_raw_node.node_id,
            "extracted_node_id": ext_id,
            "child_node_ids": list(child_ids),
            "face_count": len(child_ids),
            "resumed": True,
        }

    # Materialize JPEG bytes for the operator. m3.extract_frames writes
    # the heavy per-frame bytes to disk via the bytes_writer and the
    # IMAGE RawNode's ``content`` is left empty by design (the graph row
    # should stay light). Operators that need the bytes, like the HTTP
    # face backend, must read them back from ``source_path``.
    if not frame_raw_node.content:
        meta = frame_raw_node.metadata or {}
        src_path = meta.get("source_path")
        if src_path and Path(src_path).exists():
            frame_raw_node.content = Path(src_path).read_bytes()

    from extraction.operators.builtin import FaceDetectionExtractor
    extractor = FaceDetectionExtractor(
        backend=backend,
        min_confidence=min_confidence,
        max_faces=max_faces,
    )
    extraction = extractor.extract(frame_raw_node, ctx)
    result = op_service.persist_extraction(
        raw_node=frame_raw_node,
        operator=extractor,
        extraction=extraction,
        ctx=ctx,
        # No bytes_writer: face embeddings live directly on each
        # ExtractedNode.extracted_content (set by FACE_DETECTION_MATERIALIZATION).
        desired_node_id=ext_id,
    )
    if result.get("error"):
        raise RuntimeError(
            f"face_detection failed for {frame_raw_node.node_id}: {result['error']}"
        )

    materialized = result.get("materialized_children") or []
    child_ids = [m["child_node_id"] for m in materialized]
    return {
        "frame_node_id": frame_raw_node.node_id,
        "extracted_node_id": ext_id,
        "child_node_ids": child_ids,
        "face_count": len(child_ids),
        "resumed": False,
    }


# ---------------------------------------------------------------------------
# Driver entry point
# ---------------------------------------------------------------------------

def _build_backend(args):
    """Construct the CVBackend based on --backend flag.

    Resolve explicit CLI options, environment overrides and ``cv_service``
    in the selected YAML. Mock runs do not need a CV service or config.
    """
    backend_name = args.backend
    if backend_name == "mock":
        from extraction.operators.backends import MockBackend
        return MockBackend()

    if backend_name != "http":
        raise ValueError(f"unknown backend: {backend_name}")

    from extraction.operators.backends import HTTPBackend
    from m3.config import load_cv_service_config
    config = load_cv_service_config(
        getattr(args, "config", None),
        base_url=args.cv_service_url,
        timeout_seconds=args.cv_timeout,
        auth_token=args.cv_auth_token,
    )
    return HTTPBackend(
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        auth_token=config.auth_token,
    )


def run(args: argparse.Namespace, graph=None, op_service=None) -> int:
    """CLI entry point. Tests pass ``graph``/``op_service`` to share state
    with a fixture; production callers leave them as ``None`` and the
    driver builds fresh instances from ``--graph-store``.
    """
    from core.models import RequestContext

    if graph is None:
        from m3.load_clips import build_graph_service
        graph, store_kind = build_graph_service(args.graph_store, args.connection_string)
    else:
        store_kind = type(graph._store).__name__ if hasattr(graph, "_store") else "injected"

    if op_service is None:
        from extraction.operators import OperatorRegistry, default_registry
        from extraction.operators.builtin import register_all_defaults
        registry: OperatorRegistry = default_registry()
        register_all_defaults(registry)

        from service.operator_service import OperatorService
        op_service = OperatorService(graph, registry=registry)

    backend = _build_backend(args)

    ctx = RequestContext(
        account_id=args.account_id,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-detect-faces",
        trace_id=f"m3-faces-{uuid.uuid4().hex}",
    )
    logger.info(
        "Using graph store: %s, backend: %s, account_id=%s",
        store_kind, args.backend, ctx.account_id,
    )

    manifest_path = Path(args.frames_manifest).resolve()
    if not manifest_path.exists():
        logger.error(
            "Frames manifest not found: %s (run m3.extract_frames first)",
            manifest_path,
        )
        return 1
    # frames.jsonl is JSONL where each line is a frame extraction record:
    #   {"video_id": ..., "clip_id": ..., "video_raw_node_id": ...,
    #    "extracted_node_id": ..., "child_node_ids": [...], ...}
    # We iterate child_node_ids per line and run face_detection on each
    # frame RawNode.
    manifest_lines = manifest_path.read_text(encoding="utf-8").strip().split("\n")

    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    output_lines: list[str] = []
    total_faces = 0
    total_extractions = 0
    failed: list[dict] = []

    only_videos = set(args.video.split(",")) if args.video else None

    for line in manifest_lines:
        if not line:
            continue
        record = json.loads(line)
        video_id = record["video_id"]
        if only_videos and video_id not in only_videos:
            continue
        for frame_node_id in record.get("child_node_ids", []):
            frame_node = graph.get_raw_node(frame_node_id, ctx)
            if frame_node is None:
                logger.warning(
                    "Frame RawNode %s missing for %s clip %d",
                    frame_node_id, video_id, record.get("clip_id"),
                )
                continue
            try:
                result = detect_faces_for_frame(
                    graph=graph, op_service=op_service, ctx=ctx,
                    frame_raw_node=frame_node,
                    backend=backend,
                    min_confidence=args.min_confidence,
                    max_faces=args.max_faces,
                    force=args.force,
                )
            except Exception as exc:
                logger.exception(
                    "Face detection failed for frame %s", frame_node_id,
                )
                failed.append({
                    "video_id": video_id,
                    "clip_id": record.get("clip_id"),
                    "frame_node_id": frame_node_id,
                    "error": str(exc),
                })
                continue

            output_lines.append(json.dumps({
                "video_id": video_id,
                "clip_id": record.get("clip_id"),
                "frame_node_id": result["frame_node_id"],
                "extracted_node_id": result["extracted_node_id"],
                "child_node_ids": result["child_node_ids"],
                "face_count": result["face_count"],
                "resumed": result["resumed"],
            }, ensure_ascii=False))
            total_faces += result["face_count"]
            if not result["resumed"]:
                total_extractions += 1

    with output_jsonl.open("w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    logger.info(
        "Wrote %d extraction records (%d new, %d faces, %d failures) -> %s",
        len(output_lines), total_extractions, total_faces, len(failed), output_jsonl,
    )
    if failed:
        failed_path = output_jsonl.parent / "detect_faces_failures.json"
        with failed_path.open("w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        return 2 if not output_lines else 0
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run face_detection on every IMAGE RawNode and persist "
            "each detected face as an ExtractedNode."
        )
    )
    p.add_argument("--frames-manifest", required=True,
                   help="Path to frames.jsonl produced by m3.extract_frames")
    p.add_argument("--config", default=None,
                   help="ogmemory.yaml containing cv_service (default: $OGMEM_CONFIG or deploy/ogmemory.yaml)")
    p.add_argument("--output-jsonl", default="data/faces.jsonl",
                   help="Output manifest path (default: %(default)s)")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN; falls back to OGMEM_GRAPH_CONNECTION_STRING")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID)
    p.add_argument("--video", default=None,
                   help="Comma-separated list of video_ids to process (default: all)")
    p.add_argument("--force", action="store_true",
                   help="Re-run CV inference even when the parent "
                        "face_detection ExtractedNode is already completed. "
                        "Default: skip frames whose face_detection already "
                        "succeeded and reuse the cached child_node_ids.")
    # Backend selection
    p.add_argument("--backend", choices=("http", "mock"), default="http",
                   help="CV backend (default: %(default)s)")
    p.add_argument("--cv-service-url", default=None,
                   help="Override $OGMEM_CV_SERVICE_URL / cv_service.base_url")
    p.add_argument("--cv-timeout", type=float, default=None,
                   help="Override cv_service.timeout_seconds (fallback: 30 seconds)")
    p.add_argument("--cv-auth-token", default=None,
                   help="Optional Bearer token for the CV service")
    # Operator config — forwarded to FaceDetectionExtractor.
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--max-faces", type=int, default=20)
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
