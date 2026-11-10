"""Phase 0 (continued): Load sliced clips as RawNodes into oG-Memory.

For each video listed in the M3-Bench annotation JSON, finds the 30s
clips produced by ``m3.preprocess`` and inserts one ``RawNode`` per
clip into oG-Memory's graph via :class:`GraphService`.

The resulting memory graph contains:

* one ``RawNode`` (raw_type=VIDEO) per clip, with ``metadata`` carrying
  ``{video_id, clip_id, start_sec, duration_sec, source_path}``
* a ``data/data.jsonl`` manifest in M3-Agent format plus an extra
  ``raw_node_id`` field so downstream operator pipelines can target the
  right node

The script is **idempotent** for the per-clip RawNode creation step:
re-running it on the same data is a no-op when clip files exist and the
``--force`` flag is not set. The output manifest is always rewritten.

Usage::

    # From m3-agent working dir, default in-memory store (CI / smoke tests):
    python -m m3.load_clips --annotations data/annotations/robot.json

    # Production: store in PostgreSQL via env var:
    export OGMEM_GRAPH_CONNECTION_STRING="host=... dbname=ogmemory"
    python -m m3.load_clips --annotations data/annotations/robot.json \\
        --graph-store sql
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

from m3.config import (
    M3BENCH_ACCOUNT_ID,
    M3BENCH_AGENT_ID,
    M3BENCH_USER_ID,
    LoadConfig,
    resolve_video_path,
    video_id_from_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GraphService construction
# ---------------------------------------------------------------------------

def build_graph_service(store_kind: str, connection_string: str | None, vector_index=None):
    """Construct a :class:`GraphService` backed by the requested store.

    ``store_kind`` is one of ``"memory"`` (default; uses
    :class:`MemoryGraphStore`) or ``"sql"`` (uses :class:`SQLGraphStore`
    with the given DSN; falls back to env var
    ``OGMEM_GRAPH_CONNECTION_STRING`` if not provided).

    ``vector_index`` is an optional :class:`VectorIndex` (typically an
    :class:`OpenGaussVectorIndex` for M3). When supplied, the
    :class:`GraphService` will route ``search_by_embedding`` through
    the index instead of falling back to the store's JSONB scan. Pass
    ``None`` for callers that don't have a vector index yet (legacy
    callers, unit tests of GraphStore directly).

    Imports are deferred so this module remains importable in
    environments where :mod:`psycopg2` is unavailable.
    """
    from service.graph_service import GraphService

    if store_kind == "memory":
        from providers.graph_store import MemoryGraphStore
        return GraphService(MemoryGraphStore(), vector_index=vector_index), "memory"

    if store_kind == "sql":
        dsn = connection_string or os.environ.get("OGMEM_GRAPH_CONNECTION_STRING")
        if not dsn:
            raise RuntimeError(
                "SQL store requested but no DSN provided. Pass "
                "--connection-string or set OGMEM_GRAPH_CONNECTION_STRING."
            )
        from providers.graph_store import SQLGraphStore
        # OGMEM_GRAPH_POOL_SIZE (default 32): keep >= the parallel-clip
        # worker count or worker threads serialize on pool checkout.
        graph_pool_size = int(os.environ.get("OGMEM_GRAPH_POOL_SIZE") or 32)
        return GraphService(
            SQLGraphStore(connection_string=dsn, pool_size=graph_pool_size),
            vector_index=vector_index,
        ), "sql"

    raise ValueError(f"Unknown graph store: {store_kind!r}")


# ---------------------------------------------------------------------------
# Clip discovery
# ---------------------------------------------------------------------------

def discover_clip_files(clips_root: Path, video_id: str) -> list[Path]:
    """Return sorted list of ``<clips_root>/<video_id>/<clip_id>.mp4``.

    Missing or empty directories yield an empty list (caller logs a
    warning and skips the video).
    """
    clip_dir = clips_root / video_id
    if not clip_dir.exists():
        return []
    return sorted(p for p in clip_dir.iterdir() if p.suffix == ".mp4" and p.stat().st_size > 0)


def clip_id_from_path(clip_path: Path) -> int:
    """Extract integer clip id from a path like ``.../living_room_06/12.mp4``."""
    return int(clip_path.stem)


def clip_duration_seconds(clip_path: Path, ffprobe_bin: str = "ffprobe") -> float:
    """Return clip duration in seconds via ffprobe.

    Falls back to 30.0 (the default clip length) if ffprobe is not
    available — this lets the loader run in environments without
    ffmpeg installed when durations aren't yet needed by downstream
    operators.
    """
    try:
        from m3.preprocess import ffprobe_duration
        return ffprobe_duration(clip_path, ffprobe_bin)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.debug("ffprobe unavailable for %s: %s; assuming 30s", clip_path, exc)
        return 30.0


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def load_clips_for_video(
    graph,
    ctx,
    *,
    video_id: str,
    source_video_path: Path,
    clip_files: list[Path],
    clip_seconds: int,
    ffprobe_bin: str,
) -> list[dict]:
    """Insert one RawNode per clip; return list of ``raw_node_id`` records.

    Each record is a dict with ``video_id``, ``clip_id``, ``raw_node_id``,
    and absolute clip path. Downstream scripts use this to wire operator
    pipelines against specific clips.

    Idempotency: re-running the loader against the same data is a no-op
    for both backends. We achieve this by computing a deterministic
    ``node_id = sha1(f"{video_id}/{clip_id}")[:32]`` and probing
    ``get_node`` first; only when the node is absent do we issue a
    ``create_node`` (which on the SQL store would otherwise raise
    ``UniqueViolation``).
    """
    from core.graph_enums import RawDataType

    records: list[dict] = []
    all_pre_existing = True
    for clip_path in clip_files:
        clip_id = clip_id_from_path(clip_path)
        node_id = hashlib.sha1(f"{video_id}/{clip_id}".encode()).hexdigest()[:32]

        if graph.get_raw_node(node_id, ctx) is not None:
            logger.debug("RawNode %s already exists, skipping insert", node_id)
        else:
            all_pre_existing = False
            duration = clip_duration_seconds(clip_path, ffprobe_bin=ffprobe_bin)
            metadata = {
                "video_id": video_id,
                "clip_id": clip_id,
                "start_sec": clip_id * clip_seconds,
                "duration_sec": duration,
                "source_path": str(clip_path.resolve()),
                "source_video_path": str(source_video_path),
            }
            graph.create_raw_node(
                raw_type=RawDataType.VIDEO,
                content="",
                ctx=ctx,
                metadata=metadata,
                node_id=node_id,
            )
        records.append({
            "video_id": video_id,
            "clip_id": clip_id,
            "raw_node_id": node_id,
            "clip_path": str(clip_path.resolve()),
        })

    # Build a sequential references chain: clip 0 -> clip 1 -> clip 2 ...
    # so a graph traversal starting at any clip can walk the whole
    # video in time order without needing a separate index. Skip the
    # whole chain when every clip node was already present — re-running
    # the loader must not duplicate edges.
    if len(records) >= 2 and not all_pre_existing:
        from core.graph_enums import GraphEdgeType
        for prev, curr in zip(records, records[1:]):
            graph.create_edge(
                source_id=prev["raw_node_id"],
                target_id=curr["raw_node_id"],
                edge_type=GraphEdgeType.REFERENCES,
                ctx=ctx,
                weight=1.0,
                properties={"relation": "next_clip", "video_id": video_id},
            )

    return records


def run(args: argparse.Namespace, graph=None, op_service=None) -> int:
    cfg = LoadConfig(
        annotations_file=args.annotations,
        clips_root=args.clips_root,
        videos_root=args.videos_root,
        output_jsonl=args.output_jsonl,
        account_id=args.account_id,
        clip_seconds=args.clip_seconds,
        only_annotated=args.only_annotated,
    )

    from core.models import RequestContext
    # PipelineRunner injects a shared graph + op_service so every
    # stage sees the same instance. CLI callers leave them as None
    # and we build fresh ones (legacy behaviour, preserved for
    # backwards compatibility with tests and standalone invocations).
    if graph is None:
        graph, store_kind = build_graph_service(args.graph_store, args.connection_string)
    else:
        store_kind = type(graph._store).__name__ if hasattr(graph, "_store") else "injected"
    ctx = RequestContext(
        account_id=cfg.account_id,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-load",
        trace_id=f"m3-load-{uuid.uuid4().hex}",
    )
    logger.info("Using graph store: %s, account_id=%s", store_kind, ctx.account_id)

    annotations_path = Path(cfg.annotations_file).resolve()
    if not annotations_path.exists():
        logger.error("Annotations file not found: %s", annotations_path)
        return 1
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    clips_root = Path(cfg.clips_root).resolve()
    videos_root = Path(cfg.videos_root).resolve()
    project_root = Path.cwd()
    if not clips_root.exists():
        logger.error("Clips root not found: %s — run m3.preprocess first", clips_root)
        return 1

    output_jsonl = Path(cfg.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Pre-create the parent directories for the manifest's M3-compatible
    # paths so downstream tools (M3-Agent scripts) can write into them.
    (clips_root.parent / "memory_graphs").mkdir(parents=True, exist_ok=True)
    (clips_root.parent / "intermediate_outputs").mkdir(parents=True, exist_ok=True)

    manifest_lines: list[str] = []
    total_clips = 0
    missing_videos: list[str] = []

    # --only_annotated=False processes every *.mp4 under clips_root, not
    # only the ones listed in annotations. Useful for back-fills and
    # smoke tests where the annotation file may be incomplete.
    if cfg.only_annotated:
        videos_to_process = list(annotations.items())
    else:
        clip_dirs = sorted(p.parent for p in clips_root.glob("*/*.mp4") if p.is_file())
        videos_to_process = [(d.name, {}) for d in clip_dirs]

    for video_id, info in videos_to_process:
        source_video: Path | None = None
        declared = info.get("video_path") if isinstance(info, dict) else None
        if declared:
            source_video = resolve_video_path(
                declared, project_root=project_root, videos_root=videos_root,
            )
            if source_video is None:
                logger.warning(
                    "Source video not resolvable for %s (declared=%s); manifest will lack a real path",
                    video_id, declared,
                )
        else:
            # only_annotated=False path: derive source video path from clip stem
            stem_clip = next(iter(discover_clip_files(clips_root, video_id)), None)
            if stem_clip is not None:
                source_video = videos_root / video_id_from_path(declared) if declared else videos_root / f"{video_id}.mp4"

        clip_files = discover_clip_files(clips_root, video_id)
        if not clip_files:
            logger.warning("No clips found for %s under %s", video_id, clips_root)
            missing_videos.append(video_id)
            continue

        records = load_clips_for_video(
            graph, ctx,
            video_id=video_id,
            source_video_path=source_video if source_video else Path(),
            clip_files=clip_files,
            clip_seconds=cfg.clip_seconds,
            ffprobe_bin=args.ffprobe_bin,
        )
        total_clips += len(records)

        mem_path = (clips_root.parent / "memory_graphs" / f"{video_id}.pkl").resolve()
        manifest_lines.append(json.dumps({
            "id": video_id,
            "video_path": str(source_video) if source_video else "",
            "clip_path": str((clips_root / video_id).resolve()),
            "mem_path": str(mem_path),
            "intermediate_path": str((clips_root.parent / "intermediate_outputs" / video_id).resolve()),
            # oG-Memory extras
            "raw_nodes": records,
            "account_id": ctx.account_id,
        }, ensure_ascii=False))

    with output_jsonl.open("w", encoding="utf-8") as f:
        for line in manifest_lines:
            f.write(line + "\n")

    logger.info(
        "Loaded %d clips across %d videos; manifest at %s; missing videos: %d",
        total_clips, len(manifest_lines), output_jsonl, len(missing_videos),
    )
    if missing_videos:
        missing_path = output_jsonl.parent / "load_missing_videos.json"
        with missing_path.open("w", encoding="utf-8") as f:
            json.dump(missing_videos, f, indent=2)
        return 2 if len(missing_videos) == len(annotations) else 0
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load M3-Bench clips into oG-Memory as RawNodes.")
    p.add_argument("--annotations", required=True,
                   help="Path to M3-Bench annotation JSON")
    p.add_argument("--clips-root", default="data/clips",
                   help="Root directory of sliced clips (default: %(default)s)")
    p.add_argument("--videos-root", default="data/videos",
                   help="Root directory for original videos, used to resolve "
                        "the source path in the manifest (default: %(default)s)")
    p.add_argument("--output-jsonl", default="data/data.jsonl",
                   help="Output manifest path (default: %(default)s)")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory",
                   help="Graph store backend (default: %(default)s)")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql "
                        "(falls back to OGMEM_GRAPH_CONNECTION_STRING env var)")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID,
                   help="Tenant account id (default: %(default)s)")
    p.add_argument("--clip-seconds", type=int, default=30,
                   help="Clip length in seconds — must match the value the "
                        "slicer used; default %(default)s matches M3-Agent")
    p.add_argument("--only-annotated", dest="only_annotated",
                   action="store_true", default=True,
                   help="Process only videos listed in annotations (default: true)")
    p.add_argument("--all-clips", dest="only_annotated",
                   action="store_false",
                   help="Process every *.mp4 under clips_root, ignoring annotations")
    p.add_argument("--ffprobe-bin", default="ffprobe")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
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
