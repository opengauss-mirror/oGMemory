"""Import M3-Bench ground-truth subtitles (.srt) as voice_asr graph nodes.

M3-Bench ships per-video .srt subtitles alongside the annotations; the
benchmark questions are largely conversational (dialogue about the
video), so a vision-only pipeline cannot score meaningfully without
speech. This tool ingests the ground-truth subtitles as ``voice_asr``
ExtractedNodes through the exact same operator machinery the real
``m3.voice_asr`` driver uses (``OperatorService.persist_extraction`` +
``VOICE_ASR_MATERIALIZATION``), producing a manifest JSONL
byte-compatible with the driver's output. Downstream stages
(``m3.episodic_memory --asr-manifest``, ``m3.index_episodic
--asr-manifest``) then treat it as a perfect-ASR transcript.

Use only for benchmark validation / pipeline testing where a whisper
deployment is not available — real runs should use the driver.

Usage (inside the m3-runner container)::

    python3 -m scripts.m3_srt_asr_import \
        --srt /data/m3_data/subtitles/robot/bedroom_01.srt \
        --video bedroom_01 \
        --output-jsonl data/m3_runs/<work>/voice_asr.jsonl \
        --graph-store sql \
        --connection-string postgresql://... \
        --config deploy/ogmemory.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m3._parallel import UnitError, parallel_clips
from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID
from m3.load_clips import build_graph_service

logger = logging.getLogger(__name__)

_SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def parse_srt(path: Path) -> list[dict]:
    """Parse an SRT file into absolute-time segment dicts."""
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\r*\n", text.strip())
    segments: list[dict] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        time_match = None
        for i, ln in enumerate(lines):
            m = _SRT_TIME.search(ln)
            if m:
                time_match = m
                text_lines = lines[i + 1:]
                break
        if time_match is None:
            continue
        g = [int(x) for x in time_match.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        body = " ".join(t.strip() for t in text_lines if t.strip())
        if not body:
            continue
        segments.append({
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(end - start, 3),
            "text": body,
        })
    segments.sort(key=lambda s: s["start_sec"])
    return segments


def run(args: argparse.Namespace) -> int:
    from core.graph_enums import RawDataType
    from core.models import RequestContext
    from extraction.operators import default_registry
    from extraction.operators.builtin import register_all_defaults
    from extraction.operators.builtin.voice_asr import VoiceASRExtractor
    from extraction.operators.base import ExtractionResult
    from m3.voice_asr import segment_row
    from service.operator_service import OperatorService

    graph, store_kind = build_graph_service(args.graph_store, args.connection_string)
    logger.info("Graph store: %s", store_kind)

    registry = default_registry()
    register_all_defaults(registry)
    op_service = OperatorService(graph, registry=registry)
    # The extractor instance is used only for its operator metadata
    # (extractor_type / schema / materialization policy); extract() is
    # never called — we supply the subtitle-derived ExtractionResult.
    extractor = VoiceASRExtractor(backend="mock")

    ctx = RequestContext(
        account_id=args.account_id,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-srt-asr",
        trace_id=f"m3-srt-{uuid.uuid4().hex}",
    )

    audio_nodes = (
        graph.list_raw_nodes(RawDataType.AUDIO, ctx)
        if hasattr(graph, "list_raw_nodes") else []
    )
    if args.video:
        wanted = set(args.video.split(","))
        audio_nodes = [
            n for n in audio_nodes
            if (n.metadata or {}).get("video_id") in wanted
        ]
    logger.info("Found %d AUDIO RawNodes for %s", len(audio_nodes), args.video)

    all_segments = parse_srt(Path(args.srt))
    logger.info("Parsed %d subtitle segments from %s", len(all_segments), args.srt)

    def _unit(audio_node):
        meta = audio_node.metadata or {}
        clip_start = float(meta.get("start_sec", 0) or 0)
        duration = float(meta.get("duration_sec", args.clip_seconds) or args.clip_seconds)
        clip_end = clip_start + max(duration, args.clip_seconds)
        rel = [
            {
                **seg,
                "start_sec": round(seg["start_sec"] - clip_start, 3),
                "end_sec": round(min(seg["end_sec"], clip_end) - clip_start, 3),
            }
            for seg in all_segments
            if seg["start_sec"] < clip_end and seg["end_sec"] > clip_start
        ]
        for seg in rel:
            seg["duration_sec"] = round(seg["end_sec"] - seg["start_sec"], 3)
            seg["no_speech_prob"] = 0.0
            seg["words"] = []
            seg["asr_id"] = uuid.uuid4().hex[:16]
        extraction = ExtractionResult(
            node_id=audio_node.node_id,
            extracted_content={
                "language": args.language,
                "duration_sec": round(clip_end - clip_start, 3),
                "model": "srt-ground-truth",
                "backend": "srt_import",
                "segments": rel,
            },
            confidence=1.0 if rel else 0.0,
            schema_used="voice_asr_v1",
        )
        result = op_service.persist_extraction(
            raw_node=audio_node,
            operator=extractor,
            extraction=extraction,
            ctx=ctx,
            refresh_extracted_children=True,
        )
        error = result.get("error")
        if error:
            return UnitError(f"persist_extraction failed: {error}")
        materialized = result.get("materialized_children") or []
        rows = []
        for m, seg in zip(materialized, rel):
            rows.append(segment_row(
                video_id=meta.get("video_id"),
                clip_id=meta.get("clip_id"),
                audio_node_id=audio_node.node_id,
                segment_node_id=m["child_node_id"],
                asr_id=seg.get("asr_id", ""),
                language=args.language,
                start_sec=seg.get("start_sec", 0.0),
                end_sec=seg.get("end_sec", 0.0),
                text=seg.get("text", ""),
                word_count=len(seg.get("words") or []),
                no_speech_prob=seg.get("no_speech_prob", 0.0),
            ))
        return rows

    rows_out: list[dict] = []
    failed = 0
    with open(Path(args.output_jsonl).resolve(), "w", encoding="utf-8") as f:
        for node, result in parallel_clips(audio_nodes, _unit):
            meta = node.metadata or {}
            if isinstance(result, UnitError):
                logger.error("srt import failed for %s: %s", node.node_id, result)
                failed += 1
                continue
            for row in result:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_out.append(row)

    logger.info(
        "Imported %d subtitle segments across %d clips (%d failed) -> %s",
        len(rows_out), len(audio_nodes), failed, args.output_jsonl,
    )
    return 1 if failed else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--srt", required=True, help="Path to the .srt subtitle file")
    p.add_argument("--video", default=None,
                   help="Comma-separated video_ids to import (filter AUDIO nodes)")
    p.add_argument("--output-jsonl", default="data/voice_asr.jsonl")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="sql")
    p.add_argument("--connection-string", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID)
    p.add_argument("--clip-seconds", type=float, default=30.0,
                   help="Clip window used to convert absolute to clip-relative time")
    p.add_argument("--language", default="en")
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
