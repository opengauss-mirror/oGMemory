"""M3-Bench voice ASR driver.

Iterates over every ``AUDIO`` ``RawNode`` in the graph, runs
:class:`VoiceASRExtractor` via ``OperatorService``, and writes a
per-segment JSONL with provenance (mirrors the convention in
``m3/voice_segmentation.py``).

Backends (selected by ``--backend``):

* ``faster_whisper`` (default) — local CTranslate2 Whisper via the
  ``faster-whisper`` package. CPU-runnable.
* ``openai_http`` — POSTs to an OpenAI-compatible
  ``/audio/transcriptions`` endpoint (Whisper server on a
  separate GPU / NPU host).
* ``mock`` — deterministic stub for tests.

Output manifest (``data/voice_asr.jsonl``, one row per ASR
segment)::

    {
      "video_id": "bedroom_03",
      "clip_id": 7,
      "audio_node_id": "raw_abc123",
      "segment_node_id": "ext_def456",
      "asr_id": "asr_abcdef0123456789",
      "language": "en",
      "start_sec": 0.32,
      "end_sec": 4.18,
      "duration_sec": 3.86,
      "text": "Hello and welcome back to the show.",
      "word_count": 6,
      "no_speech_prob": 0.012,
    }

Usage::

    python -m m3.voice_asr \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --output-jsonl data/voice_asr.jsonl

    # Local Whisper with a non-default model
    python -m m3.voice_asr \\
        --graph-store memory \\
        --backend faster_whisper --model large-v3

    # HTTP backend (point at a remote Whisper server)
    python -m m3.voice_asr \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --backend openai_http --base-url http://npu:8000/v1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

from core.graph_enums import RawDataType
from core.models import RequestContext
from m3._parallel import ThreadLocalModels, UnitError, parallel_clips
from m3.audio_cache import AudioCache, signature
from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-segment row helper
# ---------------------------------------------------------------------------

def segment_row(
    *,
    video_id: str | None,
    clip_id: int | None,
    audio_node_id: str,
    segment_node_id: str,
    asr_id: str,
    language: str | None,
    start_sec: float,
    end_sec: float,
    text: str,
    word_count: int,
    no_speech_prob: float,
) -> dict:
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "audio_node_id": audio_node_id,
        "segment_node_id": segment_node_id,
        "asr_id": asr_id,
        "language": language,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": round(end_sec - start_sec, 3),
        "text": text,
        "word_count": word_count,
        "no_speech_prob": no_speech_prob,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_voice_asr(args: argparse.Namespace, graph=None) -> int:
    """Walk the graph's AUDIO RawNodes, run the operator via
    OperatorService, write per-segment rows."""
    if graph is None:
        from m3.load_clips import build_graph_service
        graph, store_kind = build_graph_service(
            args.graph_store, args.connection_string,
        )
        logger.info("Graph store: %s", store_kind)
    else:
        store_kind = (
            type(graph._store).__name__
            if hasattr(graph, "_store") else "injected"
        )

    from extraction.operators import default_registry
    from extraction.operators.builtin import register_all_defaults
    from extraction.operators.builtin.voice_asr import VoiceASRExtractor
    from service.operator_service import OperatorService

    registry = default_registry()
    register_all_defaults(registry)
    op_service = OperatorService(graph, registry=registry)

    # faster-whisper's CTranslate2 model is not safe to share across
    # threads: build one extractor per worker thread, lazily.
    extractor_factory = ThreadLocalModels(
        lambda: VoiceASRExtractor(
            backend=args.backend,
            model=args.model,
            language=args.language,
            beam_size=args.beam_size,
            word_timestamps=not args.no_word_timestamps,
            vad_filter=not args.no_vad_filter,
            base_url=args.base_url,
            api_key=args.api_key,
            device=args.device,
            timeout_sec=args.timeout_sec,
        ),
        name="voice_asr_extractor",
    )

    ctx = RequestContext(
        account_id=getattr(args, "account_id", M3BENCH_ACCOUNT_ID),
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-voice-asr",
        trace_id=f"m3-va-{uuid.uuid4().hex}",
    )

    def _list_audio(raw_type):
        # Bench-scale accounts hold >4096 AUDIO nodes; the default LIMIT
        # silently drops late-created videos' nodes and voice_asr then
        # aborts with "produced 0 segments without completed audio".
        # ``limit=None`` -> LIMIT ALL on the SQL store; other backends may
        # not accept the kwarg, so fall back to the plain call.
        try:
            return graph.list_raw_nodes(raw_type, ctx, limit=None)
        except TypeError:
            return graph.list_raw_nodes(raw_type, ctx)

    audio_nodes = _list_audio(RawDataType.AUDIO) if hasattr(graph, "list_raw_nodes") else []
    if not audio_nodes and hasattr(graph, "list_raw_nodes"):
        # Fallback: list every RawNode and filter.
        all_nodes = _list_audio(None)
        audio_nodes = [
            n for n in all_nodes
            if str(getattr(n, "raw_type", "")) == "audio"
        ]
    logger.info("Found %d AUDIO RawNodes in the graph", len(audio_nodes))
    if args.video:
        wanted = set(args.video.split(","))
        audio_nodes = [
            n for n in audio_nodes
            if (n.metadata or {}).get("video_id") in wanted
        ]
        logger.info("After --video filter (%s): %d AUDIO nodes",
                    args.video, len(audio_nodes))

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache = AudioCache(graph, ctx, "voice_asr")

    total_segments = 0
    failed = 0
    resumed_segments = 0
    cached_audio = 0
    executed_audio = 0

    def _unit(audio_node) -> dict:
        """Transcribe one AUDIO node; return rows for the main thread.

        The unit never writes the JSONL directly — the main thread
        serialises writes in input order. Returns a dict with
        ``rows``/``resumed``/``cached``/``executed``/``error`` keys; a
        raised exception surfaces as a UnitError from parallel_clips.
        """
        meta = audio_node.metadata or {}
        video_id = meta.get("video_id")
        clip_id = meta.get("clip_id")
        node_ctx = dataclasses.replace(
            ctx, trace_id=f"{ctx.trace_id}-{audio_node.node_id}"
        )

        fingerprint = signature(audio_node, args, "voice_asr", ctx.account_id)
        cached = None if args.force else cache.lookup(audio_node, fingerprint)
        if cached is not None:
            logger.info(
                "voice_asr resume: %s already has %d cached segments; "
                "skipping Whisper (pass --force to re-transcribe)",
                audio_node.node_id, len(cached),
            )
            rows = []
            for child in cached:
                ec = child.extracted_content or {}
                rows.append(segment_row(
                    video_id=video_id,
                    clip_id=clip_id,
                    audio_node_id=audio_node.node_id,
                    segment_node_id=child.node_id,
                    asr_id=ec.get("asr_id", ""),
                    language=ec.get("language"),
                    start_sec=ec.get("start_sec", 0.0),
                    end_sec=ec.get("end_sec", 0.0),
                    text=ec.get("text", ""),
                    word_count=len(ec.get("words") or []),
                    no_speech_prob=ec.get("no_speech_prob", 0.0),
                ))
            return {
                "rows": rows,
                "resumed": len(rows),
                "cached": True,
                "executed": False,
                "error": None,
            }

        cache.invalidate(audio_node)
        # Extract segments BEFORE persist_extraction so
        # lightweight_summary() doesn't strip the heavy
        # ``segments`` payload out from under us.
        extractor = extractor_factory.get()
        extraction = extractor.extract(audio_node, node_ctx)
        segments = list(
            extraction.extracted_content.get("segments", [])
        )
        language = extraction.extracted_content.get("language")
        # The materialization policy reads language per segment.
        # Preserve it so regenerated cached manifests match fresh ones.
        for segment in segments:
            segment["language"] = language
        result = op_service.persist_extraction(
            raw_node=audio_node,
            operator=extractor,
            extraction=extraction,
            ctx=node_ctx,
            refresh_extracted_children=True,
        )
        cache.finish(audio_node, fingerprint, result, len(segments))
        error = result.get("error")
        if error:
            return {
                "rows": [], "resumed": 0, "cached": False,
                "executed": True, "error": str(error),
            }
        materialized = result.get("materialized_children") or []
        if len(materialized) != len(segments):
            # Parallel path note (P3-3): the extractor normally materializes
            # exactly one child node per ASR segment, so a length mismatch
            # means the zip() below would silently drop rows — log it loudly
            # instead of losing data quietly.
            logger.warning(
                "voice_asr: materialized %d child nodes for %d ASR segments "
                "of clip %s — %d rows will be dropped",
                len(materialized), len(segments), clip_id,
                abs(len(segments) - len(materialized)),
            )
        rows = []
        for m, seg in zip(materialized, segments):
            rows.append(segment_row(
                video_id=video_id,
                clip_id=clip_id,
                audio_node_id=audio_node.node_id,
                segment_node_id=m["child_node_id"],
                asr_id=seg.get("asr_id", ""),
                language=language,
                start_sec=seg.get("start_sec", 0.0),
                end_sec=seg.get("end_sec", 0.0),
                text=seg.get("text", ""),
                word_count=len(seg.get("words") or []),
                no_speech_prob=seg.get("no_speech_prob", 0.0),
            ))
        return {
            "rows": rows, "resumed": 0, "cached": False,
            "executed": True, "error": None,
        }

    with output_path.open("w", encoding="utf-8") as f:
        for audio_node, result in parallel_clips(audio_nodes, _unit):
            if isinstance(result, UnitError):
                logger.error(
                    "voice_asr failed for %s: %s",
                    audio_node.node_id, result,
                    exc_info=result.exception,
                )
                failed += 1
                continue
            if result["error"]:
                logger.warning(
                    "voice_asr returned error for %s: %s",
                    audio_node.node_id, result["error"],
                )
                failed += 1
                continue
            if result["cached"]:
                cached_audio += 1
            if result["executed"]:
                executed_audio += 1
            for row in result["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_segments += len(result["rows"])
            resumed_segments += result["resumed"]
            f.flush()

    logger.info(
        "voice_asr wrote %d segments from %d AUDIO nodes "
        "(%d resumed from cache, %d failed) → %s",
        total_segments, len(audio_nodes), resumed_segments, failed, output_path,
    )
    cache.write_summary(output_path, video=args.video, attempted=len(audio_nodes),
                        cached=cached_audio, executed=executed_audio, failed=failed)
    # A few per-clip ASR failures (silent/corrupt segments tripping
    # faster-whisper numpy bugs) must not kill the whole video: tolerate
    # a bounded failure ratio instead of failing on the first error.
    ratio = float(os.environ.get("OGMEM_ASR_FAIL_RATIO", "0.15"))
    attempted = len(audio_nodes)
    if not attempted or ratio <= 0:
        # Zero-tolerance config must take effect, and no inputs means
        # nothing to tolerate.
        max_allowed = 0
    else:
        # Single-clip resilience floor: tolerate one failure only when one
        # clip is within the configured ratio (attempted >= 1/ratio) — for
        # tiny inputs the floor would tolerate a share far beyond the
        # ratio (1 of 2 = 50% at the default 15%), and all-failed must
        # never pass.
        floor = 1 if attempted * ratio >= 1 else 0
        max_allowed = min(max(int(attempted * ratio), floor), attempted - 1)
    if failed and failed > max_allowed:
        logger.error(
            "voice_asr: %d/%d AUDIO nodes failed (>%.0f%% tolerated) — failing run",
            failed, len(audio_nodes), ratio * 100,
        )
        return 2
    if failed:
        logger.warning(
            "voice_asr: %d/%d AUDIO nodes failed (within %.0f%% tolerance) — continuing",
            failed, len(audio_nodes), ratio * 100,
        )
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automatic speech recognition on every AUDIO RawNode.",
    )
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID,
                   help="Graph account scope (defaults to the legacy benchmark account).")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--output-jsonl", default="data/voice_asr.jsonl")
    p.add_argument("--video", default=None,
                   help="Comma-separated video_ids to limit which clips run.")
    p.add_argument("--force", action="store_true",
                   help="Re-transcribe even with a valid cache receipt. By default "
                        "reuse only complete results matching audio bytes and "
                        "inference settings; legacy/unknown caches are rebuilt.")
    # Operator config — forwarded to VoiceASRExtractor.
    p.add_argument(
        "--backend",
        choices=("faster_whisper", "openai_http", "mock"),
        default="faster_whisper",
        help="ASR backend implementation. Default: faster_whisper.",
    )
    p.add_argument("--model", default=None,
                   help="Model name (faster_whisper CTranslate2 / HF id, "
                        "or remote model id for openai_http).")
    p.add_argument("--language", default="en",
                   help="BCP-47 language code; empty string '' disables "
                        "and lets the backend auto-detect.")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--no-word-timestamps", action="store_true",
                   help="Skip per-word start/end/probability.")
    p.add_argument("--no-vad-filter", action="store_true",
                   help="Disable the backend's built-in VAD.")
    p.add_argument("--base-url", default=None,
                   help="openai_http only: base URL of the Whisper server.")
    p.add_argument("--api-key", default=None,
                   help="openai_http only: bearer token. Empty is fine for "
                        "self-hosted servers.")
    p.add_argument("--device", default="cpu",
                   help="faster_whisper only: cpu / cuda.")
    p.add_argument("--timeout-sec", type=float, default=300.0,
                   help="openai_http only: request timeout.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_voice_asr(args)


if __name__ == "__main__":
    sys.exit(main())
