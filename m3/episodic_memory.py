"""Phase 2 driver: generate per-clip episodic memory via multimodal LLM.

For each VIDEO clip (already in the graph with VIDEO RawNode + child
IMAGE RawNode frames from ``m3.extract_frames`` + optional AUDIO
RawNode child from ``m3.extract_audio``), this script:

1. Looks up the clip's frame children via the ``REFERENCES`` edges
   created by ``video_frame_extractor``'s materialization.
2. Looks up the clip's optional AUDIO child if present.
3. Calls :class:`EpisodicMemoryGenerator` (which calls the
   :class:`providers.llm.LLM`-conformant backend — default
   ``MiniMaxLLM`` pointing at ``api.minimaxi.com``) with a multimodal
   message list carrying the frames + audio.
4. Persists the LLM's structured output as the parent VIDEO's
   ``ExtractedNode.extracted_content``.

Backends (selected by ``--llm``):
  - ``minimax`` (default) — uses ``MiniMaxLLM`` with the
    ``OGMEM_MINIMAX_API_KEY`` env var. Routes to
    ``https://api.minimaxi.com/v1``.
  - ``mock`` — uses ``MockLLM`` for offline smoke runs and CI.

Output manifest ``episodic_memory.jsonl`` (one line per clip)::

    {
      "video_id": ...,
      "clip_id": ...,
      "video_raw_node_id": ...,
      "extracted_node_id": ...,
      "description": "...",
      "frame_count": 8,
      "has_audio": false,
      "resumed": false,
    }

Idempotency: deterministic parent ``ExtractedNode`` id derived from
``sha1("extracted/episodic_memory_generator/<video_raw_id>")``. Re-runs
short-circuit when the parent already exists.

Usage::

    # After extract_frames (and extract_audio if you have audio):
    export OGMEM_MINIMAX_API_KEY=...

    python -m m3.episodic_memory \\
        --video-records-manifest data/data.jsonl \\
        --frames-manifest data/frames.jsonl \\
        --output-jsonl data/episodic_memory.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import random
import re
import sys
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

from m3._parallel import UnitError, parallel_clips
from m3.config import (
    M3BENCH_ACCOUNT_ID,
    M3BENCH_AGENT_ID,
    M3BENCH_USER_ID,
    load_m3_runtime_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------

def extracted_node_id(extractor_type: str, source_raw_id: str) -> str:
    """Deterministic parent ``ExtractedNode`` id for the episodic memory."""
    seed = f"extracted/{extractor_type}/{source_raw_id}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate_episodic_for_clip(
    graph,
    op_service,
    ctx,
    *,
    video_raw_node,
    frame_nodes: list,
    audio_node,
    llm,
    max_frames: int,
    include_audio: bool,
    temperature: float,
    asr_segments: list[dict] | None = None,
    force: bool = False,
) -> dict:
    """Run :class:`EpisodicMemoryGenerator` for one VIDEO clip.

    Returns the manifest-shaped dict for this clip. The
    ``description`` is pulled from the LLM response so the manifest
    stays human-readable.

    ``asr_segments`` is the optional transcript produced by the
    ``voice_asr`` operator for this clip's audio. When non-empty
    it's threaded into the operator as the ``asr_segments`` kwarg
    so the LLM sees a timestamped transcript block in its prompt
    alongside the frames (works on any backend — text-only path).
    """
    ext_id = extracted_node_id(
        "episodic_memory_generator", video_raw_node.node_id,
    )

    from extraction.operators.builtin.episodic_memory_generator import normalise_speech_transcript

    def transcript_signature(segments):
        return sorted(json.dumps(seg, sort_keys=True) for seg in normalise_speech_transcript(segments))

    existing = graph.get_extracted_node(ext_id, ctx)
    if existing is not None:
        content = existing.extracted_content or {}
        # Compare evidence, not the legacy has_audio flag (which could be
        # true even when neither bytes nor transcripts reached the model).
        from core.graph_enums import ExtractionStatus
        is_stale = force or existing.extraction_status != ExtractionStatus.COMPLETED or (
            transcript_signature(content.get("speech_transcript"))
            != transcript_signature(asr_segments)
        ) or (include_audio and audio_node is not None and "audio_attached" not in content)
        if not is_stale:
            return {
                "video_id": (video_raw_node.metadata or {}).get("video_id"),
                "clip_id": (video_raw_node.metadata or {}).get("clip_id"),
                "video_raw_node_id": video_raw_node.node_id,
                "extracted_node_id": ext_id,
                "source_start_sec": content.get("source_start_sec"),
                "source_duration_sec": content.get("source_duration_sec"),
                "description": content.get("description", ""),
                **{key: content.get(key) or [] for key in
                   ("characters", "events", "entities", "speech_transcript")},
                "frame_count": content.get("frame_count", len(frame_nodes)),
                "has_audio": bool(content.get("has_audio")),
                "audio_attached": bool(content.get("audio_attached")),
                "has_transcript": bool(content.get("speech_transcript")),
                "resumed": True,
            }
        logger.info("Refreshing stale or forced episodic record %s", ext_id)

    from extraction.operators.builtin import EpisodicMemoryGenerator
    extractor = EpisodicMemoryGenerator(
        llm=llm,
        max_frames=max_frames,
        include_audio=include_audio,
        temperature=temperature,
    )
    extraction = extractor.extract(
        video_raw_node, ctx,
        frame_nodes=frame_nodes,
        audio_node=audio_node,
        asr_segments=list(asr_segments or []),
    )
    if existing is not None:
        # SQL create_node is INSERT, not upsert. Keep the old evidence intact
        # if generation fails, and update the same node on success.
        if extraction.error:
            raise RuntimeError(f"episodic_memory_generator failed: {extraction.error}")
        graph.mark_extraction_completed(
            ext_id, extraction.extracted_content, extraction.confidence, ctx,
        )
        result = {"extracted_node_id": ext_id,
                  "extracted_content": extraction.extracted_content}
    else:
        result = op_service.persist_extraction(
            raw_node=video_raw_node, operator=extractor,
            extraction=extraction, ctx=ctx, desired_node_id=ext_id,
        )
    if result.get("error"):
        raise RuntimeError(
            f"episodic_memory_generator failed for "
            f"{video_raw_node.node_id}: {result['error']}"
        )

    content = result.get("extracted_content") or {}
    return {
        "video_id": (video_raw_node.metadata or {}).get("video_id"),
        "clip_id": (video_raw_node.metadata or {}).get("clip_id"),
        "video_raw_node_id": video_raw_node.node_id,
        "extracted_node_id": result["extracted_node_id"],
        "source_start_sec": content.get("source_start_sec"),
        "source_duration_sec": content.get("source_duration_sec"),
        "description": content.get("description", ""),
        **{key: content.get(key) or [] for key in
           ("characters", "events", "entities", "speech_transcript")},
        "frame_count": content.get("frame_count", len(frame_nodes)),
        "has_audio": content.get("has_audio", False),
        "audio_attached": content.get("audio_attached", False),
        "has_transcript": bool(content.get("speech_transcript")),
        "resumed": False,
    }


# ---------------------------------------------------------------------------
# Driver entry point
# ---------------------------------------------------------------------------

def _build_llm(args) -> object:
    """Construct the LLM backend from CLI + ogmemory.yaml.

    Thin wrapper around :func:`m3.llm_factory.build_episodic_llm` —
    the canonical dispatch lives there so QA / episodic / judge roles
    share the same provider logic. Kept under the old name because the
    surrounding module passes an ``args`` namespace; the wrapper just
    unpacks the fields the factory expects.
    """
    from m3.llm_factory import build_episodic_llm
    return build_episodic_llm(
        args.llm,
        model=args.openai_model,
        base_url=None,
        api_key=None,
        temperature=args.temperature,
        config_path=args.config,
    )


def _index_frame_records(frames_manifest_path: Path) -> dict[str, list[str]]:
    """Build a ``video_id|clip_id -> [frame_node_id, ...]`` index.

    Reads ``frames.jsonl`` (output of ``m3.extract_frames``) and
    groups child_node_ids by ``video_id`` + ``clip_id``.
    """
    index: dict[str, list[str]] = {}
    if not frames_manifest_path.exists():
        return index
    for line in frames_manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = f"{rec.get('video_id','')}|{rec.get('clip_id','')}"
        index[key] = rec.get("child_node_ids", []) or []
    return index


def _index_audio_records(audio_manifest_path: Path | None) -> dict[str, str]:
    """Build ``video_id|clip_id -> audio_raw_node_id`` index.

    ``audio_manifest.jsonl`` is the (future) output of
    ``m3.extract_audio``. Lines look like::

        {"video_id": ..., "clip_id": ..., "audio_node_id": ...}

    Returns an empty index when the file doesn't exist yet (the
    episodic_memory driver still runs in vision-only mode).
    """
    index: dict[str, str] = {}
    if audio_manifest_path is None or not audio_manifest_path.exists():
        return index
    for line in audio_manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = f"{rec.get('video_id','')}|{rec.get('clip_id','')}"
        if rec.get("audio_node_id"):
            index[key] = rec["audio_node_id"]
    return index


def _index_asr_segments(
    graph, ctx, audio_index: dict[str, str],
) -> dict[str, list[dict]]:
    """Build ``video_id|clip_id -> [asr_segment_dict, ...]`` index.

    Walks the graph for ``voice_asr`` ExtractedNodes whose
    ``source_audio_node_id`` matches an entry in ``audio_index``.
    Returns an empty dict when no voice_asr operator has run yet.

    Each returned segment dict carries the full
    ``extracted_content`` of the child ASR ExtractedNode (start_sec,
    end_sec, text, words, speaker_id when diarization has stamped
    it in).

    Falls back to a graph-side lookup when ``audio_index`` is empty
    (e.g. the pipeline didn't pass ``--audio-manifest``): we walk the
    graph's AUDIO RawNodes, read their ``video_id``/``clip_id``
    metadata directly, and build the reverse map that way. Without
    this fallback the entire ASR path silently degrades to vision-only
    on a SQL-backed graph store.

    Scoping note (M3-Bench fix):
        The previous implementation did
        ``for n in graph.list_extracted_nodes("voice_asr", ctx):``
        which on a SQL store translated to ``SELECT ... FROM
        graph_nodes WHERE node_type='extracted' ORDER BY updated_at
        DESC LIMIT 4096``. Once the graph grew past ~10k extracted
        nodes (face + voice_asr + episodic), the LIMIT 4096 silently
        dropped the ASR nodes from every video but the most
        recently ingested — 20/23 videos received empty transcripts
        even though voice_asr had produced them.

        We now scope the ASR fetch to the ``parent_audio_node_ids``
        known from the manifest (or the manifest's reverse lookup)
        so the store can push the filter into the WHERE clause via
        ``parent_audio_node_ids``. Combined with ``limit=None``,
        this returns every ASR segment for the videos in scope
        regardless of how large the rest of the graph gets.
    """
    out: dict[str, list[dict]] = {}

    def _canon(s: str | None) -> str:
        # ``audio.jsonl`` writes the undashed 32-hex id from
        # ``_audio_child_id``; the SQL graph store persists the
        # dashed UUID form. Strip dashes + lowercase so both sides
        # compare equal.
        return s.replace("-", "").lower() if isinstance(s, str) and s else ""

    # Build a reverse map (canon audio_node_id -> "video_id|clip_id").
    # Try the manifest first; fall back to walking the graph.
    rev_audio: dict[str, str] = {_canon(v): k for k, v in audio_index.items() if v}
    # When the graph walk populates rev_audio, remember the original
    # (un-canonicalised) audio_node_ids we discovered so we can thread
    # them down to the ASR store as ``parent_audio_node_ids`` and
    # avoid the unscoped fallback (see scoped_ids below).
    walked_audio_ids: list[str] = []
    if not rev_audio:
        from core.graph_enums import RawDataType
        for audio_node in (
            graph.list_raw_nodes(RawDataType.AUDIO, ctx)
            if hasattr(graph, "list_raw_nodes") else []
        ):
            md = audio_node.metadata or {}
            vid = md.get("video_id")
            cid = md.get("clip_id")
            if vid is None or cid is None:
                continue
            rev_audio[_canon(audio_node.node_id)] = f"{vid}|{cid}"
            walked_audio_ids.append(audio_node.node_id)

    if not rev_audio:
        return out

    # JSON source IDs are compared as exact strings by SQL. Expand UUID
    # forms BEFORE filtering; normalising returned rows is too late.
    scoped_ids = []
    for aid in list(audio_index.values()) or walked_audio_ids:
        if not aid:
            continue
        variants = [aid]
        try:
            parsed = uuid.UUID(str(aid))
            variants.extend([str(parsed), parsed.hex])
        except ValueError:
            pass  # Preserve non-UUID IDs used by memory stores.
        for variant in variants:
            if variant not in scoped_ids:
                scoped_ids.append(variant)

    # Note: passing ``limit=None`` to ``list_extracted_nodes`` asks
    # the store to drop the LIMIT clause. We only do this when
    # ``scoped_ids`` is non-empty (a meaningful WHERE filter is in
    # place); with the unscoped fallback we keep the default cap.
    scoped_kwargs = {}
    if scoped_ids:
        scoped_kwargs["limit"] = None
        scoped_kwargs["parent_audio_node_ids"] = scoped_ids

    for n in graph.list_extracted_nodes("voice_asr", ctx, **scoped_kwargs):
        ec = n.extracted_content or {}
        audio_id = _canon(ec.get("source_audio_node_id"))
        if not audio_id or audio_id not in rev_audio:
            continue
        key = rev_audio[audio_id]
        out.setdefault(key, []).append({
            "start_sec": ec.get("start_sec", 0.0),
            "end_sec": ec.get("end_sec", 0.0),
            "text": ec.get("text", ""),
            "speaker_id": ec.get("speaker_id"),
            "words": ec.get("words") or [],
        })
    # Stable order per clip
    for key in out:
        out[key].sort(key=lambda s: float(s.get("start_sec", 0.0)))
    return out


def _validate_asr_manifest(path: Path, asr_index: dict, audio_index: dict) -> None:
    """Fail before LLM calls if expected speech is missing from the graph."""
    from collections import Counter

    def signature(row):
        return (str(row.get("text", "")).strip(),
                float(row.get("start_sec", 0) or 0),
                float(row.get("end_sec", 0) or 0))

    expected = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = f"{row.get('video_id')}|{row.get('clip_id')}"
        if audio_index and key not in audio_index:
            continue
        if signature(row)[0]:
            expected.setdefault(key, Counter()).update([signature(row)])
    for key, segments in expected.items():
        actual = Counter(signature(row) for row in asr_index.get(key, []))
        if segments - actual:
            raise RuntimeError(f"ASR evidence missing from graph for {key}; refusing vision-only fallback")


def _retryable_generation_error(exc: Exception) -> bool:
    """Operator errors are serialized; match explicit HTTP codes, not arbitrary numbers."""
    status = getattr(exc, "status_code", None)
    if status is None:
        match = re.search(
            r"(?:HTTP(?:/\d(?:\.\d)?)?\s+|Error code:\s*|http_code['\"]?\s*:\s*)(\d{3})",
            str(exc), re.IGNORECASE,
        )
        status = int(match.group(1)) if match else None
    return status in {408, 429, 500, 502, 503, 504, 529}


def _generate_with_retry(*, max_retries: int, retry_base_seconds: float,
                         retry_max_seconds: float, **kwargs) -> tuple[dict, int]:
    for attempt in range(max_retries + 1):
        try:
            return generate_episodic_for_clip(**kwargs), attempt
        except Exception as exc:
            if attempt == max_retries or not _retryable_generation_error(exc):
                raise
            delay = min(retry_max_seconds, retry_base_seconds * (2 ** attempt))
            delay = min(retry_max_seconds, delay * random.uniform(1.0, 1.2))
            logger.warning(
                "Transient episodic failure; retry %d/%d in %.1fs for %s: %s",
                attempt + 1, max_retries, delay, kwargs["video_raw_node"].node_id, exc,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def run(args: argparse.Namespace, graph=None, op_service=None) -> int:
    """CLI entry point. Tests pass ``graph``/``op_service`` to share state
    with a fixture; production callers leave them as ``None``."""
    from core.models import RequestContext

    output_jsonl = Path(args.output_jsonl).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_jsonl.with_suffix(".summary.json")
    failed_path = output_jsonl.parent / "episodic_memory_failures.json"
    retry_targets = None
    previous_failures = []
    if getattr(args, "retry_failed", False):
        # Read before resetting diagnostics. Failed forced refreshes may still
        # have old COMPLETED graph nodes: explicitly force these IDs again.
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous_summary.get("account_id") != args.account_id:
            raise ValueError("retry-failed requires a previous summary with the same account_id")
        previous_failures = json.loads(failed_path.read_text(encoding="utf-8"))
        if not previous_failures or any(
            "video_id" not in row or "clip_id" not in row for row in previous_failures
        ):
            raise ValueError("retry-failed requires a non-empty per-clip failure list")
        retry_targets = {(r["video_id"], str(r["clip_id"])) for r in previous_failures}
    # Replace diagnostics from previous attempts before doing any work.
    # An interrupted/exceptional attempt must not retain a success receipt.
    summary_path.write_text(
        json.dumps({"status": "running", "video_filter": args.video,
                    "account_id": args.account_id}),
        encoding="utf-8",
    )
    failed_path.write_text(json.dumps(previous_failures), encoding="utf-8")

    if graph is None:
        from m3.load_clips import build_graph_service
        graph, store_kind = build_graph_service(
            args.graph_store, args.connection_string,
        )
    else:
        store_kind = (
            type(graph._store).__name__
            if hasattr(graph, "_store") else "injected"
        )

    if op_service is None:
        from extraction.operators import default_registry
        from extraction.operators.builtin import register_all_defaults
        from service.operator_service import OperatorService
        registry = default_registry()
        register_all_defaults(registry)
        op_service = OperatorService(graph, registry=registry)

    llm = _build_llm(args)
    frames_index = _index_frame_records(Path(args.frames_manifest).resolve())
    audio_index = _index_audio_records(
        Path(args.audio_manifest).resolve() if args.audio_manifest else None,
    )
    # Build the RequestContext BEFORE the ASR index lookup so the
    # ``_index_asr_segments(graph, ctx, ...)`` call doesn't trip an
    # UnboundLocalError that gets silently swallowed (and produces
    # ``audio_attached: 0`` even when voice_asr produced segments).
    ctx = RequestContext(
        account_id=args.account_id,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-episodic",
        trace_id=f"m3-ep-{uuid.uuid4().hex}",
    )

    # When voice_asr has run for this graph, attach the per-clip
    # transcripts to the LLM call so the operator can ground its
    # description in actual speech. Builds an in-process index keyed
    # by ``video_id|clip_id``; empty when no voice_asr ExtractedNodes
    # exist (vision-only run, no ASR done yet).
    asr_index = _index_asr_segments(graph, ctx, audio_index)
    if getattr(args, "asr_manifest", None):
        _validate_asr_manifest(Path(args.asr_manifest), asr_index, audio_index)
    logger.info(
        "Using graph store: %s, LLM: %s, account_id=%s "
        "(frames_index=%d clips, audio_index=%d clips, asr_index=%d clips)",
        store_kind, args.llm or load_m3_runtime_config(args.config).episodic_llm.provider, ctx.account_id,
        len(frames_index), len(audio_index), len(asr_index),
    )

    manifest_path = Path(args.video_records_manifest).resolve()
    if not manifest_path.exists():
        logger.error(
            "Video records manifest not found: %s (run m3.load_clips first)",
            manifest_path,
        )
        failure = {"error": "missing_video_manifest", "path": str(manifest_path)}
        failed_path.write_text(json.dumps([failure]), encoding="utf-8")
        summary_path.write_text(
            json.dumps({
                "status": "failed", "errors": [failure],
                "failures_path": str(failed_path),
            }),
            encoding="utf-8",
        )
        return 1
    manifest_lines = manifest_path.read_text(encoding="utf-8").strip().split("\n")

    output_lines: list[str] = []
    total_chars = 0
    total_extractions = 0
    expected_clips = 0
    resumed_clips = 0
    retries = 0
    seen_retry_targets = set()
    failed: list[dict] = []

    only_videos = set(args.video.split(",")) if args.video else None

    # Flatten into per-clip descriptors. Child resolution (graph reads)
    # and pre-submit failure bookkeeping stay in the main thread so the
    # failure-record shapes are unchanged; the multimodal LLM call
    # dominates the worker unit. OpenAI client and SQL graph store are
    # safe to share across workers; the read-only indexes are shared.
    clips: list[dict] = []
    for line in manifest_lines:
        if not line.strip():
            continue
        record = json.loads(line)
        video_id = record["id"]
        if only_videos and video_id not in only_videos:
            continue

        for raw_record in record.get("raw_nodes", []):
            expected_clips += 1
            clip_id = raw_record.get("clip_id")
            retry_key = (video_id, str(clip_id))
            if retry_targets is not None and retry_key in retry_targets:
                seen_retry_targets.add(retry_key)
            video_node = graph.get_raw_node(raw_record["raw_node_id"], ctx)
            if video_node is None:
                logger.warning(
                    "VIDEO RawNode %s missing for %s clip %s",
                    raw_record["raw_node_id"], video_id, clip_id,
                )
                failed.append({
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "error": "missing_video_node",
                    "raw_node_id": raw_record["raw_node_id"],
                })
                continue

            # Resolve frame + audio children for this clip.
            key = f"{video_id}|{clip_id}"
            frame_ids = frames_index.get(key) or []
            frame_nodes = [
                n for n in (
                    graph.get_raw_node(fid, ctx) for fid in frame_ids
                )
                if n is not None
            ]
            audio_node = None
            audio_id = audio_index.get(key)
            if audio_id:
                audio_node = graph.get_raw_node(audio_id, ctx)

            if not frame_nodes:
                logger.warning(
                    "No frames for %s clip %s (frames manifest missing?); "
                    "run m3.extract_frames first",
                    video_id, clip_id,
                )
                failed.append({
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "error": "no_frames",
                })
                continue

            clips.append({
                "video_id": video_id,
                "clip_id": clip_id,
                "retry_key": retry_key,
                "key": key,
                "video_node": video_node,
                "frame_nodes": frame_nodes,
                "audio_node": audio_node,
            })

    def _unit(clip: dict) -> tuple[dict, int]:
        clip_ctx = dataclasses.replace(
            ctx, trace_id=f"{ctx.trace_id}-{clip['video_id']}-{clip['clip_id']}"
        )
        return _generate_with_retry(
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            retry_max_seconds=args.retry_max_seconds,
            graph=graph, op_service=op_service, ctx=clip_ctx,
            video_raw_node=clip["video_node"],
            frame_nodes=clip["frame_nodes"],
            audio_node=clip["audio_node"],
            llm=llm,
            max_frames=args.max_frames,
            include_audio=not args.no_audio,
            temperature=args.temperature,
            asr_segments=asr_index.get(clip["key"]) or [],
            force=(clip["retry_key"] in retry_targets if retry_targets is not None
                   else getattr(args, "force", False)),
        )

    for clip, result in parallel_clips(clips, _unit):
        if isinstance(result, UnitError):
            logger.error(
                "Episodic memory failed for %s clip %s: %s",
                clip["video_id"], clip["clip_id"], result,
                exc_info=result.exception,
            )
            failed.append({
                "video_id": clip["video_id"],
                "clip_id": clip["clip_id"],
                "error": str(result),
            })
            continue

        record, retry_count = result
        retries += retry_count
        output_lines.append(json.dumps(record, ensure_ascii=False))
        total_chars += len(record.get("description") or "")
        if not record["resumed"]:
            total_extractions += 1
        else:
            resumed_clips += 1

    with output_jsonl.open("w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(line + "\n")

    logger.info(
        "Wrote %d extraction records (%d new, %d description chars, "
        "%d failures) -> %s",
        len(output_lines), total_extractions, total_chars,
        len(failed), output_jsonl,
    )
    errors = list(failed)
    if retry_targets is not None and retry_targets - seen_retry_targets:
        errors.extend({
            "video_id": video, "clip_id": clip,
            "error": "retry_target_missing_from_manifest",
        } for video, clip in sorted(retry_targets - seen_retry_targets))
    if not expected_clips:
        errors.append({"error": "no_matching_clips", "video_filter": args.video})
    failed_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps({
        "status": "failed" if errors else "completed",
        "video_filter": args.video,
        "account_id": args.account_id,
        "expected_clips": expected_clips,
        "completed_clips": len(output_lines),
        "generated_clips": total_extractions,
        "resumed_clips": resumed_clips,
        "failed_clips": len(failed) + len((retry_targets or set()) - seen_retry_targets),
        "recovered_retries": retries,
        "errors": errors,
        "failures_path": str(failed_path),
    }, indent=2), encoding="utf-8")
    return 2 if errors else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run episodic_memory_generator on every VIDEO RawNode with "
            "its frame (and optional audio) children."
        )
    )
    p.add_argument(
        "--video-records-manifest", required=True,
        help="Path to data.jsonl from m3.load_clips (lists VIDEO RawNodes per clip)",
    )
    p.add_argument(
        "--frames-manifest", required=True,
        help="Path to frames.jsonl from m3.extract_frames",
    )
    p.add_argument(
        "--audio-manifest", default=None,
        help="Path to audio.jsonl from m3.extract_audio (optional; "
             "audio IDs fall back to graph metadata if absent)",
    )
    p.add_argument(
        "--output-jsonl", default="data/episodic_memory.jsonl",
    )
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None)
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID)
    p.add_argument("--video", default=None,
                   help="Comma-separated video_ids to process (default: all)")
    # LLM backend
    p.add_argument("--llm", choices=("minimax", "openai", "mock"), default=None,
                   help="LLM backend; defaults to m3.episodic_llm.provider from ogmemory.yaml")
    p.add_argument("--config", default=None,
                   help="Path to ogmemory.yaml (default: deploy/ogmemory.yaml or OGMEM_CONFIG)")
    p.add_argument("--minimax-model", default=None,
                   help="Override m3.episodic_llm.model for MiniMax")
    p.add_argument("--openai-model", default=None,
                   help="Override m3.episodic_llm.model for OpenAI")
    p.add_argument("--temperature", type=float, default=0.2)
    # Operator config
    p.add_argument("--asr-manifest", default=None, help="Validate graph transcripts against voice_asr JSONL")
    p.add_argument("--force", action="store_true", help="Regenerate cached episodic records")
    p.add_argument("--retry-failed", action="store_true",
                   help="Force prior failed clips; validate/reuse successful clips even with --force")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--retry-base-seconds", type=float, default=10.0)
    p.add_argument("--retry-max-seconds", type=float, default=60.0)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--no-audio", action="store_true",
                   help="Skip raw audio payloads; ASR transcripts are still included")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    if not 0 <= args.max_retries <= 10:
        p.error("--max-retries must be between 0 and 10")
    if not (0 <= args.retry_base_seconds <= args.retry_max_seconds <= 300):
        p.error("retry delays must satisfy 0 <= base <= max <= 300 seconds")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
