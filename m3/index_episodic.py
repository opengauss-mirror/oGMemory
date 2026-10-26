"""Build a searchable text index from M3 episodic-memory JSONL.

The episodic pipeline writes one JSON object per clip, with a natural-language
``description`` plus structured ``characters`` / ``events`` / ``entities``.
This module flattens those fields into retrieval text, embeds the text, and
writes a lightweight JSONL index consumed by :mod:`m3.search` and the QA MVP.

Usage::

    python -m m3.index_episodic \
        --episodic-jsonl data/episodic_memory.jsonl \
        --output-index data/episodic_index.jsonl \
        --embedder openai

For tests / offline smoke runs::

    python -m m3.index_episodic \
        --episodic-jsonl data/episodic_memory.jsonl \
        --output-index data/episodic_index.jsonl \
        --embedder mock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.temporal import temporal_metadata
from m3.config import load_m3_runtime_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record shaping
# ---------------------------------------------------------------------------

def stable_record_id(video_id: str, clip_id: int | str, extracted_node_id: str = "") -> str:
    """Return a deterministic index id for one episodic clip record."""
    seed = f"episodic/{video_id}/{clip_id}/{extracted_node_id}".encode()
    return hashlib.sha1(seed).hexdigest()[:32]


def _stringify_named_items(items: Any, *, name_key: str = "name") -> list[str]:
    """Convert schema-ish list items into short searchable strings."""
    out: list[str] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            parts: list[str] = []
            name = item.get(name_key)
            if isinstance(name, str) and name.strip():
                parts.append(name.strip())
            for key in ("type", "role", "summary", "description"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
            text = " - ".join(parts)
        else:
            text = ""
        if text:
            out.append(text)
    return out


def _format_transcript_block(
    asr_segments: list[dict[str, Any]],
    *,
    header: str = "Speech (verbatim):",
) -> str | None:
    """Format the per-clip ASR transcript block.

    Each line carries a ``[start-end]`` prefix so the answerer can still
    reason about timing when the transcript shows up in the retrieved
    evidence block. Returns ``None`` when no segment has non-empty
    text so callers can drop the header cleanly.

    ``header`` is configurable so the legacy "Speech transcript:" label
    is still reachable from tests that assert on it.
    """
    if not asr_segments:
        return None
    transcript_lines: list[str] = []
    for seg in asr_segments:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start_sec", 0.0) or 0.0)
        end = float(seg.get("end_sec", 0.0) or 0.0)
        speaker_id = seg.get("speaker_id")
        if speaker_id:
            transcript_lines.append(
                f"[{start:6.2f}-{end:6.2f}s] {speaker_id}: {text}",
            )
        else:
            transcript_lines.append(
                f"[{start:6.2f}-{end:6.2f}s] {text}",
            )
    if not transcript_lines:
        return None
    return f"{header}\n" + "\n".join(transcript_lines)


def build_retrieval_text(
    record: dict[str, Any],
    *,
    asr_segments: list[dict[str, Any]] | None = None,
    transcript_first: bool = True,
) -> str:
    """Flatten one episodic-memory record into retrieval text.

    The text deliberately repeats high-value labels (description, events,
    entities, characters) so keyword + vector hybrid retrieval has useful
    terms even when a model produces terse descriptions.

    When ``asr_segments`` is supplied (one dict per Whisper segment with
    ``start_sec``/``end_sec``/``text`` keys), a transcript block is
    emitted so the QA searcher can find answers that live in the speech
    track rather than the visual description. Speech is the strongest
    signal for many M3-Bench questions ("happy but exhausted", "put
    them in an album", "fetch the pictures in my bag") and dropping it
    from the index makes those questions unanswerable.

    ``transcript_first`` (default ``True``) puts the transcript block
    ABOVE the visual description so the embedder weights the verbatim
    speech more heavily than the LLM's paraphrase. Set to ``False`` to
    restore the pre-quickwin ordering (description first, transcript
    appended last) — useful for A/B tests and for the vision-only smoke
    path that has never produced transcripts.
    """
    sections: list[str] = []
    transcript_block = _format_transcript_block(asr_segments or [])
    if transcript_first and transcript_block:
        sections.append(transcript_block)

    description = record.get("description")
    if isinstance(description, str) and description.strip():
        sections.append(f"Description: {description.strip()}")

    events = _stringify_named_items(record.get("events"), name_key="type")
    if events:
        sections.append("Events: " + "; ".join(events))

    entities = _stringify_named_items(record.get("entities"))
    if entities:
        sections.append("Entities: " + "; ".join(entities))

    characters = _stringify_named_items(record.get("characters"))
    if characters:
        sections.append("Characters: " + "; ".join(characters))

    # Append the transcript at the end when transcript_first=False so
    # legacy call sites and A/B tests can compare orderings head-to-head.
    if not transcript_first and transcript_block:
        sections.append(transcript_block)

    # Keep provenance words searchable too; this helps debugging and does not
    # dominate embedding content.
    video_id = record.get("video_id") or record.get("source_video_id")
    clip_id = record.get("clip_id") if record.get("clip_id") is not None else record.get("source_clip_id")
    if video_id is not None or clip_id is not None:
        sections.append(f"Source: video={video_id} clip={clip_id}")

    return "\n".join(sections).strip()


def index_record_from_episodic(
    record: dict[str, Any],
    embedding: list[float] | None = None,
    *,
    text: str | None = None,
    asr_segments: list[dict[str, Any]] | None = None,
    transcript_first: bool = True,
    include_embedding: bool = False,
) -> dict[str, Any]:
    """Build the JSONL index record for one episodic-memory output.

    ``text`` is the pre-built retrieval string. When the caller
    (typically :func:`build_index`) has already augmented it with the
    ASR transcript, pass it through; otherwise we re-build from the
    record with the optional ``asr_segments`` so the index doesn't
    silently drop the transcript.

    ``transcript_first`` is forwarded to :func:`build_retrieval_text`
    when we have to rebuild the text here (caller didn't pre-build).
    Defaults to ``True`` so the embedder weights verbatim speech over
    the LLM's paraphrase.

    ``embedding`` is optional: when present AND ``include_embedding``
    is True, the vector is written into the JSONL record under the
    ``embedding`` key. The current default for ``include_embedding``
    is False because the canonical retrieval source is now the
    ``OpenGaussVectorIndex`` (see ``_upsert_to_vector_index``); the
    JSONL manifest is a debug / audit artifact and we don't want to
    pay the disk cost of duplicating the 1024-dim vector per clip.
    Pass ``include_embedding=True`` from the legacy ``EpisodicSearcher``
    fallback path if you still need a self-contained JSONL index.

    Without either transcript or pre-built text, the index degrades to
    vision-only (legacy behaviour).
    """
    video_id = record.get("video_id") or record.get("source_video_id") or ""
    clip_id = record.get("clip_id") if record.get("clip_id") is not None else record.get("source_clip_id")
    extracted_node_id = record.get("extracted_node_id") or ""
    if text is None:
        text = build_retrieval_text(
            record,
            asr_segments=asr_segments,
            transcript_first=transcript_first,
        )
    rec_id = stable_record_id(str(video_id), clip_id, str(extracted_node_id))
    out: dict[str, Any] = {
        "id": rec_id,
        "uri": f"m3://{video_id}/clips/{clip_id}/episodic",
        "video_id": video_id,
        "clip_id": clip_id,
        "text": text,
        "source": "episodic_memory_generator",
        "source_extracted_node_id": extracted_node_id,
        "video_raw_node_id": record.get("video_raw_node_id"),
        "frame_count": record.get("frame_count"),
        "has_audio": record.get("has_audio"),
        "metadata": {
            "description": record.get("description", ""),
            "characters": record.get("characters", []),
            "events": record.get("events", []),
            "entities": record.get("entities", []),
            **temporal_metadata(record),
            "speech_transcript": list(asr_segments or record.get("speech_transcript") or []),
            "start_sec": record.get("source_start_sec"),
            "duration_sec": record.get("source_duration_sec"),
        },
    }
    if include_embedding and embedding is not None:
        out["embedding"] = list(embedding)
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL rows from ``path``."""
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on {path}:{lineno}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"expected object on {path}:{lineno}, got {type(obj).__name__}")
        records.append(obj)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records to JSONL and return count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

def build_embedder(
    provider: str | None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    dimension: int | None = None,
    config_path: str | None = None,
):
    """Construct an Embedder for M3 index/search CLI commands.

    Defaults come from ``m3.embedding`` in ogmemory.yaml, falling back to the
    top-level ``embedding`` block. M3 does not load .env; YAML placeholders are
    resolved only if the surrounding shell already exported them.
    """
    runtime = load_m3_runtime_config(config_path)
    provider = provider or runtime.embedding_provider
    model = model or runtime.embedding_model
    base_url = base_url or runtime.embedding_base_url
    api_key = api_key or runtime.embedding_api_key
    dimension = dimension if dimension is not None else runtime.embedding_dimension

    if provider == "mock":
        from providers.embedder import MockEmbedder
        return MockEmbedder(dimension=dimension or 384)

    if provider == "openai":
        from providers.embedder import get_openai_embedder
        OpenAIEmbedder, _ = get_openai_embedder()
        # Local OpenAI-compatible embedding servers (e.g. vllm / ollama /
        # text-embeddings-inference) commonly ignore the auth header. Allow
        # an empty config key to flow through as a placeholder when a
        # custom base_url is configured, instead of failing the pipeline.
        if not api_key and base_url:
            api_key = "no-key-required"
        return OpenAIEmbedder(
            api_key=api_key,
            base_url=base_url,
            model=model or "text-embedding-ada-002",
            dimension=dimension,
            batch_size=runtime.embedding_batch_size,
        )

    if provider == "st":
        from providers.embedder.st_embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model=model or "BAAI/bge-m3")

    if provider == "volcengine":
        from providers.embedder.volcengine_embedder import VolcengineEmbedder
        return VolcengineEmbedder(
            api_key=api_key or "",
            base_url=base_url or "https://ark.cn-beijing.volces.com/api/coding/v3",
            model=model or "doubao-embedding-vision-250615",
            dimension=dimension or 1024,
        )

    raise ValueError(f"unknown embedder provider: {provider}")


def _load_asr_segments_by_clip(
    path: Path | str | None,
) -> dict[tuple[str, int | str], list[dict[str, Any]]]:
    """Load ``voice_asr.jsonl`` (if present) and group segments by (video_id, clip_id).

    The voice_asr driver writes one row per Whisper segment with the
    same shape the LLM prompt builder wants. We don't try to merge
    speaker turns here — that's a separate voice_clustering concern.
    Returns an empty dict when the manifest is missing or unreadable.
    """
    out: dict[tuple[str, int | str], list[dict[str, Any]]] = {}
    if path is None:
        return out
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        video_id = rec.get("video_id")
        clip_id = rec.get("clip_id")
        if video_id is None or clip_id is None:
            continue
        out.setdefault((str(video_id), clip_id), []).append({
            "start_sec": rec.get("start_sec", 0.0),
            "end_sec": rec.get("end_sec", 0.0),
            "text": rec.get("text", ""),
            "speaker_id": rec.get("speaker_id"),
            "words": rec.get("words") or [],
        })
    for key in out:
        out[key].sort(key=lambda s: float(s.get("start_sec", 0.0)))
    return out


def build_index(
    episodic_records: list[dict[str, Any]],
    *,
    embedder,
    batch_size: int = 64,
    skip_empty: bool = True,
    asr_manifest: Path | str | None = None,
    transcript_first: bool = True,
) -> list[dict[str, Any]]:
    """Embed episodic records and return JSON-serialisable index records.

    When ``asr_manifest`` points at a ``voice_asr.jsonl`` file, each
    clip's indexed retrieval text is augmented with its transcript so
    spoken answers are searchable. The manifest is read once up front;
    missing files silently fall back to vision-only.

    ``transcript_first`` (default ``True``) is forwarded to
    :func:`build_retrieval_text` so the embedder weights the verbatim
    speech over the LLM's paraphrase. Pass ``False`` to restore the
    pre-quickwin ordering (description first, transcript appended last).
    """
    asr_by_clip = _load_asr_segments_by_clip(asr_manifest)
    prepared: list[tuple[dict[str, Any], str]] = []
    for rec in episodic_records:
        key = (str(rec.get("video_id", "")), rec.get("clip_id"))
        text = build_retrieval_text(
            rec,
            # Graph-backed episodic transcripts may already carry diarization
            # labels; the earlier ASR export must not overwrite those labels.
            asr_segments=rec.get("speech_transcript") or asr_by_clip.get(key),
            transcript_first=transcript_first,
        )
        if skip_empty and not text:
            logger.warning(
                "Skipping empty episodic record video=%s clip=%s",
                rec.get("video_id"), rec.get("clip_id"),
            )
            continue
        prepared.append((rec, text))

    out: list[dict[str, Any]] = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        texts = [text for _rec, text in batch]
        embeddings = embedder.embed_texts(texts)
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"embedder returned {len(embeddings)} vectors for {len(batch)} texts"
            )
        for (rec, text), emb in zip(batch, embeddings):
            # Pass the pre-built text (which may include the ASR
            # transcript block) so the index record keeps it. Without
            # this, ``index_record_from_episodic`` would silently
            # re-build the text vision-only and drop the transcript.
            # Keep embeddings until upsert; run() strips them only
            # when writing the audit JSONL.
            out.append(index_record_from_episodic(
                rec, list(emb), text=text, include_embedding=True,
                asr_segments=rec.get("speech_transcript") or asr_by_clip.get(
                    (str(rec.get("video_id", "")), rec.get("clip_id"))
                ),
            ))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an M3 episodic-memory retrieval index.")
    p.add_argument("--episodic-jsonl", required=True,
                   help="Path to episodic_memory.jsonl from m3.episodic_memory")
    p.add_argument("--output-index", default="data/episodic_index.jsonl",
                   help="Output JSONL index path (default: %(default)s)")
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None,
                   help="Embedder provider; defaults to m3.embedding.provider from ogmemory.yaml")
    p.add_argument("--config", default=None,
                   help="Path to ogmemory.yaml (default: deploy/ogmemory.yaml or OGMEM_CONFIG)")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--embedding-dimension", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--asr-manifest", default=None,
                   help="Optional voice_asr.jsonl path. When present, per-clip "
                        "transcripts are appended to the retrieval text so the "
                        "QA searcher can ground answers in speech content.")
    p.add_argument("--transcript-first", dest="transcript_first", action="store_true", default=True,
                   help="Place the speech transcript block BEFORE the visual "
                        "description in the retrieval text (default: enabled). "
                        "Speeds up retrieval of speech-anchored answers; pass "
                        "--no-transcript-first to restore the legacy ordering.")
    p.add_argument("--no-transcript-first", dest="transcript_first",
                   action="store_false",
                   help="Restore the legacy description-first ordering.")
    p.add_argument("--keep-empty", action="store_true",
                   help="Index records even when the flattened retrieval text is empty")
    # --- Vector-index connection (M3 now writes embeddings to a dedicated
    # OpenGaussVectorIndex instead of mirroring them back into
    # ExtractedNode.extracted_content["embedding"]). Defaults come from
    # ``m3.vector_index.*`` (backend / table / pool) plus the shared
    # ``vector_db.connection_string`` in ogmemory.yaml; the explicit CLI
    # flags are the escape hatch for one-off experiments / dry-runs
    # against a different DB.
    p.add_argument("--vector-index-connection-string", default=None,
                   help="PostgreSQL DSN for the OpenGaussVectorIndex. "
                        "Falls back to vector_db.connection_string "
                        "so M3 reuses the same Postgres / openGauss "
                        "instance as the graph store by default.")
    p.add_argument("--vector-index-table-name", default=None,
                   help="Table name for the M3 vector index (default: "
                        "m3.vector_index.table_name or 'm3_episodic_vectors').")
    p.add_argument("--vector-index-pool-size", type=int, default=None,
                   help="Connection pool size for the OpenGaussVectorIndex "
                        "(default: m3.vector_index.pool_size or 5).")
    p.add_argument("--vector-index-backend", default=None,
                   choices=("opengauss", "pgvector", "chroma", "memory"),
                   help="VectorIndex backend selector (default: "
                        "m3.vector_index.backend or 'opengauss'). "
                        "`opengauss` is the recommended production target; "
                        "`pgvector` is the single-DB convenience alias for "
                        "PostgreSQL+pgvector. Both route to the same "
                        "OpenGaussVectorIndex implementation.")
    p.add_argument("--account-id", default="m3bench",
                   help="Account id for the per-record RequestContext "
                        "(default: %(default)s)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def run(args: argparse.Namespace, graph=None, op_service=None, vector_index=None) -> int:  # noqa: ARG001 — kept for API parity with other m3.* drivers
    """Build the M3 episodic retrieval index and write embeddings to a VectorIndex.

    Embeddings are written to a :class:`VectorIndex` (default backend
    OpenGaussVectorIndex / pgvector); the GraphService that the QA
    stage constructs later picks up the same instance via the pipeline
    runner's ``PipelineContext.vector_index`` so
    :meth:`GraphService.search_by_embedding` routes through the index
    rather than scanning the graph store's JSONB column.

    The JSONL output (``--output-index``) is an audit artifact — it
    intentionally omits the per-record embedding vector since the
    vector now lives in the dedicated index. The text + metadata
    payload is enough to debug "why did this clip surface in the
    answer" without paying the disk cost of duplicating 1024-dim
    vectors per clip (~3K clips × 4KB = ~12MB saved per run).
    """
    episodic_path = Path(args.episodic_jsonl).resolve()
    if not episodic_path.exists():
        logger.error("episodic JSONL not found: %s", episodic_path)
        return 1

    # Resolve VectorIndex. The pipeline runner injects one via the
    # ``vector_index`` kwarg so every stage shares the same instance;
    # the CLI fallback builds from the YAML config (or the explicit
    # ``--vector-index-*`` overrides).
    if vector_index is None:
        vector_index = _build_vector_index_from_args(args)
        logger.info(
            "Using vector index: %s, account_id=%s",
            type(vector_index).__name__, args.account_id,
        )

    embedder = build_embedder(
        args.embedder,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        dimension=args.embedding_dimension,
        config_path=args.config,
    )
    records = read_jsonl(episodic_path)
    asr_manifest_path = (
        Path(args.asr_manifest).resolve() if args.asr_manifest else None
    )
    if asr_manifest_path and not asr_manifest_path.exists():
        logger.warning(
            "ASR manifest %s not found; building vision-only index",
            asr_manifest_path,
        )
        asr_manifest_path = None
    index_records = build_index(
        records,
        embedder=embedder,
        batch_size=args.batch_size,
        skip_empty=not args.keep_empty,
        asr_manifest=asr_manifest_path,
        transcript_first=getattr(args, "transcript_first", True),
    )

    # Upsert each record's embedding into the VectorIndex so
    # GraphService.search_by_embedding can find it. The function is
    # best-effort — a record missing its extracted_node_id is dropped
    # silently because we can't reverse-resolve to a graph node.
    written = _upsert_to_vector_index(index_records, vector_index, args)
    if written:
        logger.info("Upserted %d embeddings to %s", written, type(vector_index).__name__)

    # Write the audit JSONL without the per-record embedding (the
    # vector lives in the VectorIndex now). Strip the field on the
    # way out so downstream consumers reading the JSONL never see a
    # stale vector that's drifted from the canonical store.
    audit_records = [
        {k: v for k, v in rec.items() if k != "embedding"}
        for rec in index_records
    ]
    out = Path(args.output_index).resolve()
    count = write_jsonl(out, audit_records)
    logger.info("Wrote %d audit records (text+metadata) -> %s", count, out)
    return 0


def _build_vector_index_from_args(args: argparse.Namespace):
    """Resolve a VectorIndex from CLI args + YAML config.

    The explicit ``--vector-index-*`` flags win so the CLI is a true
    escape hatch; otherwise we delegate to the YAML-driven helper
    (:func:`m3.vector_index.build_m3_vector_index`), which reads the
    M3-specific knobs from ``m3.vector_index.*`` and the connection
    string from the shared top-level ``vector_db.connection_string``.
    """
    from m3.config import load_m3_runtime_config
    from m3.vector_index import build_m3_vector_index

    runtime = load_m3_runtime_config(getattr(args, "config", None))
    # Apply explicit CLI overrides only when set, so the YAML remains
    # the source of truth for unspecified fields.
    conn = getattr(args, "vector_index_connection_string", None)
    if conn:
        runtime = _override(runtime, vector_index_connection_string=conn)
    table = getattr(args, "vector_index_table_name", None)
    if table:
        runtime = _override(runtime, vector_index_table_name=table)
    pool = getattr(args, "vector_index_pool_size", None)
    if pool is not None:
        runtime = _override(runtime, vector_index_pool_size=int(pool))
    backend = getattr(args, "vector_index_backend", None)
    if backend:
        runtime = _override(runtime, vector_index_backend=backend)
    return build_m3_vector_index(runtime)


def _override(runtime, **changes):
    """Return a new M3RuntimeConfig with the given fields replaced.

    ``M3RuntimeConfig`` is frozen so we can't mutate in place; this is
    the dataclasses-replacement-friendly equivalent of a one-line
    ``dataclasses.replace``. Lives here rather than in
    ``m3.config`` because no other module needs it today.
    """
    from dataclasses import replace
    return replace(runtime, **changes)


def _upsert_to_vector_index(
    index_records: list[dict[str, Any]],
    vector_index,
    args: argparse.Namespace,
) -> int:
    """Upsert each record's embedding + scope into the VectorIndex.

    For each ``index_record`` we build one :class:`IndexRecord` with:

    - ``id`` from ``IndexRecord.generate_id(uri, level=2)`` so
      re-indexing the same clip is idempotent (sha256-based, matches
      the seed-retriever convention).
    - ``uri = "m3://accounts/<account>/videos/<video>/clips/<clip>/episodic"``
      scopes record identity to the account; path segments are escaped.
      The JSONL manifest keeps its legacy source URI, not this storage key.
      Legacy unscoped vector entries should be rebuilt; they are not deleted
      here because a colliding legacy ID may now belong to another account.
    - ``level = 2`` (content-level), matching how seed_retriever
      treats context-node content.
    - ``filters = {account_id, video_id, extractor_type}`` — all
      tenant-/scope-style keys that the VectorIndex can push down to
      SQL via its ``_build_where`` allowlist.
    - ``metadata = {_embedding, extracted_node_id, video_id,
      clip_id, before_clip, source}`` — the forward channel back to
      the originating graph node (``extracted_node_id`` is what
      GraphService.search_by_embedding reads to re-fetch the
      ExtractedNode) plus the provenance keys that the QA prompt
      builder may want to surface.

    Returns the count of records successfully upserted. Records
    missing ``source_extracted_node_id`` or ``embedding`` are
    silently dropped — they cannot be reverse-resolved to a graph
    node and an embedding-less index entry would be worse than no
    entry.
    """
    if vector_index is None:
        return 0

    from core.models import IndexRecord

    payload: list[IndexRecord] = []
    for ir in index_records:
        ext_id = ir.get("source_extracted_node_id")
        emb = ir.get("embedding")
        if not ext_id or not emb:
            continue
        video_id = ir.get("video_id") or ""
        clip_id = ir.get("clip_id")
        try:
            before_clip_int: int | None = int(clip_id) if clip_id is not None else None
        except (TypeError, ValueError):
            before_clip_int = None
        from m3.index_status import episodic_uri
        uri = episodic_uri(args.account_id, video_id, clip_id)
        payload.append(IndexRecord(
            id=IndexRecord.generate_id(uri, level=2),
            uri=uri,
            level=2,
            text=ir.get("text", ""),
            filters={
                "account_id": args.account_id,
                "video_id": video_id,
                "extractor_type": "episodic_memory_generator",
            },
            metadata={
                **(ir.get("metadata") or {}),
                **temporal_metadata(ir.get("metadata") or {}),
                "_embedding": list(emb),
                "extracted_node_id": str(ext_id),
                "video_id": video_id,
                "clip_id": clip_id,
                "before_clip": before_clip_int,
                "source": "episodic_memory_generator",
            },
        ))

    if not payload:
        return 0
    vector_index.upsert(payload)
    return len(payload)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
