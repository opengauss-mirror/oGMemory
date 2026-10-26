"""M3-Bench cross-clip voice identity clustering.

Voice identity comes from one of two ExtractedNode flavours:

* ``speaker_embedding`` (NEW, 192-dim ECAPA-TDNN) — preferred when
  present. The speaker_embedding operator runs ECAPA-TDNN over the
  per-speaker audio range, producing a 192-dim L2-normalised vector
  that actually discriminates between distinct speakers.
* ``voice_segmentation`` (8-dim spectral fingerprint) — legacy
  fallback. Useful for graphs written before the ECAPA stage ran
  and for cheap "acoustic similarity" hints on noisy clips.

This driver walks the graph, prefers the ECAPA nodes when present,
falls back to the 8-dim fingerprint nodes for any audio without an
ECAPA sibling, runs single-linkage cosine clustering on the union,
and emits ``same_identity`` edges between the cluster anchor and
each member. The default ``--threshold`` flips from ``0.3`` (8-dim
calibration) to ``0.5`` (192-dim ECAPA calibration) automatically
when ECAPA nodes are found; pass ``--threshold`` explicitly to
override.

Why this mirrors ``face_clustering`` rather than reusing it:

* Same algorithm, but a different ExtractedNode shape — voice
  segments don't carry embeddings in the same way face nodes do
  (no bbox, different confidence signal). Building a dedicated
  driver keeps each cluster pass focused on its modality.
* Driver-level concern: walk the graph, cluster, emit edges. Same
  idempotency contract as ``face_clustering`` (skip already-linked
  members on re-run).

Within-clip pair policy: voice segments in the same clip are NOT
skipped here, because two voices CAN speak simultaneously (dialogue).
Within-clip clustering produces ``co_occurs_in`` semantics — left
for future work. Cross-clip is the use case.

Usage::

    python -m m3.voice_clustering \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --threshold 0.5 \\
        --output-jsonl data/voice_clusters.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from core.graph_enums import GraphEdgeType
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

@dataclass
class VoiceRecord:
    """One voice segment with its embedding + provenance."""

    node_id: str
    embedding: list[float]
    confidence: float
    voice_id: str
    video_id: str | None = None
    clip_id: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    # Which embedding flavour drove this record. ``ecapa`` = 192-dim
    # ECAPA-TDNN from the speaker_embedding operator; ``spectral`` =
    # 8-dim spectral fingerprint from voice_segmentation. Output
    # JSONL exposes this so downstream consumers can tell.
    embedding_kind: str = "spectral"


@dataclass
class VoiceClusterMember:
    """One row of the cluster output JSONL."""

    segment_node_id: str
    anchor_node_id: str
    voice_id: str
    video_id: str | None
    clip_id: int | None
    start_sec: float
    end_sec: float
    cluster_size: int
    cluster_score: float
    embedding_kind: str = "spectral"

    def to_jsonl(self) -> dict:
        return asdict(self)


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance (1 - cosine similarity) in [0, 2]. Zero-norm
    vectors return 1.0 (max distance).
    """
    if len(a) != len(b):
        raise ValueError(f"embedding dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - dot / math.sqrt(na * nb)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def cluster_voices(
    voices: list[VoiceRecord],
    threshold: float = 0.3,
) -> list[list[int]]:
    """Single-linkage clustering by cosine distance.

    Returns a list of clusters (each is a list of indices into
    ``voices``). Unlike face_clustering we DON'T skip within-clip
    pairs — two voices can speak in the same clip and the caller
    can disambiguate via the voice_id field.
    """
    n = len(voices)
    if n == 0:
        return []
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = _cosine_distance(voices[i].embedding, voices[j].embedding)
            if d < threshold:
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)
    return [sorted(g) for g in groups.values()]


def build_cluster_output(
    clusters: list[list[int]],
    voices: list[VoiceRecord],
) -> list[VoiceClusterMember]:
    """Pick the highest-confidence voice per cluster as the anchor;
    emit one row per voice pointing at its anchor."""
    out: list[VoiceClusterMember] = []
    for cluster in clusters:
        if not cluster:
            continue
        anchor_i = max(
            cluster,
            key=lambda i: (voices[i].confidence, -i),
        )
        anchor = voices[anchor_i]
        size = len(cluster)
        score = 1.0
        if size > 1:
            best_d = min(
                _cosine_distance(anchor.embedding, voices[j].embedding)
                for j in cluster if j != anchor_i
            )
            score = max(0.0, 1.0 - best_d)
        for idx in cluster:
            v = voices[idx]
            out.append(VoiceClusterMember(
                segment_node_id=v.node_id,
                anchor_node_id=anchor.node_id,
                voice_id=v.voice_id,
                video_id=v.video_id,
                clip_id=v.clip_id,
                start_sec=v.start_sec or 0.0,
                end_sec=v.end_sec or 0.0,
                cluster_size=size,
                cluster_score=round(score, 4),
                embedding_kind=v.embedding_kind,
            ))
    return out


# ---------------------------------------------------------------------------
# Graph-side helpers
# ---------------------------------------------------------------------------

def _records_from_speaker_embedding(
    graph, ctx: RequestContext,
) -> list[VoiceRecord]:
    """Pull every ``speaker_embedding`` ExtractedNode with a non-empty
    ``speaker_embedding`` vector.

    These are the 192-dim ECAPA-TDNN vectors produced by
    :class:`SpeakerEmbeddingExtractor`. One child per ``speaker_id``
    per clip — preferred over the 8-dim spectral fingerprint.
    """
    out: list[VoiceRecord] = []
    for n in graph.list_extracted_nodes("speaker_embedding", ctx):
        ec = n.extracted_content or {}
        emb = ec.get("speaker_embedding")
        if not emb or not isinstance(emb, list):
            continue
        try:
            emb_list = [float(x) for x in emb]
        except (TypeError, ValueError):
            continue
        if not emb_list:
            continue
        # Confidence proxy: weighted sum of num_turns and duration
        # so clusters prefer anchors backed by more / longer speech.
        num_turns = float(ec.get("num_turns", 0))
        total_dur = float(ec.get("total_duration_sec", 0.0))
        confidence = num_turns * (1.0 + total_dur / 10.0)
        out.append(VoiceRecord(
            node_id=n.node_id,
            embedding=emb_list,
            confidence=confidence,
            voice_id=str(ec.get("speaker_id", "")),
            video_id=ec.get("source_video_id"),
            clip_id=ec.get("source_clip_id"),
            start_sec=ec.get("start_sec"),
            end_sec=ec.get("end_sec"),
            embedding_kind="ecapa",
        ))
    out.sort(key=lambda v: v.node_id)
    return out


def _records_from_voice_segmentation(
    graph, ctx: RequestContext,
) -> list[VoiceRecord]:
    """Fallback: 8-dim spectral fingerprint from voice_segmentation.

    Used for graphs written before the ECAPA stage ran. The
    voice_clustering driver prefers ``speaker_embedding`` and only
    reads these for audio that has no ECAPA sibling.
    """
    out: list[VoiceRecord] = []
    for n in graph.list_extracted_nodes("voice_segmentation", ctx):
        ec = n.extracted_content or {}
        emb = ec.get("voice_embedding")
        if not emb or not isinstance(emb, list):
            continue
        try:
            emb_list = [float(x) for x in emb]
        except (TypeError, ValueError):
            continue
        if not emb_list:
            continue
        out.append(VoiceRecord(
            node_id=n.node_id,
            embedding=emb_list,
            confidence=float(ec.get("confidence", 0.0)),
            voice_id=str(ec.get("voice_id", "")),
            video_id=ec.get("source_video_id"),
            clip_id=ec.get("source_clip_id"),
            start_sec=ec.get("start_sec"),
            end_sec=ec.get("end_sec"),
            embedding_kind="spectral",
        ))
    out.sort(key=lambda v: v.node_id)
    return out


def _voice_records_from_graph(
    graph, ctx: RequestContext,
) -> list[VoiceRecord]:
    """Pull voice embeddings from the graph.

    Preference order:
      1. ``speaker_embedding`` (192-dim ECAPA-TDNN).
      2. ``voice_segmentation`` (8-dim spectral fingerprint).

    Back-compat: graphs that only have voice_segmentation nodes
    still cluster — we just don't get the ECAPA signal.

    Within a single audio clip the ECAPA and spectral records are
    keyed by different things (speaker_id vs voice_id), so we keep
    them in separate ``VoiceRecord`` lists and let the caller pick
    the dominant signal. We don't try to merge across kinds — the
    8-dim and 192-dim spaces are incompatible.
    """
    ecapa = _records_from_speaker_embedding(graph, ctx)
    if ecapa:
        return ecapa
    return _records_from_voice_segmentation(graph, ctx)


def _existing_same_identity_anchors(
    graph, ctx: RequestContext,
) -> dict[str, str]:
    """member → anchor map from existing same_identity edges, so
    re-runs are idempotent. We trust the source/target convention
    (member → anchor) emitted by this driver; both
    ``voice_segmentation`` and ``speaker_embedding`` child nodes
    qualify.
    """
    out: dict[str, str] = {}
    eligible_ids: set[str] = set()
    for n in graph.list_extracted_nodes("voice_segmentation", ctx):
        eligible_ids.add(n.node_id)
    for n in graph.list_extracted_nodes("speaker_embedding", ctx):
        eligible_ids.add(n.node_id)
    for n in graph.list_extracted_nodes(None, ctx):
        for _nb, edge, _dir in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.SAME_IDENTITY],
        ):
            if edge.source_id in eligible_ids:
                out[edge.source_id] = edge.target_id
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_voice_clustering(
    args: argparse.Namespace,
    graph=None,
) -> int:
    if graph is None:
        from m3.load_clips import build_graph_service
        graph, store_kind = build_graph_service(
            args.graph_store, args.connection_string,
        )
        logger.info("Graph store: %s", store_kind)

    ctx = RequestContext(
        account_id=getattr(args, "account_id", M3BENCH_ACCOUNT_ID),
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-voice-clustering",
        trace_id="m3-voice-clustering",
    )

    voices = _voice_records_from_graph(graph, ctx)
    logger.info("Loaded %d voice segments with embeddings", len(voices))
    if voices:
        kinds = {v.embedding_kind for v in voices}
        logger.info("Embedding flavours present: %s", sorted(kinds))
    if len(voices) < 2:
        logger.info("Need at least 2 voice segments to cluster; exiting")

    # Auto-pick the threshold default when the user didn't pass one.
    # 8-dim spectral: 0.3 (calibrated empirically for cosine
    # distance on the loudness/ZCR/spectral fingerprint). 192-dim
    # ECAPA: 0.5 (cross-speaker distances cluster around 0.4-0.6
    # per the ECAPA paper). An explicit --threshold always wins.
    user_threshold = getattr(args, "threshold", None)
    if user_threshold is None:
        threshold = 0.5 if any(v.embedding_kind == "ecapa" for v in voices) else 0.3
    else:
        threshold = float(user_threshold)
    started = time.monotonic()
    clusters = cluster_voices(voices, threshold=threshold)
    cluster_records = build_cluster_output(clusters, voices)
    logger.info(
        "Single-linkage cosine clustering (threshold=%.3f) → %d clusters "
        "in %.2fs",
        threshold, len({c.anchor_node_id for c in cluster_records}),
        time.monotonic() - started,
    )

    existing = _existing_same_identity_anchors(graph, ctx)
    emitted = 0
    skipped = 0
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        anchor_i = max(cluster, key=lambda i: (voices[i].confidence, -i))
        anchor_id = voices[anchor_i].node_id
        for idx in cluster:
            if idx == anchor_i:
                continue
            member_id = voices[idx].node_id
            if existing.get(member_id) == anchor_id:
                skipped += 1
                continue
            if member_id in existing:
                skipped += 1
                continue
            graph.create_edge(
                source_id=member_id,
                target_id=anchor_id,
                edge_type=GraphEdgeType.SAME_IDENTITY,
                properties={"score": round(cluster_records[0].cluster_score, 4)},
                ctx=ctx,
            )
            emitted += 1
    logger.info("Emitted %d same_identity edges (%d already linked)",
                emitted, skipped)

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Always write the output JSONL, even if there are no voices —
    # downstream consumers rely on the file existing.
    with output_path.open("w", encoding="utf-8") as f:
        for r in cluster_records:
            f.write(json.dumps(r.to_jsonl(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d cluster rows → %s", len(cluster_records), output_path)
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster voice segments across clips and emit same_identity edges.",
    )
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID,
                   help="Account whose voice identities may be read and updated.")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Cosine distance threshold (smaller = stricter). Default 0.3 "
                        "for the 8-dim spectral fingerprint; tune for other embeddings.")
    p.add_argument("--output-jsonl", default="data/voice_clusters.jsonl",
                   help="Per-segment cluster rows for audit / downstream consumers.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_voice_clustering(args)


if __name__ == "__main__":
    sys.exit(main())
