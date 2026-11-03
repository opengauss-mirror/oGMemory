"""Tests for ``m3.voice_clustering`` — cross-clip voice identity.

Three layers:

* Pure algorithm — cluster_voices, build_cluster_output, cosine
  distance. No I/O. Pins cluster shape (anchor picks highest
  confidence, score = 1 - min_distance) and verifies voice_id
  semantics (unlike face_clustering, within-clip pairs are NOT
  skipped — two voices can speak in the same clip).

* Driver — run_voice_clustering with a seeded graph. Asserts the
  same_identity edges land in the graph and re-runs are idempotent.

* Pipeline stage — voice_clustering_stage through PipelineRunner
  with voice_segmentation + voice_clustering enabled, asserting the
  stage counts match the driver output and edges land in the graph.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from core.graph_enums import GraphEdgeType, RawDataType
from core.graph_models import create_extracted_node, create_raw_node
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from m3.face_clustering import _cosine_distance as face_cosine_distance
from m3.voice_clustering import (
    VoiceClusterMember,
    VoiceRecord,
    _cosine_distance,
    build_cluster_output,
    cluster_voices,
    run_voice_clustering,
)
from providers.graph_store import MemoryGraphStore
from service.graph_service import GraphService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx(account_id=M3BENCH_ACCOUNT_ID) -> RequestContext:
    return RequestContext(
        account_id=account_id, user_id="u", agent_id="a",
        session_id="m3bench-voice-clustering-test", trace_id="t-test",
    )


def _voice(node_id: str, embedding: list[float], *,
           confidence: float = 0.9, voice_id: str = "v_a",
           video_id: str = "v_a", clip_id: int = 0) -> VoiceRecord:
    return VoiceRecord(
        node_id=node_id, embedding=embedding, confidence=confidence,
        voice_id=voice_id,
        video_id=video_id, clip_id=clip_id,
        start_sec=0.0, end_sec=1.0,
    )


# ---------------------------------------------------------------------------
# Pure algorithm
# ---------------------------------------------------------------------------

def test_cosine_distance_unit_vectors():
    """Identical → 0, orthogonal → 1, opposite → 2."""
    assert _cosine_distance([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    assert _cosine_distance([1, 0, 0], [0, 1, 0]) == pytest.approx(1.0)
    assert _cosine_distance([1, 0, 0], [-1, 0, 0]) == pytest.approx(2.0)


def test_cosine_distance_zero_norm_returns_max():
    assert _cosine_distance([0, 0, 0], [1, 0, 0]) == 1.0
    assert _cosine_distance([0, 0, 0], [0, 0, 0]) == 1.0


def test_cosine_distance_matches_face_clustering_implementation():
    """Both modules have their own cosine distance helper; verify
    they agree on a few values so the two clustering drivers
    produce identical clusters when fed the same vectors."""
    a = [1.0, 0.5, -0.3, 0.2]
    b = [0.5, 0.4, 0.1, 0.0]
    assert _cosine_distance(a, b) == pytest.approx(face_cosine_distance(a, b))


def test_cluster_voices_empty_input():
    assert cluster_voices([]) == []


def test_cluster_voices_single_voice_is_singleton():
    voices = [_voice("n1", [1.0, 0.0])]
    assert cluster_voices(voices) == [[0]]


def test_cluster_voices_two_close_pairs_grouped():
    voices = [
        _voice("n1", [1.0, 0.0]),
        _voice("n2", [0.99, 0.01]),
    ]
    clusters = cluster_voices(voices, threshold=0.4)
    assert clusters == [[0, 1]]


def test_cluster_voices_does_not_skip_within_clip_pairs():
    """Voice clustering: unlike face_clustering, two segments in the
    SAME clip ARE clustered — two voices can speak simultaneously
    and the caller disambiguates via voice_id."""
    voices = [
        _voice("n1", [1.0, 0.0], video_id="v_a", clip_id=0),
        _voice("n2", [1.0, 0.0], video_id="v_a", clip_id=0),
    ]
    clusters = cluster_voices(voices, threshold=0.4)
    # Both end up in the same cluster — caller filters via voice_id
    # to split co-occurring voices.
    assert clusters == [[0, 1]]


def test_cluster_voices_threshold_separates_distant_pairs():
    voices = [
        _voice("n1", [1.0, 0.0]),
        _voice("n2", [0.5, 0.866]),
    ]
    assert len(cluster_voices(voices, threshold=0.4)) == 2
    assert len(cluster_voices(voices, threshold=0.6)) == 1


def test_cluster_voices_transitive_three_voice_chain():
    a = [1.0, 0.0, 0.0]
    b = [0.9, 0.436, 0.0]
    c = [0.8, 0.6, 0.0]
    voices = [
        _voice("n1", a),
        _voice("n2", b),
        _voice("n3", c),
    ]
    clusters = cluster_voices(voices, threshold=0.4)
    assert clusters == [[0, 1, 2]]


def test_build_cluster_output_picks_highest_confidence_anchor():
    voices = [
        _voice("n1", [1.0, 0.0], confidence=0.5),
        _voice("n2", [1.0, 0.0], confidence=0.9),
    ]
    clusters = [[0, 1]]
    rows = build_cluster_output(clusters, voices)
    assert len(rows) == 2
    # n2 is the anchor (higher confidence).
    assert rows[0].anchor_node_id == "n2"
    assert rows[0].segment_node_id == "n1"
    assert rows[1].anchor_node_id == "n2"
    assert rows[1].segment_node_id == "n2"
    assert rows[0].cluster_size == 2
    assert rows[0].cluster_score == 1.0


def test_build_cluster_output_singleton_row_has_perfect_score():
    voices = [_voice("n1", [1.0, 0.0])]
    rows = build_cluster_output([[0]], voices)
    assert rows[0].segment_node_id == "n1"
    assert rows[0].anchor_node_id == "n1"
    assert rows[0].cluster_size == 1
    assert rows[0].cluster_score == 1.0


def test_build_cluster_output_carries_provenance():
    """Per-voice video_id + clip_id flow through to the cluster rows
    so downstream [Search] can scope by clip without a graph
    traversal."""
    voices = [
        _voice("n1", [1.0, 0.0], voice_id="voice_a", video_id="v_a", clip_id=2),
        _voice("n2", [1.0, 0.0], voice_id="voice_b", video_id="v_b", clip_id=5),
    ]
    rows = build_cluster_output([[0, 1]], voices)
    by_id = {r.segment_node_id: r for r in rows}
    assert by_id["n1"].video_id == "v_a"
    assert by_id["n1"].clip_id == 2
    assert by_id["n1"].voice_id == "voice_a"
    assert by_id["n2"].video_id == "v_b"
    assert by_id["n2"].clip_id == 5
    assert by_id["n2"].voice_id == "voice_b"


# ---------------------------------------------------------------------------
# Driver — graph I/O
# ---------------------------------------------------------------------------

def _seed_voice_segment(
    graph, ctx, *, node_id, embedding, voice_id,
    video_id="v", clip_id=0, confidence=0.9,
    start_sec=0.0, end_sec=1.0,
):
    node = create_extracted_node(
        source_raw_id=f"audio-{video_id}-{clip_id}"[:32].ljust(32, "x"),
        extractor_type="voice_segmentation",
        extraction_schema="voice_segmentation_v1",
        account_id=ctx.account_id,
        extracted_content={
            "voice_embedding": embedding,
            "voice_id": voice_id,
            "confidence": confidence,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": end_sec - start_sec,
            "source_video_id": video_id,
            "source_clip_id": clip_id,
        },
        node_id=node_id,
    )
    graph._store.create_node(node.serialize(), ctx)


def test_driver_emits_same_identity_edges_in_graph(tmp_path: Path):
    """Seed 3 voice segments across 3 clips — two identity groups.
    Assert same_identity edges for the matching pairs."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    # Identity A.
    _seed_voice_segment(graph, ctx, node_id="vs_a1",
                        embedding=[1.0, 0.0, 0.0], voice_id="va",
                        video_id="v", clip_id=0)
    _seed_voice_segment(graph, ctx, node_id="vs_a2",
                        embedding=[0.99, 0.01, 0.0], voice_id="va",
                        video_id="v", clip_id=1)
    # Identity B (far from A).
    _seed_voice_segment(graph, ctx, node_id="vs_b1",
                        embedding=[0.0, 1.0, 0.0], voice_id="vb",
                        video_id="v", clip_id=2)

    output_jsonl = tmp_path / "voice_clusters.jsonl"
    from m3.voice_clustering import parse_args
    args = parse_args([
        "--graph-store", "memory",
        "--output-jsonl", str(output_jsonl),
        "--threshold", "0.4",
    ])
    rc = run_voice_clustering(args, graph=graph)
    assert rc == 0

    # Edges (dedupe via frozenset since get_neighbors visits each
    # edge from both endpoints).
    same_id: set = set()
    for n in graph.list_extracted_nodes(None, ctx):
        for _nb, e, _d in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.SAME_IDENTITY],
        ):
            same_id.add(frozenset([e.source_id, e.target_id]))
    member_pairs = {p for p in same_id if len(p) == 2}
    assert frozenset(["vs_a2", "vs_a1"]) in member_pairs


def test_driver_is_idempotent_on_rerun(tmp_path: Path):
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    _seed_voice_segment(graph, ctx, node_id="a1",
                        embedding=[1.0, 0.0], voice_id="va")
    _seed_voice_segment(graph, ctx, node_id="a2",
                        embedding=[0.99, 0.01], voice_id="va")

    from m3.voice_clustering import parse_args
    args = parse_args([
        "--graph-store", "memory",
        "--output-jsonl", str(tmp_path / "c.jsonl"),
        "--threshold", "0.4",
    ])
    run_voice_clustering(args, graph=graph)
    edges_first = sum(
        1 for n in graph.list_extracted_nodes(None, ctx)
        for _ in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.SAME_IDENTITY],
        )
    )
    run_voice_clustering(args, graph=graph)
    edges_second = sum(
        1 for n in graph.list_extracted_nodes(None, ctx)
        for _ in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.SAME_IDENTITY],
        )
    )
    assert edges_first == edges_second


def test_driver_no_voice_embeddings_exits_cleanly(tmp_path: Path):
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    # Seed a voice_segment WITHOUT an embedding.
    _seed_voice_segment(graph, ctx, node_id="no_emb",
                        embedding=None, voice_id="vx")

    from m3.voice_clustering import parse_args
    args = parse_args([
        "--graph-store", "memory",
        "--output-jsonl", str(tmp_path / "c.jsonl"),
    ])
    rc = run_voice_clustering(args, graph=graph)
    assert rc == 0
    # File always exists (downstream consumers rely on this).
    assert (tmp_path / "c.jsonl").exists()


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------

def test_voice_clustering_stage_is_registered():
    from m3.pipeline.stages import DEFAULT_STAGE_REGISTRY
    assert "m3.voice_clustering" in DEFAULT_STAGE_REGISTRY


@pytest.mark.parametrize("account_id", [M3BENCH_ACCOUNT_ID, "isolated-pilot"])
def test_voice_clustering_stage_writes_edges_through_pipeline(tmp_path: Path, account_id):
    """End-to-end: seed two voice segments with close embeddings, run
    the stage, verify edges + JSONL."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx(account_id)
    other_ctx = _ctx("other-account")
    _seed_voice_segment(graph, other_ctx, node_id="other_a",
                        embedding=[1.0, 0.0], voice_id="other")
    _seed_voice_segment(graph, other_ctx, node_id="other_b",
                        embedding=[0.99, 0.01], voice_id="other")
    _seed_voice_segment(graph, ctx, node_id="v_a",
                        embedding=[1.0, 0.0], voice_id="va")
    _seed_voice_segment(graph, ctx, node_id="v_b",
                        embedding=[0.99, 0.01], voice_id="vb")

    from m3.pipeline.context import PipelineContext
    from m3.pipeline.spec import StageSpec
    from m3.pipeline import stages as stages_mod

    pipe_ctx = PipelineContext(
        video=tmp_path / "src.mp4",
        video_id="v", seconds=10, work_dir=tmp_path / "run",
        config_path=Path("deploy/ogmemory.yaml"),
        graph_store="memory", connection_string=None,
        account_id=account_id,
        artifacts={"voice_clusters": tmp_path / "run" / "voice_clusters.jsonl"},
        variables={}, graph=graph,
    )
    stage = StageSpec(
        id="voice_clustering", type="m3.voice_clustering",
        needs=["voice_segmentation"], enabled=True,
        params={"threshold": 0.4},
    )
    result = stages_mod.voice_clustering_stage(pipe_ctx, stage)
    assert result.ok is True
    assert result.counts["voices"] == 2
    assert result.counts["clusters"] == 1
    for node_id in ("other_a", "other_b"):
        assert not graph.get_neighbors(
            node_id, other_ctx, edge_types=[GraphEdgeType.SAME_IDENTITY],
        )

    edges = set()
    for n in graph.list_extracted_nodes(None, ctx):
        for _nb, e, _d in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.SAME_IDENTITY],
        ):
            edges.add(frozenset([e.source_id, e.target_id]))
    assert edges == {frozenset(["v_a", "v_b"])}


def test_voice_clustering_yaml_stage_position():
    """voice_clustering must depend on voice_segmentation so the
    voice embeddings exist before clustering runs."""
    import yaml
    spec = yaml.safe_load(
        Path("config/m3/pipelines/video_episodic_faces.yaml").read_text()
    )
    stage_ids = [s["id"] for s in spec["stages"]]
    assert "voice_clustering" in stage_ids
    vc = next(s for s in spec["stages"] if s["id"] == "voice_clustering")
    assert vc["type"] == "m3.voice_clustering"
    assert "voice_segmentation" in vc["needs"]
    assert vc["enabled"] is False
    assert "threshold" in vc["params"]
    assert "voice_clusters" in spec["artifacts"]
