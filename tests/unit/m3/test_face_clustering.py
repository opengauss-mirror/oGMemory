"""Tests for ``m3.face_clustering`` — cross-clip identity clustering.

Three layers:

* Pure algorithm — cluster_faces + build_cluster_output + cosine
  distance. No I/O, no graph. Pins the cluster shape (anchor picks
  highest confidence, score = 1 - min_distance, within-clip pairs
  skipped).

* Driver — run_face_clustering with a seeded graph. Asserts the
  same_identity edges land in the graph, the cluster JSONL gets
  one row per face (including singletons), and re-runs are
  idempotent.

* Pipeline stage — face_clustering_stage through PipelineRunner
  with detect_faces + face_clustering enabled, asserting the
  stage counts match the driver output and edges actually land
  in the graph.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from core.graph_enums import GraphEdgeType
from core.graph_models import create_extracted_node
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from m3.face_clustering import (
    ClusterMember,
    FaceRecord,
    _UnionFind,
    _cosine_distance,
    build_cluster_output,
    cluster_faces,
    run_face_clustering,
)
from providers.graph_store import MemoryGraphStore
from service.graph_service import GraphService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="m3bench-face-clustering-test", trace_id="t-test",
    )


def _face(node_id: str, embedding: list[float], *,
          confidence: float = 0.9, video_id: str = "v_a", clip_id: int = 0,
          bbox: list[float] | None = None) -> FaceRecord:
    return FaceRecord(
        node_id=node_id, embedding=embedding, confidence=confidence,
        bbox=bbox or [0.1, 0.1, 0.5, 0.5],
        video_id=video_id, clip_id=clip_id,
    )


# ---------------------------------------------------------------------------
# Pure algorithm
# ---------------------------------------------------------------------------

def test_cosine_distance_unit_vectors():
    """Identical vectors → distance 0; orthogonal → 1; opposite → 2."""
    assert _cosine_distance([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    assert _cosine_distance([1, 0, 0], [0, 1, 0]) == pytest.approx(1.0)
    assert _cosine_distance([1, 0, 0], [-1, 0, 0]) == pytest.approx(2.0)


def test_cosine_distance_zero_vector_returns_max():
    """Zero-norm vectors are skipped (max distance = 1.0)."""
    assert _cosine_distance([0, 0, 0], [1, 0, 0]) == 1.0
    assert _cosine_distance([0, 0, 0], [0, 0, 0]) == 1.0


def test_cosine_distance_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="dim mismatch"):
        _cosine_distance([1, 0], [1, 0, 0])


def test_cluster_faces_empty_input():
    assert cluster_faces([]) == []


def test_cluster_faces_single_face_is_singleton():
    faces = [_face("n1", [1.0, 0.0])]
    clusters = cluster_faces(faces)
    assert clusters == [[0]]


def test_cluster_faces_two_cross_clip_matches_grouped():
    """Two faces in different clips with close embeddings → one cluster."""
    faces = [
        _face("n1", [1.0, 0.0], video_id="v_a", clip_id=0),
        _face("n2", [0.99, 0.01], video_id="v_b", clip_id=3),
    ]
    clusters = cluster_faces(faces, threshold=0.4)
    assert clusters == [[0, 1]]


def test_cluster_faces_within_clip_pair_skipped():
    """Two faces in the SAME clip are not clustered even when their
    embeddings are identical — within-clip identity is captured by
    co_occurs_in (future work), not same_identity."""
    faces = [
        _face("n1", [1.0, 0.0], video_id="v_a", clip_id=0),
        _face("n2", [1.0, 0.0], video_id="v_a", clip_id=0),
    ]
    clusters = cluster_faces(faces, threshold=0.4)
    assert clusters == [[0], [1]]  # both singletons


def test_cluster_faces_threshold_separates_distant_pairs():
    """Threshold controls which pairs join. Distance ~0.5 with
    threshold 0.4 → two clusters; with threshold 0.6 → one cluster."""
    faces = [
        _face("n1", [1.0, 0.0], video_id="v_a", clip_id=0),
        _face("n2", [0.5, 0.866], video_id="v_b", clip_id=0),  # dist ~0.5
    ]
    assert len(cluster_faces(faces, threshold=0.4)) == 2
    assert len(cluster_faces(faces, threshold=0.6)) == 1


def test_cluster_faces_transitive_three_face_chain():
    """A-B close, B-C close, A-C far → all three in one cluster
    (single-linkage property)."""
    a = [1.0, 0.0, 0.0]
    # Slightly tilted toward b
    b = [0.9, 0.436, 0.0]
    # Slightly tilted toward c, building on b
    c = [0.8, 0.6, 0.0]
    faces = [
        _face("n1", a, video_id="v_a", clip_id=0),
        _face("n2", b, video_id="v_b", clip_id=0),
        _face("n3", c, video_id="v_c", clip_id=0),
    ]
    # dist(a,b) ~ 0.10, dist(b,c) ~ 0.13, dist(a,c) ~ 0.36.
    clusters = cluster_faces(faces, threshold=0.4)
    assert clusters == [[0, 1, 2]]


def test_build_cluster_output_picks_highest_confidence_anchor():
    faces = [
        _face("n1", [1.0, 0.0], confidence=0.5, video_id="v_a", clip_id=0),
        _face("n2", [1.0, 0.0], confidence=0.9, video_id="v_b", clip_id=0),
    ]
    clusters = [[0, 1]]
    rows = build_cluster_output(clusters, faces)
    assert len(rows) == 2
    # n2 is the anchor (higher confidence).
    assert rows[0].anchor_node_id == "n2"  # face n1 → anchor n2
    assert rows[0].face_node_id == "n1"
    assert rows[1].anchor_node_id == "n2"  # anchor row points at itself
    assert rows[1].face_node_id == "n2"
    # Score = 1.0 (identical embeddings).
    assert rows[0].cluster_score == 1.0
    assert rows[1].cluster_score == 1.0
    assert rows[0].cluster_size == 2


def test_build_cluster_output_singleton_row_has_perfect_score():
    """A singleton cluster (one face, no anchor comparison) gets
    cluster_score = 1.0 — the face is trivially its own identity."""
    faces = [_face("n1", [1.0, 0.0])]
    rows = build_cluster_output([[0]], faces)
    assert rows[0].face_node_id == "n1"
    assert rows[0].anchor_node_id == "n1"
    assert rows[0].cluster_size == 1
    assert rows[0].cluster_score == 1.0


# ---------------------------------------------------------------------------
# Driver — graph I/O
# ---------------------------------------------------------------------------

def _seed_face(graph, ctx, *, node_id, embedding, video_id, clip_id,
               confidence=0.9, bbox=None):
    node = create_extracted_node(
        source_raw_id=f"video-{video_id}-{clip_id}"[:32].ljust(32, "x"),
        extractor_type="face_detection",
        extraction_schema="face_detection_v1",
        account_id=ctx.account_id,
        extracted_content={
            "bbox": bbox or [0.1, 0.1, 0.5, 0.5],
            "confidence": confidence,
            "embedding": embedding,
            "_video_id": video_id,
            "_clip_id": clip_id,
        },
        node_id=node_id,
    )
    graph._store.create_node(node.serialize(), ctx)


def test_driver_emits_same_identity_edges_in_graph(tmp_path: Path):
    """End-to-end: seed 3 faces across 3 clips with two identity
    groups; run the driver; assert same_identity edges exist for
    the matching pairs."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    # Identity A: faces in clip 0 and clip 1.
    _seed_face(graph, ctx, node_id="face_a1", embedding=[1.0, 0.0, 0.0],
               video_id="v_a", clip_id=0)
    _seed_face(graph, ctx, node_id="face_a2", embedding=[0.99, 0.01, 0.0],
               video_id="v_a", clip_id=1)
    # Identity B: faces in clip 2 (far from A).
    _seed_face(graph, ctx, node_id="face_b1", embedding=[0.0, 1.0, 0.0],
               video_id="v_a", clip_id=2)

    output_jsonl = tmp_path / "clusters.jsonl"
    from m3.face_clustering import parse_args
    args = parse_args([
        "--graph-store", "memory",
        "--output-jsonl", str(output_jsonl),
        "--threshold", "0.4",
    ])
    rc = run_face_clustering(args, graph=graph)
    assert rc == 0

    # Edges: face_a2 → face_a1 (anchor) AND face_a1 → face_a1 (self-anchor).
    # Anchor for cluster {a1, a2} is the higher-confidence face.
    # Dedupe undirected — get_neighbors visits each edge from both
    # endpoints, so we use frozenset pairs.
    same_id: set = set()
    for n in graph.list_extracted_nodes(None, ctx):
        for _nb, e, _d in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.SAME_IDENTITY],
        ):
            same_id.add(frozenset([e.source_id, e.target_id]))
    member_pairs = {p for p in same_id if len(p) == 2}
    # face_a2 is the cluster anchor → face_a1; face_b1 is a
    # singleton (no same_identity edge at all).
    assert frozenset(["face_a2", "face_a1"]) in member_pairs
    assert frozenset(["face_b1", "face_b1"]) not in same_id

    # JSONL has one row per face.
    rows = [json.loads(l) for l in output_jsonl.read_text().splitlines() if l]
    assert len(rows) == 3
    by_face = {r["face_node_id"]: r for r in rows}
    assert by_face["face_a1"]["anchor_node_id"] == "face_a1"
    assert by_face["face_a2"]["anchor_node_id"] == "face_a1"
    assert by_face["face_b1"]["anchor_node_id"] == "face_b1"
    assert by_face["face_a2"]["cluster_size"] == 2


def test_driver_is_idempotent_on_rerun(tmp_path: Path):
    """Re-running the driver must NOT duplicate same_identity edges
    — the existence check skips members already linked."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    _seed_face(graph, ctx, node_id="a1", embedding=[1.0, 0.0],
               video_id="v", clip_id=0)
    _seed_face(graph, ctx, node_id="a2", embedding=[0.99, 0.01],
               video_id="v", clip_id=1)

    from m3.face_clustering import parse_args
    args = parse_args([
        "--graph-store", "memory",
        "--output-jsonl", str(tmp_path / "c.jsonl"),
        "--threshold", "0.4",
    ])
    run_face_clustering(args, graph=graph)
    # Count edges.
    edges_first = sum(
        1 for n in graph.list_extracted_nodes(None, ctx)
        for _ in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.SAME_IDENTITY],
        )
    )
    run_face_clustering(args, graph=graph)
    edges_second = sum(
        1 for n in graph.list_extracted_nodes(None, ctx)
        for _ in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.SAME_IDENTITY],
        )
    )
    assert edges_first == edges_second, (
        f"expected idempotent re-run: {edges_first} → {edges_second}"
    )


def test_driver_no_face_embeddings_exits_cleanly(tmp_path: Path):
    """No faces with embeddings → 0 clusters, exit 0, empty JSONL."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    # Seed a face WITHOUT an embedding.
    _seed_face(graph, ctx, node_id="no_emb", embedding=None,
               video_id="v", clip_id=0)

    from m3.face_clustering import parse_args
    args = parse_args([
        "--graph-store", "memory",
        "--output-jsonl", str(tmp_path / "c.jsonl"),
    ])
    rc = run_face_clustering(args, graph=graph)
    assert rc == 0
    # File exists but is empty (no clusters to write).
    assert (tmp_path / "c.jsonl").exists()
    assert (tmp_path / "c.jsonl").read_text() == ""


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------

def test_face_clustering_stage_is_registered():
    from m3.pipeline.stages import DEFAULT_STAGE_REGISTRY
    assert "m3.face_clustering" in DEFAULT_STAGE_REGISTRY


def test_face_clustering_stage_writes_edges_through_pipeline(tmp_path: Path):
    """End-to-end through PipelineRunner: seed a graph with 2 faces,
    disable everything except face_clustering, run, verify the
    stage's count + the edges it emitted."""
    graph = GraphService(MemoryGraphStore())
    ctx = _ctx()
    _seed_face(graph, ctx, node_id="face_a", embedding=[1.0, 0.0],
               video_id="v", clip_id=0)
    _seed_face(graph, ctx, node_id="face_b", embedding=[0.99, 0.01],
               video_id="v", clip_id=1)

    # Inject the seeded graph + a minimal PipelineContext.
    from m3.pipeline.context import PipelineContext
    from m3.pipeline.spec import StageSpec
    from m3.pipeline import stages as stages_mod

    pipe_ctx = PipelineContext(
        video=tmp_path / "src.mp4",
        video_id="v", seconds=10, work_dir=tmp_path / "run",
        config_path=Path("deploy/ogmemory.yaml"),
        graph_store="memory", connection_string=None,
        account_id=M3BENCH_ACCOUNT_ID,
        artifacts={"face_clusters": tmp_path / "run" / "face_clusters.jsonl"},
        variables={}, graph=graph,
    )
    stage = StageSpec(
        id="face_clustering", type="m3.face_clustering",
        needs=["detect_faces"], enabled=True,
        params={"threshold": 0.4},
    )
    result = stages_mod.face_clustering_stage(pipe_ctx, stage)
    assert result.ok is True
    assert result.counts["faces"] == 2
    assert result.counts["clusters"] == 1

    # The actual same_identity edge landed in the graph. Dedupe via
    # frozenset because get_neighbors visits each edge from both
    # endpoints.
    edges: set = set()
    for n in graph.list_extracted_nodes(None, ctx):
        for _nb, e, _d in graph.get_neighbors(
            n.node_id, ctx,
            edge_types=[GraphEdgeType.SAME_IDENTITY],
        ):
            edges.add(frozenset([e.source_id, e.target_id]))
    assert edges == {frozenset(["face_a", "face_b"])}


def test_face_clustering_yaml_stage_position():
    """The YAML must declare face_clustering after detect_faces so
    the face embeddings exist before the clustering pass runs."""
    import yaml
    spec = yaml.safe_load(
        Path("config/m3/pipelines/video_episodic_faces.yaml").read_text()
    )
    stage_ids = [s["id"] for s in spec["stages"]]
    assert "face_clustering" in stage_ids
    assert "detect_faces" in stage_ids
    assert stage_ids.index("face_clustering") > stage_ids.index("detect_faces")
    fc = next(s for s in spec["stages"] if s["id"] == "face_clustering")
    assert fc["type"] == "m3.face_clustering"
    # Disabled by default — single-video runs produce no cross-clip
    # clusters, so enabling by default just costs the O(N²) scan.
    assert fc["enabled"] is False
    assert "threshold" in fc["params"]
    # face_clusters artifact wired up.
    assert "face_clusters" in spec["artifacts"]

class TestVectorizedClusteringEquivalence:
    """WS 3.7: the numpy-blocked GEMM must be an exact-match equivalent
    of the legacy O(N²) loop (same unions, same exclusions)."""

    def _faces(self, n, seed=7):
        import random

        rng = random.Random(seed)
        faces = []
        for i in range(n):
            emb = [rng.uniform(-1, 1) for _ in range(16)]
            faces.append(FaceRecord(
                node_id=f"n{i}",
                embedding=emb,
                video_id=None if i % 11 == 0 else f"v{i % 3}",
                clip_id=None if i % 13 == 0 else i % 4,
                confidence=0.9, bbox=[0.1, 0.1, 0.5, 0.5],
            ))
        return faces

    def _legacy_clusters(self, faces, threshold):
        uf = _UnionFind(len(faces))
        for i in range(len(faces)):
            fi = faces[i]
            for j in range(i + 1, len(faces)):
                fj = faces[j]
                if (fi.video_id is not None and fj.video_id is not None
                        and fi.video_id == fj.video_id
                        and fi.clip_id is not None and fj.clip_id is not None
                        and fi.clip_id == fj.clip_id):
                    continue
                if _cosine_distance(fi.embedding, fj.embedding) < threshold:
                    uf.union(i, j)
        groups = {}
        for i in range(len(faces)):
            groups.setdefault(uf.find(i), []).append(i)
        return sorted(sorted(g) for g in groups.values())

    def test_matches_legacy_loop_exactly(self):
        faces = self._faces(120)
        for threshold in (0.3, 0.4, 0.6, 1.2):
            legacy = self._legacy_clusters(faces, threshold)
            vectorized = sorted(cluster_faces(faces, threshold=threshold))
            assert vectorized == legacy, f"mismatch at threshold={threshold}"

    def test_zero_norm_embeddings_never_cluster(self):
        faces = [
            FaceRecord(node_id="a", embedding=[0.0] * 8, bbox=[0, 0, 1, 1],
                       video_id="v", clip_id=1, confidence=0.9),
            FaceRecord(node_id="b", embedding=[0.0] * 8, bbox=[0, 0, 1, 1],
                       video_id="v", clip_id=2, confidence=0.9),
        ]
        assert cluster_faces(faces, threshold=0.4) == [[0], [1]]
        # ...and at a pathological >1 threshold the legacy behaviour
        # (distance 1.0 < 1.5 -> union) is preserved exactly.
        assert cluster_faces(faces, threshold=1.5) == [[0, 1]]

    def test_within_clip_pairs_never_merge(self):
        emb = [1.0, 0.0, 0.0]
        faces = [
            FaceRecord(node_id="a", embedding=emb, confidence=0.9, bbox=[0, 0, 1, 1], video_id="v", clip_id=1),
            FaceRecord(node_id="b", embedding=emb, confidence=0.9, bbox=[0, 0, 1, 1], video_id="v", clip_id=1),
            FaceRecord(node_id="c", embedding=[0.0, 1.0, 0.0], confidence=0.9, bbox=[0, 0, 1, 1], video_id="v", clip_id=2),
        ]
        # a-b same clip: never merged directly. c is orthogonal to both.
        # The legacy loop produces the same partition.
        assert cluster_faces(faces, threshold=0.4) == [[0], [1], [2]]
        assert self._legacy_clusters(faces, 0.4) == [[0], [1], [2]]

    def test_single_linkage_bridges_through_cross_clip(self):
        """a-b are same-clip (masked), but both merge with c cross-clip:
        single linkage still joins all three (legacy-equivalent)."""
        emb = [1.0, 0.0, 0.0]
        faces = [
            FaceRecord(node_id="a", embedding=emb, confidence=0.9, bbox=[0, 0, 1, 1], video_id="v", clip_id=1),
            FaceRecord(node_id="b", embedding=emb, confidence=0.9, bbox=[0, 0, 1, 1], video_id="v", clip_id=1),
            FaceRecord(node_id="c", embedding=emb, confidence=0.9, bbox=[0, 0, 1, 1], video_id="v", clip_id=2),
        ]
        assert cluster_faces(faces, threshold=0.4) == [[0, 1, 2]]
        assert self._legacy_clusters(faces, 0.4) == [[0, 1, 2]]

    def test_missing_coordinates_are_never_masked(self):
        emb = [1.0, 0.0, 0.0]
        faces = [
            FaceRecord(node_id="a", embedding=emb, confidence=0.9, bbox=[0, 0, 1, 1], video_id=None, clip_id=None),
            FaceRecord(node_id="b", embedding=emb, confidence=0.9, bbox=[0, 0, 1, 1], video_id="v", clip_id=1),
        ]
        # a has no coordinates: the within-clip mask must not apply.
        assert cluster_faces(faces, threshold=0.4) == [[0, 1]]

    def test_dimension_mismatch_raises(self):
        faces = [
            FaceRecord(node_id="a", embedding=[1.0, 0.0], confidence=0.9, bbox=[0, 0, 1, 1], video_id="v", clip_id=1),
            FaceRecord(node_id="b", embedding=[1.0, 0.0, 0.0], confidence=0.9, bbox=[0, 0, 1, 1], video_id="v", clip_id=2),
        ]
        import pytest

        with pytest.raises(ValueError, match="embedding dim mismatch"):
            cluster_faces(faces)

    def test_block_boundary_matches_legacy(self):
        # >2048 faces (two GEMM blocks) with identical embeddings and
        # unique clips: every pair is cross-clip and within threshold,
        # so the whole set must merge into ONE cluster — pairs spanning
        # the block boundary are computed against all columns.
        faces = [
            FaceRecord(
                node_id=f"n{i}",
                embedding=[1.0, 0.0],
                confidence=0.9, bbox=[0, 0, 1, 1],
                video_id="v",
                clip_id=i,
            )
            for i in range(2100)
        ]
        clusters = cluster_faces(faces, threshold=0.4)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2100
