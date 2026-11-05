"""Tests for :mod:`m3.resume` — graph-aware resume helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from m3 import resume
from m3.episodic_memory import extracted_node_id


# ---------------------------------------------------------------------------
# video_fully_indexed — probe logic
# ---------------------------------------------------------------------------


class _FakeNode:
    def __init__(self, node_id: str, video_id: str | None = None,
                 extraction_status: str = "completed",
                 embedding: list[float] | None = None,
                 source_raw_id: str = ""):
        self.node_id = node_id
        self.metadata = {"video_id": video_id} if video_id is not None else {}
        self.extraction_status = extraction_status
        if embedding is not None:
            self.metadata["embedding"] = embedding
        # ExtractedNodes carry the parent RawNode's id; the smart-resume
        # probe (FIX-2) matches these against the current video's raw
        # node ids. RawNodes ignore this attribute.
        self.source_raw_id = source_raw_id


def _fake_graph(
    video_nodes: list[Any],
    extracted_nodes: dict[str, Any],
    *,
    list_extracted_nodes: list[Any] | None = None,
):
    """Build a mock GraphService that returns the given VIDEO nodes and ExtractedNodes.

    ``list_extracted_nodes`` controls what
    :func:`m3.pipeline.stages._probe_from_extracted_nodes` sees when
    called with ``ctx.graph=graph``. When ``None`` (default) we fall
    back to the values in ``extracted_nodes`` so a single fixture
    works for both ``video_fully_indexed`` (per-node lookup) and the
    probe (single list call).
    """

    def list_raw_nodes(raw_type, ctx, limit=1000, offset=0):
        return video_nodes[offset:offset + limit]

    def get_extracted_node(ext_id, ctx):
        return extracted_nodes.get(ext_id)

    if list_extracted_nodes is None:
        list_extracted_nodes = list(extracted_nodes.values())

    def list_extracted_nodes_fn(extractor_type, ctx, limit=4096, offset=0):
        return list_extracted_nodes[offset:offset + limit]

    graph = MagicMock()
    graph.list_raw_nodes.side_effect = list_raw_nodes
    graph.get_extracted_node.side_effect = get_extracted_node
    graph.list_extracted_nodes.side_effect = list_extracted_nodes_fn
    return graph


def test_video_fully_indexed_returns_complete_when_all_extracted():
    """All VIDEO clips have completed ExtractedNodes with embeddings → complete=True."""
    video_nodes = [
        _FakeNode("vnode_1", video_id="bedroom_03"),
        _FakeNode("vnode_2", video_id="bedroom_03"),
    ]
    extracted_nodes = {
        extracted_node_id("episodic_memory_generator", "vnode_1"): _FakeNode(
            "vnode_1", embedding=[0.1, 0.2, 0.3]
        ),
        extracted_node_id("episodic_memory_generator", "vnode_2"): _FakeNode(
            "vnode_2", embedding=[0.4, 0.5, 0.6]
        ),
    }
    graph = _fake_graph(video_nodes, extracted_nodes)
    result = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
    assert result["complete"] is True
    assert result["video_clip_count"] == 2
    assert result["missing_extracted"] == []
    assert result["unfinished_extracted"] == []
    assert result["missing_embedding"] == []


def test_video_fully_indexed_reports_missing_extracted():
    """ExtractedNode not in graph → reported as missing, complete=False."""
    video_nodes = [_FakeNode("vnode_1", video_id="bedroom_03")]
    graph = _fake_graph(video_nodes, extracted_nodes={})
    result = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
    assert result["complete"] is False
    assert result["video_clip_count"] == 1
    assert result["missing_extracted"] == [
        extracted_node_id("episodic_memory_generator", "vnode_1")
    ]


def test_video_fully_indexed_reports_unfinished_extractions():
    """Pending/failed ExtractedNodes are not reused → reported as unfinished."""
    video_nodes = [_FakeNode("vnode_1", video_id="bedroom_03")]
    extracted_nodes = {
        extracted_node_id("episodic_memory_generator", "vnode_1"): _FakeNode(
            "vnode_1", extraction_status="pending"
        ),
    }
    graph = _fake_graph(video_nodes, extracted_nodes)
    result = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
    assert result["complete"] is False
    assert len(result["unfinished_extracted"]) == 1


def test_video_fully_indexed_reports_missing_embedding():
    """Completed ExtractedNode without embedding → reported as missing_embedding."""
    video_nodes = [_FakeNode("vnode_1", video_id="bedroom_03")]
    extracted_nodes = {
        extracted_node_id("episodic_memory_generator", "vnode_1"): _FakeNode(
            "vnode_1", extraction_status="completed", embedding=None
        ),
    }
    graph = _fake_graph(video_nodes, extracted_nodes)
    result = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
    assert result["complete"] is False
    assert len(result["missing_embedding"]) == 1


def test_video_fully_indexed_no_video_nodes_returns_incomplete():
    """No VIDEO RawNodes for this video → complete=False, no crash."""
    graph = _fake_graph([], extracted_nodes={})
    result = resume.video_fully_indexed(graph, "m3bench", "missing_video")
    assert result["complete"] is False
    assert result["video_clip_count"] == 0


def test_video_fully_indexed_filters_other_videos():
    """VIDEO RawNodes from other videos must be ignored."""
    video_nodes = [
        _FakeNode("vnode_1", video_id="bedroom_03"),
        _FakeNode("vnode_2", video_id="living_room_06"),  # different video
    ]
    extracted_nodes = {
        extracted_node_id("episodic_memory_generator", "vnode_1"): _FakeNode(
            "vnode_1", embedding=[0.1]
        ),
    }
    graph = _fake_graph(video_nodes, extracted_nodes)
    result = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
    assert result["video_clip_count"] == 1
    assert result["complete"] is True


def test_video_fully_indexed_can_skip_embedding_requirement():
    """require_embeddings=False bypasses the embedding check."""
    video_nodes = [_FakeNode("vnode_1", video_id="bedroom_03")]
    extracted_nodes = {
        extracted_node_id("episodic_memory_generator", "vnode_1"): _FakeNode(
            "vnode_1", extraction_status="completed", embedding=None
        ),
    }
    graph = _fake_graph(video_nodes, extracted_nodes)
    result = resume.video_fully_indexed(
        graph, "m3bench", "bedroom_03", require_embeddings=False
    )
    assert result["complete"] is True


# ---------------------------------------------------------------------------
# Consolidation: video_fully_indexed and the smart-resume probes share
# the per-node classification rule via _classify_extracted_node.
# ---------------------------------------------------------------------------


class TestConsolidationWithSmartResume:
    """The smart-resume probe (Layer 2) and the legacy --reuse-graph
    probe (m3.resume.video_fully_indexed) must agree on whether a
    given graph state is "done". Both delegate the per-node check to
    :func:`m3.pipeline.stages._classify_extracted_node`; these tests
    pin that contract.
    """

    def test_classifier_is_None_when_completed_with_embedding(self):
        from m3.pipeline.stages import _classify_extracted_node
        ext = _FakeNode("v1", embedding=[0.1, 0.2])
        assert _classify_extracted_node(ext, require_embeddings=True) is None
        assert _classify_extracted_node(ext, require_embeddings=False) is None

    def test_classifier_returns_missing_when_ext_is_None(self):
        from m3.pipeline.stages import _classify_extracted_node
        assert _classify_extracted_node(None) == "missing"
        assert _classify_extracted_node(None, require_embeddings=True) == "missing"

    def test_classifier_returns_unfinished_when_status_pending(self):
        from m3.pipeline.stages import _classify_extracted_node
        ext = _FakeNode("v1", extraction_status="pending", embedding=[0.1])
        assert _classify_extracted_node(ext) == "unfinished"
        assert _classify_extracted_node(ext, require_embeddings=True) == "unfinished"

    def test_classifier_returns_missing_embedding_only_when_required(self):
        from m3.pipeline.stages import _classify_extracted_node
        ext = _FakeNode("v1", extraction_status="completed", embedding=None)
        # require_embeddings=False: completed node passes
        assert _classify_extracted_node(ext, require_embeddings=False) is None
        # require_embeddings=True: completed-but-no-embedding flags
        assert _classify_extracted_node(ext, require_embeddings=True) == "missing_embedding"

    def test_video_fully_indexed_and_probe_agree_on_complete_state(self):
        """When the graph has the same set of completed EpisodicMemoryGenerator
        ExtractedNodes (with embedding), ``video_fully_indexed`` and
        the smart-resume probe must both report done. This pins that
        the consolidation didn't drift the two judgements apart."""
        from m3.pipeline.stages import _probe_from_extracted_nodes

        # 5 VIDEO nodes, each with a completed + embedded ExtractedNode
        # whose ``source_raw_id`` is the matching VIDEO raw node id.
        video_nodes = [
            _FakeNode(f"vnode_{i}", video_id="bedroom_03")
            for i in range(5)
        ]
        extracted_nodes = {
            extracted_node_id(
                "episodic_memory_generator", f"vnode_{i}"
            ): _FakeNode(
                f"vnode_{i}", embedding=[0.1 * i],
                source_raw_id=f"vnode_{i}",
            )
            for i in range(5)
        }
        graph = _fake_graph(video_nodes, extracted_nodes)

        # Legacy --reuse-graph probe
        legacy = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
        assert legacy["complete"] is True

        # Smart-resume probe. We need a minimal PipelineContext that
        # exposes ctx.graph + ctx.account_id. The probe reads the graph
        # via ctx.graph and builds its own RequestContext from
        # account_id; everything else is unused for this assertion.
        from m3.pipeline.context import PipelineContext
        from m3.pipeline.spec import StageSpec
        ctx = PipelineContext(
            video=Path("/tmp/v.mp4"),
            video_id="bedroom_03",
            seconds=0,
            work_dir=Path("/tmp"),
            config_path=Path("/tmp/cfg.yaml"),
            graph_store="memory",
            connection_string=None,
            account_id="m3bench",
            artifacts={},
            variables={},
            graph=graph,
            op_service=None,
        )
        stage = StageSpec(
            id="episodic_memory",
            type="m3.episodic_memory", enabled=True,
            needs=[], params={}, outputs={},
        )
        # episodic_memory probe checks completion only
        ep_probe = _probe_from_extracted_nodes(
            ctx, "episodic_memory_generator",
        )
        # index_episodic probe additionally requires embeddings
        ix_probe = _probe_from_extracted_nodes(
            ctx, "episodic_memory_generator", require_embeddings=True,
        )
        assert ep_probe.complete is True
        assert ix_probe.complete is True

    def test_video_fully_indexed_and_probe_agree_on_incomplete_state(self):
        """Missing embedding must trigger BOTH probes — a regression here
        would mean --reuse-graph says "rerun QA" while smart resume
        says "rerun index_episodic", which is contradictory UX."""
        from m3.pipeline.context import PipelineContext
        from m3.pipeline.spec import StageSpec
        from m3.pipeline.stages import _probe_from_extracted_nodes

        # One completed-with-embedding + one completed-without-embedding.
        video_nodes = [
            _FakeNode("vnode_1", video_id="bedroom_03"),
            _FakeNode("vnode_2", video_id="bedroom_03"),
        ]
        extracted_nodes = {
            extracted_node_id("episodic_memory_generator", "vnode_1"): _FakeNode(
                "vnode_1", embedding=[0.1],
                source_raw_id="vnode_1",
            ),
            # vnode_2 has completed extraction but NO embedding
            extracted_node_id("episodic_memory_generator", "vnode_2"): _FakeNode(
                "vnode_2", extraction_status="completed", embedding=None,
                source_raw_id="vnode_2",
            ),
        }
        graph = _fake_graph(video_nodes, extracted_nodes)

        # --reuse-graph probe: missing_embedding should make complete=False
        legacy = resume.video_fully_indexed(graph, "m3bench", "bedroom_03")
        assert legacy["complete"] is False
        assert len(legacy["missing_embedding"]) == 1

        # Smart-resume probes: episodic_memory probe says complete (the
        # extraction IS done), but index_episodic probe says incomplete
        # (the embedding isn't stamped yet).
        ctx = PipelineContext(
            video=Path("/tmp/v.mp4"),
            video_id="bedroom_03",
            seconds=0,
            work_dir=Path("/tmp"),
            config_path=Path("/tmp/cfg.yaml"),
            graph_store="memory",
            connection_string=None,
            account_id="m3bench",
            artifacts={},
            variables={},
            graph=graph,
            op_service=None,
        )
        stage = StageSpec(
            id="episodic_memory",
            type="m3.episodic_memory", enabled=True,
            needs=[], params={}, outputs={},
        )
        ep_probe = _probe_from_extracted_nodes(
            ctx, "episodic_memory_generator",
        )
        ix_probe = _probe_from_extracted_nodes(
            ctx, "episodic_memory_generator", require_embeddings=True,
        )
        assert ep_probe.complete is True
        assert ix_probe.complete is False
        # Both probes + legacy call agree on the actionable answer:
        # QA cannot reuse the graph yet because index_episodic still
        # has work to do. --reuse-graph falls through, smart resume
        # re-runs index_episodic, and the video-clip finalisation
        # matches.
        assert ix_probe.complete is False and legacy["complete"] is False


# ---------------------------------------------------------------------------
# write_resume_summary
# ---------------------------------------------------------------------------


def test_write_resume_summary_creates_minimal_summary(tmp_path: Path):
    """write_resume_summary emits a summary JSON the single-video CLI can be discovered by."""
    jsonl = tmp_path / "m3_qa_results.jsonl"
    jsonl.write_text("", encoding="utf-8")
    summary_path = resume.write_resume_summary(
        work_dir=tmp_path,
        video_id="bedroom_03",
        pipeline_name="video_episodic_full",
        pipeline_version=1,
        jsonl_path=jsonl,
        duration_sec=12.3,
        rc=0,
    )
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["video_id"] == "bedroom_03"
    assert summary["mode"] == "qa_only"
    assert summary["stages"]["qa_only_resume"]["ok"] is True
    assert summary["artifacts"]["qa_results"] == str(jsonl)


# ---------------------------------------------------------------------------
# build_graph_service — cycle workaround
# ---------------------------------------------------------------------------


def test_build_graph_service_memory_returns_memory_store():
    """In-memory store path returns a GraphService backed by MemoryGraphStore."""
    from providers.graph_store import MemoryGraphStore
    from service.graph_service import GraphService

    graph, kind = resume.build_graph_service("memory", None)
    assert kind == "memory"
    assert isinstance(graph, GraphService)
    # Underlying store is MemoryGraphStore (or its subclass).
    store = graph._store  # type: ignore[attr-defined]
    assert isinstance(store, MemoryGraphStore)


def test_build_graph_service_sql_without_dsn_raises():
    """SQL store requested but no DSN → RuntimeError."""
    import os
    os.environ.pop("OGMEM_GRAPH_CONNECTION_STRING", None)
    with pytest.raises(RuntimeError, match="no DSN provided"):
        resume.build_graph_service("sql", None)


def test_build_graph_service_sql_with_connection_string(monkeypatch):
    """SQL store path returns a GraphService backed by SQLGraphStore.

    We patch ``SQLGraphStore`` so the test doesn't try to open a real
    database connection (the real constructor eagerly runs
    ``_ensure_schema``).
    """
    from service.graph_service import GraphService

    sentinel = MagicMock()
    monkeypatch.setattr(
        "providers.graph_store.SQLGraphStore",
        lambda connection_string, **kwargs: sentinel,
    )

    graph, kind = resume.build_graph_service(
        "sql", "postgresql://user:pw@host:5432/db"
    )
    assert kind == "sql"
    assert isinstance(graph, GraphService)
    assert graph._store is sentinel  # type: ignore[attr-defined]


def test_build_graph_service_sql_pool_size_env(monkeypatch):
    """OGMEM_GRAPH_POOL_SIZE sizes the SQLGraphStore connection pool.

    The pool must be >= the parallel-clip worker count or worker
    threads serialize on pool checkout.
    """
    store_cls = MagicMock()
    monkeypatch.setattr("providers.graph_store.SQLGraphStore", store_cls)
    dsn = "postgresql://user:pw@host:5432/db"

    monkeypatch.setenv("OGMEM_GRAPH_POOL_SIZE", "17")
    resume.build_graph_service("sql", dsn)
    assert store_cls.call_args.kwargs["pool_size"] == 17

    monkeypatch.delenv("OGMEM_GRAPH_POOL_SIZE")
    resume.build_graph_service("sql", dsn)
    assert store_cls.call_args.kwargs["pool_size"] == 32


def test_build_graph_service_unknown_kind_raises():
    """Unknown graph_store value → ValueError."""
    with pytest.raises(ValueError, match="Unknown graph store"):
        resume.build_graph_service("redis", None)