"""Tests for :mod:`m3.detect_faces` driver.

The driver itself is now a thin wrapper around ``OperatorService`` +
``FaceDetectionExtractor`` materialization (kind='extracted'). Each
detected face becomes an ``ExtractedNode`` carrying bbox / confidence /
embedding / detector; the parent ``ExtractedNode`` carries the
lightweight summary. The interesting behaviour (extracted-child
creation, EXTRACTED_FROM auto-edge, lightweight summary, idempotency)
lives at the operator / materialization layer — see
``tests/unit/extraction/operators/test_materialize.py`` for the bulk
of coverage. These tests verify the M3-specific glue:

* the ``faces.jsonl`` manifest format the m3 tooling expects,
* idempotent re-run via the parent ``ExtractedNode`` lookup,
* backend selection (mock vs HTTP) at the CLI level,
* graceful failure when an operator error happens.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.graph_enums import RawDataType  # noqa: E402
from core.models import RequestContext  # noqa: E402
from m3.config import M3BENCH_ACCOUNT_ID  # noqa: E402
from m3.detect_faces import (  # noqa: E402
    detect_faces_for_frame,
    extracted_node_id,
    parse_args,
    run,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="m3bench-detect-faces-test", trace_id="t-test",
    )


def _make_memory_graph():
    from providers.graph_store import MemoryGraphStore
    from service.graph_service import GraphService
    return GraphService(MemoryGraphStore())


def _make_op_service(graph):
    from extraction.operators import default_registry
    from extraction.operators.builtin import register_all_defaults
    from service.operator_service import OperatorService
    registry = default_registry()
    register_all_defaults(registry)
    return OperatorService(graph, registry=registry)


def _persist_image_raw_node(graph, video_id, clip_id, frame_idx, jpeg_bytes, ctx):
    """Create + persist a fake IMAGE RawNode for a frame."""
    seed = f"frame/{video_id}/{clip_id}/{frame_idx}".encode()
    node_id = hashlib.sha1(seed).hexdigest()[:32]
    graph.create_raw_node(
        raw_type=RawDataType.IMAGE,
        content=jpeg_bytes,
        ctx=ctx,
        metadata={
            "video_id": video_id,
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "ts_sec": frame_idx * 0.2,
            "keyframe": True,
            "width": 160,
            "height": 120,
            "source_path": f"/tmp/{video_id}/{clip_id}/{frame_idx}.jpg",
        },
        node_id=node_id,
    )
    return node_id


def _write_frames_manifest(manifest_path: Path, records: list[dict]) -> None:
    """Persist a frames.jsonl-shaped manifest that detect_faces reads."""
    manifest_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# ID derivation (manifest compatibility)
# ---------------------------------------------------------------------------

class TestExtractedNodeId:
    def test_is_deterministic(self):
        a = extracted_node_id("face_detection", "frame-xyz")
        b = extracted_node_id("face_detection", "frame-xyz")
        assert a == b
        assert len(a) == 32

    def test_varies_with_extractor(self):
        a = extracted_node_id("face_detection", "frame-1")
        b = extracted_node_id("object_detection", "frame-1")
        assert a != b

    def test_varies_with_source(self):
        a = extracted_node_id("face_detection", "frame-1")
        b = extracted_node_id("face_detection", "frame-2")
        assert a != b


# ---------------------------------------------------------------------------
# Driver integration via detect_faces_for_frame
# ---------------------------------------------------------------------------

class TestDetectFacesForFrame:
    """End-to-end driver test using the in-process MockBackend."""

    def _setup(self, tmp_path, *, frame_count=2):
        graph = _make_memory_graph()
        op_service = _make_op_service(graph)
        ctx = _ctx()
        jpeg = b"\xff\xd8\xff\xe0fake-jpeg"  # bytes don't matter; mock backend
        frame_ids = [
            _persist_image_raw_node(graph, "vid_a", 0, i, jpeg, ctx)
            for i in range(frame_count)
        ]
        return graph, op_service, ctx, frame_ids

    def test_creates_face_extracted_nodes_per_frame(self, tmp_path):
        from extraction.operators.backends import MockBackend

        graph, op_service, ctx, frame_ids = self._setup(tmp_path, frame_count=2)
        backend = MockBackend(face_count=2)  # 2 faces per frame

        results = [
            detect_faces_for_frame(
                graph=graph, op_service=op_service, ctx=ctx,
                frame_raw_node=graph.get_raw_node(fid, ctx),
                backend=backend,
            )
            for fid in frame_ids
        ]

        for r in results:
            assert r["face_count"] == 2
            assert r["resumed"] is False
            assert len(r["child_node_ids"]) == 2
            for face_id in r["child_node_ids"]:
                face_node = graph.get_extracted_node(face_id, ctx)
                assert face_node is not None
                assert face_node.extractor_type == "face_detection"
                assert face_node.extraction_schema == "face_detection_v1"
                # bbox + confidence propagated to extracted_content
                assert "bbox" in face_node.extracted_content
                assert "confidence" in face_node.extracted_content

    def test_parent_extracted_node_has_lightweight_summary(self, tmp_path):
        from extraction.operators.backends import MockBackend

        graph, op_service, ctx, frame_ids = self._setup(tmp_path, frame_count=1)
        backend = MockBackend(face_count=3)

        result = detect_faces_for_frame(
            graph=graph, op_service=op_service, ctx=ctx,
            frame_raw_node=graph.get_raw_node(frame_ids[0], ctx),
            backend=backend,
        )

        parent = graph.get_extracted_node(result["extracted_node_id"], ctx)
        assert parent is not None
        content = parent.extracted_content
        # Heavy per-item collection is stripped; only IDs + count kept.
        assert "faces" not in content
        assert content["child_node_ids"] == result["child_node_ids"]
        assert content["child_count"] == 3

    def test_resumed_when_extracted_node_exists(self, tmp_path):
        from extraction.operators.backends import MockBackend

        graph, op_service, ctx, frame_ids = self._setup(tmp_path, frame_count=1)
        backend = MockBackend(face_count=2)

        first = detect_faces_for_frame(
            graph=graph, op_service=op_service, ctx=ctx,
            frame_raw_node=graph.get_raw_node(frame_ids[0], ctx),
            backend=backend,
        )
        assert first["resumed"] is False

        second = detect_faces_for_frame(
            graph=graph, op_service=op_service, ctx=ctx,
            frame_raw_node=graph.get_raw_node(frame_ids[0], ctx),
            backend=backend,
        )
        assert second["resumed"] is True
        assert second["child_node_ids"] == first["child_node_ids"]

    def test_min_confidence_filter_propagates_to_operator(self, tmp_path):
        from extraction.operators.backends import MockBackend

        graph, op_service, ctx, frame_ids = self._setup(tmp_path, frame_count=1)
        backend = MockBackend(face_count=3, default_confidence=0.5)

        result = detect_faces_for_frame(
            graph=graph, op_service=op_service, ctx=ctx,
            frame_raw_node=graph.get_raw_node(frame_ids[0], ctx),
            backend=backend,
            min_confidence=0.9,
        )
        # min_confidence=0.9 + default_confidence=0.5 -> 0 faces kept
        assert result["face_count"] == 0
        assert result["child_node_ids"] == []


# ---------------------------------------------------------------------------
# Driver entry point: run() with manifest
# ---------------------------------------------------------------------------

class TestRunDriver:
    """Smoke-test ``run()`` end-to-end with a synthetic frames.jsonl."""

    def _make_frames(self, tmp_path, *, frame_count=2):
        """Seed graph with IMAGE RawNodes and write a frames.jsonl manifest."""
        graph = _make_memory_graph()
        ctx = _ctx()
        jpeg = b"\xff\xd8\xff\xe0fake-jpeg"
        records = []
        for i in range(frame_count):
            frame_id = _persist_image_raw_node(graph, "vid_a", 0, i, jpeg, ctx)
            records.append({
                "video_id": "vid_a",
                "clip_id": 0,
                "video_raw_node_id": "src-vid_a-0",
                "extracted_node_id": f"fake-ext-{i}",
                "child_node_ids": [frame_id],
                "child_count": 1,
                "resumed": False,
            })
        manifest_path = tmp_path / "frames.jsonl"
        _write_frames_manifest(manifest_path, records)
        return graph, records, manifest_path

    def test_run_writes_faces_manifest_jsonl(self, tmp_path):

        graph, records, manifest_path = self._make_frames(tmp_path, frame_count=2)
        op_service = _make_op_service(graph)

        out_jsonl = tmp_path / "faces.jsonl"
        args = parse_args([
            "--frames-manifest", str(manifest_path),
            "--output-jsonl", str(out_jsonl),
            "--graph-store", "memory",
            "--backend", "mock",
        ])
        rc = run(args, graph=graph, op_service=op_service)
        assert rc == 0, f"run() returned {rc}"

        assert out_jsonl.exists()
        out_records = [
            json.loads(line) for line in out_jsonl.read_text().splitlines() if line
        ]
        # One record per frame (we had 2 frames).
        assert len(out_records) == 2
        for rec in out_records:
            assert rec["video_id"] == "vid_a"
            assert rec["clip_id"] == 0
            assert rec["frame_node_id"]
            assert rec["extracted_node_id"]
            # MockBackend default is 1 face per image.
            assert rec["face_count"] == 1
            assert rec["resumed"] is False

    def test_run_resumed_on_second_pass(self, tmp_path):

        graph, records, manifest_path = self._make_frames(tmp_path, frame_count=1)
        op_service = _make_op_service(graph)

        out_jsonl = tmp_path / "faces.jsonl"
        args = parse_args([
            "--frames-manifest", str(manifest_path),
            "--output-jsonl", str(out_jsonl),
            "--backend", "mock",
        ])

        # First pass: resumed=False
        run(args, graph=graph, op_service=op_service)
        # Second pass: resumed=True because parent ExtractedNodes exist
        run(args, graph=graph, op_service=op_service)
        records = [
            json.loads(line) for line in out_jsonl.read_text().splitlines() if line
        ]
        # Output is overwritten each run, so we only see the second
        # pass: one record with resumed=True.
        assert len(records) == 1
        assert records[0]["resumed"] is True

    def test_missing_manifest_returns_error_code(self, tmp_path):
        args = parse_args([
            "--frames-manifest", str(tmp_path / "nonexistent.jsonl"),
            "--output-jsonl", str(tmp_path / "out.jsonl"),
            "--backend", "mock",
        ])
        rc = run(args)
        assert rc == 1

    def test_unknown_backend_choice_rejected(self, tmp_path):
        # argparse rejects unknown --backend values.
        with pytest.raises(SystemExit):
            parse_args([
                "--frames-manifest", str(tmp_path / "x.jsonl"),
                "--backend", "magic",
            ])


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        args = parse_args(["--frames-manifest", "x.jsonl"])
        assert args.backend == "http"
        assert args.cv_timeout is None  # Resolved from YAML, with a 30-second fallback.
        assert args.min_confidence == 0.5
        assert args.max_faces == 20
        assert args.graph_store == "memory"

    def test_overrides(self):
        args = parse_args([
            "--frames-manifest", "x.jsonl",
            "--backend", "mock",
            "--min-confidence", "0.7",
            "--max-faces", "5",
            "--cv-service-url", "http://example.com:9000",
            "--cv-auth-token", "secret",
            "--cv-timeout", "60",
            "--video", "v1,v2",
            "-v",
        ])
        assert args.backend == "mock"
        assert args.min_confidence == 0.7
        assert args.max_faces == 5
        assert args.cv_service_url == "http://example.com:9000"
        assert args.cv_auth_token == "secret"
        assert args.cv_timeout == 60.0
        assert args.video == "v1,v2"
        assert args.verbose is True
