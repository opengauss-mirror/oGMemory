"""WS 5 — m3.wipe_stages: stage order, boundary math, SQL builders, post-ops.

DB-free: everything here exercises pure functions over STAGE_SPECS /
ORDERED_STAGES; the recursive CTE and deletes are validated by their SQL
text and bind parameters, never against a live connection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from m3 import wipe_stages as ws


# ---------------------------------------------------------------------------
# Stage order + specs
# ---------------------------------------------------------------------------


def test_stage_order_is_dependency_order():
    """Signal extraction precedes clustering precedes LLM stages precedes QA."""
    order = list(ws.ORDERED_STAGES)
    assert order.index("m3.extract_audio") < order.index("m3.extract_frames")
    assert order.index("m3.extract_frames") < order.index("m3.detect_faces")
    assert order.index("m3.detect_faces") < order.index("m3.face_clustering")
    assert order.index("m3.face_clustering") < order.index("m3.voice_segmentation")
    assert order.index("m3.voice_segmentation") < order.index("m3.voice_asr")
    assert order.index("m3.voice_asr") < order.index("m3.speaker_diarization")
    assert order.index("m3.speaker_diarization") < order.index("m3.speaker_embedding")
    assert order.index("m3.speaker_embedding") < order.index("m3.episodic_memory")
    assert order.index("m3.episodic_memory") < order.index("m3.memory.from_episodic")
    assert order.index("m3.memory.from_episodic") < order.index("m3.index_episodic")
    assert (
        order.index("m3.index_episodic")
        < order.index("m3.semantic_memory_consolidator")
        < order.index("m3.qa")
        < order.index("m3.eval")
    )
    # Every stage in the order has a spec and vice versa.
    assert set(ws.ORDERED_STAGES) == set(ws.STAGE_SPECS)


def test_specs_match_live_extractor_types():
    """The extractor/raw types in STAGE_SPECS are the ones the pipeline uses."""
    from m3.pipeline.stages import _EXTRACTOR_PARENT_RAW_TYPE

    known_extractors = set(_EXTRACTOR_PARENT_RAW_TYPE)
    specced = {
        e for s in ws.STAGE_SPECS.values() for e in s.extractor_types
    }
    assert specced <= known_extractors
    # The core M3 chain is fully covered.
    assert {
        "audio_extractor", "video_frame_extractor", "face_detection",
        "voice_segmentation", "voice_asr", "speaker_diarization",
        "speaker_embedding", "episodic_memory_generator",
    } <= specced


# ---------------------------------------------------------------------------
# Boundary math
# ---------------------------------------------------------------------------


def test_resolve_stage_accepts_short_and_full_ids():
    assert ws.resolve_stage("m3.episodic_memory") == "m3.episodic_memory"
    assert ws.resolve_stage("episodic_memory") == "m3.episodic_memory"
    with pytest.raises(ValueError, match="unknown stage"):
        ws.resolve_stage("nope")


def test_wipe_from_episodic_memory_covers_the_llm_side_only():
    stages = ws.stages_from("m3.episodic_memory")
    ids = [s.stage_id for s in stages]
    assert ids == [
        "m3.episodic_memory",
        "m3.memory.from_episodic",
        "m3.index_episodic",
        "m3.semantic_memory_consolidator",
        "m3.qa",
        "m3.eval",
    ]
    # Signal stages are NOT in range.
    assert "m3.extract_frames" not in ids
    assert "m3.detect_faces" not in ids


def test_wipe_from_extract_frames_is_a_full_media_rebuild():
    stages = ws.stages_from("m3.extract_frames")
    ids = [s.stage_id for s in stages]
    assert ids[0] == "m3.extract_frames"
    assert "m3.extract_audio" not in ids  # audio is before the boundary
    assert "m3.episodic_memory" in ids
    assert "m3.eval" in ids


def test_keep_until_complements_wipe_from():
    keep = ws.stages_after("m3.speaker_embedding")
    wipe = ws.stages_from("m3.voice_clustering")
    assert [s.stage_id for s in keep] == [s.stage_id for s in wipe]


def test_keep_until_last_stage_has_nothing_to_wipe():
    with pytest.raises(ValueError, match="nothing to wipe"):
        ws.stages_after("m3.eval")


# ---------------------------------------------------------------------------
# seed_where / plan_sql / delete_sql
# ---------------------------------------------------------------------------


def test_seed_where_episodic_boundary_seeds_extractors_and_memory():
    stages = ws.stages_from("m3.episodic_memory")
    where, params = ws.seed_where(stages)
    assert "extractor_type = ANY(%(extractor_types)s)" in where
    assert params["extractor_types"] == ["episodic_memory_generator"]
    assert "node_type = 'memory'" in where
    # No raw types in the LLM-side range.
    assert "raw_type" not in where


def test_seed_where_frames_boundary_seeds_raw_and_extractors():
    stages = ws.stages_from("m3.extract_frames")
    where, params = ws.seed_where(stages)
    assert "raw_type = ANY(%(raw_types)s)" in where
    assert "image" in params["raw_types"]
    assert "video_frame_extractor" in params["extractor_types"]
    assert "face_detection" in params["extractor_types"]


def test_seed_where_requires_owned_nodes():
    """A range owning only edge/file stages cannot seed a node wipe."""
    with pytest.raises(ValueError, match="owns no graph nodes"):
        ws.seed_where(ws.stages_from("m3.qa"))


def test_plan_and_delete_sql_share_the_recursive_cte():
    stages = ws.stages_from("m3.episodic_memory")
    plan, plan_params = ws.plan_sql(stages)
    delete, delete_params = ws.delete_sql(stages)
    for sql in (plan, delete):
        assert "WITH RECURSIVE doomed" in sql
        assert "JOIN doomed d ON c.source_raw_id = d.node_id" in sql
        # Account scoping on both seed and closure.
        assert sql.count("account_id = %(account_id)s") >= 2
    assert "DELETE FROM graph_nodes" in delete
    assert "DELETE" not in plan
    assert "COUNT(*)" in plan
    assert plan_params["account_id"] is None  # run() injects the real one
    assert delete_params["extractor_types"] == ["episodic_memory_generator"]


# ---------------------------------------------------------------------------
# post_ops
# ---------------------------------------------------------------------------


def test_post_ops_face_clustering_boundary_drops_face_edges_only():
    """boundary == face_clustering: faces survive, their clusters must not."""
    ops = ws.post_ops(ws.stages_from("m3.face_clustering"))
    names = [op["name"] for op in ops]
    assert "drop_same_identity_edges" in names
    edge_op = next(o for o in ops if o["name"] == "drop_same_identity_edges")
    # voice_clustering is also in the wiped range (after face_clustering),
    # so its speaker edges are dropped too; the *faces* themselves survive.
    assert edge_op["params"]["endpoint_types"] == [
        "face_detection", "speaker_embedding",
    ]
    # index_episodic is in range → embeddings cleared, vector rows need a flag.
    assert "clear_episodic_embeddings" in names
    assert "drop_vector_rows" not in names  # only with wipe_vector=True


def test_post_ops_voice_clustering_boundary_drops_speaker_edges():
    ops = ws.post_ops(ws.stages_from("m3.voice_clustering"))
    edge_op = next(o for o in ops if o["name"] == "drop_same_identity_edges")
    assert edge_op["params"]["endpoint_types"] == ["speaker_embedding"]


def test_post_ops_detect_faces_boundary_keeps_faces_but_drops_edges():
    """boundary ≤ face_clustering includes the clustering stage in range;
    when boundary is detect_faces the face nodes are doomed anyway — the
    edge delete is a harmless no-op next to the FK cascade."""
    ops = ws.post_ops(ws.stages_from("m3.detect_faces"))
    assert any(o["name"] == "drop_same_identity_edges" for o in ops)


def test_post_ops_episodic_boundary_has_no_edge_or_vector_ops():
    ops = ws.post_ops(ws.stages_from("m3.episodic_memory"))
    names = [op["name"] for op in ops]
    assert "drop_same_identity_edges" not in names
    assert "clear_episodic_embeddings" in names
    assert "drop_vector_rows" not in names


def test_post_ops_wipe_vector_gated():
    ops = ws.post_ops(
        ws.stages_from("m3.episodic_memory"), wipe_vector=True
    )
    vector_op = next(o for o in ops if o["name"] == "drop_vector_rows")
    assert "m3_episodic_vectors" in vector_op["sql"]
    assert "filters->>'account_id' = %(account_id)s" in vector_op["sql"]
    assert "uri LIKE 'm3://%%'" in vector_op["sql"]  # %% = psycopg2-escaped %


def test_post_ops_vector_table_overridable():
    ops = ws.post_ops(
        ws.stages_from("m3.episodic_memory"),
        vector_table="custom_vectors",
        wipe_vector=True,
    )
    vector_op = next(o for o in ops if o["name"] == "drop_vector_rows")
    assert "custom_vectors" in vector_op["sql"]


# ---------------------------------------------------------------------------
# find_qa_files
# ---------------------------------------------------------------------------


def test_find_qa_files_rglobs_artifacts(tmp_path):
    for rel in (
        "v1/m3_qa_results.jsonl",
        "_aggregate/all_m3_qa_results.jsonl",
        "_aggregate/run_report.json",
        "_aggregate/m3_eval_summary.json",
        "_aggregate/m3_eval_results.jsonl",
        "v1/frames/000001.jpg",          # not a QA artifact
        "v1/pipeline.log",               # not a QA artifact
    ):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")

    found = ws.find_qa_files(tmp_path)
    names = {p.name for p in found}
    assert names == {
        "m3_qa_results.jsonl",
        "all_m3_qa_results.jsonl",
        "run_report.json",
        "m3_eval_summary.json",
        "m3_eval_results.jsonl",
    }
    # Sorted → deterministic output.
    assert found == sorted(found)


# ---------------------------------------------------------------------------
# run() dry-run + account scoping
# ---------------------------------------------------------------------------


class _FakeCursor:
    rowcount = -1

    def __init__(self, rows):
        self._rows = rows
        self.executed: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._rows


def test_run_dry_run_never_executes_delete():
    stages = ws.stages_from("m3.episodic_memory")
    conn = type("C", (), {})()
    conn.cursor = lambda: _FakeCursor(
        [("extracted", "episodic_memory_generator", None, 12),
         ("memory", None, None, 5)]
    )
    report = ws.run(conn, account="m3bench", stages=stages, execute=False)

    assert report["execute"] is False
    assert report["doomed_total"] == 17
    assert "deleted_nodes" not in report
    assert report["boundary_stage"] == "m3.episodic_memory"
    assert report["wiped_stages"][0] == "m3.episodic_memory"


def test_run_requires_account():
    with pytest.raises(ValueError, match="mandatory"):
        ws.run(
            type("C", (), {})(),
            account="",
            stages=ws.stages_from("m3.episodic_memory"),
        )


def test_run_execute_commits_account_scoped_delete():
    stages = ws.stages_from("m3.episodic_memory")
    committed = []
    cur = _FakeCursor([("extracted", "episodic_memory_generator", None, 12)])
    conn = type("C", (), {})()
    conn.cursor = lambda: cur
    conn.commit = lambda: committed.append(True)
    report = ws.run(conn, account="m3bench", stages=stages, execute=True)

    assert report["deleted_nodes"] == -1  # the fake cursor's rowcount
    assert committed  # commit happened


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_requires_boundary():
    with pytest.raises(SystemExit):
        ws.parse_args(["--account", "m3bench"])


def test_cli_list_stages_works_without_account(capsys):
    # --list-stages is a help/inspection path: no account or DSN needed.
    assert ws.main(["--list-stages"]) == 0
    out = capsys.readouterr().out
    assert "m3.extract_audio" in out
    assert "m3.episodic_memory" in out


def test_cli_requires_account_without_list_stages():
    rc = ws.main(["--wipe-from", "m3.episodic_memory"])
    assert rc == 2


def test_cli_list_stages_prints_order(capsys):
    assert ws.main(["--list-stages"]) == 0
    out = capsys.readouterr().out
    assert "m3.extract_audio" in out
    assert "m3.episodic_memory" in out


def test_cli_wipe_from_unknown_stage_fails():
    rc = ws.main(["--account", "a", "--wipe-from", "nope",
                  "--connection-string", "postgresql://x"])
    assert rc == 2


def test_cli_requires_connection_string(monkeypatch):
    monkeypatch.delenv("OGMEM_GRAPH_CONNECTION_STRING", raising=False)
    rc = ws.main(["--account", "a", "--wipe-from", "m3.episodic_memory"])
    assert rc == 2
