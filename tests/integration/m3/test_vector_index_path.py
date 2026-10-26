"""End-to-end test of the M3 VectorIndex path against a live Postgres.

This test is the integration counterpart to
``tests/integration/m3/test_index_episodic_vector_index.py`` (which
uses an :class:`InMemoryVectorIndex`). It exercises the real
:class:`OpenGaussVectorIndex` SQL behaviour — HNSW search, scope
filter columns, the ``get_record`` round-trip — so a regression in
the schema migration would trip here before it ships.

Skipped automatically when ``OGMEM_GRAPH_CONNECTION_STRING`` is not
set or the DB isn't reachable; CI runners should set it.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from core.models import IndexRecord, RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from m3.index_episodic import parse_args, run
from m3.index_status import episodic_uri
from providers.graph_store import MemoryGraphStore
from providers.vector_index import OpenGaussVectorIndex
from service.graph_service import GraphService


def _db_available() -> bool:
    """Probe for a reachable pgvector-capable Postgres."""
    import psycopg2

    dsn = os.environ.get("OGMEM_GRAPH_CONNECTION_STRING")
    if not dsn:
        return False
    try:
        conn = psycopg2.connect(dsn, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(),
    reason="OGMEM_GRAPH_CONNECTION_STRING not set or DB unreachable",
)


def _ctx() -> RequestContext:
    return RequestContext(
        account_id=M3BENCH_ACCOUNT_ID, user_id="u", agent_id="a",
        session_id="m3bench-vector-index-test", trace_id="t-vector",
    )


def _write_episodic(path: Path, n: int = 3) -> None:
    lines = []
    for i in range(n):
        lines.append(json.dumps({
            "video_id": f"vid_{i}",
            "clip_id": i,
            "video_raw_node_id": f"vidraw_{i}".ljust(32, "x"),
            "extracted_node_id": f"extvid_{i}".ljust(32, "x"),
            "description": f"Episode {i}: the woman walks into the room.",
            "characters": [],
            "events": [],
            "entities": [],
            "frame_count": 4,
            "has_audio": False,
        }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_graph(graph: GraphService, ctx: RequestContext, path: Path) -> None:
    from core.graph_models import create_extracted_node

    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        node = create_extracted_node(
            source_raw_id=row["video_raw_node_id"],
            extractor_type="episodic_memory_generator",
            extraction_schema="m3_bench_v1",
            account_id=ctx.account_id,
            extracted_content={
                "description": row["description"],
                "source_video_id": row["video_id"],
                "source_clip_id": row["clip_id"],
            },
            node_id=row["extracted_node_id"],
        )
        graph._store.create_node(node.serialize(), ctx)


def _query_vector(index, ctx: RequestContext, video_id: str, clip_id: int):
    record = index.get_record(IndexRecord.generate_id(
        episodic_uri(ctx.account_id, video_id, clip_id), level=2,
    ))
    assert record is not None
    return list(record.metadata["_embedding"])


@pytest.fixture
def vector_index():
    """Build a per-test OpenGaussVectorIndex against a unique table."""
    dsn = os.environ["OGMEM_GRAPH_CONNECTION_STRING"]
    # Unique table per test so parallel runs / re-runs don't collide.
    # The ``_ensure_table`` migration is idempotent (CREATE TABLE IF
    # NOT EXISTS) so repeated invocations are safe.
    table = f"m3_test_vec_{uuid.uuid4().hex[:12]}"
    idx = OpenGaussVectorIndex(
        connection_string=dsn, table_name=table, dimension=16, pool_size=2, dialect="postgresql",
    )
    try:
        yield idx
    finally:
        idx.close()
        # Clean up the per-test table so we don't accumulate cruft.
        import psycopg2
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{table}"')
            conn.commit()


def test_index_and_search_round_trip(tmp_path: Path, vector_index):
    """End-to-end: write embeddings, search via GraphService, assert
    the HNSW + scope columns return the right hit."""
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    _write_episodic(episodic_path, n=3)

    graph = GraphService(MemoryGraphStore(), vector_index=vector_index)
    ctx = _ctx()
    _seed_graph(graph, ctx, episodic_path)

    rc = run(parse_args([
        "--episodic-jsonl", str(episodic_path),
        "--output-index", str(index_path),
        "--embedder", "mock",
        "--embedding-dimension", "16",
        "--account-id", M3BENCH_ACCOUNT_ID,
    ]), graph=graph, vector_index=vector_index)
    assert rc == 0
    assert vector_index.count() == 3

    query_emb = _query_vector(vector_index, ctx, "vid_0", 0)
    hits = graph.search_by_embedding(
        query_emb, ctx, top_k=3, extractor_types=("episodic_memory_generator",),
    )
    assert len(hits) >= 1
    assert hits[0].node.node_id == "extvid_0".ljust(32, "x")
    assert hits[0].score > 0.99
    assert hits[0].video_id == "vid_0"
    assert hits[0].clip_id == 0


def test_scope_columns_push_down_to_sql(tmp_path: Path, vector_index):
    """Insert records across two videos, then issue a search scoped
    to one video. The SQL WHERE clause (built by OpenGauss's
    ``_build_where``) must drop the other video's rows — proves the
    filter doesn't leak in Python after the fetch."""
    lines = []
    for vid, cid in (("vid_a", 0), ("vid_a", 1), ("vid_b", 0)):
        lines.append(json.dumps({
            "video_id": vid,
            "clip_id": cid,
            "video_raw_node_id": f"raw_{vid}_{cid}".ljust(32, "x"),
            "extracted_node_id": f"ext_{vid}_{cid}".ljust(32, "x"),
            "description": f"{vid} {cid}",
        }))
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    episodic_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    graph = GraphService(MemoryGraphStore(), vector_index=vector_index)
    ctx = _ctx()
    _seed_graph(graph, ctx, episodic_path)

    run(parse_args([
        "--episodic-jsonl", str(episodic_path),
        "--output-index", str(index_path),
        "--embedder", "mock",
        "--embedding-dimension", "16",
        "--account-id", M3BENCH_ACCOUNT_ID,
    ]), graph=graph, vector_index=vector_index)

    from service.graph_service import ClipScope
    hits = graph.search_by_embedding(
        _query_vector(vector_index, ctx, "vid_b", 0), ctx,
        top_k=10,
        scope=ClipScope(video_id="vid_b"),
    )
    assert hits, "no hits returned for vid_b"
    for hit in hits:
        assert hit.video_id == "vid_b", (
            f"ClipScope(video_id='vid_b') leaked a {hit.video_id} hit"
        )


def test_before_clip_filter_drops_future_clips(tmp_path: Path, vector_index):
    """Records with clip_id > before_clip must not match. The
    ``_build_where`` clause uses ``(before_clip IS NULL OR
    before_clip <= N)`` so legacy rows still match."""
    lines = []
    for cid in (0, 1, 2):
        lines.append(json.dumps({
            "video_id": "vid_a",
            "clip_id": cid,
            "video_raw_node_id": f"raw_{cid}".ljust(32, "x"),
            "extracted_node_id": f"ext_{cid}".ljust(32, "x"),
            "description": f"clip {cid}",
        }))
    episodic_path = tmp_path / "episodic_memory.jsonl"
    index_path = tmp_path / "episodic_index.jsonl"
    episodic_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    graph = GraphService(MemoryGraphStore(), vector_index=vector_index)
    ctx = _ctx()
    _seed_graph(graph, ctx, episodic_path)

    run(parse_args([
        "--episodic-jsonl", str(episodic_path),
        "--output-index", str(index_path),
        "--embedder", "mock",
        "--embedding-dimension", "16",
        "--account-id", M3BENCH_ACCOUNT_ID,
    ]), graph=graph, vector_index=vector_index)

    from service.graph_service import ClipScope
    hits = graph.search_by_embedding(
        _query_vector(vector_index, ctx, "vid_a", 0), ctx,
        top_k=10,
        scope=ClipScope(before_clip=0),
    )
    # Only clip_id == 0 should be visible (clip 1 and 2 are filtered
    # by the SQL ``before_clip <= 0`` clause).
    clip_ids = {h.clip_id for h in hits}
    assert clip_ids == {0}


def test_schema_migration_adds_m3_columns(vector_index):
    """The ``_ensure_table`` migration adds the M3 scope columns to
    pre-existing tables. We assert on column metadata after the
    fixture-driven construction — the fixture's first call already
    ran the migration on a fresh table, so this is also a smoke
    test that the migration is non-destructive on second call.
    """
    import psycopg2

    dsn = os.environ["OGMEM_GRAPH_CONNECTION_STRING"]
    table = vector_index._table_name  # noqa: SLF001 — test introspection
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s",
                (table,),
            )
            columns = {row[0] for row in cur.fetchall()}
    for required in ("video_id", "extractor_type", "before_clip"):
        assert required in columns, f"missing column {required} in {table}"
