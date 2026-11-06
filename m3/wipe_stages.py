"""Stage-bounded account wipe for the M3 graph (WS 5).

Rebuild only the LLM side of the M3 graph without redoing the deterministic
signal-extraction stages (frames, faces, audio, ASR, diarization). The
constructive counterpart of "use a fresh account": wipe stages at/after a
boundary in the same namespace, keep everything before it.

Design
------
``ORDERED_STAGES`` is the *dependency* order (not YAML order). Wiping stage
X wipes X and everything after it. ``STAGE_SPECS`` maps every M3 stage to
what it owns in the store (``graph_nodes.extractor_type`` / ``raw_type`` /
``node_type`` it produces). The node delete is a recursive CTE over
``graph_nodes`` following ``source_raw_id`` (extracted → raw parent);
``graph_edges`` rows cascade via their ``ON DELETE CASCADE`` foreign keys.

Memory nodes carry no ``source_raw_id`` (they hang off ExtractedNodes via
``supporting_node_ids``), so they are *seeded* directly whenever a wiped
stage owns memory nodes — the recursive walk cannot reach them.

``post_ops`` runs after the node delete:
  * ``same_identity`` edges between *surviving* nodes are dropped when the
    clustering stage that produced them is in the wiped range (the faces /
    speaker embeddings survive, their clusters must not);
  * the ``metadata.embedding`` key is cleared from surviving episodic nodes
    when ``index_episodic`` is in range (so smart-resume re-indexes);
  * vector rows are dropped from the vector table when ``--wipe-vector``.

Every write is limited to one ``account_id`` (mandatory).

Common boundaries::

    # Re-prompt the LLM-side graph, reuse all signal stages:
    python -m m3.wipe_stages --account m3bench \
        --connection-string "$DSN" --wipe-from m3.episodic_memory --yes

    # New voice model — keep frames/faces, redo everything from audio on:
    ... --wipe-from m3.voice_segmentation --yes

    # Only same_identity face clusters:
    ... --wipe-from m3.face_clustering --yes

    # Full media-to-graph rebuild (keeps the loaded VIDEO raw nodes):
    ... --wipe-from m3.extract_frames --yes

Dry-run is the default: ``plan_sql`` prints the doomed-node counts by kind.
``--yes`` executes the delete.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage vocabulary — dependency order, NOT pipeline YAML order.
# ---------------------------------------------------------------------------

ORDERED_STAGES: tuple[str, ...] = (
    "m3.extract_audio",
    "m3.extract_frames",
    "m3.detect_faces",
    "m3.face_clustering",
    "m3.voice_segmentation",
    "m3.voice_asr",
    "m3.speaker_diarization",
    "m3.speaker_embedding",
    "m3.voice_clustering",
    "m3.episodic_memory",
    "m3.memory.from_episodic",
    "m3.index_episodic",
    "m3.semantic_memory_consolidator",
    "m3.qa",
    "m3.eval",
)

_VECTOR_TABLE_DEFAULT = "m3_episodic_vectors"


@dataclass(frozen=True)
class StageWipe:
    """What one M3 stage owns in the graph/vector/file stores."""

    stage_id: str
    # ExtractedNodes produced by the stage (graph_nodes.extractor_type).
    extractor_types: tuple[str, ...] = ()
    # RawNodes produced by the stage (graph_nodes.raw_type).
    raw_types: tuple[str, ...] = ()
    # The stage creates MemoryNodes (node_type='memory'). The value is
    # the producer selector, because two stages create memory nodes and a
    # bare node_type='memory' seed cannot tell them apart (review item 2):
    #   "episodic"  — m3.memory.from_episodic per-clip memories
    #                 (tags contain 'episodic', topic 'm3:<video>')
    #   "semantic"  — m3.semantic_memory_consolidator evolved memories
    #                 (topic 'semantic:<video>')
    memory_nodes: str = ""
    # Endpoint extractor types whose same_identity edges this stage owns;
    # empty = no clustering. (faces for face_clustering, speakers for
    # voice_clustering.)
    same_identity_endpoints: tuple[str, ...] = ()
    # The stage populates the vector index / episodic node embeddings.
    vector_index: bool = False
    # The stage writes QA/eval result files.
    qa_files: bool = False
    notes: str = ""


STAGE_SPECS: dict[str, StageWipe] = {
    "m3.extract_audio": StageWipe(
        stage_id="m3.extract_audio",
        extractor_types=("audio_extractor",),
        raw_types=("audio",),
    ),
    "m3.extract_frames": StageWipe(
        stage_id="m3.extract_frames",
        extractor_types=("video_frame_extractor",),
        raw_types=("image",),
    ),
    "m3.detect_faces": StageWipe(
        stage_id="m3.detect_faces",
        extractor_types=("face_detection",),
    ),
    "m3.face_clustering": StageWipe(
        stage_id="m3.face_clustering",
        same_identity_endpoints=("face_detection",),
        notes="emits same_identity edges between face nodes",
    ),
    "m3.voice_segmentation": StageWipe(
        stage_id="m3.voice_segmentation",
        extractor_types=("voice_segmentation",),
    ),
    "m3.voice_asr": StageWipe(
        stage_id="m3.voice_asr",
        extractor_types=("voice_asr",),
    ),
    "m3.speaker_diarization": StageWipe(
        stage_id="m3.speaker_diarization",
        extractor_types=("speaker_diarization",),
    ),
    "m3.speaker_embedding": StageWipe(
        stage_id="m3.speaker_embedding",
        extractor_types=("speaker_embedding",),
    ),
    "m3.voice_clustering": StageWipe(
        stage_id="m3.voice_clustering",
        same_identity_endpoints=("speaker_embedding",),
        notes="emits same_identity edges between speaker nodes",
    ),
    "m3.episodic_memory": StageWipe(
        stage_id="m3.episodic_memory",
        extractor_types=("episodic_memory_generator",),
    ),
    "m3.memory.from_episodic": StageWipe(
        stage_id="m3.memory.from_episodic",
        memory_nodes="episodic",
    ),
    "m3.index_episodic": StageWipe(
        stage_id="m3.index_episodic",
        vector_index=True,
        notes="populates metadata.embedding + vector rows",
    ),
    "m3.semantic_memory_consolidator": StageWipe(
        stage_id="m3.semantic_memory_consolidator",
        memory_nodes="semantic",
        notes="evolved memories + evolves edges",
    ),
    "m3.qa": StageWipe(
        stage_id="m3.qa",
        qa_files=True,
    ),
    "m3.eval": StageWipe(
        stage_id="m3.eval",
        qa_files=True,
    ),
}


# ---------------------------------------------------------------------------
# Boundary math
# ---------------------------------------------------------------------------


def resolve_stage(name: str) -> str:
    """Accept ``m3.episodic_memory`` or ``episodic_memory``; validate."""
    if name in STAGE_SPECS:
        return name
    prefixed = f"m3.{name}"
    if prefixed in STAGE_SPECS:
        return prefixed
    known = ", ".join(ORDERED_STAGES)
    raise ValueError(f"unknown stage {name!r}; known stages: {known}")


def stages_from(boundary: str) -> list[StageWipe]:
    """Wiped stages for ``--wipe-from``: the boundary stage and everything after."""
    boundary = resolve_stage(boundary)
    idx = ORDERED_STAGES.index(boundary)
    return [STAGE_SPECS[s] for s in ORDERED_STAGES[idx:]]


def stages_after(keep_until: str) -> list[StageWipe]:
    """Wiped stages for ``--keep-until``: everything strictly AFTER the stage.

    ``--keep-until X`` keeps X and everything before it.
    """
    keep_until = resolve_stage(keep_until)
    idx = ORDERED_STAGES.index(keep_until)
    if idx + 1 >= len(ORDERED_STAGES):
        raise ValueError(
            f"--keep-until {keep_until} is the last stage; nothing to wipe"
        )
    return [STAGE_SPECS[s] for s in ORDERED_STAGES[idx + 1:]]


# ---------------------------------------------------------------------------
# SQL builders
# ---------------------------------------------------------------------------


def seed_where(stages: list[StageWipe]) -> tuple[str, dict[str, Any]]:
    """SQL predicate selecting the *direct* seeds of the wipe.

    Root URIs of the doomed closure: everything a wiped stage produces —
    ExtractedNodes by ``extractor_type``, RawNodes by ``raw_type``, and
    MemoryNodes (which dangle from ``supporting_node_ids`` and would
    otherwise be unreachable for the source_raw_id walk).
    """
    clauses: list[str] = []
    params: dict[str, Any] = {}
    extractors = sorted({e for s in stages for e in s.extractor_types})
    raws = sorted({r for s in stages for r in s.raw_types})
    if extractors:
        clauses.append("extractor_type = ANY(%(extractor_types)s)")
        params["extractor_types"] = extractors
    if raws:
        clauses.append("raw_type = ANY(%(raw_types)s)")
        params["raw_types"] = raws
    memory_selectors = {s.memory_nodes for s in stages if s.memory_nodes}
    if memory_selectors:
        if len(memory_selectors) > 1:
            # Wipe range covers every memory producer: a bare node_type
            # predicate is correct and cheapest.
            clauses.append("node_type = 'memory'")
        elif "episodic" in memory_selectors:
            # Only m3.memory.from_episodic is in range: seed its per-clip
            # memories, NOT the semantic consolidator's evolved memories
            # (and vice versa) — a bare node_type='memory' seed would doom
            # memories produced by stages the range promises to keep.
            clauses.append(
                "(node_type = 'memory' AND 'episodic' = ANY(tags))"
            )
        else:  # "semantic"
            # %% — escaped for psycopg2 (the seed SQL executes with named
            # params), same as the vector-wipe uri LIKE 'm3://%%' below.
            clauses.append(
                "(node_type = 'memory' AND topic LIKE 'semantic:%%')"
            )
    if not clauses:
        raise ValueError("wipe range owns no graph nodes")
    return " OR ".join(f"({c})" for c in clauses), params


_PLAN_SQL_TEMPLATE = """
WITH RECURSIVE doomed(node_id, node_type, extractor_type, raw_type) AS (
    SELECT node_id, node_type, extractor_type, raw_type
    FROM graph_nodes
    WHERE account_id = %(account_id)s AND ({seed})
    UNION
    SELECT c.node_id, c.node_type, c.extractor_type, c.raw_type
    FROM graph_nodes c
    JOIN doomed d ON c.source_raw_id = d.node_id
    WHERE c.account_id = %(account_id)s
)
SELECT node_type, extractor_type, raw_type, COUNT(*) AS doomed_count
FROM graph_nodes
WHERE account_id = %(account_id)s
  AND node_id IN (SELECT node_id FROM doomed)
GROUP BY node_type, extractor_type, raw_type
ORDER BY node_type, extractor_type, raw_type
"""

_DELETE_SQL_TEMPLATE = """
WITH RECURSIVE doomed(node_id) AS (
    SELECT node_id
    FROM graph_nodes
    WHERE account_id = %(account_id)s AND ({seed})
    UNION
    SELECT c.node_id
    FROM graph_nodes c
    JOIN doomed d ON c.source_raw_id = d.node_id
    WHERE c.account_id = %(account_id)s
)
DELETE FROM graph_nodes
WHERE account_id = %(account_id)s
  AND node_id IN (SELECT node_id FROM doomed)
"""


def plan_sql(stages: list[StageWipe]) -> tuple[str, dict[str, Any]]:
    """Full dry-run SQL counting doomed nodes by (node_type, extractor, raw)."""
    seed, params = seed_where(stages)
    return _PLAN_SQL_TEMPLATE.format(seed=seed), {**params, "account_id": None}


def delete_sql(stages: list[StageWipe]) -> tuple[str, dict[str, Any]]:
    """Delete SQL for the doomed closure (edges cascade via FK)."""
    seed, params = seed_where(stages)
    return _DELETE_SQL_TEMPLATE.format(seed=seed), {**params, "account_id": None}


# ---------------------------------------------------------------------------
# Post-delete ops
# ---------------------------------------------------------------------------

_SAME_IDENTITY_DELETE_SQL = """
DELETE FROM graph_edges e
USING graph_nodes a, graph_nodes b
WHERE e.edge_type = 'same_identity'
  AND e.account_id = %(account_id)s
  AND e.source_id = a.node_id
  AND e.target_id = b.node_id
  AND a.extractor_type = ANY(%(endpoint_types)s)
  AND b.extractor_type = ANY(%(endpoint_types)s)
"""

_CLEAR_EPISODIC_EMBEDDINGS_SQL = """
UPDATE graph_nodes
SET metadata = metadata - 'embedding', updated_at = NOW()
WHERE account_id = %(account_id)s
  AND extractor_type = 'episodic_memory_generator'
  AND metadata ? 'embedding'
"""


def post_ops(
    stages: list[StageWipe],
    *,
    vector_table: str = _VECTOR_TABLE_DEFAULT,
    wipe_vector: bool = False,
) -> list[dict[str, Any]]:
    """SQL statements to run AFTER the node delete (params without account).

    * ``same_identity`` edges between *surviving* nodes are dropped when
      the clustering stage that owns them is in the wiped range (faces /
      speaker embeddings survive an earlier boundary; their clusters must
      not). When the clustered nodes themselves are doomed the delete is a
      harmless no-op — the FK cascade already removed their edges.
    * surviving episodic nodes lose their ``metadata.embedding`` when
      ``index_episodic`` is in range, so the smart-resume probe sees them
      as un-indexed and re-runs the stage.
    * vector rows go only with an explicit ``wipe_vector``.
    """
    ops: list[dict[str, Any]] = []
    endpoints = sorted({e for s in stages for e in s.same_identity_endpoints})
    if endpoints:
        ops.append({
            "name": "drop_same_identity_edges",
            "sql": _SAME_IDENTITY_DELETE_SQL,
            "params": {"endpoint_types": endpoints},
            "when": (
                "clustering stage in wiped range — surviving cluster "
                "members' same_identity edges are stale"
            ),
        })
    if any(s.vector_index for s in stages):
        ops.append({
            "name": "clear_episodic_embeddings",
            "sql": _CLEAR_EPISODIC_EMBEDDINGS_SQL,
            "when": "index_episodic in wiped range — surviving episodic nodes re-index",
        })
        if wipe_vector:
            ops.append({
                "name": "drop_vector_rows",
                "sql": (
                    f"DELETE FROM {vector_table} "
                    "WHERE filters->>'account_id' = %(account_id)s "
                    # %% below — escaped for psycopg2 (named params present)
                    "AND uri LIKE 'm3://%%'"
                ),
                "when": "index_episodic in wiped range + --wipe-vector",
            })
    return ops


# ---------------------------------------------------------------------------
# QA/eval artifacts
# ---------------------------------------------------------------------------

_QA_FILE_NAMES = (
    "m3_qa_results.jsonl",
    "all_m3_qa_results.jsonl",
    "run_report.json",
    "m3_eval_summary.json",
    "m3_eval_results.jsonl",
)


def find_qa_files(root: Path) -> list[Path]:
    """rglob the QA/eval artifact names under ``root`` (sorted for output)."""
    files = {
        p for name in _QA_FILE_NAMES for p in root.rglob(name)
    }
    return sorted(files)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    conn,
    *,
    account: str,
    stages: list[StageWipe],
    vector_conn=None,
    vector_table: str = _VECTOR_TABLE_DEFAULT,
    clear_qa_dir: Path | None = None,
    wipe_vector: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Plan (dry-run) or execute the stage-bounded wipe for one account.

    ``conn`` is a psycopg2 connection to the graph store; ``vector_conn``
    (optional) to the vector store for ``--wipe-vector``. Dry-run is the
    default. Every statement is scoped to ``account``.
    """
    if not account:
        raise ValueError("account_id is mandatory — refusing an unscoped wipe")

    plan, params = plan_sql(stages)
    params = {**params, "account_id": account}

    report: dict[str, Any] = {
        "account_id": account,
        "boundary_stage": stages[0].stage_id if stages else None,
        "wiped_stages": [s.stage_id for s in stages],
        "execute": execute,
        "doomed_by_kind": [],
        "doomed_total": 0,
    }

    with conn.cursor() as cur:
        cur.execute(plan, params)
        rows = cur.fetchall()
    for node_type, extractor_type, raw_type, count in rows:
        report["doomed_by_kind"].append({
            "node_type": node_type,
            "extractor_type": extractor_type,
            "raw_type": raw_type,
            "count": count,
        })
        report["doomed_total"] += count

    ops = post_ops(stages, vector_table=vector_table, wipe_vector=wipe_vector)
    report["post_ops"] = [
        {k: v for k, v in op.items() if k != "sql"} for op in ops
    ]

    qa_files: list[Path] = []
    if any(s.qa_files for s in stages) and clear_qa_dir is not None:
        qa_files = find_qa_files(clear_qa_dir)
    report["qa_files"] = [str(p) for p in qa_files]

    if not execute:
        logger.info(
            "[dry-run] would delete %d nodes for account=%s from stage %s",
            report["doomed_total"], account, report["boundary_stage"],
        )
        return report

    # --- execute ---------------------------------------------------------
    delete_stmt, del_params = delete_sql(stages)
    del_params = {**del_params, "account_id": account}
    with conn.cursor() as cur:
        cur.execute(delete_stmt, del_params)
        report["deleted_nodes"] = cur.rowcount
    for op in ops:
        # drop_vector_rows runs on vector_conn below when one is provided —
        # executing it on conn too would self-deadlock (conn's uncommitted
        # delete holds the transactionid lock vector_conn then waits on,
        # while conn only commits after the vector_conn execute returns).
        if op["name"] == "drop_vector_rows" and vector_conn is not None:
            continue
        with conn.cursor() as cur:
            cur.execute(op["sql"], {**op.get("params", {}), "account_id": account})
            report.setdefault("post_op_rowcounts", {})[op["name"]] = cur.rowcount
    if vector_conn is not None:
        vector_op = next(
            (o for o in ops if o["name"] == "drop_vector_rows"), None
        )
        if vector_op is not None:
            with vector_conn.cursor() as cur:
                cur.execute(
                    vector_op["sql"], {"account_id": account}
                )
                report["post_op_rowcounts"]["drop_vector_rows"] = cur.rowcount
            # psycopg2 is not autocommit by default — without this the
            # vector delete silently rolled back at connection close.
            vector_conn.commit()
    for path in qa_files:
        try:
            path.unlink()
        except OSError as exc:  # pragma: no cover - best-effort
            logger.warning("could not remove %s: %s", path, exc)
    conn.commit()
    logger.info(
        "wiped %d nodes for account=%s from stage %s (+%d post-op groups)",
        report["deleted_nodes"], account, report["boundary_stage"], len(ops),
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="m3.wipe_stages",
        description=(
            "Stage-bounded account wipe: delete one account's graph/vector/"
            "QA state at/after a pipeline-stage boundary, keeping everything "
            "before it. Dry-run by default; pass --yes to execute."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--account", default=None,
                   help="account_id to wipe (mandatory — writes are account-scoped).")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--wipe-from", default=None, metavar="STAGE",
                       help="wipe STAGE and everything after it (e.g. m3.episodic_memory).")
    group.add_argument("--keep-until", default=None, metavar="STAGE",
                       help="keep STAGE and everything before it; wipe the rest.")
    p.add_argument("--list-stages", action="store_true",
                   help="print the stage dependency order and exit.")
    p.add_argument("--yes", action="store_true",
                   help="execute the delete (default is a dry-run).")
    p.add_argument("--connection-string",
                   default=os.environ.get("OGMEM_GRAPH_CONNECTION_STRING"),
                   help="Graph-store DSN (env OGMEM_GRAPH_CONNECTION_STRING).")
    p.add_argument("--wipe-vector", action="store_true",
                   help="Also delete the account's m3:// rows from the vector "
                        "table (requires --vector-connection-string).")
    p.add_argument("--vector-connection-string",
                   default=os.environ.get("OGMEM_VECTOR_CONNECTION_STRING"),
                   help="Vector-store DSN (env OGMEM_VECTOR_CONNECTION_STRING).")
    p.add_argument("--vector-table", default=_VECTOR_TABLE_DEFAULT,
                   help="Vector table holding the m3:// episodic rows.")
    p.add_argument("--clear-qa-dir", default=None, metavar="DIR",
                   help="Additionally delete m3_qa_results.jsonl / "
                        "all_m3_qa_results.jsonl / run_report.json / "
                        "m3_eval_summary.json under DIR (rglob).")
    args = p.parse_args(argv)
    if not args.list_stages and not (args.wipe_from or args.keep_until):
        p.error("one of the arguments --wipe-from --keep-until is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.list_stages:
        for s in ORDERED_STAGES:
            spec = STAGE_SPECS[s]
            owns = " | ".join(filter(None, [
                "ext:" + ",".join(spec.extractor_types) if spec.extractor_types else "",
                "raw:" + ",".join(spec.raw_types) if spec.raw_types else "",
                f"memory_nodes({spec.memory_nodes})" if spec.memory_nodes else "",
                "same_identity_edges" if spec.same_identity_endpoints else "",
                "vector_index" if spec.vector_index else "",
                "qa_files" if spec.qa_files else "",
            ]))
            print(f"{s:36s} {owns}")
        return 0

    if not args.account:
        print("error: --account is required", file=sys.stderr)
        return 2

    try:
        if args.wipe_from:
            stages = stages_from(args.wipe_from)
        else:
            stages = stages_after(args.keep_until)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.connection_string:
        print(
            "error: --connection-string (or OGMEM_GRAPH_CONNECTION_STRING) "
            "is required",
            file=sys.stderr,
        )
        return 2

    import psycopg2

    conn = psycopg2.connect(args.connection_string)
    vector_conn = None
    if args.wipe_vector:
        if not args.vector_connection_string:
            print(
                "error: --wipe-vector requires --vector-connection-string "
                "(or OGMEM_VECTOR_CONNECTION_STRING)",
                file=sys.stderr,
            )
            return 2
        vector_conn = psycopg2.connect(args.vector_connection_string)

    try:
        report = run(
            conn,
            account=args.account,
            stages=stages,
            vector_conn=vector_conn,
            vector_table=args.vector_table,
            clear_qa_dir=(
                Path(args.clear_qa_dir) if args.clear_qa_dir else None
            ),
            wipe_vector=args.wipe_vector,
            execute=args.yes,
        )
    finally:
        conn.close()
        if vector_conn is not None:
            vector_conn.close()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
