"""Shared resume / QA-only helpers for M3 single-video and multi-video entry points.

The probe + QA-only shortcut lives here so both:

* :mod:`scripts.run_m3_bench_subset` (multi-video driver), and
* :mod:`m3.run_video_pipeline` (single-video CLI)

can offer ``--reuse-graph`` semantics without duplicating the cycle
workaround, graph query, or QA subprocess invocation.

What this module provides
-------------------------

* :func:`build_graph_service` — build a :class:`GraphService` without
  tripping the pre-existing ``service.api`` ↔ ``core.tools`` circular
  import. Pre-loads ``core.models`` to break the cycle (same trick the
  pipeline runner uses at ``m3/pipeline/runner.py:236``).
* :func:`request_context` — uniform :class:`RequestContext` for graph
  queries, with the same agent/user ids as the QA driver.
* :func:`video_fully_indexed` — probe the SQL graph for ``VIDEO`` →
  ``episodic_memory_generator`` completeness per video. Returns a
  structured dict (counts + missing-node-id lists) instead of a bare
  bool so callers can log the breakdown.
* :func:`run_qa_only` — bypass :class:`PipelineRunner`, invoke
  :func:`m3.qa.run_react_qa` directly against the pre-loaded graph and
  write ``<work_dir>/m3_qa_results.jsonl``. No ffmpeg / Whisper / face /
  voice / LLM-summarisation / embedder calls.

The module has zero state — every function is a pure callable so the
driver and the pipeline CLI can use the same helpers in their own
control flow.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph service construction (avoids the service.api ↔ core.tools cycle)
# ---------------------------------------------------------------------------


def build_graph_service(
    graph_store: str, connection_string: str | None
) -> tuple[Any, str]:
    """Build a :class:`service.graph_service.GraphService`.

    Mirrors :func:`m3.load_clips.build_graph_service` but inlined to
    defer the imports until first call. Importing
    :mod:`m3.load_clips` at module load triggers a circular import via
    :mod:`service.api` (which itself needs :mod:`core.tools` resolving
    first). The pipeline runner pre-loads ``core.models`` to break the
    cycle; we do the same.

    Returns:
        ``(graph_service, store_kind)`` — the second element is
        ``"memory"`` or ``"sql"`` for logging.
    """
    # Pre-load core.models so the service.api <-> core.tools cycle is
    # broken before we import the graph service.
    import core.models  # noqa: F401 -- side-effect import

    from service.graph_service import GraphService

    if graph_store == "memory":
        from providers.graph_store import MemoryGraphStore

        return GraphService(MemoryGraphStore()), "memory"

    if graph_store == "sql":
        import os

        dsn = (
            connection_string
            or os.environ.get("OGMEM_GRAPH_CONNECTION_STRING")
        )
        if not dsn:
            raise RuntimeError(
                "SQL store requested but no DSN provided. Pass "
                "--connection-string or set OGMEM_GRAPH_CONNECTION_STRING."
            )
        from providers.graph_store import SQLGraphStore

        # OGMEM_GRAPH_POOL_SIZE (default 32): keep >= the parallel-clip
        # worker count or worker threads serialize on pool checkout.
        graph_pool_size = int(os.environ.get("OGMEM_GRAPH_POOL_SIZE") or 32)
        return GraphService(
            SQLGraphStore(connection_string=dsn, pool_size=graph_pool_size)
        ), "sql"

    raise ValueError(f"Unknown graph store: {graph_store!r}")


def request_context(account_id: str) -> Any:
    """Construct a :class:`RequestContext` for graph queries.

    Tenant-scoped queries need a ``RequestContext``; we use the same
    agent / user ids as the QA driver so the resulting queries look
    identical to the ones the pipeline runner emits.
    """
    from core.models import RequestContext
    from m3.config import M3BENCH_AGENT_ID, M3BENCH_USER_ID

    return RequestContext(
        account_id=account_id,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-resume",
        trace_id="m3-resume-graph-probe",
    )


# ---------------------------------------------------------------------------
# Completeness probe
# ---------------------------------------------------------------------------


def video_fully_indexed(
    graph_service: Any,
    account_id: str,
    video_id: str,
    *,
    require_embeddings: bool = True,
    page_size: int = 1000,
    vector_index: Any = None,
) -> dict[str, Any]:
    """Check whether every VIDEO clip of ``video_id`` has a completed
    ``episodic_memory_generator`` ExtractedNode and persisted vector if
    ``require_embeddings``. An explicitly supplied (or graph-attached) vector
    index is authoritative; graph embeddings are a legacy-only fallback.

    Returns a dict so callers can log the breakdown::

        {
            "complete": bool,
            "video_clip_count": int,
            "missing_extracted": list[str],   # ExtractedNode_ids missing
            "unfinished_extracted": list[str],  # still pending / failed
            "missing_embedding": list[str],    # completed but no embedding
        }

    The "complete" key is True iff all four lists are empty.

    Consistency note: this function shares its per-node "done" judgement
    with the smart-resume probe protocol (see
    :func:`m3.pipeline.stages._probe_from_extracted_nodes`). Both call
    :func:`m3.pipeline.stages._classify_extracted_node` so the "is this
    ExtractedNode completed?" rule lives in exactly one place. If you
    add a new completion criterion (e.g. a schema version check), only
    the classifier needs updating.
    """
    from core.graph_enums import RawDataType
    from m3.episodic_memory import extracted_node_id
    from m3.pipeline.stages import _classify_extracted_node

    ctx = request_context(account_id)
    if vector_index is None:
        vector_index = vars(graph_service).get("_vector_index")

    # Pull all VIDEO RawNodes for this account. The graph service doesn't
    # expose a metadata-key filter, so we paginate with a generous limit
    # and filter in Python. M3-Bench scale (100 videos × ~10 clips) fits
    # well within a single page.
    video_nodes: list[Any] = []
    offset = 0
    while True:
        page = graph_service.list_raw_nodes(
            RawDataType.VIDEO, ctx, limit=page_size, offset=offset
        )
        for n in page:
            if (n.metadata or {}).get("video_id") == video_id:
                video_nodes.append(n)
        if len(page) < page_size:
            break
        offset += page_size

    if not video_nodes:
        return {
            "complete": False,
            "video_clip_count": 0,
            "missing_extracted": [],
            "unfinished_extracted": [],
            "missing_embedding": [],
        }

    missing: list[str] = []
    unfinished: list[str] = []
    no_embed: list[str] = []
    for vn in video_nodes:
        ext_id = extracted_node_id("episodic_memory_generator", vn.node_id)
        ext = graph_service.get_extracted_node(ext_id, ctx)
        # Same classification the smart-resume probe uses — see
        # :func:`m3.pipeline.stages._classify_extracted_node`.
        category = _classify_extracted_node(
            ext, require_embeddings=require_embeddings and vector_index is None,
        )
        if category is None and require_embeddings and vector_index is not None:
            from m3.index_status import indexed_extraction
            if not indexed_extraction(vector_index, ext, vn, account_id):
                category = "missing_embedding"
        if category == "missing":
            missing.append(ext_id)
        elif category == "unfinished":
            unfinished.append(ext_id)
        elif category == "missing_embedding":
            no_embed.append(ext_id)
        elif category is None:
            pass  # completed; nothing to record
        else:  # pragma: no cover — defensive
            logger.warning(
                "video_fully_indexed: unexpected classifier output %r for %s",
                category, ext_id,
            )

    return {
        "complete": not (missing or unfinished or no_embed),
        "video_clip_count": len(video_nodes),
        "missing_extracted": missing,
        "unfinished_extracted": unfinished,
        "missing_embedding": no_embed,
    }


# ---------------------------------------------------------------------------
# QA-only execution (no PipelineRunner)
# ---------------------------------------------------------------------------


def qa_params_argv(set_values: list[str] | None) -> list[str]:
    """Map ``--set qa.params.<name>=<value>`` entries to m3.qa argv pairs.

    ``qa.params.transcript_top_k=20`` → ``["--transcript-top-k", "20"]``;
    non-``qa.params.*`` entries are ignored. Appended LAST to the qa argv
    by :func:`run_qa_only` so explicit ``--set`` overrides win over the
    driver's own defaults (argparse last-occurrence-wins).
    """
    argv: list[str] = []
    for entry in set_values or []:
        if "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        if not key.startswith("qa.params."):
            continue
        name = key[len("qa.params."):].strip()
        if not name:
            continue
        argv.extend(["--" + name.replace("_", "-"), value])
    return argv


def run_qa_only(
    *,
    graph_store: str,
    connection_string: str | None,
    account_id: str,
    config_path: Path,
    annotations_path: Path,
    video_id: str,
    work_dir: Path,
    max_iterations: int = 6,
    timeout_seconds: int = 90,
    top_k: int | None = None,
    resume: bool = False,
    verbose: bool = False,
    temporal_policy: str = "current_clip",
    lookahead_seconds: float = 0.0,
    qa_params: list[str] | None = None,
) -> dict[str, Any]:
    """Bypass :class:`PipelineRunner` and invoke ``m3.qa.run_react_qa`` directly.

    Used by both the single-video CLI and the multi-video driver when
    the graph is already complete. Writes only ``<work_dir>/m3_qa_results.jsonl``
    (and optionally ``pipeline_summary.json`` if the caller passes the
    work_dir back through :func:`write_resume_summary`).

    Returns a summary dict::

        {
            "video_id": str,
            "status": "ok" | "failed",
            "duration_sec": float,
            "jsonl": str,
            "returncode": int,
            "mode": "qa_only",
            "qa_only_seen_qids": list[str],
            "qa_only_missing_qids": list[str],
        }
    """
    from m3.qa import parse_react_qa_args, run_react_qa

    output_jsonl = work_dir / "m3_qa_results.jsonl"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        "--annotations",
        str(annotations_path),
        "--output-jsonl",
        str(output_jsonl),
        "--graph-store",
        graph_store,
        "--config",
        str(config_path),
        "--account-id",
        account_id,
        "--video",
        video_id,
        "--max-iterations",
        str(max_iterations),
        "--timeout-seconds",
        str(timeout_seconds),
        "--temporal-policy",
        temporal_policy,
        "--lookahead-seconds",
        str(lookahead_seconds),
    ]
    if connection_string:
        argv.extend(["--connection-string", connection_string])
    if top_k:
        argv.extend(["--top-k", str(top_k)])
    if resume:
        argv.append("--resume")
    if verbose:
        argv.append("--verbose")
    # Appended last: argparse last-occurrence wins, so explicit
    # ``--set qa.params.*`` retrieval knobs override the defaults above
    # (graph-aware QA-only resume used to silently ignore them).
    argv.extend(qa_params or [])

    logger.info(
        "[%s] QA-only mode against %s graph store (resume=%s)",
        video_id,
        graph_store,
        resume,
    )

    start = time.monotonic()
    # Let the QA driver construct both the graph and its configured vector
    # index. The status-check helper above intentionally builds only a graph;
    # passing that graph here bypasses indexed retrieval and interval lookup.
    rc = run_react_qa(parse_react_qa_args(argv))
    duration = time.monotonic() - start

    # Build the expected_qids set by re-loading the annotations file and
    # extracting just our video_id's qa_list. Lets the caller pass nothing
    # extra to determine completeness.
    expected_qids = _expected_qids(annotations_path, video_id)

    from m3.qa_results import completed_keys
    seen = {qid for vid, qid in completed_keys(output_jsonl) if vid == video_id}

    return {
        "video_id": video_id,
        "status": "ok"
        if (rc == 0 and expected_qids.issubset(seen))
        else "failed",
        "duration_sec": round(duration, 2),
        "jsonl": str(output_jsonl),
        "returncode": rc,
        "mode": "qa_only",
        "qa_only_seen_qids": sorted(seen),
        "qa_only_missing_qids": sorted(expected_qids - seen),
    }


def _expected_qids(annotations_path: Path, video_id: str) -> set[str]:
    """Read ``annotations_path`` and return the set of question_ids for
    ``video_id``. Best-effort — empty set on any failure (caller will
    then skip the completeness check).
    """
    try:
        annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    entry = annotations.get(video_id) or {}
    return {
        str(q.get("question_id"))
        for q in (entry.get("qa_list") or [])
        if q.get("question_id") is not None
    }


# ---------------------------------------------------------------------------
# Optional pipeline_summary.json write for single-video CLI parity
# ---------------------------------------------------------------------------


def write_resume_summary(
    *,
    work_dir: Path,
    video_id: str,
    pipeline_name: str,
    pipeline_version: int,
    jsonl_path: Path,
    duration_sec: float,
    rc: int,
) -> Path:
    """Write a minimal ``pipeline_summary.json`` after a QA-only run.

    The single-video CLI normally writes this from inside
    ``PipelineRunner.run``. When we bypass the runner, callers should
    call this so downstream tooling (``scripts/m3_eval.py`` etc.) can
    still discover the output.
    """
    summary_path = work_dir / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "pipeline": {
            "name": pipeline_name,
            "version": pipeline_version,
            "description": "QA-only resume path (--reuse-graph hit)",
        },
        "video_id": video_id,
        "selected_video": "",  # not produced by QA-only mode
        "annotations": str(annotations_path := work_dir / "annotations"),
        "data_jsonl": "",
        "frames_jsonl": "",
        "faces": None,
        "episodic_jsonl": "",
        "memory": None,
        "index": None,
        "qa_results": str(jsonl_path),
        "artifacts": {
            "qa_results": str(jsonl_path),
            "summary_json": str(summary_path),
        },
        "stages": {
            "qa_only_resume": {
                "stage_id": "qa_only_resume",
                "stage_type": "m3.resume.run_qa_only",
                "ok": rc == 0,
                "outputs": {"qa_results": str(jsonl_path)},
                "counts": {"seconds": 0, "duration_sec": round(duration_sec, 2)},
                "errors": [],
                "warnings": [],
            }
        },
        "summary_path": str(summary_path),
        "mode": "qa_only",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path
