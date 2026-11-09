#!/usr/bin/env python3
"""Run the M3 multi-modal pipeline over a subset of M3-Bench videos.

Driver for scaling the single-video ``m3.run_video_pipeline`` flow from
1 video to N videos. For each ``video_id`` in the requested subset the
script:

1. Resolves the source video path via ``m3.config.resolve_video_path``
   (so it works whether the bench data lives under ``data/videos``,
   ``data/m3/videos``, or a relocated root).
2. Spawns ``python -m m3.run_video_pipeline`` as a subprocess with the
   full pipeline spec (``video_episodic_full`` by default), the real
   M3-Bench ``robot.json`` annotations file, and ``--set
   qa.params.video=<id>`` to filter the qa_list down to this video.
3. Collects the per-video ``m3_qa_results.jsonl`` into a merged JSONL
   at ``<work-dir-root>/all_m3_qa_results.jsonl``.
4. Optionally runs the M3 judge (``m3.eval``) over the merged JSONL
   to produce ``m3_eval_summary.json``.

Resumable: by default a video is skipped if its per-video
``m3_qa_results.jsonl`` already contains records for every question
listed in the annotations file (use ``--force`` to override).

This is the Option A (subprocess loop) implementation described in the
M3-Bench multi-video plan. The alternative — a single-process
in-process orchestrator sharing one graph store across videos — is
covered by running with ``--graph-store sql`` and is left as a future
optimization.

LLM / config wiring
-------------------

The QA stage reads its LLM provider / model / API key from the
``m3.qa_llm`` block of the YAML passed via ``--config``. The driver
does NOT override this — leave ``deploy/ogmemory.yaml`` (or your
envsubst-resolved copy) as the single source of truth.

Two supported workflows:

**Workflow A — envsubst-resolved YAML (preferred)::

    set -a && source deploy/.env && set +a
    envsubst < deploy/ogmemory.yaml > data/ogmem_test_lyj.resolved.yaml
    uv run python scripts/run_m3_bench_subset.py \\
        --config data/ogmem_test_lyj.resolved.yaml \\
        --videos bedroom_03,living_room_06 \\
        --work-dir-root data/m3_runs/m3bench_smoke

The resolved YAML has hardcoded values, no ``${VAR}`` refs. The driver
detects this and skips env injection (it's a no-op).

**Workflow B — original YAML with placeholders::

    uv run python scripts/run_m3_bench_subset.py \\
        --config deploy/ogmemory.yaml \\
        ...

If ``m3.qa_llm`` in the original YAML has unresolved ``${LLM_API_KEY}``
refs and the parent shell didn't export them, the driver copies the
hardcoded ``m3.episodic_llm`` values into ``LLM_API_KEY`` /
``LLM_BASE_URL`` / etc. before spawning the subprocess. This is a
back-compat shim — workflow A is cleaner.

To override the QA provider for one specific run without editing the
YAML, pass ``--set qa.params.llm=openai`` (or any supported provider);
this is forwarded as a stage-param override.

Usage::

    ./scripts/run_m3_bench_subset.py \\
        --videos bedroom_03,living_room_06,kitchen_01 \\
        --annotations data/m3/annotations/robot.json \\
        --work-dir-root data/m3_runs/m3bench_pilot \\
        --pipeline config/m3/pipelines/video_episodic_full.yaml \\
        --seconds 120 --judge

Notes on what already works today (no code changes needed in the core
pipeline):
- ``m3.pipeline.stages.qa_stage`` (line 400) reads
  ``stage.params["annotations"]`` and falls back to the auto-generated
  ``one_video.json`` only when the param is unset.
- ``run_react_qa`` (m3/qa.py:383) splits ``--video`` on commas and
  filters ``qa_list`` records whose ``video_id`` matches, so a single
  run only answers questions for the requested video.

If you want to switch to a real multi-video graph (Option B), pass
``--graph-store sql`` plus ``--connection-string`` so all videos share
one store. The per-process driver shape stays the same.

Graph-aware resume (--reuse-graph)
-----------------------------------

When ``--graph-store sql`` is in use, pass ``--reuse-graph`` to skip the
heavy extraction stages for any video whose graph state is already
complete. The driver probes the SQL graph for:

* every ``VIDEO`` RawNode with ``metadata.video_id == <video_id>``
* a matching ``episodic_memory_generator`` ExtractedNode per VIDEO
  (deterministic id: ``sha1("extracted/episodic_memory_generator/<video_node_id>")[:32]``)
* ``extraction_status == "completed"``
* ``metadata.embedding`` populated (so ``search_by_embedding`` will work)

When all four conditions hold for every clip, the driver invokes
``m3.qa.run_react_qa`` directly with the pre-loaded graph and writes
``<work_dir_root>/<video_id>/m3_qa_results.jsonl`` — no ffmpeg / Whisper
/ face / voice / LLM-summarisation / embedder calls for that video.

This is what makes the multi-video driver safe to point at a new
``--work-dir-root`` after a full run: the heavy work is reused from
the persistent graph, only QA re-executes.

Usage::

    uv run python scripts/run_m3_bench_subset.py \\
        --config data/ogmem_test_lyj.resolved.yaml \\
        --videos bedroom_03,living_room_06 \\
        --graph-store sql --connection-string "$GRAPH_DSN" \\
        --reuse-graph \\
        --work-dir-root data/m3_runs/m3bench_smoke_v2 \\
        --judge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from m3.config import resolve_video_path  # noqa: E402
from m3.temporal import annotation_scope  # noqa: E402
from m3.resume import (  # noqa: E402
    build_graph_service as _build_graph_service,
    run_qa_only,
    video_fully_indexed,
)

logger = logging.getLogger(__name__)


def resolve_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    """Build the env passed to each per-video subprocess.

    Workflow A — envsubst-resolved YAML (preferred)::

        set -a && source deploy/.env && set +a
        envsubst < deploy/ogmemory.yaml > data/ogmem_test_lyj.resolved.yaml
        # pass --config data/ogmem_test_lyj.resolved.yaml to the driver

    The resolved YAML has hardcoded values (no ``${VAR}`` refs).
    ``m3.config._load_yaml_config`` will read those values directly, no
    env vars needed. This function detects that and returns the parent
    env unchanged.

    Workflow B — original YAML with ``${VAR}`` placeholders::

        # deploy/ogmemory.yaml still has api_key: "${LLM_API_KEY}"
        # m3.qa_llm block was empty, m3.episodic_llm has hardcoded key.
        # The QA stage would fail with "LLM_API_KEY not set" because the
        # parent shell doesn't export it.

    To make workflow B work, copy the hardcoded ``episodic_llm`` values
    into the standard env var names that ``m3.config._expand_env_ref``
    looks up (``LLM_PROVIDER``, ``LLM_API_KEY``, ``LLM_BASE_URL``,
    ``LLM_MODEL``). The QA stage then resolves its ``${LLM_API_KEY}``
    reference against the injected env. Users who don't want this can
    run their pipeline with a resolved YAML (workflow A) and the
    injection becomes a no-op.
    """
    env = os.environ.copy()
    try:
        import yaml

        cfg_path = Path(args.config).resolve()
        if not cfg_path.exists():
            return env
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        m3 = cfg.get("m3") or {}

        # Detect workflow A: no ${VAR} placeholders anywhere in m3.qa_llm
        # or m3.episodic_llm. The resolved YAML doesn't need env help.
        def has_unresolved_refs(block: dict[str, Any]) -> bool:
            for v in block.values():
                if (
                    isinstance(v, str)
                    and v.startswith("${")
                    and v.endswith("}")
                ):
                    return True
            return False

        qa_block = m3.get("qa_llm") or {}
        ep_block = m3.get("episodic_llm") or {}
        if not has_unresolved_refs(qa_block) and not has_unresolved_refs(ep_block):
            logger.debug(
                "config %s has no ${VAR} refs in m3.qa_llm/m3.episodic_llm — "
                "skipping env injection (workflow A: envsubst-resolved).",
                cfg_path,
            )
            return env

        # Workflow B: inject qa_llm (preferred) or episodic_llm values into
        # the env so the QA stage can resolve its ${LLM_*} placeholders.
        for block_name in ("qa_llm", "episodic_llm"):
            block = m3.get(block_name) or {}
            if not isinstance(block, dict):
                continue
            mapping = {
                "provider": "LLM_PROVIDER",
                "api_key": "LLM_API_KEY",
                "base_url": "LLM_BASE_URL",
                "model": "LLM_MODEL",
            }
            injected_any = False
            for yaml_key, env_key in mapping.items():
                value = block.get(yaml_key)
                if not value or (
                    isinstance(value, str)
                    and value.startswith("${")
                    and value.endswith("}")
                ):
                    continue
                env[env_key] = str(value)
                injected_any = True
            if injected_any:
                # qa_llm wins; if it's incomplete fall through to
                # episodic_llm only for fields qa_llm left unset.
                if block_name == "qa_llm":
                    break
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("could not load %s for env injection: %s", args.config, exc)
    return env


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def parse_csv(value: str | None) -> list[str]:
    """Split a comma-separated id list, stripping whitespace.

    Empty input -> empty list (lets ``--videos`` and ``--exclude`` compose).
    """
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def select_subset(
    annotations: dict[str, Any],
    include: list[str],
    exclude: list[str],
    limit: int | None,
    *,
    sample_size: int | None = None,
    seed: int = 42,
) -> list[str]:
    """Resolve the requested subset to a concrete list of video_ids.

    Order is preserved: explicit ``--videos`` is honored first, then any
    remaining video_ids from the annotation file in JSON-iteration order
    (subject to ``--exclude`` and ``--limit``).
    With ``sample_size``, sample without replacement from sorted IDs with
    questions, using a local seeded RNG and applying exclusions first.
    """
    if sample_size is not None:
        if include or limit is not None:
            raise ValueError("--sample-size cannot be combined with --videos or --limit")
        if sample_size <= 0:
            raise ValueError("--sample-size must be positive")
        unknown = sorted(set(exclude) - annotations.keys())
        if unknown:
            raise ValueError(f"--exclude references unknown video IDs: {unknown}")
        candidates = sorted(
            vid for vid, entry in annotations.items()
            if vid not in set(exclude) and entry.get("qa_list")
        )
        if sample_size > len(candidates):
            raise ValueError(
                f"Requested {sample_size} videos, but only {len(candidates)} "
                "annotated candidates remain after exclusions"
            )
        # Stable across JSON key ordering; do not mutate global RNG state.
        return random.Random(seed).sample(candidates, sample_size)
    all_ids = list(annotations.keys())
    if include:
        missing = [v for v in include if v not in annotations]
        if missing:
            raise SystemExit(
                f"--videos references ids not in annotations: {missing}"
            )
        return include
    selected = [v for v in all_ids if v not in set(exclude)]
    if limit is not None:
        selected = selected[:limit]
    return selected


def save_sample_manifest(
    path: Path, annotations: dict[str, Any], subset: list[str],
    *, seed: int, exclude: list[str],
) -> dict[str, Any]:
    """Pin the cohort; refuse to mix different samples in one run directory."""
    manifest = {
        "version": 1,
        "method": "uniform_without_replacement_sorted_ids",
        "seed": seed,
        "sample_size": len(subset),
        "exclude": sorted(set(exclude)),
        "annotations_sha256": hashlib.sha256(
            json.dumps(annotations, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "candidate_ids": sorted(
            vid for vid, entry in annotations.items()
            if vid not in set(exclude) and entry.get("qa_list")
        ),
        "selected_video_ids": subset,
        "question_counts": {
            vid: len(annotations[vid]["qa_list"]) for vid in subset
        },
        "total_questions": sum(len(annotations[vid]["qa_list"]) for vid in subset),
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                f"Sample differs from saved {path}; use a new --work-dir-root. "
                "--force does not overwrite the selected cohort."
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return manifest


# ---------------------------------------------------------------------------
# Per-video run + resume
# ---------------------------------------------------------------------------


def load_questions_for_video(
    annotations: dict[str, Any], video_id: str
) -> list[dict[str, Any]]:
    """Return the qa_list for ``video_id`` from the loaded annotations."""
    entry = annotations.get(video_id)
    if entry is None:
        raise KeyError(f"video_id {video_id!r} not in annotations")
    return list(entry.get("qa_list") or [])


def already_done(
    work_dir_root: Path,
    video_id: str,
    expected_qids: set[str],
    expected_questions: list[dict[str, Any]] | None = None,
) -> bool:
    """Has this video already produced records for every expected question?

    Returns True if the per-video JSONL exists and contains every
    ``(video_id, question_id)`` we'd run, AND every record is non-empty
    (no ``error`` field). Empty JSONL or partial coverage -> False.
    """
    jsonl = work_dir_root / video_id / "m3_qa_results.jsonl"
    if not jsonl.exists():
        return False
    seen: set[str] = set()
    saw_error = False
    expected = {
        str(q["question_id"]): q for q in (expected_questions or [])
        if q.get("question_id") is not None
    }
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(rec, dict) or rec.get("video_id") != video_id:
            return False
        qid = rec.get("question_id")
        if str(qid) not in expected_qids:
            return False
        if expected_questions is not None:
            question = expected.get(str(qid))
            if question is None or any(
                rec.get(result_key) != question.get(annotation_key)
                for result_key, annotation_key in (
                    ("question", "question"),
                    ("expected_answer", "answer"),
                    ("timestamp", "timestamp"),
                    ("before_clip", "before_clip"),
                )
            ):
                return False
        if qid:
            seen.add(str(qid))
        if rec.get("error") or not str(rec.get("predicted_answer") or "").strip():
            saw_error = True
    if saw_error:
        return False
    return expected_qids.issubset(seen)


def validate_annotation_duration(
    annotations: dict[str, Any], video_ids: Iterable[str], seconds: int,
) -> list[str]:
    """Find questions outside a trimmed run without silently changing the benchmark."""
    if seconds <= 0:
        return []
    outside = []
    for video_id in video_ids:
        for question in load_questions_for_video(annotations, video_id):
            _, cutoff = annotation_scope(timestamp=question.get("timestamp"))
            if cutoff is not None and cutoff > seconds:
                outside.append(
                    f"{video_id}/{question.get('question_id', '?')} "
                    f"at {question['timestamp']}"
                )
    return outside


# Graph-aware resume helpers (``build_graph_service``,
# ``video_fully_indexed``, ``run_qa_only``) are imported from
# :mod:`m3.resume` at module-top so the single-video CLI in
# :mod:`m3.run_video_pipeline` can share the same logic.


def build_pipeline_cmd(
    *,
    python: list[str],
    video_path: Path,
    video_id: str,
    annotations_path: Path,
    pipeline: Path,
    config_path: Path,
    seconds: int,
    work_dir: Path,
    graph_store: str,
    connection_string: str | None,
    account_id: str,
    enable_stages: list[str],
    disable_stages: list[str],
    set_values: list[str],
    extra_qa_params: list[str],
    force: bool,
    no_smart_resume: bool,
    verbose: bool,
) -> list[str]:
    """Compose the ``python -m m3.run_video_pipeline`` argv.

    The two key overrides:
      * ``--set qa.params.annotations=<robot.json>`` so the QA stage reads
        the real qa_list instead of the empty auto-generated stub.
      * ``--set qa.params.video=<video_id>`` so the adapter only answers
        this video's questions (robot.json contains all 1276 QAs across
        100 videos).

    ``python`` is a *list* (e.g. ``["uv", "run", "python"]`` or
    ``["/path/to/.venv/bin/python"]``) so the subprocess inherits the
    same venv / env that the driver was launched from. Using the system
    ``/usr/bin/python3`` will fail because it lacks the project's
    optional deps (``typing_extensions``, ``faster_whisper``,
    ``speechbrain``, etc.).
    """
    cmd: list[str] = [
        *python,
        "-m", "m3.run_video_pipeline",
        "--video", str(video_path),
        "--video-id", video_id,
        "--seconds", str(seconds),
        "--pipeline", str(pipeline),
        "--work-dir", str(work_dir),
        "--config", str(config_path),
        "--graph-store", graph_store,
        "--account-id", account_id,
        # Tell QA where the real qa_list lives and which video to answer.
        "--set", f"qa.params.annotations={annotations_path}",
        "--set", f"qa.params.video={video_id}",
    ]
    if connection_string:
        cmd.extend(["--connection-string", connection_string])
    for stage in enable_stages:
        cmd.extend(["--enable-stage", stage])
    for stage in disable_stages:
        cmd.extend(["--disable-stage", stage])
    for sv in set_values:
        cmd.extend(["--set", sv])
    for extra in extra_qa_params:
        cmd.extend(["--set", extra])
    if force:
        cmd.append("--force")
    if no_smart_resume:
        cmd.append("--no-smart-resume")
    if verbose:
        cmd.append("--verbose")
    return cmd


def run_one_video(
    *,
    video_id: str,
    video_path: Path,
    args: argparse.Namespace,
    annotations: dict[str, Any],
    expected_qids: set[str],
) -> dict[str, Any]:
    """Run the pipeline for one video; return a summary dict.

    Returns ``{"video_id": ..., "status": "ok|skipped|failed",
    "duration_sec": ..., "jsonl": str, "returncode": int}``. Caller
    can persist these summaries for an aggregate run report.
    """
    work_dir = args.work_dir_root / video_id
    work_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and already_done(
        args.work_dir_root, video_id, expected_qids,
        load_questions_for_video(annotations, video_id),
    ):
        logger.info("[%s] already complete — skipping (use --force to rerun)",
                    video_id)
        return {
            "video_id": video_id,
            "status": "skipped",
            "duration_sec": 0.0,
            "jsonl": str(work_dir / "m3_qa_results.jsonl"),
            "returncode": 0,
        }

    # Graph-aware resume: when --reuse-graph + --graph-store sql, probe
    # the graph for completeness before falling through to the heavy
    # pipeline. Only skip stages if every VIDEO clip already has a
    # completed episodic_memory_generator ExtractedNode with embedding.
    if args.reuse_graph and args.graph_store == "sql" and not args.force and not args.dry_run:
        try:
            graph, store_kind = _build_graph_service(
                args.graph_store, args.connection_string
            )
            probe = video_fully_indexed(graph, args.account_id, video_id)
            logger.info(
                "[%s] graph probe: %d clips, complete=%s "
                "(missing_extracted=%d, unfinished=%d, missing_embedding=%d)",
                video_id,
                probe["video_clip_count"],
                probe["complete"],
                len(probe["missing_extracted"]),
                len(probe["unfinished_extracted"]),
                len(probe["missing_embedding"]),
            )
            if probe["complete"]:
                return run_qa_only(
                    graph_store=args.graph_store,
                    connection_string=args.connection_string,
                    account_id=args.account_id,
                    config_path=args.config,
                    annotations_path=args.annotations,
                    video_id=video_id,
                    work_dir=work_dir,
                    max_iterations=args.qa_max_iterations,
                    timeout_seconds=args.qa_timeout,
                    top_k=args.qa_top_k,
                    resume=args.resume,
                    verbose=args.verbose,
                )
        except Exception as exc:  # pragma: no cover - best-effort
            logger.warning(
                "[%s] graph probe failed (%s) — falling back to full pipeline",
                video_id, exc,
            )
    elif args.reuse_graph and args.graph_store != "sql":
        logger.warning(
            "--reuse-graph requires --graph-store sql (current %s); "
            "ignoring --reuse-graph",
            args.graph_store,
        )

    set_values = list(args.set or [])
    extra_qa_params = [
        f"qa.params.max_iterations={args.qa_max_iterations}",
        f"qa.params.timeout_seconds={args.qa_timeout}",
        f"qa.params.output_jsonl={work_dir / 'm3_qa_results.jsonl'}",
    ]
    if args.qa_top_k:
        extra_qa_params.append(f"qa.params.top_k={args.qa_top_k}")

    python_argv = shlex.split(args.python)
    cmd = build_pipeline_cmd(
        python=python_argv,
        video_path=video_path,
        video_id=video_id,
        annotations_path=args.annotations,
        pipeline=args.pipeline,
        config_path=args.config,
        seconds=args.seconds,
        work_dir=work_dir,
        graph_store=args.graph_store,
        connection_string=args.connection_string,
        account_id=args.account_id,
        enable_stages=args.enable_stage,
        disable_stages=args.disable_stage,
        set_values=set_values,
        extra_qa_params=extra_qa_params,
        force=args.force,
        no_smart_resume=args.no_smart_resume,
        verbose=args.verbose,
    )

    log_path = work_dir / "pipeline.log"
    logger.info("[%s] running: %s", video_id, " ".join(shlex.quote(c) for c in cmd))
    logger.info("[%s] streaming logs to %s", video_id, log_path)
    start = time.monotonic()

    if args.dry_run:
        return {
            "video_id": video_id,
            "status": "dry_run",
            "duration_sec": 0.0,
            "jsonl": str(work_dir / "m3_qa_results.jsonl"),
            "returncode": 0,
            "command": cmd,
        }

    sub_env = resolve_subprocess_env(args)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=sub_env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration = time.monotonic() - start
    status = "ok" if proc.returncode == 0 else "failed"
    return {
        "video_id": video_id,
        "status": status,
        "duration_sec": round(duration, 2),
        "jsonl": str(work_dir / "m3_qa_results.jsonl"),
        "returncode": proc.returncode,
    }


# ---------------------------------------------------------------------------
# Aggregation + optional judge
# ---------------------------------------------------------------------------


def aggregate_jsonl(
    video_ids: Iterable[str],
    work_dir_root: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Concatenate per-video ``m3_qa_results.jsonl`` files into one.

    Skips missing/empty files. Returns counts for the run report.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0
    answered = 0
    failed = 0
    per_video: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as out:
        for vid in video_ids:
            jsonl = work_dir_root / vid / "m3_qa_results.jsonl"
            if not jsonl.exists():
                logger.warning("[%s] no m3_qa_results.jsonl at %s", vid, jsonl)
                continue
            count = 0
            for line in jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                out.write(line + "\n")
                count += 1
                total_records += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    failed += 1
                    continue
                if rec.get("error"):
                    failed += 1
                else:
                    answered += 1
            per_video[vid] = count
    return {
        "merged_jsonl": str(out_path),
        "total_records": total_records,
        "answered": answered,
        "failed": failed,
        "per_video": per_video,
    }


def run_judge(
    *,
    annotations: Path,
    merged_jsonl: Path,
    output_dir: Path,
    args: argparse.Namespace,
    subset: list[str] | None = None,
) -> dict[str, Any]:
    """Invoke ``python -m m3.eval`` to judge the merged JSONL.

    Returns the parsed ``m3_eval_summary.json`` if available.

    The ``subset`` list (the video_ids this run actually processed)
    MUST be forwarded as ``--video`` so the judge only scores the
    answered questions. Without this, ``m3.eval`` loads every
    question in ``annotations`` and scores un-answered ones as 0,
    producing a misleading ``overall_mean_score`` (e.g. 0.005 over
    all 1276 questions when only 26 had predictions).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    python_argv = shlex.split(args.python)
    cmd = [
        *python_argv,
        "-m", "m3.eval",
        "--annotations", str(annotations),
        "--input-qa-results", str(merged_jsonl),
        "--output-dir", str(output_dir),
        "--config", str(args.config),
        "--graph-store", args.graph_store,
        "--account-id", args.account_id,
    ]
    if args.connection_string:
        cmd.extend(["--connection-string", args.connection_string])
    # Forward the subset filters so the judge reports accuracy on the
    # questions that were actually answered (not on the global pool).
    if subset:
        cmd.extend(["--video", ",".join(subset)])
    if getattr(args, "type_filter", None):
        cmd.extend(["--type-filter", args.type_filter])
    # args.limit selects VIDEOS in this driver. m3.eval --limit selects
    # QUESTIONS; forwarding it would score only N questions, not N videos.
    # The selected video IDs above are the complete judge scope.
    # Forward judge overrides. ``--judge-llm`` is purely the provider
    # *label* used in the summary header (e.g. "minimax", "openai",
    # "deepseek", "vllm"); the actual endpoint is configured by
    # ``--judge-base-url`` / ``--judge-model`` / ``--judge-api-key`` (or
    # ``m3.judge_llm`` in ``--config``). When only the URL is given, the
    # m3.eval subprocess derives the label from the hostname, so
    # ``--judge-llm`` is optional.
    if args.judge_llm:
        cmd.extend(["--judge-provider", str(args.judge_llm)])
    if args.judge_model:
        cmd.extend(["--judge-model", str(args.judge_model)])
    if args.judge_base_url:
        cmd.extend(["--judge-base-url", str(args.judge_base_url)])
    if args.judge_api_key:
        cmd.extend(["--judge-api-key", str(args.judge_api_key)])
    if args.resume:
        cmd.append("--resume")
    logger.info("judge: %s", " ".join(shlex.quote(c) for c in cmd))
    log_path = output_dir / "judge.log"
    sub_env = resolve_subprocess_env(args)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=sub_env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    summary_path = output_dir / "m3_eval_summary.json"
    summary: dict[str, Any] = {
        "returncode": proc.returncode,
        "summary_path": str(summary_path) if summary_path.exists() else None,
    }
    if summary_path.exists():
        try:
            summary["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("could not parse %s as JSON", summary_path)
    return summary


# ---------------------------------------------------------------------------
# Video-parallel orchestration (WS 2.5)
# ---------------------------------------------------------------------------


def _judge_input_digest(
    jsonl: Path, annotations: Path, args: argparse.Namespace
) -> str:
    """Digest of everything a cached judgment depends on (review item 9).

    Predictions, annotations, the effective judge configuration (provider /
    model / base URL / type filter) and the ``--config`` YAML (it carries
    ``m3.judge_llm``). The API key is deliberately excluded: rotating a key
    must not invalidate judgments made by the same model+endpoint.
    """
    h = hashlib.sha256()
    config = getattr(args, "config", None)
    paths = [jsonl, annotations] + ([Path(config)] if config else [])
    for path in paths:
        try:
            h.update(path.read_bytes())
        except OSError:
            h.update(b"<missing:" + str(path).encode() + b">")
    judge_identity = "|".join(str(x or "") for x in (
        getattr(args, "judge_llm", None),
        getattr(args, "judge_model", None),
        getattr(args, "judge_base_url", None),
        getattr(args, "type_filter", None),
    ))
    h.update(b"judge:" + judge_identity.encode())
    return h.hexdigest()


def _write_judge_digest(
    work_dir: Path, digest: str, *, jsonl: Path, annotations: Path
) -> None:
    """Persist the digest a future resume must match to reuse the judgment."""
    def _size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    sidecar = work_dir / "m3_eval_summary.digest.json"
    try:
        sidecar.write_text(json.dumps({
            "digest": digest,
            "qa_results_bytes": _size(jsonl),
            "annotations_bytes": _size(annotations),
        }, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("could not write judge digest sidecar %s", sidecar)


def live_judge_video(
    *,
    video_id: str,
    work_dir: Path,
    args: argparse.Namespace,
    annotations_path: Path,
) -> dict[str, Any] | None:
    """Judge one video's predictions immediately after the video finishes.

    Writes per-video ``m3_eval_summary.json`` / ``m3_eval_results.jsonl``
    under ``<work-dir-root>/<video_id>/``. When a per-video summary already
    exists (resume) and ``--force`` is unset, it is reused **only when the
    recorded input digest still matches** (predictions, annotations, judge
    configuration) — repaired or partial predictions otherwise re-judge
    instead of silently aggregating stale scores (review item 9).
    """
    if args.dry_run:
        return None
    jsonl = work_dir / "m3_qa_results.jsonl"
    if not jsonl.exists():
        return None
    if not any(
        line.strip() for line in jsonl.read_text(encoding="utf-8").splitlines()
    ):
        return None
    summary_path = work_dir / "m3_eval_summary.json"
    digest = _judge_input_digest(jsonl, annotations_path, args)
    if summary_path.exists() and not args.force:
        sidecar = work_dir / "m3_eval_summary.digest.json"
        recorded = None
        try:
            recorded = json.loads(sidecar.read_text(encoding="utf-8")).get("digest")
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        if recorded == digest:
            try:
                return {
                    "status": "reused",
                    "returncode": 0,
                    "summary_path": str(summary_path),
                    "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                }
            except (OSError, json.JSONDecodeError):
                logger.warning(
                    "[%s] per-video judge summary unreadable — re-judging", video_id
                )
        else:
            logger.warning(
                "[%s] cached judge summary is stale (predictions, annotations "
                "or judge configuration changed since it was written) — "
                "re-judging", video_id,
            )
    # Reuse the existing judge driver, scoped to this one video: the
    # ``--video`` filter keeps m3.eval from scoring other videos' questions.
    result = run_judge(
        annotations=annotations_path,
        merged_jsonl=jsonl,
        output_dir=work_dir,
        args=args,
        subset=[video_id],
    )
    if result.get("returncode") == 0 and result.get("summary_path"):
        _write_judge_digest(
            work_dir, digest, jsonl=jsonl, annotations=annotations_path
        )
    return result


def merge_per_video_judges(
    subset: Iterable[str],
    work_dir_root: Path,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Merge the per-video judge rows into one ``_aggregate`` summary.

    No second judge pass: the final ``m3_eval_summary.json`` is recomputed
    from the per-video ``m3_eval_results.jsonl`` rows via ``m3.eval.aggregate``
    (the same aggregation the judge itself uses), so the result is exactly
    what a single judge pass over the merged JSONL would produce.
    """
    from m3.eval import QuestionResult, aggregate

    rows: list[QuestionResult] = []
    judge_provider = judge_model = ""
    for vid in subset:
        summary_path = work_dir_root / vid / "m3_eval_summary.json"
        results_path = work_dir_root / vid / "m3_eval_results.jsonl"
        if summary_path.exists():
            try:
                per_video = json.loads(summary_path.read_text(encoding="utf-8"))
                judge_provider = per_video.get("judge_provider") or judge_provider
                judge_model = per_video.get("judge_model") or judge_model
            except (OSError, json.JSONDecodeError):
                logger.warning("[%s] unreadable judge summary skipped in merge", vid)
        if not results_path.exists():
            continue
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(QuestionResult(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("[%s] malformed judge row skipped: %s", vid, exc)
    if not rows:
        return None
    merged = aggregate(
        rows, judge_provider=judge_provider, judge_model=judge_model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m3_eval_results.jsonl").write_text(
        "".join(
            json.dumps(r.to_jsonl(), ensure_ascii=False) + "\n" for r in rows
        ),
        encoding="utf-8",
    )
    merged["merge_of_per_video_judges"] = True
    summary_path = output_dir / "m3_eval_summary.json"
    summary_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    merged["summary_path"] = str(summary_path)
    return merged


def run_videos(
    *,
    subset: list[str],
    args: argparse.Namespace,
    annotations: dict[str, Any],
    annotations_path: Path,
    project_root: Path,
    videos_root: Path,
    on_video_done: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the per-video flow for every video in ``subset`` (WS 2.5).

    ``--video-workers 1`` (default) runs the exact serial loop of the old
    driver. With ``--video-workers N > 1`` one thread per video runs the
    full per-video flow — pipeline subprocess, then the live per-video
    judge when ``--judge`` is set — while only the shared aggregate state
    (the summaries map / run report via ``on_video_done``) is touched under
    a lock. Returns summaries in ``subset`` order regardless of completion
    order.
    """
    summaries: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def _worker(video_id: str) -> dict[str, Any]:
        try:
            declared = annotations[video_id].get("video_path", "")
            video_path = resolve_video_path(
                declared, project_root=project_root, videos_root=videos_root
            )
            if video_path is None:
                logger.error(
                    "[%s] cannot resolve video_path=%r under %s",
                    video_id, declared, videos_root,
                )
                return {
                    "video_id": video_id,
                    "status": "failed",
                    "reason": "video_not_found",
                    "declared_video_path": declared,
                }
            expected_qids = {
                str(q.get("question_id"))
                for q in load_questions_for_video(annotations, video_id)
                if q.get("question_id") is not None
            }
            try:
                summary = run_one_video(
                    video_id=video_id,
                    video_path=video_path,
                    args=args,
                    annotations=annotations,
                    expected_qids=expected_qids,
                )
            except Exception as exc:  # pragma: no cover - top-level guard
                logger.exception("[%s] driver raised: %s", video_id, exc)
                summary = {
                    "video_id": video_id,
                    "status": "failed",
                    "reason": "driver_exception",
                    "error": str(exc),
                }
            if not isinstance(summary, dict):
                summary = {"raw_result": summary}
            summary.setdefault("video_id", video_id)
            if args.judge:
                # Live per-video judge: score this video's predictions
                # immediately, don't wait for a final pass.
                try:
                    judge_result = live_judge_video(
                        video_id=video_id,
                        work_dir=Path(args.work_dir_root) / video_id,
                        args=args,
                        annotations_path=annotations_path,
                    )
                except Exception as exc:  # pragma: no cover - best-effort
                    logger.warning("[%s] live judge failed: %s", video_id, exc)
                    judge_result = {
                        "status": "failed",
                        "returncode": 1,
                        "error": str(exc),
                    }
                if judge_result is not None:
                    summary["judge"] = judge_result
            return summary
        except Exception as exc:  # pragma: no cover - top-level guard
            logger.exception("[%s] worker raised: %s", video_id, exc)
            return {
                "video_id": video_id,
                "status": "failed",
                "reason": "driver_exception",
                "error": str(exc),
            }

    def _record(summary: dict[str, Any]) -> None:
        with lock:
            summaries[summary["video_id"]] = summary
            if on_video_done is not None:
                on_video_done(summary)

    workers = max(1, getattr(args, "video_workers", 1) or 1)
    if workers <= 1:
        for video_id in subset:
            _record(_worker(video_id))
    else:
        logger.info("video-parallel run: %d workers over %d videos",
                    workers, len(subset))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_worker, video_id): video_id for video_id in subset
            }
            for future in as_completed(futures):
                _record(future.result())

    # Deterministic: subset order, independent of completion order.
    return [summaries[vid] for vid in subset if vid in summaries]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the M3 multi-modal pipeline over a subset of "
            "M3-Bench videos (default annotations: data/m3/annotations/robot.json)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotations",
                   default="data/m3/annotations/robot.json",
                   help="Path to M3-Bench annotations JSON (e.g. robot.json).")
    p.add_argument("--videos", default=None,
                   help="Comma-separated list of video_ids to run. "
                        "Default: all (subject to --limit and --exclude).")
    p.add_argument("--exclude", default=None,
                   help="Comma-separated list of video_ids to skip.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the subset size when --videos is not given.")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Uniformly sample N videos with questions, without replacement. "
                        "Incompatible with --videos/--limit; respects --exclude.")
    p.add_argument("--seed", type=int, default=42,
                   help="Local RNG seed for --sample-size.")
    p.add_argument("--selection-only", action="store_true",
                   help="With --sample-size, validate video paths and save sample_manifest.json; "
                        "do not load runtime config or run pipeline/judge.")
    p.add_argument("--work-dir-root", default="data/m3_runs/m3bench_subset",
                   help="Root directory; each video gets <work-dir-root>/<video_id>/.")
    p.add_argument("--videos-root", default="data/m3/videos",
                   help="Root directory passed to m3.config.resolve_video_path. "
                        "The M3-Bench robot.json declares paths like "
                        "'data/videos/robot/<id>.mp4' but on this layout the "
                        "actual files live under 'data/m3/videos/robot/' "
                        "(data/m3 is a symlink to /data3/m3); override when "
                        "running elsewhere.")
    p.add_argument("--pipeline",
                   default="config/m3/pipelines/video_episodic_full.yaml",
                   help="Pipeline YAML spec to run per video.")
    p.add_argument("--config", default="deploy/ogmemory.yaml",
                   help="Path to ogmemory.yaml (LLM/embedding defaults). "
                        "Used both as the subprocess --config and as the "
                        "source of LLM API keys for the env injection.")
    p.add_argument("--seconds", type=int, default=120,
                   help="Trim each video to its first N seconds (<=0 = full).")
    p.add_argument("--allow-out-of-window-questions", action="store_true",
                   help="Allow question timestamps beyond --seconds. By default "
                        "reject this mismatch; use smoke annotations or --seconds 0.")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="Required when --graph-store=sql.")
    p.add_argument("--account-id", default="m3bench")
    p.add_argument("--enable-stage", action="append", default=[],
                   help="Forwarded to run_video_pipeline --enable-stage (repeatable).")
    p.add_argument("--disable-stage", action="append", default=[],
                   help="Forwarded to run_video_pipeline --disable-stage (repeatable).")
    p.add_argument("--set", action="append", default=[],
                   help="Forwarded to run_video_pipeline --set (repeatable).")
    p.add_argument("--qa-max-iterations", type=int, default=6)
    p.add_argument("--qa-timeout", type=int, default=90)
    p.add_argument("--qa-top-k", type=int, default=None,
                   help="Forwarded as qa.params.top_k. The QA LLM provider / "
                        "model / API key come from m3.qa_llm in --config; "
                        "to override the provider for one run use "
                        "--set qa.params.llm=openai.")
    p.add_argument("--force", action="store_true",
                   help="Rerun videos even if their JSONL looks complete.")
    p.add_argument("--reuse-graph", action="store_true",
                   help="When --graph-store is sql, query the graph for "
                        "completeness before each video. If every VIDEO "
                        "clip already has a completed episodic_memory_generator "
                        "ExtractedNode (with embedding), skip the heavy "
                        "extraction stages and run QA-only against the "
                        "existing graph. Requires --graph-store sql.")
    p.add_argument("--resume", action="store_true",
                   help="Forward --resume to m3.eval during the judge pass.")
    p.add_argument("--no-smart-resume", action="store_true",
                   help="Disable the per-stage probe layer (Layer 2) and "
                        "always run every stage from scratch. Forwarded "
                        "as --no-smart-resume to run_video_pipeline.")
    p.add_argument("--video-workers", type=int, default=1,
                  help="Video-level parallelism: one thread per video, each "
                       "running the full per-video flow (pipeline subprocess "
                       "+ live per-video judge with --judge). Default 1 = the "
                       "old serial loop. Only the shared aggregate state "
                       "(run report / merged JSONL) is lock-protected.")
    p.add_argument("--judge", action="store_true",
                   help="After all videos, invoke m3.eval over the merged JSONL.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the merged JSONL + judge output "
                        "(default: <work-dir-root>/_aggregate).")
    p.add_argument("--judge-llm", default=None,
                   help="Provider label for the judge summary header "
                        "(e.g. 'minimax', 'openai', 'deepseek', 'vllm'). "
                        "Purely cosmetic — when unset, m3.eval derives "
                        "the label from --judge-base-url's hostname or "
                        "m3.judge_llm.provider in --config.")
    p.add_argument("--judge-model", default=None,
                   help="Model name for the judge LLM. Defaults to "
                        "m3.judge_llm.model in --config.")
    p.add_argument("--judge-base-url", default=None,
                   help="Base URL for the judge LLM. Defaults to "
                        "m3.judge_llm.base_url in --config.")
    p.add_argument("--judge-api-key", default=None,
                   help="API key for the judge LLM. Defaults to "
                        "m3.judge_llm.api_key or $OGMEM_MINIMAX_API_KEY.")
    p.add_argument("--python", default="uv run python",
                   help="Python invocation used for subprocesses. Defaults to "
                        "'uv run python' so the spawned pipeline inherits the "
                        "project venv (typing_extensions, faster_whisper, etc.). "
                        "Override with a path like '/abs/path/to/.venv/bin/python' "
                        "or a different launcher. Anything before the final token "
                        "is treated as the launcher (e.g. 'uv run') and tokens "
                        "after are passed through as a single argv.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print each per-video command and exit without running.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.selection_only and args.sample_size is None:
        p.error("--selection-only requires --sample-size")
    if args.sample_size is not None and (args.videos or args.limit is not None):
        p.error("--sample-size cannot be combined with --videos or --limit")
    if args.video_workers < 1:
        p.error("--video-workers must be >= 1")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    annotations_path = Path(args.annotations).resolve()
    if not annotations_path.exists():
        logger.error("annotations not found: %s", annotations_path)
        return 1
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d video entries from %s",
                len(annotations), annotations_path)

    args.annotations = annotations_path  # normalize to absolute
    args.config = Path(args.config).resolve()
    args.work_dir_root = Path(args.work_dir_root).resolve()
    args.work_dir_root.mkdir(parents=True, exist_ok=True)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else args.work_dir_root / "_aggregate"
    )

    try:
        subset = select_subset(
            annotations,
            include=parse_csv(args.videos),
            exclude=parse_csv(args.exclude),
            limit=args.limit,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    except ValueError as exc:
        logger.error("Invalid sample: %s", exc)
        return 1
    if not subset:
        logger.error("subset is empty after applying --videos/--exclude/--limit")
        return 1
    logger.info("Running subset (%d videos): %s", len(subset), subset)

    try:
        outside = validate_annotation_duration(annotations, subset, args.seconds)
    except ValueError as exc:
        logger.error("Invalid annotation timestamp: %s", exc)
        return 1
    if outside:
        message = (
            f"{len(outside)} question(s) in {annotations_path} occur after "
            f"--seconds {args.seconds}: {', '.join(outside[:10])}. "
            "Use --annotations with a matching smoke question set, or "
            "--seconds 0 for the full video. To intentionally evaluate "
            "incomplete evidence, pass --allow-out-of-window-questions."
        )
        if not args.allow_out_of_window_questions:
            logger.error(message)
            return 1
        logger.warning(message)

    project_root = REPO_ROOT
    videos_root = (project_root / args.videos_root).resolve() \
        if not Path(args.videos_root).is_absolute() \
        else Path(args.videos_root).resolve()

    sample_manifest = None
    if args.sample_size is not None:
        # Fail before launching ANY video. Do not silently resample missing
        # files: doing so changes the population and biases comparisons.
        missing = [
            vid for vid in subset
            if resolve_video_path(
                annotations[vid].get("video_path", ""),
                project_root=project_root, videos_root=videos_root,
            ) is None
        ]
        if missing:
            logger.error("Selected videos missing on disk (no resampling): %s", missing)
            return 1
        try:
            sample_manifest = save_sample_manifest(
                args.work_dir_root / "sample_manifest.json", annotations, subset,
                seed=args.seed, exclude=parse_csv(args.exclude),
            )
        except (ValueError, OSError) as exc:
            logger.error("Cannot pin sample: %s", exc)
            return 1
        logger.info("Pinned %d videos / %d questions in %s",
                    len(subset), sample_manifest["total_questions"],
                    args.work_dir_root / "sample_manifest.json")
        if args.selection_only:
            print(",".join(subset))
            return 0
    if not args.config.exists():
        logger.error("config not found: %s", args.config)
        return 1

    # WS 2.5: video-parallel run with incremental report refresh. After
    # each video finishes (any completion order) the merged JSONL and
    # run_report.json are rebuilt under the shared lock, so a crash
    # mid-run leaves a consistent, resumable report at video granularity.
    merged_jsonl = output_dir / "all_m3_qa_results.jsonl"
    run_report_path = output_dir / "run_report.json"
    run_report_path.parent.mkdir(parents=True, exist_ok=True)

    def _report_dict(summaries_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            "annotations": str(annotations_path),
            "pipeline": str(Path(args.pipeline).resolve()),
            "seconds": args.seconds,
            "graph_store": args.graph_store,
            "subset": subset,
            "sample": sample_manifest,
            "per_video": [
                summaries_map[vid] for vid in subset if vid in summaries_map
            ],
            "aggregate": aggregate_jsonl(subset, args.work_dir_root, merged_jsonl),
        }

    summaries_map: dict[str, dict[str, Any]] = {}

    def _on_video_done(summary: dict[str, Any]) -> None:
        summaries_map[summary["video_id"]] = summary
        run_report_path.write_text(
            json.dumps(
                _report_dict(summaries_map), ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )

    summaries = run_videos(
        subset=subset,
        args=args,
        annotations=annotations,
        annotations_path=annotations_path,
        project_root=project_root,
        videos_root=videos_root,
        # A dry run must not truncate existing aggregate results.
        on_video_done=None if args.dry_run else _on_video_done,
    )

    if args.dry_run:
        # Preview must not truncate existing aggregate results or run a judge.
        return 2 if any(s.get("status") == "failed" for s in summaries) else 0

    agg = aggregate_jsonl(subset, args.work_dir_root, merged_jsonl)
    report = _report_dict({s["video_id"]: s for s in summaries})
    run_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Run report: %s", run_report_path)
    logger.info("Merged JSONL: %s (%d records, %d answered, %d failed)",
                merged_jsonl, agg["total_records"], agg["answered"], agg["failed"])

    if agg["total_records"] == 0:
        report["status"] = "failed"
        report["reason"] = "no_qa_results"
        report["judge"] = {"status": "skipped", "reason": "no_qa_results"}
        run_report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        logger.error("No QA results; experiment failed. Judge was not run.")
        return 2

    if args.judge and not args.dry_run:
        # WS 2.5: the live per-video judges already ran inside run_videos;
        # the final _aggregate summary is a merge of their rows — no second
        # judge pass.
        judge_summary = merge_per_video_judges(
            subset, args.work_dir_root, output_dir
        )
        report["judge"] = judge_summary or {
            "status": "skipped",
            "reason": "no_per_video_judges",
        }
        run_report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    failed = [s for s in summaries if s.get("status") == "failed"]
    judge_failed = any(
        (s.get("judge") or {}).get("returncode", 0) != 0 for s in summaries
    )
    report["status"] = "failed" if failed or judge_failed else "completed"
    run_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return 2 if failed or judge_failed else 0


if __name__ == "__main__":
    sys.exit(main())
