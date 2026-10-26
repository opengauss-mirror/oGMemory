"""M3-Bench evaluation harness.

Drives :class:`OGMemoryControlAdapter` over every QA in an M3-Bench
annotations JSON, captures the adapter's answer and per-round
search history, then asks a judge LLM (default: MiniMax-M3 — see
``m3.judge_llm`` in ``ogmemory.yaml``) to score the predicted answer
against the ground truth. Results land in two files:

* ``m3_eval_results.jsonl`` — one row per question with predicted /
  expected answers, judge score, search history, latency.
* ``m3_eval_summary.json`` — aggregate counts and per-question-type
  buckets (mean score, p50 iterations, count).

The harness is resumable: re-running with ``--resume`` skips any
``(video_id, question_id)`` already present in the results file. This
matters for the 1276-QA × 5-rounds evaluation budget — partial runs
can be picked up without recomputing.

To re-judge an existing QA pass without re-running the adapter, pass
``--input-qa-results <jsonl>``. The judge-only path skips the embedder
/ answer-LLM / graph entirely and only invokes the judge on the
supplied predictions. Useful when the adapter run took hours and you
want to swap in a different judge model.

Usage::

    # Bench-scale run, no OpenAI key required — judge defaults to MiniMax-M3:
    python -m m3.eval \
        --annotations data/annotations/robot.json \
        --graph-store sql --connection-string "$GRAPH_DSN" \
        --embedder openai --llm minimax \
        --output-dir data/m3_eval/2026-07-22

    # …or DeepSeek (any OpenAI-compatible endpoint works — just point at it):
    python -m m3.eval \
        --annotations data/annotations/robot.json \
        --judge-base-url https://api.deepseek.com/v1 \
        --judge-model deepseek-chat \
        --judge-api-key "$DEEPSEEK_API_KEY" \
        --output-dir data/m3_eval/2026-07-22

    # …or a local vLLM / Ollama / LM Studio:
    python -m m3.eval \
        --annotations data/annotations/robot.json \
        --judge-base-url http://localhost:8000/v1 \
        --judge-model Qwen2.5-7B-Instruct \
        --judge-api-key dummy \
        --output-dir data/m3_eval/2026-07-22

    # Re-judge existing QA results with a different judge (no adapter run):
    python -m m3.eval \
        --annotations data/annotations/robot.json \
        --input-qa-results work/m3_qa_results.jsonl \
        --judge-base-url https://api.deepseek.com/v1 \
        --judge-model deepseek-chat \
        --judge-api-key "$DEEPSEEK_API_KEY" \
        --output-dir data/m3_eval/rejudge_deepseek

Offline smoke / tests::

    python -m m3.eval --annotations tests.json \\
        --graph-store memory --embedder mock --llm mock
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID
from m3.control_adapter import M3TaskContext, OGMemoryControlAdapter
from m3.index_episodic import build_embedder
from m3.qa import (
    ANSWER_SCHEMA,
    _coerce_before_clip,
    build_llm,
    iter_annotation_questions,
    load_questions,
)
from m3.temporal import add_temporal_arguments, annotation_scope

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": (
                "1.0 if predicted answer is correct, 0.5 if partially "
                "correct, 0.0 if wrong or missing."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "Brief explanation of why the predicted answer received "
                "the given score."
            ),
        },
    },
    "required": ["score", "reasoning"],
}


JUDGE_PROMPT_TEMPLATE = """You are an exacting evaluator for M3-Bench long-video QA.

The predicted answer and the expected (ground truth) answer are below.
Score the predicted answer:

* 1.0 — predicted answer is essentially correct (matches expected or
  conveys the same fact, possibly with minor wording differences)
* 0.5 — predicted answer is partially correct (right idea but wrong
  details, or missing a key part)
* 0.0 — predicted answer is wrong, contradicts the expected answer,
  or says "insufficient evidence"

Return JSON with keys: score (0/0.5/1) and reasoning.

Question: {question}
Expected answer: {expected}
Predicted answer: {predicted}
"""


def build_judge_llm(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    config_path: str | None = None,
):
    """Construct the judge LLM from CLI args + ``m3.judge_llm`` in YAML.

    Backwards-compatible wrapper around
    :func:`m3.llm_factory.build_judge_llm`. The canonical dispatch lives
    in the factory so QA / episodic / judge roles share the same
    provider logic. Kept under the old name because the eval CLI passes
    keyword args through to this function directly.
    """
    from m3.llm_factory import build_judge_llm as _factory_judge
    return _factory_judge(
        model=model,
        base_url=base_url,
        api_key=api_key,
        config_path=config_path,
    )


def judge_answer(
    *,
    judge_llm,
    question: str,
    expected: str,
    predicted: str,
) -> tuple[float, str, bool]:
    """Ask the judge LLM to score the predicted answer.

    Returns ``(score, reasoning, judge_error)``. ``judge_error`` is True
    when the judge LLM itself failed (network error, bad key, malformed
    JSON); otherwise False even when the score is 0.0 because the model
    genuinely judged the answer wrong. Falls back to ``(0.0, "<reason>",
    True)`` on judge failure — never raises, so a flaky judge can't
    abort the whole eval run. The ``judge_error`` flag lets the summary
    distinguish "model wrong" from "judge broken" — otherwise a
    misconfigured judge key would silently produce a fake 0% accuracy.
    """
    if not predicted or not predicted.strip():
        # Empty predicted is an adapter-side failure (model ran out of
        # iterations, hit a tool error, etc.) — the *adapter* sets
        # ``qr.error`` and ``qr.answered=False`` for that. From the
        # judge's perspective there's nothing to grade, but this is
        # NOT a judge-side failure: the judge never ran. Keep
        # ``judge_error=False`` so the bench summary's judge-error
        # counter stays a clean signal of "the judge itself broke".
        return 0.0, "predicted answer is empty", False
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        expected=expected or "(no expected answer)",
        predicted=predicted,
    )
    try:
        result = judge_llm.complete_json(prompt, JUDGE_SCHEMA)
    except Exception as exc:  # noqa: BLE001 — judge is best-effort
        logger.warning("Judge LLM failed: %s", exc)
        return 0.0, f"judge failed: {exc}", True
    raw_score = result.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return 0.0, f"judge returned non-numeric score: {raw_score!r}", True
    # Normalise to the {0.0, 0.5, 1.0} set the docstring promises.
    if score >= 0.75:
        score = 1.0
    elif score >= 0.25:
        score = 0.5
    else:
        score = 0.0
    reasoning = result.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = ""
    return score, reasoning.strip(), False


# ---------------------------------------------------------------------------
# Per-question result schema
# ---------------------------------------------------------------------------

@dataclass
class QuestionResult:
    video_id: str
    question_id: str
    question_index: int
    question: str
    expected_answer: str
    predicted_answer: str
    type: str = ""
    timestamp: Any = None
    before_clip: Any = None
    iterations: int = 0
    finish_reason: str = ""
    search_rounds: int = 0
    search_history: list[dict] = field(default_factory=list)
    search_stats: dict = field(default_factory=dict)
    temporal_scope: dict = field(default_factory=dict)
    judge_score: float = 0.0
    judge_reasoning: str = ""
    judge_error: bool = False
    latency_seconds: float = 0.0
    error: str = ""

    def to_jsonl(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-type aggregation
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[min(99, max(0, int(p) - 1))]


def aggregate(
    results: Iterable[QuestionResult],
    *,
    judge_provider: str,
    judge_model: str,
) -> dict:
    """Compute overall, per-type, and per-video metrics from a result stream.

    Per-video aggregation is what the bench-scale run needs: a single
    bad video shouldn't be allowed to drag the global mean without being
    visible in the summary. Judge-error counts are surfaced separately
    so a misconfigured judge API key can't masquerade as a 0% accuracy.
    """
    by_type: dict[str, list[QuestionResult]] = defaultdict(list)
    by_video: dict[str, list[QuestionResult]] = defaultdict(list)
    overall: list[QuestionResult] = []
    judge_error_count = 0
    for r in results:
        overall.append(r)
        if r.judge_error:
            judge_error_count += 1
        # type may be a string or list; coerce to a stable string.
        t = r.type
        if isinstance(t, list):
            t = ",".join(str(x) for x in t)
        elif not isinstance(t, str):
            t = str(t) if t else ""
        t = t.strip() or "(unspecified)"
        by_type[t].append(r)
        # video_id is required for the eval driver, but guard anyway.
        v = (r.video_id or "").strip() or "(unspecified)"
        by_video[v].append(r)

    def _bucket(rs: list[QuestionResult]) -> dict:
        if not rs:
            return {"count": 0}
        scores = [r.judge_score for r in rs]
        iters = [float(r.iterations) for r in rs]
        lats = [r.latency_seconds for r in rs]
        return {
            "count": len(rs),
            "answered": sum(1 for r in rs if not r.error),
            "failed": sum(1 for r in rs if r.error),
            "mean_score": statistics.fmean(scores) if scores else 0.0,
            "p50_iterations": _percentile(iters, 50),
            "p95_iterations": _percentile(iters, 95),
            "mean_latency_seconds": statistics.fmean(lats) if lats else 0.0,
            "p95_latency_seconds": _percentile(lats, 95),
        }

    overall_bucket = _bucket(overall)
    return {
        "total_questions": len(overall),
        "answered": overall_bucket.get("answered", 0),
        "failed": overall_bucket.get("failed", 0),
        "judge_error_count": judge_error_count,
        "overall_mean_score": overall_bucket.get("mean_score", 0.0),
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "by_type": {t: _bucket(rs) for t, rs in sorted(by_type.items())},
        # Per-video rollup: one bucket per video_id, sorted by count desc
        # so the highest-volume videos surface first.
        "by_video": {
            v: _bucket(rs)
            for v, rs in sorted(
                by_video.items(),
                key=lambda kv: (-len(kv[1]), kv[0]),
            )
        },
    }


# ---------------------------------------------------------------------------
# Per-question driver
# ---------------------------------------------------------------------------

def answer_with_adapter(
    *,
    adapter: OGMemoryControlAdapter,
    question: str,
    video_id: str,
    question_id: str,
    before_clip=None,
    timestamp=None,
    temporal_policy="current_clip",
    lookahead_seconds=0.0,
) -> QuestionResult:
    """Run one question through the adapter. Captures the answer and
    search history into a QuestionResult; never raises."""
    ctx = RequestContext(
        account_id=M3BENCH_ACCOUNT_ID,
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-eval",
        trace_id=f"m3-eval-{question_id}",
    )
    started = time.monotonic()
    try:
        # Validate even if the caller supplied a custom adapter.
        annotation_scope(before_clip, timestamp, enabled=temporal_policy != "unrestricted")
        task_ctx = M3TaskContext(
            question=question, video_id=video_id,
            temporal_policy=temporal_policy, lookahead_seconds=lookahead_seconds,
            annotation_before_clip=before_clip, annotation_timestamp=timestamp,
        )
        result = adapter.run(ctx, task_ctx)
        predicted = (
            result.output[0]
            if result.output else ""
        )
        return QuestionResult(
            video_id=video_id,
            question_id=question_id,
            question_index=0,  # overwritten by caller
            question=question,
            expected_answer="",  # overwritten by caller
            predicted_answer=str(predicted).strip(),
            finish_reason=result.finish_reason,
            iterations=result.iterations,
            search_history=list(task_ctx.search_history),
            search_stats=dict(task_ctx.search_stats),
            temporal_scope=dict(task_ctx.temporal_scope),
            search_rounds=len(task_ctx.search_history),
            latency_seconds=time.monotonic() - started,
            error="" if result.completed else f"loop did not complete ({result.finish_reason})",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Adapter failed for %s/%s", video_id, question_id)
        return QuestionResult(
            video_id=video_id,
            question_id=question_id,
            question_index=0,
            question=question,
            expected_answer="",
            predicted_answer="",
            latency_seconds=time.monotonic() - started,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    annotations_path = Path(args.annotations).resolve()
    if not annotations_path.exists():
        logger.error("annotations not found: %s", annotations_path)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "m3_eval_results.jsonl"
    summary_path = output_dir / "m3_eval_summary.json"

    questions = load_questions(annotations_path)
    if args.video:
        allowed = set(args.video.split(","))
        questions = [q for q in questions if q["video_id"] in allowed]
    if args.type_filter:
        wanted = set(args.type_filter.split(","))
        questions = [
            q for q in questions
            if (q.get("type") if isinstance(q.get("type"), str)
                else ",".join(q.get("type") or []))
            in wanted
        ]
    if args.limit is not None:
        questions = questions[:args.limit]
    logger.info("Loaded %d questions", len(questions))

    # Build the judge LLM from the (base_url, model, api_key) triple.
    # Defaults to MiniMax-M3 so the bench-scale run doesn't need
    # OPENAI_API_KEY. Pass --judge-base-url to point at any other
    # OpenAI-compatible endpoint (DeepSeek, vLLM, etc.).
    judge_llm = build_judge_llm(
        model=args.judge_model,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        config_path=args.config,
    )
    # Derive the summary's (judge_provider, judge_model) labels from the
    # SAME effective values build_judge_llm resolved — i.e. CLI args
    # layered on top of ``m3.judge_llm`` in ogmemory.yaml. Pre-fix this
    # block only inspected ``args.judge_*`` (which default to None /
    # "MiniMax-M3"), so a YAML-configured judge silently reported as
    # the MiniMax defaults even though the build used the YAML triple —
    # making the summary header lie about which backend scored the run.
    from m3.config import load_m3_runtime_config
    judge_cfg = load_m3_runtime_config(args.config).judge_llm
    effective_base_url = args.judge_base_url or judge_cfg.base_url
    # ``--judge-provider`` lets callers override the resolved provider
    # label without touching base_url/model — useful when an operator
    # runs an internal vLLM but wants the summary labelled "internal".
    cli_provider = getattr(args, "judge_provider", None)
    if effective_base_url:
        from urllib.parse import urlparse
        host = (urlparse(effective_base_url).hostname or "").lower()
        judge_provider = host.removeprefix("api.").removesuffix(".com") or host
    elif cli_provider or judge_cfg.provider:
        judge_provider = cli_provider or judge_cfg.provider
    else:
        # Last-resort default — preserved for callers with neither YAML
        # nor CLI override (e.g. unit tests against the historic
        # MiniMax-M3 baseline).
        judge_provider = "minimax"
    judge_model = (
        args.judge_model or judge_cfg.model or "MiniMax-M3"
    )

    # Resume support: skip (video_id, question_id) pairs already on disk.
    done_keys: set[tuple[str, str]] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done_keys.add((str(rec.get("video_id")), str(rec.get("question_id"))))
        logger.info("Resume: skipping %d already-answered questions", len(done_keys))

    # Judge-only branch: skip the adapter/graph entirely and re-score
    # an existing predictions JSONL (typically the pipeline's
    # m3_qa_results.jsonl). Useful for swapping judge models without
    # paying for a fresh adapter run. Dispatched BEFORE building the
    # graph / embedder / answer LLM so those costs are not paid when
    # the caller only needs to rejudge existing predictions.
    if args.input_qa_results:
        return _run_judge_only(
            args=args,
            questions=questions,
            judge_llm=judge_llm,
            judge_provider=judge_provider,
            judge_model=judge_model,
            results_path=results_path,
            summary_path=summary_path,
            done_keys=done_keys,
        )

    # Build the graph service the adapter will query.
    from m3.load_clips import build_graph_service
    graph, store_kind = build_graph_service(args.graph_store, args.connection_string)
    logger.info("Graph store: %s", store_kind)

    # Build the adapter (embedder + answer LLM).
    embedder = build_embedder(
        args.embedder,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        dimension=args.embedding_dimension,
        config_path=args.config,
    )
    answer_llm = build_llm(
        args.llm,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        temperature=args.temperature,
        config_path=args.config,
    )
    from m3.context_budget import load_context_budget
    adapter = OGMemoryControlAdapter(
        llm=answer_llm,
        graph=graph,
        embedder=embedder,
        max_iterations=args.max_iterations,
        no_gain_limit=getattr(args, "no_gain_limit", 3) or None,
        candidate_k=getattr(args, "candidate_k", None),
        transcript_top_k=getattr(args, "transcript_top_k", 0),
        transcript_fusion=getattr(args, "transcript_fusion", "rrf"),
        context_k=getattr(args, "context_k", 0),
        include_neighbors=getattr(args, "include_neighbors", 1),
        timeout_seconds=args.timeout_seconds,
        default_video_id=args.default_video_id,
        context_budget=load_context_budget(args.config),
    )

    mode = "a" if args.resume else "w"
    all_results: list[QuestionResult] = []
    answered = failed = skipped = 0
    with results_path.open(mode, encoding="utf-8") as f:
        for q in questions:
            key = (str(q["video_id"]), str(q["question_id"]))
            if key in done_keys:
                skipped += 1
                continue
            # Run the adapter on the question.
            qr = answer_with_adapter(
                adapter=adapter,
                question=str(q["question"]),
                video_id=str(q["video_id"]),
                question_id=str(q["question_id"]),
                before_clip=q.get("before_clip"),
                timestamp=q.get("timestamp"),
                temporal_policy=getattr(args, "temporal_policy", "current_clip"),
                lookahead_seconds=getattr(args, "lookahead_seconds", 0.0),
            )
            qr.question_index = q.get("question_index", 0)
            qr.expected_answer = str(q.get("answer") or "")
            qr.type = (
                q.get("type") if isinstance(q.get("type"), str)
                else ",".join(q.get("type") or [])
            )
            qr.timestamp = q.get("timestamp")
            qr.before_clip = q.get("before_clip")

            # Judge predicted vs expected.
            score, reasoning, judge_error = judge_answer(
                judge_llm=judge_llm,
                question=str(q["question"]),
                expected=str(q.get("answer") or ""),
                predicted=qr.predicted_answer,
            )
            qr.judge_score = score
            qr.judge_reasoning = reasoning
            qr.judge_error = judge_error
            if not qr.error:
                answered += 1
            else:
                failed += 1

            f.write(json.dumps(qr.to_jsonl(), ensure_ascii=False) + "\n")
            f.flush()
            all_results.append(qr)

    # Write the summary.
    summary = aggregate(all_results, judge_provider=judge_provider, judge_model=judge_model)
    summary["answered"] = answered
    summary["failed"] = failed
    summary["skipped"] = skipped
    summary["annotations_path"] = str(annotations_path)
    summary["results_path"] = str(results_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info(
        "Eval wrote %d results (%d answered, %d failed, %d skipped) → %s",
        len(all_results), answered, failed, skipped, results_path,
    )
    logger.info(
        "Overall mean score: %.3f — summary at %s",
        summary["overall_mean_score"], summary_path,
    )
    return 2 if failed and answered == 0 else 0


def _load_predictions_jsonl(path: Path) -> dict[tuple[str, str], dict]:
    """Read a JSONL of pre-existing QA predictions into a lookup keyed by
    ``(video_id, question_id)``.

    Accepts both the modern pipeline schema (``predicted_answer`` /
    ``expected_answer``) and the legacy ``m3.qa`` schema (``answer``).
    Silently skips malformed lines so a single bad row doesn't kill
    the rejudge.
    """
    rows: dict[tuple[str, str], dict] = {}
    bad = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            key = (str(rec.get("video_id", "")), str(rec.get("question_id", "")))
            if not key[0] or not key[1]:
                bad += 1
                continue
            rows[key] = rec
    if bad:
        logger.warning("Skipped %d malformed lines in %s", bad, path)
    return rows


def _run_judge_only(
    *,
    args: argparse.Namespace,
    questions: list[dict],
    judge_llm,
    judge_provider: str,
    judge_model: str,
    results_path: Path,
    summary_path: Path,
    done_keys: set[tuple[str, str]],
) -> int:
    """Re-judge ``--input-qa-results`` without re-running the adapter.

    For each filtered question, look up its row in the input JSONL,
    build a ``QuestionResult`` from the row's predicted_answer (and
    metadata), call the judge, and append to ``m3_eval_results.jsonl``.
    Questions missing from the input get a row with ``predicted_answer="\"\"``
    (and ``judge_error=False`` — see the empty-prediction contract in
    ``judge_answer``). Honors ``--resume``: keys already present in
    the output file are skipped.
    """
    input_path = Path(args.input_qa_results).resolve()
    if not input_path.exists():
        logger.error("--input-qa-results not found: %s", input_path)
        return 1
    predictions = _load_predictions_jsonl(input_path)
    logger.info(
        "Loaded %d pre-existing predictions from %s", len(predictions), input_path,
    )

    mode = "a" if args.resume else "w"
    all_results: list[QuestionResult] = []
    answered = failed = skipped = 0
    missing = 0
    with results_path.open(mode, encoding="utf-8") as f:
        for q in questions:
            key = (str(q["video_id"]), str(q["question_id"]))
            if key in done_keys:
                skipped += 1
                continue
            row = predictions.get(key)
            # Annotation fields win over the input JSONL for question /
            # expected_answer / type — the annotation is the canonical
            # source for the ground truth. The input row is the
            # canonical source for predicted_answer / search history /
            # adapter metadata.
            question_text = str(q.get("question") or "")
            expected = str(q.get("answer") or "")
            q_type = (
                q.get("type") if isinstance(q.get("type"), str)
                else ",".join(q.get("type") or [])
            )
            qr = QuestionResult(
                video_id=str(q["video_id"]),
                question_id=str(q["question_id"]),
                question_index=q.get("question_index", 0),
                question=question_text,
                expected_answer=expected,
                predicted_answer="",
                type=q_type,
                timestamp=q.get("timestamp"),
                before_clip=q.get("before_clip"),
            )
            if row is not None:
                qr.temporal_scope = dict(row.get("temporal_scope") or {})
                qr.predicted_answer = str(
                    row.get("predicted_answer") or row.get("answer") or ""
                ).strip()
                # Carry adapter-side metadata through if present in the row.
                qr.iterations = int(row.get("iterations") or 0)
                qr.finish_reason = str(row.get("finish_reason") or "")
                rounds = row.get("search_rounds")
                if rounds is None and row.get("search_history"):
                    rounds = len(row["search_history"])
                qr.search_rounds = int(rounds or 0)
                qr.search_history = list(row.get("search_history") or [])
                qr.search_stats = dict(row.get("search_stats") or {})
                qr.error = str(row.get("error") or "")
            else:
                missing += 1
                qr.error = "missing_prediction"
                logger.warning(
                    "no prediction for %s/%s — writing empty row", key[0], key[1],
                )

            # Judge predicted vs expected (predicted_answer may be empty
            # if the input row was missing — that's an adapter-side
            # failure, judge_error=False, see judge_answer).
            started = time.monotonic()
            score, reasoning, judge_error = judge_answer(
                judge_llm=judge_llm,
                question=question_text,
                expected=expected,
                predicted=qr.predicted_answer,
            )
            qr.judge_score = score
            qr.judge_reasoning = reasoning
            qr.judge_error = judge_error
            # In judge-only mode the only latency we have is the judge
            # call itself; the adapter latency lives in the source row
            # and isn't surfaced here.
            qr.latency_seconds = time.monotonic() - started
            if not qr.error:
                answered += 1
            else:
                failed += 1

            f.write(json.dumps(qr.to_jsonl(), ensure_ascii=False) + "\n")
            f.flush()
            all_results.append(qr)

    summary = aggregate(all_results, judge_provider=judge_provider, judge_model=judge_model)
    summary["answered"] = answered
    summary["failed"] = failed
    summary["skipped"] = skipped
    summary["missing_from_input"] = missing
    summary["status"] = "failed" if failed else "completed"
    summary["prediction_coverage"] = (
        (len(all_results) - missing) / len(all_results) if all_results else 0.0
    )
    summary["input_qa_results"] = str(input_path)
    summary["annotations_path"] = str(args.annotations)
    summary["results_path"] = str(results_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info(
        "Judge-only eval wrote %d results (%d answered, %d failed, %d "
        "skipped, %d missing from input) → %s",
        len(all_results), answered, failed, skipped, missing, results_path,
    )
    logger.info(
        "Overall mean score: %.3f — summary at %s",
        summary["overall_mean_score"], summary_path,
    )
    return 2 if failed else 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run M3-Bench evaluation: drives OGMemoryControlAdapter, "
                    "judges via GPT-4o, aggregates per question type.",
    )
    p.add_argument("--annotations", required=True,
                   help="Path to M3-Bench annotation JSON")
    p.add_argument("--output-dir", default="data/m3_eval/run",
                   help="Directory for results.jsonl + summary.json")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--config", default=None,
                   help="ogmemory.yaml path (default: deploy/ogmemory.yaml or $OGMEM_CONFIG)")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID)
    # Filtering
    p.add_argument("--video", default=None, help="Comma-separated video_ids to evaluate")
    p.add_argument("--type-filter", default=None,
                   help="Comma-separated question-type values to include")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip (video_id, question_id) pairs already in results.jsonl")
    p.add_argument("--input-qa-results", default=None,
                   help="Skip the adapter and re-judge an existing predictions "
                        "JSONL (typically m3_qa_results.jsonl from the pipeline). "
                        "Honors --video / --type-filter / --limit / --resume.")
    # Adapter config
    add_temporal_arguments(p)
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--include-neighbors", type=int, default=1)
    p.add_argument("--context-k", type=int, default=0,
                   help="Separate exact-neighbor evidence budget (0 preserves the existing result budget)")
    p.add_argument("--transcript-top-k", type=int, default=0,
                   help="Independent transcript candidates (0 disables; requires scoped memory/SQL index)")
    p.add_argument("--transcript-fusion", choices=("rrf", "weighted"), default="rrf",
                   help="Experimental vector/transcript fusion; effective only when transcript recall is enabled")
    p.add_argument("--candidate-k", type=int, default=None,
                   help="Vector candidates per query branch; promoted to at least top-k (default: legacy depth)")
    p.add_argument("--no-gain-limit", type=int, default=3,
                   help="Consecutive searches without new evidence before answer-only mode (>=2; 0 disables)")
    p.add_argument("--timeout-seconds", type=float, default=60.0)
    p.add_argument("--default-video-id", default=None,
                   help="Default video scope for [Search] when question.video_id is missing")
    # Embedder (for [Search])
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--embedding-dimension", type=int, default=None)
    # Answer LLM
    p.add_argument("--llm", choices=("mock", "openai", "minimax"), default=None)
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    # Judge LLM — pick any OpenAI-compatible endpoint via (base_url,
    # model, api_key). Defaults to MiniMax-M3 (no OpenAI key required)
    # when none of these are set. The flags are intentionally short
    # (``--judge-*`` rather than ``--judge-llm-*``) so they don't
    # prefix-collide with the answer LLM flags below.
    p.add_argument("--judge-base-url", default=None,
                   help="Base URL of the judge LLM. Defaults to MiniMax.")
    p.add_argument("--judge-api-key", default=None,
                   help="API key for the judge LLM. Defaults to "
                        "$OGMEM_MINIMAX_API_KEY.")
    p.add_argument("--judge-model", default=None,
                   help="Model name for the judge LLM. Defaults to "
                        "MiniMax-M3 when --judge-base-url is unset.")
    p.add_argument("--judge-provider", default=None,
                   help="Provider label for the summary header (e.g. "
                        "'minimax', 'openai', 'deepseek', 'mock'). "
                        "Defaults to m3.judge_llm.provider in "
                        "ogmemory.yaml; only used when no effective "
                        "base_url is resolved (so it doesn't override a "
                        "hostname-derived label).")
    # Verbosity
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.context_k < 0 or args.include_neighbors < 0:
        p.error("--context-k and --include-neighbors must be >= 0")
    if args.context_k and not args.include_neighbors:
        p.error("--context-k requires --include-neighbors > 0")
    if args.transcript_top_k < 0:
        p.error("--transcript-top-k must be >= 0")
    if args.candidate_k is not None and args.candidate_k <= 0:
        p.error("--candidate-k must be a positive integer")
    if args.no_gain_limit != 0 and args.no_gain_limit < 2:
        p.error("--no-gain-limit must be >= 2, or 0 to disable")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
