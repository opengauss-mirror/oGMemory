"""Vision-only M3-Bench QA MVP over episodic-memory retrieval.

This command reads M3-Bench annotations, retrieves relevant episodic evidence
for each question, and asks an LLM to answer directly. It intentionally avoids
full M3-Agent ReAct control for the MVP; the interface is structured so a
multi-round ``[Search]`` / ``[Answer]`` adapter can replace the direct answerer
later without changing result files.

Usage::

    python -m m3.qa \
        --annotations data/annotations/robot.json \
        --index data/episodic_index.jsonl \
        --output-jsonl data/m3_qa_results.jsonl \
        --llm openai

Offline smoke / tests::

    python -m m3.qa --annotations tests.json --index index.jsonl --llm mock
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from m3.config import M3BENCH_ACCOUNT_ID, video_id_from_path
from m3.index_episodic import build_embedder
from m3.search import EpisodicSearcher, Evidence, format_evidence_block
from m3.temporal import (
    add_temporal_arguments,
    resolve_question_scope,
    temporal_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

# Matches the literal tool tags the LLM might emit in its content stream
# (``[Search]`` / ``[Answer]``) instead of using the tool_calls channel.
# Some models echo the protocol text into the answer when they fail to
# format the tool call, which lands verbatim in ``predicted_answer`` via
# ``_parse_output``. These aren't real answers — flag them as invalid.
_PROTOCOL_SCAFFOLDING_RE = re.compile(r"\[(Search|Answer)\]", re.IGNORECASE)
_TOOL_CALL_LEAK_RE = re.compile(
    r"\b(?:Search|Answer)\s*\("
    r"|</?tool(?:s|_call)?(?:\s[^>]*)?>"
    r'|"name"\s*:\s*"(?:Search|Answer)"',
    re.IGNORECASE,
)


def _has_protocol_scaffolding(text: str) -> bool:
    """True if ``text`` contains the ``[Search]`` / ``[Answer]`` protocol tags.

    Used by both :func:`run_react_qa` (which marks the row as failed when
    producing it) and :func:`m3.pipeline.stages.qa_stage` (defense in
    depth — catches rows written by older drivers that didn't run the
    new check).
    """
    return bool(text) and bool(
        _PROTOCOL_SCAFFOLDING_RE.search(text) or _TOOL_CALL_LEAK_RE.search(text)
    )


_ANSWER_TAG_RE = re.compile(r"\[Answer\]", re.IGNORECASE)


def _salvage_scaffolding(text: str) -> str:
    """Recover the intended answer from a scaffolding-leaked string.

    The model occasionally emits its final answer as plain content —
    ``"[Answer] <text>"`` — in a tool-disabled round instead of calling
    the [Answer] tool. If the text after the LAST ``[Answer]`` tag (with
    any remaining protocol tags stripped) is non-trivial, that is the
    intended answer and should be kept rather than discarded.

    Returns ``""`` when there is nothing salvageable (e.g. the content
    was a stray ``[Search] …`` query with no ``[Answer]`` tag, or only
    punctuation remained).
    """
    matches = list(_ANSWER_TAG_RE.finditer(text))
    if not matches:
        return ""
    body = text[matches[-1].end():]
    if _TOOL_CALL_LEAK_RE.search(body):
        return ""
    cleaned = _PROTOCOL_SCAFFOLDING_RE.sub(" ", body)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n.-:|\\\"'")
    return cleaned


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "A concise final answer to the question, grounded only in the evidence.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of which evidence supports the answer.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence from 0 to 1.",
        },
    },
    "required": ["answer"],
}


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def iter_annotation_questions(annotations: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield normalized QA records from an M3-Bench annotation object."""
    for key, info in annotations.items():
        if not isinstance(info, dict):
            continue
        video_id = str(key)
        declared_path = info.get("video_path")
        if declared_path:
            video_id = video_id_from_path(str(declared_path))
        qa_list = info.get("qa_list") or info.get("questions") or []
        if not isinstance(qa_list, list):
            continue
        for idx, qa in enumerate(qa_list):
            if not isinstance(qa, dict):
                continue
            question = qa.get("question") or qa.get("query")
            if not question:
                continue
            qid = qa.get("question_id") or qa.get("id") or f"{video_id}_Q{idx:04d}"
            yield {
                "video_id": video_id,
                "question_id": qid,
                "question_index": idx,
                "question": str(question),
                "answer": qa.get("answer") or qa.get("gt_answer") or qa.get("ground_truth") or "",
                "type": qa.get("type") or qa.get("category") or [],
                "timestamp": qa.get("timestamp"),
                "before_clip": qa.get("before_clip"),
                "raw": qa,
            }


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected annotation object at {path}")
    return list(iter_annotation_questions(data))


# ---------------------------------------------------------------------------
# LLM answering
# ---------------------------------------------------------------------------

def build_llm(
    provider: str | None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    config_path: str | None = None,
):
    """Construct the answer LLM for QA from CLI + ogmemory.yaml.

    Backwards-compatible wrapper around :func:`m3.llm_factory.build_qa_llm`
    — the canonical dispatch lives there so QA / episodic / judge roles
    share the same provider logic. Kept under the old name because
    external callers (notably ``m3.eval``) do ``from m3.qa import build_llm``.
    """
    from m3.llm_factory import build_qa_llm
    return build_qa_llm(
        provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        config_path=config_path,
    )


def answer_question(
    *,
    llm,
    question: str,
    evidence: list[Evidence],
    video_id: str,
    temporal_scope: dict | None = None,
) -> dict[str, Any]:
    """Answer one question using direct retrieved evidence."""
    evidence_text = format_evidence_block(evidence)
    if not evidence_text:
        evidence_text = "(No evidence retrieved.)"
    prompt = f"""You answer questions about a long egocentric robot video.
Use ONLY the evidence clips below. If the evidence is insufficient, give the
best supported concise answer and mention uncertainty in the reasoning.

Video: {video_id}
Question: {question}
{temporal_prompt(temporal_scope)}

Evidence clips:
{evidence_text}

Return JSON with keys: answer, reasoning, confidence.
"""
    result = llm.complete_json(prompt, ANSWER_SCHEMA)
    from core.llm_output import final_text
    answer = final_text(result.get("answer"))
    if not isinstance(answer, str) or not answer.strip():
        answer = _fallback_answer_from_text(evidence_text)
    reasoning = result.get("reasoning") if isinstance(result.get("reasoning"), str) else ""
    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0 if not evidence else 0.5
    return {
        "answer": answer.strip(),
        "reasoning": reasoning.strip(),
        "confidence": float(confidence),
    }


def _fallback_answer_from_text(text: str) -> str:
    """Tiny guard for schema-incomplete mock / weak LLM outputs."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Description:"):
            return line.split(":", 1)[1].strip()
    return "Insufficient evidence."


def _coerce_before_clip(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_qa(args: argparse.Namespace) -> int:
    annotations_path = Path(args.annotations).resolve()
    if not annotations_path.exists():
        logger.error("annotations not found: %s", annotations_path)
        return 1

    # The legacy ``--index`` flag used to point at the
    # ``episodic_index.jsonl`` manifest. That file no longer carries
    # the per-record embedding (the canonical store is now the
    # OpenGaussVectorIndex), so the legacy driver reads from the
    # vector index instead. The flag is still accepted for backwards
    # compatibility — we log a deprecation warning so a stale
    # command line doesn't silently do the wrong thing.
    if getattr(args, "index", None):
        logger.warning(
            "--index is deprecated and ignored: m3.qa.run_qa now reads "
            "embeddings from the VectorIndex (m3.vector_index.* plus the "
            "shared vector_db.connection_string). The canonical retrieval "
            "source is the OpenGaussVectorIndex populated by "
            "m3.index_episodic."
        )

    questions = load_questions(annotations_path)
    if args.video:
        allowed = set(args.video.split(","))
        questions = [q for q in questions if q["video_id"] in allowed]
    if args.limit is not None:
        questions = questions[:args.limit]

    embedder = build_embedder(
        args.embedder,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
        dimension=args.embedding_dimension,
        config_path=args.config,
    )
    # Resolve the VectorIndex once and hand it to the searcher. When
    # ``--skip-vector-index`` is set (unit-test escape hatch) the
    # searcher falls back to in-memory records, which keeps the
    # legacy MVP driver testable without a live pgvector.
    if getattr(args, "skip_vector_index", False):
        searcher = EpisodicSearcher.from_records(
            records=[],
            embedder=embedder,
            hybrid_weight=args.hybrid_weight,
            account_id=args.account_id,
        )
    else:
        vi = _build_qa_vector_index(args)
        searcher = EpisodicSearcher.from_vector_index(
            vector_index=vi,
            embedder=embedder,
            hybrid_weight=args.hybrid_weight,
            account_id=args.account_id,
        )
    llm = build_llm(
        args.llm,
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        temperature=args.temperature,
        config_path=args.config,
    )

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from m3.qa_results import completed_keys
    done_keys = completed_keys(output_path) if args.resume else set()

    if args.resume:
        from m3.qa_results import separate_partial_line
        separate_partial_line(output_path)
    mode = "a" if args.resume else "w"
    answered = failed = skipped = 0
    with output_path.open(mode, encoding="utf-8") as f:
        for q in questions:
            key = (str(q["video_id"]), str(q["question_id"]))
            if key in done_keys:
                skipped += 1
                continue
            evidence = []
            temporal_scope = {}
            try:
                before_clip, before_sec, temporal_scope = resolve_question_scope(
                    q.get("before_clip"), q.get("timestamp"), enabled=args.respect_before_clip,
                    policy=getattr(args, "temporal_policy", "current_clip"),
                    lookahead_seconds=getattr(args, "lookahead_seconds", 0.0),
                    clip_lookup=lambda timestamp: searcher.get_clip_at_time(
                        video_id=q["video_id"], timestamp=timestamp,
                    ),
                )
                evidence = searcher.search(
                    q["question"],
                    video_id=q["video_id"],
                    top_k=args.top_k,
                    candidate_k=getattr(args, "candidate_k", None),
                    transcript_top_k=getattr(args, "transcript_top_k", 0),
                    transcript_fusion=getattr(args, "transcript_fusion", "rrf"),
                    context_k=getattr(args, "context_k", 0),
                    before_clip=before_clip,
                    before_sec=before_sec,
                    include_neighbors=args.include_neighbors,
                )
                answer = answer_question(
                    llm=llm,
                    question=q["question"],
                    evidence=evidence,
                    video_id=q["video_id"],
                    temporal_scope=temporal_scope,
                )
                error = ""
                answered += 1
            except Exception as exc:
                logger.exception("QA failed for %s/%s", q["video_id"], q["question_id"])
                answer = {"answer": "", "reasoning": "", "confidence": 0.0}
                error = str(exc)
                failed += 1

            out = {
                "video_id": q["video_id"],
                "question_id": q["question_id"],
                "question_index": q["question_index"],
                "question": q["question"],
                "expected_answer": q["answer"],
                "type": q["type"],
                "timestamp": q["timestamp"],
                "before_clip": q["before_clip"],
                "predicted_answer": answer["answer"],
                "reasoning": answer["reasoning"],
                "confidence": answer["confidence"],
                "evidence": [ev.to_dict() for ev in evidence],
                "temporal_scope": temporal_scope,
                "error": error,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

    logger.info(
        "QA wrote %d answered records to %s (%d failed, %d skipped)",
        answered, output_path, failed, skipped,
    )
    return 2 if failed and answered == 0 else 0


# ---------------------------------------------------------------------------
# Adapter-driven driver (the modern path)
# ---------------------------------------------------------------------------

def _build_qa_vector_index(args: argparse.Namespace):
    """Resolve a :class:`VectorIndex` for the standalone CLI QA driver.

    Shared by :func:`run_qa` (legacy MVP driver) and
    :func:`run_react_qa` (modern adapter driver). Mirrors
    :func:`m3.pipeline.runner._build_vector_index` but reads the CLI
    flag overrides first (``--vector-index-*``) before falling back
    to the YAML config.
    """
    from dataclasses import replace

    from m3.config import load_m3_runtime_config
    from m3.vector_index import build_m3_vector_index

    runtime = load_m3_runtime_config(getattr(args, "config", None))
    conn = getattr(args, "vector_index_connection_string", None)
    if conn:
        runtime = replace(runtime, vector_index_connection_string=conn)
    table = getattr(args, "vector_index_table_name", None)
    if table:
        runtime = replace(runtime, vector_index_table_name=table)
    pool = getattr(args, "vector_index_pool_size", None)
    if pool is not None:
        runtime = replace(runtime, vector_index_pool_size=int(pool))
    backend = getattr(args, "vector_index_backend", None)
    if backend:
        runtime = replace(runtime, vector_index_backend=backend)
    return build_m3_vector_index(runtime)


def run_react_qa(args: argparse.Namespace, graph=None, embedder=None, llm=None) -> int:
    """Adapter-driven QA: drives OGMemoryControlAdapter per question.

    Modern replacement for :func:`run_qa` (single-pass MVP that reads
    ``episodic_index.jsonl`` via the JSONL searcher). The adapter path
    uses ``GraphService.search_by_embedding`` against the indexed
    EpisractedNodes — embeddings land in the graph via
    ``m3.index_episodic`` (see ``test_index_episodic_persist_embeddings.py``
    for the contract).

    Falls back to building the graph / embedder / LLM from CLI args
    if any are None — same pattern as the other graph-using drivers.
    Resumable via ``--resume`` (skip ``(video_id, question_id)`` pairs
    already in the output JSONL).
    """
    from core.models import RequestContext
    from m3.config import M3BENCH_ACCOUNT_ID, M3BENCH_AGENT_ID, M3BENCH_USER_ID
    from m3.control_adapter import M3TaskContext, OGMemoryControlAdapter
    from m3.index_episodic import build_embedder as _build_embedder

    annotations_path = Path(args.annotations).resolve()
    if not annotations_path.exists():
        logger.error("annotations not found: %s", annotations_path)
        return 1

    questions = load_questions(annotations_path)
    logger.info("Loaded %d questions from %s", len(questions), annotations_path)
    if args.video:
        allowed = set(args.video.split(","))
        questions = [q for q in questions if q["video_id"] in allowed]
        logger.info("After --video filter (%s): %d questions", args.video, len(questions))
    if args.limit is not None:
        questions = questions[:args.limit]

    if graph is None:
        from m3.load_clips import build_graph_service
        # Resolve the VectorIndex from CLI args + YAML config (skip
        # entirely when ``--skip-vector-index`` is set so unit tests
        # can exercise the legacy JSONB path without needing a live
        # vector DB). In the M3-Bench hot path this builds an
        # OpenGaussVectorIndex against the same Postgres instance as
        # the graph store; the GraphService built below is then
        # constructed with that index so ``search_by_embedding`` routes
        # through HNSW + scope columns instead of JSONB scan.
        vi = None
        if not getattr(args, "skip_vector_index", False):
            vi = _build_qa_vector_index(args)
        graph, store_kind = build_graph_service(
            args.graph_store, args.connection_string,
            vector_index=vi,
        )
        logger.info(
            "Graph store: %s, vector index: %s",
            store_kind,
            type(vi).__name__ if vi is not None else "none",
        )

    if embedder is None:
        embedder = _build_embedder(
            args.embedder,
            model=args.embedding_model,
            base_url=args.embedding_base_url,
            api_key=args.embedding_api_key,
            dimension=args.embedding_dimension,
            config_path=args.config,
        )
    if llm is None:
        llm = build_llm(
            args.llm,
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            temperature=args.temperature,
            config_path=args.config,
        )

    from m3.context_budget import load_context_budget
    adapter = OGMemoryControlAdapter(
        llm=llm,
        graph=graph,
        embedder=embedder,
        max_iterations=args.max_iterations,
        no_gain_limit=getattr(args, "no_gain_limit", 3) or None,
        timeout_seconds=args.timeout_seconds,
        default_video_id=args.default_video_id,
        default_top_k=args.top_k,
        candidate_k=getattr(args, "candidate_k", None),
        transcript_top_k=getattr(args, "transcript_top_k", 0),
        transcript_fusion=getattr(args, "transcript_fusion", "rrf"),
        context_k=getattr(args, "context_k", 0),
        include_neighbors=getattr(args, "include_neighbors", 1),
        context_budget=load_context_budget(args.config),
    )

    output_path = Path(args.output_jsonl).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from m3.qa_results import completed_keys
    done_keys = completed_keys(output_path) if args.resume else set()

    if args.resume:
        from m3.qa_results import separate_partial_line
        separate_partial_line(output_path)
    mode = "a" if args.resume else "w"
    answered = failed = skipped = 0
    ctx = RequestContext(
        account_id=getattr(args, "account_id", M3BENCH_ACCOUNT_ID),
        user_id=M3BENCH_USER_ID,
        agent_id=M3BENCH_AGENT_ID,
        session_id="m3bench-qa",
        trace_id="m3-qa",
    )

    with output_path.open(mode, encoding="utf-8") as f:
        for q in questions:
            key = (str(q["video_id"]), str(q["question_id"]))
            if key in done_keys:
                skipped += 1
                continue

            task_ctx = M3TaskContext(
                question=str(q["question"]),
                video_id=str(q["video_id"]) if not args.default_video_id else args.default_video_id,
                temporal_policy=(getattr(args, "temporal_policy", "current_clip")
                                 if args.respect_before_clip else "unrestricted"),
                annotation_before_clip=q.get("before_clip"),
                annotation_timestamp=q.get("timestamp"),
                lookahead_seconds=(getattr(args, "lookahead_seconds", 0.0)
                                   if args.respect_before_clip else 0.0),
            )
            # Bounds are enforced by retrieval, not appended to the query.
            search_history: list = []
            iterations = 0
            finish_reason = ""
            warning = ""
            try:
                result = adapter.run(ctx, task_ctx)
                predicted = result.output[0] if result.output else ""
                iterations = result.iterations
                finish_reason = result.finish_reason
                search_history = list(task_ctx.search_history)
                # Three-way failure classification — mirrors
                # m3.eval.answer_with_adapter so the two drivers agree
                # on what "answered" / "failed" mean. The previous
                # code only checked exceptions, so a loop that
                # finished with finish_reason="max_iterations" or
                # content="[Answer] …" (scaffolding leak) was silently
                # recorded as answered.
                if finish_reason == "llm_error":
                    # Surface the upstream's own message + status code
                    # so the JSONL row tells operators whether it was a
                    # 400 (protocol mismatch, retry won't help) or a 500
                    # (transient, retry might). result.llm_error is the
                    # original :class:`LLMCallError` raised by
                    # ``_call_llm`` and threaded through to the result.
                    llm_err = getattr(result, "llm_error", None)
                    if llm_err is not None:
                        error = f"llm_error: {llm_err}"
                    else:
                        error = "llm_error: adapter loop terminated due to LLM transport failure"
                    predicted = ""
                    failed += 1
                elif not result.completed:
                    error = f"loop did not complete ({finish_reason})"
                    failed += 1
                elif _has_protocol_scaffolding(predicted):
                    # Salvage before discarding: when the model emits
                    # "[Answer] <text>" as content in a tool-disabled
                    # round, <text> IS the answer. Only a [Search]-only
                    # leak (no [Answer] tag) or an empty residue stays
                    # a failure.
                    salvaged = _salvage_scaffolding(predicted)
                    if len(salvaged) >= 3:
                        predicted = salvaged
                        error = ""
                        warning = (
                            "scaffolding salvaged: answer was emitted as "
                            "[Answer] content instead of a tool call"
                        )
                        answered += 1
                    else:
                        error = (
                            "predicted_answer contains [Search]/[Answer] "
                            "protocol scaffolding"
                        )
                        predicted = ""
                        failed += 1
                else:
                    error = ""
                    answered += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Adapter failed for %s/%s", q["video_id"], q["question_id"],
                )
                predicted = ""
                error = str(exc)
                failed += 1

            out = {
                "video_id": q["video_id"],
                "question_id": q["question_id"],
                "question_index": q["question_index"],
                "question": q["question"],
                "expected_answer": q["answer"],
                "type": q["type"],
                "timestamp": q["timestamp"],
                "before_clip": q["before_clip"],
                "predicted_answer": str(predicted).strip(),
                "iterations": iterations,
                "finish_reason": finish_reason,
                "search_rounds": len(search_history),
                "search_history": search_history,
                "warning": warning,
                "search_stats": dict(task_ctx.search_stats),
                "temporal_scope": dict(task_ctx.temporal_scope),
                "error": error,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

    logger.info(
        "QA wrote %d answered records to %s (%d failed, %d skipped)",
        answered, output_path, failed, skipped,
    )
    return 2 if failed else 0


def parse_react_qa_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """CLI for the adapter-driven QA driver.

    Shares most flags with :func:`parse_args` (legacy single-pass
    driver) but drops ``--index`` (we read from the graph now) and
    adds ``--graph-store``, ``--connection-string``,
    ``--max-iterations``, ``--timeout-seconds``, ``--default-video-id``.
    """
    p = argparse.ArgumentParser(
        description="Run M3-Bench QA via OGMemoryControlAdapter (graph retrieval).",
    )
    p.add_argument("--annotations", required=True)
    p.add_argument("--output-jsonl", default="data/m3_qa_results.jsonl")
    p.add_argument("--account-id", default=M3BENCH_ACCOUNT_ID,
                   help="Account scope used when retrieving the indexed video")
    p.add_argument("--graph-store", choices=("memory", "sql"), default="memory")
    p.add_argument("--connection-string", default=None,
                   help="PostgreSQL DSN for --graph-store sql")
    p.add_argument("--config", default=None,
                   help="ogmemory.yaml path (default: deploy/ogmemory.yaml or $OGMEM_CONFIG)")
    # Vector-index connection (M3 now reads embeddings from a
    # dedicated OpenGaussVectorIndex instead of the graph store's
    # JSONB column). The CLI flags mirror m3.vector_index.* plus the
    # shared vector_db.connection_string in ogmemory.yaml so the
    # same code path serves both pipeline runner and standalone CLI
    # invocations.
    p.add_argument("--vector-index-connection-string", default=None,
                   help="PostgreSQL DSN for the OpenGaussVectorIndex. "
                        "Falls back to vector_db.connection_string so M3 "
                        "reuses the same Postgres / openGauss instance as "
                        "the graph store by default.")
    p.add_argument("--vector-index-table-name", default=None,
                   help="Vector-index table name (default: "
                        "m3.vector_index.table_name or 'm3_episodic_vectors').")
    p.add_argument("--vector-index-pool-size", type=int, default=None,
                   help="OpenGaussVectorIndex connection pool size (default: 5).")
    p.add_argument("--vector-index-backend", default=None,
                   choices=("opengauss", "pgvector", "chroma", "memory"),
                   help="VectorIndex backend (default: m3.vector_index.backend or opengauss). "
                        "`opengauss` is the recommended production target; `pgvector` is "
                        "the single-DB convenience alias for PostgreSQL+pgvector "
                        "deployments that share one DB with the graph store. Both route "
                        "to the same OpenGaussVectorIndex implementation.")
    p.add_argument("--skip-vector-index", action="store_true",
                   help="Skip building a VectorIndex and force the legacy "
                        "JSONB search_by_embedding path. Useful for unit "
                        "tests that mock the graph but don't need a real "
                        "vector store; do NOT use in production runs.")
    p.add_argument("--video", default=None, help="Comma-separated video_ids to evaluate")
    p.add_argument("--limit", type=int, default=None, help="Max questions to run")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--respect-before-clip", action="store_true", default=True)
    add_temporal_arguments(p)
    p.add_argument("--ignore-before-clip", dest="respect_before_clip", action="store_false",
                   help="Disable both annotation clip and timestamp boundaries (unrestricted temporal search).")
    # Adapter
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--no-gain-limit", type=int, default=3,
                   help="Consecutive searches without new evidence before answer-only mode (>=2; 0 disables)")
    p.add_argument("--timeout-seconds", type=float, default=60.0)
    p.add_argument("--default-video-id", default=None,
                   help="Override the per-question video scope passed to [Search]")
    p.add_argument("--context-k", type=int, default=0,
                   help="Separate exact-neighbor evidence budget (0 preserves the existing result budget)")
    p.add_argument("--transcript-top-k", type=int, default=0,
                   help="Independent transcript candidates (0 disables; requires scoped memory/SQL index)")
    p.add_argument("--transcript-fusion", choices=("rrf", "weighted"), default="rrf",
                   help="Experimental vector/transcript fusion; effective only when transcript recall is enabled")
    p.add_argument("--candidate-k", type=int, default=None,
                   help="Vector candidates per query branch; promoted to at least top-k (default: legacy depth)")
    p.add_argument("--top-k", type=int, default=8,
                   help="Number of evidence hits returned per [Search] "
                        "(default 8; was 5 — the lower default hurt recall "
                        "for short benchmarks).")
    p.add_argument("--include-neighbors", type=int, default=1,
                   help="Expand ±N neighbor clips around the top hits so the "
                        "answerer can see temporal context (default 1; was 0).")
    # Embedder (for [Search])
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--embedding-dimension", type=int, default=None)
    # Answer LLM
    p.add_argument("--llm", default=None,
                   help="Answer LLM label (mock/minimax/any OpenAI-compatible); "
                        "defaults to m3.qa_llm.provider from ogmemory.yaml")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
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


def main_react_qa(argv: Iterable[str] | None = None) -> int:
    args = parse_react_qa_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_react_qa(args)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run vision-only M3 QA over episodic retrieval.")
    p.add_argument("--annotations", required=True)
    # ``--index`` was the path to ``episodic_index.jsonl``; that
    # manifest no longer carries embeddings (the canonical store is
    # the VectorIndex). Kept for backwards compatibility with
    # existing scripts — :func:`run_qa` logs a deprecation warning
    # when it's set.
    p.add_argument("--index", default=None,
                   help="[DEPRECATED] legacy episodic_index.jsonl path; "
                        "embeddings now live in the VectorIndex.")
    p.add_argument("--output-jsonl", default="data/m3_qa_results.jsonl")
    p.add_argument("--config", default=None,
                   help="Path to ogmemory.yaml (default: deploy/ogmemory.yaml or OGMEM_CONFIG)")
    p.add_argument("--video", default=None, help="Comma-separated video_ids to evaluate")
    p.add_argument("--limit", type=int, default=None, help="Max questions to run")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--account-id", default="m3bench",
                   help="Account id for the per-record RequestContext "
                        "(default: %(default)s)")
    # Retrieval
    p.add_argument("--context-k", type=int, default=0,
                   help="Separate exact-neighbor evidence budget (0 preserves the existing result budget)")
    p.add_argument("--transcript-top-k", type=int, default=0,
                   help="Independent transcript candidates (0 disables; requires scoped memory/SQL index)")
    p.add_argument("--transcript-fusion", choices=("rrf", "weighted"), default="rrf",
                   help="Experimental vector/transcript fusion; effective only when transcript recall is enabled")
    p.add_argument("--candidate-k", type=int, default=None,
                   help="Vector candidates per query branch; promoted to at least top-k (default: legacy depth)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--include-neighbors", type=int, default=1)
    p.add_argument("--respect-before-clip", action="store_true", default=True)
    add_temporal_arguments(p)
    p.add_argument("--ignore-before-clip", dest="respect_before_clip", action="store_false",
                   help="Disable both annotation clip and timestamp boundaries (unrestricted temporal search).")
    p.add_argument("--hybrid-weight", type=float, default=0.15)
    p.add_argument("--embedder", choices=("mock", "openai", "st", "volcengine"), default=None,
                   help="Embedder provider; defaults to m3.embedding.provider from ogmemory.yaml")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--embedding-base-url", default=None)
    p.add_argument("--embedding-api-key", default=None)
    p.add_argument("--embedding-dimension", type=int, default=None)
    # Vector-index connection (legacy driver reads from the
    # VectorIndex instead of the JSONL file).
    p.add_argument("--vector-index-connection-string", default=None,
                   help="PostgreSQL DSN for the OpenGaussVectorIndex. "
                        "Falls back to vector_db.connection_string.")
    p.add_argument("--vector-index-table-name", default=None,
                   help="Vector-index table name (default: 'm3_episodic_vectors').")
    p.add_argument("--vector-index-pool-size", type=int, default=None,
                   help="OpenGaussVectorIndex pool size (default: 5).")
    p.add_argument("--vector-index-backend", default=None,
                   choices=("opengauss", "pgvector", "chroma", "memory"))
    p.add_argument("--skip-vector-index", action="store_true",
                   help="Skip building a VectorIndex and use the in-memory "
                        "searcher fallback (unit tests only).")
    # Answer LLM
    p.add_argument("--llm", default=None,
                   help="Answer LLM label (mock/minimax/any OpenAI-compatible); "
                        "defaults to m3.qa_llm.provider from ogmemory.yaml")
    p.add_argument("--llm-model", default=None)
    p.add_argument("--llm-base-url", default=None)
    p.add_argument("--llm-api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
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
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_qa(args)


if __name__ == "__main__":
    sys.exit(main())
