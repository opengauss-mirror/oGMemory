"""Unit tests for the M3-Bench eval harness (``m3.eval``).

Three layers:

* Judge scoring — pure unit tests over ``judge_answer`` with a
  scripted MockLLM. Asserts the {0.0, 0.5, 1.0} normalisation and
  the empty-predicted short-circuit.
* Aggregation — pure unit tests over ``aggregate`` with synthetic
  QuestionResult lists. Asserts per-type bucketing, percentile math,
  and overall totals.
* End-to-end — an integration test that wires a tiny synthetic
  annotation + seeded graph + scripted adapter LLM + scripted judge
  LLM and asserts the produced results.jsonl + summary.json shapes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

from core.graph_models import create_extracted_node
from core.models import RequestContext
from m3.config import M3BENCH_ACCOUNT_ID
from m3.control_adapter import M3TaskContext
from m3.eval import (
    JUDGE_PROMPT_TEMPLATE,
    JUDGE_SCHEMA,
    QuestionResult,
    aggregate,
    answer_with_adapter,
    build_judge_llm,
    judge_answer,
    parse_args,
)
from providers.embedder import MockEmbedder
from providers.graph_store import MemoryGraphStore
from providers.llm import MockLLM
from service.graph_service import GraphService


# ---------------------------------------------------------------------------
# Judge scoring
# ---------------------------------------------------------------------------

class _ScriptedJudge(MockLLM):
    def __init__(self, script):
        super().__init__()
        self._script = list(script)
        self._last_prompt = ""

    def complete_json(self, prompt, schema):
        self._call_count += 1
        self._last_prompt = prompt
        if not self._script:
            return {"score": 0.0, "reasoning": "no script"}
        return self._script.pop(0)


def test_judge_returns_one_when_predicted_matches_expected():
    judge = _ScriptedJudge([
        {"score": 1.0, "reasoning": "matches"},
    ])
    score, reasoning, judge_error = judge_answer(
        judge_llm=judge,
        question="What colour is the mug?",
        expected="red",
        predicted="red",
    )
    assert score == 1.0
    assert reasoning == "matches"
    assert judge_error is False


def test_judge_returns_zero_when_predicted_empty():
    judge = _ScriptedJudge([])  # never reached
    score, reasoning, judge_error = judge_answer(
        judge_llm=judge,
        question="Q?",
        expected="something",
        predicted="",
    )
    assert score == 0.0
    assert "empty" in reasoning.lower()
    # Empty predicted is a true evaluation outcome (model gave up), not a
    # judge error — the judge never ran.
    assert judge_error is False


def test_judge_normalises_score_to_half_bucket():
    """LLMs often emit 0.7 or 0.8; the harness must snap to the
    {0.0, 0.5, 1.0} buckets so the per-type summary numbers add up."""
    judge = _ScriptedJudge([{"score": 0.7, "reasoning": "partial"}])
    score, _, _ = judge_answer(judge_llm=judge, question="q",
                               expected="e", predicted="p")
    assert score == 0.5
    judge = _ScriptedJudge([{"score": 0.85, "reasoning": "near"}])
    score, _, _ = judge_answer(judge_llm=judge, question="q",
                               expected="e", predicted="p")
    assert score == 1.0


def test_judge_handles_llm_failure_without_raising():
    class FailingJudge(MockLLM):
        def complete_json(self, prompt, schema):
            raise RuntimeError("LLM is down")
    score, reasoning, judge_error = judge_answer(
        judge_llm=FailingJudge(),
        question="q", expected="e", predicted="p",
    )
    assert score == 0.0
    assert "judge failed" in reasoning
    # And — critically — the failure is flagged so the summary can
    # distinguish "model wrong" from "judge broken".
    assert judge_error is True


def test_judge_prompt_includes_question_and_both_answers():
    judge = _ScriptedJudge([{"score": 1.0, "reasoning": "x"}])
    judge_answer(
        judge_llm=judge,
        question="Who walks in?",
        expected="Alice",
        predicted="Alice",
    )
    assert "Who walks in?" in judge._last_prompt
    assert "Alice" in judge._last_prompt


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _qr(video_id, qid, score, type_="object", iterations=2, error=""):
    return QuestionResult(
        video_id=video_id, question_id=qid, question_index=0,
        question="q", expected_answer="e",
        predicted_answer="p",
        type=type_, iterations=iterations,
        latency_seconds=1.0,
        judge_score=score, judge_reasoning="",
        error=error,
    )


def test_aggregate_buckets_by_question_type():
    # 7 questions: 3 objects @ 1.0, 2 temporals @ 0.5, 2 temporals @ 0.0.
    # Sum = 4.0 → mean = 4/7 ≈ 0.571.
    results = [
        _qr("v", f"q{i}", score=1.0, type_="object")
        for i in range(3)
    ] + [
        _qr("v", f"q{i}", score=0.5, type_="temporal")
        for i in range(2)
    ] + [
        _qr("v", f"q{i}", score=0.0, type_="temporal")
        for i in range(2)
    ]
    summary = aggregate(results, judge_provider="mock", judge_model="mock")
    assert summary["total_questions"] == 7
    assert summary["overall_mean_score"] == pytest.approx(4 / 7, abs=1e-9)
    assert summary["answered"] == 7
    assert summary["failed"] == 0
    assert set(summary["by_type"].keys()) == {"object", "temporal"}
    obj = summary["by_type"]["object"]
    assert obj["count"] == 3
    assert obj["mean_score"] == 1.0
    tmp = summary["by_type"]["temporal"]
    assert tmp["count"] == 4
    assert tmp["mean_score"] == 0.25
    assert tmp["p50_iterations"] == 2.0


def test_aggregate_handles_list_type_field():
    """Some M3-Bench annotations store ``type`` as a list of category
    strings. The aggregator should join them into a stable bucket key."""
    results = [
        _qr("v", "q1", score=1.0, type_=["object", "spatial"]),
    ]
    # The dataclass forbids list directly via the type annotation
    # (str), so coerce via attribute assignment.
    results[0].type = "object,spatial"
    summary = aggregate(results, judge_provider="mock", judge_model="mock")
    assert "object,spatial" in summary["by_type"]


def test_aggregate_handles_unspecified_type():
    results = [
        _qr("v", "q1", score=1.0, type_=""),
        _qr("v", "q2", score=0.0, type_=""),
    ]
    summary = aggregate(results, judge_provider="mock", judge_model="mock")
    assert "(unspecified)" in summary["by_type"]
    bucket = summary["by_type"]["(unspecified)"]
    assert bucket["count"] == 2


def test_aggregate_counts_errors():
    results = [
        _qr("v", "ok", score=1.0, type_="object"),
        _qr("v", "fail", score=0.0, type_="object", error="boom"),
    ]
    summary = aggregate(results, judge_provider="mock", judge_model="mock")
    assert summary["answered"] == 1
    assert summary["failed"] == 1
    assert summary["by_type"]["object"]["failed"] == 1


def test_aggregate_rolls_up_per_video():
    """Per-video accuracy is what the bench-scale run needs: a single
    bad video shouldn't be allowed to drag the global mean without
    being visible in the summary."""
    results = [
        _qr("vid_a", f"q{i}", score=1.0, type_="object") for i in range(3)
    ] + [
        _qr("vid_b", f"q{i}", score=0.0, type_="object") for i in range(2)
    ] + [
        _qr("vid_a", "q3", score=0.5, type_="temporal"),
    ]
    summary = aggregate(results, judge_provider="mock", judge_model="mock")
    assert "by_video" in summary
    assert set(summary["by_video"].keys()) == {"vid_a", "vid_b"}
    assert summary["by_video"]["vid_a"]["count"] == 4
    assert summary["by_video"]["vid_a"]["mean_score"] == pytest.approx(3.5 / 4)
    assert summary["by_video"]["vid_b"]["count"] == 2
    assert summary["by_video"]["vid_b"]["mean_score"] == 0.0
    # Sorted by count desc — vid_a (4) before vid_b (2).
    assert list(summary["by_video"].keys()) == ["vid_a", "vid_b"]


def test_aggregate_counts_judge_errors_separately():
    """A misconfigured judge API key must NOT masquerade as a 0%
    accuracy. judge_error_count is the dedicated counter."""
    results = [
        _qr("v", "ok", score=1.0, type_="object"),
        _qr("v", "wrong", score=0.0, type_="object"),
        # judge_error=True means the score here is not a real judgement;
        # it should NOT inflate ``failed`` (which is adapter errors) but
        # SHOULD show up under judge_error_count so reviewers can spot a
        # broken judge.
        _qr("v", "judge_died", score=0.0, type_="object"),
    ]
    results[2].judge_error = True
    summary = aggregate(results, judge_provider="mock", judge_model="mock")
    assert summary["judge_error_count"] == 1
    # adapter-level answered/failed unchanged — judge_error is its own axis.
    assert summary["answered"] == 3
    assert summary["failed"] == 0
    # And it surfaces per-video too so a single bad judge call doesn't
    # poison a single video's rollup silently.
    assert summary["by_video"]["v"]["count"] == 3


# ---------------------------------------------------------------------------
# End-to-end (integration)
# ---------------------------------------------------------------------------

_FFMPEG = shutil.which("ffmpeg")
_requires_ffmpeg = pytest.mark.skipif(
    _FFMPEG is None, reason="ffmpeg not available on PATH",
)


def _seed_episode(graph, ctx, video_id, clip_id, description, nid):
    node = create_extracted_node(
        source_raw_id=f"video-{video_id}-{clip_id}"[:32].ljust(32, "x"),
        extractor_type="episodic_memory_generator",
        extraction_schema="m3_bench_v1",
        account_id=ctx.account_id,
        extracted_content={"description": description, "_video_id": video_id,
                           "_clip_id": clip_id, "characters": [],
                           "events": [], "entities": []},
        node_id=nid,
    )
    graph._store.create_node(node.serialize(), ctx)
    live = graph._store.get_node(nid, ctx)
    live.set_embedding([1.0, 0.0, 0.5])
    graph.update_extracted_node(live, ctx)


class _ScriptedAdapterLLM(MockLLM):
    """MockLLM that cycles Search → Answer → content on every question.

    The eval harness builds the adapter (and its LLM) once and
    reuses it across all questions, so we can't just count absolute
    calls — we need to reset state when a new question starts. We
    detect the question boundary via ``messages`` shape: the first
    user message in a new run() always starts with "Question: …".
    When that prefix changes vs the previous run, we reset.
    """
    def __init__(self, answer_text: str):
        super().__init__()
        self._answer_text = answer_text
        self._per_question_round = 0
        self._last_question_prefix = None

    def complete_with_tools_messages(self, messages, tools=None, tool_choice="auto"):
        # Detect question boundary: the user message contains "Question: …"
        # and the prefix changes between calls.
        question_prefix = None
        for m in messages:
            if m.get("role") == "user" and "Question:" in (m.get("content") or ""):
                question_prefix = m["content"][:32]
                break
        if question_prefix and question_prefix != self._last_question_prefix:
            self._per_question_round = 0
            self._last_question_prefix = question_prefix
        self._per_question_round += 1
        self._last_messages = messages
        self._call_count += 1
        if self._per_question_round == 1:
            return (
                [{"tool": "Search", "input": {"query": "anything"}}],
                "",
            )
        if self._per_question_round == 2:
            return (
                [{"tool": "Answer",
                  "input": {"final_answer": self._answer_text}}],
                "",
            )
        return [], self._answer_text


class _ScriptedJudgeLLM(MockLLM):
    """Mock judge that returns the next score from a queue on each call."""
    def __init__(self, scores):
        super().__init__()
        self._queue = list(scores)
    def complete_json(self, prompt, schema):
        self._call_count += 1
        score = self._queue.pop(0) if self._queue else 0.0
        return {"score": score, "reasoning": f"scored {score}"}


def _write_annotations(tmp_path: Path, questions: list[dict]) -> Path:
    annotations = {"vid_test": {"video_path": "/tmp/vid_test.mp4",
                                "qa_list": questions}}
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations), encoding="utf-8")
    return path


@_requires_ffmpeg
def test_run_end_to_end_writes_results_and_summary(tmp_path: Path):
    """Full run: 3 questions, scripted adapter LLM + scripted judge,
    verify results.jsonl + summary.json shapes and per-type buckets."""
    annotations_path = _write_annotations(tmp_path, [
        {"question_id": "q1", "question": "What colour is the mug?",
         "answer": "red", "type": "object_recognition"},
        {"question_id": "q2", "question": "When did she enter?",
         "answer": "after lunch", "type": "temporal_reasoning"},
        {"question_id": "q3", "question": "Where is the dog?",
         "answer": "backyard", "type": "spatial_reasoning"},
    ])

    from m3.eval import parse_args, run
    args = parse_args([
        "--annotations", str(annotations_path),
        "--output-dir", str(tmp_path / "out"),
        "--graph-store", "memory",
        "--embedder", "mock",
        "--embedding-dimension", "8",
        "--llm", "mock",
        "--judge-base-url", "http://mock-judge",
        "--judge-model", "mock-judge-model",
        "--judge-api-key", "mock-judge-key",
    ])
    # Patch the adapter LLM factory + judge factory via monkeypatch so
    # the run() uses our scripted mocks regardless of CLI choice.
    import m3.eval as eval_mod
    monkey = pytest.MonkeyPatch()

    def _answer_llm_factory(*a, **kw):
        return _ScriptedAdapterLLM("the woman is wearing red")
    monkey.setattr(eval_mod, "build_llm", _answer_llm_factory)

    def _judge_llm_factory(*a, **kw):
        # Score 1.0 for q1 (correct), 0.5 for q2 (partial), 0.0 for q3.
        return _ScriptedJudgeLLM([1.0, 0.5, 0.0])
    monkey.setattr(eval_mod, "build_judge_llm", _judge_llm_factory)

    try:
        rc = eval_mod.run(args)
        assert rc == 0, f"eval returned {rc}"
    finally:
        monkey.undo()

    results_path = tmp_path / "out" / "m3_eval_results.jsonl"
    summary_path = tmp_path / "out" / "m3_eval_summary.json"
    assert results_path.exists()
    assert summary_path.exists()

    records = [json.loads(l) for l in results_path.read_text().splitlines() if l]
    assert len(records) == 3
    by_qid = {r["question_id"]: r for r in records}
    assert by_qid["q1"]["predicted_answer"] == "the woman is wearing red"
    assert by_qid["q1"]["judge_score"] == 1.0
    assert by_qid["q2"]["judge_score"] == 0.5
    assert by_qid["q3"]["judge_score"] == 0.0
    # Search history was captured per question.
    for r in records:
        assert r["iterations"] >= 2
        assert r["search_rounds"] >= 1
        assert r["search_stats"]["search_requests"] == r["search_rounds"]
        assert r["search_stats"]["model_rounds"] == r["iterations"]
        assert any(
            h.get("query") == "anything"
            for h in r["search_history"]
        )

    summary = json.loads(summary_path.read_text())
    assert summary["total_questions"] == 3
    assert summary["answered"] == 3
    assert summary["failed"] == 0
    assert summary["judge_provider"] == "mock-judge"
    # Per-type buckets all present.
    assert set(summary["by_type"].keys()) == {
        "object_recognition", "temporal_reasoning", "spatial_reasoning",
    }
    assert summary["by_type"]["object_recognition"]["mean_score"] == 1.0
    assert summary["by_type"]["temporal_reasoning"]["mean_score"] == 0.5
    assert summary["by_type"]["spatial_reasoning"]["mean_score"] == 0.0


def test_run_resume_skips_already_answered(tmp_path: Path):
    """Re-running with --resume must not re-answer questions already
    in results.jsonl."""
    annotations_path = _write_annotations(tmp_path, [
        {"question_id": "q1", "question": "Q1?", "answer": "A1",
         "type": "object"},
        {"question_id": "q2", "question": "Q2?", "answer": "A2",
         "type": "temporal"},
    ])
    # Pre-populate results with q1 already answered.
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "m3_eval_results.jsonl").write_text(
        json.dumps({
            "video_id": "vid_test", "question_id": "q1",
            "question_index": 0, "question": "Q1?",
            "expected_answer": "A1", "predicted_answer": "A1",
            "type": "object", "timestamp": None, "before_clip": None,
            "iterations": 2, "finish_reason": "done",
            "search_rounds": 1, "search_history": [],
            "judge_score": 1.0, "judge_reasoning": "",
            "latency_seconds": 1.0, "error": "",
        }) + "\n", encoding="utf-8",
    )

    from m3.eval import parse_args, run
    import m3.eval as eval_mod

    monkey = pytest.MonkeyPatch()
    adapter_call_count = {"n": 0}

    def _answer_llm_factory(*a, **kw):
        return _ScriptedAdapterLLM("A2")
    monkey.setattr(eval_mod, "build_llm", _answer_llm_factory)
    monkey.setattr(eval_mod, "build_judge_llm",
                  lambda *a, **kw: _ScriptedJudgeLLM([1.0]))

    # Wrap adapter.run to count calls
    original_run = eval_mod.OGMemoryControlAdapter.run
    def counting_run(self, ctx, task_ctx):
        adapter_call_count["n"] += 1
        return original_run(self, ctx, task_ctx)
    monkey.setattr(eval_mod.OGMemoryControlAdapter, "run", counting_run)

    try:
        args = parse_args([
            "--annotations", str(annotations_path),
            "--output-dir", str(out_dir),
            "--graph-store", "memory",
            "--embedder", "mock",
            "--embedding-dimension", "8",
            "--llm", "mock",
            "--judge-base-url", "http://mock-judge",
            "--judge-model", "mock-judge-model",
            "--judge-api-key", "mock-judge-key",
            "--resume",
        ])
        rc = eval_mod.run(args)
        assert rc == 0
    finally:
        monkey.undo()

    # Only q2 was newly answered (q1 was skipped from the on-disk file).
    assert adapter_call_count["n"] == 1
    records = [json.loads(l) for l in (out_dir / "m3_eval_results.jsonl").read_text().splitlines() if l]
    assert len(records) == 2
    assert {r["question_id"] for r in records} == {"q1", "q2"}


def test_run_writes_summary_with_video_and_type_filters(tmp_path: Path):
    """--video and --type-filter narrow the question set; the
    summary reflects only the filtered subset."""
    annotations_path = _write_annotations(tmp_path, [
        {"question_id": "q1", "question": "Q1", "answer": "A1", "type": "object"},
        {"question_id": "q2", "question": "Q2", "answer": "A2", "type": "temporal"},
    ])
    from m3.eval import parse_args, run
    import m3.eval as eval_mod

    monkey = pytest.MonkeyPatch()
    monkey.setattr(eval_mod, "build_llm",
                  lambda *a, **kw: _ScriptedAdapterLLM("A"))
    monkey.setattr(eval_mod, "build_judge_llm",
                  lambda *a, **kw: _ScriptedJudgeLLM([1.0]))
    try:
        args = parse_args([
            "--annotations", str(annotations_path),
            "--output-dir", str(tmp_path / "out"),
            "--graph-store", "memory",
            "--embedder", "mock",
            "--embedding-dimension", "8",
            "--llm", "mock",
            "--judge-base-url", "http://mock-judge",
            "--judge-model", "mock-judge-model",
            "--judge-api-key", "mock-judge-key",
            "--type-filter", "object",
        ])
        rc = eval_mod.run(args)
        assert rc == 0
    finally:
        monkey.undo()

    summary = json.loads((tmp_path / "out" / "m3_eval_summary.json").read_text())
    # Only the object question survived the filter.
    assert summary["total_questions"] == 1
    assert set(summary["by_type"].keys()) == {"object"}


# ---------------------------------------------------------------------------
# Judge LLM routing — guards against the bench-scale run silently
# falling back to an OpenAI-key path. The judge is no longer a
# provider enum; the user passes (base_url, model, api_key) and any
# OpenAI-compatible endpoint works.
# ---------------------------------------------------------------------------


def test_parse_args_has_no_judge_llm_provider_enum():
    """We deliberately removed the ``--judge-llm`` enum. Reviewers
    shouldn't have to remember which provider name we wired up —
    they should just point at any OpenAI-compatible URL."""
    args = parse_args(["--annotations", "x.json"])
    # The old enum field is gone; only the (base_url, model, api_key)
    # triple remains.
    assert not hasattr(args, "judge_llm")
    assert args.judge_base_url is None
    assert args.judge_model is None
    assert args.judge_api_key is None


def _write_judge_yaml(tmp_path, *, provider, api_key=None, base_url=None, model=None):
    """Write a minimal ogmemory.yaml whose m3.judge_llm block resolves
    to the given provider. Used to drive the build_judge_llm routing
    without touching the real deploy/ogmemory.yaml."""
    import yaml
    cfg = {
        "m3": {
            "judge_llm": {
                "provider": provider,
                "api_key": api_key or "",
                "base_url": base_url or "",
                "model": model or "",
                "temperature": 0.0,
                "max_tokens": 4096,
            }
        }
    }
    path = tmp_path / "ogmemory.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_build_judge_llm_with_explicit_endpoint_uses_openai_transport(monkeypatch, tmp_path):
    """When ``m3.judge_llm.provider`` is not ``minimax`` (e.g. ``openai``
    or ``deepseek``), we go through OpenAILLM with that endpoint's
    base URL + model. No ``--judge-llm`` provider enum needed."""
    captured = {}

    class _FakeOpenAILLM:
        def __init__(self, *, api_key, base_url, model, temperature=0.0,
                     max_tokens=None, json_mode=False, request_timeout_seconds=120.0):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["model"] = model
            captured["json_mode"] = json_mode

    # Stub the OpenAILLM factory (used inside build_judge_llm when the
    # resolved provider is non-minimax).
    import providers.llm as providers_llm
    monkeypatch.setattr(providers_llm, "get_openai_llm",
                        lambda: (_FakeOpenAILLM, None))

    config = _write_judge_yaml(
        tmp_path,
        provider="openai",
        api_key="yaml-deepseek-key",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )
    llm = build_judge_llm(config_path=config)
    assert isinstance(llm, _FakeOpenAILLM)
    assert captured["api_key"] == "yaml-deepseek-key"
    assert captured["base_url"] == "https://api.deepseek.com/v1"
    assert captured["model"] == "deepseek-chat"
    # json_mode=True is required for the judge's structured output.
    assert captured["json_mode"] is True


def test_build_judge_llm_cli_overrides_yaml(monkeypatch, tmp_path):
    """CLI flags win over the YAML block field-by-field. The provider
    itself is read from YAML — there's no ``--judge-llm`` flag — but
    base_url / model / api_key can be overridden per-run."""
    captured = {}

    class _FakeOpenAILLM:
        def __init__(self, *, api_key, base_url, model, temperature=0.0,
                     max_tokens=None, json_mode=False, request_timeout_seconds=120.0):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["model"] = model

    import providers.llm as providers_llm
    monkeypatch.setattr(providers_llm, "get_openai_llm",
                        lambda: (_FakeOpenAILLM, None))

    config = _write_judge_yaml(
        tmp_path,
        provider="openai",
        api_key="yaml-key",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )
    llm = build_judge_llm(
        config_path=config,
        api_key="cli-override-key",
        model="deepseek-reasoner",
    )
    assert isinstance(llm, _FakeOpenAILLM)
    assert captured["api_key"] == "cli-override-key"
    # base_url came from the YAML.
    assert captured["base_url"] == "https://api.deepseek.com/v1"
    # model came from the CLI.
    assert captured["model"] == "deepseek-reasoner"


def test_build_judge_llm_endpoint_requires_api_key(monkeypatch, tmp_path):
    """If the YAML says ``provider: openai`` but no api_key is set
    anywhere (YAML, CLI, or env), fail fast — don't silently fall
    through to MiniMax and route a 'custom' judge at the wrong URL."""
    monkeypatch.delenv("OGMEM_MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _write_judge_yaml(
        tmp_path,
        provider="openai",
        api_key="",  # intentionally empty
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )
    with pytest.raises(ValueError, match="api_key is required"):
        build_judge_llm(config_path=config)


def test_build_judge_llm_defaults_to_minimax(monkeypatch):
    """No args → MiniMaxLLM with its baked-in MiniMax defaults. This
    is the bench-scale path; OPENAI_API_KEY must NOT be required."""
    captured = {}

    class _FakeMiniMaxLLM:
        def __init__(self, *, api_key, base_url, model, temperature=0.0,
                     max_tokens=None, json_mode=False, request_timeout_seconds=120.0):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["model"] = model

    import providers.llm as providers_llm
    monkeypatch.setattr(providers_llm, "get_minimax_llm", lambda: _FakeMiniMaxLLM)

    llm = build_judge_llm()
    assert isinstance(llm, _FakeMiniMaxLLM)
    # The MiniMax class handles its own api_key fallback via
    # OGMEM_MINIMAX_API_KEY — we just pass through whatever the
    # caller (or env) supplied.
    assert captured["model"] == "MiniMax-M3"


# ---------------------------------------------------------------------------
# --input-qa-results (judge-only mode)
# ---------------------------------------------------------------------------


def _write_qa_predictions(path: Path, rows: list[dict]) -> Path:
    """Write a JSONL shaped like the pipeline's m3_qa_results.jsonl."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _write_annotations_with_video(
    tmp_path: Path, video_id: str, questions: list[dict],
) -> Path:
    """Like ``_write_annotations`` but pins ``video_id`` as the
    annotation dict key (so it round-trips through
    ``iter_annotation_questions``)."""
    annotations = {video_id: {"video_path": f"/tmp/{video_id}.mp4",
                              "qa_list": questions}}
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotations), encoding="utf-8")
    return path


def test_run_with_input_qa_results_judges_existing_predictions(
    monkeypatch, tmp_path,
):
    """``--input-qa-results`` skips the adapter entirely and only
    invokes the judge on the supplied predictions."""
    import m3.eval as eval_mod

    video_id = "vid_judge_only"
    questions = [
        {"question_id": "q1", "question_index": 0,
         "question": "What color?", "answer": "red",
         "type": "object_recognition",
         "timestamp": 0.0, "before_clip": "c0"},
        {"question_id": "q2", "question_index": 1,
         "question": "How many?", "answer": "3",
         "type": "object_recognition",
         "timestamp": 5.0, "before_clip": "c1"},
        {"question_id": "q3", "question_index": 2,
         "question": "Where?", "answer": "kitchen",
         "type": "spatial_reasoning",
         "timestamp": 10.0, "before_clip": "c2"},
    ]
    annotations_path = _write_annotations_with_video(
        tmp_path, video_id, questions,
    )
    predictions = _write_qa_predictions(
        tmp_path / "qa.jsonl",
        [
            {"video_id": video_id, "question_id": "q1",
             "predicted_answer": "the woman is wearing red",
             "iterations": 2, "search_rounds": 2,
             "search_history": [{"round": 1, "queries": ["red"]}],
             "search_stats": {"search_executions": 1, "cache_hits": 1}},
            {"video_id": video_id, "question_id": "q2",
             "predicted_answer": "three objects", "iterations": 1,
             "search_rounds": 0, "search_history": []},
            {"video_id": video_id, "question_id": "q3",
             "predicted_answer": "in the kitchen", "iterations": 3,
             "search_rounds": 1, "search_history": []},
        ],
    )

    # If we accidentally let the adapter run it would blow up — the
    # assertion here is "build_graph_service was never called".
    import m3.load_clips as load_clips_mod
    monkeypatch.setattr(
        load_clips_mod, "build_graph_service",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("adapter should not run in judge-only mode")
        ),
    )
    monkeypatch.setattr(
        eval_mod, "build_judge_llm",
        lambda *a, **kw: _ScriptedJudgeLLM([1.0, 0.5, 0.0]),
    )

    out_dir = tmp_path / "eval_out"
    args = eval_mod.parse_args([
        "--annotations", str(annotations_path),
        "--input-qa-results", str(predictions),
        "--output-dir", str(out_dir),
    ])
    rc = eval_mod.run(args)
    assert rc == 0

    results_path = out_dir / "m3_eval_results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    assert len(rows) == 3
    # The adapter side never ran — predicted_answer came from the input JSONL.
    assert rows[0]["predicted_answer"] == "the woman is wearing red"
    assert rows[0]["expected_answer"] == "red"
    assert rows[0]["judge_score"] == 1.0
    # Adapter-side metadata carried through from the input row.
    assert rows[0]["iterations"] == 2
    assert rows[0]["search_rounds"] == 2
    assert rows[0]["search_stats"] == {"search_executions": 1, "cache_hits": 1}
    assert rows[1]["search_stats"] == {}
    assert rows[1]["judge_score"] == 0.5
    assert rows[2]["judge_score"] == 0.0

    summary = json.loads((out_dir / "m3_eval_summary.json").read_text())
    assert summary["answered"] == 3
    assert summary["failed"] == 0
    assert summary["missing_from_input"] == 0
    assert summary["input_qa_results"] == str(predictions.resolve())


@pytest.mark.parametrize("empty_input", [False, True])
def test_run_with_input_qa_results_handles_missing_predictions(
    monkeypatch, tmp_path, empty_input,
):
    """If a question has no row in the input JSONL, the result row is
    written with empty predicted_answer and judge_error=False (per the
    adapter-side failure contract). The summary's
    ``missing_from_input`` counter is incremented."""
    import m3.eval as eval_mod

    video_id = "vid_partial"
    questions = [
        {"question_id": "q1", "question_index": 0,
         "question": "What color?", "answer": "red",
         "type": "object_recognition",
         "timestamp": 0.0, "before_clip": "c0"},
        {"question_id": "q2", "question_index": 1,
         "question": "How many?", "answer": "3",
         "type": "object_recognition",
         "timestamp": 5.0, "before_clip": "c1"},
    ]
    annotations_path = _write_annotations_with_video(
        tmp_path, video_id, questions,
    )
    # Only q1 has a row. q2 should be written with empty predicted.
    predictions = _write_qa_predictions(
        tmp_path / "qa.jsonl",
        [] if empty_input else [
            {"video_id": video_id, "question_id": "q1", "predicted_answer": "red"},
        ],
    )
    monkeypatch.setattr(
        eval_mod, "build_judge_llm",
        lambda *a, **kw: _ScriptedJudgeLLM([1.0, 0.0]),
    )

    out_dir = tmp_path / "eval_out"
    args = eval_mod.parse_args([
        "--annotations", str(annotations_path),
        "--input-qa-results", str(predictions),
        "--output-dir", str(out_dir),
    ])
    assert eval_mod.run(args) == 2

    rows = [json.loads(line) for line in (out_dir / "m3_eval_results.jsonl").read_text().splitlines() if line]
    assert len(rows) == 2
    # q2 has no row → empty predicted_answer, judge_error=False.
    q2 = next(r for r in rows if r["question_id"] == "q2")
    assert q2["predicted_answer"] == ""
    assert q2["judge_error"] is False
    assert q2["error"] == "missing_prediction"
    assert q2["judge_reasoning"] == "predicted answer is empty"

    summary = json.loads((out_dir / "m3_eval_summary.json").read_text())
    assert summary["missing_from_input"] == (2 if empty_input else 1)
    assert summary["failed"] == (2 if empty_input else 1)
    assert summary["answered"] == (0 if empty_input else 1)
    assert summary["prediction_coverage"] == (0.0 if empty_input else 0.5)
    assert summary["overall_mean_score"] == (0.0 if empty_input else 0.5)
    assert summary["status"] == "failed"


def test_run_with_input_qa_results_accepts_legacy_answer_field(
    monkeypatch, tmp_path,
):
    """The legacy ``m3.qa`` schema uses ``answer`` (not
    ``predicted_answer``). Accept both so rejudging older artifacts
    doesn't require a rewrite."""
    import m3.eval as eval_mod

    video_id = "vid_legacy"
    annotations_path = _write_annotations_with_video(
        tmp_path, video_id,
        [{"question_id": "q1", "question": "What color?",
          "answer": "red", "type": "object_recognition"}],
    )
    predictions = _write_qa_predictions(
        tmp_path / "qa.jsonl",
        [{"video_id": video_id, "question_id": "q1", "answer": "red shirt"}],
    )
    monkeypatch.setattr(
        eval_mod, "build_judge_llm",
        lambda *a, **kw: _ScriptedJudgeLLM([1.0]),
    )

    out_dir = tmp_path / "eval_out"
    args = eval_mod.parse_args([
        "--annotations", str(annotations_path),
        "--input-qa-results", str(predictions),
        "--output-dir", str(out_dir),
    ])
    eval_mod.run(args)

    rows = [json.loads(line) for line in (out_dir / "m3_eval_results.jsonl").read_text().splitlines() if line]
    assert rows[0]["predicted_answer"] == "red shirt"


# ---------------------------------------------------------------------------
# Regression: summary header must reflect the YAML-configured judge
# (m3.judge_llm in ogmemory.yaml). Pre-fix the summary hardcoded
# ``judge_provider="minimax" / judge_model="MiniMax-M3"`` whenever the
# operator didn't ALSO pass --judge-base-url / --judge-model on the CLI,
# even when ogmemory.yaml named a different provider+model — making the
# per-run summary lie about which backend scored the answers.
# ---------------------------------------------------------------------------


def _write_ogmem_yaml(tmp_path, m3_judge_block: str) -> Path:
    """Materialise a minimal ``ogmemory.yaml`` under tmp_path with the
    caller-supplied ``m3.judge_llm`` block. Top-level fields are
    ``${ENV}`` placeholders; the resolution in
    :func:`m3.config.load_m3_runtime_config` falls back to empty strings
    for unset placeholders, which is fine because nothing in this test
    reads top-level ``llm``/``embedding``."""
    cfg_path = tmp_path / "ogmem.yaml"
    cfg_path.write_text(
        "llm:\n"
        "  provider: \"\"\n"
        "  api_key: \"\"\n"
        "  base_url: \"\"\n"
        "  model: \"\"\n"
        "embedding:\n"
        "  provider: \"mock\"\n"
        "m3:\n"
        f"  judge_llm:\n{m3_judge_block}\n",
        encoding="utf-8",
    )
    return cfg_path


def test_run_summary_reflects_yaml_judge_with_base_url(
    monkeypatch, tmp_path,
):
    """YAML pins provider/host/model → summary header matches."""
    cfg_path = _write_ogmem_yaml(
        tmp_path,
        "    provider: openai\n"
        "    base_url: \"https://api.deepseek.com/v1\"\n"
        "    model: \"deepseek-chat\"\n"
        "    api_key: \"sk-test\"\n",
    )

    import m3.eval as eval_mod

    video_id = "vid_yaml_judge"
    annotations_path = _write_annotations_with_video(
        tmp_path, video_id,
        [{"question_id": "q1", "question": "Q?",
          "answer": "A", "type": "object_recognition"}],
    )
    predictions = _write_qa_predictions(
        tmp_path / "qa.jsonl",
        [{"video_id": video_id, "question_id": "q1",
          "predicted_answer": "A"}],
    )
    monkeypatch.setattr(
        eval_mod, "build_judge_llm",
        lambda *a, **kw: _ScriptedJudgeLLM([1.0]),
    )

    out_dir = tmp_path / "out"
    args = eval_mod.parse_args([
        "--annotations", str(annotations_path),
        "--input-qa-results", str(predictions),
        "--output-dir", str(out_dir),
        "--config", str(cfg_path),
    ])
    eval_mod.run(args)

    summary = json.loads((out_dir / "m3_eval_summary.json").read_text())
    # DeepSeek → host "deepseek" (api.X.com strip).
    assert summary["judge_provider"] == "deepseek"
    assert summary["judge_model"] == "deepseek-chat"


def test_run_summary_reflects_yaml_judge_without_base_url(
    monkeypatch, tmp_path,
):
    """YAML has ``provider: openai`` only (no explicit base_url — uses
    OpenAI's default) → the summary header falls back to the configured
    provider label, not the historic MiniMax default."""
    cfg_path = _write_ogmem_yaml(
        tmp_path,
        "    provider: openai\n"
        "    model: \"gpt-4o-mini\"\n"
        "    api_key: \"sk-test\"\n",
    )

    import m3.eval as eval_mod

    video_id = "vid_yaml_judge_no_url"
    annotations_path = _write_annotations_with_video(
        tmp_path, video_id,
        [{"question_id": "q1", "question": "Q?",
          "answer": "A", "type": "object_recognition"}],
    )
    predictions = _write_qa_predictions(
        tmp_path / "qa.jsonl",
        [{"video_id": video_id, "question_id": "q1",
          "predicted_answer": "A"}],
    )
    monkeypatch.setattr(
        eval_mod, "build_judge_llm",
        lambda *a, **kw: _ScriptedJudgeLLM([1.0]),
    )

    out_dir = tmp_path / "out2"
    args = eval_mod.parse_args([
        "--annotations", str(annotations_path),
        "--input-qa-results", str(predictions),
        "--output-dir", str(out_dir),
        "--config", str(cfg_path),
    ])
    eval_mod.run(args)

    summary = json.loads((out_dir / "m3_eval_summary.json").read_text())
    assert summary["judge_provider"] == "openai"
    assert summary["judge_model"] == "gpt-4o-mini"


def test_run_cli_judge_overrides_yaml_judge(monkeypatch, tmp_path):
    """CLI ``--judge-*`` still wins over YAML — the resolution is
    layered exactly like :func:`m3.llm_factory.build_judge_llm`."""
    cfg_path = _write_ogmem_yaml(
        tmp_path,
        "    provider: openai\n"
        "    model: \"gpt-4o-mini\"\n"
        "    base_url: \"https://api.openai.com/v1\"\n"
        "    api_key: \"sk-yaml\"\n",
    )

    import m3.eval as eval_mod

    video_id = "vid_cli_override"
    annotations_path = _write_annotations_with_video(
        tmp_path, video_id,
        [{"question_id": "q1", "question": "Q?",
          "answer": "A", "type": "object_recognition"}],
    )
    predictions = _write_qa_predictions(
        tmp_path / "qa.jsonl",
        [{"video_id": video_id, "question_id": "q1",
          "predicted_answer": "A"}],
    )
    monkeypatch.setattr(
        eval_mod, "build_judge_llm",
        lambda *a, **kw: _ScriptedJudgeLLM([1.0]),
    )

    out_dir = tmp_path / "out3"
    args = eval_mod.parse_args([
        "--annotations", str(annotations_path),
        "--input-qa-results", str(predictions),
        "--output-dir", str(out_dir),
        "--config", str(cfg_path),
        # CLI overrides: vLLM takes the summary slot.
        "--judge-base-url", "http://my-vllm:8000/v1",
        "--judge-model", "my-finetuned-judge",
    ])
    eval_mod.run(args)

    summary = json.loads((out_dir / "m3_eval_summary.json").read_text())
    # Hostname "my-vllm" survives (no api./.com prefix).
    assert summary["judge_provider"] == "my-vllm"
    assert summary["judge_model"] == "my-finetuned-judge"


def test_run_summary_default_minimax_when_nothing_configured(
    monkeypatch, tmp_path,
):
    """No YAML judge, no CLI judge, no CONFIG env var → historical
    default (``minimax`` / ``MiniMax-M3``) preserved for callers that
    depend on the legacy output shape."""
    import m3.eval as eval_mod

    video_id = "vid_default_judge"
    annotations_path = _write_annotations_with_video(
        tmp_path, video_id,
        [{"question_id": "q1", "question": "Q?",
          "answer": "A", "type": "object_recognition"}],
    )
    predictions = _write_qa_predictions(
        tmp_path / "qa.jsonl",
        [{"video_id": video_id, "question_id": "q1",
          "predicted_answer": "A"}],
    )
    monkeypatch.setattr(
        eval_mod, "build_judge_llm",
        lambda *a, **kw: _ScriptedJudgeLLM([1.0]),
    )
    # No deploy/ogmemory.yaml present in cwd — point at a guaranteed-
    # empty config so the resolution falls cleanly through to defaults.
    empty_cfg = tmp_path / "empty.yaml"
    empty_cfg.write_text(
        "llm:\n  provider: \"\"\n  api_key: \"\"\n  base_url: \"\"\n  model: \"\"\n"
        "embedding:\n  provider: \"mock\"\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out4"
    args = eval_mod.parse_args([
        "--annotations", str(annotations_path),
        "--input-qa-results", str(predictions),
        "--output-dir", str(out_dir),
        "--config", str(empty_cfg),
    ])
    eval_mod.run(args)

    summary = json.loads((out_dir / "m3_eval_summary.json").read_text())
    # Empty YAML provider resolves to "" (empty string), so the
    # ``None or judge_cfg.provider`` short-circuit kicks in: we drop
    # through to the "minimax" historical default. This protects
    # existing consumers of the summary header.
    assert summary["judge_provider"] == "minimax"
    assert summary["judge_model"] == "MiniMax-M3"


@pytest.mark.parametrize("parser_name", ["qa", "eval"])
@pytest.mark.parametrize("limit", [0, 2, 5])
def test_no_gain_cli_configuration(parser_name, limit):
    from m3.qa import parse_react_qa_args
    parser = parse_react_qa_args if parser_name == "qa" else parse_args
    assert parser(["--annotations", "x.json"]).no_gain_limit == 3
    assert parser(["--annotations", "x.json", "--no-gain-limit", str(limit)]).no_gain_limit == limit


@pytest.mark.parametrize("parser_name", ["qa", "eval"])
@pytest.mark.parametrize("limit", [-1, 1])
def test_no_gain_cli_rejects_invalid_threshold(parser_name, limit):
    from m3.qa import parse_react_qa_args
    parser = parse_react_qa_args if parser_name == "qa" else parse_args
    with pytest.raises(SystemExit):
        parser(["--annotations", "x.json", "--no-gain-limit", str(limit)])
