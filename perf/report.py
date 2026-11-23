# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Render Markdown reports from a JSONL trace produced by ``Recorder``.

CLI:
    python -m perf.report <trace.jsonl> [-o report.md]

Output sections:
    1. Run metadata
    2. Stage totals (time, tokens, $cost)
    3. Sub-span breakdown per stage
    4. Model breakdown (total tokens + $ per model)
    5. Bottleneck ranking (top sub-spans by wall time)
    6. Counters appendix (meta)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_events(path: str | os.PathLike[str]) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    # If path is a directory, glob all .jsonl files and merge them.
    # If path is a file, load it (backward compatible).
    if p.is_dir():
        paths = sorted(p.glob("*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"no .jsonl files found in {p}")
    else:
        paths = [p]

    events: list[dict] = []
    for fp in paths:
        with fp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    k = max(0, min(len(values) - 1, int(round(pct * (len(values) - 1)))))
    return values[k]


def _fmt_ms(v: float) -> str:
    return f"{v:,.1f}"


def _fmt_money(v: float) -> str:
    return f"${v:.6f}"


def _fmt_int(v: int) -> str:
    return f"{v:,}"


def _sum_tokens(tokens: dict[str, Any]) -> dict[str, int]:
    llm = tokens.get("llm", {}) or {}
    embed = tokens.get("embed", {}) or {}
    return {
        "llm_input": llm.get("input_tokens", 0),
        "llm_output": llm.get("output_tokens", 0),
        "llm_calls": llm.get("llm_calls", 0),
        "embed_tokens": embed.get("embed_tokens", 0),
        "embed_calls": embed.get("embed_calls", 0),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(events: list[dict]) -> str:
    if not events:
        return "# Perf Report\n\n_(no events)_\n"

    run_ids = sorted({e.get("run_id", "?") for e in events})
    sessions = sorted({e.get("session_id") for e in events if e.get("session_id")})

    # Stage root events have no parent span (parent_span is None).
    # Sub-span events always have a parent_span value (the stage root's span).
    stage_events = [e for e in events if e.get("parent_span") is None]
    subspan_events = [e for e in events if e.get("parent_span") is not None]

    lines: list[str] = []
    lines.append("# ContextEngine Lifecycle Perf Report")
    lines.append("")
    lines.append(f"- **run_id(s):** {', '.join(run_ids)}")
    lines.append(f"- **sessions:** {', '.join(sessions) if sessions else '(none recorded)'}")
    lines.append(f"- **total events:** {len(events)}  (stages={len(stage_events)}, sub-spans={len(subspan_events)})")
    lines.append("")

    # ---- Section 2: stage totals ----
    lines.append("## 1. Stage Totals")
    lines.append("")
    lines.append(
        "| stage | calls | total ms | P50 | P95 | P99 | LLM in | LLM out | embed tok | $ cost |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for e in stage_events:
        by_stage[e["stage"]].append(e)

    total_cost = 0.0
    for stage in sorted(by_stage.keys()):
        evs = by_stage[stage]
        walls = [e.get("wall_ms", 0.0) for e in evs]
        toks_all = [_sum_tokens(e.get("tokens", {})) for e in evs]
        llm_in = sum(t["llm_input"] for t in toks_all)
        llm_out = sum(t["llm_output"] for t in toks_all)
        embed = sum(t["embed_tokens"] for t in toks_all)
        cost = sum((e.get("cost_usd", {}) or {}).get("total", 0.0) for e in evs)
        total_cost += cost
        lines.append(
            "| {stage} | {n} | {tot} | {p50} | {p95} | {p99} | {li} | {lo} | {em} | {c} |".format(
                stage=stage,
                n=len(evs),
                tot=_fmt_ms(sum(walls)),
                p50=_fmt_ms(_percentile(walls, 0.50)),
                p95=_fmt_ms(_percentile(walls, 0.95)),
                p99=_fmt_ms(_percentile(walls, 0.99)),
                li=_fmt_int(llm_in),
                lo=_fmt_int(llm_out),
                em=_fmt_int(embed),
                c=_fmt_money(cost),
            )
        )
    lines.append("")
    lines.append(f"**Total $ cost across all stages:** {_fmt_money(total_cost)}")
    lines.append("")

    # ---- Section 3: sub-span breakdown per stage ----
    lines.append("## 2. Sub-span Breakdown")
    lines.append("")
    by_stage_sub: dict[str, list[dict]] = defaultdict(list)
    for e in subspan_events:
        by_stage_sub[e["stage"]].append(e)
    if not by_stage_sub:
        lines.append("_(no sub-spans recorded — only stage-level events)_")
        lines.append("")
    for stage in sorted(by_stage_sub.keys()):
        lines.append(f"### {stage}")
        lines.append("")
        lines.append("| span | calls | total ms | P50 | P95 | LLM calls | embed calls | $ cost |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        by_name: dict[str, list[dict]] = defaultdict(list)
        for e in by_stage_sub[stage]:
            by_name[e["span"]].append(e)
        for name in sorted(by_name.keys()):
            evs = by_name[name]
            walls = [e.get("wall_ms", 0.0) for e in evs]
            toks_all = [_sum_tokens(e.get("tokens", {})) for e in evs]
            lines.append(
                "| {n} | {c} | {tot} | {p50} | {p95} | {lc} | {ec} | {co} |".format(
                    n=name,
                    c=len(evs),
                    tot=_fmt_ms(sum(walls)),
                    p50=_fmt_ms(_percentile(walls, 0.50)),
                    p95=_fmt_ms(_percentile(walls, 0.95)),
                    lc=sum(t["llm_calls"] for t in toks_all),
                    ec=sum(t["embed_calls"] for t in toks_all),
                    co=_fmt_money(sum((e.get("cost_usd", {}) or {}).get("total", 0.0) for e in evs)),
                )
            )
        lines.append("")

    # ---- Section 4: model breakdown ----
    lines.append("## 3. Model Breakdown")
    lines.append("")
    models: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"tokens": 0, "calls": 0, "cost": 0.0}
    )
    for e in stage_events:
        toks = _sum_tokens(e.get("tokens", {}))
        if toks["llm_input"] + toks["llm_output"] + toks["llm_calls"] > 0:
            key = ("llm", e.get("llm_model") or "(unknown)")
            models[key]["tokens"] += toks["llm_input"] + toks["llm_output"]
            models[key]["calls"] += toks["llm_calls"]
            models[key]["cost"] += (e.get("cost_usd", {}) or {}).get("llm", 0.0)
        if toks["embed_tokens"] + toks["embed_calls"] > 0:
            key = ("embedding", e.get("embed_model") or "(unknown)")
            models[key]["tokens"] += toks["embed_tokens"]
            models[key]["calls"] += toks["embed_calls"]
            models[key]["cost"] += (e.get("cost_usd", {}) or {}).get("embedding", 0.0)
    if not models:
        lines.append("_(no token usage recorded)_")
    else:
        lines.append("| kind | model | total tokens | calls | $ cost |")
        lines.append("|---|---|---:|---:|---:|")
        for (kind, model), stats in sorted(models.items()):
            lines.append(
                f"| {kind} | {model} | {_fmt_int(stats['tokens'])} | {_fmt_int(stats['calls'])} | {_fmt_money(stats['cost'])} |"
            )
    lines.append("")

    # ---- Section 5: bottleneck ranking ----
    lines.append("## 4. Bottleneck Ranking (Top Sub-spans by Wall Time)")
    lines.append("")
    agg_sub: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"total_ms": 0.0, "calls": 0}
    )
    for e in subspan_events:
        key = (e["stage"], e["span"])
        agg_sub[key]["total_ms"] += e.get("wall_ms", 0.0)
        agg_sub[key]["calls"] += 1
    ranked = sorted(agg_sub.items(), key=lambda kv: -kv[1]["total_ms"])[:10]
    if not ranked:
        lines.append("_(no sub-spans recorded)_")
    else:
        lines.append("| rank | stage | span | total ms | calls |")
        lines.append("|---:|---|---|---:|---:|")
        for i, ((stage, span_name), v) in enumerate(ranked, start=1):
            lines.append(f"| {i} | {stage} | {span_name} | {_fmt_ms(v['total_ms'])} | {v['calls']} |")
    lines.append("")

    # ---- Section 6: counters / meta appendix ----
    lines.append("## 5. Counters (meta) Appendix")
    lines.append("")
    counter_keys: set[str] = set()
    for e in events:
        meta = e.get("meta") or {}
        for k, v in meta.items():
            if isinstance(v, (int, float)):
                counter_keys.add(k)
    if not counter_keys:
        lines.append("_(no numeric counters recorded)_")
    else:
        lines.append("| counter | total | avg per event |")
        lines.append("|---|---:|---:|")
        for key in sorted(counter_keys):
            values = [
                float(e.get("meta", {}).get(key, 0))
                for e in events
                if isinstance((e.get("meta") or {}).get(key), (int, float))
            ]
            if values:
                lines.append(
                    f"| {key} | {_fmt_int(int(sum(values)))} | {statistics.mean(values):,.2f} |"
                )
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a perf report from a JSONL trace")
    parser.add_argument("trace", help="Path to perf JSONL file, or a directory containing *.jsonl files for aggregation")
    parser.add_argument("-o", "--output", help="Output Markdown path (default: stdout)")
    args = parser.parse_args(argv)

    events = load_events(args.trace)
    md = render(events)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"wrote report: {args.output}")
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
