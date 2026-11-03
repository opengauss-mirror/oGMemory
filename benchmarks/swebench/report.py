#!/usr/bin/env python3
"""Report final attempts, with missing evidence distinct from zero/false."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from prepare import read_json, save_json

TOKEN_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)


def session_metrics(path):
    events, malformed = [], False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("not an event")
            events.append(item)
        except ValueError:
            malformed = True
    ends = [e for e in events if e.get("type") == "result"]
    end = ends[-1] if len(ends) == 1 else {}
    usage = end.get("usage", {})
    complete_usage = (
        bool(end)
        and not malformed
        and all(type(usage.get(k)) is int and usage[k] >= 0 for k in TOKEN_FIELDS)
    )
    calls, failed, succeeded = {}, set(), set()
    conflict = False
    for event in events:
        if event.get("type") not in {"assistant", "user"}:
            continue
        content = event.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                ident, name = block.get("id"), block.get("name")
                if not ident or not name or (ident in calls and calls[ident] != name):
                    conflict = True
                else:
                    calls[ident] = name
            if block.get("type") == "tool_result" and block.get("is_error"):
                failed.add(block.get("tool_use_id"))
            elif block.get("type") == "tool_result":
                succeeded.add(block.get("tool_use_id"))
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
    servers, plugins = init.get("mcp_servers", []), init.get("plugins", [])
    connected = any(s.get("name") == "ogmemory" and s.get("status") == "connected" for s in servers)
    plugin_names = [p.get("name") if isinstance(p, dict) else p for p in plugins]
    isolation = (
        bool(init)
        and all(s.get("name") == "ogmemory" for s in servers)
        and all(p == "og-memory" for p in plugin_names)
    )
    isolation = isolation and (
        (connected and plugin_names == ["og-memory"] and len(servers) == 1)
        or (not servers and not plugins)
    )
    return {
        "agent_completed": end.get("subtype") == "success" and end.get("is_error") is False,
        "isolation_valid": isolation,
        "sql_connected": connected,
        "sql_tool_succeeded": any(
            calls.get(i, "").startswith("mcp__ogmemory__") for i in succeeded
        ),
        "usage": {
            "evidence_status": "COMPLETE" if complete_usage else "PARTIAL",
            "total_tokens": sum(usage[k] for k in TOKEN_FIELDS) if complete_usage else None,
            **{k: usage.get(k) for k in TOKEN_FIELDS},
        },
        "actions": {
            "evidence_status": "COMPLETE" if end and not malformed and not conflict else "PARTIAL",
            "total": len(calls) if end and not malformed and not conflict else None,
            "by_tool": dict(Counter(calls.values())),
            "errored_tool_results": len(failed),
        },
    }


def final_rows(batch):
    manifest = read_json(batch / "manifest.json")
    dataset = read_json(batch / "dataset.json")
    rows = []
    for instance in dataset:
        for arm in manifest["config"]["arms"]:
            attempts = sorted(
                (batch / "jobs" / f"{arm}--{instance['instance_id']}").glob("attempt-*/result.json")
            )
            result = read_json(attempts[-1]) if attempts else {}
            # Read the saved official verdict; a completed process alone is not a verdict.
            validation = (
                attempts[-1].parent / "evaluation" / "validation.json" if attempts else None
            )
            v = read_json(validation) if validation and validation.exists() else {}
            valid = (
                result.get("status") == "completed"
                and v.get("status") == "completed"
                and type(v.get("resolved")) is bool
            )
            usage, actions = result.get("usage", {}), result.get("actions", {})
            rows.append(
                {
                    "instance_id": instance["instance_id"],
                    "arm": arm,
                    "attempt": attempts[-1].parent.name if attempts else None,
                    "status": result.get("status", "pending"),
                    "resolved": v["resolved"] if valid else None,
                    "tokens": usage.get("total_tokens")
                    if usage.get("evidence_status") == "COMPLETE"
                    else None,
                    "actions": actions.get("total")
                    if actions.get("evidence_status") == "COMPLETE"
                    else None,
                    "usage_evidence": usage.get("evidence_status", "MISSING"),
                    "action_evidence": actions.get("evidence_status", "MISSING"),
                    "error": result.get("error"),
                    "result_path": str(attempts[-1]) if attempts else None,
                }
            )
    return rows


def mean_complete(rows, field):
    if not rows or any(type(r[field]) is not int for r in rows):
        return None
    return sum(r[field] for r in rows) / len(rows)


def summarize(rows):
    summary = []
    for arm in sorted({r["arm"] for r in rows}):
        group = [r for r in rows if r["arm"] == arm]
        valid = [r for r in group if type(r["resolved"]) is bool]
        solved = [r for r in valid if r["resolved"]]
        summary.append(
            {
                "arm": arm,
                "planned": len(group),
                "valid": len(valid),
                "resolved": len(solved),
                "unresolved": len(valid) - len(solved),
                "no_verdict": len(group) - len(valid),
                "success_rate": len(solved) / len(group),
                "valid_accuracy": len(solved) / len(valid) if valid else None,
                "tokens_complete": sum(r["tokens"] is not None for r in group),
                "actions_complete": sum(r["actions"] is not None for r in group),
                "mean_tokens": mean_complete(group, "tokens"),
                "mean_actions": mean_complete(group, "actions"),
            }
        )
    return summary


def paired_comparison(rows):
    by_arm = {
        arm: {r["instance_id"]: r for r in rows if r["arm"] == arm} for arm in ("baseline", "sql")
    }
    common = sorted(set(by_arm["baseline"]) & set(by_arm["sql"]))
    ids = [
        iid
        for iid in common
        if all(
            by_arm[a][iid]["resolved"] is True
            and by_arm[a][iid]["tokens"] is not None
            and by_arm[a][iid]["actions"] is not None
            for a in by_arm
        )
    ]
    result = {
        "scope": "both_resolved_with_complete_usage_and_actions",
        "count": len(ids),
        "instance_ids": ids,
    }
    for field in ("tokens", "actions"):
        baseline = mean_complete([by_arm["baseline"][i] for i in ids], field)
        sql = mean_complete([by_arm["sql"][i] for i in ids], field)
        result[field] = {
            "baseline_mean": baseline,
            "sql_mean": sql,
            "reduction_percent": 100 * (baseline - sql) / baseline if baseline else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    batch = args.batch.resolve()
    rows = final_rows(batch)
    summary = summarize(rows)
    comparison = paired_comparison(rows)
    for name, data in [("tasks", rows), ("summary", summary)]:
        with (batch / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    save_json(
        batch / "summary.json",
        {"attempt_policy": "final_recorded_attempt", "summary": summary, "paired": comparison},
    )
    lines = [
        "# SWE-bench 测试报告",
        "",
        f"运行目录：{batch}",
        "",
        "按任务最后记录的 attempt 汇总；空值不补零。此报告可在运行中生成，未完成任务列为无判定。",
        "",
        "| 方案 | 计划 | 有效评分 | 修复成功 | 无判定 | Token 完整数 | 平均 Token | 调用完整数 | 平均调用 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary:
        values = [
            r[k]
            for k in (
                "arm",
                "planned",
                "valid",
                "resolved",
                "no_verdict",
                "tokens_complete",
                "mean_tokens",
                "actions_complete",
                "mean_actions",
            )
        ]
        lines.append("| " + " | ".join("—" if v is None else str(v) for v in values) + " |")
    lines += ["", f"双方均修复成功且用量完整的共同题目：{comparison['count']} 道。", ""]
    if comparison["count"]:
        for field in ("tokens", "actions"):
            r = comparison[field]
            reduction = r["reduction_percent"]
            reduction_text = (
                f"{reduction}%" if reduction is not None else "不计算（baseline 均值为 0）"
            )
            lines.append(
                f"- {field}: baseline={r['baseline_mean']}, SQL={r['sql_mean']}, 下降比例={reduction_text}。"
            )
    else:
        lines.append("无满足条件的共同题目，不计算 baseline 与 SQL 的效率差异。")
    lines += [
        "",
        "Token 包含缓存且只计一次，不等同于费用；工具调用包含失败调用。",
        "SQL 使用代码检索、编辑和输出过滤组合；不将差异单独归因于数据库后端。",
        "主表不包含早期重试、Embedding 或评分成本。样本不足或配置发生变化时不能外推收益。",
        "",
    ]
    (batch / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(batch / "report.md")


if __name__ == "__main__":
    main()
