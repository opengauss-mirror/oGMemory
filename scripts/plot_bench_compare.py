#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Generate comparison chart: AGFS localfs vs PostgreSQL direct."""

import json
import sys
from pathlib import Path

# Results data (from benchmark runs)
LOCALFS = {
    "write_node": {"p50": 9.5, "p95": 12.5, "p99": 13.3, "ops": 102},
    "write_merge": {"p50": 8.9, "p95": 11.3, "p99": 12.3, "ops": 117},
    "read_node": {"p50": 3.6, "p95": 5.3, "p99": 6.3, "ops": 268},
    "exists": {"p50": 0.8, "p95": 1.4, "p99": 1.8, "ops": 1181},
    "exists_miss": {"p50": 0.8, "p95": 1.3, "p99": 1.7, "ops": 1136},
    "list_children[10]": {"p50": 94.7, "p95": 105.1, "p99": 112.7, "ops": 11},
    "list_children[50]": {"p50": 127.3, "p95": 135.8, "p99": 140.9, "ops": 8},
    "list_children[100]": {"p50": 170.6, "p95": 189.1, "p99": 203.0, "ops": 6},
    "delete_node": {"p50": 0.9, "p95": 1.5, "p99": 1.8, "ops": 1006},
    "move_node": {"p50": 1.9, "p95": 3.1, "p99": 3.7, "ops": 481},
    "outbox_roundtrip": {"p50": 127.8, "p95": 144.2, "p99": 174.4, "ops": 8},
    "outbox_register": {"p50": 14.2, "p95": 17.9, "p99": 19.8, "ops": 70},
    "relation_upsert_5": {"p50": 1.8, "p95": 2.8, "p99": 3.4, "ops": 546},
    "relation_get": {"p50": 0.8, "p95": 1.3, "p99": 1.7, "ops": 1188},
    "write_read_cycle": {"p50": 14.5, "p95": 17.0, "p99": 17.9, "ops": 70},
    "write_with_relations": {"p50": 9.9, "p95": 13.3, "p99": 14.2, "ops": 99},
}

SQL_DIRECT = {
    "write_node": {"p50": 1.2, "p95": 2.2, "p99": 2.8, "ops": 775},
    "write_merge": {"p50": 1.2, "p95": 2.1, "p99": 2.2, "ops": 791},
    "read_node": {"p50": 0.1, "p95": 0.4, "p99": 0.5, "ops": 6385},
    "exists": {"p50": 0.2, "p95": 0.3, "p99": 1.4, "ops": 5197},
    "exists_miss": {"p50": 0.3, "p95": 0.5, "p99": 0.9, "ops": 3006},
    "list_children[10]": {"p50": 0.2, "p95": 0.3, "p99": 0.4, "ops": 5093},
    "list_children[50]": {"p50": 0.1, "p95": 0.3, "p99": 0.3, "ops": 5477},
    "list_children[100]": {"p50": 0.2, "p95": 0.3, "p99": 0.4, "ops": 4258},
    "delete_node": {"p50": 1.1, "p95": 1.8, "p99": 2.3, "ops": 978},
    "move_node": {"p50": 1.5, "p95": 3.1, "p99": 3.3, "ops": 543},
    "outbox_roundtrip": {"p50": 2.3, "p95": 4.7, "p99": 5.4, "ops": 381},
    "outbox_register": {"p50": 3.0, "p95": 4.9, "p99": 5.5, "ops": 339},
    "relation_upsert_5": {"p50": 1.7, "p95": 3.0, "p99": 3.5, "ops": 556},
    "relation_get": {"p50": 0.2, "p95": 0.3, "p99": 0.3, "ops": 5477},
    "write_read_cycle": {"p50": 1.5, "p95": 2.9, "p99": 3.2, "ops": 573},
    "write_with_relations": {"p50": 1.5, "p95": 2.4, "p99": 3.2, "ops": 682},
}


def bar(value_a, value_b, max_val, width=30):
    """Return two ASCII bars for comparison."""
    wa = int(value_a / max_val * width) if max_val > 0 else 0
    wb = int(value_b / max_val * width) if max_val > 0 else 0
    return "█" * wa + "░" * (width - wa), "█" * wb + "░" * (width - wb)


def main():
    names = list(LOCALFS.keys())

    # ── Section 1: P50 Latency Comparison ──────────────────────────────
    print("\n" + "=" * 85)
    print("  P50 LATENCY COMPARISON (ms) — lower is better")
    print("=" * 85)

    max_p50 = max(max(v["p50"] for v in LOCALFS.values()),
                  max(v["p50"] for v in SQL_DIRECT.values()))

    print(f"  {'Operation':<24} {'AGFS/localfs':>12} {'PG direct':>12} {'ratio':>7}  Bar")
    print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*7}  {'-'*35}")

    for name in names:
        la = LOCALFS[name]["p50"]
        lb = SQL_DIRECT[name]["p50"]
        ratio = la / lb if lb > 0 else float("inf")
        winner = "PG" if ratio > 1 else "agfs" if ratio < 1 else "="
        bar_a, bar_b = bar(la, lb, max_p50, width=15)
        print(f"  {name:<24} {la:>10.1f}ms {lb:>10.1f}ms {ratio:>6.1f}x  {bar_a} vs {bar_b}")

    # ── Section 2: Throughput Comparison ───────────────────────────────
    print("\n" + "=" * 85)
    print("  THROUGHPUT COMPARISON (ops/s) — higher is better")
    print("=" * 85)

    max_ops = max(max(v["ops"] for v in LOCALFS.values()),
                  max(v["ops"] for v in SQL_DIRECT.values()))

    print(f"  {'Operation':<24} {'AGFS/localfs':>12} {'PG direct':>12} {'ratio':>7}")
    print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*7}")

    for name in names:
        oa = LOCALFS[name]["ops"]
        ob = SQL_DIRECT[name]["ops"]
        ratio = ob / oa if oa > 0 else 0
        print(f"  {name:<24} {oa:>12.0f} {ob:>12.0f} {ratio:>6.1f}x")

    # ── Section 3: Highlights ──────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  KEY FINDINGS")
    print("=" * 85)

    # Find biggest wins and losses
    ratios = []
    for name in names:
        la = LOCALFS[name]["p50"]
        lb = SQL_DIRECT[name]["p50"]
        ratios.append((name, la / lb if lb > 0 else float("inf"), la, lb))

    ratios.sort(key=lambda x: -x[1])

    print("\n  PG direct biggest wins:")
    for name, ratio, la, lb in ratios[:5]:
        print(f"    {name:<24}  {la:.1f}ms → {lb:.1f}ms  ({ratio:.1f}x faster)")

    print("\n  Closest (smallest gap):")
    for name, ratio, la, lb in ratios[-3:]:
        print(f"    {name:<24}  {la:.1f}ms → {lb:.1f}ms  ({ratio:.1f}x)")

    # ── Section 4: list_children scaling ───────────────────────────────
    print("\n  list_children scaling (P50 ms):")
    print(f"    {'children':>10} {'AGFS/localfs':>14} {'PG direct':>14} {'speedup':>10}")
    for c in [10, 50, 100]:
        key = f"list_children[{c}]"
        la = LOCALFS[key]["p50"]
        lb = SQL_DIRECT[key]["p50"]
        print(f"    {c:>10} {la:>12.1f}ms {lb:>12.1f}ms {la/lb:>9.1f}x")

    print("\n  AGFS list_children grows linearly (file listing per child × N).")
    print("  PG list_children is O(1) — constant ~0.2ms regardless of child count.")
    print("=" * 85)


if __name__ == "__main__":
    main()
