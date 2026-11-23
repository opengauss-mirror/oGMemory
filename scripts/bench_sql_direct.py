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

"""Storage backend benchmark: direct PostgreSQL (SQLContextFS etc.)

Measures the same operations as bench_storage.py but connects directly
to PostgreSQL via SQLContextFS / SQLOutboxStore / SQLRelationStore,
bypassing the AGFS server entirely.

Usage:
    OGMEM_BENCH_DSN="postgres://user:password@localhost:5432/ogmemory?sslmode=disable" \
        PYTHONPATH=. python3 scripts/bench_sql_direct.py -o /tmp/bench_sql_direct.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import RequestContext, ContextNode, RelationEdge
from fs.sql_adapter import SQLContextFS
from commit.sql_outbox_store import SQLOutboxStore
from providers.relation_store.sql_relation_store import SQLRelationStore


# Reuse the same measurement infrastructure
@dataclass
class BenchResult:
    name: str
    iterations: int
    total_s: float
    avg_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    ops_per_s: float
    extra: dict[str, Any] = field(default_factory=dict)


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    idx = (len(sorted_data) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def _measure(func, iterations, warmup=3, name=""):
    for _ in range(warmup):
        try:
            func()
        except Exception:
            pass
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            func()
            times.append(time.perf_counter() - t0)
        except Exception:
            times.append(time.perf_counter() - t0)
    if not times:
        raise RuntimeError(f"All {iterations} iterations failed for {name}")
    times.sort()
    total = sum(times)
    ms = [t * 1000 for t in times]
    return BenchResult(
        name=name or func.__name__,
        iterations=len(times),
        total_s=total,
        avg_ms=mean(ms),
        min_ms=ms[0],
        max_ms=ms[-1],
        p50_ms=_percentile(ms, 0.50),
        p95_ms=_percentile(ms, 0.95),
        p99_ms=_percentile(ms, 0.99),
        ops_per_s=len(times) / total if total > 0 else 0,
    )


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

ACCOUNT = "bench"
USER = "u-bench"
AGENT = "a-bench"


def _ctx():
    return RequestContext(
        account_id=ACCOUNT, user_id=USER, agent_id=AGENT,
        session_id="bench-session", trace_id=str(uuid.uuid4()),
    )


def _profile_node(content_size=500):
    return ContextNode(
        uri=f"ctx://{ACCOUNT}/users/{USER}/memories/profile",
        context_type="MEMORY", category="profile", level=0,
        owner_space=f"user:{USER}",
        abstract="Benchmark user profile",
        overview="## Profile\nBenchmark user overview text for testing.",
        content="x" * content_size,
        metadata={},
    )


def _pref_node(slug, content_size=300):
    return ContextNode(
        uri=f"ctx://{ACCOUNT}/users/{USER}/memories/preferences/{slug}",
        context_type="MEMORY", category="preference", level=0,
        owner_space=f"user:{USER}",
        abstract=f"Preference: {slug}",
        overview=f"## {slug}\nPreference overview.",
        content="p" * content_size,
        metadata={},
    )


def _make_edges(uri, count=3):
    return [
        RelationEdge(
            from_uri=uri,
            to_uri=f"ctx://{ACCOUNT}/users/{USER}/memories/entities/entity_{i}",
            relation_type="related_to",
            weight=0.5 + i * 0.1,
            reason=f"test relation {i}",
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

class SQLBenchmark:
    def __init__(self, dsn: str, pool_size: int = 5):
        self.dsn = dsn
        self.fs = SQLContextFS(connection_string=dsn, pool_size=pool_size)
        self.relations = SQLRelationStore(connection_string=dsn, pool_size=pool_size)
        # SQLOutboxStore needs fs reference
        self.outbox = SQLOutboxStore(connection_string=dsn, fs=self.fs, pool_size=pool_size)
        self.results: list[BenchResult] = []
        self._cleanup()

    def _cleanup(self):
        import psycopg2
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM context_nodes WHERE uri LIKE 'ctx://bench/%'")
                cur.execute("DELETE FROM outbox_events WHERE account_id = 'bench'")
                cur.execute("DELETE FROM relation_edges WHERE from_uri LIKE 'ctx://bench/%'")
            conn.commit()
        finally:
            conn.close()

    def _run(self, name, func, iterations, warmup=3):
        r = _measure(func, iterations, warmup=warmup, name=name)
        self.results.append(r)
        return r

    def bench_write_node(self, n=100):
        ctx = _ctx()
        counter = [0]
        def fn():
            counter[0] += 1
            self.fs.write_node(_pref_node(f"bench_w_{counter[0]}"), ctx)
        return self._run("write_node", fn, n)

    def bench_write_merge(self, n=50):
        ctx = _ctx()
        self.fs.write_node(_profile_node(), ctx)
        counter = [0]
        def fn():
            counter[0] += 1
            self.fs.write_node(_profile_node(content_size=500 + counter[0]), ctx)
        return self._run("write_merge", fn, n)

    def bench_read_node(self, n=200):
        ctx = _ctx()
        node = _profile_node()
        self.fs.write_node(node, ctx)
        def fn():
            self.fs.read_node(node.uri, ctx)
        return self._run("read_node", fn, n)

    def bench_exists(self, n=300):
        ctx = _ctx()
        node = _profile_node()
        self.fs.write_node(node, ctx)
        def fn():
            self.fs.exists(node.uri, ctx)
        return self._run("exists", fn, n)

    def bench_exists_miss(self, n=200):
        ctx = _ctx()
        def fn():
            self.fs.exists(
                f"ctx://{ACCOUNT}/users/{USER}/memories/preferences/nonexistent_{uuid.uuid4().hex[:6]}",
                ctx,
            )
        return self._run("exists_miss", fn, n)

    def bench_list_children(self, child_counts=None, n=50):
        if child_counts is None:
            child_counts = [10, 50, 100]
        results = []
        ctx = _ctx()
        for count in child_counts:
            for i in range(count):
                self.fs.write_node(_pref_node(f"lc_{count}_{i}", content_size=100), ctx)
            parent_uri = f"ctx://{ACCOUNT}/users/{USER}/memories/preferences"
            def fn():
                self.fs.list_children(parent_uri, ctx)
            r = self._run(f"list_children[{count}]", fn, n, warmup=2)
            r.extra["child_count"] = count
            results.append(r)
            # cleanup
            for i in range(count):
                try:
                    self.fs.delete_node(
                        f"ctx://{ACCOUNT}/users/{USER}/memories/preferences/lc_{count}_{i}", ctx)
                except Exception:
                    pass
        return results

    def bench_delete_node(self, n=50):
        ctx = _ctx()
        for i in range(n + 5):
            self.fs.write_node(_pref_node(f"del_{i}", content_size=100), ctx)
        counter = [0]
        def fn():
            counter[0] += 1
            try:
                self.fs.delete_node(
                    f"ctx://{ACCOUNT}/users/{USER}/memories/preferences/del_{counter[0]}", ctx)
            except Exception:
                pass
        return self._run("delete_node", fn, n, warmup=0)

    def bench_move_node(self, n=30):
        ctx = _ctx()
        for i in range(n + 5):
            self.fs.write_node(_pref_node(f"mv_src_{i}", content_size=100), ctx)
        counter = [0]
        def fn():
            counter[0] += 1
            try:
                self.fs.move_node(
                    f"ctx://{ACCOUNT}/users/{USER}/memories/preferences/mv_src_{counter[0]}",
                    f"ctx://{ACCOUNT}/users/{USER}/memories/preferences/mv_dst_{counter[0]}",
                    ctx)
            except Exception:
                pass
        return self._run("move_node", fn, n, warmup=0)

    def bench_outbox_roundtrip(self, n=50):
        ctx = _ctx()
        node = _profile_node()
        self.fs.write_node(node, ctx)
        def fn():
            event = self.outbox.register_write(node, ctx)
            pending = self.outbox.list_pending(ACCOUNT)
            if pending:
                _, ev = pending[0]
                self.outbox.mark_done(ev, node.uri)
        return self._run("outbox_roundtrip", fn, n)

    def bench_outbox_register(self, n=100):
        ctx = _ctx()
        counter = [0]
        def fn():
            counter[0] += 1
            node = _pref_node(f"ob_reg_{counter[0]}", content_size=200)
            self.fs.write_node(node, ctx)
            self.outbox.register_write(node, ctx)
        return self._run("outbox_register", fn, n)

    def bench_relation_upsert(self, n=50):
        ctx = _ctx()
        uri = f"ctx://{ACCOUNT}/users/{USER}/memories/profile"
        def fn():
            self.relations.upsert_edges(_make_edges(uri, count=5), ctx)
        return self._run("relation_upsert_5", fn, n)

    def bench_relation_get(self, n=100):
        ctx = _ctx()
        uri = f"ctx://{ACCOUNT}/users/{USER}/memories/profile"
        self.relations.upsert_edges(_make_edges(uri, count=5), ctx)
        def fn():
            self.relations.get_edges(uri, ctx)
        return self._run("relation_get", fn, n)

    def bench_write_read_cycle(self, n=50):
        ctx = _ctx()
        counter = [0]
        def fn():
            counter[0] += 1
            node = _pref_node(f"wrc_{counter[0]}", content_size=400)
            self.fs.write_node(node, ctx)
            self.fs.read_node(node.uri, ctx)
        return self._run("write_read_cycle", fn, n)

    def bench_write_with_relations(self, n=50):
        ctx = _ctx()
        counter = [0]
        def fn():
            counter[0] += 1
            uri = f"ctx://{ACCOUNT}/users/{USER}/memories/preferences/wr_{counter[0]}"
            edges = _make_edges(uri, count=5)
            node = ContextNode(
                uri=uri, context_type="MEMORY", category="preference", level=0,
                owner_space=f"user:{USER}",
                abstract="With relations", overview="## Pref with edges",
                content="content " * 50, metadata={"_relations": edges},
            )
            self.fs.write_node(node, ctx)
        return self._run("write_with_relations", fn, n)

    def run_all(self):
        self._cleanup()
        print(f"\n{'=' * 70}")
        print(f"  Storage Benchmark — PostgreSQL (direct)")
        print(f"  DSN: {self.dsn.split('@')[-1]}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}\n")

        self.bench_write_node()
        self.bench_write_merge()
        self.bench_read_node()
        self.bench_exists()
        self.bench_exists_miss()
        self.bench_list_children()
        self.bench_delete_node()
        self.bench_move_node()
        self.bench_outbox_roundtrip()
        self.bench_outbox_register()
        self.bench_relation_upsert()
        self.bench_relation_get()
        self.bench_write_read_cycle()
        self.bench_write_with_relations()

        _print_results(self.results, "PostgreSQL (direct)")
        return self.results


def _print_results(results, label=""):
    header = f"  Results: {label}" if label else "  Results"
    print(f"\n{'=' * 70}")
    print(header)
    print(f"{'=' * 70}")
    print(f"  {'Name':<25} {'N':>5} {'P50':>8} {'P95':>8} {'P99':>8} {'ops/s':>8}")
    print(f"  {'-'*25} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(
            f"  {r.name:<25} {r.iterations:>5} "
            f"{r.p50_ms:>7.1f}ms {r.p95_ms:>7.1f}ms {r.p99_ms:>7.1f}ms "
            f"{r.ops_per_s:>7.0f}"
        )
    print(f"{'=' * 70}\n")


def _save_json(results, path, label):
    data = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="oG-Memory SQL direct benchmark")
    parser.add_argument(
        "--dsn",
        default=os.getenv("OGMEM_BENCH_DSN"),
        help="PostgreSQL connection string, or set OGMEM_BENCH_DSN",
    )
    parser.add_argument("-o", "--output", default="", help="Save results to JSON")
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or OGMEM_BENCH_DSN is required")

    bench = SQLBenchmark(dsn=args.dsn)
    results = bench.run_all()

    if args.output:
        _save_json(results, args.output, "sql_direct")


if __name__ == "__main__":
    main()
