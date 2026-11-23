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

"""ContextEngine 性能基准测试

⚠️ 注意：本脚本使用 MockLLM 和 MockEmbedder，不调用真实的 OpenAI API。
   因此测得的性能不包含网络延迟和 OpenAI API 响应时间。
   如需测试完整的端到端性能，请配置真实的 OpenAI API 并使用 OpenAILLM/OpenAIEmbedder。

测量各组件的性能表现，包括:
- 写入延迟 (P50, P95, P99)
- 提取性能
- Merge 性能
- 向量化性能
- 检索性能
"""

import argparse
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Dict, Any, Callable

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import RequestContext, CandidateMemory, ContextNode
from fs.agfs_adapter import AGFSContextFS
from pyagfs import AGFSClient
from service.api import MemoryWriteAPI
from providers.llm import MockLLM
from providers.embedder import MockEmbedder
from providers.embedder.openai_embedder import OpenAIEmbedder
from providers.vector_index import InMemoryVectorIndex
from commit import OutboxStore, ContextWriter, ArchiveBuilder, PolicyRouter
from extraction import Extractor


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    p50: float
    p95: float
    p99: float
    ops_per_second: float
    details: Dict[str, Any]


class PerformanceBenchmark:
    """性能基准测试器"""

    def __init__(self, agfs_base_url: str = "http://localhost:1833"):
        self.agfs_base_url = agfs_base_url
        self.client = AGFSClient(api_base_url=agfs_base_url)
        self.fs = AGFSContextFS(client=self.client, mount_prefix="/local/benchmark")
        self.llm = MockLLM()
        self.embedder = MockEmbedder(dimension=1536)
        self.vector_index = InMemoryVectorIndex(dimension=1536)
        self.outbox_store = OutboxStore(client=self.client, fs=self.fs, mount_prefix="/local/benchmark")

        # 创建测试目录
        try:
            self.client.mkdir("/local/benchmark")
        except:
            pass

    def create_context(self, account_id: str = "benchmark", user_id: str = "user") -> RequestContext:
        """创建 RequestContext"""
        return RequestContext(
            account_id=account_id,
            user_id=user_id,
            agent_id="benchmark-agent",
            session_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
        )

    def _measure(self, func: Callable, iterations: int, *args, **kwargs) -> BenchmarkResult:
        """测量函数执行时间"""
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                times.append(time.perf_counter() - start)
            except Exception as e:
                print(f"  错误: {e}")
                continue

        if not times:
            raise RuntimeError("所有迭代都失败了")

        times.sort()
        total_time = sum(times)

        return BenchmarkResult(
            name=func.__name__,
            iterations=len(times),
            total_time=total_time,
            avg_time=mean(times),
            min_time=times[0],
            max_time=times[-1],
            p50=times[len(times) // 2],
            p95=times[int(len(times) * 0.95)] if len(times) > 20 else times[-1],
            p99=times[int(len(times) * 0.99)] if len(times) > 100 else times[-1],
            ops_per_second=len(times) / total_time if total_time > 0 else 0,
            details={},
        )

    def benchmark_single_write(self, iterations: int = 100) -> BenchmarkResult:
        """基准测试 1: 单条写入延迟"""
        print(f"\n{'='*60}")
        print(f"基准测试 1: 单条写入延迟 ({iterations} 次迭代)")
        print(f"{'='*60}")

        ctx = self.create_context()

        def write_single():
            candidate = CandidateMemory(
                category="preference",
                owner_scope="user",
                routing_key=f"pref_{uuid.uuid4().hex[:8]}",
                abstract="Test preference",
                overview="## Test\n\nContent",
                content="Test content for benchmarking",
                confidence=0.9,
            )
            api = MemoryWriteAPI(fs=self.fs, llm=self.llm, outbox_store=self.outbox_store)
            return api.write_memory(candidate, ctx)

        result = self._measure(write_single, iterations)
        self._print_result(result)
        return result

    def benchmark_batch_write(self, batch_size: int = 10, iterations: int = 50) -> BenchmarkResult:
        """基准测试 2: 批量写入性能"""
        print(f"\n{'='*60}")
        print(f"基准测试 2: 批量写入性能 ({batch_size} 条/批, {iterations} 次迭代)")
        print(f"{'='*60}")

        ctx = self.create_context()

        def write_batch():
            candidates = [
                CandidateMemory(
                    category="preference",
                    owner_scope="user",
                    routing_key=f"pref_{i % 20}",
                    abstract=f"Preference {i}",
                    overview="## Test",
                    content=f"Content {i}",
                    confidence=0.9,
                )
                for i in range(batch_size)
            ]
            api = MemoryWriteAPI(fs=self.fs, llm=self.llm, outbox_store=self.outbox_store)
            return api.write_memories(candidates, ctx, parallel=False)

        result = self._measure(write_batch, iterations)
        result.details = {"batch_size": batch_size}
        self._print_result(result)
        return result

    def benchmark_extraction(self, iterations: int = 50) -> BenchmarkResult:
        """基准测试 3: LLM 提取性能"""
        print(f"\n{'='*60}")
        print(f"基准测试 3: LLM 提取性能 ({iterations} 次迭代)")
        print(f"{'='*60}")

        ctx = self.create_context()
        extractor = Extractor(self.llm)

        # 配置 mock 返回
        self.llm._mock_tool_calls = [
            {
                "tool": "extract_preference",
                "input": {
                    "routing_key": "coffee",
                    "abstract": "Likes coffee",
                    "overview": "Overview",
                    "content": "Content",
                    "confidence": 0.9,
                },
            }
        ]

        messages = [
            {"role": "user", "content": "I like coffee with milk"},
            {"role": "assistant", "content": "Got it"},
        ]

        def extract():
            return extractor.extract(messages, ctx)

        result = self._measure(extract, iterations)
        self._print_result(result)
        return result

    def benchmark_merge(self, iterations: int = 100) -> BenchmarkResult:
        """基准测试 4: Merge 性能"""
        print(f"\n{'='*60}")
        print(f"基准测试 4: Merge 性能 ({iterations} 次迭代)")
        print(f"{'='*60}")

        ctx = self.create_context()
        writer = ContextWriter(fs=self.fs, llm=self.llm, outbox_store=self.outbox_store)

        def merge():
            candidate = CandidateMemory(
                category="preference",
                owner_scope="user",
                routing_key="coffee",  # 相同 routing_key 触发 merge
                abstract="Likes coffee",
                overview="## Coffee Preference\n\nDark roast, 2 cups/day",
                content="Detailed preference content here",
                confidence=0.9,
            )
            return writer.write_candidate(candidate, ctx)

        # 首先创建初始节点
        try:
            merge()
        except:
            pass

        result = self._measure(merge, iterations)
        self._print_result(result)
        return result

    def benchmark_embedding(self, batch_sizes: List[int] = [1, 10, 50, 100]) -> List[BenchmarkResult]:
        """基准测试 5: 向量化性能 (不同批次大小)"""
        print(f"\n{'='*60}")
        print(f"基准测试 5: 向量化性能")
        print(f"{'='*60}")

        results = []

        texts = [f"This is test text number {i} for embedding benchmark." for i in range(100)]

        for batch_size in batch_sizes:
            def embed_batch():
                return self.embedder.embed_texts(texts[:batch_size])

            result = self._measure(embed_batch, 50)
            result.name = f"embedding_batch_{batch_size}"
            result.details = {"batch_size": batch_size}
            results.append(result)

            print(f"  批次大小 {batch_size}: {result.avg_time*1000:.2f}ms avg, {result.ops_per_second:.0f} ops/s")

        return results

    def benchmark_vector_search(self, index_sizes: List[int] = [100, 500, 1000, 5000]) -> List[BenchmarkResult]:
        """基准测试 6: 向量检索性能 (不同索引大小)"""
        print(f"\n{'='*60}")
        print(f"基准测试 6: 向量检索性能")
        print(f"{'='*60}")

        from core.models import TypedQuery, IndexRecord

        results = []

        for size in index_sizes:
            # 创建索引
            vi = InMemoryVectorIndex(dimension=1536)
            records = [
                IndexRecord(
                    id=f"rec_{i}",
                    uri=f"ctx://test/users/user/memories/pref_{i}",
                    level=0,
                    text=f"Text content for record {i}",
                    filters={"account_id": "test", "owner_space": "user:user"},
                )
                for i in range(size)
            ]
            vi.upsert(records)

            def search():
                query = TypedQuery(
                    text="search query",
                    context_type="MEMORY",
                    categories=["preference"],
                    account_id="test",
                    owner_space="user:user",
                )
                return vi.search(query)

            result = self._measure(search, 50)
            result.name = f"search_index_{size}"
            result.details = {"index_size": size}
            results.append(result)

            print(f"  索引大小 {size}: {result.avg_time*1000:.2f}ms avg")

        return results

    def benchmark_concurrent_writes(self, threads: int = 10, writes_per_thread: int = 50) -> BenchmarkResult:
        """基准测试 7: 并发写入性能"""
        print(f"\n{'='*60}")
        print(f"基准测试 7: 并发写入性能 ({threads} 线程)")
        print(f"{'='*60}")

        times = []
        import threading
        lock = threading.Lock()

        def write_concurrent(thread_id: int):
            thread_times = []
            ctx = self.create_context("concurrent", f"user-{thread_id}")

            for i in range(writes_per_thread):
                start = time.perf_counter()
                try:
                    candidate = CandidateMemory(
                        category="preference",
                        owner_scope="user",
                        routing_key=f"pref_{i}",
                        abstract=f"Preference {i}",
                        overview="## Test",
                        content=f"Content {i}",
                        confidence=0.9,
                    )
                    api = MemoryWriteAPI(fs=self.fs, llm=self.llm)
                    api.write_memory(candidate, ctx)
                    thread_times.append(time.perf_counter() - start)
                except Exception as e:
                    pass

            with lock:
                times.extend(thread_times)

        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(write_concurrent, i) for i in range(threads)]
            for f in futures:
                f.result()

        total_time = time.perf_counter() - start

        if times:
            times.sort()
            total_ops = threads * writes_per_thread

            result = BenchmarkResult(
                name="concurrent_writes",
                iterations=len(times),
                total_time=total_time,
                avg_time=mean(times),
                min_time=times[0],
                max_time=times[-1],
                p50=times[len(times) // 2],
                p95=times[int(len(times) * 0.95)],
                p99=times[int(len(times) * 0.99)],
                ops_per_second=total_ops / total_time,
                details={"threads": threads, "writes_per_thread": writes_per_thread},
            )
            self._print_result(result)
            return result

        raise RuntimeError("并发测试失败")

    def _print_result(self, result: BenchmarkResult):
        """打印基准测试结果"""
        print(f"\n  结果:")
        print(f"    迭代次数: {result.iterations}")
        print(f"    总耗时: {result.total_time:.3f} 秒")
        print(f"    平均延迟: {result.avg_time*1000:.2f} ms")
        print(f"    最小延迟: {result.min_time*1000:.2f} ms")
        print(f"    最大延迟: {result.max_time*1000:.2f} ms")
        print(f"    P50: {result.p50*1000:.2f} ms")
        print(f"    P95: {result.p95*1000:.2f} ms")
        print(f"    P99: {result.p99*1000:.2f} ms")
        print(f"    吞吐量: {result.ops_per_second:.2f} ops/s")

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """运行所有基准测试"""
        print(f"\n{'='*60}")
        print(f"ContextEngine 性能基准测试")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        results = []

        # 基准测试 1-4
        results.append(self.benchmark_single_write(iterations=100))
        results.append(self.benchmark_batch_write(batch_size=10, iterations=50))
        results.append(self.benchmark_extraction(iterations=50))
        results.append(self.benchmark_merge(iterations=100))

        # 基准测试 5: 向量化
        results.extend(self.benchmark_embedding(batch_sizes=[1, 10, 50, 100]))

        # 基准测试 6: 向量检索
        results.extend(self.benchmark_vector_search(index_sizes=[100, 500, 1000]))

        # 基准测试 7: 并发写入
        results.append(self.benchmark_concurrent_writes(threads=10, writes_per_thread=50))

        return results

    def print_summary(self, results: List[BenchmarkResult]):
        """打印性能摘要"""
        print(f"\n{'='*60}")
        print(f"性能基准测试摘要")
        print(f"{'='*60}\n")

        # 按类别分组
        write_results = [r for r in results if "write" in r.name]
        search_results = [r for r in results if "search" in r.name]
        other_results = [r for r in results if r not in write_results and r not in search_results]

        print("写入性能:")
        for r in write_results:
            detail = f" ({r.details.get('batch_size', '')} batch)" if 'batch' in r.name else ""
            detail = f" ({r.details.get('threads', '')} threads)" if 'concurrent' in r.name else detail
            print(f"  {r.name}{detail}: {r.ops_per_second:.1f} ops/s, P95={r.p95*1000:.1f}ms")

        print("\n检索性能:")
        for r in search_results:
            size = r.details.get('index_size', '?')
            print(f"  {r.name} (size={size}): {r.avg_time*1000:.1f}ms avg")

        print("\n其他性能:")
        for r in other_results:
            print(f"  {r.name}: {r.avg_time*1000:.1f}ms avg")


def main():
    parser = argparse.ArgumentParser(description="ContextEngine 性能基准测试")
    parser.add_argument("--test", choices=["all", "write", "batch", "extract", "merge", "embed", "search", "concurrent"],
                        default="all", help="要运行的基准测试")
    parser.add_argument("--agfs-url", default="http://localhost:1833", help="AGFS 服务地址")
    parser.add_argument("--iterations", type=int, default=100, help="迭代次数")

    args = parser.parse_args()

    bench = PerformanceBenchmark(agfs_base_url=args.agfs_url)

    if args.test == "all":
        results = bench.run_all_benchmarks()
        bench.print_summary(results)
    elif args.test == "write":
        result = bench.benchmark_single_write(args.iterations)
        bench.print_summary([result])
    elif args.test == "batch":
        result = bench.benchmark_batch_write(10, args.iterations)
        bench.print_summary([result])
    elif args.test == "extract":
        result = bench.benchmark_extraction(args.iterations)
        bench.print_summary([result])
    elif args.test == "merge":
        result = bench.benchmark_merge(args.iterations)
        bench.print_summary([result])
    elif args.test == "embed":
        results = bench.benchmark_embedding([1, 10, 50, 100])
        bench.print_summary(results)
    elif args.test == "search":
        results = bench.benchmark_vector_search([100, 500, 1000])
        bench.print_summary(results)
    elif args.test == "concurrent":
        result = bench.benchmark_concurrent_writes(10, args.iterations // 10)
        bench.print_summary([result])


if __name__ == "__main__":
    main()
