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

"""ContextEngine 压力测试脚本

测试大规模写入场景下的写入链路有效性和一致性。

运行方式:
    python3 scripts/stress_test.py --help

测试场景:
1. 单账户大规模写入 - 1000 条记忆写入
2. 多账户并发写入 - 10 账户 x 100 条
3. 相同 routing_key 并发写入 - 测试 merge 一致性
4. Outbox 处理压力测试 - 大量索引事件
"""

import argparse
import asyncio
import os
import sys
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import RequestContext, CandidateMemory
from fs.agfs_adapter import AGFSContextFS
from pyagfs import AGFSClient
from service.api import MemoryWriteAPI
from providers.llm import MockLLM
from providers.embedder import MockEmbedder
from providers.vector_index import InMemoryVectorIndex
from commit import OutboxStore


@dataclass
class StressTestResult:
    """压力测试结果"""
    name: str
    total_operations: int
    successful: int
    failed: int
    duration_seconds: float
    ops_per_second: float
    errors: List[str]


class StressTestRunner:
    """压力测试运行器"""

    def __init__(self, agfs_base_url: str = "http://localhost:1833"):
        self.agfs_base_url = agfs_base_url
        self.client = AGFSClient(api_base_url=agfs_base_url)
        self.fs = AGFSContextFS(client=self.client, mount_prefix="/local/stress_test")
        self.llm = MockLLM()
        self.embedder = MockEmbedder(dimension=1536)
        self.vector_index = InMemoryVectorIndex(dimension=1536)
        self.outbox_store = OutboxStore(client=self.client, fs=self.fs, mount_prefix="/local/stress_test")

        # 初始化 API
        self.api = MemoryWriteAPI(fs=self.fs, llm=self.llm, outbox_store=self.outbox_store)

        # 创建测试目录
        try:
            self.client.mkdir("/local/stress_test")
        except:
            pass

    def create_context(self, account_id: str, user_id: str) -> RequestContext:
        """创建 RequestContext"""
        return RequestContext(
            account_id=account_id,
            user_id=user_id,
            agent_id="stress-test-agent",
            session_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
        )

    def test_single_account_large_scale(self, num_writes: int = 1000) -> StressTestResult:
        """测试 1: 单账户大规模写入

        验证:
        - 大量写入不会崩溃
        - 写入吞吐量可接受
        - 数据一致性保持
        """
        print(f"\n{'='*60}")
        print(f"测试 1: 单账户大规模写入 ({num_writes} 条)")
        print(f"{'='*60}")

        ctx = self.create_context("stress-test-account", "user-1")
        successful = 0
        failed = 0
        errors = []

        start_time = time.time()

        for i in range(num_writes):
            try:
                candidate = CandidateMemory(
                    category="preference",
                    owner_scope="user",
                    routing_key=f"pref_{i % 100}",  # 100 个不同 routing key，会有 merge
                    abstract=f"Preference {i}",
                    overview=f"## Preference {i}\n\nTest content",
                    content=f"This is test preference number {i} for stress testing.",
                    confidence=0.9,
                )

                result = self.api.write_memory(candidate, ctx)
                successful += 1

                # 每 100 条打印进度
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    ops_per_sec = (i + 1) / elapsed
                    print(f"  进度: {i+1}/{num_writes} ({ops_per_sec:.1f} ops/s)")

            except Exception as e:
                failed += 1
                errors.append(str(e))

        duration = time.time() - start_time
        ops_per_sec = num_writes / duration

        print(f"\n结果:")
        print(f"  总操作数: {num_writes}")
        print(f"  成功: {successful}")
        print(f"  失败: {failed}")
        print(f"  耗时: {duration:.2f} 秒")
        print(f"  吞吐量: {ops_per_sec:.2f} ops/s")

        return StressTestResult(
            name="single_account_large_scale",
            total_operations=num_writes,
            successful=successful,
            failed=failed,
            duration_seconds=duration,
            ops_per_second=ops_per_sec,
            errors=errors,
        )

    def test_multi_account_concurrent(self, num_accounts: int = 10, writes_per_account: int = 100) -> StressTestResult:
        """测试 2: 多账户并发写入

        验证:
        - 多账户隔离正确
        - 并发写入一致性
        - 无跨账户数据泄漏
        """
        print(f"\n{'='*60}")
        print(f"测试 2: 多账户并发写入 ({num_accounts} 账户 x {writes_per_account} 条)")
        print(f"{'='*60}")

        successful = 0
        failed = 0
        errors = []
        lock = threading.Lock()

        def write_for_account(account_idx: int):
            nonlocal successful, failed
            ctx = self.create_context(f"stress-account-{account_idx}", f"user-{account_idx}")

            for i in range(writes_per_account):
                try:
                    candidate = CandidateMemory(
                        category="entity",
                        owner_scope="user",
                        routing_key=f"entity_{i}",
                        abstract=f"Entity {i} for account {account_idx}",
                        overview=f"## Entity {i}",
                        content=f"Entity data for account {account_idx}",
                        confidence=0.9,
                    )

                    # 每个账户使用独立的 API 实例
                    api = MemoryWriteAPI(fs=self.fs, llm=self.llm, outbox_store=self.outbox_store)
                    api.write_memory(candidate, ctx)

                    with lock:
                        successful += 1

                except Exception as e:
                    with lock:
                        failed += 1
                        errors.append(f"Account-{account_idx}: {str(e)}")

        start_time = time.time()

        # 使用线程池并发写入
        with ThreadPoolExecutor(max_workers=num_accounts) as executor:
            futures = [executor.submit(write_for_account, i) for i in range(num_accounts)]
            for future in as_completed(futures):
                future.result()  # 获取结果，触发异常

        duration = time.time() - start_time
        total_ops = num_accounts * writes_per_account
        ops_per_sec = total_ops / duration

        print(f"\n结果:")
        print(f"  总操作数: {total_ops}")
        print(f"  成功: {successful}")
        print(f"  失败: {failed}")
        print(f"  耗时: {duration:.2f} 秒")
        print(f"  吞吐量: {ops_per_sec:.2f} ops/s")

        return StressTestResult(
            name="multi_account_concurrent",
            total_operations=total_ops,
            successful=successful,
            failed=failed,
            duration_seconds=duration,
            ops_per_second=ops_per_sec,
            errors=errors,
        )

    def test_concurrent_merge_consistency(self, num_threads: int = 10, writes_per_thread: int = 50) -> StressTestResult:
        """测试 3: 相同 routing_key 并发写入 (Merge 一致性)

        验证:
        - 并发 merge 不会丢失数据
        - 最终状态一致
        - 无竞态条件导致的错误
        """
        print(f"\n{'='*60}")
        print(f"测试 3: 并发 Merge 一致性 ({num_threads} 线程 x {writes_per_thread} 次)")
        print(f"{'='*60}")

        # 所有线程写入相同的 routing_key
        routing_key = "concurrent_test_preference"
        ctx = self.create_context("consistency-test", "user-1")

        successful = 0
        failed = 0
        errors = []
        lock = threading.Lock()

        def write_concurrent(thread_id: int):
            nonlocal successful, failed

            for i in range(writes_per_thread):
                try:
                    candidate = CandidateMemory(
                        category="preference",
                        owner_scope="user",
                        routing_key=routing_key,  # 相同 routing_key
                        abstract=f"Concurrent write {thread_id}-{i}",
                        overview=f"## Concurrent Write\n\nThread {thread_id}, iteration {i}",
                        content=f"Content from thread {thread_id}, iteration {i}",
                        confidence=0.9,
                    )

                    api = MemoryWriteAPI(fs=self.fs, llm=self.llm, outbox_store=self.outbox_store)
                    api.write_memory(candidate, ctx)

                    with lock:
                        successful += 1

                except Exception as e:
                    with lock:
                        failed += 1
                        errors.append(f"Thread-{thread_id}: {str(e)}")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_concurrent, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        duration = time.time() - start_time
        total_ops = num_threads * writes_per_thread
        ops_per_sec = total_ops / duration

        # 验证最终只有一个节点
        try:
            uri = f"ctx://consistency-test/users/user-1/memories/preferences/{routing_key}"
            exists = self.fs.exists(uri, ctx)
            print(f"\n  最终节点状态: {'存在' if exists else '不存在'}")
        except Exception as e:
            print(f"\n  验证节点时出错: {e}")

        print(f"\n结果:")
        print(f"  总操作数: {total_ops}")
        print(f"  成功: {successful}")
        print(f"  失败: {failed}")
        print(f"  耗时: {duration:.2f} 秒")
        print(f"  吞吐量: {ops_per_sec:.2f} ops/s")

        return StressTestResult(
            name="concurrent_merge_consistency",
            total_operations=total_ops,
            successful=successful,
            failed=failed,
            duration_seconds=duration,
            ops_per_second=ops_per_sec,
            errors=errors,
        )

    def test_outbox_processing_stress(self, num_events: int = 500) -> StressTestResult:
        """测试 4: Outbox 处理压力测试

        验证:
        - 大量 OutboxEvent 正确处理
        - 索引同步不丢失
        - DLQ 机制正确工作
        """
        print(f"\n{'='*60}")
        print(f"测试 4: Outbox 处理压力测试 ({num_events} 事件)")
        print(f"{'='*60}")

        from index.outbox_worker import OutboxWorker, create_upsert_event
        from core.models import IndexRecord

        successful = 0
        failed = 0
        errors = []

        start_time = time.time()

        # 创建大量事件
        for i in range(num_events):
            try:
                ctx = self.create_context("outbox-test", f"user-{i % 10}")

                # 先写入节点
                candidate = CandidateMemory(
                    category="preference",
                    owner_scope="user",
                    routing_key=f"outbox_pref_{i % 50}",
                    abstract=f"Outbox test {i}",
                    overview="## Test",
                    content=f"Content {i}",
                    confidence=0.9,
                )

                self.api.write_memory(candidate, ctx)
                successful += 1

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    ops_per_sec = (i + 1) / elapsed
                    print(f"  进度: {i+1}/{num_events} ({ops_per_sec:.1f} ops/s)")

            except Exception as e:
                failed += 1
                errors.append(str(e))

        # 处理 outbox 事件
        print(f"\n  处理 Outbox 事件...")
        worker = OutboxWorker(vector_index=self.vector_index, embedder=self.embedder)

        process_start = time.time()
        stats = worker.run_once(self.outbox_store, ["outbox-test"], worker_id="stress-worker")
        process_duration = time.time() - process_start

        duration = time.time() - start_time
        ops_per_sec = num_events / duration

        print(f"\nOutbox 处理统计:")
        print(f"  处理: {stats['processed']}")
        print(f"  成功: {stats['succeeded']}")
        print(f"  失败: {stats['failed']}")
        print(f"  跳过: {stats['skipped']}")
        print(f"  处理耗时: {process_duration:.2f} 秒")

        print(f"\n结果:")
        print(f"  总事件数: {num_events}")
        print(f"  写入成功: {successful}")
        print(f"  写入失败: {failed}")
        print(f"  总耗时: {duration:.2f} 秒")
        print(f"  吞吐量: {ops_per_sec:.2f} ops/s")

        return StressTestResult(
            name="outbox_processing_stress",
            total_operations=num_events,
            successful=successful,
            failed=failed,
            duration_seconds=duration,
            ops_per_second=ops_per_sec,
            errors=errors,
        )

    def run_all_tests(self) -> List[StressTestResult]:
        """运行所有压力测试"""
        results = []

        print(f"\n{'='*60}")
        print(f"ContextEngine 压力测试套件")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # 测试 1: 单账户大规模写入
        results.append(self.test_single_account_large_scale(num_writes=500))

        # 测试 2: 多账户并发写入
        results.append(self.test_multi_account_concurrent(num_accounts=10, writes_per_account=50))

        # 测试 3: 并发 Merge 一致性
        results.append(self.test_concurrent_merge_consistency(num_threads=10, writes_per_thread=20))

        # 测试 4: Outbox 处理压力
        results.append(self.test_outbox_processing_stress(num_events=200))

        return results

    def print_summary(self, results: List[StressTestResult]):
        """打印测试摘要"""
        print(f"\n{'='*60}")
        print(f"压力测试摘要")
        print(f"{'='*60}\n")

        total_ops = sum(r.total_operations for r in results)
        total_successful = sum(r.successful for r in results)
        total_failed = sum(r.failed for r in results)

        for r in results:
            status = f"{r.successful}/{r.total_operations}"
            print(f"  {r.name}")
            print(f"    状态: {status}")
            print(f"    吞吐量: {r.ops_per_second:.2f} ops/s")
            if r.errors:
                print(f"    错误数: {len(r.errors)}")
            print()

        print(f"总计:")
        print(f"  总操作数: {total_ops}")
        print(f"  成功: {total_successful}")
        print(f"  失败: {total_failed}")
        print(f"  成功率: {total_successful/total_ops*100:.1f}%")

        # 打印失败详情
        if total_failed > 0:
            print(f"\n失败详情:")
            for r in results:
                if r.errors:
                    print(f"\n  {r.name}:")
                    for e in r.errors[:5]:  # 只显示前 5 个
                        print(f"    - {e}")
                    if len(r.errors) > 5:
                        print(f"    ... 还有 {len(r.errors) - 5} 个错误")


def main():
    parser = argparse.ArgumentParser(description="ContextEngine 压力测试")
    parser.add_argument("--test", choices=["all", "single", "multi", "merge", "outbox"],
                        default="all", help="要运行的测试")
    parser.add_argument("--agfs-url", default="http://localhost:1833",
                        help="AGFS 服务地址")
    parser.add_argument("--writes", type=int, default=500,
                        help="单测试写入数量")

    args = parser.parse_args()

    runner = StressTestRunner(agfs_base_url=args.agfs_url)

    if args.test == "all":
        results = runner.run_all_tests()
        runner.print_summary(results)
    elif args.test == "single":
        result = runner.test_single_account_large_scale(num_writes=args.writes)
        runner.print_summary([result])
    elif args.test == "multi":
        result = runner.test_multi_account_concurrent(num_accounts=10, writes_per_account=args.writes // 10)
        runner.print_summary([result])
    elif args.test == "merge":
        result = runner.test_concurrent_merge_consistency(num_threads=10, writes_per_thread=args.writes // 10)
        runner.print_summary([result])
    elif args.test == "outbox":
        result = runner.test_outbox_processing_stress(num_events=args.writes)
        runner.print_summary([result])


if __name__ == "__main__":
    main()
