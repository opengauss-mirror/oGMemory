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

"""Run an end-to-end long-context mock test for ContextEngine.

This script intentionally avoids AGFS/network dependencies by using an in-memory
ContextFS while still exercising the full write + index + retrieval pipeline.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from core.models import ContextNode, RequestContext
from service.api import MemoryWriteAPI, ReadAPI
from providers.llm import MockLLM
from providers.embedder import MockEmbedder
from providers.vector_index import InMemoryVectorIndex
from retrieval.pipeline import RetrievalPipeline, RetrievalConfig
from retrieval.query_planner import QueryPlanner
from retrieval.seed_retriever import SeedRetriever
from retrieval.hierarchical_searcher import HierarchicalSearcher
from retrieval.result_ranker import ResultRanker
from retrieval.context_reader import ContextReader
from index.index_record_builder import build_index_records


@dataclass
class InMemoryContextFS:
    """Simple in-memory ContextFS for local smoke tests."""

    _store: dict[str, ContextNode]

    def __init__(self):
        self._store = {}

    def write_node(self, node: ContextNode, ctx: RequestContext) -> None:
        self._store[node.uri] = node

    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
        return self._store[uri]

    def delete_node(self, uri: str, ctx: RequestContext) -> None:
        self._store.pop(uri, None)

    def move_node(self, from_uri: str, to_uri: str, ctx: RequestContext) -> None:
        self._store[to_uri] = self._store.pop(from_uri)

    def list_children(self, uri: str, ctx: RequestContext) -> list[str]:
        prefix = uri if uri.endswith("/") else f"{uri}/"
        return sorted([u for u in self._store.keys() if u.startswith(prefix)])

    def exists(self, uri: str, ctx: RequestContext) -> bool:
        return uri in self._store


def build_mock_tool_calls() -> list[dict]:
    """Return deterministic mock extraction results for all 7 memory types."""
    return [
        {
            "tool": "extract_profile",
            "input": {
                "abstract": "用户是资深后端工程师，负责高并发系统。",
                "overview": "长期维护 Python + Go 服务，关注可观测性与稳定性。",
                "content": "用户在多轮对话中持续描述了分布式系统、日志排障与性能优化经验。",
                "confidence": 0.95,
            },
        },
        {
            "tool": "extract_preference",
            "input": {
                "routing_key": "engineering_style",
                "abstract": "偏好先写可观测性，再做性能优化。",
                "overview": "偏好结构化日志、统一 trace_id、可重放压测。",
                "content": "用户强调 debug 时必须先具备指标和日志，再进行代码层面的热点优化。",
                "confidence": 0.91,
            },
        },
        {
            "tool": "extract_entity",
            "input": {
                "routing_key": "order_service",
                "abstract": "OrderService 是核心订单聚合服务。",
                "overview": "处理下单、库存预占、支付状态回写。",
                "content": "对话多次提及 OrderService 的慢查询、缓存击穿和幂等处理。",
                "confidence": 0.89,
            },
        },
        {
            "tool": "extract_event",
            "input": {
                "routing_key": "incident_review",
                "abstract": "线上故障复盘：定位到连接池配置错误。",
                "overview": "故障窗口约 12 分钟，根因是连接泄漏与超时重试叠加。",
                "content": "团队通过日志采样和指标对齐确认连接池参数偏小并触发级联超时。",
                "confidence": 0.9,
            },
        },
        {
            "tool": "extract_case",
            "input": {
                "routing_key": "fix_n_plus_1",
                "abstract": "案例：修复 N+1 查询导致的 RT 抖动。",
                "overview": "通过预加载与批量查询将 P99 从 3.2s 降到 320ms。",
                "content": "最终方案是 ORM eager load + SQL 索引补齐 + 缓存过期抖动。",
                "confidence": 0.93,
            },
        },
        {
            "tool": "extract_pattern",
            "input": {
                "routing_key": "postmortem_template",
                "abstract": "习惯使用标准 postmortem 模板复盘。",
                "overview": "固定包含时间线、影响面、检测、遏制、根因、行动项。",
                "content": "该模式有助于沉淀可检索经验并提高跨团队协作效率。",
                "confidence": 0.82,
            },
        },
        {
            "tool": "extract_skill",
            "input": {
                "abstract": "掌握慢查询诊断与优化技能。",
                "overview": "会使用 explain、火焰图、索引命中率分析。",
                "content": "形成了从症状采集、假设验证到回归压测的完整方法论。",
                "confidence": 0.88,
            },
        },
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    fixture_path = repo_root / "tests" / "fixtures" / "long_conversation_1000.json"
    messages = json.loads(fixture_path.read_text(encoding="utf-8"))

    fs = InMemoryContextFS()
    llm = MockLLM()
    llm._mock_tool_calls = build_mock_tool_calls()

    ctx = RequestContext(
        account_id="demo-account",
        user_id="demo-user",
        agent_id="demo-agent",
        session_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
    )

    write_api = MemoryWriteAPI(fs=fs, llm=llm)
    write_result = write_api.commit_session(messages, ctx)

    embedder = MockEmbedder(dimension=1536)
    vector_index = InMemoryVectorIndex(dimension=embedder._dimension)

    all_records = []
    for node in fs._store.values():
        all_records.extend(build_index_records(node))
    vector_index.upsert(all_records)

    cfg = RetrievalConfig()
    pipeline = RetrievalPipeline(
        planner=QueryPlanner(cfg),
        seed_retriever=SeedRetriever(vector_index, embedder, cfg),
        hierarchical_searcher=HierarchicalSearcher(vector_index, cfg),
        assembly=ResultRanker(cfg),
        config=cfg,
    )

    read_api = ReadAPI(
        pipeline=pipeline,
        read_service=ContextReader(fs=fs),
        config=cfg,
    )

    search_result = read_api.search_memory(
        query="这位工程师排查线上故障时有什么固定方法？",
        ctx=ctx,
        top_k=5,
        include_debug=True,
    )

    summary = {
        "fixture_messages": len(messages),
        "candidates_extracted": write_result["candidates_extracted"],
        "writes_completed": write_result["writes_completed"],
        "stored_nodes": len(fs._store),
        "index_records": len(all_records),
        "retrieval_hits": len(search_result.hits),
        "top_hit": search_result.hits[0].uri if search_result.hits else None,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
