# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Retrieval pipeline — 4-stage search_memory chain.

Stages:
  1. QueryPlanner    — classify query into TypedQuery[]
  2. SeedRetriever   — global L0/L1/L2 mixed vector search
  3. HierarchicalSearcher — auto-expand L0/L1 to L2 leaves
  4. ResultRanker — filter to L2, sort, top-k → RetrievedBlock[]
"""
