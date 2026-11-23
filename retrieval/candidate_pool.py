# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Candidate pool sizing helpers for retrieval stages."""

from __future__ import annotations

from core.models import RetrievalConfig


def candidate_top_k(
    config: RetrievalConfig,
    final_top_k: int | None,
    *,
    fallback_top_k: int | None = None,
) -> int:
    if not getattr(config, "candidate_pool_enabled", False):
        return max(1, int(fallback_top_k or final_top_k or config.default_top_k))

    requested = max(1, int(final_top_k or config.default_top_k))
    multiplier = max(1, int(getattr(config, "candidate_pool_multiplier", 1)))
    minimum = max(1, int(getattr(config, "min_candidate_pool_size", requested)))
    candidate_count = max(requested * multiplier, minimum)
    search_cap = max(requested, int(config.global_search_topk))
    return min(search_cap, candidate_count)
