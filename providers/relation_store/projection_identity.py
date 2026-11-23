# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Stable identity and representation metadata for relation projections."""

from __future__ import annotations

import hashlib

from core.uri_parser import strip_l2_suffix

REPRESENTATION_VERSION = "relation-target-v2"
FACTORIZED_REPRESENTATION_VERSION = "relation-target-factorized-full-v1"


def canonical_edge_id(
    account_id: str,
    source_uri: str,
    target_uri: str,
    relation: str,
) -> str:
    raw = "\0".join(
        (
            account_id,
            strip_l2_suffix(source_uri),
            strip_l2_suffix(target_uri),
            relation,
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()
