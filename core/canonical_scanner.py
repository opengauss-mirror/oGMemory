# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Public contract for bounded scans of canonical memory nodes."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from core.models import ContextNode


@runtime_checkable
class CanonicalNodeScanner(Protocol):
    """Read active canonical nodes without exposing storage internals."""

    def projection_is_quiescent(self, *, account_id: str) -> bool: ...

    def scan_active_nodes(
        self,
        *,
        account_id: str,
        owner_spaces: Sequence[str],
        page_size: int,
    ) -> Iterator[ContextNode]: ...
