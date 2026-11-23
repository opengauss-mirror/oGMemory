# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Call-local description of the prompt and tools offered to extraction."""
from dataclasses import dataclass


class UnsupportedAssemblyTemplateError(ValueError):
    """A template cannot safely be filtered using trusted metadata."""


@dataclass(frozen=True)
class AssemblyContract:
    categories: frozenset[str]
    tool_names: tuple[str, ...]
    mode: str
    scope: str
    reason: str | None = None
    source_only: bool = False
