# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Dependency-free default lexical analyzer."""

from __future__ import annotations

import re
import unicodedata

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_TERM_RE = re.compile(r"[^\W_]+(?:[-._:/][^\W_]+)*", re.UNICODE)


class DefaultLexicalAnalyzer:
    """Stable Unicode analyzer requiring only the Python standard library.

    It deliberately performs no language-specific stemming or stop-word
    removal. Deployments that need language-specific behavior can inject a
    different analyzer through the provider factory.
    """

    identity = "ogmemory-unicode-v1"

    def analyze_document(self, text: str) -> list[str]:
        return self._analyze(text)

    def analyze_query(self, text: str) -> list[str]:
        return self._analyze(text)

    @staticmethod
    def _analyze(text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text or "").casefold()
        tokens: list[str] = []
        without_cjk = _CJK_RE.sub(" ", normalized)
        tokens.extend(match.group(0) for match in _TERM_RE.finditer(without_cjk))
        for span in _CJK_RE.findall(normalized):
            if len(span) == 1:
                tokens.append(span)
            else:
                tokens.extend(span[index : index + 2] for index in range(len(span) - 1))
        return tokens
