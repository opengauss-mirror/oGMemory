# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Compression quality metrics for session archives."""

from __future__ import annotations

import re

_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}\b")
_ENTITY_STOPWORDS = {
    "Add",
    "Build",
    "Can",
    "Could",
    "Create",
    "Do",
    "Fix",
    "I",
    "Implement",
    "Make",
    "Need",
    "Please",
    "Review",
    "Set",
    "Should",
    "That",
    "The",
    "Then",
    "There",
    "These",
    "This",
    "Those",
    "Use",
    "Want",
    "We",
    "Will",
    "Would",
    "You",
    "No",
    "Ok",
    "Okay",
    "Yes",
}


class CompressionQualityEvaluator:
    def evaluate(
        self,
        messages: list[dict],
        overview: str,
        abstract: str,
        include_details: bool = False,
    ) -> dict:
        source_text = "\n".join(str(msg.get("content", "")) for msg in messages)
        compressed_text = "\n".join([overview or "", abstract or ""])
        source_chars = len(source_text)
        compressed_chars = len(compressed_text)
        source_entities = self._entities(source_text)
        retained = {
            entity for entity in source_entities
            if entity.lower() in compressed_text.lower()
        }
        if not source_entities:
            retention = 1.0
        else:
            retention = len(retained) / len(source_entities)
        if source_chars <= 0:
            compression_ratio = 1.0
        else:
            compression_ratio = min(1.0, compressed_chars / source_chars)
        metrics = {
            "information_retention_ratio": round(retention, 4),
            "entity_retention_ratio": round(retention, 4),
            "compression_ratio": round(compression_ratio, 4),
        }
        if include_details:
            metrics["details"] = {
                "source_chars": source_chars,
                "compressed_chars": compressed_chars,
                "source_entity_count": len(source_entities),
                "retained_entity_count": len(retained),
                "missing_entities": sorted(source_entities - retained),
            }
        return metrics

    def _entities(self, text: str) -> set[str]:
        return {
            match.group(0)
            for match in _ENTITY_RE.finditer(text or "")
            if match.group(0) not in _ENTITY_STOPWORDS
        }
