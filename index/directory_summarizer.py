# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Directory Summarizer: Aggregate child summaries into directory-level summary.

Used by OutboxWorker when processing UPSERT_DIRECTORY events.
Generates L0 (abstract) and L1 (overview) summaries from children.
"""

import logging
from dataclasses import dataclass

from core.interfaces import ContextFS, LLM
from core.language import detect_language

logger = logging.getLogger(__name__)


SUMMARY_PROMPT = """You are a knowledge management assistant. Aggregate child node summaries into a directory-level summary.

Directory: {directory_uri}

Child summaries:
{children_summaries}

Generate:
1. A concise abstract (100 chars max) summarizing all content themes
2. A structured overview listing key themes and highlights

Return JSON:
{{
    "abstract": "Directory abstract (100 chars max)",
    "overview": "Structured overview (Markdown allowed)"
}}

IMPORTANT: Output in {language} language.
"""


def _detect_language(texts: list[str]) -> str:
    """Detect language from a list of text strings.

    Returns:
        Language code: "zh-CN", "ja", "ko", "ru", or "en" (default)
    """
    return detect_language(" ".join(texts))


@dataclass
class DirectorySummary:
    """Generated directory summary."""

    abstract: str
    overview: str
    child_count: int
    categories: list[str] | None = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = []


@dataclass
class ChildSummary:
    """Child node summary for directory aggregation."""

    uri: str
    name: str
    abstract: str
    category: str
    is_directory: bool
    has_abstract: bool


class DirectorySummarizer:
    """Aggregate child node summaries into directory-level summary.

    Provides LLM-based summary generation and fallback methods.
    Used by DirectoryEventHandler for DAG-style processing.
    """

    def __init__(
        self,
        fs: ContextFS,
        llm: LLM,
        max_children: int = 50,
    ):
        """Initialize DirectorySummarizer.

        Args:
            fs: ContextFS for reading child nodes
            llm: LLM for generating aggregated summary
            max_children: Maximum children to include in summary
        """
        self._fs = fs
        self._llm = llm
        self._max_children = max_children

    def _generate_summary(
        self,
        directory_uri: str,
        child_summaries: list[dict],
    ) -> DirectorySummary:
        """Generate aggregated summary using LLM.

        Args:
            directory_uri: Directory URI
            child_summaries: List of child summaries with uri, abstract, category

        Returns:
            DirectorySummary with abstract and overview
        """
        summaries_text = "\n\n".join([
            f"- [{s['category']}] {s['uri']}:\n  {s['abstract']}"
            for s in child_summaries
        ])

        # Detect language from child abstracts
        language = _detect_language([s.get("abstract", "") for s in child_summaries])

        prompt = SUMMARY_PROMPT.format(
            directory_uri=directory_uri,
            children_summaries=summaries_text,
            language=language,
        )

        schema = {
            "type": "object",
            "properties": {
                "abstract": {"type": "string"},
                "overview": {"type": "string"},
            },
            "required": ["abstract", "overview"],
        }

        try:
            result = self._llm.complete_json(prompt, schema)
            abstract = result.get("abstract", "")
            overview = result.get("overview", "")
        except Exception as e:
            logger.warning("LLM summary generation failed: %s, using fallback", e)
            abstract = self._fallback_abstract(child_summaries)
            overview = self._fallback_overview(child_summaries)

        categories = list(set(s["category"] for s in child_summaries))

        return DirectorySummary(
            abstract=abstract[:100],
            overview=overview,
            child_count=len(child_summaries),
            categories=categories,
        )

    def _fallback_abstract(self, child_summaries: list[dict]) -> str:
        """Generate fallback abstract without LLM.

        Args:
            child_summaries: List of child summaries

        Returns:
            Simple aggregated abstract
        """
        language = _detect_language([s.get("abstract", "") for s in child_summaries])
        categories = set(s["category"] for s in child_summaries)

        if language == "zh-CN":
            category_str = "、".join(categories)
            return f"包含 {len(child_summaries)} 条{category_str}相关记忆"
        else:
            category_str = ", ".join(categories)
            return f"Contains {len(child_summaries)} {category_str} memories"

    def _fallback_overview(self, child_summaries: list[dict]) -> str:
        """Generate fallback overview without LLM.

        Args:
            child_summaries: List of child summaries

        Returns:
            Simple Markdown overview
        """
        language = _detect_language([s.get("abstract", "") for s in child_summaries])
        title = "# 目录内容概览\n" if language == "zh-CN" else "# Directory Overview\n"

        lines = [title]

        by_category: dict[str, list[dict]] = {}
        for s in child_summaries:
            cat = s["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(s)

        for category, items in by_category.items():
            count_label = f"({len(items)}条)" if language == "zh-CN" else f"({len(items)} items)"
            lines.append(f"## {category} {count_label}")
            for item in items:
                lines.append(f"- {item['abstract'][:50]}...")
            lines.append("")

        return "\n".join(lines)


def is_directory_uri(uri: str) -> bool:
    """Check if URI represents a directory (ends with /).

    Args:
        uri: URI to check

    Returns:
        True if URI is a directory
    """
    return uri.endswith("/")
