# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Retrieval-intent classifier for Layer 3.

Classifies search queries into intent categories to enable intent-aware retrieval
instead of raw semantic similarity search.
"""

from __future__ import annotations

import logging
import re
from typing import ClassVar

from core.enums import RetrievalIntent

logger = logging.getLogger(__name__)


class RetrievalIntentClassifier:
    """Rule-based retrieval intent classifier.

    Classifies queries into intent categories to enable intent-aware retrieval
    instead of raw semantic similarity search.

    The classifier uses regex pattern matching with deterministic fallback,
    ensuring fast (<1ms per query) and predictable behavior.
    """

    # Compile patterns once at class level for performance
    INTENT_PATTERNS: ClassVar[dict[RetrievalIntent, list[re.Pattern[str]]]] = {
        RetrievalIntent.HISTORICAL_DECISIONS: [
            re.compile(r"(?:decided|decision|chose|chosen|agreed|concluded)", re.IGNORECASE),
            re.compile(r"(?:what did|why did|how did).*(?:decide|choose|pick|conclude)", re.IGNORECASE),
            re.compile(r"(?:outcome|result|resolution)", re.IGNORECASE),
        ],
        RetrievalIntent.OPEN_ITEMS: [
            re.compile(r"(?:pending|todo|still need|remaining|left to|not yet|unfinished)", re.IGNORECASE),
            re.compile(r"(?:open loop|follow.?up|blocker|waiting for)", re.IGNORECASE),
            re.compile(r"(?:what's left|what remains|outstanding)", re.IGNORECASE),
        ],
        RetrievalIntent.REUSABLE_SKILLS: [
            re.compile(r"(?:how to|how do|steps? to|procedure|process|method)", re.IGNORECASE),
            re.compile(r"(?:best practice|pattern|approach|workflow)", re.IGNORECASE),
            re.compile(r"(?:guide|tutorial|instructions?)", re.IGNORECASE),
        ],
        RetrievalIntent.ENTITY_RELATIONS: [
            re.compile(r"(?:who is|who are|what is|relationship between)", re.IGNORECASE),
            re.compile(r"(?:works? at|belongs? to|connected to|related to)", re.IGNORECASE),
            re.compile(r"(?:team|group|department|org)", re.IGNORECASE),
        ],
        RetrievalIntent.BACKGROUND_SUPPLEMENT: [
            # Default/fallback — profile, preferences, context
            # No explicit patterns, matched when no other intent matches
        ],
    }

    # Intent priority for deterministic tiebreaking
    # Higher priority intents are checked first
    INTENT_PRIORITY: ClassVar[list[RetrievalIntent]] = [
        RetrievalIntent.HISTORICAL_DECISIONS,
        RetrievalIntent.OPEN_ITEMS,
        RetrievalIntent.REUSABLE_SKILLS,
        RetrievalIntent.ENTITY_RELATIONS,
        RetrievalIntent.BACKGROUND_SUPPLEMENT,  # Always matches (fallback)
    ]

    def classify(self, query: str) -> RetrievalIntent:
        """Classify a query into a retrieval intent.

        Args:
            query: The search query text to classify.

        Returns:
            RetrievalIntent: The classified intent. Falls back to
                BACKGROUND_SUPPLEMENT if no specific patterns match.
        """
        if not query or not query.strip():
            return RetrievalIntent.BACKGROUND_SUPPLEMENT

        # Check intents in priority order
        for intent in self.INTENT_PRIORITY:
            if intent == RetrievalIntent.BACKGROUND_SUPPLEMENT:
                # Fallback intent, always matches if we reach it
                return intent

            patterns = self.INTENT_PATTERNS.get(intent, [])
            if self._match_any(query, patterns):
                logger.debug(
                    "[IntentClassifier] query='%s' -> intent=%s",
                    query[:50] + "..." if len(query) > 50 else query,
                    intent.value,
                )
                return intent

        # Should never reach here due to BACKGROUND_SUPPLEMENT fallback
        return RetrievalIntent.BACKGROUND_SUPPLEMENT

    @staticmethod
    def _match_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
        """Check if text matches any of the given patterns.

        Args:
            text: Text to search within.
            patterns: Compiled regex patterns to match against.

        Returns:
            True if any pattern matches, False otherwise.
        """
        return any(p.search(text) for p in patterns)

    def get_intent_description(self, intent: RetrievalIntent) -> str:
        """Get human-readable description of an intent.

        Args:
            intent: The intent to describe.

        Returns:
            Description string for the intent.
        """
        descriptions = {
            RetrievalIntent.BACKGROUND_SUPPLEMENT: (
                "Background context: profile, preferences, and general context"
            ),
            RetrievalIntent.HISTORICAL_DECISIONS: (
                "Historical decisions: past choices, outcomes, and resolutions"
            ),
            RetrievalIntent.OPEN_ITEMS: (
                "Open items: pending tasks, open loops, and follow-ups"
            ),
            RetrievalIntent.REUSABLE_SKILLS: (
                "Reusable skills: procedures, patterns, and how-to guides"
            ),
            RetrievalIntent.ENTITY_RELATIONS: (
                "Entity relations: people, organizations, and relationships"
            ),
        }
        return descriptions.get(intent, "Unknown intent")
