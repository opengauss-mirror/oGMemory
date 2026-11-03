"""Stage 1 — Rule-based query planner.

Classifies a natural-language query into one or more TypedQuery objects
using keyword matching with a deterministic fallback.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from core.enums import ContextType
from core.errors import ValidationError
from core.models import RequestContext, RetrievalConfig, TypedQuery
from retrieval.intent_classifier import RetrievalIntentClassifier

logger = logging.getLogger(__name__)

_MEMORY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"偏好|背景|身份|记忆|回顾|profile|preference|background|remember",
    ]
]

_SKILL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"怎么做|步骤|命令|流程|how\s*to|workflow|procedure|instruction|tutorial",
    ]
]

_RESOURCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"文档|资料|规范|链接|document|spec|reference|link|paper|article",
    ]
]


def _match_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(p.search(text) for p in patterns)


_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"```[\s\S]*?```"),                          # markdown code blocks (sender metadata)
    re.compile(r"Sender\s*\(untrusted metadata\)\s*:\s*"),  # OpenClaw sender prefix
    re.compile(r"\[[\w\s:\-]+UTC\]\s*"),                    # timestamp like [Fri 2026-03-27 06:16 UTC]
]

_BROWSER_CONTEXT_PREFIX = re.compile(
    r"\A\s*<in-app-browser-context\b"
    r"(?=[^>]*\bsource\s*=\s*['\"]ambient-ui-state['\"])[^>]*>"
    r"[\s\S]*?</in-app-browser-context>\s*"
)
_USER_REQUEST_PREFIX = re.compile(r"\A## My request:[ \t]*(?:\r?\n|$)")


def sanitize_query(raw: str) -> str:
    """Strip upstream metadata noise, keeping only user intent."""
    text = raw
    # Codex Desktop prepends UI state to the prompt. It is not user intent
    # and can route retrieval to RESOURCE or overwhelm the query embedding.
    # Only strip a leading, explicitly marked ambient block; preserve quoted
    # examples and ordinary user text containing the request heading.
    text, removed = _BROWSER_CONTEXT_PREFIX.subn("", text, count=1)
    if removed:
        text = _USER_REQUEST_PREFIX.sub("", text, count=1)
    for pat in _NOISE_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


class QueryPlanner:
    def __init__(self, config: RetrievalConfig | None = None, llm=None) -> None:
        self.config = config or RetrievalConfig()
        self.intent_classifier = RetrievalIntentClassifier()
        self._llm = llm  # None = skip decomposition

    def plan(
        self,
        query: str,
        ctx: RequestContext,
        *,
        session_archive: dict | None = None,
        hints: dict | None = None,
        categories: list[str] | None = None,
        top_k: int | None = None,
        context_type: str | None = None,
        resource_type: str | None = None,
    ) -> list[TypedQuery]:
        query = sanitize_query(query or "")
        if not query:
            raise ValidationError("query", "query must not be empty")

        effective_top_k = top_k or self.config.default_top_k
        if effective_top_k > self.config.max_top_k:
            raise ValidationError(
                "top_k",
                f"top_k={effective_top_k} exceeds maximum {self.config.max_top_k}",
            )

        context_type_override = (context_type or "").strip().upper() or None
        valid_context_types = {member.value for member in ContextType}
        if context_type_override and context_type_override not in valid_context_types:
            raise ValidationError("context_type", "unsupported context type override")
        normalized_resource_type = (resource_type or "").strip() or None
        if normalized_resource_type and context_type_override != ContextType.RESOURCE.value:
            raise ValidationError(
                "resource_type",
                "resource_type requires context_type=RESOURCE",
            )

        if not context_type_override and categories:
            groups = [(ContextType.MEMORY.value, [c for c in categories if c != "skill"])]
            # ArchiveBuilder stores legacy tools and patterns as SKILL nodes.
            groups.append((ContextType.SKILL.value,
                           [c for c in categories if c in {"skill", "tool", "pattern"}]))
            return [self._make_query(query, kind, ctx, categories=selected,
                                     top_k=effective_top_k, resource_type=None)
                    for kind, selected in groups if selected]

        # Try semantic decomposition for complex queries
        sub_queries = self._decompose(query)
        if len(sub_queries) > 1:
            logger.info("[QueryPlanner] decomposed '%s' into %d sub-queries: %s", query, len(sub_queries), sub_queries)
            return [
                self._make_query(
                    sq,
                    context_type_override or ContextType.MEMORY.value,
                    ctx,
                    categories=categories,
                    top_k=effective_top_k,
                    resource_type=normalized_resource_type,
                )
                for sq in sub_queries
            ]

        if context_type_override:
            return [
                self._make_query(
                    query,
                    context_type_override,
                    ctx,
                    categories=categories,
                    top_k=effective_top_k,
                    resource_type=normalized_resource_type,
                )
            ]

        matched_types = self._classify(query)

        if len(matched_types) == 1:
            if (self.config.include_agent_memories and matched_types[0] == ContextType.SKILL.value
                    and ctx.agent_id and not categories):
                return [self._make_query(query, kind, ctx, categories=None,
                                         top_k=effective_top_k, resource_type=None)
                        for kind in (ContextType.SKILL.value, ContextType.MEMORY.value)]
            return [
                self._make_query(
                    query, matched_types[0], ctx,
                    categories=categories,
                    top_k=effective_top_k,
                    resource_type=None,
                )
            ]

        # No specific type matched (or multiple matched) — default to MEMORY
        # instead of triple-query to avoid 3x embedding load and timeouts.
        logger.info("[QueryPlanner] matched %d types, defaulting to MEMORY", len(matched_types))
        return [
            self._make_query(
                query,
                ContextType.MEMORY.value,
                ctx,
                categories=categories,
                top_k=effective_top_k,
                resource_type=None,
            )
        ]

    @staticmethod
    def _classify(text: str) -> list[str]:
        hits: list[str] = []
        if _match_any(text, _MEMORY_PATTERNS):
            hits.append(ContextType.MEMORY.value)
        if _match_any(text, _SKILL_PATTERNS):
            hits.append(ContextType.SKILL.value)
        if _match_any(text, _RESOURCE_PATTERNS):
            hits.append(ContextType.RESOURCE.value)
        return hits

    def _decompose(self, query: str) -> list[str]:
        """Decompose a complex query into 2-4 sub-queries via LLM.

        Returns [query] (single-element list) if decomposition is skipped
        or fails — safe to call unconditionally.
        """
        if self._llm is None or len(query.split()) < 5:
            return [query]

        try:
            result = self._llm.complete_json(
                (
                    "Decompose the following question into 2-4 independent search "
                    "sub-queries, each focusing on one aspect.\n\n"
                    f"Question: {query}"
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "sub_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 4,
                        }
                    },
                    "required": ["sub_queries"],
                },
            )
            subs = result.get("sub_queries", [])
            if isinstance(subs, list) and len(subs) >= 2 and all(isinstance(s, str) and s.strip() for s in subs):
                return subs[:4]
        except Exception:
            logger.debug("[QueryPlanner] decomposition failed, using original query")

        return [query]

    def _make_query(
        self,
        text: str,
        context_type: str,
        ctx: RequestContext,
        *,
        categories: list[str] | None,
        top_k: int,
        resource_type: str | None,
    ) -> TypedQuery:
        visible_spaces = list(ctx.visible_owner_spaces) if getattr(ctx, "visible_owner_spaces", ()) else []
        owner_space: str | list[str] | None
        # Determine owner_space based on context_type:
        # - SKILL: agent space (skills/patterns/cases are agent-scoped), but
        #   if the visible scope list is provided and excludes agent space,
        #   fall back to an impossible filter to return no results.
        # - code RESOURCE: account-wide by contract; no user/agent owner filter.
        # - other RESOURCE queries: preserve the existing visible owner scope.
        # - MEMORY: use visible scope list when provided; otherwise preserve
        #   historical behavior.
        if context_type == ContextType.SKILL.value:
            visible_agent_spaces = [space for space in visible_spaces if str(space).startswith("agent:")]
            agent_space = ctx.agent_space_name() if ctx.agent_id else ""
            if agent_space and (not visible_spaces or agent_space in visible_spaces):
                owner_space = agent_space
            elif visible_agent_spaces:
                owner_space = visible_agent_spaces
            else:
                owner_space = "__inaccessible__"
        elif (
            context_type == ContextType.RESOURCE.value
            and resource_type == "code"
        ):
            owner_space = None
        elif visible_spaces:
            owner_space = visible_spaces
        else:
            owner_space = None

        # Classify retrieval intent for Layer 3
        intent = self.intent_classifier.classify(text)

        return TypedQuery(
            text=text,
            context_type=context_type,
            categories=categories or [],
            top_k=top_k,
            account_id=ctx.account_id,
            owner_space=owner_space,
            intent=intent.value,
            resource_type=resource_type,
        )
