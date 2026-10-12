"""Archive builder for constructing ContextNode from candidates.

Takes CandidateMemory and WritePlan to produce ContextNode ready for writing.
"""

import re
from datetime import UTC, datetime

from core.enums import ContextType
from core.interfaces import LLM
from core.logging_config import get_logger
from core.models import (
    CandidateMemory,
    ContextNode,
    RequestContext,
    WritePlan,
)

logger = get_logger(__name__)


class ArchiveBuilder:
    """Builds ContextNode from candidate and write plan.

    The builder handles:
    - Constructing proper URI from WritePlan
    - Merging content for merge actions (with LLM semantic merge)
    - Setting appropriate metadata
    - Building relations
    """

    def __init__(self, llm: LLM):
        """Initialize ArchiveBuilder.

        Args:
            llm: LLM instance for semantic merge of skill content.
                 Skill content will be semantically merged (deduplicated, refined).
        """
        self._llm = llm

    def build(
        self,
        candidate: CandidateMemory,
        plan: WritePlan,
        ctx: RequestContext
    ) -> ContextNode:
        """Build ContextNode from candidate and write plan.

        Args:
            candidate: CandidateMemory with extracted content
            plan: WritePlan with action and target URI
            ctx: RequestContext for this operation

        Returns:
            ContextNode ready for ContextFS.write_node()
        """
        # Determine context type from category
        if candidate.category in ("skill", "pattern", "tool"):
            context_type = ContextType.SKILL.value
        else:
            context_type = ContextType.MEMORY.value

        # Calculate level (depth in directory tree)
        # profile/ = level 3, others = level 4
        if candidate.category == "profile":
            level = 3
        else:
            level = 4

        # Determine owner space using RequestContext methods for consistent format
        if candidate.owner_scope == "user":
            owner_space = ctx.user_space_name()
        else:
            owner_space = ctx.agent_space_name()

        # Build content based on action
        if plan.action == "merge" and plan.merged_fields:
            content = self._merge_content(candidate, plan.merged_fields)
        else:
            content = candidate.content

        # Build overview based on action (for skill_merge or overview_append)
        if plan.action == "merge" and plan.merged_fields:
            overview = self._merge_overview(candidate, plan.merged_fields)
        else:
            overview = candidate.overview

        # Build abstract based on action (for skill_merge)
        if plan.action == "merge" and plan.merged_fields.get("skill_merge"):
            abstract = plan.merged_fields.get("new_abstract", candidate.abstract)
        else:
            abstract = candidate.abstract

        # Build metadata
        metadata = {
            "created_at": datetime.now(UTC).isoformat(),
        }

        # Add expected_version for optimistic locking (from merge policies)
        if "expected_version" in plan.merged_fields:
            metadata["expected_version"] = plan.merged_fields["expected_version"]

        # Add any merged metadata
        if "usage_count" in plan.merged_fields:
            metadata["usage_count"] = plan.merged_fields["usage_count"]
        if "usage_count_increment" in plan.merged_fields:
            metadata["usage_count_increment"] = plan.merged_fields["usage_count_increment"]

        # Preserve temporal/actor/location context from extraction
        if candidate.when:
            metadata["when"] = candidate.when
        if candidate.who:
            metadata["who"] = candidate.who
        if candidate.where:
            metadata["where"] = candidate.where

        # Preserve only the opt-in graph descriptor, not arbitrary LLM metadata.
        from index.activation_projection import ACTIVATION_VERSION
        descriptor = candidate.metadata.get("update_activation")
        if (candidate.metadata.get("update_activation_version") == ACTIVATION_VERSION
                and isinstance(descriptor, str) and descriptor.strip()
                and "\n" not in descriptor.strip() and "\r" not in descriptor.strip()):
            metadata["update_activation"] = descriptor.strip()
            metadata["update_activation_version"] = ACTIVATION_VERSION

        # Preserve routing_key for metadata-enriched retrieval
        if candidate.routing_key:
            metadata["routing_key"] = candidate.routing_key

        # Preserve tool usage stats for tool category
        if candidate.tool_stats:
            metadata["tool_stats"] = candidate.tool_stats

        # Preserve provenance tracking IDs
        # On merge: merged_fields contains existing + new (deduped)
        # On create: fall back to candidate's provenance_ids
        prov = plan.merged_fields.get("provenance_ids") or candidate.provenance_ids
        if prov:
            metadata["provenance_ids"] = prov

        # Store relations in metadata for write_node
        if plan.relation_edges:
            metadata["_relations"] = plan.relation_edges

        return ContextNode(
            uri=plan.target_uri,
            context_type=context_type,
            category=candidate.category,
            level=level,
            owner_space=owner_space,
            abstract=abstract,
            overview=overview,
            content=content,
            metadata=metadata,
        )

    def _merge_overview(self, candidate: CandidateMemory, merged_fields: dict) -> str:
        """Merge overview for skill_merge or overview_append operations.

        Args:
            candidate: CandidateMemory with new overview
            merged_fields: Fields to merge from existing node

        Returns:
            Merged overview string

        Supports two merge modes:
        - skill_merge: uses existing_overview + new_overview (semantic)
        - overview_append: uses existing_overview + overview_append (direct)
        """
        existing = merged_fields.get("existing_overview", "")

        # skill_merge mode: semantic merge via new_overview
        if merged_fields.get("skill_merge"):
            new = merged_fields.get("new_overview", candidate.overview)
            if not existing:
                return new
            return f"{existing}\n\n{new}"

        # overview_append mode: direct append
        if "overview_append" in merged_fields:
            new = merged_fields.get("overview_append", candidate.overview)
            if not existing:
                return new
            return f"{existing}\n\n{new}"

        # No merge specified, use candidate overview
        return candidate.overview

    def _merge_content(self, candidate: CandidateMemory, merged_fields: dict) -> str:
        """Merge candidate content with existing data.

        Args:
            candidate: CandidateMemory with new content
            merged_fields: Fields to merge from existing node

        Returns:
            Merged content string

        Supports three merge modes:
        - skill_merge: LLM semantic merge of existing_content + new_content
        - content_append: Direct append of existing_content + content_append
        - replace: Uses candidate.content directly (for profile, etc.)
        """
        # For skill_merge operations (structured skill accumulation)
        if merged_fields.get("skill_merge"):
            existing = merged_fields.get("existing_content", "")
            new_content = merged_fields.get("new_content", candidate.content)
            if not existing:
                return new_content
            # Use LLM semantic merge
            return self._llm_merge_content(
                existing=existing,
                new=new_content,
            )

        # For append operations (AggregateTopicPolicy — preference, entity, pattern)
        if "content_append" in merged_fields:
            existing = merged_fields.get("existing_content", "")
            new = merged_fields.get("content_append", candidate.content)
            if not existing:
                return new
            # Use LLM merge to deduplicate and reconcile conflicts
            return self._llm_merge_content(existing=existing, new=new)

        # For replace operations (profile, etc.)
        if "abstract" in merged_fields:
            # New content replaces old
            return candidate.content

        # Default: use candidate content
        return candidate.content

    def _llm_merge_content(self, existing: str, new: str) -> str:
        """Use LLM to semantically merge content.

        Used for both skill accumulation and AggregateTopicPolicy merge
        (preference, entity, pattern) to avoid infinite append buildup.

        Args:
            existing: Existing content
            new: New content to merge in

        Returns:
            Merged content with duplicates removed.
        """
        schema = {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        }
        first_error: Exception | None = None
        try:
            result = self._llm.complete_json(
                prompt=self._merge_content_prompt(existing, new, strict=False),
                schema=schema,
            )
            content = self._merged_content_from_result(result)
            if content:
                return content
            raise ValueError("LLM merge returned empty content")
        except Exception as exc:
            first_error = exc
            logger.warning("LLM content merge failed; retrying with strict JSON prompt: %s", exc)

        try:
            result = self._llm.complete_json(
                prompt=self._merge_content_prompt(existing, new, strict=True),
                schema=schema,
            )
            content = self._merged_content_from_result(result)
            if content:
                return content
            raise ValueError("LLM merge retry returned empty content")
        except Exception as exc:
            logger.warning(
                "LLM content merge retry failed; falling back to deterministic append: "
                "first_error=%s retry_error=%s",
                first_error,
                exc,
            )
            return self._deterministic_append_content(existing, new)

    @staticmethod
    def _merge_content_prompt(existing: str, new: str, strict: bool) -> str:
        base = (
            "Merge these two memory descriptions into one. "
            "Remove duplicates, keep all unique details. "
            "If they conflict, prefer the newer content. "
            "Return the result as structured Markdown.\n\n"
            f"Existing:\n{existing}\n\nNew:\n{new}"
        )
        if not strict:
            return base
        return (
            "Return ONLY valid JSON matching this exact schema: "
            '{"content": "merged structured Markdown string"}.\n'
            "Do not include markdown fences, comments, explanations, or extra keys.\n\n"
            f"{base}"
        )

    @staticmethod
    def _merged_content_from_result(result: dict) -> str:
        content = result.get("content") if isinstance(result, dict) else None
        return content.strip() if isinstance(content, str) else ""

    @staticmethod
    def _deterministic_append_content(existing: str, new: str) -> str:
        paragraphs: list[str] = []
        seen: set[str] = set()
        for text in (existing, new):
            for paragraph in re.split(r"\n\s*\n", text or ""):
                cleaned = paragraph.strip()
                if not cleaned:
                    continue
                key = re.sub(r"\s+", " ", cleaned).casefold()
                if key in seen:
                    continue
                seen.add(key)
                paragraphs.append(cleaned)
        return "\n\n".join(paragraphs)
