# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Policy router for selecting the appropriate merge policy.

Routes CandidateMemory to the correct MergePolicy based on schema properties
(operation_mode, is_single_file, owner_scope) instead of hardcoded category mappings.

This enables adding new memory types via YAML schema definitions without code changes.
"""

from logging import getLogger
from typing import Optional

from core.models import RequestContext, CandidateMemory, WritePlan
from core.interfaces import ContextFS, MergePolicy
from core.uri_resolver import URIResolver
from extraction.schemas.registry import SchemaRegistry
from commit.merge_policies import (
    ProfilePolicy,
    AggregateTopicPolicy,
    AppendOnlyPolicy,
    SkillToolPolicy,
)

logger = getLogger(__name__)


class PolicyRouter:
    """Routes candidates to the appropriate merge policy based on schema properties.

    Policy selection logic (driven by schema attributes):
    - upsert + single_file → ProfilePolicy (fixed URI, always merge)
    - upsert + multi_file → AggregateTopicPolicy (merge by slug)
    - add_only → AppendOnlyPolicy (always create, unique ID)
    - agent scope + /skills/ path → SkillToolPolicy (skills/tools)

    Custom policies can be registered via register_policy() for special cases.
    """

    def __init__(
        self,
        fs: ContextFS,
        registry: SchemaRegistry | None = None,
        uri_resolver: URIResolver | None = None,
    ):
        """Initialize PolicyRouter with schema-driven policy selection.

        Args:
            fs: ContextFS instance for policy operations
            registry: SchemaRegistry for accessing memory type schemas.
                      Defaults to the YAML-backed registry.
            uri_resolver: Optional URIResolver for policy use (lazy created if needed)
        """
        self._fs = fs
        self._registry = registry or SchemaRegistry()
        self._uri_resolver = uri_resolver or URIResolver(self._registry)

        # Lazy-initialized policy instances
        self._profile_policy: Optional[ProfilePolicy] = None
        self._aggregate_topic_policy: Optional[AggregateTopicPolicy] = None
        self._append_only_policy: Optional[AppendOnlyPolicy] = None
        self._skill_tool_policy: Optional[SkillToolPolicy] = None

        # Custom policy overrides (category → MergePolicy)
        self._custom_policies: dict[str, MergePolicy] = {}

    def route(self, candidate: CandidateMemory) -> Optional[MergePolicy]:
        """Get the appropriate policy for a candidate based on schema properties.

        Policy selection logic:
        1. Check custom policy overrides first
        2. Look up schema for candidate.category
        3. Select policy based on operation_mode + schema properties:
           - add_only → AppendOnlyPolicy
           - upsert + single_file → ProfilePolicy
           - upsert + multi_file → AggregateTopicPolicy
           - agent scope + /skills/ → SkillToolPolicy

        Args:
            candidate: CandidateMemory to route

        Returns:
            MergePolicy for this candidate, or None if schema not found
        """
        # Check custom policy overrides first
        if candidate.category in self._custom_policies:
            return self._custom_policies[candidate.category]

        # Look up schema
        schema = self._registry.get_compatible(candidate.category)
        if schema is None:
            logger.warning(f"No compatible schema found for category: {candidate.category}")
            return None

        # Select policy based on schema properties
        return self._select_policy_from_schema(schema, candidate)

    def _select_policy_from_schema(
        self, schema, candidate: CandidateMemory
    ) -> Optional[MergePolicy]:
        """Select policy based on schema operation_mode and properties.

        Args:
            schema: MemoryTypeSchema for this memory type
            candidate: CandidateMemory for additional context

        Returns:
            Appropriate MergePolicy instance (lazy initialized)
        """
        # Check for skill/tool special case first
        if self._is_skill_tool_schema(schema):
            return self._get_skill_tool_policy()

        # Route based on operation_mode
        if schema.operation_mode == "add_only":
            return self._get_append_only_policy()

        if schema.operation_mode == "upsert":
            # Single file → ProfilePolicy, Multi file → AggregateTopicPolicy
            if schema.is_single_file:
                return self._get_profile_policy()
            else:
                return self._get_aggregate_topic_policy()

        logger.warning(
            f"Unknown operation_mode '{schema.operation_mode}' for {schema.memory_type}"
        )
        return None

    def _is_skill_tool_schema(self, schema) -> bool:
        """Check if schema represents a skill or tool type.

        Args:
            schema: MemoryTypeSchema to check

        Returns:
            True if this is a skill/tool schema
        """
        return (
            schema.owner_scope == "agent"
            and schema.memory_type in ("skill", "tool")
        )

    # Lazy policy initialization methods
    def _get_profile_policy(self) -> ProfilePolicy:
        """Get or create ProfilePolicy instance."""
        if self._profile_policy is None:
            self._profile_policy = ProfilePolicy(self._fs, self._uri_resolver)
        return self._profile_policy

    def _get_aggregate_topic_policy(self) -> AggregateTopicPolicy:
        """Get or create AggregateTopicPolicy instance."""
        if self._aggregate_topic_policy is None:
            self._aggregate_topic_policy = AggregateTopicPolicy(self._fs, self._uri_resolver)
        return self._aggregate_topic_policy

    def _get_append_only_policy(self) -> AppendOnlyPolicy:
        """Get or create AppendOnlyPolicy instance."""
        if self._append_only_policy is None:
            self._append_only_policy = AppendOnlyPolicy(self._fs, self._uri_resolver)
        return self._append_only_policy

    def _get_skill_tool_policy(self) -> SkillToolPolicy:
        """Get or create SkillToolPolicy instance."""
        if self._skill_tool_policy is None:
            self._skill_tool_policy = SkillToolPolicy(self._fs, self._uri_resolver)
        return self._skill_tool_policy

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan using the appropriate policy.

        Args:
            candidate: CandidateMemory to plan for
            ctx: RequestContext for this operation

        Returns:
            WritePlan from the appropriate policy

        Raises:
            ValueError: If no policy can be found for this candidate
        """
        policy = self.route(candidate)
        if policy is None:
            raise ValueError(
                f"No policy found for category '{candidate.category}'. "
                f"Ensure schema exists and operation_mode is valid."
            )

        return policy.plan(candidate, ctx)

    def register_policy(self, category: str, policy: MergePolicy) -> None:
        """Register a custom policy for a category.

        Custom policies take precedence over schema-driven routing.
        Useful for special cases that don't fit the standard policy patterns.

        Args:
            category: Memory category
            policy: MergePolicy to use for this category
        """
        self._custom_policies[category] = policy
        logger.debug(f"Registered custom policy for category: {category}")

    def has_custom_policy(self, category: str) -> bool:
        return category in self._custom_policies
