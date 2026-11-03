"""Merge policies for determining write actions.

Each policy implements the MergePolicy Protocol and decides whether
to create, merge, append, or skip a candidate memory.
"""

import uuid as uuid_lib
from datetime import UTC, datetime

from commit.routing_key import normalize_routing_key
from core.errors import ConcurrentModificationError
from core.interfaces import ContextFS, MergePolicy
from core.models import CandidateMemory, RequestContext, WritePlan
from core.uri_resolver import URIResolver
from core.validation import validate_candidate


def _merge_provenance_ids(
    existing: list[str] | None,
    incoming: list[str] | None,
) -> list[str]:
    """Merge two provenance ID lists, preserving order and deduplicating."""
    merged = (existing or []) + (incoming or [])
    return list(dict.fromkeys(merged))


def _schema_routing_fields(category: str, routing_key: str) -> dict[str, str]:
    """Build URI template fields from the candidate routing key."""
    return {"routing_key": routing_key}


class ProfilePolicy(MergePolicy):
    """Merge policy for user profile nodes.

    Behavior (CLAUDE.md §5):
    - Target URI: ctx://{account}/users/{user}/memories/profile
    - Create the profile once, then reconcile incoming facts with its current content
    - Keep the current version for optimistic concurrency control

    Profile is a fixed URI that gets continuously updated.
    """

    def __init__(self, fs: ContextFS, uri_resolver: URIResolver):
        """Initialize ProfilePolicy.

        Args:
            fs: ContextFS for checking existing nodes
            uri_resolver: URIResolver for URI construction
        """
        self._fs = fs
        self._uri_resolver = uri_resolver

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan for profile candidate.

        Args:
            candidate: CandidateMemory (must be category="profile")
            ctx: RequestContext for this operation

        Returns:
            WritePlan with the fixed target URI and the current profile snapshot
        """
        target_uri = self._uri_resolver.resolve("profile", {"routing_key": candidate.routing_key}, ctx)

        if candidate.target_uri and candidate.target_uri != target_uri:
            raise ValueError("profile update target must match its fixed URI")
        if candidate.target_uri and (type(candidate.expected_version) is not int or candidate.expected_version < 1):
            raise ValueError("profile update requires a positive read version")
        if candidate.target_uri:
            validate_candidate(candidate)
            if not all(isinstance(getattr(candidate, name), str) and getattr(candidate, name).strip()
                       for name in ("abstract", "overview", "content")):
                raise ValueError("profile update requires complete content, overview and abstract")
        exists = self._fs.exists(target_uri, ctx)
        merged_fields = {"expected_version": 0}
        if exists:
            existing_node = self._fs.read_node(target_uri, ctx)
            if candidate.target_uri:
                if candidate.expected_version != existing_node.metadata.get("version", 0):
                    raise ConcurrentModificationError(
                        target_uri, candidate.expected_version, existing_node.metadata.get("version", 0),
                    )
                if (all(getattr(candidate, name) == getattr(existing_node, name)
                        for name in ("abstract", "overview", "content"))
                        and set(candidate.provenance_ids) <= set(existing_node.metadata.get("provenance_ids", []))
                        and all(not getattr(candidate, name)
                                or getattr(candidate, name) == existing_node.metadata.get(name)
                                for name in ("when", "who", "where"))):
                    return WritePlan(action="skip", target_uri=target_uri)
            merged_fields = {
                "existing_metadata": dict(existing_node.metadata),
                "updated_at": datetime.now(UTC).isoformat(),
                "expected_version": existing_node.metadata.get("version", 0),
                "provenance_ids": _merge_provenance_ids(
                    existing_node.metadata.get("provenance_ids"), candidate.provenance_ids,
                ),
            }
            if not candidate.target_uri:
                merged_fields.update({f"existing_{name}": getattr(existing_node, name)
                                      for name in ("abstract", "overview", "content")})
        elif candidate.target_uri:
            raise ValueError("profile update target must exist")

        return WritePlan(
            action="merge" if exists else "create",
            target_uri=target_uri,
            merged_fields=merged_fields,
            relation_edges=[],
        )


class AggregateTopicPolicy(MergePolicy):
    """Merge policy for topic-based memories.

    Applies to: preference, entity, pattern

    Behavior (CLAUDE.md §5):
    - Target URI: ctx://.../{category}/{slug}
    - If exists: merge content into overview/content
    - If not exists: create new node
    - Similar slug handling: edit distance < 2 or semantic > 0.9 → merge

    Uses routing_key as slug for URI construction.
    """

    def __init__(self, fs: ContextFS, uri_resolver: URIResolver, similarity_threshold: float = 0.9):
        """Initialize AggregateTopicPolicy.

        Args:
            fs: ContextFS for checking existing nodes
            uri_resolver: URIResolver for URI construction
            similarity_threshold: Semantic similarity threshold for slug merging
        """
        self._fs = fs
        self._uri_resolver = uri_resolver
        self._similarity_threshold = similarity_threshold

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan for topic-based candidate.

        Args:
            candidate: CandidateMemory (preference, entity, or pattern)
            ctx: RequestContext for this operation

        Returns:
            WritePlan with action="create" or "merge"
        """
        # Normalize routing_key to avoid fragmentation
        normalized_key = normalize_routing_key(candidate.routing_key, candidate.category)

        routing_fields = _schema_routing_fields(candidate.category, normalized_key)

        target_uri = self._uri_resolver.resolve(candidate.category, routing_fields, ctx)

        # Check if exact URI exists
        exists = self._fs.exists(target_uri, ctx)
        if not exists and candidate.legacy_routing_key:
            legacy_key = normalize_routing_key(candidate.legacy_routing_key, candidate.category)
            legacy_fields = _schema_routing_fields(candidate.category, legacy_key)
            legacy_uri = self._uri_resolver.resolve(candidate.category, legacy_fields, ctx)
            if legacy_uri != target_uri and self._fs.exists(legacy_uri, ctx):
                target_uri, exists = legacy_uri, True
        if exists:
            # Read existing node to get version for optimistic locking
            existing_node = self._fs.read_node(target_uri, ctx)
            current_version = existing_node.metadata.get("version", 0)

            return WritePlan(
                action="merge",
                target_uri=target_uri,
                merged_fields={
                    "abstract": candidate.abstract,
                    "existing_overview": existing_node.overview,
                    "overview_append": candidate.overview,
                    "existing_content": existing_node.content,
                    "content_append": candidate.content,
                    "expected_version": current_version,
                    "provenance_ids": _merge_provenance_ids(
                        existing_node.metadata.get("provenance_ids"),
                        candidate.provenance_ids,
                    ),
                },
                relation_edges=[],
            )

        # Check for similar slugs via normalization (see routing_key.py)
        # If not found with normalized key, create new node

        return WritePlan(
            action="create",
            target_uri=target_uri,
            merged_fields={},
            relation_edges=[],
        )


class AppendOnlyPolicy(MergePolicy):
    """Merge policy for time-series events.

    Applies to: event, case

    Behavior (CLAUDE.md §5):
    - event_id/case_id globally unique (timestamp_uuid)
    - Always create, never overwrite history
    - Establish SEQUENCE relations for same-session events

    Each event is a new node - historical record preserved.
    """

    def __init__(self, fs: ContextFS, uri_resolver: URIResolver):
        """Initialize AppendOnlyPolicy.

        Args:
            fs: ContextFS for checking existing nodes
            uri_resolver: URIResolver for URI construction
        """
        self._fs = fs
        self._uri_resolver = uri_resolver

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan for event/case candidate.

        Args:
            candidate: CandidateMemory (event or case)
            ctx: RequestContext for this operation

        Returns:
            WritePlan with action="create" and unique event ID
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        unique_id = uuid_lib.uuid4().hex[:8]
        fields = {"routing_key": f"{candidate.routing_key}_{timestamp}_{unique_id}"}

        target_uri = self._uri_resolver.resolve(candidate.category, fields, ctx)

        # Always create new event nodes (never merge)
        return WritePlan(
            action="create",
            target_uri=target_uri,
            merged_fields={},
            relation_edges=[],
        )


class SkillToolPolicy(MergePolicy):
    """Merge policy for agent skills and tools.

    Applies to: skill, tool

    Behavior (CLAUDE.md §5):
    - skill URI: ctx://{account}/agents/{agent}/skills/{routing_key}
    - tool URI:  ctx://{account}/agents/{agent}/memories/tools/{routing_key}
    - Fixed URI, always exists or is created
    - Cumulative: best practices, failures, parameters appended
    - Usage stats tracked in metadata

    Skills and tools accumulate knowledge over time.
    """

    def __init__(self, fs: ContextFS, uri_resolver: URIResolver):
        """Initialize SkillToolPolicy.

        Args:
            fs: ContextFS for checking existing nodes
            uri_resolver: URIResolver for URI construction
        """
        self._fs = fs
        self._uri_resolver = uri_resolver

    def plan(self, candidate: CandidateMemory, ctx: RequestContext) -> WritePlan:
        """Generate WritePlan for skill/tool candidate.

        Args:
            candidate: CandidateMemory (skill or tool)
            ctx: RequestContext for this operation

        Returns:
            WritePlan with action="create" or "merge" (accumulate)
        """
        fields = _schema_routing_fields(candidate.category, candidate.routing_key)

        target_uri = self._uri_resolver.resolve(candidate.category, fields, ctx)

        exists = self._fs.exists(target_uri, ctx)

        if exists:
            # Read existing node to get version for optimistic locking
            existing_node = self._fs.read_node(target_uri, ctx)
            current_version = existing_node.metadata.get("version", 0)

            return WritePlan(
                action="merge",
                target_uri=target_uri,
                merged_fields={
                    "skill_merge": True,
                    "existing_content": existing_node.content,
                    "existing_overview": existing_node.overview,
                    "existing_abstract": existing_node.abstract,
                    "new_content": candidate.content,
                    "new_overview": candidate.overview,
                    "new_abstract": candidate.abstract,
                    "usage_count": existing_node.metadata.get("usage_count", 0) + 1,
                    "expected_version": current_version,
                    "provenance_ids": _merge_provenance_ids(
                        existing_node.metadata.get("provenance_ids"),
                        candidate.provenance_ids,
                    ),
                },
                relation_edges=[],
            )
        else:
            return WritePlan(
                action="create",
                target_uri=target_uri,
                merged_fields={
                    "usage_count": 1,
                },
                relation_edges=[],
            )
