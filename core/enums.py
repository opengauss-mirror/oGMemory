"""Core enumerations for ContextEngine.

Phase 0 frozen — these enums are used across all packages.
"""

from enum import Enum


class NodeStatus(str, Enum):
    """Status of a ContextNode in AGFS.

    Corresponds to .meta.json status field.
    Used by Repair Job to determine node health and recovery actions.

    See CLAUDE.md §2 AGFS Directory Spec for write order semantics.
    """
    PENDING = "PENDING"  # Node directory created, write incomplete (steps ①-③ done, ④ not done)
    ACTIVE = "ACTIVE"    # Step ④ complete, node visible for retrieval
    BROKEN = "BROKEN"    # Repair Job detected missing/corrupt files, needs manual intervention
    ARCHIVED = "ARCHIVED"  # Soft-deleted node retained for audit/recovery but not visible


class ContextType(str, Enum):
    """High-level classification of context content.

    Used by TypedQuery for index-level filtering and by Planner for query routing.
    """
    MEMORY = "MEMORY"    # User memories: profile, preferences, entities, events
    SKILL = "SKILL"      # Agent capabilities: skills, tools, procedures
    RESOURCE = "RESOURCE"  # Reference materials, documentation, external resources


class EventType(str, Enum):
    """Types of outbox events for async processing.

    Events are written to .outbox/{event_id}.json and processed by OutboxWorker.
    """
    UPSERT_CONTEXT = "UPSERT_CONTEXT"      # Create or update a ContextNode
    UPSERT_DIRECTORY = "UPSERT_DIRECTORY"  # Update directory summary from children
    DELETE_CONTEXT = "DELETE_CONTEXT"      # Remove a ContextNode
    ARCHIVE_CONTEXT = "ARCHIVE_CONTEXT"    # Soft-delete a ContextNode (mark as ARCHIVED)
    MOVE_CONTEXT = "MOVE_CONTEXT"          # Rename/relocate a ContextNode
    UPSERT_RELATION = "UPSERT_RELATION"    # Add or update relation edges
    EXTRACT_AGENT_MEMORY = "EXTRACT_AGENT_MEMORY"


class RetrievalIntent(str, Enum):
    """Intent classification for retrieval queries.

    Used by RetrievalIntentClassifier to categorize search queries
    before vector search, enabling intent-aware retrieval instead of
    raw semantic similarity search.
    """
    BACKGROUND_SUPPLEMENT = "background_supplement"  # Profile, preferences, context
    HISTORICAL_DECISIONS = "historical_decisions"    # Past decisions, case outcomes
    OPEN_ITEMS = "open_items"                        # Pending tasks, open loops
    REUSABLE_SKILLS = "reusable_skills"              # Procedures, patterns, how-to
    ENTITY_RELATIONS = "entity_relations"            # People, orgs, entity graph
