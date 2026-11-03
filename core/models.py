"""Core domain models for ContextEngine.

Phase 0 frozen — these dataclasses define the shape of all domain objects.
All models use stdlib dataclass for minimal dependencies.

Retrieval-chain models (SeedHit → SearchMemoryResult) added for the
search_memory pipeline; they follow the same conventions.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class Role(str, Enum):  # noqa: UP042 - serialized values must remain legacy-compatible
    """Minimal RBAC roles for multi-tenant control plane."""
    ROOT = "ROOT"
    ADMIN = "ADMIN"
    MEMBER = "MEMBER"


def user_space_name(user_id: str) -> str:
    """Generate owner_space identifier for a user.

    Uses colon format for consistency across the codebase.
    Format: "user:{user_id}"

    Args:
        user_id: User identifier

    Returns:
        Owner space string in format "user:{user_id}"
    """
    return f"user:{user_id}"


def agent_space_name(agent_id: str) -> str:
    """Generate owner_space identifier for an agent.

    Uses colon format for consistency across the codebase.
    Format: "agent:{agent_id}"

    Args:
        agent_id: Agent identifier

    Returns:
        Owner space string in format "agent:{agent_id}"
    """
    return f"agent:{agent_id}"


@dataclass(frozen=True)
class RequestContext:
    """Execution context for all operations.

    Enforces multi-tenant isolation by carrying account_id and owner_space.
    Passed to all ContextFS and RelationStore methods for access control.
    """
    account_id: str
    user_id: str
    agent_id: str
    session_id: str
    trace_id: str
    role: Role | str = Role.MEMBER
    visible_owner_spaces: tuple[str, ...] = ()

    def user_space_name(self) -> str:
        """Generate owner_space identifier for user scope.

        Returns owner_space in format 'user:{user_id}'.
        """
        return f"user:{self.user_id}"

    def agent_space_name(self) -> str:
        """Generate owner_space identifier for agent scope.

        Returns owner_space in format 'agent:{agent_id}'.
        """
        return f"agent:{self.agent_id}"


@dataclass
class ContextNode:
    """A node in the context knowledge graph.

    Persisted to AGFS as a directory with content.md, .relations.json,
    .abstract.md, .overview.md, .meta.json files.

    See CLAUDE.md §2 AGFS Directory Spec for file layout.
    See CLAUDE.md §1 URI Spec for URI patterns by category.
    """
    uri: str
    context_type: str  # MEMORY | SKILL | RESOURCE
    category: str  # profile|preference|entity|event|case|pattern|skill
    level: int  # Depth in directory tree (0 for root-level nodes)
    owner_space: str  # user_space_name() or agent_space_name()
    abstract: str  # ≤200 chars summary (→ .abstract.md)
    overview: str  # Structured overview (→ .overview.md)
    content: str  # Full original content (→ content.md)
    metadata: dict[str, Any] = field(default_factory=dict)  # tags / timestamps / status

    @property
    def parent_uri(self) -> str:
        """Get parent URI by removing the last path segment.

        For example:
        - "ctx://acme/users/alice/memories/preferences/coding_style"
          → "ctx://acme/users/alice/memories/preferences/"
        - "ctx://acme/users/alice/memories/profile"
          → "ctx://acme/users/alice/memories/"
        - "ctx://acme/users/alice/memories/preferences/"
          → "ctx://acme/users/alice/memories/"
        """
        if "/" not in self.uri:
            return ""

        uri = self.uri
        if uri.endswith("/"):
            uri = uri.rstrip("/")

        if "/" not in uri:
            return ""

        return uri.rsplit("/", 1)[0] + "/"


@dataclass
class RelationEdge:
    """A directed relationship between two ContextNodes.

    Persisted in .relations.json at the source node.
    Weight indicates confidence/strength (0.0-1.0).
    """
    from_uri: str
    to_uri: str
    relation_type: str  # e.g. "related_to" | "derived_from" | "contradicts" | "SEQUENCE"
    weight: float  # 0.0-1.0
    reason: str


@dataclass
class CandidateMemory:
    """Extracted candidate before merge policy decisions.

    Produced by CandidateExtractor from message history.
    Consumed by MergePolicy to produce WritePlan.
    """
    category: str  # Same as ContextNode.category
    owner_scope: str  # "user" | "agent"
    routing_key: str  # Stable logical key; occurrence nodes receive an allocated event ID
    abstract: str
    overview: str
    content: str
    confidence: float  # 0.0-1.0, below threshold triggers skip
    when: str | None = None  # Temporal context (resolved absolute date)
    who: str | None = None  # Actor context
    where: str | None = None  # Location context
    tool_stats: dict | None = None  # Tool usage stats (only for 'tool' category)
    # Speaker attribution (profile disambiguation)
    evidence_quote: str | None = None  # Verbatim quote from source text
    attributed_speaker: str | None = None  # Who said this ('user' or speaker name)
    attribution_basis: str | None = None  # self_first_person | self_named | other_named
    provenance_ids: list[str] = field(default_factory=list)
    object_name: str = ""  # Source name of the entity being matched
    scope_ref: str = ""  # Identity namespace distinguishing subjects, not evidence provenance
    target_uri: str = ""  # URI of the existing node selected for an update
    expected_version: int | None = None  # Version read from target_uri for the update
    legacy_routing_key: str = ""  # Previous validator's key for existing-node lookup
    title: str = ""  # Source-grounded label for an event or case
    complete_occurrence_context: bool = False  # Internal full snapshot: null context clears old metadata.
    write_operation_id: str = ""  # Internal durable batch receipt; never an extraction-tool argument.
    source_occurrence_id: str = ""  # Internal accepted source partition; never a tool argument.
    verified_routing_aliases: list[str] = field(default_factory=list)
    verified_object_aliases: list[str] = field(default_factory=list)
    source_contexts: list[str] = field(default_factory=list)  # Internal complete extraction-span evidence.
    node_draft_indices: list[int] = field(default_factory=list)  # Contributors in the durable draft checkpoint.


class WriteAction:
    """Constants for WritePlan action types."""
    CREATE = "create"
    MERGE = "merge"
    ARCHIVE = "archive"
    DELETE = "delete"


@dataclass
class WritePlan:
    """Output of MergePolicy — determines write action.

    Action values:
    - create: New node, does not exist
    - merge: Existing node, merge content
    - append: New version to existing node (events/cases)
    - skip: Duplicate or below confidence threshold
    """
    action: str  # create | merge | append | skip
    target_uri: str
    merged_fields: dict[str, Any] = field(default_factory=dict)  # For merge actions
    relation_edges: list[RelationEdge] = field(default_factory=list)


@dataclass
class IndexRecord:
    """A record in the vector index.

    Index emitters may create L0, L1, and L2 records as applicable.
    Leaf ContextNodes use L0 plus optional L2; directory summaries own L1.
    The id is stable: sha256(uri + ":" + str(level))[:16].

    CRITICAL: filters MUST contain account_id + owner_space.
    Missing filters cause silent cross-tenant leakage (no runtime error).
    Contract test enforces this invariant.
    """
    id: str
    uri: str
    level: int  # 0=abstract / 1=overview / 2=content
    text: str  # Text content for embedding
    filters: dict[str, Any]  # MUST contain account_id + owner_space
    metadata: dict[str, Any] = field(default_factory=dict)  # category / context_type / has_overview / has_content

    @staticmethod
    def generate_id(uri: str, level: int) -> str:
        """Generate stable IndexRecord id.

        Stable id enables idempotent upsert to vector index.
        """
        return hashlib.sha256(f"{uri}:{level}".encode()).hexdigest()[:16]


@dataclass
class OutboxEvent:
    """Event for async index synchronization.

    Written to .outbox/{event_id}.json, processed by OutboxWorker.
    Events are durable and survive process restarts (stored in AGFS).
    """
    event_id: str  # uuid4
    event_type: str  # UPSERT_CONTEXT|UPSERT_DIRECTORY|DELETE_CONTEXT|MOVE_CONTEXT|UPSERT_RELATION
    uri: str
    payload: dict[str, Any]
    status: str  # PENDING | PROCESSING | DONE | FAILED
    retry_count: int = 0
    created_at: str = ""  # ISO8601, set to datetime.utcnow().isoformat() if empty
    next_retry_at: str = ""  # ISO8601, earliest time for next retry (exponential backoff)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


@dataclass
class TypedQuery:
    """A query with type annotations for filtered retrieval.

    account_id is injected by service layer, cannot be overridden by caller.
    categories is inferred by Planner from query text.
    target_uri allows direct lookup skipping vector recall.
    intent is the retrieval-intent classification for Layer 3.
    """
    text: str  # Original query text, used for vectorization
    context_type: str  # MEMORY | SKILL | RESOURCE
    categories: list[str]  # Planner-inferred categories for pre-filtering
    target_uri: str | None = None  # If known, direct read, skip vector recall
    top_k: int = 10  # L0 recall count limit
    account_id: str = ""  # Injected by service layer
    owner_space: str | list[str] | None = None  # Further limit to user or agent scope
    intent: str | None = None  # RetrievalIntent value, for Layer 3 intent-aware retrieval
    resource_type: str | None = None  # RESOURCE subtype pre-filter (for example, "code")
    memory_directories: list[str] | None = None  # URI type directories applied before candidate limits


# ---------------------------------------------------------------------------
# Retrieval pipeline models
# ---------------------------------------------------------------------------

class RetrieverMode(str, Enum):  # noqa: UP042 - public enum compatibility
    """Controls depth of scoring: QUICK skips reranker, THINKING uses it."""
    THINKING = "thinking"
    QUICK = "quick"


@dataclass
class SeedHit:
    """Raw result from VectorIndex.

    Carries level and parent_uri needed for hierarchical expansion.
    """
    uri: str
    score: float
    level: int = 2
    parent_uri: str | None = None
    context_type: str = ""
    category: str = ""
    owner_space: str = ""
    abstract: str = ""
    has_overview: bool = False
    has_content: bool = False
    active_count: int = 0
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedResult:
    """Output of SeedRetriever — two streams after level-based split."""
    starting_points: list[SeedHit] = field(default_factory=list)
    initial_candidates: list[SeedHit] = field(default_factory=list)
    query_vector: list[float] = field(default_factory=list)
    root_uris: list[str] = field(default_factory=list)


@dataclass
class LeafHit:
    """L2 node produced by hierarchy expansion or direct L2 retrieval."""
    uri: str
    score: float
    level: int = 2
    depth: int = 0
    parent_uri: str | None = None
    category: str = ""
    owner_space: str = ""
    abstract: str = ""
    has_overview: bool = False
    has_content: bool = False
    hierarchy_path: list[str] = field(default_factory=list)
    active_count: int = 0
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedBlock:
    """Final output block returned by search_memory or read_memory."""
    uri: str
    level_hit: str = "L2"
    score: float = 0.0
    ranking_score: float | None = field(default=None, repr=False, compare=False)
    category: str = ""
    owner_space: str = ""
    abstract: str | None = None
    overview: str | None = None
    content_excerpt: str | None = None
    relations: list[RelationEdge] = field(default_factory=list)
    has_overview: bool = False
    has_content: bool = False
    match_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchMemoryResult:
    """Top-level result of search_memory — pure structured data."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    query: str = ""
    typed_queries: list[TypedQuery] = field(default_factory=list)
    hits: list[RetrievedBlock] = field(default_factory=list)
    trace: Any = None


@dataclass
class RetrievalConfig:
    """Tunables shared across all retrieval pipeline stages."""
    max_top_k: int = 50
    default_top_k: int = 10
    global_search_topk: int = 30
    candidate_pool_enabled: bool = False
    candidate_pool_multiplier: int = 3
    min_candidate_pool_size: int = 15
    result_ranker_strategy: str = "semantic"
    global_result_rerank_enabled: bool = False
    owner_scope_filter_enabled: bool = False
    max_convergence_rounds: int = 3
    score_propagation_alpha: float = 0.5
    hotness_alpha: float = 0.2
    hotness_half_life_days: float = 7.0
    default_score_threshold: float = 0.0
    # Appended to preserve the legacy positional constructor.
    hybrid_enabled: bool = False
    vector_top_k: int = 100
    lexical_top_k: int = 50
    rrf_k: int = 60
    semantic_weight: float = 1.0
    lexical_weight: float = 1.0
    expand_hierarchy: bool = True
    include_agent_memories: bool = False


# ---------------------------------------------------------------------------
# Unified assembly models
# ---------------------------------------------------------------------------

@dataclass
class BudgetTier:
    """A budget tier with priority and degradation support.

    Tiers define how token budget is allocated across different memory components:
    - priority: 0=highest (must keep), 10=lowest (first to drop)
    - min_tokens: Floor — never go below this
    - max_tokens: Ceiling — never exceed this
    - degradable: Can truncate to min when budget tight
    - expandable: Can grow beyond max when budget available
    """
    name: str
    priority: int  # 0=highest (must keep), 10=lowest (first to drop)
    min_tokens: int  # Floor — never go below this
    max_tokens: int  # Ceiling — never exceed this
    degradable: bool = True  # Can truncate to min when budget tight
    expandable: bool = False  # Can grow beyond max when budget available


@dataclass
class TokenBudget:
    """Token budget configuration with priority-tiered allocation.

    Tiers by priority:
    0. Identity (profile, preferences) — must keep, never degrade
    1. Session state (active task, constraints) — must keep, never degrade
    5. Archive history — degradable (full → overview → abstract)
    8. Working set (search results) — degradable AND expandable

    Legacy properties (archive_ratio, session_state_ratio) preserved for backward compatibility.
    """
    total: int = 128_000
    archive_ratio: float = 0.7  # Legacy compat
    session_state_ratio: float = 0.2  # Legacy compat

    # Priority tiers
    identity_tier: BudgetTier = field(default_factory=lambda: BudgetTier(
        name="identity", priority=0, min_tokens=500, max_tokens=5000,
        degradable=False, expandable=False,
    ))
    session_state_tier: BudgetTier = field(default_factory=lambda: BudgetTier(
        name="session_state", priority=1, min_tokens=1000, max_tokens=10000,
        degradable=False, expandable=False,
    ))
    archive_tier: BudgetTier = field(default_factory=lambda: BudgetTier(
        name="archive", priority=5, min_tokens=500, max_tokens=40000,
        degradable=True, expandable=False,
    ))
    working_set_tier: BudgetTier = field(default_factory=lambda: BudgetTier(
        name="working_set", priority=8, min_tokens=0, max_tokens=20000,
        degradable=True, expandable=True,
    ))

    @property
    def archive_limit(self) -> int:
        """Maximum tokens allowed for archives (legacy compat)."""
        return self.allocate().get("archive", int(self.total * self.archive_ratio))

    @property
    def session_state_limit(self) -> int:
        """Maximum tokens for session state (legacy compat)."""
        return self.allocate().get("session_state", int(self.total * self.session_state_ratio))

    def allocate(self) -> dict[str, int]:
        """Allocate budget across tiers by priority.

        Returns dict of tier_name -> allocated_tokens.
        Higher priority tiers get first claim on budget.
        """
        tiers = sorted(
            [self.identity_tier, self.session_state_tier,
             self.archive_tier, self.working_set_tier],
            key=lambda t: t.priority,
        )
        remaining = self.total
        result = {}

        # Pass 1: Allocate minimums for non-degradable tiers
        for tier in tiers:
            if not tier.degradable:
                allocated = min(tier.min_tokens, remaining)
                result[tier.name] = allocated
                remaining -= allocated

        # Pass 2: Allocate up to max for all tiers (priority order)
        for tier in tiers:
            current = result.get(tier.name, 0)
            wanted = tier.max_tokens - current
            if wanted > 0 and remaining > 0:
                extra = min(wanted, remaining)
                result[tier.name] = current + extra
                remaining -= extra

        # Pass 3: Expand expandable tiers with leftover budget
        if remaining > 0:
            for tier in tiers:
                if tier.expandable and remaining > 0:
                    current = result.get(tier.name, 0)
                    result[tier.name] = current + remaining
                    remaining = 0
                    break

        return result


@dataclass
class ArchiveRef:
    """Reference to a session archive.

    Carries metadata about the archive without loading full content.
    Used for assembly decisions and archive index display.
    """
    archive_id: str
    archive_uri: str
    abstract: str
    overview: str | None = None
    tokens: int = 0


@dataclass
class SessionArchiveMeta:
    """Metadata for a session archive.

    Stored as archive summary, used for archive selection.
    """
    session_id: str
    archive_id: str
    created_at: str
    message_count: int
    overview: str
    abstract: str


@dataclass
class ComposedContext:
    """Result of memory assembly operation.

    Defines the cognitive contract: what context is available and where it goes.
    Semantic slots map to specific prompt positions in the runtime.
    """
    # --- Semantic slots (the cognitive contract) ---
    identity_context: str = ""           # Layer 1a: Profile + stable preferences
    episodic_context: str = ""           # Layer 1b: Archive history
    session_context: str = ""            # Layer 2: Structured session state
    task_context: str = ""               # Current task description
    retrieved_evidence: str = ""         # Layer 3: Working set search results
    open_loops: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    # --- Budget metadata ---
    estimated_tokens: int = 0
    budget_used_by_slot: dict[str, int] = field(default_factory=dict)

    # --- Legacy compat (derived from semantic slots) ---
    messages: list[dict] = field(default_factory=list)
    archive_count: int = 0
    archive_included: bool = False
    system_prompt_suffix: str = ""
    stats: dict = field(default_factory=dict)
