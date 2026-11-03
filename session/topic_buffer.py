"""Session-level topic buffer for RCA pipeline.

Provides caching for stable slots and prefetch results from PrefetchManager.
This is a key component of Phase 0 (prefetch) and Phase 1 (assembly) of the
RuntimeContextAssembly (RCA) pipeline.
"""

from dataclasses import dataclass, field
from threading import RLock

from core.models import SeedHit
from retrieval.identity import memory_identity
from session.topic_detector import TopicDetection


@dataclass
class SlotContent:
    """Content stored in a named slot.

    Slots are used to cache stable content like identity, skills_summary,
    and archive_history that doesn't change frequently during a session.
    """
    content: str
    uris: list[str] = field(default_factory=list)
    tokens: int = 0
    cached_at: str = ""

    def __post_init__(self):
        from datetime import datetime, timezone
        if not self.cached_at:
            self.cached_at = datetime.now(timezone.utc).isoformat()


class SessionTopicBuffer:
    """Session-level cache for fixed slots and prefetch results.

    Manages two types of data:
    1. Stable slots (identity, skills_summary, archive_history) - cached across turns
    2. Prefetch results (pending_injection) - populated by PrefetchManager

    Also tracks which URIs have been injected to avoid duplicates in the
    working set across multiple assembly calls.
    """

    def __init__(self, session_id: str):
        """Initialize a new buffer for a session.

        Args:
            session_id: Unique session identifier
        """
        self._session_id = session_id
        self._slots: dict[str, SlotContent] = {}
        self._pending_injection: list[SeedHit] | None = None
        self._injected_uris: set[tuple[str, str]] = set()
        self._current_topic: TopicDetection | None = None
        self._injection_lock = RLock()
        self._injection_epoch = 0

    # ---------------------------------------------------------------------
    # Stable Base caching
    # ---------------------------------------------------------------------

    def get_cached_slot(self, name: str) -> SlotContent | None:
        """Get cached content for a named slot.

        Args:
            name: Slot name (e.g., "identity", "skills_summary", "archive_history")

        Returns:
            SlotContent if cached, None otherwise
        """
        return self._slots.get(name)

    def set_cached_slot(self, name: str, content: SlotContent) -> None:
        """Cache content for a named slot.

        Args:
            name: Slot name to cache
            content: SlotContent to store
        """
        self._slots[name] = content

    def invalidate(self, name: str) -> None:
        """Invalidate a cached slot.

        Args:
            name: Slot name to invalidate
        """
        if name in self._slots:
            del self._slots[name]

    def clear_all_slots(self) -> None:
        """Clear all cached slots."""
        self._slots.clear()

    def update_topic(self, detection: TopicDetection) -> bool:
        """Store the current detected topic.

        Returns True when this is the first topic or when the topic label changed.
        """
        previous = self._current_topic
        self._current_topic = detection
        if previous is None:
            return True
        return previous.label != detection.label

    def get_current_topic(self) -> TopicDetection | None:
        """Return the latest detected topic for this session."""
        return self._current_topic

    # ---------------------------------------------------------------------
    # Prefetch results
    # ---------------------------------------------------------------------

    def get_pending_injection(self) -> list[SeedHit] | None:
        """Get pending prefetch results waiting to be injected.

        Returns:
            List of SeedHit from prefetch, or None if no prefetch pending
        """
        with self._injection_lock:
            return self._pending_injection

    def injection_snapshot(self) -> tuple[list[SeedHit] | None, int]:
        """Capture the pending batch and its topic generation."""
        with self._injection_lock:
            return self._pending_injection, self._injection_epoch

    def consume_injection(self, snapshot: tuple[list[SeedHit] | None, int], delivered_uris: list[str],
                          *, delivered_units: list[dict] | None = None) -> None:
        """Confirm rendered entries while preserving a newer prefetch batch."""
        hits, epoch = snapshot
        delivered = {memory_identity(uri) for uri in delivered_uris}
        delivered.update(memory_identity(unit["uri"], unit) for unit in delivered_units or [])
        with self._injection_lock:
            if epoch != self._injection_epoch:
                return
            self._injected_uris.update(memory_identity(hit.uri, hit.metadata) for hit in hits or []
                                       if memory_identity(hit.uri, hit.metadata) in delivered)
            if self._pending_injection is hits:
                self._pending_injection = [hit for hit in hits or []
                    if memory_identity(hit.uri, hit.metadata) not in self._injected_uris] or None

    def set_pending_injection(self, hits: list[SeedHit]) -> None:
        """Set pending prefetch results from PrefetchManager.

        Args:
            hits: List of SeedHit from L0 vector recall
        """
        with self._injection_lock:
            self._pending_injection = hits

    def clear_pending_injection(self) -> None:
        """Clear pending prefetch results after consumption."""
        with self._injection_lock:
            self._pending_injection = None

    # ---------------------------------------------------------------------
    # Injection tracking
    # ---------------------------------------------------------------------

    def mark_injected(self, uris: list[str]) -> None:
        """Mark URIs as injected to avoid duplicates.

        Args:
            uris: List of URIs that were included in the working set
        """
        with self._injection_lock:
            self._injected_uris.update(memory_identity(uri) for uri in uris)

    def was_injected(self, uri: str, metadata: dict | None = None) -> bool:
        """Check if a URI was already injected in a previous turn.

        Args:
            uri: URI to check

        Returns:
            True if URI was already injected, False otherwise
        """
        with self._injection_lock:
            return memory_identity(uri, metadata) in self._injected_uris

    def clear_injected_tracking(self) -> None:
        """Clear injected URI tracking (e.g., for new session)."""
        with self._injection_lock:
            self._injected_uris.clear()
            self._injection_epoch += 1

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Get the session ID for this buffer."""
        return self._session_id

    def get_slot_names(self) -> list[str]:
        """Get list of currently cached slot names."""
        return list(self._slots.keys())

    def __repr__(self) -> str:
        return f"SessionTopicBuffer(session_id={self._session_id}, slots={len(self._slots)}, pending={len(self._pending_injection) if self._pending_injection else 0})"
