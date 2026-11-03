"""Backend-neutral contracts for lexical retrieval projections."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from core.models import IndexRecord

INDEXABLE_METADATA_FIELDS = frozenset(
    {
        "account_id",
        "owner_space",
        "context_type",
        "category",
        "parent_uri",
        "abstract",
        "has_overview",
        "has_content",
        "active_count",
        "updated_at",
        "level",
        "uri",
        "unit_key",
        "agent_memory_unit_key",
        "unit_revision",
        "agent_memory_version",
        "grounding",
    }
)

# A bounded lexical query prevents pathological expressions from monopolizing
# either an embedded or remote provider. Providers may expose a configurable
# override, but should keep a finite limit.
DEFAULT_MAX_QUERY_TERMS = 64  # Conservative operational bound, configurable.


@dataclass(frozen=True)
class LexicalRecord:
    """Minimal document projection required by a lexical provider."""

    id: str
    uri: str
    level: int
    text: str
    account_id: str
    owner_space: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LexicalHit:
    """Backend-neutral lexical search result."""

    id: str
    score: float
    metadata: dict[str, Any]


@runtime_checkable
class LexicalAnalyzer(Protocol):
    """Deterministic document/query analyzer."""

    @property
    def identity(self) -> str: ...

    def analyze_document(self, text: str) -> list[str]: ...

    def analyze_query(self, text: str) -> list[str]: ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Small capability contract shared by lexical providers."""

    @property
    def supported_filters(self) -> frozenset[str]: ...

    def upsert_many(self, records: Sequence[LexicalRecord]) -> None: ...

    def delete(self, record_ids: Sequence[str]) -> None: ...

    def delete_scope(self, *, account_id: str, owner_space: str | None = None) -> int: ...

    def begin_rebuild(self) -> str: ...

    def replace_scopes(
        self,
        *,
        rebuild_id: str,
        account_id: str,
        owner_spaces: Sequence[str],
        records: Iterable[LexicalRecord],
    ) -> bool: ...

    def abort_rebuild(self, rebuild_id: str) -> None: ...

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[LexicalHit]: ...

    def health(self) -> dict[str, Any]: ...

    def close(self, timeout: float = 5.0) -> bool: ...


@runtime_checkable
class LexicalCheckpointStore(Protocol):
    """Durable projection-validity capability required for hybrid retrieval."""

    def ready_scopes(
        self, account_id: str, owner_spaces: Sequence[str]
    ) -> frozenset[str]: ...

    def dirty_generations(
        self, account_id: str, owner_spaces: Sequence[str]
    ) -> dict[str, int]:
        """Return monotonic invalidations, including global scope state."""
        ...

    def mark_scopes_dirty(
        self, scopes: Sequence[tuple[str, str]]
    ) -> None: ...

    def mark_all_scopes_dirty(self) -> None: ...


@runtime_checkable
class HybridLexicalIndex(LexicalIndex, LexicalCheckpointStore, Protocol):
    """Exact lexical capability set required by hybrid retrieval."""


def lexical_record_from_index_record(record: IndexRecord) -> LexicalRecord:
    """Create a minimal lexical projection from the shared index record."""

    filters = dict(record.filters or {})
    account_id = str(filters.get("account_id", ""))
    owner_space = str(filters.get("owner_space", ""))
    if not account_id or not owner_space:
        raise ValueError("lexical records require account_id and owner_space")

    combined = dict(record.metadata or {})
    combined.update(filters)
    combined.update({"uri": record.uri, "level": record.level})
    if record.level == 0 and not combined.get("abstract"):
        combined["abstract"] = record.text
    metadata = {
        key: combined[key]
        for key in INDEXABLE_METADATA_FIELDS
        if key in combined and combined[key] is not None
    }
    return LexicalRecord(
        id=record.id,
        uri=record.uri,
        level=record.level,
        text=record.text,
        account_id=account_id,
        owner_space=owner_space,
        metadata=metadata,
    )


def lexical_scopes_from_index_records(
    records: Sequence[IndexRecord],
) -> tuple[tuple[str, str], ...]:
    """Return the distinct lexical scopes affected by projection records."""

    scopes: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        filters = dict(record.filters or {})
        scope = (
            str(filters.get("account_id", "")),
            str(filters.get("owner_space", "")),
        )
        if all(scope) and scope not in seen:
            seen.add(scope)
            scopes.append(scope)
    return tuple(scopes)
