"""Database-neutral contracts for scene projection and retrieval."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.models import ContextNode, RequestContext
from scene.models import PersistedScene, SceneArchive, SceneCluster, SceneSnapshot

SceneSourceState = tuple[int, object]
ResolvedArchiveSource = tuple[str, float]
SceneNodeReader = Callable[[str, RequestContext], ContextNode]


@runtime_checkable
class SceneSnapshotRepository(Protocol):
    """Persistence port for one scoped scene snapshot.

    Implementations own transactions, tenant isolation, locking, and the
    serialization format. Scene retrieval and decision code only depends on
    this port.
    """

    def load_snapshot(self, account_id: str, owner_space: str) -> SceneSnapshot: ...

    def compare_and_swap(self, expected: SceneSnapshot, replacement: SceneSnapshot) -> bool: ...


class SceneEvidenceRepository(SceneSnapshotRepository, Protocol):
    """Publication port sharing the canonical writers' transaction or mutation lock."""

    def compare_and_swap_checked(
        self, expected: SceneSnapshot, replacement: SceneSnapshot,
        check: Callable[[SceneNodeReader | None], bool],
    ) -> bool:
        """Hold canonical mutation protection across check and snapshot CAS.

        The backend acquires protection for the account/owner's evidence first,
        invokes check with a transaction-bound reader (or None for a shared
        mutation lock), and commits only when it returns True.
        A failed check returns False. All canonical writers share this protection.
        """
        ...


@dataclass(frozen=True)
class SceneDecisionInput:
    """Detached scenes and a bounded evidence window at one revision.

    Nodes include offered evidence and complete members of selected scene candidates.
    Embeddings use node URIs as keys. Adapters return caller-owned node and vector
    copies so a decision can operate independently of subsequent backend changes.
    """

    snapshot: SceneSnapshot
    nodes: tuple[ContextNode, ...]
    embeddings: dict[str, list[float]]
    unavailable_uris: tuple[str, ...] = ()


@runtime_checkable
class SceneDecisionStore(Protocol):
    """Consistent snapshot reads and atomic publication scoped by account and owner.

    Scene changes advance the monotonic scope revision. Adapters validate source
    versions during publication and own transactions across readers and writers.
    """

    def load_input(
        self,
        account_id: str,
        owner_space: str,
        *,
        node_uris: Sequence[str] | None = None,
    ) -> SceneDecisionInput:
        """Read one revision and the bounded evidence window for a job.

        Adapters should include the offered URIs plus existing scene members
        and any evidence required to compare or merge those scenes.
        """
        ...

    def compare_and_swap(
        self, expected: SceneSnapshot, replacement: SceneSnapshot,
        *, evidence_nodes: Sequence[ContextNode] = (),
        source_fingerprints: Sequence[tuple[str, str]] = (),
    ) -> bool:
        """Publish the full scene replacement when the scope revision still matches.

        Both snapshots identify the same account and owner. The replacement
        revision equals expected.revision + 1. Return True after atomic commit;
        return False on revision conflict, preserving the current state.
        Raise ValueError for a scope mismatch or invalid replacement revision.
        Include any nodes acquired by preflight tools in the same atomic source check.
        Source-only memory fingerprints capture the state when tool evidence was read.
        """
        ...


@runtime_checkable
class SceneNodeSource(Protocol):
    """Source of canonical memory nodes used to build scene projections."""

    def scan_active_nodes(
        self,
        *,
        account_id: str,
        owner_spaces: Sequence[str],
        page_size: int,
    ) -> Iterator[ContextNode]: ...


@runtime_checkable
class SceneStore(Protocol):
    """Read-side scene store contract."""

    def load_active(
        self,
        account_id: str,
        owner_space: str,
    ) -> list[PersistedScene]: ...

    def contributions(
        self,
        account_id: str,
        owner_spaces: Sequence[str],
        candidate_uris: Sequence[str],
    ) -> list[tuple[PersistedScene, str, float]]: ...

    def archive_evidence(
        self,
        account_id: str,
        scene_ids: Sequence[str],
    ) -> list[SceneArchive]: ...


@runtime_checkable
class SceneProjectionStore(SceneStore, Protocol):
    """Write-side scene projection contract.

    ``publish`` must atomically install one active version for a scope, retire
    projections absent from ``clusters``, and advance source state only after
    the complete projection succeeds. Locking is backend-owned.
    """

    def list_dirty_scopes(
        self,
        account_id: str,
        limit: int = 20,
    ) -> list[str]: ...

    def scope_is_dirty(self, account_id: str, owner_space: str) -> bool: ...

    def projection_is_quiescent(self, account_id: str) -> bool: ...

    def source_state(
        self,
        account_id: str,
        owner_space: str,
    ) -> SceneSourceState: ...

    def publish(
        self,
        account_id: str,
        owner_space: str,
        clusters: Sequence[SceneCluster],
        node_count: int,
        max_updated_at: object,
        identity_overlap: float,
    ) -> None: ...


class SceneArchiveResolver(Protocol):
    def resolve_archive_sources(
        self,
        account_id: str,
        archive_nodes: Sequence[ContextNode],
    ) -> Mapping[str, ResolvedArchiveSource]: ...


class SceneSourceExpansionStore(Protocol):
    def source_related_contexts(self, account_id: str, scene_id: str) -> set[str]: ...

    def source_related_nodes(self, account_id: str, scene_id: str) -> list[dict]: ...


class SceneEntityExpansionStore(Protocol):
    def entity_anchor_candidates(
        self,
        account_id: str,
        owner_spaces: Sequence[str],
    ) -> list[str]: ...

    def entity_anchor_nodes(
        self,
        account_id: str,
        anchor_uris: Sequence[str],
    ) -> list[dict]: ...
