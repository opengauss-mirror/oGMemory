"""Storage contract for knowledge documents."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from core.resource_models import DocumentResource, ResourceChunk


class DocumentBackendError(RuntimeError):
    """The selected knowledge backend could not complete an operation."""


@dataclass(frozen=True)
class DocumentRef:
    resource: DocumentResource
    backend: str
    namespace: str
    object_id: str


class DocumentCleanupPendingError(DocumentBackendError):
    """A caller can retry cleanup of an uploaded document using its reference."""

    def __init__(self, ref: DocumentRef) -> None:
        super().__init__("uploaded document cleanup could not be confirmed")
        self.ref = ref


@dataclass(frozen=True)
class DocumentIndexStatus:
    state: Literal["pending", "indexing", "ready", "failed", "canceled", "unknown"]
    chunk_count: int


class DocumentBackend(Protocol):
    """An adapter scoped to one backend namespace."""

    @property
    def name(self) -> str: ...

    @property
    def namespace(self) -> str: ...

    @property
    def creates_distinct_objects(self) -> bool:
        """Ingest allocates a fresh object ID for each upload and preserves published content."""
        ...

    def ingest(self, resource: DocumentResource, content: bytes) -> DocumentRef: ...

    def status(self, ref: DocumentRef) -> DocumentIndexStatus: ...

    def download(self, ref: DocumentRef) -> bytes: ...

    def delete(self, ref: DocumentRef) -> None: ...

    def search(
        self, query: str, refs: Sequence[DocumentRef], *, limit: int
    ) -> list[ResourceChunk]: ...
