"""Translate the knowledge backend contract to native RAGFlow operations."""

import logging
from collections.abc import Sequence
from typing import Literal

from core.document_backend import (
    DocumentBackendError,
    DocumentCleanupPendingError,
    DocumentIndexStatus,
    DocumentRef,
)
from core.resource_models import DocumentResource, ResourceChunk
from providers.resource_store.ragflow import (
    RagFlowDocumentNotFoundError,
    RagFlowDocumentRef,
    RagFlowError,
    RagFlowResourceAdapter,
)

logger = logging.getLogger(__name__)


class RagFlowDocumentBackend:
    name = "external/ragflow"
    creates_distinct_objects = True

    def __init__(self, adapter: RagFlowResourceAdapter, namespace: str) -> None:
        self._adapter = adapter
        self.namespace = namespace

    def ingest(self, resource: DocumentResource, content: bytes) -> DocumentRef:
        try:
            ref = self._adapter.upload(resource, content, dataset_id=self.namespace)
        except RagFlowError as exc:
            raise DocumentBackendError("document upload failed") from exc
        result = DocumentRef(resource, self.name, self.namespace, ref.document_id)
        try:
            self._adapter.start_indexing(ref)
        except RagFlowError as exc:
            try:
                self.delete(result)
            except DocumentBackendError as cleanup_error:
                logger.warning(
                    "RAGFlow cleanup pending account_id=%s resource_id=%s dataset_id=%s document_id=%s",
                    resource.account_id,
                    resource.resource_id,
                    result.namespace,
                    result.object_id,
                )
                raise DocumentCleanupPendingError(result) from cleanup_error
            raise DocumentBackendError("document indexing submission failed") from exc
        return result

    def status(self, ref: DocumentRef) -> DocumentIndexStatus:
        try:
            status = self._adapter.get_status(self._native_ref(ref))
        except RagFlowDocumentNotFoundError:
            raise FileNotFoundError("document content not found") from None
        except RagFlowError as exc:
            raise DocumentBackendError("document status unavailable") from exc
        state: Literal["pending", "indexing", "ready", "failed", "canceled", "unknown"]
        match status.ingestion_status.upper():
            case "UNSTART":
                state = "pending"
            case "RUNNING" | "SCHEDULED":
                state = "indexing"
            case "COMPLETED":
                state = "ready"
            case "FAILED":
                state = "failed"
            case "STOPPED" | "CANCELED":
                state = "canceled"
            case _:
                state = "unknown"
        return DocumentIndexStatus(state, status.chunk_count)

    def download(self, ref: DocumentRef) -> bytes:
        try:
            return self._adapter.download(self._native_ref(ref))
        except RagFlowError as exc:
            raise DocumentBackendError("document download failed") from exc

    def delete(self, ref: DocumentRef) -> None:
        native = self._native_ref(ref)
        try:
            if self._adapter.delete(native) == 1:
                return
        except RagFlowError:
            pass
        try:
            self._adapter.get_status(native)
        except RagFlowDocumentNotFoundError:
            return
        except RagFlowError as exc:
            raise DocumentBackendError("document deletion could not be confirmed") from exc
        raise DocumentBackendError("document deletion could not be confirmed")

    def search(self, query: str, refs: Sequence[DocumentRef], *, limit: int) -> list[ResourceChunk]:
        try:
            return self._adapter.search(query, [self._native_ref(ref) for ref in refs], limit=limit)
        except RagFlowError as exc:
            raise DocumentBackendError("document search failed") from exc

    def _native_ref(self, ref: DocumentRef) -> RagFlowDocumentRef:
        if (ref.backend, ref.namespace) != (self.name, self.namespace):
            raise ValueError("document reference belongs to another backend namespace")
        return RagFlowDocumentRef(ref.resource, ref.namespace, ref.object_id)
