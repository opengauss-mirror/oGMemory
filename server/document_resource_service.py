"""Knowledge document operations and provider reference management."""

import logging
from dataclasses import replace
from uuid import UUID, uuid4

from core.document_backend import (
    DocumentBackend,
    DocumentBackendError,
    DocumentCleanupPendingError,
    DocumentIndexStatus,
    DocumentRef,
)
from core.resource_models import DocumentResource, ResourceChunk
from server.document_resource_store import (
    DocumentBinding,
    DocumentResourceCommitError,
    DocumentResourceStoreError,
    SQLDocumentResourceStore,
)

logger = logging.getLogger(__name__)


class DocumentResourceService:
    def __init__(
        self,
        store: SQLDocumentResourceStore,
        backend: DocumentBackend,
        account_id: str,
    ) -> None:
        self._store = store
        self._backend = backend
        self._account_id = account_id

    def ingest(self, filename: str, media_type: str, content: bytes) -> DocumentResource:
        """Publish a document supplied by a trusted caller in this account."""
        self._require_distinct_uploads()
        resource = self._resource(uuid4(), filename, media_type, content)
        self._record_upload(self._ingest(resource, content))
        return resource

    def replace(
        self,
        resource_id: UUID,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> tuple[DocumentResource, bool]:
        current = self._active(resource_id)
        self._require_distinct_uploads()
        resource = self._resource(resource_id, filename, media_type, content)
        self._clear_retired(current)
        uploaded = self._ingest(resource, content)
        if not self._record_upload(uploaded, current=current):
            self._discard(uploaded)
            raise FileExistsError("document resource changed during replacement")
        try:
            self._clear_retired(DocumentBinding(uploaded, current.ref.object_id))
        except (DocumentBackendError, DocumentResourceStoreError):
            logger.warning("document resource cleanup pending resource_id=%s", resource_id)
            return resource, True
        return resource, False

    def delete(self, resource_id: UUID) -> None:
        binding = self._binding(resource_id)
        if binding is None:
            return
        if not binding.deleted and not self._store.mark_deleted(binding):
            raise FileExistsError("document resource changed during deletion")
        self._clear_retired(binding)
        self._backend.delete(binding.ref)
        self._store.remove(binding)

    def retry_pending_uploads(self) -> None:
        self._store.retry_pending_uploads(
            self._account_id, self._backend.name, self._backend.namespace, self._delete_upload
        )

    def status(self, resource_id: UUID) -> tuple[DocumentResource, DocumentIndexStatus]:
        ref = self._active(resource_id).ref
        return ref.resource, self._backend.status(ref)

    def download(self, resource_id: UUID) -> tuple[DocumentResource, bytes]:
        ref = self._active(resource_id).ref
        return ref.resource, self._backend.download(ref)

    def list_documents(self, *, page: int, page_size: int) -> tuple[list[DocumentResource], int]:
        return self._store.list_page(
            self._account_id,
            self._backend.name,
            self._backend.namespace,
            limit=page_size,
            offset=(page - 1) * page_size,
        )

    def search(self, query: str, *, limit: int) -> list[ResourceChunk]:
        refs = self._store.list_active(
            self._account_id, self._backend.name, self._backend.namespace
        )
        if not refs:
            return []
        return self._backend.search(query, refs, limit=limit)

    def _active(self, resource_id: UUID) -> DocumentBinding:
        binding = self._binding(resource_id)
        if binding is None or binding.deleted:
            raise FileNotFoundError("document resource not found")
        return binding

    def _binding(self, resource_id: UUID) -> DocumentBinding | None:
        binding = self._store.get(self._account_id, resource_id)
        if binding is not None and (binding.ref.backend, binding.ref.namespace) != (
            self._backend.name,
            self._backend.namespace,
        ):
            return None
        return binding

    def _resource(
        self, resource_id: UUID, filename: str, media_type: str, content: bytes
    ) -> DocumentResource:
        if not content:
            raise ValueError("document content is required")
        if len(filename) > 255 or len(media_type) > 255:
            raise ValueError("filename and media type must be at most 255 characters")
        return DocumentResource(self._account_id, resource_id, filename, media_type)

    def _require_distinct_uploads(self) -> None:
        if not getattr(self._backend, "creates_distinct_objects", False):
            raise ValueError("knowledge backend must create independent document objects")

    def _ingest(self, resource: DocumentResource, content: bytes) -> DocumentRef:
        try:
            return self._backend.ingest(resource, content)
        except DocumentCleanupPendingError as exc:
            ref = exc.ref
            if (ref.resource, ref.backend, ref.namespace) != (
                resource, self._backend.name, self._backend.namespace
            ):
                raise
            try:
                self._store.track_pending_upload(ref)
            except DocumentResourceStoreError as store_error:
                raise exc from store_error
            raise

    def _record_upload(self, ref: DocumentRef, *, current: DocumentBinding | None = None) -> bool:
        try:
            if current is not None:
                return self._store.replace(current, ref)
            self._store.add(ref)
            return True
        except DocumentResourceCommitError:
            try:
                binding = self._binding(ref.resource.resource_id)
            except DocumentResourceStoreError:
                binding = None
            if binding is not None and binding.ref == ref and not binding.deleted:
                return True
            logger.error(
                "document resource reference write uncertain resource_id=%s backend=%s "
                "namespace=%s object_id=%s",
                ref.resource.resource_id,
                ref.backend,
                ref.namespace,
                ref.object_id,
            )
            raise
        except DocumentResourceStoreError:
            self._discard(ref)
            raise

    def _clear_retired(self, binding: DocumentBinding) -> None:
        if binding.retired_object_id is not None:
            self._backend.delete(replace(binding.ref, object_id=binding.retired_object_id))
            self._store.clear_retired(binding)

    def _discard(self, ref: DocumentRef) -> None:
        try:
            self._store.discard_pending_upload(ref, self._delete_upload)
        except DocumentResourceStoreError as exc:
            raise DocumentCleanupPendingError(ref) from exc

    def _delete_upload(self, ref: DocumentRef) -> bool:
        try:
            self._backend.delete(ref)
        except DocumentBackendError:
            logger.warning(
                "document resource cleanup failed resource_id=%s backend=%s namespace=%s object_id=%s",
                ref.resource.resource_id,
                ref.backend,
                ref.namespace,
                ref.object_id,
            )
            return False
        return True
