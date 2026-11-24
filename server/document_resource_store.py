"""Persist the account-scoped document references used by document resources."""

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any
from uuid import UUID

from core.document_backend import DocumentRef
from core.knowledge_base import KnowledgeBase
from core.resource_models import DocumentResource
from fs.sql_adapter.pool import SharedConnectionPool
from server.knowledge_base_repository import KnowledgeBaseRepository, knowledge_base_lock_id

_SCHEMA_LOCK = int.from_bytes(b"ogmdocrs", "big", signed=True)
_CLEANUP_SLOT = Lock()


class DocumentResourceStoreError(RuntimeError):
    """The document reference store could not complete an operation."""


class DocumentResourceCommitError(DocumentResourceStoreError):
    """The store could not determine the outcome of a commit."""


@dataclass(frozen=True)
class DocumentBinding:
    ref: DocumentRef
    retired_object_id: str | None = None
    deleted: bool = False


class SQLDocumentResourceStore:
    def __init__(self, pool: SharedConnectionPool) -> None:
        self._pool = pool
        self._locked_connection: ContextVar[Any] = ContextVar(
            "knowledge_base_connection", default=None
        )
        self.catalog = KnowledgeBaseRepository(self._transaction)
        with self._transaction() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK,))
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_resource_bindings (
                    account_id VARCHAR(128) NOT NULL,
                    resource_id VARCHAR(36) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    media_type VARCHAR(255) NOT NULL,
                    backend VARCHAR(64) NOT NULL DEFAULT 'external/ragflow',
                    dataset_id VARCHAR(128) NOT NULL,
                    document_id VARCHAR(128) NOT NULL,
                    retired_document_id VARCHAR(128),
                    deleted BOOLEAN NOT NULL DEFAULT FALSE,
                    PRIMARY KEY (account_id, resource_id)
                )
            """)
            cur.execute(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = current_schema()
                    AND table_name = 'document_resource_bindings' AND column_name = 'backend'
                """
            )
            if cur.fetchone() is None:
                cur.execute(
                    "ALTER TABLE document_resource_bindings ADD COLUMN "
                    "backend VARCHAR(64) NOT NULL DEFAULT 'external/ragflow'"
                )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_resource_pending_uploads (
                    account_id VARCHAR(128) NOT NULL,
                    resource_id VARCHAR(36) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    media_type VARCHAR(255) NOT NULL,
                    backend VARCHAR(64) NOT NULL,
                    dataset_id VARCHAR(128) NOT NULL,
                    document_id VARCHAR(128) NOT NULL,
                    PRIMARY KEY (account_id, resource_id, backend, dataset_id, document_id)
                )
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_resource_provider_id "
                "ON document_resource_bindings (account_id, backend, dataset_id, document_id)"
            )
            self.catalog.ensure_schema(cur)

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        conn = self._locked_connection.get()
        if conn is not None:
            yield conn
            return
        try:
            conn = self._pool.get_connection()
        except Exception:
            raise DocumentResourceStoreError("document resource store is unavailable") from None
        try:
            yield conn
        finally:
            self._pool.return_connection(conn)

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        with self._connection() as conn:
            committing = False
            try:
                with conn.cursor() as cur:
                    yield cur
                committing = True
                conn.commit()
            except BaseException as exc:
                try:
                    conn.rollback()
                except Exception:
                    conn.close()
                if isinstance(exc, Exception) and not isinstance(
                    exc, (ValueError, FileExistsError)
                ):
                    if committing:
                        raise DocumentResourceCommitError("document resource commit unconfirmed") from None
                    raise DocumentResourceStoreError("document resource store operation failed") from None
                raise

    @contextmanager
    def lock_knowledge_base(self, base: KnowledgeBase) -> Iterator[None]:
        with self._session_lock(knowledge_base_lock_id(base)):
            yield

    @contextmanager
    def _session_lock(self, key: int) -> Iterator[None]:
        with self._connection() as conn:
            token = self._locked_connection.set(conn)
            locked = False
            try:
                with self._transaction() as cur:
                    cur.execute("SELECT pg_advisory_lock(%s)", (key,))
                    locked = True
                yield
            finally:
                try:
                    if locked and not conn.closed:
                        with self._transaction() as cur:
                            cur.execute("SELECT pg_advisory_unlock(%s)", (key,))
                except DocumentResourceStoreError:
                    conn.close()
                finally:
                    self._locked_connection.reset(token)

    @contextmanager
    def _publication(self, ref: DocumentRef) -> Iterator[Any]:
        account_id, _, backend, namespace, _ = _identity(ref)
        publishing = False
        try:
            with self._session_lock(_upload_lock_id(account_id, backend, namespace)):
                self._track_pending_upload(ref, publish=True)
                publishing = True
                with self._transaction() as cur:
                    yield cur
        except DocumentResourceCommitError as exc:
            if publishing:
                raise
            raise DocumentResourceStoreError("document upload registration unconfirmed") from exc

    def add(self, ref: DocumentRef) -> None:
        resource = ref.resource
        with self._publication(ref) as cur:
            cur.execute(
                """
                INSERT INTO document_resource_bindings
                    (account_id, resource_id, filename, media_type, backend, dataset_id, document_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    resource.account_id,
                    str(resource.resource_id),
                    resource.filename,
                    resource.media_type,
                    ref.backend,
                    ref.namespace,
                    ref.object_id,
                ),
            )
            self._forget_pending_upload(cur, ref)

    def get(self, account_id: str, resource_id: UUID) -> DocumentBinding | None:
        with self._transaction() as cur:
            cur.execute(
                """
                SELECT filename, media_type, backend, dataset_id, document_id, retired_document_id, deleted
                FROM document_resource_bindings
                WHERE account_id = %s AND resource_id = %s
                """,
                (account_id, str(resource_id)),
            )
            row = cur.fetchone()
        if row is None:
            return None
        filename, media_type, backend, namespace, object_id, retired_object_id, deleted = row
        resource = DocumentResource(account_id, resource_id, filename, media_type)
        ref = DocumentRef(resource, backend, namespace, object_id)
        return DocumentBinding(ref, retired_object_id, deleted)

    def list_active(
        self, account_id: str, backend: str, namespace: str, *, object_ids: list[str] | None = None
    ) -> list[DocumentRef]:
        with self._transaction() as cur:
            cur.execute(
                """
                SELECT resource_id, filename, media_type, document_id
                FROM document_resource_bindings
                WHERE account_id = %s AND backend = %s AND dataset_id = %s AND NOT deleted
                    AND (%s OR document_id = ANY(%s))
                ORDER BY resource_id
                """,
                (account_id, backend, namespace, object_ids is None, object_ids or []),
            )
            rows = cur.fetchall()
        return [
            DocumentRef(
                DocumentResource(account_id, UUID(resource_id), filename, media_type),
                backend,
                namespace,
                object_id,
            )
            for resource_id, filename, media_type, object_id in rows
        ]

    def list_page(
        self, account_id: str, backend: str, namespace: str, *, limit: int, offset: int
    ) -> tuple[list[DocumentResource], int]:
        with self._transaction() as cur:
            cur.execute(
                """
                WITH scoped_docs AS (
                    SELECT resource_id, filename, media_type
                    FROM document_resource_bindings
                    WHERE account_id = %s AND backend = %s AND dataset_id = %s AND NOT deleted
                )
                SELECT counts.total, page.resource_id, page.filename, page.media_type
                FROM (SELECT COUNT(*) AS total FROM scoped_docs) counts
                LEFT JOIN (
                    SELECT resource_id, filename, media_type FROM scoped_docs
                    ORDER BY resource_id LIMIT %s OFFSET %s
                ) page ON TRUE
                ORDER BY page.resource_id
                """,
                (account_id, backend, namespace, limit, offset),
            )
            rows = cur.fetchall()
        return (
            [
                DocumentResource(account_id, UUID(resource_id), filename, media_type)
                for _, resource_id, filename, media_type in rows
                if resource_id is not None
            ],
            int(rows[0][0]),
        )

    def has_documents(self, account_id: str, backend: str, namespace: str) -> bool:
        with self._transaction() as cur:
            cur.execute(
                "SELECT 1 FROM document_resource_bindings "
                "WHERE account_id = %s AND backend = %s AND dataset_id = %s LIMIT 1",
                (account_id, backend, namespace),
            )
            return cur.fetchone() is not None

    def replace(self, current: DocumentBinding, updated: DocumentRef) -> bool:
        expected = _identity(current.ref)
        if expected[:-1] != _identity(updated)[:-1] or updated.object_id == current.ref.object_id:
            raise ValueError(
                "replacement must bind a new object in the resource's backend namespace"
            )
        with self._publication(updated) as cur:
            cur.execute(
                """
                UPDATE document_resource_bindings
                SET filename = %s, media_type = %s, retired_document_id = document_id,
                    document_id = %s
                WHERE account_id = %s AND resource_id = %s AND backend = %s AND dataset_id = %s
                    AND document_id = %s AND NOT deleted AND retired_document_id IS NULL
                """,
                (
                    updated.resource.filename,
                    updated.resource.media_type,
                    updated.object_id,
                    *expected,
                ),
            )
            changed = cur.rowcount == 1
            if changed:
                self._forget_pending_upload(cur, updated)
            return changed

    def track_pending_upload(self, ref: DocumentRef) -> None:
        account_id, _, backend, namespace, _ = _identity(ref)
        with self._session_lock(_upload_lock_id(account_id, backend, namespace)):
            self._track_pending_upload(ref)

    def _track_pending_upload(self, ref: DocumentRef, *, publish: bool = False) -> None:
        identity = _identity(ref)
        with self._transaction() as cur:
            if self._object_is_bound(cur, ref):
                raise FileExistsError("knowledge backend object is already bound")
            cur.execute(
                "INSERT INTO document_resource_pending_uploads "
                "(account_id, resource_id, backend, dataset_id, document_id, filename, media_type) "
                "SELECT %s, %s, %s, %s, %s, %s, %s "
                "WHERE NOT EXISTS (SELECT 1 FROM document_resource_pending_uploads "
                "WHERE account_id = %s AND backend = %s AND dataset_id = %s AND document_id = %s)",
                (*identity, ref.resource.filename, ref.resource.media_type, identity[0], *identity[2:]),
            )
            if publish and cur.rowcount != 1:
                raise FileExistsError("knowledge backend object is pending cleanup")

    def forget_pending_upload(self, ref: DocumentRef) -> None:
        with self._transaction() as cur:
            self._forget_pending_upload(cur, ref)

    def _forget_pending_upload(self, cur: Any, ref: DocumentRef) -> None:
        cur.execute(
            "DELETE FROM document_resource_pending_uploads "
            "WHERE account_id = %s AND resource_id = %s AND backend = %s "
            "AND dataset_id = %s AND document_id = %s",
            _identity(ref),
        )

    def retry_pending_uploads(
        self, account_id: str, backend: str, namespace: str, discard: Callable[[DocumentRef], bool]
    ) -> None:
        with self._transaction() as cur:
            cur.execute(
                "SELECT resource_id, filename, media_type, document_id "
                "FROM document_resource_pending_uploads "
                "WHERE account_id = %s AND backend = %s AND dataset_id = %s",
                (account_id, backend, namespace),
            )
            rows = cur.fetchall()
        for resource_id, filename, media_type, object_id in rows:
            resource = DocumentResource(account_id, UUID(resource_id), filename, media_type)
            self.discard_pending_upload(DocumentRef(resource, backend, namespace, object_id), discard)

    def discard_pending_upload(self, ref: DocumentRef, discard: Callable[[DocumentRef], bool]) -> None:
        try:
            self.track_pending_upload(ref)
        except FileExistsError:
            self.forget_pending_upload(ref)
            return
        if _CLEANUP_SLOT.acquire(blocking=False):
            try:
                if discard(ref):
                    self.forget_pending_upload(ref)
            finally:
                _CLEANUP_SLOT.release()

    def _object_is_bound(self, cur: Any, ref: DocumentRef) -> bool:
        account_id, _, backend, namespace, object_id = _identity(ref)
        cur.execute(
            "SELECT 1 FROM document_resource_bindings "
            "WHERE account_id = %s AND backend = %s AND dataset_id = %s "
            "AND (document_id = %s OR retired_document_id = %s) LIMIT 1",
            (account_id, backend, namespace, object_id, object_id),
        )
        return cur.fetchone() is not None

    def clear_retired(self, binding: DocumentBinding) -> None:
        with self._transaction() as cur:
            cur.execute(
                """
                UPDATE document_resource_bindings SET retired_document_id = NULL
                WHERE account_id = %s AND resource_id = %s AND backend = %s AND dataset_id = %s
                    AND document_id = %s AND retired_document_id = %s
                """,
                (*_identity(binding.ref), binding.retired_object_id),
            )

    def mark_deleted(self, binding: DocumentBinding) -> bool:
        with self._transaction() as cur:
            cur.execute(
                """
                UPDATE document_resource_bindings SET deleted = TRUE
                WHERE account_id = %s AND resource_id = %s AND backend = %s AND dataset_id = %s
                    AND document_id = %s AND NOT deleted
                """,
                _identity(binding.ref),
            )
            return bool(cur.rowcount == 1)

    def remove(self, binding: DocumentBinding) -> None:
        with self._transaction() as cur:
            cur.execute(
                """
                DELETE FROM document_resource_bindings
                WHERE account_id = %s AND resource_id = %s AND backend = %s AND dataset_id = %s
                    AND document_id = %s AND deleted
                """,
                _identity(binding.ref),
            )


def _upload_lock_id(account_id: str, backend: str, namespace: str) -> int:
    scope = f"ogmemory-document-upload:{account_id}:{backend}:{namespace}"
    digest = hashlib.sha256(scope.encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def _identity(ref: DocumentRef) -> tuple[str, str, str, str, str]:
    return (
        ref.resource.account_id,
        str(ref.resource.resource_id),
        ref.backend,
        ref.namespace,
        ref.object_id,
    )
