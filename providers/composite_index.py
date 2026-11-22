"""Explicit composition of vector and lexical retrieval projections.

The ContextFS node and its durable outbox event remain the source of truth.
Both projections are idempotent consumers of the same ``IndexRecord``. A
partial failure retries the durable outbox event.
"""

from __future__ import annotations

import logging
import threading
import time

from core.interfaces import VectorIndex
from core.lexical import (
    HybridLexicalIndex,
    LexicalIndex,
    lexical_record_from_index_record,
    lexical_scopes_from_index_records,
)
from core.models import IndexRecord
from core.projection_ownership import check_projection_owner

logger = logging.getLogger(__name__)


def create_projection_writer(config, vector_index):
    if not config.retrieval_uses_lexical_projection:
        return vector_index

    from providers.lexical_index import create_lexical_index

    lexical = None
    try:
        lexical = create_lexical_index(config)
        if not isinstance(lexical, HybridLexicalIndex):
            raise TypeError("inverted index provider lacks durable checkpoint capabilities")
        writer = CompositeProjectionWriter(vector_index, lexical)
        writer.set_failure_callbacks(
            records_failed=lambda records: lexical.mark_scopes_dirty(lexical_scopes_from_index_records(records)),
            scope_unknown_failed=lexical.mark_all_scopes_dirty,
        )
        return writer
    except Exception:
        for resource in (lexical, vector_index):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.warning("Resource cleanup failed after lexical initialization error", exc_info=True)
        raise


class CompositeProjectionWriter:
    """Write independent projections from the same durable outbox event."""

    def __init__(
        self,
        vector: VectorIndex | None,
        lexical: LexicalIndex,
        projection_lock: threading.RLock | None = None,
    ) -> None:
        self.vector_index = vector
        self.lexical_index = lexical
        self._projection_lock = projection_lock or threading.RLock()
        self._closed = False
        self._on_records_failed = None
        self._on_scope_unknown_failed = None
        self._vector_close_event = threading.Event()
        if vector is None:
            self._vector_close_event.set()
        self._vector_close_thread: threading.Thread | None = None
        self._vector_close_result = True
        self._vector_close_error: Exception | None = None

    def set_failure_callbacks(self, *, records_failed=None, scope_unknown_failed=None) -> None:
        self._on_records_failed = records_failed
        self._on_scope_unknown_failed = scope_unknown_failed

    @staticmethod
    def _best_effort_callback(callback, *args) -> None:
        if not callable(callback):
            return
        try:
            callback(*args)
        except Exception:
            logger.warning("Projection invalidation callback failed", exc_info=True)

    def upsert(self, records: list[IndexRecord]) -> None:
        if not records:
            return
        with self._projection_lock:
            lexical_records = [
                lexical_record_from_index_record(record) for record in records
                if record.level in (0, 1, 2)
            ]
            try:
                if lexical_records:
                    self.lexical_index.upsert_many(lexical_records)
            except Exception:
                check_projection_owner()
                self._best_effort_callback(self._on_records_failed, records)
                raise
            if self.vector_index is not None:
                try:
                    self.vector_index.upsert(records)
                except Exception:
                    check_projection_owner()
                    self._best_effort_callback(self._on_records_failed, records)
                    self._best_effort_lexical_delete([record.id for record in records])
                    raise

    def _best_effort_lexical_delete(self, record_ids: list[str]) -> None:
        try:
            self.lexical_index.delete(record_ids)
        except Exception:
            self._best_effort_callback(self._on_scope_unknown_failed)

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._projection_lock:
            if self.vector_index is not None:
                self.vector_index.delete(ids)
            try:
                self.lexical_index.delete(ids)
            except Exception:
                self._best_effort_callback(self._on_scope_unknown_failed)
                raise

    def delete_account_data(self, account_id: str) -> int:
        with self._projection_lock:
            count = self.vector_index.delete_account_data(account_id) if self.vector_index is not None else None
            try:
                lexical_count = self.lexical_index.delete_scope(account_id=account_id)
            except Exception:
                self._best_effort_callback(self._on_scope_unknown_failed)
                raise
            return lexical_count if count is None else count

    def delete_by_owner_space(self, account_id: str, owner_space: str) -> int:
        with self._projection_lock:
            count = self.vector_index.delete_by_owner_space(account_id, owner_space) if self.vector_index is not None else None
            try:
                lexical_count = self.lexical_index.delete_scope(account_id=account_id, owner_space=owner_space)
            except Exception:
                self._best_effort_callback(self._on_scope_unknown_failed)
                raise
            return lexical_count if count is None else count

    def release_for_migration(self, timeout: float = 5.0) -> bool:
        """Release the projections and any migration-specific vector SDK cache."""
        return self._close(timeout, migration=True)

    def close(self, timeout: float = 5.0) -> bool:
        return self._close(timeout)

    def _close(self, timeout, *, migration=False):
        if self._closed:
            return True
        acquired = self._projection_lock.acquire(timeout=max(0.0, timeout))
        if not acquired:
            return False
        deadline = time.monotonic() + max(0.0, timeout)
        try:
            self._start_vector_close(migration=migration)
            remaining = max(0.0, deadline - time.monotonic())
            if not self._vector_close_event.wait(remaining):
                return False
            if self._vector_close_error is not None:
                raise self._vector_close_error
            lexical_ok = self.lexical_index.close(
                timeout=max(0.0, deadline - time.monotonic())
            )
            self._closed = (
                self._vector_close_result is not False
                and lexical_ok is not False
            )
            return self._closed
        finally:
            self._projection_lock.release()

    def _start_vector_close(self, *, migration=False) -> None:
        if self.vector_index is None or self._vector_close_thread is not None:
            return

        def close_vector() -> None:
            try:
                close = getattr(self.vector_index, "release_for_migration", None) if migration else None
                close = close or getattr(self.vector_index, "close", None)
                if callable(close):
                    self._vector_close_result = close()
            except Exception as exc:
                self._vector_close_error = exc
                self._vector_close_result = False
            finally:
                self._vector_close_event.set()

        self._vector_close_thread = threading.Thread(
            target=close_vector,
            name="vector-index-close",
            daemon=True,
        )
        self._vector_close_thread.start()
