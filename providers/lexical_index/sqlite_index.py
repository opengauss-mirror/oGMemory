"""SQLite FTS5 projection of the canonical records."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Any

from core.default_lexical_analyzer import DefaultLexicalAnalyzer
from core.errors import OperationDeadlineExceededError
from core.lexical import DEFAULT_MAX_QUERY_TERMS, LexicalHit, LexicalRecord
from core.memory_names import memory_directory
from core.operation_deadline import check_deadline, operation_deadline, remaining_seconds
from core.projection_ownership import check_projection_owner

logger = logging.getLogger(__name__)

_BUSY_TIMEOUT_MS = 5000
_REBUILD_LEASE_SECONDS = 3600
_REBUILD_BATCH_SIZE = 256
_DOCUMENT_COLUMNS = (
    "record_id,account_id,owner_space,context_type,category,level,memory_directory,metadata,terms"
)
_UPSERT = f"""
    INSERT INTO documents({_DOCUMENT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(record_id) DO UPDATE SET account_id=excluded.account_id,
        owner_space=excluded.owner_space,context_type=excluded.context_type,
        category=excluded.category,level=excluded.level,memory_directory=excluded.memory_directory,
        metadata=excluded.metadata,terms=excluded.terms
"""
_SCHEMA = """
    CREATE TABLE IF NOT EXISTS projection_state (
        id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL,
        all_scopes_dirty INTEGER NOT NULL, records_deleted INTEGER NOT NULL,
        schema_identity TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS account_generations (
        account_id TEXT PRIMARY KEY, generation INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS scope_checkpoints (
        account_id TEXT, owner_space TEXT, rebuilt INTEGER NOT NULL, dirty INTEGER NOT NULL,
        PRIMARY KEY(account_id, owner_space)
    );
    CREATE TABLE IF NOT EXISTS documents (
        row_id INTEGER PRIMARY KEY, record_id TEXT NOT NULL UNIQUE,
        account_id TEXT NOT NULL, owner_space TEXT NOT NULL, context_type TEXT, category TEXT,
        level INTEGER NOT NULL, memory_directory TEXT, metadata TEXT NOT NULL, terms TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS document_scope ON documents(account_id,owner_space);
    CREATE VIRTUAL TABLE IF NOT EXISTS postings USING fts5(
        terms, content='documents', content_rowid='row_id'
    );
    CREATE TRIGGER IF NOT EXISTS document_insert AFTER INSERT ON documents BEGIN
        INSERT INTO postings(rowid,terms) VALUES (new.row_id,new.terms);
    END;
    CREATE TRIGGER IF NOT EXISTS document_delete AFTER DELETE ON documents BEGIN
        INSERT INTO postings(postings,rowid,terms) VALUES ('delete',old.row_id,old.terms);
    END;
    CREATE TRIGGER IF NOT EXISTS document_update AFTER UPDATE ON documents BEGIN
        INSERT INTO postings(postings,rowid,terms) VALUES ('delete',old.row_id,old.terms);
        INSERT INTO postings(rowid,terms) VALUES (new.row_id,new.terms);
    END;
    CREATE TABLE IF NOT EXISTS rebuilds (
        id TEXT PRIMARY KEY, generation INTEGER NOT NULL, expires_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS staged_documents (
        rebuild_id TEXT NOT NULL REFERENCES rebuilds(id) ON DELETE CASCADE,
        record_id TEXT NOT NULL, account_id TEXT NOT NULL, owner_space TEXT NOT NULL,
        context_type TEXT, category TEXT, level INTEGER NOT NULL, memory_directory TEXT,
        metadata TEXT NOT NULL, terms TEXT NOT NULL, PRIMARY KEY(rebuild_id,record_id)
    );
"""


class SQLiteLexicalIndex:
    """Shared local BM25 index with transactional projection checkpoints."""

    _migration_runtime_only = True
    _migration_reads = ("health", "ready_scopes", "search",)

    supported_filters = frozenset(
        {
            "account_id",
            "owner_space",
            "context_type",
            "category",
            "level",
            "memory_directory",
        }
    )

    def __init__(
        self,
        path: str,
        *,
        max_query_terms: int = DEFAULT_MAX_QUERY_TERMS,
        max_query_chars: int = 32768,
        max_document_chars: int = 262144,
        query_timeout_ms: int = 5000,
    ) -> None:
        if not path or path == ":memory:":
            raise ValueError("SQLite lexical index requires a shared database file path")
        for name, value, limit in (
            ("max_query_terms", max_query_terms, 256),
            ("max_query_chars", max_query_chars, 262144),
            ("max_document_chars", max_document_chars, 1048576),
            ("query_timeout_ms", query_timeout_ms, 300000),
        ):
            if not 1 <= value <= limit:
                raise ValueError(f"{name} must be between 1 and {limit}")
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._analyzer = DefaultLexicalAnalyzer()
        self._max_query_terms = max_query_terms
        self._max_query_chars = max_query_chars
        self._max_document_chars = max_document_chars
        self._query_timeout_seconds = query_timeout_ms / 1000
        self._lock = threading.Lock()
        self._closed = False
        self._db = sqlite3.connect(
            self._path, timeout=_BUSY_TIMEOUT_MS / 1000, check_same_thread=False
        )
        try:
            self._enable_wal()
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript("BEGIN IMMEDIATE; " + _SCHEMA)
            identity = f"fts5-v1:{self._analyzer.identity}:chars={max_document_chars}"
            with self._db as db:
                db.execute("INSERT OR IGNORE INTO projection_state VALUES (1,0,0,0,?)", (identity,))
                stored = db.execute(
                    "SELECT schema_identity FROM projection_state WHERE id=1"
                ).fetchone()[0]
                if stored != identity:
                    raise RuntimeError(
                        "SQLite lexical index schema changed; configure a new projection path"
                    )
        except Exception:
            self._db.close()
            raise

    def _enable_wal(self) -> None:
        deadline = time.monotonic() + _BUSY_TIMEOUT_MS / 1000
        while True:
            try:
                mode = self._db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
                if mode != "wal":
                    raise RuntimeError("SQLite lexical index requires WAL journal mode")
                return
            except sqlite3.OperationalError as exc:
                remaining = deadline - time.monotonic()
                code = getattr(exc, "sqlite_errorcode", 0) & 0xFF
                if remaining <= 0 or code not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                    raise
                time.sleep(min(0.01, remaining))

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._db as db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    @staticmethod
    def _next_generation(db: sqlite3.Connection) -> int:
        db.execute("UPDATE projection_state SET generation=generation+1 WHERE id=1")
        return int(db.execute("SELECT generation FROM projection_state WHERE id=1").fetchone()[0])

    @staticmethod
    def _touch_accounts(db: sqlite3.Connection, accounts: Iterable[str], generation: int) -> None:
        db.executemany(
            "INSERT INTO account_generations VALUES (?,?) "
            "ON CONFLICT(account_id) DO UPDATE SET generation=excluded.generation",
            ((account, generation) for account in dict.fromkeys(accounts)),
        )

    def _document(self, record: LexicalRecord) -> tuple:
        if not record.account_id or not record.owner_space:
            raise ValueError("lexical records require account_id and owner_space")
        metadata = dict(record.metadata or {})
        metadata.update(
            uri=record.uri,
            level=record.level,
            account_id=record.account_id,
            owner_space=record.owner_space,
        )
        terms = self._analyzer.analyze_document((record.text or "")[: self._max_document_chars])
        return (
            record.id,
            record.account_id,
            record.owner_space,
            metadata.get("context_type"),
            metadata.get("category"),
            record.level,
            memory_directory(record.uri),
            json.dumps(metadata, ensure_ascii=False),
            " ".join(terms),
        )

    def upsert_many(self, records: Sequence[LexicalRecord]) -> None:
        if records:
            documents = [self._document(record) for record in records]
            with self._write() as db:
                check_projection_owner()
                self._touch_accounts(db, (r.account_id for r in records), self._next_generation(db))
                db.executemany(_UPSERT, documents)

    def delete(self, record_ids: Sequence[str]) -> None:
        if record_ids:
            with self._write() as db:
                check_projection_owner()
                generation = self._next_generation(db)
                db.execute(
                    "UPDATE projection_state SET records_deleted=? WHERE id=1", (generation,)
                )
                db.executemany(
                    "DELETE FROM documents WHERE record_id=?", ((key,) for key in record_ids)
                )

    def delete_scope(self, *, account_id: str, owner_space: str | None = None) -> int:
        if not account_id:
            raise ValueError("account_id is required")
        filters = {"account_id": account_id}
        if owner_space is not None:
            filters["owner_space"] = owner_space
        where, args = self._filters(filters)
        with self._write() as db:
            check_projection_owner()
            self._touch_accounts(db, (account_id,), self._next_generation(db))
            db.execute(f"DELETE FROM scope_checkpoints AS d WHERE {where}", args)
            return db.execute(f"DELETE FROM documents AS d WHERE {where}", args).rowcount

    @staticmethod
    def _filters(filters: dict[str, Any]) -> tuple[str, tuple]:
        clauses = []
        args: list[Any] = []
        for name, value in filters.items():
            values = value if isinstance(value, (tuple, list, set, frozenset)) else (value,)
            values = tuple(item for item in values if item is not None)
            clauses.append(f"d.{name} IN ({','.join('?' for _ in values)})" if values else "1=0")
            args.extend(values)
        return " AND ".join(clauses), tuple(args)

    def search(
        self, query: str, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[LexicalHit]:
        filters = dict(filters or {})
        unknown = set(filters) - self.supported_filters
        if unknown:
            raise ValueError(f"unsupported lexical filters: {sorted(unknown)}")
        if not isinstance(filters.get("account_id"), str) or not filters["account_id"]:
            raise ValueError("lexical search requires an account_id filter")
        if top_k < 1:
            return []
        with operation_deadline(self._query_timeout_seconds):
            terms = dict.fromkeys(self._analyzer.analyze_query(query[: self._max_query_chars]))
            expression = " OR ".join(f'"{term}"' for term in islice(terms, self._max_query_terms))
            if not expression:
                return []
            where, args = self._filters(filters)
            if not self._lock.acquire(timeout=remaining_seconds()):
                raise OperationDeadlineExceededError("Operation deadline exceeded")
            try:
                milliseconds = max(1, int(remaining_seconds() * 1000))
                self._db.execute(f"PRAGMA busy_timeout={milliseconds}")
                expires_at = time.monotonic() + milliseconds / 1000
                self._db.set_progress_handler(lambda: int(time.monotonic() >= expires_at), 1000)
                rows = self._db.execute(
                    "SELECT d.record_id,d.metadata,rank FROM postings "
                    "JOIN documents d ON d.row_id=postings.rowid "
                    f"WHERE postings MATCH ? AND {where} ORDER BY rank,d.row_id LIMIT ?",
                    (expression, *args, top_k),
                ).fetchall()
                check_deadline()
            except sqlite3.OperationalError:
                check_deadline()
                raise
            finally:
                try:
                    self._db.set_progress_handler(None, 0)
                    self._db.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
                finally:
                    self._lock.release()
        return [
            LexicalHit(record_id, -score, json.loads(metadata))
            for record_id, metadata, score in rows
        ]

    def begin_rebuild(self) -> str:
        rebuild_id = uuid.uuid4().hex
        now = time.time()
        with self._write() as db:
            db.execute("DELETE FROM rebuilds WHERE expires_at<=?", (now,))
            db.execute(
                "INSERT INTO rebuilds VALUES (?,?,?)",
                (rebuild_id, self._next_generation(db), now + _REBUILD_LEASE_SECONDS),
            )
        return rebuild_id

    def abort_rebuild(self, rebuild_id: str) -> None:
        with self._write() as db:
            db.execute("DELETE FROM rebuilds WHERE id=?", (rebuild_id,))

    def replace_scopes(
        self,
        *,
        rebuild_id: str,
        account_id: str,
        owner_spaces: Sequence[str],
        records: Iterable[LexicalRecord],
    ) -> bool:
        scopes = tuple(dict.fromkeys(scope for scope in owner_spaces if scope))
        if not account_id or not scopes:
            raise ValueError("account_id and owner_spaces are required")
        scope_set = frozenset(scopes)

        def staged_records():
            for record in records:
                if record.account_id != account_id or record.owner_space not in scope_set:
                    raise ValueError("replacement records must belong to the requested scopes")
                yield (rebuild_id, *self._document(record))

        staged = staged_records()
        try:
            while batch := list(islice(staged, _REBUILD_BATCH_SIZE)):
                with self._write() as db:
                    db.executemany(
                        "INSERT OR REPLACE INTO staged_documents VALUES (?,?,?,?,?,?,?,?,?,?)",
                        batch,
                    )
            with self._write() as db:
                rebuild = db.execute(
                    "SELECT r.generation, MAX(s.all_scopes_dirty,s.records_deleted,COALESCE(a.generation,0)) "
                    "FROM rebuilds r CROSS JOIN projection_state s "
                    "LEFT JOIN account_generations a ON a.account_id=? WHERE r.id=? AND r.expires_at>?",
                    (account_id, rebuild_id, time.time()),
                ).fetchone()
                publish = rebuild is not None and rebuild[0] >= rebuild[1]
                if publish:
                    where, args = self._filters({"account_id": account_id, "owner_space": scopes})
                    db.execute(f"DELETE FROM documents AS d WHERE {where}", args)
                    db.execute(
                        f"INSERT INTO documents ({_DOCUMENT_COLUMNS}) "
                        f"SELECT {_DOCUMENT_COLUMNS} FROM staged_documents WHERE rebuild_id=?",
                        (rebuild_id,),
                    )
                    generation = self._next_generation(db)
                    self._touch_accounts(db, (account_id,), generation)
                    db.executemany(
                        "INSERT INTO scope_checkpoints VALUES (?,?,?,0) "
                        "ON CONFLICT(account_id,owner_space) DO UPDATE SET rebuilt=excluded.rebuilt",
                        ((account_id, scope, generation) for scope in scopes),
                    )
                db.execute("DELETE FROM rebuilds WHERE id=?", (rebuild_id,))
                return publish
        except Exception:
            try:
                self.abort_rebuild(rebuild_id)
            except sqlite3.Error:
                logger.warning("Unable to discard SQLite lexical rebuild", exc_info=True)
            raise

    def ready_scopes(self, account_id: str, owner_spaces: Sequence[str]) -> frozenset[str]:
        if not account_id or not owner_spaces:
            return frozenset()
        where, args = self._filters({"account_id": account_id, "owner_space": owner_spaces})
        with self._lock:
            rows = self._db.execute(
                "SELECT d.owner_space FROM scope_checkpoints d CROSS JOIN projection_state s "
                f"WHERE {where} AND d.rebuilt>=MAX(d.dirty,s.all_scopes_dirty)",
                args,
            ).fetchall()
        return frozenset(scope for (scope,) in rows)

    def dirty_generations(self, account_id: str, owner_spaces: Sequence[str]) -> dict[str, int]:
        if not account_id or not owner_spaces:
            return {}
        where, args = self._filters({"account_id": account_id, "owner_space": owner_spaces})
        with self._lock:
            rows = self._db.execute(
                "SELECT d.owner_space,d.dirty,s.all_scopes_dirty FROM projection_state s "
                f"LEFT JOIN scope_checkpoints d ON {where}",
                args,
            ).fetchall()
        global_dirty = rows[0][2]
        dirty = {scope: max(generation, global_dirty) for scope, generation, _ in rows if scope}
        return {scope: dirty.get(scope, global_dirty) for scope in owner_spaces}

    def mark_scopes_dirty(self, scopes: Sequence[tuple[str, str]]) -> None:
        scopes = tuple((account, scope) for account, scope in scopes if account and scope)
        if scopes:
            with self._write() as db:
                generation = self._next_generation(db)
                self._touch_accounts(db, (account for account, _ in scopes), generation)
                db.executemany(
                    "INSERT INTO scope_checkpoints VALUES (?,?,0,?) "
                    "ON CONFLICT(account_id,owner_space) DO UPDATE SET dirty=excluded.dirty",
                    ((account, scope, generation) for account, scope in scopes),
                )

    def mark_all_scopes_dirty(self) -> None:
        with self._write() as db:
            db.execute(
                "UPDATE projection_state SET all_scopes_dirty=? WHERE id=1",
                (self._next_generation(db),),
            )

    def health(self) -> dict[str, Any]:
        with self._lock:
            records = (
                0
                if self._closed
                else self._db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            )
            return {"provider": "sqlite_fts", "available": not self._closed, "records": records}

    def close(self, timeout: float = 5.0) -> bool:
        if not self._lock.acquire(timeout=max(timeout, 0.0)):
            return False
        try:
            if not self._closed:
                self._db.close()
                self._closed = True
            return True
        finally:
            self._lock.release()
