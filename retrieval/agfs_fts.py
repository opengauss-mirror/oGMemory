"""Process-local SQLite candidate projection of authorized AGFS scopes."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import chain

from core.errors import OperationDeadlineExceededError
from core.memory_names import memory_directory
from core.operation_deadline import check_deadline, remaining_seconds

logger = logging.getLogger(__name__)

_MAX_SCOPES = 4
_REFRESH_SECONDS = 300
_RETRY_SECONDS = 30
_REFRESH_PENDING = 128
_MAX_PENDING = 512


@dataclass
class _Snapshot:
    db: sqlite3.Connection
    checked_at: float
    pending: set[str] = field(default_factory=set)
    query_lock: threading.Lock = field(default_factory=threading.Lock)
    readers: int = 0
    retired: bool = False

    def retire(self) -> None:
        self.retired = True
        if not self.readers:
            self.db.close()


@dataclass
class FTSCandidates:
    uris: list[str] | None
    status: str
    fields_truncated: bool = False
    over_budget: bool = False


def _sqlite_deadline() -> int:
    try:
        check_deadline()
    except OperationDeadlineExceededError:
        return 1
    return 0


class AGFSFullTextIndex:
    def __init__(self, documents, *, bm25: bool = False) -> None:
        self._documents = documents
        self._bm25 = bm25
        self._lock = threading.Lock()
        self._snapshots = OrderedDict()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._building_roots: tuple[str, ...] = ()
        self._during_build: set[str] = set()
        self._discard_build = False
        self._retry_at = 0.0

    def candidates(
        self, roots, ctx, terms: set[str], short_terms: set[str], *,
        search_roots: list[str] | None = None, directories: set[str] | None = None,
        max_candidates: int,
    ) -> FTSCandidates:
        key = (roots, ctx.account_id, ctx.user_id, ctx.agent_id, ctx.role, ctx.visible_owner_spaces)
        prefixes = tuple(root + "/" for root in (search_roots if search_roots is not None else roots))
        with self._lock:
            now = time.monotonic()
            snapshot = self._snapshots.get(key)
            fresh = snapshot is not None and now - snapshot.checked_at < _REFRESH_SECONDS
            if not self._stop.is_set() and self._worker is None and now >= self._retry_at and (
                not fresh or len(snapshot.pending) >= _REFRESH_PENDING
            ):
                self._building_roots = roots
                self._during_build.clear()
                self._discard_build = False
                self._worker = threading.Thread(target=self._build, args=(key, roots, ctx),
                                                name="agfs-fts", daemon=True)
                self._worker.start()
            if not fresh or len(snapshot.pending) >= _MAX_PENDING:
                status = "building" if self._worker is not None else "unavailable"
                return FTSCandidates(None, status)
            self._snapshots.move_to_end(key)
            # The snapshot stays open across publication and eviction while a reader uses it.
            snapshot.readers += 1
        try:
            uris, fields_truncated = self._query(snapshot, terms, short_terms, prefixes, directories, max_candidates + 1)
            with self._lock:
                if snapshot.retired or len(snapshot.pending) >= _MAX_PENDING:
                    return FTSCandidates(None, "unavailable")
                pending = sorted(snapshot.pending)
            pending_in_scope = (
                uri for uri in pending if uri.startswith(prefixes)
                and (not directories or memory_directory(uri) in directories)
            )
            candidates = list(dict.fromkeys(chain(pending_in_scope, uris)))
            check_deadline()
            return FTSCandidates(candidates[:max_candidates], "ready", fields_truncated,
                                 len(candidates) > max_candidates)
        except sqlite3.Error:
            check_deadline()
            with self._lock:
                if self._snapshots.get(key) is snapshot:
                    self._snapshots.pop(key).retire()
            logger.warning("AGFS FTS query failed", exc_info=True)
            return FTSCandidates(None, "unavailable")
        finally:
            with self._lock:
                snapshot.readers -= 1
                if snapshot.retired and not snapshot.readers:
                    snapshot.db.close()

    def _query(
        self, snapshot: _Snapshot, terms: set[str], short_terms: set[str],
        prefixes: tuple[str, ...], directories: set[str] | None, limit: int,
    ) -> tuple[list[str], bool]:
        scope = "(" + " OR ".join("instr(nodes.uri, ?) = 1" for _ in prefixes) + ")"
        params = prefixes
        if directories:
            scope += " AND nodes.directory IN (" + ", ".join("?" for _ in directories) + ")"
            params += tuple(sorted(directories))
        timeout = remaining_seconds()
        if not snapshot.query_lock.acquire(timeout=-1 if timeout is None else timeout):
            raise OperationDeadlineExceededError("Operation deadline exceeded")
        db = snapshot.db
        try:
            db.set_progress_handler(_sqlite_deadline, 1000)
            check_deadline()
            truncated = db.execute(
                f"SELECT 1 FROM nodes WHERE nodes.truncated = 1 AND {scope} LIMIT 1", params,
            ).fetchone() is not None
            scores = {}
            for table, values in (("postings", terms), ("short_postings", short_terms)):
                if not values:
                    continue
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in sorted(values))
                score = f"bm25({table})" if self._bm25 else f"{table}.rowid"
                order = "score, nodes.uri" if self._bm25 else f"{table}.rowid"
                for uri, value in db.execute(
                    f"SELECT nodes.uri, {score} AS score FROM {table} JOIN nodes ON nodes.id = {table}.rowid "
                    f"WHERE {table} MATCH ? AND {scope} ORDER BY {order} LIMIT ?", (expression, *params, limit),
                ):
                    scores[uri] = min(value, scores.get(uri, value))
            return sorted(scores, key=lambda uri: (scores[uri], uri)), truncated
        finally:
            db.set_progress_handler(None, 0)
            snapshot.query_lock.release()

    def changed(self, uri: str, *, subtree: bool = False) -> None:
        uri = uri.rstrip("/")
        with self._lock:
            for key, snapshot in list(self._snapshots.items()):
                if self._affects(key[0], uri, subtree):
                    if subtree:
                        self._snapshots.pop(key).retire()
                    elif len(snapshot.pending) < _MAX_PENDING:
                        snapshot.pending.add(uri)
            if self._affects(self._building_roots, uri, subtree):
                if subtree or len(self._during_build) >= _MAX_PENDING:
                    self._discard_build = True
                else:
                    self._during_build.add(uri)

    @staticmethod
    def _affects(roots: tuple[str, ...], uri: str, subtree: bool) -> bool:
        return any(uri == root or uri.startswith(root + "/") or (subtree and root.startswith(uri + "/"))
                   for root in roots)

    def _build(self, key, roots, ctx) -> None:
        db = None
        failed = False
        try:
            db = sqlite3.connect("", check_same_thread=False)
            db.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, uri TEXT NOT NULL, "
                       "directory TEXT NOT NULL, truncated INTEGER NOT NULL)")
            db.execute("CREATE INDEX truncated_nodes ON nodes(truncated) WHERE truncated = 1")
            db.execute("CREATE VIRTUAL TABLE postings USING fts5(text, content='', tokenize='trigram')")
            db.execute("CREATE VIRTUAL TABLE short_postings USING fts5(terms, content='', detail=none)")
            for rowid, (uri, document, short_terms, truncated) in enumerate(self._documents(roots, ctx), 1):
                if self._stop.is_set():
                    return
                db.execute("INSERT INTO nodes VALUES (?, ?, ?, ?)", (rowid, uri, memory_directory(uri), truncated))
                db.execute("INSERT INTO postings(rowid, text) VALUES (?, ?)", (rowid, document))
                db.execute("INSERT INTO short_postings(rowid, terms) VALUES (?, ?)", (rowid, short_terms))
            db.commit()
            with self._lock:
                if self._stop.is_set() or self._discard_build:
                    return
                previous = self._snapshots.pop(key, None)
                if previous is not None:
                    previous.retire()
                self._snapshots[key] = _Snapshot(db, time.monotonic(), self._during_build.copy())
                db = None
                if len(self._snapshots) > _MAX_SCOPES:
                    self._snapshots.popitem(last=False)[1].retire()
        except Exception:
            failed = True
            logger.warning("AGFS FTS backfill failed", exc_info=True)
        finally:
            if db is not None:
                db.close()
            with self._lock:
                if failed:
                    self._retry_at = time.monotonic() + _RETRY_SECONDS
                self._worker = None
                self._building_roots = ()

    def close(self, timeout: float) -> bool:
        self._stop.set()
        with self._lock:
            worker = self._worker
        if worker is not None:
            worker.join(timeout)
            if worker.is_alive():
                return False
        with self._lock:
            for snapshot in self._snapshots.values():
                snapshot.retire()
            self._snapshots.clear()
        return True
