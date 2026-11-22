"""AGFS content with a SQL publication catalog for shared-memory deployments."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from urllib.parse import quote, urlsplit

from psycopg2.extras import Json
from pyagfs.exceptions import AGFSClientError

from core.errors import NodeBrokenError
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from fs.sql_adapter.pool import SharedConnectionPool
from fs.sql_adapter.sql_context_fs import SQLContextFS, _resource_mutation_lock_key

logger = logging.getLogger(__name__)


class AGFSSpaceBridge(SQLContextFS):
    """Publish immutable AGFS bodies through the existing SQL node transaction."""

    def __init__(self, pool, client, mount_prefix="/local"):
        super().__init__(pool=pool)
        self._agfs = AGFSContextFS(client=client, mount_prefix=mount_prefix)
        self._stage_pool = SharedConnectionPool(pool.connection_string, pool_size=1)
        self._content_reads = ThreadPoolExecutor(max_workers=8, thread_name_prefix="space-agfs")
        self._read_lock = threading.Lock()
        self._read_futures = set()

    def _map_reads(self, function, values):
        with self._read_lock:
            futures = [self._content_reads.submit(function, value) for value in values]
            self._read_futures.update(futures)
        for future in futures:
            future.add_done_callback(self._read_finished)
        return [future.result() for future in futures]

    def _read_finished(self, future):
        with self._read_lock:
            self._read_futures.discard(future)

    def _content_directory(self, uri):
        parsed = urlsplit(uri)
        account = parsed.netloc
        if parsed.scheme != "ctx" or not account or account in {".", ".."}:
            raise ValueError("AGFS content requires a tenant-scoped ctx URI")
        node_key = hashlib.sha256(uri.encode("utf-8")).hexdigest()
        return (f"{self._agfs._mount_prefix}/accounts/{quote(account, safe='')}/"
                f".space-content/{node_key}")

    def store_shared(self, uri, content, abstract="", overview="", metadata=None, *, cur=None):
        """Stage a complete body before its SQL pointer becomes visible."""
        package = {key: value for key, value in (metadata or {}).items()
                   if key in {"package_skill_md", "package_files", "fields"}}
        payload = json.dumps({"uri": uri, "content": content, "abstract": abstract,
                              "overview": overview, "metadata": package},
                             ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        directory = self._content_directory(uri)
        path = f"{directory}/{digest}.json"
        pointer = {"uri": uri, "path": path, "sha256": digest, "bytes": len(payload)}
        stage_id = None
        if cur is not None:
            account = urlsplit(uri).netloc
            self._acquire_resource_mutation_lock(cur, account, uri)
            stage_id = hashlib.sha256(f"agfs-stage:{path}".encode()).hexdigest()[:32]
        temporary = f"{path}.{stage_id or uuid.uuid4().hex}.tmp"
        try:
            existing = self._agfs._client.read(path)
        except AGFSClientError as exc:
            if not self._missing(exc):
                raise
            if stage_id:
                self._record_stage(account, uri, stage_id, digest)
            self._agfs._mkdir_p(directory)
            self._agfs._client.write(temporary, payload)
            try:
                self._agfs._client.mv(temporary, path)
            except AGFSClientError:
                existing = self._agfs._client.read(path)
                if hashlib.sha256(existing).hexdigest() != digest:
                    raise NodeBrokenError(uri)
                self._remove_content(temporary)
        else:
            if hashlib.sha256(existing).hexdigest() != digest:
                raise NodeBrokenError(uri)
        if stage_id:
            cur.execute("DELETE FROM outbox_events WHERE event_id=%s AND account_id=%s AND event_type='RECLAIM_AGFS_STAGE'",
                        (stage_id, account))
        return {"agfs_content": pointer}

    def _record_stage(self, account, uri, stage_id, digest):
        """Commit the reclaim record before staging the body."""
        conn = self._stage_pool.get_connection()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("""INSERT INTO outbox_events(event_id,event_type,uri,account_id,payload,status)
                    SELECT %s,'RECLAIM_AGFS_STAGE',%s,%s,%s,'GC_PENDING'
                    WHERE NOT EXISTS (SELECT 1 FROM outbox_events WHERE event_id=%s)""",
                    (stage_id, uri, account, Json({"sha256": digest}), stage_id))
        finally:
            self._stage_pool.return_connection(conn)

    @staticmethod
    def _missing(exc):
        message = str(exc).lower()
        return "not found" in message or "no such file" in message

    def _load_row(self, row, *, read=None):
        uri = row["uri"]
        metadata = row.get("metadata") or {}
        pointer = metadata.get("agfs_content")
        if not isinstance(pointer, dict):
            raise NodeBrokenError(uri)
        digest = pointer.get("sha256", "")
        original_uri = pointer.get("uri", uri)
        if urlsplit(original_uri).netloc != urlsplit(uri).netloc:
            raise NodeBrokenError(uri)
        path = f"{self._content_directory(original_uri)}/{digest}.json"
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or pointer.get("path") != path):
            raise NodeBrokenError(uri)
        payload = (read or self._agfs._client.read)(path)
        if hashlib.sha256(payload).hexdigest() != digest or len(payload) != pointer.get("bytes"):
            raise NodeBrokenError(uri)
        value = json.loads(payload)
        if value.get("uri") != original_uri:
            raise NodeBrokenError(uri)
        for key in ("content", "abstract", "overview"):
            row[key] = value[key]
        row["metadata"] = {**metadata, **value.get("metadata", {})}
        return row

    def load_content(self, rows):
        """Hydrate an already-authorized bounded batch from immutable AGFS bodies."""
        if len(rows) > 2000:
            raise ValueError("AGFS content batch exceeds 2000 nodes")
        if len(rows) == 1:
            self._load_row(rows[0])
        elif rows:
            self._map_reads(self._load_row, rows)
        return rows

    def _execute_write(self, cur, node, account_id, expected_version, create_only,
                       relations_json, extra_meta, new_version, now):
        extra_meta.update(self.store_shared(node.uri, node.content, node.abstract, node.overview, node.metadata, cur=cur))
        return super()._execute_write(cur, node, account_id, expected_version, create_only,
                                      relations_json, extra_meta, new_version, now)

    def _hydrate_node(self, node, level="all"):
        row = self._load_row({"uri": node.uri, "metadata": node.metadata})
        return replace(node, content=row["content"] if level in {"content", "all"} else "",
                       abstract=row["abstract"] if level in {"abstract", "summary", "all"} else "",
                       overview=row["overview"] if level in {"overview", "summary", "all"} else "",
                       metadata=row["metadata"])

    def read_node(self, uri, ctx):
        return self._hydrate_node(super().read_node(uri, ctx))

    def hydrate_nodes(self, nodes):
        """Hydrate a page already scoped by the SQL catalog."""
        if len(nodes) == 1:
            return [self._hydrate_node(nodes[0])]
        return self._map_reads(self._hydrate_node, nodes)

    def read_nodes(self, uris, ctx):
        results = super().read_nodes(uris, ctx)

        def hydrate(item):
            uri, node = item
            if isinstance(node, Exception):
                return uri, node
            try:
                return uri, self._hydrate_node(node)
            except Exception as exc:
                return uri, exc

        return dict(self._map_reads(hydrate, results.items()))

    def read_node_fields(self, uri, ctx, level):
        return self._hydrate_node(super().read_node_fields(uri, ctx, level), level)

    def list_node_summaries_page(self, uri, ctx, *, limit=50, after_uri=None):
        nodes = super().list_node_summaries_page(uri, ctx, limit=limit, after_uri=after_uri)
        rows = self.load_content([{"uri": node.uri, "metadata": node.metadata} for node in nodes])
        return [replace(node, abstract=row["abstract"], metadata=row["metadata"])
                for node, row in zip(nodes, rows)]

    def cleanup_shared(self, uri):
        """Reclaim AGFS versions after the catalog has committed its PURGED tombstone."""
        self._remove_content(self._content_directory(uri), recursive=True)

    def _remove_content(self, path, *, recursive=False):
        try:
            self._agfs._client.rm(path, recursive=recursive)
        except AGFSClientError as exc:
            if not self._missing(exc):
                raise

    def reclaim_deleted_content(self, cur, account_id, limit):
        cur.execute("""SELECT event_id,event_type,uri,payload FROM outbox_events
            WHERE account_id=%s AND status='GC_PENDING' AND event_type IN ('RECLAIM_AGFS','RECLAIM_AGFS_STAGE')
              AND (next_retry_at IS NULL OR next_retry_at<=NOW())
            ORDER BY seq LIMIT %s FOR UPDATE SKIP LOCKED""", (account_id, limit))
        for event in cur.fetchall():
            origin = event["uri"]
            cur.execute("SELECT pg_try_advisory_xact_lock(%s) AS acquired",
                        (_resource_mutation_lock_key(account_id, origin),))
            if not cur.fetchone()["acquired"]:
                continue
            try:
                self._reclaim_content(cur, account_id, event)
            except AGFSClientError:
                logger.warning("AGFS content reclaim will be retried for event %s", event["event_id"], exc_info=True)
                cur.execute("""UPDATE outbox_events SET next_retry_at=NOW()+INTERVAL '30 seconds'
                    WHERE account_id=%s AND event_id=%s""", (account_id, event["event_id"]))
                continue
            cur.execute("DELETE FROM outbox_events WHERE account_id=%s AND event_id=%s",
                        (account_id, event["event_id"]))

    def _reclaim_content(self, cur, account, event):
        origin = event["uri"]
        if event["event_type"] == "RECLAIM_AGFS":
            cur.execute("""SELECT 1 FROM context_nodes WHERE account_id=%s
                AND (metadata->'agfs_content'->>'uri')=%s LIMIT 1""", (account, origin))
            if cur.fetchone() is None:
                self.cleanup_shared(origin)
                cur.execute("""UPDATE context_nodes SET metadata=metadata || %s::jsonb
                    WHERE account_id=%s AND uri=%s AND metadata->>'state'='PURGED'""",
                    (Json({"agfs_purged": True}), account, origin))
            return
        digest = event["payload"]["sha256"]
        path = f"{self._content_directory(origin)}/{digest}.json"
        cur.execute("""SELECT BOOL_OR(COALESCE((metadata->'agfs_content'->>'path')=%s,FALSE)) AS referenced
            FROM context_nodes WHERE account_id=%s AND (metadata->'agfs_content'->>'uri')=%s""",
            (path, account, origin))
        referenced = cur.fetchone()["referenced"]
        if referenced is None:
            self.cleanup_shared(origin)
        else:
            self._remove_content(f"{path}.{event['event_id']}.tmp")
            if not referenced:
                self._remove_content(path)

    def close(self, timeout=1.0):
        self._stage_pool.close_all()
        with self._read_lock:
            self._content_reads.shutdown(wait=False)
            pending = tuple(self._read_futures)
        return not wait(pending, timeout=timeout).not_done
