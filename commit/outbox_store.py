"""Outbox event storage for async index synchronization.

Persists OutboxEvents to AGFS for processing by OutboxWorker.
"""

import json
import uuid
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

from pyagfs import AGFSClient

from commit.code_resource_events import (
    build_code_resource_delete_event,
    build_code_resource_write_event,
)
from core.enums import EventType
from core.interfaces import ContextFS
from core.logging_config import get_logger
from core.models import ContextNode, OutboxEvent, RequestContext

logger = get_logger(__name__)
_AGENT_PROCESSING_LEASE = ContextVar("agent_processing_lease", default=None)


class OutboxPublishOutcomeUnknownError(RuntimeError):
    """Raised when an atomic event publish cannot be classified safely."""


def serialize_outbox_event(event: OutboxEvent) -> bytes:
    """Return the canonical on-disk representation of an Outbox event."""

    return json.dumps(
        {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
            "next_retry_at": event.next_retry_at,
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


class OutboxStore:
    """Persists OutboxEvents to AGFS .outbox/ directories.

    OutboxEvents are written alongside the node directory:
    /{mount_prefix}/accounts/{...}/node/.outbox/{event_id}.json

    These are processed by OutboxWorker (index package) to sync
    changes to the vector index.
    """

    _DELETE_TOMBSTONE_NAME = ".delete_tombstone.json"
    _DELETE_TOMBSTONE_PENDING = "PENDING"
    _DELETE_TOMBSTONE_ACKNOWLEDGING = "ACKNOWLEDGING"
    _NODE_PAYLOAD_NAMES = (
        "content.md",
        ".relations.json",
        ".abstract.md",
        ".overview.md",
        ".meta.json",
    )

    def __init__(self, client: AGFSClient, fs: ContextFS, mount_prefix: str = "/local"):
        """Initialize OutboxStore.

        Args:
            client: AGFS client for file operations
            fs: ContextFS for URI to path conversion (must be AGFSContextFS)
            mount_prefix: AGFS mount point prefix (default: "/local")
                         Must match the mount_prefix used by AGFSContextFS

        Raises:
            ValueError: If mount_prefix contains path traversal patterns
        """
        # Security: validate mount_prefix to prevent path traversal attacks
        if ".." in mount_prefix or not mount_prefix.startswith("/"):
            raise ValueError(f"Invalid mount_prefix: must be absolute path without '..': {mount_prefix}")

        self._client = client
        self._fs = fs
        self._mount_prefix = mount_prefix.rstrip("/")

    def register_write(
        self,
        node: ContextNode,
        ctx: RequestContext
    ) -> OutboxEvent:
        """Register a node write for index synchronization.

        Called after successful ContextFS.write_node().

        Args:
            node: ContextNode that was written
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        self._check_agent_lease()
        event = self.build_write_event(node)
        self.persist_event(event, node.uri)
        return event

    def build_write_event(self, node: ContextNode) -> OutboxEvent:
        """Build a self-contained UPSERT event without persisting it."""
        return build_code_resource_write_event(node)

    def register_delete(
        self,
        uri: str,
        ctx: RequestContext
    ) -> OutboxEvent:
        """Register a node deletion for index synchronization.

        Args:
            uri: URI of deleted node
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        event = self.build_delete_event(uri)
        self.persist_event(event, uri)
        return event

    @staticmethod
    def build_delete_event(uri: str) -> OutboxEvent:
        """Build a DELETE event containing every canonical record id."""
        return build_code_resource_delete_event(uri)

    def persist_event(
        self,
        event: OutboxEvent,
        node_uri: str | None = None,
    ) -> OutboxEvent:
        """Persist a pre-built event at its owning node Outbox."""

        self._write_event(event, node_uri or event.uri)
        return event

    def register_once(self, event: OutboxEvent, ctx: RequestContext) -> None:
        with self._agent_event_lock(event):
            node_path = self._uri_to_agfs_path(event.uri)
            if self._cleanup_completed_agent_event(event):
                return
            if not self.is_event_current(event.event_id, event.uri):
                current = ""
                for part in node_path.strip("/").split("/"):
                    current += "/" + part
                    try:
                        self._client.mkdir(current)
                    except Exception as exc:
                        if "exists" not in str(exc).lower():
                            raise
                self.persist_event(event)

    def save_payload(self, event: OutboxEvent) -> None:
        with self._agent_event_lock(event):
            self._check_agent_lease(event)
            if not self._cleanup_completed_agent_event(event):
                self.persist_event(event)

    def _agent_event_done(self, event: OutboxEvent) -> bool:
        try:
            self._client.read(self._uri_to_agfs_path(event.uri) + ".agent_memory_done")
            return True
        except Exception as exc:
            if not self._is_not_found_error(exc):
                raise
            return False

    def _agent_event_lock(self, event: OutboxEvent):
        from fs.agfs_adapter.agfs_context_fs import AGFSContextFS

        ctx = RequestContext(**event.payload["context"])
        fs = AGFSContextFS(self._client, mount_prefix=self._mount_prefix)
        return fs.resource_mutation_lock(event.uri, ctx)

    def _remove_agent_event_files(self, event: OutboxEvent) -> None:
        prefix = self._uri_to_agfs_path(event.uri) + ".outbox/" + event.event_id
        self._rm_if_present(prefix + ".processing")
        self._rm_if_present(prefix + ".json")

    def _cleanup_completed_agent_event(self, event: OutboxEvent) -> bool:
        """Finish acknowledged cleanup while holding the event mutation lock."""
        if not self._agent_event_done(event):
            return False
        self._remove_agent_event_files(event)
        return True

    @staticmethod
    def _check_agent_lease(event=None):
        current = _AGENT_PROCESSING_LEASE.get()
        if current is not None and (event is None or current[0] == event.uri):
            current[1].ensure_owned()

    @contextmanager
    def claim_agent_event(self, event: OutboxEvent, worker_id: str):
        """Hold a renewable lease from claim through checkpoint and acknowledgement."""
        from fs.agfs_adapter.agfs_context_fs import AGFSContextFS

        ctx = RequestContext(**event.payload["context"])
        fs = AGFSContextFS(self._client, mount_prefix=self._mount_prefix)
        with ExitStack() as stack:
            try:
                lease = stack.enter_context(fs.resource_mutation_lock(
                    event.uri + "-worker", ctx, timeout_seconds=0))
            except TimeoutError:
                yield False
                return
            with self._agent_event_lock(event):
                pending = []
                path = self._uri_to_agfs_path(event.uri) + ".outbox/"
                if not self._cleanup_completed_agent_event(event):
                    self._collect_outbox_events(path, [{"name": event.event_id + ".json"}], pending)
                acquired = bool(pending)
                if acquired:
                    fresh = pending[0][1]
                    event.payload, event.retry_count = fresh.payload, fresh.retry_count
                    event.status, event.next_retry_at = fresh.status, fresh.next_retry_at
                    acquired = self._try_acquire(event, event.uri, worker_id)
            token = _AGENT_PROCESSING_LEASE.set((event.uri, lease))
            try:
                yield acquired
            finally:
                _AGENT_PROCESSING_LEASE.reset(token)

    def register_archive(
        self,
        uri: str,
        ctx: RequestContext,
        *,
        record_ids: list[str] | None = None,
    ) -> OutboxEvent:
        """Register a node archive for index synchronization.

        Args:
            uri: URI of archived node
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        event_id = str(uuid.uuid4())
        event = OutboxEvent(
            event_id=event_id,
            event_type=EventType.ARCHIVE_CONTEXT.value,
            uri=uri,
            payload={"ids_to_delete": record_ids} if record_ids is not None else {},
            status="PENDING",
        )

        self._write_event(event, uri)
        return event

    def register_relation(
        self,
        edges: list,
        ctx: RequestContext
    ) -> OutboxEvent:
        """Register relation edge updates for index synchronization.

        Args:
            edges: List of RelationEdge objects
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        # Use the first edge's URI as the base location
        base_uri = edges[0].from_uri if edges else ""

        event_id = str(uuid.uuid4())
        event = OutboxEvent(
            event_id=event_id,
            event_type=EventType.UPSERT_RELATION.value,
            uri=base_uri,
            payload={
                "edges": [
                    {
                        "from_uri": e.from_uri,
                        "to_uri": e.to_uri,
                        "relation_type": e.relation_type,
                        "weight": e.weight,
                    }
                    for e in edges
                ],
            },
            status="PENDING",
        )

        if base_uri:
            self._write_event(event, base_uri)

        return event

    def register_directory(
        self,
        node: ContextNode,
        ctx: RequestContext
    ) -> OutboxEvent:
        """Register UPSERT_DIRECTORY event for node's parent directory.

        Collects sibling abstracts (max 20) and writes OutboxEvent
        with payload self-contained — indexer does NOT access AGFS.

        Args:
            node: ContextNode that was written (child of the directory)
            ctx: RequestContext for this operation

        Returns:
            OutboxEvent that was registered
        """
        dir_uri = self._node_uri_to_directory_uri(node.uri)
        abstracts = self._collect_sibling_abstracts(node.uri, ctx)

        payload = {
            "directory_uri": dir_uri,
            "child_abstracts": abstracts,
            "filters": {
                "account_id": ctx.account_id,
                "owner_space": node.owner_space,
            },
        }

        event_id = str(uuid.uuid4())
        event = OutboxEvent(
            event_id=event_id,
            event_type=EventType.UPSERT_DIRECTORY.value,
            uri=dir_uri,
            payload=payload,
            status="PENDING",
        )

        # Directory URIs (e.g. ctx://acct/users/u1/memories/cat/) lack a slug,
        # so uri_to_path() won't match. Build AGFS path directly.
        dir_agfs_path = (self._mount_prefix or "") + "/accounts/" + dir_uri.replace("ctx://", "").rstrip("/")
        outbox_path = f"{dir_agfs_path}/.outbox/{event.event_id}.json"

        # Ensure .outbox directory exists
        try:
            self._client.mkdir(f"{dir_agfs_path}/.outbox")
        except Exception:
            pass

        # Serialize and write event
        event_json = json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
        }, ensure_ascii=False, indent=2)

        self._client.write(outbox_path, event_json.encode('utf-8'))

        return event

    def _node_uri_to_directory_uri(self, node_uri: str) -> str:
        """Convert leaf node URI to parent directory URI.

        Args:
            node_uri: Full URI of a leaf node

        Returns:
            Parent directory URI with trailing slash

        Example:
            ctx://acme/users/u1/memories/preferences/coffee
            → ctx://acme/users/u1/memories/preferences/
        """
        # Remove trailing slash and final slug, add trailing slash
        parts = node_uri.rstrip("/").rsplit("/", 1)
        return parts[0] + "/"

    def _collect_sibling_abstracts(
        self,
        node_uri: str,
        ctx: RequestContext
    ) -> list[str]:
        """Collect sibling node abstracts from AGFS.

        Args:
            node_uri: URI of a node in the directory
            ctx: RequestContext for access control

        Returns:
            List of abstract strings (max 20)
        """
        dir_uri = self._node_uri_to_directory_uri(node_uri)
        try:
            siblings = self._fs.list_children(dir_uri, ctx)
        except Exception:
            return []

        abstracts = []
        for sibling_uri in siblings[:20]:
            try:
                node = self._fs.read_node(sibling_uri, ctx)
                if node and node.abstract:
                    abstracts.append(node.abstract[:200])
            except Exception:
                continue
        return abstracts

    def _uri_to_agfs_path(self, uri: str) -> str:
        """Convert ContextEngine URI to AGFS path with mount prefix.

        Handles both node URIs (ctx://acct/users/u1/memories/cat/slug)
        and directory URIs (ctx://acct/users/u1/memories/cat/).
        """
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        try:
            base_path = uri_to_path(uri)
        except ValueError:
            # Directory-level URI (no slug) — build path directly
            base_path = "/accounts/" + uri.replace("ctx://", "").rstrip("/") + "/"

        if self._mount_prefix:
            return self._mount_prefix + base_path
        return base_path

    def _write_event(self, event: OutboxEvent, node_uri: str) -> None:
        """Write OutboxEvent to .outbox/ directory using atomic write pattern.

        Uses write-to-temp + rename for atomicity:
        1. Write to temporary file .tmp_{event_id}.json
        2. Rename to {event_id}.json (atomic operation)

        Args:
            event: OutboxEvent to write
            node_uri: URI of the associated node (for path calculation)

        Raises:
            ValueError: If node_uri contains path traversal patterns
        """
        # SECURITY: Validate URI format and reject path traversal
        from fs.agfs_adapter.agfs_context_fs import parse_uri
        try:
            components = parse_uri(node_uri)
        except ValueError as e:
            raise ValueError(f"Invalid node_uri in OutboxEvent: {node_uri}") from e

        # Check each URI component for path traversal patterns
        for key, val in components.items():
            val_str = str(val)
            if ".." in val_str or val_str.startswith("/"):
                raise ValueError(
                    f"Path traversal detected in URI component '{key}': {val_str}"
                )

        # Get node path and construct outbox path (with mount_prefix)
        node_path = self._uri_to_agfs_path(node_uri)
        outbox_dir = node_path + ".outbox"
        event_path = outbox_dir + "/" + event.event_id + ".json"
        temp_path = outbox_dir + "/.tmp_" + event.event_id + ".json"

        # Ensure .outbox directory exists
        try:
            self._client.mkdir(outbox_dir)
        except Exception:
            pass  # Directory may already exist

        event_bytes = serialize_outbox_event(event)

        # Write to temporary file first
        try:
            self._client.write(temp_path, event_bytes)
        except Exception as e:
            # Temp write failed - no state change
            logger.error("Failed to write temp outbox event %s: %s", event.event_id, e)
            raise

        # Move to the final path.  AGFS exposes same-backend atomic rename as
        # ``mv``; a direct final write would expose a partially written event.
        try:
            self._client.mv(temp_path, event_path)
        except Exception as exc:
            try:
                final_raw = self._client.read(event_path)
            except Exception as inspect_exc:
                if self._is_not_found_error(inspect_exc):
                    try:
                        self._client.rm(temp_path)
                    except Exception:
                        pass
                    raise exc
                try:
                    self._client.rm(temp_path)
                except Exception:
                    pass
                raise OutboxPublishOutcomeUnknownError(
                    "AGFS Outbox publish outcome cannot be proven"
                ) from inspect_exc

            if isinstance(final_raw, bytearray):
                final_raw = bytes(final_raw)
            elif isinstance(final_raw, str):
                final_raw = final_raw.encode("utf-8")
            if final_raw == event_bytes:
                try:
                    self._client.rm(temp_path)
                except Exception:
                    pass
                return
            try:
                self._client.rm(temp_path)
            except Exception:
                pass
            raise OutboxPublishOutcomeUnknownError(
                "AGFS Outbox publish outcome cannot be proven"
            ) from exc

    @staticmethod
    def _is_not_found_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "no such file" in message or "not found" in message

    def _rm_if_present(self, path: str, *, recursive: bool = False) -> bool:
        try:
            if recursive:
                self._client.rm(path, recursive=True)
            else:
                self._client.rm(path)
            return True
        except Exception as exc:
            if self._is_not_found_error(exc):
                return False
            raise

    @staticmethod
    def _load_event_json(raw) -> dict:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            raise TypeError("Outbox event content must be bytes or text")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Outbox event content must be a JSON object")
        return data

    def cancel_pending_upserts(self, node_uri: str) -> list[str]:
        """Cancel exact-URI UPSERT events, including claimed lock files."""

        node_path = self._uri_to_agfs_path(node_uri)
        outbox_dir = node_path + ".outbox"
        try:
            entries = self._client.ls(outbox_dir + "/")
        except Exception as exc:
            if self._is_not_found_error(exc):
                return []
            raise

        cancelled: list[str] = []
        for entry in entries:
            name = str(entry.get("name", "")).rsplit("/", 1)[-1]
            is_dir = entry.get("is_dir", entry.get("isDir", False))
            if is_dir or not name.endswith(".json") or name.startswith("."):
                continue
            event_path = outbox_dir + "/" + name
            try:
                raw = self._client.read(event_path)
                data = self._load_event_json(raw)
            except Exception as exc:
                if self._is_not_found_error(exc):
                    continue
                raise
            if data.get("uri") != node_uri:
                continue
            if data.get("event_type") not in {
                EventType.UPSERT_CONTEXT.value,
                EventType.UPSERT_DIRECTORY.value,
            }:
                continue
            if data.get("status", "PENDING") not in {"PENDING", "PROCESSING"}:
                continue
            event_id = str(data.get("event_id") or name[:-5])
            self._rm_if_present(event_path)
            self._rm_if_present(outbox_dir + "/" + event_id + ".processing")
            cancelled.append(event_id)
        return cancelled

    def is_event_current(self, event_id: str, node_uri: str | None = None) -> bool:
        """Return whether an event remains live and is not suppressed by a tombstone."""

        if not node_uri:
            raise ValueError("node_uri is required for AGFS event liveness checks")
        node_path = self._uri_to_agfs_path(node_uri)
        event_path = node_path + ".outbox/" + event_id + ".json"
        try:
            raw = self._client.read(event_path)
            data = self._load_event_json(raw)
        except Exception as exc:
            if self._is_not_found_error(exc):
                return False
            raise

        if data.get("event_type") not in {
            EventType.UPSERT_CONTEXT.value,
            EventType.UPSERT_DIRECTORY.value,
        }:
            return True
        tombstone_path = node_path + ".delete_tombstone.json"
        try:
            raw = self._client.read(tombstone_path)
            marker = self._load_event_json(raw)
            return not (
                marker.get("kind") == "DELETE_TOMBSTONE"
                and marker.get("uri") == node_uri
            )
        except Exception as exc:
            if self._is_not_found_error(exc):
                return True
            raise

    def mark_processing(self, event: OutboxEvent, node_uri: str) -> None:
        """Mark an event as being processed.

        Args:
            event: OutboxEvent to update
            node_uri: URI of the associated node
        """
        node_path = self._uri_to_agfs_path(node_uri)
        event_path = node_path + ".outbox/" + event.event_id + ".json"

        # Update event status
        event.status = "PROCESSING"
        event_json = json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
        }, ensure_ascii=False, indent=2)

        self._client.write(event_path, event_json.encode('utf-8'))

    def mark_done(self, event: OutboxEvent, node_uri: str) -> None:
        """Mark an event as completed and delete it.

        Args:
            event: OutboxEvent to complete
            node_uri: URI of the associated node
        """
        node_path = self._uri_to_agfs_path(node_uri)
        event_path = node_path + ".outbox/" + event.event_id + ".json"
        processing_path = node_path + ".outbox/" + event.event_id + ".processing"

        if event.event_type == EventType.EXTRACT_AGENT_MEMORY.value:
            with self._agent_event_lock(event):
                self._check_agent_lease(event)
                audit = {key: event.payload[key] for key in ("trace_uri", "value_reviews") if key in event.payload}
                self._client.write(node_path + ".agent_memory_done", json.dumps(audit, ensure_ascii=False).encode())
                self._remove_agent_event_files(event)
            return

        if event.event_type != EventType.DELETE_CONTEXT.value:
            for path in (event_path, processing_path):
                try:
                    self._rm_if_present(path)
                except Exception:
                    pass
            return

        # Keep the live event available until payload cleanup is verified and
        # the tombstone records that acknowledgement has started.  A crash
        # after the event removal is then recoverable from ACKNOWLEDGING.
        self._verify_delete_payload_absent(node_uri)
        has_tombstone = self._transition_delete_tombstone_to_acknowledging(
            event,
            node_uri,
        )

        if not has_tombstone:
            # Generic DELETE events predate the durable resource tombstone
            # protocol.  Preserve their existing strict lock cleanup.
            self._rm_if_present(event_path)
            self._rm_if_present(processing_path)
            self._cleanup_empty_outbox_directory(node_uri)
            return

        if not self._remove_acknowledged_delete_event(event_path, node_uri):
            return

        # Once event removal succeeds, do not raise back into the worker: its
        # generic error path would recreate the DELETE event and race with
        # ACKNOWLEDGING recovery.  The marker remains a durable recovery token
        # if any best-effort cleanup below is interrupted or fails.
        try:
            self._rm_if_present(processing_path)
        except Exception as exc:
            logger.warning(
                "Failed to remove DELETE processing lock for %s: %s",
                node_uri,
                exc,
            )
            return
        try:
            self._cleanup_empty_outbox_directory(node_uri)
        except Exception as exc:
            logger.warning(
                "Failed to clean empty DELETE Outbox for %s: %s",
                node_uri,
                exc,
            )
            return
        try:
            self._rm_if_present(node_path + self._DELETE_TOMBSTONE_NAME)
        except Exception as exc:
            logger.warning(
                "Failed to remove acknowledged DELETE tombstone for %s: %s",
                node_uri,
                exc,
            )

    def _remove_acknowledged_delete_event(
        self,
        event_path: str,
        node_uri: str,
    ) -> bool:
        """Remove the live DELETE event without replaying an ambiguous delete."""

        try:
            self._client.rm(event_path)
            return True
        except Exception as exc:
            if self._is_not_found_error(exc):
                return True

            # A lost response may mean the durable remove succeeded.  Re-read
            # before propagating: recreating an event that is already absent
            # would let an old DELETE replay after lazy ACK recovery releases
            # the URI for a new write.
            try:
                self._client.read(event_path)
            except Exception as verify_exc:
                if self._is_not_found_error(verify_exc):
                    logger.warning(
                        "DELETE event removal returned an error but %s is absent",
                        node_uri,
                    )
                    return True
                logger.warning(
                    "DELETE event removal outcome is unknown for %s; "
                    "retaining ACKNOWLEDGING state",
                    node_uri,
                )
                return False

            # The event is confirmed live, so the worker may safely use its
            # normal retry path while ACKNOWLEDGING continues to block writes.
            raise

    def _verify_delete_payload_absent(self, node_uri: str) -> None:
        """Fail before acknowledgement while any node payload may remain."""

        node_path = self._uri_to_agfs_path(node_uri)
        for name in self._NODE_PAYLOAD_NAMES:
            try:
                self._client.read(node_path + name)
            except Exception as exc:
                if self._is_not_found_error(exc):
                    continue
                raise
            raise RuntimeError(
                f"cannot acknowledge DELETE_CONTEXT while payload exists: {name}"
            )

    def _transition_delete_tombstone_to_acknowledging(
        self,
        event: OutboxEvent,
        node_uri: str,
    ) -> bool:
        """Atomically publish the recoverable DELETE acknowledgement state."""

        node_path = self._uri_to_agfs_path(node_uri)
        tombstone_path = node_path + self._DELETE_TOMBSTONE_NAME
        try:
            raw = self._client.read(tombstone_path)
        except Exception as exc:
            if self._is_not_found_error(exc):
                return False
            raise

        try:
            marker = self._load_event_json(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("invalid durable delete tombstone") from exc
        if (
            marker.get("kind") != "DELETE_TOMBSTONE"
            or marker.get("uri") != node_uri
            or marker.get("delete_event_id") != event.event_id
        ):
            raise RuntimeError("durable delete tombstone does not match event")

        status = marker.get("status", self._DELETE_TOMBSTONE_PENDING)
        if status == self._DELETE_TOMBSTONE_ACKNOWLEDGING:
            return True
        if status != self._DELETE_TOMBSTONE_PENDING:
            raise RuntimeError("durable delete tombstone has invalid status")

        acknowledging = dict(marker)
        acknowledging["status"] = self._DELETE_TOMBSTONE_ACKNOWLEDGING
        temp_path = tombstone_path + f".tmp-{uuid.uuid4().hex}"
        encoded = json.dumps(
            acknowledging,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        self._client.write(temp_path, encoded)
        try:
            self._client.mv(temp_path, tombstone_path)
        except Exception:
            try:
                self._client.rm(temp_path)
            except Exception:
                pass
            raise
        return True

    def _cleanup_empty_outbox_directory(self, node_uri: str) -> None:
        """Remove an empty Outbox without recursively deleting the node."""

        node_path = self._uri_to_agfs_path(node_uri)
        outbox_dir = node_path + ".outbox"
        try:
            entries = self._client.ls(outbox_dir + "/")
            remaining = [
                entry
                for entry in entries
                if entry.get("name") not in {"", ".", ".."}
            ]
            if not remaining:
                self._rm_if_present(outbox_dir, recursive=True)
        except Exception as exc:
            if self._is_not_found_error(exc):
                return
            raise

    def list_pending(self, account_id: str) -> list[tuple[str, "OutboxEvent"]]:
        """Scan AGFS for all pending OutboxEvents under an account.

        Scans /accounts/{account_id}/**/.outbox/*.json (excluding dlq/).
        Returns list of (node_uri, OutboxEvent) tuples.
        """
        results: list[tuple[str, OutboxEvent]] = []
        prefix = self._mount_prefix + f"/accounts/{account_id}/"
        try:
            self._scan_outbox_recursive(prefix, results)
        except Exception:
            pass
        return results

    def count_unfinished(self, account_id: str | None = None) -> dict[str, int]:
        """Count every live account Outbox event without claiming it.

        This deliberately differs from :meth:`list_pending`: active processing
        locks and retry-delayed events are unfinished index work even though a
        worker must not claim them again.  The counter is used only for
        freshness reporting, so an incomplete AGFS scan raises rather than
        reporting a false idle state.
        """

        if (
            not isinstance(account_id, str)
            or not account_id.strip()
            or ".." in account_id
            or "/" in account_id
            or "\\" in account_id
        ):
            raise ValueError("account_id is invalid")
        counts = {"pending": 0, "processing": 0}
        prefix = self._mount_prefix + f"/accounts/{account_id}/"
        try:
            entries = self._client.ls(prefix)
        except Exception as exc:
            if self._is_not_found_error(exc):
                return {"pending": 0, "processing": 0, "total": 0}
            raise
        self._count_unfinished_entries(prefix, entries, counts)
        counts["total"] = counts["pending"] + counts["processing"]
        return counts

    def _count_unfinished_recursive(self, path: str, counts: dict[str, int]) -> None:
        try:
            entries = self._client.ls(path)
        except Exception as exc:
            name = path.rstrip("/").rsplit("/", 1)[-1]
            if self._is_resource_lock_directory(name) and (
                self._is_not_found_error(exc) or "no such directory" in str(exc).lower()
            ):
                return
            raise
        self._count_unfinished_entries(path, entries, counts)

    @staticmethod
    def _is_resource_lock_directory(name: str) -> bool:
        """Identify mutation lease names whose directory can disappear mid-scan."""

        if not name.startswith("."):
            return False
        for suffix in (".resource-mutation-lock", ".update-lock"):
            if name.endswith(suffix) and len(name) > len(suffix) + 1:
                return True
        for marker in (".resource-mutation-lock-", ".update-lock-"):
            prefix, separator, token = name.rpartition(marker)
            if (
                separator
                and len(prefix) > 1
                and len(token) == 32
                and all(char in "0123456789abcdef" for char in token)
            ):
                return True
        return False

    @staticmethod
    def _is_atomic_resource_staging_directory(name: str) -> bool:
        """Return whether ``name`` is a hidden atomic code publish tree."""

        prefix, separator, token = name.rpartition(".resource-create-")
        return bool(
            separator
            and prefix.startswith(".")
            and len(prefix) > 1
            and len(token) == 32
            and all(char in "0123456789abcdef" for char in token)
        )

    def _count_unfinished_entries(
        self,
        path: str,
        entries: list[dict],
        counts: dict[str, int],
    ) -> None:
        """Walk an account tree and count live Outbox records exactly once."""

        if ".." in path:
            raise ValueError("invalid Outbox scan path")
        for entry in entries:
            name = str(entry.get("name", "")).rsplit("/", 1)[-1]
            if not name or name in {".", ".."}:
                continue
            is_dir = bool(entry.get("is_dir", entry.get("isDir", False)))
            child_path = path.rstrip("/") + "/" + name
            if is_dir and name == ".outbox":
                self._count_unfinished_outbox(child_path, counts)
            elif (
                is_dir
                and name != ".outbox"
                and not self._is_atomic_resource_staging_directory(name)
            ):
                self._count_unfinished_recursive(child_path + "/", counts)

    def _count_unfinished_outbox(self, outbox_path: str, counts: dict[str, int]) -> None:
        """Count pending and processing files, including locks and retries."""

        entries = self._client.ls(outbox_path + "/")
        processing_ids = {
            str(entry.get("name", "")).rsplit("/", 1)[-1][:-11]
            for entry in entries
            if not bool(entry.get("is_dir", entry.get("isDir", False)))
            and str(entry.get("name", "")).rsplit("/", 1)[-1].endswith(".processing")
        }
        for entry in entries:
            name = str(entry.get("name", "")).rsplit("/", 1)[-1]
            is_dir = bool(entry.get("is_dir", entry.get("isDir", False)))
            if is_dir or not name.endswith(".json") or name.startswith("."):
                continue
            event_id = name[:-5]
            try:
                raw = self._client.read(outbox_path.rstrip("/") + "/" + name)
            except Exception as exc:
                if self._is_not_found_error(exc):
                    continue
                raise
            data = self._load_event_json(raw)
            if data.get("event_id") != event_id:
                raise ValueError("Outbox event id does not match its file name")
            status = data.get("status", "PENDING")
            if status == "PROCESSING" or event_id in processing_ids:
                counts["processing"] += 1
            elif status == "PENDING":
                counts["pending"] += 1
            else:
                raise ValueError("Outbox event has an invalid live status")

    def _scan_outbox_recursive(self, path: str, results: list) -> None:
        """Recursively scan directory for .outbox/ event files.

        SECURITY: Skips paths containing '..' to prevent path traversal.
        Skips events with active .processing locks (default 300s timeout).
        """
        # SECURITY: Skip paths with traversal patterns
        if ".." in path:
            return

        try:
            entries = self._client.ls(path)
        except Exception:
            return

        for entry in entries:
            name = entry.get("name", "")
            is_dir = entry.get("is_dir", entry.get("isDir", False))
            child_path = path.rstrip("/") + "/" + name
            if is_dir and name == ".outbox":
                try:
                    ev_entries = self._client.ls(child_path + "/")
                except Exception:
                    continue
                self._collect_outbox_events(child_path, ev_entries, results)
            elif (
                is_dir
                and name not in (".outbox", "dlq")
                and not self._is_atomic_resource_staging_directory(str(name))
            ):
                self._scan_outbox_recursive(child_path + "/", results)

    def _collect_outbox_events(self, outbox_path: str, entries: list[dict], results: list) -> None:
        """Collect pending OutboxEvents from a single .outbox directory."""
        for entry in entries:
            ev_name = entry.get("name", "")
            is_dir = entry.get("is_dir", entry.get("isDir", False))
            if is_dir or not ev_name.endswith(".json"):
                continue
            if ev_name.startswith("."):
                continue

            ev_path = outbox_path.rstrip("/") + "/" + ev_name
            try:
                raw = self._client.read(ev_path)
                data = json.loads(raw.decode("utf-8"))
                if "/dlq/" in ev_path:
                    continue

                event_id = data["event_id"]
                processing_path = outbox_path.rstrip("/") + "/" + event_id + ".processing"
                has_active_lock = False
                try:
                    lock_raw = self._client.read(processing_path)
                    lock_data = json.loads(lock_raw.decode("utf-8"))
                    acquired_at = lock_data.get("acquired_at", "")
                    try:
                        lock_time = datetime.fromisoformat(acquired_at)
                        elapsed = (datetime.now(UTC) - lock_time).total_seconds()
                        if elapsed < 300:
                            has_active_lock = True
                    except (ValueError, KeyError):
                        pass
                except Exception:
                    pass

                if has_active_lock:
                    continue

                next_retry_at_str = data.get("next_retry_at", "")
                if next_retry_at_str:
                    try:
                        next_retry_time = datetime.fromisoformat(next_retry_at_str)
                        if datetime.now(UTC) < next_retry_time:
                            continue
                    except ValueError:
                        pass

                event = OutboxEvent(
                    event_id=data["event_id"],
                    event_type=data["event_type"],
                    uri=data["uri"],
                    payload=data.get("payload", {}),
                    status=data.get("status", "PENDING"),
                    retry_count=data.get("retry_count", 0),
                    created_at=data.get("created_at", ""),
                    next_retry_at=next_retry_at_str,
                )
                results.append((event.uri, event))
            except Exception:
                pass

    def move_to_dlq(self, event: "OutboxEvent", node_uri: str) -> None:
        """Move a failed event to the dead letter queue."""
        self._check_agent_lease(event)
        node_path = self._uri_to_agfs_path(node_uri)
        src_path = node_path + ".outbox/" + event.event_id + ".json"
        dlq_dir = node_path + ".outbox/dlq/"
        dlq_path = dlq_dir + event.event_id + ".json"
        try:
            self._client.mkdir(dlq_dir.rstrip("/"))
        except Exception:
            pass
        event.status = "FAILED"
        data = json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
            "next_retry_at": event.next_retry_at,
        }, ensure_ascii=False, indent=2)
        self._client.write(dlq_path, data.encode("utf-8"))
        try:
            self._client.rm(src_path)
        except Exception:
            pass

    def increment_retry(
        self,
        event: "OutboxEvent",
        node_uri: str,
        next_retry_at: datetime | None = None,
    ) -> None:
        """Increment retry_count and re-write event file.

        Args:
            event: OutboxEvent to update
            node_uri: URI of the associated node
            next_retry_at: Optional datetime for scheduled retry (exponential backoff)
        """
        self._check_agent_lease(event)
        node_path = self._uri_to_agfs_path(node_uri)
        event_path = node_path + ".outbox/" + event.event_id + ".json"
        event.retry_count += 1
        event.status = "PENDING"
        if next_retry_at:
            event.next_retry_at = next_retry_at.isoformat()
        data = json.dumps({
            "event_id": event.event_id,
            "event_type": event.event_type,
            "uri": event.uri,
            "payload": event.payload,
            "status": event.status,
            "retry_count": event.retry_count,
            "created_at": event.created_at,
            "next_retry_at": event.next_retry_at,
        }, ensure_ascii=False, indent=2)
        self._client.write(event_path, data.encode("utf-8"))

    def try_acquire(
        self,
        event: "OutboxEvent",
        node_uri: str,
        worker_id: str,
        timeout_seconds: int = 300,
    ) -> bool:
        if event.event_type == EventType.EXTRACT_AGENT_MEMORY.value:
            with self._agent_event_lock(event):
                if self._cleanup_completed_agent_event(event) or not self.is_event_current(event.event_id, node_uri):
                    return False
                return self._try_acquire(event, node_uri, worker_id, timeout_seconds)
        return self._try_acquire(event, node_uri, worker_id, timeout_seconds)

    def _try_acquire(
        self,
        event: "OutboxEvent",
        node_uri: str,
        worker_id: str,
        timeout_seconds: int = 300,
    ) -> bool:
        """Atomically claim an event for processing.

        Creates .outbox/{event_id}.processing lock file.
        Returns False if already locked by another worker within timeout.

        Args:
            event: OutboxEvent to acquire
            node_uri: URI of the associated node
            worker_id: Unique identifier for the processing worker
            timeout_seconds: Seconds before a stale lock is considered expired

        Returns:
            True if lock was acquired, False if already locked
        """
        if (
            event.event_type == EventType.DELETE_CONTEXT.value
            and event.payload.get("durable_delete") is True
        ):
            # Durable DELETE acknowledgement may remove the tombstone.  Cancel
            # every older UPSERT before the DELETE can be claimed so an
            # in-flight stale write will fail its event-liveness check even
            # after acknowledgement releases the URI.
            self.cancel_pending_upserts(node_uri)

        node_path = self._uri_to_agfs_path(node_uri)
        processing_path = node_path + ".outbox/" + event.event_id + ".processing"

        # Check if lock already exists
        try:
            raw = self._client.read(processing_path)
            lock_data = json.loads(raw.decode("utf-8"))
            acquired_at = lock_data.get("acquired_at", "")

            # Parse timestamp to check if expired
            try:
                lock_time = datetime.fromisoformat(acquired_at)
                elapsed = (datetime.now(UTC) - lock_time).total_seconds()
                if elapsed < timeout_seconds:
                    # Lock is still valid
                    return False
            except (ValueError, KeyError):
                # Invalid timestamp, treat as expired
                pass
        except Exception:
            # Lock file doesn't exist, can acquire
            pass

        # Create lock file
        lock_data = {
            "worker_id": worker_id,
            "acquired_at": datetime.now(UTC).isoformat(),
        }
        self._client.write(
            processing_path,
            json.dumps(lock_data, ensure_ascii=False).encode("utf-8")
        )
        return True

    def release(self, event: "OutboxEvent", node_uri: str) -> None:
        """Release the processing lock. Called on success or exception.

        Removes the .outbox/{event_id}.processing lock file.

        Args:
            event: OutboxEvent to release
            node_uri: URI of the associated node
        """
        self._check_agent_lease(event)
        node_path = self._uri_to_agfs_path(node_uri)
        processing_path = node_path + ".outbox/" + event.event_id + ".processing"

        try:
            self._client.rm(processing_path)
        except Exception:
            # Lock file may have been removed already, silently ignore
            pass
