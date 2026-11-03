"""RepairJob: Scan and repair corrupted AGFS nodes.

Branch logic MUST match AGFS write order spec (docs/architecture/agfs-directory-spec.md).
The write order determines what the repair job can recover:

Write Order (from spec):
  ① content.md (largest file, do first)
  ② .relations.json
  ③ .abstract.md + .overview.md (parallel safe)
  ④ .meta.json with status=ACTIVE ★ COMMIT POINT
  ⑤ Register OutboxEvent to .outbox/{event_id}.json

Repair Branch Logic:
  - .meta.json missing + content.md exists → Repair (recreate .meta.json)
  - .meta.json missing + content.md missing → Skip (step ① not complete)
  - status=PENDING + files complete → Activate (update status=ACTIVE)
  - status=PENDING + files missing → Mark BROKEN (manual intervention)
  - status=ACTIVE → Skip (normal)
  - status=BROKEN → Skip (already marked, requires manual)

IMPORTANT: If write-path-dev changes write order, dfx-dev MUST update this file.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.enums import NodeStatus
from core.models import OutboxEvent

logger = logging.getLogger(__name__)

# Expected files in a complete node directory
_REQUIRED_FILES: tuple[str, ...] = (
    "content.md",
    ".relations.json",
    ".abstract.md",
    ".overview.md",
    ".meta.json",
)

# NodeStatus values used by repair branch logic.
_NODE_STATUS_PENDING = "PENDING"
_NODE_STATUS_ACTIVE = "ACTIVE"
_NODE_STATUS_BROKEN = "BROKEN"
_DELETE_TOMBSTONE_FILE = ".delete_tombstone.json"
_TOOL_OUTPUT_URI_SEGMENT = "/memories/tool_outputs/"


@dataclass
class RepairReport:
    """Report from a single node repair action."""

    uri: str
    action: str  # repaired | activated | marked_pending | marked_broken | skipped | already_broken
    reason: str
    files_present: list[str] = field(default_factory=list)
    files_missing: list[str] = field(default_factory=list)


@dataclass
class RepairJobSummary:
    """Summary of a RepairJob run."""

    total_nodes_scanned: int = 0
    nodes_repaired: int = 0
    nodes_activated: int = 0
    nodes_marked_broken: int = 0
    nodes_skipped: int = 0
    nodes_already_broken: int = 0
    outbox_events_created: int = 0


class AGFSSnapshot:
    """Read-only snapshot of AGFS for RepairJob scanning.

    Abstract interface to allow testing without real AGFS connection.
    Real implementation will use agfs-python-client.
    """

    def list_nodes_under_account(self, account_id: str) -> list[str]:
        """List all node URIs under an account.

        Returns:
            List of URIs like "ctx://acme/users/alice/memories/profile"
        """
        raise NotImplementedError

    def read_file(self, uri: str, filename: str) -> str | None:
        """Read a file from a node directory.

        Args:
            uri: Node URI
            filename: File name (e.g., ".meta.json")

        Returns:
            File content as string, or None if file doesn't exist
        """
        raise NotImplementedError

    def write_file(self, uri: str, filename: str, content: str) -> None:
        """Write a file to a node directory.

        Args:
            uri: Node URI
            filename: File name (e.g., ".meta.json")
            content: File content to write
        """
        raise NotImplementedError

    def list_files(self, uri: str) -> list[str]:
        """List all files in a node directory.

        Args:
            uri: Node URI

        Returns:
            List of filenames in the directory
        """
        raise NotImplementedError


class MemoryAGFSSnapshot(AGFSSnapshot):
    """In-memory AGFS snapshot for testing.

    Simulates AGFS structure using a dict of URI → files.
    """

    def __init__(self) -> None:
        self._storage: dict[str, dict[str, str]] = {}

    def add_node(self, uri: str, files: dict[str, str]) -> None:
        """Add a node to the snapshot."""
        self._storage[uri] = files

    def list_nodes_under_account(self, account_id: str) -> list[str]:
        """List all node URIs under an account."""
        prefix = f"ctx://{account_id}/"
        return [uri for uri in self._storage.keys() if uri.startswith(prefix)]

    def read_file(self, uri: str, filename: str) -> str | None:
        """Read a file from a node directory."""
        if uri not in self._storage:
            return None
        return self._storage[uri].get(filename)

    def write_file(self, uri: str, filename: str, content: str) -> None:
        """Write a file to a node directory."""
        if uri not in self._storage:
            self._storage[uri] = {}
        self._storage[uri][filename] = content

    def list_files(self, uri: str) -> list[str]:
        """List all files in a node directory."""
        if uri not in self._storage:
            return []
        return list(self._storage[uri].keys())


class OutboxEventSink:
    """Interface for creating OutboxEvents during repair.

    Abstract interface to allow testing without real OutboxStore.
    """

    def create_event(self, event: OutboxEvent) -> None:
        """Create an OutboxEvent for later processing.

        Args:
            event: OutboxEvent to create
        """
        raise NotImplementedError


class MemoryOutboxEventSink(OutboxEventSink):
    """In-memory event sink for testing."""

    def __init__(self) -> None:
        self.events: list[OutboxEvent] = []

    def create_event(self, event: OutboxEvent) -> None:
        """Store event in memory."""
        self.events.append(event)


class RepairJob:
    """Scan and repair corrupted AGFS nodes.

    The repair job scans all nodes under accounts and applies
    branch logic based on node state and file presence.

    Branch logic MUST correspond to AGFS write order spec.
    """

    def __init__(
        self,
        agfs_snapshot: AGFSSnapshot,
        event_sink: OutboxEventSink,
    ) -> None:
        """Initialize RepairJob.

        Args:
            agfs_snapshot: Read-only AGFS snapshot for scanning
            event_sink: Sink for creating OutboxEvents
        """
        self._agfs = agfs_snapshot
        self._event_sink = event_sink

    def scan_account(self, account_id: str) -> RepairJobSummary:
        """Scan all nodes under an account and repair as needed.

        Args:
            account_id: Account to scan

        Returns:
            RepairJobSummary with action counts
        """
        summary = RepairJobSummary()

        node_uris = self._agfs.list_nodes_under_account(account_id)
        summary.total_nodes_scanned = len(node_uris)

        for uri in node_uris:
            report = self._repair_node(uri)
            summary.outbox_events_created += self._handle_repair_report(report, uri)

            # Update summary counters
            if report.action == "repaired":
                summary.nodes_repaired += 1
            elif report.action == "activated":
                summary.nodes_activated += 1
            elif report.action == "marked_broken":
                summary.nodes_marked_broken += 1
            elif report.action == "skipped":
                summary.nodes_skipped += 1
            elif report.action == "already_broken":
                summary.nodes_already_broken += 1

        logger.info(
            f"RepairJob scan for account {account_id} complete: "
            f"{summary.nodes_repaired} repaired, {summary.nodes_activated} activated, "
            f"{summary.nodes_marked_broken} broken, {summary.nodes_skipped} skipped"
        )

        return summary

    def _repair_node(self, uri: str) -> RepairReport:
        """Apply repair logic to a single node.

        Branch logic corresponds to AGFS write order:
        - .meta.json missing → check content.md exists → repair or skip
        - status=PENDING → check files complete → activate or mark BROKEN
        - status=ACTIVE → skip
        - status=BROKEN → skip

        Args:
            uri: Node URI to repair

        Returns:
            RepairReport with action taken
        """
        files = self._agfs.list_files(uri)

        if _DELETE_TOMBSTONE_FILE in files:
            return RepairReport(
                uri=uri,
                action="skipped",
                reason="Node has a durable delete tombstone",
                files_present=files,
            )
        if _TOOL_OUTPUT_URI_SEGMENT in uri:
            return RepairReport(
                uri=uri,
                action="skipped",
                reason="Tool-output nodes are excluded from generic repair indexing",
                files_present=files,
            )

        # Branch 1: .meta.json missing
        if ".meta.json" not in files:
            return self._handle_missing_metadata(uri, files)

        # Read .meta.json
        meta_content = self._agfs.read_file(uri, ".meta.json")
        if not meta_content:
            return RepairReport(
                uri=uri,
                action="marked_broken",
                reason=".meta.json exists but is unreadable",
                files_present=files,
            )

        try:
            meta = json.loads(meta_content)
        except json.JSONDecodeError:
            return RepairReport(
                uri=uri,
                action="marked_broken",
                reason=".meta.json contains invalid JSON",
                files_present=files,
            )

        if meta.get("indexing_disabled") is True or meta.get("category") == "tool_outputs":
            return RepairReport(
                uri=uri,
                action="skipped",
                reason="Node metadata disables generic indexing",
                files_present=files,
            )

        status = meta.get("status", "")

        # Branch 2: status=PENDING
        if status == _NODE_STATUS_PENDING:
            return self._handle_pending_node(uri, files, meta)

        # Branch 3: status=ACTIVE
        if status == _NODE_STATUS_ACTIVE:
            return RepairReport(
                uri=uri,
                action="skipped",
                reason="Node is ACTIVE (normal)",
                files_present=files,
            )

        # Branch 4: status=BROKEN
        if status == _NODE_STATUS_BROKEN:
            return RepairReport(
                uri=uri,
                action="already_broken",
                reason="Node already marked BROKEN (requires manual intervention)",
                files_present=files,
            )

        if status == NodeStatus.ARCHIVED.value:
            return RepairReport(
                uri=uri,
                action="skipped",
                reason="Node is ARCHIVED (soft-deleted)",
                files_present=files,
            )

        # Unknown status
        return RepairReport(
            uri=uri,
            action="marked_broken",
            reason=f"Unknown status: {status}",
            files_present=files,
        )

    def _handle_missing_metadata(self, uri: str, files: list[str]) -> RepairReport:
        """Handle node with missing .meta.json.

        Branch logic (corresponds to write order):
        - content.md exists → step ① completed, can repair
        - content.md missing → step ① not complete, skip (normal interruption)

        Args:
            uri: Node URI
            files: List of files present

        Returns:
            RepairReport with action taken
        """
        # SECURITY: Check ALL required files from steps 1-3 before marking ACTIVE
        # Per AGFS write order spec: only mark ACTIVE when all files present
        required_before_meta = ["content.md", ".relations.json", ".abstract.md", ".overview.md"]
        missing = [f for f in required_before_meta if f not in files]

        if not missing:
            # All files from steps 1-3 present, can safely write meta.json with ACTIVE
            self._write_meta_json(uri, status=_NODE_STATUS_ACTIVE)
            return RepairReport(
                uri=uri,
                action="repaired",
                reason=".meta.json missing but all required files exist, marking ACTIVE",
                files_present=files,
            )
        elif "content.md" in files:
            # Content exists but other files missing - mark PENDING, not ACTIVE
            # This allows the write pipeline to resume from step 2
            self._write_meta_json(uri, status=_NODE_STATUS_PENDING)
            return RepairReport(
                uri=uri,
                action="marked_pending",
                reason=f".meta.json missing, files missing: {missing}, marking PENDING",
                files_present=files,
                files_missing=missing,
            )
        else:
            # Skip: content.md missing, step ① not complete
            # This is a normal interruption, nothing to repair
            return RepairReport(
                uri=uri,
                action="skipped",
                reason=".meta.json and content.md both missing (step ① not complete)",
                files_present=files,
            )

    def _handle_pending_node(
        self,
        uri: str,
        files: list[str],
        meta: dict[str, Any],
    ) -> RepairReport:
        """Handle node with status=PENDING.

        Branch logic (corresponds to write order):
        - All required files present → activate (update status=ACTIVE)
        - Some files missing → mark BROKEN (corruption detected)

        Args:
            uri: Node URI
            files: List of files present
            meta: Parsed .meta.json content

        Returns:
            RepairReport with action taken
        """
        missing = [f for f in _REQUIRED_FILES if f not in files]

        if not missing:
            # All files present, activate the node
            self._update_meta_status(uri, _NODE_STATUS_ACTIVE)
            return RepairReport(
                uri=uri,
                action="activated",
                reason="All files present, activated PENDING node",
                files_present=files,
            )
        else:
            # Files missing, mark as BROKEN
            self._update_meta_status(uri, _NODE_STATUS_BROKEN)
            return RepairReport(
                uri=uri,
                action="marked_broken",
                reason=f"PENDING node missing files: {missing}",
                files_present=files,
                files_missing=missing,
            )

    def _write_meta_json(self, uri: str, status: str) -> None:
        """Write a .meta.json file to a node directory.

        Args:
            uri: Node URI
            status: NodeStatus value
        """
        # Extract basic info from URI
        account_match = re.match(r"^ctx://([^/]+)/", uri)
        if not account_match:
            logger.error(f"Invalid URI: {uri}")
            return

        # SECURITY: Extract owner_space from URI to maintain tenant isolation
        # URI format: ctx://{account}/{owner_type}s/{owner_id}/memories/{category}/{slug}
        # or: ctx://{account}/{owner_type}s/{owner_id}/skills/{skill_name}
        owner_space = "unknown_space"  # fallback
        owner_match = re.match(r"^ctx://[^/]+/(users|agents)/([^/]+)", uri)
        if owner_match:
            owner_type = owner_match.group(1)  # "users" or "agents"
            owner_id = owner_match.group(2)
            owner_space = f"{owner_type.rstrip('s')}:{owner_id}"  # "user:alice" or "agent:bot1"

        # Determine context_type and category from URI
        if "/agents/" in uri and "/skills/" in uri:
            context_type = "SKILL"
            category = "skill"
            level = 3  # agents/{agent}/skills/{skill_name}
        elif "/agents/" in uri:
            context_type = "MEMORY"
            if "/cases/" in uri:
                category = "case"
                level = 4  # agents/{agent}/memories/cases/{case_id}
            elif "/patterns/" in uri:
                category = "pattern"
                level = 4  # agents/{agent}/memories/patterns/{slug}
            else:
                category = "case"  # Default for agent memories
                level = 4
        else:
            context_type = "MEMORY"
            if "/profile" in uri:
                category = "profile"
                level = 3  # users/{user}/memories/profile
            elif "/preferences/" in uri:
                category = "preference"
                level = 4  # users/{user}/memories/preferences/{slug}
            elif "/entities/" in uri:
                category = "entity"
                level = 4
            elif "/events/" in uri:
                category = "event"
                level = 4
            else:
                category = "profile"  # Default
                level = 3

        # SECURITY WARNING: If owner_space is still "unknown_space", this node
        # will not be searchable via vector index (filters won't match).
        # Consider marking as BROKEN if this is a production system.
        if owner_space == "unknown_space":
            logger.warning(
                f"Could not extract owner_space from URI: {uri}. "
                "Node may not be searchable. Consider manual review."
            )

        meta = {
            "uri": uri,
            "context_type": context_type,
            "category": category,
            "level": level,
            "owner_space": owner_space,
            "status": status,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "version": 1,
            "tags": ["repaired"],
        }

        content = json.dumps(meta, indent=2)
        self._agfs.write_file(uri, ".meta.json", content)

    def _update_meta_status(self, uri: str, status: str) -> None:
        """Update the status in a .meta.json file.

        Args:
            uri: Node URI
            status: New status value
        """
        meta_content = self._agfs.read_file(uri, ".meta.json")
        if not meta_content:
            self._write_meta_json(uri, status)
            return

        try:
            meta = json.loads(meta_content)
        except json.JSONDecodeError:
            self._write_meta_json(uri, status)
            return

        meta["status"] = status
        meta["updated_at"] = datetime.now(UTC).isoformat()

        if "tags" not in meta:
            meta["tags"] = []
        if "repaired" not in meta["tags"]:
            meta["tags"].append("repaired")

        content = json.dumps(meta, indent=2)
        self._agfs.write_file(uri, ".meta.json", content)

    def _handle_repair_report(self, report: RepairReport, uri: str) -> int:
        """Handle repair report by creating OutboxEvents if needed.

        Args:
            report: RepairReport from node repair
            uri: Node URI

        Returns:
            Number of OutboxEvents created
        """
        # Re-check durable no-index markers at event creation time.  This is a
        # second line of defence against a marker appearing after the initial
        # scan or a caller passing a synthetic RepairReport directly.
        files = self._agfs.list_files(uri)
        if _DELETE_TOMBSTONE_FILE in files or _TOOL_OUTPUT_URI_SEGMENT in uri:
            return 0
        meta_content = self._agfs.read_file(uri, ".meta.json")
        if meta_content:
            try:
                meta = json.loads(meta_content)
            except json.JSONDecodeError:
                meta = {}
            if (
                meta.get("indexing_disabled") is True
                or meta.get("category") == "tool_outputs"
            ):
                return 0

        # Create OutboxEvent for repaired/activated nodes
        if report.action in ("repaired", "activated"):
            event = OutboxEvent(
                event_id=str(uuid.uuid4()),
                event_type="UPSERT_CONTEXT",
                uri=uri,
                payload={
                    "reason": report.reason,
                    "repair_action": report.action,
                },
                status="PENDING",
                retry_count=0,
            )
            self._event_sink.create_event(event)
            return 1

        # Log broken nodes for alerting
        if report.action == "marked_broken":
            logger.error(
                f"RepairJob marked node as BROKEN: {uri} - {report.reason}"
            )
            # TODO: Trigger alert (integration with monitoring system)

        return 0
