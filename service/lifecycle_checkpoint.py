"""Typed lifecycle batches reuse the accepted-candidate and receipt journal."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

from service.extraction_checkpoint import ExtractionBatchJournal, RecoveryNeedsConfirmationError
from session.models import SessionMessage


def lifecycle_kind(buffer):
    batch = buffer.runtime_extensions.get("memory_extraction_recovery", {}).get("batch", {})
    if "lifecycle" not in batch:
        return None
    value = batch["lifecycle"]
    if not isinstance(value, dict) or value.get("kind") not in {"compaction", "dispose"}:
        raise RecoveryNeedsConfirmationError("pending lifecycle checkpoint is invalid")
    return value.get("kind")


def source_digest(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


class LifecycleBatchJournal(ExtractionBatchJournal):
    def __init__(self, manager, buffer, ctx, snapshot, *, kind, **kwargs):
        if kind not in {"compaction", "dispose"}:
            raise ValueError("unsupported lifecycle kind")
        if ctx.session_id != buffer.session_id:
            raise RecoveryNeedsConfirmationError("lifecycle context must match its source session")
        with buffer.extraction_lock:
            expected = [asdict(message) for message in snapshot]
            source_ids = {message.id for message in snapshot}
            actual = [asdict(message) for message in buffer.messages if message.id in source_ids]
            if len(source_ids) != len(snapshot) or actual != expected:
                raise RecoveryNeedsConfirmationError("lifecycle source snapshot changed before checkpoint")
        super().__init__(manager, buffer, ctx, snapshot,
                         lifecycle={"kind": kind, "source_digest": source_digest(
                             [asdict(message) for message in snapshot])}, **kwargs)
        lifecycle = self.batch["lifecycle"]
        if lifecycle.get("source_digest") != source_digest(self.batch["snapshot"]):
            raise RecoveryNeedsConfirmationError("pending lifecycle source digest changed")

    @property
    def snapshot(self):
        return [SessionMessage(**value) for value in self.batch["snapshot"]]

    def prepare_state(self, ttl_seconds):
        value = self.batch["lifecycle"].get("prepared")
        if value is None:
            return None
        if (not isinstance(value, dict) or not isinstance(value.get("token"), str)
                or not value["token"] or value.get("source_digest") != self.batch["lifecycle"]["source_digest"]
                or value.get("archive_id") != self.batch["archive_id"]
                or value.get("owner") != self.batch["owner"]):
            raise RecoveryNeedsConfirmationError("pending compaction token is invalid")
        try:
            created = datetime.fromisoformat(value["created_at"])
            if created.tzinfo is None:
                raise ValueError("token timestamp must include timezone")
        except (KeyError, TypeError, ValueError) as exc:
            raise RecoveryNeedsConfirmationError("pending compaction token timestamp is invalid") from exc
        if not 0 <= (datetime.now(UTC) - created).total_seconds() < ttl_seconds:
            return None
        return {"prepareToken": value["token"], "archive_id": self.batch["archive_id"]}

    def issue_prepare_token(self):
        value = {"token": uuid4().hex, "created_at": datetime.now(UTC).isoformat(),
                 "source_digest": self.batch["lifecycle"]["source_digest"],
                 "archive_id": self.batch["archive_id"], "owner": deepcopy(self.batch["owner"])}
        self.mutate(lambda batch: batch["lifecycle"].update(prepared=value))
        return value["token"]

    def archive_record(self):
        return deepcopy(self.batch["lifecycle"].get("archive"))

    def archive_begin(self, payload):
        self.mutate(lambda batch: batch["lifecycle"].update(
            archive={"status": "writing", "payload": deepcopy(payload)}))

    def archive_complete(self, uri):
        self.mutate(lambda batch: batch["lifecycle"]["archive"].update(status="complete", uri=uri))

    def finish(self, *, reset_window_state=False):
        archive = self.archive_record()
        super().finish(reset_window_state=reset_window_state)
        if archive is not None and archive.get("status") == "complete":
            self.manager._maybe_merge_archives(self.buffer.session_id, self.ctx)

    def archive_recover(self, manager):
        """An uncertain archive is confirmed only by a complete exact read."""
        record = self.archive_record()
        if record is None:
            return None
        if not isinstance(record, dict):
            raise RecoveryNeedsConfirmationError("pending archive checkpoint is invalid")
        payload = record.get("payload")
        if (record.get("status") not in {"writing", "complete"} or not isinstance(payload, dict)
                or not isinstance(payload.get("metadata"), dict)
                or not isinstance(payload.get("overview"), str) or not isinstance(payload.get("abstract"), str)):
            raise RecoveryNeedsConfirmationError("pending archive checkpoint is invalid")
        required_metadata = {
            "memory_lifecycle_source_sha256": self.batch["lifecycle"]["source_digest"],
            "memory_lifecycle_archive_id": self.batch["archive_id"],
            "owner_user_id": self.ctx.user_id, "owner_agent_id": self.ctx.agent_id,
            "session_id": self.ctx.session_id,
        }
        if (payload.get("messages") != self.batch["snapshot"] or any(
                payload["metadata"].get(key) != value for key, value in required_metadata.items())):
            raise RecoveryNeedsConfirmationError("pending archive source checkpoint changed")
        try:
            entry = manager.read_archive_for_recovery(self.buffer.session_id, self.batch["archive_id"], self.ctx)
        except Exception as exc:
            raise RecoveryNeedsConfirmationError("pending archive outcome cannot be read") from exc
        expected_metadata = payload["metadata"]
        if (entry is None or entry.archive_id != self.batch["archive_id"]
                or entry.session_id != self.buffer.session_id
                or entry.messages != payload["messages"]
                or entry.overview != payload["overview"] or entry.abstract != payload["abstract"]
                or any(entry.metadata.get(key) != value for key, value in expected_metadata.items())
                or str(entry.metadata.get("archive_status", entry.metadata.get("status", ""))).upper()
                in {"MERGED", "ARCHIVED", "DELETED"}):
            raise RecoveryNeedsConfirmationError("pending archive outcome needs confirmation")
        uri = f"ctx://{self.ctx.account_id}/sessions/{self.buffer.session_id}/history/{self.batch['archive_id']}"
        self.archive_complete(uri)
        return {"success": True, "uri": uri, "archive_id": self.batch["archive_id"]}
