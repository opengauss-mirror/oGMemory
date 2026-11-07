"""Durable source batches and accepted candidates for extraction recovery."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace

from commit.outbox_scope import current_outbox_events, restore_outbox_events
from commit.write_receipts import operation_digest
from service.model_stage import stage_digest
from core.write_contract import OUTCOMES_KEY, store_outcome, merge_source_outcome
from core.models import CandidateMemory

EXTENSION_KEY = "memory_extraction_recovery"


class RecoveryNeedsConfirmationError(RuntimeError):
    """No durable evidence distinguishes an uncommitted write from deletion."""


class RecoveryPersistenceError(RuntimeError):
    """A checkpoint was not durably acknowledged; no new write is allowed."""


def recovery_owner(ctx):
    return {"account_id": ctx.account_id, "user_id": ctx.user_id,
            "agent_id": ctx.agent_id, "session_id": ctx.session_id}


def request_message_id(ctx, request_id, index):
    payload = {"owner": recovery_owner(ctx), "request_id": request_id, "index": index}
    return "msg_request_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ExtractionBatchJournal:
    @classmethod
    def resume(cls, manager, buffer, ctx):
        extension = buffer.runtime_extensions.get(EXTENSION_KEY, {})
        batch = extension.get("batch")
        if (extension.get("version") != 1 or not isinstance(batch, dict)
                or batch.get("owner") != recovery_owner(ctx)):
            raise RecoveryNeedsConfirmationError("invalid pending extraction owner or format")
        journal = cls.__new__(cls)
        journal.manager, journal.buffer, journal.ctx = manager, buffer, ctx
        journal._archive_id = batch["archive_id"]
        restore_outbox_events(batch, ctx.account_id)
        return journal

    def __init__(self, manager, buffer, ctx, snapshot, *, archive_id,
                 messages, session_time, session_summary, tool_stats_text, client_request_id="",
                 request_messages=None, lifecycle=None, request_source_ids=None):
        self.manager, self.buffer, self.ctx = manager, buffer, ctx
        extension = buffer.runtime_extensions.get(EXTENSION_KEY, {})
        batch = extension.get("batch")
        if batch is not None:
            if (extension.get("version") != 1 or not isinstance(batch, dict)
                    or batch.get("owner") != recovery_owner(ctx)):
                raise RecoveryNeedsConfirmationError("pending extraction batch has a different owner or format")
            actual = [asdict(message) for message in snapshot]
            if actual != batch.get("snapshot"):
                raise RecoveryNeedsConfirmationError("pending extraction source snapshot changed")
            stored_lifecycle = batch.get("lifecycle")
            if ("lifecycle" in batch and (not isinstance(stored_lifecycle, dict)
                    or stored_lifecycle.get("kind") not in {"compaction", "dispose"})):
                raise RecoveryNeedsConfirmationError("pending extraction lifecycle is invalid")
            if ((stored_lifecycle or {}).get("kind") != (lifecycle or {}).get("kind")):
                raise RecoveryNeedsConfirmationError("pending extraction batch belongs to another lifecycle")
        else:
            batch = {"owner": recovery_owner(ctx), "archive_id": archive_id,
                     "client_request_id": client_request_id,
                     "request_messages": deepcopy(request_messages or []),
                     "request_source_ids": list(request_source_ids) if request_source_ids is not None else [message.id for message in snapshot],
                     "message_ids": [message.id for message in snapshot],
                     "snapshot": [asdict(message) for message in snapshot],
                     "messages": deepcopy(messages),
                     "session_time": session_time.isoformat() if hasattr(session_time, "isoformat") else session_time,
                     "session_summary": session_summary, "tool_stats_text": tool_stats_text,
                     "chunks": None, "slots": {}, "outbox_tracking_version": 1,
                     "outbox_events": []}
            if lifecycle is not None:
                batch["lifecycle"] = deepcopy(lifecycle)
            self._replace(batch)
        self._archive_id = batch["archive_id"]
        restore_outbox_events(batch, ctx.account_id)

    @property
    def batch(self):
        batch = self.buffer.runtime_extensions.get(EXTENSION_KEY, {}).get("batch")
        if (not isinstance(batch, dict) or batch.get("archive_id") != self._archive_id
                or batch.get("owner") != recovery_owner(self.ctx)):
            raise RecoveryNeedsConfirmationError("pending extraction batch changed or completed")
        return batch

    def _replace(self, batch):
        with self.buffer.session_state_save_lock:
            if batch is not None:
                batch = dict(batch)
                references = [tuple(item) for item in batch.get("outbox_events", [])]
                current = current_outbox_events(self.ctx.account_id)
                if current is None:
                    batch.pop("outbox_tracking_version", None)
                references.extend(current or [])
                batch["outbox_events"] = [list(item) for item in dict.fromkeys(references)]
            with self.buffer.extraction_lock:
                prior = self.buffer.runtime_extensions.get(EXTENSION_KEY)
                if batch is None:
                    self.buffer.runtime_extensions.pop(EXTENSION_KEY, None)
                else:
                    self.buffer.runtime_extensions[EXTENSION_KEY] = {"version": 1, "batch": deepcopy(batch)}
            if not self.manager.save_session_state(self.buffer.session_id, self.ctx):
                with self.buffer.extraction_lock:
                    if prior is None:
                        self.buffer.runtime_extensions.pop(EXTENSION_KEY, None)
                    else:
                        self.buffer.runtime_extensions[EXTENSION_KEY] = prior
                raise RecoveryPersistenceError("extraction checkpoint could not be saved")

    def mutate(self, update):
        with self.buffer.session_state_save_lock:
            batch = deepcopy(self.batch)
            update(batch)
            self._replace(batch)

    def chunks(self, factory):
        chunks = self.batch.get("chunks")
        if chunks is None:
            chunks = factory()
            self.mutate(lambda batch: batch.update(chunks=deepcopy(chunks)))
        return deepcopy(chunks)

    def checkpoint(self, slot):
        return CandidateCheckpoint(self, str(slot))

    def has_retryable_write_failure(self):
        with self.buffer.session_state_save_lock:
            statuses = {operation.get("status")
                        for slot in self.batch["slots"].values()
                        for operation in slot["operations"].values()}
            return ("rejected_before_write" in statuses
                    and statuses <= {"complete", "rejected_before_write"})

    def finish(self, *, reset_window_state=False, outcome=None, resolve_outcome=None,
               current_request_id="", current_source_ids=()):
        """Clear source claim, pending drafts and request ID in one durable state."""
        with self.buffer.session_state_save_lock:
            with self.buffer.extraction_lock:
                batch = self.batch
                rollback = self.buffer.capture_rollback_state()
                old_completed = list(self.buffer.completed_request_ids)
                old_extension = self.buffer.runtime_extensions[EXTENSION_KEY]
                old_outcomes = deepcopy(self.buffer.runtime_extensions.get(OUTCOMES_KEY))
                client_request_id = batch.get("client_request_id", "")
                receipt = deepcopy(outcome or {})
                references = [tuple(item) for item in batch.get("outbox_events", [])]
                references.extend(current_outbox_events(self.ctx.account_id) or [])
                receipt.update(request_id=client_request_id, archive_id=batch["archive_id"],
                               source_sha256=stage_digest(batch["messages"]),
                               outbox_events=[list(item) for item in dict.fromkeys(references)],
                               outbox_tracking_complete=batch.get("outbox_tracking_version") == 1)
                receipt.setdefault("source_status", "ARCHIVED")
                if batch.get("lifecycle", {}).get("source_digest"):
                    receipt["source_snapshot_sha256"] = batch["lifecycle"]["source_digest"]
                receipt.setdefault("semantic_status", "UNKNOWN")
                receipt.setdefault("projection_status", "UNKNOWN")
                receipt.setdefault("request_status", "PROJECTING")
                batch_ids = set(batch["message_ids"])
                self.buffer.remove_messages_by_id(batch_ids)
                if (batch.get("lifecycle", {}).get("kind") == "compaction"
                        and self.buffer.compaction_prepare_archive_id == batch["archive_id"]):
                    self.buffer.compaction_prepare_archive_id = ""
                if reset_window_state and not self.buffer.messages:
                    from session.models import SessionWindowState
                    self.buffer.window_state = SessionWindowState()
                if client_request_id:
                    self.buffer.mark_request_completed(client_request_id)
                pending_receipts = dict(self.buffer.runtime_extensions.get(OUTCOMES_KEY, {}).get("entries", {}))
                batch_receipt = receipt
                if client_request_id not in pending_receipts and client_request_id:
                    pending_receipts[client_request_id] = {"source_status": "BUFFERED",
                        "source_message_ids": batch.get("request_source_ids", batch["message_ids"])}
                if (current_request_id and current_request_id not in pending_receipts
                        and not batch_ids.isdisjoint(current_source_ids)):
                    pending_receipts[current_request_id] = {"source_status": "BUFFERED",
                        "source_message_ids": sorted(current_source_ids)}
                for request_id, pending_receipt in pending_receipts.items():
                    source_ids = set(pending_receipt.get("source_message_ids", []))
                    if (pending_receipt.get("source_status") == "BUFFERED" and source_ids
                            and not batch_ids.isdisjoint(source_ids)) or request_id == client_request_id:
                        merged = merge_source_outcome(pending_receipt, {**batch_receipt, "request_id": request_id}, batch["message_ids"])
                        if request_id == current_request_id and request_id != client_request_id:
                            self.buffer.mark_request_completed(request_id)
                        merged = store_outcome(self.buffer, request_id, merged,
                                               resolve_outcome=resolve_outcome)
                        if request_id == client_request_id:
                            receipt = merged
                self.buffer.runtime_extensions.pop(EXTENSION_KEY)
            if not self.manager.save_session_state(self.buffer.session_id, self.ctx):
                with self.buffer.extraction_lock:
                    self.buffer.restore_rollback_state(rollback)
                    self.buffer.completed_request_ids = old_completed
                    self.buffer.runtime_extensions[EXTENSION_KEY] = old_extension
                    if old_outcomes is None:
                        self.buffer.runtime_extensions.pop(OUTCOMES_KEY, None)
                    else:
                        self.buffer.runtime_extensions[OUTCOMES_KEY] = old_outcomes
                raise RecoveryPersistenceError("completed extraction checkpoint could not be saved")
            return receipt


class CandidateCheckpoint:
    def __init__(self, journal, slot):
        self.journal, self.slot = journal, slot

    @property
    def record(self):
        return self.journal.batch["slots"].get(self.slot)

    def load(self):
        record = self.record
        if record is None:
            return None
        try:
            candidates = [CandidateMemory(**value) for value in record["candidates"]]
            for index, candidate in enumerate(candidates):
                revision = record.get("revisions", {}).get(str(index), 0)
                suffix = f":revision:{revision}" if revision else ""
                expected = hashlib.sha256(
                    f"{self.journal.batch['archive_id']}:{self.slot}:{index}{suffix}".encode()).hexdigest()
                identity_version = record.get("operation_id_versions", {}).get(
                    str(index), record.get("operation_id_version", 1))
                if identity_version == 2:
                    expected = "receipt-v2:" + expected
                elif identity_version != 1:
                    raise ValueError("operation identity version is unsupported")
                if candidate.write_operation_id != expected:
                    raise ValueError("operation identity changed")
                if operation_digest(candidate, self.journal.ctx) != record["candidate_digests"][index]:
                    raise ValueError("accepted operation payload changed")
            return candidates, record["extracted"], record["filtered"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RecoveryNeedsConfirmationError("accepted extraction checkpoint is invalid") from exc

    def accept(self, candidates, extracted, filtered):
        if self.record is not None:
            raise RecoveryNeedsConfirmationError("accepted extraction candidates cannot be replaced")
        batch = self.journal.batch
        accepted = [replace(candidate, write_operation_id="receipt-v2:" + hashlib.sha256(
            f"{batch['archive_id']}:{self.slot}:{index}".encode()).hexdigest())
                    for index, candidate in enumerate(candidates)]
        record = {"candidates": [asdict(candidate) for candidate in accepted],
                  "candidate_digests": [operation_digest(candidate, self.journal.ctx) for candidate in accepted],
                  "extracted": extracted, "filtered": filtered, "operations": {},
                  "operation_id_version": 2, "relation_tracking": True}
        self.journal.mutate(lambda value: value["slots"].update({self.slot: record}))
        return accepted

    def restore_relation_tracking(self):
        record = self.record
        if ("relation_tracking" not in record and record["operations"]
                and len(record["operations"]) == len(record["candidates"])
                and self.journal.batch.get("outbox_tracking_version") == 1):
            self.journal.mutate(lambda batch: batch.update(outbox_tracking_version=0))

    def begin_relation_write(self):
        if self.journal.batch.get("outbox_tracking_version") != 1:
            return False
        self.journal.mutate(lambda batch: batch.update(outbox_tracking_version=0))
        return True

    def complete_relation_write(self):
        self.journal.mutate(lambda batch: batch.update(outbox_tracking_version=1))

    def load_drafts(self):
        record = self.journal.batch.get("node_drafts", {}).get(self.slot)
        return [CandidateMemory(**c) for c in record["candidates"]] if record else None

    def draft_source(self):
        return self.journal.batch.get("node_drafts", {}).get(self.slot, {}).get("messages", [])

    def save_drafts(self, candidates, messages):
        record = {"candidates": [asdict(c) for c in candidates], "messages": messages}
        self.journal.mutate(lambda batch: batch.setdefault("node_drafts", {}).update({self.slot: record}))

    def reprocessing_drafts(self, candidate):
        """Replay the contributing source spans against a refreshed target snapshot."""
        if not candidate.node_draft_indices:
            return [replace(candidate, target_uri="", expected_version=None, write_operation_id="")]
        saved = self.load_drafts() or []
        if any(type(i) is not int or not 0 <= i < len(saved) for i in candidate.node_draft_indices):
            raise RecoveryNeedsConfirmationError("node draft checkpoint references are invalid")
        return [replace(saved[i], target_uri=candidate.target_uri, expected_version=None,
                        verified_routing_aliases=list(dict.fromkeys([
                            *saved[i].verified_routing_aliases, *candidate.verified_routing_aliases])),
                        verified_object_aliases=list(dict.fromkeys([
                            *saved[i].verified_object_aliases, *candidate.verified_object_aliases])))
                for i in candidate.node_draft_indices]

    def revise_unwritten(self, index, candidate):
        if self._operation(index).get("status") != "rejected_before_write":
            raise RecoveryNeedsConfirmationError("node revision requires confirmed pre-write rejection")
        revision = self.record.get("revisions", {}).get(str(index), 0) + 1
        identity = f"{self.journal.batch['archive_id']}:{self.slot}:{index}:revision:{revision}"
        candidate = replace(candidate, write_operation_id="receipt-v2:" + hashlib.sha256(identity.encode()).hexdigest())
        def revise(batch):
            record = batch["slots"][self.slot]
            record.setdefault("revision_history", []).append({"index": index,
                "candidate": record["candidates"][index], "operation": record["operations"][str(index)]})
            record.setdefault("revisions", {})[str(index)] = revision
            record.setdefault("operation_id_versions", {})[str(index)] = 2
            record["candidates"][index] = asdict(candidate)
            record["candidate_digests"][index] = operation_digest(candidate, self.journal.ctx)
            record["operations"].pop(str(index))
        self.journal.mutate(revise)
        return candidate

    def _operation(self, index):
        return self.record["operations"].get(str(index), {})

    def completed(self, index):
        operation = self._operation(index)
        return deepcopy(operation["plan"]) if operation.get("status") == "complete" else None

    def outcome_uncertain(self, index):
        return self._operation(index).get("status") == "writing"

    def _set_operation(self, index, operation):
        self.journal.mutate(lambda batch: batch["slots"][self.slot]["operations"].update(
            {str(index): operation}))

    def begin(self, index):
        self._set_operation(index, {"status": "writing"})

    def reject(self, index, error):
        self._set_operation(index, {"status": "rejected_before_write", "error": error})

    def complete(self, index, plan):
        # Recovery needs the acknowledged action and target, not another copy of
        # full memory bodies and accumulated identity metadata in session state.
        saved = {"action": plan.action, "target_uri": plan.target_uri,
                 "merged_fields": {key: deepcopy(plan.merged_fields[key])
                                   for key in ("status", "outbox_events", "projection_evidence_known")
                                   if key in plan.merged_fields}, "relation_edges": []}
        if plan.merged_fields.get("projection_evidence_known") is False:
            self.journal.mutate(lambda batch: batch.update(outbox_tracking_version=0))
        self._set_operation(index, {"status": "complete", "plan": saved})
