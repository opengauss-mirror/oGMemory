"""Proof of one exact accepted operation, persisted atomically with its node."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from core.errors import NodeNotFoundError, OperationDeadlineExceededError
from core.models import CandidateMemory, RequestContext, WritePlan

RECEIPTS_KEY = "memory_write_receipts"


class WriteNotStartedError(RuntimeError):
    """Planning, authorization or construction rejected before durable mutation."""


class WriteDeadlineExceededError(OperationDeadlineExceededError, WriteNotStartedError):
    """A deadline expired at a proven pre-storage boundary; retry is safe."""


class WriteOutcomeUnknownError(RuntimeError):
    """The storage call failed; its durable outcome must be proven by receipt."""


def operation_digest(candidate: CandidateMemory, ctx: RequestContext) -> str:
    payload = asdict(candidate)
    payload.pop("write_operation_id")
    for name in ("verified_routing_aliases", "verified_object_aliases", "source_contexts", "node_draft_indices"):
        if not payload.get(name):
            payload.pop(name, None)
    if not payload.get("source_occurrence_id"):
        payload.pop("source_occurrence_id", None)
    evidence = {"candidate": payload, "account_id": ctx.account_id,
                "user_id": ctx.user_id, "agent_id": ctx.agent_id,
                "session_id": ctx.session_id}
    return hashlib.sha256(json.dumps(evidence, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_receipt_target(uris, candidate: CandidateMemory, ctx: RequestContext, target_uri: str):
    """Keep a custom policy's durable write inside the receipt discovery scope."""
    schema = uris.schema_registry.get_compatible(candidate.category)
    directory = uris.get_directory_uri(candidate.category, ctx).rstrip("/")
    if schema is None or not uris.validate_uri(target_uri, ctx):
        raise ValueError("durable write target is outside its authorized schema directory")
    if schema.is_single_file:
        allowed = target_uri == uris.resolve(candidate.category, {"routing_key": candidate.routing_key}, ctx)
    else:
        allowed = target_uri.rsplit("/", 1)[0] == directory
    if not allowed:
        raise ValueError("durable write target is outside its authorized schema directory")


def _receipt_candidates(fs, directory, operation_id, ctx):
    page_method = getattr(type(fs), "list_node_summaries_page", None)
    if not callable(page_method):
        yield from fs.list_children(directory, ctx)
        return
    cursor = None
    order_key = getattr(fs, "list_node_order_key", lambda uri: uri)
    while True:
        page = page_method(fs, directory, ctx, limit=100, after_uri=cursor)
        for node in page:
            receipts = node.metadata.get(RECEIPTS_KEY, {})
            if isinstance(receipts, dict) and operation_id in receipts:
                yield node.uri
        if len(page) < 100:
            return
        cursor = order_key(page[-1].uri)


def receipt_plan(fs, uris, candidate: CandidateMemory, ctx: RequestContext):
    """Trust a receipt only after a complete authorized node read.

    Generated occurrence names are deliberately never guessed. Directory scans
    remain inside one schema-owned category and one request owner.
    """
    if not candidate.write_operation_id:
        return None
    schema = uris.schema_registry.get_compatible(candidate.category)
    if schema is None or candidate.owner_scope != schema.owner_scope:
        raise ValueError("receipt category owner does not match its schema")
    if candidate.owner_scope == "agent" and not ctx.agent_id:
        raise ValueError("receipt requires an authorized agent context")
    directory = uris.get_directory_uri(candidate.category, ctx).rstrip("/")
    expected_owner = ctx.agent_space_name() if candidate.owner_scope == "agent" else ctx.user_space_name()
    digest = operation_digest(candidate, ctx)
    if schema.is_single_file:
        candidates = [uris.resolve(candidate.category, {"routing_key": candidate.routing_key}, ctx)]
    else:
        candidates = _receipt_candidates(fs, directory, candidate.write_operation_id, ctx)
    matches = []
    for uri in candidates:
        if (not uris.validate_uri(uri, ctx)
                or (uri.rsplit("/", 1)[0] != directory and not schema.is_single_file)):
            continue
        try:
            node = fs.read_node(uri, ctx)
        except NodeNotFoundError:
            continue
        if node.category != candidate.category or node.owner_space != expected_owner:
            continue
        receipts = node.metadata.get(RECEIPTS_KEY, {})
        if not isinstance(receipts, dict) or candidate.write_operation_id not in receipts:
            continue
        if receipts[candidate.write_operation_id] != digest:
            raise ValueError("operation receipt does not match the accepted candidate or owner")
        matches.append(node.uri)
    if len(matches) > 1:
        raise ValueError("operation receipt has ambiguous committed nodes")
    if matches:
        return WritePlan(action="skip", target_uri=matches[0],
                         merged_fields={"status": "operation_committed"})
    return None


def attach_receipt(fs, node, candidate: CandidateMemory, ctx: RequestContext):
    """Preserve earlier receipts, then add this operation before durable writing."""
    receipts = dict(node.metadata.get(RECEIPTS_KEY, {}))
    read = getattr(type(fs), "read_node", None)
    existing = None
    if callable(read):
        try:
            existing = fs.read_node(node.uri, ctx)
        except NodeNotFoundError:
            pass
    elif candidate.write_operation_id:
        raise RuntimeError("durable operation requires complete node reads")
    if existing is not None:
        previous_receipts = existing.metadata.get(RECEIPTS_KEY, {})
        if isinstance(previous_receipts, dict):
            receipts.update(previous_receipts)
    if candidate.write_operation_id:
        digest = operation_digest(candidate, ctx)
        previous = receipts.get(candidate.write_operation_id)
        if previous is not None and previous != digest:
            raise ValueError("operation ID is already committed for a different candidate")
        receipts[candidate.write_operation_id] = digest
    if receipts:
        node.metadata[RECEIPTS_KEY] = receipts
