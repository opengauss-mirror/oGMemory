"""Proof of one exact accepted operation, persisted atomically with its node."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from core.errors import NodeNotFoundError, OperationDeadlineExceededError
from core.models import CandidateMemory, ContextNode, RequestContext, WritePlan

RECEIPTS_KEY = "memory_write_receipts"
PROJECTION_RECEIPTS_KEY = "memory_write_projection_receipts"
LOCATED_OPERATION_PREFIX = "receipt-v2:"


def _locator_uri(uris, candidate: CandidateMemory, ctx: RequestContext) -> str:
    uris.get_directory_uri(candidate.category, ctx)
    owner_type, owner_id = (("agents", ctx.agent_id) if candidate.owner_scope == "agent"
                            else ("users", ctx.user_id))
    key = hashlib.sha256(json.dumps([candidate.category, candidate.write_operation_id]).encode()).hexdigest()
    return f"ctx://{ctx.account_id}/{owner_type}/{owner_id}/memories/.write_operations/{key}"


def located_target(fs, uris, candidate: CandidateMemory, ctx: RequestContext) -> str | None:
    """Resolve a versioned operation's authorized, content-bound target."""
    if not candidate.write_operation_id.startswith(LOCATED_OPERATION_PREFIX):
        return None
    try:
        locator = fs.read_node(_locator_uri(uris, candidate, ctx), ctx)
    except NodeNotFoundError:
        return None
    owner = ctx.agent_space_name() if candidate.owner_scope == "agent" else ctx.user_space_name()
    payload = json.loads(locator.content)
    if (locator.context_type != "RESOURCE" or locator.category != "write_operation"
            or locator.owner_space != owner or not isinstance(payload, dict)
            or payload.get("operation_id") != candidate.write_operation_id
            or payload.get("category") != candidate.category
            or payload.get("digest") != operation_digest(candidate, ctx)):
        raise ValueError("operation locator does not match the accepted candidate or owner")
    target_uri = payload.get("target_uri")
    if not isinstance(target_uri, str):
        raise ValueError("operation locator requires a target URI")
    validate_receipt_target(uris, candidate, ctx, target_uri)
    return target_uri


def bind_operation_target(fs, uris, candidate: CandidateMemory, ctx: RequestContext, target_uri: str):
    """Publish an immutable target binding under the category lock before source writing.

    Versioned operations acquire this binding before their source receipt can
    exist. The source node remains the proof of a completed operation.
    """
    if not candidate.write_operation_id.startswith(LOCATED_OPERATION_PREFIX):
        return
    schema = uris.schema_registry.get_compatible(candidate.category)
    if schema.is_single_file:
        return
    validate_receipt_target(uris, candidate, ctx, target_uri)
    existing = located_target(fs, uris, candidate, ctx)
    if existing is not None:
        if existing != target_uri:
            raise ValueError("write plan changed the durable operation target")
        return
    payload = {"operation_id": candidate.write_operation_id, "category": candidate.category,
               "digest": operation_digest(candidate, ctx), "target_uri": target_uri}
    owner = ctx.agent_space_name() if candidate.owner_scope == "agent" else ctx.user_space_name()
    fs.write_node(ContextNode(
        uri=_locator_uri(uris, candidate, ctx), context_type="RESOURCE", category="write_operation",
        level=0, owner_space=owner, abstract="", overview="",
        content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        metadata={"indexing_disabled": True, "_expected_version": 0},
    ), ctx)


def has_stored_target(plan: WritePlan) -> bool:
    """A proven stored endpoint, not an unsuccessful or unverified skip.

    Callers still authorize/hydrate target_uri before consuming its contents.
    """
    return bool(plan.target_uri) and (plan.action in {"create", "merge", "append"}
        or (plan.action == "skip" and plan.merged_fields.get("status") in
            {"operation_committed", "duplicate"}))


class WriteNotStartedError(RuntimeError):
    """Planning, authorization or construction rejected before durable mutation."""


class WriteDeadlineExceededError(OperationDeadlineExceededError, WriteNotStartedError):
    """A deadline expired at a proven pre-storage boundary; retry is safe."""


class WriteOutcomeUnknownError(RuntimeError):
    """The storage call failed; its durable outcome must be proven by receipt."""


def operation_digest(candidate: CandidateMemory, ctx: RequestContext) -> str:
    payload = asdict(candidate)
    from extraction.relation_proposals import KEY
    metadata = dict(payload.get("metadata") or {})
    metadata.pop(KEY, None)  # transient edge inputs are not a node-write payload
    payload["metadata"] = metadata
    # New opt-in annotations must not invalidate pre-graph durable receipts.
    if not payload.get("metadata"):
        payload.pop("metadata", None)
    payload.pop("write_operation_id")
    payload.pop("node_reconciliation_role", None)
    payload.pop("node_reconciliation_group", None)
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


def receipt_plan(fs, uris, candidate: CandidateMemory, ctx: RequestContext, *, outbox_store=None):
    """Trust a receipt only after a complete authorized node read.

    Versioned operations resolve their durable target binding. Legacy directory
    discovery remains inside one schema-owned category and one request owner.
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
    elif candidate.write_operation_id.startswith(LOCATED_OPERATION_PREFIX):
        target = located_target(fs, uris, candidate, ctx)
        candidates = [target] if target is not None else []
    else:
        candidates = _receipt_candidates(fs, directory, candidate.write_operation_id, ctx)
    matches = []
    projection_refs = []
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
        projection = node.metadata.get(PROJECTION_RECEIPTS_KEY, {}).get(candidate.write_operation_id)
        if isinstance(projection, dict) and projection.get("event_id"):
            if outbox_store is not None:
                status = outbox_store.get_event_status(node.uri, projection["event_id"])
                if status is None:
                    event = outbox_store.build_write_event(node)
                    event.event_id = projection["event_id"]
                    outbox_store.register_once(event, ctx)
            projection_refs.append([node.uri, projection["event_id"]])
    if len(matches) > 1:
        raise ValueError("operation receipt has ambiguous committed nodes")
    if matches:
        from commit.outbox_scope import record_persisted_event
        for uri, event_id in projection_refs:
            record_persisted_event(uri, event_id)
        return WritePlan(action="skip", target_uri=matches[0],
                         merged_fields={"status": "operation_committed", "outbox_events": projection_refs,
                                        "projection_evidence_known": bool(projection_refs)})
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
        prior_projection = existing.metadata.get(PROJECTION_RECEIPTS_KEY, {})
        if isinstance(prior_projection, dict) and prior_projection:
            node.metadata[PROJECTION_RECEIPTS_KEY] = {**prior_projection,
                **node.metadata.get(PROJECTION_RECEIPTS_KEY, {})}
    if candidate.write_operation_id:
        digest = operation_digest(candidate, ctx)
        previous = receipts.get(candidate.write_operation_id)
        if previous is not None and previous != digest:
            raise ValueError("operation ID is already committed for a different candidate")
        receipts[candidate.write_operation_id] = digest
    if receipts:
        node.metadata[RECEIPTS_KEY] = receipts
