"""Proof of one exact accepted operation, persisted atomically with its node."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from core.dreaming_evidence import node_created_at
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


def receipt_projection(node, operation_id):
    projection = node.metadata.get(PROJECTION_RECEIPTS_KEY, {}).get(operation_id)
    return projection if isinstance(projection, dict) else {}


def receipt_origin(node, operation_id):
    return receipt_projection(node, operation_id).get("origin", [node.uri, node_created_at(node.metadata)])


def _resolve_receipt_owner(fs, matches, operation_id, ctx):
    origin = receipt_origin(matches[0], operation_id)
    if (not isinstance(origin, list) or len(origin) != 2
            or not all(isinstance(value, str) and value for value in origin)
            or origin[0].rsplit("/", 1)[0] != matches[0].parent_uri.rstrip("/")
            or any(receipt_origin(node, operation_id) != origin for node in matches[1:])):
        return None
    instances = {(node.uri, node_created_at(node.metadata)): node for node in matches}
    if owner := instances.get(tuple(origin)):
        return owner
    confirmed = [node for node in matches if receipt_projection(node, operation_id).get("ownership") == "confirmed"]
    if confirmed:
        return confirmed[0] if len(confirmed) == 1 else None
    resolve = getattr(fs, "read_dreaming_alias", None)
    if callable(resolve):
        try:
            canonical = resolve(origin[0], ctx, expected_created_at=origin[1])
        except NodeNotFoundError:
            return None
        return instances.get((canonical.uri, node_created_at(canonical.metadata)))
    return None


def receipt_plan(fs, uris, candidate: CandidateMemory, ctx: RequestContext, *, outbox_store=None):
    """Trust a receipt only after a complete authorized node read.

    Versioned operations check their durable target first. Discovery follows
    merged receipts within the same schema-owned category and owner.
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
    located = not schema.is_single_file and candidate.write_operation_id.startswith(LOCATED_OPERATION_PREFIX)
    resolve_alias = getattr(fs, "read_dreaming_alias", None) if located else None

    def read_matches(candidates, *, follow_alias=False):
        for uri in candidates:
            if (not uris.validate_uri(uri, ctx)
                    or (uri.rsplit("/", 1)[0] != directory and not schema.is_single_file)):
                continue
            try:
                node = fs.read_node(uri, ctx)
            except NodeNotFoundError:
                if not follow_alias or not callable(resolve_alias):
                    continue
                try:
                    node = resolve_alias(uri, ctx)
                except NodeNotFoundError:
                    continue
                validate_receipt_target(uris, candidate, ctx, node.uri)
            if node.category != candidate.category or node.owner_space != expected_owner:
                continue
            receipts = node.metadata.get(RECEIPTS_KEY, {})
            if not isinstance(receipts, dict) or candidate.write_operation_id not in receipts:
                continue
            if receipts[candidate.write_operation_id] != digest:
                raise ValueError("operation receipt does not match the accepted candidate or owner")
            yield node

    if schema.is_single_file:
        uri = uris.resolve(candidate.category, {"routing_key": candidate.routing_key}, ctx)
        matches = list(read_matches((uri,)))
    elif located:
        target = located_target(fs, uris, candidate, ctx)
        if target is None:
            return None
        matches = list(read_matches((target,), follow_alias=True))
        if not matches and callable(resolve_alias):
            from commit.dreaming import MERGE_CATEGORIES

            if candidate.category in MERGE_CATEGORIES:
                matches = list(read_matches(_receipt_candidates(fs, directory, candidate.write_operation_id, ctx)))
    else:
        matches = list(read_matches(_receipt_candidates(fs, directory, candidate.write_operation_id, ctx)))
    if not matches:
        return None
    node = matches[0]
    if len(matches) > 1 or "origin" in receipt_projection(node, candidate.write_operation_id):
        node = _resolve_receipt_owner(fs, matches, candidate.write_operation_id, ctx)
        if node is None:
            if len(matches) > 1:
                raise ValueError("operation receipt has ambiguous committed nodes")
            raise ValueError("operation receipt owner cannot be proven")
    projection_refs = []
    projection = receipt_projection(node, candidate.write_operation_id)
    if projection.get("event_id"):
        event_id = projection["event_id"]
        if outbox_store is not None and outbox_store.get_event_status(node.uri, event_id) is None:
            event = outbox_store.build_write_event(node)
            event.event_id = event_id
            outbox_store.register_once(event, ctx)
        projection_refs.append([node.uri, event_id])
        from commit.outbox_scope import record_persisted_event

        record_persisted_event(node.uri, event_id)
    return WritePlan(action="skip", target_uri=node.uri,
                     merged_fields={"status": "operation_committed", "outbox_events": projection_refs,
                                    "projection_evidence_known": bool(projection_refs)})


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
