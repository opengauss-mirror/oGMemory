"""Versioned Dreaming mutations for AGFS memory nodes."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from uuid import NAMESPACE_OID, uuid4, uuid5

from commit.dreaming_facts import same_correction_scope
from commit.write_receipts import PROJECTION_RECEIPTS_KEY, RECEIPTS_KEY, receipt_origin, receipt_projection
from core.dreaming_evidence import EVIDENCE_KEY, matches_node_version, merge_evidence, node_created_at
from core.enums import EventType, NodeStatus
from core.errors import AccessDeniedError, NodeBrokenError, NodeNotFoundError
from core.memory_names import merge_name_aliases
from core.models import OutboxEvent, RequestContext
from core.validation import MAX_ABSTRACT_LEN, MAX_CONTENT_LEN, MAX_OVERVIEW_LEN
from fs.agfs_adapter.client import publish_staged_file
from fs.agfs_adapter.node_scanner import (
    AGFSCanonicalNodeScanner,
    published_agfs_uri,
    safe_agfs_uri,
)

_MERGE = "dreaming_merge"
_MERGED_SOURCES = "dreaming_merged_sources"
_RECEIPT_TRANSFER = "dreaming_receipt_transfer"
_SOURCE_MAPS = ("memory_observation_sources", "occurrence_replay_sources")


def _identity_keys(metadata) -> set[tuple[str, str]]:
    keys = set()
    for kind in ("routing_key", "object_name"):
        aliases = metadata.get(kind + "_aliases", ())
        variants = [metadata.get(kind)]
        if isinstance(aliases, (list, tuple)):
            variants.extend(aliases)
        keys.update((kind, value) for value in variants if isinstance(value, str) and value)
    return keys


def _same_scope(left, right) -> bool:
    return (
        left.parent_uri == right.parent_uri
        and left.owner_space == right.owner_space
        and left.category == right.category
        and left.context_type == right.context_type
        and left.metadata.get("scope_ref", "") == right.metadata.get("scope_ref", "")
    )


def _candidate_identity(node, other) -> bool:
    if node.metadata.get("dreaming_promotion") or other.metadata.get("dreaming_promotion"):
        return False
    if node.category in {"event", "case"}:
        if any(node.metadata.get(field) and other.metadata.get(field)
               and node.metadata[field] != other.metadata[field]
               for field in ("when", "who", "where")):
            return False
        identity = node.metadata.get("source_occurrence_id")
        if identity and identity == other.metadata.get("source_occurrence_id"):
            return True
        return bool(set(node.metadata.get("provenance_ids") or [])
                    & set(other.metadata.get("provenance_ids") or []))
    if node.category == "pattern":
        return True
    return bool(_identity_keys(node.metadata) & _identity_keys(other.metadata))


def candidates(fs, node, ctx, limit):
    """Read complete peers from a bounded scan of the canonical directory."""
    if limit <= 0:
        return []
    if (node.category in {"event", "case"} and not node.metadata.get("source_occurrence_id")
            and not node.metadata.get("provenance_ids")):
        return []
    if node.category not in {"event", "case", "pattern"} and not _identity_keys(node.metadata):
        return []
    peers = []
    after = None
    page_size = min(64, limit * 4)
    for _ in range(4):
        page = fs.list_identity_node_summaries_page(
            node.parent_uri, ctx, limit=page_size, after_uri=after,
        )
        for summary in page:
            if (
                summary.uri == node.uri
                or not _same_scope(node, summary)
                or not _candidate_identity(node, summary)
                or not safe_agfs_uri(summary.uri)
            ):
                continue
            try:
                peer = fs.read_node(summary.uri, ctx)
            except (NodeNotFoundError, NodeBrokenError):
                continue
            if (
                published_agfs_uri(peer, summary.uri)
                and _same_scope(node, peer)
                and _candidate_identity(node, peer)
                and peer.content
            ):
                peers.append(peer)
                if len(peers) == limit:
                    return peers
        if len(page) < page_size:
            break
        after = fs.list_node_order_key(page[-1].uri)
    return peers


def _event_id(event, step):
    return str(uuid5(NAMESPACE_OID, f"{event.event_id}:{step}"))


def _revision(metadata):
    return {"version": metadata["version"], "created_at": node_created_at(metadata)}


def merge(fs, canonical, duplicate, ctx, event, outbox, *, layers=None):
    """Persist the accepted pair in the claimed event before changing either node."""
    if (
        canonical.uri == duplicate.uri
        or event.uri not in {canonical.uri, duplicate.uri}
        or not _same_scope(canonical, duplicate)
        or any(not safe_agfs_uri(node.uri) for node in (canonical, duplicate))
    ):
        raise ValueError("Dreaming merge requires distinct nodes in the same owner and scope")
    plan = {
        "canonical_uri": canonical.uri,
        "canonical_revision": _revision(canonical.metadata),
        "source_uri": duplicate.uri,
        "source_revision": _revision(duplicate.metadata),
    }
    if not all(plan[key]["created_at"] for key in ("canonical_revision", "source_revision")):
        return False
    if layers is not None:
        plan["layers"] = _validated_layers(layers)
    with fs.memory_identity_lock(canonical.parent_uri.rstrip("/"), ctx):
        current = fs.read_node(canonical.uri, ctx)
        source = fs.read_node(duplicate.uri, ctx)
        if (
            not _same_scope(current, source)
            or _revision(current.metadata) != plan["canonical_revision"]
            or _revision(source.metadata) != plan["source_revision"]
            or not _can_archive(source, current)
        ):
            return False
        current = _settle_receipts(fs, current, ctx)
        source = _settle_receipts(fs, source, ctx)
        changes = _merge_metadata(current, source, _event_id(event, "projection"))
        if changes is None:
            return False
        outbox.ensure_processing_owned(event, event.uri)
        event.payload[_MERGE] = plan
        outbox.save_payload(event)
        _publish_canonical(fs, current, changes, plan, ctx)
        _archive_source(fs, source, current, event)
        confirm_archived_receipts(fs, source.uri, ctx)
    _project_merge(fs, event, ctx, outbox)
    return True


def resume_merge(fs, event, ctx, outbox):
    """An unacknowledged event resumes both metadata changes and projection."""
    plan = event.payload[_MERGE]
    target_uri, source_uri = plan["canonical_uri"], plan["source_uri"]
    directory = target_uri.rsplit("/", 1)[0]
    if (
        target_uri == source_uri
        or event.uri not in {target_uri, source_uri}
        or source_uri.rsplit("/", 1)[0] != directory
        or not all(map(safe_agfs_uri, (target_uri, source_uri)))
    ):
        raise ValueError("Invalid Dreaming merge plan")
    source_revision, target_revision = plan.get("source_revision"), plan.get("canonical_revision")
    if not source_revision or not target_revision:
        return False
    with fs.memory_identity_lock(directory, ctx):
        outbox.ensure_processing_owned(event, event.uri)
        try:
            target = fs.read_dreaming_alias(target_uri, ctx, expected_created_at=target_revision["created_at"])
        except NodeNotFoundError:
            target = None
        committed = target is not None and target.metadata.get(_MERGED_SOURCES, {}).get(source_uri) == source_revision
        archived = _resume_source(fs, event, plan, ctx, committed)
        if archived:
            confirm_archived_receipts(fs, source_uri, ctx)
    if not committed and not archived:
        return False
    _project_merge(fs, event, ctx, outbox, archive_source=archived, project_target=target is not None)
    return True


def _resume_source(fs, event, plan, ctx, committed):
    target_uri, source_uri = plan["canonical_uri"], plan["source_uri"]
    source_revision, target_revision = plan["source_revision"], plan["canonical_revision"]
    try:
        metadata = fs.read_node_metadata_including_inactive(source_uri, ctx)
        if node_created_at(metadata) != source_revision["created_at"]:
            return False
        if metadata.get("status") == NodeStatus.ARCHIVED.value:
            return (metadata["version"] == source_revision["version"] + 1
                    and metadata.get("dreaming_canonical_uri") == target_uri
                    and metadata.get("dreaming_canonical_created_at") == target_revision["created_at"])
        source = fs.read_node(source_uri, ctx)
        canonical = fs.read_node(target_uri, ctx)
    except NodeNotFoundError:
        return _retained_entry(fs, source_uri, source_revision["created_at"]) == [
            target_uri, target_revision["created_at"], source_revision["version"] + 1,
        ]
    if (
        not _same_scope(canonical, source)
        or _revision(source.metadata) != source_revision
        or node_created_at(canonical.metadata) != target_revision["created_at"]
        or not _can_archive(source, canonical)
    ):
        return False
    if not committed:
        if _revision(canonical.metadata) != target_revision:
            return False
        canonical = _settle_receipts(fs, canonical, ctx)
        source = _settle_receipts(fs, source, ctx)
        changes = _merge_metadata(canonical, source, _event_id(event, "projection"))
        if changes is None:
            return False
        _publish_canonical(fs, canonical, changes, plan, ctx)
    _archive_source(fs, source, canonical, event)
    return True


def _can_archive(node, canonical):
    return (
        not node.metadata.get("pinned")
        and not node.metadata.get("_relations")
        and all(edge.to_uri != node.uri for edge in canonical.metadata.get("_relations", ()))
    )


def apply_correction(fs, current, prior, ctx, prior_review):
    if (current.uri == prior.uri or current.parent_uri != prior.parent_uri
            or current.owner_space != prior.owner_space or current.category != prior.category):
        return False
    with fs.memory_identity_lock(current.parent_uri.rstrip("/"), ctx):
        active = fs.read_node(current.uri, ctx)
        previous = fs.read_node(prior.uri, ctx)
        if not matches_node_version(active.metadata, current.metadata["version"], node_created_at(current.metadata)):
            return False
        if not same_correction_scope(active, previous) and any(
                current.uri in fact.get("external_corrections", {}) for fact in prior_review["facts"].values()):
            return False
        fs.record_dreaming_review(prior, ctx, prior_review)
    return True


def demote(fs, node, ctx, archive_event_id):
    fs._ensure_accessible(node.uri, ctx)
    _publish_metadata(fs, node, {
        "status": NodeStatus.ARCHIVED.value,
        "archived_at": datetime.now(UTC).isoformat(),
        "dreaming_demotion": archive_event_id,
    })


def _merge_metadata(canonical, source, projection_event):
    current, incoming = canonical.metadata, source.metadata
    existing = current.get(RECEIPTS_KEY, {})
    receipts = dict(existing)
    projection = dict(current.get(PROJECTION_RECEIPTS_KEY, {}))
    source_binding = [source.uri, node_created_at(incoming)]
    for operation, digest in incoming.get(RECEIPTS_KEY, {}).items():
        if receipts.setdefault(operation, digest) != digest:
            return None
        origin = receipt_origin(source, operation)
        owns_receipt = (origin == source_binding
                        or receipt_projection(source, operation).get("ownership") == "confirmed")
        if operation in existing and (
            not owns_receipt or receipt_origin(canonical, operation) != origin
            or receipt_projection(canonical, operation).get("ownership") == "confirmed"
        ):
            continue
        entry = {"event_id": projection_event, "origin": origin}
        if owns_receipt:
            entry["ownership"] = "pending"
            entry["transfer_source"] = source_binding
        projection[operation] = entry
    changes = {
        **merge_name_aliases(current, incoming),
        RECEIPTS_KEY: receipts,
        PROJECTION_RECEIPTS_KEY: projection,
        "provenance_ids": list(dict.fromkeys([
            *current.get("provenance_ids", []), *incoming.get("provenance_ids", []),
        ])),
        _MERGED_SOURCES: {
            **incoming.get(_MERGED_SOURCES, {}),
            **current.get(_MERGED_SOURCES, {}),
            source.uri: _revision(incoming),
        },
    }
    evidence = merge_evidence(current.get(EVIDENCE_KEY), incoming.get(EVIDENCE_KEY),
                              references=set(changes["provenance_ids"]))
    if evidence:
        changes[EVIDENCE_KEY] = evidence
    for field in _SOURCE_MAPS:
        merged = dict(current.get(field, {}))
        for key, references in incoming.get(field, {}).items():
            merged[key] = list(dict.fromkeys([*merged.get(key, []), *references]))
        if merged:
            changes[field] = merged
    observations = [*current.get("memory_observations", []), *incoming.get("memory_observations", [])]
    if observations:
        changes["memory_observations"] = list(dict.fromkeys(observations))
    return changes


def _publish_metadata(fs, node, changes):
    expected = node.metadata["version"]

    def update(metadata):
        metadata.update(changes, version=expected + 1, updated_at=datetime.now(UTC).isoformat())

    fs._replace_node(
        fs._uri_to_agfs_path(node.uri), node, expected,
        update_metadata=update,
    )


def _validated_layers(layers):
    limits = {"abstract": MAX_ABSTRACT_LEN, "overview": MAX_OVERVIEW_LEN, "content": MAX_CONTENT_LEN}
    if not isinstance(layers, dict) or set(layers) != set(limits):
        raise ValueError("Dreaming merge requires abstract, overview and content")
    if any(not isinstance(layers[key], str) or not layers[key].strip() or len(layers[key]) > limit
           for key, limit in limits.items()):
        raise ValueError("Dreaming merge layers exceed the memory contract")
    return dict(layers)


def _publish_canonical(fs, node, changes, plan, ctx):
    layers = plan.get("layers")
    if layers is None:
        _publish_metadata(fs, node, changes)
        return
    fs.write_node(replace(
        node, **_validated_layers(layers),
        metadata={**node.metadata, **changes, "_expected_version": node.metadata["version"],
                  "_expected_created_at": node_created_at(node.metadata)},
    ), ctx)


def _archive_source(fs, node, target, event):
    _publish_metadata(fs, node, {
        "status": NodeStatus.ARCHIVED.value,
        "archived_at": datetime.now(UTC).isoformat(),
        "dreaming_canonical_uri": target.uri,
        "dreaming_canonical_created_at": node_created_at(target.metadata),
        _RECEIPT_TRANSFER: _event_id(event, "projection"),
    })


def _settle_receipts(fs, node, ctx):
    pending = set()
    for receipt in node.metadata.get(PROJECTION_RECEIPTS_KEY, {}).values():
        if not isinstance(receipt, dict) or receipt.get("ownership") != "pending":
            continue
        binding = receipt.get("transfer_source")
        if isinstance(binding, list) and len(binding) == 2 and all(isinstance(value, str) and value for value in binding):
            pending.add(tuple(binding))
    for uri, created_at in sorted(pending):
        if uri.rsplit("/", 1)[0] == node.parent_uri.rstrip("/") and safe_agfs_uri(uri):
            confirm_archived_receipts(fs, uri, ctx, source_created_at=created_at, target_uri=node.uri)
    return fs.read_node(node.uri, ctx) if pending else node


def confirm_archived_receipts(fs, source_uri, ctx, *, source_created_at=None, target_uri=None, retain_alias=False):
    try:
        metadata = fs.read_node_metadata_including_inactive(source_uri, ctx)
    except (NodeNotFoundError, ValueError):
        return
    if source_created_at and node_created_at(metadata) != source_created_at:
        return
    target = metadata.get("dreaming_canonical_uri")
    if target_uri and target != target_uri:
        return
    created_at = metadata.get("dreaming_canonical_created_at")
    event_id = metadata.get(_RECEIPT_TRANSFER)
    if (metadata.get("status") == NodeStatus.ARCHIVED.value and created_at and event_id
            and isinstance(target, str) and safe_agfs_uri(target)
            and target.rsplit("/", 1)[0] == source_uri.rsplit("/", 1)[0]):
        _confirm_receipts(fs, target, created_at, event_id, ctx)
        if retain_alias:
            from commit.outbox_store import OutboxStore

            revision = _revision(metadata)
            _retain_alias(fs, source_uri, revision["created_at"], [target, created_at, revision["version"]])
            outbox = OutboxStore(fs._client, fs, mount_prefix=fs._mount_prefix)
            _queue_archive(outbox, ctx, source_uri, revision, event_id)


def _alias_path(fs, uri):
    parent, name = fs._uri_to_agfs_path(uri).rstrip("/").rsplit("/", 1)
    return f"{parent}/.{name}.dreaming-aliases.json"


def _read_alias_history(fs, path):
    try:
        return json.loads(fs._read_file(path))
    except NodeNotFoundError:
        return {}


def _retain_alias(fs, uri, created_at, target):
    path = _alias_path(fs, uri)
    aliases = _read_alias_history(fs, path)
    if aliases.get(created_at) != target:
        aliases[created_at] = target
        staged = f"{path}.{uuid4().hex}.tmp"
        try:
            fs._write_file(staged, json.dumps(aliases, ensure_ascii=False))
            publish_staged_file(fs._client, staged, path)
        finally:
            fs._remove_file_if_present(staged)


def _retained_entry(fs, uri, created_at):
    aliases = _read_alias_history(fs, _alias_path(fs, uri))
    return next(iter(aliases.values()), None) if created_at in (None, "legacy") else aliases.get(created_at)


def retained_alias(fs, uri, ctx, created_at):
    fs._ensure_accessible(uri, ctx)
    alias = _retained_entry(fs, uri, created_at)
    if (isinstance(alias, list) and len(alias) in (2, 3) and all(isinstance(value, str) and value for value in alias[:2])
            and safe_agfs_uri(alias[0]) and alias[0].rsplit("/", 1)[0] == uri.rsplit("/", 1)[0]):
        return alias[:2]


def retained_archive_version(fs, uri, created_at):
    entry = _retained_entry(fs, uri, created_at)
    return entry[2] if isinstance(entry, list) and len(entry) == 3 else None


def archive_source_uri(event):
    source = event.payload["dreaming_archive_source"]
    if not isinstance(source, str) or not safe_agfs_uri(source) or source.rpartition("/")[0] != event.uri:
        raise ValueError("Dreaming archive source must be in its queue directory")
    if not isinstance(event.payload.get("dreaming_archive_revision"), dict):
        raise ValueError("Dreaming archive source requires an archived revision")
    return source


def bind_relation_target(fs, edge, ctx):
    if edge.target_created_at is not None:
        return edge
    try:
        target = fs.read_dreaming_alias(edge.to_uri, ctx)
    except (AccessDeniedError, NodeNotFoundError, ValueError):
        return replace(edge, target_created_at="")
    return replace(edge, to_uri=target.uri, target_created_at=node_created_at(target.metadata) or "")


def _confirm_receipts(fs, target_uri, created_at, event_id, ctx):
    try:
        node = fs.read_node(target_uri, ctx)
    except NodeNotFoundError:
        return
    if node_created_at(node.metadata) != created_at:
        return
    confirmed = {
        key: {**receipt, "ownership": "confirmed"}
        for key, receipt in node.metadata.get(PROJECTION_RECEIPTS_KEY, {}).items()
        if isinstance(receipt, dict) and receipt.get("event_id") == event_id
        and receipt.get("ownership") == "pending"
    }
    if confirmed:
        fs._replace_node(
            fs._uri_to_agfs_path(node.uri), node, node.metadata["version"],
            update_metadata=lambda metadata: metadata[PROJECTION_RECEIPTS_KEY].update(confirmed),
        )


def _project_merge(fs, event, ctx, outbox, *, archive_source=True, project_target=True):
    plan = event.payload[_MERGE]
    source_revision = plan["source_revision"]
    outbox.ensure_processing_owned(event, event.uri)
    if project_target:
        canonical = fs.read_dreaming_alias(plan["canonical_uri"], ctx,
                                         expected_created_at=plan["canonical_revision"]["created_at"])
        projection = outbox.build_write_event(canonical)
        projection.event_id = _event_id(event, "projection")
        outbox.register_once(projection, ctx)
    if archive_source:
        _queue_archive(outbox, ctx, plan["source_uri"],
                       {**source_revision, "version": source_revision["version"] + 1},
                       _event_id(event, "projection"))


def _queue_archive(outbox, ctx, source_uri, revision, projection_id):
    owner_type, owner_id = source_uri.split("/", 5)[3:5]
    outbox.register_once(OutboxEvent(
        event_id=str(uuid5(NAMESPACE_OID, f"{projection_id}:archive")),
        event_type=EventType.ARCHIVE_CONTEXT.value,
        uri=source_uri.rpartition("/")[0],
        payload={
            "task_owner_spaces": [f"{owner_type[:-1]}:{owner_id}"],
            "dreaming_archive_source": source_uri,
            "dreaming_archive_revision": revision,
        },
        status="PENDING",
    ), ctx)


def scan_nodes(fs, account_id):
    """Enumerate active user and agent memories through bounded canonical reads."""
    from pyagfs.exceptions import AGFSClientError

    from commit.dreaming import eligible

    account = f"ctx://{account_id}"
    if not safe_agfs_uri(account) or "/" in account_id:
        raise ValueError("Invalid Dreaming account")
    scanner = AGFSCanonicalNodeScanner(fs)
    for owner_dir in ("users", "agents"):
        path = f"{fs._mount_prefix}/accounts/{account_id}/{owner_dir}/"
        try:
            owners = fs._client.ls(path)
        except AGFSClientError as exc:
            if fs._is_not_found_error(exc):
                continue
            raise
        for entry in owners:
            name = entry.get("name", "")
            if (
                not entry.get("isDir")
                or not isinstance(name, str)
                or not name
                or name.startswith(".")
                or "/" in name
                or not safe_agfs_uri(f"{account}/{owner_dir}/{name}")
            ):
                continue
            user, agent = (name, "") if owner_dir == "users" else ("", name)
            ctx = RequestContext(account_id, user, agent, "", "dreaming-sweep")
            owner = f"{owner_dir[:-1]}:{name}"
            for node in scanner.scan_active_nodes(account_id=account_id, owner_spaces=[owner], page_size=32):
                if eligible(node):
                    yield node, ctx
