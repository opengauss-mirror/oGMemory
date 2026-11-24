"""Versioned Dreaming mutations for AGFS memory nodes."""

from datetime import UTC, datetime

from commit.dreaming_facts import same_correction_scope
from core.dreaming_evidence import matches_node_version, node_created_at
from core.enums import NodeStatus
from core.errors import NodeBrokenError, NodeNotFoundError
from core.models import RequestContext
from fs.agfs_adapter.node_scanner import (
    AGFSCanonicalNodeScanner,
    published_agfs_uri,
    safe_agfs_uri,
)

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


def _publish_metadata(fs, node, changes):
    expected = node.metadata["version"]

    def update(metadata):
        metadata.update(changes, version=expected + 1, updated_at=datetime.now(UTC).isoformat())

    fs._replace_node(
        fs._uri_to_agfs_path(node.uri), node, expected,
        update_metadata=update,
    )


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
