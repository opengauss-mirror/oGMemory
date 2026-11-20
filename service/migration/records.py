"""Business record contract shared by the two offline storage adapters."""

from __future__ import annotations

import json
from datetime import datetime
from urllib.parse import unquote, urlsplit

from core.models import ContextNode
from service.migration.package import PackageError

NODE_FIELDS = (
    "uri",
    "account_id",
    "owner_space",
    "category",
    "context_type",
    "level",
    "status",
    "content",
    "abstract",
    "overview",
    "metadata",
    "version",
    "created_at",
    "updated_at",
)
ARCHIVE_FIELDS = (
    "archive_id",
    "session_id",
    "account_id",
    "owner_user_id",
    "owner_agent_id",
    "abstract",
    "overview",
    "messages",
    "metadata",
    "created_at",
)
EDGE_FIELDS = ("from_uri", "to_uri", "relation_type", "weight", "reason")
MIRRORS = set(NODE_FIELDS) - {"metadata", "content", "abstract", "overview"}
RUNTIME_META = {
    "_relations",
    "_expected_version",
    "expected_version",
    "session_state_layout",
    "memory_write_receipts",
    "memory_write_projection_receipts",
}


def identity(uri):
    if not isinstance(uri, str) or len(uri) > 512 or any(ord(c) < 32 or ord(c) == 127 for c in uri):
        raise PackageError("invalid node URI")
    parsed = urlsplit(uri)
    parts = [unquote(p) for p in parsed.path.strip("/").split("/")]
    if (
        parsed.scheme != "ctx"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or any(c in parsed.netloc for c in "%\\:@")
        or parsed.netloc in {".", ".."}
        or any(not p or p in {".", ".."} or any(c in p for c in "/\\\x00%") for p in parts)
    ):
        raise PackageError("invalid node URI")
    from fs.agfs_adapter.agfs_context_fs import uri_to_path

    if parts[0] != 'spaces':
        uri_to_path(uri)
    return parsed.netloc, parts


def iso(value):
    if isinstance(value, datetime):
        value = value.isoformat()
    if not isinstance(value, str):
        raise PackageError("missing timestamp")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("timezone required")
        from datetime import UTC

        return dt.astimezone(UTC).isoformat()
    except ValueError as exc:
        raise PackageError("invalid timestamp") from exc


def normalize_relation(record):
    """Canonical JSON numbers across SQL DOUBLE PRECISION and file records.

    Call after validation; preserve caller data and collapse signed zero.
    """
    return {**record, "weight": float(record["weight"]) or 0.0}


def normalize_node(record):
    r = dict(record)
    r["metadata"] = {
        k: v for k, v in r.get("metadata", {}).items() if k not in MIRRORS | RUNTIME_META
    }
    r["created_at"], r["updated_at"] = iso(r["created_at"]), iso(r["updated_at"])
    return {k: r[k] for k in NODE_FIELDS}


def normalize_archive(record):
    r = dict(record)
    r["created_at"] = iso(r["created_at"])
    metadata = dict(r.get("metadata", {}))
    if str(metadata.get("status", "")).upper() == "MERGED":
        metadata.setdefault("archive_status", "MERGED")
    r["metadata"] = {
        k: v
        for k, v in metadata.items()
        if k
        not in MIRRORS
        | RUNTIME_META
        | {"archive_id", "session_id", "message_count", "owner_user_id", "owner_agent_id"}
    }
    return {k: r[k] for k in ARCHIVE_FIELDS}


def node_to_archive(node):
    account, parts = identity(node["uri"])
    m = dict(node["metadata"])
    if node["status"] == "MERGED":
        m.setdefault("archive_status", "MERGED")
    elif node["status"] != "ACTIVE":
        raise PackageError("unsupported archived history storage state")
    return normalize_archive(
        dict(
            archive_id=parts[3],
            session_id=parts[1],
            account_id=account,
            owner_user_id=m.get("owner_user_id", ""),
            owner_agent_id=m.get("owner_agent_id", ""),
            abstract=node["abstract"],
            overview=node["overview"],
            messages=json.loads(node["content"]),
            metadata=m,
            created_at=node["created_at"],
        )
    )


def archive_to_node(r):
    return dict(
        uri=archive_uri(r),
        account_id=r["account_id"],
        owner_space="session:" + r["session_id"],
        context_type="RESOURCE",
        category="history",
        level=0,
        status="ACTIVE",
        version=1,
        content=json.dumps(r["messages"], ensure_ascii=False),
        abstract=r["abstract"],
        overview=r["overview"],
        created_at=r["created_at"],
        updated_at=r["created_at"],
        metadata={
            **r["metadata"],
            **{k: r[k] for k in ("archive_id", "session_id", "owner_user_id", "owner_agent_id")},
            "message_count": len(r["messages"]),
        },
    )


def archive_uri(r):
    return f"ctx://{r['account_id']}/sessions/{r['session_id']}/history/{r['archive_id']}"


def node_object(r):
    return ContextNode(
        **{
            k: r[k]
            for k in (
                "uri",
                "context_type",
                "category",
                "level",
                "owner_space",
                "abstract",
                "overview",
                "content",
            )
        },
        metadata={**r["metadata"], "version": r["version"], "status": r["status"]},
    )


def write_operation_path(parts):
    """Reserved receipt namespace, scoped to a user/Agent memory root."""
    return (
        len(parts) >= 4
        and parts[0] in {"users", "agents"}
        and parts[2:4] == ["memories", ".write_operations"]
    )


def runtime_node(r):
    _, parts = identity(r["uri"])
    return write_operation_path(parts) or (
        parts[0] == "sessions" and (len(parts) < 3 or parts[2] != "history")
    )


def validate_record(kind, r):
    fields = {"nodes": NODE_FIELDS, "space_nodes": NODE_FIELDS, "archives": ARCHIVE_FIELDS, "relations": EDGE_FIELDS}[kind]
    if set(r) != set(fields):
        raise PackageError(f"{kind}: unsupported or missing record fields")
    if kind == "relations":
        a, _ = identity(r["from_uri"])
        b, _ = identity(r["to_uri"])
        if a != b or not isinstance(r["relation_type"], str) or not r["relation_type"]:
            raise PackageError("invalid relation identity")
        if (
            type(r["weight"]) not in (int, float)
            or not 0 <= r["weight"] <= 1
            or not isinstance(r["reason"], str)
        ):
            raise PackageError("invalid relation value")
        if len(r["relation_type"]) > 256:
            raise PackageError("relation type exceeds storage limit")
        return
    if not isinstance(r["metadata"], dict):
        raise PackageError("metadata must be an object")
    for field in fields:
        if field not in {"metadata", "messages", "level", "version"} and not isinstance(
            r[field], str
        ):
            raise PackageError(f"{kind}: invalid {field}")
    limits = {
        "account_id": 128,
        "owner_space": 128,
        "category": 64,
        "context_type": 32,
        "archive_id": 64,
        "session_id": 128,
        "owner_user_id": 128,
        "owner_agent_id": 128,
    }
    if any(len(r[k]) > limit for k, limit in limits.items() if k in r):
        raise PackageError("record identity exceeds storage limit")
    iso(r["created_at"])
    if kind == "archives":
        account, parts = identity(archive_uri(r))
        if account != r["account_id"] or len(parts) != 4 or not isinstance(r["messages"], list):
            raise PackageError("invalid archive")
        if not all(isinstance(m, dict) for m in r["messages"]):
            raise PackageError("invalid archived message")
        return
    account, parts = identity(r["uri"])
    if account != r["account_id"] or runtime_node(r) or parts[0] == "sessions":
        raise PackageError("invalid or runtime-only node")
    expected = {"users": "user:", "agents": "agent:", "resources": "resource:"}.get(parts[0])
    if kind == 'space_nodes':
        if parts[0] != 'spaces' or len(parts) != 5 or parts[2] != 'memories':
            raise PackageError('invalid shared-space content URI')
        expected = 'space:'
    owner = account if parts[0] == "resources" else parts[1]
    if expected is None or r["owner_space"] != expected + owner:
        raise PackageError("node owner does not match URI")
    if r["context_type"] not in {"MEMORY", "RESOURCE", "SKILL"} or r["status"] not in {
        "ACTIVE",
        "ARCHIVED",
    }:
        raise PackageError("unsupported node type or state")
    if (
        type(r["version"]) is not int
        or r["version"] < 1
        or type(r["level"]) is not int
        or r["level"] < 0
    ):
        raise PackageError("invalid node version/level")
    iso(r["updated_at"])
    from service.migration.assets import bindings

    list(bindings(r))  # Absolute source paths are never accepted in an import package.
