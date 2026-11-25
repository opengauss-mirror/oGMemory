"""Retain committed Bot archives independently of the archived node directory."""

import hashlib
import json
from uuid import uuid4

from core.dreaming_evidence import node_created_at
from core.errors import NodeNotFoundError
from fs.agfs_adapter.client import publish_staged_file
from fs.agfs_adapter.node_scanner import safe_agfs_uri


def _path(fs, uri, operation_id):
    parent = fs._uri_to_agfs_path(uri).rstrip("/").rsplit("/", 1)[0]
    key = hashlib.sha256(f"{uri}:{operation_id}".encode()).hexdigest()
    return f"{parent}/.bot-archive-{key}.json"


def read_receipt(fs, uri, ctx, operation_id, *, created_at=None):
    fs._ensure_accessible(uri, ctx)
    try:
        value = json.loads(fs._read_file(_path(fs, uri, operation_id)))
    except NodeNotFoundError:
        return None
    if (
        value.get("uri") != uri
        or value["operation"]["id"] != operation_id
        or created_at
        and value["created_at"] != created_at
    ):
        return None
    return value


def retain_receipt(fs, uri, ctx):
    fs._ensure_accessible(uri, ctx)
    try:
        metadata = fs.read_node_metadata_including_inactive(uri, ctx)
    except NodeNotFoundError:
        return
    operation = metadata.get("bot_mutation") or {}
    if metadata.get("status") != "ARCHIVED" or not operation.get("id"):
        return
    value = {
        "uri": uri,
        "created_at": node_created_at(metadata),
        "version": metadata["version"],
        "operation": operation,
    }
    path = _path(fs, uri, operation["id"])
    temporary = f"{path}.{uuid4().hex}.tmp"
    try:
        fs._write_file(temporary, json.dumps(value, ensure_ascii=False))
        publish_staged_file(fs._client, temporary, path)
    finally:
        fs._remove_file_if_present(temporary)


def archive_source_uri(event):
    uri = event.payload["canonical_archive_source"]
    if not safe_agfs_uri(uri) or uri.rpartition("/")[0] != event.uri:
        raise ValueError("Archive source must belong to its queue directory")
    return uri
