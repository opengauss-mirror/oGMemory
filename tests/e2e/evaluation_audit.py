"""Read-only native after_turn evidence for the isolated single-case service."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from evaluation_settlement import SnapshotUnstable, inventory

AUDIT_VERSION = 2


def pending_summary(recovery):
    if recovery is None:
        return None
    if not isinstance(recovery, dict):
        return {"pending": True}
    return {"pending": True, **{key: recovery[key] for key in (
        "status", "request_id", "archive_id", "retry_count", "attempts",
    ) if key in recovery}}


def state_summary(state):
    if state is None:
        return None
    return {
        "session_meta": {key: state.get("session_meta", {}).get(key) for key in (
            "account_id", "user_id", "agent_id", "session_id",
        )},
        "completed_request_ids": state.get("completed_request_ids", []),
        "buffer_message_count": len(state.get("buffer_messages", [])),
        "pending_recovery": pending_summary(
            state.get("runtime_extensions", {}).get("memory_extraction_recovery")),
    }


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def validate_projection_record(expected, actual, dimension, *, embedder):
    from core.models import IndexRecord
    from core.transcript_lexical import indexed_metadata
    from index.outbox_worker import OutboxWorker

    actual_metadata = actual.get("metadata", {}) if actual else {}
    embedding = actual_metadata.get("_embedding")
    numeric = isinstance(embedding, list) and all(type(value) in (int, float) for value in embedding)
    finite = numeric and all(math.isfinite(value) for value in embedding)
    embedding_checks = {"dimension_matches": isinstance(embedding, list) and len(embedding) == dimension,
                        "numeric_finite": finite, "nonzero": finite and any(value != 0 for value in embedding)}
    worker = OutboxWorker(vector_index=None, embedder=embedder, directory_summary_enabled=False)
    augmented = worker._augment_with_embeddings(
        [IndexRecord(**expected)], [embedding if isinstance(embedding, list) else []],
    )[0]
    native_expected = {**expected, "metadata": indexed_metadata(augmented.metadata)}
    mismatches = [key for key, value in native_expected.items() if actual is None or actual.get(key) != value]
    mismatches.extend("metadata._embedding." + key for key, valid in embedding_checks.items() if not valid)
    return {"id": expected["id"], "matches": not mismatches, "mismatched_fields": mismatches,
            "expected": native_expected, "actual": actual,
            "embedding_validation": {"configured_dimension": dimension,
                                     "configured_model": getattr(embedder, "model", None),
                                     "actual_dimension": len(embedding) if isinstance(embedding, list) else None,
                                     **embedding_checks}}


def scan_account(root):
    before = inventory(root)
    nodes, events, processing = {}, {}, []
    for name in before:
        path = root / name
        if path.name == ".meta.json":
            metadata = json.loads(path.read_text(encoding="utf-8"))
            node = {"metadata": metadata}
            for field, filename in (("abstract", ".abstract.md"), ("overview", ".overview.md"),
                                    ("content", "content.md")):
                text_path = path.parent / filename
                node[field] = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
            if metadata.get("uri"):
                nodes[metadata["uri"]] = node
        elif any(part in {".outbox", ".dlq", "dlq"} for part in path.parts):
            if path.suffix == ".processing":
                processing.append(name)
            elif path.suffix == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("event_id"):
                    events[name] = value
    after = inventory(root)
    return nodes, events, processing, before == after, after


def collect_database(service, account_id):
    from psycopg2 import sql
    from psycopg2.extras import RealDictCursor

    config = service._cfg
    index = service._vector_index
    if index is None:
        return {"available": False, "status": "UNKNOWN", "reason": "VECTOR_INDEX_NOT_INITIALIZED",
                "table": config.opengauss_table_name, "records": {},
                "records_sha256": None, "record_count": None, "read_only_transaction": None}
    result = {"available": True, "read_only_transaction": True,
              "table": config.opengauss_table_name, "records": {}}
    connection = index._get_connection()
    try:
        result["connection_backend_pid"] = connection.get_backend_pid()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor.execute("SET LOCAL statement_timeout='30000ms'")
            cursor.execute(sql.SQL("SELECT id,uri,level,text,filters,metadata FROM {} "
                                   "WHERE filters->>'account_id'=%s").format(
                                       sql.Identifier(config.opengauss_table_name)), (account_id,))
            result["records"] = {row["id"]: dict(row) for row in cursor.fetchall()}
    finally:
        try:
            connection.rollback()
        finally:
            index._return_connection(connection)
    result["records_sha256"] = digest(result["records"])
    result["record_count"] = len(result["records"])
    return result


def _collect_audit(service, ctx, params):
    from core.models import ContextNode
    from index.index_record_builder import build_index_records

    started = time.perf_counter()
    root = Path(os.environ["NODE_AUDIT_ACCOUNT_ROOT"]).resolve()
    if root.name != ctx.account_id:
        raise ValueError("Audit account must match the isolated account directory")
    nodes, events, processing, stable, initial_inventory = scan_account(root)
    database = collect_database(service, ctx.account_id)
    stable = stable and initial_inventory == inventory(root)
    hashes = {uri: digest(node) for uri, node in nodes.items()}
    receipts = {uri: sorted(node["metadata"].get("memory_write_receipts", {})) for uri, node in nodes.items()}
    baseline = params.get("baseline")
    baseline_nodes = baseline.get("nodes", {}) if baseline else {}
    changed = {uri: node for uri, node in nodes.items() if baseline_nodes.get(uri) != hashes[uri]}
    structural = {"uri", "context_type", "category", "level", "owner_space", "expected_version"}
    projections, added_receipts = [], {}
    for uri, node in changed.items():
        metadata = node["metadata"]
        prior_receipts = set((baseline or {}).get("receipts", {}).get(uri, []))
        added = {key: value for key, value in metadata.get("memory_write_receipts", {}).items()
                 if key not in prior_receipts}
        if added:
            added_receipts[uri] = added
        if metadata.get("context_type") != "MEMORY" or metadata.get("status", "ACTIVE") != "ACTIVE":
            continue
        source = ContextNode(uri=uri, context_type=metadata["context_type"], category=metadata["category"],
                             level=metadata["level"], owner_space=metadata["owner_space"],
                             abstract=node["abstract"], overview=node["overview"], content=node["content"],
                             metadata={key: value for key, value in metadata.items() if key not in structural})
        expected = build_index_records(source)
        rows = []
        for record in expected:
            fields = {name: getattr(record, name) for name in ("id", "uri", "level", "text", "filters", "metadata")}
            actual = database["records"].get(record.id)
            rows.append(validate_projection_record(fields, actual, service._cfg.opengauss_dimension,
                                                     embedder=service._embedder))
        projections.append({"uri": uri, "source_version": metadata.get("version"),
                            "records": rows, "complete": all(row["matches"] for row in rows)})

    state_uri = f"ctx://{ctx.account_id}/sessions/{ctx.session_id}/state.json"
    state_node = nodes.get(state_uri)
    state = json.loads(state_node["content"]) if state_node else None
    request_id = params.get("client_request_id", "")
    archives = {}
    expected_messages = params.get("expected_messages")
    archive_id = params.get("archive_id")
    archive_matches = []
    for uri, node in changed.items():
        if node["metadata"].get("session_id") == ctx.session_id and node["metadata"].get("category") == "history":
            messages = json.loads(node["content"])
            matches = expected_messages is not None and [
                {key: item.get(key) for key in ("role", "content")} for item in messages
            ] == [{key: item.get(key) for key in ("role", "content")} for item in expected_messages]
            identity_matches = not archive_id or node["metadata"].get("archive_id") == archive_id
            archive_matches.append(matches and identity_matches)
            archives[uri] = {"source_matches": matches, "archive_identity_matches": identity_matches,
                             "message_count": len(messages), "messages": messages,
                             "metadata": node["metadata"]}
    recovery = (state or {}).get("runtime_extensions", {}).get("memory_extraction_recovery")
    outbox_states = Counter(value.get("status", "UNKNOWN") for value in events.values())
    checks = {
        "database_available": database["available"],
        "stable_source_snapshot": stable,
        "request_completed": bool(request_id and request_id in (state or {}).get("completed_request_ids", [])),
        "buffer_empty": state is not None and state.get("buffer_messages") == [],
        "recovery_finished": state is not None and recovery is None,
        "source_archived_exactly": any(archive_matches),
        "extraction_full_quality": bool(archives) and all(
            not entry["metadata"].get("memory_extraction_degraded") for entry in archives.values()),
        "outbox_empty": not events and not processing,
        "changed_memories_projected": bool(projections) and all(row["complete"] for row in projections),
        "new_write_receipts_present": bool(added_receipts),
    }
    proof = {"source_hashes": hashes, "events": events, "processing": processing,
             "records_sha256": database["records_sha256"]}
    public_state = state_summary(state)
    if state_uri in changed:
        changed[state_uri] = {**changed[state_uri], "content": public_state}
    return {"ok": True, "audit_version": AUDIT_VERSION,
            "at": datetime.now(timezone.utc).isoformat(), "read_only": True,
            "account_id": ctx.account_id, "session_id": ctx.session_id,
            "baseline": {"nodes": hashes, "receipts": receipts,
                         "event_ids": sorted(value["event_id"] for value in events.values())},
            "source_node_count": len(nodes), "changed_node_count": len(changed), "changed_nodes": changed,
            "new_write_receipts": added_receipts, "new_write_receipt_count": sum(map(len, added_receipts.values())),
            "state": public_state, "archives": archives, "pending_recovery": pending_summary(recovery),
            "outbox": {"event_count": len(events), "states": dict(outbox_states),
                       "events": events, "processing_files": processing},
            "database": database, "projections": projections, "checks": checks,
            "complete": baseline is not None and all(checks.values()),
            "snapshot_sha256": digest(proof), "elapsed_seconds": time.perf_counter() - started}


def collect_audit(service, ctx, params):
    for attempt in range(3):
        try:
            result = _collect_audit(service, ctx, params)
        except (SnapshotUnstable, FileNotFoundError, json.JSONDecodeError):
            continue
        if result["checks"]["stable_source_snapshot"]:
            return result
    return {"ok": True, "complete": False, "read_only": True, "status": "SNAPSHOT_UNSTABLE",
            "snapshot_sha256": None, "checks": {"stable_source_snapshot": False}, "attempts": 3}


def register_routes(app, profiler):
    from flask import jsonify, request
    from server.app import _build_authenticated_context, _get_service

    @app.post("/latency_profile/audit")
    def audit_node_case_route():
        params = request.get_json(force=True, silent=True) or {}
        try:
            ctx = _build_authenticated_context(params)
            return jsonify(collect_audit(_get_service(), ctx, params))
        except Exception as exc:
            return jsonify({"ok": False, "complete": False, "read_only": True,
                            "error_type": type(exc).__name__, "error": str(exc)}), 500
