"""Read-only durable source settlement for the isolated CE evaluation account."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(block["text"] for block in value
                         if isinstance(block, dict) and isinstance(block.get("text"), str))
    return ""


class SnapshotUnstable(RuntimeError):
    """The account changed while collecting a read-only snapshot."""


def _temporary_directory(name):
    return bool(name.startswith(".") and (
        name.endswith((".resource-mutation-lock", ".update-lock"))
        or re.search(r"\.(?:resource-mutation-lock|update-lock|resource-create|create)-[0-9a-f]{32}$", name)))


def inventory(root):
    result = {}
    try:
        root.stat()
    except FileNotFoundError:
        return result

    def scan_error(error):
        if isinstance(error, FileNotFoundError):
            raise SnapshotUnstable("Source changed during directory scan") from error
        raise error

    for directory, names, files in os.walk(root, onerror=scan_error):
        names[:] = sorted(name for name in names
                          if name != ".outbox-done" and not _temporary_directory(name))
        for name in sorted(files):
            path = Path(directory) / name
            if (name not in {".meta.json", ".abstract.md", ".overview.md", "content.md"}
                    and not (any(part in {".outbox", ".dlq", "dlq"} for part in path.parts)
                             and not name.startswith(".") and path.suffix in {".json", ".processing"})):
                continue
            try:
                value = path.stat()
            except FileNotFoundError as error:
                raise SnapshotUnstable("Source changed during file scan") from error
            result[str(path.relative_to(root))] = [value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
    return result


def _collect_settlement(service, ctx, params):
    started = time.perf_counter()
    configured = os.environ.get("NODE_AUDIT_ACCOUNT_ROOT")
    root = Path(configured or f"/data/agfs/plugin/accounts/{ctx.account_id}").resolve()
    if root.name != ctx.account_id:
        raise ValueError("Settlement account differs from the isolated source root")
    expected_texts = params.get("expected_user_texts", [])
    request_ids = params.get("client_request_ids", [])
    if not expected_texts or not all(isinstance(value, str) and value for value in expected_texts):
        raise ValueError("Expected original user texts are required")
    if not request_ids or not all(isinstance(value, str) and value for value in request_ids):
        raise ValueError("Stable client request IDs are required")
    before = inventory(root)
    state, archives, outbox = None, [], []
    archived_hashes, buffered_hashes = Counter(), Counter()
    for relative in before:
        path = root / relative
        if any(part in {".outbox", ".dlq", "dlq"} for part in path.parts):
            if path.suffix in {".json", ".processing"}:
                outbox.append(relative)
            continue
        if path.name != ".meta.json":
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        uri = metadata.get("uri", "")
        source = path.parent / "content.md"
        if uri == f"ctx://{ctx.account_id}/sessions/{ctx.session_id}/state.json":
            state = json.loads(source.read_text(encoding="utf-8"))
            for message in state.get("buffer_messages", []):
                if message.get("role") == "user":
                    buffered_hashes[sha(content_text(message.get("content", "")))] += 1
        if metadata.get("session_id") == ctx.session_id and metadata.get("category") == "history":
            messages = json.loads(source.read_text(encoding="utf-8"))
            user_hashes = [sha(content_text(message.get("content", ""))) for message in messages
                           if message.get("role") == "user"]
            archived_hashes.update(user_hashes)
            archives.append({"uri": uri, "archive_id": metadata.get("archive_id"),
                             "message_count": len(messages), "user_text_sha256": user_hashes,
                             "degraded": bool(metadata.get("memory_extraction_degraded"))})
    expected = Counter(map(sha, expected_texts))
    completed = set((state or {}).get("completed_request_ids", []))
    recovery = (state or {}).get("runtime_extensions", {}).get("memory_extraction_recovery")
    meta = (state or {}).get("session_meta", {})
    checks = {
        "identity_matches": all(meta.get(key) == value for key, value in {
            "account_id": ctx.account_id, "user_id": ctx.user_id, "agent_id": ctx.agent_id,
            "session_id": ctx.session_id}.items()),
        "source_preserved": all(archived_hashes[key] >= count for key, count in expected.items()),
        "request_completed": all(value in completed for value in request_ids),
        "buffer_empty": state is not None and state.get("buffer_messages") == [],
        "recovery_finished": state is not None and recovery is None,
        "outbox_empty": not outbox,
        "extraction_full_quality": bool(archives) and all(not row["degraded"] for row in archives),
        "stable_source_snapshot": before == inventory(root),
    }
    evidence = {"archives": archives, "completed_request_ids": sorted(completed),
                "buffer_user_text_sha256": dict(buffered_hashes), "outbox_paths": outbox}
    return {"ok": True, "complete": all(checks.values()), "read_only": True,
            "at": datetime.now(timezone.utc).isoformat(), "checks": checks,
            "expected_user_text_sha256": dict(expected), "archived_user_text_sha256": dict(archived_hashes),
            "request_ids": request_ids, "identity": {"account_id": ctx.account_id, "user_id": ctx.user_id,
                                                      "agent_id": ctx.agent_id, "session_id": ctx.session_id},
            **evidence, "snapshot_sha256": sha(json.dumps(evidence, sort_keys=True, ensure_ascii=False)),
            "elapsed_seconds": time.perf_counter() - started}


def collect_settlement(service, ctx, params):
    for attempt in range(3):
        try:
            result = _collect_settlement(service, ctx, params)
        except (SnapshotUnstable, FileNotFoundError, json.JSONDecodeError):
            continue
        if result["checks"]["stable_source_snapshot"]:
            return result
    return {"ok": True, "complete": False, "read_only": True, "status": "SNAPSHOT_UNSTABLE",
            "snapshot_sha256": None, "checks": {"stable_source_snapshot": False}, "attempts": 3}


def register_routes(app, get_service):
    from flask import jsonify, request
    from server.app import _build_authenticated_context

    @app.post("/latency_profile/settle")
    def settle_evaluation_source():
        params = request.get_json(force=True, silent=True) or {}
        try:
            context = _build_authenticated_context(params)
            return jsonify(collect_settlement(get_service(), context, params))
        except ValueError as error:
            return jsonify({"ok": False, "complete": False, "read_only": True,
                            "error_type": type(error).__name__, "error": str(error)}), 400
        except Exception as error:
            return jsonify({"ok": False, "complete": False, "read_only": True,
                            "error_type": type(error).__name__, "error": str(error)}), 500
