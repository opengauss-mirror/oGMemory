#!/usr/bin/env python3
"""Ingest only BEAM source pairs via native APIs into an isolated fresh SQL run.

Run the snapshot runner's preflight first. No source replay/resume, production
patch, prompt rewrite or truncation salvage is performed by this script.
Without --execute this only reads the source parquet and reports its identity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.beam_dataset import load_source_pairs
from scripts.beam_eval_safety import (
    SnapshotDB, digest, ingestion_failure_log, require, require_eval_container, write_json,
)
from scripts.run_stage8r_beam_clean_eval import check_runtime, wait_health


def source_pairs(parquet, conversation_index=0):
    """Compatibility entrypoint for existing experiment controllers."""
    return load_source_pairs(parquet, conversation_index)


def inventory(db):
    has_projections = db.sql("SELECT to_regclass('public.edge_projection_jobs') IS NOT NULL") == "t"
    pending = "(SELECT count(*) FROM edge_projection_jobs WHERE upper(status)<>'COMPLETE')" if has_projections else "0"
    return json.loads(db.sql(f"""SELECT json_build_object(
      'nodes',(SELECT count(*) FROM context_nodes),
      'activations',(SELECT count(*) FROM context_nodes WHERE metadata ? 'update_activation'),
      'activation_vectors',(SELECT count(*) FROM vector_index WHERE level=3),
      'edges',(SELECT count(*) FROM relation_edges),
      'outbox',(SELECT count(*) FROM outbox_events),
      'outbox_failed',(SELECT count(*) FROM outbox_events WHERE upper(status)='FAILED'),
      'projection_pending',{pending},
      'archives',(SELECT count(*) FROM session_archives))"""))


def request_headers(args):
    headers = {"X-Account-ID": args.account, "X-User-ID": args.user, "X-Agent-ID": "main"}
    key = os.environ.get(args.memory_api_key_env)
    if key:
        headers["X-API-Key"] = key
    return headers


def post(args, method, payload):
    import requests
    response = requests.post(
        args.memory_url.rstrip("/") + "/api/v1/call/" + method,
        headers=request_headers(args), json=payload, timeout=args.timeout,
    )
    response.raise_for_status()
    body = response.json()
    require(isinstance(body, dict), "Memory API returned a non-object response")
    require(body.get("ok") is not False and not body.get("error") and
            body.get("status") not in ("failed", "needs_confirmation") and
            body.get("retryable") is not False, method + " reported failure; inspect retained artifacts")
    return body


def wait_idle(args, identity):
    deadline = time.monotonic() + args.timeout
    while True:
        value = post(args, "wait_until_idle", {**identity, "timeoutSeconds": 5,
                     "drainOutbox": False, "waitOutbox": True})
        if value.get("idle") is True:
            return value
        require(time.monotonic() < deadline, "Ingestion/extraction did not become idle")
        time.sleep(1)


def ingest(args):
    require(bool(re.fullmatch(r"[a-z0-9][a-z0-9-]{5,50}", args.run_id)), "Invalid unique run ID")
    conversation, pairs = source_pairs(args.parquet, args.conversation_index)
    source_hash = hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()
    report = dict(status="planned", run_id=args.run_id, database=args.database,
                  conversation_index=args.conversation_index, conversation_id=conversation,
                  source_pairs=len(pairs), completed_pairs=0, source_sha256=source_hash,
                  parquet_sha256=digest(args.parquet), read_columns=["conversation_id", "chat"],
                  source_root=str(args.source_root.resolve()), fresh=True,
                  activation_generation="native configured extraction; no experiment overrides")
    if not args.execute:
        return report
    require(not args.output_dir.exists(), "Refusing to overwrite/replay existing ingestion")
    pre = json.loads(args.preflight.read_text())
    require(pre.get("corpus_origin") == "fresh" and pre["run_id"] == args.run_id
            and pre["database"] == args.database, "Fresh preflight identity mismatch")
    info = require_eval_container(args.memory_container, args.run_id, "memory")
    require(not info["State"]["Running"], "Start from stopped preflight memory service")
    db = SnapshotDB(args.postgres_container, args.database)
    db.assert_idle()
    require(db.fingerprint() == pre["inventory"], "Database changed since zero-state preflight")
    args.output_dir.mkdir(parents=True, mode=0o700)
    report.update(status="ingesting", preflight_sha256=digest(args.preflight))
    write_json(args.output_dir / "source-manifest.json", report)
    try:
        subprocess.run(["docker", "start", args.memory_container], check=True, capture_output=True)
        wait_health(args.memory_url)
        require(check_runtime(args, db) == pre["runtime"], "Runtime changed since preflight")
        for index, pair in enumerate(pairs, 1):
            started_at = datetime.now(timezone.utc).isoformat()
            identity = {"sessionId": f"{args.run_id}-turn-{index:03d}", "accountId": args.account,
                        "userId": args.user, "agentId": "main"}
            payload = {**identity, "wait": True, "prePromptMessageCount": 0, "messages": [
                {"role": message["role"], "content": message["content"], "source_chat_id": message["id"]}
                for message in pair]}
            prefix = args.output_dir / f"turn-{index:03d}"
            write_json(prefix.with_suffix(".request.json"), payload)
            started = time.monotonic()
            result = post(args, "after_turn", payload)
            write_json(prefix.with_suffix(".after-turn.json"), result)
            require(result.get("status") != "processing", "Synchronous after_turn returned early")
            wait_idle(args, identity)
            disposed = post(args, "dispose", identity)
            write_json(prefix.with_suffix(".dispose.json"), disposed)
            require(disposed.get("reason") not in ("no_write_api", "session_not_found", "below_token_threshold"),
                    "Source pair was not fully extracted; retain failed run, do not label complete")
            wait_idle(args, identity)
            logs = subprocess.run(["docker", "logs", "--since", started_at, args.memory_container],
                                  capture_output=True, text=True, check=True)
            log_text = logs.stdout + logs.stderr
            with prefix.with_suffix(".service.log").open("x") as handle:
                handle.write(log_text)
            require(not any("ERROR" in line for line in ingestion_failure_log(log_text).splitlines()),
                    "Service reported an ingestion error; inspect retained service log")
            deadline = time.monotonic() + args.timeout
            stable = 0
            while stable < 2:
                counts = inventory(db)
                require(counts["outbox_failed"] == 0, "Failed indexing jobs; no quarantine permitted")
                settled = counts["outbox"] == 0 and counts["projection_pending"] == 0
                stable = stable + 1 if settled else 0
                require(time.monotonic() < deadline, "Index/projection jobs did not settle")
                if stable < 2:
                    time.sleep(1)
            write_json(prefix.with_suffix(".completed.json"), dict(pair=index,
                       session=identity["sessionId"], counts=counts,
                       seconds=round(time.monotonic() - started, 3)))
            report["completed_pairs"] = index
            print(f"Completed source pair {index}/{len(pairs)}: {counts['nodes']} nodes, {counts['edges']} edges", flush=True)
        report.update(status="ingested_unfrozen", final_counts=inventory(db))
    except BaseException as error:
        report.update(status="INVALID", failure_type=type(error).__name__)
        raise
    finally:
        write_json(args.output_dir / "ingestion.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("database", "memory-container", "run-id", "account", "user", "memory-url", "postgres-container"):
        parser.add_argument("--" + name, required=True)
    for name in ("source-root", "output-dir", "preflight", "parquet"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--conversation-index", type=int, default=0, help="Zero-based parquet row")
    parser.add_argument("--memory-api-key-env", default="BEAM_MEMORY_API_KEY")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--execute", action="store_true", help="Permit Docker and paid extraction work")
    args = parser.parse_args(argv)
    require(args.timeout > 0, "timeout must be positive")
    os.umask(0o077)
    result = ingest(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
