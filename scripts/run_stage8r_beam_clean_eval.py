#!/usr/bin/env python3
"""Safe BEAM runner: preflight -> external ingestion -> freeze -> run.

See scripts/README-BEAM.md. Never uses timestamp resets.
No Docker mutation or paid answering operation occurs without --execute.
Only disposable beam_eval_* databases and labelled containers are supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.beam_dataset import load_source_pairs, load_questions
from scripts.beam_official import CATEGORIES
from scripts.beam_snapshot import freeze_qualified_snapshot, resolve_snapshot
from scripts.beam_eval_safety import (
    PROTOCOL, SnapshotDB, agent_preflight, audit_session,
    collect_session, digest, docker_text, json_digest, require,
    require_eval_container, required_corpus_tables, runtime_preflight, validate_agent_mounts, write_json,
)

def response_text(body):
    return "\n".join(part["text"] for item in body.get("output", [])
        if item.get("type") == "message" for part in item.get("content", [])
        if part.get("type") == "output_text")


def expected_answer(probe):
    return next((probe[k] for k in ("answer", "ideal_answer", "ideal_response", "ideal_summary",
                                   "expected_compliance") if k in probe), None)


def wait_health(url, path="/api/v1/health"):
    import requests
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            if requests.get(url.rstrip("/") + path, timeout=3).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("Evaluation service did not become healthy")


def validate_ingestion_manifest(args):
    """Bind freezing to a completed source-only ingestion of this exact dataset."""
    require(args.ingestion_manifest is not None and args.parquet is not None,
            "Freeze requires --ingestion-manifest and --parquet")
    report = json.loads(args.ingestion_manifest.read_text())
    conversation, pairs = load_source_pairs(args.parquet, args.conversation_index)
    source_hash = hashlib.sha256(
        json.dumps(pairs, sort_keys=True).encode()).hexdigest()
    require(report.get("status") == "ingested_unfrozen",
            "Only fully completed ingestion can be frozen")
    require(report.get("run_id") == args.run_id and report.get("database") == args.database,
            "Ingestion completion identity differs from run/database")
    require(str(report.get("conversation_id")) == str(conversation)
            and report.get("conversation_index") == args.conversation_index,
            "Ingestion completed a different conversation")
    require(report.get("source_pairs") == report.get("completed_pairs") == len(pairs),
            "Ingestion did not complete every source pair")
    require(report.get("source_sha256") == source_hash
            and report.get("parquet_sha256") == digest(args.parquet)
            and report.get("preflight_sha256") == digest(args.preflight),
            "Ingestion source/dataset/preflight binding changed")
    return {"manifest_sha256": digest(args.ingestion_manifest),
            "source_pairs": len(pairs), "conversation_id": conversation,
            "conversation_index": args.conversation_index,
            "source_sha256": source_hash, "parquet_sha256": digest(args.parquet)}


def validate_snapshot_dataset(snapshot, args, conversation_id):
    """Reject QA against another conversation/dataset before any SQL restore."""
    ingestion = snapshot.get("ingestion")
    require(isinstance(ingestion, dict), "Snapshot lacks a verified ingestion completion binding")
    require(ingestion.get("parquet_sha256") == digest(args.parquet)
            and ingestion.get("conversation_index") == args.conversation_index
            and str(ingestion.get("conversation_id")) == str(conversation_id),
            "Frozen corpus and QA dataset/conversation differ")
    require(type(ingestion.get("source_pairs")) is int and ingestion["source_pairs"] > 0,
            "Frozen corpus has no completed source pairs")


def check_runtime(args, db):
    info = require_eval_container(args.memory_container, args.run_id, "memory")
    require(info["HostConfig"]["NetworkMode"] == "host", "Only host-network evaluation services are supported")
    actual = runtime_preflight(args.memory_container, args.source_root, args.database, args.account, args.user)
    require(actual["targets"][0][1] == db.sql(
        "SELECT system_identifier::text FROM pg_control_system()", admin=True),
        "Runtime and reset command target different PostgreSQL clusters")
    url = urlsplit(args.memory_url)
    require(url.scheme == "http" and url.hostname == "127.0.0.1"
            and url.port == actual["config"]["http_port"], "Memory URL does not match resolved runtime")
    actual["container_id"] = info["Id"]
    actual["image_id"] = info["Image"]
    actual["entrypoint"] = info["Config"].get("Entrypoint")
    actual["command"] = info["Config"].get("Cmd")
    return actual


def stop_memory(args):
    require_eval_container(args.memory_container, args.run_id, "memory")
    subprocess.run(["docker", "stop", args.memory_container], check=True, capture_output=True)


def stop_agent(args):
    require_eval_container(args.agent_container, args.run_id, "agent")
    subprocess.run(["docker", "stop", args.agent_container], check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "freeze", "run"])
    for name in ("database", "memory-container", "run-id", "account", "user", "memory-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--postgres-container", required=True)
    parser.add_argument("--execute", action="store_true",
                        help="Permit isolated Docker lifecycle/SQL resets and paid QA")
    for name in ("source-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("preflight", "snapshot", "source-corpus-dump", "parquet", "ingestion-manifest"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--corpus-origin", choices=["fresh", "historical"], default="fresh")
    parser.add_argument("--agent-container")
    parser.add_argument("--base-url")
    parser.add_argument("--variant", choices=["baseline", "graph"], default="baseline")
    parser.add_argument("--conversation-index", type=int, default=0,
                        help="Zero-based parquet conversation row")
    args = parser.parse_args()
    require(bool(re.fullmatch(r"[a-z0-9][a-z0-9-]{5,50}", args.run_id)), "Invalid unique run ID")
    if not args.execute:
        print(json.dumps({"status": "prepared_offline", "protocol": PROTOCOL,
                          "command": args.command, "database": args.database,
                          "variant": args.variant, "output_dir": str(args.output_dir),
                          "note": "Add --execute for Docker/SQL operations or paid QA."}))
        return
    os.umask(0o077)  # Dataset questions, evidence and raw answers are private artifacts.
    require(not args.output_dir.exists(), "Output directory already exists; refusing reuse/overwrite")
    db = SnapshotDB(args.postgres_container, args.database)
    if args.command == "preflight":
        actual = check_runtime(args, db)
        stop_memory(args)
        db.assert_idle()
        inventory = db.fingerprint()
        if args.corpus_origin == "fresh":
            required_tables = required_corpus_tables(actual["config"])
            require(all(inventory["tables"].get("public."+name, {}).get("count") == 0
                        for name in required_tables), "Fresh corpus DB must be empty and initialized")
        else:
            require(args.source_corpus_dump is not None, "Historical control requires an explicit source dump")
        args.output_dir.mkdir(parents=True)
        write_json(args.output_dir / "preflight.json", {"protocol": PROTOCOL, "run_id": args.run_id,
            "runtime": actual, "database": args.database, "inventory": inventory,
            "corpus_origin": args.corpus_origin,
            "source_dump_sha256": digest(args.source_corpus_dump) if args.source_corpus_dump else None})
        print("Preflight saved. Memory service left stopped; start explicitly for ingestion.")
        return
    if args.command == "freeze":
        require(args.preflight is not None, "Freeze requires a pre-ingestion preflight artifact")
        preflight = json.loads(args.preflight.read_text())
        require(preflight["protocol"] == PROTOCOL and preflight["run_id"] == args.run_id
                and preflight["database"] == args.database, "Preflight identity mismatch")
        completion = validate_ingestion_manifest(args)
        actual = check_runtime(args, db)
        require(actual == preflight["runtime"], "Runtime changed since preflight")
        stop_memory(args)
        manifest = freeze_qualified_snapshot(db, args.output_dir, {"run_id": args.run_id, "runtime": actual,
            "preflight_sha256": digest(args.preflight), "preflight": preflight,
            "ingestion": completion})
        print("Frozen:", manifest["dump_sha256"], "Memory service left stopped.")
        return

    require(all((args.snapshot, args.parquet, args.agent_container, args.base_url, args.variant)),
            "Run requires snapshot, parquet, agent-container, base-url and variant")
    info = require_eval_container(args.agent_container, args.run_id, "agent")
    require(info["HostConfig"]["NetworkMode"] == "host", "Agent must use host networking")
    validate_agent_mounts(info, [args.source_root, args.parquet, args.output_dir, args.snapshot])
    require_eval_container(args.memory_container, args.run_id, "memory")
    args.snapshot = resolve_snapshot(args.snapshot)
    snapshot = json.loads((args.snapshot / "snapshot.json").read_text())
    require(snapshot["run_id"] == args.run_id and snapshot["database"] == args.database,
            "Snapshot/run mismatch")
    agent, _ = agent_preflight(args.agent_container, args.account, args.user, args.memory_url, fresh=True)
    cfg = json.loads(docker_text(args.agent_container, "cat", "/home/node/.openclaw/openclaw.json"))
    url = urlsplit(args.base_url)
    require(url.scheme == "http" and url.hostname == "127.0.0.1"
            and url.port == cfg["gateway"]["port"], "Answer URL does not match isolated agent")
    import requests
    conversation_id, probes = load_questions(args.parquet, args.conversation_index)
    validate_snapshot_dataset(snapshot, args, conversation_id)
    rubric_count = sum(len(p["rubric"]) for category in CATEGORIES for p in probes[category])
    token = os.environ.get("BEAM_AGENT_TOKEN")
    require(token, "Set BEAM_AGENT_TOKEN; no insecure default")
    args.output_dir.mkdir(parents=True)
    run = {"protocol": PROTOCOL, "variant": args.variant, "run_id": args.run_id,
        "status": "running", "snapshot_sha256": digest(args.snapshot / "snapshot.json"),
        "runner_sha256": digest(Path(__file__)),
        "safety_helpers_sha256": digest(Path(__file__).with_name("beam_eval_safety.py")),
        "dataset_sha256": digest(args.parquet), "agent": agent,
        "conversation_index": args.conversation_index,
        "conversation_id": conversation_id, "rubric_count": rubric_count,
        "questions": [], "restores": []}
    write_json(args.output_dir / "snapshot-manifest.json", snapshot)
    try:
        for category in CATEGORIES:
            for index, probe in enumerate(probes[category], 1):
                # Terminate queued after-turn/retry work before restoring SQL.
                stop_agent(args)
                stop_memory(args)
                require(digest(args.snapshot / "snapshot.json") == run["snapshot_sha256"],
                        "Frozen snapshot manifest changed during evaluation")
                audit = db.restore(args.snapshot)
                subprocess.run(["docker", "start", args.memory_container], check=True, capture_output=True)
                wait_health(args.memory_url)
                require(check_runtime(args, db) == snapshot["runtime"], "Runtime changed after restart")
                require(db.fingerprint() == snapshot["inventory"], "Startup mutated frozen corpus")
                subprocess.run(["docker", "start", args.agent_container], check=True, capture_output=True)
                wait_health(args.base_url, "/health")
                current_agent, sessions = agent_preflight(args.agent_container, args.account, args.user,
                                                         args.memory_url, fresh=False)
                require(current_agent == agent, "Agent configuration changed")
                session = f"{args.run_id}-{category}-{index}"
                namespace = os.environ.get("OG_V3_QA_ATTEMPT", "")
                if namespace:
                    require(bool(re.fullmatch(r"[a-z0-9-]+", namespace)), "Unsafe QA attempt namespace")
                    session += "-" + namespace
                require(session not in sessions, "Answer session already used")
                run["restores"].append({"session": session, **audit})
                write_json(args.output_dir / f"{session}.restore.json", audit)
                started = time.monotonic()
                response = requests.post(args.base_url.rstrip("/") + "/v1/responses", headers={
                    "Authorization": f"Bearer {token}", "X-OpenClaw-Agent-ID": "main",
                    "X-OpenClaw-Session-Key": session}, json={"model": "openclaw",
                    "input": f"Answer the question directly: {probe['question']}",
                    "stream": False, "user": args.user}, timeout=1800)
                response.raise_for_status()
                body = response.json()
                # Retain the HTTP answer even if subsequent evidence capture fails.
                write_json(args.output_dir / f"{session}.response.json", body)
                compose_path = args.output_dir / f"{session}.compose.json"
                from scripts.experiment_compose_capture import capture_compose
                captured_compose = capture_compose(args.agent_container, args.memory_container, session)
                with compose_path.open("x") as handle:
                    handle.write(captured_compose)
                compose = json.loads(compose_path.read_text())
                require("stats" in compose and "retrievedEvidence" in compose, "Incomplete compose capture")
                require(not any(item.get("type") in {"function_call", "function_call_output", "tool_call"}
                                for item in body.get("output", [])), "Response contains structured tool activity")
                log = collect_session(args.agent_container, session, probe["question"])
                with (args.output_dir / f"{session}.session.jsonl").open("x") as handle:
                    handle.write(log)
                tool_audit = audit_session(log, probe["question"])
                require(bool(response_text(body).strip()), "Empty answer")
                answer_name = f"{session}.answer.json"
                write_json(args.output_dir / answer_name, {"category": category, "question_index": index,
                    "session_key": session, "question": probe["question"], "expected": expected_answer(probe),
                    "source_chat_ids": probe.get("source_chat_ids", []),
                    "rubric": probe["rubric"], "response": response_text(body),
                    "elapsed_seconds": round(time.monotonic()-started, 3), "usage": body.get("usage", {})})
                files = {p.name: digest(p) for p in args.output_dir.glob(session + ".*")}
                run["questions"].append({"category": category, "question_index": index,
                    "answer_file": answer_name, "files": files, "audit": tool_audit})
                print(category, index, "captured and audited", flush=True)
        answers = {category: [] for category in CATEGORIES}
        for row in run["questions"]:
            answers[row["category"]].append(json.loads((args.output_dir / row["answer_file"]).read_text()))
        write_json(args.output_dir / "answers.json", answers)
        run["answers_sha256"] = digest(args.output_dir / "answers.json")
        run["status"] = "completed_unjudged"
    except BaseException as exc:
        run["status"] = "INVALID"
        run["failure_type"] = type(exc).__name__
        raise
    finally:
        try:
            stop_agent(args)
            stop_memory(args)
            require(digest(args.snapshot / "snapshot.json") == run["snapshot_sha256"],
                    "Frozen snapshot manifest changed before cleanup")
            run["final_restore"] = db.restore(args.snapshot)
        except Exception:
            run["status"] = "INVALID"
            run["final_restore_failed"] = True
            raise
        finally:
            write_json(args.output_dir / "run.json", run)
        print("Both evaluation containers left stopped; retained full snapshot permits recovery.")


if __name__ == "__main__":
    main()
