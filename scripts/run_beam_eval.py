#!/usr/bin/env python3
"""Portable stage launcher for the native BEAM tools; offline unless --execute.

Example: python scripts/run_beam_eval.py preflight --setup /data/beam/setup.json
Stages: prepare-agent -> preflight -> ingest -> freeze -> qa -> grade. Each stage preserves its
own receipts and refuses existing output. --resume applies to grading only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_setup(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Setup must be a JSON object")
    for name in ("parquet", "output_root"):
        value = config.get(name)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("Use an explicit absolute path for " + name)
    output = Path(config["output_root"]).resolve()
    if output == ROOT or output in ROOT.parents or ROOT in output.parents:
        raise ValueError("Keep experiment output outside the source checkout")
    source = config.get("source_root", str(ROOT))
    if Path(source).resolve() != ROOT:
        raise ValueError("source_root must be this checkout, not a historical worktree")
    config["source_root"] = str(ROOT)
    for name in ("database", "memory_container", "postgres_container", "agent_container",
                 "run_id", "account", "user", "memory_url", "base_url"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError("Missing setup field: " + name)
    index = config.get("conversation_index", 0)
    if type(index) is not int or index < 0:
        raise ValueError("conversation_index must be a non-negative integer")
    config["conversation_index"] = index
    for name in ("memory_url", "base_url", "judge_base_url"):
        if name not in config:
            continue
        endpoint = urlsplit(config[name])
        if (endpoint.scheme not in ("http", "https") or not endpoint.hostname or
                endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
            raise ValueError("Use a plain service URL without credentials/query: " + name)
    if config.get("variant", "baseline") not in ("baseline", "graph"):
        raise ValueError("Unknown comparison label")
    return config


def stage_command(stage, config, execute=False, resume=False, verify_offline=False):
    output = Path(config["output_root"])
    command = [sys.executable, "-m", "scripts." + (
        "prepare_beam_agent" if stage == "prepare-agent" else
        "ingest_beam_fresh" if stage == "ingest" else
        "grade_beam_answers" if stage == "grade" else "run_stage8r_beam_clean_eval")]
    if stage == "prepare-agent":
        for key in ("agent_container", "run_id", "account", "user", "memory_url", "base_url"):
            command += ["--" + key.replace("_", "-"), config[key]]
        command += ["--output-dir", str(output / "prepared-agent")]
    elif stage == "grade":
        from scripts.beam_dataset import load_questions
        _, probes = load_questions(Path(config["parquet"]), config["conversation_index"])
        rubric_count = sum(len(q["rubric"]) for group in probes.values() for q in group)
        command += [str(output / "qa/answers.json"), str(output / "grading"),
                    "--questions", "20", "--rubrics", str(rubric_count),
                    "--model", config.get("judge_model", os.environ.get("BEAM_JUDGE_MODEL", "deepseek-chat")),
                    "--base-url", config.get("judge_base_url", os.environ.get(
                        "BEAM_JUDGE_BASE_URL", os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")))]
        if resume:
            command.append("--resume")
        if verify_offline:
            command.append("--verify-offline")
    else:
        if stage != "ingest":
            command.append("run" if stage == "qa" else stage)
        for key in ("database", "memory_container", "postgres_container", "run_id", "account",
                    "user", "memory_url", "source_root"):
            command.extend(["--" + key.replace("_", "-"), config[key]])
        command.extend(["--conversation-index", str(config["conversation_index"]),
                        "--parquet", config["parquet"], "--output-dir", str(output / stage)])
        if stage in ("ingest", "freeze"):
            command.extend(["--preflight", str(output / "preflight/preflight.json")])
        if stage == "ingest":
            command.extend(["--memory-api-key-env", config.get("memory_api_key_env", "BEAM_MEMORY_API_KEY")])
        if stage == "freeze":
            command.extend(["--ingestion-manifest", str(output / "ingest/ingestion.json")])
        if stage == "qa":
            command.extend(["--snapshot", str(output / "freeze"), "--agent-container", config["agent_container"],
                            "--base-url", config["base_url"], "--variant", config.get("variant", "baseline")])
    if execute:
        command.append("--execute")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("prepare-agent", "preflight", "ingest", "freeze", "qa", "grade"))
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Permit Docker/SQL mutations and paid calls")
    parser.add_argument("--resume", action="store_true", help="Continue retained grading calls only")
    parser.add_argument("--verify-offline", action="store_true", help="Replay retained grading without API access")
    args = parser.parse_args(argv)
    if (args.resume or args.verify_offline) and args.stage != "grade":
        parser.error("Only grading supports --resume/--verify-offline")
    if args.verify_offline and (args.execute or not args.resume):
        parser.error("Offline verification requires --resume and cannot use --execute")
    config = load_setup(args.setup)
    command = stage_command(args.stage, config, args.execute, args.resume, args.verify_offline)
    if not args.execute and not args.verify_offline:
        print(json.dumps({"status": "planned", "stage": args.stage, "command": command,
                          "note": "No Docker, HTTP, provider call or output write; add --execute."}, indent=2))
        return
    os.umask(0o077)
    if args.stage == "qa":
        from scripts.beam_eval_safety import agent_preflight, require, require_eval_container
        from scripts.run_stage8r_beam_clean_eval import wait_health
        receipt = Path(config["output_root"]) / "prepared-agent/preparation.json"
        require(receipt.is_file(), "Run prepare-agent --execute before starting QA")
        prepared = json.loads(receipt.read_text())
        info = require_eval_container(config["agent_container"], config["run_id"], "agent")
        require(prepared.get("status") == "prepared" and prepared.get("run_id") == config["run_id"]
                and prepared.get("container_id") == info["Id"], "Agent preparation identity changed")
        try:
            subprocess.run(["docker", "start", config["agent_container"]], check=True, capture_output=True)
            wait_health(config["base_url"], "/health")
            actual, _ = agent_preflight(config["agent_container"], config["account"], config["user"],
                                         config["memory_url"], fresh=True)
            require(actual["config_sha256"] == prepared["prepared_config_sha256"],
                    "Agent startup replaced or changed the prepared configuration")
            subprocess.run(command, cwd=ROOT, check=True)
        except BaseException:
            subprocess.run(["docker", "stop", config["agent_container"]], check=False, capture_output=True)
            raise
    else:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
