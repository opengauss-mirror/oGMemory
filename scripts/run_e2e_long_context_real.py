#!/usr/bin/env python3
"""Run a real-service (non-mock) long-context E2E for AGFS + OpenClaw bridge.

Flow:
1) Verify AGFS is reachable
2) Verify OpenAI credentials (unless --allow-mock-llm)
3) Call OpenClaw bridge lifecycle (bootstrap -> after_turn -> assemble)
4) Print an execution summary

Notes:
- This script exercises the real OpenClaw plugin bridge implementation in
  `plugins/openclaw/bridge/memory_api.py`.
- It does NOT require running the OpenClaw daemon; it invokes bridge lifecycle
  functions directly in-process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from uuid import uuid4



def _load_bridge_module(repo_root: Path):
    """Import bridge module from local plugin path."""
    plugin_dir = repo_root / "plugins/openclaw"
    bridge_file = plugin_dir / "bridge" / "memory_api.py"

    import importlib.util

    spec = importlib.util.spec_from_file_location("memory_api", bridge_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules["memory_api"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _check_agfs(agfs_base_url: str) -> None:
    from pyagfs import AGFSClient

    client = AGFSClient(api_base_url=agfs_base_url)
    client.ls("/")


def _check_openclaw_cli() -> dict:
    """Best-effort check for OpenClaw CLI availability."""
    import shutil
    import subprocess

    result = {
        "openclaw_cli_found": False,
        "openclaw_version": None,
        "openclaw_doctor": None,
    }

    binary = shutil.which("openclaw")
    if not binary:
        return result

    result["openclaw_cli_found"] = True

    try:
        version = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        result["openclaw_version"] = (version.stdout or version.stderr).strip()
    except Exception as exc:
        result["openclaw_version"] = f"error: {exc}"

    try:
        doctor = subprocess.run(
            [binary, "doctor"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        result["openclaw_doctor"] = (doctor.stdout or doctor.stderr).strip()
    except Exception as exc:
        result["openclaw_doctor"] = f"error: {exc}"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real long-context E2E via AGFS + OpenClaw bridge")
    parser.add_argument(
        "--fixture",
        default="tests/fixtures/long_conversation_1000.json",
        help="Path to long conversation fixture JSON",
    )
    parser.add_argument(
        "--agfs-base-url",
        default=os.environ.get("AGFS_BASE_URL", "http://localhost:1833"),
        help="AGFS base URL",
    )
    parser.add_argument(
        "--session-id",
        default=f"e2e-real-{uuid4()}",
        help="Session id for this test run",
    )
    parser.add_argument(
        "--allow-mock-llm",
        action="store_true",
        help="Allow fallback to mock provider if OGMEM_API_KEY is not configured",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    fixture_path = (repo_root / args.fixture).resolve()

    if not fixture_path.exists():
        print(json.dumps({"status": "error", "error": f"fixture not found: {fixture_path}"}, ensure_ascii=False))
        return 2

    if not args.allow_mock_llm and not os.environ.get("OGMEM_API_KEY"):
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "OGMEM_API_KEY is required for non-mock full-chain run (or pass --allow-mock-llm)",
                },
                ensure_ascii=False,
            )
        )
        return 2

    # Force non-mock provider by default for "real" mode
    if os.environ.get("OGMEM_API_KEY"):
        os.environ["CONTEXTENGINE_PROVIDER"] = "openai"

    os.environ["AGFS_BASE_URL"] = args.agfs_base_url

    try:
        _check_agfs(args.agfs_base_url)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "error": f"AGFS unreachable at {args.agfs_base_url}", "detail": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2

    messages = json.loads(fixture_path.read_text(encoding="utf-8"))
    bridge = _load_bridge_module(repo_root)

    # Optional visibility into OpenClaw CLI status
    openclaw_info = _check_openclaw_cli()

    bootstrap_result = bridge.bootstrap({"sessionId": args.session_id})
    after_turn_result = bridge.after_turn({"sessionId": args.session_id, "messages": messages})
    assemble_result = bridge.compose(
        {"sessionId": args.session_id, "messages": messages[-12:], "tokenBudget": 128_000}
    )
    health_result = bridge.memory_health()

    summary = {
        "status": "ok",
        "mode": "real" if os.environ.get("CONTEXTENGINE_PROVIDER") == "openai" else "mock-allowed",
        "session_id": args.session_id,
        "fixture": str(fixture_path.relative_to(repo_root)),
        "fixture_messages": len(messages),
        "bootstrap": bootstrap_result,
        "after_turn": after_turn_result,
        "assemble_estimated_tokens": assemble_result.get("estimatedTokens"),
        "assemble_system_prompt_preview": (assemble_result.get("systemPromptAddition") or "")[:300],
        "memory_health": health_result,
        "openclaw_cli": openclaw_info,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
