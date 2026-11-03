"""Live Deep Agents/oGMemory end-to-end check with a dedicated test identity.

Install: python -m pip install '.[deepagents]' 'langchain-openai>=1.1,<2'
Run: python tests/e2e/run_deepagents_smoke.py

Required environment variables:
  OGMEM_URL, OGMEM_API_KEY, OGMEM_ACCOUNT_ID, OGMEM_USER_ID, OGMEM_AGENT_ID
  LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
Optional: LLM_EXTRA_BODY (JSON), OGMEM_VERIFY_WAIT_SECONDS (default 300).
Writes durable test memory. Use a dedicated user. Reports default to the
repository's ignored .tmp/deepagents-e2e/ directory; no credentials are recorded.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="JSON report path (default: .tmp/deepagents-e2e/<run>.json)")
    parser.add_argument("--wait-seconds", type=int, help="Maximum extraction/index wait (default: 300 or OGMEM_VERIFY_WAIT_SECONDS)")
    args = parser.parse_args(argv)
    required = (
        "OGMEM_URL", "OGMEM_API_KEY", "OGMEM_ACCOUNT_ID", "OGMEM_USER_ID", "OGMEM_AGENT_ID",
        "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY",
    )
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        parser.error("Set dedicated test connection variables: " + ", ".join(missing))
    try:
        wait_seconds = args.wait_seconds if args.wait_seconds is not None else int(
            os.getenv("OGMEM_VERIFY_WAIT_SECONDS", "300")
        )
    except ValueError:
        parser.error("OGMEM_VERIFY_WAIT_SECONDS must be a positive integer")
    if wait_seconds <= 0:
        parser.error("--wait-seconds must be positive")

    # Allow direct script invocation from any working directory.
    sys.path.insert(0, str(ROOT))
    try:
        from scripts.verify_deepagents import main as verify
    except ModuleNotFoundError as exc:
        parser.error(f"Missing dependency {exc.name}; install '.[deepagents]' and langchain-openai")

    report = args.report or ROOT / ".tmp" / "deepagents-e2e" / f"{uuid4().hex}.json"
    report = report.resolve()
    report.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    verify(report_path=report, wait_seconds=wait_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
