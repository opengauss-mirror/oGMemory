#!/usr/bin/env python3
"""Render the product-owned code policy for hooks and experiment preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from code_policy import render_code_policy  # noqa: E402
from ogm_plugin_request import effective_tool_output_filter_enabled  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rendered = render_code_policy(
        read_filter_enabled=effective_tool_output_filter_enabled("Read"),
        bash_filter_enabled=effective_tool_output_filter_enabled("Bash"),
    )
    if args.json:
        sys.stdout.write(json.dumps(rendered, ensure_ascii=False, sort_keys=True))
    else:
        sys.stdout.write(str(rendered["text"]).rstrip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
