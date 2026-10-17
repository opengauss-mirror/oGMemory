"""ogmem config command - manage configuration."""

from __future__ import annotations

import json
from pathlib import Path

from cli.lib.colors import CYAN, GREEN, NC, YELLOW, fail, ok
from cli.lib.process import find_project_root


def run(args) -> int:
    """Handle config subcommands."""
    cmd = getattr(args, "config_action", "show")
    if cmd == "show":
        from providers.unified_config import get_config
        try:
            cfg = get_config()
            print(f"\n{GREEN}oG-Memory Configuration{NC}\n")
            print(json.dumps(cfg.dump_summary(), indent=2, default=str))
            print()
            return 0
        except Exception as e:
            print(fail(f"Failed: {e}"))
            return 1
    elif cmd == "init":
        root = find_project_root()
        cfg_path = root / "config" / "ogmem.yaml"
        if cfg_path.exists():
            print(fail(f"Config exists: {cfg_path}"))
            return 1
        try:
            from cli.lib.config_loader import get_reference_yaml, load_template
            tmpl = load_template(get_reference_yaml())
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(tmpl)
            print(ok(f"Created: {CYAN}{cfg_path}{NC}\n  {YELLOW}Edit API keys and settings as needed{NC}\n"))
            return 0
        except (IOError, FileNotFoundError) as e:
            print(fail(f"Failed: {e}"))
            return 1
    print(fail(f"Unknown subcommand: {cmd}"))
    return 1
