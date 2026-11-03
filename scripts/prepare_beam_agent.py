"""Preserve an explicitly configured, stopped BEAM agent across image startup.

Only insert missing empty auth keys required by the image's template-refresh
guard. Never invent or restore an answering configuration from an arbitrary
backup, and never change an existing auth value or any experiment setting.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.beam_eval_safety import (
    digest, json_digest, require, require_eval_container, validate_agent_config,
    validate_agent_mounts, write_json,
)


def guarded_config(config):
    result = deepcopy(config)
    values = result["plugins"]["entries"]["og-memory-context-engine"]["config"]
    for name in ("authApiKey", "authAccountId"):
        values.setdefault(name, "")
    return result


def prepare(args):
    info = require_eval_container(args.agent_container, args.run_id, "agent")
    require(not info["State"]["Running"], "Stop the isolated agent before preparing its configuration")
    output = args.output_dir.resolve()
    require(not output.exists(), "Refusing to overwrite agent preparation evidence")
    validate_agent_mounts(info, [output])
    mounts = [m for m in info["Mounts"] if m.get("Destination") == "/home/node/.openclaw"]
    require(len(mounts) == 1 and mounts[0].get("Type") == "bind" and mounts[0].get("RW"),
            "Agent configuration requires one writable, dedicated state bind mount")
    state = Path(mounts[0]["Source"]).resolve()
    path = state / "openclaw.json"
    require(path.is_file() and not path.is_symlink() and path.resolve().parent == state,
            "Expected a regular openclaw.json in the isolated state directory")
    before = path.read_bytes()
    original = json.loads(before)
    require(not any(value in before.decode() for value in (
        "__LLM_PROVIDER__", "__LLM_MODEL__", "__LLM_BASE_URL__", "__LLM_API_KEY__",
        "__GATEWAY_PORT__", "__OGMEM_URL__", "__OG_AUTH_API_KEY__", "__OG_AUTH_ACCOUNT_ID__")),
        "Resolve image template placeholders before preparing the agent")
    # If startup has already replaced the config, fail rather than silently
    # selecting a different model/policy or guessing which backup was intended.
    validate_agent_config(original, args.account, args.user, args.memory_url)
    require(original["gateway"]["port"] == urlsplit(args.base_url).port,
            "Agent gateway port differs from setup")
    patched = guarded_config(original)
    changed = patched != original
    mode = stat.S_IMODE(path.stat().st_mode)
    os.umask(0o077)
    output.mkdir(parents=True)
    with (output / "original.private.json").open("xb") as handle:
        handle.write(before)
    require(path.read_bytes() == before, "Agent configuration changed concurrently")
    if changed:
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=state,
                                             prefix=".beam-config-", delete=False) as handle:
                name = handle.name
                json.dump(patched, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(name, mode)
            require(path.read_bytes() == before, "Agent configuration changed concurrently")
            os.replace(name, path)
            name = None
        finally:
            if name is not None:
                Path(name).unlink(missing_ok=True)
    require(json.loads(path.read_bytes()) == patched, "Prepared config differs from the validated settings")
    report = {"status": "prepared", "run_id": args.run_id, "container_id": info["Id"],
              "configuration_changed": changed, "original_config_sha256": json_digest(original),
              "prepared_config_sha256": json_digest(patched), "file_sha256": digest(path),
              "inserted_empty_keys": [name for name in ("authApiKey", "authAccountId")
                  if name not in original["plugins"]["entries"]["og-memory-context-engine"]["config"]],
              "model_policy_and_credentials_preserved": True,
              "original_config_backup": "original.private.json"}
    write_json(output / "preparation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("agent-container", "run-id", "account", "user", "memory-url", "base-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "planned", "docker_or_provider_calls": 0,
                          "note": "Add --execute while the explicitly configured agent is stopped."}))
        return
    print(json.dumps(prepare(args), indent=2))


if __name__ == "__main__":
    main()
