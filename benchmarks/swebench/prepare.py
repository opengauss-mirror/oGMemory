#!/usr/bin/env python3
"""Prepare frozen SWE-bench inputs. Shared file/config helpers live here."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HARNESS_VERSION = "4.1.0"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def command(args, *, cwd=None, env=None, timeout=600):
    result = subprocess.run(
        [str(a) for a in args],
        cwd=cwd,
        env=env,
        timeout=timeout,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        # Do not echo arguments: subprocesses may receive private configuration.
        raise RuntimeError(
            f"{Path(str(args[0])).name} failed ({result.returncode}): {result.stderr[-2000:]}"
        )
    return result.stdout.strip()


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["work_dir"] = str((path.parent / cfg["work_dir"]).resolve())
    if not cfg.get("arms") or len(set(cfg["arms"])) != len(cfg["arms"]):
        raise ValueError("arms must be a nonempty, unique list")
    if set(cfg["arms"]) - {"baseline", "sql"}:
        raise ValueError("Only baseline and sql are supported")
    if not isinstance(cfg.get("concurrency"), int) or cfg["concurrency"] < 1:
        raise ValueError("concurrency must be a positive integer")
    ids = cfg["instances"]
    if ids != "all" and (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))):
        raise ValueError("instances must be all or a nonempty unique list")
    if not cfg["model"].get("name"):
        raise ValueError("model.name is required")
    return cfg


def expand_environment(value):
    """Expand required product settings only at execution time, never in snapshots."""
    if isinstance(value, dict):
        return {k: expand_environment(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_environment(v) for v in value]
    if not isinstance(value, str):
        return value

    def replacement(match):
        name = match[1]
        if not os.environ.get(name):
            raise ValueError(f"Required environment variable is empty: {name}")
        return os.environ[name]

    return re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", replacement, value)


def image_name(cfg, iid):
    e = cfg["evaluation"]
    # SWE-bench 4.1.0 TestSpec.instance_image_key naming for remote x86 images.
    return (
        f"{e['namespace']}/sweb.eval.x86_64.{iid.lower().replace('__', '_1776_')}:{e['image_tag']}"
    )


def selected_rows(cfg, rows):
    by_id = {r["instance_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate dataset instance IDs")
    ids = list(by_id) if cfg["instances"] == "all" else cfg["instances"]
    missing = set(ids) - by_id.keys()
    if missing:
        raise ValueError(f"Unknown instances: {sorted(missing)}")
    return [by_id[i] for i in ids]


def dataset(cfg):
    from datasets import load_dataset

    cache = Path(cfg["work_dir"]) / "inputs"
    path = cache / "dataset.json"
    source = {"dataset": cfg["dataset"], "revision": cfg["dataset_revision"], "split": "test"}
    if path.exists():
        rows = read_json(path)
        manifest = read_json(cache / "dataset-source.json")
        if manifest["source"] != source or manifest["sha256"] != digest(rows):
            raise ValueError("Dataset provenance differs; use a new work_dir")
    else:
        rows = list(load_dataset(cfg["dataset"], split="test", revision=cfg["dataset_revision"]))
        if cfg["dataset"] == "princeton-nlp/SWE-bench_Lite" and len(rows) != 300:
            raise ValueError("Expected 300 Lite test instances")
        save_json(path, rows)
        save_json(cache / "dataset-source.json", {"source": source, "sha256": digest(rows)})
    return selected_rows(cfg, rows)


def prepare_instance(cfg, row):
    iid = row["instance_id"]
    cache = Path(cfg["work_dir"]) / "inputs" / iid
    repo = cache / "repo"
    cache.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        command(["git", "init", repo])
        command(
            ["git", "remote", "add", "origin", f"https://github.com/{row['repo']}.git"], cwd=repo
        )
    for ref in dict.fromkeys([row["base_commit"], row.get("environment_setup_commit")]):
        if ref:
            command(["git", "fetch", "--depth", "1", "origin", ref], cwd=repo)
    command(["git", "checkout", "--detach", row["base_commit"]], cwd=repo)
    image = image_name(cfg, iid)
    try:
        info = json.loads(command(["docker", "image", "inspect", image]))[0]
    except RuntimeError:
        command(["docker", "pull", "--platform", "linux/amd64", image], timeout=3600)
        info = json.loads(command(["docker", "image", "inspect", image]))[0]
    if info["Architecture"] != "amd64":
        raise ValueError(f"Expected amd64 image: {image}")
    command(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--entrypoint",
            "/bin/bash",
            image,
            "-lc",
            "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && python --version",
        ],
        timeout=180,
    )
    save_json(
        cache / "prepared.json",
        {
            "instance_id": iid,
            "base_commit": row["base_commit"],
            "image": image,
            "image_id": info["Id"],
            "arch": "amd64",
        },
    )
    print(f"READY {iid}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    (work / ".gitignore").write_text("*\n", encoding="utf-8")
    rows = dataset(cfg)
    print(f"Selected {len(rows)} instances, {len(rows) * len(cfg['arms'])} jobs")
    if not args.dataset_only:
        command(["docker", "info"], timeout=30)
        for row in rows:
            prepare_instance(cfg, row)


if __name__ == "__main__":
    main()
