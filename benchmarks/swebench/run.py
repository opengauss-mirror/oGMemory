#!/usr/bin/env python3
"""Run baseline/SQL jobs, resume unfinished jobs, or grade an existing patch."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import yaml
from prepare import (
    HARNESS_VERSION,
    ROOT,
    command,
    digest,
    expand_environment,
    image_name,
    load_config,
    read_json,
    save_json,
    selected_rows,
)

BUILTIN_TOOLS = "Bash,Read,Edit,Write,Glob,Grep"


PRODUCT_PATHS = (
    "server",
    "service",
    "providers",
    "features",
    "plugins/claude-code",
    "cli",
    "core",
    "fs",
    "retrieval",
    "index",
    "commit",
    "session",
    "extraction",
    "pyproject.toml",
)


def check_sql_environment(cfg):
    # Product imports reload config/.env even after cleaning the child environment.
    # Refuse it without reading or changing the user's configuration.
    if "sql" in cfg["arms"] and (ROOT / "config" / ".env").exists():
        raise ValueError(
            "SQL benchmark requires a checkout without config/.env; use a clean "
            "checkout and supply benchmark settings through config.yaml and the "
            "documented environment variables. Existing config/.env was not changed."
        )


def source_identity():
    tracked = command(["git", "ls-files", "--", *PRODUCT_PATHS], cwd=ROOT).splitlines()
    tracked += [
        f"benchmarks/swebench/{name}"
        for name in ("prepare.py", "run.py", "report.py", "requirements.txt")
    ]
    return {
        "source_hash": digest(
            {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in tracked}
        ),
        "ogmemory_dirty": bool(command(["git", "diff", "HEAD", "--", *PRODUCT_PATHS], cwd=ROOT)),
    }


def clean_env():
    # Do not inherit another experiment's identity, model routing, or MCP settings.
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("OGMEM", "OG_", "ANTHROPIC_", "CLAUDE"))
    }


def model_env(cfg):
    model = cfg["model"]
    if model["name"] == "YOUR_MODEL_NAME":
        raise ValueError("Set model.name to a model supported by your Messages API endpoint")
    env = clean_env()
    for field, target in [
        ("base_url_env", "ANTHROPIC_BASE_URL"),
        ("api_key_env", "ANTHROPIC_AUTH_TOKEN"),
    ]:
        key = model[field]
        if not os.environ.get(key):
            raise ValueError(f"Missing model environment variable: {key}")
        env[target] = os.environ[key]
    return env


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stop_process(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


@contextlib.contextmanager
def background(args, log, *, env, cwd=ROOT):
    with Path(log).open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(
            [str(a) for a in args],
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        save_json(
            Path(log).with_suffix(".process.json"),
            {"pid": proc.pid, "command": [str(a) for a in args]},
        )
        try:
            yield proc
        finally:
            stop_process(proc)


def api(url, key, path, body=None):
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": key,
        "X-Account-ID": "swebench",
        "X-User-ID": "tester",
        "X-Agent-ID": "solver",
    }
    request = Request(
        url + path, headers=headers, data=json.dumps(body).encode() if body is not None else None
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def wait_http(url, key, proc, timeout):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if proc.poll() is not None:
            raise RuntimeError("oGMemory exited before readiness; inspect service log")
        try:
            api(url, key, "/api/v1/health")
            return
        except (URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("oGMemory readiness timeout")


def service_config(cfg, workspace, db_port, password, key, http_port, mcp_port):
    embedding = expand_environment(cfg["ogmemory"]["embedding"])
    return {
        "storage": {
            "backend": "sql",
            "dialect": "opengauss",
            "pool_size": 3,
            "connection_string": f"host=127.0.0.1 port={db_port} dbname=swebench user=swebench password={password}",
        },
        # candidate_local computes ephemeral candidate embeddings, no global code index.
        "vector_db": {"type": "memory"},
        "llm": {"provider": "mock"},
        "embedding": embedding,
        "identity": {"account_id": "swebench", "user_id": "tester", "agent_id": "solver"},
        # Configured tenant admin keys bootstrap the account in current dev.
        "auth": {"role_control_enabled": True, "admin_api_keys": [f"swebench:{key}"]},
        "service": {"http_port": http_port, "workers": 1},
        "features": {
            "code_resources_enabled": True,
            "code_sync_enabled": False,
            "code_workspace_allowed_roots": [str(workspace)],
            "code_retrieval_enabled": True,
            "code_retrieval": {"mode": "candidate_local"},
            "tool_output_filter": {"enabled": cfg["ogmemory"]["tool_output_filter"]},
        },
        "mcp": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": mcp_port,
            "default_toolset": "code",
            "enable_code_edit_tools": True,
            "enable_write_tools": False,
            "enable_delete_tools": False,
        },
    }


@contextlib.contextmanager
def sql_database(cfg, attempt, private):
    """Start an isolated openGauss database and verify its engine and PG mode."""
    import psycopg2

    name = "ogmswe-db-" + uuid.uuid4().hex[:16]
    # openGauss enforces an upper bound of 32 characters as well as complexity.
    password = "SwE!9a" + uuid.uuid4().hex[:24]
    db_env = private / "opengauss.env"
    db_env.write_text(
        f"GS_PASSWORD={password}\nGS_DB=postgres\nGS_USERNAME=swebench\n", encoding="utf-8"
    )
    db_env.chmod(0o600)
    save_json(attempt / "resources.json", {"database_container": name})
    try:
        command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--env-file",
                db_env,
                "-p",
                "127.0.0.1::5432",
                cfg["ogmemory"]["opengauss_image"],
            ]
        )
        info = json.loads(command(["docker", "inspect", name]))[0]
        db_port = info["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]
        timeout = cfg["ogmemory"]["startup_timeout_seconds"]
        until = time.monotonic() + timeout
        while True:
            state = json.loads(command(["docker", "inspect", name]))[0]["State"]
            if not state["Running"]:
                raise RuntimeError("openGauss container exited during initialization")
            try:
                # The entrypoint starts and stops a temporary database during init.
                # Wait for the final server, not that temporary listener.
                if (
                    command(["docker", "exec", name, "cat", "/proc/1/comm"], timeout=10)
                    != "gaussdb"
                ):
                    raise RuntimeError("openGauss initialization is still running")
                with contextlib.closing(
                    psycopg2.connect(
                        host="127.0.0.1",
                        port=db_port,
                        dbname="postgres",
                        user="swebench",
                        password=password,
                        connect_timeout=5,
                    )
                ) as connection:
                    with connection.cursor() as cursor:
                        cursor.execute("SELECT 1")
                break
            except (RuntimeError, psycopg2.OperationalError):
                if time.monotonic() > until:
                    raise TimeoutError(
                        "openGauss readiness timeout; check the configured image"
                    ) from None
                time.sleep(1)
        command(
            [
                "docker",
                "exec",
                name,
                "su",
                "-",
                "omm",
                "-c",
                "gsql -d postgres -v ON_ERROR_STOP=1 -c "
                "\"CREATE DATABASE swebench OWNER swebench DBCOMPATIBILITY 'PG';\"",
            ]
        )
        with contextlib.closing(
            psycopg2.connect(
                host="127.0.0.1",
                port=db_port,
                dbname="swebench",
                user="swebench",
                password=password,
                connect_timeout=5,
            )
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                cursor.execute("SHOW sql_compatibility")
                compatibility = cursor.fetchone()[0]
        if "opengauss" not in version.lower() or compatibility != "PG":
            raise RuntimeError("SQL benchmark requires openGauss with sql_compatibility=PG")
        save_json(
            attempt / "database.json",
            {
                "version": version,
                "sql_compatibility": compatibility,
                "dialect": "opengauss",
                "image_id": info["Image"],
            },
        )
        yield db_port, password
    finally:
        result = subprocess.run(["docker", "rm", "-f", "-v", name], capture_output=True, timeout=30)
        save_json(
            attempt / "cleanup.json",
            {"database_container": name, "removed": result.returncode == 0},
        )


@contextlib.contextmanager
def sql_services(cfg, attempt, workspace):
    check_sql_environment(cfg)
    key = uuid.uuid4().hex
    http_port, mcp_port = free_port(), free_port()
    private = attempt / "private"
    private.mkdir(mode=0o700)
    with sql_database(cfg, attempt, private) as (db_port, password):
        timeout = cfg["ogmemory"]["startup_timeout_seconds"]
        generated = private / "ogmemory.yaml"
        generated.write_text(
            yaml.safe_dump(
                service_config(cfg, workspace, db_port, password, key, http_port, mcp_port)
            ),
            encoding="utf-8",
        )
        generated.chmod(0o600)
        env = clean_env()
        env.update(OGMEM_CONFIG=str(generated), PYTHONPATH=str(ROOT))
        url = f"http://127.0.0.1:{http_port}"
        with background(
            [
                sys.executable,
                "-m",
                "gunicorn",
                "--workers",
                "1",
                "--timeout",
                "900",
                "--bind",
                f"127.0.0.1:{http_port}",
                "server.app:app",
            ],
            attempt / "http.log",
            env=env,
        ) as http:
            wait_http(url, key, http, timeout)
            api(url, key, "/api/v1/admin/accounts/swebench/users", {"user_id": "tester"})
            api(
                url,
                key,
                "/api/v1/admin/accounts/swebench/agents",
                {"agent_id": "solver", "owner_user_id": "tester"},
            )
            registration = api(
                url,
                key,
                "/api/v1/resources/code/sync/registrations",
                {
                    "path": str(workspace),
                    "name": "task",
                    "trigger_mode": "manual",
                    "watch_enabled": False,
                    "wait_mode": "accepted",
                },
            )
            save_json(attempt / "registration.json", registration)
            with background(
                [
                    sys.executable,
                    "-m",
                    "cli.main",
                    "mcp",
                    "--config",
                    generated,
                    "--transport",
                    "streamable-http",
                ],
                attempt / "mcp.log",
                env=env,
            ) as mcp:
                until = time.monotonic() + timeout
                while True:
                    if mcp.poll() is not None:
                        raise RuntimeError("MCP exited before readiness; inspect mcp.log")
                    try:
                        with socket.create_connection(("127.0.0.1", mcp_port), timeout=1):
                            break
                    except OSError:
                        if time.monotonic() > until:
                            raise TimeoutError("MCP readiness timeout") from None
                        time.sleep(0.5)
                yield {
                    "url": url,
                    "key": key,
                    "config": str(generated),
                    "mcp_url": f"http://127.0.0.1:{mcp_port}/mcp",
                }


@contextlib.contextmanager
def task_container(cfg, row, workspace, attempt, expected):
    identity = grading_identity(cfg, row, expected)
    name = "ogmswe-code-" + uuid.uuid4().hex[:16]
    save_json(attempt / "task-container.json", {"name": name})
    try:
        command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--platform",
                "linux/amd64",
                "--mount",
                f"type=bind,source={workspace},target=/testbed",
                "--entrypoint",
                "sleep",
                identity["image_id"],
                "infinity",
            ]
        )
        prepare_task_environment(cfg, row, name, workspace, attempt)
        yield name
    finally:
        result = subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        save_json(
            attempt / "task-container-cleanup.json",
            {"name": name, "removed": result.returncode == 0},
        )


def task_setup_script(row):
    from swebench.harness.constants import MAP_REPO_TO_INSTALL, MAP_REPO_VERSION_TO_SPECS

    specs = MAP_REPO_VERSION_TO_SPECS[row["repo"]][row["version"]]
    commands = [
        "set -euo pipefail",
        "source /opt/miniconda3/etc/profile.d/conda.sh",
        "conda activate testbed",
        "cd /testbed",
        "git config --global --add safe.directory /testbed",
    ]
    if row["repo"] in MAP_REPO_TO_INSTALL:
        commands.append(MAP_REPO_TO_INSTALL[row["repo"]])
    commands.extend(specs.get("pre_install", []))
    if specs.get("install"):
        commands.append(specs["install"])
    commands.append("python --version")
    return "\n".join(commands) + "\n"


def prepare_task_environment(cfg, row, container, workspace, attempt):
    """Rebuild editable installs after the bind mount hides image build artifacts."""
    script = task_setup_script(row)
    (attempt / "setup.sh").write_text(script, encoding="utf-8")
    with (attempt / "setup.log").open("w", encoding="utf-8") as log:
        try:
            subprocess.run(
                ["docker", "exec", "-i", "-w", "/testbed", container, "bash"],
                input=script,
                text=True,
                encoding="utf-8",
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=cfg["evaluation"].get("setup_timeout_seconds", 1800),
                check=True,
            )
        finally:
            command(
                [
                    "docker",
                    "exec",
                    container,
                    "chown",
                    "-R",
                    f"{os.getuid()}:{os.getgid()}",
                    "/testbed",
                ]
            )
    # Installation can adjust tracked build configuration. Keep generated build
    # artifacts but start the agent from the dataset's exact tracked sources.
    command(
        ["git", "restore", "--source", row["base_commit"], "--staged", "--worktree", "."],
        cwd=workspace,
    )
    generated = command(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=workspace)
    # Exclude only files produced before the agent starts, not future new files.
    paths = [p for p in generated.split("\0") if p]
    with (workspace / ".git" / "info" / "exclude").open("a", encoding="utf-8") as excludes:
        for path in paths:
            escaped = "".join("\\" + c if c in "\\*?[] #!" else c for c in path)
            excludes.write("\n/" + escaped)
        excludes.write("\n")
    save_json(attempt / "setup.json", {"status": "completed", "excluded_install_files": paths})


def solve(cfg, row, arm, attempt, workspace, container, service):
    env = model_env(cfg)
    env.update(
        CLAUDE_CONFIG_DIR=str(attempt / "private" / "claude"),
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
    )
    servers = {}
    if service:
        env.update(
            OGMEM_CONFIG=service["config"],
            OG_MEMORY_URL=service["url"],
            OG_AUTH_API_KEY=service["key"],
            OG_ACCOUNT_ID="swebench",
            OG_USER_ID="tester",
            OG_AGENT_ID="solver",
            OGMEM_CODE_MEMORY_ONLY="true",
            OGMEM_CODE_RETRIEVAL_MODE="candidate_local",
            OGMEM_AUTO_CAPTURE="false",
            OGMEM_AUTO_RECALL="true",
            OGMEM_CLAUDE_MCP_RECOVERY_ENABLED="true",
            OGMEM_TOOL_OUTPUT_FILTER_ENABLED=str(cfg["ogmemory"]["tool_output_filter"]).lower(),
        )
        servers["ogmemory"] = {
            "type": "http",
            "url": service["mcp_url"],
            "headers": {
                "X-API-Key": service["key"],
                "X-Account-ID": "swebench",
                "X-User-ID": "tester",
                "X-Agent-ID": "solver",
            },
        }
    mcp_path = attempt / "private" / "mcp.json"
    mcp_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    save_json(mcp_path, {"mcpServers": servers})
    mcp_path.chmod(0o600)
    # No gold patch, test_patch, hints, or grading test lists in the solver input.
    prompt = (
        f"Fix this repository issue. Modify files in the current workspace; do not just print a diff.\n"
        f"Do not commit changes, inspect other workspaces, or look up upstream fixes.\n"
        f"The repository is mounted at /testbed in container {container}. Run tests using:\n"
        f"docker exec -w /testbed {container} bash -lc "
        "'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && <test command>'\n"
        f"Do not alter dependencies outside this task container.\n\nIssue:\n{row['problem_statement']}\n"
    )
    (attempt / "prompt.txt").write_text(prompt, encoding="utf-8")
    args = [
        cfg["model"]["claude"],
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        cfg["model"]["name"],
        "--max-turns",
        str(cfg["model"]["max_turns"]),
        "--tools",
        BUILTIN_TOOLS,
        "--setting-sources",
        "",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--mcp-config",
        str(mcp_path),
        "--dangerously-skip-permissions",
    ]
    if service:
        args += ["--plugin-dir", str(ROOT / "plugins" / "claude-code")]
    with (
        (attempt / "session.jsonl").open("w", encoding="utf-8") as output,
        (attempt / "agent.log").open("w") as err,
    ):
        proc = subprocess.Popen(
            args,
            cwd=workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=err,
            text=True,
            encoding="utf-8",
            start_new_session=True,
        )
        save_json(attempt / "agent.process.json", {"pid": proc.pid})
        try:
            proc.communicate(prompt, timeout=cfg["model"]["timeout_seconds"])
            if proc.returncode:
                raise RuntimeError(f"Agent exited {proc.returncode}; inspect agent.log")
        finally:
            stop_process(proc)


def extract_patch(workspace, base_commit):
    # Include new files, and diff against the immutable base even if the agent committed.
    command(["git", "add", "-N", "--", "."], cwd=workspace)
    # Preserve trailing context whitespace: stripping a unified diff can corrupt it.
    return subprocess.run(
        ["git", "-c", "core.fileMode=false", "diff", "--binary", base_commit, "--"],
        cwd=workspace,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")


def evaluate_child(attempt):
    """Runs in its own process/cwd: official harness uses cwd-relative log paths."""
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.test_spec.test_spec import make_test_spec

    import docker

    data = read_json(attempt / "evaluation-input.json")
    row, cfg = data["row"], data["config"]
    patch = (attempt / "submitted.patch").read_text(encoding="utf-8")
    if not patch.strip():
        save_json(attempt / "validation.json", {"status": "empty_patch", "resolved": None})
        return
    spec = make_test_spec(
        row,
        namespace=cfg["evaluation"]["namespace"],
        arch="x86_64",
        instance_image_tag=cfg["evaluation"]["image_tag"],
    )
    # Test-spec preparation can fetch remote dependencies and take time.
    # Recheck the tag immediately before handing it to the official harness.
    grading_identity(cfg, row, read_json(attempt / "grading-environment.json"))
    prediction = {
        "instance_id": row["instance_id"],
        "model_name_or_path": "solver",
        "model_patch": patch,
    }
    run_id = uuid.uuid4().hex
    save_json(attempt / "container.json", {"name": spec.get_instance_container_name(run_id)})
    result = run_instance(
        spec,
        prediction,
        False,
        False,
        docker.from_env(),
        run_id,
        timeout=cfg["evaluation"]["timeout_seconds"],
    )
    report = Path("logs/run_evaluation") / run_id / "solver" / row["instance_id"] / "report.json"
    if not result["completed"] or not report.exists():
        raise RuntimeError("Official evaluation incomplete; inspect logs/run_evaluation")
    official = read_json(report)
    resolved = official[row["instance_id"]].get("resolved")
    if type(resolved) is not bool:
        raise RuntimeError("Official instance verdict is missing")
    save_json(
        attempt / "validation.json",
        {
            "status": "completed",
            "resolved": resolved,
            "report_reused": False,
            "official_report": str(report.resolve()),
            "report": official,
        },
    )


def grading_identity(cfg, row, expected=None):
    if sys.platform != "linux":
        raise RuntimeError("Official evaluation requires Linux or WSL2")
    version = importlib.metadata.version("swebench")
    if version != HARNESS_VERSION:
        raise ValueError(f"Official evaluation requires swebench=={HARNESS_VERSION}")
    image = image_name(cfg, row["instance_id"])
    info = json.loads(command(["docker", "image", "inspect", image]))[0]
    if info["Architecture"] != "amd64":
        raise ValueError("Official evaluation requires an amd64 image")
    identity = {"swebench_version": version, "image": image, "image_id": info["Id"]}
    if expected is not None and identity != expected:
        raise ValueError(
            "Grading version or image differs from original batch; restore the recorded environment"
        )
    return identity


def frozen_grading_identity(snapshot, row):
    prepared = next(
        p for p in snapshot["evidence"]["prepared"] if p["instance_id"] == row["instance_id"]
    )
    return {
        "swebench_version": snapshot["evidence"]["swebench_version"],
        "image": prepared["image"],
        "image_id": prepared["image_id"],
    }


def evaluate(cfg, row, patch, target, expected=None):
    identity = grading_identity(cfg, row, expected)
    target.mkdir(parents=True, exist_ok=False)
    save_json(target / "grading-environment.json", identity)
    (target / "submitted.patch").write_text(patch, encoding="utf-8")
    save_json(target / "evaluation-input.json", {"row": row, "config": cfg})
    with (target / "harness.log").open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(
            [sys.executable, __file__, "--evaluate-child", str(target)],
            cwd=target,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=cfg["evaluation"]["timeout_seconds"] + 600)
            if proc.returncode:
                raise RuntimeError("Official evaluation failed; inspect harness.log")
        finally:
            stop_process(proc)
            record = target / "container.json"
            if record.exists():
                name = read_json(record)["name"]
                # run_instance normally removes this container; also clean it on timeout.
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
    return read_json(target / "validation.json")


def run_job(cfg, row, arm, batch):
    from report import session_metrics

    job = batch / "jobs" / f"{arm}--{row['instance_id']}"
    job.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name.split("-")[1]) for p in job.glob("attempt-*") if p.is_dir()]
    attempt = job / f"attempt-{max(numbers, default=0) + 1:04d}"
    attempt.mkdir()
    result = {"instance_id": row["instance_id"], "arm": arm, "status": "running", "resolved": None}
    save_json(attempt / "result.json", result)
    workspace = attempt / "workspace"
    try:
        expected = frozen_grading_identity(read_json(batch / "manifest.json"), row)
        cache = Path(cfg["work_dir"]) / "inputs" / row["instance_id"] / "repo"
        command(["git", "clone", "--no-hardlinks", cache, workspace])
        command(["git", "checkout", "--detach", row["base_commit"]], cwd=workspace)
        context = (
            sql_services(cfg, attempt, workspace) if arm == "sql" else contextlib.nullcontext(None)
        )
        with task_container(cfg, row, workspace, attempt, expected) as container, context as service:
            solve(cfg, row, arm, attempt, workspace, container, service)
        metrics = session_metrics(attempt / "session.jsonl")
        result.update(metrics)
        if not metrics["agent_completed"]:
            raise RuntimeError("Agent has no successful terminal result")
        if not metrics["isolation_valid"] or metrics["sql_connected"] != (arm == "sql"):
            raise RuntimeError("Agent tool/plugin initialization does not match the selected arm")
        if arm == "sql" and not metrics["sql_tool_succeeded"]:
            raise RuntimeError(
                "SQL has no successful oGMemory tool result; not a valid integration run"
            )
        patch = extract_patch(workspace, row["base_commit"])
        (attempt / "produced.patch").write_text(patch, encoding="utf-8")
        validation = evaluate(cfg, row, patch, attempt / "evaluation", expected=expected)
        result.update(status=validation["status"], resolved=validation["resolved"])
    except Exception as exc:
        result.update(status="error", error=str(exc))
    finally:
        if (workspace / ".git").exists() and not (attempt / "produced.patch").exists():
            try:
                patch = extract_patch(workspace, row["base_commit"])
                (attempt / "produced.patch").write_text(patch, encoding="utf-8")
            except Exception as exc:
                result["patch_error"] = str(exc)
        if (attempt / "session.jsonl").exists():
            result.update(session_metrics(attempt / "session.jsonl"))
        save_json(attempt / "result.json", result)
    print(
        f"{arm} {row['instance_id']}: {result['status']} resolved={result['resolved']}", flush=True
    )


def preflight(cfg, rows):
    check_sql_environment(cfg)
    if os.name != "posix" or sys.platform != "linux":
        raise RuntimeError("Execution requires Linux or WSL2 with a local Docker daemon")
    if importlib.metadata.version("swebench") != HARNESS_VERSION:
        raise RuntimeError(f"Install swebench=={HARNESS_VERSION}")
    model_env(cfg)
    command(["docker", "info"], timeout=30)
    help_text = command([cfg["model"]["claude"], "--help"], timeout=30)
    for flag in ["--setting-sources", "--strict-mcp-config", "--plugin-dir", "--tools"]:
        if flag not in help_text:
            raise RuntimeError(f"Claude Code lacks {flag}; install a supported version")
    prepared = []
    for row in rows:
        p = Path(cfg["work_dir"]) / "inputs" / row["instance_id"] / "prepared.json"
        meta = read_json(p)
        if meta["base_commit"] != row["base_commit"] or meta["image"] != image_name(
            cfg, row["instance_id"]
        ):
            raise ValueError("Prepared inputs do not match selected configuration")
        actual = json.loads(command(["docker", "image", "inspect", meta["image"]]))[0]
        if actual["Id"] != meta["image_id"]:
            raise ValueError("Image changed since prepare; use a new work_dir")
        prepared.append(meta)
    embedding_identity = None
    database_image_id = None
    if "sql" in cfg["arms"]:
        embedding = expand_environment(cfg["ogmemory"]["embedding"])
        if embedding.get("provider") in {None, "mock"}:
            raise ValueError("SQL requires a real embedding provider")
        embedding_identity = {k: embedding.get(k) for k in ("provider", "model", "base_url")}
        command(
            [sys.executable, "-c", "import gunicorn, psycopg2, mcp; import server.mcp_server"],
            cwd=ROOT,
        )
        # Pull the explicitly configured image only when it is not already present.
        try:
            command(["docker", "image", "inspect", cfg["ogmemory"]["opengauss_image"]])
        except RuntimeError:
            command(["docker", "pull", cfg["ogmemory"]["opengauss_image"]], timeout=600)
        database_image_id = json.loads(
            command(["docker", "image", "inspect", cfg["ogmemory"]["opengauss_image"]])
        )[0]["Id"]
    return {
        "prepared": prepared,
        "ogmemory_commit": command(["git", "rev-parse", "HEAD"], cwd=ROOT),
        **source_identity(),
        "python_version": sys.version,
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in (
                ["swebench", "PyYAML", "datasets"]
                + (["mcp", "psycopg2-binary"] if "sql" in cfg["arms"] else [])
            )
        },
        "embedding": embedding_identity,
        "database_image_id": database_image_id,
        "claude_version": command([cfg["model"]["claude"], "--version"]),
        "swebench_version": HARNESS_VERSION,
        "model_base_url": os.environ[cfg["model"]["base_url_env"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check", action="store_true", help="Preflight only; no model calls")
    parser.add_argument(
        "--resume", type=Path, help="Continue pending jobs; never repeats terminal jobs by default"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="On resume, create new attempts for error/empty_patch jobs",
    )
    parser.add_argument(
        "--evaluate",
        type=Path,
        help="Regrade one attempt in a new directory without calling the model",
    )
    parser.add_argument("--evaluate-child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.evaluate_child:
        evaluate_child(args.evaluate_child.resolve())
        return
    if args.evaluate:
        attempt = args.evaluate.resolve()
        batch = attempt.parents[2]
        snapshot = read_json(batch / "manifest.json")
        result = read_json(attempt / "result.json")
        row = next(
            r
            for r in read_json(batch / "dataset.json")
            if r["instance_id"] == result["instance_id"]
        )
        target = attempt / ("regrade-" + uuid.uuid4().hex[:12])
        expected = frozen_grading_identity(snapshot, row)
        print(
            evaluate(
                snapshot["config"],
                row,
                (attempt / "produced.patch").read_text(encoding="utf-8"),
                target,
                expected=expected,
            )
        )
        return
    if args.config is None:
        parser.error("--config is required")
    cfg = load_config(args.config)
    input_path = Path(cfg["work_dir"]) / "inputs" / "dataset.json"
    all_rows = read_json(input_path)
    source = read_json(input_path.with_name("dataset-source.json"))
    if source["source"] != {
        "dataset": cfg["dataset"],
        "revision": cfg["dataset_revision"],
        "split": "test",
    }:
        raise ValueError("Dataset source differs; use a new work_dir")
    if digest(all_rows) != source["sha256"]:
        raise ValueError("Dataset checksum mismatch")
    rows = selected_rows(cfg, all_rows)
    evidence = preflight(cfg, rows)
    print(f"Preflight passed: {len(rows)} instances x {len(cfg['arms'])} arms")
    if args.check:
        return
    if args.resume:
        batch = args.resume.resolve()
        previous = read_json(batch / "manifest.json")
        if (
            previous["config"] != cfg
            or previous["evidence"] != evidence
            or previous["dataset_hash"] != digest(rows)
        ):
            raise ValueError("Configuration, inputs or versions changed; start a new batch")
    else:
        batch = (
            Path(cfg["work_dir"])
            / "runs"
            / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        )
        batch.mkdir(parents=True)
        save_json(
            batch / "manifest.json",
            {"config": cfg, "evidence": evidence, "dataset_hash": digest(rows)},
        )
        save_json(batch / "dataset.json", rows)
    print(f"Batch: {batch}", flush=True)
    import fcntl

    with (batch / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = []
        for row in rows:
            for arm in cfg["arms"]:
                attempts = sorted(
                    (batch / "jobs" / f"{arm}--{row['instance_id']}").glob("attempt-*/result.json")
                )
                status = read_json(attempts[-1])["status"] if attempts else None
                if status in {"completed", "empty_patch", "error"} and not (
                    args.retry_errors and status != "completed"
                ):
                    continue
                if status == "running":
                    raise RuntimeError(
                        "Interrupted running job: inspect recorded resources, clean only its resources, then mark result status=error and resume --retry-errors"
                    )
                jobs.append((row, arm))
        with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
            futures = [pool.submit(run_job, cfg, row, arm, batch) for row, arm in jobs]
            for future in futures:
                future.result()
    print(
        f"Finished. Generate report: {sys.executable} {Path(__file__).with_name('report.py')} {batch}"
    )


if __name__ == "__main__":
    main()
