"""Offline checks of benchmark evidence accounting and reproducible patch creation."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "benchmarks" / "swebench"


def load(name, filename=None):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (filename or f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


prepare = load("prepare")
report = load("report")
runner = load("swebench_reference_run", "run.py")


def session(tmp_path, events):
    p = tmp_path / "session.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return p


def terminal(**usage):
    return {"type": "result", "subtype": "success", "is_error": False, "usage": usage}


def test_usage_counts_cache_once_and_deduplicates_calls(tmp_path):
    tool = {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "id": "a", "name": "Bash"}]},
    }
    p = session(
        tmp_path,
        [
            {"type": "system", "subtype": "init", "plugins": [], "mcp_servers": []},
            tool,
            tool,
            terminal(
                input_tokens=10,
                cache_creation_input_tokens=20,
                cache_read_input_tokens=30,
                output_tokens=4,
            ),
        ],
    )
    m = report.session_metrics(p)
    assert m["usage"]["total_tokens"] == 64
    assert m["actions"]["total"] == 1
    assert m["isolation_valid"] and not m["sql_connected"]


def test_truncated_session_does_not_report_zero(tmp_path):
    m = report.session_metrics(session(tmp_path, []))
    assert m["usage"]["total_tokens"] is None
    assert m["actions"]["total"] is None
    assert not m["agent_completed"]


def test_missing_cache_field_is_partial(tmp_path):
    m = report.session_metrics(session(tmp_path, [terminal(input_tokens=1, output_tokens=2)]))
    assert m["usage"]["evidence_status"] == "PARTIAL"


def test_unrelated_plugin_invalidates_isolation(tmp_path):
    m = report.session_metrics(
        session(
            tmp_path,
            [
                {
                    "type": "system",
                    "subtype": "init",
                    "plugins": [{"name": "other"}],
                    "mcp_servers": [],
                }
            ],
        )
    )
    assert not m["isolation_valid"]


def test_final_attempt_and_official_verdict_are_required(tmp_path):
    prepare.save_json(tmp_path / "manifest.json", {"config": {"arms": ["baseline"]}})
    prepare.save_json(tmp_path / "dataset.json", [{"instance_id": "x"}])
    job = tmp_path / "jobs/baseline--x"
    prepare.save_json(job / "attempt-0001/result.json", {"status": "completed", "resolved": True})
    prepare.save_json(
        job / "attempt-0001/evaluation/validation.json", {"status": "completed", "resolved": True}
    )
    prepare.save_json(job / "attempt-0002/result.json", {"status": "error", "resolved": True})
    rows = report.final_rows(tmp_path)
    assert rows[0]["attempt"] == "attempt-0002"
    assert rows[0]["resolved"] is None
    assert report.summarize(rows)[0]["no_verdict"] == 1


def test_paired_comparison_excludes_different_success_sets():
    rows = [
        {"instance_id": i, "arm": a, "resolved": good, "tokens": n, "actions": n}
        for i, a, good, n in [
            ("x", "baseline", True, 100),
            ("x", "sql", True, 75),
            ("y", "baseline", True, 1000),
            ("y", "sql", False, 5),
        ]
    ]
    result = report.paired_comparison(rows)
    assert result["count"] == 1
    assert result["tokens"]["reduction_percent"] == 25


def test_patch_includes_new_files_and_agent_commits(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return prepare.command(["git", *args], cwd=repo)

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    (repo / "a.py").write_text("old = 1\n\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    (repo / "a.py").write_text("new = 2\n\n", encoding="utf-8")
    git("commit", "-am", "agent commit")
    (repo / "new.py").write_text("added = 1\n", encoding="utf-8")
    patch = runner.extract_patch(repo, base)
    patch_path = tmp_path / "produced.patch"
    patch_path.write_text(patch, encoding="utf-8")
    clean = tmp_path / "clean"
    prepare.command(["git", "clone", repo, clean])
    prepare.command(["git", "checkout", "--detach", base], cwd=clean)
    subprocess.run(["git", "apply", str(patch_path)], cwd=clean, check=True)
    assert (clean / "new.py").read_text() == "added = 1\n"
    assert (clean / "a.py").read_text() == "new = 2\n\n"


def test_sql_config_matches_current_product_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("SWE_EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv("SWE_EMBEDDING_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("SWE_EMBEDDING_API_KEY", "test-only")
    cfg = prepare.load_config(SCRIPTS / "config.example.yaml")
    service = runner.service_config(cfg, tmp_path, 5432, "test-only", "test-only", 8090, 8091)
    assert service["storage"]["backend"] == "sql"
    assert service["storage"]["dialect"] == "opengauss"
    assert "user=swebench" in service["storage"]["connection_string"]
    assert service["features"]["code_workspace_allowed_roots"] == [str(tmp_path)]
    assert service["features"]["code_retrieval"]["mode"] == "candidate_local"
    assert service["mcp"]["default_toolset"] == "code"
    # Validate against the actual latest-dev parser, not a duplicated schema.
    import yaml

    from providers.unified_config import OgMemConfig

    path = tmp_path / "ogmemory.yaml"
    path.write_text(yaml.safe_dump(service), encoding="utf-8")
    product = OgMemConfig.load(str(path))
    assert product.storage_backend == "sql"
    assert product.admin_api_keys == ["swebench:test-only"]
    assert product.code_retrieval_mode == "candidate_local"
    assert product.mcp_enable_code_edit_tools is True


def test_unknown_instance_is_rejected():
    with pytest.raises(ValueError, match="Unknown instances"):
        prepare.selected_rows({"instances": ["missing"]}, [{"instance_id": "x"}])


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("SWE_SQL_INTEGRATION") != "1", reason="Opt-in Docker SQL compatibility check"
)
def test_sql_account_registration_and_mcp_tools(tmp_path, monkeypatch):
    """Exercise current product APIs against a real, disposable openGauss database."""
    import asyncio
    import contextlib
    import uuid

    import yaml

    key = uuid.uuid4().hex
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "example.py").write_text("def add(a, b):\n    return a + b\n")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    cfg = prepare.load_config(SCRIPTS / "config.example.yaml")
    cfg["ogmemory"]["embedding"] = {"provider": "mock"}
    application = None
    stack = contextlib.ExitStack()
    try:
        port, password = stack.enter_context(runner.sql_database(cfg, tmp_path, private))
        generated = runner.service_config(cfg, workspace, port, password, key, 8090, 8091)
        path = tmp_path / "ogmemory.yaml"
        path.write_text(yaml.safe_dump(generated), encoding="utf-8")
        monkeypatch.setenv("OGMEM_CONFIG", str(path))
        from providers.unified_config import reset_config

        reset_config()
        from mcp import Client

        import server.app as application
        from server.mcp_server import create_mcp_server

        application._shutdown_service()
        monkeypatch.setattr(application, "_service", None)
        client = application.app.test_client()
        headers = {
            "X-API-Key": key,
            "X-Account-ID": "swebench",
            "X-User-ID": "tester",
            "X-Agent-ID": "solver",
        }
        for url, body in [
            ("/api/v1/admin/accounts/swebench/users", {"user_id": "tester"}),
            (
                "/api/v1/admin/accounts/swebench/agents",
                {"agent_id": "solver", "owner_user_id": "tester"},
            ),
            (
                "/api/v1/resources/code/sync/registrations",
                {
                    "path": str(workspace),
                    "name": "task",
                    "trigger_mode": "manual",
                    "watch_enabled": False,
                    "wait_mode": "accepted",
                },
            ),
        ]:
            response = client.post(url, json=body, headers=headers)
            assert response.status_code == 200, response.get_json()
        assert client.get("/api/v1/health").get_json()["sql"] == "connected"
        server = create_mcp_server(service=application._get_service())

        async def tools():
            async with Client(server) as connection:
                return {t.name for t in (await connection.list_tools()).tools}

        assert asyncio.run(tools()) == {"ogm_search_code", "ogm_read_memory", "ogm_edit_file"}
    finally:
        if application is not None:
            application._shutdown_service()
            from providers.unified_config import reset_config

            reset_config()
        stack.close()


def test_setup_runs_official_install_after_mount(tmp_path, monkeypatch):
    import types

    constants = types.ModuleType("swebench.harness.constants")
    constants.MAP_REPO_TO_INSTALL = {}
    constants.MAP_REPO_VERSION_TO_SPECS = {
        "example/project": {
            "1": {"pre_install": ["echo prepare"], "install": "python -m pip install -e ."}
        }
    }
    monkeypatch.setitem(sys.modules, "swebench.harness.constants", constants)
    calls = []
    monkeypatch.setattr(runner, "command", lambda args, **kw: calls.append(args) or "")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda args, **kw: calls.append((args, kw)) or types.SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(runner.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(runner.os, "getgid", lambda: 1000, raising=False)
    workspace = tmp_path / "workspace"
    (workspace / ".git/info").mkdir(parents=True)
    row = {
        "repo": "example/project",
        "version": "1",
        "instance_id": "example-1",
        "base_commit": "base",
    }
    cfg = {
        "evaluation": {"namespace": "swebench", "image_tag": "latest", "setup_timeout_seconds": 123}
    }
    expected = {"image_id": "sha256:frozen"}
    monkeypatch.setattr(runner, "grading_identity", lambda *a: expected)
    with runner.task_container(cfg, row, workspace, tmp_path, expected):
        assert (tmp_path / "setup.json").exists()
    install_index = next(i for i, c in enumerate(calls) if isinstance(c, tuple) and "input" in c[1])
    mount_index = next(i for i, c in enumerate(calls) if isinstance(c, list) and "--mount" in c)
    assert mount_index < install_index
    assert calls[mount_index][-2] == "sha256:frozen"
    script = calls[install_index][1]["input"]
    assert (
        script.index("conda activate testbed")
        < script.index("echo prepare")
        < script.index("python -m pip install -e .")
    )
    assert calls[install_index][1]["timeout"] == 123


@pytest.mark.parametrize("failure", ["timeout", "invalid_session"])
def test_failed_solver_keeps_patch_without_grading(tmp_path, monkeypatch, failure):
    import contextlib

    cache = tmp_path / "inputs/example/repo"
    cache.mkdir(parents=True)

    def git(*args):
        return prepare.command(["git", *args], cwd=cache)

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    (cache / "a.py").write_text("old = 1\n")
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")

    def solve(cfg, row, arm, attempt, workspace, container, service):
        (workspace / "a.py").write_text("fixed = 2\n")
        (workspace / "new.py").write_text("added = 3\n")
        if failure == "timeout":
            raise TimeoutError("test timeout")
        (attempt / "session.jsonl").write_text("")

    monkeypatch.setattr(runner, "solve", solve)
    monkeypatch.setattr(runner, "task_container", lambda *a: contextlib.nullcontext("test"))
    monkeypatch.setattr(
        runner, "evaluate", lambda *a, **k: pytest.fail("Invalid solve must not be graded")
    )
    batch = tmp_path / "batch"
    prepare.save_json(
        batch / "manifest.json",
        {
            "evidence": {
                "swebench_version": "4.1.0",
                "prepared": [
                    {"instance_id": "example", "image": "original", "image_id": "sha256:original"}
                ],
            }
        },
    )
    runner.run_job(
        {"work_dir": str(tmp_path)},
        {"instance_id": "example", "base_commit": base},
        "baseline",
        batch,
    )
    attempt = batch / "jobs/baseline--example/attempt-0001"
    assert prepare.read_json(attempt / "result.json")["status"] == "error"
    patch = (attempt / "produced.patch").read_text()
    assert "+fixed = 2" in patch and "+added = 3" in patch


@pytest.mark.parametrize(
    "version,image_id,arch,accepted",
    [
        ("4.1.0", "sha256:original", "amd64", True),
        ("4.2.0", "sha256:original", "amd64", False),
        ("4.1.0", "sha256:changed", "amd64", False),
        ("4.1.0", "sha256:original", "arm64", False),
    ],
)
def test_grading_checks_version_and_original_image(monkeypatch, version, image_id, arch, accepted):
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: version)
    monkeypatch.setattr(
        runner, "command", lambda *a, **k: json.dumps([{"Id": image_id, "Architecture": arch}])
    )
    cfg = {"evaluation": {"namespace": "swebench", "image_tag": "latest"}}
    row = {"instance_id": "example"}
    expected = {
        "swebench_version": "4.1.0",
        "image": prepare.image_name(cfg, "example"),
        "image_id": "sha256:original",
    }
    if accepted:
        assert runner.grading_identity(cfg, row, expected) == expected
    else:
        with pytest.raises(ValueError):
            runner.grading_identity(cfg, row, expected)


def test_setup_preserves_build_files_but_excludes_them_from_patch(tmp_path, monkeypatch):
    import types

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def git(*args):
        return prepare.command(["git", *args], cwd=workspace)

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    (workspace / "setup.py").write_text("original = 1\n")
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    real_run = subprocess.run

    def install(args, **kwargs):
        if args[0] != "docker":
            return real_run(args, **kwargs)
        if "-i" in args:
            (workspace / "setup.py").write_text("installation_adjustment = 2\n")
            (workspace / "extension.so").write_bytes(b"compiled placeholder")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", install)
    monkeypatch.setattr(runner, "task_setup_script", lambda row: "install")
    monkeypatch.setattr(runner.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(runner.os, "getgid", lambda: 1000, raising=False)
    runner.prepare_task_environment(
        {"evaluation": {}}, {"base_commit": base}, "test", workspace, tmp_path
    )
    assert (workspace / "setup.py").read_text() == "original = 1\n"
    assert (workspace / "extension.so").exists()
    assert runner.extract_patch(workspace, base) == ""
    (workspace / "new_fix.py").write_text("fix = 1\n")
    patch = runner.extract_patch(workspace, base)
    assert "new_fix.py" in patch and "extension.so" not in patch


def test_regrade_cli_passes_original_identity_without_model_credentials(tmp_path, monkeypatch):
    batch = tmp_path / "batch"
    attempt = batch / "jobs/baseline--example/attempt-0001"
    attempt.mkdir(parents=True)
    prepare.save_json(
        batch / "manifest.json",
        {
            "config": {"evaluation": {"namespace": "swebench", "image_tag": "latest"}},
            "evidence": {
                "swebench_version": "4.1.0",
                "prepared": [
                    {"instance_id": "example", "image": "original", "image_id": "sha256:original"}
                ],
            },
        },
    )
    prepare.save_json(batch / "dataset.json", [{"instance_id": "example"}])
    prepare.save_json(attempt / "result.json", {"instance_id": "example", "status": "error"})
    (attempt / "produced.patch").write_text("saved patch")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("SWE_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["run.py", "--evaluate", str(attempt)])

    def evaluate(cfg, row, patch, target, expected):
        assert patch == "saved patch"
        assert expected == {
            "swebench_version": "4.1.0",
            "image": "original",
            "image_id": "sha256:original",
        }
        assert target.parent == attempt and target.name.startswith("regrade-")
        return {"status": "completed"}

    monkeypatch.setattr(runner, "evaluate", evaluate)
    runner.main()
    assert prepare.read_json(attempt / "result.json")["status"] == "error"


def test_single_arm_report_does_not_render_missing_comparison_as_percentage(tmp_path, monkeypatch):
    prepare.save_json(tmp_path / "manifest.json", {"config": {"arms": ["baseline"]}})
    prepare.save_json(tmp_path / "dataset.json", [{"instance_id": "example"}])
    monkeypatch.setattr(sys, "argv", ["report.py", str(tmp_path)])
    report.main()
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "不计算 baseline 与 SQL 的效率差异" in text
    assert "None%" not in text and "baseline=None" not in text
    data = prepare.read_json(tmp_path / "summary.json")
    assert data["summary"][0]["planned"] == 1
    assert data["summary"][0]["valid"] == 0
    assert data["paired"]["tokens"]["reduction_percent"] is None


@pytest.mark.parametrize("entry", ["preflight", "service"])
def test_sql_rejects_repository_dotenv_before_side_effects(tmp_path, monkeypatch, entry):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    dotenv = config_dir / ".env"
    content = "OGMEM_CODE_RETRIEVAL_RESULT_LIMIT=1\n"
    dotenv.write_text(content)
    monkeypatch.setattr(runner, "command", lambda *a, **k: pytest.fail("Must reject before launch"))
    cfg = {"arms": ["baseline", "sql"]}
    with pytest.raises(ValueError, match="checkout without config/.env"):
        if entry == "preflight":
            runner.preflight(cfg, [])
        else:
            with runner.sql_services(cfg, tmp_path / "attempt", tmp_path / "workspace"):
                pytest.fail("SQL must not start")
    assert dotenv.read_text() == content
    assert not (tmp_path / "attempt").exists()
    runner.check_sql_environment({"arms": ["baseline"]})
    dotenv.unlink()
    runner.check_sql_environment(cfg)


def test_example_model_requires_explicit_selection(monkeypatch):
    cfg = prepare.load_config(SCRIPTS / "config.example.yaml")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-only")
    with pytest.raises(ValueError, match="Set model.name"):
        runner.model_env(cfg)
    cfg["model"]["name"] = "test-model"
    assert runner.model_env(cfg)["ANTHROPIC_BASE_URL"] == "https://example.invalid"


@pytest.mark.parametrize("cli_file", ["cli/main.py", "cli/commands/mcp.py"])
def test_cli_edit_changes_identity_and_prevents_resume(tmp_path, monkeypatch, cli_file):
    root = tmp_path / "product"
    root.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    for name in [
        cli_file,
        *[
            f"benchmarks/swebench/{n}"
            for n in ("prepare.py", "run.py", "report.py", "requirements.txt")
        ],
    ]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original\n")

    def git(*args):
        return prepare.command(["git", *args], cwd=root)

    git("init")
    git("config", "user.name", "test")
    git("config", "user.email", "test@example.invalid")
    git("add", ".")
    git("commit", "-m", "base")
    original_head = git("rev-parse", "HEAD")
    before = runner.source_identity()
    assert before["ogmemory_dirty"] is False
    (root / cli_file).write_text("# changed MCP startup\n")
    after = runner.source_identity()
    assert after["source_hash"] != before["source_hash"]
    assert after["ogmemory_dirty"] is True
    assert git("rev-parse", "HEAD") == original_head

    cfg = prepare.load_config(SCRIPTS / "config.example.yaml")
    cfg["work_dir"] = str(tmp_path / "work")
    rows = [{"instance_id": cfg["instances"][0]}]
    inputs = Path(cfg["work_dir"]) / "inputs"
    prepare.save_json(inputs / "dataset.json", rows)
    prepare.save_json(
        inputs / "dataset-source.json",
        {
            "source": {
                "dataset": cfg["dataset"],
                "revision": cfg["dataset_revision"],
                "split": "test",
            },
            "sha256": prepare.digest(rows),
        },
    )
    batch = tmp_path / "batch"
    prepare.save_json(
        batch / "manifest.json",
        {
            "config": cfg,
            "evidence": before,
            "dataset_hash": prepare.digest(rows),
        },
    )
    monkeypatch.setattr(runner, "load_config", lambda path: cfg)
    monkeypatch.setattr(runner, "preflight", lambda *a: runner.source_identity())
    monkeypatch.setattr(runner, "run_job", lambda *a: pytest.fail("Resume must not schedule jobs"))
    monkeypatch.setattr(sys, "argv", ["run.py", "--config", "unused", "--resume", str(batch)])
    with pytest.raises(ValueError, match="versions changed"):
        runner.main()


@pytest.mark.parametrize(
    "version,compatibility,valid",
    [
        ("openGauss 7.0.0", "PG", True),
        ("PostgreSQL 16", "PG", False),
        ("openGauss 7.0.0", "A", False),
    ],
)
def test_database_checks_engine_mode_and_cleans_up(
    tmp_path, monkeypatch, version, compatibility, valid
):
    from unittest.mock import MagicMock

    import psycopg2

    cfg = prepare.load_config(SCRIPTS / "config.example.yaml")
    private = tmp_path / "private"
    private.mkdir()
    calls = []
    states = iter(["bash", "gaussdb"])

    def command(args, **kwargs):
        calls.append(args)
        if args[1] == "inspect":
            return json.dumps(
                [
                    {
                        "Image": "sha256:test",
                        "State": {"Running": True},
                        "NetworkSettings": {"Ports": {"5432/tcp": [{"HostPort": "15432"}]}},
                    }
                ]
            )
        if args[-1] == "/proc/1/comm":
            return next(states)
        return ""

    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = [(version,), (compatibility,)]
    monkeypatch.setattr(psycopg2, "connect", lambda **kw: connection)
    monkeypatch.setattr(runner, "command", command)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    cleanup = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr(runner.subprocess, "run", cleanup)

    def exercise():
        with runner.sql_database(cfg, tmp_path, private) as (port, password):
            assert port == "15432"
            assert 8 <= len(password) <= 32
            assert all(
                any(c in alphabet for c in password)
                for alphabet in (
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    "abcdefghijklmnopqrstuvwxyz",
                    "0123456789",
                    "!@#$",
                )
            )
            evidence = prepare.read_json(tmp_path / "database.json")
            assert evidence["dialect"] == "opengauss"
            assert evidence["sql_compatibility"] == "PG"

    if valid:
        exercise()
    else:
        with pytest.raises(RuntimeError, match="requires openGauss"):
            exercise()
        assert not (tmp_path / "database.json").exists()
    assert any("DBCOMPATIBILITY 'PG'" in str(arg) for call in calls for arg in call)
    assert "GS_USERNAME=swebench" in (private / "opengauss.env").read_text()
    cleanup.assert_called_once()
    assert prepare.read_json(tmp_path / "cleanup.json")["removed"] is True


def test_current_claude_plugin_path():
    plugin = runner.ROOT / "plugins" / "claude-code"
    assert (plugin / ".claude-plugin" / "plugin.json").is_file()
    assert "plugins/claude-code" in runner.PRODUCT_PATHS


@pytest.mark.parametrize("change_at", ["before_solve", "before_grading"])
def test_normal_job_rejects_changed_image(tmp_path, monkeypatch, change_at):
    from types import SimpleNamespace

    cfg = {
        "work_dir": str(tmp_path),
        "evaluation": {"namespace": "swebench", "image_tag": "latest"},
    }
    row = {"instance_id": "example", "base_commit": "base"}
    batch = tmp_path / "batch"
    prepare.save_json(
        batch / "manifest.json",
        {
            "evidence": {
                "swebench_version": "4.1.0",
                "prepared": [
                    {
                        "instance_id": "example",
                        "image": prepare.image_name(cfg, "example"),
                        "image_id": "sha256:original",
                    }
                ],
            }
        },
    )
    current_image = "sha256:changed" if change_at == "before_solve" else "sha256:original"
    launches, solves = [], []

    def command(args, **kwargs):
        if args[:3] == ["docker", "image", "inspect"]:
            return json.dumps([{"Id": current_image, "Architecture": "amd64"}])
        if args[:2] == ["docker", "run"]:
            launches.append(args)
        return ""

    def solve(*args):
        nonlocal current_image
        solves.append(True)
        current_image = "sha256:changed"

    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "4.1.0")
    monkeypatch.setattr(runner, "command", command)
    monkeypatch.setattr(runner, "prepare_task_environment", lambda *a: None)
    monkeypatch.setattr(runner, "solve", solve)
    monkeypatch.setattr(runner, "extract_patch", lambda *a: "saved patch\n")
    monkeypatch.setattr(
        report,
        "session_metrics",
        lambda *a: {
            "agent_completed": True,
            "isolation_valid": True,
            "sql_connected": False,
        },
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("Changed image must not reach the grading process"),
    )

    runner.run_job(cfg, row, "baseline", batch)

    attempt = batch / "jobs/baseline--example/attempt-0001"
    result = prepare.read_json(attempt / "result.json")
    assert result["status"] == "error"
    assert "differs from original batch" in result["error"]
    assert not (attempt / "evaluation").exists()
    if change_at == "before_solve":
        assert not launches and not solves
    else:
        assert len(launches) == len(solves) == 1
        assert launches[0][-2] == "sha256:original"
        assert (attempt / "produced.patch").read_text() == "saved patch\n"
