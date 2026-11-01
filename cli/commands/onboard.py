"""ogmem onboard command - unified setup wizard."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

from cli.lib.colors import CYAN, GREEN, NC, YELLOW, fail, info, ok, warn
from cli.lib.prompt import confirm, input_masked, input_validated, select

# Provider defaults
PROVIDER_DEFAULTS = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini", "embedding_model": "text-embedding-ada-002", "embedding_dim": 1536},
    "volcengine": {"base_url": "https://ark.cn-beijing.volces.com/api/v3", "model": "doubao-seed-2-0-code-preview-260215", "embedding_model": "doubao-embedding-vision-250615", "embedding_dim": 1024},
    "dashscope": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus", "embedding_model": "text-embedding-v3", "embedding_dim": 1024},
    "zhipu": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash", "embedding_model": "embedding-3", "embedding_dim": 2048},
    "mock": {"base_url": "", "model": "mock", "embedding_model": "mock", "embedding_dim": 1024},
}


def _print_header(step: int, title: str) -> None:
    print(f"\n{CYAN}{'='*60}{NC}\n{CYAN}Step {step}: {title}{NC}\n{CYAN}{'='*60}{NC}\n")


# ---------------------------------------------------------------------------
# Shared steps (all modes)
# ---------------------------------------------------------------------------

def _step1_check(args) -> bool:
    _print_header(2, "System Check")
    from cli.commands.check import run as check_run
    check_run(args)
    return confirm("Continue?", default=True)


def _step2_llm(args) -> dict:
    _print_header(3, "LLM Configuration")
    if args.non_interactive:
        if not args.provider:
            print(fail("--provider required in non-interactive mode"))
            sys.exit(1)
        provider, api_key = args.provider, args.api_key or ""
        base_url = args.base_url or PROVIDER_DEFAULTS.get(provider, {}).get("base_url", "")
        model = args.model or PROVIDER_DEFAULTS.get(provider, {}).get("model", "")
    else:
        providers = list(PROVIDER_DEFAULTS.keys())
        if getattr(args, "mode", None) == "docker":
            from cli.lib.deploy_helpers import QUICKSTART_EMBEDDING_PROVIDERS
            providers = [provider for provider in providers if provider in QUICKSTART_EMBEDDING_PROVIDERS]
        previous = getattr(args, "provider", None)
        if previous and previous not in providers:
            providers.append(previous)
        provider_idx = select("LLM provider:", providers, default=providers.index(previous) if previous in providers else 0)
        provider = providers[provider_idx]
        defaults = PROVIDER_DEFAULTS.get(provider, {})
        if provider == "mock":
            api_key = ""
            base_url = ""
            model = defaults["model"]
            print(info("Mock provider selected; API key and Base URL are not required"))
        else:
            keep = provider == previous
            api_key = input_masked("API key (press Enter to keep existing or skip)") or (getattr(args, "api_key", "") or "" if keep else "")
            base_url = input_validated("Base URL", (getattr(args, "base_url", None) if keep else None) or defaults.get("base_url", ""))
            model = input_validated("Model", (getattr(args, "model", None) if keep else None) or defaults.get("model", ""))
    return {"provider": provider, "api_key": api_key, "base_url": base_url, "model": model}


def _embedding_adapter(provider: str) -> str:
    adapter = {"zhipu": "openai", "dashscope": "openai"}.get(provider, provider)
    if adapter not in {"openai", "openai-cached", "volcengine", "st", "mock"}:
        raise ValueError("Unsupported embedding provider; use openai for an OpenAI-compatible API")
    return adapter


def _step3_embedding(args, llm_config: dict) -> dict:
    _print_header(4, "Embedding Configuration")
    if args.non_interactive:
        provider = args.embedding_provider or llm_config["provider"]
        model = args.embedding_model or PROVIDER_DEFAULTS.get(provider, {}).get("embedding_model", "")
        defaults = PROVIDER_DEFAULTS.get(provider, {})
        dim = getattr(args, "embedding_dimension", None) or defaults.get("embedding_dim", 1024)
        same_provider = provider == llm_config["provider"]
        base_url = getattr(args, "embedding_base_url", None) or (
            llm_config.get("base_url", "") if same_provider else defaults.get("base_url", "")
        )
        api_key = getattr(args, "embedding_api_key", None) or (
            llm_config.get("api_key", "") if same_provider else ""
        )
    else:
        same_default = all(getattr(args, "embedding_" + key, None) in (None, llm_config.get(key)) for key in ("provider", "base_url", "api_key"))
        same = confirm("Use same provider for embedding?", default=same_default)
        if same:
            provider = llm_config["provider"]
        elif getattr(args, "mode", None) == "docker":
            from cli.lib.deploy_helpers import QUICKSTART_EMBEDDING_PROVIDERS
            providers = list(QUICKSTART_EMBEDDING_PROVIDERS)
            provider = providers[select("Embedding provider:", providers)]
        else:
            provider = input_validated("Embedding provider", getattr(args, "embedding_provider", None) or llm_config["provider"])
        defaults = PROVIDER_DEFAULTS.get(provider, {})
        keep = provider == getattr(args, "embedding_provider", None)
        model = input_validated("Embedding model", (getattr(args, "embedding_model", None) if keep else None) or defaults.get("embedding_model", ""))
        dim = int(input_validated("Dimension", str((getattr(args, "embedding_dimension", None) if keep else None) or defaults.get("embedding_dim", 1024))) or 1024)
        base_url = (
            llm_config.get("base_url", "")
            if same
            else input_validated("Embedding base URL", (getattr(args, "embedding_base_url", None) if keep else None) or defaults.get("base_url", ""))
        )
        api_key = (
            llm_config.get("api_key", "")
            if same
            else (input_masked("Embedding API key (press Enter to keep existing or skip)") or ((getattr(args, "embedding_api_key", None) or "") if keep else ""))
        )
    _embedding_adapter(provider)
    if provider != llm_config["provider"] and provider not in {"mock", "st"} and not api_key:
        raise ValueError("Different embedding provider requires its own API key (or --embedding-api-key); the LLM key will not be reused")
    return {
        "provider": provider,
        "model": model,
        "dimension": dim,
        "base_url": base_url,
        "api_key": api_key,
    }


def _vector_db_choices(args) -> list[str]:
    choices = (
        ["opengauss"] if getattr(args, "mode", None) == "docker"
        else ["opengauss", "chroma", "milvus"]
    )
    requested = getattr(args, "vector_db", None) or "opengauss"
    if requested not in choices:
        raise ValueError(
            f"Docker quickstart supports only openGauss; requested {requested!r}. "
            "Use --mode local or --mode headless for Chroma/Milvus."
        )
    return choices


def _step4_database(args) -> tuple[dict, dict]:
    """Vector DB + SQL storage configuration and schema init."""
    _print_header(5, "Database Configuration")

    # Restrict choices to the selected deployment before prompting.
    vector_types = _vector_db_choices(args)
    if getattr(args, "non_interactive", False):
        vdb_type = getattr(args, "vector_db", None) or "opengauss"
    else:
        requested = getattr(args, "vector_db", None) or "opengauss"
        labels = {"opengauss": "openGauss", "chroma": "Chroma", "milvus": "Milvus"}
        vdb_type = vector_types[
            select("Vector database:", [labels[name] for name in vector_types],
                   default=vector_types.index(requested))
        ]

    vdb = {"type": vdb_type}
    storage_backend = getattr(args, "storage_backend", None) or "sql"

    database_required = storage_backend == "sql" or vdb_type == "opengauss"
    if getattr(args, "non_interactive", False):
        sql_dialect = getattr(args, "sql_dialect", None) or "opengauss"
    elif database_required:
        sql_dialect = ["opengauss", "postgresql"][
            select(
                "SQL storage database:",
                ["openGauss", "PostgreSQL"],
                default=1 if getattr(args, "sql_dialect", None) == "postgresql" else 0,
            )
        ]
    else:
        sql_dialect = getattr(args, "sql_dialect", None) or "opengauss"

    import getpass

    current_user = getpass.getuser()
    if sql_dialect == "opengauss":
        default_conn = "host=127.0.0.1 port=5432 dbname=ogmemory user=gaussdb"
    else:
        default_conn = (
            f"host=127.0.0.1 port=5432 dbname=ogmemory user={current_user}"
        )
    interactive = not getattr(args, "non_interactive", False)
    conn_str = getattr(args, "db_connection", None)
    vector_conn = getattr(args, "vector_db_connection", None)
    if storage_backend == "sql":
        if interactive:
            if conn_str:
                conn_str = input_masked("SQL connection string (Enter to keep existing)") or conn_str
            else:
                conn_str = input_validated("Database connection string", default_conn)
        conn_str = conn_str or default_conn
    if vdb_type == "opengauss":
        if interactive:
            if storage_backend == "sql" and confirm(
                "Use SQL storage connection for vector database?",
                default=not vector_conn or vector_conn == conn_str,
            ):
                vector_conn = conn_str
            elif vector_conn:
                vector_conn = input_masked("Vector database connection string (Enter to keep existing)") or vector_conn
            else:
                vector_conn = input_validated("Vector database connection string", conn_str or default_conn)
        else:
            vector_conn = vector_conn or conn_str or default_conn
        vdb["connection_string"] = vector_conn

    db = {
        "backend": storage_backend,
        "dialect": sql_dialect,
        "connection_string": conn_str if storage_backend == "sql" else "",
        "reachable": True,
        "ready": True,
    }

    # OpenGauss vector and SQL storage both require a reachable database.
    # Docker quickstart creates and checks its own OpenGauss container.
    if (storage_backend == "sql" or vdb_type == "opengauss") and getattr(args, "mode", None) != "docker":
        from cli.lib.db_setup import check_opengauss_compatibility, check_postgresql, init_schema

        # Probe each distinct database; never silently replace a separate vector DSN.
        if vdb_type == "opengauss" and (storage_backend != "sql" or vector_conn != conn_str):
            if not vector_conn or not check_postgresql(vector_conn):
                db["ready"] = False
                db["reachable"] = False
                print(fail("Vector database is not reachable; local setup cannot continue"))
                return vdb, db
        if storage_backend != "sql":
            return vdb, db
        if conn_str and check_postgresql(conn_str):
            print(ok(f"{sql_dialect} reachable"))
            if storage_backend == "sql" and sql_dialect == "opengauss":
                if not check_opengauss_compatibility(conn_str):
                    db["ready"] = False
                    print(fail("Cannot validate openGauss PG compatibility; local setup cannot continue"))
                    print(info("Use a database created with: CREATE DATABASE ogmemory DBCOMPATIBILITY 'PG';"))
                    return vdb, db
                print(ok("openGauss PG compatibility verified"))
            if storage_backend == "sql" and (
                not getattr(args, "non_interactive", False)
                and confirm("Initialize schema?", default=True)
            ):
                if init_schema(conn_str, sql_dialect):
                    print(ok("Schema initialized"))
                else:
                    db["ready"] = False
                    print(fail("Schema initialization failed; local setup cannot continue"))
                    print(info("Resolve the database error and rerun ogmem onboard"))
        else:
            db["reachable"] = False
            print(warn(f"{sql_dialect} not reachable — ensure it is running"))
            print(info("Local OpenGauss mode cannot start until the database is reachable"))

    return vdb, db


def _step5_target(args) -> str:
    """Select deployment target."""
    _print_header(1, "Setup Mode")
    if getattr(args, "mode", None):
        return args.mode
    if getattr(args, "non_interactive", False):
        return "local"
    targets = [
        "Local OpenGauss (single machine)",
        "Docker (containerized)",
        "Existing Agent connection (native plugin / MCP configuration)",
    ]
    idx = select("Choose deployment target:", targets, default=0)
    return ["local", "docker", "plugin"][idx]


# ---------------------------------------------------------------------------
# Shared sub-steps (local & headless)
# ---------------------------------------------------------------------------

def _step_install_deps(args, embedding: dict, vdb: dict) -> bool:
    """Install Python dependencies."""
    _print_header(6, "Install Dependencies")

    extras = []
    if embedding.get("provider") == "st":
        extras.append("sentence-transformers")
    if vdb.get("type") in {"chroma", "milvus"}:
        extras.append(vdb["type"])

    if shutil.which("uv"):
        cmd = ["uv", "sync"]
        for extra in extras:
            cmd.extend(["--extra", extra])
        print(info(f"Running {' '.join(cmd)}..."))
        result = subprocess.run(cmd, check=False)
    elif confirm("Run pip install -e . ?", default=True):
        requirement = "." if not extras else ".[" + ",".join(extras) + "]"
        pip_cmd = [sys.executable, "-m", "pip", "install", "-e", requirement]
        print(info(f"Running {' '.join(pip_cmd)}..."))
        result = subprocess.run(pip_cmd, check=False)
    else:
        print(warn("Skipping dependency installation"))
        if extras:
            print(info(f"Note: you'll need to manually install extras: {', '.join(extras)}"))
        return True
    if result.returncode != 0:
        print(fail(f"Dependency installation failed with exit code {result.returncode}"))
        return False
    return True


# ---------------------------------------------------------------------------
# Mode-specific steps
# ---------------------------------------------------------------------------

def _step_docker_setup(args, llm: dict, embedding: dict, vdb: dict) -> dict:
    """Docker mode: reuse deploy/user-quickstart/deploy.sh."""
    from cli.lib.process import find_project_root

    root = find_project_root()
    result = {}
    if vdb.get("type", "opengauss") != "opengauss":
        print(fail(
            f"Docker quickstart supports only openGauss; requested {vdb['type']!r}. "
            "Use --mode local or --mode headless for Chroma/Milvus."
        ))
        result["deploy_failed"] = True
        return result
    state_root = root / ".ogmem_data" / "onboard-docker"
    env_path = state_root / "ogmemory.env"
    runtime_env_path = state_root / "ogmemory-runtime.env"
    instance_name = getattr(args, "instance_name", None) or "ogmem"

    _print_header(6, "Generate deploy.env")
    from cli.lib.deploy_helpers import (
        deploy_quickstart,
        generate_quickstart_env,
        quickstart_shell_command,
    )

    try:
        generate_quickstart_env(
            llm,
            embedding,
            instance_name=instance_name,
            opengauss_port=args.opengauss_port,
            ogmem_http_port=args.port,
            ogmem_mcp_port=args.mcp_port,
            openclaw_ce_port=args.openclaw_ce_port,
            openclaw_mcp_port=args.openclaw_mcp_port,
            output_path=env_path,
        )
    except ValueError as exc:
        print(fail(str(exc)))
        result["deploy_failed"] = True
        return result
    print(ok(f"Written: {env_path}"))
    result["env_path"] = str(env_path)
    result["stack_root"] = str(state_root)
    result["quickstart"] = str(root / "deploy" / "user-quickstart" / "deploy.sh")

    _print_header(7, "Start Docker Containers")
    if confirm(
        "Start containers now?",
        default=bool(getattr(args, "non_interactive", False)),
    ):
        rc = deploy_quickstart(
            env_file=env_path,
            stack_root=state_root,
            runtime_env_file=runtime_env_path,
        )
        if rc != 0:
            print(warn(f"user-quickstart exited with code {rc}"))
            result["deploy_failed"] = True
        else:
            print(ok("Containers started"))
    else:
        command = quickstart_shell_command(
            env_file=env_path,
            stack_root=state_root,
            runtime_env_file=runtime_env_path,
        )
        print(info(f"Skipped. Run: {command}"))

    return result


# ---------------------------------------------------------------------------
# Write configuration
# ---------------------------------------------------------------------------

def _build_config(llm: dict, embedding: dict, vdb: dict, db: dict, *, port: int = 8090, plugin_type: str = "", openclaw_config: str = "") -> dict:
    """Build the settings owned by the wizard."""
    def quote(value: object) -> str:
        return json.dumps("" if value is None else str(value), ensure_ascii=False)

    embedding_provider = _embedding_adapter(embedding["provider"])
    storage_backend = db.get("backend", "sql")
    conn_str = db.get("connection_string", "")
    sql_dialect = db.get("dialect", "postgresql")

    cfg = f"""# oG-Memory Configuration
# Generated by ogmem onboard

paths:
  data_root: .ogmem_data

llm:
  provider: {quote(llm['provider'])}
  api_key: {quote(llm.get('api_key', ''))}
  base_url: {quote(llm.get('base_url', ''))}
  model: {quote(llm['model'])}
  temperature: 0.1

embedding:
  provider: {quote(embedding_provider)}
  model: {quote(embedding['model'])}
  api_key: {quote(embedding.get('api_key', ''))}
  base_url: {quote(embedding.get('base_url', ''))}

vector_db:
  type: {quote(vdb['type'])}
  dimension: {embedding['dimension']}
"""
    if vdb["type"] == "chroma":
        cfg += "  chroma_persist_dir: .ogmem_data/chroma\n"
    elif vdb.get("connection_string"):
        cfg += f"  connection_string: {quote(vdb['connection_string'])}\n"

    cfg += "\nstorage:\n"
    cfg += f"  backend: {quote(storage_backend)}\n"
    if storage_backend == "sql":
        cfg += (
            f"  dialect: {quote(sql_dialect)}\n"
            f"  connection_string: {quote(conn_str)}\n"
            "  pool_size: 5\n"
        )

    cfg += f"""
service:
  http_port: {port}
  workers: 2

identity:
  account_id: "acct-demo"
  user_id: "u-alice"
  agent_id: "main"
"""

    if plugin_type:
        cfg += f"""
plugin:
  type: "{plugin_type}"
"""
        if openclaw_config:
            cfg += f"  openclaw_config: {quote(openclaw_config)}\n"

    import yaml
    return yaml.safe_load(cfg)




# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(args) -> int:
    print(f"\n{GREEN}oG-Memory Setup Wizard{NC}\n{YELLOW}This wizard will guide you through the configuration.{NC}\n")

    # Select the mode before any backend checks, prompts, or dependency installs.
    deployment = _step5_target(args)
    args.mode = deployment
    if deployment == "plugin":
        from cli.commands.plugin_config import run as export_plugin_config
        return export_plugin_config(args)

    try:
        _vector_db_choices(args)
    except ValueError as exc:
        print(fail(str(exc)))
        return 1

    from cli.lib.onboard_config import existing_defaults, read_config
    from cli.lib.process import find_project_root
    root = find_project_root()
    from cli.lib.local_config import resolve_local_config_path
    config_path = resolve_local_config_path(root)
    try:
        if deployment in {"local", "headless"}:
            from cli.lib.local_runtime import require_stopped
            require_stopped(root)
        existing = read_config(config_path) if deployment in {"local", "headless"} else {}
        existing_defaults(args, existing)
    except (ValueError, RuntimeError, OSError) as exc:
        print(fail(str(exc)))
        return 1

    if existing and not args.non_interactive:
        action = select("Existing local configuration:", ["Use existing configuration", "Modify configuration", "Exit"], default=0)
        if action == 2:
            return 0
        if action == 0:
            if confirm("Start services now?", default=True):
                from cli.commands.start import start_headless
                return start_headless(type('Args', (), {
                    'mode': 'local', 'daemon': True, 'dry_run': False, 'config': str(config_path),
                })())
            return 0

    # System Check
    if not _step1_check(args):
        return 1

    # LLM
    llm_config = _step2_llm(args)

    # Embedding
    try:
        embedding_config = _step3_embedding(args, llm_config)
    except ValueError as exc:
        print(fail(str(exc)))
        return 1

    # Database (Vector DB + SQL storage)
    vdb_config, db_config = _step4_database(args)

    if deployment == "docker":
        docker_setup = _step_docker_setup(args, llm_config, embedding_config, vdb_config)
        if docker_setup.get("deploy_failed") and not docker_setup.get("env_path"):
            return 1
        print(ok("\nDocker configuration complete!"))
        print(f"\n{'-'*60}")
        print(f"  Deployment:  {deployment}")
        print(f"  LLM:         {llm_config['provider']} / {llm_config['model']}")
        print(f"  Files:       {docker_setup.get('env_path', 'deploy/deploy.env')}")
        print(f"  Quickstart:  {docker_setup.get('quickstart', 'deploy/user-quickstart/deploy.sh')}")
        print(f"{'-'*60}")
        if not docker_setup.get("deploy_failed"):
            print(info(f"Data directory: {docker_setup.get('stack_root', '.ogmem_data/onboard-docker')}"))
        else:
            return 1

    elif deployment in {"local", "headless"}:
        if not db_config.get("ready", True):
            return 1
        if db_config.get("backend") == "sql" and not db_config.get("reachable", False):
            print(fail("Database is not reachable; local OpenGauss mode cannot continue"))
            return 1
        import tempfile

        import yaml

        from cli.lib.local_config import load_local_config
        from cli.lib.local_runtime import require_stopped, runtime_lock
        from cli.lib.onboard_config import changed_fields, merge_config, save_config
        value = merge_config(existing, _build_config(llm_config, embedding_config, vdb_config, db_config, port=args.port))
        changes = changed_fields(existing, value)
        print(info("Changed fields (values hidden): " + (", ".join(changes) or "none")))
        try:
            # Validate privately before replacing the saved configuration.
            import os
            previous_env = dict(os.environ)
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as staged:
                    yaml.safe_dump(value, staged)
                    staged.flush()
                    candidate = load_local_config(root, staged.name)
                if not 1 <= candidate.http_port <= 65535:
                    raise ValueError("HTTP port must be between 1 and 65535")
            finally:
                os.environ.clear()
                os.environ.update(previous_env)
            if not _step_install_deps(args, embedding_config, vdb_config):
                return 1
            with runtime_lock(root):
                require_stopped(root)
                # Recheck after the prompts, before writing or replacing the config.
                if read_config(config_path) != existing:
                    raise ValueError("Configuration changed during setup; rerun the wizard")
                saved = save_config(config_path, value)
        except (ValueError, RuntimeError, OSError) as exc:
            print(fail(str(exc)))
            return 1
        print(ok(f"Configuration {'written' if saved else 'unchanged'}: {config_path}"))
        if confirm("Start services now?", default=True):
            from cli.commands.start import start_headless
            return start_headless(type('Args', (), {
                'mode': 'headless', 'daemon': True, 'dry_run': False,
                'config': str(config_path),
            })())
        print(info("Configuration saved; service configuration has not been applied. Run 'ogmem start local' to begin."))

    return 0
