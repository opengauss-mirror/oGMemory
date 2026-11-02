#!/usr/bin/env python3
"""Prepare private, disposable BEAM configuration files; never launch services.

Default: print a plan without reading credentials or creating directories.
--execute: create a NEW directory with memory/Agent configuration, Compose and
the evaluator setup. Docker builds/starts and paid evaluation remain explicit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# BEAM-only safety headroom, written solely to a NEW private fixture YAML.
# Never mutate OgMemConfig defaults or an existing service configuration.
BEAM_EXTRACTION_MAX_TOKENS = 32768


def validate(args):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,47}", args.run_id):
        raise ValueError("Use a unique 8-48 character lowercase run ID")
    output = args.directory
    if not output.is_absolute() or output.resolve() == ROOT or ROOT in output.resolve().parents or output.resolve() in ROOT.parents:
        raise ValueError("Use an absolute NEW fixture directory outside the checkout")
    if not args.parquet.is_absolute() or args.conversation_index < 0:
        raise ValueError("Use an absolute parquet path and a non-negative conversation index")
    ports = [args.sql_port, args.memory_port, args.agent_port]
    if len(set(ports)) != 3 or any(not 1024 <= p <= 65535 for p in ports):
        raise ValueError("Use three different unprivileged ports")
    if args.max_tokens <= 0 or args.embedding_dimension <= 0 or not math.isfinite(args.temperature):
        raise ValueError("Invalid sampling limit or embedding dimension")


def build_files(args, env):
    """Materialize explicit settings using the repository's native Agent template."""
    import yaml
    from psycopg2.extensions import make_dsn
    from scripts.beam_eval_safety import native_policy_manifest, validate_agent_config
    from providers.unified_config import OgMemConfig

    required = ["BEAM_LLM_API_KEY", "BEAM_LLM_BASE_URL", "BEAM_LLM_MODEL",
                "BEAM_EMBEDDING_API_KEY", "BEAM_EMBEDDING_BASE_URL", "BEAM_EMBEDDING_MODEL",
                "BEAM_QA_API_KEY", "BEAM_QA_BASE_URL", "BEAM_QA_MODEL",
                "BEAM_AGENT_TOKEN", "BEAM_MEMORY_API_KEY", "BEAM_POSTGRES_PASSWORD"]
    missing = [name for name in required if not env.get(name)]
    if missing:
        raise ValueError("Set credentials/settings in the environment: " + ", ".join(missing))
    output = args.directory.resolve()
    account, user = "beam-eval", f"beam-conversation-{args.conversation_index + 1}"
    database = "beam_eval_" + args.run_id.replace("-", "_")
    memory_url, agent_url = f"http://127.0.0.1:{args.memory_port}", f"http://127.0.0.1:{args.agent_port}"
    dsn = make_dsn(host="127.0.0.1", port=args.sql_port, dbname=database,
                   user="postgres", password=env["BEAM_POSTGRES_PASSWORD"])
    llm = dict(provider="openai", api_key=env["BEAM_LLM_API_KEY"],
               base_url=env["BEAM_LLM_BASE_URL"], model=env["BEAM_LLM_MODEL"],
               temperature=args.temperature, max_tokens=args.max_tokens)
    if args.thinking_disabled:
        llm["extra_body"] = {"thinking": {"type": "disabled"}}
    memory = dict(llm=llm, embedding=dict(provider="openai", api_key=env["BEAM_EMBEDDING_API_KEY"],
        base_url=env["BEAM_EMBEDDING_BASE_URL"], model=env["BEAM_EMBEDDING_MODEL"], multimodal=False),
        storage=dict(backend="sql", connection_string=dsn),
        vector_db=dict(type="opengauss", connection_string=dsn, dimension=args.embedding_dimension,
                       table_name="vector_index"),
        identity=dict(account_id=account, user_id=user, agent_id="main"),
        service=dict(http_port=args.memory_port, workers=1, threads=2,
                     http_ip_allowlist=["127.0.0.1", "::1"]),
        auth=dict(role_control_enabled=True, root_api_key=env["BEAM_MEMORY_API_KEY"]),
        cache=dict(enabled=False), mcp=dict(enabled=False))
    # Explicit toggles only. Do not bypass occurrence qualification, erase
    # routing keywords or install another extraction/retrieval implementation.
    graph_on = args.profile == "graph"
    memory["graph"] = dict(backend="sql" if graph_on else "none",
        enable_expansion=graph_on, supplement_enabled=graph_on)
    memory["scene"] = dict(enabled=False, generation_enabled=False, retrieval_enabled=False)
    replacements = dict(__LLM_PROVIDER__="beam-qa", __LLM_MODEL__=env["BEAM_QA_MODEL"],
        __LLM_BASE_URL__=env["BEAM_QA_BASE_URL"], __LLM_API_KEY__=env["BEAM_QA_API_KEY"],
        __GATEWAY_PORT__=args.agent_port, __OGMEM_URL__=memory_url,
        __OG_AUTH_API_KEY__=env["BEAM_MEMORY_API_KEY"], __OG_AUTH_ACCOUNT_ID__=account)

    def render(value):
        if isinstance(value, str):
            if value in replacements:
                return replacements[value]
            # The native primary model is "__LLM_PROVIDER__/__LLM_MODEL__".
            # Replace tokens in the ORIGINAL template once; never rescan keys.
            return re.sub(r"__[A-Z_]+__", lambda m: str(replacements.get(m[0], m[0])), value)
        if isinstance(value, list):
            return [render(item) for item in value]
        if isinstance(value, dict):
            return {render(key): render(item) for key, item in value.items()}
        return value

    agent = render(json.loads((ROOT / "config/openclaw/ogmem.template.json").read_text()))
    agent["agents"]["defaults"].update(workspace="/home/node/.openclaw/workspace", skipBootstrap=True)
    agent["tools"] = {"deny": ["*"]}
    agent["gateway"]["auth"] = dict(mode="token", token=env["BEAM_AGENT_TOKEN"])
    agent["plugins"]["entries"]["og-memory-context-engine"]["config"].update(
        accountId=account, userId=user, agentId="main", autoCapture=False, autoRecall=True)
    validate_agent_config(agent, account, user, memory_url)
    names = {role: f"{args.run_id}-{role}" for role in ("postgres", "memory", "agent")}
    setup = dict(parquet=str(args.parquet), output_root=str(output / "results"),
        conversation_index=args.conversation_index, run_id=args.run_id, database=database,
        postgres_container=names["postgres"], memory_container=names["memory"], agent_container=names["agent"],
        account=account, user=user, memory_url=memory_url, base_url=agent_url, variant="baseline",
        judge_model=env.get("BEAM_JUDGE_MODEL", "deepseek-chat"),
        judge_base_url=env.get("BEAM_JUDGE_BASE_URL", "https://api.deepseek.com"))
    setup["variant"] = setup["fixture_profile"] = args.profile
    setup["native_policy"] = native_policy_manifest(OgMemConfig(llm_temperature=args.temperature))
    setup["benchmark_settings"] = dict(
        algorithm="current checkout; reviewed PR314 graph; native Dev policies",
        routing="native keyword matching; unchanged",
        graph_enabled=graph_on, scene_enabled=False,
        extraction_model=env["BEAM_LLM_MODEL"],
        extraction_temperature=args.temperature,
        native_extraction_temperature_arguments=[None, 0],
        native_extraction_effective_temperatures=[args.temperature, 0],
        extraction_max_tokens=args.max_tokens,
        extraction_limit_scope="new BEAM fixture only",
        thinking_disabled=args.thinking_disabled,
        embedding_model=env["BEAM_EMBEDDING_MODEL"],
        embedding_dimension=args.embedding_dimension,
        qa_model=env["BEAM_QA_MODEL"], qa_temperature="native agent default; not forced",
        qa_max_tokens=8192, qa_context_window=131072,
        judge_model=setup["judge_model"], judge_temperature=0,
        cache_enabled=False, mcp_enabled=False,
        deployment="dedicated SQL fixture; MemoryService owns outbox and graph indexing")
    services = {}
    for role in names:
        services[role] = dict(container_name=names[role], network_mode="host",
            labels={"anttrail.eval.run": args.run_id, "anttrail.eval.role": role}, restart="no")
    # SnapshotDB executes native psql/pg_dump inside this container on 5432.
    # Only publish the operator-selected host port; do not move its socket port.
    services["postgres"].pop("network_mode")
    services["postgres"].update(image=env.get("BEAM_POSTGRES_IMAGE", "pgvector/pgvector:pg16"),
        environment=dict(POSTGRES_USER="postgres", POSTGRES_DB=database,
                         POSTGRES_PASSWORD=env["BEAM_POSTGRES_PASSWORD"]),
        ports=[f"127.0.0.1:{args.sql_port}:5432"],
        volumes=[str(output / "postgres-data") + ":/var/lib/postgresql/data"],
        healthcheck=dict(test=["CMD-SHELL", f"pg_isready -U postgres -d {database}"],
                         interval="2s", timeout="3s", retries=30))
    services["memory"].update(image=f"beam-dev-memory:{args.run_id}",
        build=dict(context=str(ROOT), dockerfile="docker/Dockerfile.standalone", target="standalone-no-agfs"),
        depends_on={"postgres": {"condition": "service_healthy"}},
        environment=dict(OGMEM_CONFIG="/etc/ogmem/beam.yaml", OGMEM_HTTP_PORT=str(args.memory_port),
                         OGMEM_MCP_ENABLED="false", PYTHONPATH="/opt/beam-capture:/opt/ogmem",
                         BEAM_COMPOSE_CAPTURE="1"),
        volumes=[str(output / "memory.private.yaml") + ":/etc/ogmem/beam.yaml:ro",
                 str(ROOT / "scripts/beam_capture") + ":/opt/beam-capture:ro"])
    # Both arms use the same deployment. The API owns native outbox indexing;
    # a separate standalone index process cannot bind its graph projection worker.
    services["memory"]["entrypoint"] = ["/bin/sh", "/opt/ogmem/scripts/beam_sql_entrypoint.sh"]
    services["agent"].update(image=f"beam-dev-agent:{args.run_id}",
        build=dict(context=str(ROOT), dockerfile="docker/Dockerfile.openclaw"),
        environment=dict(OPENCLAW_GATEWAY_TOKEN=env["BEAM_AGENT_TOKEN"],
                         GATEWAY_PORT=str(args.agent_port)),
        volumes=[str(output / "agent-state") + ":/home/node/.openclaw"])
    return {"memory.private.yaml": yaml.safe_dump(memory, sort_keys=False),
            "agent-state/openclaw.json": json.dumps(agent, ensure_ascii=False, indent=2) + "\n",
            "setup.json": json.dumps(setup, indent=2) + "\n",
            "compose.private.yaml": yaml.safe_dump(dict(name=args.run_id, services=services), sort_keys=False)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--conversation-index", type=int, default=0)
    parser.add_argument("--sql-port", type=int, default=25432)
    parser.add_argument("--memory-port", type=int, default=18800)
    parser.add_argument("--agent-port", type=int, default=20800)
    parser.add_argument("--embedding-dimension", type=int, required=True)
    parser.add_argument("--profile", choices=("baseline", "graph"), default="baseline",
                        help="Native baseline or reviewed PR314 graph; neither rewrites routing or qualification")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=BEAM_EXTRACTION_MAX_TOKENS,
                        help="Extraction output cap in this BEAM fixture only (default: 32768; Dev remains 4096)")
    parser.add_argument("--thinking-disabled", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Write new private configuration files only")
    args = parser.parse_args(argv)
    validate(args)
    report = dict(status="planned", directory=str(args.directory), docker_calls=0, provider_calls=0,
                  extraction_temperature=args.temperature, extraction_max_tokens=args.max_tokens,
                  extraction_limit_scope="new BEAM fixture only; production defaults unchanged",
                  fixture_profile=args.profile, graph_enabled=args.profile == "graph",
                  native_extraction_temperature_arguments=[None, 0],
                  native_extraction_effective_temperatures=[args.temperature, 0],
                  note="Review configurations; current native policy, not historical-score equivalence. No services started.")
    if args.execute:
        if args.directory.exists():
            raise ValueError("Refusing to overwrite an existing fixture directory")
        files = build_files(args, os.environ)  # Validate everything before creating output.
        os.umask(0o077)
        args.directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        for name, content in files.items():
            path = args.directory / name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)
        for name in ("agent-state/workspace", "agent-state/agents/main/sessions"):
            (args.directory / name).mkdir(parents=True, exist_ok=True, mode=0o700)
        report.update(status="prepared_files_only", files=sorted(files))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
