"""Unified CLI for oG-Memory — main entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from cli import __version__


class ExplicitStore(argparse.Action):
    """Keep public parser defaults while distinguishing explicitly supplied values."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        fields = set(getattr(namespace, "_explicit_fields", ()))
        fields.add(self.dest)
        namespace._explicit_fields = fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogmem",
        description="oG-Memory — unified management CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # -- onboard --
    p_onboard = sub.add_parser("onboard", help="Interactive setup wizard")
    p_onboard.add_argument("--non-interactive", action="store_true", help="Run without prompts")
    p_onboard.add_argument(
        "--mode",
        choices=["local", "docker", "plugin", "headless"],
        help="Deployment mode (local is the single-machine OpenGauss mode)",
    )
    p_onboard.add_argument(
        "--provider", help="LLM provider (openai/volcengine/dashscope/zhipu/mock)"
    )
    p_onboard.add_argument("--api-key", help="LLM API key")
    p_onboard.add_argument("--base-url", help="LLM API base URL")
    p_onboard.add_argument("--model", help="LLM model name")
    plugin = p_onboard.add_argument_group("Existing Agent connection (plugin mode)")
    plugin.add_argument(
        "--agent",
        help="openclaw, claude-code, codex, opencode, deepseek-harness, hermes, deepagents, agent-plugins; other names use MCP",
    )
    plugin.add_argument("--ogmem-url", help="oGMemory HTTP API base URL as seen by the Agent")
    plugin.add_argument("--mcp-url", help="Full Streamable HTTP MCP URL, including its port and path")
    credentials = plugin.add_mutually_exclusive_group()
    credentials.add_argument("--ogmem-api-key", help="oGMemory service key (not an LLM key)")
    credentials.add_argument("--ogmem-api-key-env", help="Read the service key from this environment variable")
    plugin.add_argument("--account-id", help="oGMemory account ID")
    plugin.add_argument("--user-id", help="oGMemory user ID")
    plugin.add_argument("--agent-id", help="oGMemory identity for this Agent")
    plugin.add_argument("--output-dir", help="Export directory; defaults to ./ogmem-client-config/<adapter>")
    p_onboard.add_argument("--embedding-model", help="Embedding model name")
    p_onboard.add_argument("--embedding-provider", help="Embedding provider")
    p_onboard.add_argument("--embedding-base-url", help="Embedding API base URL")
    p_onboard.add_argument("--embedding-api-key", help="Embedding API key")
    p_onboard.add_argument("--embedding-dimension", type=int, help="Embedding output dimension")
    p_onboard.add_argument(
        "--vector-db",
        choices=["chroma", "opengauss"],
        help="Vector DB type (local mode defaults to opengauss)",
    )
    p_onboard.add_argument("--port", type=int, default=8090, action=ExplicitStore, help="HTTP port")
    p_onboard.add_argument("--instance-name", default="ogmem", help="Docker quickstart instance name")
    p_onboard.add_argument("--opengauss-port", type=int, default=15432, help="Docker host port for OpenGauss")
    p_onboard.add_argument("--mcp-port", type=int, default=18091, help="Docker host port for oGMemory MCP")
    p_onboard.add_argument("--openclaw-ce-port", type=int, default=18790, help="Docker host port for OpenClaw CE")
    p_onboard.add_argument("--openclaw-mcp-port", type=int, default=18791, help="Docker host port for OpenClaw MCP")
    p_onboard.add_argument(
        "--storage-backend",
        choices=["agfs", "sql"],
        default="sql", action=ExplicitStore,
        help="Storage backend (default: sql)",
    )
    p_onboard.add_argument("--db-connection", help="SQL storage connection string")
    p_onboard.add_argument("--vector-db-connection", help="Vector database connection string (defaults to SQL connection on first setup)")
    p_onboard.add_argument(
        "--sql-dialect",
        choices=["postgresql", "opengauss"],
        default="opengauss", action=ExplicitStore,
        help="SQL storage dialect (default: opengauss)",
    )

    # -- start --
    p_start = sub.add_parser("start", help="Start services")
    p_start.add_argument(
        "mode",
        nargs="?",
        default=None,
        choices=["headless", "local", "index"],
        help="What to start",
    )
    p_start.add_argument("-d", "--daemon", action="store_true", help="Run in background")
    p_start.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p_start.add_argument("-c", "--config", help="Path to ogmem.yaml")
    p_start.add_argument("--agfs-bin", help="Override AGFS binary path")
    p_start.add_argument(
        "--health-timeout", type=int, default=30, help="Health check timeout (seconds)"
    )
    # -- stop --
    p_stop = sub.add_parser("stop", help="Stop services")
    p_stop.add_argument(
        "mode",
        nargs="?",
        default=None,
        choices=["headless", "local"],
        help="What to stop",
    )

    # -- status --
    sub.add_parser("status", help="Show service status")

    # -- mcp --
    p_mcp = sub.add_parser(
        "mcp",
        help="Run MCP server",
        description=(
            "Run the MCP server. HTTP settings default to mcp.host, mcp.port, and "
            "mcp.path (OGMEM_MCP_HOST, OGMEM_MCP_PORT, and OGMEM_MCP_PATH). SSE uses "
            "the configured path for "
            "GET and /messages/ for POST. Legacy MCP sessions are process-local; use "
            "one instance or sticky routing for pre-2026 clients. SSE requires "
            "OG_ROLE_CONTROL_ENABLED=true "
            "and OG_ROOT_API_KEY or OG_ADMIN_API_KEYS; clients send the key with "
            "X-API-Key or Authorization: Bearer. Concurrent SSE sessions are capped "
            "by OGMEM_MCP_MAX_SSE_CONNECTIONS (default: 64)."
        ),
    )
    p_mcp.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    p_mcp.add_argument("-c", "--config", help="Path to ogmem.yaml")
    p_mcp.add_argument("--host", default=None, help="Host for HTTP transports")
    p_mcp.add_argument("--port", type=int, default=None, help="Port for HTTP transports")
    p_mcp.add_argument("--path", default=None, help="Path for HTTP transports")

    # -- check --
    sub.add_parser("check", help="Environment diagnostics")

    # -- logs --
    p_logs = sub.add_parser("logs", help="Tail service logs")
    p_logs.add_argument(
        "--service", choices=["agfs", "ogmem", "openclaw"], help="Which service logs"
    )
    p_logs.add_argument("-f", "--follow", action="store_true", help="Follow log output")

    # -- eval --
    p_eval = sub.add_parser("eval", help="Run LoCoMo eval pipeline")
    p_eval.add_argument("eval_mode", choices=["ingest", "qa", "judge", "stat"], help="Eval step")
    p_eval.add_argument("input", help="Input file (JSON or CSV)")
    p_eval.add_argument("--output", "-o", help="Output path")
    p_eval.add_argument("--base-url", default="http://127.0.0.1:18789")
    p_eval.add_argument("--token", help="Auth token")
    p_eval.add_argument("--sample", type=int, help="Sample index")
    p_eval.add_argument("--count", type=int, help="Number of questions")
    p_eval.add_argument("-p", "--parallel", type=int, help="Parallel workers")

    # -- config --
    p_config = sub.add_parser("config", help="Configuration management")
    config_sub = p_config.add_subparsers(dest="config_action")
    config_sub.add_parser("show", help="Show resolved config (masked)")
    config_sub.add_parser("init", help="Generate starter ogmem.yaml")

    # -- skill --
    p_skill = sub.add_parser("skill", help="Inspect published and sync generated Agent Skills")
    p_skill.add_argument(
        "--base-url",
        default=None,
        help=(
            "oGMemory HTTP base URL (default: OGMEM_HTTP_BASE_URL, "
            "OGMEM_SERVER_URL, or http://127.0.0.1:8090)"
        ),
    )
    p_skill.add_argument("--api-key", default=None, help="API key (or OGMEM_MCP_API_KEY)")
    p_skill.add_argument("--account-id", default=None, help="Account header override")
    p_skill.add_argument("--user-id", default=None, help="User header override")
    p_skill.add_argument("--agent-id", default=None, help="Agent whose skills are requested")
    p_skill.add_argument("-c", "--config", default=None, help="Config containing an optional sync target")
    skill_sub = p_skill.add_subparsers(dest="skill_action", required=True)
    p_skill_list = skill_sub.add_parser("list", help="List published skills")
    p_skill_list.add_argument("--query", default=None, help="Optional semantic/text query")
    p_skill_list.add_argument("--limit", type=int, default=50)
    p_skill_show = skill_sub.add_parser("show", help="Show one published SKILL.md")
    p_skill_show.add_argument("name")
    p_skill_show.add_argument("--source", choices=["generated", "imported"], default=None)
    p_skill_show.add_argument("--raw", action="store_true", help="Print only SKILL.md content")
    p_candidates = skill_sub.add_parser("candidates", help="Inspect held or accumulating candidates")
    p_candidates.add_argument("--state", default=None, choices=["accumulating", "held"])
    p_sync = skill_sub.add_parser("sync", help="Sync published skills to an Agent Skills directory")
    p_sync.add_argument("--target-dir", default=None)
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.add_argument("--prune", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch
    cmd = args.command

    if cmd == "check":
        from cli.commands.check import run

        return run(args)
    elif cmd == "status":
        from cli.commands.status import run

        return run(args)
    elif cmd == "mcp":
        from cli.commands.mcp import run

        return run(args)
    elif cmd == "start":
        from cli.commands.start import run

        return run(args)
    elif cmd == "stop":
        from cli.commands.stop import run

        return run(args)
    elif cmd == "onboard":
        from cli.commands.onboard import run

        return run(args)
    elif cmd == "logs":
        from cli.commands.logs import run

        return run(args)
    elif cmd == "eval":
        from cli.commands.eval_cmd import run

        return run(args)
    elif cmd == "config":
        from cli.commands.config import run

        return run(args)
    elif cmd == "skill":
        from cli.commands.skill import run
        return run(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
