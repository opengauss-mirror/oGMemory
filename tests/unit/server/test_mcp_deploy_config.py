from __future__ import annotations

import re
import runpy
from pathlib import Path

import yaml

from providers.unified_config import OgMemConfig

ROOT = Path(__file__).resolve().parents[3]

_ENTRYPOINT_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")


def _reference_env_defaults(*, include_commented: bool = False) -> dict[str, str]:
    defaults: dict[str, str] = {}
    prefix = r"^#?" if include_commented else r"^"
    assignment = re.compile(
        prefix + r'([A-Z][A-Z0-9_]*)=(?:"([^"]*)"|(.*))$'
    )
    for line in (ROOT / "config" / "deploy.env.reference").read_text(
        encoding="utf-8"
    ).splitlines():
        match = assignment.match(line.strip())
        if match:
            defaults[match.group(1)] = (
                match.group(2) if match.group(2) is not None else match.group(3)
            )
    return defaults


def _resolve_entrypoint_placeholders(text: str, values: dict[str, str]) -> str:
    return _ENTRYPOINT_PLACEHOLDER.sub(
        lambda match: values.get(match.group(1), match.group(0)),
        text,
    )


def _code_deploy_variable(name: str) -> bool:
    return name.startswith("OGMEM_CODE_") or name.startswith(
        "OGMEM_TOOL_OUTPUT_FILTER_"
    )


def test_example_deploy_keeps_auth_disabled_until_user_sets_real_key():
    cfg = yaml.safe_load((ROOT / "deploy" / "ogmemory.example.yaml").read_text(encoding="utf-8"))

    assert cfg["mcp"]["enabled"] is True
    assert cfg["auth"]["role_control_enabled"] is False
    assert "root_api_key" not in cfg["auth"]


def test_standalone_dockerfile_remains_jenkins_sed_compatible():
    dockerfile = (ROOT / "docker" / "Dockerfile.standalone").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM python:3.11-slim-bookworm")
    assert "GO_BASE_IMAGE" not in dockerfile
    assert "FROM ${" not in dockerfile
    runtime = dockerfile.split("FROM golang:", 1)[0]
    assert "go build" not in runtime
    assert "COPY agfs/agfs-server" not in dockerfile
    assert "COPY --chmod=755 docker/entrypoint-standalone.sh /opt/entrypoint.sh" in dockerfile
    assert "entrypoint-standalone-mcp.sh" not in dockerfile
    assert "ENV OGMEM_MCP_DEFAULT_TOOLSET=read-only" in dockerfile

    patched = re.sub(
        r"^.*$",
        "FROM docker.xuanyuan.run/python:3.11-slim-bookworm",
        dockerfile,
        count=1,
    )
    assert "FROM ${" not in patched


def test_standalone_dockerfile_defines_mcp_env_used_by_default_config():
    dockerfile = (ROOT / "docker" / "Dockerfile.standalone").read_text(encoding="utf-8")
    config = (ROOT / "deploy" / "ogmemory.example.yaml").read_text(encoding="utf-8")

    mcp_envs = set(re.findall(r"\$\{(OGMEM_MCP_[A-Z0-9_]+)\}", config))
    docker_envs = set(re.findall(r"^ENV (OGMEM_MCP_[A-Z0-9_]+)=", dockerfile, flags=re.M))

    assert mcp_envs <= docker_envs


def test_deploy_script_forwards_all_declared_code_memory_variables():
    deploy_script = (ROOT / "deploy" / "deploy.sh").read_text(encoding="utf-8")
    list_match = re.search(
        r"local env_vars_for_yaml=\((?P<body>.*?)\n\s*\)",
        deploy_script,
        flags=re.S,
    )
    assert list_match is not None

    forwarded = set(re.findall(r'"([A-Z][A-Z0-9_]*)"', list_match.group("body")))
    declared = {
        name for name in _reference_env_defaults() if _code_deploy_variable(name)
    }
    forwarded_code = {name for name in forwarded if _code_deploy_variable(name)}

    assert declared == forwarded_code
    assert "OGMEM_MCP_ENABLE_CODE_EDIT_TOOLS" in forwarded


def test_standalone_dockerfile_defines_defaults_for_code_memory_placeholders():
    dockerfile = (ROOT / "docker" / "Dockerfile.standalone").read_text(
        encoding="utf-8"
    )
    config = (ROOT / "deploy" / "ogmemory.example.yaml").read_text(
        encoding="utf-8"
    )
    reference_defaults = _reference_env_defaults(include_commented=True)
    docker_defaults = dict(
        re.findall(r"^ENV ([A-Z][A-Z0-9_]*)=(.*)$", dockerfile, flags=re.M)
    )
    placeholders = set(_ENTRYPOINT_PLACEHOLDER.findall(config))
    required = {
        name for name in placeholders if _code_deploy_variable(name)
    } | {"OGMEM_MCP_ENABLE_CODE_EDIT_TOOLS"}

    assert required <= docker_defaults.keys()
    assert {name: docker_defaults[name] for name in required} == {
        name: reference_defaults[name] for name in required
    }


def test_reference_defaults_resolve_code_memory_yaml_and_load(tmp_path, monkeypatch):
    defaults = _reference_env_defaults(include_commented=True)
    template = (ROOT / "deploy" / "ogmemory.example.yaml").read_text(
        encoding="utf-8"
    )
    resolved = _resolve_entrypoint_placeholders(template, defaults)
    relevant_placeholders = {
        name for name in _ENTRYPOINT_PLACEHOLDER.findall(template)
        if _code_deploy_variable(name)
    } | {"OGMEM_MCP_ENABLE_CODE_EDIT_TOOLS"}

    assert all(f"${{{name}}}" not in resolved for name in relevant_placeholders)

    for name in relevant_placeholders:
        monkeypatch.delenv(name, raising=False)
    config_path = tmp_path / "ogmemory.resolved.yaml"
    config_path.write_text(resolved, encoding="utf-8")

    cfg = OgMemConfig.load(str(config_path))

    assert cfg.code_resources_enabled is False
    assert cfg.code_retrieval_enabled is False
    assert cfg.code_retrieval_semantic_enabled is True
    assert cfg.code_retrieval_bm25_enabled is True
    assert cfg.code_retrieval_symbolic_enabled is True
    assert cfg.code_retrieval_graph_enabled is True
    assert cfg.tool_output_filter_enabled is False
    assert cfg.tool_output_filter_read_enabled is True
    assert cfg.tool_output_filter_bash_enabled is True
    assert cfg.tool_output_filter_adaptive_enabled is True
    assert cfg.tool_output_filter_read_min_chars == 8000
    assert cfg.tool_output_filter_read_max_lines == 60
    assert cfg.tool_output_filter_bash_min_chars == 8000
    assert cfg.tool_output_filter_bash_min_lines == 100
    assert cfg.tool_output_filter_test_max_lines == 120
    assert cfg.tool_output_filter_diff_max_lines == 100
    assert cfg.tool_output_filter_error_max_lines == 100
    assert cfg.tool_output_filter_generic_max_lines == 80
    assert cfg.tool_output_filter_output_max_chars == 50000
    assert cfg.tool_output_recovery_max_chars == 50000
    assert cfg.mcp_enable_code_edit_tools is False


def test_yaml_to_env_exports_code_edit_capability(monkeypatch, capsys):
    monkeypatch.setattr(
        OgMemConfig,
        "load",
        classmethod(lambda cls, _path=None: cls(mcp_enable_code_edit_tools=True)),
    )

    runpy.run_path(str(ROOT / "scripts" / "yaml_to_env.py"), run_name="__main__")

    assert 'export OGMEM_MCP_ENABLE_CODE_EDIT_TOOLS="true"' in capsys.readouterr().out


def test_candidate_local_deploy_env_values_reach_unified_config(tmp_path, monkeypatch):
    defaults = _reference_env_defaults(include_commented=True)
    template = (ROOT / "deploy" / "ogmemory.example.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "ogmemory.resolved.yaml"
    config_path.write_text(
        _resolve_entrypoint_placeholders(template, defaults),
        encoding="utf-8",
    )
    deploy_env = {
        "OGMEM_CODE_RETRIEVAL_MODE": "candidate_local",
        "OGMEM_CODE_RETRIEVAL_CANDIDATE_DISCOVERY_POOL_LIMIT": "250",
        "OGMEM_CODE_RETRIEVAL_CANDIDATE_FILE_LIMIT": "70",
        "OGMEM_CODE_RETRIEVAL_CANDIDATE_SNIPPET_LIMIT": "400",
        "OGMEM_CODE_RETRIEVAL_CANDIDATE_SEMANTIC_LIMIT": "15",
        "OGMEM_CODE_RETRIEVAL_CANDIDATE_EMBEDDING_CHARS": "3000",
        "OGMEM_CODE_RETRIEVAL_CANDIDATE_EMBEDDING_BATCH_TEXTS": "6",
        "OGMEM_CODE_RETRIEVAL_CANDIDATE_EMBEDDING_BATCH_TOKENS": "6000",
        "OGMEM_CODE_RETRIEVAL_CANDIDATE_READ_BYTES": "33554432",
        "OGMEM_CODE_RETRIEVAL_CANDIDATE_DISCOVERY_DEADLINE_MS": "14000",
        "OGMEM_CODE_RETRIEVAL_CANDIDATE_EMBEDDING_DEADLINE_MS": "300000",
        "OGMEM_CODE_RETRIEVAL_CANDIDATE_REQUEST_DEADLINE_MS": "320000",
    }
    for name, value in deploy_env.items():
        monkeypatch.setenv(name, value)

    cfg = OgMemConfig.load(str(config_path))

    assert cfg.code_retrieval_mode == "candidate_local"
    assert cfg.code_retrieval_candidate_discovery_pool_limit == 250
    assert cfg.code_retrieval_candidate_file_limit == 70
    assert cfg.code_retrieval_candidate_snippet_limit == 400
    assert cfg.code_retrieval_candidate_semantic_limit == 15
    assert cfg.code_retrieval_candidate_embedding_chars == 3000
    assert cfg.code_retrieval_candidate_embedding_batch_texts == 6
    assert cfg.code_retrieval_candidate_embedding_batch_tokens == 6000
    assert cfg.code_retrieval_candidate_read_bytes == 33554432
    assert cfg.code_retrieval_candidate_discovery_deadline_ms == 14000
    assert cfg.code_retrieval_candidate_embedding_deadline_ms == 300000
    assert cfg.code_retrieval_candidate_request_deadline_ms == 320000


def test_standalone_dockerfile_declares_runtime_dependencies():
    dockerfile = (ROOT / "docker" / "Dockerfile.standalone").read_text(encoding="utf-8")

    assert '"mcp[cli]==2.0.0"' in dockerfile
    assert "libsqlite3-0" not in dockerfile
    for dependency in ("pydantic", "python-dotenv", "json-repair", "rank-bm25", "nltk"):
        assert dependency in dockerfile


def test_standalone_dockerfile_copies_runtime_skill_package():
    dockerfile = (ROOT / "docker" / "Dockerfile.standalone").read_text(encoding="utf-8")

    assert "COPY skills/     /opt/ogmem/skills/" in dockerfile


def test_standalone_entrypoint_uses_one_process_with_threads():
    entrypoint = (ROOT / "docker" / "entrypoint-standalone.sh").read_text(encoding="utf-8")

    assert '--workers 1' in entrypoint
    assert '--worker-class gthread' in entrypoint
    assert '--threads "${OGMEM_THREADS}"' in entrypoint
    assert 'OGMEM_THREADS=${OGMEM_THREADS:-${OGMEM_WORKERS:-2}}' in entrypoint


def test_mcp_integration_guide_uses_sdk_initialize_result_field():
    guide = (ROOT / "docs" / "integrations" / "mcp.md").read_text(encoding="utf-8")

    assert "init.serverInfo.name" in guide
    assert "init.server_info" not in guide


def test_standalone_build_context_includes_cli_and_default_config():
    dockerfile = (ROOT / "docker" / "Dockerfile.standalone").read_text(encoding="utf-8")
    ignored = {
        line.strip().rstrip("/")
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    for source in ("cli", "deploy/ogmemory.example.yaml"):
        assert (ROOT / source).exists()
        assert re.search(r"^COPY " + re.escape(source) + r"/?\s", dockerfile, re.M)
        assert source not in ignored
    # Keep runtime secrets and large datasets out of the context.
    assert {"deploy/.env", "deploy/ogmemory.yaml", "data", ".git"} <= ignored


def test_standalone_sql_runtime_does_not_require_bundled_agfs():
    dockerfile = (ROOT / "docker" / "Dockerfile.standalone").read_text(encoding="utf-8")
    entrypoint = (ROOT / "docker" / "entrypoint-standalone.sh").read_text(encoding="utf-8")
    runtime = dockerfile.split("FROM golang:", 1)[0]
    assert "/opt/agfs" not in runtime
    assert "/data/agfs" not in runtime
    assert "mkdir -p /data/ogmem /etc/ogmem" in dockerfile
    assert "ogmem:ogmem /data/ogmem" in dockerfile
    assert 'FROM python:3.11-slim-bookworm AS runtime' in runtime
    assert 'FROM runtime AS standalone-agfs' in dockerfile
    assert dockerfile.strip().endswith('FROM runtime AS standalone-no-agfs')
    assert 'COPY --chown=ogmem:ogmem --from=agfs-builder /agfs-server' in dockerfile
    assert 'if [ "${OGMEM_START_BUNDLED_AGFS:-false}" = "true" ]' in entrypoint
    assert 'if [ ! -x /opt/agfs/agfs-server ]' in entrypoint
    compose = (ROOT / "deploy/docker-compose.yml").read_text()
    assert 'target: ${OGMEM_BUILD_TARGET:-standalone-no-agfs}' in compose
    assert entrypoint.count("python3 /opt/ogmem/scripts/run_index_service.py &") == 1
    assert entrypoint.count("python3 -m cli.main mcp") == 1
