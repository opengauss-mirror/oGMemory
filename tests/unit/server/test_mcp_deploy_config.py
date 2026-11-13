from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


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
    assert "COPY docker/entrypoint-standalone.sh /opt/entrypoint.sh" in dockerfile
    assert "chmod 755 /opt/entrypoint.sh" in runtime
    assert "COPY --chmod" not in dockerfile
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
