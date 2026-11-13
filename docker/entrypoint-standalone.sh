#!/bin/sh
set -e

# --- 0. Resolve ${ENV_VAR} references in config and patch connection_string ---
CONFIG_SRC="${OGMEM_CONFIG:-/etc/ogmem/config.yaml}"
CONFIG_WORK="/tmp/ogmem.resolved.yaml"

python3 << 'PYEOF'
import os
import re

import yaml
from psycopg2.extensions import make_dsn

path_src = os.environ.get("OGMEM_CONFIG", "/etc/ogmem/config.yaml")
path_dst = "/tmp/ogmem.resolved.yaml"

# Parse before interpolation: values containing quotes, newlines or YAML syntax
# remain scalar data, and connection overrides do not depend on field layout.
with open(path_src) as stream:
    config = yaml.safe_load(stream)
if not isinstance(config, dict):
    raise ValueError("configuration must be a YAML mapping")


def expand(value):
    if isinstance(value, str):
        return re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m[1], m[0]), value)
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


config = expand(config)


def selected_backend(section, field, env_name, default):
    # Match OgMemConfig: an explicit YAML value wins, otherwise use env/default.
    # Missing or null sections are valid for environment-only configuration.
    values = config.get(section)
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError(f"{section} configuration must be a mapping")
    value = values.get(field)
    if value is None:
        value = os.environ.get(env_name, default)
    return str(value).strip().casefold()


vector_type = selected_backend("vector_db", "type", "VECTOR_DB_TYPE", "chroma")
storage_backend = selected_backend("storage", "backend", "STORAGE_BACKEND", "agfs")

# VECTOR_DB_HOST_PORT is the host-published port; VECTOR_DB_PORT is the
# container-internal port and must not be used on the host network path.
# Legacy OPENGAUSS_* names remain fallbacks. Preserve mounted SQL DSNs unless
# a connection override was supplied; non-SQL vector providers are untouched.
vector_overrides = (
    "VECTOR_DB_HOST", "VECTOR_DB_HOST_PORT", "VECTOR_DB_NAME",
    "VECTOR_DB_USER", "VECTOR_DB_PASSWORD", "OPENGAUSS_HOST",
    "OPENGAUSS_PORT", "OPENGAUSS_DBNAME", "OPENGAUSS_USER", "OPENGAUSS_PASSWORD",
)
if vector_type == "opengauss" and any(name in os.environ for name in vector_overrides):
    vector = config.get("vector_db") or {}
    config["vector_db"] = vector
    vector["connection_string"] = make_dsn(
        host=os.environ.get("VECTOR_DB_HOST") or os.environ.get("OPENGAUSS_HOST", "127.0.0.1"),
        port=os.environ.get("VECTOR_DB_HOST_PORT") or os.environ.get("OPENGAUSS_PORT", "5432"),
        dbname=os.environ.get("VECTOR_DB_NAME") or os.environ.get("OPENGAUSS_DBNAME", "postgres"),
        user=os.environ.get("VECTOR_DB_USER") or os.environ.get("OPENGAUSS_USER", "gaussdb"),
        password=os.environ.get("VECTOR_DB_PASSWORD", os.environ.get("OPENGAUSS_PASSWORD", "")),
    )
    print("[entrypoint] Patched vector_db.connection_string from environment overrides")

# Storage is configured independently from the vector provider, including
# deployments using Milvus with SQL context storage.
if storage_backend == "sql" and os.environ.get("STORAGE_DB_HOST"):
    storage = config.get("storage") or {}
    config["storage"] = storage
    storage["connection_string"] = make_dsn(
        host=os.environ["STORAGE_DB_HOST"],
        port=os.environ.get("STORAGE_DB_HOST_PORT", "5432"),
        dbname=os.environ.get("STORAGE_DB_NAME", "ogmem"),
        user=os.environ.get("STORAGE_DB_USER", "ogmem"),
        password=os.environ.get("STORAGE_DB_PASSWORD", ""),
    )
    print("[entrypoint] Patched storage.connection_string from environment overrides")

# Resolved YAML contains credentials. Tighten permissions before writing even
# when the path already exists from a previous startup.
fd = os.open(path_dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as stream:
    os.fchmod(stream.fileno(), 0o600)
    yaml.safe_dump(config, stream, allow_unicode=True)

print(f"[entrypoint] Config resolved: {path_src} -> {path_dst}")
PYEOF

export OGMEM_CONFIG="${CONFIG_WORK}"

# --- 1. Load resolved YAML config into shell env (port / workers) ---
CONFIG_ENV=$(python3 /opt/ogmem/scripts/yaml_to_env.py)
eval "$CONFIG_ENV"

# Keep the shell as PID 1 so signals reach every managed service. Register
# cleanup before starting children, including bundled AGFS readiness checks.
API_PID=""
INDEX_PID=""
MCP_PID=""
AGFS_PID=""
cleanup() {
  status=$?
  trap - EXIT
  trap '' TERM INT
  echo "[entrypoint] Shutting down..."
  # Keep storage available until API, MCP and index workers have drained.
  for pid in "$API_PID" "$MCP_PID" "$INDEX_PID"; do
    if [ -n "$pid" ]; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$API_PID" "$MCP_PID" "$INDEX_PID"; do
    if [ -n "$pid" ]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [ -n "$AGFS_PID" ]; then
    kill -TERM "$AGFS_PID" 2>/dev/null || true
    wait "$AGFS_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 0' TERM INT

# Legacy configurations default to AGFS. Preserve their mounted /data/agfs;
# never silently switch an existing deployment to an empty SQL database.
if [ "${OGMEM_START_BUNDLED_AGFS:-false}" = "true" ]; then
  if [ ! -x /opt/agfs/agfs-server ]; then
    echo "[entrypoint] Local AGFS storage requires the standalone-agfs build target. For standalone-no-agfs, configure storage.backend: sql or an external AGFS server." >&2
    exit 1
  fi
  mkdir -p /data/agfs
  echo "[entrypoint] Starting bundled AGFS for legacy local storage..."
  /opt/agfs/agfs-server -c /opt/agfs/config.yaml &
  AGFS_PID=$!
  AGFS_READY=false
  for i in $(seq 1 60); do
    if ! kill -0 "$AGFS_PID" 2>/dev/null; then
      echo "[entrypoint] AGFS exited before becoming ready" >&2
      exit 1
    fi
    if curl -sf "http://127.0.0.1:1833/api/v1/files?path=/serverinfo/version" >/dev/null 2>&1; then
      AGFS_READY=true
      echo "[entrypoint] AGFS server ready"
      break
    fi
    sleep 0.5
  done
  if [ "$AGFS_READY" != "true" ]; then
    echo "[entrypoint] AGFS readiness timed out; refusing to start API" >&2
    kill "$AGFS_PID" 2>/dev/null || true
    exit 1
  fi
fi

# --- 2. Start IndexService (background) ---
echo "[entrypoint] Starting IndexService..."
python3 /opt/ogmem/scripts/run_index_service.py &
INDEX_PID=$!

# --- 3. Start MCP streamable HTTP server (background, optional) ---
if [ "${OGMEM_MCP_ENABLED:-true}" = "true" ] || [ "${OGMEM_MCP_ENABLED:-true}" = "1" ] || [ "${OGMEM_MCP_ENABLED:-true}" = "yes" ]; then
  OGMEM_MCP_BIND_HOST="${OGMEM_MCP_HOST:-0.0.0.0}"
  OGMEM_MCP_BIND_PORT="${OGMEM_MCP_PORT:-8091}"
  OGMEM_MCP_BIND_PATH="${OGMEM_MCP_PATH:-/mcp}"
  echo "[entrypoint] Starting oGMemory MCP server on ${OGMEM_MCP_BIND_HOST}:${OGMEM_MCP_BIND_PORT}${OGMEM_MCP_BIND_PATH}..."
  python3 -m cli.main mcp \
    --transport streamable-http \
    --host "${OGMEM_MCP_BIND_HOST}" \
    --port "${OGMEM_MCP_BIND_PORT}" \
    --path "${OGMEM_MCP_BIND_PATH}" &
  MCP_PID=$!
else
  echo "[entrypoint] MCP server disabled"
fi

# --- 4. Start and wait for the oGMemory HTTP API ---
# cd ensures perf_logs/ and other relative paths resolve under /opt/ogmem (writable by ogmem user)
cd /opt/ogmem
OGMEM_PORT=${OGMEM_HTTP_PORT:-8090}
OGMEM_THREADS=${OGMEM_THREADS:-${OGMEM_WORKERS:-2}}
echo "[entrypoint] Starting oGMemory HTTP server on :${OGMEM_PORT} with 1 process and ${OGMEM_THREADS} threads..."
python3 -m gunicorn \
  --config python:server.gunicorn_config \
  --bind "0.0.0.0:${OGMEM_PORT}" \
  --workers 1 \
  --worker-class gthread \
  --threads "${OGMEM_THREADS}" \
  --timeout "${OGMEM_GUNICORN_TIMEOUT:-900}" \
  --access-logfile - \
  server.app:app &
API_PID=$!
# A failed foreground service must retain its exit status while EXIT cleans
# up its siblings. Waiting as a builtin also lets TERM/INT interrupt promptly.
if wait "$API_PID"; then
  exit 0
else
  exit "$?"
fi
