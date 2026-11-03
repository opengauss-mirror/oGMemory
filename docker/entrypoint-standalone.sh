#!/bin/sh
set -e

# --- 0. Resolve ${ENV_VAR} references in config and patch connection_string ---
CONFIG_SRC="${OGMEM_CONFIG:-/etc/ogmem/config.yaml}"
CONFIG_WORK="/tmp/ogmem.resolved.yaml"

python3 << 'PYEOF'
import json, re, os, sys

path_src = os.environ.get("OGMEM_CONFIG", "/etc/ogmem/config.yaml")
path_dst = "/tmp/ogmem.resolved.yaml"

with open(path_src) as f:
    content = f.read()

def replace_env(m):
    var = m.group(1)
    val = os.environ.get(var)
    if val is not None:
        return val
    return m.group(0)

content = re.sub(r'\$\{(\w+)\}', replace_env, content)

def dsn_value(value):
    # libpq needs its own escaping, independently of YAML and regex escaping.
    if any(c.isspace() or c in "\\'\"" for c in value):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return value


# Patch vector_db.connection_string from VECTOR_DB_* env vars. The legacy
# OPENGAUSS_* spellings still work as fallbacks for older deploy scripts.
# VECTOR_DB_HOST_PORT (not VECTOR_DB_PORT) is the port we dial: compose
# runs ogmem on the host network namespace, so we reach the bridge-
# networked opengauss sibling via loopback on the HOST port (the one
# compose forwards to the openGauss container's internal port). The
# legacy OPENGAUSS_PORT env var (pre-rename) carries the same value.
# Do not replace a mounted YAML DSN unless connection overrides were supplied.
vector_overrides = (
    "VECTOR_DB_HOST", "VECTOR_DB_HOST_PORT", "VECTOR_DB_NAME",
    "VECTOR_DB_USER", "VECTOR_DB_PASSWORD", "OPENGAUSS_HOST",
    "OPENGAUSS_PORT", "OPENGAUSS_DBNAME", "OPENGAUSS_USER", "OPENGAUSS_PASSWORD",
)
if any(name in os.environ for name in vector_overrides):
    og_host   = os.environ.get("VECTOR_DB_HOST") or os.environ.get("OPENGAUSS_HOST", "127.0.0.1")
    og_port   = os.environ.get("VECTOR_DB_HOST_PORT") or os.environ.get("OPENGAUSS_PORT", "5432")
    og_dbname = os.environ.get("VECTOR_DB_NAME") or os.environ.get("OPENGAUSS_DBNAME", "postgres")
    og_user   = os.environ.get("VECTOR_DB_USER") or os.environ.get("OPENGAUSS_USER", "gaussdb")
    og_pass   = os.environ.get("VECTOR_DB_PASSWORD") or os.environ.get("OPENGAUSS_PASSWORD", "")
    og_conn = f"host={og_host} port={og_port} dbname={og_dbname} user={og_user} password={dsn_value(og_pass)}"
    # Match the connection_string that comes right after the `vector_db:` line.
    content = re.sub(
        r'(vector_db:\s*\n\s*type:\s*\S+\s*\n\s*connection_string:\s*)".*?"',
        lambda m: m.group(1) + json.dumps(og_conn),
        content,
    )
    print(f"[entrypoint] Patched vector_db.connection_string: {og_host}:{og_port}/{og_dbname}")

# Patch storage.connection_string from STORAGE_DB_* env vars (postgres).
# Only when STORAGE_DB_HOST is set — otherwise leave the YAML value as-is.
# STORAGE_DB_HOST_PORT is the host port we dial (mirrors VECTOR_DB_HOST_PORT
# above); the postgres container-internal port is hardcoded 5432, so there
# is no STORAGE_DB_PORT variable on this path.
if os.environ.get("STORAGE_DB_HOST"):
    pg_host   = os.environ.get("STORAGE_DB_HOST")
    pg_port   = os.environ.get("STORAGE_DB_HOST_PORT", "5432")
    pg_dbname = os.environ.get("STORAGE_DB_NAME", "ogmem")
    pg_user   = os.environ.get("STORAGE_DB_USER", "ogmem")
    pg_pass   = os.environ.get("STORAGE_DB_PASSWORD", "")
    pg_conn = f"host={pg_host} port={pg_port} dbname={pg_dbname} user={pg_user} password={dsn_value(pg_pass)}"
    content = re.sub(
        r'(storage:\s*\n\s*backend:\s*\S+\s*\n\s*connection_string:\s*)".*?"',
        lambda m: m.group(1) + json.dumps(pg_conn),
        content,
    )
    print(f"[entrypoint] Patched storage.connection_string: {pg_host}:{pg_port}/{pg_dbname}")

with open(path_dst, "w") as f:
    f.write(content)

print(f"[entrypoint] Config resolved: {path_src} -> {path_dst}")
PYEOF

export OGMEM_CONFIG="${CONFIG_WORK}"

# --- 1. Load resolved YAML config into shell env (port / workers) ---
CONFIG_ENV=$(python3 /opt/ogmem/scripts/yaml_to_env.py)
eval "$CONFIG_ENV"

# Legacy configurations default to AGFS. Preserve their mounted /data/agfs;
# never silently switch an existing deployment to an empty SQL database.
AGFS_PID=""
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
MCP_PID=""
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

# --- 4. Trap signals for graceful shutdown ---
cleanup() {
  echo "[entrypoint] Shutting down..."
  if [ -n "${MCP_PID}" ]; then
    kill $MCP_PID 2>/dev/null || true
  fi
  kill $INDEX_PID 2>/dev/null || true
  if [ -n "${AGFS_PID}" ]; then
    kill "$AGFS_PID" 2>/dev/null || true
    wait "$AGFS_PID" 2>/dev/null || true
  fi
  if [ -n "${MCP_PID}" ]; then
    wait $MCP_PID 2>/dev/null || true
  fi
  wait $INDEX_PID 2>/dev/null || true
  exit 0
}
trap cleanup TERM INT

# --- 5. Start oG-Memory HTTP API (foreground, main process) ---
# cd ensures perf_logs/ and other relative paths resolve under /opt/ogmem (writable by ogmem user)
cd /opt/ogmem
OGMEM_PORT=${OGMEM_HTTP_PORT:-8090}
OGMEM_THREADS=${OGMEM_THREADS:-${OGMEM_WORKERS:-2}}
echo "[entrypoint] Starting oGMemory HTTP server on :${OGMEM_PORT} with 1 process and ${OGMEM_THREADS} threads..."
exec python3 -m gunicorn \
  --config python:server.gunicorn_config \
  --bind "0.0.0.0:${OGMEM_PORT}" \
  --workers 1 \
  --worker-class gthread \
  --threads "${OGMEM_THREADS}" \
  --timeout "${OGMEM_GUNICORN_TIMEOUT:-900}" \
  --access-logfile - \
  server.app:app
