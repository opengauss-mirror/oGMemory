#!/usr/bin/env bash
# ContextEngine SQL direct mode launcher.
#
# Starts only the ContextEngine HTTP service with:
#   STORAGE_BACKEND=sql
#   VECTOR_DB_TYPE=memory   (default, or opengauss for pgvector)
#   CONTEXTENGINE_PROVIDER=mock (default)
#
# Usage:
#   cp config/ogmem.reference.yaml config/ogmem.yaml
#   # edit storage.connection_string in config/ogmem.yaml
#   ./scripts/start_sql.sh
#   ./scripts/start_sql.sh --daemon
#   ./scripts/start_sql.sh --status
#   ./scripts/start_sql.sh --stop

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CE_PORT="${OGMEM_HTTP_PORT:-8090}"
CE_PID_FILE="$PROJECT_ROOT/.ce_sql_pid"
CE_LOG="/tmp/context_engine_sql.log"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-30}"
CONFIG_PATH="${OGMEM_CONFIG:-$PROJECT_ROOT/config/ogmem.yaml}"
EXAMPLE_CONFIG_PATH="$PROJECT_ROOT/config/ogmem.reference.yaml"

RAW_STORAGE_BACKEND="${STORAGE_BACKEND:-}"
RAW_SQL_CONNECTION_STRING="${SQL_CONNECTION_STRING:-}"
RAW_VECTOR_DB_TYPE="${VECTOR_DB_TYPE:-}"
RAW_OPENGAUSS_CONNECTION_STRING="${OPENGAUSS_CONNECTION_STRING:-}"
RAW_CONTEXTENGINE_PROVIDER="${CONTEXTENGINE_PROVIDER:-}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }

load_config_defaults() {
    local config_values
    mapfile -t config_values < <("$PROJECT_ROOT/.venv/bin/python" - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = {
    "storage_backend": "",
    "sql_connection_string": "",
    "vector_db_type": "",
    "opengauss_connection_string": "",
    "llm_provider": "",
}

if path.is_file():
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            storage = data.get("storage") or {}
            vector_db = data.get("vector_db") or {}
            llm = data.get("llm") or {}
            if isinstance(storage, dict):
                result["storage_backend"] = str(storage.get("backend") or "")
                result["sql_connection_string"] = str(storage.get("connection_string") or "")
            if isinstance(vector_db, dict):
                result["vector_db_type"] = str(vector_db.get("type") or "")
                result["opengauss_connection_string"] = str(vector_db.get("connection_string") or "")
            if isinstance(llm, dict):
                result["llm_provider"] = str(llm.get("provider") or "")
    except Exception:
        section = None
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if line and not line.startswith(" "):
                section = line[:-1].strip() if line.endswith(":") else None
                continue
            if section and line.startswith("  ") and not line.startswith("    "):
                key, sep, value = line.strip().partition(":")
                if not sep:
                    continue
                value = value.strip().strip('"').strip("'")
                if section == "storage" and key == "backend":
                    result["storage_backend"] = value
                elif section == "storage" and key == "connection_string":
                    result["sql_connection_string"] = value
                elif section == "vector_db" and key == "type":
                    result["vector_db_type"] = value
                elif section == "vector_db" and key == "connection_string":
                    result["opengauss_connection_string"] = value
                elif section == "llm" and key == "provider":
                    result["llm_provider"] = value

for key in (
    "storage_backend",
    "sql_connection_string",
    "vector_db_type",
    "opengauss_connection_string",
    "llm_provider",
):
    print(result[key])
PY
)

    STORAGE_BACKEND="${config_values[0]:-${RAW_STORAGE_BACKEND:-sql}}"
    SQL_CONNECTION_STRING="${config_values[1]:-${RAW_SQL_CONNECTION_STRING:-}}"
    VECTOR_DB_TYPE="${config_values[2]:-${RAW_VECTOR_DB_TYPE:-memory}}"
    OPENGAUSS_CONNECTION_STRING="${config_values[3]:-${RAW_OPENGAUSS_CONNECTION_STRING:-}}"
    CONTEXTENGINE_PROVIDER="${config_values[4]:-${RAW_CONTEXTENGINE_PROVIDER:-mock}}"

    if [ "$VECTOR_DB_TYPE" = "opengauss" ] && [ -z "$OPENGAUSS_CONNECTION_STRING" ]; then
        OPENGAUSS_CONNECTION_STRING="$SQL_CONNECTION_STRING"
    fi

    export STORAGE_BACKEND SQL_CONNECTION_STRING VECTOR_DB_TYPE OPENGAUSS_CONNECTION_STRING CONTEXTENGINE_PROVIDER
}

health_status() {
    curl -s "http://127.0.0.1:$CE_PORT/api/v1/health" 2>/dev/null || true
}

wait_for_health() {
    local pid=$1
    local elapsed=0
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            fail "ContextEngine process died"
            return 1
        fi

        local status
        status="$(health_status)"
        if echo "$status" | grep -q '"ok"'; then
            return 0
        fi

        sleep 1
        elapsed=$((elapsed + 1))
    done
    fail "ContextEngine did not become healthy within ${HEALTH_TIMEOUT}s"
    return 1
}

check_prereqs() {
    if [ ! -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        fail "Missing virtualenv Python at $PROJECT_ROOT/.venv/bin/python"
        echo "  Create it with: python3 -m venv .venv && .venv/bin/pip install -e .[dev,sql]"
        exit 1
    fi

    load_config_defaults

    if [ ! -f "$CONFIG_PATH" ] && [ -z "$RAW_SQL_CONNECTION_STRING" ]; then
        warn "Config file not found: $CONFIG_PATH"
        echo "  Recommended:"
        echo "    cp \"$EXAMPLE_CONFIG_PATH\" \"$CONFIG_PATH\""
        echo "    # edit storage.connection_string in $CONFIG_PATH"
    fi

    if [ -z "$SQL_CONNECTION_STRING" ]; then
        fail "No PostgreSQL DSN found"
        echo "  Set storage.connection_string in $CONFIG_PATH"
        if [ -f "$EXAMPLE_CONFIG_PATH" ]; then
            echo "  Example template: $EXAMPLE_CONFIG_PATH"
        fi
        echo "  Env override still works: export SQL_CONNECTION_STRING='host=127.0.0.1 port=5432 dbname=ogmemory user=postgres password=postgres'"
        exit 1
    fi

    if ! "$PROJECT_ROOT/.venv/bin/python" -c "import psycopg2" >/dev/null 2>&1; then
        fail "psycopg2 is not installed in .venv"
        echo "  Install with: .venv/bin/pip install -e .[dev,sql]"
        exit 1
    fi

    if ! "$PROJECT_ROOT/.venv/bin/python" - <<'PY' >/dev/null 2>&1
import os
import psycopg2

dsn = os.environ["SQL_CONNECTION_STRING"]
conn = psycopg2.connect(dsn)
conn.close()
PY
    then
        fail "Cannot connect to PostgreSQL using the configured SQL DSN"
        exit 1
    fi

    if [ "$VECTOR_DB_TYPE" = "opengauss" ]; then
        if ! "$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import os
import sys
import psycopg2

dsn = os.environ.get("OPENGAUSS_CONNECTION_STRING") or os.environ["SQL_CONNECTION_STRING"]
conn = psycopg2.connect(dsn)
try:
    with conn.cursor() as cur:
        cur.execute("SELECT default_version FROM pg_available_extensions WHERE name = 'vector'")
        row = cur.fetchone()
        if row is None:
            print("pgvector extension package is not installed on this PostgreSQL instance.")
            sys.exit(2)

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
finally:
    conn.close()
PY
        then
            fail "pgvector is not ready for vector_db.type=opengauss"
            echo "  Install the pgvector extension package on PostgreSQL, then run:"
            echo "    CREATE EXTENSION IF NOT EXISTS vector;"
            exit 1
        fi
    fi
}

start_context_engine() {
    local status
    status="$(health_status)"
    if echo "$status" | grep -q '"ok"'; then
        ok "ContextEngine already running on :$CE_PORT"
        return 0
    fi

    check_prereqs

    log "Starting ContextEngine SQL mode on :$CE_PORT..."
    export STORAGE_BACKEND=sql
    export SQL_CONNECTION_STRING
    export VECTOR_DB_TYPE
    export OPENGAUSS_CONNECTION_STRING
    export CONTEXTENGINE_PROVIDER

    (
        cd "$PROJECT_ROOT"
        PYTHONPATH="$PROJECT_ROOT" ./.venv/bin/python server/app.py > "$CE_LOG" 2>&1
    ) &
    local pid=$!
    echo "$pid" > "$CE_PID_FILE"

    if wait_for_health "$pid"; then
        ok "ContextEngine up (PID $pid)"
        ok "storage_backend=sql"
        ok "config_path=$CONFIG_PATH"
        ok "vector_db_type=$VECTOR_DB_TYPE"
        ok "provider=$CONTEXTENGINE_PROVIDER"
    else
        tail -10 "$CE_LOG" || true
        return 1
    fi
}

stop_context_engine() {
    log "Stopping SQL-mode ContextEngine..."
    if [ -f "$CE_PID_FILE" ]; then
        local pid
        pid="$(cat "$CE_PID_FILE")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && ok "PID $pid stopped" || true
        fi
        rm -f "$CE_PID_FILE"
    fi

    local pids
    pids="$(lsof -ti :$CE_PORT 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null && ok "Killed processes on :$CE_PORT" || true
    fi
}

show_status() {
    local status
    status="$(health_status)"
    echo ""
    if echo "$status" | grep -q '"ok"'; then
        echo -e "  ContextEngine(SQL)  http://127.0.0.1:$CE_PORT  ${GREEN}UP${NC}"
        echo -e "  Health              $status"
    else
        echo -e "  ContextEngine(SQL)  http://127.0.0.1:$CE_PORT  ${RED}DOWN${NC}"
    fi
    echo ""
}

cd "$PROJECT_ROOT"

case "${1:-start}" in
    --stop|-s)
        stop_context_engine
        ;;
    --status)
        show_status
        ;;
    --daemon|-d)
        start_context_engine
        show_status
        ;;
    start|--start|"")
        start_context_engine
        echo ""
        ok "SQL direct mode ready"
        show_status
        echo -e "  ${YELLOW}Ctrl+C to stop${NC}"
        trap 'log "Shutting down..."; stop_context_engine' INT TERM
        wait
        ;;
    *)
        echo "Usage: $0 [--start|--stop|--status|--daemon]"
        exit 1
        ;;
esac
