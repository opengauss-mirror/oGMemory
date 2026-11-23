#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

################################################################################
# ogmem stack health check — for daily checks + before running LoCoMo
#
# Usage:
#   ./check.sh              # full health check
#   ./check.sh ogmem        # check just one service
#   ./check.sh locomo       # LoCoMo-specific check (extra endpoints)
#   ./check.sh -h           # help
#
# Exit code: 0 = all green, 1 = at least one check failed
################################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load container names from .env so the script can look them up by exact
# name (rather than guessing the suffix). Falls back to safe defaults if
# the var isn't set.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

PROFILES=(--profile with-db --profile with-openclaw)
# Defaults below MUST match `deploy/docker-compose.yml` defaults so that a
# fresh `./check.sh` works against a default `docker compose up -d` with
# no .env present. Values from .env override these.
OGMEM_PORT="${OGMEM_HTTP_PORT:-8090}"
OG_HOST_PORT="${VECTOR_DB_HOST_PORT:-5432}"
PG_HOST_PORT="${STORAGE_DB_HOST_PORT:-5432}"
GW_PORT="${GATEWAY_PORT:-18789}"
OG_CT="${VECTOR_DB_CONTAINER_NAME:-opengauss}"
OGMEM_CT="${OGMEM_CONTAINER_NAME:-ogmem}"
PG_CT="${STORAGE_DB_CONTAINER_NAME:-ogmem-postgres}"
OC_CT="${OPENCLAW_CONTAINER_NAME:-openclaw_ogmem}"

# Look up a container by name, tolerating several naming conventions:
#   * exact name:                       ogmem
#   * compose v2 with project prefix:   <project>-ogmem-1   (top-level name: ogmem → "ogmem-ogmem-1")
#   * compose v2 plain suffix:          <service>-1          (only when container_name is unset)
#   * compose v1 hash prefix:           <12hex>_ogmem
#   * user-renamed (custom):            ogmem_yuanjian
# Exact match wins so we don't accidentally latch onto an unrelated stack
# whose name happens to contain our service name as a substring (e.g. when
# looking for `ogmem`, don't pick `ogmem-trace2skill-pg-...`).
resolve_ct() {
    local ct_name="$1"
    local names
    names=$(docker ps --format '{{.Names}}')
    local exact
    exact=$(printf '%s\n' "$names" | grep -Fx "$ct_name" || true)
    if [[ -n "$exact" ]]; then
        printf '%s\n' "$exact"
        return 0
    fi
    printf '%s\n' "$names" | grep -E "(^|[-_])${ct_name}([-_]|$)" | head -1 || true
}

ok()   { printf "  \033[0;32m✓\033[0m %s\n" "$*"; }
fail() { printf "  \033[0;31m✗\033[0m %s\n" "$*"; }
warn() { printf "  \033[1;33m!\033[0m %s\n" "$*"; }
section() { printf "\n\033[1;36m=== %s ===\033[0m\n" "$*"; }

overall_ok=true

# ============================================================================
# 1. 容器状态
# ============================================================================
check_containers() {
    section "1. 容器状态"
    docker compose "${PROFILES[@]}" ps 2>&1 || { fail "docker compose ps 失败"; return 1; }

    # Look up by service using the container name from .env. We use
    # `resolve_ct` which handles the `<project-hash>_<name>` variant too.
    for svc in opengauss postgres ogmem; do
        local ct_name
        case "$svc" in
            opengauss) ct_name=$(resolve_ct "$OG_CT") ;;
            postgres)  ct_name=$(resolve_ct "$PG_CT") ;;
            ogmem)     ct_name=$(resolve_ct "$OGMEM_CT") ;;
        esac
        if [[ -z "$ct_name" ]]; then
            fail "$svc: 找不到容器（期待 $svc → 名称含 '$([ "$svc" = "postgres" ] && echo "$PG_CT" || ([ "$svc" = "ogmem" ] && echo "$OGMEM_CT" || echo "$OG_CT"))'）"
            overall_ok=false
            continue
        fi
        local status
        status=$(docker inspect --format '{{.State.Health.Status}}' "$ct_name" 2>/dev/null || echo "?")
        if [[ "$status" == "healthy" ]]; then
            ok "$svc: $ct_name healthy"
        else
            fail "$svc: $ct_name $status"
            overall_ok=false
        fi
    done

    # openclaw (opt-in via with-openclaw profile)
    local oc_ct
    oc_ct=$(resolve_ct "$OC_CT")
    if [[ -n "$oc_ct" ]]; then
        local oc_status
        oc_status=$(docker inspect --format '{{.State.Health.Status}}' "$oc_ct" 2>/dev/null || echo "?")
        if [[ "$oc_status" == "healthy" ]]; then
            ok "openclaw: $oc_ct healthy"
        else
            warn "openclaw: $oc_ct $oc_status（LoCoMo perf 不需要；eval 模式才需要）"
        fi
    else
        warn "openclaw 未启动（profile with-openclaw 未启用？）"
    fi
}

# ============================================================================
# 2. ogmem HTTP 端点
# ============================================================================
check_ogmem_http() {
    section "2. ogmem HTTP 端点 (:${OGMEM_PORT})"

    if ! command -v curl &>/dev/null; then
        warn "未安装 curl，跳过 HTTP 检查"
        return
    fi

    # Health
    local body
    body=$(curl -sS "http://127.0.0.1:${OGMEM_PORT}/api/v1/health" 2>&1)
    if [[ $? -eq 0 ]]; then
        local status backend storage sql llm
        status=$(echo "$body"  | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null)
        backend=$(echo "$body" | python3 -c "import json,sys; print(json.load(sys.stdin).get('backend','?'))" 2>/dev/null)
        storage=$(echo "$body" | python3 -c "import json,sys; print(json.load(sys.stdin).get('storage_backend','?'))" 2>/dev/null)
        sql=$(echo "$body"     | python3 -c "import json,sys; print(json.load(sys.stdin).get('sql','?'))" 2>/dev/null)
        llm=$(echo "$body"      | python3 -c "import json,sys; print(json.load(sys.stdin).get('llm','?'))" 2>/dev/null)
        if [[ "$status" == "ok" ]]; then
            ok "health: ok  (backend=$backend, storage=$storage, sql=$sql, llm=$llm)"
        else
            fail "health not ok: $body"
            overall_ok=false
        fi
    else
        fail "ogmem :${OGMEM_PORT} 不可达"
        overall_ok=false
        return
    fi

    # Bootstrap
    local boot
    boot=$(curl -sS -X POST "http://127.0.0.1:${OGMEM_PORT}/api/v1/bootstrap" \
        -H "Content-Type: application/json" \
        -d '{"user_id":"u-alice","agent_id":"main"}' 2>&1)
    if echo "$boot" | grep -q '"bootstrapped":true'; then
        ok "bootstrap: bootstrapped=true"
    else
        fail "bootstrap: $boot"
        overall_ok=false
    fi

    # Compose
    local cmp_body
    cmp_body=$(curl -sS -X POST "http://127.0.0.1:${OGMEM_PORT}/api/v1/compose" \
        -H "Content-Type: application/json" \
        -d '{"query":"ping","user_id":"u-alice","agent_id":"main"}' 2>&1)
    if echo "$cmp_body" | python3 -c "import json,sys; sys.exit(0 if 'systemPromptSuffix' in json.load(sys.stdin) else 1)" 2>/dev/null; then
        ok "compose: 响应正常"
    else
        fail "compose 异常: $cmp_body"
        overall_ok=false
    fi
}

# ============================================================================
# 3. LoCoMo 完整端点
# ============================================================================
check_locomo_endpoints() {
    section "3. LoCoMo 用到的端点 (perf_locomo_concurrency.py)"

    if ! command -v curl &>/dev/null; then
        warn "未安装 curl，跳过"
        return
    fi

    local base="http://127.0.0.1:${OGMEM_PORT}"

    # /api/v1/ingest
    local r
    r=$(curl -sS -X POST "$base/api/v1/ingest" \
        -H "Content-Type: application/json" \
        -d '{"user_id":"u-locomo","agent_id":"main","session_id":"sess_locomo","role":"user","content":"hello"}' 2>&1)
    if echo "$r" | grep -q '"ingested":true'; then
        ok "POST /api/v1/ingest"
    else
        fail "/api/v1/ingest: $r"
        overall_ok=false
    fi

    # /api/v1/ingest_batch
    r=$(curl -sS -X POST "$base/api/v1/ingest_batch" \
        -H "Content-Type: application/json" \
        -d '{"user_id":"u-locomo","agent_id":"main","session_id":"sess_locomo","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}' 2>&1)
    if echo "$r" | grep -q '"ingested":true'; then
        ok "POST /api/v1/ingest_batch"
    else
        fail "/api/v1/ingest_batch: $r"
        overall_ok=false
    fi

    # /api/v1/after_turn
    r=$(curl -sS -X POST "$base/api/v1/after_turn" \
        -H "Content-Type: application/json" \
        -d '{"user_id":"u-locomo","agent_id":"main","session_id":"sess_locomo"}' 2>&1)
    if echo "$r" | grep -q '"ok":true'; then
        ok "POST /api/v1/after_turn"
    else
        fail "/api/v1/after_turn: $r"
        overall_ok=false
    fi
}

# ============================================================================
# 4. 数据库连接
# ============================================================================
check_databases() {
    section "4. 数据库连通性"

    # Postgres (the SQL storage backend) — resolve via the same env-driven
    # helper used in check_containers (handles renamed / hash-prefixed names).
    local pg_ct
    pg_ct=$(resolve_ct "$PG_CT")
    if [[ -n "$pg_ct" ]]; then
        local pg_count
        pg_count=$(docker exec "$pg_ct" psql -U "${STORAGE_DB_USER:-ogmem}" -d "${STORAGE_DB_NAME:-ogmem}" -tA -c "SELECT count(*) FROM context_nodes;" 2>/dev/null || echo "ERR")
        if [[ "$pg_count" == "ERR" ]]; then
            fail "postgres: 连接失败"
            overall_ok=false
        else
            ok "postgres (${pg_ct}): context_nodes 行数=$pg_count"
        fi
    else
        fail "postgres 容器未运行（期望 .env 里的 STORAGE_DB_CONTAINER_NAME）"
        overall_ok=false
    fi

    # openGauss (vector DB)
    local og_ct
    og_ct=$(resolve_ct "$OG_CT")
    if [[ -n "$og_ct" ]]; then
        # openGauss listens on ${OG_PORT} inside the container (litescale images
        # default to 42356, not 5432), the DB lives at ${OG_DB_NAME} (default
        # postgres), and gsql needs the password explicitly (there's no peer
        # auth for the gaussdb OS user → DB user mapping). The `-U ${OG_USERNAME}`
        # picks the right DB user; `su - omm` keeps gsql out of the libssl path
        # bind issue we hit when running as root.
        local og_user="${VECTOR_DB_USER:-gaussdb}"
        local og_db="${VECTOR_DB_NAME:-postgres}"
        local og_port="${VECTOR_DB_PORT:-5432}"
        local og_pw="${VECTOR_DB_PASSWORD:-}"
        local og_count
        if [[ -z "$og_pw" ]]; then
            warn "openGauss ($og_ct): .env 缺 VECTOR_DB_PASSWORD, 跳过 vector_index 检查"
        else
            og_count=$(docker exec "$og_ct" su - omm -c \
                "PATH=/usr/local/opengauss/bin:\$PATH gsql -U '$og_user' -d '$og_db' -p '$og_port' -W '$og_pw' -tA -c 'SELECT count(*) FROM vector_index;'" \
                2>/dev/null | tr -d ' ' || echo "ERR")
            if [[ "$og_count" == "ERR" || -z "$og_count" ]]; then
                warn "openGauss ($og_ct): vector_index 表未创建（首次 ingest 后会自动创建）"
            else
                ok "openGauss ($og_ct): vector_index 行数=$og_count"
            fi
        fi
    else
        fail "openGauss 容器未运行"
        overall_ok=false
    fi
}

# ============================================================================
# 5. 端口监听
# ============================================================================
check_ports() {
    section "5. 端口监听"

    local checks=(
        "ogmem:3666:${OGMEM_PORT}"
        "openGauss:42356:${OG_HOST_PORT}"
        "postgres:54321:${PG_HOST_PORT}"
    )
    # Stash ss output once: with `set -o pipefail` and `grep -q`, the
    # pipe returns 141 (SIGPIPE) because grep -q exits on the first match
    # while ss is still writing, which falsely reports the port as not
    # listening. Reading ss into a variable first avoids the race.
    local listening
    listening=$(ss -tln 2>/dev/null || true)
    for entry in "${checks[@]}"; do
        local name expect_port=$entry
        name=$(echo "$entry" | cut -d: -f1)
        expect_port=$(echo "$entry" | cut -d: -f3)
        if grep -qE ":${expect_port}[[:space:]]" <<< "$listening"; then
            ok "$name :${expect_port} 监听中"
        else
            warn "$name :${expect_port} 未监听（其他实例占用了？profile 未启用？）"
        fi
    done
}

# ============================================================================
# 6. 日志末尾
# ============================================================================
check_logs() {
    section "6. 日志末尾（找 ERROR / Traceback）"

    local svc
    for svc in ogmem openclaw; do
        if docker compose "${PROFILES[@]}" ps --format '{{.Names}}' 2>/dev/null | grep -q "$svc"; then
            # Stash logs into a variable first; with `set -o pipefail` the
            # grep chain can return non-zero (no matches found) and trigger
            # errexit on the `local hits=$(...)` assignment.
            local raw hits
            raw=$(docker compose "${PROFILES[@]}" logs --tail=50 "$svc" 2>&1 || true)
            # Filter out our own log noise (decorative '===' headers, ANSI escapes)
            hits=$(printf '%s\n' "$raw" | grep -iE "error|traceback|exception" | \
                   grep -vE "===" | \
                   head -3 || true)
            if [[ -z "$hits" ]]; then
                ok "$svc: 无 error/traceback"
            else
                warn "$svc: 发现 error/traceback ↓"
                echo "$hits" | sed 's/^/      /'
            fi
        fi
    done
}

# ============================================================================
# main
# ============================================================================
case "${1:-full}" in
    full|all|"")
        check_containers
        check_ogmem_http
        check_databases
        check_ports
        check_logs
        ;;
    locomo)
        check_containers
        check_ogmem_http
        check_locomo_endpoints
        check_databases
        check_logs
        ;;
    ogmem|openclaw|opengauss|postgres)
        check_containers
        check_ogmem_http
        check_databases
        check_logs
        ;;
    -h|--help|help)
        sed -n '2,12p' "$0"
        ;;
    *)
        echo "unknown command: $1"; exit 2 ;;
esac

echo ""
if $overall_ok; then
    printf "\033[0;32m✓ 所有检查通过\033[0m\n"
    exit 0
else
    printf "\033[0;31m✗ 至少一个检查失败\033[0m\n"
    exit 1
fi
