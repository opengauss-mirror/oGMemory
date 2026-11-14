#!/usr/bin/env bash
################################################################################
# OpenClaw + oGMemory 一键部署脚本（legacy docker run path）
#
# 架构说明：
# 本脚本会按配置启动数据库容器：
# 1. PostgreSQL/pgvector 容器（storage.backend=sql 时自动启用）
# 2. openGauss 数据库容器（ENABLE_OPENGAUSS=true 时启用）
# 3. ogmemory 容器：独立的 oGMemory 服务（HTTP API，端口 8090）
# 4. openclaw-ogmemory 容器：OpenClaw Gateway（Web 界面，端口 18789）
#
# 用法:
#   1. 编辑 .env 中的配置项（从 .env.example 复制）
#   2. bash deploy.sh -password <密码>  — 部署（ENABLE_OPENGAUSS=true 时必须指定密码）
#   3. bash deploy.sh                   — 部署（ENABLE_OPENGAUSS=false 时无需密码）
#   4. bash deploy.sh --cleanup         — 清理所有已部署的容器
#   5. bash deploy.sh --status          — 查看部署状态
#
# 推荐使用 docker compose 方式（见 README.compose.md）；本脚本保留是为了向后兼容。
################################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Single env file shared with the compose path. We read `.env` first (the new
# convention, also auto-loaded by `docker compose`); if absent we fall back to
# `deploy.env` so users with an existing legacy config keep working.
ENV_FILE="${SCRIPT_DIR}/.env"
[[ ! -f "${ENV_FILE}" && -f "${SCRIPT_DIR}/deploy.env" ]] && ENV_FILE="${SCRIPT_DIR}/deploy.env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

OG_PASSWORD=""
PARSED_ACTION=""
SQL_STORAGE_ENABLED="false"
SQL_STORAGE_BACKEND=""
SQL_STORAGE_CONNECTION_STRING=""
SQL_STORAGE_HOST=""
SQL_STORAGE_PORT=""
SQL_STORAGE_DBNAME=""
SQL_STORAGE_USER=""
SQL_STORAGE_PASSWORD=""
declare -a OPENCLAW_CONTAINER_NAMES=()
declare -a OPENCLAW_INSTANCE_DESCRIPTIONS=()

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*" >&2; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "\n${CYAN}========== $* ==========${NC}" >&2; }

check_command() {
    if ! command -v "$1" &>/dev/null; then
        log_error "未找到命令: $1，请先安装后再运行此脚本"
        exit 1
    fi
}

validate_password() {
    local pwd="$1"
    local categories=0

    if [[ ${#pwd} -lt 8 ]]; then
        log_error "密码长度不足 8 个字符"
        return 1
    fi

    [[ "$pwd" =~ [A-Z] ]] && (( categories += 1 ))
    [[ "$pwd" =~ [a-z] ]] && (( categories += 1 ))
    [[ "$pwd" =~ [0-9] ]] && (( categories += 1 ))
    [[ "$pwd" =~ [\#\?\!\@\$\%\^\&\*] ]] && (( categories += 1 ))

    if (( categories < 3 )); then
        log_error "密码复杂度不够：必须包含大写字母、小写字母、数字、特殊符号(#?!@\$%^&*)中的至少三种"
        return 1
    fi

    return 0
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -password)
                if [[ -z "${2:-}" ]]; then
                    log_error "-password 参数需要指定密码值"
                    exit 1
                fi
                OG_PASSWORD="$2"
                shift 2
                ;;
            --cleanup|--status)
                PARSED_ACTION="$1"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                echo "用法: bash deploy.sh [-password <密码>] [--cleanup|--status]"
                exit 1
                ;;
        esac
    done
}

append_openclaw_instance() {
    local instance_name="$1"
    local gateway_port="$2"
    local gateway_token="$3"
    local auth_api_key="$4"
    local auth_account_id="$5"
    local home_dir="$6"

    OPENCLAW_CONTAINER_NAMES+=("$instance_name")
    OPENCLAW_INSTANCE_DESCRIPTIONS+=(
        "${instance_name}|${gateway_port}|${gateway_token}|${auth_api_key}|${auth_account_id}|${home_dir}"
    )
}

container_exists() {
    local name="$1"
    docker ps -a --format '{{.Names}}' 2>/dev/null | grep -Fxq "${name}"
}

container_is_running() {
    local name="$1"
    docker ps --format '{{.Names}}' 2>/dev/null | grep -Fxq "${name}"
}

get_target_container_names() {
    local names=("${OGMEM_CONTAINER_NAME}")
    local name

    if [[ "${SQL_STORAGE_ENABLED}" == "true" ]]; then
        names=("${POSTGRES_CONTAINER_NAME}" "${names[@]}")
    fi

    if [[ "${ENABLE_OPENGAUSS}" == "true" ]]; then
        names=("${OG_CONTAINER_NAME}" "${names[@]}")
    fi

    for name in "${OPENCLAW_CONTAINER_NAMES[@]}"; do
        names+=("${name}")
    done

    printf '%s\n' "${names[@]}"
}

build_openclaw_instances() {
    OPENCLAW_CONTAINER_NAMES=()
    OPENCLAW_INSTANCE_DESCRIPTIONS=()

    if [[ -n "${OPENCLAW_INSTANCES:-}" ]]; then
        local raw_instance instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir
        IFS=';' read -r -a raw_instances <<< "${OPENCLAW_INSTANCES}"

        for raw_instance in "${raw_instances[@]}"; do
            [[ -z "${raw_instance//[[:space:]]/}" ]] && continue

            IFS='|' read -r instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir <<< "${raw_instance}"

            instance_name="$(echo "${instance_name:-}" | xargs)"
            gateway_port="$(echo "${gateway_port:-}" | xargs)"
            gateway_token="$(echo "${gateway_token:-}" | xargs)"
            auth_api_key="$(echo "${auth_api_key:-}" | xargs)"
            auth_account_id="$(echo "${auth_account_id:-}" | xargs)"
            home_dir="$(echo "${home_dir:-}" | xargs)"

            if [[ -z "${instance_name}" || -z "${gateway_port}" || -z "${gateway_token}" || -z "${auth_api_key}" || -z "${auth_account_id}" ]]; then
                log_error "OPENCLAW_INSTANCES 格式错误: ${raw_instance}"
                log_error "期望格式: container_name|gateway_port|gateway_token|auth_api_key|auth_account_id|openclaw_home_dir"
                exit 1
            fi

            append_openclaw_instance "${instance_name}" "${gateway_port}" "${gateway_token}" "${auth_api_key}" "${auth_account_id}" "${home_dir}"
        done
    else
        append_openclaw_instance \
            "${OPENCLAW_CONTAINER_NAME}" \
            "${GATEWAY_PORT}" \
            "${OPENCLAW_GATEWAY_TOKEN}" \
            "${OG_AUTH_API_KEY:-${OG_ROOT_API_KEY:-}}" \
            "${OG_AUTH_ACCOUNT_ID:-${OG_ACCOUNT_ID:-}}" \
            "${OPENCLAW_HOME_DIR}"
    fi

    if [[ ${#OPENCLAW_CONTAINER_NAMES[@]} -eq 0 ]]; then
        log_error "未找到 OpenClaw 实例配置"
        exit 1
    fi
}

load_sql_storage_config_from_yaml() {
    local yaml_path="${SCRIPT_DIR}/ogmemory.yaml"
    SQL_STORAGE_ENABLED="false"
    SQL_STORAGE_BACKEND=""
    SQL_STORAGE_CONNECTION_STRING=""
    SQL_STORAGE_HOST=""
    SQL_STORAGE_PORT=""
    SQL_STORAGE_DBNAME=""
    SQL_STORAGE_USER=""
    SQL_STORAGE_PASSWORD=""

    if [[ ! -f "${yaml_path}" ]]; then
        return
    fi

    local parsed
    if ! parsed="$(python3 - "${yaml_path}" <<'PY'
import os
import re
import shlex
import sys
from urllib.parse import unquote, urlparse

import yaml

path = sys.argv[1]
data = yaml.safe_load(open(path, encoding="utf-8")) or {}
storage = data.get("storage") or {}

def replace_env(m):
    return os.environ.get(m.group(1), m.group(0))

backend = re.sub(r"\$\{(\w+)\}", replace_env, str(storage.get("backend") or "")).strip().lower()
connection_string = re.sub(r"\$\{(\w+)\}", replace_env, str(storage.get("connection_string") or "")).strip()

parts = {}
if connection_string.startswith(("postgres://", "postgresql://")):
    parsed = urlparse(connection_string)
    if parsed.hostname:
        parts["host"] = parsed.hostname
    if parsed.port:
        parts["port"] = str(parsed.port)
    if parsed.path and parsed.path != "/":
        parts["dbname"] = unquote(parsed.path.lstrip("/"))
    if parsed.username:
        parts["user"] = unquote(parsed.username)
    if parsed.password:
        parts["password"] = unquote(parsed.password)
else:
    for item in shlex.split(connection_string):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key in {"host", "port", "dbname", "user", "password"}:
            parts[key] = value

print(f"backend={backend}")
print(f"connection_string={connection_string}")
for key in ("host", "port", "dbname", "user", "password"):
    print(f"{key}={parts.get(key, '')}")
PY
    )"; then
        log_warn "无法解析 ${yaml_path} 中的 storage 配置，跳过 PostgreSQL 自动部署检测"
        return
    fi

    local key value
    while IFS='=' read -r key value; do
        case "${key}" in
            backend) SQL_STORAGE_BACKEND="${value}" ;;
            connection_string) SQL_STORAGE_CONNECTION_STRING="${value}" ;;
            host) SQL_STORAGE_HOST="${value}" ;;
            port) SQL_STORAGE_PORT="${value}" ;;
            dbname) SQL_STORAGE_DBNAME="${value}" ;;
            user) SQL_STORAGE_USER="${value}" ;;
            password) SQL_STORAGE_PASSWORD="${value}" ;;
        esac
    done <<< "${parsed}"

    if [[ "${SQL_STORAGE_BACKEND}" == "sql" ]]; then
        SQL_STORAGE_ENABLED="true"
        POSTGRES_HOST_PORT="${SQL_STORAGE_PORT:-${POSTGRES_HOST_PORT}}"
        POSTGRES_DB="${SQL_STORAGE_DBNAME:-${POSTGRES_DB}}"
        POSTGRES_USER="${SQL_STORAGE_USER:-${POSTGRES_USER}}"
        POSTGRES_PASSWORD="${SQL_STORAGE_PASSWORD:-${POSTGRES_PASSWORD}}"
    fi
}

load_config() {
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "配置文件不存在: $ENV_FILE"
        log_error "请先复制并编辑 .env 文件"
        exit 1
    fi

    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a

    GATEWAY_PORT="${GATEWAY_PORT:-18789}"
    OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-ogmem-default-token}"
    OPENCLAW_BIND_MODE="${OPENCLAW_BIND_MODE:-lan}"
    OGMEM_URL="${OGMEM_URL:-http://127.0.0.1:8090}"
    OG_AUTH_API_KEY="${OG_AUTH_API_KEY:-}"
    OG_AUTH_ACCOUNT_ID="${OG_AUTH_ACCOUNT_ID:-}"

    OGMEM_HTTP_PORT="${OGMEM_HTTP_PORT:-8090}"
    OGMEM_MCP_PORT="${OGMEM_MCP_PORT:-8091}"
    AGFS_DATA_DIR="${AGFS_DATA_DIR:-}"
    OG_ROOT_API_KEY="${OG_ROOT_API_KEY:-}"
    OG_ACCOUNT_ID="${OG_ACCOUNT_ID:-}"

    OG_IMAGE_REPO="${OG_IMAGE_REPO:-swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed}"
    OG_IMAGE_TAG="${OG_IMAGE_TAG:-0328}"
    OG_IMAGE="${OG_IMAGE_REPO}:${OG_IMAGE_TAG}"
    # VECTOR_DB_* are the canonical .env names; OG_* kept as legacy aliases
    # so an old deploy.env keeps working.
    OG_CONTAINER_NAME="${VECTOR_DB_CONTAINER_NAME:-${OG_CONTAINER_NAME:-opengauss}}"
    OG_NODE_NAME="${VECTOR_DB_NODE_NAME:-${OG_NODE_NAME:-gaussdb}}"
    OG_USERNAME="${VECTOR_DB_USER:-${OG_USERNAME:-gaussdb}}"
    OG_DB_NAME="${VECTOR_DB_NAME:-${OG_DB_NAME:-postgres}}"
    OG_PORT="${VECTOR_DB_PORT:-${OG_PORT:-5432}}"
    OG_HOST_PORT="${VECTOR_DB_HOST_PORT:-${OG_HOST_PORT:-5432}}"
    OG_PASSWORD="${VECTOR_DB_PASSWORD:-${OG_PASSWORD:-}}"
    OG_CPUSET_CPUS="${OG_CPUSET_CPUS:-}"
    OG_WAIT_TIMEOUT="${OG_WAIT_TIMEOUT:-120}"
    # VECTOR_DB_HOST is the vector DB's reachable address (deploy.sh uses
    # host networking, so this is a host IP / 127.0.0.1; compose sets it to
    # the service name). Legacy alias OPENGAUSS_HOST_IP / OPENGAUSS_HOST.
    VECTOR_DB_HOST="${VECTOR_DB_HOST:-${OPENGAUSS_HOST:-${OPENGAUSS_HOST_IP:-127.0.0.1}}}"

    POSTGRES_IMAGE="${POSTGRES_IMAGE:-pgvector/pgvector:pg16}"
    POSTGRES_CONTAINER_NAME="${POSTGRES_CONTAINER_NAME:-postgres_ogmemory}"
    POSTGRES_HOST_PORT="${POSTGRES_HOST_PORT:-5432}"
    POSTGRES_CONTAINER_PORT="${POSTGRES_CONTAINER_PORT:-5432}"
    POSTGRES_DB="${POSTGRES_DB:-ogmemory}"
    POSTGRES_USER="${POSTGRES_USER:-postgres}"
    POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}"
    POSTGRES_WAIT_TIMEOUT="${POSTGRES_WAIT_TIMEOUT:-60}"

    OGMEM_CONTAINER_NAME="${OGMEM_CONTAINER_NAME:-ogmem}"
    OGMEM_IMAGE="${OGMEM_IMAGE:-swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:20261114}"

    OPENCLAW_CONTAINER_NAME="${OPENCLAW_CONTAINER_NAME:-openclaw_ogmem}"
    OPENCLAW_IMAGE="${OPENCLAW_IMAGE:-swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/openclaw-ogmemory:v0.2}"
    OPENCLAW_HOME_DIR="${OPENCLAW_HOME_DIR:-}"
    OPENCLAW_INSTANCES="${OPENCLAW_INSTANCES:-}"

    OGMEM_IMAGE="$(echo "${OGMEM_IMAGE}" | xargs)"
    OPENCLAW_IMAGE="$(echo "${OPENCLAW_IMAGE}" | xargs)"

    ENABLE_OPENGAUSS="${ENABLE_OPENGAUSS:-true}"
    AUTO_HEALTH_CHECK="${AUTO_HEALTH_CHECK:-true}"
    SKIP_PULL="${SKIP_PULL:-false}"

    load_sql_storage_config_from_yaml
    build_openclaw_instances

    log_info "OpenClaw Gateway 配置: provider=${LLM_PROVIDER:-}, model=${LLM_MODEL:-}, port=${GATEWAY_PORT}"
    log_info "oGMemory 服务配置: port=${OGMEM_HTTP_PORT}, mcp_port=${OGMEM_MCP_PORT}, yaml=${SCRIPT_DIR}/ogmemory.yaml"
    if [[ "${SQL_STORAGE_ENABLED}" == "true" ]]; then
        log_info "检测到 storage.backend=sql，将部署 PostgreSQL: ${POSTGRES_CONTAINER_NAME} -> 127.0.0.1:${POSTGRES_HOST_PORT}/${POSTGRES_DB}"
    fi
    log_info "openGauss 镜像: ${OG_IMAGE}"
    [[ "${SQL_STORAGE_ENABLED}" == "true" ]] && log_info "PostgreSQL 镜像: ${POSTGRES_IMAGE}"
    log_info "ogmemory 镜像: ${OGMEM_IMAGE}"
    log_info "openclaw-ogmemory 镜像: ${OPENCLAW_IMAGE}"
    log_info "OpenClaw 实例数: ${#OPENCLAW_CONTAINER_NAMES[@]}"
}

validate_required_config() {
    local required_vars=(
        VECTOR_DB_HOST
        OPENCLAW_HOST_IP
        OG_HOST_PORT
        LLM_PROVIDER
        LLM_API_KEY
        LLM_BASE_URL
        LLM_MODEL
    )
    local var
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" || "${!var}" == your_* ]]; then
            log_error "必填配置项 $var 未设置或仍为模板值，请在 .env 中修改"
            exit 1
        fi
    done

    if [[ ! -f "${SCRIPT_DIR}/ogmemory.yaml" ]]; then
        if [[ -f "${SCRIPT_DIR}/ogmemory.example.yaml" ]]; then
            log_warn "oG-Memory 配置文件不存在: ${SCRIPT_DIR}/ogmemory.yaml"
            log_warn "自动从模板复制: cp ${SCRIPT_DIR}/ogmemory.example.yaml ${SCRIPT_DIR}/ogmemory.yaml"
            log_warn "如需自定义 LLM/Embedding/向量库等配置，请部署后编辑 ${SCRIPT_DIR}/ogmemory.yaml 并重启容器"
            cp "${SCRIPT_DIR}/ogmemory.example.yaml" "${SCRIPT_DIR}/ogmemory.yaml"
        else
            log_error "oG-Memory 配置文件不存在: ${SCRIPT_DIR}/ogmemory.yaml"
            log_error "模板也未找到: ${SCRIPT_DIR}/ogmemory.example.yaml"
            log_error "请重新运行 install.sh 或手动从仓库下载模板"
            exit 1
        fi
    fi

    load_sql_storage_config_from_yaml
    if [[ "${SQL_STORAGE_ENABLED}" == "true" ]]; then
        if [[ -z "${SQL_STORAGE_CONNECTION_STRING}" ]]; then
            log_error "检测到 storage.backend=sql，但 ogmemory.yaml 未配置 storage.connection_string"
            exit 1
        fi
        if [[ -z "${SQL_STORAGE_PASSWORD}" ]]; then
            log_error "检测到 storage.backend=sql，但 PostgreSQL 密码为空，请在 storage.connection_string 中配置 password"
            exit 1
        fi
        if [[ -n "${SQL_STORAGE_HOST}" && "${SQL_STORAGE_HOST}" != "127.0.0.1" && "${SQL_STORAGE_HOST}" != "localhost" ]]; then
            log_warn "storage.connection_string host=${SQL_STORAGE_HOST} 不是本机地址；脚本仍会按 backend=sql 启动本地 PostgreSQL 容器"
            log_warn "请确认 ogmemory 容器能通过该 host 连接到 PostgreSQL"
        fi
    fi

    if [[ -z "${OPENCLAW_INSTANCES:-}" ]]; then
        local role_control_enabled=""
        role_control_enabled="$(python3 -c "import yaml; d=yaml.safe_load(open('${SCRIPT_DIR}/ogmemory.yaml')); print(str(bool(d.get('auth',{}).get('role_control_enabled', False))).lower())" 2>/dev/null || true)"
        if [[ "${role_control_enabled}" == "true" ]]; then
            if [[ -z "${OG_AUTH_API_KEY:-${OG_ROOT_API_KEY:-}}" ]]; then
                log_error "检测到 ogmemory.yaml 已启用多租认证(auth.role_control_enabled=true)，但 .env 中未设置 OG_AUTH_API_KEY 或 OG_ROOT_API_KEY"
                log_error "请在 .env 中显式配置 OpenClaw 访问 oGMemory 使用的认证 Key"
                exit 1
            fi
            if [[ -z "${OG_AUTH_ACCOUNT_ID:-${OG_ACCOUNT_ID:-}}" ]]; then
                log_error "检测到 ogmemory.yaml 已启用多租认证(auth.role_control_enabled=true)，但 .env 中未设置 OG_AUTH_ACCOUNT_ID 或 OG_ACCOUNT_ID"
                log_error "请在 .env 中显式配置 OpenClaw 默认访问的 account_id"
                exit 1
            fi
        fi
    fi
}

preflight_check() {
    log_step "前置检查"
    check_command docker

    if ! docker info &>/dev/null; then
        log_error "Docker 服务未运行或当前用户无权限"
        exit 1
    fi
    log_info "Docker 环境正常"

    if ! command -v curl &>/dev/null; then
        log_warn "未找到 curl 命令，健康检查将不可用"
    fi
}

pull_images() {
    if [[ "${SKIP_PULL}" == "true" ]]; then
        log_info "已跳过镜像拉取（SKIP_PULL=true）"
        return
    fi

    log_step "拉取 Docker 镜像"

    if [[ "${ENABLE_OPENGAUSS}" == "true" ]]; then
        log_info "拉取 openGauss 镜像: ${OG_IMAGE}"
        docker pull "${OG_IMAGE}"
    fi

    if [[ "${SQL_STORAGE_ENABLED}" == "true" ]]; then
        log_info "拉取 PostgreSQL/pgvector 镜像: ${POSTGRES_IMAGE}"
        docker pull "${POSTGRES_IMAGE}"
    fi

    log_info "拉取 ogmemory 镜像: ${OGMEM_IMAGE}"
    docker pull "${OGMEM_IMAGE}"

    log_info "拉取 openclaw-ogmemory 镜像: ${OPENCLAW_IMAGE}"
    docker pull "${OPENCLAW_IMAGE}"
}

deploy_opengauss() {
    log_step "部署 openGauss 数据库"

    if docker ps -a --format '{{.Names}}' | grep -q "^${OG_CONTAINER_NAME}$"; then
        log_warn "容器 ${OG_CONTAINER_NAME} 已存在"
        if docker ps --format '{{.Names}}' | grep -q "^${OG_CONTAINER_NAME}$"; then
            log_info "容器正在运行，跳过创建"
            return
        fi
        log_info "容器已停止，尝试启动..."
        docker start "${OG_CONTAINER_NAME}"
        return
    fi

    local docker_run_args=(
        docker run --name "${OG_CONTAINER_NAME}" --privileged=true -d
        --label ogmem.managed=true
        -e "GS_PASSWORD=${OG_PASSWORD}"
        -e "GS_USER_PASSWORD=${OG_PASSWORD}"
        -p "${OG_HOST_PORT}:${OG_PORT}"
    )

    [[ "${OG_NODE_NAME}" != "gaussdb" ]] && docker_run_args+=(-e "GS_NODENAME=${OG_NODE_NAME}")
    [[ "${OG_USERNAME}" != "gaussdb" ]] && docker_run_args+=(-e "GS_USERNAME=${OG_USERNAME}")
    [[ "${OG_PORT}" != "5432" ]] && docker_run_args+=(-e "GS_PORT=${OG_PORT}")
    [[ "${OG_DB_NAME}" != "postgres" ]] && docker_run_args+=(-e "GS_DB=${OG_DB_NAME}")
    [[ -n "${OG_CPUSET_CPUS}" ]] && docker_run_args+=(--cpuset-cpus="${OG_CPUSET_CPUS}")

    docker_run_args+=("${OG_IMAGE}")

    log_info "启动 openGauss 容器..."
    "${docker_run_args[@]}"
    log_info "openGauss 容器已启动"
}

wait_opengauss_ready() {
    log_info "等待 openGauss 就绪（最多 ${OG_WAIT_TIMEOUT} 秒）..."
    local elapsed=0
    local interval=3

    while (( elapsed < OG_WAIT_TIMEOUT )); do
        if docker exec "${OG_CONTAINER_NAME}" su - omm -c "gsql -d ${OG_DB_NAME} -p ${OG_PORT} -c 'SELECT 1;'" &>/dev/null; then
            log_info "openGauss 已就绪（耗时 ${elapsed} 秒）"
            return 0
        fi
        if (( elapsed > 0 && elapsed % 15 == 0 )); then
            log_warn "已等待 ${elapsed} 秒，数据库仍未就绪..."
            log_warn "可手动测试: docker exec ${OG_CONTAINER_NAME} su - omm -c \"gsql -d ${OG_DB_NAME} -p ${OG_PORT} -c 'SELECT 1;'\""
        fi
        sleep "${interval}"
        elapsed=$(( elapsed + interval ))
    done

    log_error "openGauss 在 ${OG_WAIT_TIMEOUT} 秒内未就绪"
    log_error "请手动检查:"
    log_error "  docker logs ${OG_CONTAINER_NAME}"
    log_error "  docker exec ${OG_CONTAINER_NAME} su - omm -c \"gsql -l\""
    exit 1
}

deploy_postgres() {
    log_step "部署 PostgreSQL 存储数据库"

    if docker ps -a --format '{{.Names}}' | grep -q "^${POSTGRES_CONTAINER_NAME}$"; then
        log_warn "容器 ${POSTGRES_CONTAINER_NAME} 已存在"
        if docker ps --format '{{.Names}}' | grep -q "^${POSTGRES_CONTAINER_NAME}$"; then
            log_info "容器正在运行，跳过创建"
            return
        fi
        log_info "容器已停止，尝试启动..."
        docker start "${POSTGRES_CONTAINER_NAME}"
        return
    fi

    local docker_run_args=(
        docker run --name "${POSTGRES_CONTAINER_NAME}" -d
        --label ogmem.managed=true
        -e "POSTGRES_DB=${POSTGRES_DB}"
        -e "POSTGRES_USER=${POSTGRES_USER}"
        -e "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}"
        -p "${POSTGRES_HOST_PORT}:${POSTGRES_CONTAINER_PORT}"
        "${POSTGRES_IMAGE}"
    )

    log_info "启动 PostgreSQL 容器: ${POSTGRES_CONTAINER_NAME} -> 127.0.0.1:${POSTGRES_HOST_PORT}/${POSTGRES_DB}"
    "${docker_run_args[@]}"
    log_info "PostgreSQL 容器已启动"
}

wait_postgres_ready() {
    log_info "等待 PostgreSQL 就绪（最多 ${POSTGRES_WAIT_TIMEOUT} 秒）..."
    local elapsed=0
    local interval=2

    while (( elapsed < POSTGRES_WAIT_TIMEOUT )); do
        if docker exec "${POSTGRES_CONTAINER_NAME}" pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" &>/dev/null; then
            log_info "PostgreSQL 已就绪（耗时 ${elapsed} 秒）"
            return 0
        fi
        if (( elapsed > 0 && elapsed % 10 == 0 )); then
            log_warn "已等待 ${elapsed} 秒，PostgreSQL 仍未就绪..."
        fi
        sleep "${interval}"
        elapsed=$(( elapsed + interval ))
    done

    log_error "PostgreSQL 在 ${POSTGRES_WAIT_TIMEOUT} 秒内未就绪"
    log_error "请手动检查:"
    log_error "  docker logs ${POSTGRES_CONTAINER_NAME}"
    log_error "  docker exec -it ${POSTGRES_CONTAINER_NAME} psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}"
    exit 1
}

deploy_ogmem() {
    log_step "部署 oGMemory 服务"

    if docker ps -a --format '{{.Names}}' | grep -q "^${OGMEM_CONTAINER_NAME}$"; then
        log_warn "容器 ${OGMEM_CONTAINER_NAME} 已存在"
        if docker ps --format '{{.Names}}' | grep -q "^${OGMEM_CONTAINER_NAME}$"; then
            log_info "容器正在运行，跳过创建"
            return
        fi
        log_info "容器已停止，尝试启动..."
        docker start "${OGMEM_CONTAINER_NAME}"
        return
    fi

    local yaml_path="${SCRIPT_DIR}/ogmemory.yaml"
    if [[ ! -f "${yaml_path}" ]]; then
        if [[ -f "${SCRIPT_DIR}/ogmemory.example.yaml" ]]; then
            log_warn "配置文件不存在: ${yaml_path}，自动从模板复制"
            cp "${SCRIPT_DIR}/ogmemory.example.yaml" "${yaml_path}"
        else
            log_error "配置文件不存在: ${yaml_path}"
            log_error "模板 ogmemory.example.yaml 也未找到，请重新运行 install.sh"
            exit 1
        fi
    fi

    local docker_run_args=(
        docker run -d --name "${OGMEM_CONTAINER_NAME}"
        --network host
        --label ogmem.managed=true
        -v "${yaml_path}:/etc/ogmem/config.yaml:ro"
    )

    if [[ -n "${AGFS_DATA_DIR}" ]]; then
        mkdir -p "${AGFS_DATA_DIR}"
        chmod 777 "${AGFS_DATA_DIR}"
        docker_run_args+=(-v "${AGFS_DATA_DIR}:/data/agfs")
        log_info "挂载 AGFS 数据目录: ${AGFS_DATA_DIR}"
    fi

    if [[ -n "${PERF_LOGS_HOST_DIR:-}" ]]; then
        mkdir -p "${PERF_LOGS_HOST_DIR}"
        chmod 755 "${PERF_LOGS_HOST_DIR}"
        docker_run_args+=(-v "${PERF_LOGS_HOST_DIR}:/opt/ogmem/perf_logs")
        log_info "挂载 perf 日志目录: ${PERF_LOGS_HOST_DIR} -> /opt/ogmem/perf_logs"
    fi

    if [[ "${ENABLE_OPENGAUSS}" == "true" ]]; then
        # VECTOR_DB_* are the canonical client-connection env vars consumed
        # by the container entrypoint. VECTOR_DB_HOST defaults to the host
        # address the container can reach the vector DB on. VECTOR_DB_HOST_PORT
        # is the host port (the one the container dials via loopback), not
        # VECTOR_DB_PORT — that name is reserved for the openGauss
        # container-internal listen port and is not passed to ogmem.
        docker_run_args+=(-e "VECTOR_DB_HOST=${VECTOR_DB_HOST}")
        docker_run_args+=(-e "VECTOR_DB_HOST_PORT=${OG_HOST_PORT}")
        docker_run_args+=(-e "VECTOR_DB_NAME=${OG_DB_NAME}")
        docker_run_args+=(-e "VECTOR_DB_USER=${OG_USERNAME}")
        docker_run_args+=(-e "VECTOR_DB_PASSWORD=${OG_PASSWORD}")
        log_info "openGauss 连接信息已通过环境变量传入: ${VECTOR_DB_HOST}:${OG_HOST_PORT}/${OG_DB_NAME}"
    fi

    # Storage DB env vars (mirrors the VECTOR_DB_* block above). Required
    # for ogmem to build storage.connection_string. STORAGE_DB_HOST defaults
    # to 127.0.0.1 since deploy.sh runs ogmem on the host network namespace;
    # STORAGE_DB_HOST_PORT is the host port that compose / deploy.sh has
    # forwarded to the postgres container's internal 5432.
    docker_run_args+=(-e "STORAGE_DB_HOST=${STORAGE_DB_HOST:-127.0.0.1}")
    docker_run_args+=(-e "STORAGE_DB_HOST_PORT=${STORAGE_DB_HOST_PORT:-5432}")
    docker_run_args+=(-e "STORAGE_DB_NAME=${STORAGE_DB_NAME:-ogmem}")
    docker_run_args+=(-e "STORAGE_DB_USER=${STORAGE_DB_USER:-ogmem}")
    docker_run_args+=(-e "STORAGE_DB_PASSWORD=${STORAGE_DB_PASSWORD:-}")

    local env_vars_for_yaml=(
        "LLM_PROVIDER"
        "LLM_API_KEY"
        "LLM_BASE_URL"
        "LLM_MODEL"
        "OGMEM_HTTP_PORT"
        "OGMEM_WORKERS"
        "OGMEM_GUNICORN_TIMEOUT"
        "OGMEM_MCP_ENABLED"
        "OGMEM_MCP_HOST"
        "OGMEM_MCP_PORT"
        "OGMEM_MCP_PATH"
        "OGMEM_MCP_DEFAULT_TOOLSET"
        "OGMEM_MCP_MAX_SSE_CONNECTIONS"
        "OGMEM_MCP_ENABLE_WRITE_TOOLS"
        "OGMEM_MCP_ENABLE_DELETE_TOOLS"
        "OGMEM_MCP_ALLOWED_HOSTS"
        "OGMEM_MCP_ALLOWED_ORIGINS"
        "OGMEM_MCP_MAX_REQUEST_BYTES"
        "OGMEM_MCP_MAX_RESPONSE_CHARS"
        "OGMEM_MCP_COLLECTION_LIMIT"
        "OG_ACCOUNT_ID"
        "OG_USER_ID"
        "OG_AGENT_ID"
        "OG_ROLE_CONTROL_ENABLED"
        "OG_ROOT_API_KEY"
        "OG_ADMIN_API_KEYS"
        "OG_AGENT_SHARED_MODE"
        "OG_AGENT_SHARED_LIST"
        "OG_HTTP_IP_ALLOWLIST"
        "OG_HTTP_IP_ALLOWLIST_TRUST_PROXY"
        "OG_HTTP_TRUSTED_PROXIES"
        "INDEX_INTERVAL"
        "INDEX_WORKERS"
        "STORAGE_DB_CONNECTION_STRING"
    )
    for var in "${env_vars_for_yaml[@]}"; do
        if [[ -n "${!var:-}" ]]; then
            docker_run_args+=(-e "${var}=${!var}")
        fi
    done

    # Pass perf monitoring vars through when set in .env
    local perf_vars=("OGMEM_PERF_ENABLED" "OGMEM_PERF_OUT" "OGMEM_PERF_HTTP_URL" "OGMEM_PERF_RUN_ID")
    for var in "${perf_vars[@]}"; do
        if [[ -n "${!var:-}" ]]; then
            docker_run_args+=(-e "${var}=${!var}")
        fi
    done

    docker_run_args+=(-e "no_proxy=*")
    docker_run_args+=(-e "NO_PROXY=*")
    docker_run_args+=("${OGMEM_IMAGE}")

    log_info "启动 ogmemory 容器（配置通过 YAML 文件传入）..."
    "${docker_run_args[@]}"
    log_info "ogmemory 容器已启动"
}

deploy_openclaw_instance() {
    local instance_name="$1"
    local gateway_port="$2"
    local gateway_token="$3"
    local auth_api_key="$4"
    local auth_account_id="$5"
    local home_dir="$6"

    if docker ps -a --format '{{.Names}}' | grep -q "^${instance_name}$"; then
        log_warn "容器 ${instance_name} 已存在"
        if docker ps --format '{{.Names}}' | grep -q "^${instance_name}$"; then
            log_info "容器正在运行，跳过创建"
            return
        fi
        log_info "容器已停止，尝试启动..."
        docker start "${instance_name}"
        return
    fi

    local docker_run_args=(
        docker run -d --name "${instance_name}"
        --network host
        --label ogmem.managed=true
    )

    if [[ -n "${home_dir}" ]]; then
        mkdir -p "${home_dir}"
        chmod 777 "${home_dir}"
        docker_run_args+=(-v "${home_dir}:/home/node/.openclaw")
        log_info "挂载 OpenClaw 状态目录: ${home_dir}"
    fi

    docker_run_args+=(
        -e "LLM_PROVIDER=${LLM_PROVIDER}"
        -e "LLM_API_KEY=${LLM_API_KEY}"
        -e "LLM_BASE_URL=${LLM_BASE_URL}"
        -e "LLM_MODEL=${LLM_MODEL}"
        -e "GATEWAY_PORT=${gateway_port}"
        -e "OGMEM_URL=${OGMEM_URL}"
        -e "OG_AUTH_API_KEY=${auth_api_key}"
        -e "OG_AUTH_ACCOUNT_ID=${auth_account_id}"
        -e "OPENCLAW_GATEWAY_TOKEN=${gateway_token}"
    )

    [[ -n "${OPENCLAW_BIND_MODE:-}" ]] && docker_run_args+=(-e "OPENCLAW_BIND_MODE=${OPENCLAW_BIND_MODE}")
    docker_run_args+=(-e "no_proxy=*")
    docker_run_args+=(-e "NO_PROXY=*")
    docker_run_args+=("${OPENCLAW_IMAGE}")

    log_info "启动 openclaw-ogmemory 容器..."
    "${docker_run_args[@]}"
    log_info "openclaw-ogmemory 容器已启动: ${instance_name} -> account=${auth_account_id}, port=${gateway_port}"
}

deploy_openclaw() {
    log_step "部署 OpenClaw Gateway 服务"
    local spec instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir
    for spec in "${OPENCLAW_INSTANCE_DESCRIPTIONS[@]}"; do
        IFS='|' read -r instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir <<< "${spec}"
        deploy_openclaw_instance "${instance_name}" "${gateway_port}" "${gateway_token}" "${auth_api_key}" "${auth_account_id}" "${home_dir}"
    done
}

print_container_table() {
    local docker_ps_args=(docker ps --filter "name=${OG_CONTAINER_NAME}" --filter "name=${OGMEM_CONTAINER_NAME}")
    local name
    if [[ "${SQL_STORAGE_ENABLED}" == "true" ]]; then
        docker_ps_args+=(--filter "name=${POSTGRES_CONTAINER_NAME}")
    fi
    for name in "${OPENCLAW_CONTAINER_NAMES[@]}"; do
        docker_ps_args+=(--filter "name=${name}")
    done
    docker_ps_args+=(--format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")
    "${docker_ps_args[@]}" 2>/dev/null || true
}

cleanup_current_env() {
    log_step "Cleanup containers for current .env"

    local name found_any=false
    while IFS= read -r name; do
        [[ -z "${name}" ]] && continue
        if container_exists "${name}"; then
            found_any=true
            log_info "Removing container: ${name}"
            docker rm -f "${name}"
        else
            log_info "Skip missing container from current .env: ${name}"
        fi
    done < <(get_target_container_names)

    if [[ "${found_any}" != "true" ]]; then
        log_warn "No containers defined by current .env were found"
    fi

    log_info "Cleanup finished"
}

show_status_current_env() {
    log_step "Status for current .env"

    local name found_any=false
    while IFS= read -r name; do
        [[ -z "${name}" ]] && continue
        if container_exists "${name}"; then
            found_any=true
        fi

        if container_is_running "${name}"; then
            log_info "${name}: running"
        elif container_exists "${name}"; then
            log_warn "${name}: stopped"
        else
            log_warn "${name}: not found"
        fi
    done < <(get_target_container_names)

    if [[ "${found_any}" != "true" ]]; then
        log_warn "No containers defined by current .env were found"
        return
    fi

    echo ""
    print_container_table

    if command -v curl &>/dev/null; then
        echo ""
        log_info "Health checks"
        if container_is_running "${OGMEM_CONTAINER_NAME}"; then
            log_info "oGMemory: $(curl -sf "http://127.0.0.1:${OGMEM_HTTP_PORT}/api/v1/health" 2>/dev/null || echo "unreachable")"
        fi
        local spec instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir
        for spec in "${OPENCLAW_INSTANCE_DESCRIPTIONS[@]}"; do
            IFS='|' read -r instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir <<< "${spec}"
            if container_is_running "${instance_name}"; then
                log_info "OpenClaw ${instance_name}: $(curl -sf "http://127.0.0.1:${gateway_port}/health" 2>/dev/null || echo "unreachable")"
            fi
        done
    fi
}

health_check() {
    if [[ "${AUTO_HEALTH_CHECK}" != "true" ]]; then
        return
    fi

    log_step "服务健康检查"

    local elapsed=0
    local max_wait=30
    local interval=3

    log_info "等待 oGMemory 服务就绪..."
    while (( elapsed < max_wait )); do
        if curl -sf "http://127.0.0.1:${OGMEM_HTTP_PORT}/api/v1/health" &>/dev/null 2>&1; then
            log_info "oGMemory 健康检查通过"
            break
        fi
        sleep "${interval}"
        elapsed=$(( elapsed + interval ))
    done

    if (( elapsed >= max_wait )); then
        log_warn "oGMemory 健康检查未在 ${max_wait} 秒内通过，请手动确认:"
        log_warn "  curl http://${OPENCLAW_HOST_IP}:${OGMEM_HTTP_PORT}/api/v1/health"
        log_warn "  docker logs ${OGMEM_CONTAINER_NAME}"
    fi

    local spec instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir
    for spec in "${OPENCLAW_INSTANCE_DESCRIPTIONS[@]}"; do
        IFS='|' read -r instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir <<< "${spec}"
        log_info "等待 OpenClaw 服务就绪: ${instance_name}..."
        elapsed=0
        while (( elapsed < max_wait )); do
            if curl -sf "http://127.0.0.1:${gateway_port}/health" &>/dev/null 2>&1; then
                log_info "OpenClaw 健康检查通过: ${instance_name}"
                break
            fi
            sleep "${interval}"
            elapsed=$(( elapsed + interval ))
        done
        if (( elapsed >= max_wait )); then
            log_warn "OpenClaw 健康检查未在 ${max_wait} 秒内通过: ${instance_name}，请手动确认:"
            log_warn "  curl http://${OPENCLAW_HOST_IP}:${gateway_port}/health"
            log_warn "  docker logs ${instance_name}"
        fi
    done

    echo ""
    log_info "容器运行状态:"
    print_container_table
}

cleanup() {
    cleanup_current_env
    return

    log_step "清理部署的容器"

    local name found_any=false
    while IFS= read -r name; do
        [[ -z "${name}" ]] && continue
        log_info "停止并删除容器: ${name}"
        if container_exists "${name}"; then
            found_any=true
            log_info "Removing container: ${name}"
            docker rm -f "${name}"
        else
            log_info "Skip missing container from current .env: ${name}"
        fi
    done < <(get_target_container_names)

    log_info "清理完成"
    if [[ "${found_any}" != "true" ]]; then
        log_warn "No containers defined by current .env were found"
    fi

    log_info "Cleanup finished"
}

show_status() {
    show_status_current_env
    return

    log_step "部署状态"

    local name found_any=false

    if [[ -z "${managed}" ]]; then
        log_warn "未发现由本脚本管理的容器"
        return
    fi

    while IFS= read -r name; do
        [[ -z "${name}" ]] && continue
        if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
            log_info "${name}: 运行中"
        else
            log_warn "${name}: 已停止"
        fi
    done <<< "${managed}"

    echo ""
    docker ps --filter "label=ogmem.managed=true" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || true

    if command -v curl &>/dev/null; then
        echo ""
        log_info "服务健康检查:"
        log_info "oGMemory: $(curl -sf "http://127.0.0.1:${OGMEM_HTTP_PORT}/api/v1/health" 2>/dev/null || echo "未响应")"
        local spec instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir
        for spec in "${OPENCLAW_INSTANCE_DESCRIPTIONS[@]}"; do
            IFS='|' read -r instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir <<< "${spec}"
            log_info "OpenClaw ${instance_name}: $(curl -sf "http://127.0.0.1:${gateway_port}/health" 2>/dev/null || echo "未响应")"
        done
    fi
}

print_summary() {
    log_step "部署完成"

    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                  部署成功！                                  ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}容器列表:${NC}"
    if [[ "${SQL_STORAGE_ENABLED}" == "true" ]]; then
        echo -e "  - PostgreSQL: ${POSTGRES_CONTAINER_NAME} (端口: ${POSTGRES_HOST_PORT}, 数据库: ${POSTGRES_DB})"
    fi
    if [[ "${ENABLE_OPENGAUSS}" == "true" ]]; then
        echo -e "  - openGauss: ${OG_CONTAINER_NAME} (端口: ${OG_HOST_PORT})"
    fi
    echo -e "  - oGMemory: ${OGMEM_CONTAINER_NAME} (HTTP: ${OGMEM_HTTP_PORT}, MCP: ${OGMEM_MCP_PORT})"
    local spec instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir idx=1
    for spec in "${OPENCLAW_INSTANCE_DESCRIPTIONS[@]}"; do
        IFS='|' read -r instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir <<< "${spec}"
        echo -e "  - OpenClaw #${idx}: ${instance_name} (端口: ${gateway_port}, 账户: ${auth_account_id})"
        idx=$(( idx + 1 ))
    done
    echo ""
    echo -e "${GREEN}访问地址:${NC}"
    echo -e "  oGMemory API:  http://${OPENCLAW_HOST_IP}:${OGMEM_HTTP_PORT}"
    if [[ "${OGMEM_MCP_ENABLED:-true}" == "true" ]]; then
        echo -e "  oGMemory MCP:  http://${OPENCLAW_HOST_IP}:${OGMEM_MCP_PORT}${OGMEM_MCP_PATH:-/mcp}"
    else
        echo -e "  oGMemory MCP:  disabled"
    fi
    for spec in "${OPENCLAW_INSTANCE_DESCRIPTIONS[@]}"; do
        IFS='|' read -r instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir <<< "${spec}"
        echo -e "  OpenClaw Web:   http://${OPENCLAW_HOST_IP}:${gateway_port}  (token: ${gateway_token})"
    done
    echo ""
    echo -e "${GREEN}首次访问:${NC}"
    echo -e "  1. 浏览器打开上述地址"
    echo -e "  2. 输入 Gateway Token"
    echo -e "  3. 配对成功后即可使用"
    echo ""
    echo -e "${GREEN}常用命令:${NC}"
    echo -e "  查看 oGMemory 日志:  docker logs ${OGMEM_CONTAINER_NAME} -f"
    for spec in "${OPENCLAW_INSTANCE_DESCRIPTIONS[@]}"; do
        IFS='|' read -r instance_name gateway_port gateway_token auth_api_key auth_account_id home_dir <<< "${spec}"
        echo -e "  查看 OpenClaw 日志:   docker logs ${instance_name} -f"
    done
    echo -e "  重启所有服务:         docker restart $(get_target_container_names | xargs)"
    echo -e "  进入 oGMemory 容器:  docker exec -it ${OGMEM_CONTAINER_NAME} bash"
    echo ""
}

main() {
    parse_args "$@"

    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║        OpenClaw + oGMemory 一键部署脚本                    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    load_config
    preflight_check

    case "${PARSED_ACTION}" in
        --cleanup)
            cleanup_current_env
            return
            ;;
        --status)
            show_status_current_env
            return
            ;;
    esac

    validate_required_config

    if [[ "${ENABLE_OPENGAUSS}" == "true" ]]; then
        if [[ -z "${OG_PASSWORD}" ]]; then
            log_error "启用 openGauss 时必须通过 -password 参数指定数据库密码"
            echo "用法: bash deploy.sh -password <密码>"
            echo ""
            echo "密码要求："
            echo "  - 长度至少 8 个字符"
            echo "  - 包含大写字母、小写字母、数字、特殊符号(#?!@\$%^&*)中的至少三种"
            exit 1
        fi
        if ! validate_password "${OG_PASSWORD}"; then
            exit 1
        fi
        log_info "密码复杂度校验通过"
        if [[ "${SQL_STORAGE_ENABLED}" == "true" ]]; then
            log_info "部署模式: PostgreSQL + openGauss + ogmemory + openclaw-ogmemory"
        else
            log_info "部署模式: openGauss + ogmemory + openclaw-ogmemory"
        fi
    else
        if [[ "${SQL_STORAGE_ENABLED}" == "true" ]]; then
            log_info "部署模式: PostgreSQL + ogmemory + openclaw-ogmemory（ENABLE_OPENGAUSS=false）"
        else
            log_info "部署模式: ogmemory + openclaw-ogmemory（ENABLE_OPENGAUSS=false）"
        fi
    fi

    pull_images

    if [[ "${SQL_STORAGE_ENABLED}" == "true" ]]; then
        deploy_postgres
        wait_postgres_ready
    fi

    if [[ "${ENABLE_OPENGAUSS}" == "true" ]]; then
        deploy_opengauss
        wait_opengauss_ready
    fi

    deploy_ogmem
    deploy_openclaw
    health_check
    print_summary
}

main "$@"
