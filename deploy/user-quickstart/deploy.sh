#!/usr/bin/env bash
set -Eeuo pipefail

readonly ENV_FILE="${OGMEMORY_ENV_FILE:-$HOME/ogmemory.env}"
readonly STACK_ROOT="${OGMEMORY_STACK_ROOT:-$HOME}"
readonly PUBLISH_IP="127.0.0.1"
readonly OPENGAUSS_IMAGE="swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed:0328"
readonly OGMEM_IMAGE="swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:1013"
readonly OPENCLAW_IMAGE="swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/openclaw-ogmem:poc_0903"

fail() {
  echo "错误: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "缺少命令: $1"
}

require_value() {
  local name="$1"
  local value="${!name:-}"
  [[ -n "$value" ]] || fail "未配置 $name"
  [[ "$value" != *"替换为"* && "$value" != *"你的"* ]] || fail "$name 仍是占位值"
}

validate_port() {
  local name="$1"
  local value="${!name}"
  [[ "$value" =~ ^[0-9]+$ ]] || fail "$name 必须是数字"
  ((value >= 1024 && value <= 65535)) || fail "$name 必须在 1024-65535 之间"
}

random_hex() {
  local bytes="$1"
  od -An -N "$bytes" -tx1 /dev/urandom | tr -d ' \n'
}

ensure_image() {
  local image="$1"
  if docker image inspect "$image" >/dev/null 2>&1; then
    echo "镜像已存在: $image"
  else
    echo "拉取镜像: $image"
    docker pull "$image" || fail "镜像拉取失败；请联系管理员确认仓库权限和磁盘空间"
  fi
}

wait_http() {
  local name="$1"
  local url="$2"
  local container="$3"
  local attempt
  for attempt in $(seq 1 120); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name 已就绪"
      return 0
    fi
    sleep 3
  done
  docker logs --tail 100 "$container" >&2 || true
  fail "$name 启动超时"
}

show_upstream_error() {
  local response_file="$1"
  # API 错误响应通常不会包含密钥，但仍对常见敏感字段做脱敏后再显示。
  sed -E \
    's/("?(api[_-]?key|authorization|token|password|secret)"?[[:space:]]*:[[:space:]]*)"[^"]*"/\1"<redacted>"/Ig' \
    "$response_file" | head -c 2000 >&2 || true
  echo >&2
}

validate_llm_connectivity() {
  local endpoint="${LLM_BASE_URL%/}/chat/completions"
  local payload
  local response_file
  local http_code

  payload="$(printf '{"model":"%s","messages":[{"role":"user","content":"Reply exactly OK."}],"max_tokens":8,"stream":false}' "$LLM_MODEL")"
  response_file="$(mktemp)"
  echo "预检 LLM 连通性: ${LLM_BASE_URL} / ${LLM_MODEL}"
  if ! http_code="$(curl -sS --connect-timeout 10 --max-time 45 \
    -o "$response_file" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${LLM_API_KEY}" \
    -d "$payload" "$endpoint")"; then
    rm -f "$response_file"
    fail "LLM 连通性失败；请检查 DNS、网络、LLM_BASE_URL 和 API Key"
  fi
  if [[ "$http_code" != 2* ]]; then
    echo "LLM 预检返回 HTTP ${http_code}，响应如下:" >&2
    show_upstream_error "$response_file"
    rm -f "$response_file"
    fail "LLM 模型不可用: ${LLM_MODEL}"
  fi
  if ! grep -Eq '"choices"[[:space:]]*:' "$response_file"; then
    echo "LLM 预检返回成功状态，但响应中没有 choices 字段:" >&2
    show_upstream_error "$response_file"
    rm -f "$response_file"
    fail "LLM 响应格式异常: ${LLM_MODEL}"
  fi
  rm -f "$response_file"
  echo "LLM 连通性验证通过"
}

validate_embedding_connectivity() {
  local endpoint
  local payload
  local response_file
  local http_code

  if [[ "$EMBEDDING_PROVIDER" == "volcengine" ]]; then
    endpoint="${EMBEDDING_BASE_URL%/}/embeddings/multimodal"
    payload="$(printf '{"model":"%s","input":[{"type":"text","text":"deployment preflight"}],"dimensions":%s}' \
      "$EMBEDDING_MODEL" "$EMBEDDING_DIMENSION")"
  else
    endpoint="${EMBEDDING_BASE_URL%/}/embeddings"
    payload="$(printf '{"model":"%s","input":["deployment preflight"]}' "$EMBEDDING_MODEL")"
  fi
  response_file="$(mktemp)"
  echo "预检 Embedding 连通性: ${EMBEDDING_BASE_URL} / ${EMBEDDING_MODEL}"
  if ! http_code="$(curl -sS --connect-timeout 10 --max-time 45 \
    -o "$response_file" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${EMBEDDING_API_KEY}" \
    -d "$payload" "$endpoint")"; then
    rm -f "$response_file"
    fail "Embedding 连通性失败；请检查 DNS、网络、EMBEDDING_BASE_URL 和 API Key"
  fi
  if [[ "$http_code" != 2* ]]; then
    echo "Embedding 预检返回 HTTP ${http_code}，响应如下:" >&2
    show_upstream_error "$response_file"
    rm -f "$response_file"
    fail "Embedding 模型不可用: ${EMBEDDING_MODEL}"
  fi
  if ! grep -Eq '"data"[[:space:]]*:' "$response_file" \
    || ! grep -Eq '"embedding"[[:space:]]*:' "$response_file"; then
    echo "Embedding 预检返回成功状态，但响应中没有 embedding 数据:" >&2
    show_upstream_error "$response_file"
    rm -f "$response_file"
    fail "Embedding 响应格式异常: ${EMBEDDING_MODEL}"
  fi
  rm -f "$response_file"
  echo "Embedding 连通性验证通过"
}

is_true() {
  case "${1,,}" in
    true|1|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

dir_has_entries() {
  local path="$1"
  [[ -d "$path" ]] && find "$path" -mindepth 1 -maxdepth 1 -print -quit | grep -q .
}

show_persistent_directory_warning() {
  local found=0
  local path
  for path in \
    "$STACK_DIR/opengauss" \
    "$STACK_DIR/agfs" \
    "$STACK_DIR/openclaw-ce" \
    "$STACK_DIR/openclaw-mcp"; do
    if dir_has_entries "$path"; then
      [[ "$found" == 0 ]] && echo
      echo "检测到已有持久化目录: $path"
      found=1
    fi
  done
  if ((found)); then
    echo "这些目录会通过 bind mount 映射到容器；重新执行脚本只重建容器，默认不会删除本地目录。"
    echo "因此仅修改 ogmemory.env，可能仍会保留旧的 OpenClaw 状态、AGFS 数据或 openGauss 数据。"
    echo "如需重置，请使用 RESET_OPENCLAW_CONFIG=true、RESET_AGFS_DATA=true 或 RESET_OPENGAUSS_DATA=true。"
    echo "重置操作会直接删除对应目录中的全部数据，执行前请确认无需保留。"
    echo
  fi
}

remove_and_recreate_dir() {
  local path="$1"
  [[ -d "$path" ]] || return 0
  dir_has_entries "$path" || return 0
  rm -rf -- "$path" || fail "无法删除持久化目录: $path"
  mkdir -p "$path"
  echo "已删除并重置目录: $path"
}

container_env_value() {
  local container="$1"
  local name="$2"
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container" 2>/dev/null \
    | awk -F= -v key="$name" '$1 == key {print substr($0, length(key) + 2); exit}'
}

state_value() {
  local file="$1"
  local name="$2"
  awk -F= -v key="$name" '$1 == key {print substr($0, length(key) + 2); exit}' "$file"
}

container_belongs_to_stack() {
  local container="$1"
  local owner_uid
  local stack_dir
  local source

  owner_uid="$(docker inspect --format \
    '{{with index .Config.Labels "io.opengauss.ogmemory.owner-uid"}}{{.}}{{end}}' \
    "$container")"
  stack_dir="$(docker inspect --format \
    '{{with index .Config.Labels "io.opengauss.ogmemory.stack-dir"}}{{.}}{{end}}' \
    "$container")"

  if [[ -n "$owner_uid" || -n "$stack_dir" ]]; then
    [[ "$owner_uid" == "$OWNER_UID" && "$stack_dir" == "$STACK_DIR" ]]
    return
  fi

  while IFS= read -r source; do
    if [[ "$source" == "$STACK_DIR" || "$source" == "$STACK_DIR/"* ]]; then
      return 0
    fi
  done < <(docker inspect --format '{{range .Mounts}}{{println .Source}}{{end}}' "$container")
  return 1
}

[[ -f "$ENV_FILE" ]] || fail "配置文件不存在: $ENV_FILE"
chmod 600 "$ENV_FILE"

# Preserve one-shot RESET_* values from the process environment.  The env file
# is sourced below, but its values must not override an explicit command-line
# assignment such as RESET_OPENCLAW_CONFIG=true "$HOME/deploy.sh".
external_reset_openclaw_config_set="${RESET_OPENCLAW_CONFIG+x}"
external_reset_openclaw_config_value="${RESET_OPENCLAW_CONFIG-}"
external_reset_agfs_data_set="${RESET_AGFS_DATA+x}"
external_reset_agfs_data_value="${RESET_AGFS_DATA-}"
external_reset_opengauss_data_set="${RESET_OPENGAUSS_DATA+x}"
external_reset_opengauss_data_value="${RESET_OPENGAUSS_DATA-}"

# Explicit process values take precedence over the env file.
external_after_turn_threshold_set="${OGMEM_AFTER_TURN_THRESHOLD+x}"
external_after_turn_threshold_value="${OGMEM_AFTER_TURN_THRESHOLD-}"

# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ -n "$external_reset_openclaw_config_set" ]]; then
  RESET_OPENCLAW_CONFIG="$external_reset_openclaw_config_value"
fi
if [[ -n "$external_reset_agfs_data_set" ]]; then
  RESET_AGFS_DATA="$external_reset_agfs_data_value"
fi
if [[ -n "$external_reset_opengauss_data_set" ]]; then
  RESET_OPENGAUSS_DATA="$external_reset_opengauss_data_value"
fi

if [[ -n "$external_after_turn_threshold_set" ]]; then
  OGMEM_AFTER_TURN_THRESHOLD="$external_after_turn_threshold_value"
fi
# Default to short-conversation extraction for this testing quickstart only.
OGMEM_AFTER_TURN_THRESHOLD="${OGMEM_AFTER_TURN_THRESHOLD-1}"
[[ "$OGMEM_AFTER_TURN_THRESHOLD" =~ ^[1-9][0-9]*$ ]] || \
  fail "OGMEM_AFTER_TURN_THRESHOLD 必须是正整数（例如 1 或 200）"

RESET_OPENCLAW_CONFIG="${RESET_OPENCLAW_CONFIG:-false}"
RESET_AGFS_DATA="${RESET_AGFS_DATA:-false}"
RESET_OPENGAUSS_DATA="${RESET_OPENGAUSS_DATA:-false}"

for reset_name in RESET_OPENCLAW_CONFIG RESET_AGFS_DATA RESET_OPENGAUSS_DATA; do
  reset_value="${!reset_name}"
  case "${reset_value,,}" in
    true|false|1|0|yes|no|y|n) ;;
    *) fail "$reset_name 只能设置为 true/false" ;;
  esac
done

for command_name in docker curl awk df od tr seq grep chmod mkdir mktemp rm sha256sum touch find head; do
  require_command "$command_name"
done

SS_COMMAND="$(command -v ss 2>/dev/null || true)"
if [[ -z "$SS_COMMAND" && -x /usr/sbin/ss ]]; then
  SS_COMMAND="/usr/sbin/ss"
fi
[[ -n "$SS_COMMAND" ]] || fail "缺少命令: ss"

[[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]] || \
  fail "当前镜像要求 arm64 Linux"
docker info >/dev/null 2>&1 || fail "当前用户无法使用 Docker"

for name in \
  INSTANCE_NAME \
  LLM_BASE_URL LLM_MODEL LLM_API_KEY \
  EMBEDDING_PROVIDER EMBEDDING_BASE_URL EMBEDDING_MODEL \
  EMBEDDING_API_KEY EMBEDDING_DIMENSION; do
  require_value "$name"
done

[[ "$INSTANCE_NAME" =~ ^[a-z][a-z0-9-]{0,30}$ ]] || \
  fail "INSTANCE_NAME 只能使用小写字母、数字和短横线，并以字母开头"
[[ "$EMBEDDING_PROVIDER" == "openai" || "$EMBEDDING_PROVIDER" == "volcengine" ]] || \
  fail "EMBEDDING_PROVIDER 只支持 openai 或 volcengine"
[[ "$EMBEDDING_DIMENSION" =~ ^[0-9]+$ ]] || fail "EMBEDDING_DIMENSION 必须是数字"
[[ "$LLM_BASE_URL" != *"127.0.0.1"* && "$LLM_BASE_URL" != *"localhost"* ]] || \
  fail "LLM_BASE_URL 必须是容器可访问的地址，不能使用 127.0.0.1 或 localhost"
[[ "$EMBEDDING_BASE_URL" != *"127.0.0.1"* && "$EMBEDDING_BASE_URL" != *"localhost"* ]] || \
  fail "EMBEDDING_BASE_URL 必须是容器可访问的地址，不能使用 127.0.0.1 或 localhost"
[[ "$EMBEDDING_BASE_URL" != */embeddings && "$EMBEDDING_BASE_URL" != */embeddings/ ]] || \
  fail "EMBEDDING_BASE_URL 应填写 Base URL，不能包含具体的 /embeddings 接口"

port_names=(
  OPENGAUSS_HOST_PORT
  OGMEM_HTTP_HOST_PORT
  OGMEM_MCP_HOST_PORT
  OPENCLAW_CE_HOST_PORT
  OPENCLAW_MCP_HOST_PORT
)
declare -A seen_ports=()
for name in "${port_names[@]}"; do
  require_value "$name"
  validate_port "$name"
  value="${!name}"
  [[ -z "${seen_ports[$value]:-}" ]] || \
    fail "$name 与 ${seen_ports[$value]} 使用了相同端口 $value"
  seen_ports[$value]="$name"
done

readonly STACK_DIR="$STACK_ROOT/ogmemory-$INSTANCE_NAME"
readonly OGMEM_NETWORK="${INSTANCE_NAME}_ogmemory_net"
readonly OPENGAUSS_CONTAINER="${INSTANCE_NAME}_opengauss"
readonly OGMEM_CONTAINER="${INSTANCE_NAME}_ogmemory"
readonly CE_CONTAINER="${INSTANCE_NAME}_openclaw_ce"
readonly MCP_CONTAINER="${INSTANCE_NAME}_openclaw_mcp"
readonly OG_ACCOUNT_ID="${INSTANCE_NAME}-account"
readonly OG_USER_ID="${INSTANCE_NAME}-user"
readonly OG_API_AGENT_ID="${INSTANCE_NAME}-api-agent"
readonly OG_CE_AGENT_ID="${INSTANCE_NAME}-ce-agent"
readonly OG_MCP_AGENT_ID="${INSTANCE_NAME}-mcp-agent"
readonly EMBEDDING_STATE_FILE="$STACK_DIR/embedding.state"
readonly OWNER_UID="$UID"
readonly OWNER_UID_LABEL="io.opengauss.ogmemory.owner-uid"
readonly STACK_DIR_LABEL="io.opengauss.ogmemory.stack-dir"

for container in "$CE_CONTAINER" "$MCP_CONTAINER" "$OGMEM_CONTAINER" "$OPENGAUSS_CONTAINER"; do
  if docker container inspect "$container" >/dev/null 2>&1 \
    && ! container_belongs_to_stack "$container"; then
    fail "容器名称已被其他用户占用: $container；请修改 INSTANCE_NAME"
  fi
done

embedding_fingerprint="$(
  printf '%s\n' \
    "$EMBEDDING_PROVIDER" \
    "$EMBEDDING_MODEL" \
    "$EMBEDDING_DIMENSION" \
    | sha256sum | awk '{print $1}'
)"

if [[ -s "$EMBEDDING_STATE_FILE" ]]; then
  saved_fingerprint="$(state_value "$EMBEDDING_STATE_FILE" fingerprint)"
  [[ -n "$saved_fingerprint" ]] || fail "Embedding 状态文件损坏: $EMBEDDING_STATE_FILE"
  if [[ "$saved_fingerprint" != "$embedding_fingerprint" ]]; then
    echo "已有 Embedding: $(state_value "$EMBEDDING_STATE_FILE" provider) / $(state_value "$EMBEDDING_STATE_FILE" model) / $(state_value "$EMBEDDING_STATE_FILE" dimension) 维" >&2
    echo "当前 Embedding: $EMBEDDING_PROVIDER / $EMBEDDING_MODEL / $EMBEDDING_DIMENSION 维" >&2
    fail "已有实例不能更换 Embedding 配置；请使用新的 INSTANCE_NAME"
  fi
elif docker container inspect "$OGMEM_CONTAINER" >/dev/null 2>&1; then
  previous_provider="$(container_env_value "$OGMEM_CONTAINER" EMBEDDING_PROVIDER)"
  previous_model="$(container_env_value "$OGMEM_CONTAINER" EMBEDDING_MODEL)"
  previous_dimension="$(container_env_value "$OGMEM_CONTAINER" EMBEDDING_DIMENSION)"
  previous_fingerprint="$(
    printf '%s\n' \
      "$previous_provider" \
      "$previous_model" \
      "$previous_dimension" \
      | sha256sum | awk '{print $1}'
  )"
  [[ "$previous_fingerprint" == "$embedding_fingerprint" ]] || \
    fail "已有实例不能更换 Embedding 配置；请恢复原配置或使用新的 INSTANCE_NAME"
elif [[ -d "$STACK_DIR/opengauss" ]] && compgen -G "$STACK_DIR/opengauss/*" >/dev/null; then
  fail "检测到未记录 Embedding 配置的 openGauss 数据；为避免损坏索引，未继续部署"
fi

# 在创建或删除任何容器之前验证外部模型服务，避免服务启动后才发现上游不可用。
validate_llm_connectivity
validate_embedding_connectivity

docker_root="$(docker info --format '{{.DockerRootDir}}')"
echo "用户 HOME: $HOME"
echo "Docker 根目录: $docker_root"
df -h "$HOME" "$docker_root"

mkdir -p \
  "$STACK_DIR/opengauss" \
  "$STACK_DIR/agfs" \
  "$STACK_DIR/openclaw-ce" \
  "$STACK_DIR/openclaw-mcp"
chmod 700 "$STACK_DIR"
chmod 777 "$STACK_DIR/opengauss" "$STACK_DIR/agfs"
show_persistent_directory_warning

readonly SECRETS_FILE="$STACK_DIR/secrets.env"
readonly RUNTIME_ENV_FILE="${OGMEMORY_RUNTIME_ENV_FILE:-$HOME/ogmemory-runtime.env}"
if [[ ! -s "$SECRETS_FILE" ]]; then
  umask 077
  db_secret="OgMem9_$(random_hex 8)!A"
  cat > "$SECRETS_FILE" <<EOF
export OG_DB_PASSWORD='$db_secret'
export OG_ROOT_API_KEY='$(random_hex 32)'
export OG_TENANT_API_KEY='$(random_hex 32)'
export CE_GATEWAY_TOKEN='$(random_hex 32)'
export MCP_GATEWAY_TOKEN='$(random_hex 32)'
EOF
  unset db_secret
fi
chmod 600 "$SECRETS_FILE"
# shellcheck disable=SC1090
source "$SECRETS_FILE"
(( ${#OG_DB_PASSWORD} >= 8 && ${#OG_DB_PASSWORD} <= 32 )) || \
  fail "secrets.env 中的 OG_DB_PASSWORD 长度必须为 8-32 个字符"

umask 077
{
  printf 'source %q\n' "$ENV_FILE"
  printf 'export STACK_DIR=%q\n' "$STACK_DIR"
  printf 'export PUBLISH_IP=%q\n' "$PUBLISH_IP"
  printf 'export OG_ACCOUNT_ID=%q\n' "$OG_ACCOUNT_ID"
  printf 'export OG_USER_ID=%q\n' "$OG_USER_ID"
  printf 'export OG_API_AGENT_ID=%q\n' "$OG_API_AGENT_ID"
  printf 'export OG_CE_AGENT_ID=%q\n' "$OG_CE_AGENT_ID"
  printf 'export OG_MCP_AGENT_ID=%q\n' "$OG_MCP_AGENT_ID"
  printf 'source %q\n' "$SECRETS_FILE"
} > "$RUNTIME_ENV_FILE"
chmod 600 "$RUNTIME_ENV_FILE"

ensure_image "$OPENGAUSS_IMAGE"
ensure_image "$OGMEM_IMAGE"
ensure_image "$OPENCLAW_IMAGE"

declare -A own_containers=(
  ["$CE_CONTAINER"]=1
  ["$MCP_CONTAINER"]=1
  ["$OGMEM_CONTAINER"]=1
  ["$OPENGAUSS_CONTAINER"]=1
)
declare -A own_published_ports=()
declare -A other_published_ports=()
while IFS= read -r container; do
  [[ -n "$container" ]] || continue
  while IFS='|' read -r host_ip host_port; do
    [[ -n "$host_port" ]] || continue
    if [[ "$host_ip" != "0.0.0.0" && "$host_ip" != "::" && "$host_ip" != "$PUBLISH_IP" \
      && "$PUBLISH_IP" != "0.0.0.0" && "$PUBLISH_IP" != "::" ]]; then
      continue
    fi
    if [[ -n "${own_containers[$container]:-}" ]]; then
      own_published_ports[$host_port]=1
    else
      other_published_ports[$host_port]="$container"
    fi
  done < <(
    docker inspect --format \
      '{{range $port, $bindings := .HostConfig.PortBindings}}{{range $bindings}}{{printf "%s|%s\n" .HostIp .HostPort}}{{end}}{{end}}' \
      "$container"
  )
done < <(docker ps --format '{{.Names}}')

port_conflicts=0
for name in "${port_names[@]}"; do
  value="${!name}"
  conflict_source="${other_published_ports[$value]:-}"
  if [[ -z "$conflict_source" ]] \
    && "$SS_COMMAND" -H -ltn "sport = :${value}" | grep -q . \
    && [[ -z "${own_published_ports[$value]:-}" ]]; then
    conflict_source="宿主机进程"
  fi
  if [[ -n "$conflict_source" ]]; then
    echo "端口已占用: $name=$value ($conflict_source)" >&2
    port_conflicts=1
  fi
done
((port_conflicts == 0)) || fail "请修改已占用的端口后重新执行 $HOME/deploy.sh"

for container in "$CE_CONTAINER" "$MCP_CONTAINER" "$OGMEM_CONTAINER" "$OPENGAUSS_CONTAINER"; do
  docker rm -f "$container" >/dev/null 2>&1 || true
done

if is_true "$RESET_OPENCLAW_CONFIG"; then
  remove_and_recreate_dir "$STACK_DIR/openclaw-ce"
  remove_and_recreate_dir "$STACK_DIR/openclaw-mcp"
fi
if is_true "$RESET_AGFS_DATA"; then
  remove_and_recreate_dir "$STACK_DIR/agfs"
  chmod 777 "$STACK_DIR/agfs"
fi
if is_true "$RESET_OPENGAUSS_DATA"; then
  remove_and_recreate_dir "$STACK_DIR/opengauss"
  chmod 777 "$STACK_DIR/opengauss"
fi

docker network inspect "$OGMEM_NETWORK" >/dev/null 2>&1 || \
  docker network create "$OGMEM_NETWORK" >/dev/null

echo "创建 openGauss..."
docker run -d \
  --name "$OPENGAUSS_CONTAINER" \
  --label "$OWNER_UID_LABEL=$OWNER_UID" \
  --label "$STACK_DIR_LABEL=$STACK_DIR" \
  --hostname opengauss \
  --network "$OGMEM_NETWORK" \
  --network-alias opengauss \
  --restart unless-stopped \
  --privileged=true \
  -p "${PUBLISH_IP}:${OPENGAUSS_HOST_PORT}:5432" \
  -e "GS_PASSWORD=${OG_DB_PASSWORD}" \
  -e "GS_USER_PASSWORD=${OG_DB_PASSWORD}" \
  -v "$STACK_DIR/opengauss:/var/lib/opengauss" \
  "$OPENGAUSS_IMAGE" >/dev/null

for attempt in $(seq 1 120); do
  if docker exec "$OPENGAUSS_CONTAINER" \
    su - omm -c "gsql -d postgres -p 5432 -c 'SELECT 1;'" >/dev/null 2>&1; then
    echo "openGauss 已就绪"
    break
  fi
  if ((attempt == 120)); then
    docker logs --tail 100 "$OPENGAUSS_CONTAINER" >&2 || true
    fail "openGauss 启动超时"
  fi
  sleep 5
done

cat > "$STACK_DIR/ogmemory.yaml" <<'YAML'
llm:
  provider: "openai"
  api_key: "${LLM_API_KEY}"
  base_url: "${LLM_BASE_URL}"
  model: "${LLM_MODEL}"

embedding:
  provider: "${EMBEDDING_PROVIDER}"
  api_key: "${EMBEDDING_API_KEY}"
  base_url: "${EMBEDDING_BASE_URL}"
  model: "${EMBEDDING_MODEL}"

vector_db:
  type: opengauss
  connection_string: "host=opengauss port=5432 dbname=postgres user=gaussdb password=${OG_DB_PASSWORD}"
  dimension: "${EMBEDDING_DIMENSION}"
  table_name: vector_index
  pool_size: 5

service:
  http_port: 8090
  workers: 1
  threads: 2

mcp:
  enabled: true
  host: "0.0.0.0"
  port: 8091
  path: "/mcp"
  default_toolset: "all"
  enable_write_tools: true
  enable_delete_tools: false
  allowed_hosts: ["ogmemory", "localhost", "127.0.0.1"]

agfs:
  base_url: "http://127.0.0.1:1833"
  mount_prefix: "/local/plugin"

index:
  interval: 15
  workers: 1

identity:
  account_id: "${OG_ACCOUNT_ID}"
  user_id: "${OG_USER_ID}"
  agent_id: "${OG_API_AGENT_ID}"

auth:
  role_control_enabled: true
  root_api_key: "${OG_ROOT_API_KEY}"
  admin_api_keys:
    - "${OG_ADMIN_API_KEYS}"

memory:
  after_turn_threshold: "${OGMEM_AFTER_TURN_THRESHOLD}"
  rolling_compress_enabled: true

agent_memory:
  enabled: false
YAML
chmod 644 "$STACK_DIR/ogmemory.yaml"

echo "创建 oGMemory..."
docker run -d \
  --name "$OGMEM_CONTAINER" \
  --label "$OWNER_UID_LABEL=$OWNER_UID" \
  --label "$STACK_DIR_LABEL=$STACK_DIR" \
  --hostname ogmemory \
  --network "$OGMEM_NETWORK" \
  --network-alias ogmemory \
  --add-host host.docker.internal:host-gateway \
  --restart unless-stopped \
  -p "${PUBLISH_IP}:${OGMEM_HTTP_HOST_PORT}:8090" \
  -p "${PUBLISH_IP}:${OGMEM_MCP_HOST_PORT}:8091" \
  -e OGMEM_CONFIG=/etc/ogmem/config.yaml \
  -e "LLM_API_KEY=${LLM_API_KEY}" \
  -e "LLM_BASE_URL=${LLM_BASE_URL}" \
  -e "LLM_MODEL=${LLM_MODEL}" \
  -e "EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER}" \
  -e "EMBEDDING_API_KEY=${EMBEDDING_API_KEY}" \
  -e "EMBEDDING_BASE_URL=${EMBEDDING_BASE_URL}" \
  -e "EMBEDDING_MODEL=${EMBEDDING_MODEL}" \
  -e "EMBEDDING_DIMENSION=${EMBEDDING_DIMENSION}" \
  -e "OGMEM_AFTER_TURN_THRESHOLD=${OGMEM_AFTER_TURN_THRESHOLD}" \
  -e "OG_DB_PASSWORD=${OG_DB_PASSWORD}" \
  -e "OG_ROOT_API_KEY=${OG_ROOT_API_KEY}" \
  -e "OG_ACCOUNT_ID=${OG_ACCOUNT_ID}" \
  -e "OG_USER_ID=${OG_USER_ID}" \
  -e "OG_API_AGENT_ID=${OG_API_AGENT_ID}" \
  -e "OG_ADMIN_API_KEYS=${OG_ACCOUNT_ID}:${OG_TENANT_API_KEY}" \
  -e NO_PROXY='*' -e no_proxy='*' \
  -v "$STACK_DIR/ogmemory.yaml:/etc/ogmem/config.yaml:ro" \
  -v "$STACK_DIR/agfs:/data/agfs" \
  "$OGMEM_IMAGE" >/dev/null

wait_http \
  "oGMemory" \
  "http://${PUBLISH_IP}:${OGMEM_HTTP_HOST_PORT}/api/v1/health" \
  "$OGMEM_CONTAINER"

echo "验证 Embedding 服务和维度..."
docker exec -e OGMEM_CONFIG=/tmp/ogmem.resolved.yaml \
  "$OGMEM_CONTAINER" python -c '
from providers.unified_config import OgMemConfig
cfg = OgMemConfig.load("/tmp/ogmem.resolved.yaml")
vector = cfg.to_provider_config().create_embedder().embed_texts(["deployment smoke test"])[0]
if len(vector) != cfg.opengauss_dimension:
    raise SystemExit(f"Embedding dimension mismatch: expected {cfg.opengauss_dimension}, got {len(vector)}")
print(f"Embedding 验证通过: {len(vector)} 维")
' || fail "Embedding 验证失败；请检查 Provider、Base URL、模型和维度"

umask 077
cat > "$EMBEDDING_STATE_FILE" <<EOF
fingerprint=$embedding_fingerprint
provider=$EMBEDDING_PROVIDER
base_url=$EMBEDDING_BASE_URL
model=$EMBEDDING_MODEL
dimension=$EMBEDDING_DIMENSION
EOF
chmod 600 "$EMBEDDING_STATE_FILE"

for agent_id in "$OG_API_AGENT_ID" "$OG_CE_AGENT_ID" "$OG_MCP_AGENT_ID"; do
  agents_json="$(curl -fsS \
    "http://${PUBLISH_IP}:${OGMEM_HTTP_HOST_PORT}/api/v1/admin/accounts/${OG_ACCOUNT_ID}/agents" \
    -H "X-API-Key: ${OG_TENANT_API_KEY}" \
    -H "X-Account-ID: ${OG_ACCOUNT_ID}" \
    -H "X-User-ID: ${OG_USER_ID}")"
  if grep -Fq "\"agent_id\":\"${agent_id}\"" <<<"$agents_json"; then
    echo "Agent 已存在: $agent_id"
  else
    curl -fsS -X POST \
      "http://${PUBLISH_IP}:${OGMEM_HTTP_HOST_PORT}/api/v1/admin/accounts/${OG_ACCOUNT_ID}/agents" \
      -H 'Content-Type: application/json' \
      -H "X-API-Key: ${OG_TENANT_API_KEY}" \
      -H "X-Account-ID: ${OG_ACCOUNT_ID}" \
      -H "X-User-ID: ${OG_USER_ID}" \
      --data "{\"agent_id\":\"${agent_id}\"}" >/dev/null
    echo "Agent 注册完成: $agent_id"
  fi
done

cat > "$STACK_DIR/openclaw-ce.template.json" <<'JSON'
{
  "agents": {"defaults": {"skipBootstrap": true, "timeoutSeconds": 600, "model": {"primary": "custom-openai/${LLM_MODEL}"}}},
  "models": {
    "mode": "merge",
    "providers": {
      "custom-openai": {
        "baseUrl": "${LLM_BASE_URL}", "apiKey": "${LLM_API_KEY}",
        "api": "openai-completions", "timeoutSeconds": 300,
        "models": [{
          "id": "${LLM_MODEL}", "name": "${LLM_MODEL}",
          "reasoning": false, "input": ["text"],
          "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
          "contextWindow": 131072, "maxTokens": 8192
        }]
      }
    }
  },
  "gateway": {
    "mode": "local", "bind": "lan", "port": 18789,
    "auth": {"token": "${OPENCLAW_GATEWAY_TOKEN}"},
    "controlUi": {"dangerouslyAllowHostHeaderOriginFallback": true, "dangerouslyDisableDeviceAuth": true},
    "http": {"endpoints": {"responses": {"enabled": true}}}
  },
  "plugins": {
    "allow": ["og-memory-context-engine"],
    "slots": {"contextEngine": "og-memory-context-engine"},
    "entries": {
      "og-memory-context-engine": {
        "enabled": true,
        "hooks": {"allowConversationAccess": true},
        "config": {
          "mode": "remote", "memoryApiBaseUrl": "http://ogmemory:8090",
          "authApiKey": "${OGMEM_TENANT_API_KEY}", "authAccountId": "${OG_ACCOUNT_ID}",
          "accountId": "${OG_ACCOUNT_ID}", "userId": "${OG_USER_ID}", "agentId": "${OG_CE_AGENT_ID}",
          "prefetchEnabled": true, "prefetchTopK": 5,
          "compactTakeoverEnabled": true, "shortTermIndexMode": "sync"
        }
      },
      "memory-core": {"enabled": false}
    }
  }
}
JSON

cat > "$STACK_DIR/openclaw-mcp.template.json" <<'JSON'
{
  "agents": {"defaults": {"skipBootstrap": true, "timeoutSeconds": 600, "model": {"primary": "custom-openai/${LLM_MODEL}"}}},
  "models": {
    "mode": "merge",
    "providers": {
      "custom-openai": {
        "baseUrl": "${LLM_BASE_URL}", "apiKey": "${LLM_API_KEY}",
        "api": "openai-completions", "timeoutSeconds": 300,
        "models": [{
          "id": "${LLM_MODEL}", "name": "${LLM_MODEL}",
          "reasoning": false, "input": ["text"],
          "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
          "contextWindow": 131072, "maxTokens": 8192
        }]
      }
    }
  },
  "gateway": {
    "mode": "local", "bind": "lan", "port": 18789,
    "auth": {"token": "${OPENCLAW_GATEWAY_TOKEN}"},
    "controlUi": {"dangerouslyAllowHostHeaderOriginFallback": true, "dangerouslyDisableDeviceAuth": true},
    "http": {"endpoints": {"responses": {"enabled": true}}}
  },
  "mcp": {
    "servers": {
      "ogmemory": {
        "url": "http://ogmemory:8091/mcp", "transport": "streamable-http",
        "headers": {
          "X-API-Key": "${OGMEM_TENANT_API_KEY}", "X-Account-ID": "${OG_ACCOUNT_ID}",
          "X-User-ID": "${OG_USER_ID}", "X-Agent-ID": "${OG_MCP_AGENT_ID}"
        },
        "connectTimeout": 30, "timeout": 300
      }
    }
  },
  "plugins": {
    "allow": ["og-memory-context-engine"],
    "slots": {"contextEngine": "legacy"},
    "entries": {
      "og-memory-context-engine": {
        "enabled": false,
        "config": {
          "mode": "remote", "memoryApiBaseUrl": "http://ogmemory:8090",
          "authApiKey": "${OGMEM_TENANT_API_KEY}", "authAccountId": "${OG_ACCOUNT_ID}"
        }
      },
      "memory-core": {"enabled": false}
    }
  }
}
JSON

for mode in ce mcp; do
  docker run --rm -i \
    --user 0:0 \
    -v "$STACK_DIR/openclaw-$mode:/target" \
    --entrypoint sh \
    "$OPENCLAW_IMAGE" \
    -c 'cat > /target/openclaw.json && chown node:node /target/openclaw.json && chmod 600 /target/openclaw.json' \
    < "$STACK_DIR/openclaw-$mode.template.json"
done

echo "创建 OpenClaw Context Engine..."
docker run -d \
  --name "$CE_CONTAINER" \
  --label "$OWNER_UID_LABEL=$OWNER_UID" \
  --label "$STACK_DIR_LABEL=$STACK_DIR" \
  --network "$OGMEM_NETWORK" \
  --add-host host.docker.internal:host-gateway \
  --restart unless-stopped \
  -p "${PUBLISH_IP}:${OPENCLAW_CE_HOST_PORT}:18789" \
  -e "LLM_API_KEY=${LLM_API_KEY}" \
  -e "LLM_BASE_URL=${LLM_BASE_URL}" \
  -e "LLM_MODEL=${LLM_MODEL}" \
  -e "OGMEM_TENANT_API_KEY=${OG_TENANT_API_KEY}" \
  -e "OG_ACCOUNT_ID=${OG_ACCOUNT_ID}" \
  -e "OG_USER_ID=${OG_USER_ID}" \
  -e "OG_CE_AGENT_ID=${OG_CE_AGENT_ID}" \
  -e "OPENCLAW_GATEWAY_TOKEN=${CE_GATEWAY_TOKEN}" \
  -e NO_PROXY='*' -e no_proxy='*' \
  -v "$STACK_DIR/openclaw-ce:/home/node/.openclaw" \
  "$OPENCLAW_IMAGE" gateway >/dev/null

echo "创建 OpenClaw MCP..."
docker run -d \
  --name "$MCP_CONTAINER" \
  --label "$OWNER_UID_LABEL=$OWNER_UID" \
  --label "$STACK_DIR_LABEL=$STACK_DIR" \
  --network "$OGMEM_NETWORK" \
  --add-host host.docker.internal:host-gateway \
  --restart unless-stopped \
  -p "${PUBLISH_IP}:${OPENCLAW_MCP_HOST_PORT}:18789" \
  -e "LLM_API_KEY=${LLM_API_KEY}" \
  -e "LLM_BASE_URL=${LLM_BASE_URL}" \
  -e "LLM_MODEL=${LLM_MODEL}" \
  -e "OGMEM_TENANT_API_KEY=${OG_TENANT_API_KEY}" \
  -e "OG_ACCOUNT_ID=${OG_ACCOUNT_ID}" \
  -e "OG_USER_ID=${OG_USER_ID}" \
  -e "OG_MCP_AGENT_ID=${OG_MCP_AGENT_ID}" \
  -e "OPENCLAW_GATEWAY_TOKEN=${MCP_GATEWAY_TOKEN}" \
  -e NO_PROXY='*' -e no_proxy='*' \
  -v "$STACK_DIR/openclaw-mcp:/home/node/.openclaw" \
  "$OPENCLAW_IMAGE" gateway >/dev/null

wait_http \
  "OpenClaw Context Engine" \
  "http://${PUBLISH_IP}:${OPENCLAW_CE_HOST_PORT}/health" \
  "$CE_CONTAINER"
wait_http \
  "OpenClaw MCP" \
  "http://${PUBLISH_IP}:${OPENCLAW_MCP_HOST_PORT}/health" \
  "$MCP_CONTAINER"

docker exec -u node -e HOME=/home/node \
  "$MCP_CONTAINER" \
  openclaw mcp probe ogmemory --json > "$STACK_DIR/mcp-probe.json" || \
  fail "OpenClaw MCP 无法连接 oGMemory"

response_check_file="$(mktemp)"
if ! curl -fsS --max-time 300 -X POST \
  "http://${PUBLISH_IP}:${OPENCLAW_MCP_HOST_PORT}/v1/responses" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${MCP_GATEWAY_TOKEN}" \
  -H 'X-OpenClaw-Agent-ID: main' \
  -H "X-OpenClaw-Session-Key: deploy-check-${INSTANCE_NAME}" \
  -d '{"model":"openclaw","input":"Reply exactly: OK. Do not call tools.","stream":false}' \
  > "$response_check_file"; then
  docker logs --tail 100 "$MCP_CONTAINER" >&2 || true
  rm -f "$response_check_file"
  fail "OpenClaw LLM 验证失败；请检查 LLM Base URL、模型和 API Key"
fi
if ! grep -q '"status"[[:space:]]*:[[:space:]]*"completed"' "$response_check_file"; then
  docker logs --tail 100 "$MCP_CONTAINER" >&2 || true
  rm -f "$response_check_file"
  fail "OpenClaw LLM 未返回完成状态"
fi
rm -f "$response_check_file"
echo "OpenClaw LLM 验证通过"

echo
echo "部署完成"
docker ps --filter "name=${INSTANCE_NAME}_" \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
echo "配置文件: $ENV_FILE"
echo "运行配置: $RUNTIME_ENV_FILE"
echo "数据目录: $STACK_DIR"
echo "MCP 探测结果: $STACK_DIR/mcp-probe.json"
