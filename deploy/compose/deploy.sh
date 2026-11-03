#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
cd "$ROOT"
fail() { printf '%s\n' "部署失败：$*" >&2; exit 1; }
helper_mounts() {
  data_root="$(cd "$data_root" && pwd -P)"
  mounts=(-v "$ROOT:$ROOT")
  if [[ "$data_root" != "$ROOT" ]]; then mounts+=(-v "$data_root:$data_root"); fi
}
trap 'printf "%s\n" "部署失败：步骤未完成；请查看上方阶段及本实例日志。" >&2' ERR
command -v docker >/dev/null || fail '请安装 Docker。'
docker info >/dev/null 2>&1 || fail 'Docker 未运行。'
docker compose version >/dev/null || fail '请安装 Docker Compose 插件。'
[[ -f .env ]] || fail '请先复制 .env.example 为 .env 并填写配置。'
# Lock before reading generated configuration, and hold through the entire lifecycle.
# Do not unlink lock files: waiters must continue sharing the same inode.
case "${1:-start}" in
  start|stop|down)
    (umask 077; touch .ogmemory-deploy.lock)
    exec 9>>.ogmemory-deploy.lock
    if command -v flock >/dev/null; then
      flock -x 9
    elif command -v python3 >/dev/null; then
      python3 -c 'import fcntl; fcntl.flock(9, fcntl.LOCK_EX)'
    else
      fail '启动管理需要 flock（Linux util-linux）或 Python 3（macOS）。'
    fi
    ;;
esac
chmod 600 .env
# Keep Docker client connectivity/authentication, but isolate deployment inputs.
# All model, identity and custom dotenv references come from this instance's files.
compose=(env -i)
for name in PATH HOME TMPDIR DOCKER_CONFIG DOCKER_HOST DOCKER_CONTEXT DOCKER_TLS_VERIFY DOCKER_CERT_PATH DOCKER_API_VERSION XDG_RUNTIME_DIR SSH_AUTH_SOCK SSL_CERT_FILE SSL_CERT_DIR HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  case "$name" in
    HTTP_PROXY|HTTPS_PROXY|NO_PROXY|http_proxy|https_proxy|no_proxy)
      # A proxy explicitly declared in this instance takes precedence over shell transport settings.
      if grep -Eq "^[[:space:]]*(export[[:space:]]+)?${name}[[:space:]]*=" .env; then continue; fi
      ;;
  esac
  if [[ -n "${!name:-}" ]]; then compose+=("$name=${!name}"); fi
done
compose+=(docker compose --env-file .env)
if [[ -f .env.compose ]]; then compose+=(--env-file .env.compose); fi
compose+=(--project-directory "$ROOT")
command -v python3 >/dev/null || fail '部署管理需要 Python 3 解析结构化配置。'
# Compose preserves multiline interpolation in JSON; never log this output.
resolved_env=$("${compose[@]}" config --format json)
value() { printf '%s' "$resolved_env" | python3 -c 'import json,sys; m=json.load(sys.stdin); print(m.get("name", "") if sys.argv[1]=="COMPOSE_PROJECT_NAME" else m.get("x-deployment-inputs",{}).get(sys.argv[1],"").replace("$$","$"))' "$1"; }
image="$(value OGMEM_IMAGE)"
image="${image:-swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:1013}"
project="$(value COMPOSE_PROJECT_NAME)"
project="${project:-ogmemory}"
# Docker container names are atomic and shared by every client of this daemon.
case "${1:-start}" in
  start|stop|down)
    lock_name="ogmemory-deploy-guard-$project"
    lock_image="$image"
    lock_create=(docker create --name "$lock_name" --network none)
    if [[ "${1:-start}" == start ]]; then
      lock_create+=(--platform linux/arm64)
      if ! docker image inspect "$lock_image" >/dev/null 2>&1; then
        docker pull "$lock_image" >/dev/null || fail '项目互斥保护需要指定的 oGMemory 镜像。'
      fi
    else
      # Stopping an existing service must not depend on a future target image.
      lock_image=''
      existing_ids=$(docker ps -aq --filter "label=com.docker.compose.project=$project")
      for id in $existing_ids; do
        lock_image=$(docker inspect "$id" --format '{{.Image}}')
        [[ -z "$lock_image" ]] || break
      done
      if [[ -z "$lock_image" ]]; then
        lock_image=$(docker image ls -q --no-trunc | awk 'NR==1 {print}')
      fi
      if [[ -z "$lock_image" ]]; then
        existing_networks=$(docker network ls -q --filter "label=com.docker.compose.project=$project")
        [[ -z "$existing_networks" ]] || fail '没有缓存镜像可建立项目锁；未修改现有网络。'
        printf '%s\n' '本项目没有服务或网络需要停止或删除。'
        exit 0
      fi
    fi
    if ! lock_id=$("${lock_create[@]}" \
        --label "ogmemory.deploy.owner=$ROOT" --entrypoint /bin/true "$lock_image" 2>/dev/null); then
      fail "Docker 项目 $project 正在由其他启动管理操作占用(${lock_name})，请核对后重试。"
    fi
    [[ -n "$lock_id" ]] || fail '项目锁未返回容器 ID，拒绝继续。'
    release_project_lock() {
      status=$?
      trap - EXIT
      docker rm "$lock_id" >/dev/null 2>&1 || printf '%s\n' "项目锁 $lock_name 未释放，请核对后处理。" >&2
      exit "$status"
    }
    trap release_project_lock EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    ;;
esac
# Never adopt another directory's Compose project, even if its name matches.
    ids=$(docker ps -aq --filter "label=com.docker.compose.project=$project")
    for id in $ids; do
      owner=$(docker inspect "$id" --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}')
      [[ "$owner" == "$ROOT" ]] || fail '项目名已属于另一个部署目录，请修改 COMPOSE_PROJECT_NAME。'
    done
case "${1:-start}" in
  start)
    # Protect both still-running legacy services and named volumes left after down.
    legacy_volumes=$(docker volume ls -q --filter "label=com.docker.compose.project=$project")
    legacy_named=$(docker volume ls -q --filter "name=^${project}_(opengauss-data|agfs-data)$")
    [[ -z "$legacy_volumes$legacy_named" ]] || fail '检测到旧存储命名卷，禁止自动切换。请保留原部署和数据，使用独立项目名创建新实例。'
    legacy_containers=$(docker ps -aq --filter "name=^/${project}_(opengauss|ogmemory)$")
    [[ -z "$legacy_containers" ]] || fail '检测到 quickstart 旧存储实例，禁止自动接管；请保留原部署，使用独立项目名创建新实例。'
    stack_dir=$(value STACK_DIR)
    if [[ -n "$stack_dir" ]]; then
      requested_root=$(value DATA_ROOT)
      requested_root="${requested_root:-./data}"
      [[ "$requested_root" == /* ]] || requested_root="$ROOT/$requested_root"
      [[ -d "$requested_root" ]] || fail '旧存储位置与本次 DATA_ROOT 不一致，禁止自动切换。'
      requested_root=$(cd "$requested_root" && pwd -P)
      [[ "$stack_dir" == "$requested_root/$project" ]] || fail '旧存储位置与本次 DATA_ROOT/项目名不一致，禁止自动切换。'
    fi
    for id in $ids; do
      mounts=$(docker inspect "$id" --format '{{range .Mounts}}{{.Type}}|{{.Source}}|{{.Destination}}{{println}}{{end}}')
      while IFS='|' read -r mount_type mount_source mount_target; do
        case "$mount_target" in
          /var/lib/opengauss) expected=opengauss ;;
          /data/agfs) expected=agfs ;;
          *) continue ;;
        esac
        [[ "$mount_type" == bind && -n "$stack_dir" && "$mount_source" == "$stack_dir/$expected" ]] || fail '检测到旧存储或数据挂载位置变化，禁止自动切换。请先停机备份并核对原配置。'
      done <<< "$mounts"
    done
    own_ports=$(docker ps --filter "label=com.docker.compose.project=$project" --format '{{.Ports}}')
    for key in OGMEM_HTTP_HOST_PORT OGMEM_MCP_HOST_PORT; do
      port=$(value "$key")
      [[ -n "$port" ]] || { [[ "$key" == OGMEM_HTTP_HOST_PORT ]] && port=18090 || port=18091; }
      [[ "$port" =~ ^[0-9]+$ ]] || fail '宿主端口必须是数字。'
      if [[ "$own_ports" == *":$port->"* ]]; then continue; fi
      if command -v ss >/dev/null; then
        [[ $(ss -H -ltn "sport = :$port" | wc -l) -eq 0 ]] || fail "宿主端口 $port 已占用。"
      elif command -v lsof >/dev/null; then
        ! lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 || fail "宿主端口 $port 已占用。"
      else fail '端口检查需要 ss（Linux）或 lsof（macOS）。'; fi
      other_ports=$(docker ps --format '{{.Ports}}')
      [[ "$other_ports" != *":$port->"* ]] || fail "Docker 已占用宿主端口 $port。"
    done
    printf '%s\n' '阶段 1/5：拉取服务镜像'
    "${compose[@]}" pull
    printf '%s\n' '阶段 2/5：准备配置和实例凭据'
    data_root="$(value DATA_ROOT)"
    data_root="${data_root:-./data}"
    [[ "$data_root" == /* ]] || data_root="$ROOT/$data_root"
    mkdir -p "$data_root"
    helper_mounts
    printf '%s\n' "$resolved_env" | docker run --rm -i --user 0:0 --entrypoint python3 -e "DEPLOY_DATA_ROOT=$data_root" -e "DEPLOY_UID=$(id -u)" -e "DEPLOY_GID=$(id -g)" \
      "${mounts[@]}" -w "$ROOT" "$image" "$ROOT/deploy/compose/scripts/prepare.py"
    compose+=(--env-file .env.compose)
    printf '%s\n' '阶段 3/5：启动服务并等待健康检查'
    "${compose[@]}" up -d --wait --wait-timeout 900 opengauss
    "${compose[@]}" up -d --no-deps --force-recreate --wait --wait-timeout 900 ogmemory
    printf '%s\n' '阶段 4/5：初始化用户和 Agent'
    "${compose[@]}" exec -T ogmemory python3 /opt/ogmem-deploy/runtime.py bootstrap
    printf '%s\n' '阶段 5/5：服务就绪'
    "${compose[@]}" ps
    "${compose[@]}" exec -T ogmemory python3 /opt/ogmem-deploy/runtime.py connection
    ;;
  status) "${compose[@]}" ps -a ;;
  logs)
    [[ -f .env.compose ]] || fail '请先执行 start。'
    data_root="$(value DATA_ROOT)"; data_root="${data_root:-./data}"
    [[ "$data_root" == /* ]] || data_root="$ROOT/$data_root"
    helper_mounts
    "${compose[@]}" logs --tail 100 ogmemory opengauss | docker run --rm -i --user 0:0 --entrypoint python3 \
      -e "DEPLOY_HTTP_PROXY=$(value HTTP_PROXY)" -e "DEPLOY_HTTPS_PROXY=$(value HTTPS_PROXY)" \
      "${mounts[@]}" "$image" "$ROOT/deploy/compose/scripts/redact.py" "$data_root/$project/ogmemory.yaml"
    ;;
  stop) "${compose[@]}" stop ;;
  down) "${compose[@]}" down ;;
  *) fail '用法：deploy.sh start|status|logs|stop|down' ;;
esac
