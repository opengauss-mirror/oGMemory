#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'HELP'
Usage: test-locomo.sh (--sample N | --all) [--config PATH]
  --sample N  Test one LoCoMo sample (0-9).
  --all       Test all 10 samples.
  --config PATH  Test config (default: ./locomo.env).
HELP
}
mode=''
sample=''
test_config="$PWD/locomo.env"
while (($#)); do
  case "$1" in
    --sample) [[ -z "$mode" && $# -ge 2 ]] || { usage; exit 2; }; mode=sample; sample="$2"; shift 2 ;;
    --all) [[ -z "$mode" ]] || { usage; exit 2; }; mode=all; shift ;;
    --config) [[ $# -ge 2 ]] || { usage; exit 2; }; test_config="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$mode" ]] || { usage; exit 2; }
if [[ "$mode" == sample ]]; then
  [[ "$sample" =~ ^[0-9]$ ]] || { usage; exit 2; }
  export LOCOMO_SAMPLES="$sample"
else
  export LOCOMO_SAMPLES='0 1 2 3 4 5 6 7 8 9'
fi
trap 'printf "[FAIL] 执行中断（行 %s）；查看终端与运行日志。\n" "$LINENO" >&2' ERR
source "$test_config"
runtime_env="${OGMEMORY_RUNTIME_ENV_FILE:-$HOME/ogmemory-runtime.env}"
source "$runtime_env"
source "$test_config"
printf '[OK] 测试配置：%s\n[OK] 部署环境：%s\n' "$test_config" "$runtime_env"
for name in INSTANCE_NAME STACK_DIR OG_ACCOUNT_ID OG_ROOT_API_KEY MCP_GATEWAY_TOKEN LLM_API_KEY LLM_BASE_URL LLM_MODEL; do
  [[ -n "${!name:-}" ]] || { printf '[FAIL] 配置项 %s 为空\n' "$name" >&2; exit 1; }
done
printf '[OK] 服务身份与模型配置\n'
for command in docker curl sha256sum git; do
  command -v "$command" >/dev/null
  printf '[OK] 命令：%s\n' "$command"
done
docker info >/dev/null
printf '[OK] Docker 服务\n'
for container in "${INSTANCE_NAME}_ogmemory" "${INSTANCE_NAME}_openclaw_mcp"; do
  [[ "$(docker inspect --format '{{.State.Running}}' "$container")" == true ]]
  printf '[OK] 容器运行：%s\n' "$container"
done
docker network inspect "${INSTANCE_NAME}_ogmemory_net" >/dev/null
printf '[OK] Docker 网络\n'
docker exec "${INSTANCE_NAME}_ogmemory" python -c 'import requests, openai, yaml; r=requests.get("http://127.0.0.1:8090/api/v1/health", timeout=15); r.raise_for_status()'
printf '[OK] Python 评测依赖与 oGMemory HTTP 服务\n'

umask 077

export LOCOMO_RUN_ID="$(date +%Y%m%d-%H%M%S)-$$"
export LOCOMO_DIR="$STACK_DIR/locomo/$LOCOMO_RUN_ID"
export LOCOMO_CONTAINER_DIR="/tmp/locomo-$LOCOMO_RUN_ID"
export LOCOMO_MEMORY_CONTAINER="${INSTANCE_NAME}_ogmemory"
export LOCOMO_OPENCLAW_IMAGE="$(docker inspect --format '{{.Image}}' "${INSTANCE_NAME}_openclaw_mcp")"
export LOCOMO_RUNNER_IMAGE="$(docker inspect --format '{{.Image}}' "$LOCOMO_MEMORY_CONTAINER")"
mkdir -p "$LOCOMO_DIR/source" "$LOCOMO_DIR/results"
exec > >(tee "$LOCOMO_DIR/run.log") 2>&1
printf '[RUN] LoCoMO samples=%s 模型=%s\n' "$LOCOMO_SAMPLES" "$LLM_MODEL"

export LOCOMO_JUDGE_BASE_URL="${LOCOMO_JUDGE_BASE_URL:-$LLM_BASE_URL}"
export LOCOMO_JUDGE_MODEL="${LOCOMO_JUDGE_MODEL:-$LLM_MODEL}"
export LOCOMO_JUDGE_API_KEY="${LOCOMO_JUDGE_API_KEY:-$LLM_API_KEY}"
export LOCOMO_JUDGE_API_MODE="${LOCOMO_JUDGE_API_MODE:-chat}"

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$source_root/tests/e2e/eval.py" ]] && \
   git -C "$source_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  source_ref="${LOCOMO_SOURCE_REV:-HEAD}"
  export LOCOMO_SOURCE_REV="$(git -C "$source_root" rev-parse "$source_ref^{commit}")"
elif [[ "${LOCOMO_SOURCE_REV:-dev}" =~ ^[0-9a-f]{40}$ ]]; then
  source_root=''
  source_ref="$LOCOMO_SOURCE_REV"
  export LOCOMO_SOURCE_REV="$source_ref"
else
  source_root=''
  source_ref="${LOCOMO_SOURCE_REV:-dev}"
  export LOCOMO_SOURCE_REV="$(git ls-remote https://gitcode.com/opengauss/oGMemory.git "refs/heads/$source_ref" | awk 'NR==1 {print $1}')"
fi
[[ "$LOCOMO_SOURCE_REV" =~ ^[0-9a-f]{40}$ ]] || { printf '[FAIL] 无法解析评测代码版本\n' >&2; exit 1; }
for file in eval.py agent_plugin_locomo.py run_mcp_locomo.py judge_parallel.py run_openclaw_plugin_locomo.py; do
  mkdir -p "$(dirname "$LOCOMO_DIR/source/$file")"
  if [[ -n "$source_root" ]]; then
    git -C "$source_root" show "$LOCOMO_SOURCE_REV:tests/e2e/$file" > "$LOCOMO_DIR/source/$file"
  else
    curl -fsSL \
      "https://raw.gitcode.com/opengauss/oGMemory/raw/$LOCOMO_SOURCE_REV/tests/e2e/$file" \
      -o "$LOCOMO_DIR/source/$file"
  fi
  printf '[OK] 准备评测程序：%s\n' "$file"
done

curl -fsSL \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o "$LOCOMO_DIR/source/locomo10.json"
sha256sum "$LOCOMO_DIR/source/locomo10.json" > "$LOCOMO_DIR/dataset.sha256"

printf '%s\n' "$LOCOMO_SOURCE_REV" > "$LOCOMO_DIR/source-commit.txt"
docker cp "${INSTANCE_NAME}_openclaw_mcp:/home/node/.openclaw/openclaw.json" \
  "$LOCOMO_DIR/source/openclaw.base.json"
docker exec "$LOCOMO_MEMORY_CONTAINER" mkdir -p "$LOCOMO_CONTAINER_DIR"
docker cp "$LOCOMO_DIR/source/." "$LOCOMO_MEMORY_CONTAINER:$LOCOMO_CONTAINER_DIR/"
memory_owner="$(docker exec "$LOCOMO_MEMORY_CONTAINER" python -c 'import os; print(f"{os.getuid()}:{os.getgid()}")')"
docker exec --user 0 "$LOCOMO_MEMORY_CONTAINER" chown -R "$memory_owner" "$LOCOMO_CONTAINER_DIR"
printf '[OK] 容器内评测文件归属：%s\n' "$memory_owner"

docker exec -i -w "$LOCOMO_CONTAINER_DIR" -e LOCOMO_CONTAINER_DIR -e LOCOMO_SAMPLES "$LOCOMO_MEMORY_CONTAINER" python - <<'PY'
import os
from pathlib import Path
from eval import build_session_messages, load_dataset
root = Path(os.environ['LOCOMO_CONTAINER_DIR'])
data = load_dataset(root/'locomo10.json', dataset_format='locomo')
assert isinstance(data, list) and len(data) == 10, 'LoCoMO 数据应包含 10 个 sample'
for sample in map(int, os.environ['LOCOMO_SAMPLES'].split()):
    item = data[sample]
    sessions = build_session_messages(item)
    questions = item['questions']
    assert sessions and questions, f'sample {sample} 对话或题目为空'
    print(f'[OK] 数据 sample {sample}：{len(sessions)} sessions，{len(questions)} questions')
PY

run_locomo() (
  set -Eeuo pipefail
  printf '\n[RUN] 启动测试 Gateway\n'
  export LOCOMO_TEST_USER="locomo-$LOCOMO_RUN_ID"
  export LOCOMO_TEST_AGENT="$LOCOMO_TEST_USER-agent"
  export LOCOMO_TEST_GATEWAY="locomo-$LOCOMO_RUN_ID"
  export LOCOMO_GATEWAY_URL="http://$LOCOMO_TEST_GATEWAY:18789"
  export OPENCLAW_GATEWAY_TOKEN="$MCP_GATEWAY_TOKEN"
  export OGMEM_API_KEY="$LOCOMO_JUDGE_API_KEY"
  sample_dir="$LOCOMO_DIR/results"
  result_dir="$LOCOMO_CONTAINER_DIR/results"
  mkdir -p "$sample_dir/state"
  export OGMEM_LOCOMO_OGMEMORY_API_KEY="$OG_ROOT_API_KEY"

  cleanup() {
    docker logs "$LOCOMO_TEST_GATEWAY" > "$sample_dir/gateway.log" 2>&1 || true
    docker rm -f "$LOCOMO_TEST_GATEWAY" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT

  docker exec -i \
    -e LOCOMO_CONTAINER_DIR -e LOCOMO_TEST_USER -e LOCOMO_TEST_AGENT \
    -e OG_ACCOUNT_ID -e OG_ROOT_API_KEY -e MCP_GATEWAY_TOKEN \
    -e LLM_API_KEY -e LLM_BASE_URL -e LLM_MODEL \
    "$LOCOMO_MEMORY_CONTAINER" python - <<'PY' > "$sample_dir/state/openclaw.json"
import json, os
from pathlib import Path

def expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value

base = Path(os.environ['LOCOMO_CONTAINER_DIR']) / 'openclaw.base.json'
cfg = expand(json.loads(base.read_text()))
cfg['gateway']['port'] = 18789
cfg['gateway']['auth'] = {'mode': 'token', 'token': os.environ['MCP_GATEWAY_TOKEN']}
cfg['agents']['defaults']['workspace'] = '/tmp/ogmem_ws'
cfg['mcp']['servers']['ogmemory']['headers'] = {
    'X-API-Key': os.environ['OG_ROOT_API_KEY'],
    'X-Account-ID': os.environ['OG_ACCOUNT_ID'],
    'X-User-ID': os.environ['LOCOMO_TEST_USER'],
    'X-Agent-ID': os.environ['LOCOMO_TEST_AGENT'],
}
cfg['mcp']['servers']['ogmemory']['url'] = 'http://ogmemory:8091/mcp'
cfg['mcp']['servers']['ogmemory']['timeout'] = 1200
for entry in cfg.get('plugins', {}).get('entries', {}).values():
    settings = entry.get('config', {})
    if 'authApiKey' in settings:
        settings['authApiKey'] = os.environ['OG_ROOT_API_KEY']
    if 'authAccountId' in settings:
        settings['authAccountId'] = os.environ['OG_ACCOUNT_ID']
print(json.dumps(cfg, ensure_ascii=False, indent=2))
PY

  docker run -d --name "$LOCOMO_TEST_GATEWAY" \
    --network "${INSTANCE_NAME}_ogmemory_net" \
    --add-host host.docker.internal:host-gateway \
    -e LLM_API_KEY -e LLM_BASE_URL -e LLM_MODEL \
    -e OPENCLAW_GATEWAY_TOKEN \
    -v "$sample_dir/state:/home/node/.openclaw" \
    "$LOCOMO_OPENCLAW_IMAGE" gateway > "$sample_dir/container-id.txt"

  docker exec -i -e LOCOMO_GATEWAY_URL "$LOCOMO_MEMORY_CONTAINER" python -u - <<'PY'
import os, time, requests
url = os.environ['LOCOMO_GATEWAY_URL'] + '/health'
for _ in range(120):
    try:
        if requests.get(url, timeout=2).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise SystemExit('测试 Gateway 启动失败')
PY

  selection=()
  if [[ "$mode" == sample ]]; then selection=(--sample "$sample"); fi
  run_eval() {
    docker run --rm --network "${INSTANCE_NAME}_ogmemory_net" \
      --add-host host.docker.internal:host-gateway \
      --user "$(id -u):$(id -g)" --entrypoint python \
      -e OPENCLAW_GATEWAY_TOKEN -e OGMEM_API_KEY -e OGMEM_LOCOMO_OGMEMORY_API_KEY \
      -v "$LOCOMO_DIR/source:$LOCOMO_CONTAINER_DIR/source:ro" \
      -v "$sample_dir:$result_dir" \
      -v "$sample_dir/state:$LOCOMO_CONTAINER_DIR/gateway-state:ro" \
      "$LOCOMO_RUNNER_IMAGE" -u "$LOCOMO_CONTAINER_DIR/source/run_mcp_locomo.py" "$@"
  }
  for phase in ingest qa; do
    output="$result_dir/qa"
    if [[ "$phase" == ingest ]]; then output="$result_dir/ingest.txt"; fi
    printf '[RUN] %s\n' "$phase"
    run_eval \
      "$phase" "$LOCOMO_CONTAINER_DIR/source/locomo10.json" --transport gateway \
      "${selection[@]}" --output "$output" --output-dir "$result_dir/logs/$phase" \
      --run-id "$LOCOMO_RUN_ID" --ingest-record "$result_dir/ingest-record.json" \
      --base-url "$LOCOMO_GATEWAY_URL" --user "$LOCOMO_TEST_USER" --parallel 2 \
      --account-id "$OG_ACCOUNT_ID" --memory-agent-id "$LOCOMO_TEST_AGENT" \
      --ogmemory-url http://ogmemory:8090 --verify-writes \
      --gateway-state-dir "$LOCOMO_CONTAINER_DIR/gateway-state"
  done

  printf '[RUN] judge\n'
  run_eval \
    judge "$result_dir/qa.csv" --output "$result_dir/qa_judged.csv" \
    --output-dir "$result_dir/logs/judge" \
    --api-base "$LOCOMO_JUDGE_BASE_URL" --judge-model "$LOCOMO_JUDGE_MODEL" \
    --judge-profile locomo \
    --judge-api-mode "$LOCOMO_JUDGE_API_MODE"

  printf '[RUN] stat\n'
  run_eval \
    stat "$result_dir/qa_judged.csv" --output "$result_dir/summary.json" \
    --output-dir "$result_dir/logs/stat"
)

run_locomo
printf '结果目录：%s\n' "$LOCOMO_DIR/results"
