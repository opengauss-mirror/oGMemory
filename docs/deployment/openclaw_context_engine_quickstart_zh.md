# OpenClaw + oGMemory Context Engine 源码部署教程

本文用于在一台 Linux 服务器上直接运行 OpenClaw 和 oGMemory，不使用 Docker。

运行结构：

```text
OpenClaw Gateway :18789
        |
        | og-memory-context-engine 插件
        | compose / after_turn / compact
        v
oGMemory HTTP :8090
        |
        +-- AGFS :1833        记忆正文
        +-- IndexService      异步索引
        +-- Chroma            本地持久化向量库
```

本文使用：

- OpenClaw `2026.6.9`。这是当前 oGMemory `dev` 分支 `docker/Dockerfile.openclaw` 固定的兼容基线；
- oGMemory [`dev`](https://gitcode.com/opengauss/oGMemory/tree/dev) 分支；
- Python 3.11+；
- Go 1.22+；
- Chroma 本地向量库；
- AGFS 本地文件存储；
- OpenAI 兼容的 LLM 和 Embedding 接口。

OpenClaw 按本文配置仅监听回环地址；当前源码启动器会让 ContextEngine 监听 `0.0.0.0:8090`，并让 AGFS 监听所有网卡的 `1833` 端口。开始部署前，必须通过云安全组或宿主机防火墙禁止外部访问 `8090` 和 `1833`。远程使用建议通过 SSH 隧道，不直接暴露 18789、8090、1833。

# 第一章：拉起 OpenClaw

## 1.1 安装 OpenClaw

OpenClaw 官方安装器会准备兼容的 Node.js。当前官方要求 Node `22.22.3+`、`24.15+` 或 `25.9+`。

安装与 oGMemory 当前基线一致的版本：

```bash
curl -fsSL --proto '=https' --tlsv1.2 https://openclaw.ai/install.sh \
  | bash -s -- --version 2026.6.9 --no-onboard
```

确认安装：

```bash
openclaw --version
node --version
```

OpenClaw 官方参考：

- [安装](https://docs.openclaw.ai/zh-CN/install)
- [Gateway 运维](https://docs.openclaw.ai/gateway)

## 1.2 初始化模型和 Gateway

运行交互式向导：

```bash
openclaw onboard --install-daemon
```

向导中完成：

1. 选择 LLM Provider；
2. 填写 API Key；
3. 选择可用模型；
4. 安装 Linux systemd 用户服务。

无桌面的 Linux 服务器建议允许用户退出后服务继续运行：

```bash
sudo loginctl enable-linger "$(whoami)"
```

## 1.3 验证 OpenClaw

```bash
openclaw doctor
openclaw gateway status --deep --require-rpc
openclaw status
```

健康状态至少应包含：

```text
Runtime: running
Connectivity probe: ok
```

查看日志：

```bash
openclaw logs --follow
```

此时 OpenClaw 已独立运行，但尚未接入 oGMemory。

# 第二章：拉起 oGMemory

本章从源码编译 AGFS，并从源码运行 ContextEngine 和 IndexService。使用 AGFS + Chroma，不依赖 PostgreSQL 或 openGauss。

## 2.1 检查环境

```bash
git --version
python3 --version
go version
make --version
```

最低要求：

```text
Python 3.11+
Go 1.22+
```

Ubuntu / Debian 还需基础编译工具：

```bash
sudo apt-get update
sudo apt-get install -y git curl make build-essential python3-venv python3-dev
```

若系统默认 Python 低于 3.11，先通过系统包、pyenv 或其他受管方式安装 Python 3.11+，再继续。

确认端口空闲：

```bash
sudo ss -lntp | grep -E ':(1833|8090|18789)\b' || true
```

## 2.2 拉取代码

```bash
mkdir -p "$HOME/src"
cd "$HOME/src"
git clone --branch dev --single-branch https://gitcode.com/opengauss/oGMemory.git
cd oGMemory
git rev-parse HEAD
```

记录 commit。排错时必须同时提供 oGMemory commit 和 `openclaw --version`。

后续命令都在仓库根目录执行：

```bash
cd "$HOME/src/oGMemory"
export OGMEM_REPO="$PWD"
```

## 2.3 安装 Python 依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[chroma]"
```

验证 CLI：

```bash
ogmem --version
ogmem --help
```

## 2.4 编译 AGFS

```bash
make -C agfs build
test -x agfs/build/agfs-server
agfs/build/agfs-server --help | head
```

编译结果：

```text
agfs/build/agfs-server
```

## 2.5 配置 oGMemory

复制模板：

```bash
cp config/env.example config/.env
cp config/ogmem.reference.yaml config/ogmem.yaml
chmod 600 config/.env config/ogmem.yaml
```

编辑 `config/.env`：

```bash
vi config/.env
```

示例使用 OpenAI 兼容接口。替换所有示例值：

```dotenv
CONTEXTENGINE_PROVIDER=openai
OGMEM_API_KEY=替换为真实密钥
OGMEM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OGMEM_LLM_MODEL=qwen-plus

EMBEDDING_PROVIDER=openai
OGMEM_EMBEDDING_MODEL=text-embedding-v3

VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIR=.ogmem_data/chroma

AGFS_BASE_URL=http://127.0.0.1:1833
AGFS_DATA_DIR=.ogmem_data/agfs
OGMEM_HTTP_PORT=8090
```

`CONTEXTENGINE_PROVIDER` 必须写 `openai`，即使实际服务商是 DashScope、智谱或其他 OpenAI 兼容服务。服务商差异由 `OGMEM_BASE_URL`、模型名和密钥表达。

编辑 `config/ogmem.yaml`。首次运行建议使用以下关键配置：

```yaml
llm:
  provider: openai
  api_key: null
  base_url: null
  model: null
  temperature: 0.1
  max_tokens: 4096
  json_mode: false

embedding:
  provider: openai
  model: null
  base_url: null
  api_key: null
  multimodal: false

vector_db:
  type: chroma
  chroma_persist_dir: ".ogmem_data/chroma"
  chroma_collection: "contextengine"
  dimension: 1024

service:
  http_port: 8090
  workers: 2

storage:
  backend: agfs

paths:
  data_root: .ogmem_data

agfs:
  base_url: "http://127.0.0.1:1833"
  mount_prefix: "/local/plugin"

index:
  interval: 5
  workers: 1

identity:
  account_id: "acct-demo"
  user_id: "u-alice"
  agent_id: "main"

auth:
  role_control_enabled: false

memory:
  after_turn_threshold: 1
  extraction:
    mode: lazy
  prefetch_enabled: true
  prefetch_top_k: 5
```

`role_control_enabled: false` 只适用于已由安全组或防火墙隔离 `8090`、`1833` 的单机验证环境；不能把该配置直接用于可被其他主机访问的服务器。

说明：

- YAML 中的 `null` 表示从 `config/.env` 读取；
- `storage.backend: agfs` 避免 PostgreSQL 依赖；
- `vector_db.type: chroma` 提供本地持久化向量检索；
- `dimension` 必须等于 Embedding 接口实际返回维度；
- 上例 `text-embedding-v3` 按 1024 维配置；使用其他模型时必须同步修改；
- `after_turn_threshold: 1` 便于教程立即观察记忆写入；正式环境可改为默认的 `400`，或按成本调整。

检查解析后的配置。密钥会被遮蔽：

```bash
ogmem config show
```

## 2.6 启动 oGMemory

启动 AGFS 和 ContextEngine：

```bash
ogmem start local -d
```

启动异步索引服务：

```bash
ogmem start index -d
```

检查状态：

```bash
ogmem status
curl -fsS http://127.0.0.1:1833/api/v1/health
curl -fsS http://127.0.0.1:8090/api/v1/health
```

检查三个进程：

```bash
for f in .ogmem_data/agfs.pid .ogmem_data/ce.pid .ogmem_data/index.pid; do
  printf '%s: ' "$f"
  ps -p "$(cat "$f")" -o pid=,cmd=
done
```

查看日志：

```bash
tail -n 100 .ogmem_data/logs/agfs.log
tail -n 100 .ogmem_data/logs/context_engine.log
tail -n 100 .ogmem_data/logs/index.log
```

停止：

```bash
ogmem stop local
```

`ogmem stop local` 会停止 AGFS、ContextEngine 和 IndexService。

# 第三章：查看 OpenClaw 是否对接 oGMemory

## 3.1 安装本地 Context Engine 插件

确保 oGMemory 已运行：

```bash
curl -fsS http://127.0.0.1:8090/api/v1/health
```

将仓库内插件复制到 OpenClaw 扩展目录：

```bash
cd "$OGMEM_REPO"
npm test --prefix plugins/openclaw

export OGMEM_PLUGIN_DST="$HOME/.openclaw/extensions/og-memory-context-engine"
mkdir -p "$OGMEM_PLUGIN_DST"
cp -a "$OGMEM_REPO/plugins/openclaw/." "$OGMEM_PLUGIN_DST/"
```

OpenClaw `2026.6.9` 不允许 `plugins install --link` 与当前插件包的 Hook 元数据组合使用，因此本教程使用与容器镜像相同的扩展目录加载方式。修改插件源码后，重新执行上述 `cp -a` 并重启 Gateway。

## 3.2 配置 Context Engine 槽位

备份 OpenClaw 配置：

```bash
cp "$HOME/.openclaw/openclaw.json" \
  "$HOME/.openclaw/openclaw.json.bak.$(date +%Y%m%d-%H%M%S)"
```

执行以下脚本。它只合并 oGMemory 插件和 Responses API 配置，不覆盖 `openclaw onboard` 已生成的模型配置：

```bash
python3 - <<'PY'
import json
import secrets
from pathlib import Path

p = Path.home() / ".openclaw" / "openclaw.json"
cfg = json.loads(p.read_text(encoding="utf-8"))

gateway = cfg.setdefault("gateway", {})
gateway.setdefault("mode", "local")
gateway["bind"] = "loopback"
gateway.setdefault("port", 18789)
gateway.setdefault("auth", {}).setdefault("token", secrets.token_hex(32))
gateway.setdefault("http", {}).setdefault("endpoints", {}).setdefault(
    "responses", {}
)["enabled"] = True

plugins = cfg.setdefault("plugins", {})
allowed = plugins.setdefault("allow", [])
if "og-memory-context-engine" not in allowed:
    allowed.append("og-memory-context-engine")
plugins.setdefault("slots", {})["contextEngine"] = "og-memory-context-engine"
entries = plugins.setdefault("entries", {})
entries["og-memory-context-engine"] = {
    "enabled": True,
    "hooks": {"allowConversationAccess": True},
    "config": {
        "mode": "remote",
        "memoryApiBaseUrl": "http://127.0.0.1:8090",
        "accountId": "acct-demo",
        "userId": "u-alice",
        "agentId": "main",
        "prefetchEnabled": True,
        "prefetchTopK": 5,
        "compactTakeoverEnabled": True,
        "summaryMaxChars": 4000,
        "shortTermIndexMode": "sync"
    }
}
entries.setdefault("memory-core", {})["enabled"] = False

p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(p)
PY
```

关键配置必须满足：

```json
{
  "plugins": {
    "allow": [
      "og-memory-context-engine"
    ],
    "slots": {
      "contextEngine": "og-memory-context-engine"
    },
    "entries": {
      "og-memory-context-engine": {
        "enabled": true,
        "config": {
          "mode": "remote",
          "memoryApiBaseUrl": "http://127.0.0.1:8090"
        }
      },
      "memory-core": {
        "enabled": false
      }
    }
  }
}
```

## 3.3 重启并检查

```bash
openclaw doctor
openclaw gateway restart
openclaw gateway status --deep --require-rpc
openclaw plugins list --enabled
openclaw plugins inspect og-memory-context-engine --runtime --json
```

查看 Gateway 日志：

```bash
openclaw logs --follow
```

启动阶段应出现类似内容：

```text
og-memory: remote mode — verifying service connectivity...
og-memory: remote services verified
```

确认生成配置：

```bash
python3 - <<'PY'
import json
from pathlib import Path

cfg = json.loads((Path.home() / ".openclaw/openclaw.json").read_text())
print("slot:", cfg["plugins"]["slots"]["contextEngine"])
print("plugin:", cfg["plugins"]["entries"]["og-memory-context-engine"])
print("memory-core:", cfg["plugins"]["entries"].get("memory-core"))
PY
```

预期：

```text
slot: og-memory-context-engine
memory-core: {'enabled': False}
```

到这里，只证明插件注册和服务连通。下一章验证真实写入、索引和跨会话召回。

# 第四章：查看运行效果

## 4.1 准备测试身份和 Token

保持 OpenClaw Agent ID 为 `main`。不要把 oGMemory 的其他 Agent ID 写进 `X-OpenClaw-Agent-ID`。

Responses API 请求体中的 `user` 字段不会传给 Context Engine 的 `userId`。本测试使用插件支持的 `[ogmem-user: ...]` 标记显式指定身份，确保写入、轮询和跨会话读取使用同一个用户作用域。

```bash
export OGMEM_SMOKE_USER="source-smoke-$(date +%s)"
export OGMEM_WRITE_SESSION="ogmem-write-$OGMEM_SMOKE_USER"
export OGMEM_READ_SESSION="ogmem-read-$OGMEM_SMOKE_USER"

export OPENCLAW_GATEWAY_TOKEN="$(python3 - <<'PY'
import json
from pathlib import Path
cfg = json.loads((Path.home() / ".openclaw/openclaw.json").read_text())
print(cfg["gateway"]["auth"]["token"])
PY
)"
```

## 4.2 写入一条可识别记忆

```bash
curl -fsS -X POST http://127.0.0.1:18789/v1/responses \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $OPENCLAW_GATEWAY_TOKEN" \
  -H 'X-OpenClaw-Agent-ID: main' \
  -H "X-OpenClaw-Session-Key: $OGMEM_WRITE_SESSION" \
  -d "{
    \"model\": \"openclaw\",
    \"input\": \"[ogmem-user: $OGMEM_SMOKE_USER] 请长期记住这个稳定偏好：我的测试项目代号是海鸥-827，我最喜欢的发布窗口是每周四晚上九点。只回复收到。\",
    \"stream\": false
  }" | tee /tmp/ogmem-write-response.json
```

确认 OpenClaw 调用了 Context Engine：

```bash
openclaw logs --follow
```

应看到：

```text
[og-memory] assemble ...
[og-memory] compose result ...
[og-memory] afterTurn called ...
```

`assemble/compose` 表示推理前读取；`afterTurn` 表示推理后回写。

`afterTurn` 日志中的 `userId` 应等于 `$OGMEM_SMOKE_USER`，不能是 `(none)`。

## 4.3 等待记忆完成索引

在 oGMemory 仓库执行：

```bash
cd "$OGMEM_REPO"
tail -f .ogmem_data/logs/context_engine.log .ogmem_data/logs/index.log
```

也可自动轮询 `compose`，最长等待 120 秒：

```bash
for i in $(seq 1 60); do
  BODY="$(curl -fsS -X POST http://127.0.0.1:8090/api/v1/compose \
    -H 'Content-Type: application/json' \
    -d "{
      \"accountId\": \"acct-demo\",
      \"userId\": \"$OGMEM_SMOKE_USER\",
      \"agentId\": \"main\",
      \"sessionId\": \"index-probe-$OGMEM_SMOKE_USER\",
      \"prompt\": \"测试项目代号和发布窗口是什么？\",
      \"messages\": []
    }")"

  if printf '%s' "$BODY" | python3 -c \
    'import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if "海鸥-827" in json.dumps(d, ensure_ascii=False) else 1)'; then
    echo "记忆已可检索"
    break
  fi

  if [ "$i" -eq 60 ]; then
    echo "120 秒内未检索到记忆；检查 ContextEngine 和 IndexService 日志" >&2
    exit 1
  fi
  sleep 2
done
```

## 4.4 新会话召回

使用不同的 `X-OpenClaw-Session-Key`，但在输入中保持相同的 `[ogmem-user: ...]` 标记。这会验证同一用户作用域下的跨会话长期记忆，不只是当前聊天记录。

```bash
curl -fsS -X POST http://127.0.0.1:18789/v1/responses \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $OPENCLAW_GATEWAY_TOKEN" \
  -H 'X-OpenClaw-Agent-ID: main' \
  -H "X-OpenClaw-Session-Key: $OGMEM_READ_SESSION" \
  -d "{
    \"model\": \"openclaw\",
    \"input\": \"[ogmem-user: $OGMEM_SMOKE_USER] 我的测试项目代号是什么？我喜欢什么发布时间？只回答这两个信息。\",
    \"stream\": false
  }" | tee /tmp/ogmem-read-response.json
```

预期回答包含：

```text
海鸥-827
每周四晚上九点
```

## 4.5 查看落盘效果

```bash
cd "$OGMEM_REPO"

# AGFS 记忆正文和元数据
find .ogmem_data/agfs -type f | sort | tail -n 50

# Chroma 持久化索引
find .ogmem_data/chroma -maxdepth 2 -type f | sort

# 服务日志
tail -n 200 .ogmem_data/logs/context_engine.log
tail -n 200 .ogmem_data/logs/index.log
```

最终应同时满足：

1. OpenClaw 日志出现 `assemble`、`compose`、`afterTurn`；
2. ContextEngine 日志没有 `ERROR` 或 `Traceback`；
3. IndexService 进程仍在运行；
4. `.ogmem_data/agfs` 出现记忆文件；
5. `.ogmem_data/chroma` 出现持久化索引；
6. 新 OpenClaw 会话能回答“海鸥-827”和“每周四晚上九点”。

测试完成后清理 shell 中的 Token：

```bash
unset OPENCLAW_GATEWAY_TOKEN
unset OGMEM_SMOKE_USER OGMEM_WRITE_SESSION OGMEM_READ_SESSION
```

## 4.6 常见问题

### OpenClaw 返回 404

检查 Responses API：

```bash
python3 - <<'PY'
import json
from pathlib import Path
cfg = json.loads((Path.home() / ".openclaw/openclaw.json").read_text())
print(cfg.get("gateway", {}).get("http", {}).get("endpoints", {}).get("responses"))
PY
```

应输出：

```text
{'enabled': True}
```

### OpenClaw 返回 400

保持：

```text
X-OpenClaw-Agent-ID: main
```

不要把 oGMemory 的用户 ID 或记忆 Agent ID 写进该 Header。

### 插件已安装，但没有 `[og-memory]` 日志

```bash
openclaw plugins inspect og-memory-context-engine --runtime --json
openclaw doctor
openclaw gateway restart
```

检查槽位是否为 `og-memory-context-engine`，并确认 `memory-core` 已禁用。

### oGMemory 健康检查失败

```bash
cd "$OGMEM_REPO"
tail -n 200 .ogmem_data/logs/agfs.log
tail -n 200 .ogmem_data/logs/context_engine.log
```

重点检查 Python 版本、AGFS 二进制、API Key、Base URL、模型名和端口占用。

### `Unknown provider type: dashscope`

oGMemory 配置应使用：

```yaml
llm:
  provider: openai
```

然后通过 `base_url` 连接 DashScope。不要把 `dashscope` 直接写成 oGMemory 的 LLM provider。

### 对话成功，但长期记忆检索不到

```bash
cd "$OGMEM_REPO"
ps -p "$(cat .ogmem_data/index.pid)" -o pid=,cmd=
grep -E 'ERROR|Traceback|FAILED|PROCESSING' \
  .ogmem_data/logs/context_engine.log .ogmem_data/logs/index.log | tail -n 100
```

再检查：

- Embedding 模型与 `vector_db.dimension` 是否一致；
- 写入和读取是否使用同一 `accountId`、`user`、`agentId`；
- OpenClaw 是否在修改配置后重启；
- IndexService 是否启动；
- 上游 LLM/Embedding API 是否限流或拒绝请求。

## 仓库内参考

- [`plugins/openclaw/README.md`](../../plugins/openclaw/README.md)
- [`plugins/openclaw/ENV.md`](../../plugins/openclaw/ENV.md)
- [`config/ogmem.reference.yaml`](../../config/ogmem.reference.yaml)
- [`config/env.example`](../../config/env.example)
- [`agfs/Makefile`](../../agfs/Makefile)
- [`cli/commands/start.py`](../../cli/commands/start.py)
- [`scripts/run_index_service.py`](../../scripts/run_index_service.py)
