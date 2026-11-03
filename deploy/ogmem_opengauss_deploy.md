# oGMemory + openGauss + OpenClaw 部署说明

## 环境与版本要求

| 配置项 | 要求 |
| --- | --- |
| 服务器架构 | ARM64（鲲鹏） |
| Docker | 已安装 Docker Engine，且当前用户具有使用权限 |
| 操作系统 | 支持 Docker 的 Linux 发行版 |
| 用户权限 | 可执行 `docker` 命令，并可读写持久化目录 |

部署前，执行以下命令检查 Docker 版本和服务状态：

```bash
docker --version
docker info
```

> 部署镜像仅支持 ARM64 架构，不支持 x86_64。

## 镜像汇总

| 组件 | 使用场景 | 镜像 |
| --- | --- | --- |
| openGauss | 所有部署方式 | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed:0328` |
| oGMemory | 所有部署方式 | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:1013` |
| OpenClaw | Context Engine / MCP 部署 | `swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/openclaw-ogmem:poc_0903` |

按以下顺序完成部署：

1. 部署 openGauss。
2. 部署 oGMemory。
3. 部署并验证 OpenClaw。

## 一键快速安装部署

如需一键完成 openGauss、oGMemory、Context Engine 和 MCP 容器的安装与验证，请参考 [oGMemory 一键部署与体验指南](https://gitcode.com/opengauss/oGMemory/tree/dev/deploy/user-quickstart)。

## 第一部分：openGauss

### 1. 创建数据库目录

创建宿主机数据目录并配置写权限。openGauss 数据将持久化至 `/your_directory/opengauss`：

```bash
mkdir -p /your_directory/opengauss
chmod 777 /your_directory/opengauss
```

### 2. 启动 openGauss 容器

执行以下命令启动 openGauss 容器：

```bash
docker run --name opengauss --network host --privileged=true -d \
  -e 'GS_PASSWORD=<OPENGAUSS_PASSWORD>' \
  -e 'GS_PORT=<OPENGAUSS_PORT>' \
  -v /your_directory/opengauss:/var/lib/opengauss \
  swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/opengauss-distributed:0328
```

参数说明：

- `<OPENGAUSS_PASSWORD>`：设置 openGauss 数据库密码，并在 `ogmemory.yaml` 中使用相同值。
- `<OPENGAUSS_PORT>`：设置 openGauss 监听端口，例如 `5432` 或 `15432`。若宿主机的 `5432` 端口已被占用，请使用其他空闲端口。


## 第二部分：oGMemory

### 1. 创建 oGMemory 数据目录

按下表规划 oGMemory 配置文件和 AGFS 数据目录：

| 宿主机路径 | 容器路径 | 用途 |
| --- | --- | --- |
| `/your_directory/ogmemory.yaml` | `/etc/ogmem/config.yaml` | oGMemory 配置文件 |
| `/your_directory/ogmem` | `/data/agfs` | AGFS 记忆数据目录 |

首次部署时创建 AGFS 数据目录并赋予写权限：

```bash
mkdir -p /your_directory/ogmem
chmod 777 /your_directory/ogmem
```

### 2. 准备配置文件

执行以下命令直接创建完整配置文件。将尖括号中的占位符替换为实际值。其中，`<OPENGAUSS_PORT>` 和 `<OPENGAUSS_PASSWORD>` 必须与第一部分启动 openGauss 时使用的 `GS_PORT` 和 `GS_PASSWORD` 一致：

```bash
cat > /your_directory/ogmemory.yaml <<'YAML'
llm:
  provider: openai
  api_key: "<YOUR_LLM_API_KEY>"
  base_url: "<YOUR_LLM_BASE_URL>"
  model: "<YOUR_LLM_MODEL>"
  temperature: 0.7
  max_tokens: 4096

embedding:
  provider: openai
  api_key: "<YOUR_EMBEDDING_API_KEY>"
  base_url: "<YOUR_EMBEDDING_BASE_URL>"
  model: "<YOUR_EMBEDDING_MODEL>"
  multimodal: false

vector_db:
  type: opengauss
  connection_string: "host=127.0.0.1 port=<OPENGAUSS_PORT> dbname=postgres user=gaussdb password=<OPENGAUSS_PASSWORD>"
  dimension: <EMBEDDING_DIMENSION>
  table_name: vector_index
  pool_size: 5

service:
  http_port: 8090
  workers: 1
  threads: 2

mcp:
  enabled: true
  host: "127.0.0.1"
  port: 8091
  path: "/mcp"
  default_toolset: "all"
  enable_write_tools: true
  enable_delete_tools: false
  allowed_hosts:
    - "127.0.0.1"
    - "localhost"

agfs:
  base_url: "http://127.0.0.1:1833"
  mount_prefix: "/local/plugin"

index:
  interval: 15
  workers: 1

identity:
  account_id: "acct-demo"
  user_id: "u-alice"
  agent_id: "main"

auth:
  role_control_enabled: true
  root_api_key: "<OG_ROOT_API_KEY>"
  admin_api_keys:
    - "acct-demo:<OG_TENANT_API_KEY>"

sharing:
  agent_shared_mode: "off"

cache:
  enabled: true
  max_size: 1000
YAML
```

配置要求：

- `<YOUR_LLM_API_KEY>`、`<YOUR_LLM_BASE_URL>` 和 `<YOUR_LLM_MODEL>` 分别对应 LLM 服务的 API Key、OpenAI-compatible Base URL 和模型名。
- `<YOUR_EMBEDDING_API_KEY>`、`<YOUR_EMBEDDING_BASE_URL>` 和 `<YOUR_EMBEDDING_MODEL>` 分别对应 Embedding 服务的 API Key、Base URL 和模型名。若 Embedding 与 LLM 共用服务，可使用相同的 API Key 和 Base URL。
- `<OPENGAUSS_PORT>` 和 `<OPENGAUSS_PASSWORD>` 必须与第一部分的 `GS_PORT` 和 `GS_PASSWORD` 一致。
- `<EMBEDDING_DIMENSION>` 必须与 Embedding 模型的输出维度一致，例如 `1024`。

### 3. 启动 oGMemory 容器

```bash
docker run -d --name ogmem --network host \
  -e OGMEM_CONFIG=/etc/ogmem/config.yaml \
  -v /your_directory/ogmemory.yaml:/etc/ogmem/config.yaml:ro \
  -v /your_directory/ogmem:/data/agfs \
  swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/ogmemory:1013
```

参数说明：

- `--network host` 使 oGMemory 直接使用宿主机网络。
- oGMemory HTTP API 默认监听宿主机 `8090` 端口。
- 配置文件以只读方式挂载到容器的 `/etc/ogmem/config.yaml`。
- 记忆数据写入宿主机目录 `/your_directory/ogmem`。


### 4. 验证 oGMemory

执行以下命令检查容器状态：

```bash
docker ps --filter name=ogmem
```

若容器状态异常，执行以下命令检查启动日志：

```bash
docker logs -f ogmem
```

执行健康检查：

```bash
curl http://127.0.0.1:8090/api/v1/health
```

修改配置后，重启容器以使配置生效：

```bash
docker restart ogmem
```

### 5. 打开管理界面

在浏览器中访问 `http://<服务器IP>:8090/memory-viewer/`，将服务器 IP 和端口替换为实际部署值。在 **连接设置** 中填写部署时配置的 oGMemory API Key，点击 **连接**，即可通过左侧 **记忆** 查看目录和内容。

页面操作和常见问题见 [管理界面使用指南](../docs/memory_viewer_guide.md)。

## 第三部分：OpenClaw

启动 OpenClaw 前，确保 openGauss 和 oGMemory 已完成部署。

> OpenClaw 部署请使用“镜像汇总”中列出的镜像。

### 1. 准备 OpenClaw 配置

设置工作目录、模型、认证和容器参数。将占位值替换为实际值：

```bash
export OGMEM_STACK='/path/to/ogmemory-stack'
mkdir -p "$OGMEM_STACK"
export LLM_PROVIDER='openai'
export LLM_API_KEY='替换为真实 API Key'
export LLM_BASE_URL='替换为 OpenAI-compatible 地址'
export LLM_MODEL='替换为模型名'
export OG_ROOT_API_KEY='替换为 oGMemory Root Key'
export OG_TENANT_API_KEY='替换为 acct-demo 租户级 Admin Key'
export OG_ACCOUNT_ID='acct-demo'
export OG_USER_ID='ce-user'
export OG_CE_AGENT_ID='ce-agent'
export OG_MCP_AGENT_ID='mcp-agent'
export OPENCLAW_IMAGE='swr.cn-north-4.myhuaweicloud.com/kunpeng-ai/openclaw-ogmem:poc_0903'
export OPENCLAW_CONTAINER_NAME='openclaw_ce'
export OPENCLAW_HOME_DIR="$OGMEM_STACK/openclaw-ce"
export GATEWAY_PORT='18790'
export OPENCLAW_GATEWAY_TOKEN='替换为 CE Gateway Token'
export MCP_GATEWAY_TOKEN='替换为 MCP Gateway Token'
export OGMEM_URL='http://127.0.0.1:8090'
export OG_AUTH_API_KEY="$OG_TENANT_API_KEY"
export OG_AUTH_ACCOUNT_ID="$OG_ACCOUNT_ID"
```

Root Key 具有跨租户最高权限，仅限用于管理、初始化和紧急诊断。生产环境中，禁止向 Context Engine 或 MCP Agent 配置 Root Key。请使用与 `OG_AUTH_ACCOUNT_ID` 或 `X-Account-ID` 匹配的租户级 Admin/Member Key。上述配置以 `acct-demo` 的 Admin Key 为例。仅在本机临时调试时，可将 `OG_TENANT_API_KEY` 临时设置为 Root Key。

### 2. 配置 Context Engine

执行以下脚本生成 Context Engine 版 OpenClaw 配置：

```bash
mkdir -p "$OPENCLAW_HOME_DIR"
chmod 755 "$OPENCLAW_HOME_DIR"

python3 - <<'PY'
import json
import os
from pathlib import Path

cfg = {
    "agents": {"defaults": {"skipBootstrap": True, "timeoutSeconds": 600,
                             "model": {"primary": f"custom-openai/{os.environ['LLM_MODEL']}"}},
               "list": [{"id": os.environ["OG_CE_AGENT_ID"], "default": True,
                         "workspace": "/home/node/.openclaw/workspace"}]},
    "models": {"mode": "merge", "providers": {"custom-openai": {
        "baseUrl": os.environ["LLM_BASE_URL"], "apiKey": os.environ["LLM_API_KEY"],
        "api": "openai-completions", "timeoutSeconds": 300,
        "models": [{"id": os.environ["LLM_MODEL"], "name": os.environ["LLM_MODEL"], "reasoning": False,
                    "input": ["text"], "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 131072, "maxTokens": 8192}],
    }}},
    "gateway": {"mode": "local", "bind": "lan", "port": int(os.environ["GATEWAY_PORT"]),
                "auth": {"token": os.environ["OPENCLAW_GATEWAY_TOKEN"]},
                "controlUi": {"dangerouslyAllowHostHeaderOriginFallback": True,
                              "dangerouslyDisableDeviceAuth": True},
                "http": {"endpoints": {"responses": {"enabled": True}}}},
    "plugins": {"allow": ["og-memory-context-engine"],
                "slots": {"contextEngine": "og-memory-context-engine"},
                "entries": {"og-memory-context-engine": {"enabled": True,
                    "hooks": {"allowConversationAccess": True},
                    "config": {"mode": "remote", "memoryApiBaseUrl": os.environ["OGMEM_URL"],
                               "authApiKey": os.environ["OG_TENANT_API_KEY"],
                               "authAccountId": os.environ["OG_ACCOUNT_ID"],
                               "accountId": os.environ["OG_ACCOUNT_ID"],
                               "userId": os.environ["OG_USER_ID"],
                               "agentId": os.environ["OG_CE_AGENT_ID"], "prefetchEnabled": True,
                               "prefetchTopK": 5, "compactTakeoverEnabled": True,
                               "shortTermIndexMode": "sync"}},
                    "memory-core": {"enabled": False}}},
}

out = Path(os.environ["OPENCLAW_HOME_DIR"]) / "openclaw.template.json"
rendered = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
if "${" in rendered:
    raise SystemExit("OpenClaw 配置中存在未替换的环境变量")
out.write_text(rendered, encoding="utf-8")
out.chmod(0o600)
print(out)
PY

docker run --rm -i \
  --user 0:0 \
  -v "$OPENCLAW_HOME_DIR:/target" \
  --entrypoint sh \
  "$OPENCLAW_IMAGE" \
  -c 'cat > /target/openclaw.json && chown node:node /target/openclaw.json && chmod 600 /target/openclaw.json' \
  < "$OPENCLAW_HOME_DIR/openclaw.template.json"

rm -f "$OPENCLAW_HOME_DIR/openclaw.template.json"
```

### 3. 启动 Context Engine

执行以下命令启动 Context Engine 容器：

```bash
mkdir -p "$OPENCLAW_HOME_DIR"

docker run -d \
  --name "$OPENCLAW_CONTAINER_NAME" \
  --network host \
  --label ogmem.managed=true \
  -v "$OPENCLAW_HOME_DIR:/home/node/.openclaw" \
  -e "LLM_PROVIDER=$LLM_PROVIDER" \
  -e "LLM_API_KEY=$LLM_API_KEY" \
  -e "LLM_BASE_URL=$LLM_BASE_URL" \
  -e "LLM_MODEL=$LLM_MODEL" \
  -e "GATEWAY_PORT=$GATEWAY_PORT" \
  -e "OGMEM_URL=$OGMEM_URL" \
  -e "OG_AUTH_API_KEY=$OG_AUTH_API_KEY" \
  -e "OG_AUTH_ACCOUNT_ID=$OG_AUTH_ACCOUNT_ID" \
  -e "OG_TENANT_API_KEY=$OG_TENANT_API_KEY" \
  -e "OG_ACCOUNT_ID=$OG_ACCOUNT_ID" \
  -e "OG_USER_ID=$OG_USER_ID" \
  -e "OG_CE_AGENT_ID=$OG_CE_AGENT_ID" \
  -e "OPENCLAW_GATEWAY_TOKEN=$OPENCLAW_GATEWAY_TOKEN" \
  -e 'NO_PROXY=*' \
  -e 'no_proxy=*' \
  "$OPENCLAW_IMAGE" gateway
```

检查容器和 Gateway 状态：

```bash
docker ps --filter "name=$OPENCLAW_CONTAINER_NAME"
curl -fsS "http://127.0.0.1:${GATEWAY_PORT}/health"
```

### 4. 启动 MCP

为 MCP 版 OpenClaw 创建独立配置目录。MCP 与 Context Engine 不得共用 OpenClaw HOME：

```bash
mkdir -p "$OGMEM_STACK/openclaw-mcp"
chmod 755 "$OGMEM_STACK/openclaw-mcp"
```

执行以下脚本生成 OpenClaw 配置：

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

cfg = {
    "agents": {
        "defaults": {
            "skipBootstrap": True,
            "timeoutSeconds": 600,
            "model": {"primary": f"custom-openai/{os.environ['LLM_MODEL']}"},
        },
        "list": [{
            "id": os.environ["OG_MCP_AGENT_ID"],
            "default": True,
            "workspace": "/home/node/.openclaw/workspace",
        }],
    },
    "models": {
        "mode": "merge",
        "providers": {
            "custom-openai": {
                "baseUrl": os.environ["LLM_BASE_URL"],
                "apiKey": os.environ["LLM_API_KEY"],
                "api": "openai-completions",
                "timeoutSeconds": 300,
                "models": [{
                    "id": os.environ["LLM_MODEL"],
                    "name": os.environ["LLM_MODEL"],
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                    },
                    "contextWindow": 131072,
                    "maxTokens": 8192,
                }],
            },
        },
    },
    "gateway": {
        "mode": "local",
        "bind": "lan",
        "port": 18789,
        "auth": {"token": os.environ["MCP_GATEWAY_TOKEN"]},
        "controlUi": {
            "dangerouslyAllowHostHeaderOriginFallback": True,
            "dangerouslyDisableDeviceAuth": True,
        },
        "http": {"endpoints": {"responses": {"enabled": True}}},
    },
    "mcp": {
        "servers": {
            "ogmemory": {
                "url": "http://127.0.0.1:8091/mcp",
                "transport": "streamable-http",
                "headers": {
                    "X-API-Key": os.environ["OG_TENANT_API_KEY"],
                    "X-Account-ID": os.environ["OG_ACCOUNT_ID"],
                    "X-User-ID": os.environ["OG_USER_ID"],
                    "X-Agent-ID": os.environ["OG_MCP_AGENT_ID"],
                },
                "connectTimeout": 30,
                "timeout": 300,
            },
        },
    },
    "plugins": {
        "allow": ["og-memory-context-engine"],
        "slots": {"contextEngine": "legacy"},
        "entries": {
            "og-memory-context-engine": {
                "enabled": False,
                "config": {
                    "mode": "remote",
                    "memoryApiBaseUrl": "http://127.0.0.1:8090",
                    "authApiKey": os.environ["OG_TENANT_API_KEY"],
                    "authAccountId": os.environ["OG_ACCOUNT_ID"],
                },
            },
            "memory-core": {"enabled": False},
        },
    },
}

out = Path(os.environ["OGMEM_STACK"]) / "openclaw-mcp" / "openclaw.template.json"
rendered = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
if "${" in rendered:
    raise SystemExit("OpenClaw 配置中存在未替换的环境变量")
out.write_text(rendered, encoding="utf-8")
out.chmod(0o600)
print(out)
PY

docker run --rm -i \
  --user 0:0 \
  -v "$OGMEM_STACK/openclaw-mcp:/target" \
  --entrypoint sh \
  "$OPENCLAW_IMAGE" \
  -c 'cat > /target/openclaw.json && chown node:node /target/openclaw.json && chmod 600 /target/openclaw.json' \
  < "$OGMEM_STACK/openclaw-mcp/openclaw.template.json"

rm -f "$OGMEM_STACK/openclaw-mcp/openclaw.template.json"
```

OpenClaw 连接 oGMemory MCP 服务后会通过交互自动发现可用工具，无需在 `tools.alsoAllow` 中静态列出工具名称。

启动 MCP 版 OpenClaw：

```bash
docker run -d \
  --name openclaw_mcp \
  --restart unless-stopped \
  --network host \
  -v "$OGMEM_STACK/openclaw-mcp:/home/node/.openclaw" \
  -e "LLM_API_KEY=$LLM_API_KEY" \
  -e "LLM_BASE_URL=$LLM_BASE_URL" \
  -e "LLM_MODEL=$LLM_MODEL" \
  -e "OG_TENANT_API_KEY=$OG_TENANT_API_KEY" \
  -e "MCP_GATEWAY_TOKEN=$MCP_GATEWAY_TOKEN" \
  -e 'NO_PROXY=*' \
  -e 'no_proxy=*' \
  "$OPENCLAW_IMAGE" gateway
```

### 5. 验证两种接入

在服务器终端设置要验证的容器；这些变量也用于[操作指南](user-quickstart/TEST_GUIDE_zh.md#1-操作准备)：

```bash
export OGMEM_CONTAINER='ogmem'
export CE_CONTAINER="$OPENCLAW_CONTAINER_NAME"
export MCP_CONTAINER='openclaw_mcp'
```

先从两个 OpenClaw 容器分别调用配置的模型服务：

```bash
for container in "$CE_CONTAINER" "$MCP_CONTAINER"; do
  agent_id="$(docker exec -u node -e HOME=/home/node "$container" node -e '
    const cfg = require("/home/node/.openclaw/openclaw.json");
    const agents = cfg.agents?.list ?? [];
    process.stdout.write(agents.find(agent => agent.default)?.id ?? agents[0]?.id ?? "main");
  ')"
  docker exec -u node -e HOME=/home/node "$container" openclaw agent \
    --agent "$agent_id" \
    --session-id "model-check-$(date +%s)" \
    --message '只回复：模型连接正常' --json
done
```

两个容器都应返回“模型连接正常”。出现 `EADDRNOTAVAIL` 等连接错误时，检查报错容器的模型域名解析、IPv4/IPv6 出站网络和模型服务访问权限，修复后重跑模型请求。查看 Gateway 错误和容器重启次数：

```bash
for container in "$CE_CONTAINER" "$MCP_CONTAINER"; do
  docker inspect "$container" --format '{{.Name}} status={{.State.Status}} restarts={{.RestartCount}}'
  docker logs "$container" --since 10m 2>&1 | tail -n 100
done
```

随后检查 MCP 工具连接：

```bash
curl -fsS http://127.0.0.1:18789/health

docker exec -u node -e HOME=/home/node \
  "$MCP_CONTAINER" openclaw mcp probe ogmemory --json
```

验证结果应满足以下条件：`diagnostics` 为空，且工具列表包含：

```text
ogmemory__ogm_query_memory
ogmemory__ogm_read_memory
ogmemory__ogm_write_memory
```

检查 Context Engine Gateway：

```bash
curl -fsS "http://127.0.0.1:${GATEWAY_PORT}/health"
```

按操作指南完成[Context Engine 测试](user-quickstart/TEST_GUIDE_zh.md#2-context-engine-测试)和[MCP 测试](user-quickstart/TEST_GUIDE_zh.md#3-mcp-测试)，并核对以下结果：

| 接入方式 | 通过条件 |
| --- | --- |
| Context Engine | 短、长内容触发提取或 `/compact` 完成后，在 oGMemory 中核对写入的记忆；切换新会话后，正确召回短内容偏好和长内容中的各项事实。 |
| MCP | 模型实际调用 oGMemory 工具完成写入，按检索返回的 URI 读取并核对内容；新会话正确召回，更新后再次读取和查询，得到更新后的要求。 |

在[管理界面](../docs/memory_viewer_guide.md)中选择部署时的 Account、User 和 Agent，核对记忆内容；MCP 还可使用 `ogm_read_memory` 读取检索返回的 URI。保存工具返回和存储核对结果，作为部署验收记录。

### 6. 配置目录判断

不同镜像可能使用不同的运行用户和 HOME 目录。执行以下命令确认实际配置：

```bash
docker inspect openclaw_mcp --format 'default_user={{json .Config.User}}'
docker exec openclaw_mcp sh -lc \
  'echo "exec_user=$(id -un) home=$HOME"; find /root /home -path "*/.openclaw/openclaw.json" 2>/dev/null'
```

上述 OpenClaw 镜像中，Gateway 以 `node` 用户运行，配置目录为 `/home/node/.openclaw`。因此，执行 probe 时应显式指定用户和 HOME 目录：

```bash
docker exec -u node -e HOME=/home/node \
  openclaw_mcp openclaw mcp probe ogmemory --json
```

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| `Unknown provider type` | `llm.provider` 使用 `openai`、`openai-cached` 或 `mock`。即使后端是兼容 OpenAI 协议的模型服务，LLM provider 也通常填写 `openai`。 |
| `OPENGAUSS_CONNECTION_STRING is required` | 使用 openGauss 时必须在 `vector_db.connection_string` 填写完整连接串。 |
| 无法连接 openGauss | 确认 `opengauss` 容器已启动，并且 `connection_string` 中的端口、密码分别和 `GS_PORT`、`GS_PASSWORD` 一致。 |
| 宿主机端口被占用 | host 网络模式下请勿使用 `-p`。将 openGauss 的 `GS_PORT` 设置为空闲端口，并将 `connection_string` 中的 `port` 修改为相同值。 |
| Embedding 维度不匹配 | 修改 `vector_db.dimension`，设置为 Embedding 模型实际输出维度。 |
| AGFS 写入失败 | 检查 `/your_directory/ogmem` 是否存在且容器用户有写权限。 |
