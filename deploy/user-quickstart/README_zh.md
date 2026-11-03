# oGMemory 一键部署与体验指南

oGMemory 是基于 openGauss 构建的语义记忆引擎，为 Claude Code、OpenClaw、Codex 等 CLI Agent 提供跨会话的长期记忆能力。它在对话的六个关键阶段（消息接收、启动组装、工具调用、回合结束、压缩、会话结束）拦截并处理上下文，把对话里的偏好、事实和技能沉淀为结构化记忆，供后续会话检索复用，让 Agent 记住用户在旧会话中说过的内容。

以下章节介绍部署步骤，以及三种典型的写入检索方式。

## 1. 一键部署

### 1.1 配置

在待部署的 ARM64 服务器上执行以下命令，将环境变量模板和部署脚本下载到当前用户的 HOME 目录：

```bash
curl -fL https://raw.atomgit.com/opengauss/oGMemory/raw/dev/deploy/user-quickstart/ogmemory.env -o "$HOME/ogmemory.env"

curl -fL https://raw.atomgit.com/opengauss/oGMemory/raw/dev/deploy/user-quickstart/deploy.sh -o "$HOME/deploy.sh"

chmod 600 "$HOME/ogmemory.env"
chmod 700 "$HOME/deploy.sh"
```

确认文件已下载：

```bash
ls -l "$HOME/ogmemory.env" "$HOME/deploy.sh"
```

编辑环境变量文件：

```bash
vi "$HOME/ogmemory.env"
```

修改以下内容：

```bash
# 实例名称：只使用小写字母、数字和短横线；同一服务器上必须唯一。
export INSTANCE_NAME='codelab'

# 宿主机端口：同一服务器上必须唯一。
export OPENGAUSS_HOST_PORT='15432'
export OGMEM_HTTP_HOST_PORT='18090'
export OGMEM_MCP_HOST_PORT='18091'
export OPENCLAW_CE_HOST_PORT='18790'
export OPENCLAW_MCP_HOST_PORT='18791'

# OpenAI 兼容的 LLM 服务。
export LLM_BASE_URL='https://llm.example.com/v1'
export LLM_MODEL='your-llm-model'
export LLM_API_KEY='your-llm-api-key'

# Embedding Base URL 不要包含具体的 /embeddings 接口。
# 火山引擎豆包使用 volcengine；其他 OpenAI 兼容服务通常使用 openai。
export EMBEDDING_PROVIDER='openai'
export EMBEDDING_BASE_URL='https://embedding.example.com/v1'
export EMBEDDING_MODEL='your-embedding-model'
export EMBEDDING_API_KEY='your-embedding-api-key'
export EMBEDDING_DIMENSION='1024'

# 测试场景：短对话也触发记忆提取，必须是正整数。
export OGMEM_AFTER_TURN_THRESHOLD='1'
```

保存并退出 `vi`：按 `Esc`，输入 `:wq`，按回车。

`OGMEM_AFTER_TURN_THRESHOLD` 控制待处理 token 达到多少时触发记忆提取。
quickstart 默认值为 `1`（旧环境文件未设置该变量时也使用 `1`），便于短对话测试；
较小的值会增加模型调用频率。需要降低调用频率时可设为 `200` 或其他正整数。
提取和索引仍是异步过程，“记住了”不代表后台处理已经完成；新会话检索前应等待处理完成。

修改环境文件后重新执行部署脚本生效，也可以单次覆盖文件中的值：

```bash
OGMEM_AFTER_TURN_THRESHOLD=200 "$HOME/deploy.sh"
```

配置优先级为：进程环境变量 > `ogmemory.env` > 默认值 `1`。
单次覆盖不修改环境文件；后续部署会重新按以上优先级取值。
这只改变 quickstart 的默认配置，其他部署方式的默认阈值不变。


### 1.2 执行部署

在终端执行：

```bash
chmod 600 "$HOME/ogmemory.env"
chmod 700 "$HOME/deploy.sh"
"$HOME/deploy.sh"
```

脚本会实际调用一次 OpenClaw Responses 接口验证 LLM 配置。成功后会显示四个容器及其端口。数据、生成配置和密钥保存在：$HOME/ogmemory-<实例名称>/。

正式创建容器前，脚本还会先调用一次 LLM 的 `chat/completions` 和 Embedding 的 `embeddings` 接口，验证地址、Key 和模型是否可用。上游校验失败时不会创建或删除容器，便于尽早修正配置。

如果实例名称已被其他用户使用，脚本会在删除容器前停止。请修改 `INSTANCE_NAME` 后重新执行。

### 1.3 修改配置并重建（可选）

需要更换 LLM 配置、Embedding API Key、模型服务地址或端口时，修改 `$HOME/ogmemory.env`，再次执行：

```bash
"$HOME/deploy.sh"
```

注意：以下目录会通过 Docker bind mount 映射到宿主机，并且默认保留在实例目录中：

- `openclaw-ce`、`openclaw-mcp`：OpenClaw 配置、设备配对和运行状态；
- `agfs`：记忆文件和 AGFS 数据；
- `opengauss`：数据库和向量索引数据。

重新执行脚本默认只删除并重建容器，不删除这些目录。因此仅修改 `ogmemory.env` 可能仍保留旧的本地状态。脚本每次会重新生成 OpenClaw 的主配置，但不会自动清空其余持久化文件。

如确认需要重置某类本地状态，可显式执行（会直接删除对应目录中的全部数据）：

```bash
RESET_OPENCLAW_CONFIG=true "$HOME/deploy.sh"  # 重置 OpenClaw 配置和状态
RESET_AGFS_DATA=true "$HOME/deploy.sh"        # 重置 AGFS 记忆数据
RESET_OPENGAUSS_DATA=true "$HOME/deploy.sh"   # 重置数据库和向量索引
```

脚本启动时会列出检测到的持久化目录；不确定时不要设置这些重置开关。重置前请确认数据无需保留。不要删除实例目录中的 `secrets.env`，否则访问密钥会改变。

脚本完成全部预检后重建本实例的容器，保留 openGauss、oGMemory 和 OpenClaw 数据。不要删除实例目录中的 `secrets.env`，否则访问密钥会改变。

已有实例不能修改 `EMBEDDING_PROVIDER`、`EMBEDDING_MODEL` 或 `EMBEDDING_DIMENSION`。这些配置决定向量空间；混用不同模型会导致检索失真，维度变化会导致向量写入失败。需要更换时，请设置新的 `INSTANCE_NAME` 和一组未占用端口，创建独立实例。

## 2. 写入、检索与交互体验

### 2.1 HTTP API 

#### 写入记忆

在终端执行：

```bash
source "$HOME/ogmemory-runtime.env"

curl -fsS -X POST "http://127.0.0.1:${OGMEM_HTTP_HOST_PORT}/api/v1/after_turn" -H 'Content-Type: application/json' -H "X-API-Key: ${OG_TENANT_API_KEY}" -H "X-Account-ID: ${OG_ACCOUNT_ID}" -H "X-User-ID: ${OG_USER_ID}" -d "{\"accountId\": \"${OG_ACCOUNT_ID}\", \"userId\": \"${OG_USER_ID}\", \"agentId\": \"${OG_API_AGENT_ID}\", \"sessionId\": \"api-write-001\", \"messages\": [{\"role\": \"user\", \"content\": \"我喜欢用 Python 和 uv 管理项目。\"}, {\"role\": \"assistant\", \"content\": \"记住了。\"}], \"forceExtract\": true}"
```

#### 检索记忆

等待后台索引后，单独执行：

```bash
source "$HOME/ogmemory-runtime.env"

sleep 20

curl -fsS -X POST "http://${PUBLISH_IP}:${OGMEM_HTTP_HOST_PORT}/api/v1/compose" -H 'Content-Type: application/json' -H "X-API-Key: ${OG_TENANT_API_KEY}" -H "X-Account-ID: ${OG_ACCOUNT_ID}" -H "X-User-ID: ${OG_USER_ID}" -d "{\"accountId\": \"${OG_ACCOUNT_ID}\", \"userId\": \"${OG_USER_ID}\", \"agentId\": \"${OG_API_AGENT_ID}\", \"sessionId\": \"api-read-001\", \"messages\": [{\"role\": \"user\", \"content\": \"我偏好什么开发语言和包管理工具？\"}], \"prompt\": \"我偏好什么开发语言和包管理工具？\"}"
```

返回内容应包含 Python 和 uv 相关记忆。

### 2.2 OpenClaw Context Engine

#### 写入记忆

```bash
source "$HOME/ogmemory-runtime.env"

export CE_WRITE_SESSION="ce-write-$(date +%s)"

curl -fsS -X POST "http://${PUBLISH_IP}:${OPENCLAW_CE_HOST_PORT}/v1/responses" -H 'Content-Type: application/json' -H "Authorization: Bearer ${CE_GATEWAY_TOKEN}" -H 'X-OpenClaw-Agent-ID: main' -H "X-OpenClaw-Session-Key: ${CE_WRITE_SESSION}" -d '{"model":"openclaw","input":"记住：我每周五下午写周报。只回复：记住了。","stream":false}'
```

#### 检索记忆

使用新会话检索，避免答案来自当前聊天上下文：

```bash
source "$HOME/ogmemory-runtime.env"

sleep 20
export CE_READ_SESSION="ce-read-$(date +%s)"

curl -fsS -X POST "http://${PUBLISH_IP}:${OPENCLAW_CE_HOST_PORT}/v1/responses" -H 'Content-Type: application/json' -H "Authorization: Bearer ${CE_GATEWAY_TOKEN}" -H 'X-OpenClaw-Agent-ID: main' -H "X-OpenClaw-Session-Key: ${CE_READ_SESSION}" -d '{"model":"openclaw","input":"我每周什么时候写周报？请直接回答。","stream":false}'
```

回答应包含"周五下午"。

#### TUI 交互

启动 Context Engine 的交互终端：

```bash
source "$HOME/ogmemory-runtime.env"

docker exec -it -u node -e HOME=/home/node \
  "${INSTANCE_NAME}_openclaw_ce" \
  openclaw tui \
  --url ws://127.0.0.1:18789 \
  --token "$CE_GATEWAY_TOKEN" \
  --session ce-tui
```

进入后依次输入：

```text
记住：我常用的测试框架是 pytest，提交代码前会运行完整测试。
/new
我提交代码前会做什么？常用什么测试框架？
/exit
```

`/new` 会切换到新会话。新会话仍能回答，说明信息来自 oGMemory。

### 2.3 OpenClaw MCP

#### 验证连接与能力发现

在终端执行：

```bash
source "$HOME/ogmemory-runtime.env"

docker exec -u node -e HOME=/home/node \
  "${INSTANCE_NAME}_openclaw_mcp" \
  openclaw mcp probe ogmemory --json
```

结果应无连接错误，并列出服务端动态返回的工具和资源能力。

#### 写入记忆

```bash
source "$HOME/ogmemory-runtime.env"

export MCP_WRITE_SESSION="mcp-write-$(date +%s)"

curl -fsS -X POST "http://${PUBLISH_IP}:${OPENCLAW_MCP_HOST_PORT}/v1/responses" -H 'Content-Type: application/json' -H "Authorization: Bearer ${MCP_GATEWAY_TOKEN}" -H 'X-OpenClaw-Agent-ID: main' -H "X-OpenClaw-Session-Key: ${MCP_WRITE_SESSION}" -d '{"model":"openclaw","input":"请先发现 ogmemory MCP 服务提供的能力，再使用合适能力记住：我的代码评审时间是每周二上午。完成后只回复写入结果。","stream":false}'
```

#### 检索记忆

使用新会话单独执行：

```bash
source "$HOME/ogmemory-runtime.env"

sleep 20
export MCP_READ_SESSION="mcp-read-$(date +%s)"

curl -fsS -X POST "http://${PUBLISH_IP}:${OPENCLAW_MCP_HOST_PORT}/v1/responses" -H 'Content-Type: application/json' -H "Authorization: Bearer ${MCP_GATEWAY_TOKEN}" -H 'X-OpenClaw-Agent-ID: main' -H "X-OpenClaw-Session-Key: ${MCP_READ_SESSION}" -d '{"model":"openclaw","input":"请使用 ogmemory MCP 服务提供的合适能力查询：我的代码评审时间是什么时候？请直接回答。","stream":false}'
```

回答应包含"每周二上午"。

#### TUI 交互与更多能力

启动 MCP 版交互终端：

```bash
source "$HOME/ogmemory-runtime.env"

docker exec -it -u node -e HOME=/home/node \
  "${INSTANCE_NAME}_openclaw_mcp" \
  openclaw tui \
  --url ws://127.0.0.1:18789 \
  --token "$MCP_GATEWAY_TOKEN" \
  --session mcp-tui
```

进入后可依次输入：

```text
请使用ogmemory中的合适能力记住：我负责支付模块，偏好小步提交；发布前必须完成回归测试。
/new
我负责哪个模块？提交和发布前分别有什么偏好或要求？请通过 ogmemory 检索后回答。
请通过 ogmemory 检索与"上线前质量检查"语义相关的记忆，并说明匹配到哪些信息。
请使用合适能力把我的发布要求更新为：发布前必须完成回归测试和安全扫描。
/new
我现在发布前必须完成哪些检查？请通过 ogmemory 检索后回答。
/exit
```

这组操作依次体验能力发现、记忆写入、跨会话检索、语义检索和记忆更新。实际能力以 MCP 连接后返回的结果为准。

### 2.4 查看运行状态

```bash
source "$HOME/ogmemory-runtime.env"

docker ps --filter "name=${INSTANCE_NAME}_" \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

查看某个服务最近日志：

```bash
source "$HOME/ogmemory-runtime.env"

docker logs "${INSTANCE_NAME}_ogmemory" --since 10m
docker logs "${INSTANCE_NAME}_openclaw_ce" --since 10m
docker logs "${INSTANCE_NAME}_openclaw_mcp" --since 10m
```

## Agent 工具与经验记忆

部署模板默认关闭 Agent 记忆抽取。需要启用时，按 [Agent 记忆部署说明](../agent-memory.md) 修改挂载 YAML、重启服务并验证配置和 Trace 抽取结果。
