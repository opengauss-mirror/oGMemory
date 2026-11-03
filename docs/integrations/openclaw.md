# OpenClaw 对接

## 1. 前置条件

- Node.js 18+
- OpenClaw 已安装
- remote 模式：oGMemory HTTP 服务已启动
- local 模式：本机可运行 AGFS 和 oGMemory Python 服务

## 2. 安装插件

```bash
cd plugins/openclaw
openclaw plugins install --link .
openclaw gateway restart
```

插件 ID：`og-memory-context-engine`；类型：`context-engine`。

## 3. Remote 模式

`~/.openclaw/openclaw.json`：

```jsonc
{
  "plugins": {
    "slots": {
      "contextEngine": "og-memory-context-engine"
    },
    "entries": {
      "og-memory-context-engine": {
        "enabled": true,
        "hooks": {
          "allowConversationAccess": true
        },
        "config": {
          "mode": "remote",
          "memoryApiBaseUrl": "https://ogmemory.example.com",
          "authApiKey": "<tenant-key>",
          "authAccountId": "acct-1",
          "accountId": "acct-1",
          "userId": "user-1",
          "agentId": "openclaw",
          "traceTimeoutMs": 3000,
          "traceCaptureContent": true,
          "prefetchEnabled": true,
          "prefetchTopK": 5,
          "compactTakeoverEnabled": true,
          "summaryMaxChars": 4000,
          "shortTermIndexMode": "sync"
        }
      }
    }
  }
}
```

`memoryApiBaseUrl` 不带 `/api/v1`。

## 4. Local 模式

```jsonc
{
  "plugins": {
    "slots": {
      "contextEngine": "og-memory-context-engine"
    },
    "entries": {
      "og-memory-context-engine": {
        "enabled": true,
        "hooks": {
          "allowConversationAccess": true
        },
        "config": {
          "mode": "local",
          "projectRoot": "/absolute/path/oGMemory",
          "agfsServerDir": "/absolute/path/oGMemory/agfs",
          "agfsDataDir": "/absolute/path/ogmemory-data",
          "agfsPort": 1833,
          "contextEnginePort": 8090,
          "startupTimeoutMs": 60000,
          "accountId": "acct-1",
          "userId": "user-1",
          "agentId": "openclaw",
          "llmProvider": "openai",
          "llmApiKey": "<model-key>",
          "llmBaseUrl": "https://api.openai.com/v1",
          "llmModel": "gpt-4o-mini",
          "embeddingProvider": "openai",
          "embeddingApiKey": "<embedding-key>",
          "embeddingBaseUrl": "https://api.openai.com/v1",
          "embeddingModel": "text-embedding-ada-002"
        }
      }
    }
  }
}
```

local 模式由 OpenClaw Gateway：

1. 检查 AGFS/ContextEngine 健康状态。
2. 端口未占用时启动 AGFS。
3. 启动 oGMemory ContextEngine HTTP 服务。
4. 把实际 HTTP 地址写入运行时配置。

Local 模式中，插件配置里显式填写的 LLM/Embedding 字段优先于
`projectRoot/config/ogmem.yaml` 中的同名字段；未填写的字段仍使用 YAML、环境变量或默认值。

## 5. 配置参数

| 参数 | 模式 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `mode` | 全部 | `remote` | `local` 或 `remote` |
| `memoryApiBaseUrl` | remote | `http://127.0.0.1:8090` | oGMemory HTTP 基址；环境变量 `OGMEM_API_URL` |
| `authApiKey` | remote | 空 | 租户 Key；`OG_AUTH_API_KEY` |
| `authAccountId` | remote | 空 | 鉴权 account；`OG_AUTH_ACCOUNT_ID` |
| `accountId` | 全部 | 配置值 | 请求体 accountId |
| `userId` | 全部 | 配置值 | 默认 userId；可由消息 tag/调用参数覆盖 |
| `agentId` | 全部 | 配置值 | 默认 Agent ID |
| `traceTimeoutMs` | 全部 | `3000` | 每次 Trace 上传超时 |
| `traceCaptureContent` | 全部 | `true` | Trace 是否带脱敏内容 |
| `commitTokenThreshold` | 全部 | `20000` | 兼容字段；当前服务端不按此值提交，抽取阈值由 `OGMEM_AFTER_TURN_THRESHOLD` 控制 |
| `prefetchEnabled` | 全部 | `false` | compose 前调用 prefetch |
| `prefetchTopK` | 全部 | `5` | prefetch 候选数 |
| `compactTakeoverEnabled` | 全部 | `true` | oGMemory 是否返回最终 compact 结果 |
| `summaryMaxChars` | 全部 | `4000` | compact summary 字符上限 |
| `shortTermIndexMode` | 全部 | `sync` | `sync`、`async`、`off` |
| `agfsServerDir` | local | 无 | AGFS Go 源码目录；`AGFS_SERVER_DIR` |
| `agfsPort` | local | `1833` | `AGFS_PORT` |
| `agfsDataDir` | local | 无 | `AGFS_DATA_DIR` |
| `contextEnginePort` | local | `8090` | `CONTEXTENGINE_PORT` |
| `projectRoot` | local | 无 | `CONTEXTENGINE_PROJECT_ROOT` |
| `startupTimeoutMs` | local | `60000` | 每个服务启动等待上限 |
| `llmProvider` | local | `mock` | `openai` 或 `mock` |
| `llmApiKey` | local | 空 | `OGMEM_API_KEY` |
| `llmBaseUrl` | local | 空 | `OGMEM_BASE_URL` |
| `llmModel` | local | `gpt-4o-mini` | `OGMEM_LLM_MODEL` |
| `embeddingProvider` | local | LLM provider | `EMBEDDING_PROVIDER` |
| `embeddingApiKey` | local | LLM Key | `OGMEM_EMBEDDING_API_KEY` |
| `embeddingBaseUrl` | local | LLM URL | `OGMEM_EMBEDDING_BASE_URL` |
| `embeddingModel` | local | `text-embedding-ada-002` | `OGMEM_EMBEDDING_MODEL` |

## 6. 会话访问权限

非内置插件必须配置：

```json
{
  "hooks": {"allowConversationAccess": true}
}
```

该字段放在插件 entry 下，与 `config` 同级。缺失时，OpenClaw 阻止 `before_agent_run`、`agent_end` 获取原始 prompt/assistant 文本；自动 compose/Trace 内容捕获不完整。

## 7. HTTP 鉴权与报文

插件请求头：

```text
Content-Type: application/json
X-API-Key: <authApiKey>          # 配置时发送
X-Account-ID: <authAccountId>    # 配置时发送
X-User-ID: <params.userId>       # 当前请求有 userId 时发送
```

HTTP 超时：`after_turn` 300 秒；其余 Context Engine 方法 120 秒；Trace 使用 `traceTimeoutMs`。

## 8. Context Engine 方法

### 8.1 `ingest(params)`

原样调用：

```http
POST /api/v1/ingest

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "openclaw",
  "sessionId": "session-1",
  "messages": [
    {"role": "user", "content": "message"}
  ]
}
```

### 8.2 `assemble(params)`

流程：

1. 从消息中的用户 tag 提取 userId；否则用调用参数或配置 userId。
2. 补齐 accountId、agentId。
3. `prefetchEnabled=true` 时调用 `/prefetch`。
4. 调用 `/compose`。
5. 将 Profile、Archive、原会话、Session State、Working Set 组装成 OpenClaw messages。

prefetch：

```http
POST /api/v1/prefetch

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "openclaw",
  "sessionId": "session-1",
  "messages": [{"role":"user","content":"当前问题"}],
  "prefetchTopK": 5
}
```

compose：

```http
POST /api/v1/compose

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "openclaw",
  "sessionId": "session-1",
  "messages": [{"role":"user","content":"当前问题"}]
}
```

oGMemory 不可用时，assemble 返回原始 `params.messages`，`estimatedTokens: 0`，主 Agent 继续。

### 8.3 `afterTurn(params)`

```http
POST /api/v1/after_turn

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "openclaw",
  "sessionId": "session-1",
  "messages": [
    {"role":"user","content":"用户消息"},
    {"role":"assistant","content":"助手回复"}
  ]
}
```

该调用执行增量缓冲和达到阈值后的长期记忆抽取。

### 8.4 `compact(params)`

插件总是先 prepare：

```http
POST /api/v1/prepare_compaction

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "openclaw",
  "sessionId": "session-1",
  "summaryMaxChars": 4000,
  "shortTermIndexMode": "sync"
}
```

`compactTakeoverEnabled=true`：

```http
POST /api/v1/compact

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "openclaw",
  "sessionId": "session-1",
  "summaryMaxChars": 4000,
  "shortTermIndexMode": "sync",
  "prepareToken": "<one-time-token>"
}
```

插件把响应 `result.summary`、`tokensBefore`、`firstKeptEntryId` 写入 OpenClaw compaction boundary。

`compactTakeoverEnabled=false`：prepare 仍执行增量抽取，但最终压缩委托 OpenClaw runtime；不调用 oGMemory `/compact`。

### 8.5 `dispose()`

```http
POST /api/v1/dispose

{
  "accountId": "acct-1",
  "userId": "user-1",
  "agentId": "openclaw",
  "sessionId": "session-1"
}
```

插件使用已记录的当前 session ID，以及最近一次 `afterTurn` 的 account、user、agent 身份。服务端释放 session 资源并派发未处理消息的后台刷新。

## 9. Trace Hook

| OpenClaw Hook | canonical event |
| --- | --- |
| `session_start` | `session_start` |
| `before_agent_run` | `user_message` |
| `before_tool_call` | `tool_start` |
| `after_tool_call` 成功 | `tool_success` |
| `after_tool_call` 失败 | `tool_failure` |
| `agent_end` | `turn_end` |
| `before_compaction` | `pre_compact` |
| `session_end` | `session_end` |

请求：

```http
POST /api/v1/trace/events

{
  "request_id": "event-id",
  "timestamp": "2026-08-03T17:00:00+08:00",
  "account_id": "acct-1",
  "user_id": "user-1",
  "agent_id": "openclaw",
  "session_id": "session-1",
  "event": {
    "event_type": "tool_success",
    "object_id": "tool-call-id",
    "terminal_state": "success",
    "metadata": {"tool": "write"}
  }
}
```

Trace：

- 非阻塞发送。
- 每个事件最多尝试 2 次。
- 每次超时 `traceTimeoutMs`。
- 内容上传前脱敏。
- 失败记录日志，Agent 主流程继续。

## 10. 验证

```bash
openclaw doctor
npm test --prefix plugins/openclaw
curl http://127.0.0.1:8090/api/v1/health
```

运行日志必须出现：

```text
[og-memory] ... compose ...
[og-memory] ... after_turn ...
[og-memory] ... prepare_compaction ...
```

验证顺序：

1. 第一轮声明一个偏好。
2. 确认 `/after_turn` 收到 user/assistant。
3. 新会话询问该偏好。
4. 确认 `/compose` 返回 `retrievedEvidence` 或 `_ogmem` 消息。
5. 触发压缩，确认 prepareToken 被消费且返回 summary。

## 11. 切回默认 Context Engine

从 `plugins.slots.contextEngine` 删除 `og-memory-context-engine`，禁用对应 entry，重启 OpenClaw。
