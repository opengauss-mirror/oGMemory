# 通用 Agent Plugin 对接 oGMemory

本教程沿用仓库接入文档的顺序：准备服务、获取插件、配置客户端、体验跨会话记忆、排查问题。
OpenClaw、DSH、Claude Code、Codex 共用同一个代理和同一个 Skill。
本教程使用各客户端的 MCP 接入口，不要求它们都支持直接安装 Agent Plugins 1.0。

## 1. 准备 oGMemory 服务

先按 [部署与体验指南](../../deploy/user-quickstart/README_zh.md) 部署服务，
再按 [MCP 教程](mcp.md) 开启 Streamable HTTP 和角色控制。
服务端至少开启记忆写入，示例配置片段：

```yaml
mcp:
  enabled: true
  port: 8091
  path: /mcp
  enable_write_tools: true
  enable_delete_tools: false
```

准备以下信息：

| 参数 | 示例 | 说明 |
| --- | --- | --- |
| MCP URL | `https://memory.example.com:18091/mcp` | 完整 MCP 地址，注意容器端口映射 |
| Tenant Key | 管理员提供 | oGMemory 服务密钥，不是 LLM 密钥 |
| Account ID | `customer-a` | 租户 |
| User ID | `alice` | 当前记忆用户 |
| Agent ID | `customer-a-agent` | 服务端已授权的 Agent |

客户端不需要数据库或模型凭据。服务端单独配置抽取模型和向量模型。
远程地址使用 HTTPS。`127.0.0.1` 指 Agent 进程所在机器；容器和 WSL 应特别核对。
插件仅暴露三个工具，底层 MCP 的其他能力不会透出。

## 2. 获取插件与生成配置

在 Agent 机器安装 Node.js 22+，并取得当前发布版本的 `plugins/agent-plugins/` 目录或 ZIP。
插件无 npm 依赖。使用仓库时，在包含本功能的分支或发布标签下操作。
不要在功能尚未合入时从旧 `dev` 下载插件。

```bash
node --version
```

已有 oGMemory CLI 时可生成配置：

```bash
# 预先在当前进程环境配置 OG_AUTH_API_KEY
ogmem onboard --mode plugin --agent agent-plugins --non-interactive \
  --mcp-url https://memory.example.com:18091/mcp \
  --ogmem-api-key-env OG_AUTH_API_KEY \
  --account-id customer-a --user-id alice --agent-id customer-a-agent
```

产物位于 `ogmem-client-config/agent-plugins/`。将 `agent-plugin.json` 放到
Agent 机器的 `~/.ogmemory/agent-plugin.json`。如果 CLI 不在本机，直接复制
插件中的 `config.example.json` 到上述位置并填写即可，无需安装 Python 后端。

自定义路径使用 `OGMEM_AGENT_PLUGIN_CONFIG`。配置文件示例：

```json
{
  "mcp_url": "https://memory.example.com:18091/mcp",
  "api_key": "<tenant-key>",
  "account_id": "customer-a",
  "user_id": "alice",
  "agent_id": "customer-a-agent",
  "timeout_ms": 120000
}
```

真实文件存放在插件目录外，仅当前用户可读；POSIX 可执行 `chmod 600`，Windows 使用用户私有目录和 ACL。
配置生成器不会安装插件、改动已有 Agent 配置或重启 Agent。

先自检：

```bash
node /absolute/path/plugins/agent-plugins/scripts/doctor.mjs
```

应显示 `connected: true`，并且只包含：

```text
ogm_query_memory
ogm_read_memory
ogm_write_memory
```

自检通过后选择下面一个客户端流程。已有原生 oGMemory 自动写回插件时，
请在新 profile/工作目录测试，避免同时对同一轮对话执行两次写入。

## 3. Claude Code

### 3.1 注册 MCP

在测试项目的 `.mcp.json` 中合并以下条目，保留其他 MCP 配置：

```json
{
  "mcpServers": {
    "ogmemory": {
      "type": "stdio",
      "command": "node",
      "args": ["/absolute/path/plugins/agent-plugins/servers/mcp-proxy.mjs"],
      "env": {
        "OGMEM_AGENT_PLUGIN_CONFIG": "/absolute/path/private/agent-plugin.json"
      }
    }
  }
}
```

Windows 路径可写为 `D:/tools/plugins/agent-plugins/servers/mcp-proxy.mjs`，不要使用 `${PLUGIN_ROOT}`；
这个变量只适用于加载标准 manifest 的客户端。
也可使用 `claude mcp add-json ogmemory '<单个服务器 JSON 对象>'` 注册；
PowerShell 包装脚本对引号和 `--` 的处理可能不同，文件配置更容易检查。

### 3.2 加载 Skill 并验证

把 `plugins/agent-plugins/skills/ogmemory-memory/` 复制到测试项目的
`.claude/skills/ogmemory-memory/`。启动 Claude Code，按客户端提示信任项目 MCP。

建议在项目 `CLAUDE.md` 加入以下规则，避免仅发现 Skill 却未加载其写入状态约定：

```text
For persistent memory requests, invoke the ogmemory-memory skill before taking action.
Follow its write-status rules. Do not store these facts in local auto-memory files.
```

```bash
claude mcp list
```

预期 `ogmemory` 显示 `Connected`。在会话中查看 `/mcp`，确认三个工具。
再按第 7 节完成保存与查询。`Connected` 只证明握手，不证明模型已调用工具。

## 4. Codex

### 4.1 注册 MCP

在 Codex 的 `config.toml` 中添加以下服务器条目，保留已有配置：

```toml
[mcp_servers.ogmemory]
command = "node"
args = ["/absolute/path/plugins/agent-plugins/servers/mcp-proxy.mjs"]
startup_timeout_sec = 30
tool_timeout_sec = 180

[mcp_servers.ogmemory.env]
OGMEM_AGENT_PLUGIN_CONFIG = "/absolute/path/private/agent-plugin.json"
```

### 4.2 加载 Skill 并验证

将 Skill 目录复制到测试项目的 `.agents/skills/ogmemory-memory/`，
或将规则合并到项目 `AGENTS.md`。重新连接 Codex MCP 后检查：

```bash
codex mcp list
```

配置列表仅证明条目存在。在会话中检查 MCP 连接状态和工具列表，执行第 7 节测试。

无交互 `codex exec` 验收还需要预先授权测试工具。仅设置 `approval_policy = "never"`
不会自动批准 MCP 写入，可能返回 `MCP tool call requires approval`。
对于已经明确授权的隔离测试身份，可增加下面的服务器级配置；日常交互使用可保留审批：

```toml
[mcp_servers.ogmemory.tools.ogm_query_memory]
approval_mode = "approve"
[mcp_servers.ogmemory.tools.ogm_read_memory]
approval_mode = "approve"
[mcp_servers.ogmemory.tools.ogm_write_memory]
approval_mode = "approve"
```

本包与仓库 `plugins/ogmemory/` 的原生 Hooks 插件是两种接入方式，
不要使用原生插件的 marketplace 命令来假定安装了本包。

## 5. OpenClaw

适用于提供原生 `openclaw mcp` 客户端命令的版本，先执行 `openclaw mcp --help` 确认。
OpenClaw 自身可能要求比插件更高的 Node.js 版本，应满足两者要求。

### 5.1 配置 MCP

在选定 profile 的 `openclaw.json` 中合并：

```json
{
  "mcp": {
    "servers": {
      "ogmemory": {
        "command": "node",
        "args": ["/absolute/path/plugins/agent-plugins/servers/mcp-proxy.mjs"],
        "env": {
          "OGMEM_AGENT_PLUGIN_CONFIG": "/absolute/path/private/agent-plugin.json"
        },
        "requestTimeoutMs": 180000,
        "connectionTimeoutMs": 30000
      }
    }
  }
}
```

这一步不配置 `plugins.slots.contextEngine`，也不安装仓库的原生 Context Engine 插件。

### 5.2 加载 Skill 并验证

复制 Skill 到对应 OpenClaw 工作区的 `skills/ogmemory-memory/`，
按照现有管理方式重载该 profile/Gateway，并运行：

```bash
openclaw mcp probe ogmemory --json
```

预期 `diagnostics` 为空，业务工具数为 3，名字可能带 `ogmemory__` 前缀。
使用允许 MCP 工具的 tool profile；`minimal` 或显式 deny 可能隐藏工具。
probe 只验证连接和发现，随后仍需在新会话中执行第 7 节。
如 CLI 报状态数据库 maintenance 错误，先处理 OpenClaw 自身状态，不要删数据库。

## 6. DeepSeek Harness（DSH）

此流程使用 DSH 的 `@deepseek-ai/dsh-mcp-client`，不使用
`@opengauss/ogmemory-dsh-plugin` 原生生命周期适配器。

### 6.1 添加配置补丁

以下补丁格式按仓库现有 DSH 教程的 profile 机制编写。保存为 `ogmemory-mcp.patch.yml`：

```yaml
- insert:
    - id: ogmemory-mcp
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        transport: stdio
        serverName: ogmemory
        command: node
        args:
          - /absolute/path/plugins/agent-plugins/servers/mcp-proxy.mjs
        env:
          OGMEM_AGENT_PLUGIN_CONFIG: /absolute/path/private/agent-plugin.json
        toolCallTimeoutMs: 180000
        failOnStartupError: true
```

DSH 会清理子进程的部分环境变量，因此显式传入配置文件路径。
先确认目标 profile 能解析该 MCP 客户端包；依赖版本跟随已安装 DSH，避免混装不兼容版本。

```bash
dsh --profile web --patch /absolute/path/ogmemory-mcp.patch.yml --dump-config
dsh --profile web --patch /absolute/path/ogmemory-mcp.patch.yml
```

配置检查中应出现 `ogmemory-mcp` 条目。按 profile 支持的方式加载随包 Skill，
或把其规则加入该 Agent 的系统指令。不要仅因为配置导出成功就认为 MCP 已连接。

### 6.2 验证

DSH 的模型工具名通常为：

```text
mcp__ogmemory__ogm_query_memory
mcp__ogmemory__ogm_read_memory
mcp__ogmemory__ogm_write_memory
```

确认只有三个工具，并进行第 7 节测试。DSH 自身的模型配置沿用
[现有 DSH 教程](deepseek-harness.md)；本包不配置模型提供方。

## 7. 体验跨会话记忆

在独立测试身份下发送：

```text
请使用 oGMemory 记住：plugin-demo-20261006 项目的发布窗口是每周四北京时间20点，
包管理工具统一使用 uv。请实际调用写入工具，等结果返回后告诉我保存状态。
```

检查工具调用参数包含当前对话 `session_id`，以及 `force_extract=true`、`wait=true`。
以返回 `ok`、`status`、`stats` 为准：`accepted/processing` 不能当作已完成；
`completed` 但写入数为 0 也不能直接声称新增了记忆。

然后新建会话，不粘贴上一轮内容，发送：

```text
请查询 oGMemory：plugin-demo-20261006 项目什么时候发布，用什么包管理工具？
请读取命中条目的详情再回答，并说明是否有实际检索证据。
```

同一用户使用相同 Account/User 身份，使用服务端授权的 Agent；每个新会话换一个业务 session ID。
记忆可见范围仍由服务端决定，不能随便切换身份模拟共享。

自研 Agent 可通过 MCP SDK 启动同一代理：初始化 → `tools/list` →
把工具 schema 提供给模型 → 执行 `tools/call` → 将结果返回模型。
业务会话 ID 由自研 Agent 管理。可参考仓库
[`run_agent_plugin_smoke.py`](../../tests/e2e/run_agent_plugin_smoke.py) 的真实协议调用，
该脚本会写入测试数据，不是无副作用自检。

## 8. 故障排查

### 8.1 完整客户端的无交互模型验收

先按前文注册 MCP、安装 Skill、配置模型，在隔离测试目录中运行。
下面的 `SAVE_PROMPT` 和 `RECALL_PROMPT` 分别使用第 7 节的保存和查询文本；
查询文本只包含项目标识和问题，不重复待检索的事实。两条命令分别启动全新会话。

```bash
# Claude Code：MCP 配置指向本插件，仅预授权测试所需工具。
claude -p "$SAVE_PROMPT" --strict-mcp-config --mcp-config .mcp.json \
  --permission-mode dontAsk --allowedTools 'Skill,Read,mcp__ogmemory__ogm_query_memory,mcp__ogmemory__ogm_read_memory,mcp__ogmemory__ogm_write_memory' \
  --output-format stream-json --verbose > claude-save.jsonl
claude -p "$RECALL_PROMPT" --strict-mcp-config --mcp-config .mcp.json \
  --permission-mode dontAsk --allowedTools 'Skill,Read,mcp__ogmemory__ogm_query_memory,mcp__ogmemory__ogm_read_memory,mcp__ogmemory__ogm_write_memory' \
  --output-format stream-json --verbose > claude-recall.jsonl

# Codex：先按 4.2 配置隔离测试身份的工具权限。
codex exec --skip-git-repo-check --json "$SAVE_PROMPT" > codex-save.jsonl
codex exec --skip-git-repo-check --json "$RECALL_PROMPT" > codex-recall.jsonl

# DSH：每次 headless 命令创建独立会话。
dsh --profile headless --patch ./ogmemory-mcp.patch.yml "$SAVE_PROMPT"
dsh --profile headless --patch ./ogmemory-mcp.patch.yml "$RECALL_PROMPT"

# OpenClaw：使用两个不同 UUID，不使用 --deliver。
openclaw agent --local --session-id "$SAVE_UUID" --message "$SAVE_PROMPT" --json
openclaw agent --local --session-id "$RECALL_UUID" --message "$RECALL_PROMPT" --json
```

验收依据是客户端日志中的实际工具调用及返回：保存调用 `ogm_write_memory`，
新会话调用 `ogm_query_memory`，需要核对全文时调用 `ogm_read_memory`；
不能仅凭最终回复中的“已保存”判定成功。DSH 的工具轨迹在其会话持久化日志中；
OpenClaw 的 JSON 结果包含成功工具列表与会话标识。

本次使用豆包 `doubao-seed-2.1-lite`：Codex 使用方舟 `/api/coding/v3` 的 Responses 协议，
DSH/OpenClaw 使用该地址的 Chat Completions 协议；Claude Code 的
`ANTHROPIC_BASE_URL` 使用同一方舟服务的 `https://ark.cn-beijing.volces.com/api/coding`。
模型密钥通过进程环境传入，不填入插件 MCP 配置。详见
[方舟 Coding Plan 文档](https://www.volcengine.com/docs/82379/1928262)。

### 8.2 常见问题

| 现象 | 检查项 |
| --- | --- |
| 找不到 node | 使用 Node 可执行文件绝对路径；检查 Agent 服务进程的 PATH |
| 配置缺失 | 配置路径必须在 Agent 机器存在，且有读取权限 |
| 401/403 | 服务密钥、租户、用户和 Agent 授权；不要替换成模型密钥 |
| 只有两个工具 | 服务端是否启用记忆写入；连接成功并不意味着可写 |
| 超时 | 服务端模型耗时；代理及客户端都需足够超时；写入结果未知时先查询 |
| 有工具但不调用 | 是否加载 Skill；检查模型工具能力及客户端权限 |
| Codex 无交互写入需要审批 | `never` 不等于批准；见 4.2 的已授权测试工具配置 |
| OpenClaw 状态库 ownership/maintenance 错误 | 本次 Windows 2026.9.8 遇到该问题；WSL 隔离 profile 可完成模型验收，勿删除其他实例的状态库 |
| 新会话查不到 | 是否写入完成、索引是否就绪、身份与作用域是否一致 |
| MCP 404/会话失效 | 完整路径、反向代理、服务重启；重新连接，不自动重放写入 |
| 服务端 500 | 查看服务端日志；代理不透传可能含敏感信息的 HTTP 错误正文 |

代理每个进程串行执行请求，无离线队列、自动重连或后台资源订阅。
请求取消不保证撤销服务端已接受的写入。

## 9. 相关文档

- [MCP 服务与鉴权](mcp.md)
- [接入配置生成](plugin-config.md)
- [原生 Claude Code 插件](claude-code.md)
- [原生 Codex 插件](codex.md)
- [原生 OpenClaw 插件](openclaw.md)
- [原生 DSH 插件](deepseek-harness.md)

规范与客户端参考：[Agent Plugins](https://agent-plugins.org/)、
[OpenAI MCP 文档](https://developers.openai.com/learn/docs-mcp)、
[OpenClaw MCP 注册表](https://docs.openclaw.ai/cli/mcp/registry)。
