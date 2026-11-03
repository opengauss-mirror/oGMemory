# 客户端 Agent 对接 oGMemory

为已有 Agent 导出配置：`ogmem onboard --mode plugin`。
七种原生适配器之外的 Agent 自动回退到 HTTP MCP。参见[接入配置生成](plugin-config.md)。

| 接入方式 | 文档 | 入口 | 适用场景 |
| --- | --- | --- | --- |
| HTTP REST API | [HTTP 接口](http-rest.md) | `http://<host>:8090/api/v1/*` | 自研、跨语言 Agent |
| Python API | [Python API](python-api.md) | `MemoryService`、`MemoryWriteAPI`、`ReadAPI` | Python 进程内集成 |
| MCP | [MCP 接入](mcp.md) | stdio、Streamable HTTP、SSE | MCP 客户端 |
| 通用 Agent Plugin | [记忆检索与写入](agent-plugins.md) | Agent Plugins 1.0 / stdio 代理 + Skill | OpenClaw、DSH、Claude Code、Codex 等 MCP Agent |
| Claude Code | [Claude Code 插件](claude-code.md) | Claude Code Hooks | Claude Code |
| Codex | [Codex 插件](codex.md) | 全生命周期 Codex Hooks | Codex App、CLI、IDE |
| OpenCode | [OpenCode 插件](opencode.md) | OpenCode lifecycle hooks | OpenCode 1.15.11+ |
| DeepSeek Harness | [DSH 插件](deepseek-harness.md) | DSH bundle 与生命周期 Hook | DSH 0.1.0-rc.6 |
| Hermes Agent | [Hermes MemoryProvider](hermes.md) | Hermes Memory Provider Hook | Hermes Agent |
| Deep Agents | [Deep Agents 中间件](deepagents.md) | LangChain AgentMiddleware | LangChain Python Deep Agents |
| OpenClaw | [OpenClaw Context Engine](openclaw.md) | OpenClaw Context Engine | OpenClaw |
| QwenPaw | [QwenPaw 插件](qwenpaw.md) | 官方中间件与 canonical Trace | QwenPaw 2.1.x |

## 选择规则

- 自研 Agent：HTTP。
- Python 进程内调用：Python API。
- 客户端支持 MCP：本地用 stdio，远程用 Streamable HTTP，旧客户端用 SSE。
- Claude Code、Codex、OpenCode、DeepSeek Harness、Hermes Agent、OpenClaw、QwenPaw：使用对应插件。
- LangChain Deep Agents：使用 `OGMemoryMiddleware`。

## 共用身份字段

| 字段 | 含义 |
| --- | --- |
| `accountId` | 租户 ID |
| `userId` | 用户 ID |
| `agentId` | Agent ID；必须是服务端已授权的作用域 |
| `sessionId` | 会话 ID；同一会话稳定复用，不同会话隔离 |

启用角色控制后，HTTP 请求头为：

```text
X-API-Key: <tenant-key>
X-Account-ID: <account-id>
X-User-ID: <user-id>
X-Agent-ID: <agent-id>   # MCP 可用；REST 的 agentId 主要放请求体
```

远程服务使用 HTTPS。记忆内容按不可信数据处理，不得覆盖系统指令。
