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
| DeepSeek Harness | [DSH 插件](deepseek-harness.md) | DSH bundle 与生命周期 Hook | DSH 0.1.5-rc.2 |
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

## 无向量模式

在服务端的 `config/ogmem.yaml` 或 `OGMEM_CONFIG` 指定的文件中，从下面两种倒排配置中选一种，更新 `storage` 和 `retrieval` 下的对应字段。两种配置均由 AGFS 保存记忆原文，保留现有 `agfs` 配置和会话抽取所需的 `llm`，可省略 `embedding` 和 `vector_db`。各 Agent 共用同一套服务端配置。

### AGFS + 内置 SQLite FTS5

```yaml
storage:
  backend: agfs
retrieval:
  mode: vector_free
  hybrid_enabled: false
  inverted_index:
    provider: sqlite_fts
  agfs_fts_ranker: literal
```

各服务进程按访问范围构建临时 FTS5 索引，覆盖 L0、L1、L2 和部分节点元数据，无需配置 SQLite 文件路径。默认由 AGFS 字面评分，设 `agfs_fts_ranker: sqlite_bm25` 可改用 FTS5 的 BM25 排序。

### AGFS + openGauss 原生 BM25

```yaml
storage:
  backend: agfs
retrieval:
  mode: vector_free
  hybrid_enabled: false
  inverted_index:
    provider: opengauss_native
    options:
      connection_string: "host=DB_HOST port=15432 dbname=ogmemory user=gaussdb password=CHANGE_ME"
```

将连接串改成服务可访问的 openGauss 地址。目标库须支持原生 BM25，账号须有建表和建索引权限。倒排持久化在 openGauss，查询命中后从 AGFS 读取原文并校验权限。索引回填完成前，服务使用 AGFS 字面扫描并标记结果为 `partial`。

**仅用 AGFS 字面扫描**时，将 `retrieval.inverted_index.provider` 显式设为 `none`，省略 `options`。服务不创建或更新倒排索引，查询直接扫描 AGFS。省略 provider 时默认使用 `sqlite_fts`。

配置支持 `native` 和 `http` 的 AGFS 存储。会话写入仍由 `after_turn` 抽取并归档，现有写权限继续适用。`compose`、`prefetch`、REST 记忆 API、Python `MemoryService` 和 MCP 工具都使用服务端的检索配置。配置项见[记忆检索模式](../OGMEMORY_ENV.md#记忆检索模式)。

独立部署可连接现有 AGFS，或用 `standalone-agfs` 镜像启动内置 AGFS。同一存储的服务进程与 Outbox Worker 应读取相同配置，修改后重启。切回向量模式时，为无向量模式下新增或更新的记忆重建向量索引。

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
