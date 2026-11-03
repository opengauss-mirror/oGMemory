# Agent 接入配置生成

`ogmem onboard --mode plugin` 为已经安装的 Agent 生成 oGMemory 接入配置。
oGMemory 服务由 local 或 docker 流程独立部署。生成器只依赖 Python 标准库，
不要求本机安装目标 Agent、数据库驱动或后端模型依赖，也不访问网络。

## 使用

```bash
# OG_AUTH_API_KEY 应预先设置为服务管理员提供的 oGMemory 租户密钥
ogmem onboard --mode plugin --agent openclaw --non-interactive \
  --ogmem-url https://memory.example.com \
  --ogmem-api-key-env OG_AUTH_API_KEY \
  --account-id acct-1 --user-id user-1 --agent-id openclaw

# 已有 Agent 没有原生适配器时，生成 Streamable HTTP MCP 配置
ogmem onboard --mode plugin --agent my-agent --non-interactive \
  --mcp-url https://memory.example.com:18091/mcp \
  --ogmem-api-key-env OG_AUTH_API_KEY \
  --account-id acct-1 --user-id user-1 --agent-id my-agent
```

不传 `--non-interactive` 时可使用向导。显式 `--mode`、`--agent` 和连接参数优先，
只询问缺失的参数。未指定 Agent 的非交互 plugin 模式默认使用 OpenClaw。

## 支持范围与产物

| `--agent` | 产物 | 在 Agent 机器上的应用位置 |
| --- | --- | --- |
| `openclaw` | `openclaw.config.patch.json` | 合并到当前 profile 的 OpenClaw 配置，只包含 plugins 字段 |
| `claude-code` / `claude` | `claude.settings.patch.json` | 合并 env 字段到项目或用户的 Claude settings |
| `codex` | `codex-plugin.json` | `~/.ogmemory/codex-plugin.json` 或 `OG_MEMORY_CODEX_CONFIG` |
| `opencode` | `opencode.config.patch.json`、`ogmemory-config.json` | 追加插件声明，并配置 `~/.config/opencode/ogmemory-config.json` |
| `deepseek-harness` / `dsh` | `ogmemory.env.sh`、`ogmemory.env.ps1` | 在启动对应 DSH profile 的进程环境加载 |
| `hermes` | `hermes.config.patch.yaml`、两种环境脚本 | 合并 memory.provider 配置，设置启动 Hermes 的进程环境 |
| `deepagents` / `deepagent` | Python 接入示例、两种环境脚本 | 传给 `create_deep_agent(middleware=[...])`，见 [Deep Agents](deepagents.md) |
| `mcp` 或任意未适配名称 | `mcp.config.json` | 支持 HTTP MCP 和自定义 Headers 的客户端设置 |
| `agent-plugins` | `agent-plugin.json` | `~/.ogmemory/agent-plugin.json` 或 `OGMEM_AGENT_PLUGIN_CONFIG`，搭配通用插件包 |

每份导出都包含 `README.md`，说明插件获取、配置应用和验证步骤。
原生插件代码必须已安装或按说明加载，JSON 配置本身不能代替插件代码。
通用插件同样要求 `--mcp-url`，只提供记忆检索、详情读取和写入；
四个客户端的配置及验证步骤见 [通用 Agent Plugin](agent-plugins.md)。
不同 MCP 客户端的设置结构可能不同；导出的 URL 和 Headers 可以用于其对应设置。
MCP 工具接入不承诺自动召回、轮次写回或压缩接管，这些能力需要原生生命周期适配。

## 参数与地址

- `--ogmem-url`：Agent 可访问的 HTTP API 基址，不含 `/api/v1`。默认 `http://127.0.0.1:8090`；`--port` 可调整默认端口。
- `--mcp-url`：完整的 Streamable HTTP MCP 地址，包含实际端口和路径。非交互 MCP 导出必须指定，不能从 REST 地址推断。
- `--ogmem-api-key` / `--ogmem-api-key-env`：服务鉴权二选一。`--api-key` 是后端模型密钥，不用于插件接入。
- `--account-id`、`--user-id`、`--agent-id`：使用服务端授权的身份。未指定时分别为 `acct-demo`、`u-alice`、适配器名称，仅用于默认开发环境。
- `--output-dir`：默认 `./ogmem-client-config/<adapter>/`。重复导出相同内容可直接复用；已有不同内容时拒绝覆盖，请选择新的导出目录。

HTTP API 和 MCP 可以使用不同端口，例如本机 API 8090、MCP 8091，或 Docker 映射后的 API 18090、MCP 18091。
`127.0.0.1` 指向 Agent 所在的机器/容器。导出可以在服务端生成，再传到 Agent 机器应用。
OpenCode 和 DeepSeek Harness 的原生适配器要求远程地址使用 HTTPS，生成器会检查这一约束。

原生本地开发服务关闭鉴权时可不填服务密钥；HTTP MCP 需要开启角色控制并使用租户密钥。
未填 MCP 密钥时会生成 `<OGMEM_TENANT_API_KEY>` 占位符，应用前必须替换。
导出会包含提供的密钥，文件在 POSIX 下以 0600 创建，默认导出目录已加入 gitignore。

## 完成与验证

导出成功只表示配置文件已生成。安装状态、激活状态和连接状态均标记为未验证。

1. 在 Agent 机器安装/加载对应插件，按导出的 README 合并配置；保留现有的模型、网关、其它插件设置。
2. OpenClaw 的 contextEngine 槽位和 Hermes 的 memory provider 有唯一归属，应用前检查已有配置。
3. 由用户按现有运行方式重载 Agent。Plugin 模式没有专用的 `ogmem start/stop` 命令，生成器也不会启动或停止 Agent。
4. 原生插件检查 HTTP API 及一次真实对话的召回/写回；MCP 检查初始化、鉴权、工具列表和一次查询。
5. 如需部署服务，另行使用 local/docker 流程。接入配置不会修改 `config/ogmem.yaml`。
