# oGMemory 通用 Agent Plugin

提供三个工具：`ogm_query_memory` 检索、`ogm_read_memory` 读取详情、
`ogm_write_memory` 写入。随包 `ogmemory-memory` Skill 指导 Agent 主动使用。
本包不自动捕获对话，不接管 Context Engine，也不提供删除、Skill 管理或诊断工具。

## 1. 准备

- Agent 机器安装 Node.js 22+，无需 npm install。
- 独立部署 oGMemory MCP 服务，准备完整 MCP URL、租户密钥和授权身份。
- 如需写入，在服务端开启 `mcp.enable_write_tools`。
- 远程地址使用 HTTPS；本机 loopback 可使用 HTTP。

## 2. 配置

复制 `config.example.json` 为 `~/.ogmemory/agent-plugin.json`，替换所有占位符。
也可将 `OGMEM_AGENT_PLUGIN_CONFIG` 设为私有配置文件的绝对路径。
Windows 默认位置为 `%USERPROFILE%/.ogmemory/agent-plugin.json`。

不要把真实配置放进插件目录。配置优先级为环境变量高于 JSON 文件；
可覆盖变量见 [README.md](README.md)。修改配置后重新连接。
MCP 地址必须包含实际端口和路径，不能用 REST 健康检查地址代替。
同步抽取较慢时将 `timeout_ms` 设为 `120000`，同时增加客户端工具超时。

## 3. 加载

支持 Agent Plugins 1.0 的客户端加载本目录。普通 MCP 客户端配置：

```json
{
  "mcpServers": {
    "ogmemory": {
      "command": "node",
      "args": ["/absolute/path/plugins/agent-plugins/servers/mcp-proxy.mjs"],
      "env": {"OGMEM_AGENT_PLUGIN_CONFIG": "/absolute/path/private/agent-plugin.json"}
    }
  }
}
```

然后通过客户端的 Skill 机制加载 `skills/ogmemory-memory/SKILL.md`，
或将其使用规则加入项目指令。MCP 负责提供工具，Skill 负责指导使用时机。
不要同时启用另一个自动写回同一对话的原生插件。

## 4. 验证

```bash
node /absolute/path/plugins/agent-plugins/scripts/doctor.mjs
```

退出码 0 表示三个工具齐全；2 表示已连接但有工具缺失；1 表示配置或连接失败。
自检不读写记忆。再用一条独特测试偏好完成“保存 → 新会话查询 → 读详情”。
只有服务返回的实际结果能证明写入状态；超时不代表写入失败，不要自动重试。

四个客户端的逐步操作见仓库 [接入教程](../../docs/integrations/agent-plugins.md)。
ZIP 内对应文件是 `CLIENT-GUIDE.md`。
