# oGMemory OpenCode 插件安装

## 前置条件

- OpenCode 1.15.11 或更高版本
- Node.js 18 或更高版本
- 已启动 oGMemory HTTP 服务，默认地址为 `http://127.0.0.1:8090`

## npm 安装

在 `%USERPROFILE%\.config\opencode\opencode.json` 中加入：

```json
{
  "plugin": ["@opengauss/ogmemory-opencode-plugin"]
}
```

OpenCode 会自动安装配置中的 npm 插件。修改后重启 OpenCode。

## 源码安装（Windows PowerShell）

在 oGMemory 仓库根目录执行：

```powershell
$target = Join-Path $env:USERPROFILE ".config\opencode\plugins"
New-Item -ItemType Directory -Force (Join-Path $target "ogmemory") | Out-Null
Copy-Item "plugins/opencode\wrappers\ogmemory.js" (Join-Path $target "ogmemory.js")
Copy-Item "plugins/opencode\index.mjs","plugins/opencode\package.json" (Join-Path $target "ogmemory")
Copy-Item -Recurse "plugins/opencode\lib" (Join-Path $target "ogmemory\lib")
```

最终目录为：

```text
%USERPROFILE%\.config\opencode\plugins\
├── ogmemory.js
└── ogmemory\
    ├── index.mjs
    ├── package.json
    └── lib\
```

## 配置

创建 `%USERPROFILE%\.config\opencode\ogmemory-config.json`：

```json
{
  "enabled": true,
  "autoCapture": true,
  "autoRecall": true,
  "endpoint": "http://127.0.0.1:8090",
  "apiKey": "",
  "accountId": "",
  "userId": "",
  "agentId": "opencode",
  "traceCaptureContent": true
}
```

可以使用 `OG_MEMORY_URL`、`OG_AUTH_API_KEY`、`OG_AUTH_ACCOUNT_ID`、`OG_AUTH_USER_ID` 和 `OG_AUTH_AGENT_ID` 环境变量覆盖文件配置。设置 `OGMEM_TRACE_CAPTURE_CONTENT=0` 后，Trace 只记录元数据。

本机回环地址可以使用 HTTP；远程 oGMemory 服务必须使用 HTTPS。将 `enabled` 设为 `false` 会禁用所有 Hook 和网络请求。

只读取已有记忆、不自动写入新对话时，将 `autoCapture` 设为 `false`，保留 `enabled=true` 和 `autoRecall=true`。关闭 capture 后不创建消息写入缓存，也不会在轮次完成、空闲、压缩、错误、会话删除或退出时调用 `/api/v1/after_turn`。`autoRecall=false` 则关闭自动读取及系统提示词注入，不影响自动写入。

两个开关默认均为 `true`。未在 JSON 中配置时可使用 `OGMEM_AUTO_CAPTURE` 和 `OGMEM_AUTO_RECALL` 环境变量；显式 JSON 布尔值优先于环境变量。环境变量值 `0`、`false`、`off`、`no`（不区分大小写）表示关闭。修改配置后重启 OpenCode。

这些开关控制原生插件的自动记忆流程。会话、工具、压缩和错误 Trace 仍会上传；关闭 capture 后，依赖消息缓存的 `user_message`、`assistant_message` 和 `turn_end` Trace 不再生成。显式 MCP 记忆工具仍保留各自的读写行为。

## 行为边界

插件直接消费 OpenCode 生命周期事件，不读取或分页扫描 OpenCode transcript。尚未成功写入的消息只在当前 OpenCode 进程内存中保留，并在下一个轮次边界重试。Trace 异步发送，不阻塞用户消息或工具 Hook；发送失败会记录脱敏告警，但首版不提供磁盘队列、跨进程锁或崩溃恢复。

子 Agent 会话保持独立的 session ID；当 OpenCode 提供父会话信息时，`session_start` Trace 会携带 `parent_session_id`。

## 验证

启动 OpenCode 后完成一轮对话，并确认：

1. oGMemory 收到 `/api/v1/compose`。
2. 轮次结束后收到 `/api/v1/after_turn`，且同时包含 user 和 assistant。
3. `/api/v1/trace/events` 中存在 `user_message`、`assistant_message` 和 `turn_end`。
4. 使用工具时存在 `tool_start` 与终态工具事件。
