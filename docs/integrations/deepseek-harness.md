# DeepSeek Harness 对接

oGMemory 插件为 DeepSeek Harness（DSH）提供自动记忆召回、对话写回和跨会话记忆能力。

## 1. 准备 oGMemory 服务

在服务器上按照 [oGMemory 一键部署与体验指南](../../deploy/user-quickstart/README_zh.md) 完成部署，并通过 HTTPS 发布 HTTP API。

准备以下连接信息：

| 参数 | 示例 | 用途 |
| --- | --- | --- |
| oGMemory 地址 | `https://memory.example.com` | HTTP API 基址 |
| Tenant Key | `ogm_xxx` | API 鉴权 |
| Account ID | `customer-a` | 租户标识 |
| User ID | `alice` | 记忆归属 |
| Agent ID | `customer-a-api-agent` | oGMemory 中已注册的 Agent 身份 |

访问健康检查接口：

```bash
curl -fsS https://memory.example.com/api/v1/health
```

响应中的 `status` 为 `ok` 后继续安装 DSH。

## 2. 安装并初始化 DSH

安装 Git、Node.js `22.19.0+` 或 `24.0.0+`，以及 pnpm，然后安装插件适配的 DSH 版本：

```bash
npm install -g pnpm
npm install -g @deepseek-ai/dsh@0.1.5-rc.2
git --version
pnpm --version
dsh --version
```

插件 `0.1.1` 支持 DSH CLI `0.1.5-rc.2`、核心包 `0.1.5-rc.2`、Cordis `4.0.2`，
提供已完成对话的自动写回和跨会话记忆召回。

启动 Web UI：

```bash
dsh --profile web
```

在浏览器中打开 DSH 显示的地址，按照引导配置模型提供方并完成一次普通对话。完成后按 `Ctrl+C` 结束进程。

## 3. 安装 oGMemory 插件

将插件安装到 DSH 的 `web` profile：

```bash
dsh plugin --profile web add https://gitcode.com/opengauss/oGMemory.git#dev
```

查看 profile 配置：

```bash
dsh --profile web --dump-config
```

配置中应包含：

```text
# == @opengauss/ogmemory-dsh-plugin
id: ogmemory-lifecycle
name: '@opengauss/ogmemory-dsh-plugin'
```

## 4. 配置连接

在启动 DSH 的运行环境中设置：

```bash
export OGMEM_URL='https://memory.example.com'
export OGMEM_DSH_API_KEY='ogm_xxx'
export OGMEM_ACCOUNT_ID='customer-a'
export OGMEM_USER_ID='alice'
export OGMEM_AGENT_ID='customer-a-api-agent'

dsh --profile web
```

Windows PowerShell：

```powershell
$env:OGMEM_URL='https://memory.example.com'
$env:OGMEM_DSH_API_KEY='ogm_xxx'
$env:OGMEM_ACCOUNT_ID='customer-a'
$env:OGMEM_USER_ID='alice'
$env:OGMEM_AGENT_ID='customer-a-api-agent'

dsh --profile web
```

长期运行时，将这些变量写入 DSH 启动服务的环境配置。

同一用户在不同会话中使用相同的 Account ID 和 User ID。不同用户使用不同的 User ID。

## 5. 体验跨会话记忆

体验前确认自动采集与召回均开启。如果 DSH 已运行，先按 `Ctrl+C` 退出，设置环境变量后重新启动：

```bash
export OGMEM_AUTO_CAPTURE=true
export OGMEM_AUTO_RECALL=true
dsh --profile web
```

在 DSH 页面新建会话，发送：

```text
请记住：我的项目发布窗口是每周四 20:00。只回复“已记住”。
```

回复后保持 DSH 运行，插件会在后台自动写回、提取并建立索引。
建议先等待 1–2 分钟，实际处理耗时取决于模型服务和索引速度。

保持相同的 Account ID、User ID 和 Agent ID，新建会话并发送：

```text
我的项目发布窗口是什么时候？只回复时间。
```

应回答 `每周四 20:00`。展开回复的过程区，查看
`上下文注入 ogmemory-lifecycle`（英文界面为 `Context injection ogmemory-lifecycle`）。
也可在 `Trajectory` 标签页查看注入的 `<ogmemory-context>` 内容。

同时满足“新会话准确回答发布窗口”和“出现 oGMemory 上下文注入”，即表示跨会话记忆验证通过。
后台仍在处理时，继续等待，随后另建新会话提问。重复测试时使用新的项目名称和发布窗口。

## 6. 插件运行流程

| DSH 阶段 | oGMemory 行为 |
| --- | --- |
| 回答问题前 | 自动召回并注入相关记忆 |
| 对话轮次完成 | 后台保存对话，提取记忆并建立索引 |
| 会话结束 | 完成待写入对话的提交并清理会话状态 |

安装并配置插件后，用户正常对话即可使用自动记忆与跨会话召回。

## 7. Trace 采集

插件默认在 `turn/end` 后读取本轮持久化事件，按顺序上报到
`/api/v1/trace/events`：用户消息、助手消息、工具开始、工具成功或失败，以及
`turn_end`。会话开始和结束分别上报 `session_start`、`session_end`。
工具调用通过 `tool_use_id` 关联；服务端在 `turn_end` 汇总本轮 Trace。

Trace 采用独立持久化队列，保存在会话身份目录下的 `traces` 子目录，按会话
顺序发送并重试；恢复后复用事件 ID。消息和工具内容上传前脱敏，每个内容字段
最多保留 10,000 字符。

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `traceEnabled` / `OGMEM_DSH_TRACE` | `true` | Trace 采集开关 |
| `traceCaptureContent` / `OGMEM_DSH_TRACE_CAPTURE_CONTENT` | `true` | 关闭后采集事件结构和内容长度 |
| `traceTimeoutMs` | `3000` | 单次上报超时，单位毫秒 |

`captureToolResults` 控制工具文本进入记忆写回；Trace 工具事件由上述 Trace 配置控制。

## 8. 自动记忆控制（可选）

若只需召回已有记忆，退出当前 DSH 后设置以下变量并重新启动：

```bash
export OGMEM_AUTO_CAPTURE=false
export OGMEM_AUTO_RECALL=true
dsh --profile web
```

两个开关默认均开启。`OGMEM_AUTO_CAPTURE` 控制自动写回和会话 dispose；关闭时不启动记忆队列，
已有待写入文件保留，不会在后台重放，重新开启后恢复处理。`OGMEM_AUTO_RECALL` 独立控制自动召回。
配置字段分别为 `autoCapture`、`autoRecall`，环境变量优先；`1`、`true`、`yes`（忽略大小写和首尾空格）表示开启，其余显式值关闭。
原有 `writebackEnabled`、`composeEnabled`、`disposeEnabled` 仍生效，需要对应开关同时开启才执行。
修改环境后重启 DSH。Trace 以及显式工具保持独立，写入工具仍由 `writeToolsEnabled` 控制。

关闭采集后，新对话不会自动写入，延长等待也不会生成新记忆。再次执行第 5 节跨会话体验时，
须将两个变量重新设为 `true` 并重启 DSH，然后发送新的记忆内容。
