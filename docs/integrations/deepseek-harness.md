# DeepSeek Harness 对接

oGMemory 插件为 DeepSeek Harness（DSH）提供自动记忆召回、对话写回和跨会话记忆能力。

在浏览器中使用插件可按第 1–5 节操作。在命令行运行一次性任务可按
[Headless](#headless) 完成安装、配置、跨会话验证和卸载。

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

## 2. 安装 DSH 并初始化 Web UI

安装 Git、Node.js `22.19.0+` 或 `24.0.0+`，以及 pnpm。已有 DSH 环境可通过
`dsh --version` 查看版本，并在第 3 步选择对应的插件。新安装 `0.1.5-rc.2` 的示例如下。

```bash
npm install -g pnpm
npm install -g @deepseek-ai/dsh@0.1.5-rc.2
git --version
pnpm --version
dsh --version
```

DSH CLI 和核心包 `0.1.5-rc.2` 对应插件 `0.1.1`，`0.2.1-rc.2` 对应插件 `0.2.1`。

启动 Web UI：

```bash
dsh --profile web
```

在浏览器中打开 DSH 显示的地址，按照引导配置模型提供方并完成一次普通对话。完成后按 `Ctrl+C` 结束进程。

## 3. 将 oGMemory 插件安装到 Web profile

DSH `0.1.5-rc.2` 直接从仓库安装插件 `0.1.1`。

```bash
dsh plugin --profile web add https://gitcode.com/opengauss/oGMemory.git#dev
```

DSH `0.2.1-rc.2` 使用插件 `0.2.1` 的安装包。在同一 `dev` 分支的本地仓库中，
使用 Node.js 和 `tar` 构建新版安装包，然后用目标 DSH 的 CLI 安装。

```bash
cd plugins/deepseek-harness
node scripts/pack.mjs 0.2.1-rc.2
dsh plugin --profile web add ./dist/opengauss-ogmemory-dsh-plugin-0.2.1.tgz
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

## 4. 配置连接并启动 Web UI

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

## 5. 在 Web UI 体验跨会话记忆

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

## Headless

以下 Bash 示例使用 DSH `0.2.1-rc.2` 和插件 `0.2.1`。先配置可用的模型提供方，
并准备第 1 节所列的 oGMemory 地址、Tenant Key 和身份信息。示例中的 `dsh` 指目标安装的 CLI。
使用 AgentBench 时，用同一安装目录的 `node_modules/.bin/dsh` 管理插件和查看配置，
用任务包装器启动下面的两次 headless 任务。安装、验证和运行都使用相同的 `DSH_HOME`。

**安装并检查配置。** 在仓库根目录中构建安装包，再将其加入 `headless` profile。

```bash
node plugins/deepseek-harness/scripts/pack.mjs 0.2.1-rc.2
dsh plugin --profile headless add ./plugins/deepseek-harness/dist/opengauss-ogmemory-dsh-plugin-0.2.1.tgz
dsh --profile headless --dump-config | grep -F ogmemory-lifecycle
```

输出中应出现 `ogmemory-lifecycle`。如果使用 DSH `0.1.5-rc.2`，安装命令为
`dsh plugin --profile headless add https://gitcode.com/opengauss/oGMemory.git#dev`。

**配置连接。** 在运行任务的同一终端中设置以下变量，替换服务地址和身份示例值。
按提示输入该租户的 oGMemory Tenant Key。

```bash
export OGMEM_URL='https://memory.example.com'
export OGMEM_ACCOUNT_ID='customer-a'
export OGMEM_USER_ID='alice'
export OGMEM_AGENT_ID='customer-a-api-agent'
read -rsp 'oGMemory Tenant Key: ' OGMEM_DSH_API_KEY && echo
export OGMEM_DSH_API_KEY
export OGMEM_AUTO_CAPTURE=true OGMEM_AUTO_RECALL=true
```

**验证跨会话记忆。** 第一次运行生成一个新的测试项目和随机口令，并请模型记住它。

```bash
project=$(node -p "require('node:crypto').randomUUID()")
marker=$(node -p "require('node:crypto').randomUUID()")
dsh --profile headless "请记住：项目 ${project} 的验收口令是 ${marker}。只回复已记住。"
```

等 oGMemory 完成本轮写回、提取和索引，再运行第二条任务。两次独立启动均省略
`--session-id`，DSH 会创建两个不同的会话。

```bash
dsh --profile headless "项目 ${project} 的验收口令是什么？只回复口令。"
printf '预期口令：%s\n' "$marker"
```

第二次输出应包含本次生成的口令。在第二个会话的轨迹中确认 `<ogmemory-context>`
包含同一口令，即可核实本次回答使用了插件召回的内容。

**卸载。** 如需移除本次安装，使用同一个 CLI 和 profile。

```bash
dsh plugin --profile headless remove @opengauss/ogmemory-dsh-plugin
```
