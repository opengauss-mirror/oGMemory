# Codex 接入 oGMemory

Codex 插件支持连接远端或本机 oGMemory 服务。

## 1. 准备环境

需要以下软件：

- 支持插件的 Codex Desktop 或 Codex CLI
- Python 3.11 或更高版本，用于运行插件 Hook

Windows Codex App 用户按以下顺序操作：准备可访问的 oGMemory 服务 → 在 Windows 创建配置文件 → 在 Windows PowerShell 安装插件 → 完全退出并重开 App → 信任 Hook 并验证跨会话记忆。

插件通过 HTTP 连接服务。Windows App 可连接远端服务，也可连接 WSL2 中的本机服务。第 3 节的服务部署命令在 Linux、WSL2 或 macOS 环境执行；Windows 侧执行第 2、4、5 节。Windows 与 WSL2 的用户主目录和插件安装位置分别独立，配置文件应创建在运行 Codex App 的 Windows 用户目录中。

## 2. 配置远端服务

在用户主目录创建 `.ogmemory/codex-plugin.json`：

```json
{
  "url": "https://memory.example.com",
  "api_key": "<tenant-key>",
  "account_id": "<account-id>",
  "user_id": "<user-id>",
  "agent_id": "codex"
}
```

配置要求：

- `url` 使用 HTTPS，末尾不包含 `/api/v1`
- 未启用鉴权时，`api_key` 使用空字符串
- `account_id`、`user_id` 和 `agent_id` 使用服务端配置的身份

`~/.ogmemory/codex-plugin.json` 中的 `~` 表示当前用户主目录。不要提交包含密钥的配置文件。

### Windows Codex App 配置文件

在 Windows PowerShell 执行：

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.ogmemory" | Out-Null
notepad "$env:USERPROFILE\.ogmemory\codex-plugin.json"
```

粘贴上面的 JSON，替换服务地址、API Key 和身份字段后保存。实际路径为 `C:\Users\<Windows用户名>\.ogmemory\codex-plugin.json`，确认文件扩展名为 `.json`。

如果服务部署在本机 WSL2，`url` 填 Windows 能访问的 HTTP 地址，例如 `http://127.0.0.1:8090`。先在 Windows PowerShell 检查实际端口：

```powershell
Test-NetConnection 127.0.0.1 -Port 8090
```

`TcpTestSucceeded` 应为 `True`；这只验证端口连通，服务状态在 WSL2 中用 `ogmem status` 检查。若端口不通，检查服务启动状态、监听地址和 WSL2 网络设置，再配置可访问地址。

## 3. 部署并配置本机服务

本节在 Linux、WSL2 或 macOS 的终端执行。当前本机生命周期管理使用 `fcntl` 文件锁及 Unix 进程查询；Windows 用户通过 WSL2 部署服务。

安装 Git，克隆并配置 oGMemory：

```text
git clone -b dev https://gitcode.com/opengauss/oGMemory.git
cd oGMemory
python -m pip install -e .
ogmem onboard --mode headless
```

若引导程序中未立即启动服务，执行：

```text
ogmem start headless -d
```

检查服务：

```text
ogmem status
```

预期结果包含：

```text
ContextEngine   http://127.0.0.1:8090  UP
```

命令中的 `python` 指向本机 Python 3.11+ 解释器；若系统命令名为 `python3`，使用 `python3`。

本机默认服务可省略配置文件，插件使用以下默认值：

```json
{
  "url": "http://127.0.0.1:8090",
  "api_key": "",
  "account_id": "acct-demo",
  "user_id": "u-codex",
  "agent_id": "codex"
}
```

也可使用环境变量覆盖配置文件：

| 环境变量 | 配置字段 |
| --- | --- |
| `OG_MEMORY_URL` | `url` |
| `OG_AUTH_API_KEY` | `api_key` |
| `OG_MEMORY_ACCOUNT_ID` | `account_id` |
| `OG_MEMORY_USER_ID` | `user_id` |
| `OG_MEMORY_AGENT_ID` | `agent_id` |

## 4. 安装插件

### Windows Codex App

打开 Windows PowerShell，先检查 Git、Python 和 Codex：

```powershell
git --version
python --version
Get-Command codex -ErrorAction SilentlyContinue
```

Python 应为 3.11+。Windows Hook 使用 `python` 命令；安装 Python 时启用 PATH，并在修改 PATH 后完全退出并重开 Codex App。

若 PowerShell 能找到 `codex`，使用下方通用安装命令。若提示“无法将 codex 项识别为 cmdlet”，可查找 App 随附的可执行文件：

```powershell
$codexBinRoot = Join-Path $env:LOCALAPPDATA 'OpenAI\Codex\bin'
$codexExe = Get-ChildItem -LiteralPath $codexBinRoot -Filter codex.exe -Recurse -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $codexExe) {
    throw '未找到 App 随附的 codex.exe。请安装支持插件的 Codex CLI，重开 PowerShell 后运行通用安装命令。'
}

& $codexExe --version
& $codexExe plugin --help
& $codexExe plugin marketplace add https://gitcode.com/opengauss/oGMemory.git --ref dev --sparse .agents/plugins --sparse plugins/ogmemory
& $codexExe plugin add ogmemory@ogmemory
& $codexExe plugin list
```

上述 App 安装路径取决于安装方式和版本；找不到时安装 [Codex CLI](https://developers.openai.com/codex/cli)。`plugin --help` 应显示插件管理子命令；若版本缺少该命令，升级相应客户端。

CLI 安装与 App 应使用同一 Windows 用户及同一 `CODEX_HOME`（默认 `%USERPROFILE%\.codex`）。插件市场配置和安装结果写入该 Codex 用户目录，App 重启后加载。

### 通用安装命令

添加仓库插件市场并安装 `ogmemory`：

```text
codex plugin marketplace add https://gitcode.com/opengauss/oGMemory.git --ref dev --sparse .agents/plugins --sparse plugins/ogmemory
codex plugin add ogmemory@ogmemory
codex plugin list
```

预期插件列表包含：

```text
ogmemory@ogmemory
```

重启 Codex。打开 `/hooks`，检查并信任 `ogmemory` 的 11 个生命周期 Hook。

Windows App 用户完全退出并重新启动 App，再打开新任务执行 `/hooks`。如果入口不可用，检查 App 版本是否支持文档所需的插件及 Hook 功能。

### Windows 常见问题

| 现象 | 处理 |
| --- | --- |
| PowerShell 提示找不到 `codex` | 使用上面的 App 可执行文件查找命令，或安装 Codex CLI |
| `plugin` 子命令不可用 | 升级所使用的 Codex CLI 或 App |
| Hook 提示找不到 `python` | 安装 Python 3.11+ 并加入 Windows PATH，完全退出并重开 App |
| CLI 列出插件，App 尚未加载 | 核对 Windows 用户、`CODEX_HOME`，重启 App 并检查 `/hooks` |
| Hook 已加载，记忆未召回 | 检查 Hook 信任、服务连通性、配置文件路径及身份字段，再执行第 5 节验证 |

## 5. 验证跨会话记忆

在第一个 Codex 任务中输入：

```text
请记住：Codex 插件验收码是 OG-CODEX-2026。只回复“已记录”。
```

预期回复：

```text
已记录
```

该轮完成后，新建 Codex 任务并输入：

```text
Codex 插件验收码是什么？只回复验收码。
```

预期回复：

```text
OG-CODEX-2026
```

若结果不符，检查 `/hooks` 信任状态、服务地址、API Key 和身份字段。本机部署还可检查：

```text
ogmem status
ogmem logs --service ogmem
```

## 自动记忆控制

`OGMEM_AUTO_CAPTURE` 和 `OGMEM_AUTO_RECALL` 默认均为 `true`，分别控制自动写入和自动召回。
也可在 `~/.ogmemory/codex-plugin.json` 配置 `"auto_capture": false` 和 `"auto_recall": true`；环境变量优先。
值忽略大小写和首尾空格，`1`、`true`、`yes` 表示开启，其余显式值表示关闭。修改后重启 Codex。

```powershell
$env:OGMEM_AUTO_CAPTURE='false'
$env:OGMEM_AUTO_RECALL='true'
```

PowerShell 的 `$env:` 设置只影响当前终端及其后启动的子进程。从开始菜单启动 Codex App 时，建议将 `auto_capture`、`auto_recall` 写入 Windows 用户的 `codex-plugin.json`，然后完全退出并重开 App；App 继承的环境变量仍优先于配置文件。

关闭自动写入后，不缓存新提示词、不写入完成轮次，也不调用 bootstrap、压缩、子代理写入和 dispose 接口。
自动召回仍可运行，Trace 仍独立上报；此开关不用于关闭 Trace 内容采集。
已有提示词和压缩缓存保留，重新开启后同一轮次或压缩生命周期仍可能提交这些数据。
两个开关均关闭时，只保留 Trace 上报。

## 生命周期映射

| Codex 阶段 | oGMemory 接口 | 用途 |
| --- | --- | --- |
| `SessionStart` | `/bootstrap`、`/compose` | 初始化会话并注入相关记忆 |
| `UserPromptSubmit` | `/compose` | 记录用户输入并召回当前任务记忆 |
| `PreToolUse` | `/compose` | 记录工具意图并召回相关约束 |
| `PermissionRequest` | `/trace/events` | 记录权限申请上下文 |
| `PostToolUse` | `/trace/events` | 记录有界的工具输入与输出 |
| `PreCompact` | `/prepare_compaction` | 准备提取和压缩状态 |
| `PostCompact` | `/compact` | 提交压缩状态 |
| `SubagentStart` | `/prepare_subagent_spawn`、`/compose` | 初始化子代理并注入相关记忆 |
| `SubagentStop` | `/on_subagent_ended` | 记录子代理结果并结束生命周期 |
| `Stop` | `/after_turn` | 持久化完整用户与助手轮次 |
| `SessionEnd` | `/dispose` | 刷新待处理记忆并释放会话 |

所有阶段同时写入 `/api/v1/trace/events`。轨迹内容有长度上限，并对常见凭据字段脱敏。Hook 失败时不会阻塞 Codex；注入的记忆作为不可信上下文处理。
