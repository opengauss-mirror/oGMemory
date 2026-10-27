# QwenPaw 对接

oGMemory 通过 QwenPaw 官方插件接口提供自动召回、对话写回、跨会话记忆、主动检索和 Agent Trace 采集。适用于 QwenPaw 2.1 起提供的 `PluginApi` 与 AgentScope 中间件接口，后续版本沿用这些接口时可继续使用。

已验证 QwenPaw 2.1.0（AgentScope 2.0.4.post1）和 QwenPaw 2.2.2.post1（AgentScope 2.0.10）。安装命令使用满足最低版本要求的当前版本。

以下从首次安装开始，使用独立 Python 环境和明确的数据目录。Linux、macOS、WSL 使用 Bash 命令；Windows 使用 PowerShell 命令。

## 1. 准备运行环境和连接信息

安装 Python 3.12 和 Git，可分别从 [Python 官网](https://www.python.org/downloads/)与 [Git 官网](https://git-scm.com/downloads)获取。本文命令使用 Python 3.12；其他 Python 版本按所安装 QwenPaw 的要求选择。

先验证 Python。Linux、macOS、WSL：

```bash
python3.12 --version
git --version
```

Windows PowerShell：

```powershell
py -3.12 --version
git --version
```

准备两组连接信息，分别用于模型调用和记忆服务：

| 连接 | 需要准备 | 获取方式 |
| --- | --- | --- |
| 模型提供方 | 模型名称、API Key；自定义提供方还需要 Base URL | 向模型服务提供方申请，在 QwenPaw 初始化向导中填写 |
| oGMemory | 服务地址、Tenant Key、Account ID、User ID、Agent ID | 按[部署与体验指南](../../deploy/user-quickstart/README_zh.md)部署服务，或向服务管理员获取 |

oGMemory 可以部署在另一台机器上，服务地址应能从运行 QwenPaw 的机器访问。本文使用以下示例值；配置时替换为实际值：

| 参数 | 示例 | 用途 |
| --- | --- | --- |
| oGMemory 地址 | `https://memory.example.com` | HTTP API 基址 |
| Tenant Key | `ogm_xxx` | oGMemory API 鉴权 |
| Account ID | `customer-a` | 租户标识 |
| User ID | `alice` | 记忆归属 |
| Agent ID | `customer-a-api-agent` | 服务端已授权的 Agent 身份 |

## 2. 安装并初始化 QwenPaw

创建工作目录、Python 虚拟环境，并指定 QwenPaw 数据目录。

Linux、macOS、WSL：

```bash
mkdir -p "$HOME/qwenpaw-ogmemory"
cd "$HOME/qwenpaw-ogmemory"
python3.12 -m venv .venv
source .venv/bin/activate
export QWENPAW_WORKING_DIR="$PWD/data"
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$HOME\qwenpaw-ogmemory" | Out-Null
Set-Location "$HOME\qwenpaw-ogmemory"
py -3.12 -m venv .venv
$env:Path = "$(Join-Path $PWD '.venv\Scripts');$env:Path"
$env:QWENPAW_WORKING_DIR = Join-Path $PWD 'data'
```

在上述终端继续执行，两个平台命令相同：

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade "qwenpaw>=2.1.0" "httpx>=0.27.0"
qwenpaw --version
qwenpaw init
```

`qwenpaw --version` 用于确认实际安装版本。初始化向导开头会打印 `Working dir`，应指向刚设置的 `qwenpaw-ogmemory/data`。按向导配置模型提供方、API Key 和模型；自定义端点还需填写 Base URL。需要重新配置模型时，执行 `qwenpaw models config`；查看配置使用 `qwenpaw models list`。

初始化后，工作目录中包含：

```text
qwenpaw-ogmemory/
├── .venv/                      # Python 环境和 qwenpaw 命令
├── data/                       # QWENPAW_WORKING_DIR
│   ├── config.json             # 全局配置、Agent 路径和插件配置
│   └── workspaces/
│       └── default/
│           └── agent.json      # 默认 Agent 的配置
└── data.secret/                # QwenPaw 凭据目录
```

## 3. 获取并安装 oGMemory 插件

保持终端位于第 2 节创建的 `qwenpaw-ogmemory` 目录，克隆 oGMemory 源码。以下命令在 Bash 和 PowerShell 中相同：

```bash
git clone --depth 1 --branch dev https://gitcode.com/opengauss/oGMemory.git oGMemory
qwenpaw plugin install ./oGMemory/plugins/qwenpaw
qwenpaw plugin list
qwenpaw plugin info ogmemory
```

`--branch dev` 指定源码版本；使用其他已包含此插件的发布标签或分支时，将 `dev` 替换为对应名称。目录结构如下：

```text
qwenpaw-ogmemory/
├── .venv/
├── data/
└── oGMemory/
    ├── docs/
    └── plugins/
        └── qwenpaw/
            ├── __init__.py
            ├── plugin.json
            ├── plugin.py
            └── client.py
```

安装输出应包含 `installed successfully`。列表中应出现 ID 为 `ogmemory` 的插件。安装命令将源码复制到 `QWENPAW_WORKING_DIR` 下，本文对应 `qwenpaw-ogmemory/data/plugins/ogmemory/`；实际目标路径以安装输出的 `Location` 为准。

## 4. 配置 oGMemory 连接和 Agent

### 4.1 配置服务连接

用文本编辑器打开 `qwenpaw-ogmemory/data/config.json`，在 JSON 顶层合并以下 `plugins` 配置，并将五个示例值替换为第 1 节取得的实际值。已有 `plugins` 对象时，在其中添加 `ogmemory`；保留其他配置项，并检查相邻字段之间的逗号。

```json
{
  "plugins": {
    "ogmemory": {
      "endpoint": "https://memory.example.com",
      "api_key": "ogm_xxx",
      "account_id": "customer-a",
      "user_id": "alice",
      "agent_id": "customer-a-api-agent"
    }
  }
}
```

保存为 UTF-8 JSON。同一用户的不同会话保持相同 `account_id` 和 `user_id`。以上示例适用于单用户；多用户服务可省略 `user_id`，由插件使用 QwenPaw 当前请求的用户 ID。

### 4.2 配置 Agent 的记忆与上下文策略

首次初始化的默认 Agent 配置文件为 `qwenpaw-ogmemory/data/workspaces/default/agent.json`。打开该文件，在已有 `running` 对象中设置：

```json
{
  "running": {
    "memory_manager_backend": "none",
    "context_manager_backend": "light"
  }
}
```

保留 `running` 中其他字段。长期记忆交给 oGMemory，本地上下文裁剪和压缩使用 QwenPaw 的上下文策略。

使用其他 Agent 时，打开 `data/config.json`，在 `agents.profiles` 中找到目标 Agent，使用其 `workspace_dir` 下的 `agent.json`。例如：

```json
{
  "agents": {
    "active_agent": "default",
    "profiles": {
      "default": {
        "id": "default",
        "workspace_dir": "/actual/path/to/qwenpaw-ogmemory/data/workspaces/default"
      }
    }
  }
}
```

此段仅说明如何定位文件；`workspace_dir` 使用初始化生成的实际路径。

### 4.3 检查配置和连接

在第 2 节的终端执行以下跨平台命令，确认配置文件是有效 JSON：

```bash
python -m json.tool ./data/config.json
python -m json.tool ./data/workspaces/default/agent.json
```

从配置中读取服务地址，访问健康检查接口：

```bash
python -c "import json, pathlib, urllib.request; p=pathlib.Path('data/config.json'); c=json.loads(p.read_text(encoding='utf-8')); u=c['plugins']['ogmemory']['endpoint'].rstrip('/')+'/api/v1/health'; print(urllib.request.urlopen(u, timeout=10).read().decode())"
```

响应中 `status` 应为 `ok`。API Key、Account ID 和 Agent 授权是否正确，还需通过第 6 节的实际对话验证。

### 4.4 使用环境变量配置连接

需要通过进程环境注入连接信息时，可用以下方式代替第 4.1 节的插件配置。仍需完成第 4.2 节的 Agent 配置。

Linux、macOS、WSL：

```bash
export OG_MEMORY_URL="https://memory.example.com"
export OG_AUTH_API_KEY="ogm_xxx"
export OG_AUTH_ACCOUNT_ID="customer-a"
export OG_AUTH_USER_ID="alice"
export OG_AUTH_AGENT_ID="customer-a-api-agent"
```

Windows PowerShell：

```powershell
$env:OG_MEMORY_URL = "https://memory.example.com"
$env:OG_AUTH_API_KEY = "ogm_xxx"
$env:OG_AUTH_ACCOUNT_ID = "customer-a"
$env:OG_AUTH_USER_ID = "alice"
$env:OG_AUTH_AGENT_ID = "customer-a-api-agent"
```

在同一终端启动 QwenPaw，进程才能继承这些变量。环境变量优先于 `config.json` 的插件配置。使用此方式检查健康状态时，在浏览器打开实际服务地址的 `/api/v1/health`，确认响应中的 `status` 为 `ok`。

## 5. 启动 QwenPaw

在已设置虚拟环境和 `QWENPAW_WORKING_DIR` 的终端执行：

```bash
qwenpaw app --host 127.0.0.1 --port 8088
```

保持该终端运行，在同一机器的浏览器打开 [http://127.0.0.1:8088](http://127.0.0.1:8088)，进入默认 Agent 的对话页面。先发送一条普通消息，确认模型可以正常回复。

以后新开终端时，先恢复 Python 环境和数据目录，再启动服务。

Linux、macOS、WSL：

```bash
cd "$HOME/qwenpaw-ogmemory"
source .venv/bin/activate
export QWENPAW_WORKING_DIR="$PWD/data"
qwenpaw app --host 127.0.0.1 --port 8088
```

Windows PowerShell：

```powershell
Set-Location "$HOME\qwenpaw-ogmemory"
$env:Path = "$(Join-Path $PWD '.venv\Scripts');$env:Path"
$env:QWENPAW_WORKING_DIR = Join-Path $PWD 'data'
qwenpaw app --host 127.0.0.1 --port 8088
```

使用第 4.4 节环境变量方式配置连接时，在 `qwenpaw app` 前再次设置连接变量。编辑配置后，通过终端 `Ctrl+C` 停止服务，再使用上述命令启动。

## 6. 验证跨会话记忆

在对话页面发送：

```text
请记住：项目 Atlas 的固定发布窗口是每周四 20:00。只回复“已记住”。
```

包含 `remember`、`记住`、`记下` 或 `保存到记忆` 的用户消息会触发抽取；普通消息按服务端配置的阈值累积。等待回复以及 oGMemory 服务端写入、抽取和索引完成。随后在同一 Agent 下新建会话，发送：

```text
项目 Atlas 的固定发布窗口是什么时候？只回复时间。
```

预期回复包含“每周四 20:00”。两个会话应使用相同的 oGMemory Account ID 和 User ID。

也可在新会话中显式调用检索工具：

```text
请使用 ogmemory_search 查询项目 Atlas 的发布窗口。
```

确认工具返回相关记忆，并在 oGMemory 服务端会话详情中检查用户消息、助手消息、工具调用和轮次结束事件。

## 7. 升级插件

在第 2 节配置的终端、`qwenpaw-ogmemory` 目录下拉取同一分支的新版源码，再更新插件：

```bash
git -C ./oGMemory pull --ff-only
qwenpaw plugin install ./oGMemory/plugins/qwenpaw --force
qwenpaw plugin info ogmemory
```

确认版本后重启 QwenPaw，使运行进程使用更新后的插件和配置。

## 8. 接入流程

| QwenPaw 阶段 | oGMemory 行为 |
| --- | --- |
| 首次使用会话 | `bootstrap` 初始化会话，记录 `session_start` |
| 模型调用前 | `compose` 检索并注入记忆，同一轮工具循环复用检索结果 |
| 对话完成 | `after_turn` 保存用户和助手文本，记录 `turn_end` |
| 工具执行 | 记录 `tool_start`、`tool_success`、`tool_failure` |
| 工具确认 | 记录 `permission_request`，确认恢复后由原生回复结束事件关闭轮次 |
| 上下文压缩 | 记录 `pre_compact`、`post_compact` |
| 关闭或卸载插件 | 排空 Trace 队列，记录 `session_end` 并提交会话 |

Trace 使用现有 `/api/v1/trace/events` 协议，可用于会话重建和 Skill 沉淀。检索内容作为外部数据注入。Trace 正文在本地脱敏和限长。

## 9. 超时与排查

以下插件配置项放在 `data/config.json` 的 `plugins.ogmemory` 对象中：

| 环境变量 | 插件配置 | 默认值 |
| --- | --- | --- |
| `OGMEMORY_TIMEOUT_SECONDS` | `write_timeout_seconds` | `120` 秒 |
| `OGMEMORY_BOOTSTRAP_TIMEOUT_SECONDS` | `bootstrap_timeout_seconds` | `5` 秒 |
| `OGMEMORY_COMPOSE_TIMEOUT_SECONDS` | `compose_timeout_seconds` | `5` 秒 |
| `OGMEMORY_TRACE_TIMEOUT_SECONDS` | `trace_timeout_seconds` | `3` 秒 |
| `OGMEM_TRACE_CAPTURE_CONTENT` | `trace_capture_content` | `true` |

原有 `OGMEMORY_URL`、`OGMEMORY_API_KEY`、`OGMEMORY_ACCOUNT_ID`、`OGMEMORY_USER_ID`、`OGMEMORY_AGENT_ID` 也可使用；标准 `OG_*` 变量优先。

| 现象 | 检查步骤 |
| --- | --- |
| 找不到 `qwenpaw` 命令 | 重新执行第 5 节中恢复 Python 环境的命令，再运行 `qwenpaw --version` |
| 插件列表或配置与预期不一致 | 核对当前终端的 `QWENPAW_WORKING_DIR`，确认安装和启动使用同一个 `data` 目录 |
| 普通对话无法回复 | 检查 QwenPaw 的模型提供方、模型名称、Base URL 和模型 API Key |
| oGMemory 连接或鉴权失败 | 从 QwenPaw 所在机器检查服务地址，再核对 Tenant Key、Account ID 和 Agent 授权 |
| 写入较慢 | 按服务端抽取耗时调整写入超时；待重试批次保存在 QwenPaw 会话状态中 |
| 跨会话召回为空 | 确认写入、抽取和索引完成，并核对两个会话的 Account ID 和 User ID |

开发及评测命令见[插件 README](../../plugins/qwenpaw/README.md)。
