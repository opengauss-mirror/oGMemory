# Hermes Agent 对接

oGMemory 以原生 Hermes `MemoryProvider` 接入，为 Hermes 提供自动记忆召回、对话写回、跨会话记忆和主动记忆工具。

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

响应中的 `status` 为 `ok` 后继续安装 Hermes。

## 2. 安装并初始化 Hermes

Linux、macOS 或 WSL2：

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

按照安装向导配置模型提供方，然后确认 Hermes 可以完成普通对话：

```bash
hermes
```

完成后输入 `/exit` 返回终端。

## 3. 安装 oGMemory Memory Provider

将 provider 下载到当前 Hermes profile：

```bash
export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
mkdir -p "$HERMES_HOME/plugins/ogmemory"

curl -fL \
  https://raw.gitcode.com/opengauss/oGMemory/raw/dev/plugins/hermes/ogmemory/__init__.py \
  -o "$HERMES_HOME/plugins/ogmemory/__init__.py"

curl -fL \
  https://raw.gitcode.com/opengauss/oGMemory/raw/dev/plugins/hermes/ogmemory/plugin.yaml \
  -o "$HERMES_HOME/plugins/ogmemory/plugin.yaml"
```

安装后的目录结构：

```text
$HERMES_HOME/plugins/ogmemory/
├── __init__.py
└── plugin.yaml
```

## 4. 配置连接

运行 Hermes Memory Provider 配置向导：

```bash
hermes memory setup
```

选择 `ogmemory`，依次填写：

```text
Endpoint: https://memory.example.com
API Key: ogm_xxx
Account ID: customer-a
User ID: alice
Agent ID: customer-a-api-agent
```

查看状态：

```bash
hermes memory status
```

当前 provider 应显示为 `ogmemory`，连接状态应为可用。

配置也可以通过 Hermes 运行环境提供：

```dotenv
OGMEMORY_ENDPOINT=https://memory.example.com
OGMEMORY_API_KEY=ogm_xxx
OGMEMORY_ACCOUNT_ID=customer-a
OGMEMORY_USER_ID=alice
OGMEMORY_AGENT_ID=customer-a-api-agent
```

同一用户在不同会话中使用相同的 Account ID 和 User ID。不同用户使用不同的 User ID。

## 5. 体验跨会话记忆

### 自动记忆控制

```bash
export OGMEM_AUTO_CAPTURE=false
export OGMEM_AUTO_RECALL=true
hermes
```

两个开关默认均开启，分别控制自动写入和自动召回。也可在 `$HERMES_HOME/ogmemory.json` 设置
`"auto_capture": false`、`"auto_recall": true`，环境变量优先。`1`、`true`、`yes`
（忽略大小写和首尾空格）开启，其余显式值关闭；修改后重启 Hermes。

关闭自动写入后，跳过 `sync_turn`、Hermes 原生记忆写入的自动镜像及自动压缩/会话提交，
不启动后台写入线程，不重放或删除已有恢复文件。重新开启后恢复待写入数据。
`ogmemory_remember` 和 `ogmemory_search` 显式工具不受这两个自动开关限制。

启动 Hermes：

```bash
hermes
```

发送：

```text
请记住：我的项目发布窗口是每周四 20:00。只回复“已记住”。
```

回复完成后输入 `/new` 新建会话，再发送：

```text
我的项目发布窗口是什么时候？只回复时间。
```

Hermes 会从 oGMemory 召回记忆并回答 `每周四 20:00`。

## 6. 使用记忆工具

Provider 会为 Hermes 注册以下工具：

| 工具 | 用途 |
| --- | --- |
| `ogmemory_search` | 主动检索偏好、决策、项目、事件和历史对话 |
| `ogmemory_remember` | 立即抽取并保存长期记忆 |

主动保存：

```text
请使用 ogmemory_remember 记住：项目 Atlas 的代码冻结时间是每周三 18:00。
```

主动检索：

```text
请使用 ogmemory_search 查询项目 Atlas 的代码冻结时间。
```

## 7. 接入流程

| Hermes 阶段 | oGMemory 行为 |
| --- | --- |
| 推理开始 | `prefetch` 调用 `/api/v1/compose`，注入相关记忆 |
| 对话轮次完成 | `sync_turn` 调用 `/api/v1/after_turn`，抽取并保存对话 |
| 会话结束、切换或压缩 | 完成写入、索引和会话提交 |
| `ogmemory_search` | 查询长期记忆 |
| `ogmemory_remember` | 立即抽取并持久化指定内容 |

Hermes 对接完成后，每轮对话会自动参与 oGMemory 的记忆生命周期，无需在提示词中手动调用接口。
