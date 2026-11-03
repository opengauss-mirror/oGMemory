# oGMemory 操作与功能验证

## 1. 操作准备

准备 oGMemory HTTP 服务地址、API Key，以及要使用的账户、用户和 Agent。TUI 命令在 OpenClaw 的运行环境中执行。

使用 Docker 时，在对应组件所在服务器查看容器名称：

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
```

将以下名称替换为实际容器名，在服务器终端执行：

```bash
export OGMEM_CONTAINER='ogmem'
export CE_CONTAINER='openclaw_ce'
export MCP_CONTAINER='openclaw_mcp'
```

OpenClaw 从各自的配置中读取 Gateway 端口、令牌和默认 Agent。[管理界面](#4-管理界面核验) 在浏览器中访问。

| 容器 | 用途 |
| --- | --- |
| `$OGMEM_CONTAINER` | 提供记忆写入、检索和 MCP 服务 |
| `$CE_CONTAINER` | 通过 Context Engine 自动提取、压缩和召回记忆 |
| `$MCP_CONTAINER` | 通过 MCP 管理记忆和 Agent Skill |

## 2. Context Engine 测试

使用容器时，在服务器终端执行：

```bash
docker exec -it -u node -e HOME=/home/node \
  "$CE_CONTAINER" openclaw tui --session "ce-tui-$(date +%s)"
```

OpenClaw 直接安装时，使用 Context Engine Gateway 的运行账户和配置环境执行：

```bash
openclaw tui --session "ce-tui-$(date +%s)"
```

进入 TUI 后，执行以下两组测试。

### 2.1 短内容记忆

记忆提取在待处理内容达到配置阈值时自动触发，也可执行 `/compact` 主动处理。提取和索引异步完成，切换新会话前先确认记忆可读取和检索：

```text
记住：我习惯在周二上午做代码评审。
/compact
```

等待 `/compact` 完成，在 [管理界面核验](#4-管理界面核验) 中确认对应偏好记忆已可读取和检索，再输入：

```text
/new
我习惯什么时候做代码评审？
```

回答和偏好记忆正文应包含“周二上午”。

### 2.2 长内容记忆

输入一段完整的项目背景，再在新会话中按主题检索：

```text
/new
请记住我的项目背景：我负责一个面向企业客户的订单平台，核心服务使用 Python，数据库使用 openGauss。团队采用小步提交，每个变更都要经过代码评审。发布前先执行单元测试和接口回归，再完成安全扫描；如果数据库结构发生变化，还必须验证升级和回滚脚本。正常发布时间是每周四晚上，遇到高风险变更则推迟到下一个发布窗口。线上告警由值班人员先确认影响范围，严重故障需要在十五分钟内通知负责人并建立处置群。
/compact
```

等待 `/compact` 完成，在管理界面确认项目记忆已可读取和检索，再输入：

```text
/new
请根据记忆概括我负责的项目、技术栈和协作方式。
我发布前需要完成哪些检查？数据库结构变化时还要做什么？
我的正常发布时间是什么时候？高风险变更和严重故障分别如何处理？
/exit
```

回答应能从长内容中分别召回项目背景、发布检查、发布时间和故障处理要求。

在管理界面中核对订单平台的记忆正文。回答和存储内容应包含 Python、openGauss、单元测试、接口回归、安全扫描、升级和回滚脚本、周四晚上以及十五分钟内通知负责人。

## 3. MCP 测试

使用容器时，在服务器终端执行：

```bash
docker exec -it -u node -e HOME=/home/node \
  "$MCP_CONTAINER" openclaw tui --session "mcp-tui-$(date +%s)"
```

OpenClaw 直接安装时，使用 MCP Gateway 的运行账户和配置环境执行：

```bash
openclaw tui --session "mcp-tui-$(date +%s)"
```

### 3.1 写入与跨会话检索

进入 TUI 后依次输入：

```text
请先发现 ogmemory MCP 服务提供的能力，再调用 ogm_write_memory 记住：我负责支付模块，偏好小步提交，发布前必须完成回归测试。使用当前会话 ID，设置 force_extract=true、wait=true，并展示工具返回的状态。
```

工具返回 `ok=true`、`status=completed` 后，继续输入：

```text
/new
请通过 ogm_query_memory 查询：我负责哪个模块？提交和发布前分别有什么要求？展示命中的 URI，再调用 ogm_read_memory 读取对应正文。
```

新会话应能回答“支付模块”“小步提交”和“回归测试”，工具返回的 URI 和正文应对应这些事实。

### 3.2 语义检索与记忆更新

继续输入：

```text
请通过 ogmemory 检索与“上线前质量检查”语义相关的记忆，并概括结果。
请调用 ogm_write_memory 保存我的最新要求：发布前必须完成回归测试和安全扫描。使用当前会话 ID，设置 force_extract=true、wait=true，并展示工具返回的状态。
```

工具返回 `ok=true`、`status=completed` 后，继续输入：

```text
/new
请通过 ogm_query_memory 查询：我现在发布前必须完成哪些检查？展示命中的 URI，并调用 ogm_read_memory 读取反映最新要求的正文。
```

回答应先匹配到发布前的质量检查要求，更新后再返回“回归测试”和“安全扫描”。

### 3.3 自动分类与按类型查询

继续输入：

```text
请调用 ogm_write_memory 记住：极光验收项目使用 Python 和 openGauss，联调窗口是每周三下午三点，联调前执行接口回归。使用当前会话 ID，设置 force_extract=true、wait=true，并展示工具返回的状态。
```

工具返回 `ok=true`、`status=completed` 后，输入：

```text
请调用 ogm_query_memory，query 为“极光验收项目的技术栈和联调安排”，types=["entity"]，top_k=5，展示命中的 type、URI 和内容。
请再查询“极光验收项目的联调安排”，展示命中的 URI 和正文。
```

按类型查询应包含 `type=entity` 的项目记忆，项目名称和技术栈与输入一致；联调安排查询应能找到“周三下午三点”和“接口回归”。记忆类型由服务从原文中提取。

### 3.4 记忆清单、分页与分层读取

```text
请调用 ogm_list_memories，limit=1，展示本页记忆、count、truncated 和 next_cursor。若 truncated=true，使用返回的 next_cursor 作为 cursor 再读取一页。
请对上一节查询返回的项目记忆 URI 分别调用 ogm_read_memory，level 依次为 abstract、overview、content，展示三次工具结果。
```

每页 `count` 应与本页条目数一致。分层读取应分别返回摘要、概览和正文，正文包含项目名称与技术栈。

### 3.5 Agent Skill 安装、发现与读取

MCP 服务连接已配置当前账户的 Admin Key，并启用可写工具集。工具列表应包含 `ogm_install_skill`、`ogm_list_skills`、`ogm_search_skills`、`ogm_read_skill` 和 `ogm_update_skill`。

以下示例使用名称 `guide-release-check`；重复测试时换一个新名称，并在后续步骤中使用相同名称。

```text
请调用 ogm_install_skill，data={"name":"guide-release-check","description":"发布前回归与安全检查清单","content":"# 发布检查清单\n\n1. 执行接口回归。\n2. 执行安全扫描。"}，展示返回的 name、source、action、version 和 URI。
请调用 ogm_list_skills 查看技能目录，再调用 ogm_search_skills 搜索“发布前回归与安全检查”。
请用目录返回的 name 和 source 调用 ogm_read_skill，展示技能正文。
```

安装结果应为 `status=success`、`action=create`、`version=1`。目录和搜索应能找到该 Skill，`source=imported`，正文包含接口回归和安全扫描两步。

### 3.6 Agent Skill 更新与复用

```text
请调用 ogm_update_skill，name="guide-release-check"，提交完整 data={"name":"guide-release-check","description":"发布前回归与安全检查清单","content":"# 发布检查清单\n\n1. 执行接口回归。\n2. 执行安全扫描。\n3. 验证数据库升级和回滚脚本。"}，展示返回的 action 和 version。
/new
请通过 ogm_search_skills 查找“发布前回归与安全检查”，选择 name="guide-release-check"、source="imported" 的技能，调用 ogm_read_skill 读取正文，再按技能给出发布检查清单。
/exit
```

更新结果应为 `status=success`、`action=update`、`version=2`；新会话读取的正文和回答都应包含新增的升级、回滚脚本检查。

## 4. 管理界面核验

### 4.1 打开管理界面

在浏览器中打开 `http://<服务地址>:<HTTP端口>/memory-viewer/`，填写实际部署的服务地址和端口。使用容器端口映射时，填写宿主机端口。

### 4.2 核对数据与运行状态

在 **连接设置** 中填写 oGMemory API Key，选择对应账户。账户连接和页面操作见 [管理界面使用指南](../../docs/memory_viewer_guide.md)。

| 功能 | 操作 | 通过条件 |
| --- | --- | --- |
| 记忆浏览 | 打开 **记忆**，选择对应账户、用户或 Agent，定位偏好或项目记忆 | 摘要、概览和正文与写入或更新的事实一致 |
| 语义检索 | 打开 **检索**，查询“上线前质量检查”或“极光验收项目” | 命中相应记忆，打开正文可核对事实 |
| Agent Skill | 打开 **Agent Skill**，选择当前 Agent 的 `imported` 目录 | 找到 `guide-release-check` 或自定义名称的 Skill，正文包含更新后的三步检查 |
| 模型用量 | 打开 **控制台**，查看执行操作时段的调用与 Token 用量 | 能看到模型调用记录或用量变化 |
| 组件监控 | 打开 **监控**，查看服务、存储、模型和索引状态 | 组件状态正常，已写入内容可检索 |
| 后台任务 | 打开 **任务中心**，按任务类型和状态筛选，核对更新时间、关联资源和处理进度 | 关联记忆最终可检索、可读取，内容与输入事实一致 |

## 5. 常用命令

- `/compact`：短内容未达到自动提取阈值时，主动触发提取和持久化。
- `/new`：开始新会话，用于验证跨会话记忆。
- `/exit`：退出 TUI。
