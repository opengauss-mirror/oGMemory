# oGMemory + OpenClaw 部署运维指南

## 1. 架构总览

```
┌─────────────────────────────────┐
│         用户浏览器 / API 调用     │
│   http://<server-ip>:<port>     │
└──────────┬──────────────────────┘
           │ WebSocket / HTTP
           ▼
┌─────────────────────────────────┐
│     OpenClaw 容器                │
│     ogmem-openclaw:<tag>        │
│     端口: 可配置(默认 18789)     │
│                                 │
│  ┌─────────────────────────┐    │
│  │ og-memory-context-engine│    │
│  │ (Node.js 插件，已内置)   │    │
│  │  HTTP fetch ──────────────────┐
│  └─────────────────────────┘    ││
└─────────────────────────────────┘│
                                   │
┌──────────────────────────────────▼┐
│     oGMemory 容器                │
│     ogmem-server:<tag>            │
│     端口: 8090                    │
│                                   │
│  ┌──────────┐ ┌──────────────┐   │
│  │ Flask/   │ │ IndexService │   │
│  │ Gunicorn │ │ (后台轮询)    │   │
│  └────┬─────┘ └──────┬───────┘   │
│       │              │           │
│  ┌────▼──────────────▼────┐      │
│  │    AGFS Server (:1833) │      │
│  │    文件存储引擎          │      │
│  └────────────────────────┘      │
└───────────┬──────────────────────┘
            │ volume mount
            ▼
┌───────────────────────┐
│   宿主机磁盘           │
│   /path/to/agfs-data   │
└───────────────────────┘

┌───────────────────────┐
│   openGauss 数据库     │
│   端口: 8799           │
│   (外部依赖)           │
└───────────────────────┘
```

**两个镜像，各司其职：**

| 镜像 | 内置内容 | 职责 |
|------|---------|------|
| `ogmem-openclaw` | alpine/openclaw + oGMemory 插件 + 默认配置 + entrypoint | AI Agent 网关，提供前端 UI 和对话能力，插件已内置无需挂载 |
| `ogmem-server` | Python 运行时 + 检索/写入引擎 + AGFS + IndexService | 长期记忆 HTTP 服务，提供语义检索和记忆写入 |

> 外部依赖仅 **openGauss 数据库**（需提前部署）。

---

## 2. 前置依赖

### 2.1 openGauss 数据库

oGMemory 需要 openGauss 作为向量数据库（已内置向量扩展，无需手动安装）。

```bash
# 确认数据库可用
gsql -U <user> -W <password> -d <dbname> -p <port> -r
# 执行 \q 退出
```

> 表 `vector_index` 会在 oGMemory 首次启动时自动创建，无需手动建表。

### 2.2 LLM API Key

需要一个兼容 OpenAI 接口的 LLM 服务（如 DashScope / 智谱 / DeepSeek 等）。

### 2.3 Docker 环境

```bash
docker --version   # 确认 Docker 已安装
```

### 2.4 镜像准备

| 镜像 | 获取方式 |
|------|---------|
| `alpine/openclaw:latest` | `docker load < opencl.tar`（用于构建 ogmem-openclaw） |
| `ogmem-server:<tag>` | 本地构建（见第 3 节） |
| `ogmem-openclaw:<tag>` | 本地构建（见第 3 节） |

---

## 3. 构建镜像

### 3.1 源码目录结构

```
oGMemory/
├── config/                           # 统一配置模板目录
│   ├── ogmem.reference.yaml          # 统一参考模板（所有模式）
│   ├── deploy.env.reference          # Docker 部署环境变量模板
│   ├── agfs/config.reference.yaml    # AGFS 配置参考
│   └── openclaw/ogmem.template.json  # OpenClaw 配置模板
├── docker/
│   ├── Dockerfile.standalone         # ogmem-server 镜像 Dockerfile
│   ├── Dockerfile.openclaw           # ogmem-openclaw 镜像 Dockerfile
│   ├── entrypoint-standalone.sh      # ogmem-server 容器入口脚本
│   └── entrypoint-openclaw.sh        # ogmem-openclaw 容器入口脚本
├── server/                          # HTTP API 服务
├── core/                            # 数据模型
├── retrieval/                       # 检索链路
├── extraction/                      # 记忆提取
├── commit/                          # 写入链路
├── index/                           # 索引服务
├── providers/                       # 外部服务适配
├── fs/                              # AGFS 适配
├── scripts/                         # 运维脚本
├── agfs/                            # AGFS 二进制
└── plugins/openclaw/  # OpenClaw 插件（内置到 ogmem-openclaw 镜像）
    ├── index.js                     # 插件入口（HTTP 模式）
    ├── openclaw.plugin.json
    └── package.json
```

### 3.2 构建 oGMemory 镜像

```bash
cd /path/to/oGMemory

# 构建 ogmem-server
docker build --network host -f docker/Dockerfile.standalone -t ogmem-server:v3 .
```

### 3.3 构建 OpenClaw + 插件镜像

```bash
# 确保 alpine/openclaw:latest 已加载
docker images | grep alpine/openclaw
# 如果没有：docker load < /path/to/opencl.tar && docker tag <image-id> alpine/openclaw:latest

# 构建 ogmem-openclaw
docker build --network host -f docker/Dockerfile.openclaw -t ogmem-openclaw:v2 .
```

### 3.4 验证镜像

```bash
docker images | grep -E "ogmem|openclaw"
# 预期看到：
# ogmem-openclaw    v2      ...   ~2.3GB
# ogmem-server      v3      ...   ~308MB
# alpine/openclaw   latest  ...   ~2.3GB（base 镜像，保留不删）
```

---

## 4. 完整部署流程（Quick Start）

### Step 1：确认 openGauss 可用

```bash
gsql -U <user> -W '<password>' -d <dbname> -p <port> -r
# 执行 \q 退出
```

### Step 2：准备数据目录

```bash
# AGFS 文件存储目录
mkdir -p /path/to/agfs-data
chmod 777 /path/to/agfs-data

# OpenClaw 状态持久化目录（可选，用于保存对话历史）
mkdir -p /path/to/openclaw-home
chmod 777 /path/to/openclaw-home
```

### Step 3：启动 oGMemory

```bash
docker run -d \
  --name ogmem-server \
  --network host \
  -v /path/to/agfs-data:/data/agfs \
  -e OGMEM_API_KEY="<your-api-key>" \
  -e OGMEM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" \
  -e OGMEM_EMBEDDING_MODEL="text-embedding-v2" \
  -e OGMEM_LLM_MODEL="glm-5" \
  -e "OPENGAUSS_CONNECTION_STRING=host=127.0.0.1 port=8799 dbname=postgres user=<user> password=<password>" \
  ogmem-server:v3
```

等待 10 秒后验证：

```bash
curl http://127.0.0.1:8090/api/v1/health
# 预期：{"status":"ok", "agfs":true, "llm":"glm-5", ...}
```

### Step 4：启动 OpenClaw

```bash
docker run -d \
  --name openclaw \
  --network host \
  -v /path/to/openclaw-home:/home/node/.openclaw \
  -e LLM_API_KEY="<your-api-key>" \
  -e LLM_PROVIDER="dashscope" \
  -e LLM_MODEL="glm-5" \
  -e LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" \
  -e OGMEM_URL="http://127.0.0.1:8090" \
  -e GATEWAY_PORT="18789" \
  ogmem-openclaw:v2
```

> **注意**：如果 18789 端口被占用，通过 `GATEWAY_PORT` 环境变量指定其他端口。

等待 15 秒后验证：

```bash
# 确认启动成功
docker logs openclaw 2>&1 | grep -E "listening|og-memory"
# 预期看到：
# [gateway] [og-memory] mode=http url=http://127.0.0.1:8090
# [gateway] listening on ws://127.0.0.1:18789 ...
```

### Step 5：测试 HTTP API

```bash
curl -s -X POST http://127.0.0.1:18789/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"openclaw","input":"你好，今天天气怎么样？"}'
```

### Step 6：验证 oGMemory 工作

```bash
# 检查 oG-Memory 是否被调用
docker logs ogmem-server 2>&1 | grep -E "compose|assemble"
# 预期：首次 compose 可能返回 0 hits，后续多轮对话会逐步积累

# 检查 OpenClaw 插件连接状态
docker logs openclaw 2>&1 | grep "og-memory"
# 预期：[plugins] [og-memory] mode=http url=http://127.0.0.1:8090
```

### Step 7：通过前端访问（可选）

浏览器打开：`http://<server-ip>:<GATEWAY_PORT>`

> 使用 `--auth none` 模式，无需配对 Token，直接进入对话界面。

---

## 5. 环境变量参考

### 5.1 ogmem-server 环境变量

#### 必填参数

| 变量 | 说明 | 示例 |
|------|------|------|
| `OGMEM_API_KEY` | LLM/Embedding 服务的 API Key | `sk-xxxxxxxxxxxx` |
| `OPENGAUSS_CONNECTION_STRING` | openGauss 连接串（libpq 格式） | `host=127.0.0.1 port=8799 dbname=postgres user=xxx password=xxx` |

#### 可选参数（有默认值）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OGMEM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | LLM API 地址 |
| `OGMEM_EMBEDDING_MODEL` | `text-embedding-v2` | Embedding 模型名 |
| `OGMEM_LLM_MODEL` | （无默认） | LLM 模型名，用于记忆提取 |
| `OGMEM_HTTP_PORT` | `8090` | HTTP API 监听端口 |
| `OGMEM_WORKERS` | `2` | Gunicorn worker 进程数 |
| `OG_ACCOUNT_ID` | `acct-demo` | 默认租户 ID |
| `AGFS_BASE_URL` | `http://127.0.0.1:1833` | AGFS 内部地址（一般不改） |
| `VECTOR_DB_TYPE` | `opengauss` | 向量数据库类型 |
| `INDEX_INTERVAL` | `15` | IndexService 轮询间隔（秒） |

#### 数据卷

| 宿主机路径 | 容器路径 | 用途 |
|-----------|---------|------|
| `/path/to/agfs-data` | `/data/agfs` | AGFS 文件存储（记忆 .md 文件） |

> 此目录需确保容器内用户有写权限。首次使用时执行 `chmod 777 /path/to/agfs-data`。

### 5.2 ogmem-openclaw 环境变量

#### 必填参数

| 变量 | 说明 | 示例 |
|------|------|------|
| `LLM_API_KEY` | LLM 服务的 API Key | `sk-xxxxxxxxxxxx` |
| `OGMEM_URL` | oGMemory HTTP API 地址 | `http://127.0.0.1:8090` |

#### 可选参数（有默认值）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | `dashscope` | LLM 提供商名称 |
| `LLM_MODEL` | `glm-5` | LLM 模型名 |
| `LLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | LLM API 地址 |
| `GATEWAY_PORT` | `18789` | OpenClaw Gateway 监听端口 |

#### 数据卷

| 宿主机路径 | 容器路径 | 用途 |
|-----------|---------|------|
| `/path/to/openclaw-home` | `/home/node/.openclaw` | OpenClaw 配置和状态持久化（对话历史等） |

> 如果不挂载此目录，容器会自动生成配置，但容器重启后对话历史丢失。

---

## 6. ogmem-openclaw 镜像原理

### 6.1 Dockerfile.openclaw 做了什么

```dockerfile
FROM alpine/openclaw:latest

# 将 oGMemory 插件内置到镜像中（无需运行时挂载）
COPY plugins/openclaw/ /app/extensions/og-memory-context-engine/

# 内置默认 openclaw.json 模板（含占位符，entrypoint 运行时注入真实值）
COPY config/openclaw/ogmem.template.json /opt/defaults/openclaw.json

# 自定义 entrypoint：生成配置 → 启动 Gateway
COPY docker/entrypoint-openclaw.sh /opt/entrypoint-openclaw.sh
```

### 6.2 entrypoint-openclaw.sh 逻辑

启动时自动完成以下步骤：

1. **检查配置文件**：如果 `/home/node/.openclaw/openclaw.json` 不存在，从模板生成
2. **注入环境变量**：将 `LLM_API_KEY`、`LLM_MODEL`、`OGMEM_URL`、`GATEWAY_PORT` 等替换进模板
3. **启动 Gateway**：`exec node /app/openclaw.mjs gateway --allow-unconfigured --auth none`

> `--auth none` 使 HTTP API 和前端无需配对即可直接使用，适合自动化部署。

### 6.3 通信链路

```
用户发送消息
      │
      ▼
OpenClaw Gateway (Node.js)
      │
      ▼
og-memory-context-engine 插件 (index.js)
      │
      ├─── prefetch (可选，对话前) ──► POST http://ogmem:8090/api/v1/prefetch
      │     ↳ 预取候选记忆并暂存到会话 TopicBuffer
      │
      ├─── compose (对话前) ──► POST http://ogmem:8090/api/v1/compose
      │     ↳ 返回分层 system messages（Profile / Archive / Session State / Working Set）
      │
      ├─── LLM 生成回复 ◄──── 注入记忆上下文
      │
      ├─── afterTurn (对话后) ──► POST http://ogmem:8090/api/v1/after_turn
      │     ↳ 从对话中提取新记忆，写入存储 + openGauss
      │
      └─── compact (上下文接近上限) ──► prepare_compaction → compact
            ↳ 先增量抽取并发放 prepareToken，再同步提交/归档并返回压缩摘要
```

---

## 7. oGMemory HTTP API

| 端点 | 方法 | 说明 | 调用时机 |
|------|------|------|---------|
| `/api/v1/health` | GET | 健康检查 | 运维监控 |
| `/api/v1/compose` | POST | 检索并组装分层上下文 | 每轮对话前 |
| `/api/v1/prefetch` | POST | 预取候选记忆并暂存到会话 TopicBuffer | compose 前，可选 |
| `/api/v1/after_turn` | POST | 提取并写入新记忆 | 每轮对话后 |
| `/api/v1/ingest` | POST | 单条消息注入 | OpenClaw 消息处理 |
| `/api/v1/prepare_compaction` | POST | compact 前增量抽取，返回一次性 `prepareToken` | compact 前 |
| `/api/v1/compact` | POST | 同步提交/归档并返回压缩上下文 | OpenClaw 触发 |
| `/api/v1/dispose` | POST | 清理会话资源 | 会话结束 |
| `/api/v1/session_working_set` | GET/POST | 查看活跃会话工作集 | 运维排查 |
| `/api/v1/evict_idle_sessions` | POST | 按 `maxIdleSeconds` 淘汰空闲会话 | 运维维护 |

> 运维注意：`prepare_compaction` 当前会在 HTTP worker 内同步执行增量抽取。大上下文或慢 LLM 会阻塞该请求；生产部署应设置合理的网关超时和 worker 数量，并通过两阶段 compact 避免最终 compact 阶段重复抽取。

**compose 请求示例：**

```bash
curl -X POST http://127.0.0.1:8090/api/v1/compose \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "周末想跟男朋友出去玩"}],
    "prompt": "周末想跟男朋友出去玩",
    "sessionId": "test-session",
    "accountId": "acct-demo",
    "userId": "u-alice"
  }'
```

---

## 8. 常用运维命令

```bash
# 查看所有相关容器
docker ps | grep -E "ogmem|openclaw"

# 查看 oGMemory 日志（实时跟踪）
docker logs -f ogmem-server

# 查看 OpenClaw 日志
docker logs -f openclaw

# 重启（不丢失数据）
docker restart ogmem-server
docker restart openclaw

# 停止并删除容器（镜像和数据不受影响）
docker stop ogmem-server openclaw
docker rm ogmem-server openclaw

# 查看 AGFS 存储的记忆文件
ls /path/to/agfs-data/plugin/accounts/

# 查看 openGauss 中的向量索引数据
gsql -U <user> -W '<password>' -d <dbname> -p <port> -c "SELECT count(*) FROM vector_index;"
```

---

## 9. 故障排查

| 问题 | 检查方法 | 解决方案 |
|------|---------|---------|
| oGMemory health 返回异常 | `docker logs ogmem-server` | 检查 openGauss 连接串和 AGFS 启动日志 |
| OpenClaw 日志无 `[og-memory]` | `docker logs openclaw \| grep plugin` | 确认镜像版本正确（应为 `ogmem-openclaw`，不是原生 `alpine/openclaw`） |
| compose 返回 0 hits | `docker logs ogmem-server \| grep -E "compose\|assemble"` | 正常现象（首次使用无历史记忆），多轮对话后会积累 |
| `permission denied` 写入失败 | `docker logs ogmem-server \| grep permission` | `chmod 777 /path/to/agfs-data` |
| 端口被占用 | `ss -tlnp \| grep <port>` | 通过 `GATEWAY_PORT` 环境变量指定其他端口 |
| `OPENGAUSS_CONNECTION_STRING is required` | 检查 `docker run` 参数 | 确保 `-e` 参数中包含此连接串 |
| `Network is unreachable` (LLM) | `curl https://dashscope.aliyuncs.com` | 确认 `OGMEM_BASE_URL` 正确且网络可达 |
| LLM 返回 `Arrearage` | 检查 API Key 余额 | 阿里云账户欠费，请充值或更换 Key |
| `Config invalid` 启动失败 | `docker logs openclaw` | 检查错误提示的 JSON 字段，或删除 openclaw-home 让容器重新生成配置 |

---

## 10. 实际部署示例

以下是在当前服务器上已验证通过的完整命令：

```bash
# === 前提：openGauss 已在 127.0.0.1:8799 运行 ===

# 1. 准备目录
mkdir -p /data1/sundechao/ogmem/fresh-deploy/agfs-data
mkdir -p /data1/sundechao/ogmem/fresh-deploy/openclaw-home
chmod 777 /data1/sundechao/ogmem/fresh-deploy/agfs-data
chmod 777 /data1/sundechao/ogmem/fresh-deploy/openclaw-home

# 2. 启动 oGMemory
docker run -d \
  --name ogmem-server \
  --network host \
  -v /data1/sundechao/ogmem/fresh-deploy/agfs-data:/data/agfs \
  -e OGMEM_API_KEY="sk-bc4cceb67d01420f828aeabb0e27ac29" \
  -e OGMEM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" \
  -e OGMEM_EMBEDDING_MODEL="text-embedding-v2" \
  -e OGMEM_LLM_MODEL="glm-5" \
  -e "OPENGAUSS_CONNECTION_STRING=host=127.0.0.1 port=8799 dbname=postgres user=sundechao password=Huawei@123" \
  ogmem-server:v3

# 3. 等待 oGMemory 就绪
sleep 12
curl http://127.0.0.1:8090/api/v1/health

# 4. 启动 OpenClaw（端口 18800，避开被占用的 18789）
docker run -d \
  --name openclaw \
  --network host \
  -v /data1/sundechao/ogmem/fresh-deploy/openclaw-home:/home/node/.openclaw \
  -e LLM_API_KEY="sk-bc4cceb67d01420f828aeabb0e27ac29" \
  -e LLM_PROVIDER="dashscope" \
  -e LLM_MODEL="glm-5" \
  -e LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" \
  -e OGMEM_URL="http://127.0.0.1:8090" \
  -e GATEWAY_PORT="18800" \
  ogmem-openclaw:v2

# 5. 等待 OpenClaw 就绪
sleep 20
docker logs openclaw 2>&1 | grep -E "listening|og-memory"

# 6. 测试
curl -s -X POST http://127.0.0.1:18800/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"openclaw","input":"你好"}'
```
