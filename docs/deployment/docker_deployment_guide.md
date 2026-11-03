# oGMemory Docker 镜像构建与部署指南

> 文档入口：[文档总览](../README.md)。本页保留镜像构建、参数、容器架构和故障排查；Compose 部署见 [Compose 指南](../../deploy/compose/README_zh.md)。

## 1. 前置条件

| 依赖项 | 要求 |
|--------|------|
| Docker | 20.10+ |
| 架构 | arm64（基础镜像为 `alpine/openclaw:latest`，Debian bookworm arm64） |
| openGauss | 外部运行，需开启 vector 扩展 |
| 网络 | 构建时需要访问外网（PyPI / apt） |

## 2. 目录结构

```
oGMemory/
├── config/                     # 统一配置模板目录
│   ├── ogmem.reference.yaml    # 统一参考模板
│   ├── deploy.env.reference    # Docker 部署环境变量模板
│   ├── agfs/config.reference.yaml  # AGFS 配置参考
│   └── openclaw/ogmem.template.json # OpenClaw 配置模板
├── docker/
│   ├── Dockerfile.standalone   # ogmem-server 镜像
│   ├── Dockerfile.openclaw     # ogmem-openclaw 镜像
│   ├── entrypoint-standalone.sh  # ogmem-server 容器入口脚本
│   └── entrypoint-openclaw.sh    # ogmem-openclaw 容器入口脚本
├── agfs/
│   └── agfs-server             # AGFS 二进制文件
├── core/                       # oGMemory 核心模块
├── retrieval/                  # 检索链路
├── extraction/                 # 记忆抽取
├── commit/                     # 写入管线
├── index/                      # 索引调度
├── providers/                  # LLM/Embedding/VectorDB 适配
├── service/                    # IndexService
├── fs/                         # AGFS 文件系统适配
├── scripts/                    # 启动脚本（run_index_service.py）
└── plugins/openclaw/  # OpenClaw 插件
```

## 3. 构建镜像

```bash
cd oGMemory

# 构建（使用 host 网络加速拉取依赖）
docker build --network host -t ogmem-openclaw:latest .

```

**构建过程说明：**

1. 基于 `alpine/openclaw:latest`（含 Node.js + OpenClaw 运行时）
2. 安装 Python 3.11 + pip + 编译依赖（`gcc`, `libpq-dev`）
3. 安装 Python 依赖：`openai`, `psycopg2-binary`, `pyagfs`, `rich`, `requests`, `apscheduler`
4. 编译完成后清理编译工具，减小镜像体积
5. 复制 oGMemory 源码、OpenClaw 插件、AGFS 二进制、配置文件
6. 最终以 `node` 用户运行

**构建耗时：** 首次约 2-5 分钟（取决于网络），后续利用缓存约 30 秒。

## 4. 启动容器

### 4.1 最小启动命令

```bash
docker run -d \
  --name ogmem \
  --network host \
  -v /path/to/agfs-data:/data/agfs \
  -e OGMEM_API_KEY=<your-api-key> \
  -e OGMEM_LLM_MODEL=<model-name> \
  -e "OPENGAUSS_CONNECTION_STRING=host=127.0.0.1 port=8799 dbname=postgres user=<user> password=<pass>" \
  ogmem-openclaw:latest
```

### 4.2 完整启动命令（推荐）

```bash
docker run -d \
  --name ogmem \
  --network host \
  -v /data/ogmem/agfs:/data/agfs \
  -v /data/ogmem/openclaw-state:/home/node/.openclaw \
  -e OGMEM_API_KEY=sk-xxxxxxxxxxxx \
  -e OGMEM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  -e OGMEM_EMBEDDING_MODEL=text-embedding-v2 \
  -e OGMEM_LLM_MODEL=glm-5 \
  -e "OPENGAUSS_CONNECTION_STRING=host=127.0.0.1 port=8799 dbname=postgres user=sundechao password=xxxxx" \
  -e OPENCLAW_GATEWAY_TOKEN=my-secret-token \
  -e OG_ACCOUNT_ID=acct-prod \
  -e INDEX_INTERVAL=30 \
  ogmem-openclaw:latest
```

### 4.3 常用操作

```bash
# 查看日志
docker logs ogmem -f

# 查看容器状态
docker ps --filter name=ogmem

# 重启容器
docker restart ogmem

# 停止并删除
docker stop ogmem && docker rm ogmem

# 进入容器调试
docker exec -it ogmem bash
```

## 5. 配置参数说明

### 5.1 环境变量（-e）

#### 必填参数

| 环境变量 | 说明 | 示例 |
|----------|------|------|
| `OGMEM_API_KEY` | LLM / Embedding API 密钥 | `sk-bc4cceb67d01420f...` |
| `OGMEM_LLM_MODEL` | LLM 模型名称，注入到 OpenClaw 配置中 | `glm-5`, `qwen-plus` |
| `OPENGAUSS_CONNECTION_STRING` | openGauss 数据库连接串 | `host=127.0.0.1 port=8799 dbname=postgres user=xx password=xx` |

#### 可选参数（有默认值）

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `OGMEM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI 兼容 API 地址 |
| `OGMEM_EMBEDDING_MODEL` | `text-embedding-v2` | Embedding 模型名称 |
| `OPENCLAW_GATEWAY_TOKEN` | `test-token-123` | 前端配对 Token（浏览器首次访问时输入） |
| `OG_ACCOUNT_ID` | `acct-demo` | 租户隔离标识 |
| `OG_USER_ID` | `u-alice` | 用户标识 |
| `OG_AGENT_ID` | `main` | Agent 标识 |
| `INDEX_INTERVAL` | `15` | IndexService 轮询间隔（秒） |
| `AGFS_BASE_URL` | `http://127.0.0.1:1833` | AGFS 服务地址（容器内自动启动） |
| `AGFS_MOUNT_PREFIX` | `/local/plugin` | AGFS 挂载路径前缀 |
| `CONTEXTENGINE_PROVIDER` | `openai` | LLM 提供者类型 |
| `VECTOR_DB_TYPE` | `opengauss` | 向量数据库类型 |

### 5.2 数据卷挂载（-v）

| 容器路径 | 用途 | 是否必须 | 注意事项 |
|----------|------|----------|----------|
| `/data/agfs` | AGFS 文件存储（记忆 .md 文件） | **必须** | 宿主目录需要 `chmod 777` 或 chown 为 UID=1000 |
| `/home/node/.openclaw` | OpenClaw 状态持久化（配对信息、会话） | 推荐 | 挂载后重启容器不需要重新配对前端 |

### 5.3 端口

| 端口 | 协议 | 用途 |
|------|------|------|
| `18789` | HTTP/WebSocket | OpenClaw 主服务（前端 + API） |
| `1833` | HTTP | AGFS 文件服务（容器内部通信，一般不需要暴露） |
| `8799` | TCP | openGauss 数据库（外部依赖，非容器端口） |

## 6. 容器内部架构

容器启动时由 `entrypoint.sh` 编排三个进程：

```
entrypoint.sh
  │
  ├─ [后台] AGFS Server (:1833)     ← 文件存储服务，等待就绪后继续
  ├─ [后台] IndexService             ← 异步索引（Outbox → Embedding → openGauss）
  └─ [前台] OpenClaw Gateway (:18789) ← 主进程，前端 + AI Agent
```

**启动顺序：**

1. 初始化 `openclaw.json`（从默认模板复制，注入 API Key 和模型名）
2. 启动 AGFS，轮询等待 `:1833` 端口就绪
3. 启动 IndexService（后台守护，每 `INDEX_INTERVAL` 秒扫描 `.outbox/` 事件）
4. 启动 OpenClaw Gateway（前台主进程，接收 `SIGTERM` 时触发优雅关闭）

## 7. 首次访问前端

1. 浏览器访问 `http://<服务器IP>:18789`
2. 首次访问会提示输入 **Gateway Token**
3. 输入 `OPENCLAW_GATEWAY_TOKEN` 环境变量的值（默认 `test-token-123`）
4. 配对成功后即可正常使用

**注意：** 如果未挂载 `/home/node/.openclaw` 数据卷，每次重启容器都需要清除浏览器 LocalStorage 并重新配对。

## 8. openGauss 数据库准备

容器启动时会自动创建 `vector_index` 表（如果不存在）。表结构：

```sql
CREATE TABLE IF NOT EXISTS vector_index (
    id VARCHAR(16) PRIMARY KEY,
    uri VARCHAR(512) NOT NULL,
    level INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    filters JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

推荐手动创建索引以加速查询：

```sql
-- HNSW 向量索引
CREATE INDEX IF NOT EXISTS idx_vector_index_embedding_hnsw
  ON vector_index USING hnsw (embedding vector_cosine_ops);

-- JSONB 过滤索引
CREATE INDEX IF NOT EXISTS idx_vector_index_filters_account
  ON vector_index ((filters->>'account_id'));

CREATE INDEX IF NOT EXISTS idx_vector_index_filters_gin
  ON vector_index USING gin (filters);
```

## 9. 故障排查

| 现象 | 原因 | 解决方法 |
|------|------|----------|
| `permission denied` 写入 AGFS | 挂载目录权限不足 | `chmod 777 /path/to/agfs-data` |
| `OPENGAUSS_CONNECTION_STRING is required` | 缺少数据库连接串 | 添加 `-e OPENGAUSS_CONNECTION_STRING=...` |
| `Network is unreachable` (LLM 调用) | 容器无法访问外网 API | 检查 DNS 和网络，确认 `OGMEM_BASE_URL` 正确 |
| 前端 `ERR_CONNECTION_REFUSED` | 端口转发断开 | Cursor 中 `Ctrl+Shift+P` → Forward a Port → `18789` |
| 前端配对失败 / 白屏 | 浏览器缓存旧 Token | 清除 LocalStorage 后重新输入 Gateway Token |
| `address already in use :1833` | 宿主机已有 AGFS 进程占用端口 | `kill` 宿主机的 AGFS 进程或改用 `-p` 端口映射 |
| `.processing` 文件残留 | 旧版本 bug（已修复） | `docker exec ogmem find /data/agfs -name "*.processing" -delete` |
