# oGMemory 配置参考

> 从 v0.9 起支持 **YAML 配置文件**，环境变量前缀已统一为 `OGMEM_`（旧 `OPENAI_` 前缀不再支持）。

---

## 配置方式

### 方式一：YAML 配置文件（推荐）

源码运行时，程序会自动按以下顺序查找配置：

1. `OGMEM_CONFIG` 指定的 YAML
2. `config/ogmem.yaml`（项目根目录下）
3. 容器内 `/etc/ogmem/config.yaml`

本地开发最简单的做法是运行 `ogmem onboard` 或 `ogmem config init`，会自动在 `config/` 下生成 `ogmem.yaml`。

将配置写入一个 YAML 文件，Docker 启动时挂载到容器的 `/etc/ogmem/config.yaml`：

```bash
docker run -d --name ogmemory --network host \
  -v ./ogmemory.yaml:/etc/ogmem/config.yaml \
  ogmemory:v0.9
```

也可通过 `OGMEM_CONFIG` 环境变量指定其他路径：

```bash
docker run -d --name ogmemory --network host \
  -e OGMEM_CONFIG=/data/my-config.yaml \
  -v ./my-config.yaml:/data/my-config.yaml \
  ogmemory:v0.9
```

完整 YAML 示例见 `config/ogmem.reference.yaml`。

### 方式二：环境变量

不挂载 YAML 文件时，通过 `-e` 传递环境变量。环境变量前缀统一为 `OGMEM_`。

### 优先级

```
YAML 文件值  >  环境变量  >  代码默认值
```

YAML 中未设置的字段会自动回退到对应的环境变量。

---

## YAML 配置结构

```yaml
llm:
  provider: openai          # mock / openai / openai-cached
  api_key: "sk-xxx"
  # api_key_command:
  #   command: ["/usr/local/bin/op", "read", "op://team/ogmemory/api-key"]
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  model: "doubao-seed-2-0-code-preview-260215"
  temperature: 0.7
  max_tokens: 4096

embedding:
  model: "doubao-embedding-vision-250615"
  base_url: "https://ark.cn-beijing.volces.com/api/v3"
  api_key: null              # 缺省回退到 llm.api_key
  # api_key_command:
  #   command: ["/usr/local/bin/op", "read", "op://team/ogmemory/embedding-api-key"]
  multimodal: true

vector_db:
  type: memory                   # quickstart: memory; full persistence: opengauss / chroma
  connection_string: "host=127.0.0.1 port=8799 dbname=postgres user=xxx password=xxx"  # 仅 type=opengauss 时使用
  dimension: 1024
  table_name: vector_index
  pool_size: 5

service:
  http_port: 8090
  workers: 2

storage:
  backend: sql                   # agfs / sql
  dialect: postgresql            # postgresql / opengauss
  connection_string: "host=127.0.0.1 port=5432 dbname=ogmemory user=postgres password=postgres"
  pool_size: 5

agfs:
  base_url: "http://127.0.0.1:1833"
  mount_prefix: "/local/plugin"

index:
  interval: 15
  workers: 1

identity:
  account_id: "acct-demo"
  user_id: "u-alice"
  agent_id: "main"

memory:
  after_turn_threshold: 400
  rolling_compress_enabled: true
  rolling_compress_fallback_enabled: false
  compact_prepare_token_ttl: 300
  directory_summary_enabled: false
  summary_max_chars: 4000
  archive_max_count: 10
  archive_merge_threshold: 10
  prefetch_enabled: false
  prefetch_top_k: 5
  session_state_bridge_enabled: true
  session_state_sync_interval_turns: 1
  topic_detection_enabled: false
  compression_quality_enabled: false
  compression_quality_persist_metadata: false

cache:
  enabled: true
  max_size: 1000
```

---

## 环境变量对照表

下表列出每个 YAML 字段对应的环境变量和默认值。

### LLM 配置

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `llm.provider` | `CONTEXTENGINE_PROVIDER` | `openai` (Dockerfile) / `mock` (代码) | 否 | Provider 类型：`mock` / `openai` / `openai-cached` |
| `llm.api_key` | `OGMEM_API_KEY` | 无 | **是** | LLM 和 Embedding 共用的 API Key |
| `llm.api_key_command` | `OGMEM_API_KEY_CMD` / `OGMEM_API_KEY_COMMAND` | 无 | 否 | 惰性执行命令并读取 stdout 作为 API Key。命令必须使用绝对路径，且可执行文件必须命中代码内置白名单 |
| `llm.base_url` | `OGMEM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 否 | LLM API Base URL。不以 `/v1` `/v2` `/v3` 结尾时自动追加 `/v1` |
| `llm.model` | `OGMEM_LLM_MODEL` | `gpt-4o-mini` | 否 | LLM 模型名称 |
| `llm.temperature` | `LLM_TEMPERATURE` | `0.7` | 否 | LLM 采样温度 |
| `llm.max_tokens` | `LLM_MAX_TOKENS` | `4096` | 否 | LLM 最大输出 token 数 |

### Embedding 配置

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `embedding.model` | `OGMEM_EMBEDDING_MODEL` | `text-embedding-ada-002` | 否 | Embedding 模型名称 |
| `embedding.base_url` | `OGMEM_EMBEDDING_BASE_URL` | 回退到 `llm.base_url` | 否 | Embedding API Base URL |
| `embedding.api_key` | `OGMEM_EMBEDDING_API_KEY` | 回退到 `llm.api_key` | 否 | Embedding API Key |
| `embedding.api_key_command` | `OGMEM_EMBEDDING_API_KEY_CMD` / `OGMEM_EMBEDDING_API_KEY_COMMAND` | 回退到 `llm.api_key` | 否 | 惰性执行命令并读取 stdout 作为 Embedding API Key。命令必须使用绝对路径，且可执行文件必须命中代码内置白名单 |
| `embedding.multimodal` | `OGMEM_EMBEDDING_MULTIMODAL` | `false` | 否 | 使用 `/embeddings/multimodal` 端点。设为 `true` / `1` / `yes` 启用 |

### 向量数据库

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `vector_db.type` | `VECTOR_DB_TYPE` | `opengauss` (Dockerfile) / `chroma` (代码) | 否 | `memory` 内存 / `chroma` 本地持久化 / `opengauss` 数据库向量索引 |
| `vector_db.connection_string` | `VECTOR_DB_CONNECTION_STRING`（旧名 `OPENGAUSS_CONNECTION_STRING` 仍生效） | 无 | type=opengauss 时**必填** | 连接串格式：`host=... port=... dbname=... user=... password=...` |
| `vector_db.dimension` | `VECTOR_DB_DIMENSION`（旧名 `OPENGAUSS_DIMENSION`） | `1024` | 否 | 向量维度，须与 Embedding 模型一致 |
| `vector_db.table_name` | `VECTOR_DB_TABLE_NAME`（旧名 `OPENGAUSS_TABLE_NAME`） | `vector_index` | 否 | 向量存储表名 |
| `vector_db.pool_size` | `VECTOR_DB_POOL_SIZE`（旧名 `OPENGAUSS_POOL_SIZE`） | `5` | 否 | 数据库连接池大小 |

### 主存储后端

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `storage.backend` | `STORAGE_BACKEND` | `sql` | 否 | 固定为 `sql`（PostgreSQL 直连存储） |
| `storage.dialect` | `SQL_DIALECT` | `postgresql` | 否 | SQL 方言：`postgresql` / `opengauss`；openGauss 数据库必须使用 `DBCOMPATIBILITY='PG'` |
| `storage.connection_string` | `STORAGE_DB_CONNECTION_STRING`（旧名 `SQL_CONNECTION_STRING` 仍生效） | 无 | **必填** | PostgreSQL libpq DSN，例如 `host=127.0.0.1 port=5432 dbname=ogmemory user=postgres password=postgres` |
| `storage.pool_size` | `STORAGE_DB_POOL_SIZE`（旧名 `SQL_POOL_SIZE`） | `5` | 否 | SQL storage 连接池大小 |

### 服务配置

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `service.http_port` | `OGMEM_HTTP_PORT` | `8090` | 否 | HTTP API 监听端口 |
| `service.workers` | `OGMEM_WORKERS` | `2` | 否 | Gunicorn worker 数量 |
| `service.http_ip_allowlist` | `OG_HTTP_IP_ALLOWLIST` | 空 | 否 | HTTP IP/CIDR 白名单 |
| `service.http_ip_allowlist_trust_proxy` | `OG_HTTP_IP_ALLOWLIST_TRUST_PROXY` | `false` | 否 | 是否信任代理转发头 |
| `service.http_trusted_proxies` | `OG_HTTP_TRUSTED_PROXIES` | 空 | 否 | 仅当 `REMOTE_ADDR` 命中这些代理 IP/CIDR 时，才信任 `X-Forwarded-For` / `X-Real-IP` |

### AGFS

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `agfs.base_url` | `AGFS_BASE_URL` | `http://127.0.0.1:1833` | 否 | 容器内自启动，通常无需修改 |
| `agfs.mount_prefix` | `AGFS_MOUNT_PREFIX` | `/local/plugin` | 否 | AGFS 挂载路径前缀 |

### Index Service

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `index.interval` | `INDEX_INTERVAL` | `30` | 否 | 轮询间隔（秒） |
| `index.workers` | `INDEX_WORKERS` | `1` | 否 | 索引 worker 数量 |

### 身份/多租户

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `identity.account_id` | `OG_ACCOUNT_ID` | `acct-demo` | 否 | 默认 account ID |
| `identity.user_id` | `OG_USER_ID` | `u-alice` | 否 | 默认 user ID |
| `identity.agent_id` | `OG_AGENT_ID` | `main` | 否 | 默认 agent ID |

### 记忆与压缩

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `memory.after_turn_threshold` | `OGMEM_AFTER_TURN_THRESHOLD` | `400` | 否 | 会话缓冲区待处理 token 达到阈值后触发 after_turn 增量抽取 |
| `memory.rolling_compress_enabled` | `OGMEM_ROLLING_COMPRESS_ENABLED` | `true` | 否 | 启用 Layer 2 滚动会话压缩 |
| `memory.rolling_compress_fallback_enabled` | `OGMEM_ROLLING_COMPRESS_FALLBACK_ENABLED` | `false` | 否 | LLM 压缩不可用时使用规则 fallback |
| `memory.compact_prepare_token_ttl` | `OGMEM_COMPACT_PREPARE_TOKEN_TTL` | `300` | 否 | `prepare_compaction` 返回 token 的有效期，单位秒 |
| `memory.directory_summary_enabled` | `OGMEM_DIRECTORY_SUMMARY_ENABLED` | `false` | 否 | 启用目录摘要生成 |
| `memory.summary_max_chars` | `OGMEM_SUMMARY_MAX_CHARS` | `4000` | 否 | compact 返回 summary 的最大字符数 |
| `memory.archive_max_count` | `OGMEM_ARCHIVE_MAX_COUNT` | `10` | 否 | archive 合并后的目标保留数量 |
| `memory.archive_merge_threshold` | `OGMEM_ARCHIVE_MERGE_THRESHOLD` | `10` | 否 | archive 数量达到阈值后触发合并 |
| `memory.prefetch_enabled` | `OGMEM_PREFETCH_ENABLED` | `false` | 否 | 启用 `/prefetch`，将候选记忆暂存到 SessionTopicBuffer |
| `memory.prefetch_top_k` | `OGMEM_PREFETCH_TOP_K` | `5` | 否 | 预取候选数量上限 |
| `memory.session_state_bridge_enabled` | `OGMEM_SESSION_STATE_BRIDGE_ENABLED` | `true` | 否 | 将持久化 SessionState 同步到 SessionWindowState |
| `memory.session_state_sync_interval_turns` | `OGMEM_SESSION_STATE_SYNC_INTERVAL_TURNS` | `1` | 否 | SessionState 桥接同步的最小轮次间隔 |
| `memory.topic_detection_enabled` | `OGMEM_TOPIC_DETECTION_ENABLED` | `false` | 否 | 启用实验性话题检测 |
| `memory.compression_quality_enabled` | `OGMEM_COMPRESSION_QUALITY_ENABLED` | `false` | 否 | 计算 archive 压缩质量指标 |
| `memory.compression_quality_persist_metadata` | `OGMEM_COMPRESSION_QUALITY_PERSIST_METADATA` | `false` | 否 | 将压缩质量指标写入 archive metadata |

### 缓存

| YAML 路径 | 环境变量 | 默认值 | 必填 | 说明 |
|-----------|---------|-------|------|------|
| `cache.enabled` | `OGMEM_CACHE_ENABLED` | `true` | 否 | 启用 Embedder/LLM 缓存 |
| `cache.max_size` | `OGMEM_CACHE_MAX_SIZE` | `1000` | 否 | 缓存条目上限 |

---

## 启动示例

### YAML 文件方式（推荐）

```bash
docker run -d --name ogmemory --network host \
  -v ./ogmemory.yaml:/etc/ogmem/config.yaml \
  ogmemory:v0.9
```

### 环境变量方式

```bash
docker run -d --name ogmemory --network host \
  -e OGMEM_API_KEY=your-api-key \
  -e OGMEM_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3 \
  -e OGMEM_LLM_MODEL=doubao-seed-2-0-code-preview-260215 \
  -e OGMEM_EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3 \
  -e OGMEM_EMBEDDING_MODEL=doubao-embedding-vision-250615 \
  -e OGMEM_EMBEDDING_MULTIMODAL=true \
  -e VECTOR_DB_DIMENSION=1024 \
  -e VECTOR_DB_CONNECTION_STRING="host=127.0.0.1 port=8799 dbname=postgres user=sundechao password=xxx" \
  ogmemory:v0.9
```

### SQL 直连最小本地启动

如果目标是验证 `storage.backend=sql` 路径，而不是完整 AGFS / OpenClaw 部署，最小配置可以收敛到：

```bash
pip install -e .[dev,sql]
cp config/ogmem.reference.yaml config/ogmem.yaml

# 编辑 config/ogmem.yaml 里的 storage.connection_string
./scripts/start_sql.sh
```

说明：
- 这个模式只要求本机 PostgreSQL 可连通。
- `./scripts/start_sql.sh` 会优先读取 `config/ogmem.yaml`，不必额外导出环境变量。
- PostgreSQL 参数写在 `storage.connection_string`。
- `vector_db.type: memory` 避免额外依赖 Chroma 或 openGauss vector 扩展。
- `llm.provider: mock` 避免依赖外部 LLM API。

---

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/compose` | POST | 组装当前轮次上下文（由 OpenClaw context engine 调用） |
| `/api/v1/prefetch` | POST | 在 compose 前预取候选记忆并暂存到会话 TopicBuffer |
| `/api/v1/after_turn` | POST | 对话轮结束后回调（记忆提取 + 存储） |
| `/api/v1/ingest` | POST | 消息摄入 |
| `/api/v1/ingest_batch` | POST | 批量消息摄入 |
| `/api/v1/prepare_compaction` | POST | 执行 compact 前增量抽取，返回一次性 `prepareToken` |
| `/api/v1/compact` | POST | 同步提交/归档会话并返回压缩上下文；支持 `prepareToken` |
| `/api/v1/bootstrap` | POST | 引擎初始化 |
| `/api/v1/dispose` | POST | 释放资源 |
| `/api/v1/session_working_set` | GET/POST | 查看按访问时间排序的活跃会话 |
| `/api/v1/evict_idle_sessions` | POST | 按 `maxIdleSeconds` 淘汰空闲会话 |

> 运维注意：`prepare_compaction` 会在返回 token 前执行本轮增量抽取。当前实现中这一步仍在 HTTP worker 内同步运行，大上下文或慢 LLM 会阻塞该请求；生产侧应配置合理的网关超时、worker 数量，并优先让 OpenClaw 使用两阶段 compact，避免在最终 compact 阶段重复抽取。
