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

### 记忆检索模式

在服务端的 `config/ogmem.yaml`（或 `OGMEM_CONFIG` 指定的文件）配置。HTTP、MCP 和各 Agent 插件共用服务端的设置。

| YAML 路径 | 环境变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `retrieval.mode` | `OGMEM_RETRIEVAL_MODE` | `default` | `default` 保留向量写入和检索。`vector_free` 要求 `storage.backend: agfs`，写入和检索均跳过 embedding，可删除 `embedding` 与 `vector_db`，保留抽取所需的 `llm`。 |
| `retrieval.inverted_index.provider` | `OGMEM_RETRIEVAL_INVERTED_INDEX_PROVIDER` | 见下文 | `none`、`sqlite_fts` 或 `opengauss_native`。显式选择索引后，索引维护独立于混合检索开关。 |
| `retrieval.hybrid_enabled` | `OGMEM_RETRIEVAL_HYBRID_ENABLED` | `false` | `default` 模式下，`true` 融合向量与倒排结果，`false` 仅查向量。`vector_free` 直接查选定的倒排，示例统一设为 `false`。 |
| `retrieval.agfs_fts_ranker` | `OGMEM_RETRIEVAL_AGFS_FTS_RANKER` | `literal` | 仅用于 `vector_free` + `sqlite_fts`。`literal` 对 AGFS 回读结果做字面评分，`sqlite_bm25` 按 FTS5 的 BM25 顺序排序。 |
| `retrieval.inverted_index.options.path` | — | — | `default` + `sqlite_fts` 必填。使用 API 与索引 Worker 共同读写的持久化 SQLite 文件路径。容器部署应共享可写目录。`vector_free` + `sqlite_fts` 无需该项。 |
| `retrieval.inverted_index.options.connection_string` | — | 见下文 | `opengauss_native` 的数据库连接。优先取该项，其次取 `vector_db.connection_string`，最后取 `storage.connection_string`。目标库须支持原生 BM25，建议 AGFS 部署显式填写该项。 |
| `retrieval.inverted_index.options.table_name` | — | `lexical_index_opengauss_native` | `opengauss_native` 使用的倒排表名。首次建表和建索引时，数据库账号须在目标 schema 拥有相应权限。 |

**本地参考配置**使用 `storage.backend: sql`、`storage.dialect: opengauss`、`retrieval.mode: default`、`hybrid_enabled: false` 和 `vector_db.type: chroma`，即 openGauss 保存记忆原文，Chroma 承载向量检索，倒排索引默认关闭。首次执行本地向导时，若未显式设置混合检索环境变量，向导会在探测到原生 BM25 和 schema 权限后配置 `opengauss_native` 并开启混合检索。未加载配置文件且未设置 `STORAGE_BACKEND` 时，代码的主存储回退值是 `agfs`。

**未通过配置或环境变量指定 provider**时，`vector_free` 默认选择 `sqlite_fts`，旧的 `agfs_fts_enabled: false` 会选择 `none`。`default` 在混合关闭时选择 `none`，开启时缺省选择 `opengauss_native`。整个 `inverted_index` 块省略且混合开启时，会读取旧的 `retrieval.lexical`。新配置建议明确指定 provider。旧的 `retrieval.mode: agfs` 仍解析为 `vector_free`。

#### AGFS 无向量 + 内置 SQLite FTS5

替换现有配置中的对应块。示例连接 HTTP AGFS，`mount_prefix` 要与 AGFS 实际挂载一致。使用 native AGFS 时保留原有 `agfs` 配置。

```yaml
storage:
  backend: agfs
agfs:
  transport: http
  base_url: "http://127.0.0.1:1833"
  mount_prefix: "/local/plugin"
retrieval:
  mode: vector_free
  hybrid_enabled: false
  inverted_index:
    provider: sqlite_fts
  agfs_fts_ranker: literal
```

AGFS 保存记忆原文和元数据。各服务进程按访问范围构建临时 FTS5 索引，覆盖 L0、L1、L2 与部分节点元数据，重启后按需重建。仅需 AGFS 字面扫描时，将 `provider` 改为 `none`。两种配置均可移除 `embedding` 和 `vector_db`。请关闭 `shared_spaces.enabled`、`features.code_retrieval_enabled`、`scene.enabled` 和 `skills.sedimentation.enabled`，这些功能缺省关闭。

#### AGFS 无向量 + openGauss BM25

沿用上例的 `storage` 和 `agfs`，将 `retrieval` 改成以下内容。AGFS 继续保存原文，openGauss 持久化倒排。回填期间可走 AGFS 字面扫描。

```yaml
retrieval:
  mode: vector_free
  hybrid_enabled: false
  inverted_index:
    provider: opengauss_native
    options:
      connection_string: "host=127.0.0.1 port=15432 dbname=ogmemory user=gaussdb password=CHANGE_ME"
```

#### 向量 + 倒排

继续使用原有的 `storage`、`embedding` 和 `vector_db` 配置。下面用持久化 SQLite 做倒排，也可改用上一例的 `opengauss_native` 及对应 `options`。

```yaml
retrieval:
  mode: default
  hybrid_enabled: true
  inverted_index:
    provider: sqlite_fts
    options:
      path: /data/ogmemory/lexical.db
```

设 `hybrid_enabled: false` 可让查询只用向量，明确配置的倒排索引仍随写入更新。改动 provider 或连接后，让 API 及对应的索引 Worker 读取相同配置并重启。切回向量模式时，为无向量期间新增或更新的记忆重建向量索引。

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
| `storage.backend` | `STORAGE_BACKEND` | 本地参考配置和向导为 `sql` | 否 | `sql` 或 `agfs`。 |
| `storage.dialect` | `SQL_DIALECT` | 本地参考配置和向导为 `opengauss` | 否 | openGauss 数据库须使用 `DBCOMPATIBILITY='PG'`。使用 PostgreSQL 时显式配置 `postgresql`。 |
| `storage.connection_string` | `STORAGE_DB_CONNECTION_STRING`（旧名 `SQL_CONNECTION_STRING` 仍生效） | 无 | `backend: sql` 时必填 | libpq DSN，例如 `host=127.0.0.1 port=15432 dbname=ogmemory user=gaussdb password=CHANGE_ME`。 |
| `storage.pool_size` | `STORAGE_DB_POOL_SIZE`（旧名 `SQL_POOL_SIZE`） | `5` | 否 | SQL storage 连接池大小 |

团队任务与 AGFS 正文存储采用 openGauss 7.0.0-RC3、PG/UTF8 基线；版本能力和实际请求见[团队记忆 SQL 存储兼容性](deployment/team-memory-sql-compatibility.md)。

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
  -e VECTOR_DB_CONNECTION_STRING="host=127.0.0.1 port=8799 dbname=postgres user=your_username password=xxx" \
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
- 先准备可连接的 openGauss 数据库，创建业务库时指定 `DBCOMPATIBILITY='PG'`。
- `./scripts/start_sql.sh` 会优先读取 `config/ogmem.yaml`，不必额外导出环境变量。
- 主存储连接参数写在 `storage.connection_string`。
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

---

## 并行与可靠性参数（parallel-reliability knobs）

以下环境变量在并行化与可靠性改造（spec `parallel_dockerize.md`，Workstream 2–4）中引入，用于在有界资源下控制并发度、连接池大小、重试预算和降级行为。

| 环境变量 | 默认值 | 作用位置 | 用途 |
|---------|--------|---------|------|
| `OGMEM_M3_MAX_WORKERS` | 8 | `m3/_parallel.env_max_workers` | 每片段 + 每问题的线程池大小（驱动、QA、eval judge） |
| `OGMEM_M3_FACE_WORKERS` | 4 | `m3/_parallel.face_max_workers` | 每帧人脸检测扇出上限（cv_service 串行化 detect） |
| `OGMEM_GRAPH_POOL_SIZE` | 32 | `m3/load_clips`、`m3/resume` | SQL graph-store 连接池；保持 ≥ worker 数 |
| `OGMEM_VECTOR_POOL_SIZE` | 5（或 YAML） | `m3/config.load_m3_runtime_config` | Vector-index 连接池 |
| `OGMEM_EMBED_MAX_CONCURRENCY` | 8 | `providers/embedder/openai_embedder` | 限制在途 embed 调用数 |
| `OGMEM_EMBED_RETRIES` | 2 | embedder | 瞬态 embed 失败的抖动退避尝试次数 |
| `OGMEM_EMBED_BASE_DELAY` | 1.0 | embedder | 基础退避延迟（秒） |
| `OGMEM_EXTRACT_MAX_WORKERS` | 4 | `extraction/tools` | Phase-2 span dual-run 并行度 |
| `OGMEM_EXTRACT_SPAN_TIMEOUT` | 180 | `extraction/tools` | 一个并行 span 块的等待预算（秒）——每当任一 span 完成时预算重置，因此一个病态滞后者只有在**没有任何 span 完成超过该秒数**时才会被丢弃（不是严格的每-span 硬截止时间） |
| `OGMEM_EXTRACT_SPAN_RETRIES` | 1 | `extraction/tools` | 丢弃 span 前的额外尝试次数 |
| `OGMEM_LLM_TIMEOUT` | 120 | `providers/llm/openai_llm` | OpenAI 客户端读/连接超时（秒） |
| `OGMEM_ASR_FAIL_RATIO` | 0.15 | `m3/voice_asr`、`m3/pipeline/stages` | 每片段 ASR 失败容忍比率，超过则判定该视频失败 |
| `OGMEM_TENANT_SCOPE` | m3bench | compose（ogmem service） | 后台 worker 的多租户隔离守卫（**尚未接入**：等待 WS 1.3 compose 变更；在此之前此表项仅作为规格记录） |
| `ORTSESSION_INTRA_OP_THREADS` | 8 | `scripts/cv_service` | onnxruntime intra-op 线程上限（避免 192 线程抖动） |
| `OMP_NUM_THREADS` | 8 | compose（runner） | pyannote 的 torch intra-op 线程（比默认快 23×） |

> **调参建议：** 在嵌入式服务饱和时，先将 `OGMEM_EMBED_MAX_CONCURRENCY` 调低（如 4）以减少 429/超时；`OGMEM_EXTRACT_SPAN_TIMEOUT` 在慢 LLM 下可调至 300–600；`OGMEM_GRAPH_POOL_SIZE` 应 ≥ `OGMEM_M3_MAX_WORKERS` 以避免连接饥饿。
