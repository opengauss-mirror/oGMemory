# oGMemory 对接 Milvus

Milvus 作为 `VectorIndex` Provider 接入现有工厂，与 Chroma 同层。记忆正文、会话和 outbox 仍由 ContextFS/AGFS 管理；IndexService 使用现有 Embedder 生成向量，再写入 Milvus。上层抽取和检索流程不需要选择数据库类型。

## 本地源码运行

```bash
uv sync --extra milvus
# 或 python -m pip install -e '.[milvus]'
```

在现有 oGMemory YAML 配置中替换向量库段落：

```yaml
vector_db:
  type: milvus
  dimension: 1024  # 必须与实际 embedding 输出一致
  milvus_uri: http://127.0.0.1:19530
  milvus_collection: contextengine
  milvus_db_name: default
  milvus_timeout: 30
```

凭据通过 `MILVUS_TOKEN` 提供。YAML 字段优先于环境变量；支持 `MILVUS_URI`、`MILVUS_COLLECTION`、`MILVUS_DB_NAME`、`MILVUS_TIMEOUT`。数据库须预先存在；集合不存在时自动创建，已有集合必须具有相同字段、维度和 COSINE 索引。也支持 `ogmem onboard --vector-db milvus --storage-backend agfs`，Milvus 地址使用上述配置或环境变量。

切换 Provider 不会迁移已有 Chroma 数据。已有数据应通过现有索引重建流程重新写入新集合，不能直接把 Chroma 目录挂载给 Milvus。

## Docker 部署（连接已有 Milvus）

本目录复用现有 oGMemory ARM64 基础镜像中的 AGFS 与 Python 运行环境，覆盖当前源码并安装 pymilvus。其他架构可用 `--build-arg OGMEM_BASE_IMAGE=<相应架构的oGMemory镜像>`。

```bash
cp deploy/milvus/ogmemory.example.yaml deploy/milvus/ogmemory.yaml
mkdir -p deploy/milvus/data
# 根据镜像 ogmem 用户的 uid/gid 配置 data 目录写权限。
# 修改 YAML 中的模型、embedding、维度及 Milvus 地址。
# 通过环境提供 OGMEM_API_KEY、OG_ROOT_API_KEY 和
# OG_ADMIN_API_KEYS=milvus-demo:<独立随机管理员密钥>。
docker compose -f deploy/milvus/compose.yaml up -d --build
```

默认发布 `127.0.0.1:18092`。配置和数据目录可以用 `OGMEM_MILVUS_CONFIG`、`OGMEM_MILVUS_DATA` 覆盖，子网可用 `OGMEM_MILVUS_SUBNET` 覆盖。容器中的 `127.0.0.1` 仅指容器自身；连接另一个 Milvus 容器时，将客户端加入 Milvus 所在网络，并用服务名连接。`host.docker.internal` 依赖宿主网关可达；若宿主机的 docker0 已删除，应使用实际可达的网关或容器网络。此 Compose 不创建或重启外部 Milvus。

账户通过 `auth.admin_api_keys`/`OG_ADMIN_API_KEYS` 初始化；root key 本身不创建账户。用户和 Agent 通过现有管理 API 注册。样例采用 AGFS，不启用依赖 SQL 的混合检索。

## 版本与验证边界

当前真实端到端验证组合为 Milvus 2.6.13 + pymilvus 2.6.17（ARM64 Standalone）。Python extra 的 SDK 安装范围不代表服务端兼容矩阵；Lite、其他 Milvus 小版本、Distributed/Kubernetes 和托管服务需独立验收。当前索引策略为 HNSW/COSINE，参数保持固定；不自动为 Lite 或托管服务切换索引。

Docker 默认固定 pymilvus 2.6.17，可通过 `PYMILVUS_VERSION` 构建参数显式覆盖；生产镜像不安装 pytest。需要在镜像内运行集成测试时，使用 `--build-arg INSTALL_TEST_DEPS=true` 并挂载 tests 目录。

## Provider 行为

- VARCHAR 主键沿用 `IndexRecord.id`，upsert 幂等；向量为显式 `FLOAT_VECTOR`，使用 HNSW/COSINE。
- account、owner、URI、level、目录及资源类型在 Milvus 内过滤后再取 top-k。未知过滤字段报错，避免静默忽略范围限制。
- 元数据以 JSON 保存，grounding 保留结构；`_embedding` 单独存储。`memory_directory` 由规范 URI 计算。
- 搜索、计数和 embedding 读取使用 Strong consistency；支持多进程 HTTP/Worker 读写可见性。所有 RPC 显式传入 `milvus_timeout`，不仅限制连接建立。
- 首次启动分别确保 schema、索引和加载状态，重启能补建上次初始化未完成的索引；只容忍并发重复创建，不吞掉网络或权限错误。
- 搜索不返回完整正文，摘要随 metadata 持久化；账户/owner 删除使用服务端过滤，返回删除前可见记录数，不将全量 ID 拉回客户端。并发写入时该计数不承诺与实际删除数原子一致。
- 支持 `search_children`、按原文匹配的 `get_embeddings`、`get_record`、账户/owner 批量删除和连接关闭。
- embedding 必须由现有 Embedder 提供；缺失、零向量、非有限数和维度错误均拒绝。在写入任何一批前检查整个请求中的 UTF-8 字节长度：text 最大 65,535，id 最大 512，范围字段最大 8,192，metadata JSON 最大 65,536。超限明确报错，不静默截断。
- Milvus 与 AGFS 之间仍使用现有 outbox 机制；本接入不提供跨库事务。

## 验证

```bash
uv run --extra dev --extra milvus pytest tests/unit/providers/test_milvus_adapter.py -q
OGMEM_TEST_MILVUS_URI=http://127.0.0.1:19530 \
  uv run --extra dev --extra milvus pytest tests/integration/test_milvus_vector_index.py -q
```

集成测试仅删除其自行创建的 UUID 集合，覆盖真实 CRUD、过滤顺序、跨租户/owner 隔离、重连持久化、schema 维度拒绝及超过 1,000 条记录的分页读取/清理。

完整端到端脚本应在独立测试部署中运行，读取私有 `OGMEM_CONFIG`。需要为配置中的账户及 `<account_id>-isolation` 初始化 admin key，以便创建测试身份。`--write` 会调用真实模型并写入虚构记忆：

```bash
python scripts/milvus_memory_smoke.py --write
# 记录输出的 marker，重启 oGMemory 后：
python scripts/milvus_memory_smoke.py --marker '<marker>'
# 配置 mcp.enable_delete_tools: true 后，可验证 HTTP 删除及 Milvus 索引移除：
python scripts/milvus_memory_smoke.py --marker '<marker>' --delete
```

脚本通过普通用户凭据执行 after_turn、跨会话 query/read 和删除，强制要求 entity 类型包含项目标记和 Python 属性；归档或会话摘要命中不能代替实体记忆。直接查询 Milvus 确认索引存在/移除，并使用第二账户验证 HTTP 隔离。脚本不输出凭据。
