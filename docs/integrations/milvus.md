# Milvus 向量索引

Milvus 与 Chroma、openGauss 一样，通过现有 `VectorIndex` Provider 工厂接入。
统一配置 `vector_db.type` 决定使用哪个向量后端；正文、会话和 outbox 继续由现有存储与索引流程管理。

## 配置

在 oGMemory 的现有运行环境中安装可选依赖：

```bash
uv sync --extra milvus
# 或 python -m pip install -e '.[milvus]'
```

使用现有配置文件（参见 `config/ogmem.reference.yaml`），修改向量库段落：

```yaml
vector_db:
  type: milvus
  dimension: 1024  # 必须匹配实际 embedding 输出
  milvus_uri: http://127.0.0.1:19530
  milvus_collection: contextengine
  milvus_db_name: default
  milvus_timeout: 30
```

凭据通过 `MILVUS_TOKEN` 注入，不提交到配置文件。数据库须预先存在；集合和缺失的索引会在启动时创建。
当前使用 HNSW/COSINE，已有集合的字段类型、维度和 VARCHAR 字节容量必须兼容。

沿用 oGMemory 现有服务启动方式和部署配置，无需单独的 Milvus 版 oGMemory Dockerfile 或 Compose。
Milvus 服务由外部部署管理；配置其可达地址。容器中的 `127.0.0.1` 指向容器自身，跨容器连接应使用可解析的服务名或其他可达入口。
现有运行镜像如果未包含 pymilvus，需要在该运行环境安装上述可选依赖。

| YAML 字段 | 环境变量 | 默认值 |
|---|---|---|
| `vector_db.type` | `VECTOR_DB_TYPE` | `chroma` |
| `vector_db.dimension` | `VECTOR_DB_DIMENSION` | `1024` |
| `vector_db.milvus_uri` | `MILVUS_URI` | `http://localhost:19530` |
| `vector_db.milvus_collection` | `MILVUS_COLLECTION` | `contextengine` |
| `vector_db.milvus_db_name` | `MILVUS_DB_NAME` | `default` |
| `vector_db.milvus_timeout` | `MILVUS_TIMEOUT` | `30` 秒 |
| `vector_db.milvus_token` | `MILVUS_TOKEN` | 空 |

YAML 中显式配置的字段优先于环境变量。向量后端切换不会自动迁移历史 Chroma 索引，已有记忆应按现有索引重建流程重新索引。

## 行为与兼容性

- 支持幂等 upsert、向量检索、子节点检索、embedding 复用、记录读取和账户/owner 范围删除。
- 租户、owner、URI、层级、目录和资源类型在 Milvus 内过滤后再执行 top-k。
- 读取采用 Strong consistency，每个适配器 RPC 显式传递超时；启动支持索引缺失恢复及并发创建。
- embedding 转换为实际存储的 FLOAT32 后校验有限且非零，写入前检查 UTF-8 字节上限；超限明确报错，不静默截断。范围删除不将全部 ID 拉回应用内存。
- `get_record` 恢复已持久化的范围字段，标量列优先于元数据中的旧副本，避免读取后重写改变隔离或检索范围。
- 保留 JSON 元数据和 grounding。AGFS/SQL 存储配置由现有 `storage` 段独立决定，Milvus 只负责向量索引。

目前实测组合为 Milvus 2.6.13 + pymilvus 2.6.17、ARM64 Standalone。
可选依赖的 SDK 版本范围不代表服务端兼容矩阵；其他版本、Lite、Distributed/Kubernetes 和托管服务需分别验证。

## 自动化测试

单元测试位于 `tests/unit/providers/test_milvus_adapter.py`，真实服务测试位于 `tests/integration/test_milvus_vector_index.py`。
设置 `OGMEM_TEST_MILVUS_URI` 后可运行真实服务测试；测试仅清理自行创建的 UUID 集合，覆盖隔离、分页读取、范围删除、索引恢复和并发启动。
