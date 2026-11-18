# Standalone 升级兼容与向量表迁移

## AGFS 旧部署

Standalone **默认不编译或打包 AGFS**。需要内置 AGFS 的旧部署必须显式选择
`standalone-agfs` 构建目标；该目标从仓库 `agfs/` 编译，无需宿主机预编译。
新部署保留 SQL 默认后端。选择 `storage.backend=agfs` 后，带绑定的镜像默认在进程内读写；
已有 `agfs.base_url` 继续选择 HTTP。仅当使用 HTTP 且 `agfs.base_url` 为
`http://127.0.0.1:1833`（或 `localhost:1833`）时启动内置服务，并等待其就绪；
配置错误或服务启动失败时不会继续启动 API。

- 没有配置 `storage.backend` 的旧 YAML 仍按 `agfs` 处理。
- 保留旧版 `/data/agfs` 卷挂载；镜像保留 `AGFS_MOUNT_PREFIX=/local/plugin`
  的旧默认值。YAML 中显式指定的 `agfs.mount_prefix` 优先，不要改动已有前缀。
- `storage.backend=sql` 不启动 AGFS；外部 AGFS URL 也不启动本地服务。
- **不要为了升级直接把已有 AGFS 部署改为 SQL**：这不是数据迁移，会使旧文件不可见。

升级步骤：

1. 停止旧应用写入，备份原 YAML、环境变量和 AGFS 宿主机数据目录。
2. 构建时显式指定 `--target standalone-agfs`；Compose 用户在 `deploy/.env`
   设置 `OGMEM_BUILD_TARGET=standalone-agfs` 后重新构建。保留原数据卷和原配置。
3. 查看 `/api/v1/health` 的 `agfs_transport`，读取一条原有记忆，再写入测试记忆。
4. 重启容器，再次读取两条记忆，确认持久化。失败时用原镜像、原配置回滚，
   不要初始化或清空旧数据目录。

已有 SQL 部署不需要修改存储配置。新的示例配置仍可选择 SQL。

本次回归使用实际编译的 AGFS，验证 `/local/plugin` 下升级前文件可读、写入新文件、
服务重启后两者均可读：

```bash
(cd agfs && go build -o /tmp/ogmem-agfs-test ./cmd/server)
OGMEM_TEST_AGFS_BINARY=/tmp/ogmem-agfs-test \
  .venv/bin/python -m pytest tests/integration/test_standalone_agfs_compatibility.py -q
```

## 旧向量表的一次性回填

索引构造仍确保表、列和索引存在，但**不再执行任何历史行回填 UPDATE**。
首次新增列/索引的 DDL 仍可能耗时或等待锁；这与每次启动扫描旧数据是不同操作。

对于还没有显式 `account_id`、`owner_space`、`node_uri`、`node_id` 的旧记录：

1. 备份向量数据库并安排维护窗口，停止写入进程。
2. 如需回填 `node_id`，先确保同一数据库/当前 schema 中已有正确的 `context_nodes`。
   独立向量数据库不存在该表时将跳过关联，不会跨数据库关联；也不会跨账户关联。
3. 设置 DSN 环境变量并执行命令（密码不放命令行参数；表名和维度用实际值）：

```bash
export VECTOR_DB_DSN='host=... port=... dbname=... user=... password=...'
python3 scripts/migrate_vector_index.py --table vector_index --dimension 1536
```

在容器中执行时，脚本路径为 `/opt/ogmem/scripts/migrate_vector_index.py`。
这会在一个事务中执行回填并写入 `ogmem_vector_migrations` 完成记录。
失败会回滚，可修复后重试；成功后重复执行只检查迁移记录，不扫描业务数据。
回填不会覆盖已有账户，也不会修复此前已被覆盖的向量内容。
运行完成后不要继续用旧版本写入缺少显式列的数据。

这是全量维护迁移，可能持锁和产生较多 WAL，不应在业务高峰运行。
如果此前已完成回填，也请运行一次以记录迁移完成状态。

### 可复现启动基准

仅针对 PostgreSQL + pgvector，在独立随机 schema 中生成旧数据，完成后自动删除该
schema。建议使用临时数据库：

```bash
export VECTOR_DB_DSN='host=... dbname=... user=... password=...'
git show <升级前提交>:providers/vector_index/opengauss_index.py > /tmp/vector-before.py
python3 scripts/benchmark_vector_startup.py --rows 1000000 \
  --baseline-file /tmp/vector-before.py
```

基准包含完整构造函数耗时（不含初次建表、创建索引及造数），使用三维向量以控制
测试资源开销；不是生产高维 ANN 吞吐量测试，也不代表原生 openGauss 的性能。

2026-10-09 本地临时 PostgreSQL 16 + pgvector 实测（ARM64、tmpfs 数据目录、
1,000,000 行、3 维向量；同一张既有表，首次建表/索引和造数不计入）：

| 实现 | 第 1 次构造 | 第 2 次构造 | 第 3 次构造 |
|---|---:|---:|---:|
| 修复后（仍为待迁移旧行） | 0.0122 s | 0.0107 s | 0.0104 s |
| 修复前 | 219.8371 s | 1.0298 s | 0.7697 s |

修复前第一次包含历史行回填，后两次仍会运行回填/关联查询；修复后各次均不执行
回填，并验证账户列仍为空的旧行数保持 1,000,000。数值仅反映此测试环境，不能外推
到生产存储、不同维度或 openGauss；迁移开销被移到显式维护命令，并非消失。

## SQL-only test image (without AGFS or the Go build)

For new SQL-backed test deployments, build from the repository root:

```sh
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.standalone -t ogmemory:test-no-agfs .
```

Compose also defaults to `standalone-no-agfs`; no opt-out setting is needed.
Remove any prior `OGMEM_BUILD_TARGET` override or set it to `standalone-no-agfs`, then run:

```sh
docker compose --env-file deploy/.env -f deploy/docker-compose.yml build ogmem
docker compose --env-file deploy/.env -f deploy/docker-compose.yml up -d ogmem
```

Set `storage.backend: sql` and a working `storage.connection_string` in the mounted
`deploy/ogmemory.yaml`. Keep your vector database configuration valid as well.
This does not provide an in-memory replacement for either database. The Python
base image and Python/Debian dependencies still need to be available to the build.

The default target is `standalone-no-agfs`. Use BuildKit (modern Docker's build
backend) so the unused Go/AGFS stages are skipped. To explicitly include AGFS:

```sh
DOCKER_BUILDKIT=1 docker build --target standalone-agfs -f docker/Dockerfile.standalone -t ogmemory:with-agfs .
```

For Compose, set `OGMEM_BUILD_TARGET=standalone-agfs` before building.
 Selecting the no-AGFS target does not migrate AGFS data. A legacy
configuration requesting local AGFS fails fast with an explanatory error; use
SQL for a new test deployment or configure an externally managed AGFS server.
