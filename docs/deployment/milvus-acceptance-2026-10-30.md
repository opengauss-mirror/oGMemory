# Milvus 代码审查修复与复测（2026-10-30）

关联需求：[Issue #87](https://gitcode.com/opengauss/oGMemory/issues/87)。

## 当前状态

代码已同步到 `origin/dev` 基线 `0d3494b945e26a228841316cfe3278c21ecc827f`。
本地相关测试 **783 passed、38 skipped**；远端真实 Milvus 集成测试 **5 passed**。
修复后的全模型链路复测被外部模型服务额度限制阻断，不能据此声明完整链路已重新通过。

2026-10-29 首版真实模型写入、跨会话实体召回、AGFS 读取、应用容器重建及 HTTP 删除结果见[历史报告](milvus-acceptance-2026-10-29.md)。这份历史结果不能替代修复后更严格的 entity 验收。

## 实现与修复

1. 将配置的 `milvus_timeout` 显式传给所有适配器 RPC，包含启动、搜索、写入、删除、计数和 iterator。复现表明 SDK 构造时的 timeout 并没有传给后续搜索调用。
2. 拆分集合 schema 创建、schema 校验、索引确保和加载。已有 schema 缺少索引时可以恢复；只容忍 Milvus 的明确重复创建错误，权限、连接和资源错误保留原始异常。
3. 新验收脚本只允许 entity 命中，必须读取到包含项目标记和 Python 属性的持久化正文。session_archive 和 session_summary 不再能让测试误通过。
4. 在任何批次写入前检查整个 upsert 请求的 UTF-8 字节长度：text 65,535、id 512、范围字段 8,192、metadata JSON 65,536。超限报错，不静默截断。
5. 搜索移除完整正文返回，摘要存入 metadata；按账户/owner 删除改为服务端过滤，不再在应用内累积全部 ID。协议返回删除前 Strong read 的计数，不承诺并发修改下的原子计数。
6. Docker 默认固定 pymilvus 2.6.17，pytest 改为可选构建依赖；本地私有部署 YAML 加入 Git/Docker 忽略规则。

HNSW/COSINE 和 Strong consistency 保持现有策略。本次未引入 Lite/托管服务自动适配，也未未经测量放宽一致性。

## 测试覆盖

| 测试 | 结果 |
|---|---|
| Provider、CLI main/onboard、严格验收脚本单测 | 783 passed、38 skipped |
| 真实 Milvus CRUD、幂等更新、过滤后 top-k、租户/owner 隔离 | 通过 |
| 真实 Milvus JSON/代码元数据和过滤转义 | 通过 |
| 1,005 条写入、分页 embedding 读取、owner 范围删除 | 通过 |
| 保留数据后移除索引，再启动 Provider 自动补建 | 通过 |
| 四个客户端同时首次初始化同一新集合 | 通过 |
| RPC 超时透传、异常保留、UTF-8 超限整批预校验 | 单测通过 |
| 归档/摘要不能替代实体、缺少项目属性不能通过验收 | 单测通过 |
| 改动 Python 文件 Ruff、新增适配器与脚本 mypy/Bandit | 通过 |
| uv.lock 一致性和 diff 空白检查 | 通过 |

38 个跳过项来自未启用的可选依赖/后端。远端测试镜像未装 pytest-asyncio，出现 `asyncio_mode` 配置警告；这些 Milvus 测试为同步测试，5 项均实际执行。

## 远端验证环境

- ARM64 / Milvus 2.6.13 / pymilvus 2.6.17 / Standalone。
- 测试镜像 `ogmemory-milvus:reviewed-20261030`，复用前次镜像的 AGFS 与 Python 依赖，覆盖最新 dev 和修复代码；未修改共享 Milvus 服务。
- 应用重用前次独立 AGFS 目录与集合；集成测试创建 UUID 集合，并在结束后清理。
- 远端日志保存在 `/data3/ogmemory-milvus-20261029/`：`provider-tests-reviewed.log`、`build-reviewed.log`、`health-reviewed.json`、`e2e-reviewed-write.log`、`e2e-reviewed-write-2.log`。

完整模型复测先遇到连接异常，随后 embedding 服务明确返回 `429 AccountQuotaExceeded`（周额度耗尽）。没有改用 Mock，也没有把这次失败记为端到端成功。

## 复现

```bash
uv sync --extra dev --extra milvus
uv run pytest tests/unit/providers tests/unit/cli/test_main.py \
  tests/unit/cli/test_onboard.py tests/unit/deployment/test_milvus_memory_smoke.py -q
OGMEM_TEST_MILVUS_URI=http://127.0.0.1:19530 \
  uv run pytest tests/integration/test_milvus_vector_index.py -q
```

模型服务恢复后，在隔离部署中重新执行：

```bash
python scripts/milvus_memory_smoke.py --write
python scripts/milvus_memory_smoke.py --marker '<marker>'
python scripts/milvus_memory_smoke.py --marker '<marker>' --delete
```

其他 Milvus 小版本、Lite、Distributed/Kubernetes、托管服务，以及压力/容量/长期故障恢复，仍属于未验证范围。切换 Provider 不自动迁移历史 Chroma 索引；AGFS/Milvus 同步继续使用现有 outbox。
