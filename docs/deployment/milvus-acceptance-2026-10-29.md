# Milvus 对接与远端端到端验收（2026-10-29）

> 历史记录：本文件对应首版实现。2026-10-30 的代码审查修复、最新 dev 基线和复测边界见 [后续报告](milvus-acceptance-2026-10-30.md)。历史源码哈希和镜像 ID 不代表当前提交。

## 结论

已基于获取时最新的 `origin/dev` 实现 Milvus Provider，并在远端 ARM64 服务器验证完整真实模型链路。代码位于独立工作树，未修改原工作目录的未提交文件，也未推送上游。

- 基线：`8ddd2b031cc3bba78189310d50455807e59d576a`
- 本地分支：`codex/milvus-provider`
- 工作目录：`/root/.codex/worktrees/milvus-integration/oG-Memory`
- 部署说明：`deploy/milvus/README.md`

## 实现范围

新增 `providers/vector_index/milvus_index.py`，沿用现有 `VectorIndex` 协议、Provider 工厂、Embedder 和 IndexService/outbox 流程。补齐 YAML/环境变量配置及 ProviderConfig 双向投影、CLI 数据库选择、可选依赖和 uv.lock、Dockerfile/Compose/配置示例。

适配器采用显式 schema、VARCHAR 主键、FLOAT_VECTOR 与 HNSW/COSINE。租户、owner、层级、父节点、URI、目录和资源类型在 Milvus 搜索前过滤；JSON 保留 grounding 和业务元数据。支持幂等更新、embedding 原文一致性校验、分页读取、按账户/owner 删除、重连持久化以及已有集合 schema/维度/距离度量校验。缺失或非法 embedding 明确失败。

## 远端部署

| 项目 | 实测值 |
|---|---|
| SSH | `jcp@123.60.114.206:10008` |
| 宿主机 / 架构 | `openGauss237` / `aarch64` |
| 专用目录 | `/data3/ogmemory-milvus-20261029` |
| 应用容器 | `ogmemory-milvus-e2e-20261029` |
| 应用镜像标签 | `ogmemory-milvus:dev-8ddd2b0-final` |
| 最终镜像 ID | `sha256:e0b424561e1019e1ed2ef7830ac008f8533fb35020489ca1c06512c395551aa3` |
| HTTP | 宿主机 `http://127.0.0.1:28120/api/v1/health` |
| Milvus | 复用健康运行的 `milvus-standalone`，实测版本 `pkg/v2.6.13` |
| pymilvus | `2.6.17` |
| 网络 | 应用加入已有 `milvus-latest-net`，通过容器名连接 Milvus |
| 专用集合 | `ogmemory_e2e_20261029`，1024 维 |
| 正文/会话/outbox | 专用目录下的 AGFS 持久化存储 |
| 真实 LLM | `doubao-seed-2.1-lite` |
| 真实 embedding | `doubao-embedding-vision` |

复用了远端现有模型配置，仅在远端私有配置文件中处理凭据。测试账户及集合独立；没有重启共享 Milvus，也没有清理其他应用的数据。测试容器保留运行，专用配置包含私有凭据，不纳入源码交付。

## 验证结果

| 检查 | 结果 |
|---|---|
| 本地 Provider 单元测试 + CLI main/onboard | **737 passed, 38 skipped**；跳过项涉及未安装的可选后端依赖 |
| 远端最终镜像真实 Milvus 集成测试 | **3 passed** |
| >1,000 条记录 | 1,005 条批量写入、分页 embedding 读取及 owner 清理通过 |
| 真实模型抽取 | `status=completed, writes_completed=1, writes_failed=0` |
| 跨会话召回与正文读取 | 通过，使用普通用户凭据 |
| Milvus 索引存在性 | 通过，按公开 URI 转换后的内部 URI 核实 |
| 第二账户 HTTP 查询 | 无结果，隔离验证通过 |
| 重建应用容器后召回 | 通过，复用同一 AGFS 目录和 Milvus 集合 |
| HTTP 删除与异步索引移除 | 通过，随后核实对应 URI 索引记录数为 0 |
| 新增代码 Ruff / mypy / Bandit | 通过 |
| uv.lock 一致性、YAML 加载、git diff --check | 通过 |

最终完整写入使用 marker `青柚Milvus验收9c3280cd6a`，召回实体 URI：

```text
ogmemory://memories/entity/%E9%9D%92%E6%9F%9Amilvus%E9%AA%8C%E6%94%B69c3280cd6a
```

当时专用账户共有 22 条索引记录（含此前验收的正文、摘要和归档）。删除该实体后剩余 20 条；这些数值是验收数据数量，不是性能或吞吐指标。最终重建后另用 `青柚Milvus验收b12711e165` 核实保留数据可读。

## 可复现命令

本地：

```bash
uv sync --extra dev --extra milvus
uv run pytest tests/unit/providers tests/unit/cli/test_main.py tests/unit/cli/test_onboard.py -q
OGMEM_TEST_MILVUS_URI=http://127.0.0.1:19530 \
  uv run pytest tests/integration/test_milvus_vector_index.py -q
```

远端容器内部的完整链路脚本：

```bash
python3 scripts/milvus_memory_smoke.py --write
python3 scripts/milvus_memory_smoke.py --marker '<上一步marker>'
python3 scripts/milvus_memory_smoke.py --marker '<上一步marker>' --delete
```

`--write` 产生模型调用费用和虚构测试数据；`--delete` 需要配置 `mcp.enable_delete_tools: true`。脚本使用配置中的 root key 初始化用户/Agent，再切换普通用户凭据执行记忆操作；所有账户必须先通过 `auth.admin_api_keys` 初始化。

远端验收日志位于 `/data3/ogmemory-milvus-20261029/`：

- `provider-tests-final.log`：最终镜像真实 Milvus 测试。
- `e2e-write-final.log`：完整写入、召回、正文读取和 HTTP 租户隔离。
- `e2e-restart.log`：首次重建容器后的召回。
- `e2e-delete.log`：HTTP 删除及 Milvus 索引移除。
- `e2e-final-restart.log`：最终镜像重建后保留数据的召回。
- `health-final.json`、`build-final.log`：最终健康检查和镜像构建。

## 问题与验证边界

1. 远端 SSH 22 端口关闭，实际使用 10008；宿主机 docker0 已不存在，默认地址池也已耗尽。最终通过现有 Milvus 容器网络直连，无需调整共享服务。
2. 首次模型请求发生连接瞬断，HTTP 返回 503。现有抽取恢复机制随后完成持久化；保留首次失败日志，并在后续独立写入中获得完整成功响应。没有用 Mock 替代模型链路。
3. HTTP 返回 `ogmemory://`，向量库保存 `ctx://`。验收脚本使用现有 URI 转换函数核实索引，避免把两种 URI 直接比较。
4. 本次验证覆盖应用容器重建和 Provider 重连；没有重启共享 Milvus，也没有进行压力、容量、长期稳定性或不同 Milvus 大版本兼容性测试。
5. 切换 Provider 不自动迁移历史 Chroma 索引；正文仍归 AGFS，跨库同步沿用现有 outbox，不声明跨库事务。
6. 全文件 Ruff 检查涉及的既有 `providers/config.py` 和 `cli/commands/onboard.py` 各有原有命名/导入排序问题；新增文件与适配器工厂检查通过，未扩展本次范围去改写存量代码。

## 部署源码校验

已在本地与最终运行容器分别计算 SHA-256，结果一致：

```text
2eba507a268be79278651ccc2c30e88f1cb1f5bc9a383e32c9153ea8a63d473c  providers/vector_index/milvus_index.py
7b15e8631533ff2bdcc0dd08b17d051b1dd4db3ea1c20b1976bf2d83b8039c66  scripts/milvus_memory_smoke.py
```
