# 团队记忆 SQL 存储兼容性

## 版本与能力

团队任务元数据与 AGFS 正文存储的最低 openGauss 支持与验收基线为 **openGauss-lite 7.0.0-RC3（build `2b515af1`）**，数据库使用 **PG 兼容模式、UTF8 编码**。部署时设置 `SQL_DIALECT=opengauss`。

`OpenGaussDialect.supports_batch_claim=False` 选择现有 outbox 的逐条领取、轮询调度，以及图投影队列的 `FOR UPDATE` 路径。团队任务队列采用独立的领取事务与租约，使用数据库支持的 `SKIP LOCKED` 跳过占用行。该语法能力由下面的真实数据库集成用例验证。

| 团队队列方法 | 行锁 | 执行结果 |
| --- | --- | --- |
| `dispatch_source_events` | `FOR UPDATE SKIP LOCKED` | 来源事件分发成功，返回 `1` |
| `claim_job` | `FOR UPDATE SKIP LOCKED` | 领取任务，状态 `RUNNING` |
| `maintain_due_items` | `FOR UPDATE OF i SKIP LOCKED` | 记忆依次转为 `EXPIRED`、`PURGED` |
| `AGFSSpaceBridge.reclaim_deleted_content` | `FOR UPDATE SKIP LOCKED` | 正文回收，`agfs_purged=true` |

2026-11-17 在 WSL 实跑 openGauss-lite 7.0.0-RC3（build `2b515af1`）及 AGFS：完整 schema 初始化、上述四条生产查询、生成发布、ACL、身份撤销与移动回归共 **26 passed**。双连接验证：连接 A 锁住首行，连接 B 使用 `SKIP LOCKED` 领取下一行；A 释放锁后，首行可被正常领取。

5.0.0、6.0.0 的四条生产队列查询及双连接跳锁均实测通过。完整 schema 则有单独的版本要求：6.0.0 在既有 `ALTER TABLE relation_edges ADD COLUMN IF NOT EXISTS` 处报语法错误。完整团队存储按上面的 7.0.0-RC3 基线部署。

## 复验

准备专用测试数据库和 AGFS 实例。数据库角色需具备建表权限，AGFS `/local` 挂载可读写。

```sql
CREATE DATABASE team_queue_test TEMPLATE template0
    ENCODING 'UTF8' DBCOMPATIBILITY 'PG';
```

在仓库根目录设置测试连接：

```bash
export OGMEM_SPACE_TEST_DSN='host=127.0.0.1 port=5432 dbname=team_queue_test user=<test-user> password=<test-password>'
export TEST_OPENGAUSS_CONNECTION_STRING="$OGMEM_SPACE_TEST_DSN"
export AGFS_BASE_URL='http://127.0.0.1:1833'
export OGMEM_SPACE_TEST_CONFIG='/tmp/team-queue-test.json'
python - <<'PY'
import json, os
from pathlib import Path
Path(os.environ['OGMEM_SPACE_TEST_CONFIG']).write_text(json.dumps({
    'agfs_url': os.environ['AGFS_BASE_URL']
}))
PY
python -m pytest -q -s -o addopts= \
  tests/integration/test_space_queue_compat.py \
  tests/integration/test_space_job_review.py \
  tests/integration/test_space_move_projection.py \
  tests/integration/test_space_control_plane.py
```

用例自动创建 schema，并打印实际生产方法的请求和结果；生命周期关键回显为：

```text
request {'method': 'dispatch_source_events', 'uri': 'ctx://...'} response 1
request {'method': 'claim_job/complete_job', 'action': 'ADD'} response {'items': [{'state': 'ACTIVE', ...}], ...}
request {'method': 'maintain_due_items', 'item_id': '...'} response {'state': 'EXPIRED', 'reclaimed': None}
request {'method': 'maintain_due_items', 'item_id': '...'} response {'state': 'PURGED', 'reclaimed': 'true'}
26 passed
```
