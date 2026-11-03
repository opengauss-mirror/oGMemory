# oGMemory 异步索引同步设计

> 版本: v2.0  
> 日期: 2026-03-20  
> 状态: 设计稿

## 1. 概述

本文档描述 oGMemory 异步索引同步的设计方案，采用**调度器定期轮询 + 软锁并发控制**模式，复用现有的 `OutboxWorker.run_once()` 实现。

### 1.1 设计目标

- **异步解耦**: 写入路径与索引更新解耦，不影响主流程性能
- **可靠处理**: 事件持久化到文件系统，支持故障恢复
- **并发安全**: 通过 `.processing` 软锁实现多 Worker 并发抢占
- **简单可靠**: 复用现有组件，最小化新增代码

### 1.2 现有组件复用

| 组件 | 文件 | 说明 |
|------|------|------|
| `OutboxStore` | `commit/outbox_store.py` | 事件持久化，**已实现** |
| `OutboxWorker` | `index/outbox_worker.py` | 核心处理逻辑，**已实现** |
| `run_once()` | `index/outbox_worker.py` | 单轮消费循环，**已实现** |

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              oGMemory 异步索引同步架构                               │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                  写入路径 (已实现)                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ContextWriter.write_candidate()                                                   │
│            │                                                                        │
│            ▼                                                                        │
│   OutboxStore.register_write()                                                      │
│            │                                                                        │
│            ▼                                                                        │
│   ┌─────────────────────────────┐                                                   │
│   │  .outbox/{event_id}.json    │  ◄─── 持久化到AGFS文件系统                        │
│   │  {                          │                                                   │
│   │    "event_id": "uuid",      │                                                   │
│   │    "event_type": "UPSERT",  │                                                   │
│   │    "status": "PENDING",     │                                                   │
│   │    "payload": {...}         │                                                   │
│   │  }                          │                                                   │
│   └─────────────────────────────┘                                                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ 定时触发
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              调度器 (需实现)                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌───────────────────────────────────────────────────────────────────────────┐     │
│   │                     APScheduler                                            │     │
│   │                                                                           │     │
│   │   配置:                                                                    │     │
│   │   - trigger: interval, seconds=30                                         │     │
│   │   - max_instances: 1  (防止并发执行)                                        │     │
│   │   - misfire_grace_time: 10                                                │     │
│   │                                                                           │     │
│   │   调用:                                                                    │     │
│   │   worker.run_once(                                                        │     │
│   │       outbox_store=store,                                                 │     │
│   │       account_ids=get_active_account_ids(),                               │     │
│   │       worker_id="worker-1"                                                │     │
│   │   )                                                                       │     │
│   │                                                                           │     │
│   └───────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           run_once() 内部流程 (已实现)                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌───────────────────────────────────────────────────────────────────────────┐     │
│   │                                                                           │     │
│   │   1. list_pending(account_id)                                             │     │
│   │      └── 扫描 .outbox/ 目录，自动过滤有活跃 .processing 锁的事件            │     │
│   │                                                                           │     │
│   │   2. try_acquire(event, node_uri, worker_id)                              │     │
│   │      └── 写入 .processing 软锁文件                                         │     │
│   │      └── 已被其他 Worker 占用则跳过                                        │     │
│   │                                                                           │     │
│   │   3. process_event(event)                                                 │     │
│   │      └── 调 Embedder 生成向量                                              │     │
│   │      └── 调 VectorIndex.upsert 写入                                       │     │
│   │                                                                           │     │
│   │   4. 结果路由:                                                             │     │
│   │      ├── 成功 → mark_done                                                 │     │
│   │      ├── 可重试 → increment_retry + release                               │     │
│   │      └── 超限 → move_to_dlq                                               │     │
│   │                                                                           │     │
│   │   5. 异常时自动 release 释放锁                                             │     │
│   │                                                                           │     │
│   └───────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
│   返回: {"processed": 12, "succeeded": 10, "failed": 1,                           │
│          "moved_to_dlq": 0, "skipped": 1}                                          │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 多 Worker 并发模型

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              多 Worker 并发处理                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   每次写入操作生成一个新的 event_id，一个节点可能有多个待处理事件                       │
│                                                                                     │
│   ┌───────────────────────────────────────────────────────────────────────────┐     │
│   │  节点 A: profile/.outbox/           节点 B: settings/.outbox/              │     │
│   │                                                                           │     │
│   │  .outbox/                          .outbox/                               │     │
│   │  ├── a1b2c3d4.json                 ├── e5f6g7h8.json                      │     │
│   │  ├── a1b2c3d4.processing ◄─ W1     ├── e5f6g7h8.processing ◄─ W2          │     │
│   │  ├── i9j0k1l2.json                 └── m3n4o5p6.json                       │     │
│   │  └── i9j0k1l2.processing ◄─ W2         (无锁，可被抢占)                    │     │
│   │                                                                           │     │
│   └───────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
│   ┌─────────────────────┐         ┌─────────────────────┐                         │
│   │     Worker-1        │         │     Worker-2        │                         │
│   │   (Thread-1)        │         │   (Thread-2)        │                         │
│   │                     │         │                     │                         │
│   │  扫描所有 .outbox/   │         │  扫描所有 .outbox/   │                         │
│   │                     │         │                     │                         │
│   │  try_acquire(a1b2)  │         │  try_acquire(e5f6)  │                         │
│   │  → 成功，处理中      │         │  → 成功，处理中      │                         │
│   │                     │         │                     │                         │
│   │  try_acquire(i9j0)  │         │  try_acquire(i9j0)  │                         │
│   │  → 失败，跳过        │         │  → 成功，处理中      │                         │
│   │  (已被 W2 锁定)      │         │                     │                         │
│   │                     │         │                     │                         │
│   └─────────────────────┘         └─────────────────────┘                         │
│                                                                                     │
│   锁粒度: 每个 event_id 独立加锁，同一节点的多个事件可被不同 Worker 并行处理           │
│   锁超时: .processing 文件超过 300 秒后自动失效，任何 Worker 都可以重新获取            │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Payload 格式约定

每个事件文件的格式固定如下：

```json
{
  "event_id": "uuid-v4",
  "event_type": "UPSERT_CONTEXT",
  "uri": "ctx://{account}/users/{user}/memories/profile",
  "payload": {
    "records": [
      {
        "id": "sha256(uri:level)[:16]",
        "uri": "ctx://...",
        "level": 0,
        "text": "一句话摘要文本",
        "filters": {
          "account_id": "acme-corp",
          "owner_space": "user_space_alice"
        },
        "metadata": { "category": "profile", "context_type": "MEMORY" }
      },
      { "level": 1, "text": "结构化概述..." },
      { "level": 2, "text": "完整正文..." }
    ]
  },
  "status": "PENDING",
  "retry_count": 0,
  "created_at": "2026-03-20T10:00:00+00:00"
}
```

每个节点最多产生 3 条 record（L0 摘要 / L1 概述 / L2 正文）。`id` 字段由 `sha256(uri:level)` 稳定生成，同一节点同一层级的 id 永远相同，VectorIndex 的 upsert 天然幂等。

---

## 4. AGFS 目录结构

```
ctx://{account}/users/{user}/memories/profile/
  ├── .abstract.md          ← L0 文本来源
  ├── .overview.md          ← L1 文本来源
  ├── content.md            ← L2 文本来源
  ├── .meta.json            ← status=ACTIVE 表示节点写入完整
  ├── .relations.json
  └── .outbox/
        ├── a1b2c3d4.json           ← PENDING 事件文件 (event_id = a1b2c3d4)
        ├── a1b2c3d4.processing     ← 软锁文件（Worker 写入）
        ├── e5f6g7h8.json           ← 另一个事件（同一节点多次写入）
        └── dlq/
              └── i9j0k1l2.json     ← 超过最大重试次数后移入
```

**说明：**
- 每次写入操作生成一个新的 `event_id` (UUID)，因此一个节点可能有多个待处理事件
- 锁文件命名：`{event_id}.processing`，每个事件独立加锁
- 同一节点的多个事件可被不同 Worker 并行处理（因为 VectorIndex.upsert 是幂等的）

---

## 5. 已实现 API

以下全部已实现，直接导入使用即可：

```python
from commit.outbox_store import OutboxStore
from index.outbox_worker import OutboxWorker

worker = OutboxWorker(vector_index=your_index, embedder=your_embedder)
store  = OutboxStore(client=agfs_client)

# 单轮轮询——从调度器里调用这个
stats = worker.run_once(
    outbox_store=store,
    account_ids=["acme-corp", "other-tenant"],
    worker_id="worker-1",   # 每个线程用唯一的 worker_id
)
# 返回：{"processed": 12, "succeeded": 10, "failed": 1,
#        "moved_to_dlq": 0, "skipped": 1}
```

---

## 6. 需交付内容

### 6.1 调度器

调度器负责定期触发 `run_once()`，是整个异步索引同步的入口。

#### 6.1.1 核心配置

```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED

scheduler = BackgroundScheduler()
scheduler.add_job(
    func=lambda: worker.run_once(store, get_active_account_ids(), "worker-1"),
    trigger="interval",
    seconds=30,              # 轮询间隔
    max_instances=1,         # 防止并发执行
    misfire_grace_time=10,   # 错过执行时间的容忍秒数
    coalesce=True,           # 错过的多次执行合并为一次
)
scheduler.start()
```

#### 6.1.2 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trigger` | `"interval"` | 固定间隔触发 |
| `seconds` | `30` | 轮询间隔，根据事件产生频率调整 |
| `max_instances` | `1` | 同一任务最多同时运行实例数，防止重叠执行 |
| `misfire_grace_time` | `10` | 错过执行时间后的容忍秒数，超过则跳过 |
| `coalesce` | `True` | 多次错过的执行合并为一次 |

#### 6.1.3 调度器生命周期

```python
class OutboxScheduler:
    """调度器封装，管理生命周期和错误处理"""
    
    def __init__(
        self,
        worker: OutboxWorker,
        outbox_store: OutboxStore,
        get_account_ids: Callable[[], list[str]],
        worker_id: str,
        interval_seconds: int = 30,
    ):
        self._worker = worker
        self._store = outbox_store
        self._get_account_ids = get_account_ids
        self._worker_id = worker_id
        self._interval = interval_seconds
        self._scheduler = BackgroundScheduler()
        self._running = False
    
    def start(self) -> None:
        """启动调度器"""
        if self._running:
            return
        
        self._scheduler.add_job(
            func=self._run_once_wrapped,
            trigger="interval",
            seconds=self._interval,
            max_instances=1,
            misfire_grace_time=10,
            coalesce=True,
            id="outbox_worker",
            name=f"outbox_worker_{self._worker_id}",
        )
        
        # 注册事件监听
        self._scheduler.add_listener(
            self._on_job_error,
            EVENT_JOB_ERROR | EVENT_JOB_MISSED
        )
        
        self._scheduler.start()
        self._running = True
        logger.info(f"Scheduler started, worker_id={self._worker_id}")
    
    def stop(self, wait: bool = True) -> None:
        """停止调度器
        
        Args:
            wait: 是否等待当前任务完成
        """
        if not self._running:
            return
        
        self._scheduler.shutdown(wait=wait)
        self._running = False
        logger.info(f"Scheduler stopped, worker_id={self._worker_id}")
    
    def _run_once_wrapped(self) -> dict:
        """包装 run_once，添加日志和错误处理"""
        try:
            account_ids = self._get_account_ids()
            stats = self._worker.run_once(
                outbox_store=self._store,
                account_ids=account_ids,
                worker_id=self._worker_id,
            )
            logger.info(f"run_once completed: {stats}")
            return stats
        except Exception as e:
            logger.error(f"run_once failed: {e}", exc_info=True)
            raise
    
    def _on_job_error(self, event) -> None:
        """任务执行错误回调"""
        if event.exception:
            logger.error(
                f"Job {event.job_id} failed: {event.exception}",
                exc_info=event.exception
            )
        elif event.code == EVENT_JOB_MISSED:
            logger.warning(f"Job {event.job_id} missed")
```

#### 6.1.4 获取活跃账户列表

`get_active_account_ids()` 需要自行实现，推荐方式：

**方式一：枚举 AGFS 目录**

```python
def get_active_account_ids() -> list[str]:
    """从 AGFS 枚举所有活跃账户"""
    accounts_dir = f"{mount_prefix}/accounts"
    try:
        entries = agfs_client.list(accounts_dir)
        return [e.name for e in entries if e.is_dir]
    except Exception as e:
        logger.error(f"Failed to list accounts: {e}")
        return []
```

#### 6.1.5 多 Worker 部署

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              多 Worker 部署架构                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌───────────────────────────────────────────────────────────────────────────┐     │
│   │                           AGFS (共享存储)                                   │     │
│   │                                                                           │     │
│   │   /accounts/acme-corp/users/alice/memories/profile/.outbox/               │     │
│   │   /accounts/acme-corp/users/bob/memories/settings/.outbox/                │     │
│   │   /accounts/other-tenant/users/charlie/memories/notes/.outbox/            │     │
│   │                                                                           │     │
│   └───────────────────────────────────────────────────────────────────────────┘     │
│                                        ▲                                            │
│                    ┌───────────────────┼───────────────────┐                        │
│                    │                   │                   │                        │
│   ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐          │
│   │     Thread-1        │ │     Thread-2        │ │     Thread-3        │          │
│   │                     │ │                     │ │                     │          │
│   │  worker_id: "w1"    │ │  worker_id: "w2"    │ │  worker_id: "w3"    │          │
│   │  interval: 30s      │ │  interval: 30s      │ │  interval: 30s      │          │
│   │                     │ │                     │ │                     │          │
│   │  APScheduler        │ │  APScheduler        │ │  APScheduler        │          │
│   │  BackgroundScheduler│ │  BackgroundScheduler│ │  BackgroundScheduler│          │
│   │                     │ │                     │ │                     │          │
│   └─────────────────────┘ └─────────────────────┘ └─────────────────────┘          │
│                                                                                     │
│   并发控制:                                                                          │
│   - 每个线程独立运行调度器，通过 .processing 软锁实现并发控制                         │
│   - worker_id 必须唯一，用于标识锁的持有者                                            │
│   - 锁超时 300 秒，防止 Worker 崩溃后事件无法被处理                                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**多线程 Worker 启动示例：**

```python
import threading
import uuid

def start_worker_thread(worker_id: str, interval_seconds: int = 30):
    """启动单个 Worker 线程"""
    scheduler = OutboxScheduler(
        worker=worker,
        outbox_store=outbox_store,
        get_account_ids=get_active_account_ids,
        worker_id=worker_id,
        interval_seconds=interval_seconds,
    )
    scheduler.start()
    return scheduler

# 启动多个 Worker 线程
worker_count = 3
schedulers = []
for i in range(worker_count):
    worker_id = f"worker-{i+1}"
    t = threading.Thread(
        target=start_worker_thread,
        args=(worker_id, 30),
        name=f"outbox-worker-{i+1}",
        daemon=True,
    )
    t.start()
    schedulers.append(t)

print(f"Started {worker_count} worker threads")
```

**worker_id 生成方式：**

```python
import os
import uuid

def get_worker_id() -> str:
    """获取唯一的 worker_id"""
    return f"worker-{uuid.uuid4().hex[:8]}"
```

#### 6.1.6 优雅关闭

```python
import signal
import sys

scheduler = OutboxScheduler(...)

def signal_handler(signum, frame):
    """处理 SIGTERM/SIGINT 信号，优雅关闭"""
    logger.info(f"Received signal {signum}, shutting down...")
    scheduler.stop(wait=True)  # 等待当前任务完成
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

scheduler.start()
```

#### 6.1.7 监控指标

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SchedulerMetrics:
    """调度器运行指标"""
    worker_id: str
    last_run_time: datetime | None
    last_run_stats: dict | None
    total_processed: int
    total_succeeded: int
    total_failed: int
    total_dlq: int
    consecutive_errors: int
    is_running: bool

# 在 _run_once_wrapped 中更新指标
def _run_once_wrapped(self) -> dict:
    self._metrics.last_run_time = datetime.now()
    try:
        stats = self._worker.run_once(...)
        self._metrics.last_run_stats = stats
        self._metrics.total_processed += stats["processed"]
        self._metrics.total_succeeded += stats["succeeded"]
        self._metrics.total_failed += stats["failed"]
        self._metrics.total_dlq += stats["moved_to_dlq"]
        self._metrics.consecutive_errors = 0
        return stats
    except Exception as e:
        self._metrics.consecutive_errors += 1
        raise
```

### 6.2 补充测试

在现有测试文件里补充以下用例：

**`tests/unit/commit/test_outbox_store.py`**

| 测试用例 | 验证内容 |
|---|---|
| `test_try_acquire_returns_false_when_locked` | 已有有效锁时返回 False |
| `test_try_acquire_respects_timeout` | 锁超过 300 秒后可以被抢占 |
| `test_release_removes_processing_file` | release 后 .processing 文件消失 |
| `test_list_pending_skips_locked_events` | list_pending 不返回有活跃锁的事件 |

**`tests/unit/index/test_outbox_worker.py`**

以下两个用例已提交，验证通过即可，不需要重写：
- `test_run_once_skips_acquired_events`
- `test_run_once_releases_lock_on_exception`


## 7. 集成方式

### 7.1 使用 IndexService（推荐）

`IndexService` 封装了 `OutboxScheduler` 的生命周期管理，提供更简洁的 API：

```python
from pyagfs import AGFSClient
from commit.outbox_store import OutboxStore
from providers.embedder import MockEmbedder
from providers.vector_index import InMemoryVectorIndex
from service import IndexService, setup_signal_handlers

# 初始化依赖
agfs_client = AGFSClient()
outbox_store = OutboxStore(agfs_client)
embedder = MockEmbedder()
vector_index = InMemoryVectorIndex()

# 获取活跃账户列表
def get_active_account_ids():
    return ["acme-corp", "other-tenant"]

# 创建并启动 IndexService
service = IndexService(
    outbox_store=outbox_store,
    embedder=embedder,
    vector_index=vector_index,
    get_account_ids=get_active_account_ids,
    interval_seconds=30,
    worker_count=3,
)

# 设置信号处理（优雅关闭）
setup_signal_handlers(service)

# 启动服务
service.start()
print("IndexService started, press Ctrl+C to stop")

# 保持主线程运行
import signal
try:
    while True:
        signal.pause()
except KeyboardInterrupt:
    service.stop()
```

### 7.2 使用单例模式

```python
from service import init_index_service, get_index_service

# 初始化全局实例
init_index_service(
    outbox_store=outbox_store,
    embedder=embedder,
    vector_index=vector_index,
    get_account_ids=get_active_account_ids,
)

# 获取并启动
service = get_index_service()
service.start()

# 获取运行统计
stats = service.get_aggregated_stats()
# {"worker_count": 3, "total_processed": 100, "total_succeeded": 95, ...}
```

### 7.3 作为独立进程运行

```bash
# 使用 mock embedder（测试）
python -m examples.run_index_service --mock --workers 3 --interval 30

# 使用 OpenAI embedder（生产）
python -m examples.run_index_service --workers 3 --interval 30
```

---

## 8. 与 ContextEngine 集成

### 8.1 完整集成示例

```python
from pyagfs import AGFSClient
from core.interfaces import ContextFS
from commit.outbox_store import OutboxStore
from providers.embedder import MockEmbedder
from providers.vector_index import InMemoryVectorIndex
from providers.llm import MockLLM
from service import (
    MemoryWriteAPI,
    init_write_api,
    IndexService,
    init_index_service,
)

# 1. 初始化 AGFS 和存储
agfs_client = AGFSClient()
context_fs: ContextFS = ...  # 你的 ContextFS 实现
outbox_store = OutboxStore(agfs_client)

# 2. 初始化写入 API（会自动注册 outbox 事件）
llm = MockLLM()
write_api = init_write_api(context_fs, llm, outbox_store)

# 3. 初始化索引服务（会消费 outbox 事件）
embedder = MockEmbedder()
vector_index = InMemoryVectorIndex()

def get_active_account_ids():
    # 从 AGFS 或配置获取活跃账户
    return ["acme-corp"]

init_index_service(
    outbox_store=outbox_store,
    embedder=embedder,
    vector_index=vector_index,
    get_account_ids=get_active_account_ids,
)

# 4. 启动索引服务
get_index_service().start()

# 5. 写入操作会自动触发异步索引
from core.models import RequestContext

ctx = RequestContext(account_id="acme-corp", user_id="alice")
write_api.commit_session(
    messages=[{"role": "user", "content": "I prefer dark mode"}],
    ctx=ctx,
)
# 写入完成后，outbox 事件会被 IndexService 异步处理
```

### 8.2 数据流

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              完整数据流                                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌───────────────────────────────────────────────────────────────────────────┐     │
│   │                           写入路径                                          │     │
│   │                                                                           │     │
│   │   commit_session()                                                        │     │
│   │        │                                                                  │     │
│   │        ▼                                                                  │     │
│   │   ContextWriter.write_candidate()                                         │     │
│   │        │                                                                  │     │
│   │        ├──► 写入 AGFS 节点文件                                             │     │
│   │        │                                                                  │     │
│   │        └──► OutboxStore.register_write()                                  │     │
│   │                  │                                                        │     │
│   │                  └──► 创建 .outbox/{event_id}.json                        │     │
│   │                                                                           │     │
│   └───────────────────────────────────────────────────────────────────────────┘     │
│                                        │                                            │
│                                        │ 异步                                       │
│                                        ▼                                            │
│   ┌───────────────────────────────────────────────────────────────────────────┐     │
│   │                           索引路径                                          │     │
│   │                                                                           │     │
│   │   IndexService (多线程)                                                    │     │
│   │        │                                                                  │     │
│   │        ▼                                                                  │     │
│   │   OutboxScheduler (每 30 秒)                                               │     │
│   │        │                                                                  │     │
│   │        ▼                                                                  │     │
│   │   OutboxWorker.run_once()                                                 │     │
│   │        │                                                                  │     │
│   │        ├──► list_pending() 扫描待处理事件                                  │     │
│   │        │                                                                  │     │
│   │        ├──► try_acquire() 获取锁                                          │     │
│   │        │                                                                  │     │
│   │        ├──► Embedder.embed() 生成向量                                     │     │
│   │        │                                                                  │     │
│   │        ├──► VectorIndex.upsert() 写入索引                                 │     │
│   │        │                                                                  │     │
│   │        └──► mark_done() / move_to_dlq()                                   │     │
│   │                                                                           │     │
│   └───────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```


