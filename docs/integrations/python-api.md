# Python API 对接

## 1. 安装与配置

```bash
pip install -e .
```

SQL 模式：

```bash
pip install -e ".[sql]"
cp config/ogmem.reference.yaml config/ogmem.yaml
```

```yaml
storage:
  backend: sql
  connection_string: "host=127.0.0.1 port=5432 dbname=ogmemory user=postgres password=postgres"

llm:
  provider: openai
  model: gpt-4o-mini
```

LLM、Embedding 和存储参数也可使用环境变量，见 [`../OGMEMORY_ENV.md`](../OGMEMORY_ENV.md)。

Python 有两层入口：

- `MemoryService`：自动从 `OgMemConfig` 组装存储、LLM、Embedding、索引和 SessionManager；优先使用。
- `MemoryWriteAPI`、`ReadAPI`：依赖注入式底层 API；用于自定义组件。

## 2. `RequestContext`

所有底层读写都必须传 `RequestContext`。

```python
from core.models import RequestContext, Role

ctx = RequestContext(
    account_id="acct-1",
    user_id="user-1",
    agent_id="agent-1",
    session_id="session-1",
    trace_id="trace-1",
    role=Role.MEMBER,
    visible_owner_spaces=("user:user-1", "agent:agent-1"),
)
```

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `account_id` | string | 是 | 租户边界 |
| `user_id` | string | 是 | 用户作用域 |
| `agent_id` | string | 是 | Agent 作用域 |
| `session_id` | string | 是 | 会话 ID |
| `trace_id` | string | 是 | 本次调用链 ID |
| `role` | `Role`/string | 否 | 默认 `MEMBER` |
| `visible_owner_spaces` | tuple | 否 | 允许检索的 user/agent 空间 |

不要复用其他用户的 `RequestContext`。

## 3. 高层入口：`MemoryService`

### 3.1 初始化

```python
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService

config = OgMemConfig.load("config/ogmem.yaml")
memory = MemoryService(config=config)
```

`MemoryService` 延迟初始化依赖；首次读写时建立存储、索引和模型对象。

### 3.2 构造上下文

```python
params = {
    "accountId": "acct-1",
    "userId": "user-1",
    "agentId": "agent-1",
    "sessionId": "session-1",
}
ctx = memory.build_context(params)
```

`build_context(params, identity=None) -> RequestContext`：

- `sessionId` 缺省为 `unknown`。
- `accountId`、`userId`、`agentId` 缺省取配置值。
- 传 `ResolvedIdentity` 时执行角色与 Agent 可见性检查。

### 3.3 compose

```python
result = memory.compose({
    **params,
    "_ctx": ctx,
    "prompt": "用户偏好的开发语言是什么？",
    "messages": [
        {"role": "user", "content": "用户偏好的开发语言是什么？"}
    ],
    "tokenBudget": 32000,
})

model_messages = result["messages"]
```

参数和返回字段与 [HTTP `compose`](http-rest.md#4-post-apiv1compose) 相同。`_ctx` 是 Python 内部字段，不序列化到 HTTP。

### 3.4 after_turn

```python
result = memory.after_turn({
    **params,
    "_ctx": ctx,
    "messages": [
        {"role": "user", "content": "以后 Python 项目统一使用 uv。"},
        {"role": "assistant", "content": "收到。"},
    ],
    "forceExtract": True,
})
```

返回状态：

- `accumulating`：写入缓冲，未达到抽取阈值。
- `processing`：抽取任务处理中。
- `completed`：抽取、写入完成。
- `no_new_messages`：没有增量消息。

### 3.5 两阶段 compact

```python
prepared = memory.prepare_compaction({**params, "_ctx": ctx})

result = memory.compact({
    **params,
    "_ctx": ctx,
    "prepareToken": prepared["prepareToken"],
    "tokenBudget": 32000,
    "summaryMaxChars": 4000,
    "shortTermIndexMode": "sync",
})
```

`prepareToken` 是一次性 token；重复消费会抛 `ValueError("invalid prepare token")`。

### 3.6 dispose 与关闭

```python
memory.dispose({**params, "_ctx": ctx})
memory.wait_until_idle({**params, "_ctx": ctx})
memory.shutdown()
```

`dispose()` 非阻塞；进程退出前先用 `wait_until_idle()` 等待待处理记忆写入完成，再调用 `shutdown()` 关闭线程、连接池和后台 worker。

## 4. `MemoryWriteAPI`

### 4.1 初始化

直接初始化：

```python
from fs.sql_adapter import SQLContextFS
from providers.config import ProviderConfig
from service.api import MemoryWriteAPI

provider = ProviderConfig.from_env()
fs = SQLContextFS(
    connection_string=(
        "host=127.0.0.1 port=5432 dbname=ogmemory "
        "user=postgres password=postgres"
    )
)

write_api = MemoryWriteAPI(
    fs=fs,
    llm=provider.create_llm(),
    outbox_store=None,
    schema_registry=None,
    vector_index=None,
    embedder=None,
    uri_resolver=None,
    internal_tool_usage_tracker=None,
    extraction_mode="eager",
)
```

| 构造参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `fs` | `ContextFS` | 是 | 持久化实现 |
| `llm` | `LLM` | 是 | 抽取模型 |
| `outbox_store` | `OutboxStore` | 否 | 索引事件存储；不传时不会注册异步索引事件 |
| `schema_registry` | `SchemaRegistry` | 否 | 不传则自动创建 |
| `vector_index` | `VectorIndex` | 否 | 抽取前已有记忆预取 |
| `embedder` | `Embedder` | 否 | 查询向量生成 |
| `uri_resolver` | `URIResolver` | 否 | URI 路由 |
| `internal_tool_usage_tracker` | tracker | 否 | 内部工具统计 |
| `extraction_mode` | string | 否 | `eager` 或 `lazy`；默认 `eager` |

完整生产初始化优先使用 `MemoryService.get_write_api()`，它会自动提供 Outbox、索引和 Embedder。

### 4.2 `commit_session`

```python
result = write_api.commit_session(
    messages=[
        {"role": "user", "content": "我喜欢 Python。"},
        {"role": "assistant", "content": "收到。"},
    ],
    ctx=ctx,
    confidence_threshold=0.5,
    wait=True,
    session_time=None,
    session_summary="",
    tool_stats_text="",
    archive_id=None,
)
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `messages` | `list[dict]` | 必填 | role/content 消息 |
| `ctx` | `RequestContext` | 必填 | 租户上下文 |
| `confidence_threshold` | float | `0.5` | 低于阈值的候选被过滤 |
| `wait` | bool | `True` | 当前实现主流程同步；False 时结果附带已完成 task 信息 |
| `session_time` | datetime/string | `None` | 时间消歧基准 |
| `session_summary` | string | `""` | 已抽取内容摘要，减少重复 |
| `tool_stats_text` | string | `""` | 工具使用统计文本 |
| `archive_id` | string | 自动生成 | 来源归档 ID |

返回：

```json
{
  "archive_id": "archive-id",
  "candidates_extracted": 2,
  "candidates_filtered": 0,
  "writes_completed": 2,
  "writes_skipped": 0,
  "writes_failed": 0,
  "plans": [
    {
      "action": "create",
      "target_uri": "ctx://acct-1/users/user-1/memories/preferences/python",
      "merged_fields": {}
    }
  ]
}
```

### 4.3 `commit_session_async`

```python
task_id = write_api.commit_session_async(
    messages=[{"role": "user", "content": "我喜欢 Python。"}],
    ctx=ctx,
    confidence_threshold=0.5,
)

status = write_api.get_task_status(task_id)
```

状态结构：

```json
{"status": "processing", "result": null}
```

```json
{"status": "completed", "result": {"writes_completed": 1}}
```

```json
{"status": "failed", "error": "error message"}
```

### 4.4 `write_memory`

已构造结构化候选时跳过 LLM 抽取：

```python
from core.models import CandidateMemory

candidate = CandidateMemory(
    category="preference",
    owner_scope="user",
    routing_key="python-package-manager",
    abstract="用户偏好 uv",
    overview="Python 项目统一使用 uv 管理依赖。",
    content="用户明确要求以后 Python 项目统一使用 uv。",
    confidence=0.98,
    when="2026-08-03",
    who="user-1",
    where=None,
)

result = write_api.write_memory(candidate, ctx)
```

`CandidateMemory` 必填参数：`category`、`owner_scope`、`routing_key`、`abstract`、`overview`、`content`、`confidence`。`owner_scope` 为 `user` 或 `agent`；`confidence` 为 `0.0-1.0`。

返回：

```json
{
  "action": "create",
  "target_uri": "ctx://...",
  "merged_fields": {}
}
```

批量写入：

```python
results = write_api.write_memories(
    candidates=[candidate1, candidate2],
    ctx=ctx,
    parallel=True,
)
```

### 4.5 `write_raw_chunk`

把原始对话块写成 `session_archive`，用于补充结构化抽取可能漏掉的细节：

```python
result = write_api.write_raw_chunk(
    messages=[{"role": "user", "content": "原始消息"}],
    ctx=ctx,
    chunk_index=0,
    session_id="session-1",
    session_time="2026-08-03T17:00:00+08:00",
)
```

返回 `{"action":"create|merge|skip","target_uri":"..."}`。

## 5. `ReadAPI`

生产用法：

```python
read_api = memory.get_read_api()
```

### 5.1 `search_memory`

```python
result = read_api.search_memory(
    query="Python 包管理偏好",
    ctx=ctx,
    top_k=5,
    categories=["preference", "pattern"],
    target_uri=None,
    session_archive=None,
    score_threshold=0.2,
    include_debug=False,
    mode="QUICK",
    fill_content_for_top_k=3,
)

for hit in result.hits:
    print(hit.uri, hit.score, hit.abstract, hit.content_excerpt)
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `query` | string | 必填 | 非空查询 |
| `ctx` | `RequestContext` | 必填 | 租户上下文 |
| `top_k` | integer | `10` | 不能超过配置 `retrieval.max_top_k` |
| `categories` | list | `None` | 类别过滤 |
| `target_uri` | string | `None` | 限定 URI；不得跨 account |
| `session_archive` | dict | `None` | 会话归档提示 |
| `score_threshold` | float | `None` | 最低分 |
| `include_debug` | bool | `False` | 是否保留检索 trace |
| `mode` | string | `QUICK` | `QUICK` 或 `THINKING` |
| `fill_content_for_top_k` | integer | `0` | 为前 N 条填充正文摘要 |

`SearchMemoryResult` 字段：`request_id`、`query`、`typed_queries`、`hits`、`trace`。

服务配置为 `memory.extraction.engine: deterministic` 时，通过 `memory.get_read_api()` 执行的非定向记忆查询可从 `result.message_evidence` 读取原文证据和已核验事实。带类别、目录或目标 URI 限定的查询仍按原范围返回记忆。常规 `memory.compose(params)` 把同一证据检索路径的结果按预算并入 `retrievedEvidence`。

每个 `RetrievedBlock` 包含：`uri`、`score`、`ranking_score`、`category`、`owner_space`、`abstract`、`overview`、`content_excerpt`、`relations`、`metadata`。

### 5.2 `read_memory`

```python
block = read_api.read_memory(
    uri=result.hits[0].uri,
    ctx=ctx,
)

print(block.content_excerpt)
```

`uri` 使用内部 `ctx://` URI。面向外部客户端的 `ogmemory://` URI 转换由 MCP 层负责。

## 6. 完整进程内 Agent 示例

```python
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService

memory = MemoryService(OgMemConfig.load("config/ogmem.yaml"))
identity = {
    "accountId": "acct-1",
    "userId": "user-1",
    "agentId": "agent-1",
    "sessionId": "session-1",
}
ctx = memory.build_context(identity)

def before_model(prompt, history):
    return memory.compose({
        **identity,
        "_ctx": ctx,
        "prompt": prompt,
        "messages": history + [{"role": "user", "content": prompt}],
    })["messages"]

def after_model(prompt, answer):
    return memory.after_turn({
        **identity,
        "_ctx": ctx,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    })

try:
    messages_for_model = before_model("继续昨天的任务", [])
    # answer = call_model(messages_for_model)
    answer = "..."
    after_model("继续昨天的任务", answer)
finally:
    memory.dispose({**identity, "_ctx": ctx})
    memory.wait_until_idle({**identity, "_ctx": ctx})
    memory.shutdown()
```
