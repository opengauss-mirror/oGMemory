# DirectorySummarizer 执行流程文档

## 概述

`DirectorySummarizer` 提供 LLM 摘要生成和 fallback 方法，被 `DirectoryEventHandler` 用于 DAG 风格的目录处理。

### 核心职责

1. **生成聚合摘要**: 调用 LLM 将多个子节点摘要聚合成一个目录级摘要
2. **Fallback 机制**: 当 LLM 失败时，提供简单的摘要生成方法

### 关键组件

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DirectorySummarizer                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  依赖:                                                                     │
│  ├── ContextFS: 读取子节点                                             │
│  └── LLM: 生成聚合摘要                                                  │
│                                                                             │
│  主要方法:                                                                 │
│  ├── _generate_summary(): LLM 摘要生成                                   │
│  ├── _fallback_abstract(): Fallback abstract                             │
│  └── _fallback_overview(): Fallback overview                             │
│                                                                             │
│  输出:                                                                     │
│  └── DirectorySummary: 目录摘要对象                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## DAG 执行模式

目录摘要现在通过 `DirectoryEventHandler` 以 DAG（有向无环图）方式处理：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  DirectoryEventHandler DAG 处理流程                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: UPSERT_DIRECTORY(root_dir) 事件                                     │
│                                                                             │
│  步骤 1: 递归派发所有子目录                                                 │
│  └── _dispatch_dir() 递归处理所有子目录                                    │
│                                                                             │
│  步骤 2: 等待子目录摘要完成                                                 │
│  └── pending == 0 时触发父目录摘要生成                                     │
│                                                                             │
│  步骤 3: 生成父目录摘要                                                     │
│  ├── 收集子节点信息                                                        │
│  ├── 调用 DirectorySummarizer._generate_summary()                         │
│  └── LLM 失败时使用 fallback                                              │
│                                                                             │
│  步骤 4: 写入 ContextFS                                                    │
│  └── write_node(directory_node, ctx)                                      │
│                                                                             │
│  步骤 5: 向量化存储                                                         │
│  └── vector_index.upsert(record)                                          │
│                                                                             │
│  输出: DirectoryEventResult                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 时序图

### DAG 处理时序

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        DirectoryEventHandler DAG 处理时序                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘

OutboxWorker           DirectoryEventHandler        DirectorySummarizer      ContextFS        LLM      VectorIndex
     │                       │                           │                    │              │            │
     │ process_directory_event│                           │                    │              │            │
     ├──────────────────────>│                           │                    │              │            │
     │                       │                           │                    │              │            │
     │                       │ _dispatch_dir(root)       │                    │              │            │
     │                       ├──────────────────────────>│                    │              │            │
     │                       │                           │                    │              │            │
     │                       │ list_children()           │                    │              │            │
     │                       ├──────────────────────────────────────────────────>│            │            │
     │                       │<──────────────────────────────────────────────────┤            │            │
     │                       │                           │                    │              │            │
     │                       │ [递归派发子目录]           │                    │              │            │
     │                       │                           │                    │              │            │
     │                       │ [等待 pending == 0]       │                    │              │            │
     │                       │                           │                    │              │            │
     │                       │ _generate_summary()       │                    │              │            │
     │                       ├──────────────────────────>│                    │              │            │
     │                       │                           │                    │              │            │
     │                       │                           │ complete_json()    │              │            │
     │                       │                           ├───────────────────────────────────>│            │
     │                       │                           │<───────────────────────────────────┤            │
     │                       │                           │                    │              │            │
     │                       │ DirectorySummary          │                    │              │            │
     │                       │<──────────────────────────┤                    │              │            │
     │                       │                           │                    │              │            │
     │                       │ write_node()              │                    │              │            │
     │                       ├──────────────────────────────────────────────────>│            │            │
     │                       │                           │                    │              │            │
     │                       │ upsert()                  │                    │              │            │
     │                       ├─────────────────────────────────────────────────────────────────────────────>│
     │                       │                           │                    │              │            │
     │ DirectoryEventResult   │                           │                    │              │            │
     │<──────────────────────┤                           │                    │              │            │
```

### LLM 失败时 Fallback

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        LLM 失败时使用 Fallback                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘

DirectoryEventHandler      DirectorySummarizer              LLM
     │                           │                          │
     │ _generate_summary()       │                          │
     ├──────────────────────────>│                          │
     │                           │                          │
     │                           │ complete_json()          │
     │                           ├─────────────────────────>│
     │                           │                          │
     │                           │ 抛出异常                 │
     │                           │<─────────────────────────┤
     │                           │                          │
     │                           │ _fallback_abstract()     │
     │                           │ _fallback_overview()     │
     │                           │                          │
     │ DirectorySummary          │                          │
     │<──────────────────────────┤                          │
```

---

### _process_upsert_directory 完整运行逻辑
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  _process_upsert_directory 执行流程                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 前置检查                                                                 │
│  ├── 检查 self._fs 和 self._llm 是否可用                                    │
│  └── 检查 ctx (RequestContext) 是否提供                                     │
│                                                                             │
│  2. 创建 DirectoryEventHandler                                              │
│  └── handler = DirectoryEventHandler(fs, llm, embedder, vector_index)      │
│                                                                             │
│  3. 调用 handler.process_directory_event(event, ctx)                        │
│  │                                                                          │
│  │  3.1 验证 URI 是目录 (以 / 结尾)                                         │
│  │  3.2 运行 asyncio.run(_run_dag())                                        │
│  │                                                                          │
│  │  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  │ DAG 执行流程                                                        │   │
│  │  │                                                                     │   │
│  │  │ _dispatch_dir(root_uri, parent_uri=None)                           │   │
│  │  │   ├── list_children() 获取子节点列表                               │   │
│  │  │   ├── 区分子目录 (children_dirs) 和文件 (file_children)            │   │
│  │  │   ├── 创建 DirNode:                                                 │   │
│  │  │   │   ├── children_dirs: 子目录 URI 列表                           │   │
│  │  │   │   ├── file_children: 文件 URI 列表                             │   │
│  │  │   │   ├── pending: 待完成子节点数 = len(children_dirs) + len(file_children) │
│  │  │   │   └── children_summaries: 子目录摘要缓存                       │   │
│  │  │   ├── 并发派发:                                                     │   │
│  │  │   │   ├── _file_task() 处理文件 (立即完成)                         │   │
│  │  │   │   └── _dispatch_dir() 递归处理子目录                           │   │
│  │  │   └── pending == 0 时触发 _schedule_summary()                       │   │
│  │  │                                                                     │   │
│  │  │ _on_file_done(parent_uri, file_uri)                                │   │
│  │  │   └── pending -= 1, 若 pending == 0 则触发 _summary_task()         │   │
│  │  │                                                                     │   │
│  │  │ _on_child_done(parent_uri, child_uri, child_summary)               │   │
│  │  │   ├── 更新 children_summaries[idx] = child_summary                 │   │
│  │  │   └── pending -= 1, 若 pending == 0 则触发 _summary_task()         │   │
│  │  │                                                                     │   │
│  │  │ _summary_task(dir_uri) [当 pending == 0 时执行]                    │   │
│  │  │   ├── _collect_file_summaries(): 读取文件节点的 abstract           │   │
│  │  │   ├── _collect_children_summaries(): 获取子目录摘要                │   │
│  │  │   ├── _generate_summary_with_fallback():                           │   │
│  │  │   │   ├── 调用 LLM 生成摘要                                        │   │
│  │  │   │   └── LLM 失败时使用 fallback                                  │   │
│  │  │   ├── 创建 ContextNode 并 write_node() 写入 FS                     │   │
│  │  │   ├── _vectorize_directory():                                       │   │
│  │  │   │   ├── 构建 L0 (abstract) 和 L1 (overview) IndexRecord          │   │
│  │  │   │   ├── embedder.embed_texts() 生成向量                          │   │
│  │  │   │   └── vector_index.upsert() 存储                               │   │
│  │  │   └── 通知父目录 _on_child_done(parent_uri, ...)                   │   │
│  │  │                                                                     │   │
│  │  │ 根目录完成后:                                                        │   │
│  │  │   └── self._root_done.set() 结束 DAG 执行                          │   │
│  │  └───────────────────────────────────────────────────────────────────┘   │
│  │                                                                          │
│  4. 返回 WorkerResult                                                       │
│  └── success=True, records_upserted=stats.completed_dirs                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 关键方法说明

### `_generate_summary(directory_uri, child_summaries)`

**功能**: 使用 LLM 生成聚合摘要

**参数**:
- `directory_uri`: 目录 URI
- `child_summaries`: 子节点摘要列表，每个元素包含 `uri`, `abstract`, `category`

**返回值**:
- `DirectorySummary`: 包含 abstract、overview、child_count、categories

**异常处理**:
- LLM 失败时自动使用 fallback 方法生成摘要

---

### `_fallback_abstract(child_summaries)`

**功能**: 在没有 LLM 时生成简单的 abstract

**逻辑**:
```python
categories = set(s["category"] for s in child_summaries)
category_str = "、".join(categories)
return f"包含 {len(child_summaries)} 条{category_str}相关记忆"
```

**示例输出**:
```
包含 5 条preference、event相关记忆
```

---

### `_fallback_overview(child_summaries)`

**功能**: 在没有 LLM 时生成简单的 Markdown overview

**逻辑**:
```python
# 按分类分组
by_category = {}
for s in child_summaries:
    cat = s["category"]
    if cat not in by_category:
        by_category[cat] = []
    by_category[cat].append(s)

# 生成 Markdown
lines = ["# 目录内容概览\n"]
for category, items in by_category.items():
    lines.append(f"## {category} ({len(items)}条)")
    for item in items:
        lines.append(f"- {item['abstract'][:50]}...")
```

**示例输出**:
```markdown
# 目录内容概览

## preference (3条)
- Prefers PEP8 coding style...
- Uses VS Code as primary editor...
- Dark theme preference...

## event (2条)
- Weekly team meeting on Monday...
- Project deadline on Friday...
```

---

## 数据结构

### DirectorySummary

```python
@dataclass
class DirectorySummary:
    abstract: str          # 目录摘要（不超过200字）
    overview: str          # 结构化概述（Markdown格式）
    child_count: int       # 子节点数量
    categories: list[str]  # 分类列表（去重后）
```

### ChildSummary

```python
@dataclass
class ChildSummary:
    uri: str              # 子节点 URI
    name: str             # 子节点名称
    abstract: str         # 子节点摘要
    category: str         # 子节点分类
    is_directory: bool    # 是否为目录
    has_abstract: bool    # 是否有摘要
```

---

## 辅助函数

### `is_directory_uri(uri)`

**功能**: 检查 URI 是否表示目录（以 `/` 结尾）

**示例**:
```python
is_directory_uri("ctx://acme/users/alice/memories/preferences/")  # True
is_directory_uri("ctx://acme/users/alice/memories/profile")       # False
```

---

## 与其他组件的关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        组件依赖关系                                        │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────┐
                    │   OutboxWorker    │
                    └─────────┬─────────┘
                              │
                              │ process_directory_event()
                              ▼
                    ┌───────────────────┐
                    │DirectoryEventHandler│
                    └─────────┬─────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │DirectorySummarizer│ │   ContextFS   │ │  VectorIndex  │
    └───────┬───────┘ └───────────────┘ └───────────────┘
            │
            ├─────────────────┐
            │                 │
            ▼                 ▼
    ┌───────────────┐ ┌───────────────┐
    │      LLM      │ │   ContextFS   │
    └───────────────┘ └───────────────┘
```

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_children` | 50 | 单个目录最大子节点数量 |
| `max_concurrent_llm` | 10 | 最大并发 LLM 调用数 |

---

## 错误处理

1. **子节点读取失败**: 跳过该节点，记录警告日志
2. **LLM 调用失败**: 使用 fallback 方法生成摘要
3. **目录为空**: 不生成摘要
4. **所有子节点读取失败**: 不生成摘要
