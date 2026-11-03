# Operator Service 架构设计

本文档描述 oG-Memory 中可插拔多模态抽取算子服务（Operator Service）的当前实现：在异构图记忆系统之上，提供算子注册中心、DAG 编排、CV 后端抽象与 HTTP API。

> 上游概念（节点类型、ExtractedNode、PageRank、支撑链）见 [`memory_graph.md`](memory_graph.md)。

---

## 1. 设计目标

- **可插拔**：新算子通过代码 `register()` 或 YAML 配置文件挂载，不改框架代码。
- **可编排**：JSON Spec 描述 DAG，支持顺序 / 并行 / 条件分支。
- **后端无关**：CV 类算子通过 `CVBackend` Protocol 抽象，测试用 `MockBackend`、轻量本地用 `OpenCVBackend`、生产用 `HTTPBackend`（对接自建 / YOLOv8 / Roboflow 等）。
- **异步友好**：长编排走后台线程 + `task_id` 轮询，不阻塞 HTTP。
- **零侵入**：扩展 `MultimodalExtractor` Protocol（增 `info` 属性），已有的 `PromptExtractor` 兼容。
- **图谱自动写入**：编排产生的抽取结果由 `OperatorService` 直接落图为 `ExtractedNode` + `extracted_from` 边，复用现有 PageRank / 检索机制。

---

## 2. 分层架构

```
   ┌────────────────────────────────────────────────────────────┐
   │             HTTP API  (server/app.py)                      │
   │  GET  /api/v1/operators[/<op>]                            │
   │  POST /api/v1/operators/<op>/run                          │
   │  POST /api/v1/operators/pipelines/run[_async]             │
   │  GET  /api/v1/operators/tasks/<task_id>                   │
   └─────────────────────────┬──────────────────────────────────┘
                             │
   ┌─────────────────────────▼──────────────────────────────────┐
   │        OperatorService  (service/operator_service.py)      │
   │  · run_operator / run_pipeline / run_pipeline_async       │
   │  · 自动把每步抽取结果写入 GraphService (ExtractedNode)      │
   │  · 后台线程 + TaskStore 跟踪异步任务                       │
   └──────────┬───────────────────────────┬────────────────────┘
              │                           │
   ┌──────────▼────────────┐    ┌──────────▼──────────────────┐
   │ OperatorRegistry      │    │ GraphService  (existing)    │
   │ · code + YAML 注册    │    │ · create_extracted_node()   │
   │ · default_registry()  │    │ · 自动建 extracted_from 边   │
   └──────────┬────────────┘    └─────────────────────────────┘
              │
   ┌──────────▼─────────────────────────────────────────────────┐
   │            OperatorPipeline  (DAG 编排)                    │
   │  顺序 + 并行 + 条件分支 + JSON Spec 解析 + 安全 when 表达式│
   └──────────┬─────────────────────────────────────────────────┘
              │
   ┌──────────▼──────────────────┬──────────────────────────────┐
   │  6 个内置算子 (builtin/)     │  CVBackend Protocol            │
   │  image_caption               │  ├─ MockBackend   (无依赖桩)   │
   │  object_detection            │  ├─ OpenCVBackend (Haar/HOG)  │
   │  face_detection              │  └─ HTTPBackend   (转发远端)  │
   │  pose_detection              │                                │
   │  ocr                         │                                │
   │  document_structure          │                                │
   │  + PromptExtractor           │                                │
   └──────────────────────────────┴──────────────────────────────┘
```

---

## 3. 核心抽象（`extraction/operators/base.py`）

### 3.1 `MultimodalExtractor` Protocol

每个算子必须暴露：

| 成员 | 作用 |
|------|------|
| `supported_types: list[RawDataType]` | 该算子能处理的原始数据类型 |
| `extractor_type: str` | 全局唯一 ID，如 `"object_detection"` |
| `schema_name: str` | 抽取结果使用的 schema 名 |
| `info: OperatorInfo` | 公共元数据（含 `config_schema` JSON Schema） |
| `extract(raw_node, ctx) -> ExtractionResult` | 同步执行 |
| `extract_batch(raw_nodes, ctx) -> list[ExtractionResult]` | 批量（默认逐条调用 `extract`） |

**Duck-typing 检查**：Python 的 `@runtime_checkable` 不支持 Protocol 属性，故提供 `is_multimodal_extractor(obj)` 显式校验 4 个必备成员。

### 3.2 `ExtractionResult`

```python
node_id: str           # 来源 RawNode 的 ID
extracted_content: dict
confidence: float      # 0-1
schema_used: str
error: str | None      # 失败信息；非 None 时该结果被视作失败
```

### 3.3 `OperatorInfo`

注册时对外暴露的元数据，供 `GET /api/v1/operators` 返回：

- `operator_type`, `display_name`, `description`, `version`
- `supported_raw_types`, `schema_name`
- `config_schema: dict | None` —— 该算子参数的 JSON Schema，HTTP API 可据此校验客户端输入

注册时优先用显式传入的 `info`，否则用算子自身的 `info` 属性，最后兜底 `default_info(operator)` —— 保证老的、只实现了三个属性的算子仍可注册。

---

## 4. 注册中心（`extraction/operators/registry.py`）

### 4.1 代码注册

```python
from extraction.operators import default_registry
default_registry().register(MyOperator(...))
```

### 4.2 YAML 注册（`config/operators.yaml`）

```yaml
operators:
  - class_path: extraction.operators.builtin.object_detection.ObjectDetectionExtractor
    kwargs:
      confidence_threshold: 0.5
      max_objects: 50
```

`OperatorRegistry.load_from_config(path)` 按 dotted import path 实例化算子并 `register()`。这样部署时换模型 / 换后端只需改 YAML、不改代码。

`extraction/operators/builtin/__init__.py` 提供 `register_all_defaults(registry)` 一键注册全部 7 个内置算子（含 `PromptExtractor`），供 CLI / 测试使用。生产路径一般走 YAML。

### 4.3 默认单例

进程级 `default_registry()` 单例（`threading.RLock` 保护），并提供 `reset_default_registry()` 给测试使用。

### 4.4 关键方法

| 方法 | 作用 |
|------|------|
| `register(op, info=None, override=False)` | 注册；`override=True` 允许替换同名条目 |
| `unregister(op_type)` | 移除 |
| `get(op_type)` / `try_get(op_type)` | 按 ID 取算子实例 |
| `get_info(op_type)` | 取公共元数据 |
| `list(raw_type=None)` | 按 raw_type 过滤列出 |
| `find_for_raw_type(raw_type)` | 返回该 raw_type 可用的算子实例列表 |
| `load_from_config(path)` | 从 YAML 加载 |

---

## 5. DAG 编排（`extraction/operators/pipeline.py`）

### 5.1 Pipeline Spec

```python
{
  "steps": [
    {"operator": "image_caption", "alias": "cap", "params": {...}},
    {"operator": "object_detection"},
    {
      "parallel": [
        {"operator": "face_detection"},
        {"operator": "pose_detection"}
      ]
    },
    {"operator": "ocr", "when": "results.cap.confidence < 0.5"}
  ]
}
```

每步 `StepSpec` 字段：
- `operator`: 单算子模式的算子类型 ID
- `params`: 该步传入算子的参数（写入 `extracted_content["_params"]`）
- `alias`: 用于在 `when` 表达式中被引用
- `when`: 条件表达式（语法见 5.3）
- `sub_steps`: 非空时表示 parallel 分组

链式构造：`OperatorPipeline().add("image_caption", alias="cap").when("...").parallel("face_detection", "pose_detection")`。

### 5.2 执行语义

- 顺序执行顶层 `steps`；遇到 `parallel` 子步用 `ThreadPoolExecutor(max_workers=4)` 并发。
- 单步失败被捕获进 `StepResult.extraction.error`，不中断后续步骤；并行子步之间不互相可见（不读 peer 结果）。
- 被 `when` 跳过的步会产生 `skipped=True` 的占位 `StepResult`，便于上层统一处理。
- 顶层 `PipelineResult` 聚合 `steps`、`errors`、`raw_node_id`，并提供 `by_alias()` 给后续条件引用。

### 5.3 安全的 `when` 表达式

仅支持受限语法：

```
<expr>     ::= <term> ( ("and"|"or") <term> )*
<term>     ::= "not" <term>
             | "True" | "False"
             | <ident> ( <op> <literal> )?
<ident>    ::= "results" "." <alias> ( "." <attr> )?
<attr>     ::= "ok" | "confidence" | "skipped"
<op>       ::= "<" | "<=" | ">" | ">=" | "==" | "!="
<literal>  ::= number | "True" | "False" | "None"
```

由 `_tokenize` + 递归下降解析器实现，**不允许** 任意 Python 表达式（防止 HTTP 输入注入）。任何不符合语法的 token 抛 `PipelineSpecError`。

### 5.4 关键类型

- `StepSpec` / `PipelineSpec`：spec 数据类
- `StepResult` / `PipelineResult`：执行结果数据类，含 `to_dict()` 序列化方法
- `PipelineSpecError`：spec / `when` 解析错误

---

## 6. CV 后端抽象（`extraction/operators/backends/`）

### 6.1 `CVBackend` Protocol

| 方法 | 输入 | 输出 |
|------|------|------|
| `name` | — | 稳定字符串 ID（`"mock"`/`"opencv"`/`"http"`） |
| `detect_objects(image: bytes, **kwargs)` | 编码后的图片字节 | `list[Detection]` |
| `detect_faces(image: bytes, **kwargs)` | 同上 | 同上 |
| `detect_pose(image: bytes, **kwargs)` | 同上 | 同上（关键点写入 `metadata['keypoints']`） |
| `health_check() -> bool` | — | 是否就绪 |

### 6.2 `Detection` 数据类

```python
label: str
confidence: float
bbox: tuple[float, float, float, float]   # (x1,y1,x2,y2) 归一化到 [0,1]
metadata: dict[str, Any]
```

**坐标全部归一化**，与输入图片分辨率无关。

### 6.3 后端实现

| 后端 | 用途 | 依赖 |
|------|------|------|
| `MockBackend` | 测试 / CI / 默认占位 | 无 |
| `OpenCVBackend` | 轻量本地推理（Haar / HOG / DNN） | `opencv-python` + `numpy`（可选） |
| `HTTPBackend` | 转发到远端 CV 服务 | 任意 HTTP client |

后端 **必须线程安全**，因为并行编排会跨线程调用。

---

## 7. 内置算子（`extraction/operators/builtin/`）

| 算子 ID | RawDataType | 默认后端 | 输出 |
|---------|-------------|----------|------|
| `prompt_extractor` | `PROMPT` | 无（纯文本规则 + 可选 LLM） | intent / subject / style / format / entities |
| `image_caption` | `IMAGE` | Hash 桩（可注入任意 callable） | `description` |
| `object_detection` | `IMAGE` | `MockBackend` | 检测到的物体列表 |
| `face_detection` | `IMAGE` | `MockBackend` | 人脸检测结果 |
| `pose_detection` | `IMAGE` | `MockBackend` | 人体 + keypoints |
| `ocr` | `IMAGE` | `MockBackend` | 文本 + bbox |
| `document_structure` | `DOCUMENT` | 无（纯 Python 文本解析） | 按标题切分章节 |
| `video_frame_extractor` | `VIDEO` | 无（OpenCV 抽帧，无模型） | 抽样帧（base64 JPEG）+ 关键帧标记 |

每个算子构造函数都接收 `backend: CVBackend | None`（如适用）和若干阈值 / 限制参数，方便在 YAML 中配置。

参考配置：`config/operators.reference.yaml` —— 列出全部内置算子的示例 kwargs，并演示如何用 `HTTPBackend` 替换 Mock。

`video_frame_extractor` 是个例外：它不依赖 `CVBackend`，而是直接用 OpenCV 抽帧，输出 `extracted_content["frames"]` 是一个包含 `{frame_idx, ts_sec, keyframe, width, height, image_bytes_b64}` 的列表。下游 `face_detection` / `image_caption` / `object_detection` 算子可以基于每帧的 base64 JPEG 字节继续处理。算子从 `raw_node.metadata["source_path"]`（M3-Bench 加载约定）读取视频文件路径，而不是 `raw_node.content`，因为每个 30s clip 可能几百 MB。

---

## 8. OperatorService（`service/operator_service.py`）

`OperatorService` 是连接算子与图谱的 **唯一** 入口。算子本身不知道图的存在；图也不知道算子的存在；一切由 service 编排。

### 8.1 依赖

```
OperatorService(graph_service, registry=None, task_store=None)
                ↑                ↑                ↑
                │                │                └─ 默认 InMemoryTaskStore
                │                └─ 默认 default_registry()
                └─ 复用现有 GraphService（解析 RawNode、写 ExtractedNode）
```

### 8.2 同步 API

- `list_operators(raw_type=None) -> list[dict]` —— 所有 / 按 raw_type 过滤的算子元数据。
- `describe_operator(op_type) -> dict` —— 单个算子的详情（含 config_schema）。
- `run_operator(raw_node_id, op_type, ctx, params=None, write_to_graph=True)` —— 单算子执行：
  1. `graph_service.get_raw_node(...)` 拿到 `RawNode`（租户隔离）
  2. `registry.get(op_type).extract(raw_node, ctx)` 执行
  3. 若 `write_to_graph=True` 且无错，`graph_service.create_extracted_node(...)` 落图（自动建 `extracted_from` 边）
- `run_pipeline(raw_node_id, pipeline_spec, ctx, write_to_graph=True)` —— DAG 编排，对每个非跳过 / 非失败步骤落图为独立的 `ExtractedNode`，返回 `{extracted_node_id, operator, schema_used}` 列表。

### 8.3 异步 API

- `run_pipeline_async(...) -> task_id`：在后台线程跑完整流水线，立即返回 `task_id`。
- `get_task_status(task_id) -> dict | None`：轮询返回 `TaskRecord.to_dict()`（含 `status`/`result`/`error`/`submitted_at`/`finished_at`/`trace_id`）。

线程与 task_id 在内部 `dict` 中追踪（`threading.Lock`），daemon 线程不阻塞进程退出。

### 8.4 错误

- `OperatorServiceError` —— 基类
- `RawNodeNotFoundError` —— `raw_node_id` 不存在或不可见

---

## 9. 异步任务跟踪（`extraction/operators/task_store.py`）

### 9.1 `TaskStore` Protocol

```
submit(*, raw_node_id, operator_service, trace_id) -> TaskRecord
mark_running(task_id) -> TaskRecord | None
mark_completed(task_id, result) -> TaskRecord | None
mark_failed(task_id, error) -> TaskRecord | None
get(task_id) -> TaskRecord | None
```

### 9.2 `InMemoryTaskStore`

- 线程安全的 dict 实现，单进程可用
- 状态机：`processing` → `completed` / `failed`
- `prune(older_than_seconds=86400)` —— 清理过期任务

接口预留替换点：多进程 / 跨实例部署可实现 SQL / Redis 后端，无需改 `OperatorService`。

---

## 10. HTTP API（`server/app.py`）

所有路由挂在 `/api/v1/operators` 之下：

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | `/api/v1/operators?raw_type=image` | 列出算子 |
| GET | `/api/v1/operators/<op>` | 描述单个算子（含 config_schema） |
| POST | `/api/v1/operators/<op>/run` | 运行单算子，`{raw_node_id, params?, write_to_graph?}` |
| POST | `/api/v1/operators/pipelines/run` | 同步运行流水线 |
| POST | `/api/v1/operators/pipelines/run_async` | 异步，返回 `task_id` |
| GET | `/api/v1/operators/tasks/<task_id>` | 轮询任务状态 |

所有路由通过 `_build_authenticated_context(params)` 构造 `RequestContext`（沿用项目既有鉴权）；错误由 `_error_response` 统一包装。

`_get_operator_service()` 是懒初始化入口：
1. 构造 `GraphService`
2. 调 `register_all_defaults(...)` 注册内置算子（受 try/except 保护，不阻断启动）
3. 若配置 `operators_config_path` 指向的 YAML 存在，调 `load_from_config(...)` 覆盖 / 追加
4. 包装为 `OperatorService` 单例

---

## 11. 典型调用流程

### 11.1 库内调用

```python
from core.models import RequestContext
from service.graph_service import GraphService
from service.operator_service import OperatorService
from providers.graph_store import MemoryGraphStore
from extraction.operators import default_registry
from extraction.operators.builtin import register_all_defaults

registry = default_registry()
register_all_defaults(registry)

graph = GraphService(MemoryGraphStore())
svc = OperatorService(graph, registry=registry)
ctx = RequestContext(account_id="acc1", user_id="u1", agent_id="a1",
                     session_id="s1", trace_id="t1")

# 1) 先把图片作为 RawNode 入图
raw = graph.create_raw_node(
    raw_type=RawDataType.IMAGE, content=b"...",  # 实际是图片字节或 URL
    ctx=ctx,
)

# 2) 单算子：人脸检测 + 自动写 ExtractedNode
result = svc.run_operator(
    raw_node_id=raw.node_id,
    operator_type="face_detection",
    ctx=ctx,
    params={"min_confidence": 0.7},
)
# result["extracted_node_id"] -> 新建的 ExtractedNode

# 3) 流水线：先 caption，confidence 高才走 OCR
task_id = svc.run_pipeline_async(
    raw_node_id=raw.node_id,
    pipeline_spec={
        "steps": [
            {"operator": "image_caption", "alias": "cap"},
            {"operator": "ocr", "when": "results.cap.confidence < 0.5"},
        ]
    },
    ctx=ctx,
)

# 4) 轮询
status = svc.get_task_status(task_id)
```

### 11.2 HTTP 调用

```bash
# 列出所有支持 IMAGE 的算子
curl http://localhost:8080/api/v1/operators?raw_type=image

# 描述算子
curl http://localhost:8080/api/v1/operators/face_detection

# 运行单算子
curl -X POST http://localhost:8080/api/v1/operators/face_detection/run \
     -H 'Content-Type: application/json' \
     -d '{"raw_node_id":"<id>","params":{"min_confidence":0.7}}'

# 异步流水线
curl -X POST http://localhost:8080/api/v1/operators/pipelines/run_async \
     -H 'Content-Type: application/json' \
     -d '{"raw_node_id":"<id>","pipeline":{"steps":[{"operator":"image_caption","alias":"cap"}]}}'
# -> {"task_id":"...","status":"processing"}

# 轮询
curl http://localhost:8080/api/v1/operators/tasks/<task_id>
```

---

## 12. 持久化与状态

- **算子元数据**：进程内 `OperatorRegistry`（不持久化；启动时由 `register_all_defaults` + YAML 重建）。
- **抽取结果**：通过 `GraphService.create_extracted_node` 落到 `graph_nodes` / `graph_edges`（参见 `memory_graph.md` § 持久化）。
- **异步任务**：进程内 `InMemoryTaskStore`（线程安全 dict）。进程重启即丢失 —— 适合 dev / 单进程生产；多进程 / 持久化场景需替换为 SQL / Redis 后端。
- **CV 后端**：取决于所选实现（如 `HTTPBackend` 无状态；`OpenCVBackend` 加载 Haar / HOG 模型到内存）。

---

## 13. 测试覆盖

`tests/unit/extraction/operators/` + `tests/unit/service/` + `tests/unit/server/`：

| 测试文件 | 覆盖 | 用例数 |
|----------|------|--------|
| `test_backends.py` | Mock / OpenCV / HTTP 后端的 detect 与 health_check | 26 |
| `test_builtin_operators.py` | 6 个内置算子 + PromptExtractor 的 extract / 配置 | 26 |
| `test_registry.py` | 代码 / YAML 注册、查找、覆盖、单例 | 18 |
| `test_pipeline.py` | Spec 解析、并行、when 表达式（含非法语法） | 23 |
| `test_task_store.py` | 状态机、prune、并发安全 | 8 |
| `test_operator_service.py` | run_operator / run_pipeline / 异步 + 图谱写入 | 19 |
| `test_operators_api.py` | 6 个 HTTP 路由端到端 | 14 |
| `test_graph.py::TestPromptExtractor` | PromptExtractor 的 4 个抽取维度 | 4 |

合计 **138 个** 算子相关用例。可以用下面的命令运行测试

```bash
uv run python -m pytest tests/unit/extraction/operators/ \
 tests/unit/service/test_operator_service.py \
 tests/unit/server/test_operators_api.py \
 tests/unit/test_graph.py -v
```

`test_graph.py` 还包含 23 个图本体用例（`TestGraphModels` 7 + `TestMemoryGraphStore` 9 + `TestGraphService` 7），命令共收集 **161 个** 用例。

端到端覆盖：图谱写入、并行编排、异步任务、HTTP 全流程。

---

## 14. 已知边界与扩展点

- **任务持久化**：`InMemoryTaskStore` 单进程，重启即丢；`TaskStore` Protocol 预留 SQL / Redis 后端接口。
- **算子远程注册**：未实现 `POST /api/v1/operators/register`（运行时 class_path 注入）；当前通过 YAML 文件变更 + 重启生效。
- **算子市场**：未实现元信息搜索 / 标签 / 评分。
- **结果缓存**：相同 `raw_node_id + params` 的结果不命中（每次重跑），未做去重。
- **流水线表达力**：`when` 表达式只支持受限语法；不支持跨流水线状态共享 / 嵌套流水线 / Map（per-item fan-out）等模式。
- **流式 / 视频**：`video_frame_extractor` 已经实现（Phase 1+）；当前返回的 `extracted_content["frames"]` 由调用方负责落图为独立 `IMAGE` RawNode（`m3.load_clips` 已经在每个 30s clip 上建好 VIDEO 节点，调用方再把抽帧结果接进去）。完整「视频 → 帧 RawNode → 抽取 → episodic」的端到端 pipeline 还在 Phase 1 收尾。
- **CV 后端覆盖率**：`OpenCVBackend` 受限于 OpenCV 内置模型；高精度场景需 `HTTPBackend` 接 YOLOv8 / Roboflow 等。
- **认证 / 限流**：HTTP 路由沿用项目既有 `_build_authenticated_context`；没有专门的算子级速率限制。
