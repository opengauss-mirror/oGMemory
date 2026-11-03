# oGMemory 特性能力与使用指南

oGMemory 帮助 Agent 保存和查找长期记忆、在连续任务中使用上下文，并从执行过程积累可复用的经验与技能。应用可对接代码和多模态材料，通过统一身份在多个客户端之间使用记忆。

## 特性总览

| 特性 | 使用效果 |
| --- | --- |
| [记忆写入](#记忆写入) | 将原话、对话和业务交互保存为记忆，持续补充和更新事实。 |
| [记忆检索](#记忆检索) | 按问题查找相关记忆，读取回答所需的摘要与详情。 |
| [记忆管理](#记忆管理) | 浏览、盘点和读取记忆，按用户要求遗忘指定内容。 |
| [上下文管理](#上下文管理) | 跨会话使用记忆，压缩长对话和工具结果，控制模型输入。 |
| [技能学习与复用](#技能学习与复用) | 导入、学习和维护操作技能，在后续任务中复用。 |
| [工具与经验学习](#工具与经验学习) | 从任务执行过程学习工具约束和解决方法。 |
| [场景记忆](#场景记忆) | 关联同一活动的多项事实，支持综合理解和查询。 |
| [记忆关系网络](#记忆关系网络) | 连接跨记忆的依赖与业务关系，沿关联路径补充回答所需事实。 |
| [代码理解与编辑](#代码理解与编辑) | 定位代码实现、读取源码证据并执行精确编辑。 |
| [多模态记忆](#多模态记忆) | 处理文档、图像、音频和视频，检索内容并追溯来源。 |
| [记忆共享与权限管理](#记忆共享与权限管理) | 为团队划分知识范围，共享指定记忆和技能。 |
| [可视化管理](#可视化管理) | 查看记忆、技能、场景、处理进度及模型用量。 |

## 使用准备

准备已运行的 oGMemory HTTP 服务、LLM、Embedding 和持久化存储。服务启动与连接配置见 [快速开始](QUICKSTART.md)、[配置说明](OGMEMORY_ENV.md)。配置片段合并到服务使用的 YAML 文件后，重启服务使其生效；明确支持热更新的场景配置可通过管理接口修改。

SQL 主存储和向量索引使用同一数据库连接与匹配的方言。HTTP、MCP 和索引服务使用相同的存储、身份和特性配置。

在服务配置中启用角色控制并设置管理密钥，重启 HTTP 和 MCP 服务：

```yaml
auth:
  role_control_enabled: true
  root_api_key: "<服务端 Root Key>"
```

管理密钥也可通过 `auth.admin_api_keys` 配置。

| 信息 | 用途 |
| --- | --- |
| HTTP 地址 | 访问记忆、会话、代码资源和管理接口 |
| MCP 地址 | MCP 客户端发现并调用工具 |
| 账户、用户、Agent | 标识数据归属和可见范围 |
| API Key | 认证当前调用者，执行已授权操作 |
| 会话 ID | 关联连续对话、写入与工具结果 |

在客户端设置实际服务地址与身份：

```bash
export OGMEM_CONFIG='<服务配置文件路径>'
export REPO_DIR='<oGMemory源码目录>'
export RUN_ID="$(python -c 'import uuid; print(uuid.uuid4().hex[:10])')"
export OGMEM_URL='http://127.0.0.1:8090'
export OGMEM_MCP_URL='http://127.0.0.1:8091/mcp'
export OG_ACCOUNT_ID='acct-dev'
export OG_USER_ID='user-dev'
export OG_AGENT_ID='agent-dev'
export OG_AUTH_API_KEY='<oGMemory API Key>'
```

在源码目录安装示例依赖，后续 Python 命令使用同一环境：

```bash
cd "$REPO_DIR"
python -m pip install -e .
```

创建实际账户、用户和已授权 Agent，示例统一使用所属账户的 Admin Key。`OG_ACCOUNT_ID`、`OG_USER_ID`、`OG_AGENT_ID` 与该密钥对应的身份保持一致。服务端 Root Key 用于管理账户；`OG_AUTH_API_KEY` 填写账户 Admin Key，与模型服务的 LLM、Embedding 密钥分别配置。

每项特性的 Python 示例按顺序在同一 Python 会话中执行，先运行下面的公共调用代码。Shell 准备命令在启动该会话前执行，使导出的变量可被 Python 读取。

```python
import json
import os
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE = os.environ["OGMEM_URL"].rstrip("/")
ACCOUNT = os.environ["OG_ACCOUNT_ID"]
USER = os.environ["OG_USER_ID"]
AGENT = os.environ["OG_AGENT_ID"]
RUN_ID = os.environ["RUN_ID"]
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": os.environ["OG_AUTH_API_KEY"],
    "X-Account-ID": ACCOUNT,
    "X-User-ID": USER,
    "X-Agent-ID": AGENT,
}

def api(method, path, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(BASE + path, data=data, headers=HEADERS, method=method)
    with urlopen(request, timeout=600) as response:
        return json.load(response)

def wait_idle(session, timeout=180, *, agent=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = api("POST", "/api/v1/call/wait_until_idle", {
            "agentId": AGENT if agent is None else agent, "sessionId": session, "timeoutSeconds": 10,
            "drainOutbox": False, "waitOutbox": True,
        })
        if result.get("idle"):
            return result
        if result.get("outbox", {}).get("failed", 0):
            raise RuntimeError(result)
        time.sleep(1)
    raise TimeoutError(f"会话处理超时：{result}")

def remember(text, session, request_id):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        result = api("POST", "/api/v1/memories/write", {
            "agentId": AGENT, "sessionId": session, "text": text,
            "client_request_id": request_id, "force_extract": True, "wait": True,
        })
        if result.get("ok") and result.get("status") in {"completed", "duplicate"}:
            wait_idle(session)
            return result
        if not result.get("retryable"):
            raise RuntimeError(result)
        time.sleep(1)
    raise TimeoutError(f"写入处理超时：{result}")

def find_memory(query, marker, types=None, timeout=90, *, agent=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        params = {"agentId": AGENT if agent is None else agent, "query": query, "top_k": 20}
        if types:
            params["types"] = types
        try:
            found = api("POST", "/api/v1/memories/query", params)
        except HTTPError as error:
            detail = json.loads(error.read())
            if error.code == 500 and detail.get("error") == "required lexical backfill is not ready":
                time.sleep(1)
                continue
            raise
        for hit in found["hits"]:
            if marker in (hit.get("abstract", "") + hit.get("content_excerpt", "")):
                return hit
        time.sleep(1)
    raise TimeoutError(f"记忆索引等待超时：{marker}")

scope = api("GET", "/api/v1/memory-viewer/scope")
assert scope["role_control_active"] and scope["admin_mode"], scope
assert (scope["account_id"], scope["user_id"]) == (ACCOUNT, USER), scope
assert any(row["agent_id"] == AGENT for row in scope["agents"]), scope
print(scope["account_id"], scope["user_id"], AGENT)
print(api("GET", "/api/v1/health"))
```

将示例的 `acct-dev`、`user-dev`、`agent-dev` 替换为相应身份。查询返回的 URI、游标、任务 ID 和 revision 直接用于后续调用。写入以返回状态、持久化正文和检索结果共同验证；异步接收响应后，待处理与索引完成再验证召回。

MCP 客户端按工具名调用服务，并通过 `arguments` 传入参数；客户端展示的工具名可能带服务名前缀。每项效果验证创建自己的示例事实或输入材料，使用 `RUN_ID` 区分运行。文件类能力的路径由调用方传入，生成器先创建样例文件和输出目录，再提交对应接口。

## 记忆写入

### 能力介绍

oGMemory 将用户原话、对话消息和业务交互转为可长期使用的记忆，支持单条保存、完整轮次提交和批量消息接入。应用提交原始内容后，服务提取信息、识别类型并保存摘要、概览与正文。

应用可持续提交新的项目事实、工作偏好和业务进展，使用会话标识关联连续交互，请求标识支持客户端重试去重。

| 输入内容 | 典型记忆 |
| --- | --- |
| 身份与背景 | 用户画像 |
| 工作、沟通和开发习惯 | 偏好 |
| 项目、系统和组织信息 | 实体 |
| 会议、行动和发生事项 | 事件 |
| 问题、处理过程与结果 | 案例 |
| 可重复使用的规律和方法 | 模式 |

### 使用指南

启用写入接口：

```yaml
mcp:
  enable_write_tools: true
```

创建独立项目事实，保存技术栈和发布要求：

```python
project = f"研发发布-{RUN_ID}"
session = f"write-session-{RUN_ID}"
initial = remember(
    f"请记住：{project}项目使用 Python 和 openGauss，发布前必须完成接口回归和安全扫描。",
    session, f"write-initial-{RUN_ID}",
)
print(initial["status"], initial["stats"])
```

Agent 完成一轮交互后，可一次提交有序消息。每条消息使用 `role`、`content`，并可携带来源标识与时间：

```python
turn = api("POST", "/api/v1/after_turn", {
    "agentId": AGENT, "sessionId": f"write-turn-{RUN_ID}",
    "messages": [
        {"role": "user", "content": f"{project}项目的发布窗口为每周四晚上八点。"},
        {"role": "assistant", "content": "发布窗口已记录。"},
    ],
    "forceExtract": True, "wait": True,
    "clientRequestId": f"write-turn-request-{RUN_ID}",
})
print(turn["ok"], turn["status"], turn.get("writes_completed", 0))
```

HTTP `/api/v1/memories/write` 接收 `text`；MCP `ogm_write_memory` 接收 `text` 或 `messages`。用户要求保存时设置 `force_extract=true`，需要等待结果时设置 `wait=true`。生命周期接口 `/after_turn` 使用对应的 `forceExtract` 和 `clientRequestId` 字段。

批量接入历史消息使用 `/api/v1/ingest_batch`，随后通过 `/after_turn` 提交轮次。相同请求重试保持会话、请求标识和内容一致，新的交互使用新的请求标识。

### 效果验证

写入返回 `ok=true`、`status=completed` 后，等待关联处理结束，再在新会话中查询并读取：

```python
wait_idle(session)
wait_idle(f"write-turn-{RUN_ID}")
selected = find_memory(
    f"{project}项目使用什么技术栈、发布前有哪些要求？", project, types=["entity"],
)
detail = api("POST", "/api/v1/memories/read", {
    "agentId": AGENT, "uri": selected["uri"], "level": "content",
})
print(selected["type"], selected["uri"], detail["content"])
assert all(word in detail["content"] for word in (project, "Python", "openGauss", "接口回归", "安全扫描"))
window = find_memory(
    f"{project}项目的发布窗口是什么？", "每周四晚上八点", types=["event", "session_archive"],
)
window_detail = api("POST", "/api/v1/memories/read", {
    "agentId": AGENT, "uri": window["uri"], "level": "content",
})
assert project in window_detail["content"]
print(window_detail["content"])
```

实体正文包含生成的项目名称、Python、openGauss、接口回归和安全扫描。发布窗口查询返回该项目每周四晚上八点发布的记录。写入统计中的 `candidates_extracted`、`writes_completed` 和 `writes_failed` 用于核对处理结果；已完成请求的相同重试可返回 `status=duplicate`。

## 记忆检索

### 能力介绍

应用使用自然语言寻找相关记忆，支持语义查询、记忆类型筛选、结果数量和相关性阈值控制。命中结果包含相关性分数、摘要和精确 URI，可继续读取概览或正文，帮助 Agent 按回答需要获取信息。

检索可结合向量语义与关键词匹配：语义查询适合改写表达，关键词匹配适合项目代号、接口名和精确术语。摘要、概览和正文的分层组织用于定位内容并控制读取量。

### 使用指南

先创建一条独立事实，再用不同表达查询：

```python
project = f"search-{RUN_ID}"
session = f"search-write-{RUN_ID}"
remember(
    f"请记住：{project}项目负责支付流水核对，每天输出交易差异报告。",
    session, f"search-seed-{RUN_ID}",
)
wait_idle(session)
selected = find_memory(
    f"{project}项目如何对账并报告差异？", project, types=["entity"],
)
print(selected["uri"], selected["score"], selected["abstract"])
for level in ("abstract", "overview", "content"):
    print(api("POST", "/api/v1/memories/read", {
        "agentId": AGENT, "uri": selected["uri"],
        "level": level, "max_chars": 12000,
    }))
```

`types` 选择要查询的记忆类型；`top_k` 为 `1–20`，`score_threshold` 设置最低相关性分数。读取的 `level` 可选 `abstract`、`overview`、`content` 或 `all`；`max_chars` 为 `1–50000`，同时受服务响应上限约束。

SQL 主存储采用具备原生 BM25 能力的 openGauss 时，可启用混合检索。主存储、向量索引和词法索引使用同一数据库，并在服务与索引 Worker 中应用以下配置：

```yaml
retrieval:
  hybrid_enabled: true
  vector_top_k: 100
  lexical_top_k: 50
  rrf_k: 60
  semantic_weight: 1.0
  lexical_weight: 1.0
  lexical:
    provider: opengauss_native
```

启用混合检索并重启服务后，对同一条自建项目事实执行两种查询：

```python
exact = find_memory(project, project, types=["entity"])
semantic = find_memory(
    f"{project}项目如何对账并报告差异？", project, types=["entity"],
)
print(exact["uri"], exact["score"])
print(semantic["uri"], semantic["score"])
assert exact["uri"] == semantic["uri"]
```

首次查询会触发关键词索引回填，`find_memory` 等待回填就绪后继续查询。

MCP 客户端调用 `ogm_query_memory` 和 `ogm_read_memory`；其身份和会话参数为 `agent_id`、`session_id`。查询返回的 URI 原样用于详情读取。

### 效果验证

“对账并报告差异”应命中写入的“支付流水核对、交易差异报告”。命中项包含生成的项目名称、URI、分数和摘要；三次读取分别返回摘要、概览和正文，正文对应同一项目事实。

启用混合检索后，分别使用精确项目代号和自然语言改写提问，核对相同项目的命中 URI、排名与正文。响应中的 `truncated` 用于检查返回长度，相关性分数用于排序与筛选。

## 记忆管理

### 能力介绍

应用可以浏览当前身份可见的记忆，按类型或分页盘点内容，读取完整详情，并根据用户要求遗忘指定记忆。目录、URI 和元数据帮助开发者定位具体条目，管理操作沿用账户、用户与 Agent 的访问范围。

用户要求撤销保留信息时，对确认过的精确 URI 执行遗忘。管理界面提供相同内容的目录浏览与详情查看。

### 使用指南

启用写入与删除，创建两个独立主题的偏好：

```yaml
mcp:
  enable_write_tools: true
  enable_delete_tools: true
```

```python
from urllib.parse import urlencode

subjects = [f"manage-review-{RUN_ID}", f"manage-docs-{RUN_ID}"]
facts = ["我偏好周二上午做代码评审", "我偏好周三下午整理开发文档"]
expected_uris = set()
for index, (subject, fact) in enumerate(zip(subjects, facts)):
    session = f"manage-write-{index}-{RUN_ID}"
    remember(f"请记住：在{subject}项目中，{fact}。",
             session, f"manage-request-{index}-{RUN_ID}")
    wait_idle(session)
    selected = find_memory(subject, subject, types=["preference"])
    expected_uris.add(selected["uri"])
```

偏好属于用户目录，分页请求使用 `type=preference`；`agentId` 用于选择 Agent 目录。将每页数量设为 1，使用返回的 `next_cursor` 继续读取：

```python
cursor = None
visible_uris = set()
while True:
    params = {"type": "preference", "limit": 1}
    if cursor:
        params["cursor"] = cursor
    page = api("GET", "/api/v1/memories?" + urlencode(params))
    visible_uris.update(item["uri"] for item in page["memories"])
    assert page["count"] == len(page["memories"])
    print(page["count"], page["truncated"], page["next_cursor"])
    if expected_uris <= visible_uris or not page["truncated"]:
        break
    cursor = page["next_cursor"]
assert expected_uris <= visible_uris
print("已找到两个示例偏好")
```

MCP 对应工具为 `ogm_list_memories`，使用 `limit` 和 `cursor` 翻页。确认要遗忘的条目后，读取正文并提交目标 URI：

```python
for uri in sorted(expected_uris):
    detail = api("POST", "/api/v1/memories/read", {
        "agentId": AGENT, "uri": uri, "level": "content",
    })
    print(uri, detail["content"])

target_uri = sorted(expected_uris)[0]
confirmed = api("POST", "/api/v1/memories/read", {
    "agentId": AGENT, "uri": target_uri, "level": "content",
})
print(confirmed)
forgotten = api("POST", "/api/v1/memories/delete", {
    "agentId": AGENT, "sessionId": f"manage-forget-{RUN_ID}",
    "uri": target_uri, "reason": "用户要求撤销该示例偏好",
})
print(forgotten)
```

### 效果验证

两个自建主题的 URI 应出现在分页结果中，单页 `count` 等于该页实际条目数。读取正文分别对应代码评审与文档整理偏好。

遗忘响应应包含 `ok=true`、`deleted=true`、`mode=archive` 和确认过的 URI；审计记录中的目标、原因和会话应与请求一致。

## 上下文管理

### 能力介绍

应用在模型请求前组装相关记忆和会话进展，在长对话中压缩历史，在会话结束时提交和释放运行状态。Agent 因而能够跨会话使用已保存事实，并在有限上下文窗口内继续完成任务。

上下文管理同时处理较大的工具结果：模型接收有界摘要与必要证据，完整原文可按需搜索或分段读取。

### 使用指南

开启预取并配置候选数量：

```yaml
memory:
  prefetch_enabled: true
  prefetch_top_k: 5
tool_result_externalization:
  enabled: true
  first_view_enabled: true
```

创建独立项目事实，在新会话中组装模型消息：

```python
project = f"context-{RUN_ID}"
write_session = f"context-write-{RUN_ID}"
remember(
    f"请记住：{project}项目每周四晚上八点发布，发布前完成接口回归和安全扫描。",
    write_session, f"context-seed-{RUN_ID}",
)
wait_idle(write_session)
session = f"context-read-{RUN_ID}"
question = f"{project}项目的发布安排和检查要求是什么？"
api("POST", "/api/v1/prefetch", {
    "agentId": AGENT, "sessionId": session, "query": question, "prefetchTopK": 5,
})
composed = api("POST", "/api/v1/compose", {
    "agentId": AGENT, "sessionId": session, "prompt": question,
    "messages": [{"role": "user", "content": question}], "tokenBudget": 16000,
})
model_messages = composed["messages"]
print(composed["estimatedTokens"], composed["budgetUsedBySlot"])
print(model_messages)
```

将 `model_messages` 交给业务模型。轮次完成后，使用 `/after_turn` 提交用户、助手和工具消息；长会话先准备压缩，再使用返回的令牌完成压缩：

```python
api("POST", "/api/v1/ingest_batch", {
    "agentId": AGENT, "sessionId": session,
    "messages": [
        {"role": "user", "content": f"{project}项目升级采用先灰度、后全量的顺序。灰度前运行接口回归，检查错误率与数据库查询延迟，完成升级和回滚脚本验证。"},
        {"role": "assistant", "content": "先完成灰度验证，再由负责人确认全量发布。"},
        {"role": "user", "content": "负责人确认后进入全量发布，全量后继续观察错误率和查询延迟，并保留回滚窗口。"},
    ],
})
prepared = api("POST", "/api/v1/prepare_compaction", {
    "agentId": AGENT, "sessionId": session,
    "summaryMaxChars": 2000, "shortTermIndexMode": "sync",
})
compacted = api("POST", "/api/v1/compact", {
    "agentId": AGENT, "sessionId": session, "prepareToken": prepared["prepareToken"],
    "tokenBudget": 8000, "summaryMaxChars": 2000, "shortTermIndexMode": "sync",
})
print(compacted)
```

大工具结果处理可配置 `tool_result_externalization.enabled=true` 和 `first_view_enabled=true`；适配器在工具返回后、模型请求前调用 `/api/v1/govern_first_view_tool_result`，向接口提供真实会话、回合及工具调用标识。下面自行生成日志作为输入：

```python
from urllib.parse import quote, urlencode

artifact_session = f"context-artifact-{RUN_ID}"
log = "INFO operation completed\n" * 1000 + f"request={RUN_ID} status=200\n"
governed = api("POST", "/api/v1/govern_first_view_tool_result", {
    "agentId": AGENT, "sessionId": artifact_session, "adapterId": "developer-host",
    "turnId": f"turn-{RUN_ID}", "toolCallId": f"call-{RUN_ID}",
    "toolName": "generate_log", "content": log,
})
base = f"/api/v1/sessions/{quote(artifact_session, safe='')}/tool-results"
artifacts = api("GET", base + "?" + urlencode({"agentId": AGENT}))
artifact_id = artifacts["artifacts"][0]["artifact_id"]
original = api("GET", f"{base}/{artifact_id}?" + urlencode({
    "agentId": AGENT, "offset_bytes": 0, "length_bytes": 32768,
}))
print(governed["content"])
print("原文字符数：", len(original["content"]))
print("原文一致：", original["content"] == log)
assert original["content"] == log
```

文件读取和命令输出也可生成精简内容，并保留可恢复的原文。启用工具输出处理：

```yaml
features:
  tool_output_filter:
    enabled: true
```

创建源码文件并执行差异比较，再分别提交实际文件内容和命令输出：

```python
from pathlib import Path
import subprocess
import tempfile

with tempfile.TemporaryDirectory(prefix="ogmem-guide-") as directory:
    source_path = Path(directory) / "sample.py"
    updated_path = Path(directory) / "updated.py"
    source = "def total():\n    value = 0\n" + "".join(
        f"    value += {index}\n" for index in range(600)
    ) + "    return value\n"
    source_path.write_text(source)
    updated_path.write_text(source.replace("value +=", "value -="))
    diff = subprocess.run(
        ["git", "diff", "--no-index", "--", str(source_path), str(updated_path)],
        capture_output=True, text=True,
    )
    requests = [
        ("filter_read", {"relative_path": source_path.name, "content": source_path.read_text()}, source),
        ("filter_bash", {"command": "git diff --no-index sample.py updated.py", "output": diff.stdout,
                         "exit_code": diff.returncode}, diff.stdout),
    ]
    for operation, payload, original_text in requests:
        payload.update(agentId=AGENT, sessionId=f"filter-{RUN_ID}")
        filtered = api("POST", f"/api/v1/{operation}", payload)
        restored = api("POST", "/api/v1/tool_output_original", {
            "agentId": AGENT, "memory_uri": filtered["memory_uri"], "max_chars": 50000,
        })
        assert filtered["ok"] and filtered["was_shortened"]
        assert restored["original"] == original_text
        print(operation, len(original_text), len(filtered["filtered_output"]), "原文一致")
```

会话结束时提交待处理内容并释放运行状态：

```python
disposed = api("POST", "/api/v1/dispose", {
    "agentId": AGENT, "sessionId": session, "wait": True,
})
print(disposed)
wait_idle(session)
```

### 效果验证

- 新会话的 `compose.messages` 包含独立项目的发布安排和检查要求，可作为业务模型的输入。
- `estimatedTokens` 与 `budgetUsedBySlot` 展示上下文占用；摘要、概览与正文按请求预算参与组装。
- 压缩返回 `ok=true`、`compacted=true`；`result.summary` 保留项目的灰度、全量、负责人确认和回滚要求，Token 估算值反映压缩效果。
- 日志治理返回 Artifact 引用，原文读取的 `content` 与自行生成的 `log` 一致；超过单次读取长度时，使用 `next_request` 分段获取。
- 文件内容和命令差异输出精简后字符数减少，使用返回的 `memory_uri` 读取原文，打印“原文一致”。会话释放返回 `disposed=true`，后续等待返回 `idle=true`。

## 技能学习与复用

### 能力介绍

将稳定的操作方法保存为 Agent Skill，供后续任务按需发现和调用。技能可由开发者导入，也可从成功任务的执行记录中学习，经质量评估和审核后发布。已有技能支持更新、附件管理和删除，适合积累发布检查、数据处理、代码审查等可重复执行的工作方法。

客户端先按任务描述搜索技能，再读取匹配技能的步骤和附件。导入技能保存在 `imported` 目录，学习生成的技能保存在 `generated` 目录；两类技能使用统一的目录、搜索和读取入口。

### 使用指南

技能管理使用账户 Admin Key。通过 HTTP 提交完整技能包；MCP 对应 `ogm_install_skill`、`ogm_list_skills`、`ogm_search_skills`、`ogm_read_skill`、`ogm_update_skill`。MCP 安装和更新启用 `read-write` 工具集或 `mcp.enable_write_tools=true`。

下面创建自含 Python 脚本的分批摘要技能，校验并安装，更新正文和附件后，从服务取回已保存脚本，用新的记录数量执行：

```python
import subprocess
import sys
from urllib.parse import quote, urlencode

skill_name = f"batch-digest-{RUN_ID}"
skill_path = "/api/v1/skills/" + quote(skill_name, safe="")
agent_query = urlencode({"agent_id": AGENT})
program = r"""
import hashlib, json, sys
records = json.loads(sys.argv[1])
max_batch_size = int(sys.argv[2])
batches = []
covered = []
for start in range(0, len(records), max_batch_size):
    batch = records[start:start + max_batch_size]
    batches.append({
        "count": len(batch),
        "sha256": hashlib.sha256("\n".join(batch).encode()).hexdigest(),
        "first_record": batch[0], "last_record": batch[-1],
    })
    covered.extend(batch)
assert covered == records
assert sum(item["count"] for item in batches) == len(records)
print(json.dumps({"batches": batches, "verified_total": len(covered)}))
"""
skill_text = (f"---\nname: {skill_name}\ndescription: 按顺序分批计算记录摘要并校验完整覆盖。\n---\n\n"
              "# 分批摘要\n\n1. 将JSON记录列表和每批上限作为参数传入 scripts/digest.py。\n"
              "2. 按输入顺序计算各批SHA256摘要，输出批次边界和记录数。")
package = {"files": [
    {"path": "SKILL.md", "content": skill_text},
    {"path": "scripts/digest.py", "content": program, "media_type": "text/x-python"},
]}
validation = api("POST", "/api/v1/skills/validate?" + agent_query, {"data": package})
installed = api("POST", "/api/v1/skills?" + agent_query, {"data": package})
matches = api("GET", "/api/v1/skills?" + urlencode({"agent_id": AGENT, "q": skill_name, "limit": 5}))
original = api("GET", skill_path + "?" + urlencode({"agent_id": AGENT, "source": "imported"}))
assert validation["valid"]
assert installed["action"] == "create" and installed["version"] == 1
assert any(item["name"] == skill_name for item in matches["skills"])
assert "SHA256" in original["content"]
package["files"][0]["content"] += "\n3. 核对各批记录总数，并确认覆盖顺序与输入一致。"
package["files"].append({
    "path": "assets/checklist.json", "content":
    '{"checks":["ordered_coverage","count_sum"]}', "media_type": "application/json",
})
updated = api("PUT", skill_path + "?" + agent_query, {"data": package})
latest = api("GET", skill_path + "?" + urlencode({"agent_id": AGENT, "source": "imported"}))
attachment = api("GET", skill_path + "/assets/checklist.json?" + agent_query)
saved_script = api("GET", skill_path + "/scripts/digest.py?" + agent_query)
assert updated["action"] == "update" and updated["version"] == 2
assert "覆盖顺序与输入一致" in latest["content"]
assert json.loads(attachment["content"])["checks"] == ["ordered_coverage", "count_sum"]

def run_batches(records, max_batch_size):
    completed = subprocess.run(
        [sys.executable, "-", json.dumps(records), str(max_batch_size)],
        input=saved_script["content"], check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)

new_records = [f"new-{index:03d}" for index in range(18)]
reused = run_batches(new_records, 8)
assert [item["count"] for item in reused["batches"]] == [8, 8, 2]
assert reused["verified_total"] == len(new_records)
print(latest["name"], updated["version"], attachment["content"])
print([item["count"] for item in reused["batches"]], reused["verified_total"])
```

技能包也可采用 ZIP 上传。更新时提交完整包，包内名称与目标技能一致。MCP 删除使用 `ogm_delete_skill`，HTTP 删除使用 `DELETE /api/v1/skills/<名称>?source=imported&agent_id=<Agent>`；MCP 删除入口启用 `mcp.enable_delete_tools=true` 或 `all` 工具集。

自动技能学习使用 PostgreSQL SQL 存储，并启用：

```yaml
skills:
  sedimentation:
    enabled: true
    worked_examples_enabled: true
    min_evidence_turns: 1
    min_distinct_sessions: 1
    turn_score_threshold: 0.85
    publish_score_threshold: 0.90
```

上报完整的成功执行过程，包括用户目标、实际执行的代码和参数、工具输出、完成回复及轮次结束事件。下面创建独立 Agent，实际分三批计算250条记录的摘要，再查看该执行过程的技能候选评估：

```python
import hashlib, time
from urllib.parse import urlencode, quote

learning_agent = f"skill-learner-{RUN_ID}"
api("POST", f"/api/v1/admin/accounts/{ACCOUNT}/agents", {
    "agent_id": learning_agent, "owner_user_id": USER,
})
session = f"skill-learning-{RUN_ID}"
tool_name = f"digest_batch_{RUN_ID}"
records = [f"record-{index:03d}" for index in range(250)]

def digest_batch(batch):
    if not 1 <= len(batch) <= 100:
        raise ValueError("每批记录数必须为1到100")
    digest = hashlib.sha256("\n".join(batch).encode()).hexdigest()
    return {"count": len(batch), "sha256": digest,
            "first_record": batch[0], "last_record": batch[-1]}

events = [{"event_type": "user_message", "prompt":
           f"任务{RUN_ID}：将250条样例记录分批计算SHA256摘要，每批最多100条。核对每批记录数和总覆盖范围，保存可复用方法。"}]
results = []
for index, start in enumerate(range(0, len(records), 100)):
    batch = records[start:start + 100]
    result = digest_batch(batch)
    results.append(result)
    call_id = f"digest-{RUN_ID}-{index}"
    events += [
        {"event_type": "tool_start", "tool_name": tool_name,
         "tool_use_id": call_id, "tool_input": {"batch": batch}},
        {"event_type": "tool_success", "tool_name": tool_name,
         "tool_use_id": call_id, "tool_response": result},
    ]
assert [item["count"] for item in results] == [100, 100, 50]
assert sum(item["count"] for item in results) == len(records)
answer = ("按输入顺序以100为步长切片，依次计算每批记录的SHA256摘要，"
          "各批条数为[100,100,50]，合计250。每批核对count、first_record和last_record，"
          "确认覆盖范围连续，再处理下一批；最终核对各批count之和等于原始记录总数。")
events += [
    {"event_type": "assistant_message", "assistant_message": answer},
    {"event_type": "turn_end", "status": "ok", "assistant_message": answer},
    {"event_type": "session_end", "status": "ok"},
]
for index, event in enumerate(events):
    received = api("POST", "/api/v1/trace/events", {
        "agent_id": learning_agent, "session_id": session,
        "request_id": f"{RUN_ID}-skill-learning-{index}", "event": event,
    })
    assert received["ok"], received

candidate = None
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    candidates = api("GET", "/api/v1/skill-candidates?" +
                     urlencode({"agent_id": learning_agent}))["candidates"]
    if candidates:
        candidate = candidates[0]
        break
    time.sleep(3)
assert candidate and candidate["evidence_count"] >= 1
assert candidate["summary"]
print(candidate["summary"], candidate["state"], candidate["evidence_count"])
print(candidate["reason"])
```

### 效果验证

导入校验返回 `valid=true`，安装返回 `action=create`、`version=1`，搜索命中自建技能名称。更新后 `version=2`，正文增加完整覆盖检查，附件的 `checks` 包含 `ordered_coverage` 和 `count_sum`。

执行的 Python 脚本来自保存后的技能附件。输入18条新记录、每批上限8条，返回批次条数 `[8, 8, 2]` 和 `verified_total=18`。

自动学习入口接收真实执行记录后，候选摘要描述分批 SHA256 计算与完整覆盖方法，`evidence_count` 至少为1。`state` 和 `reason` 展示当前质量评估结论；通过审核的技能进入 `generated` 目录，使用统一的技能搜索与读取接口。

## 工具与经验学习

### 能力介绍

从 Agent 完成任务的过程学习工具用法和处理经验，帮助后续任务复用已验证的参数约束、调用顺序和解决方法。工具知识记录适用条件与可核验事实，任务经验记录可迁移的处理策略；查询时可返回知识单元、修订号及来源证据。

例如，Agent 完成分页读取后，可学习游标的传递方式；后续面对相同工具或同类任务时，先检索已有经验，再据此组织调用，并通过来源定位核对约束。

### 使用指南

启用：

```yaml
agent_memory:
  enabled: true
```

使用账户 Admin Key 创建独立 Agent，执行两页读取，将实际过程提交给学习入口。等待知识可检索后，读取工具知识和来源，再按游标策略重新遍历样例记录：

```python
from urllib.parse import urlencode

tool_agent = f"tool-learner-{RUN_ID}"
api("POST", f"/api/v1/admin/accounts/{ACCOUNT}/agents", {
    "agent_id": tool_agent, "owner_user_id": USER,
})
session = f"tool-learning-{RUN_ID}"
tool_name = f"page_reader_{RUN_ID}"
items = [f"{RUN_ID}-item-{index}" for index in range(7)]

def read_page(cursor=0, limit=4):
    stop = min(cursor + limit, len(items))
    return {"items": items[cursor:stop],
            "next_cursor": stop if stop < len(items) else None}

first = read_page()
second = read_page(first["next_cursor"])
actual = first["items"] + second["items"]
summary = f"{tool_name}沿返回的next_cursor继续读取，合并得到7条有序记录。"
events = [{"event_type": "user_message", "prompt": f"读取{RUN_ID}的全部记录并核对分页结果。"}]
for index, (cursor, page) in enumerate([(0, first), (first["next_cursor"], second)]):
    call_id = f"page-{RUN_ID}-{index}"
    events += [
        {"event_type": "tool_start", "tool_name": tool_name,
         "tool_use_id": call_id, "tool_input": {"cursor": cursor, "limit": 4}},
        {"event_type": "tool_success", "tool_name": tool_name,
         "tool_use_id": call_id, "tool_response": page},
    ]
events += [
    {"event_type": "assistant_reasoning", "reasoning":
     f"{tool_name}返回的游标是下一页起点，沿游标继续读取并合并，可保持顺序并完整遍历数据。"},
    {"event_type": "assistant_message", "assistant_message": summary},
    {"event_type": "turn_end", "status": "ok", "assistant_message": summary},
]
for index, event in enumerate(events):
    api("POST", "/api/v1/trace/events", {
        "agent_id": tool_agent, "session_id": session,
        "request_id": f"{RUN_ID}-tool-learning-{index}", "event": event,
    })

wait_idle(session, agent=tool_agent)
hit = find_memory(f"{tool_name}如何使用next_cursor遍历全部记录？",
                  tool_name, ["tool_evidence"], timeout=180, agent=tool_agent)
args = {"agentId": tool_agent, "uri": hit["uri"], "level": "content", "max_chars": 10000}
if hit.get("unit_key"):
    args["unit_key"] = hit["unit_key"]
knowledge = api("POST", "/api/v1/memories/read", args)
assert actual == items
assert "cursor" in knowledge["content"].lower()
assert knowledge.get("grounding"), knowledge
print("TOOL_KNOWLEDGE", json.dumps(knowledge, ensure_ascii=False))
replayed, cursor = [], 0
while cursor is not None:
    page = read_page(cursor)
    replayed += page["items"]
    cursor = page["next_cursor"]
assert replayed == items
experience = find_memory("如何沿分页游标完整遍历记录并保持顺序？", "cursor", ["experience"], timeout=180, agent=tool_agent)
print("EXPERIENCE", json.dumps(experience, ensure_ascii=False))
print("PAGING_REUSE_OK", len(replayed))
```

MCP 使用 `ogm_query_memory` 和 `ogm_read_memory`。命中包含 `unit_key` 时，将该标识原样用于读取，获取对应知识单元及其来源。

### 效果验证

两页结果按顺序合并后与创建的七条 `items` 一致。工具知识描述 `cursor`、`limit`、`next_cursor` 的实际行为，`unit_key`、`unit_revision` 和 `grounding` 可定位到提交的工具参数与响应。

经验检索返回沿游标继续读取和按顺序合并的方法，并附带处理结论的来源。再次遍历得到相同的七条记录。

## 场景记忆

### 能力介绍

把同一活动、任务或主题的多条记忆关联为完整场景，便于回答需要综合多项事实的问题。例如，将发布评审、接口回归和回滚演练关联到一次发布准备，检索时同时获得活动进展及相关证据。

场景支持自动学习，也支持开发者创建、编辑、查询和归档。每个场景保存标题、描述、时间边界、成员记忆及来源，成员变化后可以继续维护对应活动的内容。

### 使用指南

使用 SQL ContextFS 并启用场景功能：

```yaml
scene:
  enabled: true
  options:
    analysis_trigger_tokens: 4096
    max_pending_nodes: 32
    max_llm_calls_per_batch: 3
    lexical_enabled: false
```

自行写入发布评审与回滚演练事实，逐次等待写入及索引完成。使用返回的规范 URI 创建场景，更新后重新读取成员和描述：

```python
from datetime import date
from urllib.parse import urlencode

session = f"scene-{RUN_ID}"
project = f"scenerelease{RUN_ID}"
today = date.today().isoformat()
review = remember(f"请保存已完成活动：{today}，{project}完成发布评审。开发团队执行了接口回归，42条接口用例全部通过；发布负责人确认接口兼容性检查通过，评审结论为允许继续准备发布。",
                  session + "-review", f"{RUN_ID}-scene-review")
rollback = remember(f"请保存已完成活动：{today}，{project}完成回滚演练。开发团队验证数据库回滚脚本，演练中数据恢复成功，服务在17秒内恢复正常；发布负责人确认回滚方案验证通过，等待执行发布。",
                    session + "-rollback", f"{RUN_ID}-scene-rollback")
find_memory(f"{project}发布准备", RUN_ID, ["event"])
selected_uris = set()
members = []
cursor = None
while True:
    params = {"type": "event", "limit": 100}
    if cursor:
        params["cursor"] = cursor
    page = api("GET", "/api/v1/memories?" + urlencode(params))
    for row in page["memories"]:
        detail = api("POST", "/api/v1/memories/read", {
            "agentId": AGENT, "uri": row["uri"], "level": "content", "max_chars": 10000,
        })
        if project in detail["content"]:
            members.append(row["context_uri"])
            selected_uris.add(row["uri"])
    cursor = page.get("next_cursor")
    if not cursor:
        break
assert members
content = "\n".join(api("POST", "/api/v1/memories/read", {
    "agentId": AGENT, "uri": uri, "level": "content", "max_chars": 10000,
})["content"] for uri in selected_uris)
assert "42" in content and "17" in content, content
owner = f"user:{USER}"
list_path = "/api/v1/scenes?" + urlencode({"owner_space": owner, "limit": 100})
snapshot = api("GET", list_path)
created = api("POST", "/api/v1/scenes", {
    "owner_space": owner, "revision": snapshot["revision"],
    "title": project + "发布准备", "description": "发布评审、接口回归与回滚演练。",
    "member_uris": members,
})
scene_id = created["scene"]["scene_id"]
updated = api("PUT", f"/api/v1/scenes/{scene_id}", {
    "owner_space": owner, "revision": created["revision"],
    "description": "接口回归和回滚演练完成，等待发布。",
})
saved = api("GET", list_path)
scene = next(item for item in saved["items"] if item["scene_id"] == scene_id)
assert updated["revision"] > created["revision"] > snapshot["revision"]
assert {item["context_uri"] for item in scene["cluster"]["members"]} == set(members)
assert scene["cluster"]["summary"] == "接口回归和回滚演练完成，等待发布。"
found = api("POST", "/api/v1/memories/query", {
    "agentId": AGENT, "sessionId": f"scene-read-{RUN_ID}",
    "query": f"{project}发布前已经完成哪些准备？", "types": ["event"], "top_k": 10,
})
assert selected_uris.issubset({hit["uri"] for hit in found["hits"]})
print("SCENE_OK", scene_id, saved["revision"], len(members))
print(content)
```

自动关联通过 `POST /api/v1/scenes/flush` 提交 `{"owner_space": owner}`，推动已写入证据的分析。等待接口携带 `waitScenes=true` 可查看场景处理结果。`GET/PUT /api/v1/scenes/configuration` 管理场景参数，配置更新使用返回的 `revision`；`DELETE /api/v1/scenes/<场景ID>` 携带当前 `revision` 和 `owner_space` 归档指定场景。

### 效果验证

事件正文包含42条接口用例通过和服务17秒恢复两项事实。创建、更新响应中的 `revision` 依次增加，重新读取的场景描述为“接口回归和回滚演练完成，等待发布”，成员与选定的规范 URI 一致。

查询发布准备问题时，结果包含两项活动事实的记忆。场景成员保存记忆版本和来源引用，可沿关联记忆逐项核对活动详情。

## 记忆关系网络

### 能力介绍

记忆关系网络将多次交互中的项目、系统和事件记忆连接起来，表达依赖、归属、支持和引用等有向关系。

写入新记忆时，服务结合已有相关记忆建立关联，保留关系类型、权重和来源依据。检索时，在已有召回结果的基础上补充直接关联或两跳路径上的事实，帮助 Agent 整合分散信息，理解数据流、系统依赖和业务背景。关联记忆遵循原有访问权限。

例如，账单导出服务依赖账单生成服务，账单生成服务依赖数据库。围绕导出服务提问时，关系网络可补充服务依赖、数据库名称和部署位置，为 Agent 判断数据的最终存储位置提供上下文。

## 代码理解与编辑

### 能力介绍

将代码仓库接入记忆服务后，Agent 可以按自然语言、符号或关键词定位实现，读取带文件和行号的代码证据，再对选定位置执行精确编辑。代码变化后更新资源和索引，使后续查询获取新的实现。

### 使用指南

在服务端创建独立代码样例：

```bash
export CODE_ROOT="$(mktemp -d)"
mkdir -p "$CODE_ROOT/src"
cat > "$CODE_ROOT/src/math_utils.py" <<'PY'
def calculate_total(price, quantity):
    return price * quantity
PY
```

将生成的 `CODE_ROOT` 填入服务的代码访问范围，重启服务。注册和编辑使用账户 Admin Key；HTTP 与 MCP 服务使用同一配置和存储：

```yaml
features:
  code_resources_enabled: true
  code_workspace_allowed_roots: ["<生成的 CODE_ROOT>"]
  code_sync_enabled: true
  code_retrieval_enabled: true
mcp:
  enable_code_edit_tools: true
```

注册样例、等待索引可见，然后搜索函数并读取片段：

```python
registration = api("POST", "/api/v1/resources/code/sync/registrations", {
    "path": os.environ["CODE_ROOT"], "name": f"code-{RUN_ID}",
    "trigger_mode": "manual", "watch_enabled": False,
    "wait_mode": "index_visible", "agentId": AGENT,
})
sync = api("POST", "/api/v1/resources/code/sync", {
    "root_id": registration["root_id"],
    "wait_mode": "index_visible", "agentId": AGENT,
})
found = api("POST", "/api/v1/resources/code/search", {
    "query": "calculate_total",
    "scope": {"root_slugs": [registration["resource_root_slug"]]},
    "budget": {"result_limit": 3, "snippet_chars": 1200, "batch_chars": 3600},
    "agentId": AGENT,
})
selected = next(item for item in found["evidence"]
                if "src/math_utils.py" in item["locator"])
snippet = api("POST", "/api/v1/resources/code/snippet", {
    "uri": selected["uri"], "max_chars": 1200, "agentId": AGENT,
})
assert sync["index_visible"], sync["diagnostics"]
assert "return price * quantity" in snippet["content"]
print(snippet["content"])
```

使用 MCP SDK 调用 `ogm_edit_file`，将检索返回的 `locator` 原样传入，修改唯一的返回表达式：

```python
import asyncio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def edit_code():
    async with httpx.AsyncClient(headers=HEADERS, timeout=600) as client:
        async with streamable_http_client(
            os.environ["OGMEM_MCP_URL"], http_client=client,
        ) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                result = await session.call_tool("ogm_edit_file", {
                    "locator": selected["locator"],
                    "old_string": "return price * quantity",
                    "new_string": "return round(price * quantity, 2)",
                    "replace_all": False,
                })
                assert result.structured_content["ok"], result
                print(result.structured_content)

asyncio.run(edit_code())
```

等待同步后，重新搜索并用新结果的 URI 读取代码：

```python
synced = api("POST", "/api/v1/resources/code/sync", {
    "root_id": registration["root_id"],
    "wait_mode": "index_visible", "agentId": AGENT,
})
assert synced["index_visible"], synced["diagnostics"]
after = api("POST", "/api/v1/resources/code/search", {
    "query": "calculate_total",
    "scope": {"root_slugs": [registration["resource_root_slug"]]},
    "agentId": AGENT,
})
selected_after = next(row for row in after["evidence"]
                      if "src/math_utils.py" in row["locator"])
new_snippet = api("POST", "/api/v1/resources/code/snippet", {
    "uri": selected_after["uri"], "max_chars": 1200, "agentId": AGENT,
})
assert "return round(price * quantity, 2)" in new_snippet["content"]
print(new_snippet["content"])
```

应用可用 `scope.paths`、`glob_patterns`、`grep_terms` 收敛检索范围。持续跟踪代码变化时，将工作区注册为 `trigger_mode=hybrid`、`watch_enabled=true`。`full` 检索模式使用持久化索引；`candidate_local` 模式对单个以 `manual` 注册且关闭监听的工作区执行请求内候选检索。

### 效果验证

- `sync.index_visible=true`，搜索结果的定位符指向生成的 `src/math_utils.py` 及行号。
- 片段包含 `calculate_total` 和原始乘法表达式，URI 与选中证据一致。
- 编辑后文件中的表达式为 `round(price * quantity, 2)`；资源和索引完成更新后，再次搜索与读取返回新表达式。
- 检索结果的 `diagnostics` 可核对召回路由、耗时和生效预算。

## 多模态记忆

### 能力介绍

将文档、图像、音频和视频转成可检索的结构化信息与记忆，保留原始材料、抽取结果和记忆之间的来源关系。应用可以查询文档内容、识别画面中的对象、检索语音和视频事件，并按片段或时间核对答案。

| 输入 | 可处理的信息 |
| --- | --- |
| 文档与文本 | 章节、内容结构和语义信息 |
| 图像 | 图像描述、目标、人脸、姿态和文字 |
| 音频 | 转写、说话人、语音片段及关联信息 |
| 视频 | 情景描述、事件、实体、人物、时间及跨片段问答 |

### 使用指南

应用选择相应算子或流水线，传入自己的原始材料。文档样例直接创建输入，并把抽取结果写成带来源链的记忆：

```python
from core.models import RequestContext
from core.graph_enums import RawDataType
from providers.graph_store import MemoryGraphStore
from service.graph_service import GraphService
from service.operator_service import OperatorService
from extraction.operators import OperatorRegistry
from extraction.operators.builtin.document_extractor import DocumentExtractor

ctx = RequestContext(account_id=ACCOUNT, user_id=USER, agent_id=AGENT,
                     session_id=f"document-{RUN_ID}", trace_id=f"document-{RUN_ID}")
graph = GraphService(MemoryGraphStore())
registry = OperatorRegistry()
registry.register(DocumentExtractor())
operators = OperatorService(graph_service=graph, registry=registry)
raw = graph.create_raw_node(
    raw_type=RawDataType.DOCUMENT,
    content="# 发布流程\n\n发布前执行接口回归、安全扫描和数据库升级验证。\n\n"
            "# 回滚流程\n\n回滚后验证服务健康状态和关键业务接口。", ctx=ctx,
)
extracted = operators.run_operator(raw.node_id, "document_structure", ctx)
memory = graph.create_memory_node(
    summary=f"发布流程-{RUN_ID}", content=json.dumps(extracted["extracted_content"], ensure_ascii=False),
    supporting_node_ids=[extracted["extracted_node_id"]], topic="release", ctx=ctx,
)
hits = graph.search_memories(query="发布流程", ctx=ctx)
chain = graph.get_supporting_chain(memory.node_id, ctx)
print(extracted, hits, chain)
```

持久化图使用 `SQLGraphStore`。HTTP 通过 `/api/v1/operators` 发现已配置算子，再调用单算子或 `/api/v1/operators/pipelines/run` 执行组合处理。长任务调用 `run_async`，使用返回的 `task_id` 查询进度。

视频流程使用 FFmpeg 和 Python 抽帧依赖 `opencv-python-headless`、`numpy`。在多模态处理配置中填写以下参数，模型密钥通过环境变量传入：

| 配置 | 用途 |
| --- | --- |
| `m3.episodic_llm` | 支持图像输入的视觉模型；填写 `provider`、`model`、`base_url`、`api_key` |
| `m3.qa_llm` | 问答模型；填写 `provider`、`model`、`base_url`、`api_key` |
| `m3.embedding` | 检索模型；填写 `provider`、`model`、`base_url`、`api_key`、`dimension`，维度与模型实际输出一致 |
| `m3.graph_store.connection_string` | 持久化原始节点、抽取节点和来源关系的数据库连接 |
| `vector_db.connection_string` | 向量索引数据库连接 |
| `m3.vector_index.backend` | openGauss 使用 `opengauss`，安装 pgvector 扩展的 PostgreSQL 使用 `pgvector` |

M3 的 Embedding 支持 `openai`、`st`、`volcengine` 适配器。火山引擎多模态 Embedding 选择 `volcengine`，执行 `python -m pip install "volcengine-python-sdk[ark]"`；`dimension` 设置为该模型支持的输出维度。

配置中的模型名称、地址、密钥和数据库连接对应以下环境变量；填写所选模型的实际值：

```bash
export OGMEM_VISION_MODEL='<支持图像输入的模型名称>'
export OGMEM_VISION_BASE_URL='<视觉模型的 OpenAI 兼容地址>'
export OGMEM_VISION_API_KEY='<视觉模型密钥>'
export OGMEM_QA_MODEL='<问答模型名称>'
export OGMEM_QA_BASE_URL='<问答模型的 OpenAI 兼容地址>'
export OGMEM_QA_API_KEY='<问答模型密钥>'
export OGMEM_M3_EMBEDDING_PROVIDER='<openai、st 或 volcengine>'
export OGMEM_M3_EMBEDDING_MODEL='<Embedding 模型名称>'
export OGMEM_M3_EMBEDDING_BASE_URL='<Embedding 服务地址>'
export OGMEM_M3_EMBEDDING_API_KEY='<Embedding 密钥>'
export OGMEM_M3_EMBEDDING_DIMENSION='<模型支持的输出维度>'
export OGMEM_GRAPH_CONNECTION_STRING='<图存储数据库连接>'
export OGMEM_M3_VECTOR_CONNECTION_STRING='<向量索引数据库连接>'
export OGMEM_M3_VECTOR_BACKEND='<opengauss 或 pgvector>'
```

创建多模态处理配置：

```bash
export M3_CONFIG="$(mktemp)"
cat > "$M3_CONFIG" <<'YAML'
m3:
  episodic_llm:
    provider: openai
    model: ${OGMEM_VISION_MODEL}
    base_url: ${OGMEM_VISION_BASE_URL}
    api_key: ${OGMEM_VISION_API_KEY}
  qa_llm:
    provider: openai
    model: ${OGMEM_QA_MODEL}
    base_url: ${OGMEM_QA_BASE_URL}
    api_key: ${OGMEM_QA_API_KEY}
  embedding:
    provider: ${OGMEM_M3_EMBEDDING_PROVIDER}
    model: ${OGMEM_M3_EMBEDDING_MODEL}
    base_url: ${OGMEM_M3_EMBEDDING_BASE_URL}
    api_key: ${OGMEM_M3_EMBEDDING_API_KEY}
    dimension: ${OGMEM_M3_EMBEDDING_DIMENSION}
  graph_store:
    connection_string: ${OGMEM_GRAPH_CONNECTION_STRING}
  vector_index:
    backend: ${OGMEM_M3_VECTOR_BACKEND}
vector_db:
  connection_string: ${OGMEM_M3_VECTOR_CONNECTION_STRING}
YAML
```

在源码环境中创建已知画面和问题：

```bash
cd "$REPO_DIR"
python -m pip install 'opencv-python-headless>=4.8,<5' 'numpy>=1.26'
WORK_DIR="$(mktemp -d)"
ffmpeg -f lavfi -i color=c=blue:s=320x240:d=5 \
  -vf 'drawbox=x=100:y=70:w=120:h=100:color=yellow:t=fill' \
  -c:v mpeg4 -pix_fmt yuv420p "$WORK_DIR/example.mp4"
python -m m3.run_video_pipeline \
  --video "$WORK_DIR/example.mp4" --video-id "video-$RUN_ID" \
  --seconds 5 --clip-seconds 5 --work-dir "$WORK_DIR/memory" \
  --pipeline "$REPO_DIR/config/m3/pipelines/video_episodic.yaml" \
  --config "$M3_CONFIG" --graph-store sql --account-id "$OG_ACCOUNT_ID" \
  --no-audio --disable-stage extract_audio --build-index
python - "$WORK_DIR/questions.json" "video-$RUN_ID" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({sys.argv[2]: {"qa_list": [
    {"question_id": "q1", "question": "画面中矩形是什么颜色？"}
]}}, ensure_ascii=False), encoding="utf-8")
PY
python -c 'from m3.qa import main_react_qa; raise SystemExit(main_react_qa())' \
  --annotations "$WORK_DIR/questions.json" --video "video-$RUN_ID" \
  --config "$M3_CONFIG" --graph-store sql --account-id "$OG_ACCOUNT_ID" \
  --output-jsonl "$WORK_DIR/qa.jsonl"
python - "$WORK_DIR/qa.jsonl" <<'PY'
import json, sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()[0])
assert "黄" in result["predicted_answer"] or "yellow" in result["predicted_answer"].lower()
print(result["question"], result["predicted_answer"])
print(result["search_rounds"], result["search_history"])
PY
```

音视频联合处理选择 `video_episodic_full.yaml`，安装 `cv`、`asr` 扩展并配置 CV、语音识别和说话人模型；QA 阶段传入生成的问题文件。问答可通过 `m3.qa_context` 配置上下文预算、检索次数和证据数量。参数见 [视频流水线](m3_resume.md)、[问答配置](m3_bounded_context.md)。

### 效果验证

- 文档抽取返回两个章节，分别包含发布和回滚流程；查询可找到创建的记忆。
- 来源链包含原始文档、抽取节点及其支撑的记忆。
- 视频处理产物写入新建工作目录，情景描述与答案对应蓝色背景、黄色矩形。
- 处理调用方提供的音视频时，将转写、事件与人物逐项对照原材料；答案可定位到对应时间片段。
- `qa.jsonl` 中的 `predicted_answer` 包含黄色矩形的结论；`search_rounds`、`search_history` 给出实际检索次数、问题和证据节点。

## 记忆共享与权限管理

### 能力介绍

应用可以为多个租户、用户和 Agent 分配独立的记忆空间，并把团队需要的知识共享给指定范围。用户画像和偏好归属用户，Agent 技能和执行知识归属 Agent。多个客户端使用相同身份时，可以持续使用同一份记忆。

Root 管理多个租户，Admin 管理所属租户的用户、Agent 和共享内容，Member 使用授权的记忆与技能。共享名单适合团队操作规范、公共工具知识和部门技能库，访问行为可通过审计记录核对。

### 使用指南

在服务配置中启用 `auth.role_control_enabled=true`，为实际账户配置 Admin Key。调用方通过 `X-API-Key`、`X-Account-ID` 和 `X-User-ID` 传递身份，通过请求中的 Agent 标识选择知识空间。

使用账户 Admin Key 创建两个独立用户，以及归属于其中一个用户的 Agent：

```python
shared_agent = f"shared-{RUN_ID}"
reader_ids = [f"reader-a-{RUN_ID}", f"reader-b-{RUN_ID}"]
readers = [api("POST", f"/api/v1/admin/accounts/{ACCOUNT}/users", {
    "user_id": user_id, "role": "MEMBER",
}) for user_id in reader_ids]
created_agent = api("POST", f"/api/v1/admin/accounts/{ACCOUNT}/agents", {
    "agent_id": shared_agent, "owner_user_id": reader_ids[0],
})
sharing_configuration = {
    "sharing": {
        "agent_shared_mode": "user",
        "agent_shared_list": [shared_agent],
    }
}
print(sharing_configuration)
```

将生成的 Agent 名称加入服务配置的共享名单，设置 `sharing.agent_shared_mode=user`，重新加载服务。应用将用户创建响应中的 `user_key` 保存在该用户的私有配置中，用于后续认证。

接着向共享 Agent 安装一个独立团队技能：

```python
skill_name = f"team-check-{RUN_ID}"
installed = api("POST", f"/api/v1/skills?agent_id={shared_agent}", {
    "data": {
        "name": skill_name,
        "description": "团队发布检查",
        "content": "# 发布检查\n\n1. 执行回归测试。\n2. 执行安全扫描。",
    }
})
```

分别使用两个新用户的密钥读取共享技能：

```python
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

shared_content = {}
for user_id, user in zip(reader_ids, readers):
    query = urlencode({"agent_id": shared_agent, "source": "imported"})
    request = Request(
        BASE + f"/api/v1/skills/{skill_name}?" + query,
        headers={
            "X-API-Key": user["user_key"], "X-Account-ID": ACCOUNT,
            "X-User-ID": user_id, "X-Agent-ID": shared_agent,
        },
    )
    with urlopen(request, timeout=30) as response:
        shared_content[user_id] = json.load(response)
print({user_id: result["content"] for user_id, result in shared_content.items()})
```

客户端接入相同规则：配置实际账户、用户和 Agent，向每个用户提供其访问密钥。团队知识通过共享 Agent 使用，个人记忆通过用户身份使用。

### 效果验证

- 用户列表包含新建的两个用户，角色均为 `MEMBER`；Agent 列表中的 `owner_user_id` 对应创建者。
- 共享配置接口 `GET /api/v1/admin/config/agent-sharing` 返回共享模式和生成的 Agent 名称。
- 两个用户读取到相同的 `skill_name`，正文均包含回归测试和安全扫描。
- `GET /api/v1/admin/accounts/{ACCOUNT}/audit-logs` 可核对新建用户、Agent 及相关管理操作，目标名称与生成的标识一致。

## 可视化管理

### 能力介绍

管理界面集中提供记忆浏览、语义检索、技能和场景查看，以及模型用量、组件状态和任务进度。开发者可以从回答中使用的事实定位到存储正文，检查当前数据归属，并结合处理记录分析记忆效果与运行开销。

### 使用指南

先创建独立记忆和技能，作为界面操作的输入：

```python
viewer_subject = f"界面项目-{RUN_ID}"
written = remember(
    f"请记住：{viewer_subject} 使用 uv 管理 Python 依赖，发布前执行接口回归。",
    f"viewer-write-{RUN_ID}", f"viewer-request-{RUN_ID}",
)
viewer_skill = f"viewer-check-{RUN_ID}"
installed = api("POST", f"/api/v1/skills?agent_id={AGENT}", {
    "data": {
        "name": viewer_skill, "description": "界面操作验证清单",
        "content": "# 检查清单\n\n1. 执行接口回归。\n2. 核对服务健康状态。",
    }
})
ready = wait_idle(f"viewer-write-{RUN_ID}")
print(written["status"], installed["name"], ready["idle"])
```

浏览器打开实际服务地址的 `/memory-viewer/`。在 **连接设置** 中填写 API Key，选择 `ACCOUNT`、`USER` 和 `AGENT` 对应空间。

| 操作目标 | 界面操作 |
| --- | --- |
| 查看记忆 | 打开 **记忆**，定位 `viewer_subject`，查看摘要、概览、正文和元数据 |
| 按语义查找 | 打开 **检索**，查询项目名称和“依赖管理、上线前检查”，选择命中项读取详情 |
| 查看技能 | 打开 **Agent Skill**，在 Agent 的 `skills/imported` 目录中选择 `viewer_skill` |
| 查看场景 | 打开 **场景记忆**，选择用户或 Agent，查看场景描述、时间与关联记忆 |
| 查看用量 | 打开 **控制台**，查看执行操作时段的模型调用和 Token 消耗 |
| 检查组件 | 打开 **监控**，查看服务、存储、模型、索引和任务队列状态 |
| 跟踪处理 | 打开 **任务中心**，按任务类型与状态筛选，核对关联资源、更新时间和处理详情 |

验证场景视图时，启用 `scene.enabled=true`，创建一条独立事件并关联场景：

```python
from datetime import date
from urllib.parse import urlencode

scene_subject = f"界面发布-{RUN_ID}"
scene_session = f"viewer-scene-{RUN_ID}"
remember(
    f"{date.today().isoformat()}，{scene_subject}完成接口回归并通过发布评审。",
    scene_session, f"viewer-scene-request-{RUN_ID}",
)
ready = wait_idle(scene_session)
members, cursor = [], None
while True:
    query = {"type": "event", "limit": 100}
    if cursor:
        query["cursor"] = cursor
    page = api("GET", "/api/v1/memories?" + urlencode(query))
    for row in page["memories"]:
        if RUN_ID not in (row.get("abstract", "") + row["id"]):
            continue
        detail = api("POST", "/api/v1/memories/read", {
            "uri": row["uri"], "level": "all", "agentId": AGENT,
        })
        content = detail["content"]
        if RUN_ID in content and "界面发布" in content:
            members.append(row["context_uri"])
    cursor = page.get("next_cursor")
    if not cursor:
        break
assert ready["idle"] and members
owner = f"user:{USER}"
snapshot = api("GET", "/api/v1/scenes?" + urlencode({"owner_space": owner}))
scene = api("POST", "/api/v1/scenes", {
    "owner_space": owner, "revision": snapshot["revision"],
    "title": scene_subject, "description": "接口回归与发布评审。",
    "member_uris": members,
})
print(scene["scene"]["scene_id"])
```

在场景视图中选择生成的标题，打开关联记忆。用量和工具统计也可通过 `GET /api/v1/token_stats`、`GET /api/v1/stats/tool-usage` 获取，便于应用统一展示。

### 效果验证

- 记忆正文包含 `viewer_subject`、`uv` 和接口回归要求；语义查询返回相同事实与对应 URI。
- 技能正文包含接口回归和服务健康检查，数据所在 Agent 与请求身份一致。
- 场景标题为 `scene_subject`，成员详情包含该事件的发生日期和评审事实。
- 模型用量与创建、提取和检索操作的调用记录对应；组件状态和任务详情可用于跟踪相关处理。
- 处理完成后的记忆可以从目录和检索结果读取，页面显示内容与接口返回一致。

## 接入方式

各接入方式使用相同的账户、用户和 Agent 身份。自研应用可选 HTTP 或 Python；协议客户端使用 MCP；已有 Agent 使用对应插件或中间件。

| 方式 | 接入入口 | 操作步骤 |
| --- | --- | --- |
| HTTP | `/api/v1/memories/*` 和会话接口 | [HTTP API](integrations/http-rest.md) |
| Python | `MemoryService`、`MemoryWriteAPI`、`ReadAPI` | [Python API](integrations/python-api.md) |
| MCP | stdio、Streamable HTTP、SSE | [MCP 接入](integrations/mcp.md) |
| OpenClaw | Context Engine 插件 | [OpenClaw](integrations/openclaw.md) |
| Claude Code | 原生 Hook 插件 | [Claude Code](integrations/claude-code.md) |
| Codex | 原生插件与 Hooks | [Codex](integrations/codex.md) |
| OpenCode | 原生生命周期插件 | [OpenCode](integrations/opencode.md) |
| DeepSeek Harness | DSH 插件 | [DeepSeek Harness](integrations/deepseek-harness.md) |
| Hermes | MemoryProvider 插件 | [Hermes](integrations/hermes.md) |
| QwenPaw | 原生插件和 Agent 中间件 | [QwenPaw](integrations/qwenpaw.md) |
| Deep Agents | `OGMemoryMiddleware` | [Deep Agents](integrations/deepagents.md) |
| 通用 Agent Plugin | MCP 代理与记忆 Skill | [通用插件](integrations/agent-plugins.md) |

使用配置生成器为客户端创建连接配置，输出目录由调用方指定：

```bash
cd "$REPO_DIR"
export CLIENT_CONFIG_DIR="$(mktemp -d)"
python -m cli.main onboard --mode plugin --agent mcp --non-interactive \
  --ogmem-url "$OGMEM_URL" --mcp-url "$OGMEM_MCP_URL" \
  --ogmem-api-key-env OG_AUTH_API_KEY \
  --account-id "$OG_ACCOUNT_ID" --user-id "$OG_USER_ID" --agent-id "$OG_AGENT_ID" \
  --output-dir "$CLIENT_CONFIG_DIR"
python - "$CLIENT_CONFIG_DIR" <<'PY'
import json, sys
from pathlib import Path
folder = Path(sys.argv[1])
config = json.loads((folder / "mcp.config.json").read_text(encoding="utf-8"))
print("服务项：", list(config["mcpServers"]))
print("安装步骤：", folder / "README.md")
PY
```

原生适配器可选择 `openclaw`、`claude-code`、`codex`、`opencode`、`deepseek-harness`、`hermes`、`deepagents`。MCP 客户端使用 `--agent mcp --mcp-url "$OGMEM_MCP_URL"`。安装插件、加载生成配置后，在新会话保存带唯一标识的事实，再切换会话查询并核对服务端正文。

远程 MCP 使用启用角色控制的服务，并按访问地址配置 `allowed_hosts`。MCP 客户端通过以下配置开放记忆读写，从工具列表确认所需能力：

```yaml
mcp:
  enabled: true
  default_toolset: read-write
```

`ogm_write_memory`、`ogm_query_memory`、`ogm_read_memory` 对应记忆写入、检索与读取；`ogm_list_memories` 提供清单；Skill 工具用于技能管理。资源 `ogmemory://types` 公布记忆类型，`ogmemory://status` 返回服务状态。删除和代码编辑分别通过 `mcp.enable_delete_tools`、`mcp.enable_code_edit_tools` 开启。

自研 Agent 在模型请求前调用 `compose`，使用返回的 `messages`；完成轮次后提交 `after_turn`；需要压缩时调用 `prepare_compaction` 和 `compact`。Python 进程内通过相应的 `MemoryService` 方法执行同一流程，结合已认证的 `RequestContext` 传递身份。Deep Agents 将中间件加入 `create_deep_agent(middleware=[...])`，以 `thread_id` 区分会话。


使用准备阶段的连接参数发现 MCP 工具和资源：

```python
import asyncio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def inspect_mcp():
    async with httpx.AsyncClient(headers=HEADERS, timeout=600) as client:
        async with streamable_http_client(
            os.environ["OGMEM_MCP_URL"], http_client=client,
        ) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools = await session.list_tools()
                resources = await session.list_resources()
                print([tool.name for tool in tools.tools])
                print([str(resource.uri) for resource in resources.resources])
                status = await session.read_resource("ogmemory://status")
                print(status.contents[0].text)

asyncio.run(inspect_mcp())
```

工具列表中可找到 `ogm_write_memory`、`ogm_query_memory`、`ogm_read_memory`；资源包含 `ogmemory://types`、`ogmemory://status`。状态资源返回 `status=ok` 和当前开放的工具集。
