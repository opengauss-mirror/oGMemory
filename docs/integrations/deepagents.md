# LangChain Deep Agents 对接

oGMemory 提供 `OGMemoryMiddleware`，直接接入 Python `deepagents` 的
`create_deep_agent(middleware=[...])`。支持同步 `invoke/stream` 和异步
`ainvoke/astream`，自动召回长期记忆、完成轮次写回、上传 Agent 执行轨迹，以及跨会话召回。
已验证 Deep Agents 0.7.23、LangChain 1.4.4、LangGraph 1.2.14。

## 安装与配置

先部署 [oGMemory HTTP 服务](http-rest.md)，确认 `/api/v1/health` 返回正常。
在 oGMemory 仓库根目录安装可选依赖：

```bash
python -m pip install '.[deepagents]'
```

普通安装不引入 Deep Agents。模型服务凭据由 Deep Agents 单独配置；默认模型
使用 `ANTHROPIC_API_KEY`，也可向 `create_deep_agent(model=...)` 传入已有模型。

轨迹上传默认随 `auto_capture=True` 开启，无需单独配置客户端。
如需从轨迹中抽取 Agent 记忆，在服务端 `ogmemory.yaml` 中开启：

```yaml
agent_memory:
  enabled: true
```

## 最小接入

```python
import os
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from integrations.deepagents import OGMemoryMiddleware

memory = OGMemoryMiddleware(
    endpoint="https://memory.example.com",  # 不含 /api/v1
    api_key=os.environ["OGMEM_API_KEY"],
    account_id="customer-a",
    user_id="alice",
    agent_id="customer-a-api-agent",  # 使用服务端授权的 Agent
)
agent = create_deep_agent(middleware=[memory], checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "conversation-001"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "以后 Python 项目统一使用 uv。"}]},
    config=config,
)
print(result["messages"][-1].content)
# 同一会话继续 invoke 时复用 config，只传入新的用户消息。
# 真正结束会话时调用；不要每轮 dispose。
memory.dispose("conversation-001")
```

异步调用使用相同 middleware：

```python
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "我偏好的 Python 包管理工具是什么？"}]},
    config={"configurable": {"thread_id": "conversation-002"}},
)
await memory.adispose("conversation-002")
```

同一用户跨会话保持 `account_id/user_id/agent_id` 相同，使用不同 `thread_id`。
缺失或空的 `thread_id` 会报错，防止会话串写。身份绑定在 middleware 实例上；
多用户应用应为不同身份建立独立 Agent/middleware，并使用独立的 checkpoint
作用域，不能把其他用户的对话历史传入当前实例。

可运行交互示例：`python -m examples.deepagents_memory`。

### 工具调用与轨迹上传

沿用上面的 `memory` 配置，为 Agent 添加工具并发起一次调用：

```python
from pathlib import Path
from langchain_core.tools import tool

@tool
def list_project_files(directory: str = ".") -> str:
    """列出目录中的文件名。"""
    return "\n".join(sorted(path.name for path in Path(directory).iterdir()))

agent = create_deep_agent(
    tools=[list_project_files], middleware=[memory], checkpointer=InMemorySaver()
)
result = agent.invoke(
    {"messages": [{"role": "user", "content":
        "请调用 list_project_files 列出当前目录文件，再概括项目结构。"}]},
    config={"configurable": {"thread_id": "tool-conversation-001"}},
)
print(result["messages"][-1].content)
memory.dispose("tool-conversation-001")
```

预期 Agent 实际调用工具，并根据工具结果回答。完整轮次结束时，中间件自动向
`/api/v1/after_turn` 提交普通 `messages` 和独立的 `traceMessages`，保留本轮输入、
助手输出、工具调用 ID、工具名称、参数、结果及明确的错误状态；无文本的助手
工具调用消息也会保留。无需手动构造或上传轨迹。

响应中的 `trace_captured=true` 且 `trace_status=CAPTURED` 表示轨迹已保存；
`false/PENDING` 或缺少确认按 `fail_open` 告警或抛错。HTTP 成功和轨迹保存均不代表
Agent 记忆抽取完成；还需等待后台处理。没有可复用知识的轨迹允许不产出记忆。

### E2E 接入测试

使用真实模型及 oGMemory 服务运行：

```bash
python -m pip install '.[deepagents]' 'langchain-openai>=1.1,<2'
# 先加载专用测试用户的 OGMEM_URL/API_KEY/ACCOUNT_ID/USER_ID/AGENT_ID
# 以及模型的 LLM_BASE_URL/MODEL/API_KEY 环境变量。
python tests/e2e/run_deepagents_smoke.py --wait-seconds 300
```

脚本通过真实 `create_deep_agent` 自动写回唯一项目的发布窗口，等待服务端抽取和
索引完成后，在独立同步、异步会话中检查召回答案及实际模型上下文，并确认
召回记忆没有写入消息历史。失败退出码非零；成功输出 `PASS`。
JSON 报告默认保存在已被 Git 忽略的 `.tmp/deepagents-e2e/`，可用 `--report` 指定路径。
测试会为配置的用户留下持久记忆，请使用专用测试身份。

### LoCoMo 评测

`tests/e2e/run_deepagents_locomo.py` 复用 `eval.py` 的数据选择、写入、问答、
判分和统计。通过 `--host-factory 模块:函数` 接入已挂载 `OGMemoryMiddleware`
的应用。工厂模块需能在评测 Python 环境导入。

```bash
python tests/e2e/run_deepagents_locomo.py ingest "$DATA" \
  --host-factory my_app.eval_host:create_host --run-id "$RUN_ID" --sample 0 --verify-writes
```

工厂接收评测参数，其中 `ogmemory_url` 是本轮代理地址，身份字段和
`runtime_root` 对应本轮测试。返回提供 `prepare()`、`invoke(prompt, key)` 和
`close()` 的宿主对象。宿主将每个逻辑会话键映射为实际会话，等待本轮钩子执行后
返回 `(回答文本, token 用量)`。串行宿主使用 `--parallel 1`。

运行 `qa` 时沿用数据和运行身份，代理默认拦截问答写回。
`judge` 和 `stat` 读取评测产物，无需宿主工厂。严格写入核验会匹配原文指纹；
每段写入后通过 `wait_until_idle` 检查服务端状态。若部署版本采用不同的归档判据，
宿主可以实现 `settle(session_id)`，等待记忆可召回并返回含 `session_id` 的证据字典，
未就绪时抛出异常。更换宿主工厂时使用新的评测身份和写入记录。
四阶段参数见[记忆评测教程](../testing/memory-evaluation-guide-zh.md)。

## 环境变量与配置导出

`OGMemoryMiddleware.from_env()` 读取：

| 环境变量 | 含义 |
| --- | --- |
| `OGMEM_URL` | HTTP API 基址，默认 `http://127.0.0.1:8090` |
| `OGMEM_API_KEY` | 租户服务密钥，本地无鉴权时可留空 |
| `OGMEM_ACCOUNT_ID`、`OGMEM_USER_ID`、`OGMEM_AGENT_ID` | 必须设置的服务身份 |
| `OGMEM_AUTO_RECALL`、`OGMEM_AUTO_CAPTURE` | 默认 `true`；`1/true/yes` 开启，其余显式值关闭 |

也可用已有 CLI 生成连接脚本和 Python 接入样例：

```bash
ogmem onboard --mode plugin --agent deepagents --non-interactive \
  --ogmem-url https://memory.example.com --ogmem-api-key-env OG_AUTH_API_KEY \
  --account-id customer-a --user-id alice --agent-id customer-a-api-agent
```

`deepagent` 是别名。生成后按照导出 README 在 Agent 机器加载环境脚本。

## 运行行为与验证

- 每次用户输入开始时调用 `/compose`，仅取 `_ogmem: true` 的记忆内容，
  标记为不可信历史数据后临时注入模型请求。记忆不写入对话消息列表。
- 完成轮次后调用 `/after_turn`，通过 `messages` 写回本轮用户输入及新生成的助手/工具文本，
  同时通过 `traceMessages` 上传本轮执行轨迹；召回上下文不进入轨迹。
  不重复上传 checkpoint 中的旧历史；不上传图片、音频、系统消息和推理块。
  写回保留消息 ID，使连续相同文本的不同消息仍能被服务端区分。
  `return_direct=True` 工具直接返回并正常结束的轮次也会写回；未完成的工具调用不会写回。
  checkpoint 恢复后沿用本轮轨迹与请求标识，客户端不自动重试。
  参数解析失败是已结束的失败调用；有框架错误结果时保留原始参数与关联失败轨迹。
  已完成轮次的轨迹无法完整关联时，仅写回普通消息并告警，不补造调用或结果。
- `dispose/adispose` 在会话结束时调用 `/dispose`，服务端异步刷新剩余内容。
- HTTP 超时、错误状态或无效响应默认记录不含内容/凭据的告警并继续主对话。
  可设置 `fail_open=False` 让错误抛出。轨迹准备异常发生于已完成轮次时，先写回普通消息，
  再按 `fail_open` 告警或抛错；输入/身份校验及模型、工具自身异常仍按原流程处理。
  失败写入不会在客户端持久化或自动重试。
- `timeout` 默认 30 秒，`token_budget` 默认 6000，`force_extract` 默认 `False`。
  `auto_recall` 和 `auto_capture` 可分别关闭。根 Agent 的 middleware 不自动装到
  独立子 Agent；需要时为子 Agent 显式配置，且使用独立会话 ID。

验证跨会话记忆时，先发送具体偏好，结束会话，等待服务端抽取及索引完成，
再用新的 `thread_id` 提问。`after_turn` 返回 `accumulating/processing` 或
`dispose` 成功并不代表记忆已可检索。演示时可设置 `force_extract=True`，
仍须等待后台处理完成。

中间件不替换 Deep Agents 的文件系统、checkpointer 或默认摘要机制，也不接管
oGMemory 两阶段压缩。已观察到的本轮工具调用与结果保存在 checkpoint 私有状态中，
不会被后续摘要或工具结果卸载覆盖；普通 `messages` 仍使用结束时可见的本轮文本。
自定义 middleware 若在本中间件观察前删除消息，无法通过轨迹上传还原这些内容。
