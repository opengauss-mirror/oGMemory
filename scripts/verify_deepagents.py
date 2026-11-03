"""Verify Deep Agents against a real model and oGMemory HTTP service.

Uses OGMEM_* identity/connection variables and LLM_BASE_URL/MODEL/API_KEY.
Creates durable test memory for the configured user. Prefer a dedicated user.
Run after installing .[deepagents] and langchain-openai.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from integrations.deepagents import OGMemoryMiddleware


class ObservedMemory(OGMemoryMiddleware):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.calls: list[dict[str, Any]] = []

    def _record(self, path, payload, result):
        self.calls.append(
            {
                "path": path,
                "session_id": payload["sessionId"],
                "message_count": len(payload.get("messages", [])),
                "status": result.get("status"),
                "ok": result.get("ok"),
            }
        )
        print("HTTP", json.dumps(self.calls[-1], ensure_ascii=False), flush=True)

    def _post(self, path, payload):
        result = super()._post(path, payload)
        self._record(path, payload, result)
        return result

    async def _apost(self, path, payload):
        result = await super()._apost(path, payload)
        self._record(path, payload, result)
        return result


class ObserveModel(AgentMiddleware):
    def __init__(self):
        self.memory_contexts: list[str] = []

    def _record(self, request):
        contexts = [
            m.content
            for m in request.messages
            if isinstance(m.content, str) and "<ogmemory-context>" in m.content
        ]
        self.memory_contexts.extend(contexts)

    def wrap_model_call(self, request, handler):
        self._record(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._record(request)
        return await handler(request)


def main(*, report_path: Path | None = None, wait_seconds: int | None = None) -> dict[str, Any]:
    report_path = report_path or Path(os.getenv("OGMEM_VERIFY_REPORT", "deepagents-verification.json"))
    if wait_seconds is None:
        wait_seconds = int(os.getenv("OGMEM_VERIFY_WAIT_SECONDS", "300"))
    if wait_seconds <= 0:
        raise ValueError("wait_seconds must be positive")
    run_id = uuid4().hex[:12]
    project = "DA验证" + run_id
    hour = 9 + int(run_id[:2], 16) % 12
    minute = int(run_id[2:4], 16) % 60
    expected = f"每周四 {hour:02d}:{minute:02d}"
    query = f"项目{project}的发布窗口是什么？只回复日期和时间。"
    memory = ObservedMemory.from_env(force_extract=True, fail_open=False, timeout=120)
    observer = ObserveModel()
    extra_body = json.loads(os.getenv("LLM_EXTRA_BODY", "{}"))
    base_url = os.environ["LLM_BASE_URL"].rstrip("/")
    model = ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        base_url=base_url,
        api_key=os.environ["LLM_API_KEY"],
        temperature=0,
        timeout=120,
        max_retries=1,
        extra_body=extra_body,
    )
    agent = create_deep_agent(
        model=model, middleware=[memory, observer], checkpointer=InMemorySaver()
    )
    write_session = "da-write-" + run_id
    read_session = "da-read-" + run_id
    async_session = "da-async-" + run_id
    report: dict[str, Any] = {
        "run_id": run_id,
        "endpoint": memory.endpoint,
        "identity": memory.identity,
        "model": os.environ["LLM_MODEL"],
        "project": project,
        "expected": expected,
        "versions": {
            name: version(name)
            for name in ["deepagents", "langchain", "langgraph", "langchain-openai"]
        },
    }
    print("VERIFY", json.dumps(report, ensure_ascii=False), flush=True)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"请记住：项目{project}的发布窗口是{expected}。只回复‘已记住’。",
                }
            ]
        },
        config={"configurable": {"thread_id": write_session}},
    )
    report["write_answer"] = result["messages"][-1].content
    writes = [c for c in memory.calls if c["path"] == "after_turn"]
    assert len(writes) == 1, "Expected exactly one automatic turn capture"
    assert writes[0]["ok"] is True, "Service did not acknowledge the automatic write"
    assert writes[0]["message_count"] == 2, "Write must contain the current user and assistant messages"
    memory.dispose(write_session)
    print("WRITE_ANSWER", report["write_answer"], flush=True)

    # Poll the real service for extraction/index readiness, rather than assuming
    # that an accepted after_turn response means the memory is searchable.
    evidence = ""
    deadline = time.monotonic() + wait_seconds
    with httpx.Client(timeout=120) as client:
        while time.monotonic() < deadline:
            response = client.post(
                memory.endpoint + "/api/v1/compose",
                headers=memory.headers,
                json={
                    **memory.identity,
                    "sessionId": read_session,
                    "prompt": query,
                    "messages": [{"role": "user", "content": query}],
                    "tokenBudget": 6000,
                },
            )
            response.raise_for_status()
            evidence = memory._context(response.json())
            if run_id in evidence and f"{hour:02d}:{minute:02d}" in evidence:
                print("MEMORY_INDEX_READY", flush=True)
                break
            print("WAITING_FOR_EXTRACTION_AND_INDEX", flush=True)
            time.sleep(10)
        else:
            raise RuntimeError("Test memory was not searchable before timeout")

    memory.auto_capture = False
    observer.memory_contexts.clear()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": read_session}},
    )
    report["sync_answer"] = result["messages"][-1].content
    assert f"{hour:02d}:{minute:02d}" in str(report["sync_answer"]), (
        "Sync answer did not recall the stored release time"
    )
    assert any(run_id in context for context in observer.memory_contexts), (
        "Sync model did not receive the test memory"
    )
    assert not any("<ogmemory-context>" in str(m.content) for m in result["messages"])
    memory.dispose(read_session)
    print("SYNC_RECALL_ANSWER", report["sync_answer"], flush=True)

    async def verify_async():
        observer.memory_contexts.clear()
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": async_session}},
        )
        assert any(run_id in context for context in observer.memory_contexts), (
            "Async model did not receive the test memory"
        )
        assert not any("<ogmemory-context>" in str(m.content) for m in result["messages"]), (
            "Async recall context was persisted in conversation history"
        )
        await memory.adispose(async_session)
        return result["messages"][-1].content

    report["async_answer"] = asyncio.run(verify_async())
    assert f"{hour:02d}:{minute:02d}" in str(report["async_answer"]), (
        "Async answer did not recall the stored release time"
    )
    print("ASYNC_RECALL_ANSWER", report["async_answer"], flush=True)
    report.update({"status": "passed", "evidence": evidence[:3000], "http_calls": memory.calls})
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("PASS: real model + automatic capture + cross-session sync/async recall", flush=True)
    print("REPORT:", report_path.resolve(), flush=True)
    return report


if __name__ == "__main__":
    main()
