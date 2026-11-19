from __future__ import annotations

import json
from datetime import timedelta

import pytest

from memorybot.models import ChatInput
from memorybot.runtime import BotWorker
from memorybot.store import LeaseLost
from tests.integration.test_memorybot_runtime import execute, source, task
from tests.integration.test_memorybot_runtime import runtime as runtime

pytestmark = pytest.mark.integration


def chapter_model(runtime):
    calls = []

    def complete(messages, emit):
        value = json.loads(messages[-1]["content"])
        calls.append(value)
        materials = value["materials"]
        if isinstance(materials[0], dict):
            item = materials[0]
            text = f"### 关键事实\n\n{item['text']}\n\n### 待办\n\n按原安排执行。"
        else:
            text = "\n\n".join(materials)
        return [], text

    runtime.llm.stream_hook = complete
    return calls


def test_ten_documents_generate_persistent_chapters_and_complete_handbook(runtime):
    files = [
        source(runtime, f"资料{number}.md", f"负责人林{number}，11月{number}日完成验收。")
        for number in range(1, 11)
    ]
    calls = chapter_model(runtime)
    item = task(
        runtime,
        "chat",
        ChatInput(
            text="整理成十章交接手册",
            template_id="handbook",
            file_ids=[file["file_id"] for file in files],
        ).model_dump(mode="json"),
    )
    result = execute(runtime, item)
    for number in range(1, 11):
        assert f"## {number}. 资料{number}.md" in result["result"]["text"]
        assert f"负责人林{number}" in result["result"]["text"]
    assert len(calls) == 11
    assert len(result["result"]["citations"]) == 10
    budget = runtime.store.get(
        runtime.ctx.account_id,
        "bot_step",
        {
            "task_id": item["task_id"],
            "step_no": 1,
        },
    )["output"]
    assert budget["model_calls"] == 11 and len(budget["model_calls_by_scope"]) == 11
    steps = runtime.store.task_steps(runtime.ctx.account_id, item["task_id"], runtime.ctx.user_id)
    chapters = [step for step in steps if step["label"] == "生成手册章节"]
    assert [step["summary"]["completed"] for step in chapters] == list(range(1, 11))
    merged = next(step for step in steps if step["label"] == "合并手册")
    assert merged["summary"]["completed"] == merged["summary"]["total"] == 10
    artifact = runtime.store.get(
        runtime.ctx.account_id,
        "bot_file",
        {
            "file_id": result["result"]["file_ids"][0],
        },
    )
    assert (
        runtime.files.read(runtime.ctx.account_id, artifact["storage_key"]).decode()
        == result["result"]["text"]
    )
    assert len(artifact["metadata"]["memory_sources"]) == 10


def test_long_document_segments_rejoin_into_one_complete_chapter(runtime):
    file = source(
        runtime,
        "长资料.md",
        "早期安排：林悦负责。\n" + "事实记录。" * 250 + "\n最终日期：11月12日。",
    )
    runtime.worker.config.max_input_tokens = 2300
    calls = chapter_model(runtime)
    result = execute(
        runtime,
        task(
            runtime,
            "chat",
            ChatInput(
                template_id="handbook",
                file_ids=[file["file_id"]],
            ).model_dump(mode="json"),
        ),
    )
    assert len([call for call in calls if isinstance(call["materials"][0], dict)]) >= 2
    assert "早期安排：林悦负责。" in result["result"]["text"]
    assert "最终日期：11月12日。" in result["result"]["text"]
    assert "## 1. 长资料.md" in result["result"]["text"]


def test_handbook_reuses_completed_chapters_after_worker_handoff(runtime, monkeypatch):
    files = [source(runtime, f"交接{index}.md", f"第{index}份资料完成。") for index in range(1, 4)]
    calls = chapter_model(runtime)
    item = task(
        runtime,
        "chat",
        ChatInput(
            template_id="handbook",
            file_ids=[file["file_id"] for file in files],
        ).model_dump(mode="json"),
    )
    append_event = runtime.store.append_event
    transferred = []

    def transfer(tx, task_id, kind, payload):
        result = append_event(tx, task_id, kind, payload)
        if kind == "progress" and payload.get("phase") == "handbook_chapter" and not transferred:
            transferred.append(task_id)
            raise LeaseLost("handoff at completed chapter boundary")
        return result

    monkeypatch.setattr(runtime.store, "append_event", transfer)
    runtime.worker.run_once()
    first = runtime.store.get_task(runtime.ctx.account_id, item["task_id"])
    monkeypatch.setattr(runtime.store, "append_event", append_event)
    with runtime.store.transaction(runtime.ctx.account_id) as tx:
        tx.update(
            "bot_task",
            {"task_id": item["task_id"]},
            {"lease_until": tx.now() - timedelta(seconds=1)},
        )
    peer = BotWorker(runtime.store, runtime.gateway, runtime.files, config=runtime.worker.config)
    peer.run_once()
    result = runtime.store.get_task(runtime.ctx.account_id, item["task_id"])
    assert result["state"] == "succeeded" and result["attempt"] == first["attempt"] + 1
    assert len(calls) == 4
    steps = runtime.store.task_steps(runtime.ctx.account_id, item["task_id"], runtime.ctx.user_id)
    chapters = [step for step in steps if step["label"] == "生成手册章节"]
    assert chapters[0]["attempt"] == 1 and chapters[-1]["attempt"] == 2


def test_handbook_processes_full_source_and_preserves_every_resource_reference(runtime):
    content = "资料事实。" * 16000 + "\n最终批准人：林悦。"
    file = source(runtime, "完整资料.md", content)
    fragments = []

    def summarize(messages, emit):
        data = json.loads(messages[-1]["content"])
        if isinstance(data["materials"][0], dict):
            fragment = data["materials"][0]["text"]
            fragments.append(fragment)
            text = "资料事实已核对。" + ("最终批准人：林悦。" if "最终批准人" in fragment else "")
        else:
            text = "\n".join(data["materials"])
        return [], text

    runtime.llm.stream_hook = summarize
    result = execute(
        runtime,
        task(
            runtime,
            "chat",
            ChatInput(
                template_id="handbook",
                file_ids=[file["file_id"]],
            ).model_dump(mode="json"),
        ),
    )
    assert "".join(fragments) == content
    assert len(file["metadata"]["memory_sources"]) > 5
    assert "最终批准人：林悦。" in result["result"]["text"]
    for ref in file["metadata"]["memory_sources"]:
        assert ref["uri"] in result["result"]["text"]
