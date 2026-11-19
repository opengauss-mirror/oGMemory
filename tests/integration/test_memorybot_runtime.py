from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from types import SimpleNamespace

import pytest
import yaml

from core.models import ContextNode, RequestContext, Role
from memorybot.config import BotConfig
from memorybot.files import AGFSFileStore, parse_text
from memorybot.gateway import MemoryGateway
from memorybot.models import ChatInput, new_id
from memorybot.runtime import BotWorker
from memorybot.service import BotService
from memorybot.store import BotStore, LeaseLost
from memorybot.templates import TemplateRegistry
from providers.llm.mock_llm import MockLLM
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService

pytestmark = pytest.mark.integration


class DeterministicLLM(MockLLM):
    def __init__(self):
        super().__init__()
        self.extract = False
        self.messages = []
        self.planned_calls = []
        self.stream_hook = None

    def complete_with_tools_messages(self, messages, tools=None, **kwargs):
        self.messages.append(messages)
        if self.planned_calls:
            return [self.planned_calls.pop(0)], ""
        if self.stream_hook:
            return self.stream_hook(messages, kwargs.get("on_text_delta"))
        text = "# 青禾周报\n\n青禾项目接口已完成。下一步安排联调。"
        cited = re.findall(
            r'"citation_id": "(S[0-9]+)"',
            "\n".join(str(message.get("content", "")) for message in messages),
        )
        if cited:
            text += " [" + cited[0] + "]"
        callback = kwargs.get("on_text_delta")
        if callback:
            for index in range(0, len(text), 8):
                callback(text[index : index + 8])
        return [], text

    def complete_with_tools(self, prompt, tools, **kwargs):
        if self.extract and any(tool.get("name") == "extract_entity" for tool in tools):
            return [
                {
                    "tool": "extract_entity",
                    "input": {
                        "owner_scope": "user",
                        "routing_key": "qinghe",
                        "object_name": "青禾项目",
                        "abstract": "青禾项目由我负责。",
                        "overview": "- 青禾项目由我负责。",
                        "content": "青禾项目由我负责。",
                        "confidence": 0.99,
                    },
                }
            ]
        return []


@pytest.fixture
def runtime():
    dsn = os.environ.get("MEMORYBOT_TEST_DSN")
    agfs_url = os.environ.get("MEMORYBOT_TEST_AGFS_URL")
    if not dsn or not agfs_url:
        pytest.skip(
            "MEMORYBOT_TEST_DSN and MEMORYBOT_TEST_AGFS_URL select isolated integration services"
        )
    dialect = os.environ.get("MEMORYBOT_TEST_DIALECT", "postgresql")
    store = BotStore(dsn, dialect=dialect)
    account = new_id("runtime")
    cfg = OgMemConfig(
        storage_backend="sql",
        sql_connection_string=dsn,
        sql_dialect=dialect,
        sql_pool_size=10,
        vector_db_type="opengauss",
        vector_db_dialect=dialect,
        opengauss_connection_string=dsn,
        opengauss_dimension=384,
        opengauss_table_name="memorybot_test_vector",
        provider="mock",
        embedding_provider="mock",
        index_workers=0,
        enable_cache=False,
        dual_pass_enabled=False,
        account_id=account,
        user_id="user",
        agent_id="memorybot",
    )
    memory = MemoryService(cfg)
    llm = DeterministicLLM()
    memory._llm = llm
    ctx = RequestContext(account, "user", "memorybot", "s_qinghe", "test", role=Role.ADMIN)
    api_key = memory.get_key_manager().create_account(account, ctx.user_id)
    memory.get_tenant_admin_service().create_agent(
        ctx, account, ctx.agent_id, owner_user_id=ctx.user_id
    )
    memory.get_read_api(start_outbox_worker=False)
    gateway = MemoryGateway(
        memory,
        lambda task: replace(
            ctx,
            session_id=task["accepted_authz"].get("memory_session_id", task["task_id"]),
        ),
    )
    files = AGFSFileStore(agfs_url, "/memorybot")
    config = BotConfig(
        files_base_url=agfs_url,
        file_timeout_seconds=900,
        task_timeout_seconds=120,
    )
    store.claim = partial(store.claim, account_id=account)
    worker = BotWorker(store, gateway, files, config=config)
    with store.transaction(account, ctx.user_id) as tx:
        tx.insert(
            "bot_session",
            {
                "session_id": ctx.session_id,
                "title": "青禾",
                "agent_id": ctx.agent_id,
                "scope": {
                    "memory_scope": "user",
                    "resource_ids": [],
                },
            },
        )
    value = SimpleNamespace(
        store=store,
        memory=memory,
        gateway=gateway,
        files=files,
        worker=worker,
        ctx=ctx,
        llm=llm,
        api_key=api_key,
    )
    yield value
    for file in store.list(account, "bot_file"):
        if file.get("storage_key"):
            files.delete(account, file["storage_key"])
    memory.shutdown()
    with store.transaction(account) as tx:
        for table in (
            "bot_request",
            "bot_delivery",
            "bot_forget",
            "bot_file",
            "bot_event",
            "bot_step",
            "bot_task",
            "bot_schedule",
            "bot_session",
        ):
            tx.delete(table, {})
        tx.cursor.execute(
            "DELETE FROM memorybot_test_vector WHERE filters->>'account_id'=%s",
            (account,),
        )
        tx.cursor.execute("DELETE FROM outbox_events WHERE account_id=%s", (account,))
        tx.cursor.execute("DELETE FROM context_nodes WHERE account_id=%s", (account,))
        tx.cursor.execute("DELETE FROM cp_audit_logs WHERE account_id=%s", (account,))
        tx.cursor.execute("DELETE FROM cp_agents WHERE account_id=%s", (account,))
        tx.cursor.execute("DELETE FROM cp_users WHERE account_id=%s", (account,))
        tx.cursor.execute("DELETE FROM cp_accounts WHERE account_id=%s", (account,))
    memory._sql_pool.close_all()
    store.close()


def task(runtime, kind, data, *, file=None, **extra):
    ctx = runtime.ctx
    with runtime.store.transaction(ctx.account_id, ctx.user_id) as tx:
        item = runtime.store.create_task(
            tx,
            {
                "kind": kind,
                "request_id": new_id("request"),
                "session_id": ctx.session_id if kind == "chat" else None,
                "input": data,
                "accepted_authz": {
                    "agent_id": ctx.agent_id,
                    "memory_session_id": ctx.session_id,
                    "session_generation": 1,
                },
                **extra,
            },
        )
        if file is not None:
            tx.insert("bot_file", {**file, "task_id": item["task_id"]})
        return item


def execute(runtime, item):
    for _ in range(12):
        runtime.worker.run_once()
        value = runtime.store.get_task(runtime.ctx.account_id, item["task_id"])
        if value["state"] in {
            "succeeded",
            "failed",
            "waiting_confirmation",
            "cancelled",
        }:
            assert value["state"] == "succeeded", value.get("error")
            return value
        with runtime.store.transaction(runtime.ctx.account_id) as tx:
            tx.update(
                "bot_task",
                {"task_id": item["task_id"]},
                {"next_attempt_at": datetime.now(UTC) - timedelta(seconds=1)},
            )
    pytest.fail("task remains active")


def source(runtime, name="notes.md", text="青禾项目接口已完成。\n下一步安排联调。"):
    file_id = new_id("f")
    publication = runtime.files.publish(runtime.ctx.account_id, file_id, text.encode(), name=name)
    item = task(
        runtime,
        "ingest",
        {"file_id": file_id},
        file={**publication, "file_id": file_id, "purpose": "source", "source_kind": "upload"},
    )
    execute(runtime, item)
    return runtime.store.get(runtime.ctx.account_id, "bot_file", {"file_id": file_id})


def memory_node(runtime, text="青禾项目周五发布。"):
    ctx = runtime.ctx
    node = ContextNode(
        f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/entities/qinghe",
        "MEMORY",
        "entity",
        2,
        ctx.user_space_name(),
        text,
        text,
        text,
        {"object_name": "青禾项目", "routing_key": "qinghe"},
    )
    fs = runtime.memory._get_context_fs()
    fs.write_node(node, ctx)
    runtime.memory._get_outbox_store(fs).register_write(node, ctx)
    runtime.memory.drain_outbox_sync(ctx.account_id)
    return node.uri


def test_agfs_shared_bytes_ranges_and_parsed_text(runtime):
    file_id = new_id("f")
    data = "青禾项目\n完成接口。".encode()
    saved = runtime.files.publish(runtime.ctx.account_id, file_id, data, name="qinghe.md")
    assert (
        runtime.files.read(runtime.ctx.account_id, saved["storage_key"], offset=0, size=6)
        == data[:6]
    )
    parsed = parse_text(data, "qinghe.md")
    key = runtime.files.write_parsed(runtime.ctx.account_id, saved["storage_key"], parsed)
    assert json.loads(runtime.files.read(runtime.ctx.account_id, key))["text"] == data.decode()
    assert saved == runtime.files.publish(runtime.ctx.account_id, file_id, data, name="qinghe.md")
    runtime.files.delete(runtime.ctx.account_id, saved["storage_key"])


def test_ingest_chat_artifact_revision_and_capture(runtime):
    original = source(runtime)
    assert original["state"] == "ready"
    assert original["metadata"]["memory_sources"][0]["indexed_version"] == 1
    first = execute(
        runtime,
        task(
            runtime,
            "chat",
            ChatInput(
                text="整理青禾周报",
                file_ids=[original["file_id"]],
                template_id="weekly_report",
            ).model_dump(mode="json"),
        ),
    )
    assert "青禾" in first["result"]["text"]
    assert first["result"]["citations"][0]["uri"].startswith("ctx://")
    artifact = runtime.store.get(
        runtime.ctx.account_id, "bot_file", {"file_id": first["result"]["file_ids"][0]}
    )
    assert "青禾" in runtime.files.read(runtime.ctx.account_id, artifact["storage_key"]).decode()
    revised = execute(
        runtime,
        task(
            runtime,
            "chat",
            ChatInput(text="增加下周计划", file_ids=[artifact["file_id"]]).model_dump(mode="json"),
        ),
    )
    updated = runtime.store.get(
        runtime.ctx.account_id,
        "bot_file",
        {"file_id": revised["result"]["file_ids"][0]},
    )
    assert updated["previous_file_id"] == artifact["file_id"]
    children = runtime.store.list(
        runtime.ctx.account_id,
        "bot_task",
        {"parent_task_id": revised["task_id"], "kind": "memory_write"},
    )
    capture = execute(runtime, children[0])
    assert capture["memory_status"] == "unchanged"


def test_csv_template_outputs_exact_totals(runtime):
    file = source(runtime, "amounts.csv", "项目,金额\n甲,10\n甲,15\n乙,20\n")
    result = execute(
        runtime,
        task(
            runtime,
            "chat",
            ChatInput(
                file_ids=[file["file_id"]],
                template_id="csv_summary",
                output_format="csv",
            ).model_dump(mode="json"),
        ),
    )
    artifact = runtime.store.get(
        runtime.ctx.account_id, "bot_file", {"file_id": result["result"]["file_ids"][0]}
    )
    csv_text = runtime.files.read(runtime.ctx.account_id, artifact["storage_key"]).decode()
    assert "甲,25" in csv_text and "乙,20" in csv_text and "总计,45" in csv_text


def test_memory_list_correction_and_forget_share_canonical_store(runtime):
    uri = memory_node(runtime)
    values = runtime.gateway.list_memories(runtime.ctx, scope={"memory_scope": "user"})
    assert uri in {value["uri"] for value in values}
    corrected = execute(
        runtime,
        task(
            runtime,
            "memory_write",
            {
                "action": "correct",
                "memory_uri": uri,
                "new_text": "青禾项目改为周四发布。",
                "source_refs": [],
            },
        ),
    )
    assert corrected["memory_status"] == "searchable"
    assert runtime.gateway.read_memory(runtime.ctx, uri)["text"] == "青禾项目改为周四发布。"
    forgotten = task(
        runtime,
        "memory_write",
        {"action": "forget", "memory_uri": uri, "source_refs": []},
    )
    with runtime.store.transaction(runtime.ctx.account_id, runtime.ctx.user_id) as tx:
        tx.insert(
            "bot_forget",
            {
                "forget_id": new_id("forget"),
                "task_id": forgotten["task_id"],
                "memory_uri": uri,
                "operation_id": new_id("op"),
                "source_refs": [
                    {
                        "file_id": None,
                        "task_id": None,
                        "uri": uri,
                        "quote": "青禾项目改为周四发布。",
                    }
                ],
            },
        )
    execute(runtime, forgotten)
    assert runtime.memory._get_context_fs().archived_node_version(uri, runtime.ctx) == 3
    assert (
        runtime.store.get(runtime.ctx.account_id, "bot_forget", {"task_id": forgotten["task_id"]})[
            "state"
        ]
        == "effective"
    )


def test_remember_runs_real_after_turn_and_projection(runtime):
    runtime.llm.extract = True
    result = execute(
        runtime,
        task(
            runtime,
            "memory_write",
            {"action": "remember", "new_text": "我负责青禾项目。", "source_refs": []},
        ),
    )
    assert result["memory_status"] == "searchable"
    values = runtime.gateway.list_memories(runtime.ctx)
    assert any("青禾" in row["text"] for row in values)
    remembered = next(
        row for row in values if "青禾" in row["text"] and row["category"] == "entity"
    )
    assert remembered["source_refs"][0]["task_id"] == result["task_id"]
    with runtime.store.transaction(runtime.ctx.account_id, runtime.ctx.user_id) as tx:
        assert runtime.gateway.has_task_reference(tx, result) is True


def test_completed_capture_receipt_reloads_in_a_fresh_service(runtime):
    runtime.llm.extract = True
    item = task(runtime, "memory_write", {"action": "remember", "new_text": "我负责青禾项目。"})
    operation_id = new_id("capture")
    frozen = {"messages": [{"role": "user", "content": "我负责青禾项目。"}]}
    original = runtime.gateway.capture(item, frozen["messages"], operation_id)
    fresh = MemoryService(runtime.memory._cfg)
    fresh._llm = DeterministicLLM()
    gateway = MemoryGateway(fresh, runtime.gateway.context_resolver)
    try:
        restored = gateway.capture_recheck(item, operation_id, frozen)
        assert restored["memory_status"] == "stored"
        assert [row["uri"] for row in restored["sources"]] == [
            row["uri"] for row in original["sources"]
        ]
        assert restored["receipt"]["client_request_id"] == operation_id
        assert restored["receipt"]["status"] == "completed"
    finally:
        fresh.shutdown()
        fresh._sql_pool.close_all()


def test_confirmed_capture_continues_to_searchable(runtime):
    runtime.llm.extract = True
    item = task(runtime, "memory_write", {"action": "remember", "new_text": "我负责青禾项目。"})
    claimed = runtime.store.claim("receipt-handoff", lease_seconds=60)
    identity = (runtime.ctx.account_id, item["task_id"], claimed["owner_id"], claimed["attempt"])
    operation_id = new_id("capture")
    frozen = {"messages": [{"role": "user", "content": "我负责青禾项目。"}]}
    runtime.gateway.capture(claimed, frozen["messages"], operation_id)
    runtime.store.checkpoint(
        *identity,
        1,
        "retrieve",
        {},
        state="succeeded",
        output={"model_calls": 0, "tool_calls": 0, "used_seconds": 0},
    )
    runtime.store.checkpoint(
        *identity,
        2,
        "memory_submit",
        frozen,
        state="waiting_confirmation",
        operation_id=operation_id,
    )
    runtime.store.wait_confirmation(
        *identity,
        {
            "code": "OUTCOME_UNKNOWN",
            "message": "恢复已提交的记忆回执",
            "details": {"operation_id": operation_id},
        },
    )
    service = BotService(
        runtime.store, runtime.memory, runtime.files, runtime.worker.config, gateway=runtime.gateway
    )
    with runtime.store.transaction(runtime.ctx.account_id, runtime.ctx.user_id) as tx:
        service.cancel_task(
            runtime.ctx, tx, item["task_id"], operation_id=operation_id, decision="recheck"
        )
    recovered = execute(runtime, item)
    assert recovered["memory_status"] == "searchable"
    assert recovered["attempt"] == 2
    step = runtime.store.list(
        runtime.ctx.account_id, "bot_step", {"task_id": item["task_id"], "kind": "memory_submit"}
    )[0]
    assert step["operation_id"] == operation_id
    assert step["state"] == "succeeded"


def test_capture_provenance_protects_parent_and_memory_task(runtime):
    runtime.llm.extract = True
    parent = execute(
        runtime,
        task(runtime, "chat", ChatInput(text="我负责青禾项目。").model_dump(mode="json")),
    )
    child = runtime.store.list(
        runtime.ctx.account_id,
        "bot_task",
        {"parent_task_id": parent["task_id"], "kind": "memory_write"},
    )[0]
    child = execute(runtime, child)
    assert child["memory_status"] == "searchable"
    with runtime.store.transaction(runtime.ctx.account_id, runtime.ctx.user_id) as tx:
        assert runtime.gateway.has_task_reference(tx, parent) is True
        assert runtime.gateway.has_task_reference(tx, child) is True


def test_published_skill_template_is_frozen_at_execution(runtime):
    declaration = TemplateRegistry().get("weekly_report")
    declaration.update(
        id="qinghe_weekly", name="青禾专用周报", executor="model", tools=["read_memory"]
    )
    ctx = runtime.ctx
    content = (
        "---\n"
        + yaml.safe_dump(
            {
                "name": "qinghe-weekly",
                "description": "整理青禾进展",
                "memorybot_template": declaration,
            },
            allow_unicode=True,
        )
        + "---\n先写青禾结论，再列联调计划。\n"
    )
    runtime.memory.add_managed_skill(ctx, content)
    result = execute(
        runtime,
        task(
            runtime,
            "chat",
            ChatInput(text="整理青禾周报", template_id="qinghe_weekly").model_dump(mode="json"),
        ),
    )
    steps = runtime.store.list(
        ctx.account_id, "bot_step", {"task_id": result["task_id"], "kind": "retrieve"}
    )
    frozen = next(
        step["output"] for step in steps if step.get("output", {}).get("id") == "qinghe_weekly"
    )
    assert (
        frozen["source_uri"]
        == f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills/imported/qinghe-weekly"
    )
    assert len(frozen["revision"]) == 64 and frozen["instruction"] == "先写青禾结论，再列联调计划。"
    assert frozen["tools"] == ["read_memory"]


def test_completed_model_step_replays_after_lease_transfer(runtime, monkeypatch):
    item = task(
        runtime,
        "chat",
        ChatInput(text="整理青禾进度", template_id="weekly_report").model_dump(mode="json"),
    )
    finish = runtime.store.finish

    def transfer(*args, **kwargs):
        raise LeaseLost("worker handoff")

    monkeypatch.setattr(runtime.store, "finish", transfer)
    runtime.worker.run_once()
    with runtime.store.transaction(runtime.ctx.account_id) as tx:
        tx.update(
            "bot_task",
            {"task_id": item["task_id"]},
            {"lease_until": tx.now() - timedelta(seconds=1)},
        )
    monkeypatch.setattr(runtime.store, "finish", finish)
    recovered = execute(runtime, item)
    assert recovered["attempt"] == 2
    assert len(runtime.llm.messages) == 1
    events = runtime.store.list(
        runtime.ctx.account_id,
        "bot_event",
        {"task_id": item["task_id"], "type": "answer.reset"},
    )
    assert events[0]["payload"] == {"attempt": 2, "replaced_attempt": 1}


def test_model_stream_is_visible_while_model_step_runs(runtime):
    item = task(runtime, "chat", ChatInput(text="我可以上传资料并整理周报吗？").model_dump())
    chunks = ["可以上传 TXT、Markdown、CSV，", "处理完成后选择资料，再使用周报模板。"]

    def streamed(messages, emit):
        assert messages[-1]["content"] == item["input"]["text"]
        assert "上传资料" in messages[0]["content"] and "周报" in messages[0]["content"]
        emit(chunks[0])
        events = runtime.store.list(
            runtime.ctx.account_id,
            "bot_event",
            {"task_id": item["task_id"], "type": "answer.delta"},
        )
        assert events[0]["payload"]["text"] == chunks[0]
        steps = runtime.store.list(
            runtime.ctx.account_id,
            "bot_step",
            {"task_id": item["task_id"], "kind": "model"},
        )
        assert steps[0]["state"] == "running"
        snapshot = runtime.store.task_steps(
            runtime.ctx.account_id, item["task_id"], runtime.ctx.user_id
        )
        assert [row["label"] for row in snapshot] == ["检索记忆", "生成回复"]
        assert snapshot[-1]["state"] == "running"
        assert snapshot[-1]["event_seq"] < events[0]["event_seq"]
        emit(chunks[1])
        return [], "".join(chunks)

    runtime.llm.stream_hook = streamed
    result = execute(runtime, item)
    assert result["result"]["text"] == "".join(chunks)
    assert result["result"]["citations"] == []
    events = runtime.store.list(
        runtime.ctx.account_id,
        "bot_event",
        {"task_id": item["task_id"], "type": "answer.delta"},
        order_by="event_seq ASC",
    )
    assert "".join(event["payload"]["text"] for event in events) == result["result"]["text"]
    phases = runtime.store.list(
        runtime.ctx.account_id,
        "bot_event",
        {"task_id": item["task_id"], "type": "progress"},
        order_by="event_seq ASC",
    )
    assert [phase["payload"]["message"] for phase in phases] == [
        "正在检索记忆",
        "正在生成回复",
    ]
    snapshot = runtime.store.task_steps(
        runtime.ctx.account_id, item["task_id"], runtime.ctx.user_id
    )
    assert [row["state"] for row in snapshot] == ["succeeded", "succeeded"]
    assert snapshot[-1]["summary"] == {
        "character_count": len(result["result"]["text"]),
        "tool_call_count": 0,
    }
    assert all(
        row["finished_at"] >= row["started_at"] and row["elapsed_ms"] >= 0 for row in snapshot
    )
    transitions = runtime.store.events(
        runtime.ctx.account_id, item["task_id"], user_id=runtime.ctx.user_id
    )
    transitions = [event for event in transitions if event["type"].startswith("step.")]
    assert [event["type"] for event in transitions] == [
        "step.started",
        "step.completed",
        "step.started",
        "step.completed",
    ]
    assert [
        event["payload"] for event in transitions if event["type"] == "step.completed"
    ] == snapshot


def test_streamed_tool_round_resets_draft_and_records_used_source(runtime):
    uri = memory_node(runtime)
    rounds = []

    def streamed(messages, emit):
        rounds.append(messages)
        if len(rounds) == 1:
            emit("正在读取青禾资料")
            return [{"tool": "read_memory", "input": {"uri": uri}}], "正在读取青禾资料"
        source_match = re.search(
            r'"citation_id": "(S[0-9]+)"', "\n".join(row["content"] for row in messages)
        )
        answer = f"青禾项目周五发布。[{source_match[1]}]"
        emit(answer)
        return [], answer

    runtime.llm.stream_hook = streamed
    item = task(runtime, "chat", ChatInput(text="青禾何时发布？").model_dump())
    result = execute(runtime, item)
    events = runtime.store.list(
        runtime.ctx.account_id,
        "bot_event",
        {"task_id": item["task_id"]},
        order_by="event_seq ASC",
    )
    draft_events = [event for event in events if event["type"].startswith("answer.")]
    assert [event["type"] for event in draft_events] == [
        "answer.delta",
        "answer.reset",
        "answer.delta",
    ]
    assert draft_events[-1]["payload"]["text"] == result["result"]["text"]
    assert result["result"]["citations"][0]["uri"] == uri
    assert result["result"]["citations"][0]["id"] in result["result"]["text"]
    steps = runtime.store.task_steps(runtime.ctx.account_id, item["task_id"], runtime.ctx.user_id)
    assert [row["label"] for row in steps] == [
        "检索记忆",
        "生成回复",
        "读取记忆",
        "生成回复",
    ]
    assert steps[2]["summary"]["tool_name"] == "read_memory"
    assert steps[2]["summary"]["source_count"] == 1
    assert steps[2]["summary"]["sources"][0]["uri"] == uri


def test_stop_retains_streamed_partial_reply(runtime):
    item = task(runtime, "chat", ChatInput(text="整理青禾进展").model_dump())
    visible = "已完成接口联调。"

    def streamed(messages, emit):
        emit(visible)
        with runtime.store.transaction(runtime.ctx.account_id, runtime.ctx.user_id) as tx:
            tx.update("bot_task", {"task_id": item["task_id"]}, {"cancel_requested": True})
        emit("后续工作" * 128)
        return [], visible

    runtime.llm.stream_hook = streamed
    runtime.worker.run_once()
    result = runtime.store.get_task(runtime.ctx.account_id, item["task_id"])
    assert result["state"] == "cancelled"
    assert result["result"]["text"] == visible
    steps = runtime.store.task_steps(runtime.ctx.account_id, item["task_id"], runtime.ctx.user_id)
    assert steps[-1]["state"] == "cancelled"
    assert steps[-1]["finished_at"] >= steps[-1]["started_at"]
    events = runtime.store.list(
        runtime.ctx.account_id,
        "bot_event",
        {"task_id": item["task_id"], "type": "step.cancelled"},
    )
    assert events[0]["payload"] == steps[-1]


def test_step_retry_keeps_identity_and_finishes_with_latest_attempt(runtime):
    item = task(runtime, "chat", ChatInput(text="整理资料").model_dump())
    attempts = []

    def streamed(messages, emit):
        attempts.append(messages)
        if len(attempts) == 1:
            raise RuntimeError("temporary provider issue")
        emit("资料整理完成。")
        return [], "资料整理完成。"

    runtime.llm.stream_hook = streamed
    runtime.worker.run_once()
    failed = runtime.store.task_steps(runtime.ctx.account_id, item["task_id"], runtime.ctx.user_id)[
        -1
    ]
    assert failed["state"] == "failed" and failed["summary"]["error_code"] == "STEP_FAILED"
    with runtime.store.transaction(runtime.ctx.account_id, runtime.ctx.user_id) as tx:
        tx.update("bot_task", {"task_id": item["task_id"]}, {"next_attempt_at": tx.now()})
    result = execute(runtime, item)
    completed = runtime.store.task_steps(
        runtime.ctx.account_id, item["task_id"], runtime.ctx.user_id
    )[-1]
    assert completed["state"] == "succeeded" and completed["attempt"] == 2
    assert completed["step_id"] == failed["step_id"]
    assert completed["event_seq"] > failed["event_seq"]
    assert result["result"]["text"] == "资料整理完成。"


@pytest.mark.parametrize("schedule_count", [0, 1])
def test_schedule_lookup_returns_tool_receipt_and_completes_reply(runtime, schedule_count):
    next_run = datetime.now(UTC) + timedelta(days=1)
    if schedule_count:
        with runtime.store.transaction(runtime.ctx.account_id, runtime.ctx.user_id) as tx:
            tx.insert(
                "bot_schedule",
                {
                    "schedule_id": new_id("schedule"),
                    "session_id": runtime.ctx.session_id,
                    "name": "晨间提醒",
                    "rule": {"type": "interval", "interval_seconds": 86400},
                    "timezone": "Asia/Shanghai",
                    "task_input": ChatInput(text="提醒检查进展").model_dump(),
                    "accepted_authz": {},
                    "next_run_at": next_run,
                },
            )
    runtime.llm.planned_calls.append(
        {"id": "call_schedule_lookup", "tool": "list_schedules", "input": {}}
    )
    answers = []

    def streamed(messages, emit):
        assistant, receipt = messages[-2:]
        assert assistant["role"] == "assistant"
        assert assistant["tool_calls"] == [
            {
                "id": "call_schedule_lookup",
                "type": "function",
                "function": {"name": "list_schedules", "arguments": "{}"},
            }
        ]
        assert receipt["role"] == "tool" and receipt["tool_call_id"] == "call_schedule_lookup"
        data = json.loads(receipt["content"])
        assert len(data["schedules"]) == schedule_count and data["has_more"] is False
        if schedule_count:
            plan = data["schedules"][0]
            assert plan["name"] == "晨间提醒" and plan["timezone"] == "Asia/Shanghai"
            assert datetime.fromisoformat(plan["next_run_at"].replace("Z", "+00:00")) == next_run
            text = "晨间提醒将在下次计划时间执行。"
        else:
            text = "当前没有提醒计划。"
        answers.append(text)
        emit(text)
        return [], text

    runtime.llm.stream_hook = streamed
    item = task(
        runtime,
        "chat",
        ChatInput(text="请查询我已有的提醒计划，列出名称和下次执行时间。").model_dump(),
    )
    result = execute(runtime, item)
    assert result["result"]["text"] == answers[0] and len(answers) == 1
    steps = runtime.store.task_steps(runtime.ctx.account_id, item["task_id"], runtime.ctx.user_id)
    assert [row["label"] for row in steps] == [
        "检索记忆",
        "生成回复",
        "查询计划",
        "生成回复",
    ]
    assert steps[2]["summary"] == {
        "tool_name": "list_schedules",
        "schedule_count": schedule_count,
    }
    assert steps[-1]["state"] == "succeeded"
    budget = runtime.store.get(
        runtime.ctx.account_id, "bot_step", {"task_id": item["task_id"], "step_no": 1}
    )["output"]
    assert budget["model_calls"] == 2 and budget["tool_calls"] == 1


def test_queued_followups_preserve_prior_completed_conversation(runtime):
    first = task(runtime, "chat", ChatInput(text="青禾负责人是林悦。").model_dump())
    execute(runtime, first)
    current = task(runtime, "chat", ChatInput(text="她负责哪个项目？").model_dump())
    for index in range(32):
        task(runtime, "chat", ChatInput(text=f"后续工作 {index}").model_dump())

    def streamed(messages, emit):
        assert messages[-1]["content"] == "她负责哪个项目？"
        assert any(
            row["role"] == "user" and row["content"] == "青禾负责人是林悦。" for row in messages
        )
        answer = "林悦负责青禾项目。"
        emit(answer)
        return [], answer

    runtime.llm.stream_hook = streamed
    result = execute(runtime, current)
    assert result["result"]["text"] == "林悦负责青禾项目。"


def test_schedule_new_risk_updates_baseline_and_delivers(runtime):
    uri = memory_node(runtime, "青禾风险：接口联调延期一天。")
    ctx = runtime.ctx
    schedule_id = new_id("schedule")
    with runtime.store.transaction(ctx.account_id, ctx.user_id) as tx:
        tx.insert(
            "bot_schedule",
            {
                "schedule_id": schedule_id,
                "session_id": ctx.session_id,
                "name": "青禾风险",
                "task_input": ChatInput(text="检查青禾风险").model_dump(mode="json"),
                "accepted_authz": {
                    "agent_id": ctx.agent_id,
                    "memory_session_id": ctx.session_id,
                    "session_generation": 1,
                },
                "rule": {"type": "interval", "interval_seconds": 60},
                "timezone": "Asia/Shanghai",
                "condition": {"type": "new_risk", "project": "青禾"},
                "next_run_at": tx.now() - timedelta(seconds=1),
            },
        )
    runtime.worker.run_once()
    baseline = runtime.store.get(ctx.account_id, "bot_schedule", {"schedule_id": schedule_id})[
        "baseline"
    ]
    assert len(baseline["items"]) == 1
    execute(
        runtime,
        task(
            runtime,
            "memory_write",
            {
                "action": "correct",
                "memory_uri": uri,
                "new_text": "青禾风险：接口联调延期一天。\n青禾风险：测试环境待确认。",
                "source_refs": [],
            },
        ),
    )
    with runtime.store.transaction(ctx.account_id) as tx:
        tx.update(
            "bot_schedule",
            {"schedule_id": schedule_id},
            {"next_run_at": tx.now() - timedelta(seconds=1)},
        )
    runtime.worker.run_once()
    tasks = runtime.store.list(
        ctx.account_id,
        "bot_task",
        {"schedule_id": schedule_id},
        order_by="created_at DESC",
    )
    result = execute(runtime, tasks[0])
    deliveries = runtime.store.list(
        ctx.account_id, "bot_delivery", {"source_task_id": result["task_id"]}
    )
    assert len(deliveries) == 1
    execute(runtime, runtime.store.get_task(ctx.account_id, deliveries[0]["task_id"]))
    sent = runtime.store.get(
        ctx.account_id, "bot_delivery", {"delivery_id": deliveries[0]["delivery_id"]}
    )
    assert sent["state"] == "sent" and sent["provider_receipt"]["provider"] == "web"
    assert (
        len(
            runtime.store.get(ctx.account_id, "bot_schedule", {"schedule_id": schedule_id})[
                "baseline"
            ]["items"]
        )
        == 2
    )


def test_conversation_creates_reminder_and_pauses_it(runtime):
    due = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    runtime.llm.planned_calls.append(
        {
            "tool": "create_schedule",
            "input": {
                "name": "青禾上线检查",
                "task_input": {"text": "检查青禾上线"},
                "rule": {"type": "once", "at": due},
            },
        }
    )
    first = execute(
        runtime,
        task(
            runtime,
            "chat",
            ChatInput(text="明天提醒我检查青禾上线").model_dump(mode="json"),
        ),
    )
    schedules = runtime.store.list(runtime.ctx.account_id, "bot_schedule")
    assert (
        schedules[0]["timezone"] == "Asia/Shanghai"
        and schedules[0]["session_id"] == first["session_id"]
    )
    runtime.llm.planned_calls.append(
        {
            "tool": "update_schedule",
            "input": {
                "schedule_id": schedules[0]["schedule_id"],
                "expected_revision": 1,
                "action": "pause",
            },
        }
    )
    execute(
        runtime,
        task(
            runtime,
            "chat",
            ChatInput(text="暂停青禾上线检查提醒").model_dump(mode="json"),
        ),
    )
    updated = runtime.store.get(
        runtime.ctx.account_id,
        "bot_schedule",
        {"schedule_id": schedules[0]["schedule_id"]},
    )
    assert updated["state"] == "paused" and updated["revision"] == 2


def test_conversation_correction_queues_real_memory_action(runtime):
    uri = memory_node(runtime)
    runtime.llm.planned_calls.append(
        {
            "tool": "memory_action",
            "input": {
                "action": "correct",
                "memory_uri": uri,
                "new_text": "青禾项目周四发布。",
            },
        }
    )
    conversation = execute(
        runtime,
        task(
            runtime,
            "chat",
            ChatInput(text="更正青禾发布时间为周四").model_dump(mode="json"),
        ),
    )
    children = runtime.store.list(
        runtime.ctx.account_id,
        "bot_task",
        {"parent_task_id": conversation["task_id"], "kind": "memory_write"},
    )
    assert len(children) == 1 and children[0]["input"]["action"] == "correct"
    execute(runtime, children[0])
    assert runtime.gateway.read_memory(runtime.ctx, uri)["text"] == "青禾项目周四发布。"
    assert (
        runtime.store.get_task(runtime.ctx.account_id, conversation["task_id"])["memory_status"]
        == "searchable"
    )


def test_conversation_read_tools_use_authorized_memory_and_persist_budget(runtime):
    uri = memory_node(runtime)
    runtime.llm.planned_calls.extend(
        [
            {"tool": "search_memory", "input": {"query": "青禾"}},
            {"tool": "read_memory", "input": {"uri": uri}},
        ]
    )
    result = execute(
        runtime,
        task(runtime, "chat", ChatInput(text="青禾什么时候发布").model_dump(mode="json")),
    )
    budget = runtime.store.get(
        runtime.ctx.account_id, "bot_step", {"task_id": result["task_id"], "step_no": 1}
    )["output"]
    assert budget["model_calls"] == 3 and budget["tool_calls"] == 2
    assert uri in {citation["uri"] for citation in result["result"]["citations"]}


def test_archived_memory_cancellation_keeps_receipt_and_can_be_remembered(runtime, monkeypatch):
    uri = memory_node(runtime)
    app = BotService(
        runtime.store,
        runtime.memory,
        runtime.files,
        runtime.worker.config,
        gateway=runtime.gateway,
    )
    with runtime.store.transaction(runtime.ctx.account_id, runtime.ctx.user_id) as tx:
        _, accepted = app.memory_action(
            runtime.ctx,
            tx,
            {
                "action": "forget",
                "memory_uri": uri,
                "source_refs": [],
                "fact_scope": {"mode": "entry"},
            },
            new_id("request"),
        )
    item = accepted["task"]
    projection = runtime.gateway.projection_ready
    monkeypatch.setattr(runtime.gateway, "projection_ready", lambda *_: False)
    runtime.worker.run_once()
    monkeypatch.setattr(runtime.gateway, "projection_ready", projection)
    pending = runtime.store.get_task(runtime.ctx.account_id, item["task_id"])
    assert pending["state"] == "retry_wait", pending.get("error")
    with runtime.store.transaction(runtime.ctx.account_id, runtime.ctx.user_id) as tx:
        app.cancel_task(runtime.ctx, tx, item["task_id"])
    forgotten = runtime.store.get(
        runtime.ctx.account_id, "bot_forget", {"task_id": item["task_id"]}
    )
    assert forgotten["state"] == "effective" and forgotten["source_refs"]
    runtime.llm.extract = True
    with runtime.store.transaction(runtime.ctx.account_id, runtime.ctx.user_id) as tx:
        _, accepted = app.memory_action(
            runtime.ctx,
            tx,
            {
                "action": "remember",
                "memory_uri": uri,
                "new_text": "我负责青禾项目。",
                "source_refs": [],
                "fact_scope": {"mode": "entry"},
            },
            new_id("request"),
        )
    execute(runtime, accepted["task"])
    assert (
        runtime.store.get(runtime.ctx.account_id, "bot_forget", {"task_id": item["task_id"]})[
            "state"
        ]
        == "superseded"
    )


def test_shared_object_scan_reports_backup_protection(runtime):
    files, account = runtime.files, runtime.ctx.account_id
    file_id = new_id("f")
    storage_key = files.publish(account, file_id, b"shared source", name="source.txt")["storage_key"]
    directory = files._path(account, storage_key).rsplit("/", 1)[0]
    marker = f"{directory}/backup.json"
    try:
        files.client.write(marker, b'{"backup_refs":["snapshot-1"]}')
        cursor, found = None, None
        for _ in range(50):
            page = files.list_objects(limit=100, cursor=cursor)
            found = next((row for row in page["items"] if row["file_id"] == file_id), None)
            cursor = page["next_cursor"]
            if found is not None or not cursor:
                break
        assert found is not None, f"Published object {file_id} was not listed"
        assert found["backup_protected"] is True
        assert found["modified_at"]
    finally:
        if any(entry["name"] == "backup.json" for entry in files._entries(directory)):
            files.client.rm(marker)
        files.delete(account, storage_key)


def test_running_file_cancellation_publishes_terminal_file_state(runtime, monkeypatch):
    file_id = new_id("f")
    saved = runtime.files.publish(
        runtime.ctx.account_id, file_id, "青禾资料".encode(), name="source.txt"
    )
    item = task(
        runtime,
        "ingest",
        {"file_id": file_id},
        file={**saved, "file_id": file_id, "purpose": "source", "source_kind": "upload"},
    )
    read = runtime.files.read

    def read_and_cancel(*args, **kwargs):
        data = read(*args, **kwargs)
        with runtime.store.transaction(runtime.ctx.account_id) as tx:
            tx.update("bot_task", {"task_id": item["task_id"]}, {"cancel_requested": True})
        return data

    monkeypatch.setattr(runtime.files, "read", read_and_cancel)
    runtime.worker.run_once()
    assert runtime.store.get_task(runtime.ctx.account_id, item["task_id"])["state"] == "cancelled"
    assert (
        runtime.store.get(runtime.ctx.account_id, "bot_file", {"file_id": file_id})["state"]
        == "failed"
    )


@pytest.mark.parametrize("category", ["tool", "experience"])
def test_structured_agent_correction_updates_units_revisions_and_projection(runtime, category):
    from extraction.agent_memory import (
        memory_units,
        node_memory_fields,
        render_agent_memory,
    )
    from index.index_record_builder import build_index_records

    ctx = runtime.ctx
    units = [
        {
            "key": "batch",
            "title": "青禾批次",
            "content": "青禾每批处理20个请求。",
            "grounding": [{"uri": f"ctx://{ctx.account_id}/sessions/source/history/first"}],
        }
    ]
    if category == "tool":
        units.append(
            {
                "key": "cursor",
                "title": "青禾分页",
                "content": "沿用返回的游标。",
                "grounding": units[0]["grounding"],
            }
        )
    fields = {"tool_name": "qinghe", "units": units} if category == "tool" else units[0]
    directory = "tool_units" if category == "tool" else "experiences"
    uri = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/{directory}/qinghe"
    node = ContextNode(
        uri,
        "MEMORY",
        category,
        4,
        ctx.agent_space_name(),
        "青禾批次说明",
        "青禾处理规范",
        render_agent_memory(category, fields),
        {
            "routing_key": "qinghe",
            "agent_memory_version": 1,
            "structured_fields": fields,
            "agent_memory": {
                "version": 1,
                "applied_operations": [],
                "revisions": [],
                "producer_agent_ids": [ctx.agent_id],
                "units": {
                    unit["key"]: {
                        "revision": 1,
                        "grounding": unit["grounding"],
                        "withdrawn": False,
                    }
                    for unit in units
                },
            },
        },
    )
    fs = runtime.memory._get_context_fs()
    fs.write_node(node, ctx)
    runtime.memory._get_outbox_store(fs).register_write(node, ctx)
    runtime.memory.drain_outbox_sync(ctx.account_id)
    corrected = execute(
        runtime,
        task(
            runtime,
            "memory_write",
            {
                "action": "correct",
                "memory_uri": uri,
                "new_text": "青禾每批处理50个请求。",
                "source_refs": [],
            },
        ),
    )
    saved = fs.read_node(uri, ctx)
    active = memory_units(category, node_memory_fields(saved))
    assert [unit["content"] for unit in active] == ["青禾每批处理50个请求。"]
    assert saved.metadata["version"] == 2
    assert saved.metadata["agent_memory"]["units"]["batch"]["revision"] == 2
    if category == "tool":
        assert saved.metadata["agent_memory"]["units"]["cursor"]["withdrawn"] is True
    records = build_index_records(saved)
    for record in records:
        actual = runtime.memory._get_shared_vector_index().get_record(record.id)
        assert actual.text == record.text and actual.metadata["canonical_revision"] == 2
    assert active[0]["grounding"][0]["role"] == "user"
    assert active[0]["grounding"][0]["origin"] == "memorybot_explicit_correction"
    with runtime.store.transaction(ctx.account_id, ctx.user_id) as tx:
        assert runtime.gateway.has_task_reference(tx, corrected) is True
