"""Recover committed AGFS forget operations after the source node is removed."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from types import SimpleNamespace

import pytest
from flask import Flask

from core.dreaming_evidence import node_created_at
from core.models import ContextNode
from index.index_record_builder import build_index_records
from index.outbox_worker import OutboxWorker
from index.vector_projection import VectorProjection
from memorybot.config import BotConfig
from memorybot.gateway import WorkPendingError
from memorybot.http import create_blueprint
from memorybot.runtime import BotWorker
from memorybot.worker import create_service
from server.memory_service import MemoryService
from tests.integration.test_memorybot_runtime import runtime  # noqa: F401

pytestmark = pytest.mark.integration


class WorkerStopped(BaseException):
    pass


@pytest.fixture
def agfs_runtime(runtime):  # noqa: F811
    ctx = runtime.ctx
    cfg = replace(
        runtime.memory._cfg,
        storage_backend="agfs",
        role_control_enabled=True,
        agfs_base_url=runtime.files.client.api_base,
        agfs_mount_prefix=f"/local/{ctx.account_id}",
        root_api_key="archive-recovery-root",
    )
    runtime.files.client.mkdir(cfg.agfs_mount_prefix)
    memory = MemoryService(cfg)
    memory._llm = runtime.llm
    service = None
    try:
        key = memory.get_key_manager().create_account(ctx.account_id, ctx.user_id)
        memory.get_tenant_admin_service().create_agent(
            ctx,
            ctx.account_id,
            ctx.agent_id,
            owner_user_id=ctx.user_id,
        )
        service = create_service(
            memory,
            BotConfig(
                enabled=True,
                connection_string=cfg.sql_connection_string,
                files_base_url=cfg.agfs_base_url,
            ),
        )
        service.store.claim = partial(service.store.claim, account_id=ctx.account_id)
        worker = BotWorker(service.store, service.gateway, service.files)
        app = Flask(__name__)
        app.register_blueprint(create_blueprint(lambda: service))
        client = app.test_client()
        auth = {"X-API-Key": key}
        yield SimpleNamespace(
            ctx=ctx, memory=memory, service=service, worker=worker, client=client, auth=auth
        )
    finally:
        if service is not None:
            service.store.close()
        memory.shutdown()


@pytest.mark.parametrize("replacement", [None, "pending", "indexed"])
@pytest.mark.parametrize("cancel", [False, True], ids=["resume", "cancel"])
@pytest.mark.parametrize("stop_after", ["archive", "sql"])
def test_committed_forget_recovers_after_source_removal(
    agfs_runtime,
    replacement,
    cancel,
    stop_after,
    monkeypatch,
):
    ctx, memory, service = agfs_runtime.ctx, agfs_runtime.memory, agfs_runtime.service
    worker, client, auth = agfs_runtime.worker, agfs_runtime.client, agfs_runtime.auth
    uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/entities/launch"
    text = "发布负责人是林悦。"
    node = ContextNode(uri, "MEMORY", "entity", 2, ctx.user_space_name(), text, text, text)
    fs = memory._get_context_fs()
    fs.write_node(node, ctx)
    memory._get_outbox_store(fs).register_write(node, ctx)
    memory.drain_outbox_sync(ctx.account_id)
    original = fs.read_node(uri, ctx)
    record_ids = [record.id for record in build_index_records(original)]
    response = client.post(
        "/api/v1/bot/memory-actions",
        headers=auth | {"Idempotency-Key": "forget"},
        json={"action": "forget", "memory_uri": uri},
    )
    assert response.status_code == 202
    task_id = response.json["data"]["task"]["task_id"]
    apply = service.gateway.mutate_target

    def interrupted(*args, **kwargs):
        if stop_after == "sql":
            apply(*args, **kwargs)
        raise WorkerStopped()

    with monkeypatch.context() as process:
        if stop_after == "archive":
            process.setattr("fs.agfs_adapter.archive_receipts.retain_receipt", interrupted)
        else:
            process.setattr(service.gateway, "mutate_target", interrupted)
        with pytest.raises(WorkerStopped):
            worker.run_once()
    claimed = service.store.get_task(ctx.account_id, task_id)
    with service.store.transaction(ctx.account_id, ctx.user_id) as tx:
        [step] = tx.list("bot_step", {"task_id": task_id, "kind": "memory_apply"})
    frozen = step["input"]
    assert fs.archived_node_version(uri, ctx) == 2
    fs.delete_node(uri, ctx)
    if replacement:
        new_text = "发布负责人变更为陈默。"
        updated = replace(node, abstract=new_text, overview=new_text, content=new_text, metadata={})
        fs.write_node(updated, ctx)
        if replacement == "indexed":
            memory._get_outbox_store(fs).register_write(updated, ctx)
            memory.drain_outbox_sync(ctx.account_id)
        assert node_created_at(fs.read_node(uri, ctx).metadata) != frozen["target_created_at"]
    if cancel:
        cancelled = client.post(
            f"/api/v1/bot/tasks/{task_id}/cancel",
            headers=auth | {"Idempotency-Key": "cancel"},
            json={},
        )
        assert cancelled.status_code == 202
    with service.store.transaction(ctx.account_id, ctx.user_id) as tx:
        tx.update(
            "bot_task",
            {"task_id": task_id},
            {
                "lease_until": datetime.now(UTC) - timedelta(seconds=60),
            },
        )
    worker.run_once()
    memory.drain_outbox_sync(ctx.account_id)
    forget = service.store.get(ctx.account_id, "bot_forget", {"task_id": task_id})
    assert forget["state"] == "effective"
    assert forget["source_refs"] == frozen["source_refs"]
    records = [service.gateway.index.get_record(record_id) for record_id in record_ids]
    if replacement:
        assert all(record.text == new_text for record in records)
    else:
        assert records == [None] * len(record_ids)
    with service.store.transaction(ctx.account_id, ctx.user_id) as tx:
        receipt = service.gateway.mutate_target(tx, claimed, frozen, step["operation_id"])
    assert receipt["node_created_at"] == frozen["target_created_at"]
    assert service.gateway.projection_ready(claimed, [receipt])


@pytest.mark.parametrize("category", ["tool", "experience"])
def test_forget_finishes_after_retired_agent_units_are_cleaned(agfs_runtime, category, monkeypatch):
    from extraction.agent_memory import render_agent_memory

    active = agfs_runtime
    ctx, memory, service = active.ctx, active.memory, active.service
    fs, index = service.gateway.fs, service.gateway.index
    directory = "tool_units" if category == "tool" else "experiences"
    uri = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/{directory}/rollout"

    def node_with_unit(key):
        unit = {"key": key, "title": key, "content": f"发布规则采用{key}。", "grounding": []}
        units = [unit]
        if category == "tool":
            units.append({**unit, "key": "shared", "title": "shared"})
        fields = {"tool_name": "rollout", "units": units} if category == "tool" else unit
        return ContextNode(
            uri,
            "MEMORY",
            category,
            4,
            ctx.agent_space_name(),
            key,
            key,
            render_agent_memory(category, fields),
            {
                "routing_key": "rollout",
                "agent_memory_version": 1,
                "structured_fields": fields,
                "agent_memory": {
                    "version": 1,
                    "applied_operations": [],
                    "revisions": [],
                    "producer_agent_ids": [ctx.agent_id],
                    "units": {
                        item["key"]: {"revision": 1, "grounding": [], "withdrawn": False}
                        for item in units
                    },
                },
            },
        )

    original = node_with_unit("original")
    fs.write_node(original, ctx)
    outbox = service.gateway.outbox
    outbox.register_write(original, ctx)
    memory.drain_outbox_sync(ctx.account_id)
    original_ids = {record.id for record in build_index_records(fs.read_node(uri, ctx))}
    request = {"action": "forget", "memory_uri": uri}
    accepted = active.client.post(
        "/api/v1/bot/memory-actions",
        headers=active.auth | {"Idempotency-Key": "forget-units"},
        json=request,
    )
    assert accepted.status_code == 202
    task_id = accepted.json["data"]["task"]["task_id"]
    apply = service.gateway.mutate_target

    def stop_after_archive(*args, **kwargs):
        apply(*args, **kwargs)
        raise WorkerStopped()

    with monkeypatch.context() as stopped:
        stopped.setattr(service.gateway, "mutate_target", stop_after_archive)
        with pytest.raises(WorkerStopped):
            active.worker.run_once()
    [(_, archive)] = [
        (owner, event)
        for owner, event in outbox.list_pending(ctx.account_id)
        if event.event_type == "ARCHIVE_CONTEXT"
    ]

    def unavailable_projection(*args):
        raise RuntimeError("projection cleanup temporarily unavailable")

    with monkeypatch.context() as unavailable:
        unavailable.setattr(VectorProjection, "delete", unavailable_projection)
        unavailable.setattr(OutboxWorker, "calculate_backoff", lambda *_: timedelta(hours=1))
        failed = memory.drain_outbox_sync(ctx.account_id)
    assert failed["failed"] == 1
    fs.delete_node(uri, ctx)
    replacement = node_with_unit("replacement")
    fs.write_node(replacement, ctx)
    outbox.register_write(replacement, ctx)
    memory.drain_outbox_sync(ctx.account_id)
    expected = build_index_records(fs.read_node(uri, ctx))
    retired = original_ids - {record.id for record in expected}
    assert retired
    for record in expected:
        assert index.get_record(record.id).text == record.text
    with service.store.transaction(ctx.account_id, ctx.user_id) as tx:
        tx.update(
            "bot_task",
            {"task_id": task_id},
            {"lease_until": datetime.now(UTC) - timedelta(seconds=60)},
        )
    active.worker.run_once()
    response = active.client.get(f"/api/v1/bot/tasks/{task_id}", headers=active.auth)
    assert response.status_code == 200
    print(
        json.dumps(
            {
                "method": "POST",
                "path": "/api/v1/bot/memory-actions",
                "request": request,
                "accepted": accepted.status_code,
                "task_response": {
                    "status": response.status_code,
                    "state": response.json["data"]["state"],
                    "memory_status": response.json["data"]["memory_status"],
                },
                "archive_failures": failed["failed"],
                "retired_vectors": {key: index.get_record(key).text for key in retired},
            },
            ensure_ascii=False,
        )
    )
    assert response.json["data"]["state"] == "retry_wait"
    with service.store.transaction(ctx.account_id, ctx.user_id) as tx:
        [step] = tx.list("bot_step", {"task_id": task_id, "kind": "memory_apply"})
        with pytest.raises(WorkPendingError):
            service.gateway.verify_projection(
                tx, service.store.get_task(ctx.account_id, task_id), [step["output"]]
            )
    with outbox.processing_lock(archive.uri):
        outbox.persist_event(replace(archive, retry_count=1, next_retry_at=""))
    memory.drain_outbox_sync(ctx.account_id)
    with service.store.transaction(ctx.account_id, ctx.user_id) as tx:
        tx.update(
            "bot_task",
            {"task_id": task_id},
            {"next_attempt_at": datetime.now(UTC) - timedelta(seconds=1)},
        )
    active.worker.run_once()
    finished = active.client.get(f"/api/v1/bot/tasks/{task_id}", headers=active.auth)
    assert finished.json["data"]["state"] == "succeeded"
    assert [index.get_record(key) for key in retired] == [None] * len(retired)
    for record in expected:
        assert index.get_record(record.id).text == record.text
    print(
        json.dumps(
            {
                "method": "GET",
                "path": f"/api/v1/bot/tasks/{task_id}",
                "status": finished.status_code,
                "state": finished.json["data"]["state"],
                "retired_vectors": [index.get_record(key) for key in retired],
                "replacement_vectors": {
                    record.id: index.get_record(record.id).text for record in expected
                },
            },
            ensure_ascii=False,
        )
    )
