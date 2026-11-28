"""Exercise a Bot deployment with AGFS canonical memory and a vector database."""

import io
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from types import SimpleNamespace

import pytest
import yaml
from flask import Flask

from core.models import ContextNode
from index.index_record_builder import build_index_records
from memorybot.config import BotConfig
from memorybot.http import create_blueprint
from memorybot.runtime import BotWorker
from memorybot.worker import create_service
from server.memory_service import MemoryService
from tests.integration.test_memorybot_runtime import execute, memory_node, runtime  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("shared_spaces", "correction_stop"),
    [
        (False, None),
        (True, None),
        (False, "before_canonical_commit"),
        (False, "before_event_publish"),
        (False, "cancel_before_event_publish"),
    ],
    ids=[
        "agfs",
        "agfs-sql-catalog",
        "before-canonical-commit",
        "before-event-publish",
        "cancel-before-event-publish",
    ],
)
def test_memorybot_agfs_vector(runtime, tmp_path, shared_spaces, correction_stop, monkeypatch):  # noqa: F811
    ctx = runtime.ctx
    config = replace(
        runtime.memory._cfg,
        storage_backend="agfs",
        agfs_base_url=runtime.files.client.api_base,
        agfs_mount_prefix=f"/local/{ctx.account_id}",
        extraction_mode="eager",
        shared_spaces_enabled=shared_spaces,
        role_control_enabled=True,
        root_api_key="test-memorybot-agfs-root",
    )
    runtime.files.client.mkdir(config.agfs_mount_prefix)
    path = tmp_path / "ogmem.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "bot": {"enabled": True, "files": {"base_url": config.agfs_base_url}},
            }
        ),
        encoding="utf-8",
    )
    bot_config = BotConfig.load(str(path), config)
    memory = MemoryService(config)
    memory._llm = runtime.llm
    service = None
    try:
        if shared_spaces:
            key = runtime.api_key
        else:
            key = memory.get_key_manager().create_account(ctx.account_id, ctx.user_id)
            memory.get_tenant_admin_service().create_agent(
                ctx,
                ctx.account_id,
                ctx.agent_id,
                owner_user_id=ctx.user_id,
            )
        service = create_service(memory, bot_config)
        service.store.claim = partial(service.store.claim, account_id=ctx.account_id)
        worker = BotWorker(service.store, service.gateway, service.files, config=bot_config)
        active = SimpleNamespace(ctx=ctx, memory=memory, worker=worker, store=service.store)
        app = Flask(__name__)
        app.register_blueprint(create_blueprint(lambda: service))
        client = app.test_client()
        auth = {"X-API-Key": key}
        prefix = "/api/v1/bot"

        def submit(path, payload=None, *, data=None, request_id):
            args = {"json": payload} if data is None else {"data": data}
            return client.post(
                prefix + path, headers={**auth, "Idempotency-Key": request_id}, **args
            )

        uri = memory_node(active)
        browsed = client.get(prefix + "/memories", headers=auth)
        assert browsed.status_code == 200
        assert browsed.json["data"]["items"][0]["uri"] == uri
        fs = memory._get_context_fs()
        agent_ctx = replace(ctx, agent_id="release")
        memory.get_tenant_admin_service().create_agent(
            ctx,
            ctx.account_id,
            agent_ctx.agent_id,
            owner_user_id=ctx.user_id,
        )
        agent_uri = f"ctx://{ctx.account_id}/agents/release/memories/cases/rollout"
        example = "灰度发布采用双人复核。"
        agent_node = ContextNode(
            agent_uri,
            "MEMORY",
            "case",
            2,
            agent_ctx.agent_space_name(),
            example,
            example,
            example,
        )
        fs.write_node(agent_node, agent_ctx)
        memory._get_outbox_store(fs).register_write(agent_node, agent_ctx)
        memory.drain_outbox_sync(ctx.account_id)
        for query in (
            {"memory_scope": "authorized"},
            {"memory_scope": "authorized", "q": "灰度发布"},
            {"uri": agent_uri},
        ):
            response = client.get(prefix + "/memories", headers=auth, query_string=query)
            assert response.status_code == 200
            assert agent_uri in {item["uri"] for item in response.json["data"]["items"]}

        uploaded = submit(
            "/files",
            data={"file": (io.BytesIO("青禾接口已完成。".encode()), "notes.txt")},
            request_id="upload",
        )
        assert uploaded.status_code == 202
        file_info = uploaded.json["data"]
        execute(active, file_info["task"])
        detail = client.get(prefix + "/files/" + file_info["file"]["file_id"], headers=auth)
        assert detail.json["data"]["state"] == "ready"
        source_uri = detail.json["data"]["metadata"]["memory_sources"][0]["uri"]
        assert fs.read_node(source_uri, ctx).content == "青禾接口已完成。"

        session = submit(
            "/sessions", {"title": "青禾", "agent_id": ctx.agent_id}, request_id="session"
        )
        assert session.status_code == 201
        sid = session.json["data"]["session_id"]
        runtime.llm.extract = True
        chat = submit(
            f"/sessions/{sid}/tasks",
            {"text": "整理青禾接口进展", "file_ids": [file_info["file"]["file_id"]]},
            request_id="chat",
        )
        assert chat.status_code == 202
        answer = execute(active, chat.json["data"])
        assert "青禾项目接口已完成" in answer["result"]["text"]
        [capture] = service.store.list(
            ctx.account_id,
            "bot_task",
            {"parent_task_id": answer["task_id"], "kind": "memory_write"},
        )
        assert execute(active, capture)["memory_status"] == "searchable"
        prior_version = fs.read_node(uri, ctx).metadata["version"]

        corrected = submit(
            "/memory-actions",
            {
                "action": "correct",
                "memory_uri": uri,
                "new_text": "青禾项目改为周四发布。",
            },
            request_id="correct",
        )
        assert corrected.status_code == 202
        if correction_stop:

            class WorkerStopped(BaseException):
                pass

            write = service.gateway.fs.write_node
            pending_at_write = []

            def interrupt_write(node, owner_ctx):
                pending_at_write.extend(
                    event.payload["canonical_write_revision"]
                    for _, event in service.gateway.outbox.list_pending(ctx.account_id)
                    if event.uri == uri and "canonical_write_revision" in event.payload
                )
                if "before_event_publish" in correction_stop:
                    write(node, owner_ctx)
                raise WorkerStopped()

            task_id = corrected.json["data"]["task"]["task_id"]
            with monkeypatch.context() as process:
                process.setattr(service.gateway.fs, "write_node", interrupt_write)
                with pytest.raises(WorkerStopped):
                    worker.run_once()
            print(
                json.dumps(
                    {
                        "method": "POST",
                        "path": prefix + "/memory-actions",
                        "request": {
                            "action": "correct",
                            "memory_uri": uri,
                            "new_text": "青禾项目改为周四发布。",
                        },
                        "status": corrected.status_code,
                        "stop": correction_stop,
                        "canonical_version": fs.read_node(uri, ctx).metadata["version"],
                        "queued_at_canonical_write": pending_at_write,
                    },
                    ensure_ascii=False,
                )
            )
            assert pending_at_write == []
            with service.store.transaction(ctx.account_id, ctx.user_id) as tx:
                tx.update(
                    "bot_task",
                    {"task_id": task_id},
                    {"lease_until": datetime.now(UTC) - timedelta(seconds=60)},
                )
            if correction_stop == "cancel_before_event_publish":
                cancelled = submit(f"/tasks/{task_id}/cancel", {}, request_id="cancel-correction")
                assert cancelled.status_code == 202
                worker.run_once()
                memory.drain_outbox_sync(ctx.account_id)
                assert service.store.get_task(ctx.account_id, task_id)["state"] == "cancelled"
                node = fs.read_node(uri, ctx)
                for record in build_index_records(node):
                    assert (
                        memory._get_shared_vector_index().get_record(record.id).text == record.text
                    )
                return
        corrected_task = execute(active, corrected.json["data"]["task"])
        assert corrected_task["memory_status"] == "searchable"
        node = fs.read_node(uri, ctx)
        assert node.metadata["version"] == prior_version + 1
        assert node.content == "青禾项目改为周四发布。"
        index = memory._get_shared_vector_index()
        for record in build_index_records(node):
            assert index.get_record(record.id).text == record.text
        with service.store.transaction(ctx.account_id, ctx.user_id) as tx:
            assert service.gateway.has_task_reference(tx, corrected_task)
            assert service.gateway.has_file_reference(
                tx, file_info["file"] | {"account_id": ctx.account_id}
            )
            if not shared_spaces:
                [step] = tx.list(
                    "bot_step", {"task_id": corrected_task["task_id"], "kind": "memory_apply"}
                )
                assert (
                    service.gateway.mutate_target(
                        tx, corrected_task, step["input"], step["operation_id"]
                    )
                    == step["output"]
                )

        forgotten = submit(
            "/memory-actions",
            {
                "action": "forget",
                "memory_uri": uri,
            },
            request_id="forget",
        )
        assert forgotten.status_code == 202
        forget_task = execute(active, forgotten.json["data"]["task"])
        assert forget_task["memory_status"] == "searchable"
        assert fs.archived_node_version(uri, ctx) == prior_version + 2
        for record in build_index_records(node):
            assert index.get_record(record.id) is None

        if not shared_spaces:
            accepted = submit(
                "/memory-actions",
                {
                    "action": "forget",
                    "memory_uri": agent_uri,
                },
                request_id="forget-agent",
            )
            assert accepted.status_code == 202
            task_id = accepted.json["data"]["task"]["task_id"]
            claimed = service.store.claim("before-restart")
            assert claimed["task_id"] == task_id
            identity = (ctx.account_id, task_id, "before-restart", claimed["attempt"])
            target = service.gateway.prepare_target(claimed, agent_uri)
            forget = service.store.get(ctx.account_id, "bot_forget", {"task_id": task_id})
            frozen = {
                **claimed["input"],
                "target_version": target["node_version"],
                "target_sha256": target["sha256"],
                "source_refs": [
                    {"file_id": None, "task_id": None, "uri": agent_uri, "quote": example}
                ],
            }
            step = service.store.checkpoint(
                *identity,
                1,
                "memory_apply",
                frozen,
                operation_id=forget["operation_id"],
            )
            with service.store.execution(*identity) as tx:
                service.gateway.mutate_target(tx, claimed, frozen, step["operation_id"])
            cancelled = submit(f"/tasks/{task_id}/cancel", {}, request_id="cancel-forget-agent")
            assert cancelled.status_code == 202
            with service.store.execution(*identity) as tx:
                service.store._terminal(tx, tx.task, "cancelled", None, None)
            assert (
                service.store.get(ctx.account_id, "bot_forget", {"task_id": task_id})["state"]
                == "effective"
            )
            memory.drain_outbox_sync(ctx.account_id)
            assert fs.archived_node_version(agent_uri, agent_ctx) == 2
            for record in build_index_records(agent_node):
                assert index.get_record(record.id) is None
    finally:
        if service is not None:
            service.store.close()
        memory.shutdown()
