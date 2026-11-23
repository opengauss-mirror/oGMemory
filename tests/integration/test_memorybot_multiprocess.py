"""Independent worker and HTTP processes sharing the integration database."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import requests

from memorybot.config import BotConfig
from memorybot.models import jsonable
from memorybot.store import BotStore
from tests.integration.test_memorybot_runtime import runtime as runtime
from tests.integration.test_memorybot_runtime import task

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def _report(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False), encoding="utf-8")


def _scoped_store(account):
    store = BotStore(
        os.environ["MEMORYBOT_TEST_DSN"],
        dialect=os.environ.get("MEMORYBOT_TEST_DIALECT", "postgresql"),
    )
    transaction = store.transaction
    store.transaction = lambda account_id=None, user_id=None: transaction(
        account_id or account, user_id
    )
    store.claim = partial(store.claim, account_id=account)
    return store


def _worker_process(mode, account, report):
    import threading

    from core.models import RequestContext, Role
    from memorybot.files import AGFSFileStore
    from memorybot.gateway import MemoryGateway
    from memorybot.runtime import BotWorker
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService
    from tests.integration.test_memorybot_runtime import DeterministicLLM

    store = _scoped_store(account)
    dsn = os.environ["MEMORYBOT_TEST_DSN"]
    dialect = os.environ.get("MEMORYBOT_TEST_DIALECT", "postgresql")
    memory = MemoryService(
        OgMemConfig(
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
    )
    llm = DeterministicLLM()
    memory._llm = llm
    gateway = MemoryGateway(
        memory,
        lambda row: RequestContext(
            account, "user", "memorybot", "s_qinghe", row["task_id"], role=Role.ADMIN
        ),
    )
    retrievals = []
    retrieve = gateway.retrieve

    def recorded_retrieve(row, *args, **kwargs):
        retrievals.append(row["task_id"])
        return retrieve(row, *args, **kwargs)

    gateway.retrieve = recorded_retrieve
    if mode == "worker-first":

        def pause_at_model(messages, on_delta):
            _report(report, {"pid": os.getpid(), "retrievals": retrievals})
            threading.Event().wait(30)
            return [], "已恢复"

        llm.stream_hook = pause_at_model
    files = AGFSFileStore(os.environ["MEMORYBOT_TEST_AGFS_URL"], "/memorybot")
    worker = BotWorker(
        store,
        gateway,
        files,
        config=BotConfig(lease_seconds=4, renew_seconds=1, poll_seconds=0.1),
    )
    worker.maintenance.next_check = time.monotonic() + 60
    started = time.monotonic()
    try:
        while time.monotonic() - started < 30:
            worker.run_once()
            rows = store.list(account, "bot_task")
            if rows and all(row["state"] == "succeeded" for row in rows):
                _report(
                    report,
                    {
                        "pid": os.getpid(),
                        "retrievals": retrievals,
                        "elapsed_seconds": time.monotonic() - started,
                        "tasks": [
                            {
                                key: row[key]
                                for key in ("task_id", "kind", "seq", "attempt", "state")
                            }
                            for row in rows
                        ],
                    },
                )
                return
            time.sleep(0.1)
        raise RuntimeError("Worker completion deadline reached")
    finally:
        memory.shutdown()
        memory._sql_pool.close_all()
        store.close()


def _child(mode, account, report, start="0"):
    if mode.startswith("worker-"):
        return _worker_process(mode, account, report)
    store = _scoped_store(account)
    if mode == "trigger":
        while time.time() < float(start):
            time.sleep(0.02)
        created = store.trigger_due()
        _report(report, {"pid": os.getpid(), "task_ids": [row["task_id"] for row in created]})
        store.close()
        return
    from flask import Flask
    from werkzeug.serving import make_server

    from memorybot.http import create_blueprint
    from memorybot.service import BotService
    from tests.integration.test_memorybot_service import IdentityProvider, MemoryGateway

    service = BotService(
        store, IdentityProvider(account, "user"), None, BotConfig(), gateway=MemoryGateway()
    )
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(lambda: service))
    server = make_server("127.0.0.1", 0, app, threaded=True)
    _report(report, {"pid": os.getpid(), "port": server.server_port})
    try:
        server.serve_forever()
    finally:
        server.server_close()
        store.close()


@pytest.fixture
def processes(tmp_path):
    children = []

    def start(mode, account, *, at=0):
        index = len(children)
        report = tmp_path / f"{index}-{mode}.json"
        log = tmp_path / f"{index}-{mode}.log"
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tests.integration.test_memorybot_multiprocess",
                    mode,
                    account,
                    str(report),
                    str(at),
                ],
                cwd=ROOT,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        children.append(process)
        return process, report, log

    yield start
    for process in children:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)


def _ready(child, *, timeout=30):
    process, report, log = child
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if report.is_file():
            try:
                return json.loads(report.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        if process.poll() is not None:
            pytest.fail(f"Process exited {process.returncode}: {log.read_text(encoding='utf-8')}")
        time.sleep(0.05)
    pytest.fail(f"Process report deadline: {log.read_text(encoding='utf-8')}")


def test_worker_lease_resume_session_order_and_http_handoff(runtime, processes, record_property):
    store, account = runtime.store, runtime.ctx.account_id
    first = task(runtime, "chat", {"text": "整理本周进展"})
    second = task(runtime, "chat", {"text": "继续说明下一步"})
    original = processes("worker-first", account)
    first_report = _ready(original)
    assert first_report["retrievals"] == [first["task_id"]]
    saved = store.get(account, "bot_step", {"task_id": first["task_id"], "step_no": 2})
    assert saved["state"] == "succeeded"
    assert store.get_task(account, second["task_id"])["state"] == "queued"
    original[0].terminate()
    original[0].wait(timeout=10)
    lease_until = store.get_task(account, first["task_id"])["lease_until"]
    recovered = processes("worker-resume", account)
    recovered_report = _ready(recovered)
    recovered[0].wait(timeout=10)
    assert recovered[0].returncode == 0
    assert recovered_report["pid"] != first_report["pid"]
    assert recovered_report["retrievals"] == [second["task_id"]]
    first_done = store.get_task(account, first["task_id"])
    second_done = store.get_task(account, second["task_id"])
    assert (first_done["attempt"], second_done["attempt"]) == (2, 1)
    assert first_done["finished_at"] >= lease_until
    second_started = next(
        event["created_at"]
        for event in store.events(account, second["task_id"])
        if event["type"] == "task.updated" and event["payload"].get("state") == "running"
    )
    assert first_done["finished_at"] <= second_started
    resumed = store.get(account, "bot_step", {"task_id": first["task_id"], "step_no": 2})
    assert resumed["operation_id"] == saved["operation_id"]
    assert resumed["output"] == saved["output"]
    assert resumed["updated_at"] == saved["updated_at"]

    servers = [processes("http", account), processes("http", account)]
    addresses = [_ready(server) for server in servers]
    assert addresses[0]["pid"] != addresses[1]["pid"]
    path = f"/api/v1/bot/tasks/{first['task_id']}"
    headers = {"X-API-Key": "integration-user-key"}
    urls = [f"http://127.0.0.1:{address['port']}{path}" for address in addresses]
    with requests.get(urls[0] + "/events", headers=headers, stream=True, timeout=10) as response:
        response.raise_for_status()
        first_event = next(line for line in response.iter_lines() if line.startswith(b"id: "))
    cursor = first_event[4:].decode()
    with requests.get(
        urls[1] + "/events", headers={**headers, "Last-Event-ID": cursor}, timeout=10
    ) as response:
        response.raise_for_status()
        replay = [
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        ]
    expected = store.events(account, first["task_id"], int(cursor.rsplit(":", 1)[1]), limit=200)
    assert [row["event_seq"] for row in replay] == [row["event_seq"] for row in expected]
    assert replay[-1]["event_seq"] == first_done["next_event_seq"] - 1
    snapshots = [requests.get(url, headers=headers, timeout=10).json()["data"] for url in urls]
    assert snapshots[0]["steps"] == snapshots[1]["steps"]
    assert snapshots[0]["result"]["text"] == first_done["result"]["text"]
    record_property("worker_pids", [first_report["pid"], recovered_report["pid"]])
    record_property("worker_resume_seconds", round(recovered_report["elapsed_seconds"], 3))
    record_property("http_pids", [address["pid"] for address in addresses])
    record_property("replayed_events", len(replay))


def test_two_schedulers_trigger_one_occurrence(runtime, processes, record_property):
    store, ctx = runtime.store, runtime.ctx
    with store.transaction(ctx.account_id, ctx.user_id) as tx:
        due = tx.now() + timedelta(seconds=1)
        schedule = tx.insert(
            "bot_schedule",
            {
                "schedule_id": "once",
                "session_id": ctx.session_id,
                "name": "汇总进展",
                "task_input": {"text": "生成进展汇总"},
                "accepted_authz": {"agent_id": ctx.agent_id},
                "rule": {"type": "once", "at": due.isoformat()},
                "timezone": "Asia/Shanghai",
                "next_run_at": due,
            },
        )
    at = time.time() + 3
    children = [processes("trigger", ctx.account_id, at=at) for _ in range(2)]
    reports = [_ready(child) for child in children]
    for child in children:
        child[0].wait(timeout=10)
        assert child[0].returncode == 0
    assert reports[0]["pid"] != reports[1]["pid"]
    ids = [task_id for report in reports for task_id in report["task_ids"]]
    assert len(ids) == 1
    rows = store.list(ctx.account_id, "bot_task", {"schedule_id": schedule["schedule_id"]})
    assert [row["task_id"] for row in rows] == ids
    assert rows[0]["scheduled_at"] == due
    assert (
        store.get(ctx.account_id, "bot_schedule", {"schedule_id": "once"})["state"] == "completed"
    )
    record_property("scheduler_pids", [report["pid"] for report in reports])
    record_property("created_occurrences", len(rows))


if __name__ == "__main__":
    _child(*sys.argv[1:])
