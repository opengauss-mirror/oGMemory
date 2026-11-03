"""Run with OGMEM_TEST_USAGE_DSN against a disposable PostgreSQL database."""

import json
import multiprocessing
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from urllib.request import urlopen
from uuid import uuid4

import pytest


def _service(dsn, dialect="postgresql"):
    from providers.unified_config import OgMemConfig
    from server.memory_service import MemoryService

    return MemoryService(
        config=OgMemConfig(
            storage_backend="sql",
            sql_connection_string=dsn,
            sql_dialect=dialect,
            provider="openai",
            openai_api_key="local-test",
            openai_llm_model="test-llm",
            embedding_provider="openai",
            openai_embedding_model="test-embed",
            opengauss_dimension=2,
        )
    )


def _produce(dsn, account, owner, index, dialect="postgresql"):
    from providers.token_tracker import usage_scope

    service = _service(dsn, dialect)
    with usage_scope(account, owner):
        if index:
            from core.models import OutboxEvent
            from index.outbox_worker import OutboxWorker

            provider = service._get_shared_embedder()
            response = SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 2.0])], usage=SimpleNamespace(total_tokens=7)
            )
            provider._client = SimpleNamespace(
                embeddings=SimpleNamespace(create=lambda **kw: response)
            )
            worker = OutboxWorker(SimpleNamespace(upsert=lambda records: None), provider)
            uri = f"ctx://{account}/agents/helper/memories/patterns/test"
            event = OutboxEvent(
                uuid4().hex,
                "UPSERT_CONTEXT",
                uri,
                {
                    "records": [
                        {
                            "id": "test",
                            "uri": uri,
                            "level": 0,
                            "text": "test",
                            "filters": {"account_id": account, "owner_space": owner},
                        }
                    ]
                },
                "PENDING",
            )
            for _ in range(15):
                with usage_scope("", ""):
                    assert worker.process_event(event).success
        else:
            provider = service.get_llm()
            response = SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=3,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                ),
                model="test-llm",
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
            )
            provider._client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))
            )
            for _ in range(15):
                provider.complete_json("test", {"type": "object"})


def _serve(dsn, account, pipe, dialect="postgresql"):
    from flask import Flask
    from werkzeug.serving import make_server

    from core.models import RequestContext, Role
    from server.memory_viewer import create_memory_viewer_blueprint

    service = _service(dsn, dialect)
    service.get_auth_service = lambda: SimpleNamespace(role_control_active=lambda: True)
    ctx = RequestContext(
        account_id=account,
        user_id="alice",
        agent_id="",
        session_id="test",
        trace_id="test",
        role=Role.MEMBER,
        visible_owner_spaces=("user:alice", "agent:helper"),
    )
    app = Flask(__name__)
    app.register_blueprint(
        create_memory_viewer_blueprint(
            lambda: service, lambda _: ctx, lambda exc: ({"error": str(exc)}, 500)
        )
    )
    server = make_server("127.0.0.1", 0, app)
    pipe.send(server.server_port)
    try:
        while not pipe.poll():
            server.timeout = 0.2
            server.handle_request()
    finally:
        server.server_close()


@pytest.fixture
def ledger():
    dsn = os.environ.get("OGMEM_TEST_USAGE_DSN")
    if not dsn:
        pytest.skip("OGMEM_TEST_USAGE_DSN is required for real PostgreSQL multi-process tests")
    from fs.sql_adapter.model_usage_store import SQLModelUsageStore
    from fs.sql_adapter.pool import SharedConnectionPool

    pool = SharedConnectionPool(dsn)
    store = SQLModelUsageStore(pool)
    account = "usage-test-" + uuid4().hex
    yield dsn, store, account
    conn = pool.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM model_usage_events WHERE account_id = ANY(%s)",
                ([account, account + "-other"],),
            )
        conn.commit()
    finally:
        pool.return_connection(conn)
        pool.close_all()


def test_providers_visible_to_independent_api_and_after_restart(ledger):
    dsn, store, account = ledger
    mp = multiprocessing.get_context("spawn")
    writers = [
        mp.Process(target=_produce, args=(dsn, account, owner, index))
        for owner, index in [
            ("user:alice", False),
            ("user:alice", False),
            ("agent:helper", True),
            ("agent:helper", True),
        ]
    ]
    writers.append(mp.Process(target=_produce, args=(dsn, account + "-other", "user:alice", False)))
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(60)
        assert writer.exitcode == 0
    payloads = [_read_from_api(dsn, account) for _ in range(2)]
    assert payloads[0] == payloads[1]
    metrics = payloads[0]
    assert metrics["cumulative"]["llm"]["calls"] == 30
    assert metrics["cumulative"]["llm"]["input_tokens"] == 300
    assert metrics["cumulative"]["llm"]["output_tokens"] == 90
    assert metrics["cumulative"]["llm"]["cache_read"] == 60
    assert metrics["cumulative"]["embedding"] == {"calls": 30, "total_tokens": 210}
    assert metrics["today"]["total"] == 600
    assert len(metrics["series"]) == 14
    assert store.metrics(1, account, ("user:alice",))["cumulative"]["total_tokens"] == 390
    assert store.metrics(1, account, ("agent:helper",))["cumulative"]["total_tokens"] == 210


def _read_from_api(dsn, account, dialect="postgresql"):
    mp = multiprocessing.get_context("spawn")
    parent, child = mp.Pipe()
    api = mp.Process(target=_serve, args=(dsn, account, child, dialect))
    api.start()
    try:
        assert parent.poll(30)
        port = parent.recv()
        with urlopen(
            f"http://127.0.0.1:{port}/api/v1/memory-viewer/metrics?days=14&accountId={account}-other",
            timeout=10,
        ) as response:
            payload = json.load(response)
    finally:
        parent.send("stop")
        api.join(10)
        if api.is_alive():
            api.terminate()
            api.join()
        parent.close()
        child.close()
    assert api.exitcode == 0
    return payload


def _duplicate(dsn, event):
    from fs.sql_adapter.model_usage_store import SQLModelUsageStore
    from fs.sql_adapter.pool import SharedConnectionPool

    SQLModelUsageStore(SharedConnectionPool(dsn)).record(event)


def test_concurrent_replay_and_utc_history(ledger):
    dsn, store, account = ledger
    event = {
        "id": uuid4().hex,
        "at": datetime.now(UTC) - timedelta(days=2),
        "model": "test-embed",
        "kind": "embedding",
        "allocations": {
            (account, "user:alice"): {"embed_tokens": 20, "embed_calls": 1},
            (account, "agent:helper"): {"embed_tokens": 10, "embed_calls": 1},
        },
    }
    mp = multiprocessing.get_context("spawn")
    processes = [mp.Process(target=_duplicate, args=(dsn, event)) for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    metrics = store.metrics(14, account)
    assert metrics["cumulative"]["embedding"] == {"calls": 1, "total_tokens": 30}
    assert metrics["today"]["total"] == 0
    assert metrics["series"][-3]["embedding_input"] == 30
    assert store.metrics(1, account)["cumulative"]["embedding"]["total_tokens"] == 30


def test_missing_usage_failures_cache_and_distributed_tokens(ledger):
    _, store, account = ledger
    from providers.token_tracker import TokenTracker, distributed_usage_scope, usage_scope

    tracker = TokenTracker("test-model")
    tracker.set_usage_store(store)
    with usage_scope(account, "user:alice"):
        tracker.record_llm(0, 0, usage_known=False)
        tracker.record_failure("llm")
        tracker.record_cache_hit("llm")
        with distributed_usage_scope({(account, "user:alice"): 2, (account, "agent:helper"): 1}):
            tracker.record_embed(31)
    tracker.reset()
    metrics = store.metrics(1, account)["cumulative"]
    assert metrics["llm"]["calls"] == 1
    assert metrics["embedding"] == {"calls": 1, "total_tokens": 31}
    assert (
        metrics["failed_calls"]
        == metrics["missing_usage_calls"]
        == metrics["local_cache_hits"]
        == 1
    )
    assert (
        store.metrics(1, account, ("user:alice",))["cumulative"]["embedding"]["total_tokens"] == 21
    )
    assert (
        store.metrics(1, account, ("agent:helper",))["cumulative"]["embedding"]["total_tokens"]
        == 10
    )


def _index_batch(dsn, account):
    from core.models import IndexRecord, OutboxEvent
    from index.outbox_worker import OutboxWorker

    provider = _service(dsn)._get_shared_embedder()
    provider._client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 2.0]) for _ in kw["input"]],
                usage=SimpleNamespace(total_tokens=9),
            )
        )
    )
    worker = OutboxWorker(SimpleNamespace(upsert=lambda records: None), provider)
    batch = []
    for tenant, space in [
        (account, "user:alice"),
        (account, "agent:helper"),
        (account + "-other", "user:bob"),
    ]:
        uri = f"ctx://{tenant}/resources/test"
        event = OutboxEvent(uuid4().hex, "UPSERT_CONTEXT", uri, {}, "PENDING")
        record = IndexRecord(uuid4().hex, uri, 0, "x", {"account_id": tenant, "owner_space": space})
        batch.append((event, uri, [record]))
    stats = dict(processed=0, succeeded=0, failed=0, moved_to_dlq=0, skipped=0)
    worker._process_upsert_batch(batch, SimpleNamespace(mark_done=lambda *args: None), stats)
    assert stats["succeeded"] == 3


def test_index_batch_aggregates_distinct_calls_and_preserves_resource_owners(ledger):
    dsn, store, account = ledger
    process = multiprocessing.get_context("spawn").Process(target=_index_batch, args=(dsn, account))
    process.start()
    process.join(30)
    assert process.exitcode == 0
    assert store.metrics(1, account)["cumulative"]["embedding"] == {"calls": 1, "total_tokens": 6}
    assert (
        store.metrics(1, account, ("user:alice",))["cumulative"]["embedding"]["total_tokens"] == 3
    )
    assert (
        store.metrics(1, account, ("agent:helper",))["cumulative"]["embedding"]["total_tokens"] == 3
    )
    assert store.metrics(1, account + "-other")["cumulative"]["embedding"] == {
        "calls": 1,
        "total_tokens": 3,
    }


def test_uncertain_commit_retries_same_event_without_counting_twice(ledger, monkeypatch):
    import psycopg2

    _, store, account = ledger
    original = store.pool.get_connection
    disconnected = False

    class Connection:
        def __init__(self, conn):
            self.conn = conn

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def commit(self):
            nonlocal disconnected
            self.conn.commit()
            if not disconnected:
                disconnected = True
                raise psycopg2.OperationalError("lost commit acknowledgement")

    monkeypatch.setattr(store.pool, "get_connection", lambda: Connection(original()))
    event = {
        "id": uuid4().hex,
        "at": datetime.now(UTC),
        "model": "test",
        "kind": "llm",
        "allocations": {(account, "user:alice"): {"input_tokens": 8, "llm_calls": 1}},
    }
    store.record(event)
    monkeypatch.setattr(store.pool, "get_connection", original)
    assert store.metrics(1, account)["cumulative"]["llm"]["calls"] == 1
    assert store.metrics(1, account)["cumulative"]["total_tokens"] == 8


def test_success_without_usage_and_failed_provider_calls(ledger):
    dsn, store, account = ledger
    from providers.token_tracker import usage_scope

    service = _service(dsn)
    provider = service.get_llm()
    response = SimpleNamespace(
        usage=None,
        model="test-llm",
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
    )
    client = SimpleNamespace(create=lambda **kw: response)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=client))
    with usage_scope(account, "user:alice"):
        provider.complete_json("test", {"type": "object"})

        def failed(**kwargs):
            raise RuntimeError("upstream unavailable")

        client.create = failed
        with pytest.raises(RuntimeError):
            provider.complete_json("test", {"type": "object"})
    totals = store.metrics(1, account)["cumulative"]
    assert totals["llm"]["calls"] == 1
    assert totals["missing_usage_calls"] == 1
    assert totals["failed_calls"] == 1
    assert totals["total_tokens"] == 0


def test_metrics_connection_recovers_after_backend_disconnect(ledger, monkeypatch):
    import psycopg2

    from fs.sql_adapter.model_usage_store import SQLModelUsageStore
    from fs.sql_adapter.pool import SharedConnectionPool

    dsn, _, account = ledger
    pool = SharedConnectionPool(dsn, pool_size=1)
    store = SQLModelUsageStore(pool)
    borrow = pool.get_connection

    def disconnect():
        conn = borrow()
        with psycopg2.connect(dsn) as admin:
            with admin.cursor() as cur:
                cur.execute("SELECT pg_terminate_backend(%s)", (conn.get_backend_pid(),))
        return conn

    monkeypatch.setattr(pool, "get_connection", disconnect)
    try:
        with pytest.raises(psycopg2.Error):
            store.metrics(1, account)
        conn = borrow(timeout=0.1)
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone() == (1,)
        pool.return_connection(conn)
    finally:
        pool.close_all()


@pytest.mark.parametrize("contention", ["pool", "table"])
def test_embedding_persistence_respects_deadline(ledger, contention):
    import psycopg2

    from fs.sql_adapter.model_usage_store import SQLModelUsageStore
    from fs.sql_adapter.pool import SharedConnectionPool
    from providers.embedder.openai_embedder import OpenAIEmbedder
    from providers.token_tracker import usage_scope

    dsn, _, account = ledger
    pool = SharedConnectionPool(dsn, pool_size=1)
    store = SQLModelUsageStore(pool)
    provider = OpenAIEmbedder(api_key="test", model="test", dimension=2)
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                usage=SimpleNamespace(total_tokens=7), data=[SimpleNamespace(embedding=[1.0, 2.0])]
            )
        )
    )
    client.with_options = lambda **kw: client
    provider._client = client
    provider.token_tracker.set_usage_store(store)
    if contention == "pool":
        held = pool.get_connection()

        def release():
            pool.return_connection(held)
    else:
        held = psycopg2.connect(dsn)
        with held.cursor() as cur:
            cur.execute("LOCK TABLE model_usage_events IN ACCESS EXCLUSIVE MODE")
        release = held.close
    timer = threading.Timer(0.5, release)
    timer.start()
    try:
        start = time.monotonic()
        with usage_scope(account, "user:alice"), pytest.raises(TimeoutError):
            provider.embed_texts_with_timeout(["test"], timeout_seconds=0.05)
        assert time.monotonic() - start < 0.3
    finally:
        timer.join()
    try:
        with usage_scope(account, "user:alice"):
            assert provider.embed_texts_with_timeout(["test"], timeout_seconds=2) == [[1.0, 2.0]]
        assert store.metrics(1, account)["cumulative"]["embedding"] == {
            "calls": 1,
            "total_tokens": 7,
        }
    finally:
        pool.close_all()


def test_llm_concurrent_initialization_publishes_bound_tracker(ledger, monkeypatch):
    from providers.token_tracker import usage_scope

    dsn, store, account = ledger
    service = _service(dsn)
    binding = threading.Event()
    release = threading.Event()
    requesting = threading.Event()

    def bind():
        binding.set()
        assert release.wait(5)
        return store

    monkeypatch.setattr(service, "_get_model_usage_store", bind)

    def invoke():
        requesting.set()
        llm = service.get_llm()
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
                        model="test",
                        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
                    )
                )
            )
        )
        with usage_scope(account, "user:alice"):
            return llm.complete_json("test", {"type": "object"})

    with ThreadPoolExecutor(2) as workers:
        initializing = workers.submit(service.get_llm)
        assert binding.wait(5)
        caller = workers.submit(invoke)
        assert requesting.wait(5)
        time.sleep(0.05)
        release.set()
        initializing.result(timeout=5)
        caller.result(timeout=5)
    assert store.metrics(1, account)["cumulative"]["llm"]["calls"] == 1


def test_index_cached_batch_attributes_each_input(ledger):
    from core.models import IndexRecord, OutboxEvent
    from index.outbox_worker import OutboxWorker
    from providers.embedder.openai_embedder import CachedOpenAIEmbedder

    _, store, account = ledger
    provider = CachedOpenAIEmbedder(api_key="test", model="test", dimension=2)
    provider._cache["aaaa"] = [1.0, 2.0]
    provider._client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                usage=SimpleNamespace(total_tokens=8),
                data=[SimpleNamespace(embedding=[1.0, 2.0]) for _ in kw["input"]],
            )
        )
    )
    provider.token_tracker.set_usage_store(store)
    worker = OutboxWorker(SimpleNamespace(upsert=lambda records: None), provider)
    batch = []
    for tenant, text in [(account, "aaaa"), (account + "-other", "bbbb")]:
        uri = f"ctx://{tenant}/resources/test"
        event = OutboxEvent(uuid4().hex, "UPSERT_CONTEXT", uri, {}, "PENDING")
        record = IndexRecord(
            uuid4().hex, uri, 0, text, {"account_id": tenant, "owner_space": "user:alice"}
        )
        batch.append((event, uri, [record]))
    stats = dict(processed=0, succeeded=0, failed=0, moved_to_dlq=0, skipped=0)
    worker._process_upsert_batch(batch, SimpleNamespace(mark_done=lambda *args: None), stats)
    assert stats["succeeded"] == 2
    cached = store.metrics(1, account)["cumulative"]
    fetched = store.metrics(1, account + "-other")["cumulative"]
    assert (cached["local_cache_hits"], cached["embedding"]) == (1, {"calls": 0, "total_tokens": 0})
    assert (fetched["local_cache_hits"], fetched["embedding"]) == (
        0,
        {"calls": 1, "total_tokens": 8},
    )


@pytest.mark.parametrize("kind", ["multimodal", "volcengine"])
@pytest.mark.parametrize(
    "usage,expected,missing",
    [
        ({}, 0, 1),
        ({"total_tokens": None}, 0, 1),
        ({"total_tokens": 0}, 0, 0),
        ({"prompt_tokens": 7}, 7, 0),
        ({"total_tokens": None, "prompt_tokens": 7}, 7, 0),
    ],
)
def test_embedding_usage_fields_and_per_input_owners(
    ledger, monkeypatch, kind, usage, expected, missing
):
    import httpx

    from providers.embedder.openai_embedder import OpenAIEmbedder
    from providers.embedder.volcengine_embedder import VolcengineEmbedder
    from providers.token_tracker import TokenTracker, distributed_usage_scope

    _, store, account = ledger
    if kind == "multimodal":
        provider = OpenAIEmbedder(
            api_key="test", base_url="http://test/v1", model="test", dimension=2, multimodal=True
        )
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, json={"usage": usage, "data": {"embedding": [1.0, 2.0]}}
                )
            )
        )
        monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    else:
        provider = object.__new__(VolcengineEmbedder)
        provider._model_name = "test"
        provider._dimension = 2
        provider.token_tracker = TokenTracker("test")
        provider._client = SimpleNamespace(
            multimodal_embeddings=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    usage=usage, data=SimpleNamespace(embedding=[1.0, 2.0])
                )
            )
        )
    provider.token_tracker.set_usage_store(store)
    inputs = [((tenant, "user:alice"), 1) for tenant in (account, account + "-other")]
    with distributed_usage_scope(dict(inputs), inputs=inputs):
        assert provider.embed_texts(["a", "b"]) == [[1.0, 2.0], [1.0, 2.0]]
    for tenant in (account, account + "-other"):
        totals = store.metrics(1, tenant)["cumulative"]
        assert totals["embedding"] == {"calls": 1, "total_tokens": expected}
        assert totals["missing_usage_calls"] == missing


def test_database_handshake_deadline_releases_connection_capacity(ledger):
    from fs.sql_adapter.pool import SharedConnectionPool

    dsn, _, _ = ledger
    release = threading.Event()
    accepted = threading.Event()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(5)
        port = listener.getsockname()[1]

        def serve():
            conn, _ = listener.accept()
            with conn:
                accepted.set()
                release.wait(5)

        thread = threading.Thread(target=serve)
        thread.start()
        pool = SharedConnectionPool(f"host=127.0.0.1 port={port} dbname=test", pool_size=1)
        try:
            start = time.monotonic()
            with pytest.raises(TimeoutError), pool.deadline_connection(start + 0.1):
                pass
            assert time.monotonic() - start < 0.4
            assert accepted.wait(1)
        finally:
            release.set()
            thread.join(5)
    pool._connection_string = dsn
    with pool.deadline_connection(time.monotonic() + 2) as conn:
        assert conn.closed == 0
    pool.close_all()


def test_deadline_commit_retry_preserves_event_identity(ledger, monkeypatch):
    import psycopg2

    from fs.sql_adapter import model_usage_store

    _, store, account = ledger
    original = model_usage_store.wait_for_postgres
    retried = False

    def lost_ack(conn, deadline):
        nonlocal retried
        original(conn, deadline)
        if (
            conn.get_transaction_status() == psycopg2.extensions.TRANSACTION_STATUS_IDLE
            and not retried
        ):
            retried = True
            raise psycopg2.OperationalError("lost commit acknowledgement")

    monkeypatch.setattr(model_usage_store, "wait_for_postgres", lost_ack)
    event = {
        "id": uuid4().hex,
        "at": datetime.now(UTC),
        "model": "test",
        "kind": "embedding",
        "allocations": {(account, "user:alice"): {"embed_tokens": 8, "embed_calls": 1}},
    }
    store.record(event, deadline=time.monotonic() + 2)
    assert retried
    assert store.metrics(1, account)["cumulative"]["embedding"] == {"calls": 1, "total_tokens": 8}


@pytest.mark.parametrize("kind", ["embedding", "llm"])
def test_cache_hit_survives_eviction_during_persistence(ledger, kind):
    from providers.embedder.openai_embedder import CachedOpenAIEmbedder
    from providers.llm.openai_llm import CachedOpenAILLM
    from providers.token_tracker import usage_scope

    _, store, account = ledger
    entered, release = threading.Event(), threading.Event()
    if kind == "embedding":
        provider = CachedOpenAIEmbedder(api_key="test", dimension=2, cache_max_size=1)
        provider._client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    usage=SimpleNamespace(total_tokens=2),
                    data=[SimpleNamespace(embedding=[1.0, 2.0]) for _ in kw["input"]],
                )
            )
        )

        def call(text):
            return provider.embed_texts([text])
    else:
        provider = CachedOpenAILLM(api_key="test", model="test", cache_max_size=1)
        provider._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: SimpleNamespace(
                        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
                        model="test",
                        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'), finish_reason="stop")],
                    )
                )
            )
        )

        def call(text):
            return provider.complete_json(text, {"type": "object"})

    def invoke(text):
        with usage_scope(account, "user:alice"):
            return call(text)

    expected = invoke("a")

    def record(event):
        if event["kind"].endswith("_cache"):
            entered.set()
            assert release.wait(5)
        store.record(event)

    provider.token_tracker.set_usage_store(SimpleNamespace(record=record))
    with ThreadPoolExecutor(2) as workers:
        cached = workers.submit(invoke, "a")
        try:
            assert entered.wait(5)
            workers.submit(invoke, "b").result(timeout=5)
        finally:
            release.set()
        assert cached.result(timeout=5) == expected
    assert store.metrics(1, account)["cumulative"]["local_cache_hits"] == 1


@pytest.mark.parametrize("transport", ["http", "mcp", "typed"])
def test_candidate_code_search_persists_request_owner(ledger, monkeypatch, tmp_path, transport):
    import server.app as app_module
    import server.mcp_tools as mcp_tools
    import server.memory_service as service_module
    from core.models import RequestContext
    from retrieval.code_retrieval import ActiveCodeSearchRequest
    from service.code_workspace_sync import ResolvedCodeWorkspace

    dsn, store, account = ledger
    service = _service(dsn)
    service._cfg.code_retrieval_enabled = True
    service._cfg.code_retrieval_mode = "candidate_local"
    source = tmp_path / "target.py"
    source.write_text("def parse_target(value):\n    return value\n", encoding="utf-8")
    workspace = ResolvedCodeWorkspace(
        root_id="root-1", root_slug="repo", root_fingerprint="f" * 64,
        physical_root=tmp_path, max_files=300, language_hints={},
    )
    monkeypatch.setattr(service, "get_control_plane_store", lambda: None)
    monkeypatch.setattr(
        service_module, "CodeWorkspaceRegistryResolver",
        lambda *args, **kwargs: SimpleNamespace(resolve=lambda **kw: workspace),
    )
    monkeypatch.setattr("retrieval.bm25_index._ensure_nltk", lambda: None)
    monkeypatch.setattr("retrieval.bm25_index._tokenize", lambda text: text.lower().split())
    provider = service._get_candidate_code_embedder()
    inputs = []

    def create(**kwargs):
        inputs.append(kwargs["input"])
        return SimpleNamespace(
            usage=SimpleNamespace(total_tokens=7),
            data=[SimpleNamespace(embedding=[1.0, 2.0]) for _ in kwargs["input"]],
        )

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    client.with_options = lambda **kwargs: client
    provider._client = client
    monkeypatch.setattr(app_module, "_service", service)
    for tenant, agent in [(account, ""), (account + "-other", "helper")]:
        ctx = RequestContext(tenant, "alice", agent, "test", uuid4().hex)
        monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: ctx)
        monkeypatch.setattr(mcp_tools, "build_mcp_context", lambda *args, **kwargs: ctx)
        before = len(inputs)
        if transport == "http":
            response = app_module.app.test_client().post(
                "/api/v1/resources/code/search", json={"query": "parse_target"},
            )
            assert response.status_code == 200
            assert response.json["evidence"]
        elif transport == "mcp":
            assert "parse_target" in mcp_tools.search_code(service, query="parse_target")
        else:
            assert service.search_active(ActiveCodeSearchRequest("parse_target"), ctx).evidence
        calls = len(inputs) - before
        assert calls == 2
        owner = ctx.agent_space_name() if agent else ctx.user_space_name()
        totals = store.metrics(1, tenant, (owner,))["cumulative"]["embedding"]
        assert totals == {"calls": calls, "total_tokens": calls * 7}
    assert store.metrics(1, account)["cumulative"]["embedding"] == {
        "calls": 2, "total_tokens": 14,
    }
    service._get_shared_sql_pool().close_all()


def test_database_dns_obeys_deadline(tmp_path):
    import shutil
    import subprocess
    import sys

    if not sys.platform.startswith("linux") or not shutil.which("cc"):
        pytest.skip("Linux and a C compiler are required for the libc DNS delay fixture")
    source = tmp_path / "slow_dns.c"
    library = tmp_path / "slow_dns.so"
    source.write_text('''
#define _GNU_SOURCE
#include <dlfcn.h>
#include <netdb.h>
#include <string.h>
#include <unistd.h>
int getaddrinfo(const char *node, const char *service,
                const struct addrinfo *hints, struct addrinfo **result) {
    int (*resolve)(const char *, const char *, const struct addrinfo *, struct addrinfo **);
    resolve = dlsym(RTLD_NEXT, "getaddrinfo");
    if (node && strcmp(node, "usage-timeout.invalid") == 0) {
        usleep(1000000);
        node = "127.0.0.1";
    }
    return resolve(node, service, hints, result);
}
''', encoding="utf-8")
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-ldl", "-o", str(library)], check=True)
    script = '''
import time
from fs.sql_adapter.pool import SharedConnectionPool
pool = SharedConnectionPool("host=usage-timeout.invalid dbname=test", pool_size=1)
start = time.monotonic()
try:
    with pool.deadline_connection(start + 0.05):
        pass
except Exception:
    elapsed = time.monotonic() - start
    print(f"DNS deadline elapsed={elapsed:.3f}s", flush=True)
    assert elapsed < 0.4, elapsed
else:
    raise AssertionError("DNS delay must exhaust the deadline")
assert pool._semaphore.acquire(blocking=False)
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "LD_PRELOAD": str(library)},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_deadline_connection_supports_high_file_descriptors(ledger):
    _, store, account = ledger
    store.pool.close_all()
    files = []
    try:
        while not files or files[-1].fileno() < 1100:
            files.append(open(os.devnull, "rb"))
        event = {
            "id": uuid4().hex, "at": datetime.now(UTC), "model": "test",
            "kind": "embedding",
            "allocations": {(account, "user:alice"): {"embed_tokens": 8, "embed_calls": 1}},
        }
        store.record(event, deadline=time.monotonic() + 2)
        assert store.metrics(1, account)["cumulative"]["embedding"]["calls"] == 1
    finally:
        for file in files:
            file.close()


def test_cached_embedding_batch_uses_one_database_transaction(ledger, monkeypatch):
    from contextlib import contextmanager

    from providers.embedder.openai_embedder import CachedOpenAIEmbedder
    from providers.token_tracker import distributed_usage_scope

    _, store, account = ledger
    provider = CachedOpenAIEmbedder(api_key="test", dimension=2)
    texts = [str(i) for i in range(128)]
    provider._cache.update(dict.fromkeys(texts, [1.0, 2.0]))
    provider.token_tracker.set_usage_store(store)
    original = store.pool.deadline_connection
    connections = []

    @contextmanager
    def counted(deadline):
        with original(deadline) as conn:
            connections.append(conn)
            yield conn

    monkeypatch.setattr(store.pool, "deadline_connection", counted)
    inputs = [((account if i % 2 else account + "-other", "user:alice"), 1)
              for i in range(len(texts))]
    with distributed_usage_scope(dict(inputs), inputs=inputs):
        assert provider.embed_texts(texts, timeout_seconds=5) == [[1.0, 2.0]] * len(texts)
    assert len(connections) == 1
    for tenant in (account, account + "-other"):
        assert store.metrics(1, tenant)["cumulative"]["local_cache_hits"] == 64


@pytest.mark.parametrize("bounded", [False, True])
def test_usage_batch_replay_preserves_individual_events(ledger, bounded):
    _, store, account = ledger
    events = [{
        "id": uuid4().hex, "at": datetime.now(UTC), "model": "test",
        "kind": "embedding_cache",
        "allocations": {(account, "user:alice"): {"local_cache_hits": 1}},
    } for _ in range(8)]
    for _ in range(2):
        store.record_many(events, deadline=time.monotonic() + 2 if bounded else None)
    assert store.metrics(1, account)["cumulative"]["local_cache_hits"] == 8


def test_opengauss_provider_processes_and_deadline_replay():
    from fs.sql_adapter.model_usage_store import SQLModelUsageStore
    from fs.sql_adapter.pool import SharedConnectionPool

    dsn = os.environ.get("OGMEM_OPENGAUSS_SQL_DSN")
    if not dsn:
        pytest.skip("OGMEM_OPENGAUSS_SQL_DSN is required")
    pool = SharedConnectionPool(dsn, dialect="opengauss")
    store = SQLModelUsageStore(pool)
    account = "usage-test-" + uuid4().hex
    mp = multiprocessing.get_context("spawn")
    writers = [
        mp.Process(target=_produce, args=(dsn, account, owner, index, "opengauss"))
        for owner, index in [("user:alice", False), ("agent:helper", True)]
    ]
    try:
        for writer in writers:
            writer.start()
        for writer in writers:
            writer.join(30)
            assert writer.exitcode == 0
        payloads = [_read_from_api(dsn, account, "opengauss") for _ in range(2)]
        assert payloads[0] == payloads[1]
        assert payloads[0]["cumulative"]["llm"]["calls"] == 15
        assert payloads[0]["cumulative"]["embedding"] == {"calls": 15, "total_tokens": 105}
        event = {
            "id": uuid4().hex,
            "at": datetime.now(UTC),
            "model": "test",
            "kind": "embedding",
            "allocations": {(account, "agent:helper"): {"embed_tokens": 8, "embed_calls": 1}},
        }
        for _ in range(2):
            store.record(event, deadline=time.monotonic() + 2)
        assert store.metrics(1, account)["cumulative"]["embedding"] == {
            "calls": 16,
            "total_tokens": 113,
        }
        events = [dict(event, id=uuid4().hex) for _ in range(4)]
        pool._connection_string = dsn.replace('host=127.0.0.1', 'host=localhost')
        for _ in range(2):
            store.record_many(events, deadline=time.monotonic() + 2)
        assert store.metrics(1, account)["cumulative"]["embedding"] == {
            "calls": 20, "total_tokens": 145,
        }
    finally:
        for writer in writers:
            if writer.is_alive():
                writer.terminate()
                writer.join()
        conn = pool.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM model_usage_events WHERE account_id=%s", (account,))
            conn.commit()
        finally:
            pool.return_connection(conn)
            pool.close_all()
