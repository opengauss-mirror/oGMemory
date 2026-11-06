"""Request-scoped references identify durable AGFS Outbox events directly."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

from commit.outbox_scope import capture_outbox_events, current_outbox_events
from commit.outbox_store import OutboxStore
from core.models import ContextNode, RequestContext
from fs.agfs_adapter.agfs_context_fs import AGFSContextFS
from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient


class CountingAGFSClient(_AtomicMemoryAGFSClient):
    api_base = "memory://request-outbox"

    def __init__(self):
        super().__init__()
        self.read_paths = []
        self.list_paths = []

    def read(self, path):
        self.read_paths.append(str(path))
        return super().read(path)

    def ls(self, path):
        self.list_paths.append(str(path))
        return super().ls(path)


def request_context(account="outbox-request"):
    return RequestContext(account_id=account, user_id="reader", agent_id="agent",
                          session_id="session-one", trace_id="trace-one")


def memory_node(ctx, key="receiver", text="The receiver was repaired."):
    return ContextNode(
        uri=f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/entities/{key}",
        context_type="MEMORY", category="entity", level=0, owner_space=ctx.user_space_name(),
        abstract=text, overview=text, content=text,
    )


def make_runtime():
    client = CountingAGFSClient()
    fs = AGFSContextFS(client)
    return client, fs, OutboxStore(client, fs)


def test_concurrent_request_captures_keep_their_own_durable_events():
    client, fs, store = make_runtime()
    ready = Barrier(2)

    @capture_outbox_events
    def publish(account):
        ctx = request_context(account)
        node = memory_node(ctx)
        fs.write_node(node, ctx)
        event = store.register_write(fs.read_node(node.uri, ctx), ctx)
        ready.wait()
        references = current_outbox_events(account)
        loaded = store.list_pending_events(references)
        return event, references, loaded

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, ["request-alpha", "request-bravo"]))
    for event, references, loaded in results:
        assert references == [(event.uri, event.event_id)]
        assert [(uri, item.event_id) for uri, item in loaded] == references
    assert current_outbox_events() is None
    assert client.list_paths == []


def test_nested_capture_accumulates_references_and_filters_account():
    _, fs, store = make_runtime()

    def publish(ctx, key):
        node = memory_node(ctx, key)
        fs.write_node(node, ctx)
        return store.register_write(fs.read_node(node.uri, ctx), ctx)

    @capture_outbox_events
    def child():
        event = publish(request_context("child-account"), "child")
        return event, current_outbox_events()

    @capture_outbox_events
    def parent():
        first = publish(request_context(), "first")
        inner, inner_references = child()
        second = publish(request_context("second-account"), "second")
        assert inner_references == [(first.uri, first.event_id), (inner.uri, inner.event_id)]
        assert current_outbox_events() == [(first.uri, first.event_id), (inner.uri, inner.event_id),
                                           (second.uri, second.event_id)]
        assert current_outbox_events(first.uri.split("/")[2]) == [(first.uri, first.event_id)]

    parent()
    assert current_outbox_events() is None


def test_persisted_reference_reads_are_deduplicated_and_survive_store_restart():
    client, fs, store = make_runtime()
    ctx = request_context()
    node = memory_node(ctx)
    fs.write_node(node, ctx)

    @capture_outbox_events
    def publish():
        event = store.register_write(fs.read_node(node.uri, ctx), ctx)
        store.persist_event(event, node.uri)
        return event, current_outbox_events(ctx.account_id)

    event, references = publish()
    assert references == [(node.uri, event.event_id)]
    restarted = OutboxStore(client, AGFSContextFS(client))
    client.read_paths.clear()
    client.list_paths.clear()
    loaded = restarted.list_pending_events(references + references)
    event_path = store._uri_to_agfs_path(node.uri) + ".outbox/" + event.event_id + ".json"
    assert client.read_paths.count(event_path) == 1
    assert client.list_paths == []
    assert len(loaded) == 1
    assert loaded[0][0] == node.uri
    assert loaded[0][1] == replace(event, created_at=event.created_at or "")
