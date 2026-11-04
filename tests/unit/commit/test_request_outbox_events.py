"""Request-scoped references identify durable AGFS Outbox events directly."""


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

