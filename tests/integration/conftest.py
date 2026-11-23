"""Shared fixtures for integration tests."""

import pytest

from core.errors import NodeNotFoundError
from core.models import ContextNode, RequestContext


class InMemoryContextFS:
    """In-memory ContextFS for integration tests.

    Simulates the full ContextFS protocol without requiring AGFS or database.
    Raises NodeNotFoundError to match real adapter behavior.

    # TODO: Replace with real AGFS/SQL ContextFS once persistence layer
    # stabilizes. This mock does not test actual I/O, concurrency, or
    # crash-recovery semantics.
    """

    _migration_reads = ("exists", "list_children", "read_node", "read_nodes")

    def __init__(self):
        self._store: dict[str, ContextNode] = {}

    def write_node(self, node: ContextNode, ctx: RequestContext) -> None:
        self._store[node.uri] = node

    def read_node(self, uri: str, ctx: RequestContext) -> ContextNode:
        if uri not in self._store:
            raise NodeNotFoundError(uri)
        return self._store[uri]

    def delete_node(self, uri: str, ctx: RequestContext) -> None:
        self._store.pop(uri, None)

    def archive_node(self, uri: str, ctx: RequestContext) -> None:
        if uri in self._store:
            self._store[uri].metadata["status"] = "ARCHIVED"

    def move_node(self, from_uri: str, to_uri: str, ctx: RequestContext) -> None:
        self._store[to_uri] = self._store.pop(from_uri)

    def list_children(self, uri: str, ctx: RequestContext) -> list[str]:
        prefix = uri if uri.endswith("/") else f"{uri}/"
        return sorted(u for u in self._store if u.startswith(prefix))

    def exists(self, uri: str, ctx: RequestContext) -> bool:
        return uri in self._store

    @property
    def stored_uris(self) -> list[str]:
        return sorted(self._store.keys())


@pytest.fixture
def memory_fs():
    """Provide an InMemoryContextFS for integration tests."""
    return InMemoryContextFS()
