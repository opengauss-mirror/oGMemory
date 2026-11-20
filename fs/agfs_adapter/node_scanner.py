from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from urllib.parse import unquote

from core.errors import NodeBrokenError, NodeNotFoundError
from core.memory_names import is_agfs_search_directory
from core.models import ContextNode, RequestContext
from core.operation_deadline import check_deadline
from retrieval.identity import canonical_uri
from retrieval.owner_scope import canonical_owner_space

from .agfs_context_fs import uri_to_path


_UNSAFE_COMPONENT = re.compile(r"[/\\\x00-\x1f\x7f]")


def safe_agfs_uri(uri: str) -> bool:
    parts = [unquote(part) for part in uri.removeprefix("ctx://").split("/")]
    return uri.startswith("ctx://") and all(
        part and part not in {".", ".."} and not _UNSAFE_COMPONENT.search(part) for part in parts
    )


def published_agfs_uri(node: ContextNode, listed_uri: str) -> str | None:
    uri = canonical_uri(node.uri)
    if safe_agfs_uri(uri) and unquote(uri) == unquote(listed_uri):
        return uri
    return None


class AGFSCanonicalNodeScanner:
    """Stream canonical AGFS nodes within requested backfill owner scopes."""

    def __init__(self, fs, *, outbox=None) -> None:
        self._fs = fs
        self._outbox = outbox

    def projection_is_quiescent(self, *, account_id: str) -> bool | None:
        return self._outbox.projection_is_quiescent(account_id=account_id) if self._outbox is not None else True

    def prepare_lexical_projection(self, *, account_id: str) -> bool:
        return self._outbox.prepare_lexical_projection(account_id=account_id) if self._outbox is not None else True

    def scan_active_nodes(
        self, *, account_id: str, owner_spaces: Sequence[str], page_size: int,
    ) -> Iterator[ContextNode]:
        if page_size < 1:
            raise ValueError("page_size must be positive")
        account = f"ctx://{account_id}"
        for owner_space in dict.fromkeys(owner_spaces):
            kind, _, owner = canonical_owner_space(owner_space).partition(":")
            user_id = owner if kind == "user" else ""
            agent_id = owner if kind == "agent" else ""
            session_id = owner if kind == "session" else ""
            ctx = RequestContext(account_id, user_id, agent_id, session_id, "lexical-backfill")
            if user_id:
                roots = [f"{account}/users/{user_id}/memories"]
            elif agent_id:
                roots = [f"{account}/agents/{agent_id}/memories", f"{account}/agents/{agent_id}/skills"]
            elif session_id:
                roots = [f"{account}/sessions/{session_id}/history"]
            elif kind == "resource" and owner == account_id:
                roots = [f"{account}/resources"]
            else:
                continue
            directories = [iter(roots)]
            while directories:
                check_deadline()
                uri = next(directories[-1], None)
                if uri is None:
                    directories.pop()
                    continue
                if not safe_agfs_uri(uri):
                    continue
                try:
                    uri_to_path(uri)
                except ValueError:
                    continue
                if is_agfs_search_directory(uri):
                    directories.append(iter(self._fs.list_children_including_inactive(uri, ctx)))
                try:
                    node = self._fs.read_node(uri, ctx)
                except (NodeNotFoundError, NodeBrokenError):
                    continue
                if node.owner_space == owner_space and published_agfs_uri(node, uri):
                    yield node
