"""AGFS implementation of RelationStore protocol.

Reads and writes relation edges to .relations.json files
stored alongside ContextNode directories in AGFS.
"""

import json
from contextlib import nullcontext
from dataclasses import asdict, replace

from pyagfs import AGFSClient
from pyagfs.exceptions import AGFSClientError

from core.models import RequestContext, RelationEdge
from core.interfaces import RelationStore
from core.errors import AccessDeniedError, NodeNotFoundError


class AGFSRelationStore(RelationStore):
    """AGFS-based relation store using .relations.json files.

    Relations are stored in the node directory:
    {mount_prefix}/accounts/{...}/node/.relations.json

    Format: JSON array of RelationEdge objects.
    """

    def __init__(self, client: AGFSClient, mount_prefix: str = "/local", *, fs=None):
        """Initialize AGFSRelationStore.

        Args:
            client: AGFS Python SDK client
            mount_prefix: AGFS mount point prefix (default: "/local")
                         Must match the mount_prefix used by AGFSContextFS
            fs: Canonical context store for coordinated Dreaming relation writes.
        """
        self._client = client
        self._mount_prefix = mount_prefix.rstrip("/")
        self._fs = fs

    def get_edges(self, uri: str, ctx: RequestContext) -> list[RelationEdge]:
        """Get all outgoing edges from a node.

        Args:
            uri: ContextEngine URI of the source node
            ctx: RequestContext for access control

        Returns:
            List of RelationEdge objects. Empty list if node has no relations
            or doesn't exist.

        Raises:
            AccessDeniedError: If URI account doesn't match context account
        """
        edges = self._read_edges(uri, ctx)
        if getattr(self._fs, "dreaming_enabled", False) is not True:
            return edges
        resolved = {}
        for edge in edges:
            try:
                target = self._fs.resolve_dreaming_relation_target(edge.to_uri, ctx, edge.target_created_at)
            except (AccessDeniedError, NodeNotFoundError):
                continue
            resolved[(target, edge.relation_type)] = replace(edge, to_uri=target)
        return list(resolved.values())

    def _read_edges(self, uri, ctx):
        from fs.agfs_adapter.agfs_context_fs import parse_uri
        try:
            components = parse_uri(uri)
            if components['account'] != ctx.account_id:
                raise AccessDeniedError(
                    uri, ctx.account_id,
                    f"URI belongs to account '{components['account']}'"
                )
        except ValueError:
            raise AccessDeniedError(uri, ctx.account_id, "Invalid URI format")

        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        node_path = uri_to_path(uri)
        relations_path = f"{self._mount_prefix}{node_path}.relations.json"

        try:
            content = self._client.read(relations_path)
            if not content:
                return []

            data = json.loads(content.decode('utf-8'))
            return [
                RelationEdge(
                    from_uri=edge["from_uri"],
                    to_uri=edge["to_uri"],
                    relation_type=edge["relation_type"],
                    weight=edge["weight"],
                    reason=edge["reason"],
                    target_created_at=edge.get("target_created_at"),
                )
                for edge in data
            ]

        except (AGFSClientError, json.JSONDecodeError, KeyError):
            # Return empty list if file doesn't exist or is invalid
            return []

    def upsert_edges(self, edges: list[RelationEdge], ctx: RequestContext) -> None:
        """Add or update relation edges.

        Idempotent: same edge (from_uri, to_uri, relation_type) updates
        the existing edge.

        Args:
            edges: List of RelationEdge objects to upsert
            ctx: RequestContext for access control

        Raises:
            AccessDeniedError: If any edge's source URI doesn't match context account
        """
        if not edges:
            return

        # SECURITY: Validate account access for all edges
        from fs.agfs_adapter.agfs_context_fs import parse_uri
        for edge in edges:
            try:
                components = parse_uri(edge.from_uri)
                if components['account'] != ctx.account_id:
                    raise AccessDeniedError(
                        edge.from_uri, ctx.account_id,
                        f"Edge source URI belongs to account '{components['account']}'"
                    )
            except ValueError:
                raise AccessDeniedError(edge.from_uri, ctx.account_id, "Invalid URI format")

        # Group edges by source URI
        edges_by_source = {}
        for edge in edges:
            edges_by_source.setdefault(edge.from_uri, []).append(edge)

        dreaming = getattr(self._fs, "dreaming_enabled", False) is True
        for source_uri, source_edges in edges_by_source.items():
            lock = (self._fs.memory_identity_lock(source_uri.rsplit("/", 1)[0], ctx)
                    if dreaming else nullcontext())
            with lock:
                if dreaming:
                    from fs.agfs_adapter.dreaming import bind_relation_target

                    resolved = self._fs.read_dreaming_alias(source_uri, ctx)
                    source_uri = resolved.uri
                    source_edges = [bind_relation_target(self._fs, replace(edge, from_uri=source_uri), ctx)
                                    for edge in source_edges]
                self._write_edges(source_uri, source_edges, ctx)

    def _write_edges(self, source_uri, source_edges, ctx):
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        path = f"{self._mount_prefix}{uri_to_path(source_uri)}.relations.json"
        edges = {
            (edge.from_uri, edge.to_uri, edge.relation_type, edge.target_created_at): edge
            for edge in [*self._read_edges(source_uri, ctx), *source_edges]
        }
        rows = [{key: value for key, value in asdict(edge).items() if value is not None} for edge in edges.values()]
        content = json.dumps(rows, ensure_ascii=False, indent=2)
        self._client.write(path, content.encode("utf-8"))

    def get_one_hop(self, uri: str, ctx: RequestContext, limit: int = 3) -> list[RelationEdge]:
        """Get top-N outgoing edges sorted by weight (descending).

        Args:
            uri: ContextEngine URI of the source node
            ctx: RequestContext for access control
            limit: Maximum number of edges to return (default: 3)

        Returns:
            List of RelationEdge objects sorted by weight (descending),
            limited to `limit` items. Empty list if node has no relations
            or doesn't exist.

        Raises:
            AccessDeniedError: If URI account doesn't match context account
        """
        all_edges = self.get_edges(uri, ctx)
        if not all_edges:
            return []
        sorted_edges = sorted(all_edges, key=lambda e: e.weight, reverse=True)
        return sorted_edges[:limit]
