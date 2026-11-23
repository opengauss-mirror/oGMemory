# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""AGFS implementation of RelationStore protocol.

Reads and writes relation edges to .relations.json files
stored alongside ContextNode directories in AGFS.
"""

import json
from typing import Optional

from pyagfs import AGFSClient
from pyagfs.exceptions import AGFSClientError

from core.models import RequestContext, RelationEdge
from core.interfaces import RelationStore
from core.errors import AccessDeniedError


class AGFSRelationStore(RelationStore):
    """AGFS-based relation store using .relations.json files.

    Relations are stored in the node directory:
    {mount_prefix}/accounts/{...}/node/.relations.json

    Format: JSON array of RelationEdge objects.
    """

    def __init__(self, client: AGFSClient, mount_prefix: str = "/local"):
        """Initialize AGFSRelationStore.

        Args:
            client: AGFS Python SDK client
            mount_prefix: AGFS mount point prefix (default: "/local")
                         Must match the mount_prefix used by AGFSContextFS
        """
        self._client = client
        self._mount_prefix = mount_prefix.rstrip("/")

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
        # SECURITY: Validate account access
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
            if edge.from_uri not in edges_by_source:
                edges_by_source[edge.from_uri] = []
            edges_by_source[edge.from_uri].append(edge)

        # Upsert for each source URI
        from fs.agfs_adapter.agfs_context_fs import uri_to_path

        for source_uri, source_edges in edges_by_source.items():
            node_path = uri_to_path(source_uri)
            relations_path = f"{self._mount_prefix}{node_path}.relations.json"

            # Read existing edges
            existing = self.get_edges(source_uri, ctx)

            # Build lookup dict: (from_uri, to_uri, relation_type) → edge
            existing_lookup = {
                (e.from_uri, e.to_uri, e.relation_type): e
                for e in existing
            }

            # Update with new edges
            for edge in source_edges:
                key = (edge.from_uri, edge.to_uri, edge.relation_type)
                existing_lookup[key] = edge

            # Convert back to list and write
            all_edges = [
                {
                    "from_uri": e.from_uri,
                    "to_uri": e.to_uri,
                    "relation_type": e.relation_type,
                    "weight": e.weight,
                    "reason": e.reason,
                }
                for e in existing_lookup.values()
            ]

            content = json.dumps(all_edges, ensure_ascii=False, indent=2)
            self._client.write(relations_path, content.encode('utf-8'))

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
