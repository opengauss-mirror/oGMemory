"""ContextReader — ACL-checked URI reads with canonical L0, L1, L2 and metadata."""

from __future__ import annotations

import logging
from dataclasses import replace

from core.errors import AccessDeniedError, NodeBrokenError, NodeNotFoundError, ValidationError
from core.interfaces import ContextFS
from core.models import RequestContext, RetrievedBlock
from retrieval.identity import canonical_uri

logger = logging.getLogger(__name__)


def canonical_agent_node(node):
    """Read current and legacy Agent bodies through the canonical fields."""
    if node.metadata.get("agent_memory_version") != 1:
        return node
    from extraction.agent_memory import memory_units, node_memory_fields, render_agent_memory

    fields = node_memory_fields(node)
    grounding = [source for unit in memory_units(node.category, fields)
                 for source in unit.get("grounding", [])]
    return replace(node, content=render_agent_memory(node.category, fields, grounding=grounding),
                   metadata={**node.metadata, "structured_fields": fields, "grounding": grounding})


def project_memory_unit(node, unit_key: str):
    """Project a node already read with ACL into its current reusable unit."""
    from extraction.agent_memory import memory_units, node_memory_fields, render_agent_memory

    if node.metadata.get("agent_memory_version") != 1:
        raise ValueError("unit_key requires an Agent memory unit")
    fields = node_memory_fields(node)
    unit = next((item for item in memory_units(node.category, fields)
                 if item["key"] == unit_key), None)
    if unit is None:
        raise ValueError("Memory unit was not found")
    selected = {**fields, "units": [unit]} if node.category == "tool" else unit
    state = node.metadata.get("agent_memory", {}).get("units", {}).get(unit_key, {})
    return replace(
        node, abstract=unit["title"], overview="",
        content=render_agent_memory(node.category, selected, grounding=unit.get("grounding")),
        metadata={**node.metadata, "structured_fields": selected,
                  "unit_key": unit_key, "agent_memory_unit_key": unit_key,
                  "grounding": unit.get("grounding", []),
                  "unit_revision": state.get("revision", 1)},
    )


class ContextReader:
    def __init__(self, fs: ContextFS | None = None, *, storage_backend: str = "agfs") -> None:
        self._fs = fs
        self._storage_backend = storage_backend

    def is_visible(self, uri: str, ctx: RequestContext) -> bool:
        """Validate an index hit against the current node status and access scope."""
        uri = canonical_uri(uri)
        try:
            self._check_acl(uri, ctx)
            exists = getattr(self._fs, "exists", None)
            return bool(exists(uri, ctx)) if callable(exists) else True
        except (AccessDeniedError, PermissionError, NodeNotFoundError, NodeBrokenError, FileNotFoundError):
            return False

    def read(
        self,
        uri: str,
        ctx: RequestContext,
        *,
        unit_key: str | None = None,
    ) -> RetrievedBlock:
        self._check_acl(uri, ctx)

        block = RetrievedBlock(uri=uri, level_hit="L2")

        if self._fs is None:
            logger.warning("[ContextReader] no ContextFS configured")
            return block

        try:
            # Strip file suffix from L2 URIs (e.g. "/content.md") — read_node
            # expects the directory URI, not the file URI stored in the vector index.
            read_uri = uri
            if read_uri.endswith("/content.md"):
                read_uri = read_uri[: -len("/content.md")]
            node = self._fs.read_node(read_uri, ctx)
        except (AccessDeniedError, PermissionError, NodeNotFoundError, NodeBrokenError, FileNotFoundError):
            raise
        except Exception as exc:
            logger.warning("[ContextReader] read_node failed for %s: %s", uri, exc, exc_info=True)
            return block

        if unit_key is not None:
            node = project_memory_unit(node, unit_key)
        else:
            node = canonical_agent_node(node)

        block.category = node.category
        block.owner_space = node.owner_space
        block.abstract = node.abstract
        block.overview = node.overview
        block.content_excerpt = node.content
        block.metadata = dict(node.metadata)
        block.has_overview = bool(node.overview)
        block.has_content = bool(node.content)

        return block

    def _check_acl(self, uri: str, ctx: RequestContext) -> None:
        # SECURITY: RequestContext is mandatory for all read operations
        # Per CLAUDE.md §8: "唯一强制注入 RequestContext 的位置是 service 层"
        if ctx is None:
            raise ValidationError("ctx", "RequestContext is required for read operation")

        if ctx.account_id is None or not ctx.account_id.strip():
            raise ValidationError("account_id", "account_id is required for read operation")

        prefix = f"ctx://{ctx.account_id}/"
        if uri.startswith("ctx://") and not uri.startswith(prefix):
            raise AccessDeniedError(uri, ctx.account_id, "account mismatch")

        if uri.startswith(f"{prefix}resources/"):
            return

        # Owner space isolation: strategy depends on storage backend.
        #
        # AGFS mode: strict check — only allow access to the caller's own
        #   user/agent space via ctx.user_id / ctx.agent_id.
        #
        # SQL mode: relaxed check — visible_owner_spaces includes shared
        #   agents resolved by TenantAdminService, so users can read
        #   memories belonging to agents they have been granted access to.
        if self._storage_backend == "sql":
            vos = ctx.visible_owner_spaces
            if vos is not None and len(vos) > 0:
                _owner_match = False
                for space in vos:
                    kind, _, oid = space.partition(":")
                    space_prefix = f"ctx://{ctx.account_id}/{kind}s/{oid}/"
                    if uri.startswith(space_prefix):
                        _owner_match = True
                        break
                if not _owner_match:
                    raise AccessDeniedError(uri, ctx.account_id, "owner_space mismatch")
            else:
                # Fallback: strict owner_id check consistent with sql_context_fs
                if uri.startswith(f"ctx://{ctx.account_id}/users/"):
                    user_prefix = f"ctx://{ctx.account_id}/users/{ctx.user_id}/"
                    if not uri.startswith(user_prefix):
                        raise AccessDeniedError(uri, ctx.account_id, "owner_space mismatch")
                if ctx.agent_id and uri.startswith(f"ctx://{ctx.account_id}/agents/"):
                    agent_prefix = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/"
                    if not uri.startswith(agent_prefix):
                        raise AccessDeniedError(uri, ctx.account_id, "owner_space mismatch")
        else:
            if uri.startswith(f"ctx://{ctx.account_id}/users/"):
                user_prefix = f"ctx://{ctx.account_id}/users/{ctx.user_id}/"
                if not uri.startswith(user_prefix):
                    raise AccessDeniedError(uri, ctx.account_id, "owner_space mismatch")
            if ctx.agent_id and uri.startswith(f"ctx://{ctx.account_id}/agents/"):
                agent_prefix = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/"
                if not uri.startswith(agent_prefix):
                    raise AccessDeniedError(uri, ctx.account_id, "owner_space mismatch")
