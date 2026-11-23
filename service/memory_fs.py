# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""MemoryFS - File system-like interface for browsing memory nodes.

Provides a familiar file system API for agents to browse and inspect memories.
This is a read-only interface - no write operations are exposed.

Designed for dev branch (write path) to allow basic memory verification and browsing.
For advanced search (vector similarity), use phase1 branch with ReadAPI.
"""

from typing import Optional, Dict, Any, List
from urllib.parse import unquote

from core.interfaces import ContextFS
from core.models import RequestContext
from core.errors import AccessDeniedError, NodeNotFoundError


class MemoryFS:
    """File system-like interface for browsing memory nodes.

    Provides familiar file system operations for agents to explore stored memories:
    - List directories (categories, memory nodes)
    - Get node metadata (stat)
    - Read abstracts (brief summaries)
    - Check existence

    All operations enforce multi-tenant isolation via RequestContext.

    Examples:
        >>> from service.memory_fs import MemoryFS
        >>> fs = MemoryFS(agfs_context_fs)
        >>>
        >>> # List user's memory categories
        >>> categories = fs.list_memories(ctx, "/users/alice/memories")
        >>> # ["profile", "preferences", "entities", "events"]
        >>>
        >>> # List memories in a category
        >>> preferences = fs.list_memories(ctx, "/users/alice/memories/preferences")
        >>> # ["coding_style", "tools", ...]
        >>>
        >>> # Get memory info
        >>> info = fs.stat(ctx, "/users/alice/memories/profile")
        >>> print(f"Created: {info['created_at']}, Size: {info['size']}")
        >>>
        >>> # Read just the abstract (brief summary)
        >>> abstract = fs.read_abstract(ctx, "/users/alice/memories/profile")
    """

    def __init__(self, fs: ContextFS):
        """Initialize MemoryFS with a ContextFS implementation.

        Args:
            fs: ContextFS implementation for reading nodes
        """
        self._fs = fs

    def _path_to_uri(self, path: str, ctx: RequestContext) -> str:
        """Convert a memory path to full URI using context.

        Args:
            path: Memory path (e.g., "/users/alice/memories/profile")
            ctx: RequestContext containing account_id

        Returns:
            Full ContextEngine URI (e.g., "ctx://acct/users/alice/memories/profile")

        Raises:
            AccessDeniedError: If full URI's account doesn't match context
            ValueError: If path contains traversal patterns
        """
        # SECURITY: Reject path traversal patterns
        # Check both raw path and URL-decoded path to prevent encoding bypass
        if ".." in path or ".." in unquote(path):
            raise ValueError(f"Invalid path contains traversal patterns: {path}")

        # SECURITY: Validate full URI account matches context
        if path.startswith("ctx://"):
            import re
            match = re.match(r'^ctx://([^/]+)/', path)
            if match:
                uri_account = match.group(1)
                if uri_account != ctx.account_id:
                    from core.errors import AccessDeniedError
                    raise AccessDeniedError(
                        path, ctx.account_id,
                        f"URI account '{uri_account}' does not match context account"
                    )
            return path

        # Remove leading slash and split
        parts = [p for p in path.split("/") if p]

        if not parts:
            raise ValueError(f"Invalid path: {path}")

        # Determine scope and build URI
        if parts[0] == "users":
            # /users/{user}/memories/{category}/{slug}
            # → ctx://{account}/users/{user}/memories/{category}/{slug}
            if len(parts) < 2:
                raise ValueError(f"Invalid user path: {path}")

            user_id = parts[1]
            if len(parts) >= 3 and parts[2] == "memories":
                # /users/{user}/memories → list memories directory
                # /users/{user}/memories/{category} → list category or access node
                if len(parts) == 3:
                    # Exactly /users/{user}/memories - return directory URI
                    return f"ctx://{ctx.account_id}/users/{user_id}/memories"
                else:
                    # /users/{user}/memories/{category} or /users/{user}/memories/{category}/{slug}
                    # Skip parts[2] ("memories") when joining remaining
                    remaining = "/".join(parts[3:])
                    return f"ctx://{ctx.account_id}/users/{user_id}/memories/{remaining}"
            else:
                # /users/{user} → list their memories
                return f"ctx://{ctx.account_id}/users/{user_id}/memories"

        elif parts[0] == "agents":
            # /agents/{agent}/memories/{category}/{slug}
            if len(parts) < 2:
                raise ValueError(f"Invalid agent path: {path}")

            agent_id = parts[1]
            if len(parts) >= 3 and parts[2] == "memories":
                if len(parts) == 3:
                    # Exactly /agents/{agent}/memories - return directory URI
                    return f"ctx://{ctx.account_id}/agents/{agent_id}/memories"
                else:
                    # /agents/{agent}/memories/{category} or /agents/{agent}/memories/{category}/{slug}
                    remaining = "/".join(parts[3:])
                    return f"ctx://{ctx.account_id}/agents/{agent_id}/memories/{remaining}"
            else:
                return f"ctx://{ctx.account_id}/agents/{agent_id}/memories"

        else:
            raise ValueError(f"Invalid path: {path}")

    def _uri_to_path(self, uri: str) -> str:
        """Convert a URI to a display path.

        Args:
            uri: Full ContextEngine URI

        Returns:
            Display path (e.g., "/users/alice/memories/profile")
        """
        if not uri.startswith("ctx://"):
            return uri

        # Remove ctx:// and return as path
        return "/" + uri[6:]

    def list(self, path: str, ctx: RequestContext) -> List[Dict[str, Any]]:
        """List contents at a memory path.

        Args:
            path: Memory path (e.g., "/users/alice/memories",
                   "/users/alice/memories/preferences")
            ctx: RequestContext for access control

        Returns:
            List of child items with metadata:
            [
                {"name": "profile", "type": "node", "category": "profile"},
                {"name": "preferences", "type": "directory", "category": "preferences"},
                ...
            ]

        Raises:
            AccessDeniedError: If path not accessible
            NodeNotFoundError: If path doesn't exist
        """
        uri = self._path_to_uri(path, ctx)

        # Get children using ContextFS
        child_uris = self._fs.list_children(uri, ctx)

        # Determine if we're listing the memories root (children are categories)
        # or a specific category (children are memory nodes)
        is_memories_root = uri.endswith("/memories") or path.rstrip("/").endswith("/memories")

        # Parse and format results
        result = []
        for child_uri in child_uris:
            try:
                # Extract name from URI
                # Format: ctx://{account}/users/{user}/memories/{category}/{slug}
                # or: ctx://{account}/users/{user}/memories/profile
                if "/memories/" in child_uri:
                    # Get the part after /memories/
                    after_memories = child_uri.split("/memories/")[1]
                    full_name = after_memories.rstrip("/")

                    # Check if it's a category directory or specific node
                    has_slash = "/" in full_name
                    is_node = not has_slash or full_name == "profile"

                    # If we're at memories root, items are categories (directories)
                    # even if they don't have a slash (like "profile", "preferences")
                    if is_memories_root:
                        is_node = False  # All items at root are category directories
                        # For root, items like "profile" are still categories
                        # but can be listed as nodes since they're special
                        if full_name == "profile":
                            is_node = True  # profile is a special case - both category and node
                        else:
                            is_node = False

                    # Extract category and name
                    if has_slash:
                        # Nested path like "preferences/coding_style"
                        parts = full_name.split("/")
                        category = parts[0]
                        name = parts[1]  # Last part is the actual name
                    else:
                        # Simple name like "profile" or "preferences"
                        category = full_name
                        name = full_name

                    result.append({
                        "name": name,
                        "uri": child_uri,
                        "type": "node" if is_node else "directory",
                        "category": category,
                    })
            except Exception:
                # Skip malformed URIs
                continue

        return result

    def list_memories(self, ctx: RequestContext, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """List memories with category filtering.

        Convenience method that defaults to user's memories if no path specified.

        Args:
            ctx: RequestContext for access control
            path: Optional path (defaults to user's memories root)

        Returns:
            List of memory items with metadata
        """
        if path is None:
            path = f"/users/{ctx.user_id}/memories"
        return self.list(path, ctx)

    def stat(self, path: str, ctx: RequestContext) -> Dict[str, Any]:
        """Get metadata about a memory node.

        Args:
            path: Memory path or URI
            ctx: RequestContext for access control

        Returns:
            Dict with metadata:
            {
                "uri": "ctx://...",
                "path": "/users/...",
                "category": "profile|preference|...",
                "created_at": "2026-03-19T...",
                "updated_at": "2026-03-19T...",
                "version": 1,
                "size": 1234,  # content length
                "has_overview": true,
                "has_content": true,
            }

        Raises:
            AccessDeniedError: If not accessible
            NodeNotFoundError: If doesn't exist
        """
        uri = self._path_to_uri(path, ctx)

        # Read the node
        node = self._fs.read_node(uri, ctx)

        return {
            "uri": node.uri,
            "path": self._uri_to_path(node.uri),
            "category": node.category,
            "context_type": node.context_type,
            "level": node.level,
            # SECURITY: owner_space removed to prevent user enumeration
            "created_at": node.metadata.get("created_at", ""),
            "updated_at": node.metadata.get("updated_at", ""),
            "version": node.metadata.get("version", 1),
            "size": len(node.content),
            "abstract_length": len(node.abstract),
            "overview_length": len(node.overview),
            "has_overview": bool(node.overview),
            "has_content": bool(node.content),
        }

    def read_abstract(self, path: str, ctx: RequestContext) -> str:
        """Read just the abstract (brief summary) of a memory.

        This is a lightweight operation - doesn't fetch full content.

        Args:
            path: Memory path or URI
            ctx: RequestContext for access control

        Returns:
            Abstract text (≤200 chars)

        Raises:
            AccessDeniedError: If not accessible
            NodeNotFoundError: If doesn't exist
        """
        uri = self._path_to_uri(path, ctx)
        node = self._fs.read_node(uri, ctx)
        return node.abstract

    def read_overview(self, path: str, ctx: RequestContext) -> str:
        """Read the overview (structured summary) of a memory.

        Args:
            path: Memory path or URI
            ctx: RequestContext for access control

        Returns:
            Overview text (structured summary)

        Raises:
            AccessDeniedError: If not accessible
            NodeNotFoundError: If doesn't exist
        """
        uri = self._path_to_uri(path, ctx)
        node = self._fs.read_node(uri, ctx)
        return node.overview

    def exists(self, path: str, ctx: RequestContext) -> bool:
        """Check if a memory node exists.

        Args:
            path: Memory path or URI
            ctx: RequestContext for access control

        Returns:
            True if node exists and is ACTIVE, False otherwise
        """
        uri = self._path_to_uri(path, ctx)
        return self._fs.exists(uri, ctx)

    def get_categories(self, ctx: RequestContext, owner_type: str = "user") -> List[str]:
        """Get available memory categories for a user/agent.

        Args:
            ctx: RequestContext
            owner_type: "user" or "agent"

        Returns:
            List of category names (e.g., ["profile", "preferences", "entities", ...])
        """
        if owner_type == "user":
            path = f"/users/{ctx.user_id}/memories"
        else:
            path = f"/agents/{ctx.agent_id}/memories"

        try:
            children = self.list(path, ctx)
            # Extract unique categories
            categories = []
            for child in children:
                if child["type"] == "directory" or child["name"] == "profile":
                    categories.append(child["category"])
            return categories
        except (NodeNotFoundError, AccessDeniedError):
            return []

    def get_summary(self, ctx: RequestContext) -> Dict[str, Any]:
        """Get a summary of memory storage for the current context.

        Args:
            ctx: RequestContext

        Returns:
            Dict with memory statistics:
            {
                "user_memories": {
                    "profile": 1,
                    "preferences": 3,
                    "entities": 5,
                    ...
                },
                "agent_memories": {
                    "cases": 2,
                    "patterns": 1,
                    ...
                },
                "total_nodes": 12
            }
        """
        summary = {
            "user_memories": {},
            "agent_memories": {},
            "total_nodes": 0,
        }

        # Count user memories
        try:
            user_path = f"/users/{ctx.user_id}/memories"
            children = self.list(user_path, ctx)
            for child in children:
                if child["type"] == "directory":
                    # Count items in this category
                    try:
                        cat_items = self.list(f"{user_path}/{child['name']}", ctx)
                        summary["user_memories"][child["category"]] = len(cat_items)
                        summary["total_nodes"] += len(cat_items)
                    except Exception:
                        pass
                elif child["name"] == "profile":
                    summary["user_memories"]["profile"] = 1
                    summary["total_nodes"] += 1
        except Exception:
            pass

        # Count agent memories
        try:
            agent_path = f"/agents/{ctx.agent_id}/memories"
            children = self.list(agent_path, ctx)
            for child in children:
                if child["type"] == "directory":
                    try:
                        cat_items = self.list(f"{agent_path}/{child['name']}", ctx)
                        summary["agent_memories"][child["category"]] = len(cat_items)
                        summary["total_nodes"] += len(cat_items)
                    except Exception:
                        pass
        except Exception:
            pass

        return summary
