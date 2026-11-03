"""URI resolver for memory type schemas.

Resolves Jinja2 templates in MemoryTypeSchema to concrete URIs based on
RequestContext and field values. Provides sanitization to prevent
directory traversal attacks.
"""

import unicodedata
from logging import getLogger

import jinja2

from core.models import RequestContext
from core.validation import validate_utf8
from extraction.schemas.registry import SchemaRegistry

logger = getLogger(__name__)


class URIResolver:
    """Resolves memory type schemas to concrete URIs.

    Uses Jinja2 templates defined in MemoryTypeSchema to generate
    directory paths and filenames. Automatically injects account_id,
    user_id, and agent_id from RequestContext, along with any
    additional field values provided.

    Example:
        resolver = URIResolver(registry)
        uri = resolver.resolve("profile", {}, ctx)
        # → "ctx://acme/users/alice/memories/profile/content.md"
    """

    def __init__(self, registry: SchemaRegistry):
        """Initialize URIResolver.

        Args:
            registry: SchemaRegistry for accessing MemoryTypeSchema definitions
        """
        self._registry = registry
        self._jinja_env = jinja2.Environment(autoescape=False)

    @property
    def schema_registry(self) -> SchemaRegistry:
        return self._registry

    def resolve(self, memory_type: str, fields: dict, ctx: RequestContext) -> str:
        """Render directory + filename_template → complete URI.

        Automatically injects: account_id, user_id, agent_id + any values
        from fields dict. Context identities remain exact; content fields are
        sanitized for use as path slugs and cannot replace context identities.

        Args:
            memory_type: Memory type identifier (e.g., "profile", "preference")
            fields: Additional field values for template rendering (e.g., {"routing_key": "coffee"})
            ctx: RequestContext providing account_id, user_id, agent_id

        Returns:
            Complete URI string (e.g., "ctx://acme/users/alice/memories/preferences/coffee.md")

        Raises:
            ValueError: If memory_type is not found in registry
        """
        schema = self._registry.get(memory_type)
        if schema is None:
            raise ValueError(f"Unknown memory_type: {memory_type}")

        context_vars = self._context_template_vars(ctx, schema.owner_scope)
        template_vars = {
            k: self._sanitize(str(v))
            for k, v in fields.items()
            if v is not None and k not in context_vars
        }
        template_vars.update(context_vars)

        # Render directory template
        dir_path = self._jinja_env.from_string(schema.directory).render(**template_vars)

        # For single-file schemas (e.g., profile: content.md), the directory IS the node URI.
        # For multi-file schemas (e.g., preference: {{ routing_key }}.md), append the slug part.
        if not schema.is_single_file:
            filename = self._jinja_env.from_string(schema.filename_template).render(**template_vars)
            # Strip .md extension — the URI is the node identifier, not a file path
            if filename.endswith(".md"):
                filename = filename[:-3]
            if self._is_empty_filename_slug(filename):
                filename = template_vars.get("routing_key", "")
            return f"{dir_path.rstrip('/')}/{filename}"

        return dir_path.rstrip('/')

    def get_directory_uri(self, memory_type: str, ctx: RequestContext) -> str:
        """Render only the directory portion of the URI.

        Useful for listing or checking existence of all memories of a type.

        Args:
            memory_type: Memory type identifier
            ctx: RequestContext providing account_id, user_id, agent_id

        Returns:
            Directory URI string (e.g., "ctx://acme/users/alice/memories/profile/")

        Raises:
            ValueError: If memory_type is not found in registry
        """
        schema = self._registry.get(memory_type)
        if schema is None:
            raise ValueError(f"Unknown memory_type: {memory_type}")

        template_vars = self._context_template_vars(ctx, schema.owner_scope)

        return self._jinja_env.from_string(schema.directory).render(**template_vars)

    @staticmethod
    def _context_template_vars(ctx: RequestContext, owner_scope: str) -> dict[str, str]:
        """Keep authenticated identities as exact, unencoded URI segments."""
        values = {
            "account_id": ctx.account_id,
            "user_id": ctx.user_id,
            "agent_id": ctx.agent_id,
        }
        required_owner = "agent_id" if owner_scope == "agent" else "user_id"
        for name, value in values.items():
            if not isinstance(value, str) or (not value and name in {"account_id", required_owner}):
                raise ValueError(f"{name} must be one safe URI path segment")
            validate_utf8(value, name)
            # Backends compare raw identities, so encoded principal syntax is
            # unsupported. Dot paths/separators are normalized by filesystem
            # routing; controls cannot be represented safely by the backends.
            # Reject '..' anywhere to retain validate_uri's existing boundary.
            if (value == "." or ".." in value
                    or any(char in value for char in "/\\%")
                    or any(unicodedata.category(char) in {"Cc", "Cs"} for char in value)):
                raise ValueError(f"{name} must be one safe URI path segment")
        return values

    def validate_uri(self, uri: str, ctx: RequestContext) -> bool:
        """Check if URI is within the schema-declared namespace.

        Validates that:
        - URI does not contain ".." (directory traversal)
        - URI starts with the expected account prefix

        Args:
            uri: URI string to validate
            ctx: RequestContext for account_id verification

        Returns:
            True if URI is valid, False otherwise
        """
        # Block directory traversal attempts
        if ".." in uri:
            return False

        # Ensure URI is within the correct account namespace
        prefix = f"ctx://{ctx.account_id}/"
        return uri.startswith(prefix)

    @staticmethod
    def _sanitize(value: str) -> str:
        """Sanitize a value for use in URI paths.

        Prevents directory traversal and ensures consistent formatting:
        - Removes ".." (parent directory references)
        - Replaces "/" and "\\" with "_"
        - Strips leading/trailing whitespace
        - Replaces spaces with underscores
        - Converts to lowercase

        Args:
            value: Raw string value

        Returns:
            Sanitized string safe for use in URI paths
        """
        # Remove directory traversal patterns
        value = value.replace("..", "")

        # Replace path separators with underscore
        value = value.replace("/", "_").replace("\\", "")

        # Clean up whitespace
        value = value.strip().replace(" ", "_")

        # Convert to lowercase for consistency
        return value.lower()

    @staticmethod
    def _is_empty_filename_slug(filename: str) -> bool:
        """Return True when a rendered filename has no useful slug content."""
        return not filename or not any(ch.isalnum() for ch in filename)
