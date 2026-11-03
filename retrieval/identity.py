"""Stable identity for a memory node or one independently retrievable unit."""


def canonical_uri(uri: str) -> str:
    uri = uri.rstrip("/")
    for suffix in ("/content.md", "/.abstract.md", "/.overview.md"):
        if uri.endswith(suffix):
            return uri[:-len(suffix)]
    return uri


def memory_unit_key(metadata: dict | None) -> str:
    metadata = metadata or {}
    return str(metadata.get("unit_key") or metadata.get("agent_memory_unit_key") or "")


def memory_identity(uri: str, metadata: dict | None = None) -> tuple[str, str]:
    return canonical_uri(uri), memory_unit_key(metadata)
