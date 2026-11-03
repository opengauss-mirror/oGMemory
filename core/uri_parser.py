"""Normalize index record URIs; canonical parsing stays with storage adapters."""


def strip_l2_suffix(uri: str) -> str:
    """Remove the index-only content suffix before canonical reads/dedup."""
    suffix = "/content.md"
    return uri[:-len(suffix)] if uri.endswith(suffix) else uri
