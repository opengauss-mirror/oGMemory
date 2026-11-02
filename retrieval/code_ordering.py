"""Source ordering independent of physical workspace/resource identity."""

from collections.abc import Mapping
from typing import Any


def code_source_key(record: Mapping[str, Any]) -> tuple:
    """Break code ranking ties without changing URI-based lookup or isolation.

    Index hits and source records share the same source metadata. Legacy
    records without a source path retain their identity-based ordering.
    """
    metadata = record.get("metadata") or record
    path = str(metadata.get("relative_path") or "").replace("\\", "/")
    if not path:
        return (1, str(record.get("uri") or metadata.get("uri") or record.get("id") or ""))
    return (
        0,
        str(metadata.get("resource_root_slug") or ""),
        path,
        int(metadata.get("start_line") or 0),
        int(metadata.get("end_line") or 0),
        str(metadata.get("symbol") or ""),
        str(metadata.get("chunk_hash") or ""),
    )
