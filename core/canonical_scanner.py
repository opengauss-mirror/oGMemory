"""Public contract for bounded scans of canonical memory nodes."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from core.models import ContextNode


@runtime_checkable
class CanonicalNodeScanner(Protocol):
    """Read active canonical nodes without exposing storage internals."""

    def projection_is_quiescent(self, *, account_id: str) -> bool | None:
        """Report projected writes, or return None when background preparation is needed."""
        ...

    def scan_active_nodes(
        self,
        *,
        account_id: str,
        owner_spaces: Sequence[str],
        page_size: int,
    ) -> Iterator[ContextNode]: ...
