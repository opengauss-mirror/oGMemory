"""Call-local description of the prompt and tools offered to extraction."""
from dataclasses import dataclass


class UnsupportedAssemblyTemplateError(ValueError):
    """A template cannot safely be filtered using trusted metadata."""


@dataclass(frozen=True)
class AssemblyContract:
    categories: frozenset[str]
    tool_names: tuple[str, ...]
    mode: str
    scope: str
    reason: str | None = None
    source_only: bool = False
