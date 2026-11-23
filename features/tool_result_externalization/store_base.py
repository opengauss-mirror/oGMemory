"""Artifact store contract and shared byte-range helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import RequestContext

from .models import ArtifactMetadata


class ArtifactNotVisibleError(LookupError):
    """The artifact does not exist or is not visible to the full owner tuple."""


class ArtifactStoreUnavailableError(OSError):
    """The configured Artifact backend is temporarily unavailable."""


class ToolResultExternalizationDisabledError(LookupError):
    """Tool-result APIs are unavailable because the optional feature is off."""


def require_safe_path_component(value: str, *, field: str) -> str:
    """Validate one human-readable storage path component.

    oGMemory keeps Session identifiers readable while rejecting ambiguous
    traversal components. Filesystem and AGFS Artifact stores enforce this
    rule before joining account and Session identifiers into a path.
    """
    if not isinstance(value, str) or not value:
        raise ArtifactNotVisibleError("not_found_or_not_visible")
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or (len(value) >= 2 and value[0].isalpha() and value[1] == ":")
    ):
        raise ArtifactNotVisibleError("not_found_or_not_visible")
    return value


def require_session_owner(session_id: str, ctx: RequestContext) -> None:
    if not session_id or session_id != ctx.session_id:
        raise ArtifactNotVisibleError("not_found_or_not_visible")


def metadata_visible(metadata: ArtifactMetadata, session_id: str, ctx: RequestContext) -> bool:
    return (
        metadata.account_id == ctx.account_id
        and metadata.session_id == session_id == ctx.session_id
        and metadata.owner_user_id == ctx.user_id
        and metadata.owner_agent_id == ctx.agent_id
    )


def artifact_reuse_matches(
    existing: ArtifactMetadata,
    incoming: ArtifactMetadata,
    existing_output: str,
    incoming_output: str,
) -> bool:
    """Return whether an immutable Artifact write is an exact logical retry."""
    identity_fields = (
        "artifact_id",
        "account_id",
        "session_id",
        "owner_user_id",
        "owner_agent_id",
        "turn_id",
        "source_message_id",
        "tool_call_id",
        "part_path",
        "occurrence_index",
        "sha256",
    )
    return (
        all(
            getattr(existing, field) == getattr(incoming, field)
            for field in identity_fields
        )
        and existing_output == incoming_output
    )


def decode_utf8_range(data: bytes, offset_bytes: int, length_bytes: int) -> dict:
    if offset_bytes < 0 or length_bytes < 1:
        raise ValueError("offset_bytes must be non-negative and length_bytes must be positive")
    actual_offset = min(offset_bytes, len(data))
    while actual_offset < len(data) and data[actual_offset] & 0xC0 == 0x80:
        actual_offset += 1
    requested_end = min(len(data), actual_offset + length_bytes)
    actual_end = requested_end
    while actual_end < len(data) and data[actual_end] & 0xC0 == 0x80:
        actual_end += 1
    content_bytes = data[actual_offset:actual_end]
    content = content_bytes.decode("utf-8")
    return {
        "content": content,
        "actual_offset_bytes": actual_offset,
        "actual_length_bytes": len(content_bytes),
        "next_offset_bytes": actual_end,
        "decoded_chars": len(content),
        "eof": actual_end >= len(data),
    }


class ToolResultArtifactStore(ABC):
    _migration_reads = ("list", "read_all", "read_metadata", "read_range")

    @abstractmethod
    def write(self, metadata: ArtifactMetadata, output_text: str, ctx: RequestContext) -> bool:
        """Commit output and metadata; return True when an identical artifact was reused."""

    @abstractmethod
    def read_all(self, artifact_id: str, session_id: str, ctx: RequestContext) -> tuple[ArtifactMetadata, str]:
        pass

    def read_metadata(
        self,
        artifact_id: str,
        session_id: str,
        ctx: RequestContext,
    ) -> ArtifactMetadata:
        """Read owner-validated metadata without requiring callers to load content."""
        metadata, _output = self.read_all(artifact_id, session_id, ctx)
        return metadata

    @abstractmethod
    def list(self, session_id: str, ctx: RequestContext, limit: int = 50, cursor: str | None = None) -> dict:
        pass

    def read_range(
        self,
        artifact_id: str,
        session_id: str,
        ctx: RequestContext,
        offset_bytes: int,
        length_bytes: int,
    ) -> dict:
        metadata, output = self.read_all(artifact_id, session_id, ctx)
        result = decode_utf8_range(output.encode("utf-8"), offset_bytes, length_bytes)
        return {"metadata": metadata.to_dict(), "ref": metadata.ref, **result}

    @abstractmethod
    def delete_session_artifacts(self, session_id: str, ctx: RequestContext) -> int:
        pass


class UnavailableToolResultArtifactStore(ToolResultArtifactStore):
    """Fail-closed store used when an enabled backend cannot initialize."""

    def __init__(self, error: Exception):
        self.error = error

    def _raise(self):
        raise ArtifactStoreUnavailableError(
            "tool-result artifact store is unavailable"
        ) from self.error

    def write(self, metadata, output_text, ctx):
        self._raise()

    def read_all(self, artifact_id, session_id, ctx):
        self._raise()

    def read_metadata(self, artifact_id, session_id, ctx):
        self._raise()

    def list(self, session_id, ctx, limit=50, cursor=None):
        self._raise()

    def delete_session_artifacts(self, session_id, ctx):
        self._raise()
