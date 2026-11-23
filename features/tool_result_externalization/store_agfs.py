"""AGFS tool-result store using metadata-last visibility."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import PurePosixPath

from core.models import RequestContext

from .models import ArtifactMetadata
from .store_base import (
    ArtifactNotVisibleError,
    ArtifactStoreUnavailableError,
    ToolResultArtifactStore,
    artifact_reuse_matches,
    decode_utf8_range,
    metadata_visible,
    require_safe_path_component,
    require_session_owner,
)

_ARTIFACT_ID = re.compile(r"^tr_[a-f0-9]{16,64}(?:_[a-f0-9]{8,16})?$")
_EXPLICIT_NOT_FOUND = (
    re.compile(
        r"(?:^|\b)no such (?:file|directory|file or directory)(?:\b|$)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|\b)file not found(?:\s*:|\s*$)", re.IGNORECASE),
    re.compile(r"(?:^|\b)directory not found(?:\s*:|\s*$)", re.IGNORECASE),
    re.compile(r"^not found$", re.IGNORECASE),
    re.compile(
        r"^(?:[a-z][a-z0-9_-]*:\s+)?/[^\r\n]*:\s+not found$",
        re.IGNORECASE,
    ),
)
_EXPLICIT_ALREADY_EXISTS = (
    re.compile(r"(?:^|\b)already exists(?:\b|$)", re.IGNORECASE),
    re.compile(r"(?:^|\b)(?:file|directory) exists(?:\b|$)", re.IGNORECASE),
)


def _is_explicit_not_found(exc: Exception) -> bool:
    if isinstance(exc, (FileNotFoundError, KeyError)):
        return True
    if getattr(exc, "status_code", None) == 404:
        return True
    message = str(exc).strip()
    return any(pattern.search(message) for pattern in _EXPLICIT_NOT_FOUND)


def _raise_read_error(exc: Exception) -> None:
    if _is_explicit_not_found(exc):
        raise ArtifactNotVisibleError("not_found_or_not_visible") from exc
    _raise_store_unavailable(exc)


def _raise_store_unavailable(exc: Exception) -> None:
    raise ArtifactStoreUnavailableError(
        "tool-result artifact store is unavailable"
    ) from exc


def _is_explicit_already_exists(exc: Exception) -> bool:
    if isinstance(exc, FileExistsError):
        return True
    if getattr(exc, "status_code", None) == 409:
        return True
    message = str(exc).strip()
    return any(pattern.search(message) for pattern in _EXPLICIT_ALREADY_EXISTS)


class AGFSToolResultArtifactStore(ToolResultArtifactStore):
    _migration_internal_writes = ("_mkdir_p",)

    def __init__(self, client, mount_prefix: str = "/local"):
        self._client = client
        self._mount_prefix = mount_prefix.rstrip("/")
        self._lock = threading.RLock()
        self._concurrent_commit_wait_seconds = 0.25
        self._concurrent_commit_poll_seconds = 0.01

    def _base(self, session_id: str, ctx: RequestContext) -> str:
        require_session_owner(session_id, ctx)
        account_id = require_safe_path_component(ctx.account_id, field="account_id")
        user_id = require_safe_path_component(ctx.user_id, field="user_id")
        agent_id = require_safe_path_component(ctx.agent_id, field="agent_id")
        safe_session_id = require_safe_path_component(session_id, field="session_id")
        path = (
            PurePosixPath("/accounts")
            / account_id
            / "user"
            / user_id
            / "agents"
            / agent_id
            / "sessions"
            / safe_session_id
            / "tool-results"
        )
        return f"{self._mount_prefix}{path}" if self._mount_prefix else str(path)

    def _dir(self, artifact_id: str, session_id: str, ctx: RequestContext) -> str:
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        return f"{self._base(session_id, ctx)}/{artifact_id}"

    def _mkdir_p(self, path: str) -> None:
        current = ""
        # The first component is the AGFS mount root (for example ``/local``).
        # Any remaining mount-prefix components are ordinary directories and
        # must be created for a fresh isolated namespace.
        mount_root_parts = [part for part in self._mount_prefix.split("/") if part]
        skipped_mount_roots = 1 if mount_root_parts else 0
        for index, part in enumerate(part for part in path.split("/") if part):
            current += "/" + part
            if index < skipped_mount_roots:
                continue
            try:
                self._client.mkdir(current)
            except Exception as exc:
                if not _is_explicit_already_exists(exc):
                    raise

    def _read_bytes(self, path: str) -> bytes:
        try:
            return bytes(self._client.read(path))
        except Exception as exc:
            _raise_read_error(exc)

    def _read_metadata(
        self,
        artifact_id: str,
        session_id: str,
        ctx: RequestContext,
    ) -> ArtifactMetadata:
        directory = self._dir(artifact_id, session_id, ctx)
        raw = self._read_bytes(f"{directory}/metadata.json")
        try:
            metadata = ArtifactMetadata.from_dict(json.loads(raw.decode("utf-8")))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
        ) as exc:
            raise ArtifactNotVisibleError("not_found_or_not_visible") from exc
        if not metadata_visible(metadata, session_id, ctx):
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        return metadata

    def _reuse_committed(
        self,
        metadata: ArtifactMetadata,
        output_text: str,
        ctx: RequestContext,
    ) -> bool:
        existing, existing_output = self.read_all(
            metadata.artifact_id, metadata.session_id, ctx,
        )
        if artifact_reuse_matches(
            existing, metadata, existing_output, output_text,
        ):
            return True
        raise FileExistsError("artifact identity conflict")

    def _await_concurrent_commit(
        self,
        metadata: ArtifactMetadata,
        output_text: str,
        ctx: RequestContext,
    ) -> bool:
        deadline = time.monotonic() + self._concurrent_commit_wait_seconds
        while True:
            try:
                return self._reuse_committed(metadata, output_text, ctx)
            except ArtifactNotVisibleError as exc:
                if time.monotonic() >= deadline:
                    raise ArtifactStoreUnavailableError(
                        "tool-result artifact write is already in progress"
                    ) from exc
                time.sleep(self._concurrent_commit_poll_seconds)

    def write(self, metadata: ArtifactMetadata, output_text: str, ctx: RequestContext) -> bool:
        directory = self._dir(metadata.artifact_id, metadata.session_id, ctx)
        if not metadata_visible(metadata, metadata.session_id, ctx):
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        output_bytes = output_text.encode("utf-8")
        if hashlib.sha256(output_bytes).hexdigest() != metadata.sha256:
            raise ValueError("artifact output does not match metadata")
        with self._lock:
            try:
                return self._reuse_committed(metadata, output_text, ctx)
            except ArtifactNotVisibleError:
                pass

            parent = directory.rsplit("/", 1)[0]
            self._mkdir_p(parent)
            try:
                # AGFS mkdir is the cross-process claim. Unlike _mkdir_p(),
                # this call must not swallow an already-exists response.
                self._client.mkdir(directory)
            except Exception as exc:
                if _is_explicit_already_exists(exc):
                    return self._await_concurrent_commit(
                        metadata, output_text, ctx,
                    )
                _raise_store_unavailable(exc)

            try:
                self._client.write(
                    f"{directory}/_writing.json",
                    json.dumps({
                        "artifact_id": metadata.artifact_id,
                        "writer_token": uuid.uuid4().hex,
                        "started_at": datetime.now(UTC).isoformat(),
                    }).encode("utf-8"),
                )
            except Exception as exc:
                # This writer exclusively created the directory, so a marker
                # failure can safely release the otherwise uncollectable empty
                # claim. Once the marker exists, later failures are instead
                # left for grace-period stale-write cleanup.
                try:
                    self._client.rm(directory, recursive=True)
                except Exception:
                    pass
                _raise_store_unavailable(exc)
            self._client.write(f"{directory}/output.txt", output_bytes)
            verified = self._read_bytes(f"{directory}/output.txt")
            if len(verified) != metadata.original_bytes \
                    or hashlib.sha256(verified).hexdigest() != metadata.sha256:
                raise IOError("artifact output verification failed")
            # Commit point: readers ignore directories without metadata.json.
            self._client.write(
                f"{directory}/metadata.json",
                json.dumps(metadata.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8"),
            )
            try:
                self._client.rm(f"{directory}/_writing.json")
            except Exception:
                pass
        return False

    def read_all(self, artifact_id: str, session_id: str, ctx: RequestContext) -> tuple[ArtifactMetadata, str]:
        directory = self._dir(artifact_id, session_id, ctx)
        metadata = self._read_metadata(artifact_id, session_id, ctx)
        output_bytes = self._read_bytes(f"{directory}/output.txt")
        try:
            output = output_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactNotVisibleError("not_found_or_not_visible") from exc
        if len(output_bytes) != metadata.original_bytes \
                or hashlib.sha256(output_bytes).hexdigest() != metadata.sha256:
            raise ArtifactNotVisibleError("not_found_or_not_visible")
        return metadata, output

    def read_metadata(
        self,
        artifact_id: str,
        session_id: str,
        ctx: RequestContext,
    ) -> ArtifactMetadata:
        return self._read_metadata(artifact_id, session_id, ctx)

    def read_range(
        self,
        artifact_id: str,
        session_id: str,
        ctx: RequestContext,
        offset_bytes: int,
        length_bytes: int,
    ) -> dict:
        if offset_bytes < 0 or length_bytes < 1:
            raise ValueError("offset_bytes must be non-negative and length_bytes must be positive")
        directory = self._dir(artifact_id, session_id, ctx)
        metadata = self._read_metadata(artifact_id, session_id, ctx)
        base_offset = min(offset_bytes, metadata.original_bytes)
        try:
            raw = (
                bytes(self._client.read(
                    f"{directory}/output.txt",
                    offset=base_offset,
                    # Up to three leading continuation bytes may be skipped and
                    # up to three trailing bytes may finish the last codepoint.
                    size=length_bytes + 6,
                ))
                if base_offset < metadata.original_bytes
                else b""
            )
        except Exception as exc:
            _raise_read_error(exc)
        try:
            decoded = decode_utf8_range(raw, 0, length_bytes)
        except UnicodeDecodeError as exc:
            raise ArtifactNotVisibleError("not_found_or_not_visible") from exc
        decoded["actual_offset_bytes"] += base_offset
        decoded["next_offset_bytes"] += base_offset
        decoded["eof"] = decoded["next_offset_bytes"] >= metadata.original_bytes
        return {"metadata": metadata.to_dict(), "ref": metadata.ref, **decoded}

    def list(self, session_id: str, ctx: RequestContext, limit: int = 50, cursor: str | None = None) -> dict:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        try:
            entries = self._client.ls(self._base(session_id, ctx))
        except Exception as exc:
            if _is_explicit_not_found(exc):
                entries = []
            else:
                raise ArtifactStoreUnavailableError(
                    "tool-result artifact store is unavailable"
                ) from exc
        names = sorted(
            (entry.get("name", "") for entry in entries if entry.get("isDir", False)),
            reverse=True,
        )
        if cursor:
            try:
                start = names.index(cursor) + 1
            except ValueError:
                raise ValueError("invalid cursor") from None
        else:
            start = 0
        artifacts = []
        next_cursor = None
        truncated = False
        for position, name in enumerate(names[start:], start=start):
            try:
                metadata = self._read_metadata(name, session_id, ctx)
            except ArtifactNotVisibleError:
                continue
            artifacts.append(metadata.to_dict() | {"ref": metadata.ref})
            next_cursor = name
            if len(artifacts) >= limit:
                truncated = position + 1 < len(names)
                break
        return {
            "artifacts": artifacts,
            "truncated": truncated,
            "next_cursor": next_cursor if truncated else None,
        }

    def delete_session_artifacts(self, session_id: str, ctx: RequestContext) -> int:
        base = self._base(session_id, ctx)
        with self._lock:
            try:
                entries = self._client.ls(base)
            except Exception as exc:
                if _is_explicit_not_found(exc):
                    return 0
                _raise_store_unavailable(exc)
            artifact_ids = [
                entry.get("name", "")
                for entry in entries
                if entry.get("isDir", False)
                and _ARTIFACT_ID.fullmatch(entry.get("name", ""))
            ]
            try:
                self._client.rm(base, recursive=True)
            except Exception as exc:
                if not _is_explicit_not_found(exc):
                    _raise_store_unavailable(exc)
            return len(artifact_ids)

    def cleanup_stale_writes(
        self,
        session_id: str,
        ctx: RequestContext,
        grace_seconds: int,
        *,
        now: datetime | None = None,
    ) -> int:
        """Remove only old marker-only directories for the current owner tuple."""
        if grace_seconds < 1:
            raise ValueError("grace_seconds must be positive")
        now = now or datetime.now(UTC)
        try:
            entries = self._client.ls(self._base(session_id, ctx))
        except Exception as exc:
            if _is_explicit_not_found(exc):
                return 0
            _raise_store_unavailable(exc)
        deleted = 0
        for entry in entries:
            artifact_id = str(entry.get("name", ""))
            if not entry.get("isDir", False) or not _ARTIFACT_ID.fullmatch(artifact_id):
                continue
            directory = self._dir(artifact_id, session_id, ctx)
            # Metadata is the commit point. Never delete a directory whenever
            # metadata is visible, even if a best-effort marker removal failed.
            try:
                self._client.read(f"{directory}/metadata.json")
                continue
            except Exception as exc:
                if not _is_explicit_not_found(exc):
                    _raise_store_unavailable(exc)
            try:
                marker_raw = bytes(self._client.read(f"{directory}/_writing.json"))
            except Exception as exc:
                if _is_explicit_not_found(exc):
                    continue
                _raise_store_unavailable(exc)
            try:
                marker = json.loads(marker_raw.decode("utf-8"))
                started = datetime.fromisoformat(
                    str(marker.get("started_at", "")).replace("Z", "+00:00")
                )
                if started.tzinfo is None:
                    started = started.replace(tzinfo=UTC)
            except Exception:
                continue
            if (now - started).total_seconds() < grace_seconds:
                continue
            try:
                self._client.rm(directory, recursive=True)
            except Exception as exc:
                if _is_explicit_not_found(exc):
                    continue
                _raise_store_unavailable(exc)
            deleted += 1
        return deleted
