"""Code RESOURCE node construction and lifecycle orchestration.

This module owns the durable code-resource lifecycle contract.  It does not
reuse the generic memory/resource writer because code resources are
account-scoped and must coordinate ContextFS changes with durable Outbox delete
semantics.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from commit.outbox_store import OutboxPublishOutcomeUnknownError
from core.enums import ContextType
from core.errors import NodeNotFoundError
from core.models import ContextNode, OutboxEvent, RequestContext
from core.resource_identity import (
    ResourceIdentityError,
    compute_chunk_key,
    compute_file_key,
    compute_root_fingerprint,
    compute_workspace_root_id,
    digest_slug,
    guard_full_key_collision,
    normalize_resource_relative_path,
    resolve_root_slug,
    resource_owner_space,
    select_nodes_by_metadata,
    validate_root_binding,
)
from extraction.code_processor import (
    SUPPORTED_CODE_EXTENSIONS,
    CodeChunk,
    CodeFileChunker,
    detect_language,
    read_code_source,
    validate_language_hint,
)
from extraction.code_workspace import iter_code_files, normalize_language_hints

logger = logging.getLogger(__name__)

DEFAULT_CODE_RESOURCE_MAX_FILES = 5000
_ROOT_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_resource_root_slug(value: str) -> str:
    """Return a single-segment, deterministic root slug."""

    normalized = _ROOT_SLUG_UNSAFE.sub("-", str(value or "").strip()).strip("-._")
    if not normalized:
        raise ResourceIdentityError("resource root name must contain a safe character")
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ResourceIdentityError("resource root name must be a single URI segment")
    return normalized


def _graph_metadata(chunk: CodeChunk) -> dict[str, Any]:
    metadata = chunk.metadata or {}
    graph = metadata.get("graph")
    return dict(graph) if isinstance(graph, dict) else {}


def _bounded_abstract(chunk: CodeChunk) -> str:
    label = " ".join(part for part in (chunk.symbol_kind, chunk.symbol) if part).strip()
    return (label or f"{chunk.language} code")[:200]


def _build_overview(relative_path: str, chunk: CodeChunk) -> str:
    fields = [
        f"path: {relative_path}",
        f"language: {chunk.language}",
        f"lines: {chunk.start_line}-{chunk.end_line}",
    ]
    if chunk.signature:
        fields.append(f"signature: {chunk.signature}")
    graph = _graph_metadata(chunk)
    for key in ("imports", "calls", "extends", "contains"):
        values = graph.get(key)
        if isinstance(values, list) and values:
            fields.append(f"{key}: {', '.join(str(value) for value in values[:12])}")
    return "\n".join(fields)


def build_code_resource_nodes(
    *,
    account_id: str,
    workspace_root: str | Path,
    file_path: str | Path,
    resource_root_slug: str,
    root_fingerprint: str | None = None,
    tags: Iterable[str] = (),
    description: str | None = None,
    chunker: CodeFileChunker | None = None,
    language_hint: str | None = None,
    source_filter: Callable[[str], bool] | None = None,
    retain_file_source: bool = True,
) -> tuple[str, str, list[ContextNode]]:
    """Build the complete deterministic node set for one source file.

    Returns ``(file_key, file_slug, nodes)``.  No persistent writes occur in
    this function, which lets refresh finish construction before deleting any
    previous chunk.
    """

    root = Path(workspace_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("code workspace root is not a directory")
    source_path = Path(file_path).expanduser()
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve(strict=True)
    if not source_path.is_file():
        raise ValueError("code refresh path is not a regular file")

    relative_path = normalize_resource_relative_path(root, source_path)
    extension = source_path.suffix.lower()
    normalized_hint = validate_language_hint(language_hint)
    if extension not in SUPPORTED_CODE_EXTENSIONS and normalized_hint is None:
        raise ValueError("unsupported code source extension")

    fingerprint = root_fingerprint or compute_root_fingerprint(root)
    file_key = compute_file_key(fingerprint, resource_root_slug, relative_path)
    file_slug = digest_slug(file_key)
    language = detect_language(source_path, language_hint=normalized_hint)
    source = read_code_source(source_path)
    # Reuse the chunker source for request-local filtering without another read.
    if source_filter is not None and not source_filter(source):
        return file_key, file_slug, []
    chunks = (chunker or CodeFileChunker()).chunk(source, source_path, language)
    if not chunks:
        raise ValueError("code source produced no indexable chunks")

    normalized_tags = [str(tag) for tag in tags if str(tag).strip()]
    nodes: list[ContextNode] = []
    for chunk in chunks:
        chunk_key = compute_chunk_key(
            file_key,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            symbol=chunk.symbol,
            chunk_hash=chunk.chunk_hash,
        )
        chunk_slug = digest_slug(chunk_key)
        metadata: dict[str, Any] = {
            "resource_type": "code",
            "resource_category": "code",
            "resource_root_slug": resource_root_slug,
            "root_fingerprint": fingerprint,
            "file_key": file_key,
            "file_slug": file_slug,
            "relative_path": relative_path,
            "where": relative_path,
            "source_extension": extension,
            "language": chunk.language,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol": chunk.symbol,
            "symbol_kind": chunk.symbol_kind,
            "signature": chunk.signature,
            "chunk_hash": chunk.chunk_hash,
            "chunk_key": chunk_key,
            "graph": _graph_metadata(chunk),
        }
        for key in ("implementation_start_line", "symbol_start_line", "symbol_end_line", "partial_source_line"):
            if key in (chunk.metadata or {}):
                metadata[key] = chunk.metadata[key]
        if normalized_tags:
            metadata["tags"] = normalized_tags
        if description:
            metadata["description"] = str(description)
        nodes.append(
            ContextNode(
                uri=f"ctx://{account_id}/resources/{chunk_slug}",
                context_type=ContextType.RESOURCE.value,
                category="code",
                level=4,
                owner_space=resource_owner_space(account_id),
                abstract=_bounded_abstract(chunk),
                overview=_build_overview(relative_path, chunk),
                content=chunk.content,
                metadata=metadata,
            )
        )
    nodes.sort(key=lambda node: node.uri)
    if retain_file_source:
        # Store one bounded file snapshot outside index metadata; chunks may omit source.
        nodes[0].metadata["file_source"] = source
    return file_key, file_slug, nodes


class CodeResourceLifecycleService:
    """Coordinate initial ingestion and durable code RESOURCE mutations.

    ``bootstrap`` imports the parseable files currently visible in a workspace
    for initial population.  Registered workspace synchronization owns repeated
    convergence and uses ``refresh`` and ``delete_path`` for changed or removed
    files.
    """

    def __init__(self, *, context_fs, outbox_store, config) -> None:
        if context_fs is None:
            raise RuntimeError("storage backend unavailable for code resources")
        if outbox_store is None:
            raise RuntimeError("Outbox unavailable for code resources")
        self._fs = context_fs
        self._outbox = outbox_store
        self._cfg = config
        self._chunker = CodeFileChunker(
            target_chars=int(getattr(config, "code_resources_target_chars", 2500)),
            overlap_chars=int(getattr(config, "code_resources_overlap_chars", 300)),
            max_chars=int(getattr(config, "code_resources_chunk_max_chars", 4000)),
        )
        self._lock = threading.RLock()

    @staticmethod
    def _has_real_method(instance, name: str) -> bool:
        return callable(getattr(type(instance), name, None))

    def _require_write_enabled(self) -> None:
        if not bool(getattr(self._cfg, "code_resources_enabled", False)):
            raise ValueError("code resource writes are disabled")

    def _resource_nodes(self, ctx: RequestContext) -> list[ContextNode]:
        resource_root = f"ctx://{ctx.account_id}/resources/"
        child_uris = self._fs.list_children(resource_root, ctx)
        nodes: list[ContextNode] = []
        for child_uri in sorted(set(child_uris)):
            uri = child_uri.rstrip("/")
            if self._has_real_method(self._fs, "is_tombstoned") and self._fs.is_tombstoned(
                uri, ctx
            ):
                continue
            try:
                node = self._fs.read_node(uri, ctx)
            except NodeNotFoundError:
                continue
            if node.context_type == ContextType.RESOURCE.value:
                nodes.append(node)
        return nodes

    @staticmethod
    def _code_nodes(nodes: Iterable[ContextNode]) -> list[ContextNode]:
        return [
            node
            for node in nodes
            if node.context_type == ContextType.RESOURCE.value
            and node.metadata.get("resource_type") == "code"
        ]

    @staticmethod
    def _default_root_slug(root: Path, fingerprint: str) -> str:
        candidate = root.name or f"root-{fingerprint[:12]}"
        return normalize_resource_root_slug(candidate)

    def _resolve_binding(
        self,
        *,
        root: Path,
        name: str | None,
        ctx: RequestContext,
        existing_code_nodes: list[ContextNode],
    ) -> tuple[str, str]:
        fingerprint = compute_root_fingerprint(root)
        if name is not None:
            root_slug = normalize_resource_root_slug(name)
        else:
            reserved = self._read_backend_binding(
                ctx=ctx,
                root_fingerprint=fingerprint,
            )
            root_slug = (
                reserved[1]
                if reserved is not None
                else resolve_root_slug(
                    existing_code_nodes,
                    root_fingerprint=fingerprint,
                )
                or self._default_root_slug(root, fingerprint)
            )
        return self._reserve_binding(
            ctx=ctx,
            root_slug=root_slug,
            root_fingerprint=fingerprint,
            existing_code_nodes=existing_code_nodes,
        )

    def _read_backend_binding(
        self,
        *,
        ctx: RequestContext,
        root_slug: str | None = None,
        root_fingerprint: str | None = None,
    ) -> tuple[str, str, str] | None:
        if not self._has_real_method(self._fs, "read_code_workspace_root_binding"):
            return None
        binding = self._fs.read_code_workspace_root_binding(
            ctx=ctx,
            root_slug=root_slug,
            root_fingerprint=root_fingerprint,
        )
        if binding is None:
            return None
        if (
            not isinstance(binding, tuple)
            or len(binding) != 3
            or not all(isinstance(value, str) and value for value in binding)
        ):
            raise RuntimeError("storage backend returned an invalid workspace root binding")
        return binding

    def _reserve_binding(
        self,
        *,
        ctx: RequestContext,
        root_slug: str,
        root_fingerprint: str,
        existing_code_nodes: list[ContextNode],
    ) -> tuple[str, str]:
        compute_workspace_root_id(ctx.account_id, root_fingerprint)
        validate_root_binding(
            existing_code_nodes,
            root_slug=root_slug,
            root_fingerprint=root_fingerprint,
        )
        if self._has_real_method(self._fs, "reserve_code_workspace_root_binding"):
            binding = self._fs.reserve_code_workspace_root_binding(
                ctx=ctx,
                root_slug=root_slug,
                root_fingerprint=root_fingerprint,
            )
            if (
                not isinstance(binding, tuple)
                or len(binding) != 3
                or binding[1:] != (root_slug, root_fingerprint)
            ):
                raise RuntimeError("storage backend reserved an invalid workspace root binding")
        return root_slug, root_fingerprint

    def reserve_workspace_binding(
        self,
        *,
        root_slug: str,
        root_fingerprint: str,
        ctx: RequestContext,
    ) -> tuple[str, str]:
        """Reserve a binding without requiring the physical root to be accessible."""

        normalized_slug = normalize_resource_root_slug(root_slug)
        with self._lock:
            return self._reserve_binding(
                ctx=ctx,
                root_slug=normalized_slug,
                root_fingerprint=root_fingerprint,
                existing_code_nodes=self._code_nodes(self._resource_nodes(ctx)),
            )

    def resolve_reserved_workspace_binding(
        self,
        *,
        ctx: RequestContext,
        root_slug: str | None = None,
        root_fingerprint: str | None = None,
    ) -> tuple[str, str] | None:
        """Resolve a reserved binding by slug or fingerprint without filesystem access."""

        if (root_slug is None) == (root_fingerprint is None):
            raise ValueError("exactly one root binding lookup key is required")
        normalized_slug = (
            normalize_resource_root_slug(root_slug) if root_slug is not None else None
        )
        if root_fingerprint is not None:
            compute_workspace_root_id(ctx.account_id, root_fingerprint)
        with self._lock:
            code_nodes = self._code_nodes(self._resource_nodes(ctx))
            binding = self._read_backend_binding(
                ctx=ctx,
                root_slug=normalized_slug,
                root_fingerprint=root_fingerprint,
            )
            if binding is not None:
                _, resolved_slug, resolved_fingerprint = binding
                validate_root_binding(
                    code_nodes,
                    root_slug=resolved_slug,
                    root_fingerprint=resolved_fingerprint,
                )
                return resolved_slug, resolved_fingerprint
            if root_fingerprint is not None:
                historical_slug = resolve_root_slug(
                    code_nodes,
                    root_fingerprint=root_fingerprint,
                )
                if historical_slug is None:
                    return None
                validate_root_binding(
                    code_nodes,
                    root_slug=historical_slug,
                    root_fingerprint=root_fingerprint,
                )
                return historical_slug, root_fingerprint
            fingerprints = {
                str(node.metadata.get("root_fingerprint"))
                for node in code_nodes
                if node.metadata.get("resource_root_slug") == normalized_slug
                and isinstance(node.metadata.get("root_fingerprint"), str)
                and node.metadata.get("root_fingerprint")
            }
            if not fingerprints:
                return None
            if len(fingerprints) != 1:
                raise ResourceIdentityError(
                    "resource root slug is bound to multiple root fingerprints"
                )
            historical_fingerprint = next(iter(fingerprints))
            validate_root_binding(
                code_nodes,
                root_slug=str(normalized_slug),
                root_fingerprint=historical_fingerprint,
            )
            return str(normalized_slug), historical_fingerprint

    def resolve_workspace_binding(
        self,
        *,
        workspace_root: str | Path,
        ctx: RequestContext,
        name: str | None = None,
    ) -> tuple[str, str]:
        """Resolve and reserve the canonical binding before constructing source nodes."""

        try:
            root = Path(workspace_root).expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError("code workspace was not found") from exc
        if not root.is_dir():
            raise ValueError("workspace_root must be a directory")
        with self._lock:
            return self._resolve_binding(
                root=root,
                name=name,
                ctx=ctx,
                existing_code_nodes=self._code_nodes(self._resource_nodes(ctx)),
            )

    def _prepare_node_write(
        self,
        node: ContextNode,
        ctx: RequestContext,
    ) -> bool:
        """Validate deterministic identity and attach its version precondition."""

        if self._has_real_method(self._fs, "is_tombstoned") and self._fs.is_tombstoned(
            node.uri, ctx
        ):
            raise RuntimeError("code resource URI is pending durable deletion")
        if not self._fs.exists(node.uri, ctx):
            return False
        existing = self._fs.read_node(node.uri, ctx)
        guard_full_key_collision(
            existing.metadata,
            key_field="chunk_key",
            expected_key=str(node.metadata["chunk_key"]),
            uri=node.uri,
        )
        current_version = existing.metadata.get("version")
        if (
            isinstance(current_version, bool)
            or not isinstance(current_version, int)
            or current_version < 1
        ):
            raise RuntimeError("existing code resource version is invalid")
        node.metadata["_expected_version"] = current_version
        return True

    def _write_node(self, node: ContextNode, ctx: RequestContext) -> None:
        """Publish one code node only through the backend atomic protocol.

        A backend that cannot commit the node and its UPSERT event in one
        transaction is rejected.  Publishing the node before registering its
        event would leave durable content without corresponding index work.
        """
        if not self._has_real_method(self._fs, "write_node_with_outbox"):
            raise RuntimeError(
                "code resource backend requires atomic Outbox writes"
            )
        if not self._has_real_method(self._outbox, "build_write_event"):
            raise RuntimeError(
                "code resource backend requires atomic Outbox event construction"
            )
        event = self._outbox.build_write_event(node)
        self._fs.write_node_with_outbox(node, ctx, event)

    def _write_nodes(
        self,
        nodes: Iterable[ContextNode],
        ctx: RequestContext,
    ) -> tuple[int, int]:
        created = 0
        updated = 0
        for node in nodes:
            existed = self._prepare_node_write(node, ctx)
            if not existed:
                node.metadata["_expected_version"] = 0
            self._write_node(node, ctx)
            if existed:
                updated += 1
            else:
                created += 1
        return created, updated

    def _build_delete_events(self, uris: Iterable[str]) -> list[OutboxEvent]:
        if not self._has_real_method(self._outbox, "build_delete_event"):
            return []
        return [self._outbox.build_delete_event(uri) for uri in uris]

    def _delete_nodes(self, uris: Iterable[str], ctx: RequestContext) -> int:
        targets = sorted(set(uri.rstrip("/") for uri in uris if uri))
        if not targets:
            return 0

        if self._has_real_method(self._fs, "delete_nodes_with_outbox"):
            events = self._build_delete_events(targets)
            if len(events) != len(targets):
                raise RuntimeError("SQL durable delete requires pre-built Outbox events")
            return int(self._fs.delete_nodes_with_outbox(targets, ctx, events))

        required_fs_methods = (
            "resource_mutation_lock",
            "is_tombstoned",
            "write_delete_tombstone",
            "clear_delete_tombstone",
            "delete_node_payload",
        )
        required_outbox_methods = (
            "build_delete_event",
            "persist_event",
        )
        if not all(
            self._has_real_method(self._fs, method)
            for method in required_fs_methods
        ) or not all(
            self._has_real_method(self._outbox, method)
            for method in required_outbox_methods
        ):
            raise RuntimeError(
                "code resource backend requires atomic Outbox deletes"
            )

        deleted = 0
        for uri in targets:
            with self._fs.resource_mutation_lock(uri, ctx) as mutation_lease:
                mutation_lease.ensure_owned()
                if self._fs.is_tombstoned(uri, ctx):
                    continue
                if not self._fs.exists(uri, ctx):
                    continue

                event = self._outbox.build_delete_event(uri)
                event.payload["durable_delete"] = True
                tombstone_payload = {
                    "delete_event_id": event.event_id,
                    "record_ids": list(event.payload.get("ids_to_delete") or []),
                }
                self._fs.write_delete_tombstone(
                    uri,
                    ctx,
                    tombstone_payload,
                    mutation_lease=mutation_lease,
                )
                mutation_lease.ensure_owned()
                try:
                    self._outbox.persist_event(event, uri)
                except OutboxPublishOutcomeUnknownError:
                    # The event may already be durable. Retain the tombstone
                    # so no later writer can race the uncertain DELETE.
                    raise
                except Exception:
                    self._fs.clear_delete_tombstone(
                        uri,
                        ctx,
                        mutation_lease=mutation_lease,
                    )
                    raise
                mutation_lease.ensure_owned()
                self._fs.delete_node_payload(
                    uri,
                    ctx,
                    mutation_lease=mutation_lease,
                )
                mutation_lease.ensure_owned()
                deleted += 1
        return deleted

    def bootstrap(
        self,
        *,
        path: str,
        ctx: RequestContext,
        name: str | None = None,
        tags: Iterable[str] = (),
        description: str | None = None,
        max_files: int | None = None,
        language_hints: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Import the current parseable files for initial workspace population.

        This operation writes the files found by the bounded scan.  It does not
        reconcile nodes for files absent from that scan; registered workspace
        synchronization owns repeated snapshot convergence.
        """

        self._require_write_enabled()
        with self._lock:
            try:
                root = Path(path).expanduser().resolve(strict=True)
            except FileNotFoundError as exc:
                raise FileNotFoundError("code workspace was not found") from exc
            if not root.is_dir():
                raise ValueError("code workspace path must be a directory")

            all_nodes = self._resource_nodes(ctx)
            code_nodes = self._code_nodes(all_nodes)
            configured_limit = int(
                getattr(self._cfg, "code_resources_max_files", DEFAULT_CODE_RESOURCE_MAX_FILES)
            )
            limit = configured_limit if max_files is None else int(max_files)
            if limit < 1:
                raise ValueError("max_files must be a positive integer")
            if limit > configured_limit:
                raise ValueError("max_files exceeds configured cap")
            normalized_hints = normalize_language_hints(language_hints)
            root_slug, fingerprint = self._resolve_binding(
                root=root,
                name=name,
                ctx=ctx,
                existing_code_nodes=code_nodes,
            )

            files_scanned = 0
            files_skipped = 0
            created = 0
            updated = 0
            for source_path in iter_code_files(
                root,
                max_files=limit,
                language_hints=normalized_hints,
            ):
                files_scanned += 1
                try:
                    relative_path = normalize_resource_relative_path(root, source_path)
                    _, _, nodes = build_code_resource_nodes(
                        account_id=ctx.account_id,
                        workspace_root=root,
                        file_path=source_path,
                        resource_root_slug=root_slug,
                        root_fingerprint=fingerprint,
                        tags=tags,
                        description=description,
                        chunker=self._chunker,
                        language_hint=normalized_hints.get(relative_path),
                    )
                except (OSError, UnicodeError, ValueError) as exc:
                    files_skipped += 1
                    logger.warning(
                        "code_resource_file_skipped account=%s relative_path=%s reason=%s trace=%s",
                        ctx.account_id,
                        normalize_resource_relative_path(root, source_path),
                        type(exc).__name__,
                        ctx.trace_id,
                    )
                    continue
                node_created, node_updated = self._write_nodes(nodes, ctx)
                created += node_created
                updated += node_updated

            if created + updated == 0:
                raise ValueError("code workspace contains no parseable code")
            return {
                "uri": f"ctx://{ctx.account_id}/resources/{root_slug}",
                "status": "PENDING",
                "files_scanned": files_scanned,
                "files_skipped": files_skipped,
                "nodes_created": created,
                "nodes_updated": updated,
                "message": "Code directory ingested",
            }

    def refresh(
        self,
        *,
        path: str,
        workspace_root: str,
        ctx: RequestContext,
        name: str | None = None,
        tags: Iterable[str] = (),
        description: str | None = None,
        language_hint: str | None = None,
    ) -> dict[str, Any]:
        self._require_write_enabled()
        normalized_hint = validate_language_hint(language_hint)
        with self._lock:
            try:
                root = Path(workspace_root).expanduser().resolve(strict=True)
            except FileNotFoundError as exc:
                raise FileNotFoundError("code workspace was not found") from exc
            if not root.is_dir():
                raise ValueError("workspace_root must be a directory")

            relative_path = normalize_resource_relative_path(root, path)
            all_nodes = self._resource_nodes(ctx)
            code_nodes = self._code_nodes(all_nodes)
            root_slug, fingerprint = self._resolve_binding(
                root=root,
                name=name,
                ctx=ctx,
                existing_code_nodes=code_nodes,
            )
            file_key = compute_file_key(fingerprint, root_slug, relative_path)
            file_slug = digest_slug(file_key)
            previous = select_nodes_by_metadata(code_nodes, file_key=file_key)
            previous_uris = {node.uri for node in previous}

            requested_path = Path(path).expanduser()
            if not requested_path.is_absolute():
                requested_path = root / requested_path
            if not requested_path.exists():
                deleted = self._delete_nodes(previous_uris, ctx)
                return {
                    "uri": f"ctx://{ctx.account_id}/resources/{file_slug}",
                    "status": "PENDING",
                    "nodes_deleted": deleted,
                    "nodes_created": 0,
                    "nodes_updated": 0,
                    "message": "Code file removed",
                }

            _, _, new_nodes = build_code_resource_nodes(
                account_id=ctx.account_id,
                workspace_root=root,
                file_path=requested_path,
                resource_root_slug=root_slug,
                root_fingerprint=fingerprint,
                tags=tags,
                description=description,
                chunker=self._chunker,
                language_hint=normalized_hint,
            )
            created, updated = self._write_nodes(new_nodes, ctx)
            new_uris = {node.uri for node in new_nodes}
            deleted = self._delete_nodes(previous_uris - new_uris, ctx)
            return {
                "uri": f"ctx://{ctx.account_id}/resources/{file_slug}",
                "status": "PENDING",
                "nodes_deleted": deleted,
                "nodes_created": created,
                "nodes_updated": updated,
                "message": "Code file refreshed",
            }

    def delete_path(
        self,
        *,
        path: str,
        workspace_root: str,
        ctx: RequestContext,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Durably remove one known file identity without reading its source.

        Sync uses this for a file that was previously imported but is now
        ignored or unsupported while it still exists on disk.  The operation
        shares the lifecycle's root binding and durable delete ordering.
        """

        self._require_write_enabled()
        with self._lock:
            try:
                root = Path(workspace_root).expanduser().resolve(strict=True)
            except FileNotFoundError as exc:
                raise FileNotFoundError("code workspace was not found") from exc
            if not root.is_dir():
                raise ValueError("workspace_root must be a directory")

            relative_path = normalize_resource_relative_path(root, path)
            all_nodes = self._resource_nodes(ctx)
            code_nodes = self._code_nodes(all_nodes)
            root_slug, fingerprint = self._resolve_binding(
                root=root,
                name=name,
                ctx=ctx,
                existing_code_nodes=code_nodes,
            )
            file_key = compute_file_key(fingerprint, root_slug, relative_path)
            file_slug = digest_slug(file_key)
            previous = select_nodes_by_metadata(code_nodes, file_key=file_key)
            deleted = self._delete_nodes((node.uri for node in previous), ctx)
            return {
                "uri": f"ctx://{ctx.account_id}/resources/{file_slug}",
                "status": "PENDING",
                "nodes_deleted": deleted,
                "nodes_created": 0,
                "nodes_updated": 0,
                "message": "Code file removed",
            }

    def delete(self, *, slug: str, ctx: RequestContext) -> dict[str, Any]:
        with self._lock:
            normalized_slug = str(slug or "").strip()
            if not normalized_slug or "/" in normalized_slug or "\\" in normalized_slug:
                raise ValueError("resource slug must be a single URI segment")

            all_nodes = self._resource_nodes(ctx)
            code_nodes = self._code_nodes(all_nodes)
            targets = [
                node
                for node in code_nodes
                if node.metadata.get("file_slug") == normalized_slug
                or node.metadata.get("resource_root_slug") == normalized_slug
            ]
            if not targets:
                exact_uri = f"ctx://{ctx.account_id}/resources/{normalized_slug}"
                if any(node.uri.rstrip("/") == exact_uri for node in all_nodes):
                    raise ValueError("resource is not a code resource")
                raise FileNotFoundError("code resource was not found")

            deleted = self._delete_nodes((node.uri for node in targets), ctx)
            return {
                "uri": f"ctx://{ctx.account_id}/resources/{normalized_slug}",
                "status": "PENDING",
                "nodes_deleted": deleted,
                "message": "Code resource deletion accepted",
            }


__all__ = [
    "CodeResourceLifecycleService",
    "build_code_resource_nodes",
    "normalize_resource_root_slug",
]
