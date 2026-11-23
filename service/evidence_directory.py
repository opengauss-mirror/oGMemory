"""Source authority for directory summaries derived from deterministic cards."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from uuid import NAMESPACE_URL, uuid5

from core.errors import NodeNotFoundError
from core.evidence import digest
from core.models import OutboxEvent
from service.evidence_store import EvidenceStore


def source_refs(node):
    return node.metadata.get("evidence_source_refs", []) + (node.metadata.get("deterministic") or {}).get("source_refs", [])


def directory_lock(fs, uri, ctx):
    lock = getattr(type(fs), "memory_identity_lock", None)
    return lock(fs, uri.rstrip("/"), ctx) if callable(lock) else nullcontext()


def summary_source_refs(fs, node, ctx, seen=None):
    """Recover lineage when aggregating a legacy cached directory summary."""
    from service.evidence_lifecycle import ensure_sources_active

    ensure_sources_active(node, fs, ctx)
    if node.metadata.get("node_type") != "directory" or "evidence_source_refs" in node.metadata:
        return source_refs(node)
    seen = set() if seen is None else seen
    identity = node.uri.rstrip("/")
    if identity in seen:
        raise NodeNotFoundError(node.uri)
    seen.add(identity)
    # Serialize the cached body + descendant snapshot with legacy invalidation.
    with directory_lock(fs, node.uri, ctx):
        current = ensure_sources_active(fs.read_node(node.uri, ctx), fs, ctx)
        if current.abstract != node.abstract or current.overview != node.overview:
            raise NodeNotFoundError(node.uri)
        refs = []
        for uri in fs.list_children(node.uri, ctx):
            if uri.rstrip("/") != identity:
                refs.extend(summary_source_refs(fs, fs.read_node(uri, ctx), ctx, seen.copy()))
        return refs


@contextmanager
def source_publication_locks(store, node, ctx, *, lock_directory):
    with store.source_session_locks(source_refs(node), ctx):
        lock = (directory_lock(store.fs, node.uri, ctx)
                if lock_directory else nullcontext())
        with lock:
            yield


def read_publication_revision(fs, uri, ctx, revision):
    current = fs.read_node(uri, ctx)
    if current.metadata.get("version") != revision:
        raise NodeNotFoundError(uri)
    return current


@contextmanager
def publication_guard(fs, node, ctx, *, revision=None):
    """Keep all source locks through the final write, after model work finishes."""
    store = EvidenceStore(fs)
    refs = source_refs(node)
    with source_publication_locks(store, node, ctx, lock_directory=True):
        if not store.verify_refs(refs, ctx):
            raise NodeNotFoundError(node.uri)
        if revision is not None:
            current = read_publication_revision(fs, node.uri, ctx, revision)
            if (current.metadata.get("evidence_directory_invalidated")
                    or current.abstract != node.abstract or current.overview != node.overview
                    or source_refs(current) != refs):
                raise NodeNotFoundError(node.uri)
        yield


def invalidate_directories(store, ctx, outbox):
    """Scrub ancestors, including legacy summaries without source references."""
    manifest, _ = store.manifest(ctx)
    directories = set()
    for uri in manifest["supports"]:
        root = uri.split("/memories/", 1)[0] + "/memories/"
        parent = uri.rsplit("/", 1)[0] + "/"
        while parent.startswith(root):
            directories.add(parent)
            if parent == root:
                break
            parent = parent.rstrip("/").rsplit("/", 1)[0] + "/"
    for uri in sorted(directories):
        with store.session_lock(ctx), directory_lock(store.fs, uri, ctx):
            manifest, version = store.manifest(ctx)
            completed = set(manifest.get("directory_cleanup_completed", []))
            if uri in completed:
                continue
            try:
                node = store.fs.read_node(uri, ctx)
            except NodeNotFoundError:
                node = None
            # A newly rebuilt summary containing only active sources (or only
            # ordinary memories) is safe. Legacy summaries have no lineage key.
            safe = (node is not None and "evidence_source_refs" in node.metadata
                    and not node.metadata.get("evidence_directory_invalidated")
                    and store.verify_refs(source_refs(node), ctx))
            if not safe and node is not None:
                if not node.metadata.get("evidence_directory_invalidated"):
                    store.fs.write_node(replace(node, abstract="", overview="", content="", metadata={
                        "node_type": "directory", "evidence_source_refs": [],
                        "evidence_directory_invalidated": True,
                        "_expected_version": node.metadata.get("version", 0)}), ctx)
            rebuild = not safe and node is not None
        # Registration acquires the worker's processing lock. Never wait for it
        # while holding source/directory locks used by directory publication.
        if rebuild and outbox is not None:
            outbox.register_once(OutboxEvent(
                str(uuid5(NAMESPACE_URL, digest("evidence-directory-cleanup", store.session_uri(ctx), uri))),
                "ARCHIVE_CONTEXT", uri, {"evidence_directory_cleanup": True}, "PENDING"), ctx)
        with store.session_lock(ctx):
            manifest, version = store.manifest(ctx)
            if rebuild:
                manifest["directory_rebuild_pending"] = sorted(set(manifest.get("directory_rebuild_pending", [])) | {uri})
            completed = set(manifest.get("directory_cleanup_completed", []))
            completed.add(uri)
            manifest["directory_cleanup_completed"] = sorted(completed)
            store._write(store.session_uri(ctx), manifest, version, ctx)


def schedule_directory_rebuilds(store, ctx, outbox):
    """Rebuild from surviving memories only after leaf cleanup has finished."""
    if outbox is None:
        return
    with store.session_lock(ctx):
        manifest, _ = store.manifest(ctx)
        directories = manifest.get("directory_rebuild_pending", [])
    roots = [uri for uri in directories if not any(uri != parent and uri.startswith(parent) for parent in directories)]
    for uri in roots:
        outbox.register_once(OutboxEvent(
            str(uuid5(NAMESPACE_URL, digest("evidence-directory-rebuild", store.session_uri(ctx), uri))),
            "UPSERT_DIRECTORY", uri, {"evidence_rebuild_directories": directories}, "PENDING"), ctx)
    with store.session_lock(ctx):
        manifest, version = store.manifest(ctx)
        if directories:
            manifest["directory_rebuild_pending"] = sorted(set(manifest.get("directory_rebuild_pending", [])) - set(directories))
            store._write(store.session_uri(ctx), manifest, version, ctx)
