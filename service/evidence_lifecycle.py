"""Source revocation and canonical read protection, independent of active engine."""

import time
from dataclasses import replace
from itertools import chain
from uuid import NAMESPACE_URL, uuid5

from core.errors import NodeNotFoundError
from core.evidence import deterministic_payload, digest, evidence_uri
from core.models import OutboxEvent
from extraction.deterministic.extractor import render_records
from service.evidence_store import EvidenceStore, leaf_publication_lock


def ensure_sources_active(node, fs, ctx):
    if node.metadata.get("evidence_directory_invalidated"):
        raise NodeNotFoundError(node.uri)
    derived = node.metadata.get("evidence_source_refs", [])
    data = node.metadata.get("deterministic")
    if data is None and not derived:
        return node
    refs = data.get("source_refs", []) if data is not None else []
    if data is not None and not refs:
        raise NodeNotFoundError(node.uri)
    kind, _, owner = node.owner_space.partition(":")
    root = f"ctx://{ctx.account_id}/users/{owner}/memories"
    if kind != "user" or not (node.uri.rstrip("/") == root or node.uri.startswith(root + "/")):
        raise NodeNotFoundError(node.uri)
    owner_ctx = replace(ctx, user_id=owner)
    store = EvidenceStore(fs)
    if not store.verify_refs(chain(derived, refs), ctx, owner_ctx=owner_ctx):
        raise NodeNotFoundError(node.uri)
    return node


class EvidenceLifecycle:
    def __init__(self, fs, outbox=None):
        self.store = EvidenceStore(fs)
        self.outbox = outbox

    def _discover_unpublished(self, ctx):
        """Find pre-publication copies without changing the ingestion visibility point."""
        with self.store.session_lock(ctx):
            manifest, version = self.store.manifest(ctx)
            if manifest is None:
                if not any(list(self.store.iter_uris(evidence_uri(ctx, directory), ctx, 1))
                           for directory in ("sources", "windows")):
                    return None
                manifest = {"session_id": ctx.session_id, "status": "revoked",
                            "sources": [], "batches": {}, "supports": {}}
                self.store._write(self.store.session_uri(ctx), manifest, version, ctx)
        deadline = time.monotonic() + 20
        for directory, prefix in (("sources", "source"), ("windows", "window")):
            while True:
                with self.store.session_lock(ctx):
                    manifest, version = self.store.manifest(ctx)
                    if manifest.get(prefix + "_discovery_complete"):
                        break
                    uris = list(self.store.iter_uris(
                        evidence_uri(ctx, directory), ctx, 100,
                        after_uri=manifest.get(prefix + "_discovery_cursor")))
                    keys = set(manifest["sources"])
                    for uri in uris:
                        value, _ = self.store._read(uri, ctx)
                        if value is None:
                            continue
                        if directory == "sources":
                            source = value.get("source", {})
                            if (source.get("session_id") == ctx.session_id
                                    and source.get("account_id") == ctx.account_id
                                    and source.get("owner_space") == ctx.user_space_name()
                                    and uri == evidence_uri(ctx, "sources", source.get("source_key", ""))):
                                keys.add(source["source_key"])
                        elif (value.get("session_id") == ctx.session_id and value.get("batch_id")
                                and uri == evidence_uri(ctx, "windows", digest(
                                    ctx.account_id, ctx.user_space_name(), ctx.session_id, value["batch_id"]))):
                            manifest["batches"][value["batch_id"]] = value["sources"]
                    manifest["sources"] = sorted(keys)
                    manifest[prefix + "_discovery_complete"] = len(uris) < 100
                    if uris:
                        manifest[prefix + "_discovery_cursor"] = uris[-1]
                    self.store._write(self.store.session_uri(ctx), manifest, version, ctx)
                if time.monotonic() >= deadline and not manifest[prefix + "_discovery_complete"]:
                    raise TimeoutError("evidence source discovery pending; retry session deletion")
        return manifest

    def delete_session(self, ctx):
        self.store.revoke_session(ctx)
        manifest = self._discover_unpublished(ctx)
        if manifest is None:
            return {"sources_revoked": 0, "memories_cleaned": 0}
        from service.evidence_directory import invalidate_directories
        invalidate_directories(self.store, ctx, self.outbox)
        # Revoke first, then follow the same processing → source → target order as writes.
        cleaned = 0
        for uri in sorted(manifest["supports"]):
            with leaf_publication_lock(self.store, self.outbox, uri, ctx, self.store.fs.memory_identity_lock):
                manifest, version = self.store.manifest(ctx)
                completed = set(manifest.get("cleanup_completed", []))
                if uri in completed:
                    continue
                try:
                    node = self.store.fs.read_node(uri, ctx)
                except NodeNotFoundError:
                    node = None
                if node is not None and node.metadata.get("deterministic"):
                    data = node.metadata["deterministic"]
                    resolve = self.store.reference_reader(ctx)
                    remaining = [r for r in data["fact_records"]
                                 if all(resolve(ref) is not None for ref in r["source_refs"])]
                    if remaining:
                        if remaining != data["fact_records"]:
                            abstract, overview, content = render_records(node.metadata.get("object_name", node.category), remaining)
                            data = deterministic_payload(remaining, existing=data)
                            updated = replace(node, abstract=abstract, overview=overview, content=content,
                                metadata={**node.metadata, "deterministic": data,
                                          "_expected_version": node.metadata.get("version", 0)})
                            self.store.fs.write_node(updated, ctx)
                            node = self.store.fs.read_node(uri, ctx)
                        if self.outbox is not None:
                            event = self.outbox.build_write_event(node)
                            event.event_id = str(uuid5(NAMESPACE_URL, digest(
                                "evidence-cleanup", self.store.session_uri(ctx), uri, node.metadata.get("version", 0))))
                            self.outbox.register_once(event, ctx)
                    else:
                        if self.outbox is not None:
                            event = OutboxEvent(
                                event_id=str(uuid5(NAMESPACE_URL, digest("evidence-cleanup", self.store.session_uri(ctx), uri))),
                                event_type="ARCHIVE_CONTEXT", uri=uri, payload={}, status="PENDING")
                            self.outbox.register_once(event, ctx)
                        self.store.fs.archive_node(uri, ctx)
                completed.add(uri)
                manifest["cleanup_completed"] = sorted(completed)
                self.store._write(self.store.session_uri(ctx), manifest, version, ctx)
                cleaned += 1
        from service.evidence_directory import schedule_directory_rebuilds
        schedule_directory_rebuilds(self.store, ctx, self.outbox)
        # Remove copied message bodies, retaining durable revocation/control records.
        with self.store.session_lock(ctx):
            for key in manifest["sources"]:
                uri = evidence_uri(ctx, "sources", key)
                source, version = self.store._read(uri, ctx)
                if source is not None and source.get("status") != "revoked":
                    self.store._write(uri, {"status": "revoked", "source_key": key}, version, ctx)
            for batch_id in manifest["batches"]:
                window_uri = evidence_uri(ctx, "windows", digest(ctx.account_id, ctx.user_space_name(), ctx.session_id, batch_id))
                window, version = self.store._read(window_uri, ctx)
                if window is not None and window.get("status") != "revoked":
                    self.store._write(window_uri, {**window, "status": "revoked"}, version, ctx)
        return {"sources_revoked": len(manifest["sources"]), "memories_cleaned": cleaned}
