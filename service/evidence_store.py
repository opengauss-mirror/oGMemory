"""User-scoped evidence projections using existing ContextFS versions and locks.

A manifest publication is the visibility point. Interrupted writes can be replayed;
revocation remains durable even after original session data has been removed.
"""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from functools import lru_cache
from itertools import islice

from core.errors import (
    AccessDeniedError,
    NodeNotFoundError,
    OperationDeadlineExceededError,
    ValidationError,
)
from core.evidence import content_hash, digest, evidence_uri, message_source, reassemble_messages
from core.evidence_timing import stage, timed
from core.models import ContextNode, RequestContext
from core.operation_deadline import check_deadline, deadline_expired


def snapshot_messages(snapshot, ctx, fs):
    """Freeze source order across archive batches; reused message IDs retain order."""
    messages = [{"id": message.id, "role": message.role, "content": message.content,
                 "recorded_at": message.created_at} for message in snapshot
                if message.role in {"user", "assistant"} and message.content.strip()]
    result, _ = EvidenceStore(fs).reserve_messages(messages, ctx)
    return result


@contextmanager
def leaf_publication_lock(store, outbox, uri, ctx, target_lock):
    """Acquire leaf locks in the same order for writes and revocation."""
    processing = (outbox.processing_lock(uri) if callable(
        getattr(type(outbox), "processing_lock", None)) else nullcontext())
    with processing, store.session_lock(ctx), target_lock(uri.rsplit("/", 1)[0], ctx):
        yield


class EvidenceStore:
    def __init__(self, fs):
        self.fs = fs

    @contextmanager
    def source_session_locks(self, refs, ctx):
        with ExitStack() as stack:
            for session in sorted({ref["session_id"] for ref in refs}):
                stack.enter_context(self.session_lock(ctx, session_id=session))
            yield

    @contextmanager
    def session_lock(self, ctx: RequestContext, *, session_id=None):
        lock = getattr(type(self.fs), "memory_identity_lock", None)
        if not callable(lock):
            raise RuntimeError("evidence requires ContextFS cross-process identity locks")
        source_ctx = replace(ctx, session_id=session_id) if session_id is not None else ctx
        with lock(self.fs, self.session_uri(source_ctx), ctx):
            yield

    @staticmethod
    def session_uri(ctx):
        if not ctx.session_id:
            raise ValidationError("session_id", "evidence requires a stable session")
        return evidence_uri(ctx, "sessions", digest(ctx.account_id, ctx.user_space_name(), ctx.session_id))

    @timed("node_read")
    def _read(self, uri, ctx, *, owner_ctx=None):
        try:
            node = self.fs.read_node(uri, ctx)
        except NodeNotFoundError:
            return None, 0
        return self._decode(node, owner_ctx or ctx)

    @staticmethod
    def _decode(node, ctx):
        if isinstance(node, Exception):
            if isinstance(node, NodeNotFoundError):
                return None, 0
            raise node
        if node.owner_space != ctx.user_space_name() or node.category != "evidence":
            raise AccessDeniedError(node.uri, ctx.account_id, "invalid evidence owner or node type")
        with stage("json_parse"):
            value = json.loads(node.content)
        if value.get("schema_version") != 1:
            raise ValidationError("evidence", "unsupported evidence schema")
        return value, node.metadata.get("version", 0)

    def iter_uris(self, directory, ctx, limit, *, after_uri=None):
        """Yield bounded child URIs in storage order, resuming after a prior URI."""
        fs = self.fs
        pager = getattr(type(fs), "list_node_summaries_page", None)
        if callable(pager):
            cursor = getattr(fs, "list_node_order_key", lambda uri: uri)(after_uri) if after_uri else None
            count = 0
            while count < limit:
                with stage("list"):
                    page = pager(fs, directory, ctx, limit=min(100, limit - count), after_uri=cursor)
                if not page:
                    return
                for node in page:
                    yield node.uri
                    count += 1
                cursor = getattr(fs, "list_node_order_key", lambda uri: uri)(page[-1].uri)
        else:
            yield from [uri for uri in sorted(fs.list_children(directory, ctx))
                        if after_uri is None or uri > after_uri][:limit]

    def node_pages(self, uris, ctx, *, read_limit, deadline=None):
        """Bounded prefetch only; the caller still admits items under its budgets."""
        reader = getattr(type(self.fs), "read_nodes", None)
        iterator = iter(uris)
        remaining = read_limit
        while remaining > 0:
            page = list(islice(iterator, min(32, remaining) if callable(reader) else 1))
            if not page:
                return
            if deadline_expired(deadline):
                yield [(page[0], None)]  # Caller observes its deadline before resolving.
                return
            with stage("node_batch_read"):
                if callable(reader):
                    nodes = reader(self.fs, page, ctx)
                else:
                    try:
                        nodes = {page[0]: self.fs.read_node(page[0], ctx)}
                    except Exception as exc:
                        nodes = {page[0]: exc}
            yield [(uri, nodes[uri]) for uri in page]
            remaining -= len(page)
        # Let the caller observe its existing limit sentinel without reading it.
        for uri in islice(iterator, 1):
            yield [(uri, None)]

    def iter_sources(self, uris, ctx, *, read_limit, deadline=None):
        if not callable(getattr(type(self.fs), "read_nodes", None)):
            for uri in uris:
                yield uri, lambda uri=uri: self.read_source(uri.rsplit("/", 1)[1], ctx)
            return
        for page in self.node_pages(uris, ctx, read_limit=read_limit, deadline=deadline):
            values, sessions = {}, {}
            for uri, node in page:
                if node is None:
                    values[uri] = None
                    continue
                try:
                    value, version = self._decode(node, ctx)
                    values[uri] = (value, version)
                    if value is not None and value.get("status") == "active":
                        session_ctx = replace(ctx, session_id=value["source"]["session_id"])
                        sessions[self.session_uri(session_ctx)] = session_ctx
                except Exception as exc:
                    values[uri] = exc
            manifests = {}
            for manifest_page in self.node_pages(sessions, ctx, read_limit=len(sessions), deadline=deadline):
                for uri, node in manifest_page:
                    try:
                        value, _ = self._decode(node, ctx)
                        if value is not None and value.get("session_id") != sessions[uri].session_id:
                            raise ValidationError("evidence", "manifest session mismatch")
                        manifests[uri] = value
                    except Exception as exc:
                        manifests[uri] = exc

            def manifest_for(request_ctx, *, session_id, manifests=manifests):
                value = manifests[self.session_uri(replace(request_ctx, session_id=session_id))]
                if isinstance(value, Exception):
                    raise value
                return value, 0

            def resolve(uri, values=values, manifest_for=manifest_for):
                value = values[uri]
                if isinstance(value, Exception):
                    raise value
                if value is None:
                    return None
                return self._validate_source(uri.rsplit("/", 1)[1], *value, ctx, manifest_for)

            for uri, _ in page:
                yield uri, lambda uri=uri, resolve=resolve: resolve(uri)

    def _write(self, uri, value, version, ctx):
        self.fs.write_node(ContextNode(
            uri=uri, context_type="RESOURCE", category="evidence", level=2,
            owner_space=ctx.user_space_name(), abstract="", overview="",
            content=json.dumps({**value, "schema_version": 1}, ensure_ascii=False, sort_keys=True),
            metadata={"indexing_disabled": True, "_expected_version": version},
        ), ctx)

    def manifest(self, ctx, *, session_id=None, owner_ctx=None):
        source_ctx = owner_ctx or ctx
        if session_id is not None:
            source_ctx = replace(source_ctx, session_id=session_id)
        data, version = self._read(self.session_uri(source_ctx), ctx, owner_ctx=owner_ctx)
        if data is not None and data.get("session_id") != source_ctx.session_id:
            raise ValidationError("evidence", "manifest session mismatch")
        return data, version

    def _message_order(self, ctx, manifest):
        """Read under session_lock; reservations do not publish source bodies."""
        if manifest is not None and manifest["status"] != "active":
            raise ValidationError("source", "session evidence has been revoked")
        uri = evidence_uri(ctx, "lifecycle", digest("evidence-message-order-v1", self.session_uri(ctx)))
        value, version = self._read(uri, ctx)
        positions = (value or {}).get("positions", {}).copy()
        # Seed older sessions that predate durable reservations.
        for key in (manifest or {}).get("sources", []) if value is None else ():
            source = self.read_source(key, ctx)
            if source is not None:
                positions[source["message_id"]] = {"sequence": source["sequence"],
                                                   "verified": source["sequence_verified"]}
        return uri, value, version, positions

    def reserve_messages(self, messages, ctx, *, historical=False):
        """Atomically freeze order shared by live snapshots and archive backfill."""
        with self.session_lock(ctx):
            manifest, _ = self.manifest(ctx)
            uri, previous, version, positions = self._message_order(ctx, manifest)
            occupied = {item["sequence"] for item in positions.values()}
            next_sequence = max(occupied, default=-1) + 1
            result, unpositioned = [], []
            for message in messages:
                identity = message.get("id") or message.get("message_id")
                if identity not in positions:
                    position = message.get("sequence") if historical else next_sequence
                    verified = type(position) is int and position >= 0 and position not in occupied
                    if not verified:
                        position = next_sequence
                    positions[identity] = {"sequence": position, "verified": verified}
                    occupied.add(position)
                    next_sequence = max(next_sequence, position + 1)
                item = positions[identity]
                original = {**message, "sequence": item["sequence"]}
                message_source(original, ctx)  # Validate the entire batch before reserving.
                result.append(original)
                if not item["verified"]:
                    unpositioned.append(identity)
            value = {"positions": positions, "schema_version": 1}
            if value != previous:
                self._write(uri, value, version, ctx)
            return result, unpositioned

    def register(self, messages: list[dict], ctx: RequestContext, *, batch_id: str, archive_id: str | None = None,
                 unpositioned_ids=()) -> list[dict]:
        """Publish a frozen batch. No embeddings, LLM calls or ordinary index events."""
        if not isinstance(batch_id, str) or not batch_id:
            raise ValidationError("batch_id", "a durable source batch ID is required")
        unpositioned_ids = set(unpositioned_ids)
        sources = [message_source(message, ctx).to_dict()
                   for message in reassemble_messages(messages)]
        # JSON-normalize tuples before comparing immutable persisted payloads.
        sources = json.loads(json.dumps(sources, ensure_ascii=False))
        window_key = digest(ctx.account_id, ctx.user_space_name(), ctx.session_id, batch_id)
        with self.session_lock(ctx):
            manifest, version = self.manifest(ctx)
            if manifest is None:
                manifest = {"session_id": ctx.session_id, "status": "active",
                            "sources": [], "batches": {}, "supports": {}}
            if manifest["status"] != "active":
                raise ValidationError("source", "session evidence has been revoked")
            keys = [source["source_key"] for source in sources]
            if batch_id in manifest["batches"] and manifest["batches"][batch_id] != keys:
                raise ValidationError("source", "frozen source batch changed")
            order_uri, previous_order, order_version, positions = self._message_order(ctx, manifest)
            occupied = {item["sequence"]: identity for identity, item in positions.items()}
            for source in sources:
                identity, sequence = source["message_id"], source["sequence"]
                if (identity in positions and positions[identity]["sequence"] != sequence
                        or sequence in occupied and occupied[sequence] != identity):
                    raise ValidationError("sequence", "source sequence conflicts with a frozen message position")
                positions.setdefault(identity, {"sequence": sequence, "verified": identity not in unpositioned_ids})
                occupied[sequence] = identity
                if not positions[identity]["verified"]:
                    unpositioned_ids.add(identity)
            # Check every immutable source before the first write.
            pending = []
            for source in sources:
                uri = evidence_uri(ctx, "sources", source["source_key"])
                current, source_version = self._read(uri, ctx)
                if current is not None:
                    if current.get("source") != source or current.get("status") != "active":
                        raise ValidationError("source", "source identity conflict or revocation")
                    if archive_id and archive_id not in current.get("archive_ids", []):
                        pending.append((uri, source, source_version))
                else:
                    pending.append((uri, source, source_version))
            order = {"positions": positions, "schema_version": 1}
            if order != previous_order:
                self._write(order_uri, order, order_version, ctx)
            for uri, source, source_version in pending:
                current, _ = self._read(uri, ctx)
                archive_ids = set((current or {}).get("archive_ids", []))
                if archive_id:
                    archive_ids.add(archive_id)
                order_unknown = (current.get("sequence_verified") is False if current is not None
                                 else source["message_id"] in unpositioned_ids)
                # Synthetic slots provide stable replay identities, not message adjacency.
                value = {"source": source, "status": "active", "archive_ids": sorted(archive_ids)}
                if order_unknown:
                    value["sequence_verified"] = False
                self._write(uri, value, source_version, ctx)
            window_uri = evidence_uri(ctx, "windows", window_key)
            window, window_version = self._read(window_uri, ctx)
            expected = {"batch_id": batch_id, "session_id": ctx.session_id, "sources": keys,
                        "status": "active", "schema_version": 1}
            if window is not None and window != expected:
                raise ValidationError("source", "frozen evidence window changed")
            if window is None:
                self._write(window_uri, expected, window_version, ctx)
            if batch_id not in manifest["batches"]:
                manifest["sources"] = sorted(set(manifest["sources"]) | set(keys))
                manifest["batches"][batch_id] = keys
                self._write(self.session_uri(ctx), manifest, version, ctx)
        return sources

    @timed("source_validate")
    def read_source(self, key: str, ctx: RequestContext, *, owner_ctx=None, deadline=None) -> dict | None:
        def check_read_budget():
            if deadline_expired(deadline):
                raise OperationDeadlineExceededError("Evidence source read deadline exceeded")

        # The owner locates evidence; only the original caller authorizes reads.
        source_ctx = owner_ctx or ctx
        check_read_budget()
        value, version = self._read(evidence_uri(source_ctx, "sources", key), ctx, owner_ctx=owner_ctx)

        def manifest_for(source_ctx, *, session_id):
            check_read_budget()
            return self.manifest(ctx, session_id=session_id, owner_ctx=source_ctx)

        return self._validate_source(key, value, version, source_ctx, manifest_for)

    def _validate_source(self, key, value, version, ctx, manifest_reader):
        if value is None or value.get("status") != "active":
            return None
        source = value["source"]
        if (source["account_id"] != ctx.account_id or source["owner_space"] != ctx.user_space_name()
                or source["source_key"] != key
                or digest(ctx.account_id, ctx.user_space_name(), source["session_id"], source["message_id"]) != key
                or content_hash(source["text"]) != source["content_hash"]):
            raise ValidationError("source", "corrupt evidence identity or content")
        manifest, _ = manifest_reader(ctx, session_id=source["session_id"])
        if manifest is None or manifest["status"] != "active" or key not in manifest["sources"]:
            return None
        for unit in source["units"]:
            start, end = unit["parent_span"]
            if (not 0 <= start < end <= len(source["text"])
                    or source["text"][start:end] != unit["text"]
                    or unit["evidence_unit_id"] != digest(key, start, end)):
                raise ValidationError("source", "corrupt evidence span")
        return {**source, "source_node_version": version, "archive_ids": value.get("archive_ids", []),
                "sequence_verified": value.get("sequence_verified", True)}

    @timed("ref_verify")
    def verify_ref(self, ref: dict, ctx: RequestContext, *, owner_ctx=None) -> bool:
        """Check only persisted source authority, never accept a card's quote as proof."""
        source = self.read_source(ref["source_key"], ctx, owner_ctx=owner_ctx)
        return self.ref_matches_source(ref, source)

    def reference_reader(self, ctx, *, owner_ctx=None):
        """One validation pass with one caller and owner. Start fresh after a lock boundary."""
        @lru_cache(maxsize=8)
        def load(key):
            source = (self.read_source(key, ctx) if owner_ctx is None
                      else self.read_source(key, ctx, owner_ctx=owner_ctx))
            check_deadline()
            units = {unit["evidence_unit_id"]: unit for unit in source["units"]} if source else {}
            return source, units

        @timed("ref_verify")
        def resolve(ref):
            check_deadline()
            source, units = load(ref["source_key"])
            unit = units.get(ref["evidence_unit_id"])
            if unit is not None and self._ref_matches_unit(ref, source, unit):
                return source, unit
            return None

        return resolve

    def verify_refs(self, refs, ctx, *, owner_ctx=None):
        resolve = self.reference_reader(ctx, owner_ctx=owner_ctx)
        return all(resolve(ref) is not None for ref in refs)

    @staticmethod
    def ref_matches_source(ref, source):
        return source is not None and any(
            unit["evidence_unit_id"] == ref["evidence_unit_id"]
            and EvidenceStore._ref_matches_unit(ref, source, unit) for unit in source["units"])

    @staticmethod
    def _ref_matches_unit(ref, source, unit):
        start, end = ref["span"]
        return (source["content_hash"] == ref["content_hash"]
                and unit["parent_span"][0] <= start < end <= unit["parent_span"][1]
                and source["text"][start:end] == ref["quote"])

    def add_support(self, target_uri: str, refs: list[dict], ctx: RequestContext):
        """Persist reverse references before accepting a memory write."""
        with self.session_lock(ctx):
            manifest, version = self.manifest(ctx)
            if manifest is None or manifest["status"] != "active":
                raise ValidationError("source", "source session is unavailable")
            published = set(manifest["sources"])
            resolve = self.reference_reader(ctx)
            for ref in refs:
                if ref["source_key"] not in published or resolve(ref) is None:
                    raise ValidationError("source", "memory source is not active or quote differs")
            support = sorted({ref["source_key"] for ref in refs})
            current = manifest["supports"].get(target_uri, [])
            merged = sorted(set(current) | set(support))
            if merged != current:
                check_deadline()
                manifest["supports"][target_uri] = merged
                self._write(self.session_uri(ctx), manifest, version, ctx)

    def revoke_session(self, ctx: RequestContext) -> dict | None:
        """Make all session sources unreadable before removing canonical originals."""
        with self.session_lock(ctx):
            manifest, version = self.manifest(ctx)
            if manifest is None:
                manifest = {"session_id": ctx.session_id, "status": "active",
                            "sources": [], "batches": {}, "supports": {}}
            if manifest["status"] == "active":
                manifest["status"] = "revoked"
                self._write(self.session_uri(ctx), manifest, version, ctx)
            return manifest
