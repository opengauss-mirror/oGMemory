"""MemoryBot operations on AGFS canonical nodes and their vector projections."""

from dataclasses import replace
from itertools import islice

from psycopg2.extras import Json
from pyagfs.exceptions import AGFSClientError

from core.dreaming_evidence import matches_node_version, node_created_at
from core.errors import ConcurrentModificationError, NodeNotFoundError
from core.models import OutboxEvent, RequestContext
from fs.agfs_adapter.agfs_context_fs import parse_uri
from fs.agfs_adapter.archive_receipts import read_receipt, retain_receipt
from index.index_record_builder import build_index_records
from memorybot.gateway import MemoryGateway, WorkPendingError, _digest, _node_uri
from memorybot.models import BotError


class AGFSMemoryGateway(MemoryGateway):
    def __init__(self, memory_service, context_resolver, *, fs):
        super().__init__(memory_service, context_resolver, fs=fs)
        self.outbox = memory_service._get_outbox_store(fs)

    def _validate_storage(self):
        self._require_vector_index()
        self._atomic_projection = False

    def read_memory(self, ctx, uri):
        ctx = self._node_context(ctx, uri)
        pool = self.service._get_shared_sql_pool()
        return self._read_memory(ctx, uri, pool.get_connection, pool.return_connection)

    @staticmethod
    def _agent_ids(ctx):
        visible = [space[6:] for space in ctx.visible_owner_spaces if space.startswith("agent:")]
        return tuple(dict.fromkeys(agent for agent in (ctx.agent_id, *visible) if agent))

    def _node_context(self, ctx, uri):
        owner = parse_uri(_node_uri(uri))
        if (
            owner["account"] == ctx.account_id
            and owner["owner_type"] == "agents"
            and owner["owner_id"] in self._agent_ids(ctx)
        ):
            return replace(ctx, agent_id=owner["owner_id"])
        return ctx

    @staticmethod
    def _recovery_context(uri, account_id, user_id):
        owner = parse_uri(uri)
        if owner["account"] != account_id:
            return None
        return RequestContext(
            account_id,
            user_id,
            owner["owner_id"] if owner["owner_type"] == "agents" else "",
            "",
            "memorybot-recovery",
        )

    def _recorded_metadata(self, uri, account_id, user_id):
        ctx = self._recovery_context(uri, account_id, user_id)
        return self._metadata(uri, ctx) if ctx else {}

    def recover_memory_write(self, task, step):
        if step["input"].get("action") != "correct":
            return self.committed_forget(task, step)
        frozen = step["input"]
        uri = frozen["memory_uri"]
        ctx = self._recovery_context(uri, task["account_id"], task["user_id"])
        if ctx is None:
            return None
        with self.outbox.processing_lock(uri, timeout_seconds=5) as lease:
            try:
                node = self.fs.read_node(uri, ctx)
            except NodeNotFoundError:
                return None
            operation = node.metadata.get("bot_mutation") or {}
            if operation.get("id") == step["operation_id"] and matches_node_version(
                node.metadata, frozen["target_version"] + 1, frozen.get("target_created_at")
            ):
                event = self._correction_event(
                    node, operation, node.metadata["version"], node_created_at(node.metadata)
                )
                lease.ensure_owned()
                self.outbox.register_once(event, ctx)
        return None

    def committed_forget(self, task, step):
        if step["input"].get("action") != "forget":
            return None
        uri = step["input"]["memory_uri"]
        ctx = self._recovery_context(uri, task["account_id"], task["user_id"])
        if ctx is None:
            return None
        receipt = read_receipt(
            self.fs,
            uri,
            ctx,
            step["operation_id"],
            created_at=step["input"].get("target_created_at"),
        )
        if receipt is not None:
            return receipt["operation"].get("source_refs", step["input"]["source_refs"])
        metadata = self._recorded_metadata(
            uri,
            task["account_id"],
            task["user_id"],
        )
        if (
            metadata.get("status") == "ARCHIVED"
            and (metadata.get("bot_mutation") or {}).get("id") == step["operation_id"]
        ):
            return step["input"]["source_refs"]
        return None

    def _projection_matches(self, ctx, receipts):
        matches = super()._projection_matches
        for receipt in receipts:
            owner_ctx = self._node_context(ctx, receipt["uri"])
            if receipt.get("action") == "forget" and receipt.get("operation_id"):
                retained = read_receipt(
                    self.fs,
                    receipt["uri"],
                    owner_ctx,
                    receipt["operation_id"],
                    created_at=receipt["node_created_at"],
                )
                if retained is None or retained["version"] != receipt["node_version"]:
                    raise BotError("MEMORY_CHANGED", "归档回执已变化", 409)
                try:
                    replacement = self.fs.read_node(receipt["uri"], owner_ctx)
                except NodeNotFoundError:
                    if any(
                        self.index.get_record(record_id) for record_id in receipt["ids_to_delete"]
                    ):
                        return False
                    continue
                replacement_ids = {record.id for record in build_index_records(replacement)}
                receipt = {
                    "uri": replacement.uri,
                    "node_version": replacement.metadata["version"],
                    "sha256": _digest(replacement.content),
                    "retired_ids": sorted(set(receipt["ids_to_delete"]) - replacement_ids),
                }
            if not matches(owner_ctx, [receipt]):
                return False
        return True

    def _listed_memories(self, ctx, scope):
        roots = [(f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories", ctx)]
        if scope.get("memory_scope") == "authorized":
            roots.extend(
                (f"ctx://{ctx.account_id}/agents/{agent}/memories", replace(ctx, agent_id=agent))
                for agent in self._agent_ids(ctx)
            )
        for root, owner_ctx in roots:
            for directory in self.fs.list_children_including_inactive(root, owner_ctx):
                if directory.endswith("/profile"):
                    yield directory
                else:
                    yield from self.fs.list_children(directory, owner_ctx)

    def list_memories(self, ctx, q="", scope=None):
        scope = scope or {}
        if q:
            user_space = ctx.user_space_name()
            contexts = [replace(ctx, visible_owner_spaces=(user_space,))]
            if scope.get("memory_scope") == "authorized":
                contexts.extend(
                    replace(ctx, agent_id=agent, visible_owner_spaces=(f"agent:{agent}",))
                    for agent in self._agent_ids(ctx)
                )
            values = {}
            for owner_ctx in contexts:
                values.update(
                    (row["uri"], row) for row in super().list_memories(owner_ctx, q, scope)
                )
            return list(values.values())
        return self._read_memory_values(ctx, scope, sorted(self._listed_memories(ctx, scope)))

    def list_memory_page(self, ctx, q, scope, hidden, *, limit, offset):
        if q:
            return super().list_memory_page(ctx, q, scope, hidden, limit=limit, offset=offset)
        uris = sorted(set(self._listed_memories(ctx, scope)).difference(hidden))
        values = (
            value
            for uri in uris
            for value in self._read_memory_values(ctx, scope, [uri])
            if value["context_type"] == "MEMORY"
        )
        page = list(islice(values, offset, offset + limit + 1))
        return page[:limit], len(page) > limit

    def _metadata(self, uri, ctx):
        try:
            return self.fs.read_node_metadata_including_inactive(uri, ctx)
        except NodeNotFoundError:
            return {}

    def _has_reference(self, tx, item, *, file_id=None, task_id=None):
        reference = {"file_id": file_id} if file_id else {"task_id": task_id}
        tx.cursor.execute(
            "SELECT s.user_id,s.kind,s.input,s.output FROM bot_step s JOIN bot_task t "
            "ON t.account_id=s.account_id AND t.task_id=s.task_id WHERE s.account_id=%s AND ("
            "(s.kind='memory_apply' AND (s.input->'source_refs' @> %s::jsonb "
            "OR s.input->'file'->>'file_id'=%s OR s.task_id=%s OR t.parent_task_id=%s)) "
            "OR (s.kind='memory_submit' AND (s.task_id=%s OR t.parent_task_id=%s "
            "OR t.input->'source_refs' @> %s::jsonb)))",
            (
                item["account_id"],
                Json([reference]),
                file_id,
                task_id,
                task_id,
                task_id,
                task_id,
                Json([reference]),
            ),
        )
        account = item["account_id"]
        for row in tx.cursor.fetchall():
            user = row["user_id"]
            value, output = row["input"], row["output"] or {}
            if row["kind"] == "memory_submit":
                if any(
                    self._recorded_metadata(source["uri"], account, user)
                    for source in output.get("sources", [])
                ):
                    return True
                continue
            if value.get("file"):
                uri = self._source_uri(account, user, value["file"]["file_id"], value["index"])
            else:
                uri = value["memory_uri"]
            if any(
                all(ref.get(key) == expected for key, expected in reference.items())
                for ref in self._recorded_metadata(uri, account, user).get("bot_source_refs", [])
            ):
                return True
        return False

    def has_file_reference(self, tx, file):
        return self._has_reference(tx, file, file_id=file["file_id"])

    def has_task_reference(self, tx, task):
        return self._has_reference(tx, task, task_id=task["task_id"])

    def _create_node(self, node, ctx):
        try:
            self.fs.write_node(node, ctx)
        except ConcurrentModificationError:
            existing = self.fs.read_node(node.uri, ctx)
            if (
                existing.content != node.content
                or existing.metadata.get("bot_source_refs") != node.metadata["bot_source_refs"]
            ):
                raise BotError("MEMORY_CHANGED", "资料目标已变化", 409)
            return existing
        return self.fs.read_node(node.uri, ctx)

    def publish_source(self, tx, task, file, text, operation_id, *, chunk_index=0):
        ctx = self.context(task)
        receipts = []
        try:
            for index, offset in enumerate(range(0, max(1, len(text)), 12000), chunk_index):
                node = self._source_node(ctx, file, text[offset : offset + 12000], index)
                stored = self._create_node(node, ctx)
                event = self.outbox.build_write_event(stored)
                event.event_id = _digest(f"{operation_id}:{index}")[:36]
                self.outbox.register_once(event, ctx)
                receipts.append(
                    {
                        "event_id": event.event_id,
                        "uri": stored.uri,
                        "action": "ingest",
                        "node_version": stored.metadata["version"],
                        "sha256": _digest(stored.content),
                    }
                )
        except AGFSClientError as exc:
            raise WorkPendingError("等待 AGFS 发布资料") from exc
        return {"sources": receipts}

    def mutate_target(self, tx, task, frozen, operation_id):
        uri, action = frozen["memory_uri"], frozen["action"]
        ctx = self._node_context(self.context(task), uri)
        expected_version = frozen["target_version"] + 1
        try:
            saved = (
                read_receipt(
                    self.fs,
                    uri,
                    ctx,
                    operation_id,
                    created_at=frozen.get("target_created_at"),
                )
                if action == "forget"
                else None
            )
            metadata = (
                {
                    "version": saved["version"],
                    "bot_mutation": saved["operation"],
                    "created_at": saved["created_at"],
                }
                if saved
                else self._metadata(uri, ctx)
            )
            operation = metadata.get("bot_mutation") or {}
            if operation.get("id") != operation_id:
                node = self.fs.read_node(uri, ctx)
                if not matches_node_version(
                    node.metadata,
                    frozen["target_version"],
                    frozen.get("target_created_at"),
                ):
                    raise BotError("MEMORY_CHANGED", "记忆目标已变化", 409)
                event_id = _digest(f"{operation_id}:{task['attempt']}")[:36]
                prior_ids = (
                    self._archive_ids(node, ctx)
                    if action == "forget"
                    else [record.id for record in build_index_records(node)]
                )
                operation = {"id": operation_id, "prior_ids": prior_ids, "event_id": event_id}
                if action == "forget":
                    self._check_target(node, frozen)
                    operation["source_refs"] = frozen.get("source_refs", [])
                    event = self._archive_event(
                        operation,
                        uri,
                        expected_version,
                        node_created_at(node.metadata),
                    )
                    # Match delete and projection lock order before publishing the durable archive.
                    with (
                        self.fs.memory_identity_lock(node.parent_uri.rstrip("/"), ctx),
                        self.outbox.processing_lock(event.uri, timeout_seconds=5),
                        self.outbox.processing_lock(uri, timeout_seconds=5) as lease,
                    ):
                        self.outbox.register_once(event, ctx)
                        lease.ensure_owned()
                        self.fs.archive_node_version(node, ctx, bot_mutation=operation)
                else:
                    current = replace(node, metadata=dict(node.metadata))
                    sources = []
                    self._correct_node(node, ctx, frozen, operation_id, task, sources.append)
                    if current.content != node.content:
                        self._check_target(current, frozen)
                    created_at = node_created_at(current.metadata)
                    node.metadata.update(bot_mutation=operation, _expected_created_at=created_at)
                    with self.outbox.processing_lock(uri, timeout_seconds=5) as lease:
                        for source in sources:
                            self._create_node(source, ctx)
                        lease.ensure_owned()
                        self.fs.write_node(node, ctx)
                metadata = self._metadata(uri, ctx)
            if (
                metadata.get("version") != expected_version
                or metadata.get("bot_mutation") != operation
            ):
                raise BotError("MEMORY_CHANGED", "记忆目标已变化", 409)

            retired_ids = []
            if action == "forget":
                # Upgrade an already-committed archive from a previous worker as well.
                if saved is None:
                    retain_receipt(self.fs, uri, ctx)
                self.outbox.register_once(
                    self._archive_event(
                        operation,
                        uri,
                        expected_version,
                        node_created_at(metadata),
                    ),
                    ctx,
                )
                digest = frozen["target_sha256"]
            else:
                with self.outbox.processing_lock(uri, timeout_seconds=5) as lease:
                    node = self.fs.read_node(uri, ctx)
                    if (
                        node.metadata.get("version") != expected_version
                        or node.metadata.get("bot_mutation") != operation
                    ):
                        raise BotError("MEMORY_CHANGED", "记忆目标已变化", 409)
                    # The committed node retains the event ID across publication retries.
                    event = self._correction_event(
                        node, operation, expected_version, node_created_at(node.metadata)
                    )
                    lease.ensure_owned()
                    self.outbox.register_once(event, ctx)
                    updated_ids = {record["id"] for record in event.payload["records"]}
                    retired_ids = sorted(set(operation["prior_ids"]) - updated_ids)
                    digest = _digest(node.content)
        except (AGFSClientError, ConcurrentModificationError, TimeoutError) as exc:
            raise WorkPendingError("等待 AGFS 提交记忆") from exc
        return {
            "event_id": operation["event_id"],
            "uri": uri,
            "action": action,
            "node_version": expected_version,
            **(
                {"operation_id": operation_id, "node_created_at": node_created_at(metadata)}
                if action == "forget"
                else {}
            ),
            "sha256": digest,
            "ids_to_delete": operation["prior_ids"] if action == "forget" else [],
            "retired_ids": retired_ids,
            "source_refs": frozen.get("source_refs", []),
        }

    def _correction_event(self, node, operation, version, created_at):
        event = self.outbox.build_write_event(node)
        event.event_id = operation["event_id"]
        event.payload["canonical_write_revision"] = {"version": version, "created_at": created_at}
        updated_ids = {record["id"] for record in event.payload["records"]}
        event.payload["ids_to_delete"] = sorted(
            set(event.payload.get("ids_to_delete", ())).union(
                set(operation["prior_ids"]) - updated_ids
            )
        )
        return event

    @staticmethod
    def _archive_event(operation, uri, version, created_at):
        return OutboxEvent(
            operation["event_id"],
            "ARCHIVE_CONTEXT",
            uri.rpartition("/")[0],
            {
                "ids_to_delete": operation["prior_ids"],
                "operation_id": operation["id"],
                "canonical_archive_source": uri,
                "canonical_archive_revision": {
                    "version": version,
                    "created_at": created_at,
                },
            },
            "PENDING",
        )
