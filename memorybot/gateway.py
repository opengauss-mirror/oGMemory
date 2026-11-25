"""Authorized access to the repository's real memory and model services."""

from __future__ import annotations

import hashlib
from copy import copy, deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

from core.errors import NodeNotFoundError
from core.models import ContextNode, OutboxEvent, Role
from memorybot.models import BotError
from retrieval.owner_scope import expand_owner_space_aliases


class OutcomeUnknownError(RuntimeError):
    pass


class WorkPendingError(RuntimeError):
    pass


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _node_uri(uri):
    return uri[:-11] if uri.endswith("/content.md") else uri.rstrip("/")


def create_gateway(memory_service, context_resolver):
    from fs.agfs_adapter.agfs_context_fs import AGFSContextFS

    fs = memory_service._get_context_fs()
    if isinstance(fs, AGFSContextFS):
        from memorybot.agfs_gateway import AGFSMemoryGateway

        return AGFSMemoryGateway(memory_service, context_resolver, fs=fs)
    return MemoryGateway(memory_service, context_resolver, fs=fs)


class MemoryGateway:
    def __init__(self, memory_service, context_resolver, *, fs=None):
        self.service = memory_service
        self.context_resolver = context_resolver
        if type(memory_service.get_llm()).__module__.endswith("mock_llm"):
            raise ValueError("memoryBot 需要配置真实模型服务")
        self.fs = fs if fs is not None else memory_service._get_context_fs()
        self.index = memory_service._get_shared_vector_index()
        self._validate_storage()

    def _validate_storage(self):
        from fs.agfs_adapter.space_bridge import AGFSSpaceBridge
        from fs.sql_adapter.session_mutation import _dsn_identity
        from fs.sql_adapter.sql_context_fs import SQLContextFS
        from providers.vector_index.opengauss_index import OpenGaussVectorIndex

        if not isinstance(self.fs, SQLContextFS):
            raise ValueError("memoryBot 需要 SQLContextFS 与同库 OpenGaussVectorIndex")
        self._atomic_projection = isinstance(self.index, OpenGaussVectorIndex) and (
            _dsn_identity(self.fs._connection_string)
            == _dsn_identity(self.index._connection_string)
        )
        if isinstance(self.fs, AGFSSpaceBridge):
            self._require_vector_index()
        elif not self._atomic_projection:
            raise ValueError("memoryBot canonical 与向量索引必须使用同一 SQL 连接配置")

    def _require_vector_index(self):
        if not callable(getattr(self.index, "get_record", None)):
            raise ValueError("memoryBot 的 AGFS 部署需要可按记录读取的向量索引")

    def context(self, task):
        return self.context_resolver(task)

    def templates(self, ctx):
        from memorybot.templates import TemplateRegistry

        return TemplateRegistry.from_service(self.service, ctx)

    @staticmethod
    def has_file_reference(tx, file):
        from psycopg2.extras import Json

        tx.cursor.execute(
            "SELECT 1 FROM context_nodes WHERE account_id=%s AND metadata->'bot_source_refs' @> %s::jsonb LIMIT 1",
            (file["account_id"], Json([{"file_id": file["file_id"]}])),
        )
        return tx.cursor.fetchone() is not None

    @staticmethod
    def has_task_reference(tx, task):
        from psycopg2.extras import Json

        tx.cursor.execute(
            "SELECT 1 FROM context_nodes WHERE account_id=%s AND metadata->'bot_source_refs' @> %s::jsonb LIMIT 1",
            (task["account_id"], Json([{"task_id": task["task_id"]}])),
        )
        if tx.cursor.fetchone():
            return True
        tx.cursor.execute(
            "SELECT 1 FROM bot_step s JOIN bot_task t ON t.account_id=s.account_id AND t.task_id=s.task_id "
            "CROSS JOIN jsonb_array_elements(COALESCE(s.output->'sources','[]'::jsonb)) source "
            "JOIN context_nodes n ON n.account_id=s.account_id AND n.uri=source->>'uri' "
            "WHERE s.account_id=%s AND s.user_id=%s AND s.kind='memory_submit' AND s.state='succeeded' "
            "AND (t.task_id=%s OR t.parent_task_id=%s) LIMIT 1",
            (task["account_id"], task["user_id"], task["task_id"], task["task_id"]),
        )
        return tx.cursor.fetchone() is not None

    @staticmethod
    def _value(node):
        from core.dreaming_evidence import node_created_at

        metadata = node.metadata
        return {
            "uri": node.uri,
            "title": str(metadata.get("title") or node.abstract or node.uri.rsplit("/", 1)[-1])[
                :128
            ],
            "text": node.content,
            "source_refs": list(metadata.get("bot_source_refs", [])),
            "updated_at": str(metadata.get("updated_at") or ""),
            "source_time": metadata.get("source_time") or metadata.get("created_at"),
            "node_version": int(metadata.get("version", 1)),
            "node_created_at": node_created_at(metadata),
            "sha256": _digest(node.content),
            "category": node.category,
            "context_type": node.context_type,
            "owner_space": node.owner_space,
        }

    def read_memory(self, ctx, uri):
        return self._read_memory(ctx, uri, self.fs._get_connection, self.fs._return_connection)

    def _read_memory(self, ctx, uri, get_connection, return_connection):
        from fs.sql_adapter.pool import PoolAdapterMixin

        value = self._value(self.fs.read_node(_node_uri(uri), ctx))
        conn = get_connection()
        try:
            PoolAdapterMixin.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute("SET LOCAL app.bot_worker = 'off'")
                return self._source_refs(cur, ctx, value)
        finally:
            conn.rollback()
            return_connection(conn)

    @staticmethod
    def _source_refs(cur, ctx, value):
        from psycopg2.extras import Json

        cur.execute(
            "SELECT t.parent_task_id,t.task_id,s.input,t.input FROM bot_step s JOIN bot_task t "
            "ON t.account_id=s.account_id AND t.task_id=s.task_id "
            "WHERE s.account_id=%s AND s.user_id=%s AND s.kind='memory_submit' AND s.state='succeeded' "
            "AND s.output->'sources' @> %s::jsonb ORDER BY s.created_at DESC LIMIT 20",
            (ctx.account_id, ctx.user_id, Json([{"uri": value["uri"]}])),
        )
        for parent, task_id, accepted, task_input in cur.fetchall():
            refs = task_input.get("source_refs") or [
                {
                    "file_id": None,
                    "uri": None,
                    "task_id": parent or task_id,
                    "quote": "\n".join(
                        message["content"]
                        for message in accepted.get("messages", [])
                        if message["role"] == "user"
                    )[:1024],
                }
            ]
            value["source_refs"].extend(ref for ref in refs if ref not in value["source_refs"])
        value["source_refs"] = value["source_refs"][:100]
        return value

    def list_memories(self, ctx, q="", scope=None):
        scope = scope or {}
        if q:
            hits = self.service.get_read_api().search_memory(q, ctx, top_k=30).hits
            uris = list(dict.fromkeys(_node_uri(hit.uri) for hit in hits))
        else:
            fs = self.service._get_context_fs()
            directories, _ = self._memory_directories(fs, ctx, scope)
            uris = []
            for directory in directories:
                if directory.endswith("/profile"):
                    uris.append(directory)
                uris.extend(fs.list_children(directory, ctx))
        return self._read_memory_values(ctx, scope, uris)

    @staticmethod
    def _memory_directories(fs, ctx, scope):
        roots = [f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/"]
        owners = [ctx.user_space_name()]
        if scope.get("memory_scope") == "authorized":
            for space in ctx.visible_owner_spaces:
                if space.startswith("agent:"):
                    roots.append(f"ctx://{ctx.account_id}/agents/{space[6:]}/memories/")
                    owners.append(space)
        prefixes = tuple(roots)
        directories = [uri for uri in fs.count_memory_children(ctx) if uri.startswith(prefixes)]
        return directories, owners

    def list_memory_page(self, ctx, q, scope, hidden, *, limit, offset):
        if q:
            values = [row for row in self.list_memories(ctx, q, scope) if row["uri"] not in hidden]
            return values[offset : offset + limit], len(values) > offset + limit

        fs = self.service._get_context_fs()
        directories, owners = self._memory_directories(fs, ctx, scope)
        if not directories:
            return [], False
        profiles = [directory for directory in directories if directory.endswith("/profile")]
        owner_aliases = expand_owner_space_aliases(owners)
        conn = fs._get_connection()
        try:
            fs.bind_tenant(conn, ctx.account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT uri FROM context_nodes WHERE account_id=%s AND status='ACTIVE' "
                    "AND context_type='MEMORY' AND owner_space=ANY(%s) AND uri<>ALL(%s) "
                    "AND (uri=ANY(%s) OR regexp_replace(uri, '/[^/]+$', '')=ANY(%s)) "
                    "ORDER BY uri LIMIT %s OFFSET %s",
                    (
                        ctx.account_id,
                        owner_aliases,
                        list(hidden),
                        profiles,
                        directories,
                        limit + 1,
                        offset,
                    ),
                )
                uris = [row[0] for row in cur.fetchall()]
        finally:
            conn.rollback()
            fs._return_connection(conn)
        return self._read_memory_values(ctx, scope, uris[:limit]), len(uris) > limit

    def _read_memory_values(self, ctx, scope, uris):
        values = []
        for uri in uris:
            try:
                value = self.read_memory(ctx, uri)
            except NodeNotFoundError:
                continue
            if self.in_scope(value, ctx, scope):
                values.append(value)
        return values

    @staticmethod
    def in_scope(value, ctx, scope):
        from retrieval.owner_scope import canonical_owner_space

        owner = canonical_owner_space(value["owner_space"])
        if value.get("context_type", "MEMORY") == "RESOURCE":
            uri = _node_uri(value["uri"])
            selected = scope.get("resource_ids")
            if selected is not None and all(uri != _node_uri(item) for item in selected):
                return False
            owners, uris = scope.get("resource_owners"), scope.get("resource_uris")
            if owners is not None or uris is not None:
                return owner in (owners or ()) or any(uri == _node_uri(item) for item in uris or ())
            return selected is not None
        if scope.get("memory_scope", "user") == "user" and owner != ctx.user_space_name():
            return False
        if scope.get("memory_owners") is not None:
            return owner in scope["memory_owners"]
        return True

    def retrieve(self, task, messages, scope, excluded):
        ctx = self.context(task)
        query = task["input"].get("text") or "工作进展"
        raw = self.service.compose(
            {
                "_ctx": ctx,
                "sessionId": ctx.session_id,
                "agentId": ctx.agent_id,
                "messages": messages,
                "query": query,
                "tokenBudget": 60000,
            }
        )
        values = self.list_memories(ctx, query, scope)
        seen = {_node_uri(item["uri"]) for item in values}
        for uri in scope.get("resource_ids", []):
            if _node_uri(uri) in seen:
                continue
            value = self.read_memory(ctx, uri)
            if self.in_scope(value, ctx, scope):
                values.append(value)
                seen.add(_node_uri(value["uri"]))
        denied = {row["memory_uri"] for row in excluded}
        quotes = [
            ref["quote"]
            for row in excluded
            for ref in row.get("source_refs", [])
            if ref.get("quote")
        ]
        values = [
            item
            for item in values
            if item["uri"] not in denied and not any(q in item["text"] for q in quotes)
        ]
        assembled = [row for row in raw.get("messages", messages) if not row.get("_ogmem")]
        if excluded:
            assembled = [
                {
                    **row,
                    "content": self.filter_text(str(row.get("content", "")), excluded),
                }
                for row in assembled
            ]
        return {
            "messages": [
                {"role": row["role"], "content": str(row.get("content", ""))} for row in assembled
            ],
            "sources": values[:100],
        }

    @staticmethod
    def filter_text(text, excluded):
        for row in excluded:
            for ref in row.get("source_refs", []):
                if ref.get("quote"):
                    text = text.replace(ref["quote"], "[该信息已遗忘]")
        return text

    def complete(self, task, messages, *, max_output_tokens=8000, tools=None, on_text_delta=None):
        self.context(task)
        llm = copy(self.service.get_llm())
        if type(llm).__module__.endswith("mock_llm"):
            raise RuntimeError("memoryBot 需要配置真实模型")
        if hasattr(llm, "_max_tokens"):
            llm._max_tokens = max_output_tokens
        calls, text = llm.complete_with_tools_messages(
            messages, tools=tools, on_text_delta=on_text_delta
        )
        return {
            "calls": [
                {
                    "id": call.get("id"),
                    "tool": call["tool"],
                    "input": call.get("input", {}),
                }
                for call in calls
            ],
            "text": text,
        }

    def tool(self, task, name, arguments, scope, excluded):
        ctx = self.context(task)
        if name == "search_memory":
            values = self.list_memories(ctx, str(arguments.get("query", "")), scope)
        elif name == "read_memory":
            value = self.read_memory(ctx, str(arguments["uri"]))
            values = [value] if self.in_scope(value, ctx, scope) else []
        else:
            raise ValueError("工具尚未发布")
        denied = {row["memory_uri"] for row in excluded}
        return {
            "sources": [
                {**item, "text": self.filter_text(item["text"], excluded)[:16000]}
                for item in values
                if item["uri"] not in denied
            ][:20]
        }

    def capture(self, task, messages, operation_id, *, timeout=900):
        ctx = self.context(task)
        response = self.service.after_turn(
            {
                "_ctx": ctx,
                "sessionId": ctx.session_id,
                "agentId": ctx.agent_id,
                "clientRequestId": operation_id,
                "messages": messages,
                "forceExtract": True,
                "wait": True,
                "timeoutSeconds": timeout,
                "prePromptMessageCount": 0,
            }
        )
        status = response.get("status")
        request_status = response.get("request_status")
        if request_status == "NEEDS_CONFIRMATION" or status == "needs_confirmation":
            raise OutcomeUnknownError(response.get("error", "记忆写入需要核对"))
        if request_status in {"ACCEPTED", "RUNNING", "RECOVERING"} or status in {
            "processing",
            "queued",
            "accumulating",
        }:
            raise WorkPendingError("记忆处理进行中")
        if status == "duplicate":
            from service.extraction_receipts import read as read_receipt

            with self.service._session_mutation_scope(ctx):
                buffer = self.service.get_session_manager().get_or_create(ctx.session_id, ctx=ctx)
                response = read_receipt(buffer, ctx, operation_id)
            if response is None:
                raise OutcomeUnknownError("原请求已完成；完整写入回执需要核对")
        if response.get("ok") is False:
            if response.get("retryable"):
                raise WorkPendingError("记忆处理等待重试")
            raise RuntimeError(response.get("error", "记忆写入未完成"))
        sources = []
        for plan in response.get("plans", []):
            stored = plan.get("action") in {"create", "merge", "append"} or (
                plan.get("action") == "skip"
                and (plan.get("merged_fields") or {}).get("status")
                in {"operation_committed", "duplicate"}
            )
            if stored and plan.get("target_uri"):
                sources.append(self.read_memory(ctx, plan["target_uri"]))
        return {
            "receipt": response,
            "sources": sources,
            "memory_status": "stored" if sources else "unchanged",
        }

    def capture_recheck(self, task, operation_id, frozen):
        from service.extraction_receipts import read as read_receipt

        ctx = self.context(task)
        with self.service._session_mutation_scope(ctx):
            buffer = self.service.get_session_manager().get_or_create(ctx.session_id, ctx=ctx)
            batch = buffer.runtime_extensions.get("memory_extraction_recovery", {}).get("batch", {})
            confirmed = read_receipt(buffer, ctx, operation_id, include_accepted=True)
            if batch.get("client_request_id") != operation_id and confirmed is None:
                raise OutcomeUnknownError("原请求回执待核对，请先恢复对应提取检查点或完成回执")
        return self.capture(task, frozen["messages"], operation_id)

    def prepare_target(self, task, uri):
        return self.read_memory(self.context(task), uri)

    def _correct_agent(self, node, ctx, frozen, operation_id, task, write_source):
        from commit.agent_memory import AgentMemoryPolicy
        from extraction.agent_memory import (
            AgentMemoryCandidate,
            agent_source_id,
            memory_units,
            node_memory_fields,
            render_agent_memory,
        )

        fields = node_memory_fields(node)
        units = memory_units(node.category, fields)
        key = (
            units[0]["key"]
            if units
            else next(iter(node.metadata["agent_memory"]["units"]), "explicit-correction")
        )
        title = frozen["new_text"].strip().splitlines()[0].lstrip("# ")[:128]
        source_uri = f"ctx://{ctx.account_id}/users/{ctx.user_id}/memories/documents/correction_{_digest(operation_id)[:32]}"
        source = ContextNode(
            source_uri,
            "RESOURCE",
            "document",
            2,
            ctx.user_space_name(),
            title,
            "",
            frozen["new_text"],
            {
                "indexing_disabled": True,
                "bot_source_refs": [
                    {
                        "file_id": None,
                        "uri": None,
                        "task_id": task.get("parent_task_id") or task["task_id"],
                        "quote": frozen["new_text"][:1024],
                    }
                ],
                "_expected_version": 0,
            },
        )
        write_source(source)
        source_id = agent_source_id(source_uri)
        binding = {
            "uri": source_uri,
            "content_hash": _digest(source.content),
            "resource_version": "1",
            "role": "user",
            "origin": "memorybot_explicit_correction",
            "lane": node.category,
        }
        if node.category == "tool":
            binding["tool_name"] = fields["tool_name"]
        citation = {"source_id": source_id}
        corrected = {
            "key": key,
            "title": title,
            "content": frozen["new_text"],
            "grounding": [citation],
        }
        incoming = (
            {"tool_name": fields["tool_name"], "units": [corrected]}
            if node.category == "tool"
            else corrected
        )
        affected = [key, *(unit["key"] for unit in units[1:])]
        prior = units[0] if units else None
        changed = key in node.metadata["agent_memory"]["units"] and (
            prior is None
            or any(prior[field] != corrected[field] for field in ("key", "title", "content"))
        )
        context = {
            "operation_id": operation_id,
            "base_units": {
                unit_key: state["revision"]
                for unit_key, state in node.metadata["agent_memory"]["units"].items()
            },
            "removals": [
                {
                    "key": unit["key"],
                    "reason": "用户替换整条记忆",
                    "grounding": [citation],
                }
                for unit in units[1:]
            ],
            "corrections": ([{"key": key, "reason": "用户显式更正整条记忆"}] if changed else []),
            "sources": {source_id: binding},
            "grounding": {unit_key: [{**binding, "source_id": source_id}] for unit_key in affected},
        }
        candidate = AgentMemoryCandidate(
            category=node.category,
            owner_scope="agent",
            routing_key=node.metadata["routing_key"],
            abstract=title,
            overview="",
            content=frozen["new_text"],
            confidence=1.0,
            structured_fields=incoming,
            memory_context=context,
            provenance_ids=[source_uri],
            write_operation_id=operation_id,
        )
        reader = SimpleNamespace(
            exists=lambda uri, request: uri == node.uri,
            read_node=lambda uri, request: deepcopy(node),
        )
        resolver = SimpleNamespace(resolve=lambda category, values, request: node.uri)
        policy_ctx = replace(ctx, agent_id=node.owner_space.removeprefix("agent:"))
        plan = AgentMemoryPolicy(reader, resolver).plan(candidate, policy_ctx)
        if plan.action != "skip":
            node.metadata.update(
                structured_fields=plan.merged_fields["agent_memory_fields"],
                agent_memory=plan.merged_fields["agent_memory_context"],
                provenance_ids=plan.merged_fields["provenance_ids"],
                derived_from=list(
                    dict.fromkeys([*node.metadata.get("derived_from", []), source_uri])
                ),
                agent_memory_content_version=3,
                agent_memory_layers_version=2,
            )
            node.content = render_agent_memory(node.category, node.metadata["structured_fields"])

    @classmethod
    def _check_target(cls, node, frozen):
        current = cls._value(node)
        if (
            current["node_version"] != frozen["target_version"]
            or current["sha256"] != frozen["target_sha256"]
        ):
            raise BotError(
                "MEMORY_CHANGED",
                "记忆已更新，请读取最新内容",
                409,
                {
                    "memory_uri": node.uri,
                    "current_version": current["node_version"],
                    "current_sha256": current["sha256"],
                },
            )
        return current["node_version"]

    @staticmethod
    def _archive_ids(node, ctx):
        from index.index_record_builder import agent_memory_index_ids, build_index_records

        if node.category == "profile":
            raise PermissionError("档案事实请通过更正维护")
        if node.owner_space.startswith("agent:") and ctx.role not in {Role.ADMIN, Role.ROOT}:
            raise PermissionError("代理记忆归档需要管理员权限")
        if node.metadata.get("agent_memory_version") == 1:
            return agent_memory_index_ids(node)
        return [record.id for record in build_index_records(node)]

    def _correct_node(self, node, ctx, frozen, operation_id, task, write_source):
        node.content = frozen["new_text"]
        if node.metadata.get("agent_memory_version") == 1:
            self._correct_agent(node, ctx, frozen, operation_id, task, write_source)
        node.abstract = node.content[:200]
        node.overview = node.content[:4000]
        node.metadata.update(
            _expected_version=frozen["target_version"],
            bot_source_refs=[
                *frozen.get("source_refs", []),
                {
                    "file_id": None,
                    "uri": None,
                    "task_id": task.get("parent_task_id") or task["task_id"],
                    "quote": frozen["new_text"][:1024],
                },
            ][-100:],
        )

    def _read_transaction_node(self, cur, uri, ctx):
        node = self.fs.read_node_with_cursor(cur, uri, ctx)
        hydrate = getattr(self.fs, "hydrate_nodes", None)
        return hydrate([node])[0] if hydrate else node

    def mutate_target(self, tx, task, frozen, operation_id):
        """Apply a targeted edit and its outbox on the fenced Bot transaction."""
        from psycopg2.extras import Json

        ctx = self.context(task)
        fs = self.fs
        uri = frozen["memory_uri"]
        with tx.connection.cursor() as cur:
            cur.execute(
                "SELECT version FROM context_nodes WHERE account_id=%s AND uri=%s FOR UPDATE",
                (ctx.account_id, uri),
            )
            node = self._read_transaction_node(cur, uri, ctx)
            prior_version = self._check_target(node, frozen)
            action = frozen["action"]
            if action == "forget":
                event = OutboxEvent(
                    hashlib.sha256(operation_id.encode()).hexdigest()[:36],
                    "ARCHIVE_CONTEXT",
                    uri,
                    {"ids_to_delete": self._archive_ids(node, ctx)},
                    "PENDING",
                )
                cur.execute(
                    "UPDATE context_nodes SET status='ARCHIVED',version=version+1,updated_at=NOW() WHERE account_id=%s AND uri=%s",
                    (ctx.account_id, uri),
                )
                version = prior_version + 1
            else:

                def write_source(source):
                    fs._execute_write(
                        cur, source, ctx.account_id, 0, True, *fs._build_row_data(source)
                    )

                self._correct_node(node, ctx, frozen, operation_id, task, write_source)
                values = fs._build_row_data(node)
                fs._execute_write(cur, node, ctx.account_id, prior_version, False, *values)
                event = self.service._get_outbox_store(fs).build_write_event(node)
                event.event_id = hashlib.sha256(operation_id.encode()).hexdigest()[:36]
                version = values[2]
            cur.execute(
                "INSERT INTO outbox_events (event_id,event_type,uri,account_id,payload,status,retry_count,created_at) VALUES (%s,%s,%s,%s,%s,'PENDING',0,NOW())",
                (
                    event.event_id,
                    event.event_type,
                    uri,
                    ctx.account_id,
                    Json(event.payload),
                ),
            )
            return {
                "event_id": event.event_id,
                "uri": uri,
                "node_version": version,
                "action": action,
                "records": event.payload.get("records", []),
                "ids_to_delete": event.payload.get("ids_to_delete", []),
                "retired_ids": (
                    sorted(
                        set(event.payload.get("ids_to_delete", []))
                        - {record["id"] for record in event.payload.get("records", [])}
                    )
                    if action == "correct"
                    else []
                ),
                "sha256": _digest(node.content),
                "source_refs": frozen.get("source_refs", []),
            }

    @staticmethod
    def _source_uri(account_id, user_id, file_id, chunk_index):
        return (
            f"ctx://{account_id}/users/{user_id}/memories/documents/bot_{file_id}_{chunk_index + 1}"
        )

    @classmethod
    def _source_node(cls, ctx, file, content, chunk_index):
        return ContextNode(
            cls._source_uri(ctx.account_id, ctx.user_id, file["file_id"], chunk_index),
            "RESOURCE",
            "document",
            2,
            ctx.user_space_name(),
            content[:200],
            content[:2000],
            content,
            {
                "title": file["name"],
                "bot_source_refs": [
                    {
                        "file_id": file["file_id"],
                        "task_id": None,
                        "uri": None,
                        "quote": content[:1024],
                    }
                ],
                "source_time": str(file.get("created_at") or datetime.now(UTC).isoformat()),
                "_expected_version": 0,
            },
        )

    def publish_source(self, tx, task, file, text, operation_id, *, chunk_index=0):
        from psycopg2.extras import Json

        ctx = self.context(task)
        fs = self.service._get_context_fs()
        receipts = []
        with tx.connection.cursor() as cur:
            for index, start in enumerate(range(0, max(1, len(text)), 12000)):
                index += chunk_index
                content = text[start : start + 12000]
                node = self._source_node(ctx, file, content, index)
                uri = node.uri
                values = fs._build_row_data(node)
                fs._execute_write(cur, node, ctx.account_id, 0, True, *values)
                event = self.service._get_outbox_store(fs).build_write_event(node)
                event.event_id = hashlib.sha256(f"{operation_id}:{index}".encode()).hexdigest()[:36]
                cur.execute(
                    "INSERT INTO outbox_events (event_id,event_type,uri,account_id,payload,status,retry_count,created_at) VALUES (%s,%s,%s,%s,%s,'PENDING',0,NOW())",
                    (
                        event.event_id,
                        event.event_type,
                        uri,
                        ctx.account_id,
                        Json(event.payload),
                    ),
                )
                receipts.append(
                    {
                        "event_id": event.event_id,
                        "uri": uri,
                        "node_version": 1,
                        "sha256": _digest(content),
                        "action": "ingest",
                        "records": [
                            {"id": record["id"], "sha256": _digest(record["text"])}
                            for record in event.payload["records"]
                        ],
                    }
                )
        return {"sources": receipts}

    def projection_ready(self, task, receipts):
        ctx = self.context(task)
        self.service.drain_outbox_sync(account_id=ctx.account_id)
        return self._projection_matches(ctx, receipts)

    def _projection_matches(self, ctx, receipts):
        from index.index_record_builder import build_index_records

        for receipt in receipts:
            if receipt.get("action") == "forget":
                if self.fs.archived_node_version(receipt["uri"], ctx) != receipt["node_version"]:
                    raise BotError("MEMORY_CHANGED", "归档目标已变化", 409)
                if any(
                    self.index.get_record(record_id) is not None
                    for record_id in receipt["ids_to_delete"]
                ):
                    return False
                continue
            node = self.fs.read_node(receipt["uri"], ctx)
            if (
                int(node.metadata.get("version", 1)) != receipt["node_version"]
                or _digest(node.content) != receipt["sha256"]
            ):
                raise BotError("MEMORY_CHANGED", "记忆目标已变化", 409)
            for expected in build_index_records(node):
                actual = self.index.get_record(expected.id)
                if (
                    actual is None
                    or actual.text != expected.text
                    or actual.filters.get("account_id") != ctx.account_id
                    or actual.filters.get("owner_space") != expected.filters["owner_space"]
                    or actual.metadata.get("canonical_revision")
                    != expected.metadata.get("canonical_revision")
                ):
                    return False
            if any(
                self.index.get_record(record_id) is not None
                for record_id in receipt.get("retired_ids", [])
            ):
                return False
        return True

    def verify_projection(self, tx, task, receipts):
        if not self._atomic_projection:
            if not self._projection_matches(self.context(task), receipts):
                raise WorkPendingError("记忆索引正在更新")
            return

        from psycopg2 import sql

        from fs.sql_adapter.session_mutation import _dsn_identity
        from index.index_record_builder import build_index_records

        if _dsn_identity(tx.store.pool._connection_string) != _dsn_identity(
            self.fs._connection_string
        ):
            raise ValueError("memoryBot 任务和canonical必须使用同一 SQL 连接配置")
        ctx = self.context(task)
        with tx.connection.cursor() as cur:
            for receipt in receipts:
                cur.execute(
                    "SELECT version,status FROM context_nodes WHERE account_id=%s AND uri=%s FOR SHARE",
                    (ctx.account_id, receipt["uri"]),
                )
                row = cur.fetchone()
                if row is None or row[0] != receipt["node_version"]:
                    raise BotError("MEMORY_CHANGED", "记忆目标已变化", 409)
                if receipt.get("action") == "forget":
                    if row[1] != "ARCHIVED":
                        raise BotError("MEMORY_CHANGED", "归档目标已变化", 409)
                    expected = []
                    ids = receipt["ids_to_delete"]
                else:
                    node = self._read_transaction_node(cur, receipt["uri"], ctx)
                    if _digest(node.content) != receipt["sha256"]:
                        raise BotError("MEMORY_CHANGED", "记忆内容已变化", 409)
                    expected = build_index_records(node)
                    ids = [record.id for record in expected] + receipt.get("retired_ids", [])
                cur.execute(
                    sql.SQL(
                        "SELECT id,text,filters,metadata FROM {} WHERE id=ANY(%s) FOR SHARE"
                    ).format(sql.Identifier(self.index._table_name)),
                    (ids,),
                )
                actual = {value[0]: value[1:] for value in cur.fetchall()}
                if receipt.get("action") == "forget":
                    if actual:
                        raise WorkPendingError("归档索引正在更新")
                if any(record_id in actual for record_id in receipt.get("retired_ids", [])):
                    raise WorkPendingError("旧记忆单元索引正在更新")
                for record in expected:
                    value = actual.get(record.id)
                    if (
                        not value
                        or value[0] != record.text
                        or value[1].get("account_id") != ctx.account_id
                        or value[1].get("owner_space") != record.filters["owner_space"]
                        or value[2].get("canonical_revision")
                        != record.metadata.get("canonical_revision")
                    ):
                        raise WorkPendingError("记忆索引正在更新")

    @staticmethod
    def citation(source):
        return {
            "id": source["citation_id"],
            "uri": source["uri"],
            "title": source.get("title", source["uri"].rsplit("/", 1)[-1])[:128],
            "quote": source.get("text", "")[:1024],
            "source_time": source.get("source_time") or None,
        }


READ_TOOLS = [
    {
        "name": "list_schedules",
        "description": "查询当前用户已有的提醒和周期计划，返回名称、状态、下次执行时间和时区。空列表表示当前没有计划。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "search_memory",
        "description": "搜索当前授权记忆与资料",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_memory",
        "description": "读取授权来源原文",
        "input_schema": {
            "type": "object",
            "properties": {"uri": {"type": "string"}},
            "required": ["uri"],
        },
    },
]
