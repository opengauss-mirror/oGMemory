"""Authorized conversation inputs, material loading and citation registration."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath

from memorybot.files import parse_text
from memorybot.gateway import WorkPendingError
from memorybot.models import BotError
from memorybot.service import parsed_mime_type
from memorybot.templates import TemplateRegistry


class ConversationContext:
    def __init__(self, execution):
        self.execution = execution
        self.attachment_source_versions = {}

    def load(self):
        ctx = self.prepare()
        self.load_context(ctx)
        self.load_documents()

    def prepare(self):
        execution = self.execution
        self.data = execution.task["input"]
        self.session = execution.row("bot_session", {"session_id": execution.task["session_id"]})
        self.excluded = execution.excluded()
        denied_files = {
            ref["file_id"]
            for row in self.excluded
            for ref in row["source_refs"]
            if ref.get("file_id")
        }
        self.selected_files = [
            execution.row("bot_file", {"file_id": file_id})
            for file_id in self.data.get("file_ids", [])
            if file_id not in denied_files
        ]
        self.template_id = self.data.get("template_id")
        self.template = execution.step(
            "retrieve",
            {"template_id": self.template_id},
            lambda value, *_: (
                (
                    execution.worker.templates
                    or execution.gateway.templates(execution.gateway.context(execution.task))
                ).validate(
                    value["template_id"],
                    self.data.get("text", ""),
                    [{"mime_type": parsed_mime_type(file)} for file in self.selected_files],
                    self.data.get("output_format", "markdown"),
                )
                if value["template_id"]
                else {}
            ),
        )
        scope = dict(self.session["scope"])
        attached_uris = (
            ref["uri"]
            for file in self.selected_files
            for ref in file["metadata"].get("memory_sources", [])
        )
        scope["resource_ids"] = list(dict.fromkeys([*scope.get("resource_ids", []), *attached_uris]))
        # Replay the frozen template with its source permissions rechecked.
        execution.active_template, execution.template_scope = self.template, scope
        ctx = execution.current()
        self.execution_scope = (
            TemplateRegistry.execution_scope(self.template, ctx, scope) if self.template else scope
        )
        return ctx

    def load_context(self, ctx):
        execution = self.execution
        denied_tasks = {
            ref["task_id"]
            for row in self.excluded
            for ref in row["source_refs"]
            if ref.get("task_id")
        }
        with execution.store.transaction(execution.account, execution.user) as tx:
            tx.cursor.execute(
                "SELECT * FROM bot_task WHERE account_id=%s AND user_id=%s "
                "AND session_id=%s AND kind='chat' AND seq<%s ORDER BY seq DESC LIMIT 30",
                (
                    execution.account,
                    execution.user,
                    self.session["session_id"],
                    execution.task["seq"],
                ),
            )
            history = [dict(row) for row in tx.cursor.fetchall()]
        messages = []
        for prior in reversed(history):
            if prior["task_id"] in denied_tasks:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": execution.gateway.filter_text(
                        prior["input"].get("text", ""), self.excluded
                    ),
                }
            )
            if prior.get("result"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": execution.gateway.filter_text(
                            prior["result"]["text"], self.excluded
                        ),
                    }
                )
        messages.append({"role": "user", "content": self.data.get("text", "")})
        # Tie cached retrieval to the current authorization and forgetting scope.
        authorization_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "excluded": [
                        {"uri": row["memory_uri"], "source_refs": row["source_refs"]}
                        for row in self.excluded
                    ],
                    "scope": self.execution_scope,
                    "visible_owner_spaces": sorted(ctx.visible_owner_spaces),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.retrieved = execution.step(
            "retrieve",
            {
                "messages": messages,
                "scope": self.execution_scope,
                "authorization_fingerprint": authorization_fingerprint,
            },
            lambda value, *_: execution.gateway.retrieve(
                execution.task, value["messages"], value["scope"], self.excluded
            ),
        )
        self.sources, self.sources_by_uri = ([], {})
        self.register_sources(self.retrieved["sources"])

    def register_sources(self, values):
        execution = self.execution
        denied = {row["memory_uri"] for row in self.excluded}
        quotes = {
            ref["quote"]
            for row in self.excluded
            for ref in row.get("source_refs", ())
            if ref.get("quote")
        }
        registered = []
        for source in values:
            uri = source["uri"]
            if uri in denied:
                continue
            if quotes and any(
                quote in text
                for text in (
                    source["text"],
                    source.get("title") or "",
                    *(ref.get("quote") or "" for ref in source.get("source_refs") or ()),
                )
                for quote in quotes
            ):
                continue
            if not execution.gateway.in_scope(
                source, execution.gateway.context(execution.task), self.execution_scope
            ):
                continue
            current = self.sources_by_uri.get(uri)
            if current is not None and source.get("node_version", 1) <= current.get("node_version", 1):
                source = current
            self.require_source_version(source, self.attachment_source_versions.get(uri))
            if current is None:
                current = {**source, "citation_id": f"S{len(self.sources) + 1}"}
                self.sources.append(current)
                self.sources_by_uri[uri] = current
            elif source is not current:
                citation_id = current["citation_id"]
                current.clear()
                current.update(source, citation_id=citation_id)
            registered.append(current)
        return registered

    @staticmethod
    def require_source_version(source, expected):
        if expected is not None and source.get("node_version") != expected:
            raise BotError("REVISION_CONFLICT", "所选资料的记忆来源已更新，请重新生成资料", 409)

    def load_documents(self):
        execution = self.execution
        self.documents = []
        self.document_citations = []
        for selected in self.selected_files:
            file_id = selected["file_id"]
            references = []
            for ref in selected["metadata"].get("memory_sources", []):
                source = execution.gateway.read_memory(
                    execution.gateway.context(execution.task), ref["uri"]
                )
                if not execution.gateway.in_scope(
                    source, execution.gateway.context(execution.task), self.execution_scope
                ):
                    raise BotError("PERMISSION_DENIED", "所选资料来源超出模板当前读取范围", 403)
                version = ref.get("node_version")
                self.require_source_version(source, version)
                if version is not None:
                    self.attachment_source_versions[source["uri"]] = version
                references.append(source)
            document = execution.step(
                "retrieve",
                {"file_id": file_id, "scope": self.execution_scope},
                lambda value, *_: self.document(value["file_id"]),
            )
            document["text"] = execution.gateway.filter_text(document["text"], self.excluded)
            self.documents.append(document)
            self.document_citations.extend(self.register_sources(references))

    def document(self, file_id):
        execution = self.execution
        file = execution.row("bot_file", {"file_id": file_id})
        if file["state"] != "ready":
            raise WorkPendingError("资料正在处理")
        execution.gateway.context(execution.task)
        key = file["storage_key"]
        if file["purpose"] == "source":
            parsed_key = str(PurePosixPath(key).with_name("parsed.json"))
            parsed = json.loads(execution.files.read(execution.account, parsed_key))
        else:
            parsed = parse_text(execution.files.read(execution.account, key), file["name"])
        return {
            "file_id": file_id,
            "name": file["name"],
            "mime_type": parsed.get("mime_type", file["mime_type"]),
            "text": parsed["text"],
            "purpose": file["purpose"],
            "memory_sources": file["metadata"].get("memory_sources", []),
        }
