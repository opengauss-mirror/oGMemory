"""Conversation output, artifacts and atomic follow-up publication."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

from memorybot.handbook import split_text
from memorybot.models import BotError, jsonable, new_id


class ConversationResult:
    def __init__(self, execution, context):
        self.execution = execution
        self.context = context

    def publish(self, text, csv_body, memory_action_used, stream_result):
        cited_ids = self.publish_result(text, stream_result)
        artifacts = self.publish_artifact(text, csv_body, cited_ids)
        self.complete(artifacts, memory_action_used)

    def publish_result(self, text, stream_result):
        execution = self.execution
        if not text.strip():
            raise RuntimeError("模型尚未生成完整正文")
        cited_ids = dict.fromkeys(re.findall(r"\[(S[1-9][0-9]*)\]", text))
        sources_by_id = {source["citation_id"]: source for source in self.context.sources}
        citations = [
            execution.gateway.citation(sources_by_id[citation_id])
            for citation_id in cited_ids
            if citation_id in sources_by_id
        ][:100]
        execution.partial = {
            "text": text.encode("utf-8")[:524288].decode("utf-8", errors="ignore"),
            "citations": citations,
            "file_ids": [],
        }
        if stream_result:
            for chunk in split_text(execution.partial["text"], 8192):
                execution.event(
                    "answer.delta",
                    {"attempt": execution.task["attempt"], "text": chunk},
                )
        return cited_ids

    def publish_artifact(self, text, csv_body, cited_ids):
        execution = self.execution
        artifacts = []
        if self.context.template_id or self.context.data.get("file_ids"):
            output_format = self.context.data.get("output_format", "markdown")
            content = (
                csv_body or re.sub(r"^```(?:csv)?\s*|\s*```$", "", text.strip())
                if output_format == "csv"
                else text
            )
            file_id = (
                "f_"
                + hashlib.sha256((execution.task["task_id"] + ":artifact").encode()).hexdigest()[
                    :32
                ]
            )
            extension = "csv" if output_format == "csv" else "md"
            # Publish shared bytes before committing artifact metadata with the task result.
            publication = execution.step(
                "file_publish",
                {
                    "file_id": file_id,
                    "text": content,
                    "name": f"{self.context.template_id or 'revision'}.{extension}",
                },
                lambda value, *_: execution.files.publish(
                    execution.account,
                    value["file_id"],
                    value["text"].encode("utf-8"),
                    name=value["name"],
                ),
            )
            previous_file_id = next(
                (row["file_id"] for row in self.context.documents if row["purpose"] == "artifact"),
                None,
            )
            artifacts.append(
                {
                    **publication,
                    "file_id": file_id,
                    "task_id": execution.task["task_id"],
                    "purpose": "artifact",
                    "source_kind": "generated",
                    "previous_file_id": previous_file_id,
                    "scope": self.context.session["scope"],
                    "state": "ready",
                    "metadata": {
                        "summary": text[:4096],
                        "source_time": None,
                        "parsed_ranges": [],
                        "memory_sources": [
                            {
                                "uri": source["uri"],
                                "node_version": source.get("node_version"),
                                "indexed_version": source.get("node_version"),
                            }
                            for source in self.context.sources
                            if source["citation_id"] in cited_ids
                        ],
                    },
                }
            )
            execution.partial["file_ids"] = [file_id]
        return artifacts

    def complete(self, artifacts, memory_action_used):
        execution = self.execution
        baseline, notify = self.risk_baseline(self.context.sources)
        children, deliveries = ([], [])
        user_text = execution.gateway.filter_text(
            self.context.data.get("text", ""), self.context.excluded
        )
        if user_text.strip() and (not memory_action_used):
            children.append(
                {
                    "kind": "memory_write",
                    "request_id": hashlib.sha256(
                        (execution.task["task_id"] + ":capture").encode()
                    ).hexdigest(),
                    "input": {
                        "action": "capture",
                        "messages": [
                            {
                                "id": execution.task["task_id"] + "_user",
                                "role": "user",
                                "content": user_text,
                                "created_at": jsonable(execution.task["created_at"]),
                            }
                        ],
                    },
                    "memory_status": "pending",
                }
            )
        target = self.context.data.get("delivery") or {
            "channel": "web",
            "target_id": self.context.session["session_id"],
        }
        if target["channel"] != "web":
            raise BotError("DELIVERY_UNAVAILABLE", "交付目标必须为 Web 会话", 409)
        if notify and execution.task.get("schedule_id"):
            child_id, delivery_id = (new_id("t"), new_id("d"))
            children.append(
                {
                    "task_id": child_id,
                    "kind": "delivery",
                    "request_id": delivery_id,
                    "input": {"delivery_id": delivery_id},
                }
            )
            deliveries.append(
                {
                    "channel": "web",
                    "target_id": target.get("target_id") or self.context.session["session_id"],
                    "delivery_id": delivery_id,
                    "task_id": child_id,
                    "source_task_id": execution.task["task_id"],
                    "operation_id": new_id("op"),
                    "content": execution.partial,
                }
            )

        def publish(tx):
            if (
                children
                and children[0]["kind"] == "memory_write"
                and (not tx.task["cancel_requested"])
            ):
                execution.store.set_memory_status(tx, execution.task["task_id"], "pending")
            if baseline is not None and (not tx.task["cancel_requested"]):
                execution.store.update_schedule_baseline(tx, execution.task, baseline)

        execution.current()
        # Commit follow-up tasks with the reply so background work has a durable entry point.
        execution.store.finish(
            *execution.identity,
            execution.partial,
            files=artifacts,
            children=children,
            deliveries=deliveries,
            mutate=publish,
        )

    def risk_baseline(self, sources):
        execution = self.execution
        snapshot = execution.task.get("schedule_snapshot") or {}
        if not snapshot.get("condition"):
            return (None, True)
        items = []
        seen = set()
        project = snapshot["condition"].get("project")
        for source in sources:
            if source.get("category") not in {
                "entity",
                "event",
                "case",
                "document",
                "tool",
                "experience",
                "pattern",
            }:
                continue
            if (
                project
                and source.get("project") != project
                and (project not in source["text"])
                and (project not in source.get("title", ""))
            ):
                continue
            for line in source["text"].splitlines():
                fact_hash = hashlib.sha256(line.strip().encode()).hexdigest()
                if (
                    re.search("风险|risk|阻塞|延期|待确认", line, re.IGNORECASE)
                    and fact_hash not in seen
                ):
                    seen.add(fact_hash)
                    items.append(
                        {
                            "uri": source["uri"],
                            "node_version": source.get("node_version", 1),
                            "fact_hash": fact_hash,
                        }
                    )
        prior = snapshot.get("baseline") or {}
        old = {item["fact_hash"] for item in prior.get("items", [])}
        notify = bool(prior) and any(item["fact_hash"] not in old for item in items)
        return (
            {
                "task_id": execution.task["task_id"],
                "checked_at": datetime.now(UTC).isoformat(),
                "items": items,
            },
            notify,
        )
