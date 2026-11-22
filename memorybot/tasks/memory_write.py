"""Memory writes and projection receipt completion."""

from memorybot.models import jsonable


def run(execution):
    task_input = execution.task["input"]
    action = task_input["action"]
    if action in {"correct", "forget"}:
        target = execution.step(
            "retrieve",
            {"memory_uri": task_input["memory_uri"]},
            lambda value, *_: execution.gateway.prepare_target(execution.task, value["memory_uri"]),
        )
        mutation_input = {
            **task_input,
            "target_version": target["node_version"],
            "target_sha256": target["sha256"],
        }
        source_refs = list(mutation_input.get("source_refs") or target.get("source_refs") or [])
        if not source_refs:
            source_refs = [
                {
                    "file_id": None,
                    "task_id": None,
                    "uri": target["uri"],
                    "quote": target["text"][:1024],
                }
            ]
        mutation_input["source_refs"] = source_refs
        operation_id = None
        if action == "forget":
            record = execution.row("bot_forget", {"task_id": execution.task["task_id"]})
            operation_id = record["operation_id"]

        def apply_target(value, op, waiting, tx):
            receipt = execution.gateway.mutate_target(tx, execution.task, value, op)
            if value["action"] == "forget":
                tx.update(
                    "bot_forget",
                    {"task_id": execution.task["task_id"]},
                    {"source_refs": value["source_refs"]},
                )
            return receipt

        receipt = execution.step(
            "memory_apply",
            mutation_input,
            apply_target,
            atomic=True,
            operation_id=operation_id,
        )
        execution.memory_status("stored")
        execution.project([receipt])
        receipts = [receipt]
        status, text = "searchable", (
            "记忆已更正，可用于后续问答。"
            if action == "correct"
            else "已遗忘这条记忆，原始资料保留。"
        )
    else:
        messages = task_input.get("messages") or [
            {
                "id": execution.task["task_id"] + "_user",
                "role": "user",
                "content": "请记住：" + task_input["new_text"],
                "created_at": jsonable(execution.task["created_at"]),
            }
        ]
        if action == "capture":
            excluded = execution.excluded()
            denied_tasks = {
                ref["task_id"]
                for row in excluded
                for ref in row["source_refs"]
                if ref.get("task_id")
            }
            messages = (
                []
                if execution.task.get("parent_task_id") in denied_tasks
                else [
                    {
                        **row,
                        "content": execution.gateway.filter_text(row["content"], excluded),
                    }
                    for row in messages
                ]
            )
        # Recheck the original receipt so retries resume the same capture operation.
        capture_result = execution.step(
            "memory_submit",
            {"messages": messages},
            lambda value, op, waiting: (
                {"sources": [], "memory_status": "unchanged"}
                if not value["messages"]
                else (
                    execution.gateway.capture_recheck(execution.task, op, value)
                    if waiting
                    else execution.gateway.capture(
                        execution.task,
                        value["messages"],
                        op,
                        timeout=execution.config.file_timeout_seconds,
                    )
                )
            ),
        )
        status = capture_result["memory_status"]
        if capture_result["sources"]:
            # Searchability requires the index to catch up with the stored content.
            execution.memory_status("stored")
            execution.project(capture_result["sources"])
            status = "searchable"
        receipts = capture_result["sources"]
        text = (
            "记忆已保存，可用于后续问答。" if status == "searchable" else "记忆已检查，无需更新。"
        )

    def complete(tx):
        if receipts:
            execution.gateway.verify_projection(tx, execution.task, receipts)
        execution._memory_status(tx, status)
        if action == "forget":
            tx.update(
                "bot_forget",
                {"task_id": execution.task["task_id"]},
                {"state": "effective", "source_refs": source_refs},
            )
        if action == "remember" and task_input.get("memory_uri") and status == "searchable":
            tx.update(
                "bot_forget",
                {"memory_uri": task_input["memory_uri"], "state": "effective"},
                {
                    "state": "superseded",
                    "relearn_request_id": execution.task["request_id"],
                },
            )

    execution.store.finish(
        *execution.identity,
        {"text": text, "citations": [], "file_ids": []},
        mutate=complete,
    )
