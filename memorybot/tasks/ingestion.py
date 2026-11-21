"""Source ingestion and shared-file publication."""

from memorybot.files import parse_text
from memorybot.models import jsonable


def run(execution):
    source_file = execution.row("bot_file", {"file_id": execution.task["input"]["file_id"]})

    def parse(value, *_):
        if source_file["source_kind"] == "url":
            page = execution.files.fetch(source_file["source_url"])
            publication = execution.files.publish(
                execution.account,
                source_file["file_id"],
                page["data"],
                name=page["name"],
                mime_type=page["mime_type"],
            )
            parsed = {
                "text": page["text"],
                "mime_type": "text/plain",
                "summary": page["text"][:4096],
                "parsed_ranges": [
                    {
                        "unit": "line",
                        "start": 1,
                        "end": max(1, len(page["text"].splitlines())),
                    }
                ],
            }
            publication["source_url"] = page["source_url"]
        else:
            parsed = parse_text(
                execution.files.read(execution.account, source_file["storage_key"]),
                source_file["name"],
            )
            publication = {
                key: source_file[key]
                for key in (
                    "storage_key",
                    "size_bytes",
                    "sha256",
                    "name",
                    "mime_type",
                )
            }
        execution.files.write_parsed(execution.account, publication["storage_key"], parsed)
        return {"parsed": parsed, "publication": publication}

    # Persist shared content before indexing so every chunk references durable bytes.
    parse_result = execution.step("file_publish", {"file_id": source_file["file_id"]}, parse)
    source_file.update(parse_result["publication"])
    receipts = []
    source_text = parse_result["parsed"]["text"]
    for chunk_index, offset in enumerate(range(0, max(1, len(source_text)), 12000)):
        receipts.extend(
            execution.step(
                "memory_apply",
                {
                    "file": source_file,
                    "text": source_text[offset : offset + 12000],
                    "index": chunk_index,
                },
                lambda value, op, waiting, tx: execution.gateway.publish_source(
                    tx,
                    execution.task,
                    value["file"],
                    value["text"],
                    op,
                    chunk_index=value["index"],
                ),
                atomic=True,
            )["sources"]
        )
        execution.event(
            "progress",
            {
                "phase": "ingest",
                "message": "已登记资料分块",
                "completed": min(offset + 12000, len(source_text)),
                "total": len(source_text),
            },
        )
    metadata = {
        "summary": parse_result["parsed"]["summary"],
        "parsed_mime_type": parse_result["parsed"]["mime_type"],
        "source_time": jsonable(source_file["created_at"]),
        "parsed_ranges": parse_result["parsed"]["parsed_ranges"],
        "memory_sources": [
            {
                "uri": row["uri"],
                "node_version": row["node_version"],
                "indexed_version": None,
            }
            for row in receipts
        ],
    }
    with execution.store.execution(*execution.identity) as tx:
        tx.update(
            "bot_file",
            {"file_id": source_file["file_id"]},
            {**parse_result["publication"], "metadata": metadata},
        )
    execution.project(receipts)
    for source in metadata["memory_sources"]:
        source["indexed_version"] = source["node_version"]
    result = {
        "text": parse_result["parsed"]["summary"],
        "citations": [],
        "file_ids": [source_file["file_id"]],
    }

    # Recheck projections at commit so a ready file has searchable source chunks.
    def publish(tx):
        if tx.task["cancel_requested"]:
            return
        execution.gateway.verify_projection(tx, execution.task, receipts)
        tx.update(
            "bot_file",
            {"file_id": source_file["file_id"]},
            {"state": "ready", "metadata": metadata},
        )

    execution.store.finish(*execution.identity, result, mutate=publish)
