from __future__ import annotations

import json
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import PurePosixPath
from threading import Thread

import pytest

from memorybot import files as bot_files
from memorybot.models import ChatInput, FileCreate, new_id
from memorybot.service import BotService
from tests.integration.test_memorybot_runtime import execute
from tests.integration.test_memorybot_runtime import runtime as runtime

pytestmark = pytest.mark.integration


@pytest.fixture
def webpage_url(monkeypatch):
    class ReleasePage(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                "<html><title>发布资料</title><p>林悦负责青禾发布。</p></html>".encode()
            )

    with HTTPServer(("127.0.0.1", 0), ReleasePage) as server:
        monkeypatch.setattr(
            bot_files,
            "_PinnedHTTPS",
            lambda _host, _port, _address, timeout: HTTPConnection(
                *server.server_address, timeout=timeout
            ),
        )
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield "https://93.184.216.34/release"
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize(
    ("template_id", "legacy_metadata"), [("weekly_report", True), ("handbook", False)]
)
def test_public_webpage_parsed_text_generates_template_artifact(
    runtime, webpage_url, template_id, legacy_metadata
):
    ctx = runtime.ctx
    service = BotService(
        runtime.store,
        runtime.memory,
        runtime.files,
        runtime.worker.config,
        gateway=runtime.gateway,
    )
    with runtime.store.transaction(ctx.account_id, ctx.user_id) as tx:
        status, accepted = service.create_file(
            ctx,
            tx,
            FileCreate(source_kind="url", source_url=webpage_url).model_dump(mode="json"),
            new_id("request"),
        )
    assert status == 202
    execute(runtime, accepted["task"])
    file_id = accepted["file"]["file_id"]
    file = runtime.store.get(ctx.account_id, "bot_file", {"file_id": file_id})
    assert file["state"] == "ready" and file["mime_type"] == "text/html"
    assert file["metadata"]["parsed_mime_type"] == "text/plain"
    original = runtime.files.read(ctx.account_id, file["storage_key"])
    assert b"<html" in original.lower()
    parsed_key = str(PurePosixPath(file["storage_key"]).with_name("parsed.json"))
    parsed = json.loads(runtime.files.read(ctx.account_id, parsed_key))
    assert parsed["mime_type"] == "text/plain" and parsed["text"] == "林悦负责青禾发布。"
    if legacy_metadata:
        metadata = dict(file["metadata"])
        metadata.pop("parsed_mime_type")
        with runtime.store.transaction(ctx.account_id, ctx.user_id) as tx:
            tx.update("bot_file", {"file_id": file_id}, {"metadata": metadata})

    def summarize(messages, emit):
        if template_id == "handbook":
            materials = json.loads(messages[-1]["content"])["materials"]
            text = "\n".join(item["text"] if isinstance(item, dict) else item for item in materials)
        else:
            materials = json.loads(messages[1]["content"].split("\n", 1)[1])
            text = materials["documents"][0]["text"]
            text += "\n" + " ".join(f"[{item['citation_id']}]" for item in materials["sources"])
        output = "# 网页资料\n\n" + text
        if emit:
            emit(output)
        return [], output

    runtime.llm.stream_hook = summarize
    with runtime.store.transaction(ctx.account_id, ctx.user_id) as tx:
        status, chat = service.create_chat(
            ctx,
            tx,
            ctx.session_id,
            ChatInput(
                text="整理这份网页资料。", template_id=template_id, file_ids=[file_id]
            ).model_dump(mode="json"),
            new_id("request"),
        )
    assert status == 202
    completed = execute(runtime, chat)
    artifact_id = completed["result"]["file_ids"][0]
    artifact = runtime.store.get(ctx.account_id, "bot_file", {"file_id": artifact_id})
    assert artifact["state"] == "ready"
    assert parsed["text"] in runtime.files.read(ctx.account_id, artifact["storage_key"]).decode()
    source = runtime.store.get(ctx.account_id, "bot_file", {"file_id": file_id})
    assert source["mime_type"] == "text/html"
    assert runtime.files.read(ctx.account_id, source["storage_key"]) == original
