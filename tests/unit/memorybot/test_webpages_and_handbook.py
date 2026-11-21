import http.client
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from memorybot import files
from memorybot.config import BotConfig
from memorybot.handbook import build_handbook


@pytest.mark.parametrize("charset", ["gb18030", "gbk", None])
def test_webpage_unicode_request_preserves_escapes_and_decodes_html(monkeypatch, charset):
    text = "交接资料，负责人林悦。"
    page = f"<html><title>投产核对</title><p>{text}</p></html>".encode(charset or "utf-8-sig")
    requests = []

    class PageHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            content_type = f"text/html; charset={charset}" if charset else "text/html"
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *args):
            pass

    connect = http.client.HTTPConnection
    resolve = socket.getaddrinfo

    def resolve_docs(host, *args, **kwargs):
        if host == "docs.example":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.4.4", 80))]
        return resolve(host, *args, **kwargs)

    with ThreadingHTTPServer(("127.0.0.1", 0), PageHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setattr(files.socket, "getaddrinfo", resolve_docs)
            monkeypatch.setattr(
                files.http.client,
                "HTTPConnection",
                lambda _address, _port, timeout: connect(
                    "127.0.0.1", server.server_port, timeout=timeout
                ),
            )
            result = files.fetch_page("http://docs.example/资料.html?q=交接&next=%2Fdocs")
        finally:
            server.shutdown()
            thread.join()

    assert requests == ["/%E8%B5%84%E6%96%99.html?q=%E4%BA%A4%E6%8E%A5&next=%2Fdocs"]
    assert result["data"] == page
    assert result["text"] == text
    assert result["name"] == "投产核对"


def test_handbook_rebatches_large_intermediate_summary_and_keeps_chapters():
    class Execution:
        config = BotConfig(max_input_tokens=2000, max_output_tokens=8000)
        task = {"input": {"text": "整理投产核对资料。"}}
        summary_calls = 0

        def model(self, messages, *, tools, stream, progress):
            input_bytes = sum(len(row["content"].encode("utf-8")) for row in messages)
            assert input_bytes // 3 + 1 <= self.config.max_input_tokens
            example = json.loads(messages[-1]["content"])
            if progress["phase"] == "handbook_summary":
                self.summary_calls += 1
                text = "投产核对总览。" + ("证" * 2350 if self.summary_calls == 1 else "")
            else:
                text = example["materials"][0]["title"] + " [S1] " + "A" * 2450
            return {"calls": [], "text": text}

        def step(self, _kind, value, callback):
            return callback(value)

        def event(self, *args):
            pass

    uri = "ctx://a/users/u/memories/documents/guide"
    documents = [
        dict(file_id=f"f{i}", name=f"第{i}份", text="已备份。", memory_sources=[{"uri": uri}])
        for i in range(1, 7)
    ]
    execution = Execution()
    result = build_handbook(
        execution,
        documents,
        {"instruction": "保留事实和来源。"},
        [{"uri": uri, "citation_id": "S1"}],
    )

    assert execution.summary_calls > len(documents)
    assert "投产核对总览。" in result
    assert all(f"## {i}. 第{i}份" in result for i in range(1, 7))
    assert f"[S1] {uri}" in result
