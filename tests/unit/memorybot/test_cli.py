"""Positive CLI journeys against an authenticated HTTP contract fixture."""

from __future__ import annotations

import json
import threading
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from cli.main import main


@pytest.fixture
def bot_http(monkeypatch, request):
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    stream_mode = getattr(request, "param", False)
    truncate_stream = stream_mode is True
    pending_child = stream_mode == "pending_child"
    calls = []
    stream_connections = []
    artifact = "客户,合计\n甲,25\n乙,20\n".encode()
    schedule = {
        "schedule_id": "p_friday",
        "revision": 3,
        "state": "active",
        "rule": {"type": "weekly", "weekdays": [2], "local_time": "17:00:00"},
        "task_input": {
            "text": "生成周报",
            "file_ids": ["f_notes"],
            "template_id": "weekly_report",
            "output_format": "markdown",
            "delivery": {"channel": "web", "binding_id": None, "target_id": "s_qinghe"},
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, data, status=200):
            raw = json.dumps(
                {"request_id": self.headers.get("Idempotency-Key", "read_1"), "data": data},
                ensure_ascii=False,
            ).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def route(self):
            parsed = urlsplit(self.path)
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            content_type = self.headers.get("Content-Type", "")
            body = json.loads(raw) if raw and "application/json" in content_type else None
            calls.append(
                {
                    "method": self.command,
                    "path": parsed.path,
                    "query": parse_qs(parsed.query),
                    "headers": dict(self.headers),
                    "body": body,
                    "raw": raw,
                }
            )
            path = parsed.path.removeprefix("/api/v1/bot")
            if path.endswith("/content"):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Length", str(len(artifact)))
                self.end_headers()
                self.wfile.write(artifact)
            elif path.endswith("/events"):
                stream_connections.append(self.headers)
                if pending_child:
                    events = [(1, "task.completed", {"state": "succeeded"})]
                elif len(stream_connections) == 1:
                    events = [(1, "progress", {"message": "已读取资料", "phase": "context"})]
                else:
                    events = [
                        (2, "answer.delta", {"attempt": 1, "text": "负责人是林悦。"}),
                        (3, "task.completed", {"state": "succeeded"}),
                        (4, "memory.updated", {"memory_status": "searchable"}),
                    ]
                text = "".join(
                    f"id: t_answer:{seq}\nevent: {kind}\ndata: "
                    + json.dumps(
                        {"task_id": "t_answer", "event_seq": seq, "payload": value},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                    for seq, kind, value in events
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if truncate_stream and len(stream_connections) == 1:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self.wfile.write(f"{len(text):X}\r\n".encode() + text + b"\r\n10\r\npartial")
                    self.close_connection = True
                    return
                self.send_header("Content-Length", str(len(text)))
                self.end_headers()
                self.wfile.write(text)
            elif path == "/tasks/t_answer":
                if pending_child:
                    self.respond({
                        "task_id": "t_answer",
                        "state": "succeeded",
                        "last_event_seq": 2,
                        "memory_status": "pending",
                        "pending_memory_confirmations": [{
                            "task_id": "t_memory",
                            "confirmation": {"operation_id": "op_capture", "actions": ["recheck"]},
                        }],
                    })
                    return
                self.respond(
                    {
                        "task_id": "t_answer",
                        "state": "succeeded" if len(stream_connections) > 1 else "running",
                        "last_event_seq": 4 if len(stream_connections) > 1 else 1,
                        "result": {"text": "负责人是林悦。", "citations": [], "file_ids": []},
                    }
                )
            elif path.startswith("/tasks/"):
                self.respond({"task_id": "t_weekly", "state": "queued", "confirmation": body})
            elif path == "/sessions" and self.command == "POST":
                self.respond({"session_id": "s_qinghe", **body}, 201)
            elif path == "/sessions/s_qinghe/tasks":
                self.respond({"task_id": "t_weekly", "state": "queued", "input": body}, 202)
            elif path == "/sessions/s_qinghe":
                if self.command == "PATCH":
                    self.respond({"session_id": "s_qinghe", "revision": 4, **body})
                    return
                self.respond(
                    {
                        "session": {"session_id": "s_qinghe", "title": "青禾", "revision": 3},
                        "messages": [{"task_id": "t_weekly", "seq": 1}],
                        "next_cursor": None,
                    }
                )
            elif path == "/files" and self.command == "POST":
                self.respond(
                    {
                        "file": {"file_id": "f_notes", "state": "processing"},
                        "task": {"task_id": "t_ingest", "state": "queued"},
                    },
                    202,
                )
            elif path == "/schedules" and self.command == "POST":
                self.respond({"schedule_id": "p_friday", "revision": 1, **body}, 201)
            elif path == "/schedules/p_friday" and self.command == "GET":
                self.respond(schedule)
            elif path == "/schedules/p_friday":
                self.respond({**schedule, "revision": 4, "state": "paused"})
            elif path == "/memory-actions":
                self.respond(
                    {
                        "task": {"task_id": "t_memory", "state": "queued"},
                        "forget_id": "forget_1" if body["action"] == "forget" else None,
                    },
                    202,
                )
            elif path == "/config":
                self.respond({"enabled": True, "max_file_bytes": 52428800})
            else:
                self.respond({"items": [], "next_cursor": None})

    for method in ("GET", "POST", "PATCH", "DELETE"):
        setattr(Handler, "do_" + method, Handler.route)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    yield base, calls, artifact, stream_connections
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def invoke(server, *args):
    return main(["bot", "--base-url", server[0], "--api-key", "user-test-key", *args])


def test_create_session_and_submit_work_with_stable_request(bot_http, capsys):
    assert invoke(bot_http, "session", "create", "发布准备", "--memory-scope", "user") == 0
    assert json.loads(capsys.readouterr().out)["data"]["session_id"] == "s_qinghe"
    assert (
        invoke(
            bot_http,
            "--request-id",
            "req_weekly",
            "chat",
            "s_qinghe",
            "整理周报",
            "--file-id",
            "f_notes",
            "--template-id",
            "weekly_report",
        )
        == 0
    )
    call = bot_http[1][-1]
    assert call["headers"]["Authorization"] == "Bearer user-test-key"
    assert call["headers"]["Idempotency-Key"] == "req_weekly"
    assert call["body"]["file_ids"] == ["f_notes"]
    assert call["body"]["template_id"] == "weekly_report"
    assert json.loads(capsys.readouterr().out)["data"]["task_id"] == "t_weekly"


def test_read_paginated_history_and_templates(bot_http, capsys):
    assert (
        invoke(bot_http, "session", "show", "s_qinghe", "--limit", "10", "--cursor", "page_2") == 0
    )
    assert bot_http[1][-1]["query"] == {"limit": ["10"], "cursor": ["page_2"]}
    assert json.loads(capsys.readouterr().out)["data"]["messages"][0]["seq"] == 1
    assert invoke(bot_http, "template", "list", "--query", "周报") == 0
    assert bot_http[1][-1]["query"]["q"] == ["周报"]


def test_upload_source_and_download_exact_artifact(bot_http, tmp_path, capsys):
    source = tmp_path / "input.csv"
    source.write_bytes("客户,金额\n甲,10\n甲,15\n乙,20\n".encode())
    assert invoke(bot_http, "file", "upload", str(source), "--memory-scope", "user") == 0
    call = bot_http[1][-1]
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {call['headers']['Content-Type']}\r\n\r\n".encode() + call["raw"]
    )
    parts = {
        part.get_param("name", header="content-disposition"): part for part in message.iter_parts()
    }
    assert parts["file"].get_payload(decode=True) == source.read_bytes()
    assert json.loads(parts["scope"].get_payload(decode=True)) == {
        "memory_scope": "user",
        "resource_ids": [],
    }
    capsys.readouterr()
    output = tmp_path / "result.csv"
    assert invoke(bot_http, "file", "download", "f_result", "--output", str(output)) == 0
    assert output.read_bytes() == bot_http[2]
    assert json.loads(capsys.readouterr().out)["size_bytes"] == len(bot_http[2])


def test_create_pause_and_update_weekly_plan(bot_http, capsys):
    assert (
        invoke(
            bot_http,
            "schedule",
            "create",
            "s_qinghe",
            "周五周报",
            "生成周报",
            "--weekdays",
            "5",
            "--local-time",
            "17:00:00",
            "--timezone",
            "Asia/Shanghai",
        )
        == 0
    )
    body = bot_http[1][-1]["body"]
    assert body["rule"] == {"type": "weekly", "weekdays": [5], "local_time": "17:00:00"}
    assert body["task_input"]["text"] == "生成周报"
    capsys.readouterr()
    assert invoke(bot_http, "schedule", "pause", "p_friday") == 0
    assert bot_http[1][-1]["body"] == {"action": "pause", "expected_revision": 3}
    capsys.readouterr()
    assert (
        invoke(
            bot_http,
            "schedule",
            "update",
            "p_friday",
            "--text",
            "生成客户版周报",
            "--revision",
            "3",
        )
        == 0
    )
    update = bot_http[1][-1]["body"]
    assert update["task_input"]["text"] == "生成客户版周报"
    assert update["task_input"]["file_ids"] == ["f_notes"]
    for option, value, weekdays, local_time in (
        ("--weekdays", "5", [5], "17:00:00"),
        ("--local-time", "18:00:00", [2], "18:00:00"),
    ):
        assert invoke(bot_http, "schedule", "update", "p_friday", option, value) == 0
        assert bot_http[1][-1]["body"]["rule"] == {
            "type": "weekly", "weekdays": weekdays, "local_time": local_time,
        }
    assert update["expected_revision"] == 3


def test_memory_correction_forget_and_explicit_remember(bot_http, capsys):
    uri = "ctx://acct/users/user/memories/preference"
    assert invoke(bot_http, "memory", "correct", uri, "喜欢乌龙茶", "--source-file", "f_notes") == 0
    body = bot_http[1][-1]["body"]
    assert body["source_refs"] == [
        {"file_id": "f_notes", "task_id": None, "uri": None, "quote": ""}
    ]
    assert body["new_text"] == "喜欢乌龙茶"
    capsys.readouterr()
    assert invoke(bot_http, "memory", "forget", uri) == 0
    assert json.loads(capsys.readouterr().out)["data"]["forget_id"] == "forget_1"
    assert invoke(bot_http, "memory", "remember", "现在喜欢红茶", "--uri", uri) == 0
    assert bot_http[1][-1]["body"]["action"] == "remember"
    assert bot_http[1][-1]["body"]["memory_uri"] == uri


def test_public_bot_capabilities(bot_http, capsys):
    assert invoke(bot_http, "config") == 0
    assert json.loads(capsys.readouterr().out)["data"] == {
        "enabled": True,
        "max_file_bytes": 52428800,
    }


@pytest.mark.parametrize("bot_http", [False, True], indirect=True)
def test_stream_reconnects_with_cursor_and_reads_final_answer(bot_http, capsys, monkeypatch):
    monkeypatch.setattr("cli.bot.time.sleep", lambda seconds: None)
    assert invoke(bot_http, "task", "events", "t_answer") == 0
    assert len(bot_http[3]) == 2
    assert bot_http[3][1]["Last-Event-ID"] == "t_answer:1"
    output = capsys.readouterr().out
    assert "已读取资料" in output
    assert "负责人是林悦。" in output
    assert '"state": "succeeded"' in output
    assert '"memory_status": "searchable"' in output


def test_task_cancel_and_receipt_recheck(bot_http, capsys):
    assert invoke(bot_http, "task", "cancel", "t_weekly") == 0
    assert bot_http[1][-1]["body"] == {}
    capsys.readouterr()
    assert invoke(bot_http, "task", "confirm", "t_weekly", "--operation-id", "op_delivery") == 0
    assert bot_http[1][-1]["body"] == {"operation_id": "op_delivery", "decision": "recheck"}


@pytest.mark.parametrize("bot_http", ["pending_child"], indirect=True)
def test_stream_close_reports_pending_memory_confirmation(bot_http, capsys):
    assert invoke(bot_http, "task", "events", "t_answer") == 0
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"event":')
    ]
    assert [event["event"] for event in events] == ["task.completed", "snapshot"]
    [pending] = events[-1]["data"]["pending_memory_confirmations"]
    assert invoke(
        bot_http, "task", "confirm", pending["task_id"],
        "--operation-id", pending["confirmation"]["operation_id"],
    ) == 0
    assert bot_http[1][-1]["path"] == "/api/v1/bot/tasks/t_memory/confirm"
    assert bot_http[1][-1]["body"] == {"operation_id": "op_capture", "decision": "recheck"}


def test_delivery_receipts_and_exact_memory_source(bot_http, capsys):
    assert invoke(bot_http, "delivery", "list", "--source-task-id", "t_weekly") == 0
    assert bot_http[1][-1]["path"] == "/api/v1/bot/deliveries"
    assert bot_http[1][-1]["query"]["source_task_id"] == ["t_weekly"]
    capsys.readouterr()
    uri = "ctx://acct/users/user/memories/qinghe"
    assert invoke(bot_http, "memory", "list", "--uri", uri) == 0
    assert bot_http[1][-1]["query"]["uri"] == [uri]


def test_authorized_agent_and_session_management(bot_http, capsys):
    assert invoke(bot_http, "agent", "list") == 0
    assert bot_http[1][-1]["path"] == "/api/v1/bot/agents"
    assert invoke(bot_http, "session", "create", "周报", "--agent-id", "analyst") == 0
    assert bot_http[1][-1]["body"]["agent_id"] == "analyst"
    assert invoke(bot_http, "session", "rename", "s_qinghe", "周五回顾") == 0
    assert bot_http[1][-1]["body"] == {"title": "周五回顾", "expected_revision": 3}
    for action, archived in (("archive", True), ("restore", False)):
        assert invoke(bot_http, "session", action, "s_qinghe", "--revision", "4") == 0
        assert bot_http[1][-1]["body"] == {"archived": archived, "expected_revision": 4}
    assert invoke(bot_http, "session", "list", "--archived", "true", "--query", "回顾") == 0
    assert bot_http[1][-1]["query"] == {"limit": ["20"], "archived": ["true"], "q": ["回顾"]}
    assert invoke(bot_http, "session", "show", "s_qinghe", "--history-order", "desc") == 0
    assert bot_http[1][-1]["query"]["history_order"] == ["desc"]


def test_agent_templates_and_background_task_stop(bot_http, capsys):
    assert (
        invoke(
            bot_http,
            "template",
            "list",
            "--agent-id",
            "analyst",
            "--limit",
            "2",
            "--cursor",
            "next-page",
        )
        == 0
    )
    assert bot_http[1][-1]["query"] == {
        "agent_id": ["analyst"],
        "limit": ["2"],
        "cursor": ["next-page"],
    }
    assert (
        invoke(
            bot_http,
            "task",
            "confirm",
            "t_memory",
            "--operation-id",
            "op_memory",
            "--decision",
            "cancel",
        )
        == 0
    )
    assert bot_http[1][-1]["body"] == {"operation_id": "op_memory", "decision": "cancel"}
    assert invoke(bot_http, "memory", "remember", "偏好简洁回复", "--agent-id", "analyst") == 0
    assert bot_http[1][-1]["query"] == {"agent_id": ["analyst"]}


def test_backup_export_verify_release_arguments(monkeypatch, capsys, tmp_path):
    calls = []

    def run_backup(action, **kwargs):
        calls.append({"action": action, **kwargs})
        return {"action": action, "file_count": 2}

    monkeypatch.setattr("memorybot.backup.run_backup", run_backup)
    manifest = str(tmp_path / "manifest.json")
    assert (
        main(
            [
                "bot",
                "backup",
                "export",
                "--account-id",
                "team",
                "--backup-id",
                "daily",
                "--manifest",
                manifest,
                "--retain-days",
                "180",
                "-c",
                "ogmem.yaml",
            ]
        )
        == 0
    )
    assert calls[-1] == {
        "action": "export",
        "config_path": "ogmem.yaml",
        "account_id": "team",
        "backup_id": "daily",
        "manifest_path": manifest,
        "retain_days": 180,
    }
    assert json.loads(capsys.readouterr().out)["action"] == "export"
    for action in ("verify", "release"):
        assert main(["bot", "backup", action, "--manifest", manifest, "-c", "ogmem.yaml"]) == 0
        assert calls[-1]["action"] == action
        assert calls[-1]["manifest_path"] == manifest
        assert json.loads(capsys.readouterr().out)["action"] == action
