"""Authenticated memoryBot commands for the shared HTTP API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import uuid
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

if TYPE_CHECKING:
    import requests

_TERMINAL = {"succeeded", "cancelled", "failed"}
_DEFAULT_LOCAL_TIME = "09:00:00"


def _events_finished(task: dict[str, Any], after: int) -> bool:
    return task["state"] == "waiting_confirmation" or (
        task["state"] in _TERMINAL
        and task.get("memory_status") not in {"pending", "stored"}
        and after >= task["last_event_seq"]
    )


def _paging(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--cursor")


def _scope_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--memory-scope", choices=["user", "authorized"], default="user")
    parser.add_argument("--resource-uri", action="append", default=[])


def _input_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--file-id", action="append", default=[])
    parser.add_argument("--template-id")
    parser.add_argument("--output-format", choices=["markdown", "csv"], default="markdown")


def _rule_options(parser: argparse.ArgumentParser, *, required: bool) -> None:
    rule = parser.add_mutually_exclusive_group(required=required)
    rule.add_argument("--at", help="One-time ISO 8601 date including timezone")
    rule.add_argument("--interval", type=int, help="Interval in seconds")
    rule.add_argument("--weekdays", help="ISO weekdays separated by commas, e.g. 1,5")
    parser.add_argument("--local-time", default=_DEFAULT_LOCAL_TIME if required else None)
    parser.add_argument("--timezone", default="Asia/Shanghai" if required else None)


def register_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("bot", help="Chat, files, plans and long-term memory")
    parser.add_argument("--base-url", help="HTTP origin (or OGMEM_HTTP_BASE_URL)")
    parser.add_argument("--api-key", help="User API key (or OGMEM_BOT_API_KEY)")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--request-id",
        help="Stable idempotency key. Session and schedule updates require an explicit revision.",
    )
    commands = parser.add_subparsers(dest="bot_command", required=True)

    agent = commands.add_parser("agent", help="List authorized native Agents")
    agent.add_subparsers(dest="action", required=True).add_parser("list")
    backup = commands.add_parser("backup", help="Export and verify shared-file backup manifests")
    backup_sub = backup.add_subparsers(dest="action", required=True)
    for action in ("export", "verify", "release"):
        item = backup_sub.add_parser(action)
        item.add_argument("-c", "--config", help="Path to ogmem.yaml")
        item.add_argument("--manifest", required=True, help="Local backup manifest path")
        if action == "export":
            item.add_argument("--account-id", required=True)
            item.add_argument("--backup-id", required=True)
            item.add_argument("--retain-days", type=int, default=90)

    session = commands.add_parser("session", help="Create and read conversations")
    session_sub = session.add_subparsers(dest="action", required=True)
    create = session_sub.add_parser("create")
    create.add_argument("title")
    create.add_argument("--agent-id", default="memorybot")
    _scope_options(create)
    listing = session_sub.add_parser("list")
    listing.add_argument("--query")
    listing.add_argument("--archived", choices=["true", "false"], default="false")
    _paging(listing)
    show = session_sub.add_parser("show")
    show.add_argument("session_id")
    show.add_argument("--history-order", choices=["asc", "desc"])
    _paging(show)
    for action in ("rename", "archive", "restore"):
        item = session_sub.add_parser(action)
        item.add_argument("session_id")
        item.add_argument("--revision", type=int, help="Expected revision (default: fetch current)")
        if action == "rename":
            item.add_argument("title")

    chat = commands.add_parser("chat", help="Submit work to a conversation")
    chat.add_argument("session_id")
    chat.add_argument("text", nargs="?", default="")
    chat.add_argument("--follow", action="store_true", help="Follow progress to completion")
    _input_options(chat)

    task = commands.add_parser("task", help="Inspect, follow, cancel or confirm a task")
    task_sub = task.add_subparsers(dest="action", required=True)
    for action in ("show", "cancel", "events", "confirm"):
        item = task_sub.add_parser(action)
        item.add_argument("task_id")
        if action == "events":
            item.add_argument("--after", type=int, default=0)
        if action == "confirm":
            item.add_argument("--operation-id", required=True)
            item.add_argument("--decision", choices=["recheck", "cancel"], default="recheck")

    file = commands.add_parser("file", help="Upload sources and download artifacts")
    file_sub = file.add_subparsers(dest="action", required=True)
    listing = file_sub.add_parser("list")
    _paging(listing)
    listing.add_argument("--purpose", choices=["source", "artifact"])
    listing.add_argument("--state", choices=["processing", "ready", "failed"])
    listing.add_argument("--task-id")
    for action in ("show", "download"):
        item = file_sub.add_parser(action)
        item.add_argument("file_id")
        if action == "download":
            item.add_argument("--output", "-o", required=True, help="Destination file")
    upload = file_sub.add_parser("upload")
    upload.add_argument("path")
    upload.add_argument("--agent-id")
    _scope_options(upload)
    url = file_sub.add_parser("add-url")
    url.add_argument("url")
    url.add_argument("--agent-id")
    _scope_options(url)

    template = commands.add_parser("template", help="List available work templates")
    templates = template.add_subparsers(dest="action", required=True)
    listing = templates.add_parser("list")
    listing.add_argument("--query")
    listing.add_argument("--output-format", choices=["markdown", "csv"])
    listing.add_argument("--agent-id", default="memorybot")
    listing.add_argument("--memory-scope", choices=["user", "authorized"])
    listing.add_argument("--resource-uri", action="append", default=[])
    _paging(listing)

    schedule = commands.add_parser("schedule", help="Create, update and follow plans")
    schedule_sub = schedule.add_subparsers(dest="action", required=True)
    listing = schedule_sub.add_parser("list")
    listing.add_argument("--state", choices=["active", "paused", "cancelled", "completed"])
    listing.add_argument("--session-id")
    _paging(listing)
    create = schedule_sub.add_parser("create")
    create.add_argument("session_id")
    create.add_argument("name")
    create.add_argument("text")
    _rule_options(create, required=True)
    _input_options(create)
    for action in ("show", "runs", "pause", "resume", "cancel", "update"):
        item = schedule_sub.add_parser(action)
        item.add_argument("schedule_id")
        if action == "runs":
            _paging(item)
        elif action in {"pause", "resume", "cancel", "update"}:
            item.add_argument(
                "--revision", type=int, help="Expected revision (default: fetch current)"
            )
        if action == "update":
            item.add_argument("--name")
            item.add_argument("--text")
            item.add_argument("--body", help="UTF-8 JSON file containing additional updated fields")
            _rule_options(item, required=False)

    memory = commands.add_parser("memory", help="Read, correct, forget or remember facts")
    memory_sub = memory.add_subparsers(dest="action", required=True)
    listing = memory_sub.add_parser("list")
    listing.add_argument("--query")
    listing.add_argument("--uri", help="Read an exact authorized source URI")
    listing.add_argument("--memory-scope", choices=["user", "authorized"], default="user")
    _paging(listing)
    for action in ("correct", "forget", "remember"):
        item = memory_sub.add_parser(action)
        if action != "remember":
            item.add_argument("uri")
        else:
            item.add_argument(
                "--uri", help="Original forgotten memory URI to explicitly remember again"
            )
        if action != "forget":
            item.add_argument("text")
        item.add_argument("--source-file", action="append", default=[])
        item.add_argument("--source-task", action="append", default=[])
        item.add_argument("--source-uri", action="append", default=[])
        item.add_argument("--agent-id")

    delivery = commands.add_parser("delivery", help="Read notification delivery receipts")
    delivery_sub = delivery.add_subparsers(dest="action", required=True)
    listing = delivery_sub.add_parser("list")
    listing.add_argument("--source-task-id")
    listing.add_argument("--task-id")
    listing.add_argument("--state")
    _paging(listing)
    commands.add_parser("config", help="Read public Bot capabilities")
    worker = commands.add_parser("worker", help="Run the background task worker")
    worker.add_argument("-c", "--config", help="Path to ogmem.yaml")


class BotAPIError(RuntimeError):
    def __init__(self, status: int, payload: dict[str, Any]):
        self.status = status
        self.payload = payload
        error = payload.get("error") or {}
        super().__init__(
            f"{error.get('code', status)}: {error.get('message', 'HTTP request failed')}"
        )


class BotClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0):
        import requests

        base_url = base_url.rstrip("/")
        self.base_url = base_url if base_url.endswith("/api/v1/bot") else base_url + "/api/v1/bot"
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    @classmethod
    def from_args(cls, args: Any) -> BotClient:
        base_url = (
            args.base_url
            or os.environ.get("OGMEM_HTTP_BASE_URL")
            or os.environ.get("OGMEM_SERVER_URL")
            or "http://127.0.0.1:8090"
        )
        api_key = (
            args.api_key
            or os.environ.get("OGMEM_BOT_API_KEY")
            or os.environ.get("OGMEM_MCP_API_KEY")
            or os.environ.get("OG_AUTH_API_KEY")
            or ""
        )
        return cls(base_url, api_key, timeout=args.timeout)

    def close(self) -> None:
        self.session.close()

    @staticmethod
    def _check(response: requests.Response) -> None:
        if response.ok:
            return
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": {"message": response.reason}}
        raise BotAPIError(response.status_code, payload)

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        params: dict[str, Any] | None = None,
        request_id: str | None = None,
        files: Any = None,
        form: Any = None,
    ) -> dict[str, Any]:
        headers = {"Idempotency-Key": request_id or uuid.uuid4().hex} if method != "GET" else {}
        response = self.session.request(
            method,
            self.base_url + path,
            json=body,
            params=params,
            files=files,
            data=form,
            headers=headers,
            timeout=self.timeout,
        )
        with response:
            self._check(response)
            return cast(dict[str, Any], response.json())

    def download(self, file_id: str, destination: Path) -> dict[str, Any]:
        destination = destination.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            with self.session.get(
                self.base_url + f"/files/{_id(file_id)}/content", stream=True, timeout=self.timeout
            ) as response:
                self._check(response)
                size = 0
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        stream.write(chunk)
                        size += len(chunk)
                temporary.replace(destination)
                return {"file_id": file_id, "path": str(destination.resolve()), "size_bytes": size}
        finally:
            temporary.unlink(missing_ok=True)

    def events(self, task_id: str, after: int = 0) -> Iterator[dict[str, Any]]:
        import requests

        path = f"/tasks/{_id(task_id)}"
        retries = 0
        completed_event_seen = False
        while True:
            headers = {"Accept": "text/event-stream"}
            if after:
                headers["Last-Event-ID"] = f"{task_id}:{after}"
            try:
                with self.session.get(
                    self.base_url + path + "/events",
                    params={"after": after},
                    headers=headers,
                    timeout=self.timeout,
                    stream=True,
                ) as response:
                    if response.status_code == 410:
                        snapshot = self.request("GET", path)["data"]
                        after = snapshot["last_event_seq"]
                        yield {"event": "snapshot", "data": snapshot}
                        if _events_finished(snapshot, after):
                            return
                        continue
                    self._check(response)
                    response.encoding = "utf-8"
                    for event in parse_events(response.iter_lines(decode_unicode=True)):
                        seq = event["data"].get("event_seq", after)
                        if seq <= after:
                            continue
                        after = seq
                        retries = 0
                        completed_event_seen |= event["event"] == "task.completed"
                        yield event
                snapshot = self.request("GET", path)["data"]
                if _events_finished(snapshot, after):
                    if not completed_event_seen or snapshot["state"] == "waiting_confirmation":
                        yield {"event": "snapshot", "data": snapshot}
                    return
                time.sleep(1)
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ):
                retries += 1
                if retries >= 5:
                    raise
                time.sleep(min(retries * 2, 10))


def parse_events(lines: Any) -> Iterator[dict[str, Any]]:
    event: dict[str, Any] = {"event": "message"}
    data: list[str] = []
    for line in lines:
        if not line:
            if data:
                event["data"] = json.loads("\n".join(data))
                yield event
            event, data = {"event": "message"}, []
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "data":
                data.append(value)
            elif field in {"event", "id"}:
                event[field] = value


def _id(value: str) -> str:
    return quote(value, safe="")


def _scope(args: Any) -> dict[str, Any]:
    return {
        "memory_scope": args.memory_scope,
        "resource_ids": args.resource_uri,
    }


def _input(args: Any) -> dict[str, Any]:
    return {
        "text": args.text,
        "file_ids": args.file_id,
        "template_id": args.template_id,
        "output_format": args.output_format,
    }


def _rule(args: Any, current=None, supplied=None) -> dict[str, Any] | None:
    if args.at:
        return {"type": "once", "at": args.at}
    if args.interval is not None:
        return {"type": "interval", "interval_seconds": args.interval}
    if args.weekdays is None and (current is None or args.local_time is None):
        return None
    weekly = supplied if isinstance(supplied, dict) and supplied.get("type") == "weekly" else {}
    if (args.weekdays is None and "weekdays" not in weekly) or (
        args.local_time is None and "local_time" not in weekly
    ):
        if args.request_id:
            raise ValueError("--request-id requires explicit weekdays and local time for weekly updates")
        previous = current()
        if previous.get("type") == "weekly":
            weekly = {**previous, **weekly}
    weekdays = (
        [int(day) for day in args.weekdays.split(",")]
        if args.weekdays is not None
        else weekly.get("weekdays")
    )
    if weekdays is None:
        raise ValueError("--local-time requires --weekdays when creating a weekly rule")
    local_time = (
        args.local_time if args.local_time is not None else weekly.get("local_time", _DEFAULT_LOCAL_TIME)
    )
    return {"type": "weekly", "weekdays": weekdays, "local_time": local_time}


def _revision(args: Any, current) -> int:
    if args.revision is not None:
        return args.revision
    if args.request_id:
        raise ValueError("--request-id requires --revision for updates")
    return current()["revision"]


def _params(args: Any, *names: str) -> dict[str, Any]:
    values = {name: getattr(args, name, None) for name in ("limit", "cursor", *names)}
    return {
        "q" if key == "query" else key: value for key, value in values.items() if value is not None
    }


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def execute(client: BotClient, args: Any) -> Any:
    group = args.bot_command
    action: str = getattr(args, "action", "")

    def send(method: str, path: str, body: Any = None, params: Any = None) -> Any:
        return client.request(method, path, body=body, params=params, request_id=args.request_id)

    if group == "config":
        return send("GET", "/config")
    if group == "agent":
        return send("GET", "/agents")
    if group == "session":
        if action == "create":
            return send(
                "POST",
                "/sessions",
                {"title": args.title, "scope": _scope(args), "agent_id": args.agent_id},
            )
        if action == "list":
            return send("GET", "/sessions", params=_params(args, "query", "archived"))
        path = f"/sessions/{_id(args.session_id)}"
        if action in {"rename", "archive", "restore"}:
            revision = _revision(args, lambda: send("GET", path)["data"]["session"])
            update = {"expected_revision": revision}
            update.update(
                {"title": args.title} if action == "rename" else {"archived": action == "archive"}
            )
            return send("PATCH", path, update)
        return send("GET", path, params=_params(args, "history_order"))
    if group == "chat":
        result = send("POST", f"/sessions/{_id(args.session_id)}/tasks", _input(args))
        if args.follow:
            _print(result)
            for event in client.events(result["data"]["task_id"]):
                print(json.dumps(event, ensure_ascii=False), flush=True)
            return send("GET", f"/tasks/{_id(result['data']['task_id'])}")
        return result
    if group == "task":
        path = f"/tasks/{_id(args.task_id)}"
        if action == "events":
            for event in client.events(args.task_id, args.after):
                print(json.dumps(event, ensure_ascii=False), flush=True)
            return send("GET", path)
        if action == "show":
            return send("GET", path)
        task_body = (
            {"operation_id": args.operation_id, "decision": args.decision}
            if action == "confirm"
            else {}
        )
        return send("POST", path + "/" + action, task_body)
    if group == "file":
        if action == "list":
            return send("GET", "/files", params=_params(args, "purpose", "state", "task_id"))
        if action == "upload":
            source_path = Path(args.path).expanduser()
            mime = {".md": "text/markdown", ".csv": "text/csv", ".txt": "text/plain"}.get(
                source_path.suffix.lower(),
                mimetypes.guess_type(source_path.name)[0] or "application/octet-stream",
            )
            with source_path.open("rb") as stream:
                return client.request(
                    "POST",
                    "/files",
                    request_id=args.request_id,
                    files={"file": (source_path.name, stream, mime)},
                    form={"scope": json.dumps(_scope(args), ensure_ascii=False)},
                    params=_params(args, "agent_id"),
                )
        if action == "add-url":
            return send(
                "POST",
                "/files",
                {"source_kind": "url", "source_url": args.url, "scope": _scope(args)},
                params=_params(args, "agent_id"),
            )
        if action == "download":
            return client.download(args.file_id, Path(args.output))
        return send("GET", f"/files/{_id(args.file_id)}")
    if group == "template":
        return send(
            "GET",
            "/templates",
            params=_params(
                args, "query", "output_format", "agent_id", "memory_scope", "resource_uri"
            ),
        )
    if group == "schedule":
        if action == "create":
            return send(
                "POST",
                "/schedules",
                {
                    "session_id": args.session_id,
                    "name": args.name,
                    "task_input": _input(args),
                    "rule": _rule(args),
                    "timezone": args.timezone,
                },
            )
        if action == "list":
            return send("GET", "/schedules", params=_params(args, "state", "session_id"))
        path = f"/schedules/{_id(args.schedule_id)}"
        if action in {"show", "runs"}:
            return send("GET", path + ("/runs" if action == "runs" else ""), params=_params(args))

        @cache
        def current():
            return send("GET", path)["data"]

        schedule_body: dict[str, Any] = {"action": action}
        if action == "update":
            if args.body:
                schedule_body.update(json.loads(Path(args.body).read_text(encoding="utf-8")))
                schedule_body["action"] = "update"
            if args.name is not None:
                schedule_body["name"] = args.name
            if args.text is not None:
                task_input = schedule_body.get("task_input")
                if task_input is None:
                    if args.request_id:
                        raise ValueError("--request-id with --text requires task_input in --body")
                    task_input = current()["task_input"]
                schedule_body["task_input"] = {**task_input, "text": args.text}
            rule = _rule(args, lambda: current()["rule"], schedule_body.get("rule"))
            if rule:
                schedule_body["rule"] = rule
            if args.timezone is not None:
                schedule_body["timezone"] = args.timezone
        if schedule_body.get("expected_revision") is None:
            schedule_body["expected_revision"] = _revision(args, current)
        return send("PATCH", path, schedule_body)
    if group == "memory":
        if action == "list":
            return send("GET", "/memories", params=_params(args, "query", "uri", "memory_scope"))
        sources = []
        for kind in ("file", "task", "uri"):
            for value in getattr(args, "source_" + kind):
                source = {"file_id": None, "task_id": None, "uri": None, "quote": ""}
                source[kind if kind == "uri" else kind + "_id"] = value
                sources.append(source)
        memory_body = {"action": action, "source_refs": sources, "fact_scope": {"mode": "entry"}}
        if args.uri:
            memory_body["memory_uri"] = args.uri
        if action != "forget":
            memory_body["new_text"] = args.text
        return send("POST", "/memory-actions", memory_body, params=_params(args, "agent_id"))
    if group == "delivery":
        return send(
            "GET",
            "/deliveries",
            params=_params(args, "source_task_id", "task_id", "state"),
        )
    raise ValueError(f"Unknown bot command: {group}")


def run(args: Any) -> int:
    if args.bot_command == "backup":
        from memorybot.backup import run_backup

        _print(
            run_backup(
                args.action,
                config_path=args.config,
                account_id=getattr(args, "account_id", None),
                backup_id=getattr(args, "backup_id", None),
                manifest_path=args.manifest,
                retain_days=getattr(args, "retain_days", 90),
            )
        )
        return 0
    if args.bot_command == "worker":
        from memorybot.worker import run_worker

        result = run_worker(config_path=args.config)
        return result if isinstance(result, int) else 0
    import requests

    client = BotClient.from_args(args)
    try:
        _print(execute(client, args))
        return 0
    except BotAPIError as exc:
        print(json.dumps(exc.payload, ensure_ascii=False), file=sys.stderr)
        return 1
    except (requests.RequestException, OSError, ValueError) as exc:
        print(f"ogmem bot: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        client.close()
