import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from capture import capture, get
from ogm_plugin_request import Client
from recall import recall
from state import Queue, deliver, digest


def log(message):
    print(f"[ogmemory-zcode] {message}", file=sys.stderr)


def queue_for(client):
    root = Path(os.environ.get("ZCODE_PLUGIN_DATA") or Path.home() / ".ogmemory/zcode")
    return Queue(root / "state.sqlite", digest([client.url, client.identity]),
                 max(16, int(os.environ.get("OGMEM_ZCODE_QUEUE_LIMIT", "10000"))),
                 max(1, int(os.environ.get("OGMEM_ZCODE_RETENTION_DAYS", "7"))))


def run_worker(queue, client):
    deadline = time.monotonic() + 65
    while time.monotonic() < deadline:
        row = queue.claim()
        if row:
            deliver(queue, client, row, log)
            continue
        due = queue.next_due()
        if due is None or due - time.time() >= deadline - time.monotonic():
            return
        time.sleep(max(0.1, min(1, due - time.time())))


def start_worker(queue):
    opts = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with (queue.path.parent / "delivery.log").open("ab") as output:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--drain"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=output,
                         **opts)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--drain", action="store_true")
    parser.add_argument("--resume-delivery", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--session")
    args = parser.parse_args()
    client = Client()
    if args.query is not None:
        print(recall(client, {"hookEventName": "UserPromptSubmit", "prompt": args.query},
                     {**client.identity, "sessionId": args.session or "zcode-manual"}))
        return
    if args.drain or args.resume_delivery:
        queue = queue_for(client)
        if args.resume_delivery:
            queue.resume()
        run_worker(queue, client)
        return
    event = json.load(sys.stdin.buffer)
    session = get(event, "sessionId", "session_id")
    name = get(event, "hookEventName", "hook_event_name")
    if not isinstance(session, str) or not session.strip():
        log("Hook input lacks a session ID")
        return
    identity = {**client.identity, "sessionId": session}
    try:
        if client.capture or client.trace:
            queue = queue_for(client)
            if capture(queue, event, identity, client.trace, client.capture,
                       client.trace_content, client.force_extract) is False:
                log("Capture skipped because the delivery queue is full")
            due = queue.next_due()
            if due is not None and due - time.time() < 65:
                start_worker(queue)
    except Exception as exc:
        log(f"Capture unavailable ({type(exc).__name__})")
    if client.recall and name in {"SessionStart", "UserPromptSubmit"}:
        context = recall(client, event, identity)
        if context:
            print(json.dumps({"hookSpecificOutput": {"hookEventName": name,
                                                     "additionalContext": context}}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        if os.name != "nt":
            os.umask(0o077)
        main()
    except Exception as exc:
        log(f"Hook unavailable ({type(exc).__name__})")
