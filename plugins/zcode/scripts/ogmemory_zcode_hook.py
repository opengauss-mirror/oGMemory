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


WORKER_RUN_SECONDS = 65


def log(message):
    print(f"[ogmemory-zcode] {message}", file=sys.stderr)


def queue_for(client):
    root = Path(os.environ.get("ZCODE_PLUGIN_DATA") or Path.home() / ".ogmemory/zcode")
    return Queue(root / "state.sqlite", digest([client.url, client.identity]),
                 max(16, int(os.environ.get("OGMEM_ZCODE_QUEUE_LIMIT", "10000"))),
                 max(1, int(os.environ.get("OGMEM_ZCODE_RETENTION_DAYS", "7"))))


def run_worker(queue, client, token):
    if not queue.renew_worker(token):
        return
    deadline = time.monotonic() + WORKER_RUN_SECONDS
    try:
        while time.monotonic() < deadline:
            due = queue.next_due()
            if due is None or due - time.time() >= deadline - time.monotonic():
                break
            wait = due - time.time()
            if wait > 0:
                time.sleep(min(1, wait))
                continue
            if not queue.renew_worker(token):
                return
            row = queue.claim()
            if row:
                deliver(queue, client, row, log)
    finally:
        queue.release_worker(token)
    # A Hook may have enqueued work while this worker was about to exit.
    # Recheck after releasing the lease so either caller can reserve the next worker.
    start_worker(queue)


def start_worker(queue):
    token = queue.reserve_worker(WORKER_RUN_SECONDS)
    if token is None:
        return
    opts = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    try:
        with (queue.path.parent / "delivery.log").open("ab") as output:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--drain", token],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=output,
                             **opts)
    except OSError:
        queue.release_worker(token)
        raise


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--drain", nargs="?", const="")
    parser.add_argument("--resume-delivery", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--session")
    args = parser.parse_args()
    client = Client()
    if args.query is not None:
        print(recall(client, {"hookEventName": "UserPromptSubmit", "prompt": args.query},
                     {**client.identity, "sessionId": args.session or "zcode-manual"}))
        return
    if args.drain is not None or args.resume_delivery:
        queue = queue_for(client)
        if args.resume_delivery:
            queue.resume()
        token = args.drain or queue.reserve_worker(WORKER_RUN_SECONDS)
        if token:
            run_worker(queue, client, token)
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
