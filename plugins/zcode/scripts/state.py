import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class Queue:
    def __init__(self, path, partition, limit=10000, retention_days=7):
        self.path = path
        self.partition = partition
        self.limit = limit
        self.retention_seconds = retention_days * 86400
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS seen (
                partition TEXT, id TEXT, PRIMARY KEY(partition, id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS turn (
                partition TEXT, session TEXT, id TEXT, prompt TEXT, stopped INTEGER DEFAULT 0,
                PRIMARY KEY(partition, session, id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS receipt (
                seq INTEGER PRIMARY KEY, summary TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS delivery (
                seq INTEGER PRIMARY KEY, partition TEXT, session TEXT, channel TEXT,
                payload TEXT, state TEXT DEFAULT 'pending', due REAL DEFAULT 0,
                lease TEXT, expires REAL DEFAULT 0, attempts INTEGER DEFAULT 0)""")
            db.execute("""CREATE INDEX IF NOT EXISTS delivery_lane
                       ON delivery(partition, session, channel, state, seq)""")
            db.execute("""CREATE TABLE IF NOT EXISTS capture_gap (
                partition TEXT, session TEXT, PRIMARY KEY (partition, session))""")
            for table in ("seen", "turn"):
                if "created" not in {column[1] for column in db.execute(f"PRAGMA table_info({table})")}:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN created REAL NOT NULL DEFAULT 0")

    def prune(self, db):
        cutoff = time.time() - self.retention_seconds
        for table in ("seen", "turn"):
            db.execute(f"DELETE FROM {table} WHERE partition=? AND created<?", (self.partition, cutoff))
        db.execute("""DELETE FROM receipt WHERE seq IN
                      (SELECT seq FROM delivery WHERE partition=? AND state='done' AND due<?)""",
                   (self.partition, cutoff))
        db.execute("DELETE FROM delivery WHERE partition=? AND state='done' AND due<?",
                   (self.partition, cutoff))

    @contextmanager
    def transaction(self):
        with closing(sqlite3.connect(self.path, timeout=2)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    def enqueue(self, db, session, channel, payload):
        db.execute("INSERT INTO delivery(partition,session,channel,payload) VALUES(?,?,?,?)",
                   (self.partition, session, channel, json.dumps(payload, ensure_ascii=False)))

    def claim(self):
        now = time.time()
        with self.transaction() as db:
            row = db.execute("""SELECT seq, session, channel, payload, attempts FROM delivery d
                WHERE partition=? AND state IN ('pending','verify') AND due<=? AND expires<=?
                AND NOT EXISTS (SELECT 1 FROM delivery p
                    WHERE p.partition=d.partition AND p.session=d.session AND p.channel=d.channel
                    AND p.seq<d.seq AND p.state!='done' AND d.channel!='observe')
                ORDER BY seq LIMIT 1""", (self.partition, now, now)).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            current = db.execute("SELECT state FROM delivery WHERE seq=?", (row[0],)).fetchone()[0]
            # Treat a worker interrupted during POST as having an unknown outcome.
            next_state = "verify" if row[2] == "write" else current
            db.execute("UPDATE delivery SET lease=?,expires=?,state=? WHERE seq=?",
                       (token, now + 90, next_state, row[0]))
            return (*row, current, token)

    def finish(self, seq, token, state, delay=0, summary=None):
        with self.transaction() as db:
            changed = db.execute("""UPDATE delivery SET state=?, due=?, expires=0, lease=NULL,
                          attempts=attempts+1, payload=CASE WHEN ?='done' THEN '{}' ELSE payload END
                          WHERE seq=? AND lease=?""",
                       (state, time.time() + delay, state, seq, token)).rowcount
            if changed and summary:
                db.execute("INSERT OR REPLACE INTO receipt VALUES(?,?)", (seq, json.dumps(summary)))

    def next_due(self):
        with self.transaction() as db:
            return db.execute("""SELECT MIN(MAX(due,expires)) FROM delivery d
                WHERE partition=? AND state IN ('pending','verify')
                AND NOT EXISTS (SELECT 1 FROM delivery p WHERE p.partition=d.partition
                    AND p.session=d.session AND p.channel=d.channel
                    AND p.seq<d.seq AND p.state!='done' AND d.channel!='observe')""", (self.partition,)).fetchone()[0]

    def resume(self):
        with self.transaction() as db:
            db.execute("UPDATE delivery SET state='verify',due=0 WHERE partition=? AND state='blocked'",
                       (self.partition,))


def retry_delay(attempts, retry_after):
    delay = min(60, 2 ** min(attempts, 6))
    if retry_after:
        try:
            return max(delay, int(retry_after))
        except ValueError:
            try:
                until = parsedate_to_datetime(retry_after)
                return max(delay, (until - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return delay


def write_status(payload):
    return "sessions/{}/write-requests/{}".format(
        quote(payload["sessionId"], safe=""), quote(payload["client_request_id"], safe=""))


def accepted(body):
    return (body.get("ok", True) is not False and body.get("accepted", True) is not False
            and (body.get("accepted") is True
                 or body.get("request_status") in {"ACCEPTED", "SUCCEEDED"}
                 or body.get("status") in {"accumulating", "completed", "no_new_messages"}))


def deliver(queue, client, row, log):
    seq, session, channel, raw, attempts, state, token = row
    payload = json.loads(raw)
    status_payload = payload.get("write", payload)
    status_path = write_status(status_payload) if channel != "trace" else None
    next_state = "verify" if channel == "write" else "pending"

    def complete_write(body):
        if not body.get("terminal") and (
                body.get("retry_action") == "poll" or body.get("status") == "processing"
                or body.get("semantic_status") == "PENDING"):
            with queue.transaction() as db:
                current = db.execute("SELECT lease FROM delivery WHERE seq=?", (seq,)).fetchone()
                if current and current[0] == token:
                    queue.enqueue(db, session, "observe", {"write": status_payload})
                    db.execute("UPDATE delivery SET state='done', payload='{}', lease=NULL, expires=0 WHERE seq=?", (seq,))
            return
        if body.get("semantic_status") in {"FAILED", "PARTIAL"} or body.get("writes_failed"):
            log("oGMemory reported partial or failed memory extraction")
        queue.finish(seq, token, "done", summary=write_summary(status_payload, body))

    try:
        if channel == "observe" or (channel == "write" and state == "verify"):
            code, body, retry_after = client.request(status_path)
            action = body.get("retry_action") or (body.get("error_detail") or {}).get("retry_action")
            unknown = code == 200 and body.get("request_status") == "UNKNOWN" and body.get("accepted") is False
            if channel == "observe" and code == 200 and action != "retry_same_request":
                if body.get("terminal") or action == "manual_review":
                    if body.get("request_status") != "SUCCEEDED":
                        log("oGMemory write observation requires attention")
                    queue.finish(seq, token, "done", summary=write_summary(status_payload, body))
                else:
                    queue.finish(seq, token, "pending", retry_delay(attempts, retry_after))
                return
            if channel == "write" and code == 200 and accepted(body) and action != "retry_same_request":
                complete_write(body)
                return
            if code != 404 and not unknown and action != "retry_same_request":
                if code in {400, 401, 403} or action == "manual_review":
                    log(f"{channel} status requires attention (HTTP {code})")
                    queue.finish(seq, token, "blocked")
                else:
                    queue.finish(seq, token, next_state, retry_delay(attempts, retry_after))
                return

        endpoint = "trace/events" if channel == "trace" else "after_turn"
        code, body, retry_after = client.request(endpoint, status_payload, timeout=3 if channel == "trace" else 30)
        if 200 <= code < 300:
            if channel == "trace" and body.get("ok") is True:
                turn = body.get("turn") or {}
                queue.finish(seq, token, "done", summary={
                    "event_type": payload["event"]["event_type"],
                    "event_index": body.get("event_index"),
                    "turn_uri": turn.get("uri"),
                    "agent_memory_event_id": turn.get("agent_memory_event_id"),
                })
                return
            if accepted(body):
                if channel == "observe":
                    queue.finish(seq, token, "pending", 2)
                elif channel == "write":
                    complete_write(body)
                else:
                    queue.finish(seq, token, "done")
                return
        action = body.get("retry_action") or (body.get("error_detail") or {}).get("retry_action")
        if action == "manual_review" and body.get("terminal"):
            log(f"{channel} delivery requires attention (HTTP {code})")
            queue.finish(seq, token, "blocked", summary=write_summary(status_payload, body))
            return
        if code in {408, 429} or code >= 500 or action == "retry_same_request" or body.get("retryable") is True:
            queue.finish(seq, token, next_state, retry_delay(attempts, retry_after))
        else:
            log(f"{channel} delivery requires attention (HTTP {code})")
            queue.finish(seq, token, "blocked")
    except (OSError, ValueError) as exc:
        log(f"{channel} delivery interrupted ({type(exc).__name__})")
        queue.finish(seq, token, next_state, retry_delay(attempts, None))


def write_summary(payload, body):
    fields = ("request_status", "source_status", "semantic_status", "projection_status", "status")
    return {"request_id": payload.get("client_request_id"), **{key: body.get(key) for key in fields}}
