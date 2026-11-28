import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote


WORKER_LEASE_SECONDS = 120
SCHEDULABLE = """
    FROM delivery d
    WHERE d.partition=? AND d.state IN ('pending','verify')
    AND NOT EXISTS (SELECT 1 FROM delivery p WHERE p.partition=d.partition
        AND p.session=d.session AND p.channel=d.channel
        AND p.seq<d.seq AND p.state!='done' AND d.channel!='observe')
"""


@dataclass(frozen=True)
class ClaimedDelivery:
    seq: int
    session: str
    channel: str
    payload: str
    attempts: int
    previous_state: str
    lease_token: str


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
            db.execute("""CREATE TABLE IF NOT EXISTS delivery_worker (
                partition TEXT PRIMARY KEY, token TEXT NOT NULL, expires REAL NOT NULL)""")
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
        with self.transaction() as db:
            now = time.time()
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT d.seq, d.session, d.channel, d.payload, d.attempts, d.state AS previous_state "
                + SCHEDULABLE + " AND MAX(d.due,d.expires)<=? ORDER BY d.seq LIMIT 1",
                (self.partition, now),
            ).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            job = ClaimedDelivery(**dict(row), lease_token=token)
            # Treat a worker interrupted during POST as having an unknown outcome.
            next_state = "verify" if job.channel == "write" else job.previous_state
            db.execute("UPDATE delivery SET lease=?,expires=?,state=? WHERE seq=?",
                       (token, now + 90, next_state, job.seq))
            return job

    def _finish(self, db, job, state, delay=0):
        return db.execute("""UPDATE delivery SET state=?, due=?, expires=0, lease=NULL,
                      attempts=attempts+1, payload=CASE WHEN ?='done' THEN '{}' ELSE payload END
                      WHERE partition=? AND seq=? AND lease=?""",
                   (state, time.time() + delay, state, self.partition, job.seq, job.lease_token)).rowcount

    def finish(self, job, state, delay=0, summary=None):
        with self.transaction() as db:
            if self._finish(db, job, state, delay) and summary:
                db.execute("INSERT OR REPLACE INTO receipt VALUES(?,?)", (job.seq, json.dumps(summary)))

    def observe_write(self, job, payload):
        with self.transaction() as db:
            if self._finish(db, job, "done"):
                self.enqueue(db, job.session, "observe", {"write": payload})

    def _next_due(self, db):
        return db.execute("SELECT MIN(MAX(d.due,d.expires)) " + SCHEDULABLE,
                          (self.partition,)).fetchone()[0]

    def next_due(self):
        with closing(sqlite3.connect(self.path, timeout=2)) as db:
            return self._next_due(db)

    def reserve_worker(self, wait_seconds):
        with self.transaction() as db:
            now = time.time()
            if db.execute("SELECT 1 FROM delivery_worker WHERE partition=? AND expires>?",
                          (self.partition, now)).fetchone():
                return None
            due = self._next_due(db)
            if due is None or due >= now + wait_seconds:
                return None
            token = uuid.uuid4().hex
            db.execute("INSERT OR REPLACE INTO delivery_worker VALUES(?,?,?)",
                       (self.partition, token, now + WORKER_LEASE_SECONDS))
            return token

    def renew_worker(self, token):
        with self.transaction() as db:
            now = time.time()
            return bool(db.execute("""UPDATE delivery_worker SET expires=?
                WHERE partition=? AND token=? AND expires>?""",
                (now + WORKER_LEASE_SECONDS, self.partition, token, now)).rowcount)

    def release_worker(self, token):
        with self.transaction() as db:
            db.execute("DELETE FROM delivery_worker WHERE partition=? AND token=?", (self.partition, token))

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


def deliver(queue, client, job, log):
    channel = job.channel
    payload = json.loads(job.payload)
    status_payload = payload.get("write", payload)
    status_path = write_status(status_payload) if channel != "trace" else None
    next_state = "verify" if channel == "write" else "pending"

    def complete_write(body):
        if not body.get("terminal") and (
                body.get("retry_action") == "poll" or body.get("status") == "processing"
                or body.get("semantic_status") == "PENDING"):
            queue.observe_write(job, status_payload)
            return
        if body.get("semantic_status") in {"FAILED", "PARTIAL"} or body.get("writes_failed"):
            log("oGMemory reported partial or failed memory extraction")
        queue.finish(job, "done", summary=write_summary(status_payload, body))

    try:
        if channel == "observe" or (channel == "write" and job.previous_state == "verify"):
            code, body, retry_after = client.request(status_path)
            action = body.get("retry_action") or (body.get("error_detail") or {}).get("retry_action")
            unknown = code == 200 and body.get("request_status") == "UNKNOWN" and body.get("accepted") is False
            if channel == "observe" and code == 200 and action != "retry_same_request":
                if body.get("terminal") or action == "manual_review":
                    if body.get("request_status") != "SUCCEEDED":
                        log("oGMemory write observation requires attention")
                    queue.finish(job, "done", summary=write_summary(status_payload, body))
                else:
                    queue.finish(job, "pending", retry_delay(job.attempts, retry_after))
                return
            if channel == "write" and code == 200 and accepted(body) and action != "retry_same_request":
                complete_write(body)
                return
            if code != 404 and not unknown and action != "retry_same_request":
                if code in {400, 401, 403} or action == "manual_review":
                    log(f"{channel} status requires attention (HTTP {code})")
                    queue.finish(job, "blocked")
                else:
                    queue.finish(job, next_state, retry_delay(job.attempts, retry_after))
                return

        endpoint = "trace/events" if channel == "trace" else "after_turn"
        code, body, retry_after = client.request(endpoint, status_payload, timeout=3 if channel == "trace" else 30)
        if 200 <= code < 300:
            if channel == "trace" and body.get("ok") is True:
                turn = body.get("turn") or {}
                queue.finish(job, "done", summary={
                    "event_type": payload["event"]["event_type"],
                    "event_index": body.get("event_index"),
                    "turn_uri": turn.get("uri"),
                    "agent_memory_event_id": turn.get("agent_memory_event_id"),
                })
                return
            if accepted(body):
                if channel == "observe":
                    queue.finish(job, "pending", 2)
                elif channel == "write":
                    complete_write(body)
                else:
                    queue.finish(job, "done")
                return
        action = body.get("retry_action") or (body.get("error_detail") or {}).get("retry_action")
        if action == "manual_review" and body.get("terminal"):
            log(f"{channel} delivery requires attention (HTTP {code})")
            queue.finish(job, "blocked", summary=write_summary(status_payload, body))
            return
        if code in {408, 429} or code >= 500 or action == "retry_same_request" or body.get("retryable") is True:
            queue.finish(job, next_state, retry_delay(job.attempts, retry_after))
        else:
            log(f"{channel} delivery requires attention (HTTP {code})")
            queue.finish(job, "blocked")
    except (OSError, ValueError) as exc:
        log(f"{channel} delivery interrupted ({type(exc).__name__})")
        queue.finish(job, next_state, retry_delay(job.attempts, None))


def write_summary(payload, body):
    fields = ("request_status", "source_status", "semantic_status", "projection_status", "status")
    return {"request_id": payload.get("client_request_id"), **{key: body.get(key) for key in fields}}
