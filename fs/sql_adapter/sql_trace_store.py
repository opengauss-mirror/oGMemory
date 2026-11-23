# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# oGMemory is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""PostgreSQL/openGauss persistence for agent lifecycle traces."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime

from fs.sql_adapter.pool import PoolAdapterMixin
from skills.artifact import validate_skill_publication

try:
    import psycopg2
    from psycopg2.extras import Json

    _HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None
    Json = None
    _HAS_PSYCOPG2 = False

logger = logging.getLogger(__name__)

_EVENT_COLUMNS = (
    "event_id, uri, account_id, owner_space, agent_id, session_id, "
    "hook_type, event_index, event_time, content, abstract, overview"
)


def _json_value(value):
    return Json(value)


def _insert_trace_session_events(
    cur,
    *,
    account_id: str,
    evidence_uri: str,
    owner_space: str,
    agent_id: str,
    session_id: str,
    after_event_index: int,
    stop_event_index: int,
    write_batch_id: str = "",
) -> int:
    """Project one complete turn into its indexed event stream."""
    cur.execute(
        """
        INSERT INTO trace_session_events
            (account_id, evidence_uri, ordinal, event)
        SELECT %s, %s,
               ROW_NUMBER() OVER (ORDER BY event_index),
               content
        FROM trace_events
        WHERE account_id = %s AND session_id = %s
          AND owner_space = %s AND agent_id = %s
          AND event_index > %s AND event_index <= %s
          AND COALESCE(content->>'write_batch_id', '') = %s
        ORDER BY event_index
        """,
        (
            account_id,
            evidence_uri,
            account_id,
            session_id,
            owner_space,
            agent_id,
            after_event_index,
            stop_event_index,
            write_batch_id,
        ),
    )
    return int(cur.rowcount)


def _event_uri(account_id: str, session_id: str, event_id: str) -> str:
    return f"ctx://{account_id}/traces/{session_id}/events/{event_id}"


def _turn_id(
    account_id: str,
    owner_space: str,
    agent_id: str,
    session_id: str,
    stop_event_index: int,
) -> str:
    identity = "\0".join((account_id, owner_space, agent_id, session_id, str(stop_event_index)))
    return f"turn_{hashlib.sha256(identity.encode()).hexdigest()[:32]}"


def _event_preview(content: dict, limit: int) -> str:
    event_type = str(content.get("event_type") or "")
    if event_type == "user_message":
        preview = str(content.get("prompt") or "")
    elif event_type in {"tool_start", "tool_success", "tool_failure"}:
        tool_name = str(content.get("tool_name") or "unknown_tool")
        preview = f"[{content.get('hook_type') or event_type}] {tool_name}"
    elif (
        event_type in {"assistant_message", "turn_end"}
        and content.get("assistant_message") is not None
    ):
        preview = str(content.get("assistant_message") or "")
    else:
        preview = f"[{content.get('hook_type') or event_type or 'hook_event'}]"
    return preview.replace("\n", " ")[:limit]


class SQLTraceEventStore(PoolAdapterMixin):
    """Store individual hook events and finalized per-turn aggregates."""

    def __init__(
        self,
        connection_string: str | None = None,
        pool_size: int = 5,
        pool=None,
        max_turn_events: int = 128,
        dialect: str = "postgresql",
    ) -> None:
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SQLTraceEventStore. "
                "Install with: pip install psycopg2-binary"
            )
        if max_turn_events < 2 or max_turn_events > 1024:
            raise ValueError("max_turn_events must be between 2 and 1024")
        self._max_turn_events = max_turn_events
        self._init_pool(pool, connection_string, pool_size, dialect)
        self._ensure_table()

    def _get_connection(self):
        """Borrow a standalone connection when no shared pool was supplied."""
        if self._pool:
            conn = self._pool.pop()
            if conn.closed == 0:
                return conn
        return psycopg2.connect(self._connection_string)

    def _return_connection(self, conn) -> None:
        """Return a standalone connection to this store's bounded idle pool."""
        if len(self._pool) < self._pool_size and conn.closed == 0:
            self._pool.append(conn)
        else:
            conn.close()

    def _ensure_table(self) -> None:
        from fs.sql_adapter.schema import ensure_schema

        conn = self._get_connection()
        try:
            ensure_schema(conn, self.sql_dialect)
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(f"Failed to ensure trace schema: {exc}") from exc
        finally:
            self._return_connection(conn)

    def insert(
        self,
        *,
        event_id: str,
        account_id: str,
        owner_space: str,
        agent_id: str,
        session_id: str,
        hook_type: str,
        content: dict,
        event_time: datetime | None = None,
    ) -> dict:
        """Persist one hook event idempotently by ``event_id``."""
        if not event_id:
            raise ValueError("event_id is required")
        if len(event_id) > 64:
            raise ValueError("event_id exceeds 64 characters")
        if not account_id:
            raise ValueError("account_id is required")
        if not session_id or session_id == "unknown":
            raise ValueError("valid session_id is required")
        if len(session_id) > 128:
            raise ValueError("session_id exceeds 128 characters")
        if len(hook_type) > 64:
            raise ValueError("hook_type exceeds 64 characters")
        if not isinstance(content, dict):
            raise ValueError("trace content must be an object")

        event_time = event_time or datetime.now(UTC)
        uri = _event_uri(account_id, session_id, event_id)
        abstract = _event_preview(content, 200)
        overview = _event_preview(content, 500)

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                inserted, row = self.sql_dialect.insert_trace_event(
                    cur,
                    (
                        event_id,
                        uri,
                        account_id,
                        owner_space,
                        agent_id,
                        session_id,
                        hook_type,
                        event_time,
                        _json_value(content),
                        abstract,
                        overview,
                    ),
                )
                if row is None:
                    cur.execute(
                        f"SELECT {_EVENT_COLUMNS} FROM trace_events "
                        "WHERE account_id = %s AND event_id = %s",
                        (account_id, event_id),
                    )
                    row = cur.fetchone()
                if row is None:
                    raise RuntimeError("trace event was not persisted")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

        return {
            "ok": True,
            "event_id": row[0],
            "uri": row[1],
            "event_index": int(row[7]),
            "inserted": inserted,
        }

    def finalize_session(
        self,
        *,
        account_id: str,
        owner_space: str,
        agent_id: str,
        session_id: str,
        stop_event_index: int,
    ) -> dict:
        """Aggregate exactly one turn-end-bounded turn into ``trace_sessions``."""
        if not session_id or session_id == "unknown":
            raise ValueError("valid session_id is required")
        if stop_event_index <= 0:
            raise ValueError("stop_event_index must be positive")

        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_time, COALESCE(content->>'write_batch_id', '')
                    FROM trace_events
                    WHERE account_id = %s AND session_id = %s
                      AND owner_space = %s AND agent_id = %s
                      AND event_index = %s
                      AND hook_type IN ('turn_end', 'Stop')
                    """,
                    (
                        account_id,
                        session_id,
                        owner_space,
                        agent_id,
                        stop_event_index,
                    ),
                )
                stop_row = cur.fetchone()
                if stop_row is None:
                    raise ValueError("turn-end event not found")

                write_batch_id = stop_row[1]
                prev_stop_index = None
                if not write_batch_id:
                    cur.execute(
                        """
                        SELECT MAX(event_index)
                        FROM trace_events
                        WHERE account_id = %s AND session_id = %s
                          AND owner_space = %s AND agent_id = %s
                          AND hook_type IN ('turn_end', 'Stop') AND event_index < %s
                          AND COALESCE(content->>'write_batch_id', '') = ''
                        """,
                        (
                            account_id,
                            session_id,
                            owner_space,
                            agent_id,
                            stop_event_index,
                        ),
                    )
                    previous_row = cur.fetchone()
                    prev_stop_index = previous_row[0] if previous_row else None

                events, total_event_count, started_at, ended_at = self._fetch_turn_events(
                    cur,
                    account_id,
                    owner_space,
                    agent_id,
                    session_id,
                    int(prev_stop_index or 0),
                    stop_event_index,
                    write_batch_id,
                )
                if not events:
                    raise RuntimeError("turn-end-bounded trace turn is empty")

                turn_id = _turn_id(
                    account_id,
                    owner_space,
                    agent_id,
                    session_id,
                    stop_event_index,
                )
                uri = f"ctx://{account_id}/traces/{session_id}/turns/{turn_id}"
                abstract = self._session_abstract(events)
                metadata = self._session_metadata(events, total_event_count)
                stored_content = [event["content"] for event in events]
                persisted_uri = self.sql_dialect.upsert_trace_session(
                    cur,
                    (
                        turn_id,
                        uri,
                        account_id,
                        owner_space,
                        agent_id,
                        session_id,
                        started_at or stop_row[0],
                        ended_at or stop_row[0],
                        total_event_count,
                        abstract,
                        abstract,
                        _json_value(stored_content),
                        _json_value(metadata),
                        stop_event_index,
                        prev_stop_index,
                    ),
                )
                cur.execute(
                    """
                    DELETE FROM trace_session_events
                    WHERE account_id = %s AND evidence_uri = %s
                    """,
                    (account_id, uri),
                )
                _insert_trace_session_events(
                    cur,
                    account_id=account_id,
                    evidence_uri=uri,
                    owner_space=owner_space,
                    agent_id=agent_id,
                    session_id=session_id,
                    after_event_index=int(prev_stop_index or 0),
                    stop_event_index=stop_event_index,
                    write_batch_id=write_batch_id,
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)

        return {
            "ok": True,
            "uri": persisted_uri or uri,
            "turn_id": turn_id,
            "event_count": total_event_count,
            "stored_event_count": len(events),
            "stop_event_index": stop_event_index,
            "prev_stop_event_index": prev_stop_index,
            "status": "finalized",
        }

    def enqueue_skill_candidate_from_trace(
        self,
        *,
        account_id: str,
        agent_id: str,
        session_id: str,
        turn_id: str,
        trace_uri: str,
    ) -> dict:
        """Idempotently make one finalized trace available to skill sedimentation."""
        candidate_id = f"ctx://{account_id}/skills/candidates/{turn_id}"
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                self.sql_dialect.ensure_skill_candidate(
                    cur,
                    (candidate_id, account_id, agent_id),
                )
                self.sql_dialect.ensure_skill_candidate_evidence(
                    cur,
                    (account_id, candidate_id, trace_uri, session_id),
                )
                self.sql_dialect.touch_skill_work_account(cur, account_id)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return {"ok": True, "candidate_id": candidate_id}

    def claim_skill_candidates(
        self,
        *,
        account_id: str,
        worker_id: str,
        limit: int,
        lease_seconds: int,
    ) -> list[dict]:
        """Lease finalized turns for skill assessment without double work."""
        if not worker_id or len(worker_id) > 128:
            raise ValueError("valid worker_id is required")
        if min(limit, lease_seconds) < 1:
            raise ValueError("limit and lease_seconds must be positive")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH candidates AS (
                        SELECT c.candidate_id
                        FROM skill_candidates c
                        JOIN skill_candidate_evidence e
                          ON e.account_id = c.account_id
                         AND e.candidate_id = c.candidate_id
                        JOIN trace_sessions t
                          ON t.account_id = e.account_id
                         AND t.uri = e.evidence_uri
                        WHERE c.account_id = %s AND t.status = 'finalized'
                          AND c.next_attempt_at <= NOW()
                          AND (
                              c.state = 'pending'
                              OR (
                                  c.state = 'processing' AND c.lease_until <= NOW()
                                  AND c.capability_key IS NULL
                                  AND NOT (COALESCE(c.assessment, '{}'::jsonb)
                                           ? 'staged_publication')
                              )
                          )
                        ORDER BY t.ended_at, c.candidate_id
                        FOR UPDATE SKIP LOCKED
                        LIMIT %s
                    )
                    UPDATE skill_candidates AS c
                    SET state = 'processing',
                        attempts = attempts + 1,
                        lease_owner = %s,
                        lease_until = NOW() + (%s * INTERVAL '1 second'),
                        error = NULL,
                        updated_at = NOW()
                    FROM candidates, skill_candidate_evidence e, trace_sessions t
                    WHERE c.candidate_id = candidates.candidate_id
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND t.account_id = e.account_id
                      AND t.uri = e.evidence_uri
                    RETURNING t.turn_id, t.uri, t.account_id,
                              t.owner_space, t.agent_id, t.session_id,
                              t.metadata, c.attempts, c.assessment
                    """,
                    (account_id, limit, worker_id, lease_seconds),
                )
                rows = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return [
            {
                "turn_id": row[0],
                "uri": row[1],
                "account_id": row[2],
                "owner_space": row[3],
                "agent_id": row[4],
                "session_id": row[5],
                "metadata": row[6] if isinstance(row[6], dict) else {},
                "skill_attempts": int(row[7]),
                "assessment": row[8] if isinstance(row[8], dict) else {},
            }
            for row in rows
        ]

    def read_skill_trace_event_page(
        self,
        *,
        account_id: str,
        evidence_uri: str,
        after_ordinal: int,
        limit: int,
    ) -> dict:
        """Read one bounded, ordered page from a candidate's finalized trace."""
        if after_ordinal < 0 or limit < 1:
            raise ValueError("after_ordinal must be non-negative and limit must be positive")
        conn = self._get_connection()
        backfilled = False
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM trace_session_events
                    WHERE account_id = %s AND evidence_uri = %s
                    LIMIT 1
                    """,
                    (account_id, evidence_uri),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        """
                        SELECT content, owner_space, agent_id, session_id,
                               stop_event_index,
                               COALESCE(prev_stop_event_index, 0),
                               event_count
                        FROM trace_sessions t
                        WHERE t.account_id = %s AND t.uri = %s
                          AND t.status = 'finalized'
                        FOR UPDATE
                        """,
                        (account_id, evidence_uri),
                    )
                    content_row = cur.fetchone()
                    cur.execute(
                        """
                        SELECT 1
                        FROM trace_session_events
                        WHERE account_id = %s AND evidence_uri = %s
                        LIMIT 1
                        """,
                        (account_id, evidence_uri),
                    )
                    if content_row is not None and cur.fetchone() is None:
                        projected_count = _insert_trace_session_events(
                            cur,
                            account_id=account_id,
                            evidence_uri=evidence_uri,
                            owner_space=content_row[1],
                            agent_id=content_row[2],
                            session_id=content_row[3],
                            after_event_index=int(content_row[5]),
                            stop_event_index=int(content_row[4]),
                            write_batch_id=content_row[0][-1].get("write_batch_id", "")
                            if content_row[0] else "",
                        )
                        if projected_count != int(content_row[6]):
                            cur.execute(
                                """
                                DELETE FROM trace_session_events
                                WHERE account_id = %s AND evidence_uri = %s
                                """,
                                (account_id, evidence_uri),
                            )
                            cur.execute(
                                """
                                INSERT INTO trace_session_events
                                    (account_id, evidence_uri, ordinal, event)
                                SELECT %s, %s, item.ordinality, item.event
                                FROM jsonb_array_elements(%s::jsonb)
                                    WITH ORDINALITY AS item(event, ordinality)
                                """,
                                (
                                    account_id,
                                    evidence_uri,
                                    _json_value(content_row[0]),
                                ),
                            )
                        backfilled = cur.rowcount > 0
                cur.execute(
                    """
                    SELECT event, ordinal
                    FROM trace_session_events
                    WHERE account_id = %s AND evidence_uri = %s
                      AND ordinal > %s
                    ORDER BY ordinal
                    LIMIT %s
                    """,
                    (account_id, evidence_uri, after_ordinal, limit + 1),
                )
                rows = cur.fetchall()
            if backfilled:
                conn.commit()
            else:
                conn.rollback()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        visible = rows[:limit]
        events = [
            {"ordinal": int(row[1]), "event": row[0] if isinstance(row[0], dict) else {}}
            for row in visible
        ]
        return {
            "events": events,
            "next_ordinal": events[-1]["ordinal"] if events else after_ordinal,
            "done": len(rows) <= limit,
        }

    def save_skill_compaction_checkpoint(
        self,
        *,
        account_id: str,
        uri: str,
        worker_id: str,
        checkpoint: dict,
        lease_seconds: int,
    ) -> bool:
        """Persist a bounded sanitized compaction cursor under the active lease."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET assessment = COALESCE(c.assessment, '{}'::jsonb)
                                     || jsonb_build_object('compaction', %s::jsonb),
                        lease_until = NOW() + (%s * INTERVAL '1 second'),
                        updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s AND c.lease_owner = %s
                      AND c.state = 'processing'
                      AND c.lease_until > NOW()
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = %s
                    """,
                    (json.dumps(checkpoint), lease_seconds, account_id, worker_id, uri),
                )
                updated = cur.rowcount == 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return updated

    def renew_skill_leases(
        self,
        *,
        account_id: str,
        uris: list[str],
        worker_id: str,
        lease_seconds: int,
    ) -> int:
        """Extend leases owned by one worker immediately before slow external work."""
        if not uris:
            return 0
        if not worker_id or len(worker_id) > 128 or lease_seconds < 1:
            raise ValueError("valid worker_id and positive lease_seconds are required")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET lease_until = NOW() + (%s * INTERVAL '1 second'),
                        updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = ANY(%s)
                      AND c.state = 'processing' AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (lease_seconds, account_id, uris, worker_id),
                )
                changed = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return int(changed)

    def mark_skill_assessed(
        self,
        *,
        account_id: str,
        uri: str,
        worker_id: str,
        capability_key: str,
        assessment: dict,
    ) -> bool:
        return self._finish_skill_processing(
            account_id=account_id,
            uri=uri,
            worker_id=worker_id,
            state="assessed",
            capability_key=capability_key,
            assessment=assessment,
        )

    def reject_skill_candidate(
        self,
        *,
        account_id: str,
        uri: str,
        worker_id: str,
        state: str,
        error: str,
    ) -> bool:
        """Reject only a turn still owned by the caller's active lease."""
        if state not in {"skipped", "held"}:
            raise ValueError("rejected candidate state must be skipped or held")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET state = %s, error = %s,
                        lease_owner = NULL, lease_until = NULL,
                        updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = %s
                      AND c.state = 'processing' AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (state, error[:2000], account_id, uri, worker_id),
                )
                changed = cur.rowcount == 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return changed

    def mark_skill_terminal(
        self,
        *,
        account_id: str,
        uris: list[str],
        worker_id: str,
        state: str,
        skill_uri: str | None = None,
        error: str | None = None,
    ) -> int:
        """Terminalize only evidence still owned by ``worker_id``."""
        if state not in {"consumed", "held", "skipped"}:
            raise ValueError("invalid terminal skill state")
        if not uris:
            return 0
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                if state == "consumed":
                    self.sql_dialect.insert_skill_example_evidence(
                        cur,
                        (skill_uri, account_id, uris, worker_id),
                    )
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET state = %s, skill_uri = %s, error = %s,
                        consumed_at = CASE WHEN %s = 'consumed' THEN NOW() ELSE NULL END,
                        lease_owner = NULL, lease_until = NULL,
                        updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = ANY(%s)
                      AND c.state = 'processing'
                      AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (
                        state,
                        skill_uri,
                        (error or "")[:2000] or None,
                        state,
                        account_id,
                        uris,
                        worker_id,
                    ),
                )
                changed = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return int(changed)

    def claim_skill_evidence(
        self,
        *,
        account_id: str,
        agent_id: str,
        capability_key: str,
        worker_id: str,
        min_evidence_turns: int,
        min_distinct_sessions: int,
        max_evidence_turns: int,
        lease_seconds: int,
    ) -> list[dict]:
        """Atomically lease an eligible capability batch to one compiler.

        A transaction-scoped advisory lock serializes contenders for the same
        tenant, agent, and capability. Selection favours one turn per session
        before taking additional turns, preserving evidence diversity under
        the configured bound. Expired compilation leases are reclaimable.
        """
        if not worker_id or len(worker_id) > 128:
            raise ValueError("valid worker_id is required")
        if not capability_key:
            raise ValueError("capability_key is required")
        if min_evidence_turns < 1 or min_distinct_sessions < 1:
            raise ValueError("evidence thresholds must be positive")
        if max_evidence_turns < max(min_evidence_turns, min_distinct_sessions):
            raise ValueError("max_evidence_turns must cover both evidence thresholds")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")

        lock_digest = hashlib.sha256("\0".join((account_id, agent_id)).encode()).digest()
        lock_key = int.from_bytes(lock_digest[:8], "big", signed=True)
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_try_advisory_xact_lock(%s)",
                    (lock_key,),
                )
                if not bool((cur.fetchone() or (False,))[0]):
                    conn.rollback()
                    return []
                cur.execute(
                    """
                    SELECT capability_key
                    FROM skill_candidates
                    WHERE account_id = %s AND agent_id = %s
                      AND capability_key IS NOT NULL
                      AND state IN ('assessed', 'processing')
                      AND COALESCE(assessment, '{}'::jsonb)
                          ? 'staged_publication'
                    ORDER BY updated_at, capability_key
                    LIMIT 1
                    """,
                    (account_id, agent_id),
                )
                staged_row = cur.fetchone()
                if staged_row is not None and str(staged_row[0]) != capability_key:
                    conn.rollback()
                    return []
                # The advisory lock only serializes claim transactions.  A
                # compilation lease outlives that transaction, so refuse to
                # form a second cohort while another worker still owns one
                # for the same tenant/agent catalog. Semantic deduplication is
                # a scan-then-write operation, so different capability labels
                # must not compile concurrently into duplicate skills.
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM skill_candidates
                        WHERE account_id = %s AND agent_id = %s
                          AND capability_key IS NOT NULL
                          AND state = 'processing'
                          AND lease_until > NOW()
                    )
                    """,
                    (account_id, agent_id),
                )
                if bool((cur.fetchone() or (False,))[0]):
                    conn.rollback()
                    return []
                eligible = """
                    c.account_id = %s AND c.agent_id = %s
                    AND c.capability_key = %s
                    AND c.next_attempt_at <= NOW()
                    AND (
                        c.state = 'assessed'
                        OR (c.state = 'processing' AND c.lease_until <= NOW())
                    )
                """
                cur.execute(
                    f"""
                    SELECT COUNT(*), COUNT(DISTINCT e.session_id),
                           BOOL_OR(
                               COALESCE(c.assessment->>'eligibility_state', 'eligible')
                                   = 'pending_support'
                           ) AS has_pending_support
                    FROM skill_candidates c
                    JOIN skill_candidate_evidence e
                      ON e.account_id = c.account_id
                     AND e.candidate_id = c.candidate_id
                    WHERE {eligible}
                    """,
                    (account_id, agent_id, capability_key),
                )
                count, session_count, has_pending_support = cur.fetchone() or (0, 0, False)
                required_evidence = max(
                    min_evidence_turns,
                    2 if has_pending_support else min_evidence_turns,
                )
                if int(count) < required_evidence or int(session_count) < min_distinct_sessions:
                    cur.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM skill_candidates
                            WHERE account_id = %s AND agent_id = %s
                              AND capability_key = %s
                              AND state = 'processing' AND lease_until <= NOW()
                              AND COALESCE(assessment, '{}'::jsonb) ? 'staged_publication'
                        )
                        """,
                        (account_id, agent_id, capability_key),
                    )
                    if not bool((cur.fetchone() or (False,))[0]):
                        conn.rollback()
                        return []
                cur.execute(
                    f"""
                    WITH ranked AS (
                        SELECT c.candidate_id, e.evidence_uri, e.session_id, t.ended_at,
                               COALESCE(c.assessment, '{{}}'::jsonb)
                                   ? 'staged_publication' AS is_staged,
                               ROW_NUMBER() OVER (
                                   PARTITION BY e.session_id ORDER BY t.ended_at, e.evidence_uri
                               ) AS session_rank
                        FROM skill_candidates c
                        JOIN skill_candidate_evidence e
                          ON e.account_id = c.account_id
                         AND e.candidate_id = c.candidate_id
                        JOIN trace_sessions t
                          ON t.account_id = e.account_id
                         AND t.uri = e.evidence_uri
                        WHERE {eligible}
                    ), selected AS (
                        SELECT candidate_id
                        FROM ranked
                        ORDER BY is_staged DESC,
                                 CASE WHEN session_rank = 1 THEN 0 ELSE 1 END,
                                 ended_at, evidence_uri
                        LIMIT %s
                    )
                    UPDATE skill_candidates AS c
                    SET state = 'processing',
                        compile_attempts = compile_attempts + 1,
                        lease_owner = %s,
                        lease_until = NOW() + (%s * INTERVAL '1 second'),
                        error = NULL,
                        updated_at = NOW()
                    FROM selected, skill_candidate_evidence e, trace_sessions t
                    WHERE c.account_id = %s
                      AND c.candidate_id = selected.candidate_id
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND t.account_id = e.account_id
                      AND t.uri = e.evidence_uri
                    RETURNING t.uri, e.session_id,
                              CASE
                                  WHEN COALESCE(c.assessment, '{{}}'::jsonb)
                                       ? 'source_events'
                                  THEN NULL
                                  ELSE t.content
                              END AS legacy_content,
                              c.assessment, c.compile_attempts
                    """,
                    (
                        account_id,
                        agent_id,
                        capability_key,
                        max_evidence_turns,
                        worker_id,
                        lease_seconds,
                        account_id,
                    ),
                )
                rows = cur.fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return [
            {
                "uri": row[0],
                "session_id": row[1],
                "content": (
                    row[3].get("source_events", [])
                    if isinstance(row[3], dict) and "source_events" in row[3]
                    else (row[2] if isinstance(row[2], list) else [])
                ),
                "assessment": row[3] if isinstance(row[3], dict) else {},
                "staged_publication": (
                    row[3].get("staged_publication") if isinstance(row[3], dict) else None
                ),
                "skill_compile_attempts": int(row[4]),
            }
            for row in rows
        ]

    def retry_skill_evidence(
        self,
        *,
        account_id: str,
        uris: list[str],
        worker_id: str,
        max_attempts: int,
        backoff_base_seconds: int,
        backoff_cap_seconds: int,
        error: str,
    ) -> str:
        """Release an owned batch with separately bounded staged-write recovery."""
        if not uris:
            return "assessed"
        if min(max_attempts, backoff_base_seconds, backoff_cap_seconds) < 1:
            raise ValueError("retry limits must be positive")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(MAX(c.compile_attempts), 0),
                           COALESCE(MAX(
                               (COALESCE(c.assessment, '{}'::jsonb)
                                   ->>'staged_recovery_attempts')::int
                           ), 0),
                           BOOL_OR(
                               COALESCE(c.assessment, '{}'::jsonb)
                                   ? 'staged_publication'
                           )
                    FROM skill_candidates c
                    JOIN skill_candidate_evidence e
                      ON e.account_id = c.account_id
                     AND e.candidate_id = c.candidate_id
                    WHERE c.account_id = %s AND e.evidence_uri = ANY(%s)
                      AND c.state = 'processing' AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (account_id, uris, worker_id),
                )
                row = cur.fetchone() or (0, 0, False)
                compile_attempts = int(row[0])
                recovery_attempts = int(row[1]) + 1
                staged = bool(row[2])
                effective_attempts = recovery_attempts if staged else compile_attempts
                terminal = effective_attempts >= max_attempts
                state = "held" if terminal else ("processing" if staged else "assessed")
                delay = min(
                    backoff_cap_seconds,
                    backoff_base_seconds * (2 ** max(0, effective_attempts - 1)),
                )
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET state = %s,
                        assessment = CASE WHEN %s THEN
                            jsonb_set(
                                COALESCE(assessment, '{}'::jsonb),
                                '{staged_recovery_attempts}', %s::jsonb, true
                            )
                            ELSE assessment
                        END,
                        next_attempt_at = NOW() + (%s * INTERVAL '1 second'),
                        lease_owner = NULL,
                        lease_until = CASE
                            WHEN %s AND %s <> 'held'
                            THEN NOW() + (%s * INTERVAL '1 second') ELSE NULL
                        END,
                        error = %s, updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = ANY(%s)
                      AND c.state = 'processing' AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (
                        state,
                        staged,
                        str(recovery_attempts),
                        delay,
                        staged,
                        state,
                        delay,
                        error[:2000],
                        account_id,
                        uris,
                        worker_id,
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return state

    def release_cancelled_skill_evidence(
        self,
        *,
        account_id: str,
        uris: list[str],
        worker_id: str,
    ) -> int:
        """Release a cancelled compile claim without consuming a retry attempt."""
        if not uris:
            return 0
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET state = 'assessed',
                        compile_attempts = GREATEST(compile_attempts - 1, 0),
                        next_attempt_at = NOW(),
                        lease_owner = NULL, lease_until = NULL,
                        error = NULL, updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = ANY(%s)
                      AND c.state = 'processing' AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (account_id, uris, worker_id),
                )
                changed = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return int(changed)

    def retry_skill_candidate(
        self,
        *,
        account_id: str,
        uri: str,
        worker_id: str,
        attempts: int,
        max_attempts: int,
        backoff_base_seconds: int,
        backoff_cap_seconds: int,
        error: str,
    ) -> str:
        """Release a candidate with separately bounded staged-write recovery."""
        if min(max_attempts, backoff_base_seconds, backoff_cap_seconds) < 1:
            raise ValueError("retry limits must be positive")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(
                               (c.assessment->>'staged_recovery_attempts')::int, 0
                           ),
                           COALESCE(c.assessment, '{}'::jsonb) ? 'staged_publication'
                    FROM skill_candidates c
                    JOIN skill_candidate_evidence e
                      ON e.account_id = c.account_id AND e.candidate_id = c.candidate_id
                    WHERE c.account_id = %s AND e.evidence_uri = %s
                      AND c.state = 'processing' AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (account_id, uri, worker_id),
                )
                row = cur.fetchone() or (0, False)
                recovery_attempts = int(row[0]) + 1
                staged = bool(row[1])
                effective_attempts = recovery_attempts if staged else attempts
                terminal = effective_attempts >= max_attempts
                state = "held" if terminal else ("processing" if staged else "pending")
                delay = min(
                    backoff_cap_seconds,
                    backoff_base_seconds * (2 ** max(0, effective_attempts - 1)),
                )
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET state = %s,
                        assessment = CASE WHEN %s THEN
                            jsonb_set(
                                COALESCE(assessment, '{}'::jsonb),
                                '{staged_recovery_attempts}', %s::jsonb, true
                            )
                            ELSE assessment
                        END,
                        next_attempt_at = NOW() + (%s * INTERVAL '1 second'),
                        lease_owner = NULL,
                        lease_until = CASE
                            WHEN %s AND %s <> 'held'
                            THEN NOW() + (%s * INTERVAL '1 second') ELSE NULL
                        END,
                        error = %s, updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = %s
                      AND c.state = 'processing' AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (
                        state,
                        staged,
                        str(recovery_attempts),
                        delay,
                        staged,
                        state,
                        delay,
                        error[:2000],
                        account_id,
                        uri,
                        worker_id,
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return state

    def reset_stale_skill_publication(
        self,
        *,
        account_id: str,
        uris: list[str],
        worker_id: str,
        requeue_candidate: bool,
        error: str,
        max_attempts: int,
        backoff_base_seconds: int,
        backoff_cap_seconds: int,
    ) -> str:
        """Bound stale rebases and discard only the obsolete staged payload."""
        if not uris:
            return "lost"
        if min(max_attempts, backoff_base_seconds, backoff_cap_seconds) < 1:
            raise ValueError("retry limits must be positive")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(MAX(
                        CASE WHEN %s THEN c.attempts ELSE c.compile_attempts END
                    ), 0)
                    FROM skill_candidates c
                    JOIN skill_candidate_evidence e
                      ON e.account_id = c.account_id AND e.candidate_id = c.candidate_id
                    WHERE c.account_id = %s AND e.evidence_uri = ANY(%s)
                      AND c.state = 'processing' AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (requeue_candidate, account_id, uris, worker_id),
                )
                attempts = int((cur.fetchone() or (0,))[0])
                terminal = attempts >= max_attempts
                state = "held" if terminal else ("pending" if requeue_candidate else "assessed")
                delay = min(
                    backoff_cap_seconds,
                    backoff_base_seconds * (2 ** max(0, attempts - 1)),
                )
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET state = %s,
                        capability_key = CASE WHEN %s THEN NULL ELSE capability_key END,
                        assessment = COALESCE(assessment, '{}'::jsonb) - 'staged_publication',
                        next_attempt_at = NOW() + (%s * INTERVAL '1 second'),
                        lease_owner = NULL, lease_until = NULL,
                        error = %s, updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = ANY(%s)
                      AND c.state = 'processing' AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (
                        state,
                        requeue_candidate,
                        delay,
                        error[:2000],
                        account_id,
                        uris,
                        worker_id,
                    ),
                )
                changed = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return state if changed == len(uris) else "lost"

    def release_cancelled_skill_candidate(
        self,
        *,
        account_id: str,
        uri: str,
        worker_id: str,
    ) -> bool:
        """Release a cancelled assessment claim without consuming a retry attempt."""
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET state = CASE
                            WHEN COALESCE(c.assessment, '{}'::jsonb) ? 'staged_publication'
                            THEN 'processing' ELSE 'pending'
                        END,
                        attempts = GREATEST(attempts - 1, 0),
                        next_attempt_at = NOW(),
                        lease_owner = NULL,
                        lease_until = CASE
                            WHEN COALESCE(c.assessment, '{}'::jsonb) ? 'staged_publication'
                            THEN NOW() ELSE NULL
                        END,
                        error = NULL, updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = %s
                      AND c.state = 'processing' AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (account_id, uri, worker_id),
                )
                changed = cur.rowcount == 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return changed

    def list_ready_skill_capabilities(
        self,
        *,
        account_id: str,
        min_evidence_turns: int,
        min_distinct_sessions: int,
        limit: int,
    ) -> list[dict]:
        """Return assessed capability cohorts ready for compilation.

        This read path is the durable recovery source when a process exits
        after assessment but before compilation is claimed.
        """
        if min_evidence_turns < 1 or min_distinct_sessions < 1:
            raise ValueError("evidence thresholds must be positive")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.agent_id, c.capability_key,
                           COUNT(*) AS evidence_count,
                           COUNT(DISTINCT e.session_id) AS distinct_session_count
                    FROM skill_candidates c
                    JOIN skill_candidate_evidence e
                      ON e.account_id = c.account_id
                     AND e.candidate_id = c.candidate_id
                    WHERE c.account_id = %s
                      AND c.capability_key IS NOT NULL
                      AND c.next_attempt_at <= NOW()
                      AND (
                          c.state = 'assessed'
                          OR (
                              c.state = 'processing' AND c.lease_until <= NOW()
                              AND COALESCE(c.assessment, '{}'::jsonb)
                                  ? 'staged_publication'
                          )
                      )
                    GROUP BY c.agent_id, c.capability_key
                    HAVING BOOL_OR(
                               COALESCE(c.assessment, '{}'::jsonb)
                                   ? 'staged_publication'
                           )
                        OR (
                            COUNT(*) >= %s
                            AND COUNT(DISTINCT e.session_id) >= %s
                            AND (
                                COUNT(*) >= 2
                                OR NOT BOOL_OR(
                                    COALESCE(
                                        c.assessment->>'eligibility_state', 'eligible'
                                    ) = 'pending_support'
                                )
                            )
                        )
                    ORDER BY BOOL_OR(
                                 COALESCE(c.assessment, '{}'::jsonb)
                                     ? 'staged_publication'
                             ) DESC,
                             MIN(c.created_at), c.agent_id, c.capability_key
                    LIMIT %s
                    """,
                    (
                        account_id,
                        min_evidence_turns,
                        min_distinct_sessions,
                        limit,
                    ),
                )
                rows = cur.fetchall()
            conn.rollback()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return [
            {
                "agent_id": row[0],
                "capability_key": row[1],
                "evidence_count": int(row[2]),
                "distinct_session_count": int(row[3]),
            }
            for row in rows
        ]

    def count_skill_distinct_sessions(
        self,
        *,
        account_id: str,
        skill_uri: str,
        evidence_uris: list[str],
        worker_id: str,
    ) -> int:
        """Count historical and currently leased evidence sessions exactly once."""
        if not skill_uri or not evidence_uris:
            raise ValueError("skill_uri and evidence_uris are required")
        if not worker_id or len(worker_id) > 128:
            raise ValueError("valid worker_id is required")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(DISTINCT e.session_id)
                    FROM skill_candidates c
                    JOIN skill_candidate_evidence e
                      ON e.account_id = c.account_id
                     AND e.candidate_id = c.candidate_id
                    WHERE c.account_id = %s
                      AND (
                          (c.state = 'consumed' AND c.skill_uri = %s)
                          OR (
                              e.evidence_uri = ANY(%s)
                              AND c.state = 'processing'
                              AND c.lease_owner = %s
                              AND c.lease_until > NOW()
                          )
                      )
                    """,
                    (account_id, skill_uri, evidence_uris, worker_id),
                )
                count = int((cur.fetchone() or (0,))[0])
            conn.rollback()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return count

    def count_skill_example_distinct_sessions(
        self,
        *,
        account_id: str,
        skill_uri: str,
        examples: list[dict[str, object]],
        worker_id: str,
    ) -> dict[str, int]:
        """Count historical and currently leased sessions for each exact example."""
        if not skill_uri or not examples:
            raise ValueError("skill_uri and examples are required")
        if not worker_id or len(worker_id) > 128:
            raise ValueError("valid worker_id is required")
        normalized = []
        for item in examples:
            example_id = item.get("example_id") if isinstance(item, dict) else None
            example_hash = item.get("example_hash") if isinstance(item, dict) else None
            evidence_uris = item.get("evidence_uris") if isinstance(item, dict) else None
            if (
                example_id not in {"primary", "branch-1", "branch-2"}
                or not isinstance(example_hash, str)
                or len(example_hash) != 64
                or any(char not in "0123456789abcdef" for char in example_hash)
                or not isinstance(evidence_uris, list)
                or any(not isinstance(uri, str) or not uri for uri in evidence_uris)
                or len(set(evidence_uris)) != len(evidence_uris)
            ):
                raise ValueError("invalid worked example evidence key")
            normalized.append((example_id, example_hash, evidence_uris))
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            counts = {}
            with conn.cursor() as cur:
                for example_id, example_hash, evidence_uris in normalized:
                    cur.execute(
                        """
                        SELECT COUNT(DISTINCT sessions.session_id)
                        FROM (
                            SELECT mapped.session_id
                            FROM skill_example_evidence AS mapped
                            WHERE mapped.account_id = %s
                              AND mapped.skill_uri = %s
                              AND mapped.example_id = %s
                              AND mapped.example_hash = %s
                            UNION ALL
                            SELECT evidence.session_id
                            FROM skill_candidates AS candidate
                            JOIN skill_candidate_evidence AS evidence
                              ON evidence.account_id = candidate.account_id
                             AND evidence.candidate_id = candidate.candidate_id
                            WHERE candidate.account_id = %s
                              AND evidence.evidence_uri = ANY(%s)
                              AND candidate.state = 'processing'
                              AND candidate.lease_owner = %s
                              AND candidate.lease_until > NOW()
                        ) AS sessions
                        """,
                        (
                            account_id,
                            skill_uri,
                            example_id,
                            example_hash,
                            account_id,
                            evidence_uris,
                            worker_id,
                        ),
                    )
                    key = f"{example_id}:{example_hash}"
                    counts[key] = int((cur.fetchone() or (0,))[0])
            conn.rollback()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return counts

    def stage_skill_publication(
        self,
        *,
        account_id: str,
        uris: list[str],
        worker_id: str,
        publication: dict,
    ) -> int:
        """Durably stage a publication on evidence owned by the compiler."""
        if not uris:
            return 0
        if not worker_id or len(worker_id) > 128:
            raise ValueError("valid worker_id is required")
        publication = validate_skill_publication(publication)
        capability_key = publication.get("capability_key")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET capability_key = COALESCE(c.capability_key, %s),
                        assessment = COALESCE(assessment, '{}'::jsonb)
                                     || jsonb_build_object(
                                         'staged_publication', %s::jsonb
                                     ),
                        updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = ANY(%s)
                      AND c.state = 'processing'
                      AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (capability_key, _json_value(publication), account_id, uris, worker_id),
                )
                changed = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return int(changed)

    def release_unstaged_skill_evidence(
        self, *, account_id: str, uris: list[str], worker_id: str
    ) -> int:
        """Return evidence accidentally co-claimed with a staged recovery batch."""

        if not uris:
            return 0
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET state = 'assessed',
                        compile_attempts = GREATEST(compile_attempts - 1, 0),
                        lease_owner = NULL, lease_until = NULL,
                        updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = ANY(%s)
                      AND c.state = 'processing' AND c.lease_owner = %s
                    """,
                    (account_id, uris, worker_id),
                )
                changed = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return int(changed)

    def list_skill_candidates(
        self,
        *,
        account_id: str,
        agent_id: str,
        state: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Expose safe candidate summaries without trace content or identifiers."""
        state_map = {None: ("assessed", "held"), "accumulating": ("assessed",), "held": ("held",)}
        if state not in state_map:
            raise ValueError("state must be accumulating or held")
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.capability_key, c.state,
                           COUNT(*) AS evidence_count,
                           COUNT(DISTINCT e.session_id) AS session_count,
                           MAX(c.assessment->>'summary') AS summary,
                           MAX(COALESCE(c.error, c.assessment->>'reason')) AS reason
                    FROM skill_candidates c
                    JOIN skill_candidate_evidence e
                      ON e.account_id = c.account_id
                     AND e.candidate_id = c.candidate_id
                    WHERE c.account_id = %s AND c.agent_id = %s
                      AND c.state = ANY(%s)
                    GROUP BY c.capability_key, c.state
                    ORDER BY MAX(c.updated_at) DESC
                    LIMIT %s
                    """,
                    (account_id, agent_id, list(state_map[state]), limit),
                )
                rows = cur.fetchall()
            conn.rollback()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return [
            {
                "capability_key": row[0] or "unclassified",
                "state": "accumulating" if row[1] == "assessed" else row[1],
                "evidence_count": int(row[2]),
                "distinct_session_count": int(row[3]),
                "summary": row[4] or "",
                "reason": row[5] or "",
            }
            for row in rows
        ]

    def list_pending_skill_accounts(
        self,
        *,
        account_ids: list[str] | None = None,
        limit: int = 1000,
    ) -> list[str]:
        """Filter an authorized account set to tenants with actionable work.

        The caller supplies account IDs already visible through the control
        plane. Each probe binds RLS independently and rolls the transaction
        back before the connection is reused, avoiding any tenant-wide bypass.
        """
        if limit < 1 or limit > 10000:
            raise ValueError("limit must be between 1 and 10000")
        pending: list[str] = []
        conn = self._get_connection()
        try:
            if account_ids is None:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT account_id FROM skill_work_accounts ORDER BY updated_at ASC LIMIT %s",
                        (limit,),
                    )
                    unique_accounts = [str(row[0]) for row in cur.fetchall()]
                conn.rollback()
            else:
                unique_accounts = list(dict.fromkeys(str(item) for item in account_ids if item))[
                    :limit
                ]
            for account_id in unique_accounts:
                self.bind_tenant(conn, account_id)
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT account_id FROM skill_work_accounts "
                        "WHERE account_id = %s FOR UPDATE",
                        (account_id,),
                    )
                    if cur.fetchone() is None:
                        conn.rollback()
                        continue
                    cur.execute(
                        """
                        SELECT
                            EXISTS (
                                SELECT 1 FROM skill_candidates
                                WHERE account_id = %s
                                  AND next_attempt_at <= NOW()
                                  AND (
                                      state IN ('pending', 'assessed')
                                      OR (
                                          state = 'processing'
                                          AND lease_until <= NOW()
                                      )
                                  )
                            ),
                            EXISTS (
                                SELECT 1 FROM skill_candidates
                                WHERE account_id = %s
                                  AND state IN ('pending', 'assessed', 'processing')
                            )
                        """,
                        (account_id, account_id),
                    )
                    has_work, has_nonterminal = cur.fetchone() or (False, False)
                    if bool(has_work):
                        pending.append(account_id)
                    if bool(has_nonterminal):
                        cur.execute(
                            "UPDATE skill_work_accounts SET updated_at = NOW() WHERE account_id = %s",
                            (account_id,),
                        )
                    else:
                        cur.execute(
                            "DELETE FROM skill_work_accounts WHERE account_id = %s",
                            (account_id,),
                        )
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return pending

    def count_skill_work_accounts(self) -> int:
        """Return the durable scheduler backlog for health/operations reporting."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM skill_work_accounts")
                count = int((cur.fetchone() or (0,))[0])
            conn.rollback()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return count

    def _finish_skill_processing(
        self,
        *,
        account_id: str,
        uri: str,
        worker_id: str,
        state: str,
        capability_key: str | None = None,
        assessment: dict | None = None,
    ) -> bool:
        conn = self._get_connection()
        try:
            self.bind_tenant(conn, account_id)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE skill_candidates AS c
                    SET state = %s, capability_key = %s,
                        assessment = COALESCE(assessment, '{}'::jsonb)
                                     || (%s::jsonb - 'staged_publication'),
                        lease_owner = NULL,
                        lease_until = NULL, updated_at = NOW()
                    FROM skill_candidate_evidence e
                    WHERE c.account_id = %s
                      AND e.account_id = c.account_id
                      AND e.candidate_id = c.candidate_id
                      AND e.evidence_uri = %s
                      AND c.state = 'processing' AND c.lease_owner = %s
                      AND c.lease_until > NOW()
                    """,
                    (
                        state,
                        capability_key,
                        _json_value(assessment or {}),
                        account_id,
                        uri,
                        worker_id,
                    ),
                )
                changed = cur.rowcount == 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._return_connection(conn)
        return changed

    def _fetch_turn_events(
        self,
        cur,
        account_id: str,
        owner_space: str,
        agent_id: str,
        session_id: str,
        previous_stop_index: int,
        stop_event_index: int,
        write_batch_id: str = "",
    ) -> tuple[list[dict], int, datetime | None, datetime | None]:
        cur.execute(
            """
            SELECT COUNT(*), MIN(event_time), MAX(event_time)
            FROM trace_events
            WHERE account_id = %s AND session_id = %s
              AND owner_space = %s AND agent_id = %s
              AND event_index > %s AND event_index <= %s
              AND COALESCE(content->>'write_batch_id', '') = %s
            """,
            (
                account_id,
                session_id,
                owner_space,
                agent_id,
                previous_stop_index,
                stop_event_index,
                write_batch_id,
            ),
        )
        aggregate = cur.fetchone() or (0, None, None)
        total_event_count = int(aggregate[0] or 0)
        first_row = None
        suffix_limit = self._max_turn_events
        if total_event_count > self._max_turn_events and self._max_turn_events > 1:
            cur.execute(
                """
                SELECT event_index, hook_type, event_time, content, abstract
                FROM trace_events
                WHERE account_id = %s AND session_id = %s
                  AND owner_space = %s AND agent_id = %s
                  AND event_index > %s AND event_index <= %s
                  AND COALESCE(content->>'write_batch_id', '') = %s
                ORDER BY event_index
                LIMIT 1
                """,
                (
                    account_id,
                    session_id,
                    owner_space,
                    agent_id,
                    previous_stop_index,
                    stop_event_index,
                    write_batch_id,
                ),
            )
            first_row = cur.fetchone()
            suffix_limit -= 1
        cur.execute(
            """
            SELECT event_index, hook_type, event_time, content, abstract
            FROM trace_events
            WHERE account_id = %s AND session_id = %s
              AND owner_space = %s AND agent_id = %s
              AND event_index > %s AND event_index <= %s
              AND COALESCE(content->>'write_batch_id', '') = %s
            ORDER BY event_index DESC
            LIMIT %s
            """,
            (
                account_id,
                session_id,
                owner_space,
                agent_id,
                previous_stop_index,
                stop_event_index,
                write_batch_id,
                suffix_limit,
            ),
        )
        rows = cur.fetchall()
        rows.reverse()
        if first_row is not None and (not rows or first_row[0] != rows[0][0]):
            rows.insert(0, first_row)
        events = [
            {
                "event_index": int(index),
                "hook_type": hook_type,
                "event_time": event_time,
                "content": content if isinstance(content, dict) else {},
                "abstract": abstract or "",
            }
            for index, hook_type, event_time, content, abstract in rows
        ]
        return events, total_event_count, aggregate[1], aggregate[2]

    @staticmethod
    def _session_abstract(events: list[dict]) -> str:
        first_user = next(
            (
                event["abstract"]
                for event in events
                if event["content"].get("event_type") == "user_message"
            ),
            "",
        )
        last_assistant = next(
            (
                event["abstract"]
                for event in reversed(events)
                if event["content"].get("event_type") == "assistant_message"
                or (
                    event["content"].get("event_type") == "turn_end"
                    and event["content"].get("assistant_message") is not None
                )
            ),
            "",
        )
        if first_user and last_assistant:
            return f"{first_user} → {last_assistant}"[:500]
        return (first_user or last_assistant)[:500]

    @staticmethod
    def _session_metadata(events: list[dict], total_event_count: int) -> dict:
        hook_counts: dict[str, int] = {}
        tools: set[str] = set()
        tool_call_count = 0
        for event in events:
            hook_type = event["hook_type"]
            hook_counts[hook_type] = hook_counts.get(hook_type, 0) + 1
            content = event["content"]
            if content.get("event_type") in {
                "tool_start",
                "tool_success",
                "tool_failure",
            }:
                tool_call_count += 1
                tool_name = content.get("tool_name")
                if isinstance(tool_name, str) and tool_name:
                    tools.add(tool_name)
        return {
            "event_count": total_event_count,
            "stored_event_count": len(events),
            "truncated": total_event_count > len(events),
            "stored_tool_event_count": tool_call_count,
            "stored_tools_used": sorted(tools),
            "stored_hook_event_counts": hook_counts,
        }
