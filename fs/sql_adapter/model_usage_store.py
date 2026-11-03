"""Shared, append-only provider usage ledger (one identity per actual call)."""

from datetime import UTC, datetime, timedelta

import psycopg2
from psycopg2.errors import UniqueViolation

from fs.sql_adapter.pool import wait_for_postgres
from providers.token_tracker import _empty_daily_counters

COUNTERS = tuple(_empty_daily_counters()) + (
    "failed_calls",
    "missing_usage_calls",
    "local_cache_hits",
)


class SQLModelUsageStore:
    def __init__(self, pool):
        self.pool = pool
        conn = pool.get_connection()
        try:
            with conn.cursor() as cur:
                if pool.sql_dialect.table_exists(cur, "model_usage_events"):
                    conn.rollback()
                    return
                # Serialize first-start DDL across API, MCP and Index.
                cur.execute("SELECT pg_advisory_xact_lock(734810291)")
                columns = ", ".join(f"{name} BIGINT NOT NULL DEFAULT 0" for name in COUNTERS)
                cur.execute(f"""CREATE TABLE IF NOT EXISTS model_usage_events (
                    event_id TEXT NOT NULL, account_id TEXT NOT NULL,
                    owner_space TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
                    occurred_at TIMESTAMPTZ NOT NULL, {columns},
                    PRIMARY KEY (event_id, account_id, owner_space))""")
                cur.execute("""CREATE INDEX IF NOT EXISTS model_usage_scope_time
                    ON model_usage_events(account_id, owner_space, occurred_at)""")
            conn.commit()
        finally:
            self._release(conn)

    def _release(self, conn):
        try:
            conn.rollback()
        except psycopg2.Error:
            conn.close()
        finally:
            self.pool.return_connection(conn)

    def _insert_events(self, cur, events, deadline=None):
        def execute(query, params=None):
            cur.execute(query, params)
            if deadline is not None:
                wait_for_postgres(cur.connection, deadline)

        execute("SET LOCAL statement_timeout = '5s'")
        lock_timeout = self.pool.sql_dialect.lock_timeout_parameter
        if lock_timeout:
            execute(f"SET LOCAL {lock_timeout} = '5s'")
        rows = []
        for event in events:
            for scope, counters in sorted(event["allocations"].items()):
                values = (
                    event["id"],
                    *scope,
                    event["model"],
                    event["kind"],
                    event["at"],
                    *(counters.get(k, 0) for k in COUNTERS),
                )
                rows.append(cur.mogrify('(' + ','.join(['%s'] * len(values)) + ')', values))
        execute(
            b"INSERT INTO model_usage_events "
            b"(event_id,account_id,owner_space,model,kind,occurred_at,"
            + ",".join(COUNTERS).encode()
            + b") VALUES " + b','.join(rows)
        )

    def record(self, event, *, deadline=None):
        """Retry persistence with the same event ID, including uncertain commits."""
        self.record_many([event], deadline=deadline)

    def record_many(self, events, *, deadline=None):
        """Persist and replay an atomic batch with stable per-event identities."""
        if not events:
            return
        for attempt in range(3):
            conn = None
            try:
                if deadline is not None:
                    with self.pool.deadline_connection(deadline) as bounded:
                        with bounded.cursor() as cur:
                            cur.execute("BEGIN")
                            wait_for_postgres(bounded, deadline)
                            self._insert_events(cur, events, deadline)
                            cur.execute("COMMIT")
                            wait_for_postgres(bounded, deadline)
                    return
                conn = self.pool.get_connection()
                with conn.cursor() as cur:
                    self._insert_events(cur, events)
                conn.commit()
                return
            except UniqueViolation:
                return
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                if conn is not None:
                    conn.close()
                if attempt == 2:
                    raise
            finally:
                if conn is not None:
                    self._release(conn)

    def metrics(self, days, account_id, owner_spaces=()):
        """Read cumulative and UTC daily totals from one consistent SQL statement."""
        days = max(1, min(90, int(days)))
        today = datetime.now(UTC).date()
        dates = [(today - timedelta(days=i)).isoformat() for i in reversed(range(days))]
        where = "account_id = %s"
        params = [account_id]
        if owner_spaces:
            where += " AND owner_space = ANY(%s)"
            params.append(list(owner_spaces))
        sums = ",".join(
            (
                f"COUNT(DISTINCT CASE WHEN {k}>0 THEN event_id END)"
                if k
                in {
                    "llm_calls",
                    "embed_calls",
                    "failed_calls",
                    "missing_usage_calls",
                    "local_cache_hits",
                }
                else f"SUM({k})"
            )
            + f" AS {k}"
            for k in COUNTERS
        )
        conn = self.pool.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""SELECT (occurred_at AT TIME ZONE 'UTC')::date,
                    model,kind,{sums} FROM model_usage_events WHERE {where}
                    GROUP BY GROUPING SETS (
                        ((occurred_at AT TIME ZONE 'UTC')::date,model,kind), (model,kind))""",
                    params,
                )
                rows = cur.fetchall()
        finally:
            self._release(conn)
        daily = {d: dict.fromkeys(COUNTERS, 0) for d in dates}
        total = dict.fromkeys(COUNTERS, 0)
        models = []
        for day, model, kind, *values in rows:
            counters = dict(zip(COUNTERS, (int(v or 0) for v in values)))
            if day is None:
                models.append({"model": model, "kind": kind, **counters})
                target = total
            else:
                target = daily.get(day.isoformat())
            if target is not None:
                for k, value in counters.items():
                    target[k] += value
        series = [
            dict(
                date=d,
                llm_input=c["input_tokens"],
                llm_output=c["output_tokens"],
                embedding_input=c["embed_tokens"],
                total=c["input_tokens"] + c["output_tokens"] + c["embed_tokens"],
                llm_calls=c["llm_calls"],
                embedding_calls=c["embed_calls"],
            )
            for d, c in daily.items()
        ]
        return {
            "enabled": True,
            "persistent": True,
            "timezone": "UTC",
            "scope": {"account_id": account_id, "owner_spaces": list(owner_spaces)},
            "models": {
                "llm": ", ".join(sorted({m["model"] for m in models if m["kind"] == "llm"})),
                "embedding": ", ".join(
                    sorted({m["model"] for m in models if m["kind"] == "embedding"})
                ),
            },
            "by_model": models,
            "today": series[-1],
            "series": series,
            "cumulative": {
                "total_tokens": total["input_tokens"]
                + total["output_tokens"]
                + total["embed_tokens"],
                "llm": {
                    **{
                        k: total[k]
                        for k in ("input_tokens", "output_tokens", "cache_read", "cache_write")
                    },
                    "total_tokens": total["input_tokens"] + total["output_tokens"],
                    "calls": total["llm_calls"],
                },
                "embedding": {"total_tokens": total["embed_tokens"], "calls": total["embed_calls"]},
                "failed_calls": total["failed_calls"],
                "missing_usage_calls": total["missing_usage_calls"],
                "local_cache_hits": total["local_cache_hits"],
            },
        }
