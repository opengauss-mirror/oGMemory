"""openGauss SQL-storage dialect."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from fs.sql_adapter.dialect import OPENGAUSS
from fs.sql_adapter.postgresql_dialect import PostgreSQLDialect

_POLICY_TABLE_RE = re.compile(r"CREATE POLICY\s+\w+\s+ON\s+(\w+)", re.IGNORECASE)
_WITH_CHECK_RE = re.compile(
    r"\n\s*WITH CHECK \(account_id = current_setting\('app\.account_id'\)\)",
    re.IGNORECASE,
)


def _tenant_write_constraints(table_names: list[str]) -> str:
    return "\n".join(
        f"""
DO $$ BEGIN
    ALTER TABLE {table_name}
        ADD CONSTRAINT {table_name}_tenant_write_guard
        CHECK (account_id = current_setting('app.account_id')) NOT VALID;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
"""
        for table_name in table_names
    )


class OpenGaussDialect(PostgreSQLDialect):
    name = OPENGAUSS
    supports_batch_claim = False
    # Ordinary roles cannot set lockwait_timeout. statement_timeout also bounds
    # statements waiting for locks, without requiring an administrative role.
    lock_timeout_parameter = None

    def ensure_extraction_work_account(self, cursor: Any, account_id: str) -> None:
        cursor.execute("MERGE INTO extraction_work_accounts target "
                       "USING (SELECT %s::varchar AS account_id) source "
                       "ON (target.account_id = source.account_id) "
                       "WHEN NOT MATCHED THEN INSERT (account_id) VALUES (source.account_id)", (account_id,))

    def prepare_schema(self, schema_sql: str) -> str:
        table_names = list(dict.fromkeys(_POLICY_TABLE_RE.findall(schema_sql)))
        sql = schema_sql.replace(
            "current_setting('app.account_id', true)",
            "current_setting('app.account_id')",
        )
        sql = _WITH_CHECK_RE.sub("", sql)
        return sql + "\n" + _tenant_write_constraints(table_names)

    def validate_connection(self, cursor: Any) -> None:
        cursor.execute("SHOW sql_compatibility")
        row = cursor.fetchone()
        compatibility = str(row[0] if row else "").strip().upper()
        if compatibility != "PG":
            raise RuntimeError(
                "openGauss SQL storage requires a database created with DBCOMPATIBILITY 'PG'"
            )

    def ensure_workspace_root(self, cursor: Any, account_id: str, root_id: str) -> None:
        cursor.execute(
            "MERGE INTO code_workspace_sync_roots target "
            "USING (SELECT %s::varchar AS account_id, %s::varchar AS root_id) source "
            "ON (target.account_id = source.account_id AND target.root_id = source.root_id) "
            "WHEN NOT MATCHED THEN INSERT (account_id, root_id) "
            "VALUES (source.account_id, source.root_id)",
            (account_id, root_id),
        )

    def write_context_node(
        self, cursor: Any, values: tuple[Any, ...], *, create_only: bool
    ) -> bool:
        matched_sql = ""
        if not create_only:
            matched_sql = """
                WHEN MATCHED THEN UPDATE SET
                    status = 'ACTIVE', content = source.content,
                    abstract = source.abstract, overview = source.overview,
                    relations = source.relations, metadata = source.metadata,
                    version = source.version, updated_at = NOW()
            """
        cursor.execute(
            f"""
            MERGE INTO context_nodes target
            USING (
                SELECT %s::varchar AS uri, %s::varchar AS account_id,
                       %s::varchar AS owner_space, %s::varchar AS category,
                       %s::varchar AS context_type, %s::integer AS level,
                       %s::text AS content, %s::text AS abstract,
                       %s::text AS overview, %s::jsonb AS relations,
                       %s::jsonb AS metadata, %s::integer AS version,
                       %s::timestamptz AS created_at
            ) source ON (target.uri = source.uri)
            {matched_sql}
            WHEN NOT MATCHED THEN INSERT
                (uri, account_id, owner_space, category, context_type, level,
                 status, content, abstract, overview, relations, metadata,
                 version, created_at, updated_at)
            VALUES
                (source.uri, source.account_id, source.owner_space,
                 source.category, source.context_type, source.level, 'ACTIVE',
                 source.content, source.abstract, source.overview,
                 source.relations, source.metadata, source.version,
                 source.created_at, NOW())
            """,
            values,
        )
        return not create_only or cursor.rowcount > 0

    @staticmethod
    def _moved_uri(
        value: str,
        from_uri: str,
        to_uri: str,
        from_prefix: str,
        to_prefix: str,
    ) -> str:
        if value == from_uri:
            return to_uri
        if value.startswith(from_prefix):
            return to_prefix + value[len(from_prefix) :]
        return value

    def rewrite_moved_relations(
        self,
        cursor: Any,
        *,
        from_uri: str,
        to_uri: str,
        from_prefix: str,
        to_prefix: str,
        json_wrapper: Callable[[Any], Any],
    ) -> None:
        cursor.execute(
            "SELECT uri, relations FROM context_nodes WHERE relations::text LIKE %s FOR UPDATE",
            ("%" + from_uri + "%",),
        )
        for node_uri, raw_relations in cursor.fetchall():
            relations = (
                json.loads(raw_relations)
                if isinstance(raw_relations, str)
                else list(raw_relations or [])
            )
            changed = False
            for relation in relations:
                for key in ("from_uri", "to_uri"):
                    old_value = str(relation.get(key, ""))
                    new_value = self._moved_uri(
                        old_value,
                        from_uri,
                        to_uri,
                        from_prefix,
                        to_prefix,
                    )
                    if new_value != old_value:
                        relation[key] = new_value
                        changed = True
            if changed:
                cursor.execute(
                    "UPDATE context_nodes SET relations = %s, updated_at = NOW() WHERE uri = %s",
                    (json_wrapper(relations), node_uri),
                )

    def table_exists(self, cursor: Any, table_name: str) -> bool:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_class AS relation
                WHERE relation.relname = %s
                  AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')
                  AND pg_catalog.pg_table_is_visible(relation.oid)
            )
            """,
            (table_name,),
        )
        row = cursor.fetchone()
        return bool(row and row[0])

    def insert_session_archive(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            """
            MERGE INTO session_archives target
            USING (
                SELECT %s::varchar AS archive_id, %s::varchar AS session_id,
                       %s::varchar AS account_id, %s::text AS abstract,
                       %s::text AS overview, %s::jsonb AS messages,
                       %s::jsonb AS metadata, %s::timestamptz AS created_at,
                       %s::varchar AS owner_user_id,
                       %s::varchar AS owner_agent_id
            ) source
            ON (target.account_id = source.account_id
                AND target.session_id = source.session_id
                AND target.archive_id = source.archive_id)
            WHEN NOT MATCHED THEN INSERT
                (archive_id, session_id, account_id, abstract, overview,
                 messages, metadata, created_at, owner_user_id, owner_agent_id)
            VALUES
                (source.archive_id, source.session_id, source.account_id,
                 source.abstract, source.overview, source.messages,
                 source.metadata, source.created_at, source.owner_user_id,
                 source.owner_agent_id)
            """,
            values,
        )

    def upsert_relation_edge(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            """
            MERGE INTO relation_edges target
            USING (
                SELECT %s::varchar AS from_uri, %s::varchar AS to_uri,
                       %s::varchar AS relation_type,
                       %s::double precision AS weight, %s::text AS reason,
                       %s::varchar AS account_id
            ) source
            ON (target.from_uri = source.from_uri
                AND target.to_uri = source.to_uri
                AND target.relation_type = source.relation_type)
            WHEN MATCHED THEN UPDATE SET
                weight = source.weight, reason = source.reason, updated_at = NOW()
            WHEN NOT MATCHED THEN INSERT
                (from_uri, to_uri, relation_type, weight, reason, account_id)
            VALUES
                (source.from_uri, source.to_uri, source.relation_type,
                 source.weight, source.reason, source.account_id)
            """,
            values,
        )

    def insert_tool_result_artifact(self, cursor: Any, values: tuple[Any, ...]) -> bool:
        cursor.execute(
            """
            MERGE INTO tool_result_artifacts target
            USING (
                SELECT %s::varchar AS account_id, %s::varchar AS session_id,
                       %s::varchar AS owner_user_id,
                       %s::varchar AS owner_agent_id,
                       %s::varchar AS artifact_id, %s::text AS output_text,
                       %s::jsonb AS metadata, %s::varchar AS content_sha256,
                       %s::timestamptz AS created_at
            ) source
            ON (target.account_id = source.account_id
                AND target.session_id = source.session_id
                AND target.owner_user_id = source.owner_user_id
                AND target.owner_agent_id = source.owner_agent_id
                AND target.artifact_id = source.artifact_id)
            WHEN NOT MATCHED THEN INSERT
                (account_id, session_id, owner_user_id, owner_agent_id,
                 artifact_id, output_text, metadata, content_sha256, created_at)
            VALUES
                (source.account_id, source.session_id, source.owner_user_id,
                 source.owner_agent_id, source.artifact_id, source.output_text,
                 source.metadata, source.content_sha256, source.created_at)
            """,
            values,
        )
        return cursor.rowcount > 0

    def insert_audit_log(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            """
            MERGE INTO cp_audit_logs target
            USING (
                SELECT %s::varchar AS log_id, %s::varchar AS account_id,
                       %s::varchar AS actor, %s::varchar AS target_name,
                       %s::varchar AS action, %s::timestamptz AS event_time,
                       %s::varchar AS result, %s::varchar AS trace_id,
                       %s::jsonb AS details
            ) source ON (target.log_id = source.log_id)
            WHEN NOT MATCHED THEN INSERT
                (log_id, account_id, actor, target, action, timestamp,
                 result, trace_id, details)
            VALUES
                (source.log_id, source.account_id, source.actor,
                 source.target_name, source.action, source.event_time,
                 source.result, source.trace_id, source.details)
            """,
            values,
        )

    def notify_outbox(self, cursor: Any, channel: str, payload: str) -> None:
        del cursor, channel, payload

    def insert_trace_event(self, cursor: Any, values: tuple[Any, ...]) -> tuple[bool, Any | None]:
        cursor.execute(
            """
            MERGE INTO trace_events target
            USING (
                SELECT %s::varchar AS event_id, %s::varchar AS uri,
                       %s::varchar AS account_id, %s::varchar AS owner_space,
                       %s::varchar AS agent_id, %s::varchar AS session_id,
                       %s::varchar AS hook_type, %s::timestamptz AS event_time,
                       %s::jsonb AS content, %s::text AS abstract,
                       %s::text AS overview
            ) source
            ON (target.account_id = source.account_id
                AND target.event_id = source.event_id)
            WHEN NOT MATCHED THEN INSERT
                (event_id, uri, account_id, owner_space, agent_id, session_id,
                 hook_type, event_time, content, abstract, overview)
            VALUES
                (source.event_id, source.uri, source.account_id,
                 source.owner_space, source.agent_id, source.session_id,
                 source.hook_type, source.event_time, source.content,
                 source.abstract, source.overview)
            """,
            values,
        )
        return cursor.rowcount > 0, None

    def upsert_trace_session(self, cursor: Any, values: tuple[Any, ...]) -> str | None:
        cursor.execute(
            """
            MERGE INTO trace_sessions target
            USING (
                SELECT %s::varchar AS turn_id, %s::varchar AS uri,
                       %s::varchar AS account_id, %s::varchar AS owner_space,
                       %s::varchar AS agent_id, %s::varchar AS session_id,
                       %s::timestamptz AS started_at,
                       %s::timestamptz AS ended_at,
                       %s::integer AS event_count, %s::text AS abstract,
                       %s::text AS overview, %s::jsonb AS content,
                       %s::jsonb AS metadata, %s::bigint AS stop_event_index,
                       %s::bigint AS prev_stop_event_index
            ) source
            ON (target.account_id = source.account_id
                AND target.turn_id = source.turn_id)
            WHEN MATCHED THEN UPDATE SET
                ended_at = source.ended_at,
                event_count = source.event_count,
                abstract = source.abstract,
                overview = source.overview,
                content = source.content,
                metadata = source.metadata,
                status = 'finalized',
                updated_at = NOW()
            WHEN NOT MATCHED THEN INSERT
                (turn_id, uri, account_id, owner_space, agent_id, session_id,
                 started_at, ended_at, event_count, abstract, overview,
                 content, metadata, status, stop_event_index,
                 prev_stop_event_index)
            VALUES
                (source.turn_id, source.uri, source.account_id,
                 source.owner_space, source.agent_id, source.session_id,
                 source.started_at, source.ended_at, source.event_count,
                 source.abstract, source.overview, source.content,
                 source.metadata, 'finalized', source.stop_event_index,
                 source.prev_stop_event_index)
            """,
            values,
        )
        return None

    def ensure_skill_candidate(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            """
            MERGE INTO skill_candidates target
            USING (
                SELECT %s::varchar AS candidate_id,
                       %s::varchar AS account_id,
                       %s::varchar AS agent_id
            ) source
            ON (target.account_id = source.account_id
                AND target.candidate_id = source.candidate_id)
            WHEN NOT MATCHED THEN INSERT
                (candidate_id, account_id, agent_id, state)
            VALUES
                (source.candidate_id, source.account_id,
                 source.agent_id, 'pending')
            """,
            values,
        )

    def ensure_skill_candidate_evidence(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            """
            MERGE INTO skill_candidate_evidence target
            USING (
                SELECT %s::varchar AS account_id,
                       %s::varchar AS candidate_id,
                       %s::varchar AS evidence_uri,
                       %s::varchar AS session_id
            ) source
            ON (target.account_id = source.account_id
                AND target.candidate_id = source.candidate_id
                AND target.evidence_uri = source.evidence_uri)
            WHEN NOT MATCHED THEN INSERT
                (account_id, candidate_id, evidence_uri,
                 evidence_type, session_id)
            VALUES
                (source.account_id, source.candidate_id, source.evidence_uri,
                 'trace', source.session_id)
            """,
            values,
        )

    def touch_skill_work_account(self, cursor: Any, account_id: str) -> None:
        cursor.execute(
            """
            MERGE INTO skill_work_accounts target
            USING (SELECT %s::varchar AS account_id) source
            ON (target.account_id = source.account_id)
            WHEN MATCHED THEN UPDATE SET updated_at = NOW()
            WHEN NOT MATCHED THEN INSERT (account_id, updated_at)
            VALUES (source.account_id, NOW())
            """,
            (account_id,),
        )

    def insert_skill_example_evidence(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            """
            MERGE INTO skill_example_evidence target
            USING (
                SELECT candidate.account_id, %s AS skill_uri,
                       item->>'example_id' AS example_id,
                       item->>'example_hash' AS example_hash,
                       evidence.evidence_uri, evidence.session_id
                FROM skill_candidates AS candidate
                JOIN skill_candidate_evidence AS evidence
                  ON evidence.account_id = candidate.account_id
                 AND evidence.candidate_id = candidate.candidate_id
                CROSS JOIN LATERAL jsonb_array_elements(
                    COALESCE(
                        candidate.assessment
                            -> 'staged_publication'
                            -> 'example_evidence_keys',
                        '[]'::jsonb
                    )
                ) AS item
                CROSS JOIN LATERAL jsonb_array_elements_text(
                    COALESCE(item->'evidence_uris', '[]'::jsonb)
                ) AS source(evidence_uri)
                WHERE candidate.account_id = %s
                  AND evidence.evidence_uri = ANY(%s)
                  AND evidence.evidence_uri = source.evidence_uri
                  AND candidate.state = 'processing'
                  AND candidate.lease_owner = %s
                  AND candidate.lease_until > NOW()
            ) source
            ON (target.account_id = source.account_id
                AND target.skill_uri = source.skill_uri
                AND target.example_id = source.example_id
                AND target.example_hash = source.example_hash
                AND target.evidence_uri = source.evidence_uri)
            WHEN NOT MATCHED THEN INSERT
                (account_id, skill_uri, example_id, example_hash,
                 evidence_uri, session_id)
            VALUES
                (source.account_id, source.skill_uri, source.example_id,
                 source.example_hash, source.evidence_uri, source.session_id)
            """,
            values,
        )
