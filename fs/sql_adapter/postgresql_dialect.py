"""PostgreSQL SQL-storage dialect."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fs.sql_adapter.dialect import POSTGRESQL, SQLDialect


class PostgreSQLDialect(SQLDialect):
    name = POSTGRESQL

    def ensure_extraction_work_account(self, cursor: Any, account_id: str) -> None:
        cursor.execute("INSERT INTO extraction_work_accounts (account_id) VALUES (%s) "
                       "ON CONFLICT (account_id) DO NOTHING", (account_id,))

    def ensure_workspace_root(self, cursor: Any, account_id: str, root_id: str) -> None:
        cursor.execute(
            "INSERT INTO code_workspace_sync_roots (account_id, root_id) "
            "VALUES (%s, %s) ON CONFLICT (account_id, root_id) DO NOTHING",
            (account_id, root_id),
        )

    def write_context_node(
        self, cursor: Any, values: tuple[Any, ...], *, create_only: bool
    ) -> bool:
        if create_only:
            cursor.execute(
                """
                INSERT INTO context_nodes
                    (uri, account_id, owner_space, category, context_type,
                     level, status, content, abstract, overview,
                     relations, metadata, version, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE', %s, %s, %s,
                        %s, %s, %s, %s, NOW())
                ON CONFLICT (uri) DO NOTHING
                """,
                values,
            )
            return cursor.rowcount > 0

        cursor.execute(
            """
            INSERT INTO context_nodes
                (uri, account_id, owner_space, category, context_type,
                 level, status, content, abstract, overview,
                 relations, metadata, version, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE', %s, %s, %s,
                    %s, %s, %s, %s, NOW())
            ON CONFLICT (uri) DO UPDATE SET
                status       = 'ACTIVE',
                content      = EXCLUDED.content,
                abstract     = EXCLUDED.abstract,
                overview     = EXCLUDED.overview,
                relations    = EXCLUDED.relations,
                metadata     = EXCLUDED.metadata,
                version      = EXCLUDED.version,
                updated_at   = NOW()
            """,
            values,
        )
        return True

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
        del json_wrapper
        cursor.execute(
            """
            UPDATE context_nodes
            SET relations = (
                    SELECT COALESCE(jsonb_agg(
                        jsonb_set(
                            jsonb_set(elem,
                                '{from_uri}',
                                CASE
                                    WHEN elem->>'from_uri' = %s
                                        THEN to_jsonb(%s::text)
                                    WHEN elem->>'from_uri' LIKE %s
                                        THEN to_jsonb(concat(%s,
                                            substring(elem->>'from_uri' from %s)))
                                    ELSE elem->'from_uri'
                                END
                            ),
                            '{to_uri}',
                            CASE
                                WHEN elem->>'to_uri' = %s
                                    THEN to_jsonb(%s::text)
                                WHEN elem->>'to_uri' LIKE %s
                                    THEN to_jsonb(concat(%s,
                                        substring(elem->>'to_uri' from %s)))
                                ELSE elem->'to_uri'
                            END
                        )
                    ), '[]'::jsonb)
                    FROM jsonb_array_elements(relations) AS elem
                ),
                updated_at = NOW()
            WHERE relations::text LIKE %s
            """,
            (
                from_uri,
                to_uri,
                from_prefix + "%",
                to_prefix,
                len(from_prefix) + 1,
                from_uri,
                to_uri,
                from_prefix + "%",
                to_prefix,
                len(from_prefix) + 1,
                "%" + from_uri + "%",
            ),
        )

    def table_exists(self, cursor: Any, table_name: str) -> bool:
        cursor.execute("SELECT to_regclass(%s)", (table_name,))
        row = cursor.fetchone()
        return bool(row and row[0] is not None)

    def insert_session_archive(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            """
            INSERT INTO session_archives
                (archive_id, session_id, account_id, abstract, overview,
                 messages, metadata, created_at, owner_user_id, owner_agent_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (account_id, session_id, archive_id) DO NOTHING
            """,
            values,
        )

    def upsert_relation_edge(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            """
            INSERT INTO relation_edges
                (from_uri, to_uri, relation_type, weight, reason, account_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (from_uri, to_uri, relation_type) DO UPDATE SET
                weight = EXCLUDED.weight,
                reason = EXCLUDED.reason,
                updated_at = NOW()
            """,
            values,
        )

    def insert_tool_result_artifact(self, cursor: Any, values: tuple[Any, ...]) -> bool:
        cursor.execute(
            """
            INSERT INTO tool_result_artifacts
                (account_id, session_id, owner_user_id, owner_agent_id,
                 artifact_id, output_text, metadata, content_sha256, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT
                (account_id, session_id, owner_user_id, owner_agent_id, artifact_id)
            DO NOTHING
            RETURNING artifact_id
            """,
            values,
        )
        return cursor.fetchone() is not None

    def insert_audit_log(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            "INSERT INTO cp_audit_logs "
            "(log_id, account_id, actor, target, action, timestamp, result, "
            " trace_id, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (log_id) DO NOTHING",
            values,
        )

    def notify_outbox(self, cursor: Any, channel: str, payload: str) -> None:
        cursor.execute("SELECT pg_notify(%s, %s)", (channel, payload))

    def insert_trace_event(self, cursor: Any, values: tuple[Any, ...]) -> tuple[bool, Any | None]:
        cursor.execute(
            """
            INSERT INTO trace_events
                (event_id, uri, account_id, owner_space, agent_id, session_id,
                 hook_type, event_time, content, abstract, overview)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (account_id, event_id) DO NOTHING
            RETURNING event_id, uri, account_id, owner_space, agent_id,
                      session_id, hook_type, event_index, event_time,
                      content, abstract, overview
            """,
            values,
        )
        row = cursor.fetchone()
        return row is not None, row

    def upsert_trace_session(self, cursor: Any, values: tuple[Any, ...]) -> str | None:
        cursor.execute(
            """
            INSERT INTO trace_sessions
                (turn_id, uri, account_id, owner_space, agent_id, session_id,
                 started_at, ended_at, event_count, abstract, overview,
                 content, metadata, status, stop_event_index,
                 prev_stop_event_index)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, 'finalized', %s, %s)
            ON CONFLICT (account_id, turn_id) DO UPDATE SET
                ended_at = EXCLUDED.ended_at,
                event_count = EXCLUDED.event_count,
                abstract = EXCLUDED.abstract,
                overview = EXCLUDED.overview,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                status = 'finalized',
                updated_at = NOW()
            RETURNING uri
            """,
            values,
        )
        row = cursor.fetchone()
        return str(row[0]) if row else None

    def ensure_skill_candidate(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            """
            INSERT INTO skill_candidates
                (candidate_id, account_id, agent_id, state)
            VALUES (%s, %s, %s, 'pending')
            ON CONFLICT (account_id, candidate_id) DO NOTHING
            """,
            values,
        )

    def ensure_skill_candidate_evidence(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            """
            INSERT INTO skill_candidate_evidence
                (account_id, candidate_id, evidence_uri, evidence_type, session_id)
            VALUES (%s, %s, %s, 'trace', %s)
            ON CONFLICT (account_id, candidate_id, evidence_uri) DO NOTHING
            """,
            values,
        )

    def touch_skill_work_account(self, cursor: Any, account_id: str) -> None:
        cursor.execute(
            """
            INSERT INTO skill_work_accounts (account_id, updated_at)
            VALUES (%s, NOW())
            ON CONFLICT (account_id) DO UPDATE SET updated_at = EXCLUDED.updated_at
            """,
            (account_id,),
        )

    def insert_skill_example_evidence(self, cursor: Any, values: tuple[Any, ...]) -> None:
        cursor.execute(
            """
            INSERT INTO skill_example_evidence (
                account_id, skill_uri, example_id, example_hash,
                evidence_uri, session_id
            )
            SELECT candidate.account_id, %s,
                   item->>'example_id', item->>'example_hash',
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
            ON CONFLICT (
                account_id, skill_uri, example_id, example_hash, evidence_uri
            ) DO NOTHING
            """,
            values,
        )
