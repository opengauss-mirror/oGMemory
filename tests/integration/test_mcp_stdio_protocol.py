from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import anyio
import psycopg2
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from fs.sql_adapter.sql_trace_store import SQLTraceEventStore

ROOT = Path(__file__).resolve().parents[2]
_ENV_PASSTHROUGH = {
    "COMSPEC",
    "HOME",
    "PATH",
    "PATHEXT",
    "PYTHONPATH",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "VIRTUAL_ENV",
}


@pytest.mark.integration
def test_mcp_stdio_protocol_exposes_tools_and_resources():
    anyio.run(_run_stdio_protocol_check)


async def _run_stdio_protocol_check() -> None:
    env = _clean_subprocess_env()
    sql_dsn = _required_sql_dsn()
    account_id = "acct-protocol-stdio"
    env.update(
        {
            "VECTOR_DB_TYPE": "memory",
            "OGMEM_MCP_ENABLED": "true",
            "OGMEM_MCP_DEFAULT_TOOLSET": "all",
            "OGMEM_MCP_ENABLE_WRITE_TOOLS": "true",
            "OGMEM_MCP_ENABLE_DELETE_TOOLS": "true",
            "OGMEM_CONFIG": str(ROOT / "__missing_mcp_test_config__.yaml"),
            "OGMEM_SESSION_STATE_BRIDGE_ENABLED": "false",
            "OG_ACCOUNT_ID": account_id,
            "OG_USER_ID": "root",
            "OG_AGENT_ID": "main",
        }
    )
    env.update({"STORAGE_BACKEND": "sql", "SQL_CONNECTION_STRING": sql_dsn})
    _seed_reviewed_skill(sql_dsn, account_id=account_id, agent_id="main")
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "cli.main", "mcp", "--transport", "stdio"],
        cwd=str(ROOT),
        env=env,
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            assert init.server_info.name == "oGMemory"

            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            assert tool_names == {
                "ogm_list_memories",
                "ogm_query_memory",
                "ogm_read_memory",
                "ogm_search_code",
                "ogm_search_skills",
                "ogm_list_skills",
                "ogm_read_skill",
                "ogm_list_skill_candidates",
                "ogm_install_skill",
                "ogm_update_skill",
                "ogm_delete_skill",
                "ogm_write_memory",
                "ogm_delete_memory",
            }
            assert all("." not in name for name in tool_names)

            read_result = await session.call_tool("ogm_read_memory", {"uri": "ogmemory://types"})
            read_payload = _tool_json(read_result)
            assert "types" in read_payload

            query_result = await session.call_tool("ogm_query_memory", {"query": "anything", "top_k": 1})
            query_payload = _tool_json(query_result)
            assert query_payload["hits"] == []

            list_result = await session.call_tool("ogm_list_memories", {})
            list_payload = _tool_json(list_result)
            assert list_payload["count"] == len(list_payload["memories"])

            legacy_calls = (
                ("ogm.query_memory", {"query": "legacy alias must be rejected", "top_k": 1}),
                ("ogm.read_memory", {"uri": "ogmemory://types"}),
                (
                    "ogm.write_memory",
                    {
                        "session_id": "legacy-name-rejection",
                        "text": "legacy alias must be rejected",
                        "wait": False,
                        "client_request_id": "mcp-e2e-legacy-write-1",
                    },
                ),
                (
                    "ogm.delete_memory",
                    {
                        "uri": "ogmemory://memories/preference/legacy-name-rejection",
                        "reason": "legacy alias must be rejected",
                        "client_request_id": "mcp-e2e-legacy-delete-1",
                    },
                ),
            )
            for tool_name, arguments in legacy_calls:
                legacy_result = await session.call_tool(tool_name, arguments)
                assert legacy_result.is_error is True
                error_text = "\n".join(
                    content.text
                    for content in legacy_result.content
                    if getattr(content, "type", None) == "text"
                )
                assert "Unknown tool" in error_text
                assert tool_name in error_text

            listed = _tool_json(await session.call_tool("ogm_list_skills", {}))
            searched = _tool_json(await session.call_tool(
                "ogm_search_skills", {"query": "protocol review", "top_k": 1}
            ))
            read_skill = _tool_json(await session.call_tool(
                "ogm_read_skill", {"name": "protocol-review"}
            ))
            candidates = _tool_json(await session.call_tool(
                "ogm_list_skill_candidates", {}
            ))
            assert any(row["name"] == "protocol-review" for row in listed["skills"])
            assert searched["skills"][0]["name"] == "protocol-review"
            assert read_skill["content"].startswith("---\nname: protocol-review")
            assert any(
                row["capability_key"] == "protocol-review-candidate"
                for row in candidates["candidates"]
            )

            resources = await session.list_resources()
            resource_uris = {str(resource.uri) for resource in resources.resources}
            assert {
                "ogmemory://status", "ogmemory://types", "ogmemory://skills",
                "ogmemory://skill-candidates",
            } <= resource_uris

            templates = await session.list_resource_templates()
            template_uris = {
                template.uri_template for template in templates.resource_templates
            }
            assert "ogmemory://memories/{memory_type}" in template_uris
            assert "ogmemory://memories/{memory_type}/{memory_id}" in template_uris
            assert "ogmemory://memories/pattern/agent/{memory_id}" in template_uris
            assert "ogmemory://skills/{name}" in template_uris

            status_payload = _resource_json(await session.read_resource("ogmemory://status"))
            types_payload = _resource_json(await session.read_resource("ogmemory://types"))
            skills_payload = _resource_json(
                await session.read_resource("ogmemory://skills")
            )
            skill_payload = _resource_json(
                await session.read_resource("ogmemory://skills/protocol-review")
            )
            candidate_payload = _resource_json(
                await session.read_resource("ogmemory://skill-candidates")
            )
            assert status_payload["status"] == "ok"
            assert "types" in types_payload
            protocol_skill = next(
                row for row in skills_payload["skills"] if row["name"] == "protocol-review"
            )
            assert protocol_skill["uri"] == "ogmemory://skills/generated/protocol-review"
            assert skill_payload["name"] == "protocol-review"
            assert skill_payload["content"].startswith("---\nname: protocol-review")
            protocol_candidate = next(
                row for row in candidate_payload["candidates"]
                if row["capability_key"] == "protocol-review-candidate"
            )
            assert protocol_candidate == {
                "capability_key": "protocol-review-candidate",
                "state": "accumulating",
                "evidence_count": 1,
                "distinct_session_count": 1,
                "summary": "Safe candidate summary.",
                "reason": "Awaiting more evidence.",
            }
            _assert_no_trace_provenance(
                skills_payload, skill_payload, candidate_payload
            )


def _clean_subprocess_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _ENV_PASSTHROUGH
    }


def _required_sql_dsn() -> str:
    dsn = os.environ.get("OGMEM_TEST_SQL_DSN")
    if not dsn:
        pytest.fail(
            "OGMEM_TEST_SQL_DSN is required; protocol acceptance must exercise "
            "the DB-backed skill tools and may not silently downgrade"
        )
    return dsn


def _tool_json(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    content = result.content[0]
    return json.loads(content.text)


def _resource_json(result: Any) -> dict[str, Any]:
    content = result.contents[0]
    return json.loads(content.text)


def _assert_no_trace_provenance(*payloads: dict[str, Any]) -> None:
    serialized = json.dumps(payloads, ensure_ascii=False).lower()
    assert "raw_trace_secret" not in serialized
    assert "secret-protocol-session" not in serialized
    assert '"trace_id"' not in serialized
    assert '"session_id"' not in serialized


def _seed_reviewed_skill(dsn: str, *, account_id: str, agent_id: str) -> None:
    SQLTraceEventStore(connection_string=dsn, pool_size=1)
    candidate_id = f"protocol-candidate-{account_id}"
    uri = f"ctx://{account_id}/agents/{agent_id}/skills/protocol-review"
    content = "---\nname: protocol-review\ndescription: Review protocol behavior.\n---\n# Protocol review\n"
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET app.account_id = %s", (account_id,))
            cur.execute(
                "DELETE FROM trace_sessions WHERE account_id = %s AND turn_id = %s",
                (account_id, "protocol-candidate-turn"),
            )
            cur.execute(
                "DELETE FROM skill_candidates WHERE account_id = %s AND candidate_id = %s",
                (account_id, candidate_id),
            )
            cur.execute("DELETE FROM context_nodes WHERE uri = %s", (uri,))
            cur.execute(
                """
                INSERT INTO context_nodes
                    (uri, account_id, owner_space, category, context_type, level,
                     status, content, abstract, overview, relations, metadata, version)
                VALUES (%s, %s, %s, 'skill', 'SKILL', 0, 'ACTIVE', %s, %s, %s,
                        '[]'::jsonb, %s::jsonb, 1)
                """,
                (uri, account_id, f"agent:{agent_id}", content,
                 "Review protocol behavior.", "Agent skill: protocol-review",
                 json.dumps({"ogmemory_generated": True,
                             "publication_status": "published", "review_score": .95})),
            )
            cur.execute(
                """
                INSERT INTO trace_sessions
                    (turn_id, uri, account_id, owner_space, agent_id, session_id,
                     started_at, ended_at, event_count, abstract, overview,
                     content, metadata, status, stop_event_index)
                VALUES
                    ('protocol-candidate-turn', %s, %s, %s, %s,
                     'secret-protocol-session', NOW() - INTERVAL '1 minute', NOW(),
                     1, '', '', %s::jsonb, '{}'::jsonb, 'finalized', 1)
                """,
                (
                    f"ctx://{account_id}/agents/{agent_id}/traces/protocol-candidate-turn",
                    account_id,
                    f"agent:{agent_id}",
                    agent_id,
                    json.dumps([{"event": "RAW_TRACE_SECRET"}]),
                ),
            )
            cur.execute(
                """
                INSERT INTO skill_candidates
                    (candidate_id, account_id, agent_id, capability_key,
                     state, assessment)
                VALUES (%s, %s, %s, 'protocol-review-candidate',
                        'assessed', %s::jsonb)
                """,
                (
                    candidate_id,
                    account_id,
                    agent_id,
                    json.dumps(
                        {
                            "summary": "Safe candidate summary.",
                            "reason": "Awaiting more evidence.",
                        }
                    ),
                ),
            )
            cur.execute(
                """
                INSERT INTO skill_candidate_evidence
                    (account_id, candidate_id, evidence_uri, evidence_type, session_id)
                VALUES (%s, %s, %s, 'trace', 'secret-protocol-session')
                """,
                (
                    account_id,
                    candidate_id,
                    f"ctx://{account_id}/agents/{agent_id}/traces/protocol-candidate-turn",
                ),
            )
        conn.commit()
    finally:
        conn.close()
