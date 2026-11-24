"""The existing query and compose entry points deliver evidence by engine."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

import server.app as app_module
from core.models import RetrievedBlock, SearchMemoryResult
from providers.unified_config import OgMemConfig
from retrieval.evidence.context import render_evidence_context
from server.mcp_tools import query_memory
from server.memory_service import MemoryService
from service.api import ReadAPI
from tests.unit.evidence.test_sources import CTX, message
from tests.unit.evidence.test_writer import write_candidates

QUERY = "What did I visit in Paris?"
QUOTE = "I visited Paris."
LEGACY = RetrievedBlock(
    uri="ctx://account/users/alice/memories/entities/paris/content.md",
    category="entity", owner_space="user:alice", abstract="Paris travel", score=.8,
)


@pytest.fixture
def service(setup):
    fs, _, _, _ = setup
    write_candidates(setup, [message(QUOTE)])
    memory = MemoryService(config=OgMemConfig(extraction_engine="deterministic"))
    memory._get_context_fs = Mock(return_value=fs)
    pipeline = Mock()
    pipeline.run.side_effect = lambda query, *_args, **_kwargs: SearchMemoryResult(
        query=query, hits=[LEGACY],
    )
    api = ReadAPI(pipeline, message_evidence_provider=memory._retrieve_message_evidence)
    memory.get_read_api = Mock(return_value=api)
    return memory


def test_python_query_returns_both_paths_and_respects_scoped_searches(service):
    api = service.get_read_api()
    result = api.search_memory(QUERY, CTX, top_k=1)
    assert result.hits == [LEGACY]
    assert [item["text"] for item in result.message_evidence["evidence"]] == [QUOTE]
    assert [item["quote"] for item in result.message_evidence["facts"]] == [QUOTE]
    for scope in ({"categories": ["entity"]}, {"target_uri": LEGACY.uri}, {"context_type": "RESOURCE"}):
        assert api.search_memory(QUERY, CTX, **scope).message_evidence is None


async def _public_query_payloads(service, monkeypatch, query):
    from mcp import Client

    from server.mcp_server import create_mcp_server

    monkeypatch.setattr(app_module, "_service", service)
    monkeypatch.setattr(app_module, "_build_authenticated_context", lambda params: CTX)
    params = {"query": query, "top_k": 1}
    result = app_module.app.test_client().post("/api/v1/memories/query", json=params)
    assert result.status_code == 200
    payload = result.get_json()
    monkeypatch.setattr("server.mcp_tools.build_mcp_context", lambda *_args, **_kwargs: CTX)
    async with Client(create_mcp_server(service=service, config=service._cfg)) as client:
        response = await client.call_tool("ogm_query_memory", params)
    assert not response.is_error
    return payload, response.structured_content


@pytest.mark.asyncio
async def test_http_and_mcp_query_responses_round_trip_to_renderer(service, monkeypatch):
    payload, mcp = await _public_query_payloads(service, monkeypatch, QUERY)
    assert payload["hits"] == mcp["hits"] and payload["hits"][0]["type"] == "entity"
    assert payload["message_evidence"] == mcp["message_evidence"]
    assert payload["message_evidence"]["evidence"][0]["text"] == QUOTE
    rendered = render_evidence_context(payload)
    assert QUOTE in rendered.text and not rendered.diagnostics["legacy_grouping"]
    assert render_evidence_context(mcp) == rendered
    assert render_evidence_context(mcp["message_evidence"], query=mcp["query"]) == rendered
    assert QUOTE in render_evidence_context(mcp["message_evidence"]).text
    filtered = query_memory(service, query=QUERY, request_context=CTX, types=["entity"])
    assert filtered["hits"] == mcp["hits"] and "message_evidence" not in filtered
    assert render_evidence_context(filtered).text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("expires", [False, True])
async def test_http_and_mcp_only_expose_empty_evidence_when_partial(service, monkeypatch, expires):
    from time import monotonic

    import retrieval.evidence.auxiliary as auxiliary

    fuse_auxiliary = auxiliary.fuse_auxiliary

    def expire_before_fusion(*args, deadline):
        return fuse_auxiliary(*args, deadline=monotonic())

    if expires:
        monkeypatch.setattr(auxiliary, "fuse_auxiliary", expire_before_fusion)

    http, mcp = await _public_query_payloads(service, monkeypatch, "2026-05-01")
    assert http["hits"] == mcp["hits"] and http["hits"][0]["type"] == "entity"
    for payload in (http, mcp):
        if expires:
            assert payload["message_evidence"] == {
                "evidence": [], "facts": [], "groups": [], "partial": True, "truncated": False,
            }
        else:
            assert "message_evidence" not in payload
        assert payload["truncated"] is False
        assert render_evidence_context(payload).text == ""


def test_public_query_rendering_keeps_independent_groups_when_budget_is_tight(service, setup):
    setup[2].register([message("I visited Rome.", id="rome", sequence=12)], CTX, batch_id="rome")
    payload = query_memory(service, query="Where did I visit Paris and Rome?", request_context=CTX)
    evidence = payload["message_evidence"]
    assert len(evidence["groups"]) == 2 and all(len(group) == 1 for group in evidence["groups"])
    full = render_evidence_context(payload)
    bounded = render_evidence_context(payload, max_chars=len(full.text) - 1)
    assert len(full.blocks) == 2 and len(bounded.blocks) == 1


def test_query_preserves_legacy_response_when_evidence_disabled_or_unavailable(service):
    service._cfg.extraction_engine = "llm"
    expected = query_memory(service, query=QUERY, request_context=CTX)
    assert expected["hits"] and "message_evidence" not in expected
    service._get_context_fs.assert_not_called()
    service._cfg.extraction_engine = "deterministic"
    service._get_context_fs.side_effect = TimeoutError("storage")
    actual = query_memory(service, query=QUERY, request_context=CTX)
    assert {k: v for k, v in actual.items() if k != "request_id"} == {
        k: v for k, v in expected.items() if k != "request_id"
    }


def test_account_and_user_scope_and_revocation_apply_to_public_query(service, setup):
    for ctx in (replace(CTX, user_id="other"), replace(CTX, account_id="other")):
        assert "message_evidence" not in query_memory(service, query=QUERY, request_context=ctx)
    setup[2].revoke_session(CTX)
    assert "message_evidence" not in query_memory(service, query=QUERY, request_context=CTX)


def test_compose_injects_source_checked_evidence_in_existing_slot(service):
    from session.models import SessionWindowState

    service._tool_result_externalizer = None
    service._tool_result_governance = None
    service._search_working_set = Mock(return_value=([], [], {}))
    service._read_profile = Mock(return_value="")
    service._collect_archives = Mock(return_value=([], []))
    service._get_session_state = Mock(return_value=SessionWindowState())
    service.get_session_manager().save_session_state = Mock(return_value=True)
    params = {"query": QUERY, "_ctx": CTX}
    result = service.compose(params)
    text = result["retrievedEvidence"]
    assert "## Message Evidence" in text and "untrusted historical data" in text and QUOTE in text
    assert result["messages"][-1] == {"role": "user", "content": text, "_ogmem": True}
    assert "## Message Evidence" not in service.compose({**params, "tokenBudget": 1})["retrievedEvidence"]
    service._cfg.extraction_engine = "llm"
    assert service.compose(params)["retrievedEvidence"] == ""
