from __future__ import annotations

import json

import pytest

from core.errors import AccessDeniedError, NodeNotFoundError
from core.models import (
    ContextNode,
    RequestContext,
    RetrievalConfig,
    RetrievedBlock,
    Role,
    SearchMemoryResult,
)
from providers.unified_config import OgMemConfig
from server.auth import ResolvedIdentity
from server.mcp_tools import (
    delete_memory,
    list_memories,
    list_skill_candidates,
    list_skills,
    query_memory,
    read_memory,
    read_skill,
    search_skills,
)
from skills.artifact import internal_skill_uri, is_governed_skill


class _FakeAuth:
    def role_control_active(self):
        return False


class _FakeStdioAuth:
    def role_control_active(self):
        return True

    def resolve_identity(self, headers):
        assert headers["X-API-Key"] == "member-key"
        return ResolvedIdentity(
            role=Role.MEMBER,
            account_id="acct-1",
            user_id="alice",
        )


class _FakeReadAPI:
    def __init__(self, hits=None, cfg=None):
        self.search_kwargs = None
        self._cfg = cfg or RetrievalConfig()
        self.hits = hits or [
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/preferences/coding_style",
                score=0.9,
                category="preference",
                abstract="Prefers vitest for TypeScript tests.",
                has_content=True,
            )
        ]

    def search_memory(self, query, ctx, **kwargs):
        self.search_kwargs = {"query": query, "ctx": ctx, **kwargs}
        return SearchMemoryResult(
            request_id="req-1",
            query=query,
            hits=self.hits,
        )


class _FakeFS:
    def read_node(self, uri, ctx):
        return ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space="user:alice",
            abstract="Prefers vitest.",
            overview="Testing framework preference.",
            content="Use vitest by default for TypeScript projects.",
            metadata={"version": 2, "updated_at": "2026-06-23T00:00:00Z"},
        )


class _FieldAwareFS(_FakeFS):
    def __init__(self):
        self.read_node_calls = []
        self.read_node_fields_calls = []

    def read_node(self, uri, ctx):
        self.read_node_calls.append(uri)
        return super().read_node(uri, ctx)

    def read_node_fields(self, uri, ctx, level):
        self.read_node_fields_calls.append((uri, level))
        node = super().read_node(uri, ctx)
        return ContextNode(
            uri=node.uri,
            context_type=node.context_type,
            category=node.category,
            level=node.level,
            owner_space=node.owner_space,
            abstract=node.abstract if level in {"abstract", "all"} else "",
            overview=node.overview if level in {"overview", "all"} else "",
            content=node.content if level in {"content", "all"} else "",
            metadata=node.metadata,
        )


class _FallbackPatternFS:
    def read_node(self, uri, ctx):
        if uri == "ctx://acct-1/users/alice/memories/patterns/review":
            raise FileNotFoundError(uri)
        if uri != "ctx://acct-1/agents/bot/memories/patterns/review":
            raise AssertionError(f"unexpected URI: {uri}")
        return ContextNode(
            uri=uri,
            context_type="MEMORY",
            category="pattern",
            level=0,
            owner_space="agent:bot",
            abstract="Legacy agent review pattern.",
            overview="Review pattern overview.",
            content="Use the legacy agent review pattern.",
            metadata={"version": 2, "updated_at": "2026-06-23T00:00:00Z"},
        )


class _FakeService:
    def __init__(self, cfg=None, hits=None):
        self._cfg = cfg or OgMemConfig()
        self.read_api = _FakeReadAPI(
            hits,
            RetrievalConfig(max_top_k=getattr(self._cfg, "retrieval_max_top_k", 50)),
        )
        self.fs = _FakeFS()

    def get_auth_service(self):
        return _FakeAuth()

    @property
    def default_agent_id(self):
        return self._cfg.agent_id

    @property
    def skill_search_scan_limit(self):
        return self._cfg.skill_search_scan_limit

    def build_context(self, params, identity=None):
        return RequestContext(
            account_id="acct-1",
            user_id="alice",
            agent_id=params.get("agentId") or "bot",
            session_id=params.get("sessionId") or "s1",
            trace_id="t1",
        )

    def get_read_api(self):
        return self.read_api

    def _get_context_fs(self):
        return self.fs

class _CollectionFS(_FakeFS):
    def __init__(self):
        self.contexts = []

    def list_children(self, uri, ctx):
        self.contexts.append(ctx)
        if uri.endswith("/memories/preferences"):
            return ["ctx://acct-1/users/alice/memories/preferences/coding_style"]
        return []

    def exists(self, uri, ctx):
        return False


class _CollectionService(_FakeService):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.fs = _CollectionFS()


class _SkillFS(_FakeFS):
    def list_children(self, uri, ctx):
        assert uri == "ctx://acct-1/agents/bot/skills"
        return ["ctx://acct-1/agents/bot/skills/generated/review-code"]

    def read_node(self, uri, ctx):
        return ContextNode(
            uri="ctx://acct-1/agents/bot/skills/generated/review-code",
            context_type="SKILL",
            category="skill",
            level=0,
            owner_space="agent:bot",
            abstract="Review code before production use.",
            overview="Agent skill: review-code",
            content="---\nname: review-code\ndescription: Review code.\n---\n# Review code\n",
            metadata={"version": 1, "ogmemory_generated": True,
                      "publication_status": "published", "review_score": 0.95},
        )


class _SkillService(_FakeService):
    def __init__(self, hits=None):
        super().__init__(
            hits=hits
            or [
                RetrievedBlock(
                    uri="ctx://acct-1/agents/bot/skills/generated/review-code",
                    score=0.95,
                    category="skill",
                    abstract="Review code before production use.",
                )
            ]
        )
        self.fs = _SkillFS()
        self.skill_contexts = []

    def list_published_skills(self, ctx, *, limit=50, cursor=None):
        self.skill_contexts.append(ctx)
        base = f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/skills"
        after = internal_skill_uri(ctx.account_id, ctx.agent_id, cursor) if cursor else None
        if hasattr(self.fs, "list_node_summaries_page"):
            nodes = self.fs.list_node_summaries_page(base, ctx, limit=limit + 1, after_uri=after)
        else:
            uris = sorted(self.fs.list_children(base, ctx))
            nodes = [self.fs.read_node(uri, ctx) for uri in uris if after is None or uri > after]
        rows = [
            {"name": node.uri.rsplit("/", 1)[-1], "description": node.abstract,
             "updated_at": node.metadata.get("updated_at", "")}
            for node in nodes if is_governed_skill(node)
        ]
        payload = {"skills": rows[:limit], "truncated": len(rows) > limit}
        if payload["truncated"]:
            payload["next_cursor"] = rows[limit - 1]["name"]
        return payload

    def read_published_skill(self, ctx, name, source=None):
        self.skill_contexts.append(ctx)
        node = self.fs.read_node(internal_skill_uri(ctx.account_id, ctx.agent_id, name), ctx)
        if not is_governed_skill(node):
            raise FileNotFoundError(name)
        return node

    def list_skill_candidates(self, ctx, state=None):
        self.skill_contexts.append(ctx)
        return [
            {
                "capability_key": "review-code",
                "state": state or "accumulating",
                "evidence_count": 4,
                "distinct_session_count": 2,
                "summary": "Review changes safely.",
                "reason": "Awaiting more evidence.",
            }
        ]


def _authenticated_skill_service(
    *, default_agent_id: str = "main"
) -> tuple[_SkillService, ResolvedIdentity]:
    service = _SkillService()
    service._cfg = OgMemConfig(agent_id=default_agent_id)
    service.get_auth_service = lambda: _FakeStdioAuth()
    service.build_context = lambda params, identity=None: RequestContext(
        account_id=identity.account_id,
        user_id=identity.user_id,
        agent_id=params.get("agentId") or "",
        session_id="s1",
        trace_id="t1",
        role=identity.role,
    )
    identity = ResolvedIdentity(role=Role.MEMBER, account_id="acct-1", user_id="alice")
    return service, identity


def test_query_memory_returns_external_uri_only():
    service = _FakeService()

    payload = query_memory(service, query="testing", types=["preference"], top_k=3)

    assert payload["hits"][0]["uri"] == "ogmemory://memories/preference/coding_style"
    assert "ctx://" not in str(payload)
    assert service.read_api.search_kwargs["categories"] == ["preference"]
    assert service.read_api.search_kwargs["top_k"] == 3


def test_list_memories_returns_first_page_in_native_summary_shape():
    payload = list_memories(_CollectionService())

    assert payload["count"] == 1
    assert payload["memories"] == [
        {
            "uri": "ogmemory://memories/preference/coding_style",
            "type": "preference",
            "id": "coding_style",
            "abstract": "Prefers vitest.",
            "updated_at": "2026-06-23T00:00:00Z",
            "has_content": True,
        }
    ]
    assert payload["truncated"] is False
    assert "untrusted" in payload["untrusted_data_notice"].lower()


def test_list_memories_rejects_limit_too_small_for_one_item():
    max_chars = 250
    service = _CollectionService(OgMemConfig(mcp_max_response_chars=max_chars))

    with pytest.raises(ValueError, match="too small for one ogm_list_memories item"):
        list_memories(service)


def test_list_memories_rejects_limit_smaller_than_response_metadata():
    service = _CollectionService(OgMemConfig(mcp_max_response_chars=1))

    with pytest.raises(ValueError, match="too small for ogm_list_memories metadata"):
        list_memories(service)


def test_list_memories_paginates_without_skipping_response_truncated_nodes():
    nodes = [
        ContextNode(
            uri=f"ctx://acct-1/users/alice/memories/preferences/item-{index:03d}",
            context_type="MEMORY",
            category="preference",
            level=0,
            owner_space="user:alice",
            abstract=f"Preference {index}",
            overview="",
            content="",
            metadata={"has_content": True},
        )
        for index in range(501)
    ]

    class _PagedFS(_CollectionFS):
        def __init__(self):
            super().__init__()
            self.requests = []

        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            self.contexts.append(ctx)
            self.requests.append((uri, limit, after_uri))
            if not uri.endswith("/memories/preferences"):
                return []
            page = [node for node in nodes if after_uri is None or node.uri > after_uri]
            return page[:limit]

        def read_node(self, uri, ctx):
            raise AssertionError("paged listing must not load full node content")

    service = _CollectionService()
    service.fs = _PagedFS()

    payload = list_memories(service, limit=100)

    assert payload["count"] == 100
    assert len(payload["memories"]) == 100
    assert payload["memories"][0]["id"] == "item-000"
    assert payload["memories"][-1]["id"] == "item-099"
    assert payload["next_cursor"] == payload["memories"][-1]["uri"]
    assert payload["truncated"] is True
    preference_requests = [
        request
        for request in service.fs.requests
        if request[0].endswith("/memories/preferences")
    ]
    assert preference_requests == [
        (
            "ctx://acct-1/users/alice/memories/preferences",
            101,
            None,
        )
    ]

    service.fs.requests.clear()
    second = list_memories(service, limit=100, cursor=payload["next_cursor"])

    assert second["memories"][0]["id"] == "item-100"
    assert second["memories"][-1]["id"] == "item-199"
    assert second["count"] == 100
    preference_requests = [
        request
        for request in service.fs.requests
        if request[0].endswith("/memories/preferences")
    ]
    assert preference_requests == [
        (
            "ctx://acct-1/users/alice/memories/preferences",
            101,
            "ctx://acct-1/users/alice/memories/preferences/item-099",
        )
    ]

    capped = list_memories(service, limit=500)

    assert 0 < len(capped["memories"]) < 500
    assert capped["count"] == len(capped["memories"])
    assert capped["next_cursor"] == capped["memories"][-1]["uri"]
    assert len(json.dumps(capped, ensure_ascii=False)) <= 50000

    after_cap = list_memories(service, limit=500, cursor=capped["next_cursor"])
    last_index = int(capped["memories"][-1]["id"].removeprefix("item-"))
    assert after_cap["memories"][0]["id"] == f"item-{last_index + 1:03d}"

    exhausted = list_memories(
        service,
        cursor="ogmemory://memories/preference/item-500",
    )
    assert exhausted["memories"] == []
    assert exhausted["count"] == 0
    assert exhausted["truncated"] is False
    assert "next_cursor" not in exhausted


@pytest.mark.parametrize(
    "names,expected",
    [
        (["atlas", "张三", "李雷", "王五"], ["atlas", "张三", "李雷", "王五"]),
        (["%E6%9D%8E%E9%9B%B7", "atlas", "bravo"], ["李雷", "atlas", "bravo"]),
        (["%E6%9D%8E%E9%9B%B7", "atlas", "李雷"], ["李雷", "atlas", "李雷"]),
        (["%e6%9d%8e%e9%9b%b7", "atlas", "李雷"], ["李雷", "atlas", "李雷"]),
    ],
)
def test_list_memories_paginates_mixed_unicode_names(names, expected):
    nodes = [
        ContextNode(
            uri=f"ctx://acct-1/users/alice/memories/entities/{name}",
            context_type="MEMORY", category="entity", level=0,
            owner_space="user:alice", abstract=name, overview="", content="",
        )
        for name in names
    ]

    class _PagedFS(_CollectionFS):
        def exists(self, uri, ctx):
            return any(node.uri == uri for node in nodes)

        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            if not uri.endswith("/memories/entities"):
                return []
            return [
                node for node in nodes
                if after_uri is None or node.uri > after_uri
            ][:limit]

    service = _CollectionService()
    service.fs = _PagedFS()
    seen = []
    cursor = None
    for _ in names:
        page = list_memories(service, limit=1, cursor=cursor)
        seen.extend(row["id"] for row in page["memories"])
        if not page["truncated"]:
            break
        cursor = page["next_cursor"]
    else:
        pytest.fail("memory pagination did not finish")

    assert seen == expected


def test_list_memories_response_limit_preserves_stored_cursor():
    names = ["%E6%9D%8E%E9%9B%B7", "atlas", "李雷"]
    nodes = [
        ContextNode(
            uri=f"ctx://acct-1/users/alice/memories/entities/{name}",
            context_type="MEMORY", category="entity", level=0,
            owner_space="user:alice", abstract="x" * 120, overview="", content="",
        )
        for name in names
    ]

    class _PagedFS(_CollectionFS):
        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            if not uri.endswith("/memories/entities"):
                return []
            return [node for node in nodes if after_uri is None or node.uri > after_uri][:limit]

    service = _CollectionService()
    service.fs = _PagedFS()
    first_page = list_memories(service, limit=1)
    response_limit = len(json.dumps(first_page, ensure_ascii=False)) + 1
    service._cfg = OgMemConfig(mcp_max_response_chars=response_limit)

    capped = list_memories(service, limit=3)
    assert capped["count"] == 1
    assert capped["next_cursor"] == first_page["next_cursor"]
    second = list_memories(service, limit=3, cursor=capped["next_cursor"])
    assert second["memories"][0]["id"] == "atlas"


def test_list_memories_validates_pagination_arguments():
    service = _CollectionService()

    with pytest.raises(ValueError, match="limit must be between 1 and 500"):
        list_memories(service, limit=0)
    with pytest.raises(ValueError, match="limit must be between 1 and 500"):
        list_memories(service, limit=501)
    with pytest.raises(ValueError, match="cursor must reference"):
        list_memories(service, cursor="ogmemory://memories/preference")


def test_list_memories_merges_storage_pages_in_external_uri_order():
    nodes_by_directory = {
        "/memories/cases": [
            ContextNode(
                uri="ctx://acct-1/users/alice/memories/cases/c1",
                context_type="MEMORY",
                category="case",
                level=0,
                owner_space="user:alice",
                abstract="Case",
                overview="",
                content="",
            )
        ],
        "/users/alice/memories/patterns": [
            ContextNode(
                uri=f"ctx://acct-1/users/alice/memories/patterns/{name}",
                context_type="MEMORY",
                category="pattern",
                level=0,
                owner_space="user:alice",
                abstract=name,
                overview="",
                content="",
            )
            for name in ("aardvark", "zebra")
        ],
        "/agents/bot/memories/patterns": [
            ContextNode(
                uri="ctx://acct-1/agents/bot/memories/patterns/alpha",
                context_type="MEMORY",
                category="pattern",
                level=0,
                owner_space="agent:bot",
                abstract="alpha",
                overview="",
                content="",
            )
        ],
        "/memories/preferences": [
            ContextNode(
                uri="ctx://acct-1/users/alice/memories/preferences/p1",
                context_type="MEMORY",
                category="preference",
                level=0,
                owner_space="user:alice",
                abstract="Preference",
                overview="",
                content="",
            )
        ],
    }

    class _MergedFS(_CollectionFS):
        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            nodes = next(
                (rows for suffix, rows in nodes_by_directory.items() if uri.endswith(suffix)),
                [],
            )
            return [
                node for node in nodes if after_uri is None or node.uri > after_uri
            ][:limit]

    service = _CollectionService()
    service.fs = _MergedFS()

    first = list_memories(service, limit=3, agent_id="bot")
    second = list_memories(
        service,
        limit=3,
        cursor=first["next_cursor"],
        agent_id="bot",
    )

    assert [row["uri"] for row in first["memories"]] == [
        "ogmemory://memories/case/c1",
        "ogmemory://memories/pattern/aardvark",
        "ogmemory://memories/pattern/agent/alpha",
    ]
    assert [row["uri"] for row in second["memories"]] == [
        "ogmemory://memories/pattern/zebra",
        "ogmemory://memories/preference/p1",
    ]
    assert second["truncated"] is False


def test_list_memories_uses_config_default_for_authenticated_stdio(monkeypatch):
    service, _ = _authenticated_skill_service(default_agent_id="bot")
    service.fs = _CollectionFS()
    monkeypatch.setenv("OGMEM_MCP_API_KEY", "member-key")

    list_memories(service)

    assert service.fs.contexts
    assert all(ctx.agent_id == "bot" for ctx in service.fs.contexts)


def test_query_memory_requests_content_fill_for_top_three_hits():
    service = _FakeService()

    query_memory(service, query="testing", top_k=5)

    assert service.read_api.search_kwargs["fill_content_for_top_k"] == 3


def test_query_memory_returns_filled_overview_and_content_excerpt():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/preferences/coding_style",
                score=0.9,
                category="preference",
                abstract="Prefers vitest for TypeScript tests.",
                overview="Testing framework preference.",
                content_excerpt="Use vitest by default for TypeScript projects.",
                has_content=True,
            )
        ],
    )

    payload = query_memory(service, query="testing", top_k=3)

    assert payload["hits"][0]["overview"] == "Testing framework preference."
    assert payload["hits"][0]["content_excerpt"] == "Use vitest by default for TypeScript projects."


def test_query_memory_returns_public_temporal_metadata_fields():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/events/support_group",
                score=0.9,
                category="event",
                abstract="Caroline joined the LGBTQ support group.",
                has_content=True,
                metadata={
                    "when": "2023-05-07",
                    "who": ["Caroline"],
                    "where": "community center",
                    "routing_key": "internal",
                },
            )
        ],
    )

    payload = query_memory(service, query="When did Caroline join the support group?", top_k=3)

    assert payload["hits"][0]["when"] == "2023-05-07"
    assert payload["hits"][0]["who"] == ["Caroline"]
    assert payload["hits"][0]["where"] == "community center"
    assert "routing_key" not in payload["hits"][0]


def test_query_memory_does_not_expose_internal_ranking_score():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/events/support_group",
                score=0.7,
                ranking_score=0.95,
                category="event",
                abstract="Support group note",
            )
        ],
    )

    payload = query_memory(service, query="support group", top_k=1)

    assert "ranking_score" not in payload["hits"][0]


def test_query_memory_temporal_query_overfetches_but_returns_requested_top_k():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri=f"ctx://acct-1/users/alice/memories/events/event_{i}",
                score=1.0 - i / 100,
                category="event",
                abstract=f"Event {i}",
                has_content=True,
            )
            for i in range(15)
        ],
    )

    payload = query_memory(service, query="When did Caroline join the support group?", top_k=5)

    assert service.read_api.search_kwargs["top_k"] == 15
    assert service.read_api.search_kwargs["fill_content_for_top_k"] == 3
    assert len(payload["hits"]) == 5


def test_query_memory_temporal_overfetch_respects_configured_max_top_k():
    service = _FakeService(cfg=OgMemConfig(retrieval_max_top_k=20))

    query_memory(service, query="When did Caroline join the support group?", top_k=20)

    assert service.read_api.search_kwargs["top_k"] == 20


def test_query_memory_non_temporal_query_keeps_requested_top_k():
    service = _FakeService()

    query_memory(service, query="Caroline support group notes", top_k=5)

    assert service.read_api.search_kwargs["top_k"] == 5


def test_skill_tools_use_progressive_disclosure_and_no_trace_payloads():
    service = _SkillService()

    listing = list_skills(service, agent_id="bot")
    detail = read_skill(service, name="review-code", agent_id="bot")
    candidates = list_skill_candidates(service, state="accumulating", agent_id="bot")

    assert listing["skills"] == [
        {
            "name": "review-code",
            "description": "Review code before production use.",
            "uri": "ogmemory://skills/generated/review-code",
            "updated_at": "",
        }
    ]
    assert "content" not in listing["skills"][0]
    assert detail["content"].startswith("---\nname: review-code")
    assert detail["review_status"] == "published"
    assert "trace" not in str(detail).lower()
    assert candidates["candidates"][0]["evidence_count"] == 4
    assert "session_id" not in str(candidates)


def test_skill_tools_keep_source_in_field_for_managed_package():
    service = _SkillService()
    service.list_published_skills = lambda ctx, limit, cursor: {
        "skills": [{
            "name": "Package_One",
            "description": "Imported package.",
            "source": "imported",
            "updated_at": "",
        }],
        "truncated": False,
    }
    service.read_published_skill = lambda ctx, name, source=None: ContextNode(
        uri="ctx://acct-1/agents/bot/skills/imported/Package_One",
        context_type="SKILL",
        category="skill",
        level=0,
        owner_space="agent:bot",
        abstract="Imported package.",
        overview="Agent skill: Package_One",
        content="# Imported\n",
        metadata={"ogmemory_managed": True, "publication_status": "published"},
    )

    listing = list_skills(service, agent_id="bot")
    detail = read_skill(
        service, name="Package_One", source="imported", agent_id="bot"
    )

    assert listing["skills"][0]["uri"] == "ogmemory://skills/imported/Package_One"
    assert listing["skills"][0]["source"] == "imported"
    assert detail["uri"] == "ogmemory://skills/imported/Package_One"
    assert detail["source"] == "imported"


def test_search_skills_uses_only_reviewed_catalog():
    service = _SkillService()

    payload = search_skills(service, query="review changes", top_k=3, agent_id="bot")

    assert payload["skills"][0]["uri"] == "ogmemory://skills/generated/review-code"
    assert service.read_api.search_kwargs is None
    assert "request_id" not in payload


@pytest.mark.parametrize("query", ["zzz", "C", "!!!"])
def test_search_skills_omits_zero_score_rows(query):
    service = _SkillService()

    payload = search_skills(service, query=query, top_k=3, agent_id="bot")

    assert payload["skills"] == []
    assert payload["truncated"] is False


def test_skill_tools_use_transport_agent_header_before_config_default():
    service = _SkillService()
    service.build_context = lambda params, identity=None: RequestContext(
        account_id="acct-1",
        user_id="alice",
        agent_id=params.get("agentId") or "config-default",
        session_id="s1",
        trace_id="t1",
    )
    kwargs = {"transport_headers": {"X-Agent-ID": "bot"}}

    search_skills(service, query="review", **kwargs)
    list_skills(service, **kwargs)
    read_skill(service, name="review-code", **kwargs)
    list_skill_candidates(service, **kwargs)

    assert service.skill_contexts
    assert {ctx.agent_id for ctx in service.skill_contexts} == {"bot"}


def test_skill_tools_use_config_default_for_authenticated_stdio(monkeypatch):
    service, _ = _authenticated_skill_service(default_agent_id="bot")
    monkeypatch.setenv("OGMEM_MCP_API_KEY", "member-key")

    search_skills(service, query="review")
    list_skills(service)
    read_skill(service, name="review-code")
    list_skill_candidates(service)

    assert service.skill_contexts
    assert {ctx.agent_id for ctx in service.skill_contexts} == {"bot"}


def test_read_skill_requires_agent_scope_for_http():
    service, identity = _authenticated_skill_service()

    with pytest.raises(ValueError, match="agent_id is required to read skills"):
        read_skill(
            service,
            name="review-code",
            allow_stdio_env_auth=False,
            resolved_identity=identity,
        )

    assert service.skill_contexts == []


def test_list_skill_candidates_requires_agent_scope_for_http():
    service, identity = _authenticated_skill_service()

    with pytest.raises(ValueError, match="agent_id is required to list skill candidates"):
        list_skill_candidates(
            service,
            allow_stdio_env_auth=False,
            resolved_identity=identity,
        )

    assert service.skill_contexts == []


def test_skill_catalog_hides_legacy_unreviewed_skill_nodes():
    service = _SkillService()
    service.fs.read_node = lambda uri, ctx: ContextNode(
        uri="ctx://acct-1/agents/bot/skills/generated/legacy",
        context_type="SKILL",
        category="skill",
        level=0,
        owner_space="agent:bot",
        abstract="Legacy generic extraction.",
        overview="Legacy generic extraction.",
        content="Do something.",
    )

    assert list_skills(service, agent_id="bot")["skills"] == []


def test_skill_catalog_cursor_paginates_without_loss_or_duplication():
    service = _SkillService()
    nodes = [
        ContextNode(
            uri=f"ctx://acct-1/agents/bot/skills/generated/skill-{index:03d}",
            context_type="SKILL", category="skill", level=0,
            owner_space="agent:bot", abstract=f"Skill {index}", overview="",
            content="skill", metadata={"ogmemory_generated": True,
                                       "publication_status": "published", "review_score": .95},
        )
        for index in range(120)
    ]

    class _PagedFS:
        def list_children(self, uri, ctx):
            return [node.uri for node in nodes]

        def read_node(self, uri, ctx):
            return next(node for node in nodes if node.uri == uri)

        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            page = [node for node in nodes if after_uri is None or node.uri > after_uri]
            return page[:limit]

    service.fs = _PagedFS()
    first = list_skills(service, agent_id="bot", limit=100)
    second = list_skills(
        service, agent_id="bot", limit=100, cursor=first["next_cursor"]
    )

    names = [row["name"] for row in first["skills"] + second["skills"]]
    assert len(names) == 120
    assert len(set(names)) == 120
    assert first["truncated"] is True
    assert second["truncated"] is False


def test_skill_catalog_exact_page_is_not_truncated_without_more_rows():
    service = _SkillService()
    nodes = [
        ContextNode(
            uri=f"ctx://acct-1/agents/bot/skills/generated/skill-{index:03d}",
            context_type="SKILL", category="skill", level=0,
            owner_space="agent:bot", abstract=f"Skill {index}", overview="",
            content="skill", metadata={"ogmemory_generated": True,
                                       "publication_status": "published", "review_score": .95},
        )
        for index in range(100)
    ]

    class _PagedFS:
        def list_children(self, uri, ctx):
            return [node.uri for node in nodes]

        def read_node(self, uri, ctx):
            return next(node for node in nodes if node.uri == uri)

        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            page = [node for node in nodes if after_uri is None or node.uri > after_uri]
            return page[:limit]

    service.fs = _PagedFS()
    payload = list_skills(service, agent_id="bot", limit=100)

    assert len(payload["skills"]) == 100
    assert payload["truncated"] is False
    assert "next_cursor" not in payload


def test_skill_search_enforces_configured_catalog_scan_limit():
    service = _SkillService()
    service._cfg.skill_search_scan_limit = 1
    nodes = [
        ContextNode(
            uri=f"ctx://acct-1/agents/bot/skills/generated/review-{index}",
            context_type="SKILL", category="skill", level=0,
            owner_space="agent:bot", abstract=f"Review {index}", overview="",
            content="skill", metadata={"ogmemory_generated": True,
                                       "publication_status": "published", "review_score": .95},
        )
        for index in range(2)
    ]

    class _FS:
        def list_children(self, uri, ctx):
            return [node.uri for node in nodes]

        def read_node(self, uri, ctx):
            return next(node for node in nodes if node.uri == uri)

    service.fs = _FS()

    payload = search_skills(service, query="review", top_k=2, agent_id="bot")

    assert len(payload["skills"]) == 1
    assert payload["truncated"] is True


def test_skill_search_ranks_chinese_description_for_chinese_query():
    service = _SkillService()
    nodes = [
        ContextNode(
            uri="ctx://acct-1/agents/bot/skills/generated/code-review", context_type="SKILL",
            category="skill", level=0, owner_space="agent:bot",
            abstract="用于代码审查和发布前检查", overview="", content="skill",
            metadata={"ogmemory_generated": True, "publication_status": "published",
                      "review_score": .95},
        ),
        ContextNode(
            uri="ctx://acct-1/agents/bot/skills/generated/data-export", context_type="SKILL",
            category="skill", level=0, owner_space="agent:bot",
            abstract="用于导出数据报表", overview="", content="skill",
            metadata={"ogmemory_generated": True, "publication_status": "published",
                      "review_score": .95},
        ),
    ]

    class _FS:
        def list_node_summaries_page(self, uri, ctx, *, limit, after_uri=None):
            page = [node for node in nodes if after_uri is None or node.uri > after_uri]
            return page[:limit]

        def read_node(self, uri, ctx):
            return next(node for node in nodes if node.uri == uri)

    service.fs = _FS()

    payload = search_skills(service, query="代码审查", top_k=1, agent_id="bot")

    assert payload["skills"][0]["name"] == "code-review"
    assert payload["skills"][0]["score"] > 0


def test_query_memory_returns_session_summary_hits():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/session_summaries/summary_1/content.md",
                score=0.8,
                category="session_summary",
                abstract="Caroline attended an LGBTQ support group.",
                has_content=True,
            )
        ],
    )

    payload = query_memory(service, query="Caroline support group", top_k=3)

    assert payload["hits"][0]["uri"] == "ogmemory://memories/session_summary/summary_1"
    assert payload["hits"][0]["type"] == "session_summary"


def test_query_memory_returns_scoped_uri_for_legacy_agent_pattern_hit():
    service = _FakeService(
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/agents/bot/memories/patterns/review",
                score=0.8,
                category="pattern",
                abstract="Legacy agent review pattern.",
                has_content=True,
            )
        ],
    )

    payload = query_memory(service, query="review", types=["pattern"], top_k=3)

    assert payload["hits"][0]["uri"] == "ogmemory://memories/pattern/agent/review"
    assert payload["hits"][0]["type"] == "pattern"
    assert payload["hits"][0]["id"] == "review"


def test_query_memory_rejects_request_larger_than_configured_limit():
    service = _FakeService(OgMemConfig(mcp_max_request_bytes=16))

    try:
        query_memory(service, query="x" * 64)
    except ValueError as exc:
        assert "mcp_max_request_bytes" in str(exc)
    else:
        raise AssertionError("query_memory should reject oversized requests")


def test_query_memory_applies_configured_response_cap_to_hit_abstracts():
    service = _FakeService(OgMemConfig(mcp_max_response_chars=10))

    payload = query_memory(service, query="testing", types=["preference"], top_k=3)

    assert payload["hits"][0]["abstract"] == "Prefers vi"
    assert payload["truncated"] is True


def test_query_memory_stops_adding_hits_after_response_budget_is_exhausted():
    service = _FakeService(
        OgMemConfig(mcp_max_response_chars=1),
        hits=[
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/preferences/first",
                score=0.9,
                category="preference",
                abstract="first",
                has_content=True,
            ),
            RetrievedBlock(
                uri="ctx://acct-1/users/alice/memories/preferences/second",
                score=0.8,
                category="preference",
                abstract="second",
                has_content=True,
            ),
        ],
    )

    payload = query_memory(service, query="testing", top_k=2)

    assert [hit["id"] for hit in payload["hits"]] == ["first"]
    assert payload["hits"][0]["abstract"] == "f"
    assert payload["truncated"] is True


@pytest.mark.parametrize("budget", [1, 4, 8, 12, 512])
def test_query_memory_bounds_all_summary_fields_after_one_field_is_clipped(budget):
    service = _FakeService(
        OgMemConfig(mcp_max_response_chars=budget),
        hits=[RetrievedBlock(
            uri="ctx://acct-1/users/alice/memories/entities/project",
            score=0.9, category="entity", abstract="项目摘要",
            overview="概览" * 300, content_excerpt="正文🙂" * 300, has_content=True,
        )],
    )

    payload = query_memory(service, query="项目", top_k=1)

    row = payload["hits"][0]
    assert sum(len(row.get(field, "")) for field in ("abstract", "overview", "content_excerpt")) <= budget
    assert row["uri"] == "ogmemory://memories/entity/project"
    assert payload["truncated"] is True


def test_query_memory_rejects_target_uri_style_direct_reads():
    service = _FakeService()

    payload = query_memory(service, query="testing", types=None)

    assert "target_uri" not in service.read_api.search_kwargs
    assert payload["hits"][0]["has_content"] is True


def test_read_memory_reads_external_uri():
    service = _FakeService()

    payload = read_memory(
        service,
        uri="ogmemory://memories/preference/coding_style",
        level="abstract",
    )

    assert payload["uri"] == "ogmemory://memories/preference/coding_style"
    assert payload["abstract"] == "Prefers vitest."
    assert "content" not in payload
    assert payload["truncated"] is False


def test_read_and_delete_memory_find_encoded_legacy_uri():
    stored_uri = "ctx://acct-1/users/alice/memories/preferences/coffee%20shop"
    public_uri = "ogmemory://memories/preference/coffee%20shop"

    class _StoredFS:
        def read_node(self, uri, ctx):
            if uri != stored_uri:
                raise NodeNotFoundError(uri)
            return ContextNode(
                uri=uri, context_type="MEMORY", category="preference", level=0,
                owner_space="user:alice", abstract="Coffee shop", overview="",
                content="Coffee shop", metadata={},
            )

    service = _FakeService()
    service.fs = _StoredFS()
    ctx = RequestContext("acct-1", "alice", "bot", "s1", "t1")
    assert read_memory(service, uri=public_uri, request_context=ctx)["content"] == "Coffee shop"

    deleted = []
    def forget_memory(uri, context, **kwargs):
        if uri != stored_uri:
            raise NodeNotFoundError(uri)
        deleted.append(uri)
        return {"ok": True, "deleted": True}

    service.forget_memory = forget_memory
    assert delete_memory(service, uri=public_uri, request_context=ctx)["deleted"] is True
    assert deleted == [stored_uri]


def test_read_memory_uses_field_level_reader_when_available():
    service = _FakeService()
    service.fs = _FieldAwareFS()

    payload = read_memory(
        service,
        uri="ogmemory://memories/preference/coding_style",
        level="abstract",
    )

    assert payload["abstract"] == "Prefers vitest."
    assert "content" not in payload
    assert service.fs.read_node_fields_calls == [
        ("ctx://acct-1/users/alice/memories/preferences/coding_style", "abstract")
    ]
    assert service.fs.read_node_calls == []


def test_read_memory_returns_canonical_uri_after_legacy_pattern_fallback():
    service = _FakeService()
    service.fs = _FallbackPatternFS()

    payload = read_memory(
        service,
        uri="ogmemory://memories/pattern/review",
        level="content",
    )

    assert payload["uri"] == "ogmemory://memories/pattern/agent/review"
    assert payload["type"] == "pattern"
    assert payload["id"] == "review"
    assert payload["content"] == "Use the legacy agent review pattern."


def test_read_memory_routes_skill_uri_through_governed_catalog():
    service = _SkillService()

    payload = read_memory(
        service,
        uri="ogmemory://skills/review-code",
        level="content",
        agent_id="bot",
    )

    assert payload["uri"] == "ogmemory://skills/generated/review-code"
    assert payload["type"] == "skill"
    assert payload["content"].startswith("---\nname: review-code")


def test_read_memory_routes_imported_skill_file_uri_to_package_reader():
    service = _SkillService()
    calls = []
    service.read_managed_skill_file = lambda ctx, name, path: calls.append(
        (ctx.agent_id, name, path)
    ) or {
        "name": "run.py",
        "path": path,
        "uri": f"ogmemory://skills/imported/{name}/{path}",
        "source": "imported",
        "media_type": "text/x-python",
        "content": "print('ok')\n",
        "encoding": "utf-8",
    }

    payload = read_memory(
        service,
        uri="ogmemory://skills/imported/review-code/scripts/run.py",
        agent_id="bot",
    )

    assert calls == [("bot", "review-code", "scripts/run.py")]
    assert payload["content"] == "print('ok')\n"
    assert payload["untrusted_data_notice"]


def test_read_memory_truncates_imported_binary_skill_file_to_valid_base64():
    service = _SkillService()
    service._cfg = OgMemConfig(mcp_max_response_chars=10)
    service.read_managed_skill_file = lambda ctx, name, path: {
        "uri": f"ogmemory://skills/imported/{name}/{path}",
        "content_base64": "QUJDREVGRw==",
        "encoding": "base64",
    }

    payload = read_memory(
        service,
        uri="ogmemory://skills/imported/review-code/assets/data.bin",
        agent_id="bot",
    )

    assert payload["content_base64"] == "QUJDREVG"
    assert payload["truncated"] is True


def test_read_memory_rejects_legacy_ungoverned_skill_uri():
    service = _SkillService()
    service.fs.read_node = lambda uri, ctx: ContextNode(
        uri="ctx://acct-1/agents/bot/skills/generated/legacy",
        context_type="SKILL",
        category="skill",
        level=0,
        owner_space="agent:bot",
        abstract="Legacy generic extraction.",
        overview="Legacy generic extraction.",
        content="Do something.",
    )

    try:
        read_memory(
            service,
            uri="ogmemory://memories/skill/legacy",
            level="content",
            agent_id="bot",
        )
    except Exception as exc:
        assert "not found" in str(exc).lower() or "visible" in str(exc).lower()
    else:
        raise AssertionError("read_memory should reject ungoverned skill nodes")


def test_read_memory_uses_configured_response_cap():
    service = _FakeService(OgMemConfig(mcp_max_response_chars=5))

    payload = read_memory(
        service,
        uri="ogmemory://memories/preference/coding_style",
        level="content",
        max_chars=100,
    )

    assert payload["content"] == "Use v"
    assert payload["truncated"] is True


def test_read_memory_rejects_internal_uri():
    service = _FakeService()

    try:
        read_memory(service, uri="ctx://acct-1/users/alice/memories/preferences/coding")
    except ValueError as exc:
        assert "ctx://" in str(exc) or "ogmemory" in str(exc)
    else:
        raise AssertionError("read_memory should reject internal ctx:// URIs")


def test_delete_memory_rejects_skill_uri():
    service = _SkillService()

    try:
        delete_memory(
            service,
            uri="ogmemory://skills/generated/review-code",
            agent_id="bot",
        )
    except ValueError as exc:
        assert "skills cannot be deleted" in str(exc)
    else:
        raise AssertionError("delete_memory should reject generated skill URIs")


def test_read_memory_truncates_large_fields():
    service = _FakeService()

    payload = read_memory(
        service,
        uri="ogmemory://memories/preference/coding_style",
        level="content",
        max_chars=10,
    )

    assert payload["content"] == "Use vitest"
    assert payload["truncated"] is True


def test_read_memory_preserves_collection_truncation_flag():
    service = _CollectionService(OgMemConfig(mcp_max_response_chars=5))

    payload = read_memory(
        service,
        uri="ogmemory://memories/preference",
        max_chars=100,
    )

    assert payload["memories"][0]["abstract"] == "Prefe"
    assert payload["truncated"] is True


@pytest.mark.parametrize('failures', [1, 2])
def test_query_connection_retry_is_bounded_and_never_becomes_no_hits(monkeypatch, failures):
    import psycopg2
    service = _FakeService()
    original = service.read_api.search_memory
    calls = []
    def search(*args, **kwargs):
        calls.append(True)
        if len(calls) <= failures:
            raise psycopg2.OperationalError('server closed the connection unexpectedly')
        return original(*args, **kwargs)
    monkeypatch.setattr(service.read_api, 'search_memory', search)
    if failures == 1:
        assert query_memory(service, query='testing preference')['hits']
    else:
        with pytest.raises(RuntimeError, match='database connection unavailable after one retry'):
            query_memory(service, query='testing preference')
    assert len(calls) == 2


def test_query_does_not_retry_statement_errors(monkeypatch):
    import psycopg2
    from unittest.mock import Mock
    service = _FakeService()
    search = Mock(side_effect=psycopg2.ProgrammingError('invalid SQL'))
    monkeypatch.setattr(service.read_api, 'search_memory', search)
    with pytest.raises(psycopg2.ProgrammingError):
        query_memory(service, query='testing preference')
    assert search.call_count == 1


def test_query_never_retries_inside_session_mutation(monkeypatch):
    import psycopg2
    from unittest.mock import Mock
    service = _FakeService()
    search = Mock(side_effect=psycopg2.OperationalError('owning connection lost'))
    monkeypatch.setattr(service.read_api, 'search_memory', search)
    monkeypatch.setattr('fs.sql_adapter.session_mutation.active_session_guard', lambda: object())
    with pytest.raises(psycopg2.OperationalError):
        query_memory(service, query='testing preference')
    assert search.call_count == 1
