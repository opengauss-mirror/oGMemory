"""Type-specific merge instructions must not add a new write failure gate."""
from dataclasses import replace

import pytest

from commit.archive_builder import ArchiveBuilder
from core.models import CandidateMemory
from extraction.schemas.registry import SchemaRegistry


@pytest.mark.parametrize("category", ["entity", "preference"])
def test_card_upgrade_preserves_legacy_overview_facts(category):
    from core.models import RequestContext
    from extraction.card_format import validate_memory_card
    from service.api import MemoryWriteAPI
    from tests.unit.commit.test_memory_identity import MemoryFS
    from tests.unit.server.test_memory_card_recovery import card
    ctx = RequestContext("overview-upgrade", "reader", "agent", "s", "t")
    old = replace(card(), category=category, overview="## Topic\nRadio work.\n- Minimum temperature: -5°C.")
    content = "# Radio work\n\n" + ("## Details\n" if category == "entity" else "") + "- Works on radios.\n- Uses a blue case."
    class Model:
        def complete_json(self, *args, **kwargs):
            return {"content": content}
    fs = MemoryFS()
    api = MemoryWriteAPI(fs, Model(), directory_summary_enabled=False)
    uri = api.write_memory(old, ctx)["target_uri"]
    api._writer._archive_builder._card_format_enabled = True
    incoming = replace(old, overview="- Uses a blue case.", content=content)
    api.write_memory(incoming, ctx)
    stored = fs.read_node(uri, ctx)
    validate_memory_card(stored)
    assert stored.overview == "- ## Topic\n- Radio work.\n- Minimum temperature: -5°C.\n\n- Uses a blue case."
    assert stored.metadata["version"] == 2


class Recorder:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def complete_json(self, prompt, schema):
        self.calls.append(prompt)
        if self.fail:
            raise ValueError("malformed response")
        return {"content": "Rae prefers tea without sugar."}


@pytest.mark.parametrize("category", ["entity", "preference", "pattern"])
def test_merge_uses_actual_configured_yaml_and_keeps_one_normal_call(category):
    registry = SchemaRegistry()
    original = registry.get(category)
    registry.register(replace(original, description="CUSTOM TYPE CONTRACT: retain verified qualifications"))
    llm = Recorder()
    builder = ArchiveBuilder(llm, schema_registry=registry)
    candidate = CandidateMemory(category=category, owner_scope=original.owner_scope,
        routing_key="tea", abstract="Tea", overview="Tea choice", content="Rae prefers unsweetened tea.",
        confidence=.9, who="Rae")
    before = replace(candidate)
    result = builder._merge_content(candidate, {
        "existing_content": "Rae prefers tea.", "content_append": candidate.content})
    assert result == "Rae prefers tea without sugar."
    assert len(llm.calls) == 1
    assert "CUSTOM TYPE CONTRACT" in llm.calls[0]
    assert next(f.description for f in original.fields if f.name == "content") in llm.calls[0]
    assert "Rae prefers tea." in llm.calls[0] and candidate.content in llm.calls[0]
    assert candidate == before


def test_failed_typed_merge_keeps_existing_fallback_and_all_distinct_facts():
    llm = Recorder(fail=True)
    builder = ArchiveBuilder(llm, schema_registry=SchemaRegistry())
    candidate = CandidateMemory(category="entity", owner_scope="user", routing_key="rae",
        abstract="Rae", overview="Rae", content="Rae lives in Bristol.", confidence=.9)
    result = builder._merge_content(candidate, {
        "existing_content": "Rae is an engineer.", "content_append": candidate.content})
    assert result == "Rae is an engineer.\n\nRae lives in Bristol."
    assert len(llm.calls) == 2
    assert all("memory_type_contract" in prompt for prompt in llm.calls)


def test_new_node_content_is_not_sent_for_optional_rewrite():
    llm = Recorder()
    builder = ArchiveBuilder(llm, schema_registry=SchemaRegistry())
    candidate = CandidateMemory(category="entity", owner_scope="user", routing_key="rae",
        abstract="Rae", overview="Rae", content="Rae is an engineer.", confidence=.9)
    assert builder._merge_content(candidate, {}) == candidate.content
    assert llm.calls == []
