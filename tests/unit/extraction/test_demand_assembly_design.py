"""Acceptance boundaries for category-only YAML assembly."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from core.errors import LLMOutputTruncatedError, ToolCallProtocolError
from extraction.prompt_assembly import build_selection_request, run_selection_request
from extraction.prompts import PromptManager


def template_data():
    return {
        "assembly_version": 1,
        "selection": {
            "instruction": "Choose categories only.",
            "types": {"entity": "Other subjects", "profile": "Current user's own identity"},
            "classification_rules": "Use source facts.",
            "input": "INPUT:\n{{ runtime_input_json }}",
        },
        "system_prompt": [
            {"id": "common", "text": "COMMON {{ name }}\n"},
            {"id": "entity", "types": ["entity"], "text": "ENTITY\n"},
            {"id": "profile", "types": ["profile"], "text": "PROFILE\n"},
        ],
        "examples": [
            {"id": "common", "text": "PUBLIC\n"},
            {"id": "both", "requires_types": ["profile", "entity"], "text": "BOTH\n"},
        ],
        "output_instruction": "Return memory.",
    }


@pytest.fixture
def manager(tmp_path):
    (tmp_path / "extraction.yaml").write_text(yaml.safe_dump(template_data()), encoding="utf-8")
    return PromptManager(tmp_path)


@pytest.fixture
def catalog():
    schemas = [
        SimpleNamespace(
            memory_type=n, tool_name="extract_" + n, description="DO NOT COPY TOOL DESCRIPTION"
        )
        for n in ("entity", "profile")
    ]
    registry = SimpleNamespace(list_enabled=lambda: schemas)
    tools = [
        {"name": s.tool_name, "description": s.description, "input_schema": {}} for s in schemas
    ]
    return registry, tools


def test_one_render_selects_before_rendering_and_keeps_unlabelled_blocks(manager):
    assert (
        manager.render("extraction", "system_prompt", categories={"entity"}, name="Ada")
        == "COMMON Ada\nENTITY\n"
    )
    assert manager.render("extraction", "examples", categories={"entity"}) == "PUBLIC\n"
    assert (
        manager.render("extraction", "examples", categories={"entity", "profile"})
        == "PUBLIC\nBOTH\n"
    )
    assert (
        manager.render("extraction", "system_prompt", name="Ada") == "COMMON Ada\nENTITY\nPROFILE\n"
    )


def test_selection_prompt_is_exact_yaml_render_and_preserves_literal_source(manager, catalog):
    registry, tools = catalog
    source = 'user: literal {{ secret }} and "quoted"'
    request = build_selection_request(registry, tools, source, "user-a", manager=manager)
    runtime = {
        "current_user": "user-a",
        "catalog": ["entity", "profile"],
        "source": source,
        "tool_statistics": "",
    }
    assert request.prompt == manager.render(
        "extraction", "selection", runtime_input_json=json.dumps(runtime, ensure_ascii=False)
    )
    assert "{{ secret }}" in request.prompt
    assert "DO NOT COPY TOOL DESCRIPTION" not in request.prompt
    assert set(request.schema["properties"]) == {"categories"}


@pytest.mark.parametrize(
    "bad,kind",
    [
        ({}, "schema_validation"),
        ({"categories": ["unknown"]}, "schema_validation"),
        ({"categories": ["entity", "entity"]}, "schema_validation"),
        ({"categories": "entity"}, "schema_validation"),
        ({"categories": [], "shared_tools": []}, "schema_validation"),
        (ValueError("invalid JSON"), "invalid_json"),
        (ToolCallProtocolError("invalid JSON"), "invalid_json"),
        (LLMOutputTruncatedError("length"), "truncated"),
    ],
)
def test_invalid_response_has_one_yaml_feedback_retry(manager, catalog, bad, kind):
    request = build_selection_request(*catalog, "user: project", "user-a", manager=manager)
    llm = SimpleNamespace(complete_json=Mock(side_effect=[bad, {"categories": ["entity"]}]))
    result = run_selection_request(llm, request)
    assert result.scope == "scoped" and result.categories == frozenset({"entity"})
    assert llm.complete_json.call_count == 2
    calls = llm.complete_json.call_args_list
    first = json.loads(calls[0].args[0].split("INPUT:\n", 1)[1])
    second = json.loads(calls[1].args[0].split("INPUT:\n", 1)[1])
    feedback = second.pop("validation_feedback")
    assert feedback["kind"] == kind
    assert first == second
    assert calls[0].kwargs["schema"] == calls[1].kwargs["schema"]


def test_second_invalid_response_falls_back_without_third_call(manager, catalog):
    request = build_selection_request(*catalog, "user: project", "user-a", manager=manager)
    llm = SimpleNamespace(complete_json=Mock(side_effect=[{}, {"categories": ["unknown"]}]))
    result = run_selection_request(llm, request)
    assert result.scope == "full" and result.categories == frozenset({"entity", "profile"})
    assert llm.complete_json.call_count == 2


def test_legal_empty_does_not_retry(manager, catalog):
    request = build_selection_request(*catalog, "thanks", "user-a", manager=manager)
    llm = SimpleNamespace(complete_json=Mock(return_value={"categories": []}))
    assert run_selection_request(llm, request).scope == "no_task"
    llm.complete_json.assert_called_once()


@pytest.mark.parametrize("field", ["instruction", "types", "classification_rules", "input"])
def test_missing_selection_field_fails_before_cache_and_names_location(tmp_path, field):
    data = template_data()
    data["selection"].pop(field)
    (tmp_path / "extraction.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    manager = PromptManager(tmp_path)
    with pytest.raises(ValueError, match="extraction/selection." + field):
        manager.load("extraction")
    assert "extraction" not in manager._cache


def test_defined_type_without_specific_block_remains_supported(manager):
    data = manager.load("extraction")
    data["system_prompt"] = [data["system_prompt"][0]]
    assert (
        manager.render("extraction", "system_prompt", categories={"entity"}, name="Ada")
        == "COMMON Ada\n"
    )


def test_extractor_rejects_manager_without_category_render_contract():
    from extraction.tools import Extractor

    class Legacy:
        def render(self, template_name, section):
            return "legacy"

    with pytest.raises(ValueError, match="render"):
        Extractor(Mock(), prompt_manager=Legacy())


def test_compliant_render_override_keeps_custom_output_and_selected_rules():
    class Custom(PromptManager):
        def render(self, template_name, section, *, categories=None, **variables):
            text = super().render(template_name, section, categories=categories, **variables)
            return text + "\nCUSTOM OUTPUT" if section == "output_instruction" else text

    from core.models import RequestContext
    from extraction.prompt_assembly import assemble_prepared
    from extraction.tools import Extractor, _Span

    llm = Mock()
    llm.complete_json.return_value = {"categories": ["entity"]}
    ctx = RequestContext("a", "u", "agent", "s", "t")
    extractor = Extractor(llm, prompt_manager=Custom())
    prepared = extractor._prepare_span(
        _Span(0, 0), [{"role": "user", "content": "Project uses SQL"}], "English", None, "", "", ctx
    )
    request = assemble_prepared(extractor, prepared, ctx)
    assert request.contract.scope == "scoped"
    assert "CUSTOM OUTPUT" in request.prompt
    assert "## entity" in request.prompt
    assert "── profile (" not in request.prompt


def test_lazy_assembly_snapshot_is_the_complete_sent_tool_list():
    from core.models import RequestContext
    from core.uri_resolver import URIResolver
    from extraction.prompt_assembly import assemble_prepared
    from extraction.schemas.registry import SchemaRegistry
    from extraction.tools import Extractor, _Span

    registry, fs, llm = SchemaRegistry(), Mock(), Mock()
    llm.complete_json.return_value = {"categories": ["entity"]}
    llm.complete_with_tools_messages.return_value = ([], "[]")
    ctx = RequestContext("a", "u", "agent", "s", "t")
    extractor = Extractor(
        llm, mode="lazy", fs=fs, schema_registry=registry, uri_resolver=URIResolver(registry)
    )
    prepared = extractor._prepare_span(
        _Span(0, 0), [{"role": "user", "content": "Project uses SQL"}], "English", None, "", "", ctx
    )
    request = assemble_prepared(extractor, prepared, ctx)
    assert request.contract.tool_names == tuple(t["name"] for t in request.tools)
    assert {
        "read",
        "list",
        "get_relations",
        "get_access_stats",
        "extract_entity",
        "edit_memory",
        "append_memory",
    } == set(request.contract.tool_names)
    extractor._structure_prepared_span(prepared, ctx)
    assert llm.complete_with_tools_messages.call_args.kwargs["tools"] == request.tools


def test_yaml_merge_keeps_explicit_override_semantics(tmp_path):
    template = template_data()
    template.pop("system_prompt")
    data = yaml.safe_dump(template, sort_keys=False)
    data += "\nbase: &base\n  id: inherited\n  text: INHERITED\n"
    data += "system_prompt:\n- <<: *base\n  id: caller\n  text: CALLER\n"
    (tmp_path / "extraction.yaml").write_text(data)
    manager = PromptManager(tmp_path)
    assert manager.render("extraction", "system_prompt", categories={"entity"}) == "CALLER"
    assert manager.load("extraction")["system_prompt"][0]["id"] == "caller"


@pytest.mark.parametrize("version", [None, 2, "caller metadata"])
def test_version_and_unrelated_metadata_do_not_expand_validation(tmp_path, version):
    data = template_data()
    data.pop("assembly_version")
    if version is not None:
        data["assembly_version"] = version
    data["system_prompt"][0]["note"] = {"owner": "caller"}
    data["identity_block"] = None
    (tmp_path / "extraction.yaml").write_text(yaml.safe_dump(data))
    manager = PromptManager(tmp_path)
    assert manager.supports_assembly("extraction")
    assert (
        manager.render("extraction", "system_prompt", categories={"entity"}, name="Ada")
        == "COMMON Ada\nENTITY\n"
    )
    assert manager.load("extraction")["system_prompt"][0]["note"] == {"owner": "caller"}


def test_static_assembly_validation_runs_once_per_loaded_template(manager, monkeypatch):
    from extraction.prompts.manager import validate_prompt_manager

    validate = Mock(wraps=manager._validate_components)
    monkeypatch.setattr(manager, "_validate_components", validate)
    manager.load("extraction")
    validate_prompt_manager(manager)
    for _ in range(2):
        assert manager.supports_assembly("extraction")
        manager.render("extraction", "system_prompt", categories={"entity"}, name="Ada")
        manager.render("extraction", "examples", categories={"entity"})
    assert validate.call_count == 1


def test_pattern_rules_follow_user_observation_schema():
    manager = PromptManager()
    rules = manager.render("extraction", "system_prompt", categories={"pattern"})
    assert "observations about recurring user behavior or conversation patterns" in rules
    assert "under what circumstances to follow what process" not in rules
    examples = manager.render("extraction", "examples", categories={"pattern"})
    assert "Repeated user behavior" in examples


def test_retry_can_finish_as_legal_empty_task(manager, catalog):
    request = build_selection_request(*catalog, "thanks", "user-a", manager=manager)
    llm = SimpleNamespace(complete_json=Mock(side_effect=[{}, {"categories": []}]))
    result = run_selection_request(llm, request)
    assert result.scope == "no_task" and not result.categories
    assert llm.complete_json.call_count == 2


def test_compliant_injected_manager_needs_only_load_and_unified_render():
    from core.models import RequestContext
    from extraction.prompt_assembly import assemble_prepared
    from extraction.tools import Extractor, _Span

    class Manager:
        def __init__(self):
            self.inner = PromptManager()

        def load(self, template_name):
            return self.inner.load(template_name)

        def render(self, template_name, section, *, categories=None, **variables):
            return self.inner.render(template_name, section, categories=categories, **variables)

    llm = Mock()
    llm.complete_json.return_value = {"categories": ["entity"]}
    ctx = RequestContext("a", "u", "agent", "s", "t")
    extractor = Extractor(llm, prompt_manager=Manager())
    prepared = extractor._prepare_span(
        _Span(0, 0), [{"role": "user", "content": "Project uses SQL"}], "English", None, "", "", ctx
    )
    request = assemble_prepared(extractor, prepared, ctx)
    assert request.contract.scope == "scoped"
    assert "── profile (" not in request.prompt
    llm.complete_json.assert_called_once()


@pytest.mark.parametrize(
    "invalid_text", ["{% if %}", '{{ "x" | unknown_filter }}', "{{ unknown_variable }}"]
)
def test_selection_template_errors_stop_before_model_request(tmp_path, catalog, invalid_text):
    from extraction.prompt_assembly import select_assembly

    data = template_data()
    data["selection"]["instruction"] += invalid_text
    (tmp_path / "extraction.yaml").write_text(yaml.safe_dump(data))
    llm = Mock()
    with pytest.raises(ValueError, match="extraction/selection"):
        select_assembly(
            llm, *catalog, "I prefer black coffee.", "user-a", manager=PromptManager(tmp_path)
        )
    llm.complete_json.assert_not_called()


def test_builtin_selection_instructions_are_english():
    text = PromptManager().render("extraction", "selection", runtime_input_json="{}")
    assert text.isascii()
    assert "current_user" in text and "{{ runtime_input_json }}" not in text


@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("override", ["subclass", "instance"])
def test_injected_load_override_is_validated_after_caller_changes(tmp_path, invalid, override):
    from extraction.tools import Extractor

    (tmp_path / "extraction.yaml").write_text(yaml.safe_dump(template_data()))

    def customize(data):
        if invalid:
            data["examples"][0]["requires_types"] = [{}]
        else:
            data["system_prompt"][0]["text"] = "CALLER COMMON\n"
        return data

    class CallerManager(PromptManager):
        def load(self, template_name):
            return customize(super().load(template_name))

    if override == "subclass":
        manager = CallerManager(tmp_path)
    else:
        manager = PromptManager(tmp_path)
        original = manager.load
        manager.load = lambda name: customize(original(name))
    llm = Mock()
    if invalid:
        with pytest.raises(ValueError, match="extraction/examples/common"):
            Extractor(llm, prompt_manager=manager)
        llm.complete_json.assert_not_called()
    else:
        Extractor(llm, prompt_manager=manager)
        assert (
            manager.render("extraction", "system_prompt", categories={"entity"})
            == "CALLER COMMON\nENTITY\n"
        )
