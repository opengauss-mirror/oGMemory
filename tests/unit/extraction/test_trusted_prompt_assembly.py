"""Behavioral checks for trusted prompt assembly with the existing selector."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
import yaml

from core.models import RequestContext
from core.uri_resolver import URIResolver
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.prompt_assembly import assemble_prepared
from extraction.prompts import PromptManager
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor, _Span

CTX = RequestContext("acct", "user", "agent", "session", "trace")


def prepared(categories=("entity",), mode="eager"):
    registry = SchemaRegistry()
    llm = Mock()
    llm.complete_json.return_value = {"categories": list(categories)}
    extractor = Extractor(llm, mode=mode, fs=Mock(), schema_registry=registry,
                          uri_resolver=URIResolver(registry))
    task = extractor._prepare_span(
        _Span(0, 0, categories=list(categories)),
        [{"role": "user", "content": "Project Beacon uses PostgreSQL."}],
        "English", None, "", "", CTX)
    return extractor, task


def test_eager_runtime_does_not_require_unoffered_profile():
    extractor, task = prepared()
    request = assemble_prepared(extractor, task, CTX)
    prompt, tools = request.prompt, request.tools
    assert "extract_profile" not in {t["name"] for t in tools}
    assert "Emit exactly ONE extract_profile" not in prompt
    assert "Project Beacon uses PostgreSQL." in prompt


@pytest.mark.parametrize("section", ["examples", "system_prompt"])
def test_invalid_injected_components_are_configuration_errors(section, tmp_path):
    registry = SchemaRegistry()
    manager = PromptManager()
    data = manager.load("extraction")
    data[section][1]["id"] = data[section][0]["id"]
    (tmp_path / "extraction.yaml").write_text(yaml.safe_dump(data))
    manager = PromptManager(tmp_path)
    llm = Mock()
    llm.complete_with_tools_messages.return_value = ([], "[]")
    from extraction.assembly_contract import UnsupportedAssemblyTemplateError

    with pytest.raises(UnsupportedAssemblyTemplateError):
        ExtractionReActLoop(llm, Mock(), registry, URIResolver(registry), manager)


def test_unknown_category_template_restores_full_scope():
    extractor, task = prepared(("future",), mode="lazy")
    registry = extractor._schema_registry
    registry.register(replace(registry.get("event"), memory_type="future"))
    from extraction.tool_builder import build_extraction_tools
    extractor._extraction_tools = build_extraction_tools(registry)
    extractor._llm.complete_with_tools_messages.return_value = ([], "[]")
    extractor._structure_prepared_span(task, CTX)
    request = extractor._llm.complete_with_tools_messages.call_args.kwargs
    assert {"extract_future", "extract_profile"} <= {t["name"] for t in request["tools"]}


@pytest.mark.parametrize("scope", ["scoped", "full"])
def test_lazy_preserves_output_language_and_example_omission(scope):
    extractor, task = prepared(
        categories=("unknown",) if scope == "full" else ("entity",), mode="lazy"
    )
    task.span.prompt_parts = replace(task.span.prompt_parts, include_examples=False)
    task.prompt = task.span.prompt_parts.render()
    extractor._llm.complete_with_tools_messages.return_value = ([], "[]")
    extractor._structure_prepared_span(task, CTX)
    text = extractor._llm.complete_with_tools_messages.call_args.kwargs["messages"][0]["content"]
    assert "Please output all abstract/overview/content fields in English." in text
    assert "## entity Example" not in text
    assert ("── profile (" in text) == (scope == "full")


def test_lazy_reassembly_preserves_appended_feedback():
    extractor, task = prepared(mode="lazy")
    request = assemble_prepared(extractor, task, CTX)
    prompt, _ = request.prompt, request.tools
    task.prompt = prompt + "\nCORRECTION_FEEDBACK_123"
    extractor._llm.complete_with_tools_messages.return_value = ([], "[]")
    extractor._structure_prepared_span(task, CTX)
    messages = extractor._llm.complete_with_tools_messages.call_args.kwargs["messages"]
    assert "CORRECTION_FEEDBACK_123" in str(messages)


def test_lazy_render_failure_publishes_full_contract_before_eager_recovery(monkeypatch):
    from core.errors import ToolCallProtocolError

    extractor, task = prepared(mode="lazy")
    original = extractor._prompt_manager.render

    def render(template, section, **variables):
        if section == "examples" and variables.get("categories") is not None:
            raise ValueError("caller rendering failure")
        return original(template, section, **variables)

    monkeypatch.setattr(extractor._prompt_manager, "render", render)
    extractor._llm.complete_with_tools_messages.side_effect = ToolCallProtocolError("mock failure")
    extractor._structure_span_eager = Mock(return_value=[])
    extractor._structure_prepared_span(task, CTX)
    request = extractor._structure_span_eager.call_args
    assert "extract_profile" in {t["name"] for t in request.kwargs["extraction_tools"]}
    assert "── profile (" in request.args[1]
    assert task.assembly_state.contract.scope == "full"


def test_preference_scope_keeps_original_eligibility_boundaries():
    extractor, task = prepared(("preference",))
    request = assemble_prepared(extractor, task, CTX)
    prompt, _ = request.prompt, request.tools
    assert "Stable identity attribute → profile beats preference" in prompt
    assert "Explicit time dimension → event beats preference" in prompt


@pytest.mark.parametrize("section,key,invalid", [
    ("system_prompt", "types", None),
    ("system_prompt", "types", ["entity", "entity"]),
    ("system_prompt", "types", [42]),
    ("examples", "requires_types", None),
    ("examples", "requires_types", "entity"),
    ("examples", "requires_types", [{}]),
])
def test_malformed_component_dependencies_fail_at_injection(tmp_path, section, key, invalid):
    from extraction.assembly_contract import UnsupportedAssemblyTemplateError

    data = PromptManager().load("extraction")
    data[section][0][key] = invalid
    (tmp_path / "extraction.yaml").write_text(yaml.safe_dump(data))
    llm = Mock()
    with pytest.raises(UnsupportedAssemblyTemplateError):
        Extractor(llm, prompt_manager=PromptManager(tmp_path))
    llm.complete_json.assert_not_called()


def test_node_processing_instruction_survives_compact_json_parts():
    from extraction.prompt_assembly import compact_prepared_prompt

    extractor, _ = prepared()
    extractor._node_processing_enabled = True
    task = extractor._prepare_span(
        _Span(0, 0, categories=["entity"]),
        [{"role": "user", "content": "Project Beacon uses PostgreSQL."}],
        "English", None, "", "", CTX)
    request = assemble_prepared(extractor, task, CTX)
    assert "TOPIC_DRAFTS:" in compact_prepared_prompt(task, request.contract)


def test_custom_lazy_full_fallback_preserves_rendered_template_variables():
    extractor, _ = prepared(mode="lazy")
    extractor._prompt_manager.load("extraction")["system_prompt"].append(
        {"id": "caller_reference", "text": "\nCUSTOM_REFERENCE {{ session_summary }}"})
    task = extractor._prepare_span(
        _Span(0, 0, categories=["entity"]),
        [{"role": "user", "content": "Project Beacon uses PostgreSQL."}],
        "English", None, "saved reference", "", CTX)
    extractor._llm.complete_with_tools_messages.return_value = ([], "[]")
    extractor._structure_prepared_span(task, CTX)
    text = extractor._llm.complete_with_tools_messages.call_args.kwargs["messages"][0]["content"]
    assert "CUSTOM_REFERENCE saved reference" in text
