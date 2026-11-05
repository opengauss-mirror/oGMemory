"""YAML components are selected before rendering through the public manager."""
from hashlib import sha256

import pytest
import yaml

from extraction.assembly_contract import AssemblyContract, UnsupportedAssemblyTemplateError
from extraction.prompt_blocks import render_blocks
from extraction.prompts import PromptManager


def decision(*categories, scope="scoped"):
    return AssemblyContract(frozenset(categories), (), "eager", scope)


def component_manager(tmp_path):
    data = {
        "assembly_version": 1,
        "selection": {"instruction": "Choose types", "classification_rules": "Subject boundary", "types": {"profile": "User", "entity": "Other subjects"}},
        "system_prompt": [
            {"id": "common", "text": "COMMON {{ name }}\n"},
            {"id": "boundary", "types": ["profile", "entity"], "text": "BOUNDARY\n"},
            {"id": "profile", "types": ["profile"], "text": "{{ must_not_render() }}\n"},
            {"id": "entity", "types": ["entity"], "text": "ENTITY {{ name }}\n"},
        ],
        "examples": [
            {"id": "public", "text": "PUBLIC\n"},
            {"id": "mixed", "requires_types": ["profile", "entity"], "text": "BOTH\n"},
            {"id": "entity", "requires_types": ["entity"], "text": "ENTITY EXAMPLE\n"},
        ],
        "output_instruction": "Language {{ language }}",
    }
    (tmp_path / "custom.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return PromptManager(tmp_path)


def test_selects_rules_before_render_and_keeps_actual_caller_edits(tmp_path):
    manager = component_manager(tmp_path)
    assert manager.supports_assembly("custom")
    manager.load("custom")["system_prompt"].insert(1, {"id": "caller", "text": "CALLER {{ name }}\n"})
    assert manager.render_selected("custom", "system_prompt", decision=decision("entity"), name="Ada") == "COMMON Ada\nCALLER Ada\nBOUNDARY\nENTITY Ada\n"


def test_examples_require_all_positive_types(tmp_path):
    manager = component_manager(tmp_path)
    assert manager.render_selected("custom", "examples", decision=decision("entity")) == "PUBLIC\nENTITY EXAMPLE\n"
    assert manager.render_selected("custom", "examples", decision=decision("entity", "profile")) == "PUBLIC\nBOTH\nENTITY EXAMPLE\n"


def test_full_render_and_selected_full_use_same_text_and_variables(tmp_path):
    manager = component_manager(tmp_path)
    expected = "COMMON Ada\nBOUNDARY\nPROFILE\nENTITY Ada\n"
    manager.load("custom")["system_prompt"][2]["text"] = "PROFILE\n"
    assert manager.render("custom", "system_prompt", name="Ada") == expected
    assert manager.render_selected("custom", "system_prompt", decision=decision(scope="full"), name="Ada") == expected
    assert manager.render_selected("custom", "output_instruction", decision=decision("entity"), language="English") == "Language English"
    assert "Choose types" not in expected


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "wrong_type", "unsupported_version", "wrong_field"])
def test_invalid_components_disable_selection_but_full_text_survives(tmp_path, mutation):
    manager = component_manager(tmp_path)
    data = manager.load("custom")
    data["system_prompt"][2]["text"] = "PROFILE\n"
    if mutation == "duplicate":
        data["system_prompt"][1]["id"] = "common"
    elif mutation == "unknown":
        data["system_prompt"][1]["types"] = ["future"]
    elif mutation == "wrong_type":
        data["system_prompt"][1]["types"] = "entity"
    elif mutation == "unsupported_version":
        data["assembly_version"] = 2
    else:
        data["examples"][0]["types"] = ["entity"]
    assert not manager.supports_assembly("custom")
    assert "COMMON Ada" in manager.render("custom", "system_prompt", name="Ada")
    with pytest.raises(UnsupportedAssemblyTemplateError):
        manager.render_selected("custom", "system_prompt", decision=decision("entity"))


def test_legacy_manager_override_requires_explicit_component_optin():
    class Legacy(PromptManager):
        def render(self, template_name, section, **variables):
            return "CUSTOM " + super().render(template_name, section, **variables)

    class ComponentAware(Legacy):
        def render_selected(self, template_name, section, *, decision, **variables):
            return "SELECTED CUSTOM " + super().render_selected(template_name, section, decision=decision, **variables)

    legacy = Legacy()
    assert not legacy.supports_assembly("extraction")
    assert legacy.render("extraction", "examples").startswith("CUSTOM ")
    with pytest.raises(UnsupportedAssemblyTemplateError):
        render_blocks(legacy, "examples", decision("entity"))
    aware = ComponentAware()
    assert aware.supports_assembly("extraction")
    assert render_blocks(aware, "examples", decision("entity")).startswith("SELECTED CUSTOM ")


def test_instance_render_override_also_disables_inherited_assembly():
    manager = PromptManager()
    manager.render = lambda *args, **kw: "INSTANCE CUSTOM"
    assert not manager.supports_assembly("extraction")


def test_legacy_string_template_keeps_complete_render(tmp_path):
    (tmp_path / "legacy.yaml").write_text('system_prompt: "Hi {{ name }}"\noutput_instruction: JSON\n', encoding="utf-8")
    manager = PromptManager(tmp_path)
    assert not manager.supports_assembly("legacy")
    assert manager.render("legacy", "system_prompt", name="Ada") == "Hi Ada"


def test_selection_contains_all_classification_boundaries_without_entering_extraction():
    manager = PromptManager()
    text = manager.render("extraction", "selection")
    for category in ("profile", "preference", "entity", "event", "case", "pattern", "skill", "tool"):
        assert category in text
    assert "BY the user AND ABOUT the user" in text
    assert "shared_tools" in text
    assert "SELECT CAPABILITIES" not in manager.render("extraction", "system_prompt")


def test_entity_quote_examples_do_not_require_profile():
    text = PromptManager().render_selected("extraction", "examples", decision=decision("entity"))
    assert "## entity Example" in text
    assert "Forwarded quote about someone else" in text
    assert "Report about another person" in text
    assert "Group chat with multiple speakers" not in text


@pytest.mark.parametrize(
    "template,section,digest",
    [('extraction',
      'system_prompt',
      '52465162da1330155cf232b690ba1b05967f3675ea4794837636226f75cb7cd4'),
     ('extraction', 'examples', 'b61e484caab11e18cbfc51979803500f05812df02f2e92e7fe08c749fde9407e'),
     ('extraction',
      'identity_block',
      'ba8647353850983cb22f5de5914a9eee9a9f846b54635769336f88df26150257'),
     ('extraction',
      'lazy_tools_instruction',
      '114e0a0f0758453e2b8ee6ad90ce420399ecf00cc2b19e98ea9e89ae21aff25d'),
     ('extraction',
      'conversation_header',
      'df8d6584633fdc8321c1b15582656b0ef38e9d8cc2d7568151f095c57d3f474a'),
     ('extraction',
      'output_instruction',
      '780274147a9df1bdda08dd231040d5da38136e94d90dafdcc50b17ada2a837c0'),
     ('extraction_graph',
      'system_prompt',
      'ca475978e16494b5ad8fd1a27818df428348a2e8940f8f9530431f03213f6e62'),
     ('extraction_graph',
      'examples',
      '7dfc69502e1b119666a688ce4123d234f2cb2a75dedb80e0eb6d42be0df698c4'),
     ('extraction_graph',
      'identity_block',
      'a74858936790b186d418ed190d79d1f7a974b8328d6314403b9b497f3457e9c9'),
     ('extraction_graph',
      'lazy_tools_instruction',
      'b4053022a5c0c8775f528cd6672b091d31ff8aeeacbb0fc64ddb0cd84f4b801a'),
     ('extraction_graph',
      'conversation_header',
      'df8d6584633fdc8321c1b15582656b0ef38e9d8cc2d7568151f095c57d3f474a'),
     ('extraction_graph',
      'output_instruction',
      '12d9e985314c714f627ecfbf6c20db2f44740414d7024578039d8b2b74c1eb1e')],
)
def test_full_template_migration_preserves_original_render(template, section, digest):
    text = PromptManager().render(template, section, **{'user_id': 'alice', 'account_id': 'acct', 'agent_id': 'agent', 'session_summary': 'Saved reference', 'session_time': '2026-11-05', 'day_of_week': 'Thursday', 'output_language': 'English'})
    assert sha256(text.encode()).hexdigest() == digest
    assert PromptManager().supports_assembly(template)
