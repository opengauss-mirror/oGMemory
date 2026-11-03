"""Optional demonstrations must not consume the full-source extraction budget."""
import json

import pytest
import extraction.occurrence_qualification as qualification

from core.errors import ExtractionIncompleteError
from extraction.occurrence_qualification import prompt_fits
from extraction.tools import Extractor
from tests.unit.extraction.test_lazy_fallback_budget import SOURCE, setup


def test_small_prompt_retains_examples(monkeypatch):
    extractor = Extractor(object())
    render = extractor._prompt_manager.render
    monkeypatch.setattr(extractor._prompt_manager, "render", lambda name, section, **values:
                        "Example: Alice prefers tea." if section == "examples"
                        else render(name, section, **values))
    prompt = extractor._build_prompt("en", SOURCE)
    assert extractor._prompt_manager.render("extraction", "examples") in prompt
    assert prompt_fits(prompt, {"tools": extractor._extraction_tools})


def test_complete_source_and_identity_survive_optional_example_elision(monkeypatch):
    extractor, llm, span, ctx, _ = setup(monkeypatch, large=False)
    conversation = SOURCE * 200
    prompt = extractor._build_prompt("en", conversation, ctx=ctx)
    assert prompt.endswith(conversation)
    assert prompt_fits(prompt, {"tools": extractor._extraction_tools})
    assert extractor._prompt_manager.render("extraction", "examples") not in prompt
    assert extractor._update_instruction in prompt
    assert extractor._prompt_manager.render("extraction", "identity_block", user_id=ctx.user_id) in prompt
    assert len(extractor._structure_span_lazy(span, prompt, ctx)) == 1
    assert conversation in llm.calls[0]["prompt"]


def test_source_identical_to_template_is_not_removed(monkeypatch):
    extractor = Extractor(object())
    render = extractor._prompt_manager.render
    monkeypatch.setattr(extractor._prompt_manager, "render", lambda name, section, **values:
                        "Example: Alice prefers tea." if section == "examples"
                        else render(name, section, **values))
    examples = extractor._prompt_manager.render("extraction", "examples")
    conversation = examples + "\nUSER: This exact text is my source."
    full_prompt = extractor._build_prompt("en", conversation)
    request_chars = (len(json.dumps(full_prompt, ensure_ascii=False))
                     + len(json.dumps({"tools": extractor._extraction_tools}, ensure_ascii=False)))
    monkeypatch.setattr(qualification, "_CONTEXT_CHARS",
                        request_chars + qualification._OUTPUT_CHARS - len(examples))
    prompt = extractor._build_prompt("en", conversation)
    assert prompt.endswith(conversation)
    assert prompt.count(examples) == 1
    assert prompt_fits(prompt, {"tools": extractor._extraction_tools})


def test_source_over_budget_still_rejected_without_provider(monkeypatch):
    extractor, llm, span, ctx, _ = setup(monkeypatch, large=False)
    source = SOURCE * 5000
    prompt = extractor._build_prompt("en", source, ctx=ctx)
    assert prompt.endswith(source)
    with pytest.raises(ExtractionIncompleteError, match="budget"):
        extractor._structure_span_lazy(span, prompt, ctx)
    assert not llm.calls
