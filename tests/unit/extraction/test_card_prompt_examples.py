"""Positive demonstrations satisfy the same card contract as extracted candidates."""

import re

import pytest
import yaml

from core.models import CandidateMemory
from core.validation import validate_candidate
from extraction.card_format import validate_memory_card
from extraction.prompts.manager import PromptManager
from extraction.schemas.registry import SchemaRegistry
from extraction.tools import Extractor


def examples():
    text = PromptManager().render("extraction", "examples")
    matches = re.findall(r"^.*extract_(entity|preference)[^\n]*:\n((?: {2,}[^\n]*\n)+)", text, re.M)
    return [(category, yaml.safe_load(fields)) for category, fields in matches]


@pytest.mark.parametrize("category,fields", examples(), ids=lambda value: value if isinstance(value, str) else value["routing_key"])
def test_demonstrated_card_validates(category, fields):
    candidate = CandidateMemory(category=category, owner_scope="user", **fields)
    validated = validate_candidate(candidate)
    validate_memory_card(validated)
    assert validated.content == fields["content"]


def test_full_examples_fit_supported_topic_registry(tmp_path):
    registry = SchemaRegistry(str(tmp_path))
    defaults = SchemaRegistry()
    for category in ("entity", "preference"):
        registry.register(defaults.get(category))
    extractor = Extractor(object(), schema_registry=registry)
    prompt = extractor._build_prompt("en", "A short conversation.")
    assert extractor._prompt_manager.render("extraction", "examples") in prompt
    assert len(examples()) == 7
