from dataclasses import replace

from extraction.card_format import validate_memory_card
from extraction.source_attribution import _entity_preference_card
from tests.unit.server.test_memory_card_recovery import card


def test_routed_preference_preserves_facts_in_entity_section():
    preference = replace(card(), category="preference", content="# Radio\n- Nora prefers radio work.")
    content = _entity_preference_card(preference, "Nora")
    entity = replace(preference, category="entity", content=content)
    validate_memory_card(entity)
    assert content == "# Nora\n\n## Preferences\n- Nora prefers radio work."
