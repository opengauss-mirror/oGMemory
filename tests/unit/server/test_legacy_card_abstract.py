from dataclasses import replace

import pytest

from core.models import RequestContext
from extraction.card_format import validate_memory_card
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.server.test_extraction_batch_recovery import params, setup_service
from tests.unit.server.test_memory_card_recovery import card, enable


@pytest.mark.parametrize("category", ["entity", "preference"])
@pytest.mark.parametrize("separator", ["\n", "\r\n"])
def test_duplicate_observation_migrates_legacy_abstract(category, separator):
    ctx = RequestContext("legacy-abstract", "reader", "agent", "source", "proof")
    original = replace(card(), category=category,
                       abstract=separator.join(["Nora is a radio engineer.", "Works in quiet rooms."]))
    if category == "preference":
        original = replace(original, content="# Radio work\n\n- Prefers quiet rooms for radio work.")
    incoming = replace(original, abstract="Nora is a radio engineer.")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[incoming])
    uri = writer.write_memory(original, ctx)["target_uri"]
    enable(service, writer)

    assert service.after_turn(params(ctx))["status"] == "completed"

    stored = fs.read_node(uri, ctx)
    validate_memory_card(stored)
    assert stored.abstract == "Nora is a radio engineer. Works in quiet rooms."
    assert stored.overview == original.overview
    assert stored.content == original.content
    assert stored.metadata["version"] == 2
    assert len(stored.metadata["provenance_ids"]) == 2
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    restarted, next_writer, _ = setup_service(fs, ctx, entries=[incoming])
    enable(restarted, next_writer)
    assert restarted.after_turn(params(ctx))["status"] == "duplicate"
