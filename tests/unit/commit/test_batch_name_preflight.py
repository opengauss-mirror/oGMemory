from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.errors import ExtractionIncompleteError
from core.models import RequestContext
from service.extraction_checkpoint import CandidateCheckpoint
from tests.unit.commit.test_memory_identity import MemoryFS, MergeLLM, api, memory


def prepare(fs):
    ctx = RequestContext("name-batch", "reader", "agent", "session", "trace")
    writer = api(fs, MergeLLM())
    drafts = [memory("atlas_painting", content="Atlas is a red painting."),
              memory("atlas_painting", object_name="Atlas (painting)", content="Atlas was painted by Nora.")]
    journal = SimpleNamespace(batch={"archive_id": "archive", "slots": {}}, ctx=ctx)
    journal.mutate = lambda fn: fn(journal.batch)
    checkpoint = CandidateCheckpoint(journal, "all")
    writer._pipeline.extract = lambda *a, **k: drafts
    return writer, ctx, drafts, checkpoint


def test_ambiguous_batch_rejected_before_accept_and_public_write_then_can_reextract():
    fs = MemoryFS()
    writer, ctx, drafts, checkpoint = prepare(fs)
    original = deepcopy(drafts)
    with pytest.raises(ExtractionIncompleteError, match="identity"):
        writer.commit_session([], ctx, candidate_checkpoint=checkpoint)
    assert checkpoint.load() is None and fs.nodes == {} and drafts == original
    drafts[0].object_name = "Atlas (painting)"
    result = writer.commit_session([], ctx, candidate_checkpoint=checkpoint)
    assert result["writes_completed"] == 2 and result["writes_failed"] == 0
    assert len(fs.nodes) == 1
    replay = writer.commit_session([], ctx, candidate_checkpoint=checkpoint)
    assert replay["writes_failed"] == 0 and len(fs.nodes) == 1


def test_old_accepted_batch_is_not_rewritten():
    fs = MemoryFS()
    writer, ctx, drafts, checkpoint = prepare(fs)
    checkpoint.accept(drafts, len(drafts), 0)
    frozen = deepcopy(checkpoint.record["candidates"])
    result = writer.commit_session([], ctx, candidate_checkpoint=checkpoint)
    assert result["writes_failed"] == 1 and len(fs.nodes) == 1
    assert checkpoint.record["candidates"] == frozen


def test_distinct_scopes_keep_separate_nodes():
    fs = MemoryFS()
    writer, ctx, drafts, checkpoint = prepare(fs)
    drafts[0].scope_ref = "Project A"
    drafts[1].scope_ref = "Project B"
    result = writer.commit_session([], ctx, candidate_checkpoint=checkpoint)
    assert result["writes_failed"] == 0 and len(fs.nodes) == 2


def test_existing_exact_object_identity_does_not_create_false_collision():
    fs = MemoryFS()
    writer, ctx, drafts, checkpoint = prepare(fs)
    writer.write_memory(memory("atlas_painting", object_name="Atlas (painting)"), ctx)
    drafts[0].object_name = "Atlas (painting)"
    result = writer.commit_session([], ctx, candidate_checkpoint=checkpoint)
    assert result["writes_failed"] == 0 and len(fs.nodes) == 1


@pytest.mark.parametrize("target_alias", [True, False])
def test_projected_names_match_writer_for_target_alias_and_filename_template(target_alias):
    fs = MemoryFS()
    writer, ctx, drafts, checkpoint = prepare(fs)
    if target_alias:
        result = writer.write_memory(memory("atlas_painting", object_name="Atlas painting"), ctx)
        node = fs.read_node(result["target_uri"], ctx)
        drafts[:] = [memory("atlas_painting", object_name="Atlas (painting)",
                            target_uri=node.uri, expected_version=node.metadata["version"],
                            content="Atlas is red and old."),
                     memory("atlas_painting", object_name="Atlas (painting)", content="Atlas was painted by Nora.")]
    else:
        registry = writer._writer._identity_resolver.schemas
        registry.register(replace(registry.get("entity"), filename_template="item_{{ routing_key }}.md"))
        drafts[:] = [memory("topic", content="Earlier fact."), memory("topic", content="Later fact.")]
    original = deepcopy(fs.nodes)
    writer._writer._identity_resolver.preflight_new_batch(drafts, ctx)
    assert fs.nodes == original
    result = writer.commit_session([], ctx, candidate_checkpoint=checkpoint)
    assert result["writes_completed"] == 2 and result["writes_failed"] == 0
    assert len(fs.nodes) == 1


def test_direct_name_match_precedes_other_node_alias():
    fs = MemoryFS()
    writer, ctx, drafts, checkpoint = prepare(fs)
    writer.write_memory(memory("atlas", object_name="Atlas"), ctx)
    other = writer.write_memory(memory("globe", object_name="Globe"), ctx)
    node = fs.read_node(other["target_uri"], ctx)
    writer.write_memory(memory("globe", object_name="Atlas", content="Globe is also called Atlas.",
                               target_uri=node.uri, expected_version=node.metadata["version"]), ctx)
    drafts[:] = [memory("atlas", object_name="Atlas", content="Atlas was painted by Nora.")]
    result = writer.commit_session([], ctx, candidate_checkpoint=checkpoint)
    assert result["writes_completed"] == 1 and result["writes_failed"] == 0
    assert len(fs.nodes) == 2


def test_bound_batch_reads_each_target_once_and_projects_successive_names():
    fs = MemoryFS()
    writer, ctx, drafts, checkpoint = prepare(fs)
    first = writer.write_memory(memory("atlas", object_name="Atlas"), ctx)["target_uri"]
    second = writer.write_memory(memory("globe", object_name="Globe"), ctx)["target_uri"]
    writer.write_memory(memory("portrait", object_name="Portrait"), ctx)
    drafts[:] = [memory("atlas", object_name="Atlas painting", target_uri=first, expected_version=1),
                 memory("globe", object_name="Globe painting", target_uri=second, expected_version=1),
                 memory("atlas", object_name="Atlas artwork", target_uri=first, expected_version=2)]
    reads = []
    read_node = fs.read_node

    def record_read(uri, request_ctx):
        reads.append(uri)
        return read_node(uri, request_ctx)

    fs.read_node = record_read
    writer._writer._identity_resolver.preflight_new_batch(drafts, ctx)
    assert reads == [first, second]
