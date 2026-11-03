"""New batch ordering must not invalidate its own read-bound updates."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.models import RequestContext
from service.extraction_checkpoint import CandidateCheckpoint
from tests.unit.commit.test_memory_identity import MemoryFS, api, memory


class PreserveFacts:
    def complete_json(self, prompt, schema):
        return {"content": "Melanie lives in Utrecht, works as an engineer, and likes chess."}


def prepare(fs, *, checkpointed=True):
    ctx = RequestContext(account_id="batch", user_id="reader", agent_id="agent",
                         session_id="source", trace_id="batch")
    writer = api(fs, PreserveFacts())
    uri = writer.write_memory(memory("melanie", content="Melanie lives in Utrecht."), ctx)["target_uri"]
    delta = memory("melanie", content="Melanie likes chess.")
    bound = replace(delta, content="Melanie lives in Utrecht and works as an engineer.",
                    target_uri=uri, expected_version=1)
    writer._pipeline.extract = lambda *args, **kwargs: [delta, bound]
    journal = SimpleNamespace(batch={"archive_id": "batch-source", "slots": {}}, ctx=ctx)
    journal.mutate = lambda fn: fn(journal.batch)
    checkpoint = CandidateCheckpoint(journal, "all") if checkpointed else None
    return writer, ctx, uri, delta, bound, checkpoint


@pytest.mark.parametrize("checkpointed", [False, True])
def test_new_batch_bound_snapshot_precedes_source_delta(checkpointed):
    fs = MemoryFS()
    writer, ctx, uri, delta, bound, checkpoint = prepare(fs, checkpointed=checkpointed)
    result = writer.commit_session([], ctx, archive_id="batch-source", candidate_checkpoint=checkpoint)
    assert result["writes_completed"] == 2 and result["writes_failed"] == 0
    node = fs.read_node(uri, ctx)
    assert node.metadata["version"] == 3
    assert all(fact in node.content for fact in ("Utrecht", "engineer", "chess"))
    assert bound.expected_version == 1 and delta.target_uri == ""
    if checkpoint:
        frozen = deepcopy(checkpoint.record)
        assert checkpoint.load()[0][0].target_uri == uri
        replay = writer.commit_session([], ctx, candidate_checkpoint=checkpoint)
        assert replay["writes_failed"] == 0
        assert fs.read_node(uri, ctx).metadata["version"] == 3
        assert checkpoint.record == frozen


def test_old_accepted_order_remains_frozen():
    fs = MemoryFS()
    writer, ctx, uri, delta, bound, checkpoint = prepare(fs)
    checkpoint.accept([delta, bound], 2, 0)
    frozen = deepcopy(checkpoint.record["candidates"])
    result = writer.commit_session([], ctx, candidate_checkpoint=checkpoint)
    assert result["writes_failed"] == 1
    assert checkpoint.record["candidates"] == frozen


def test_external_change_still_rejects_bound_snapshot():
    fs = MemoryFS()
    writer, ctx, uri, delta, bound, checkpoint = prepare(fs)
    writer.write_memory(memory("melanie", content="An external update."), ctx)
    result = writer.commit_session([], ctx, candidate_checkpoint=checkpoint)
    assert result["writes_failed"] == 1
    assert checkpoint.load()[0][0].expected_version == 1
