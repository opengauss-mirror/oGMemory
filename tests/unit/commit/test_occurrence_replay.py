"""Source-backed occurrence replay through the public write API."""
from copy import deepcopy
from dataclasses import replace

import pytest

from core.models import RequestContext
from core.validation import MAX_MEMORY_KEY_BYTES, validate_candidate
from tests.unit.commit.test_memory_identity import DecisionLLM, MemoryFS, api, memory


@pytest.fixture
def ctx():
    return RequestContext("replay", "reader", "agent", "first-session", "trace")


def occurrence(category="event", **changes):
    return replace(memory("equipment_inspection", category=category,
                          content="A technician inspected the equipment.",
                          when="2026-10-01 09:00", who="Technician", where="Workshop",
                          scope_ref="Project A", title="Equipment inspection"),
                   owner_scope="agent" if category == "case" else "user", **changes)


@pytest.mark.parametrize("category", ["event", "case"])
@pytest.mark.parametrize("title_only", [False, True])
def test_explicit_occurrence_update_preserves_revised_title_and_sources(ctx, category, title_only):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    first = occurrence(category)
    uri = writer.write_memory(first, ctx)["target_uri"]
    updated = replace(first, target_uri=uri, expected_version=1,
                      title="Equipment inspection and approved repair",
                      abstract="Inspection and approved repair",
                      overview="The equipment was inspected and its repair approved.",
                      content=first.content + " The repair was approved.",
                      provenance_ids=["source:repair-approval"])
    if title_only:
        updated = replace(updated, abstract=first.abstract, overview=first.overview,
                          content=first.content)

    result = writer.write_memory(updated, replace(ctx, session_id="follow-up"))

    assert result["target_uri"] == uri
    stored = fs.read_node(uri, ctx)
    assert stored.metadata["title"] == updated.title
    assert stored.metadata["version"] == 2
    assert stored.content == updated.content
    assert set(stored.metadata["provenance_ids"]) == {
        *first.provenance_ids, *updated.provenance_ids,
    }
    assert len(fs.nodes) == 1


@pytest.mark.parametrize("category,receipt", [
    ("event", "e47c41b73d724c47046e279be52d543c22eb0423a68eb544492acd6e609608d4"),
    ("case", "f2f7ccb982662abf755bc6e0c5c3d0f28cda6f8923a9194aa40bb5aa452b7e99"),
])
def test_runtime_snapshot_contract_preserves_pre_upgrade_replay_receipts(ctx, category, receipt):
    fs = MemoryFS()
    draft = occurrence(category)
    uri = api(fs, DecisionLLM()).write_memory(draft, ctx)["target_uri"]
    # Durable receipt produced by the pre-qualification public CandidateMemory contract.
    fs.nodes[uri].metadata["occurrence_replay_sources"] = {receipt: draft.provenance_ids}
    before = deepcopy(fs.nodes)

    result = api(fs, DecisionLLM()).write_memory(draft, replace(ctx, session_id="retry"))

    assert result["action"] == "skip"
    assert result["target_uri"] == uri
    assert fs.nodes == before


@pytest.mark.parametrize("category", ["event", "case"])
@pytest.mark.parametrize("complete", [False, True])
def test_complete_occurrence_context_distinguishes_withdrawal_from_omission(ctx, category, complete):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    first = occurrence(category)
    uri = writer.write_memory(first, ctx)["target_uri"]
    update = replace(first, target_uri=uri, expected_version=1, where=None,
                     complete_occurrence_context=complete,
                     content="The inspection occurred. The previously recorded location is unconfirmed.",
                     overview="The inspection location is unconfirmed.",
                     provenance_ids=["source:location-withdrawal"])

    writer.write_memory(update, ctx)

    stored = fs.read_node(uri, ctx)
    assert stored.metadata.get("where") == (None if complete else first.where)
    assert stored.metadata["when"] == first.when and stored.metadata["who"] == first.who
    assert stored.metadata["version"] == 2


@pytest.mark.parametrize("category", ["event", "case"])
def test_replay_survives_new_writer_session_and_generated_name(ctx, category):
    fs = MemoryFS()
    draft = occurrence(category)
    uri = api(fs, DecisionLLM()).write_memory(draft, ctx)["target_uri"]
    result = api(fs, DecisionLLM()).write_memory(
        replace(draft, routing_key="inspection_report", confidence=.95),
        replace(ctx, session_id="retry-session"),
    )
    assert result["action"] == "skip"
    assert result["target_uri"] == uri
    assert len(fs.nodes) == 1
    assert fs.read_node(uri, ctx).content == draft.content


@pytest.mark.parametrize("changes", [
    {"content": "A technician inspected a different piece of equipment."},
    {"when": "2026-10-01 17:00"}, {"who": "Another technician"},
    {"where": "Another workshop"}, {"scope_ref": "Project B"},
    {"overview": "Inspection was cancelled."}, {"title": "Cancelled inspection"},
    {"attributed_speaker": "Another speaker"},
    {"evidence_quote": "This is another inspection."},
    {"provenance_ids": ["source:another-message"]},
    {"provenance_ids": []},
])
def test_replay_does_not_collapse_distinct_evidence_or_occurrences(ctx, changes):
    fs = MemoryFS()
    writer = api(fs, DecisionLLM())
    first = occurrence()
    uri = writer.write_memory(first, ctx)["target_uri"]
    other = writer.write_memory(replace(first, **changes), ctx)
    assert other["action"] == "create"
    assert other["target_uri"] != uri
    assert len(fs.nodes) == 2


def test_exact_replay_is_a_read_only_noop(ctx):
    fs = MemoryFS()
    draft = occurrence()
    uri = api(fs, DecisionLLM()).write_memory(draft, ctx)["target_uri"]
    before = deepcopy(fs.nodes)
    result = api(fs, DecisionLLM()).write_memory(draft, ctx)
    assert result["action"] == "skip"
    assert result["target_uri"] == uri
    assert fs.nodes == before


def test_old_replay_does_not_revert_a_versioned_update(ctx):
    fs = MemoryFS()
    draft = occurrence()
    uri = api(fs, DecisionLLM()).write_memory(draft, ctx)["target_uri"]
    updated = replace(draft, target_uri=uri, expected_version=1,
                      content=draft.content + " The repair was approved.",
                      abstract="Inspection and approved repair", overview="Repair approved.",
                      provenance_ids=["source:approval"])
    api(fs, DecisionLLM()).write_memory(updated, ctx)
    before = deepcopy(fs.nodes)
    result = api(fs, DecisionLLM()).write_memory(draft, replace(ctx, session_id="retry"))
    assert result["action"] == "skip"
    assert result["target_uri"] == uri
    assert fs.nodes == before


def test_replay_is_isolated_between_users_and_agents(ctx):
    fs = MemoryFS()
    for category in ("event", "case"):
        draft = occurrence(category)
        first = api(fs, DecisionLLM()).write_memory(draft, ctx)
        other_ctx = replace(ctx, user_id="other") if category == "event" else replace(ctx, agent_id="other")
        second = api(fs, DecisionLLM()).write_memory(draft, other_ctx)
        assert first["target_uri"] != second["target_uri"]
    assert len(fs.nodes) == 4


def test_partial_source_replay_and_missing_occurrence_metadata(ctx):
    fs = MemoryFS()
    draft = occurrence(when=None, who=None, where=None,
                       provenance_ids=["source:one", "source:two"])
    uri = api(fs, DecisionLLM()).write_memory(draft, ctx)["target_uri"]
    result = api(fs, DecisionLLM()).write_memory(
        replace(draft, provenance_ids=["source:two"]), ctx)
    assert result["action"] == "skip"
    assert result["target_uri"] == uri
    assert fs.read_node(uri, ctx).metadata["provenance_ids"] == draft.provenance_ids


def test_source_overlap_does_not_prove_a_complete_replay(ctx):
    fs = MemoryFS()
    draft = occurrence(provenance_ids=["source:one", "source:two"])
    api(fs, DecisionLLM()).write_memory(draft, ctx)
    result = api(fs, DecisionLLM()).write_memory(
        replace(draft, provenance_ids=["source:two", "source:three"]), ctx)
    assert result["action"] == "create"
    assert len(fs.nodes) == 2
@pytest.mark.parametrize("category", ["event", "case"])
def test_unsourced_occurrences_are_not_replay_proof(ctx, category):
    fs = MemoryFS()
    draft = occurrence(category, provenance_ids=[])
    first = api(fs, DecisionLLM()).write_memory(draft, ctx)
    second = api(fs, DecisionLLM()).write_memory(draft, ctx)
    assert first["target_uri"] != second["target_uri"]
    assert len(fs.nodes) == 2


def test_receipt_lookup_uses_all_metadata_pages_without_reading_old_bodies(ctx):
    class SummaryFS(MemoryFS):
        def __init__(self):
            super().__init__()
            self.pages = 0
            self.reads = 0

        def read_node(self, uri, request):
            self.reads += 1
            return super().read_node(uri, request)

        def list_identity_node_summaries_page(self, directory, request, *, limit, after_uri=None):
            self.pages += 1
            uris = [uri for uri in self.list_children(directory, request)
                    if not after_uri or uri > after_uri][:limit]
            return [replace(deepcopy(self.nodes[uri]), content="", overview="") for uri in uris]

    fs = SummaryFS()
    draft = occurrence()
    uri = api(fs, DecisionLLM()).write_memory(draft, ctx)["target_uri"]
    node = fs.nodes.pop(uri)
    last_uri = uri.rsplit("/", 1)[0] + "/zz_last"
    node.uri = last_uri
    fs.nodes[last_uri] = node
    for index in range(100):
        unrelated = deepcopy(node)
        unrelated.uri = uri.rsplit("/", 1)[0] + f"/first_{index:03}"
        unrelated.metadata.pop("occurrence_replay_sources")
        fs.nodes[unrelated.uri] = unrelated
    fs.pages = fs.reads = 0
    result = api(fs, DecisionLLM()).write_memory(draft, ctx)
    assert result["action"] == "skip"
    assert result["target_uri"] == last_uri
    assert fs.pages == 2
    assert fs.reads == 0
    assert len(fs.nodes) == 101


def test_legacy_nodes_without_receipts_are_not_guessed_or_modified(ctx):
    fs = MemoryFS()
    draft = occurrence()
    first_uri = api(fs, DecisionLLM()).write_memory(draft, ctx)["target_uri"]
    fs.nodes[first_uri].metadata.pop("occurrence_replay_sources")
    before = deepcopy(fs.nodes[first_uri])
    result = api(fs, DecisionLLM()).write_memory(draft, ctx)
    assert result["action"] == "create"
    assert result["target_uri"] != first_uri
    assert fs.nodes[first_uri] == before


def test_ambiguous_replay_receipts_fail_without_changing_nodes(ctx):
    fs = MemoryFS()
    draft = occurrence()
    uri = api(fs, DecisionLLM()).write_memory(draft, ctx)["target_uri"]
    duplicate = deepcopy(fs.nodes[uri])
    duplicate.uri += "_duplicate"
    fs.nodes[duplicate.uri] = duplicate
    before = deepcopy(fs.nodes)
    with pytest.raises(ValueError, match="ambiguous"):
        api(fs, DecisionLLM()).write_memory(draft, ctx)
    assert fs.nodes == before


@pytest.mark.parametrize("category", ["event", "case"])
@pytest.mark.parametrize("routing_key", ["inspection", "检查" * 30])
def test_long_scope_creates_independent_occurrences_and_replays(ctx, category, routing_key):
    fs = MemoryFS()
    scope = "Project " + "设备互联与运行状态追踪平台升级验证-" * 5
    draft = validate_candidate(occurrence(category, routing_key=routing_key, scope_ref=scope))
    writer = api(fs, DecisionLLM())
    first = writer.write_memory(draft, ctx)
    assert first["action"] == "create"
    uri = first["target_uri"]
    node = fs.read_node(uri, ctx)
    assert len(uri.rsplit("/", 1)[-1].encode("utf-8")) <= MAX_MEMORY_KEY_BYTES
    assert node.content == draft.content
    assert node.metadata["scope_ref"] == scope
    assert node.metadata["provenance_ids"] == draft.provenance_ids
    assert list(node.metadata["occurrence_replay_sources"].values()) == [draft.provenance_ids]

    other = replace(draft, scope_ref=scope + " B")
    second = writer.write_memory(other, ctx)
    assert second["action"] == "create"
    assert second["target_uri"] != uri
    assert fs.read_node(second["target_uri"], ctx).metadata["scope_ref"] == other.scope_ref
    before = deepcopy(fs.nodes)
    replay_writer = api(fs, DecisionLLM())
    for candidate, expected_uri in ((draft, uri), (other, second["target_uri"])):
        result = replay_writer.write_memory(
            replace(candidate, routing_key="inspection_report"), replace(ctx, session_id="retry"))
        assert result["action"] == "skip"
        assert result["target_uri"] == expected_uri
    assert fs.nodes == before
