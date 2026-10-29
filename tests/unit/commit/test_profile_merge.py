"""Profile updates reconcile a complete node through the public write API."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Lock

from core.models import ContextNode, RequestContext
from tests.unit.commit.test_memory_identity import MemoryFS, api, memory


def profile_context():
    return RequestContext(account_id="profile-test", user_id="reader", agent_id="assistant",
                          session_id="profile-update", trace_id="trace")


def profile(key, value):
    return replace(memory(key, category="profile", content=value), provenance_ids=[f"source:{key}"])


def result(content, *, abstract=None, overview=None):
    return {"content": content, "abstract": abstract or content, "overview": overview or content}


class ProfileLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def complete_json(self, prompt, schema):
        self.calls.append((json.loads(prompt.split("\n\n", 1)[1]), schema))
        return next(self.responses)


def test_profile_merges_same_key_facts_and_different_key_correction():
    fs, ctx = MemoryFS(), profile_context()
    initial = "Name: Nora. Occupation: technician."
    combined = initial + " Location: Utrecht."
    corrected = "Name: Nora. Occupation: engineer. Location: Utrecht."
    llm = ProfileLLM(result(initial), result(combined), result(corrected))
    writer = api(fs, llm)
    candidates = [profile("identity", "Name: Nora."),
                  profile("identity", "Occupation: technician."),
                  profile("identity", "Location: Utrecht."),
                  profile("job", "Occupation: engineer.")]
    for candidate in candidates:
        uri = writer.write_memory(candidate, ctx)["target_uri"]
    node = fs.read_node(uri, ctx)
    assert len(fs.nodes) == 1
    assert node.content == node.overview == node.abstract == corrected
    assert node.metadata["version"] == 4
    assert node.metadata["routing_key"] == "identity"
    assert node.metadata["provenance_ids"] == ["source:identity", "source:job"]
    assert len(llm.calls) == 3
    for call, previous, incoming in zip(llm.calls, ["Name: Nora.", initial, combined], candidates[1:]):
        payload, schema = call
        assert payload["existing"]["content"] == previous
        assert payload["incoming"]["content"] == incoming.content
        assert set(schema["required"]) == {"abstract", "overview", "content"}


def test_existing_profile_keeps_metadata_and_sources():
    fs, ctx = MemoryFS(), profile_context()
    uri = "ctx://profile-test/users/reader/memories/profile"
    old = "Name: Nora. Location: Utrecht."
    fs.write_node(ContextNode(
        uri=uri, context_type="MEMORY", category="profile", level=3,
        owner_space=ctx.user_space_name(), abstract="Nora in Utrecht.", overview=old, content=old,
        metadata={"routing_key": "identity", "created_at": "2026-01-02T00:00:00+00:00",
                  "provenance_ids": ["source:legacy"], "custom_tag": "imported"},
    ), ctx)
    merged = old + " Occupation: engineer."
    llm = ProfileLLM(result(merged))
    api(fs, llm).write_memory(profile("occupation", "Occupation: engineer."), ctx)
    node = fs.read_node(uri, ctx)
    assert node.content == merged
    assert set(node.metadata) == {"routing_key", "created_at", "updated_at", "version",
                                  "provenance_ids", "custom_tag"}
    assert node.metadata["custom_tag"] == "imported"
    assert node.metadata["created_at"] == "2026-01-02T00:00:00+00:00"
    assert node.metadata["provenance_ids"] == ["source:legacy", "source:occupation"]
    assert node.metadata["version"] == 2


def test_exact_profile_replay_keeps_sources_with_zero_model_calls():
    fs, ctx, llm = MemoryFS(), profile_context(), ProfileLLM()
    candidate = profile("identity", "Name: Nora.")
    uri = api(fs, llm).write_memory(candidate, ctx)["target_uri"]
    api(fs, llm).write_memory(replace(candidate, routing_key="name", provenance_ids=["source:other"]), ctx)
    node = fs.read_node(uri, ctx)
    assert len(llm.calls) == 0
    assert node.content == candidate.content
    assert node.metadata["provenance_ids"] == ["source:identity", "source:other"]


def test_profile_summaries_and_full_content_come_from_one_merge():
    fs, ctx = MemoryFS(), profile_context()
    old = "Supported earlier detail. " * 240
    new = "Supported later detail. " * 240
    expected = result(old + "\n\n" + new, abstract="Earlier and later details.",
                      overview="Overview covering both sources.")
    llm = ProfileLLM(expected)
    writer = api(fs, llm)
    first = replace(profile("identity", old), abstract="Earlier details.", overview="Earlier overview.")
    second = replace(profile("identity", new), abstract="Later details.", overview="Later overview.")
    writer.write_memory(first, ctx)
    uri = writer.write_memory(second, ctx)["target_uri"]
    node = fs.read_node(uri, ctx)
    assert {field: getattr(node, field) for field in expected} == expected
    assert len(llm.calls) == 1


def test_profile_applies_repeated_state_transition_in_one_session():
    fs, ctx = MemoryFS(), profile_context()
    name = "Name: Nora."
    technician = name + " Occupation: technician."
    engineer = name + " Occupation: engineer."
    llm = ProfileLLM(result(technician), result(engineer), result(technician))
    writer = api(fs, llm)
    writer.write_memory(profile("identity", name), ctx)
    for state in ("technician", "engineer", "technician"):
        uri = writer.write_memory(profile("occupation", f"Occupation: {state}."), ctx)["target_uri"]
    assert fs.read_node(uri, ctx).content == technician
    assert fs.read_node(uri, ctx).metadata["version"] == 4
    assert len(llm.calls) == 3


def test_concurrent_profile_creation_replans_with_committed_content():
    class ConcurrentFS(MemoryFS):
        def __init__(self):
            super().__init__()
            self.creates = Barrier(2)
            self.write_lock = Lock()

        def write_node(self, node, ctx):
            if node.metadata.get("_expected_version") == 0:
                self.creates.wait(timeout=5)
            with self.write_lock:
                super().write_node(node, ctx)

    fs, ctx = ConcurrentFS(), profile_context()
    combined = "Name: Nora. Location: Utrecht."
    llm = ProfileLLM(result(combined))
    candidates = [profile("identity", "Name: Nora."), profile("identity", "Location: Utrecht.")]
    writers = [api(fs, llm), api(fs, llm)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda pair: pair[0].write_memory(pair[1], ctx), zip(writers, candidates)))
    assert len({item["target_uri"] for item in results}) == 1
    node = fs.read_node(results[0]["target_uri"], ctx)
    assert node.content == combined
    assert node.metadata["version"] == 2
    assert len(llm.calls) == 1
    payload, _ = llm.calls[0]
    assert {payload["existing"]["content"], payload["incoming"]["content"]} == {
        candidate.content for candidate in candidates
    }
