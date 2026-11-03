"""Maintain complete units and source history across repeated trace writes."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from jsonschema import validate

from commit.agent_memory import AgentMemoryArchiveBuilder, AgentMemoryPolicy
from core.uri_resolver import URIResolver
from extraction.agent_memory import AgentMemoryExtractor, agent_memory_registry, memory_units
from tests.unit.server.test_agent_memory import (
    CTX,
    ExtractionLLM,
    NodeFS,
    _MemoryAGFS,
    finalized_trace,
)


def _candidate(fs, client, llm, ctx, category):
    store, turn = finalized_trace(client, ctx)
    page = store.read_skill_trace_event_page(
        account_id=ctx.account_id, evidence_uri=turn["uri"], after_ordinal=0, limit=32,
    )
    return next(c for c in AgentMemoryExtractor(fs, llm).extract_candidates(
        turn["uri"], page["events"], ctx,
    ) if c.category == category)


def _replay(history):
    item, grounding = None, []
    for entry in history:
        if "added_grounding" in entry:
            grounding.extend(deepcopy(entry["added_grounding"]))
        else:
            item, grounding = deepcopy(entry["after"]), deepcopy(entry["after_grounding"])
        if item is not None:
            item["grounding"] = deepcopy(grounding)
    return item


@pytest.mark.parametrize("action", ["correction", "remove"])
def test_hidden_tool_unit_is_refreshed_and_committed(action):
    client, llm = _MemoryAGFS(), ExtractionLLM()
    fs = NodeFS(client)
    candidate = _candidate(fs, client, llm, CTX, "tool")
    original = candidate.structured_fields["units"][0]
    candidate.structured_fields["units"] = [
        {**deepcopy(original), "key": key, "content": (key + "约束" * 4000)[:8000]}
        for key in ("alpha", "beta")
    ]
    validate({"changes": [{"routing_key": candidate.routing_key, **candidate.structured_fields}]},
             AgentMemoryExtractor(fs, llm)._response_schema(memory_type="tool"))
    candidate.memory_context["grounding"] = {
        key: deepcopy(candidate.memory_context["grounding"]["batch_limit"])
        for key in ("alpha", "beta")
    }
    policy = AgentMemoryPolicy(fs, URIResolver(agent_memory_registry()))
    builder = AgentMemoryArchiveBuilder(llm)
    initial = builder.build(candidate, policy.plan(candidate, CTX), CTX)
    fs.write_node(initial, CTX)

    class MaintenanceLLM(ExtractionLLM):
        def complete_with_tools(self, prompt, tools, tool_choice):
            calls = super().complete_with_tools(prompt, tools, tool_choice)
            if tools[0]["name"] == "maintain_agent_memory" and prompt.startswith("Extract only tool;"):
                change = calls[0]["input"]["changes"][0]
                unit = {**change["units"][0], "key": "beta", "content": "Use the revised beta limit."}
                change.update(
                    units=[unit] if action == "correction" else [],
                    corrections=[{"key": "beta", "reason": "The current contract updates beta."}]
                    if action == "correction" else [],
                    removals=[{"key": "beta", "reason": "The current contract withdraws beta.",
                               "grounding": unit["grounding"]}] if action == "remove" else [],
                )
            return calls

    llm = MaintenanceLLM()
    ctx = replace(CTX, session_id="maintenance")
    change = _candidate(fs, client, llm, ctx, "tool")
    prompts = [prompt for name, prompt in llm.calls
               if name == "maintain_agent_memory" and prompt.startswith("Extract only tool;")]
    assert len(prompts) == 2
    snapshot = json.loads(prompts[1].split("Current maintenance state:\n")[1]
                          .split("\nPrevious proposal")[0])[0]
    assert [unit["key"] for unit in snapshot["fields"]["units"]] == ["beta"]
    assert snapshot["fields"]["units"][0]["content"] == candidate.structured_fields["units"][1]["content"]
    assert snapshot["unit_revisions"] == {"alpha": 1, "beta": 1}

    updated = builder.build(change, policy.plan(change, ctx), ctx)
    fs.write_node(updated, ctx)
    saved = fs.read_node(initial.uri, ctx)
    units = saved.metadata["structured_fields"]["units"]
    assert units[0] == initial.metadata["structured_fields"]["units"][0]
    assert saved.metadata["agent_memory"]["units"]["beta"]["revision"] == 2
    assert saved.metadata["agent_memory"]["revisions"][-1]["action"] == action
    if action == "correction":
        assert units[1]["content"] == "Use the revised beta limit."
    else:
        assert [unit["key"] for unit in units] == ["alpha"]
        assert saved.metadata["agent_memory"]["units"]["beta"]["withdrawn"] is True


@pytest.mark.parametrize("category", ["tool", "experience"])
def test_repeated_evidence_preserves_sources_with_linear_history(category):
    client, llm = _MemoryAGFS(), ExtractionLLM()
    fs = NodeFS(client)
    policy = AgentMemoryPolicy(fs, URIResolver(agent_memory_registry()))
    builder = AgentMemoryArchiveBuilder(llm)
    sizes, sources = {}, []
    for number in range(32):
        ctx = replace(CTX, session_id=f"evidence-{number:02d}")
        candidate = _candidate(fs, client, llm, ctx, category)
        key = memory_units(category, candidate.structured_fields)[0]["key"]
        sources.extend(candidate.memory_context["grounding"][key])
        plan = policy.plan(candidate, ctx)
        node = builder.build(candidate, plan, ctx)
        fs.write_node(node, ctx)
        if number + 1 in (16, 32):
            sizes[number + 1] = len(json.dumps(node.metadata, ensure_ascii=False).encode())
    saved = fs.read_node(node.uri, ctx)
    state = saved.metadata["agent_memory"]
    assert state["units"][key]["revision"] == 32
    assert state["units"][key]["grounding"] == sources
    assert memory_units(category, saved.metadata["structured_fields"])[0]["grounding"] == sources
    assert len(state["revisions"]) == 32
    assert len(state["applied_operations"]) == 32
    assert _replay(state["revisions"]) == memory_units(category, saved.metadata["structured_fields"])[0]
    assert policy.plan(candidate, ctx).action == "skip"
    print(f"{category} metadata bytes: {sizes}; doubling ratio={sizes[32] / sizes[16]:.3f}")
    assert sizes[32] < sizes[16] * 2.5


@pytest.mark.parametrize("category", ["tool", "experience"])
def test_source_deltas_preserve_legacy_history_and_content_revisions(category):
    client, llm = _MemoryAGFS(), ExtractionLLM()
    fs = NodeFS(client)
    policy = AgentMemoryPolicy(fs, URIResolver(agent_memory_registry()))
    builder = AgentMemoryArchiveBuilder(llm)

    def save(candidate):
        node = builder.build(candidate, policy.plan(candidate, ctx), ctx)
        fs.write_node(node, ctx)
        return fs.read_node(node.uri, ctx)

    for number in range(3):
        ctx = replace(CTX, session_id=f"legacy-{number}")
        candidate = _candidate(fs, client, llm, ctx, category)
        node = save(candidate)
        unit = memory_units(category, node.metadata["structured_fields"])[0]
        if number == 1:
            state = node.metadata["agent_memory"]
            entry = state["revisions"][-1]
            entry.pop("added_grounding")
            entry.update(before=deepcopy(state["revisions"][0]["after"]), after=deepcopy(unit),
                         before_grounding=deepcopy(state["revisions"][0]["after_grounding"]),
                         after_grounding=deepcopy(unit["grounding"]))
            legacy_history = deepcopy(state["revisions"])
            fs.write_node(node, ctx)
    state = node.metadata["agent_memory"]
    assert state["revisions"][:2] == legacy_history
    assert _replay(state["revisions"]) == unit
    key = unit["key"]
    assert state["revisions"][-1]["added_grounding"] == candidate.memory_context["grounding"][key]

    memory_units(category, candidate.structured_fields)[0]["content"] = "Apply the revised API limit."
    candidate.memory_context.update(operation_id="correct", base_units={key: 3},
                                    corrections=[{"key": key, "reason": "The contract updates this limit."}])
    corrected = save(candidate)
    revision = corrected.metadata["agent_memory"]["revisions"][-1]
    assert revision["before"] == unit
    assert revision["before_grounding"] == unit["grounding"]
    assert revision["after"]["content"] == "Apply the revised API limit."
    assert revision["after_grounding"] == candidate.memory_context["grounding"][key]
    assert revision["reason"] == "The contract updates this limit."

    withdrawn = deepcopy(candidate)
    withdrawn.structured_fields = ({"tool_name": candidate.structured_fields["tool_name"], "units": []}
                                   if category == "tool" else {})
    withdrawn.memory_context.update(operation_id="withdraw", base_units={key: 4}, corrections=[],
        removals=[{"key": key, "reason": "The feature was withdrawn.",
                   "grounding": memory_units(category, candidate.structured_fields)[0]["grounding"]}])
    removed = save(withdrawn)
    state = removed.metadata["agent_memory"]
    assert _replay(state["revisions"]) is None
    assert state["revisions"][-1]["before"] == revision["after"]
    assert state["revisions"][-1]["reason"] == "The feature was withdrawn."

    candidate.memory_context.update(operation_id="revive", base_units={key: 5},
                                    corrections=[{"key": key, "reason": "The feature is available again."}])
    revived = save(candidate)
    state = revived.metadata["agent_memory"]
    assert state["units"][key]["revision"] == 6
    assert _replay(state["revisions"]) == memory_units(category, revived.metadata["structured_fields"])[0]
    assert state["revisions"][:2] == legacy_history
    candidate.memory_context.update(operation_id="same-source", base_units={key: 6}, corrections=[])
    assert policy.plan(candidate, ctx).action == "skip"
