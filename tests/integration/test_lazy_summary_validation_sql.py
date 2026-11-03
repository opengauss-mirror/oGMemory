"""Real SQL replay of lazy final-output summary validation and recovery."""

import json
import time
from copy import deepcopy

import pytest

from core.errors import ToolCallProtocolError
from core.models import ContextNode, SeedHit
from core.uri_resolver import URIResolver
from extraction.schemas.registry import SchemaRegistry
from service.api import MemoryWriteAPI
from service.extraction_checkpoint import EXTENSION_KEY
from session.session_state import SessionState
from tests.integration.test_session_mutation_sql import database as database, ctx as ctx, service_pair
from tests.unit.extraction.test_node_processing import node_payload
from tests.unit.extraction.test_prefetch import FakeEmbedder, FakeVectorIndex


OVERSIZED = ("Caroline draws belonging and purpose from LGBTQ+ community experiences, wants counseling/mental-health "
             "work, learns piano, plans advocacy involvement, and shares supportive friendship and nature-trip plans with Melanie.")


def replay_lazy_summary(make, ctx, operations, snapshots, source, *, recovery=False):
    service, old_writer, manager, archive, outbox = service_pair(make, ctx)
    manager._get_agfs = None
    fs = old_writer._fs
    operations, snapshots = deepcopy(operations), deepcopy(snapshots)
    old_prefix = next(item["input"]["target_uri"].split("/memories/")[0]
                      for item in operations if item["input"].get("target_uri"))
    prefix = f"ctx://{ctx.account_id}/users/{ctx.user_id}"
    operations = json.loads(json.dumps(operations).replace(old_prefix, prefix))
    snapshots = json.loads(json.dumps(snapshots).replace(old_prefix, prefix))
    for snapshot in snapshots:
        metadata = {key: value for key, value in snapshot["metadata"].items() if key != "expected_version"}
        node = ContextNode(snapshot["uri"], "MEMORY", "entity", 2, ctx.user_space_name(),
                           snapshot["abstract"], snapshot["overview"], snapshot["content"], metadata)
        fs.write_node(node, ctx)
        assert fs.read_node(node.uri, ctx).metadata["version"] == snapshot["metadata"]["version"]

    class Model:
        allow_repair = not recovery

        def __init__(self):
            self.calls = []
            self.feedback = []
            self.drafts = []

        def complete_json(self, prompt, schema=None, **kwargs):
            properties = (schema or {}).get("properties", {})
            if "spans" in properties:
                self.calls.append("spans")
                return {"spans": [{"start": 0, "end": 0, "categories": ["entity", "event"]}]}
            if "repairs" in properties:
                self.calls.append("repair")
                payload = json.loads(prompt.split("\n", 1)[1])
                self.feedback.append(payload)
                return {"repairs": [{"operation_index": item["operation_index"],
                                     **{field: "Caroline values community, counseling, music, and friendship."
                                        if self.allow_repair else OVERSIZED
                                        for field in item["fields_to_summarize"]}}
                                    for item in payload["operations"]]}
            if "groups" in properties:
                self.calls.append("event_groups")
                payload = json.loads(prompt.split("\n", 1)[1])
                return {"groups": [{"members": [index],
                                    **{field: draft.get(field) for field in
                                       ("abstract", "overview", "content", "title", "when", "who", "where")},
                                    "scope_ref": draft.get("scope_ref", "")}
                                   for index, draft in enumerate(payload["drafts"])]}
            if "operations" in properties:
                self.calls.append("fallback")
                raise ToolCallProtocolError("scripted fallback remains unavailable until durable retry")
            raise AssertionError(list(properties))

        def complete_with_tools_messages(self, messages, tools, tool_choice):
            self.calls.append("lazy_final" if tools is None else "lazy_read")
            if tools is None:
                return [], json.dumps(operations)
            return [{"id": "read-" + str(index), "name": "read", "input": {"uri": item["uri"]}}
                    for index, item in enumerate(snapshots)], ""

        def complete_with_tools(self, **kwargs):
            if not kwargs["prompt"].startswith("NODE_PROCESSING:"):
                self.calls.append("eager")
                return [{"tool": item["name"], "input": item["input"]} for item in operations]
            self.calls.append("nodes")
            payload = node_payload(kwargs)
            self.drafts.extend(deepcopy(payload["drafts"]))
            reads = {node["uri"]: node for node in payload["nodes"]}
            decisions = []
            for draft in payload["drafts"]:
                target = draft["target_uri"]
                content = draft["content"].replace("\n\nOriginal abstract detail:\n", "\n\n## Original abstract detail\n- ")
                decisions.append({"draft_ids": [draft["id"]], "action": "update" if target else "create",
                                  "target": target, "routing_key": draft["routing_key"],
                                  "object_name": draft["object_name"], "abstract": draft["abstract"],
                                  "overview": [draft["abstract"]],
                                  "edits": [{"search": reads[target]["content"], "replacement": content}] if target else [],
                                  "sections": [] if target else [{"heading": "Recorded facts", "facts": [
                                      line[2:] for line in content.splitlines() if line.startswith("- ")]}]})
            return [{"tool": "process_memory_nodes", "input": {"operations": decisions}}]

    model, registry = Model(), SchemaRegistry()
    index = FakeVectorIndex()
    index.hits = [SeedHit(uri=item["uri"], category="entity", abstract=item["abstract"], score=.99, level=2)
                  for item in snapshots]
    writer = MemoryWriteAPI(fs, model, outbox, schema_registry=registry, uri_resolver=URIResolver(registry),
                            vector_index=index, embedder=FakeEmbedder(), extraction_mode="lazy",
                            node_processing_enabled=True, card_format_enabled=True,
                            dual_pass_enabled=False, directory_summary_enabled=False)
    service._write_api = writer
    service._cfg.node_processing_enabled = service._cfg.card_format_enabled = True

    def state():
        return json.loads(fs.read_node(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx).content)

    try:
        response = service.after_turn({"_ctx": ctx, "sessionId": ctx.session_id, "wait": False,
                                       "messages": [{"role": "user", "content": source}]})
        assert manager.get_or_create(ctx.session_id).extraction_done_event.wait(10)
        first = state()
        first_batch = first.get("runtime_extensions", {}).get(EXTENSION_KEY, {}).get("batch")
        if recovery and first_batch and first_batch.get("retry_task"):
            model.allow_repair = True
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and EXTENSION_KEY in state().get("runtime_extensions", {}):
                time.sleep(.05)
        final = state()
        nodes = []
        for category in ("entity", "event"):
            directory = writer._writer._identity_resolver.uris.get_directory_uri(category, ctx)
            nodes.extend(fs.read_node(uri, ctx) for uri in fs.list_children(directory, ctx))
        target = next(node for node in nodes if node.uri == operations[0]["input"]["target_uri"])
        return {"response": response, "initial_batch": first_batch,
                "final_batch": final.get("runtime_extensions", {}).get(EXTENSION_KEY),
                "buffer_messages": len(final.get("buffer_messages", [])),
                "archives": len(archive.list_archives(ctx.session_id, ctx)), "node_count": len(nodes),
                "target_version": target.metadata["version"], "target_abstract": target.abstract,
                "original_body_preserved": operations[0]["input"]["content"] in target.content,
                "original_abstract_preserved": operations[0]["input"]["abstract"] in target.content,
                "calls": model.calls, "repair_feedback": model.feedback, "drafts": model.drafts}
    finally:
        service._extraction_retry.close(timeout=5)


@pytest.mark.parametrize("recovery", [False, True])
def test_lazy_final_summary_repair_preserves_authorized_body_and_settles(database, ctx, recovery):
    _, make = database
    uri = "ctx://old/users/reader/memories/entities/caroline"
    snapshot = {"uri": uri, "abstract": "Caroline learns piano.", "overview": "- Learns piano.",
                "content": "# Caroline\n\n## Activities\n- Learns piano.",
                "metadata": {"version": 2, "routing_key": "caroline", "object_name": "Caroline", "owner_scope": "user"}}
    body = snapshot["content"] + "\n- Plans a nature trip with Melanie."
    operation = {"name": "extract_entity", "input": {"target_uri": uri, "expected_version": 2,
                 "routing_key": "caroline", "object_name": "Caroline", "abstract": OVERSIZED,
                 "overview": "- Learns piano.\n- Plans a nature trip with Melanie.", "content": body, "confidence": .95}}
    result = replay_lazy_summary(make, ctx, [operation], [snapshot], "Caroline plans a nature trip with Melanie.",
                                 recovery=recovery)
    assert result["archives"] == result["node_count"] == 1
    assert result["target_version"] == 3
    assert result["buffer_messages"] == 0
    assert result["final_batch"] is None
    assert len(result["target_abstract"]) <= 200
    assert result["original_body_preserved"] and result["original_abstract_preserved"]
    if recovery:
        assert result["initial_batch"]["retry_task"]["status"] == "queued"
    else:
        assert result["calls"].count("repair") == 1
