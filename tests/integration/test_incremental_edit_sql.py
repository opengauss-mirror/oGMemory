"""Read-bound text updates retain identity through durable SQL writes."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from uuid import uuid4

import pytest

from core.errors import ConcurrentModificationError, ToolCallProtocolError
from core.models import RequestContext
from core.uri_resolver import URIResolver
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import parse_tool_call
from service.api import MemoryWriteAPI
from tests.integration.test_occurrence_replay_sql import close_stores, stores
from tests.unit.commit.test_memory_identity import DecisionLLM, memory
from tests.unit.commit.test_profile_merge import profile
from tests.unit.extraction.test_incremental_edit_flow import composed_profile_operations
from tests.unit.extraction.test_prefetch import FakeEmbedder, FakeVectorIndex


def context():
    return RequestContext("incremental-" + uuid4().hex, "reader", "agent", "first", "trace")


def writer(fs, outbox):
    return MemoryWriteAPI(fs, DecisionLLM(), outbox, directory_summary_enabled=False)


def original():
    return memory("阀门", object_name="阀门", scope_ref="Project A",
                  content="Pressure: 5 bar. Serial: 7.", who="Nora", where="Lab A")


def operation(node, *, text=None, search="5 bar", replacement="6 bar", summary=None):
    summary = summary or "Pressure: 6 bar. Serial: 7."
    fields = {"target_uri": node.uri, "expected_version": node.metadata["version"],
              "abstract": summary, "overview": summary, "confidence": .9}
    if text is None:
        fields.update(search=search, replacement=replacement)
        return "edit_memory", fields
    fields["text"] = text
    return "append_memory", fields


def candidate(node, ctx, **changes):
    name, fields = operation(node, **changes)
    parsed = parse_tool_call(name, fields, SchemaRegistry(), read_nodes={node.uri: node}, ctx=ctx)
    assert parsed is not None
    return parsed[2]


def test_edit_and_append_survive_reconnect_without_repeated_text():
    ctx = context()
    fs, outbox = stores()
    try:
        uri = writer(fs, outbox).write_memory(original(), ctx)["target_uri"]
        node = fs.read_node(uri, ctx)
        edited = replace(candidate(node, ctx), provenance_ids=["source:correction"])
        result = writer(fs, outbox).write_memory(edited, ctx)
        assert result["target_uri"] == uri
    finally:
        close_stores(fs, outbox)

    fs, outbox = stores()
    try:
        node = fs.read_node(uri, ctx)
        assert node.content == "Pressure: 6 bar. Serial: 7."
        assert node.metadata["version"] == 2
        label = node.metadata["routing_key"]
        extra = "Calibration passed."
        summary = "Pressure: 6 bar. Serial: 7. Calibration passed."
        appended = replace(candidate(node, ctx, text=extra, summary=summary),
                           provenance_ids=["source:calibration"])
        writer(fs, outbox).write_memory(appended, ctx)
    finally:
        close_stores(fs, outbox)

    fs, outbox = stores()
    try:
        node = fs.read_node(uri, ctx)
        assert node.content == "Pressure: 6 bar. Serial: 7.\nCalibration passed."
        assert node.metadata["version"] == 3
        assert node.metadata["routing_key"] == label
        assert (node.metadata["object_name"], node.metadata["scope_ref"],
                node.metadata["who"], node.metadata["where"]) == ("阀门", "Project A", "Nora", "Lab A")
        assert set(node.metadata["provenance_ids"]) == {
            "prov:1:archive:sample:message-one", "source:correction", "source:calibration"}
        before = len(outbox.list_pending(ctx.account_id))
        replay = replace(candidate(node, ctx, text=extra, summary=summary),
                         provenance_ids=["source:calibration"])
        assert writer(fs, outbox).write_memory(replay, ctx)["action"] == "skip"
        assert fs.read_node(uri, ctx).metadata["version"] == 3
        assert len(outbox.list_pending(ctx.account_id)) == before == 3
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("user_id", ["reader", "user_2", "élodie", "用户"])
def test_same_principal_profile_edit_and_append_survive_sql_reconnect(user_id):
    ctx = replace(context(), user_id=user_id)
    body = "Residence: Utrecht. Occupation: technician."
    fs, outbox = stores()
    try:
        uri = writer(fs, outbox).write_memory(profile("identity", body), ctx)["target_uri"]
        node = fs.read_node(uri, ctx)
        name, fields = operation(node, search="technician", replacement="engineer",
                                 summary="Residence: Utrecht. Occupation: engineer.")
        fields.update(evidence_quote="I am an engineer.", attributed_speaker="user",
                      attribution_basis="self_first_person")
        parsed = parse_tool_call(name, fields, SchemaRegistry(), read_nodes={uri: node}, ctx=ctx)
        assert writer(fs, outbox).write_memory(replace(parsed[2], provenance_ids=["source:edit"]), ctx)["target_uri"] == uri
    finally:
        close_stores(fs, outbox)

    fs, outbox = stores()
    try:
        node = fs.read_node(uri, ctx)
        assert node.content == body.replace("technician", "engineer")
        assert node.metadata["version"] == 2
        name, fields = operation(node, text="Keeps a checklist.",
                                 summary="Lives in Utrecht, works as an engineer and keeps a checklist.")
        fields.update(evidence_quote="I keep a checklist.", attributed_speaker="user",
                      attribution_basis="self_first_person")
        parsed = parse_tool_call(name, fields, SchemaRegistry(), read_nodes={uri: node}, ctx=ctx)
        assert writer(fs, outbox).write_memory(replace(parsed[2], provenance_ids=["source:append"]), ctx)["target_uri"] == uri
    finally:
        close_stores(fs, outbox)

    fs, outbox = stores()
    try:
        stored = fs.read_node(uri, ctx)
        assert stored.content == body.replace("technician", "engineer") + "\nKeeps a checklist."
        assert (stored.uri, stored.owner_space, stored.category, stored.context_type) == (
            uri, ctx.user_space_name(), "profile", "MEMORY")
        assert stored.metadata["routing_key"] == "identity" and stored.metadata["version"] == 3
        assert set(stored.metadata["provenance_ids"]) == {"source:identity", "source:edit", "source:append"}
        assert len(outbox.list_pending(ctx.account_id)) == 3
    finally:
        close_stores(fs, outbox)


def test_two_connections_cannot_overwrite_each_others_read_version():
    ctx = context()
    fs, outbox = stores()
    try:
        uri = writer(fs, outbox).write_memory(original(), ctx)["target_uri"]
        node = fs.read_node(uri, ctx)
    finally:
        close_stores(fs, outbox)
    updates = [candidate(node, ctx), candidate(node, ctx, search="Serial: 7", replacement="Serial: 8",
                                              summary="Pressure: 5 bar. Serial: 8.")]
    barrier = Barrier(2)

    def submit(update):
        fs, outbox = stores()
        try:
            target = writer(fs, outbox)
            barrier.wait(timeout=10)
            try:
                return "written", target.write_memory(update, ctx)
            except ConcurrentModificationError:
                return "conflict", None
            except ValueError as exc:
                assert "changed since extraction" in str(exc)
                return "conflict", None
        finally:
            close_stores(fs, outbox)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, updates))
    assert sorted(status for status, _ in results) == ["conflict", "written"]
    fs, outbox = stores()
    try:
        stored = fs.read_node(uri, ctx)
        assert stored.metadata["version"] == 2
        assert stored.content in {update.content for update in updates}
        assert len(outbox.list_pending(ctx.account_id)) == 2
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("category", ["profile", "preference", "pattern"])
def test_supported_schema_update_preserves_facts_after_sql_reconnect(category):
    ctx = context()
    body = "Nora reviews notes every Friday."
    first = profile("identity", body) if category == "profile" else memory(
        "weekly_review", category=category, content=body)
    fs, outbox = stores()
    try:
        uri = writer(fs, outbox).write_memory(first, ctx)["target_uri"]
        node = fs.read_node(uri, ctx)
        _, fields = operation(node, text="Keeps a checklist.",
                              summary="Nora reviews notes every Friday and keeps a checklist.")
        fields.update(evidence_quote="I also keep a checklist.", attributed_speaker="user",
                      attribution_basis="self_first_person")
        parsed = parse_tool_call("append_memory", fields, SchemaRegistry(),
                                 read_nodes={uri: node}, ctx=ctx)
        assert parsed is not None
        writer(fs, outbox).write_memory(replace(parsed[2], provenance_ids=["source:checklist"]), ctx)
        identity, owner = node.metadata["routing_key"], node.owner_space
    finally:
        close_stores(fs, outbox)
    fs, outbox = stores()
    try:
        stored = fs.read_node(uri, ctx)
        assert stored.content == body + "\nKeeps a checklist."
        assert stored.category == category and stored.owner_space == owner
        assert stored.metadata["routing_key"] == identity and stored.metadata["version"] == 2
        assert stored.context_type == ("SKILL" if category == "pattern" else "MEMORY")
        assert "source:checklist" in stored.metadata["provenance_ids"]
        assert len(outbox.list_pending(ctx.account_id)) == 2
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("case,threshold", [("attribution", .5), ("low", .5), ("custom", .85)])
def test_public_session_cannot_promote_rejected_parts_of_composed_edit(case, threshold):
    class SessionProvider:
        operations = []

        def complete_json(self, prompt, schema):
            if "spans" in schema.get("properties", {}):
                return {"spans": [{"start": 0, "end": 0, "reason": "current facts",
                                   "categories": ["profile"]}]}
            return {"operations": []}

        def complete_with_tools(self, **kwargs):
            return [{"tool": item["name"], "input": item["input"]} for item in self.operations]

    ctx = context()
    initial = "Name: Nora. Residence: Utrecht. Occupation: technician."
    fs, outbox = stores()
    try:
        provider = SessionProvider()
        registry = SchemaRegistry()
        target = MemoryWriteAPI(fs, provider, outbox, schema_registry=registry,
                               vector_index=FakeVectorIndex(), embedder=FakeEmbedder(),
                               uri_resolver=URIResolver(registry), directory_summary_enabled=False)
        uri = target.write_memory(profile("identity", initial), ctx)["target_uri"]
        provider.operations = composed_profile_operations(
            uri, third_party=case == "attribution", confidence=.8 if case == "custom" else .1)
        result = target.commit_session(
            [{"id": "m1", "role": "user", "content": "Bob lives in Lisbon. I might move to Lisbon. I now work as an engineer."}],
            ctx, archive_id="incremental-guard", confidence_threshold=threshold)
        assert result["writes_completed"] == 0 and result["writes_failed"] == 0
    finally:
        close_stores(fs, outbox)
    fs, outbox = stores()
    try:
        stored = fs.read_node(uri, ctx)
        assert stored.content == initial and stored.metadata["version"] == 1
        assert len(outbox.list_pending(ctx.account_id)) == 1
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("bad", ["unread", "version", "identity", "owner", "ambiguous"])
def test_rejected_incremental_operation_does_not_write_or_enqueue(bad):
    ctx = context()
    fs, outbox = stores()
    try:
        first = original()
        if bad == "ambiguous":
            first = replace(first, content="Pressure: 5 bar. Alarm at 5 bar.")
        uri = writer(fs, outbox).write_memory(first, ctx)["target_uri"]
        node = fs.read_node(uri, ctx)
        name, fields = operation(node)
        snapshots = {uri: node}
        request = ctx
        if bad == "unread":
            snapshots = {}
        elif bad == "version":
            fields["expected_version"] += 1
        elif bad == "identity":
            fields["scope_ref"] = "Project B"
        elif bad == "owner":
            request = replace(ctx, user_id="other")
        try:
            parsed = parse_tool_call(name, fields, SchemaRegistry(), read_nodes=snapshots, ctx=request)
        except ToolCallProtocolError:
            parsed = None
        assert parsed is None
        assert fs.read_node(uri, ctx).content == first.content
        assert fs.read_node(uri, ctx).metadata["version"] == 1
        assert len(outbox.list_pending(ctx.account_id)) == 1
    finally:
        close_stores(fs, outbox)
