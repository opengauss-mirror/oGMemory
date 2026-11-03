"""Qualified occurrence updates use durable versioned writes and atomic outbox."""

from dataclasses import replace
from uuid import uuid4

import pytest

from core.errors import ExtractionIncompleteError
from core.models import CandidateMemory, RequestContext
from core.uri_resolver import URIResolver
from extraction.occurrence_qualification import qualify_occurrence_drafts
from extraction.schemas.registry import SchemaRegistry
from service.api import MemoryWriteAPI
from tests.integration.test_occurrence_replay_sql import close_stores, stores

SOURCE = "Add to that Project Orion inspection: the approved repair took two hours."


class QualificationLLM:
    def __init__(self, *, valid=True, evidence=SOURCE, snapshot_overrides=None):
        self.valid = valid
        self.evidence = evidence
        self.snapshot_overrides = snapshot_overrides or {}

    def complete_json(self, prompt, schema):
        if "decision" in schema["properties"]:
            return {
                "decision": "same", "target_index": 0, "identity_evidence": self.evidence,
                "snapshot": {
                    "abstract": "Inspection and two-hour approved repair",
                    "overview": "The gauge measured 5 bar. The approved repair took two hours.",
                    "content": "Nora inspected the gauge in Lab A. It measured 5 bar. "
                               "The approved repair took two hours.",
                    "title": "Inspection and approved repair", "when": "2026-10-03",
                    "who": "Nora", "where": "Lab A",
                    **self.snapshot_overrides,
                },
            }
        return {"valid": self.valid}


def draft(category):
    return CandidateMemory(
        category=category, owner_scope="agent" if category == "case" else "user",
        routing_key="gauge_inspection", abstract="Nora inspected the gauge",
        overview="The gauge measured 5 bar.",
        content="Nora inspected the gauge in Lab A. It measured 5 bar.", confidence=.9,
        title="Gauge inspection", scope_ref="Project Orion", when="2026-10-03",
        who="Nora", where="Lab A", provenance_ids=["source:inspection"],
    )


def qualify(fs, ctx, first, uri, *, valid=True):
    schemas = SchemaRegistry()
    incoming = replace(first, routing_key="approved_repair", title="Repair duration",
                       abstract="The approved repair took two hours.",
                       overview="The approved repair took two hours.",
                       content="The approved repair took two hours.",
                       provenance_ids=["source:repair"])
    return qualify_occurrence_drafts(
        [incoming], "Session Time: 2026-10-04\nuser: " + SOURCE, SOURCE, {uri: 1},
        fs, URIResolver(schemas), schemas, ctx, QualificationLLM(valid=valid),
    )[0]


@pytest.mark.parametrize("category", ["event", "case"])
def test_qualified_update_survives_sql_reconnect_with_one_identity(category):
    ctx = RequestContext("qualification-" + uuid4().hex, "reader", "agent", "first", "trace")
    fs, outbox = stores()
    first = draft(category)
    try:
        writer = MemoryWriteAPI(fs, QualificationLLM(), outbox, directory_summary_enabled=False)
        uri = writer.write_memory(first, ctx)["target_uri"]
        qualified = qualify(fs, replace(ctx, session_id="follow-up"), first, uri)
        assert qualified.target_uri == uri and qualified.expected_version == 1
        result = writer.write_memory(qualified, replace(ctx, session_id="follow-up"))
        assert result["target_uri"] == uri and result["action"] == "merge"
    finally:
        close_stores(fs, outbox)

    fs, outbox = stores()
    try:
        stored = fs.read_node(uri, ctx)
        assert stored.content == qualified.content
        assert "5 bar" in stored.content and "two hours" in stored.content
        assert stored.metadata["title"] == "Inspection and approved repair"
        assert stored.metadata["scope_ref"] == "Project Orion"
        assert stored.metadata["version"] == 2
        assert set(stored.metadata["provenance_ids"]) == {"source:inspection", "source:repair"}
        assert len(fs.list_children(uri.rsplit("/", 1)[0], ctx)) == 1
        assert len(outbox.list_pending(ctx.account_id)) == 2
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("category", ["event", "case"])
def test_sql_competing_write_cannot_be_overwritten_by_qualified_old_snapshot(category):
    ctx = RequestContext("qualification-race-" + uuid4().hex, "reader", "agent", "first", "trace")
    fs, outbox = stores()
    other_fs, other_outbox = stores()
    try:
        first = draft(category)
        writer = MemoryWriteAPI(fs, QualificationLLM(), outbox, directory_summary_enabled=False)
        uri = writer.write_memory(first, ctx)["target_uri"]
        qualified = qualify(fs, ctx, first, uri)
        competing = replace(first, target_uri=uri, expected_version=1,
                            content=first.content + " A safety review was also completed.",
                            provenance_ids=["source:safety-review"])
        other = MemoryWriteAPI(other_fs, QualificationLLM(), other_outbox,
                               directory_summary_enabled=False)
        other.write_memory(competing, replace(ctx, session_id="other-session"))

        with pytest.raises(ValueError, match="memory changed since extraction"):
            writer.write_memory(qualified, replace(ctx, session_id="follow-up"))

        stored = fs.read_node(uri, ctx)
        assert stored.content == competing.content
        assert stored.metadata["version"] == 2
        assert len(fs.list_children(uri.rsplit("/", 1)[0], ctx)) == 1
        assert len(outbox.list_pending(ctx.account_id)) == 2
    finally:
        close_stores(fs, outbox)
        close_stores(other_fs, other_outbox)


@pytest.mark.parametrize("category", ["event", "case"])
def test_unverified_qualification_leaves_sql_node_and_outbox_unchanged(category):
    ctx = RequestContext("qualification-invalid-" + uuid4().hex, "reader", "agent", "first", "trace")
    fs, outbox = stores()
    try:
        first = draft(category)
        writer = MemoryWriteAPI(fs, QualificationLLM(), outbox, directory_summary_enabled=False)
        uri = writer.write_memory(first, ctx)["target_uri"]
        with pytest.raises(ExtractionIncompleteError):
            qualify(fs, ctx, first, uri, valid=False)

        assert fs.read_node(uri, ctx).metadata["version"] == 1
        assert fs.read_node(uri, ctx).content == first.content
        assert len(outbox.list_pending(ctx.account_id)) == 1
    finally:
        close_stores(fs, outbox)


@pytest.mark.parametrize("category", ["event", "case"])
def test_qualified_location_withdrawal_clears_sql_context_and_survives_reconnect(category):
    ctx = RequestContext("qualification-clear-" + uuid4().hex, "reader", "agent", "first", "trace")
    fs, outbox = stores()
    source = "For that Project Orion gauge inspection on October 3, Lab A was incorrect; location is unknown."
    llm = QualificationLLM(evidence=source, snapshot_overrides={
        "where": None, "title": "Gauge inspection with location unconfirmed",
        "abstract": "Nora's gauge inspection recorded 5 bar; location unconfirmed.",
        "overview": "Nora recorded 5 bar on October 3. The location is unknown.",
        "content": "Nora inspected the gauge on October 3 and recorded 5 bar. The location is unknown.",
    })
    try:
        first = draft(category)
        writer = MemoryWriteAPI(fs, llm, outbox, directory_summary_enabled=False)
        uri = writer.write_memory(first, ctx)["target_uri"]
        schemas = SchemaRegistry()
        incoming = replace(first, where=None, content=source, provenance_ids=["source:location-correction"])
        qualified, = qualify_occurrence_drafts(
            [incoming], "user: " + source, source, {uri: 1}, fs,
            URIResolver(schemas), schemas, ctx, llm,
        )
        writer.write_memory(qualified, replace(ctx, session_id="follow-up"))
    finally:
        close_stores(fs, outbox)

    fs, outbox = stores()
    try:
        stored = fs.read_node(uri, ctx)
        assert "where" not in stored.metadata
        assert stored.metadata["when"] == first.when and stored.metadata["who"] == first.who
        assert stored.metadata["version"] == 2
        assert stored.content == qualified.content
        assert set(stored.metadata["provenance_ids"]) == {"source:inspection", "source:location-correction"}
        assert len(outbox.list_pending(ctx.account_id)) == 2
    finally:
        close_stores(fs, outbox)
