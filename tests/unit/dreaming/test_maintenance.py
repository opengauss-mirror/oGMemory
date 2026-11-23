import json
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from commit.dreaming import Dreaming, maintenance_event, recall_eligible, review
from commit.dreaming_facts import correction_plan
from commit.dreaming_promotion import (
    DEMOTION, PROMOTION, _identity, _peers, _promotion_destination, _publish, _witnesses, reconcile_promotion,
)
from commit.outbox_store import OutboxStore
from commit.sql_outbox_store import SQLOutboxStore
from core.errors import NodeNotFoundError
from core.models import ContextNode, OutboxEvent, RequestContext
from index.outbox_worker import OutboxWorker
from tests.unit.dreaming.test_structured_evidence import (
    _archive,
    _bind,
    _evidence,
    _message,
    _node,
    _reviewed_berth,
)


def test_demotion_saves_its_checkpoint_with_sql_outbox():
    node, _ = _reviewed_berth("A7")
    node.metadata[PROMOTION] = {"parents": []}
    ctx = RequestContext("account", "user", "agent", "session", "request")
    fs = Mock()
    outbox = Mock(spec=SQLOutboxStore)
    event = maintenance_event(node.uri, node.metadata["version"])
    with patch("commit.dreaming_promotion.resume_promotion") as resume:
        reconcile_promotion(fs, outbox, node, ctx, event=event)

    outbox.save_payload.assert_called_once_with(event)
    assert event.payload[DEMOTION] == node.metadata["version"]
    resume.assert_called_once_with(fs, outbox, event, ctx)


def test_demotion_retry_preserves_the_recreated_node():
    node, _ = _reviewed_berth("A7")
    original, recreated = "2026-06-01T00:00:00+00:00", "2026-06-02T00:00:00+00:00"
    node.metadata.update(created_at=recreated, dreaming_promotion={"parents": []})
    ctx = RequestContext("account", "user", "agent", "session", "request")
    event = maintenance_event(node.uri, 1, created_at=original)
    event.payload[DEMOTION] = 1
    fs, outbox = Mock(), Mock()
    nodes = {node.uri: node}
    fs.read_node.side_effect = lambda uri, _: nodes[uri]
    fs.demote_dreaming_node.side_effect = lambda target, *_: nodes.pop(target.uri)

    Dreaming(fs, outbox=outbox, enqueue_after_projection=True).process(event, ctx)

    assert fs.read_node(node.uri, ctx) is node


def test_rebuilt_promotion_gets_a_fresh_projection():
    original, _ = _reviewed_berth("A7")
    fact = {"content": original.content}
    witnesses = [{"uri": original.uri, "weight": 4, "occurrence": source, "source_ref": source}
                 for source in ("first", "second")]
    ctx = RequestContext("account", "user", "agent", "session", "request")
    fs, outbox = Mock(), Mock()
    fs.archived_node_version.return_value = None
    fs.memory_identity_lock.return_value = nullcontext()
    generations = iter(("2026-06-01T00:00:00+00:00", "2026-06-02T00:00:00+00:00"))
    nodes = {}

    def read(uri, _):
        if uri not in nodes:
            raise NodeNotFoundError(uri)
        return nodes[uri]

    def create(node, _):
        node.metadata.update(version=1, created_at=next(generations))
        nodes[node.uri] = node

    fs.read_node.side_effect = read
    fs.write_node.side_effect = create
    projections = {}
    outbox.build_write_event.side_effect = lambda node: OutboxEvent("", "UPSERT_CONTEXT", node.uri, {}, "PENDING")
    outbox.register_once.side_effect = lambda event, _: projections.setdefault(event.event_id, event.uri)
    with patch("commit.dreaming_promotion._supports", return_value=True):
        _publish(fs, outbox, original, fact, witnesses, ("pattern", "patterns"), ctx)
        uri, = nodes
        del nodes[uri]
        for _ in range(2):
            _publish(fs, outbox, original, fact, witnesses, ("pattern", "patterns"), ctx)

    assert list(projections.values()) == [uri, uri]


def test_promotion_retries_when_witness_revalidation_recovers():
    source, _ = _reviewed_berth("A7")
    source.uri = "ctx://account/users/user/memories/events/dock"
    source.category = "event"
    fact = {"content": source.content, "evidence": [{"supports": True}]}
    source.metadata["dreaming_review"] = {"version": 1, "facts": {"berth": fact}}
    witnesses = [{"uri": source.uri, "weight": 4, "occurrence": ref, "source_ref": ref, "source_uri": ref}
                 for ref in ("first", "second")]
    ctx = RequestContext("account", "user", "agent", "session", "request")
    fs, outbox = Mock(), Mock()
    fs.memory_identity_lock.side_effect = lambda *_: nullcontext()
    fs.archived_node_version.return_value = None
    fs.list_identity_node_summaries_page.return_value = []
    fs.read_node.side_effect = [NodeNotFoundError("promotion destination"), source]
    outbox.build_write_event.return_value = OutboxEvent("", "UPSERT_CONTEXT", source.uri, {}, "PENDING")
    with (patch("commit.dreaming_promotion._supports", side_effect=[False, True]),
          patch("commit.dreaming_promotion._witnesses", side_effect=lambda *_a, **_kw: iter(witnesses))):
        with pytest.raises(RuntimeError, match="promotion evidence changed"):
            reconcile_promotion(fs, outbox, source, ctx)
        reconcile_promotion(fs, outbox, source, ctx)

    fs.write_node.assert_called_once()
    outbox.register_once.assert_called_once()


def test_promotion_retries_until_archive_projection_finishes():
    source, _ = _reviewed_berth("A7")
    fact = {"content": source.content}
    witnesses = [{"uri": source.uri, "weight": 4, "occurrence": ref, "source_ref": ref}
                 for ref in ("first", "second")]
    ctx = RequestContext("account", "user", "agent", "session", "request")
    fs, outbox = Mock(), Mock()
    fs.memory_identity_lock.return_value = nullcontext()
    fs.archived_node_version.return_value = 1
    fs.read_node_metadata_including_inactive.return_value = {DEMOTION: "archive", "version": 1}
    nodes = {}

    def read(uri, _):
        if uri not in nodes:
            raise NodeNotFoundError(uri)
        return nodes[uri]

    def publish(node, _):
        node.metadata["version"] = node.metadata["_expected_version"] + 1
        nodes[node.uri] = node

    fs.read_node.side_effect = read
    fs.write_node.side_effect = publish
    outbox.get_event_status.side_effect = ["PROCESSING", "DONE"]
    outbox.build_write_event.side_effect = lambda node: OutboxEvent("", "UPSERT_CONTEXT", node.uri, {}, "PENDING")
    with patch("commit.dreaming_promotion._supports", return_value=True):
        with pytest.raises(RuntimeError, match="archive projection"):
            _publish(fs, outbox, source, fact, witnesses, ("pattern", "patterns"), ctx)
        _publish(fs, outbox, source, fact, witnesses, ("pattern", "patterns"), ctx)

    promoted, = nodes.values()
    projection, _ = outbox.register_once.call_args.args
    assert (promoted.metadata["version"], projection.uri) == (2, promoted.uri)


@pytest.mark.parametrize("status", ["PENDING", "FAILED"])
def test_archive_block_preserves_other_promotion_in_the_same_event(status):
    node = _node("Alice", "berth A7", "cabinet B6", category="event")
    node.uri = "ctx://account/users/user/memories/events/dock"
    blocked = {"content": "berth A7", "evidence": [{"supports": True}]}
    healthy = {"content": "cabinet B6", "evidence": [{"supports": True}]}
    node.metadata["dreaming_review"] = {"version": 1, "facts": {"berth": blocked, "cabinet": healthy}}
    blocked_uri, _ = _promotion_destination(node, _identity(node, blocked), "patterns")
    healthy_uri, _ = _promotion_destination(node, _identity(node, healthy), "patterns")
    witnesses = [{"uri": node.uri, "weight": 4, "occurrence": ref, "source_ref": ref, "source_uri": ref}
                 for ref in ("first", "second")]
    fs, outbox, manager = Mock(), Mock(), Mock()
    written = {}

    def read(uri, _):
        if uri not in written:
            raise NodeNotFoundError(uri)
        return written[uri]

    def write(node, _):
        node.metadata["version"] = 1
        written[node.uri] = node

    fs.read_node.side_effect = read
    fs.write_node.side_effect = write
    fs.memory_identity_lock.side_effect = lambda *_: nullcontext()
    fs.archived_node_version.side_effect = lambda uri, _: 1 if uri == blocked_uri else None
    fs.read_node_metadata_including_inactive.return_value = {DEMOTION: "old-archive", "version": 1}
    outbox.get_event_status.return_value = status
    outbox.build_write_event.side_effect = lambda node: OutboxEvent("", "UPSERT_CONTEXT", node.uri, {}, "PENDING")
    manager.process.side_effect = lambda event, ctx: reconcile_promotion(fs, outbox, node, ctx, event=event)
    ctx = RequestContext("account", "user", "agent", "session", "request")
    event = maintenance_event(node.uri, 1)
    worker = OutboxWorker(None, None, dreaming=manager)
    with (patch("commit.dreaming_promotion._supports", return_value=True),
          patch("commit.dreaming_promotion._witnesses", side_effect=lambda *_a, **_kw: iter(witnesses))):
        result = worker.process_event(event, ctx)

    assert written[healthy_uri].content == healthy["content"]
    assert event.payload["dreaming_promotion_blocked"][blocked_uri]["status"] == status
    assert result.moved_to_dlq == (status == "FAILED")
    assert result.deferred == (status == "PENDING")


def test_rebuilt_parent_with_copied_proofs_triggers_demotion():
    claim = "Harbor mooring: A7"
    parent = _node("Alice", claim, category="event")
    parent.uri = "ctx://account/users/user/memories/events/mooring"
    parent.metadata["created_at"] = "2026-06-01T00:00:00+00:00"
    archives = {ident: _archive(ident, _message(ident, claim)) for ident in ("observed", "confirmed")}
    sources = {}
    for ident, archive in archives.items():
        uri = f"ctx://account/sessions/{ident}/history/{ident}"
        sources[uri] = ContextNode(uri, "SESSION", "archive", 4, "user:user", "", "", json.dumps(archive.messages),
                                   {"version": 1, "status": "ACTIVE", "owner_user_id": "user"})
        _bind(parent, archive, [_evidence(claim, "Alice", claim)])
    parent.metadata["dreaming_review"] = review(parent, datetime.now(UTC), read_source=archives.get)
    fs, outbox = Mock(), Mock()
    fs.find_dreaming_archive_uri.side_effect = lambda ident, *args, **kw: f"ctx://account/sessions/{ident}/history/{ident}"
    nodes = {**sources, parent.uri: parent}
    fs.read_node.side_effect = lambda uri, _: nodes[uri]
    ctx = RequestContext("account", "user", "agent", "session", "request")
    key, = parent.metadata["dreaming_review"]["facts"]
    witnesses = list(_witnesses(fs, parent, key, parent.metadata["dreaming_review"]["facts"][key], ctx))
    assert len(witnesses) == 2
    promoted = replace(parent, uri="ctx://account/users/user/memories/patterns/mooring", category="pattern",
                       metadata={"version": 1, PROMOTION: {"parents": witnesses}})
    nodes[promoted.uri] = promoted
    replacement = replace(parent, metadata={**parent.metadata, "created_at": "2026-06-02T00:00:00+00:00"})
    replacement.metadata["dreaming_review"] = review(replacement, datetime.now(UTC), read_source=archives.get)
    nodes[parent.uri] = replacement
    fs.demote_dreaming_node.side_effect = (
        lambda node, _, event_id: node.metadata.update(status="ARCHIVED", dreaming_demotion=event_id)
    )
    fs.read_node_metadata_including_inactive.side_effect = lambda uri, _: nodes[uri].metadata

    Dreaming(fs, outbox=outbox, enqueue_after_projection=True).process(maintenance_event(promoted.uri, 1), ctx)

    assert promoted.metadata.get("status") == "ARCHIVED"


def test_unchanged_daily_promotion_reviews_reuse_the_last_completed_inputs():
    claim = "Harbor mooring: A7"
    archive = _archive("observed", _message("message", claim))
    source_uri = "ctx://account/sessions/session/history/observed"
    source = ContextNode(source_uri, "SESSION", "archive", 4, "user:user", "", "", json.dumps(archive.messages),
                         {"version": 1, "status": "ACTIVE", "owner_user_id": "user"})
    nodes = {}
    for index in range(40):
        node = _node("Alice", claim, category="event")
        node.uri = f"ctx://account/users/user/memories/events/{index:02}"
        _bind(node, archive, [_evidence(claim, "Alice", claim)])
        nodes[node.uri] = node
    fs, outbox = Mock(), Mock()
    fs.find_dreaming_archive_uri.return_value = source_uri
    fs.read_node.side_effect = lambda uri, _: source if uri == source_uri else nodes[uri]
    fs.read_node_metadata_including_inactive.side_effect = NodeNotFoundError("promotion destination")
    fs.record_dreaming_review.side_effect = lambda node, _, result: node.metadata.update(dreaming_review=result)
    fs.list_node_order_key.side_effect = lambda uri: uri
    counts, saved, completed = [], {}, set()
    outbox.save_payload.side_effect = lambda event: saved.update({event.event_id: deepcopy(event.payload)})
    outbox.get_event_status.side_effect = lambda _, event_id: "DONE" if event_id in completed else "FAILED"
    fail_once = True

    def page(_directory, _ctx, *, limit, after_uri):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise OSError("promotion peers unavailable")
        selected = [node for uri, node in nodes.items() if not after_uri or uri > after_uri][:limit]
        counts[-1] += len(selected)
        return selected

    fs.list_identity_node_summaries_page.side_effect = page
    ctx = RequestContext("account", "user", "agent", "session", "request")
    manager = Dreaming(fs, outbox=outbox, enqueue_after_projection=True, read_source=lambda *_: archive)
    start = datetime(2026, 11, 21, tzinfo=UTC)
    for days in range(3):
        now = start + timedelta(days=days)
        counts.append(0)
        with patch("commit.dreaming.datetime", wraps=datetime) as clock:
            clock.now.return_value = now
            for node in nodes.values():
                event = maintenance_event(node.uri, 1, day=now.date())
                if fail_once:
                    with pytest.raises(OSError, match="promotion peers unavailable"):
                        manager.process(event, ctx)
                    event = replace(event, payload=saved[event.event_id])
                manager.process(event, ctx)
                if days or node.uri != next(reversed(nodes)):
                    completed.add(event.event_id)
    print(f"nodes={len(nodes)}, first_review_rows={counts[0]}, unconfirmed_review_rows={counts[1]}, "
          f"stable_review_rows={counts[2]}")
    assert counts[0] > 0 and counts[1:] == [len(nodes), 0]


def test_demotion_rediscovers_a_pair_from_existing_evidence():
    claim = "Harbor mooring: A7"
    parents, nodes, archives = [], {}, {}
    root = "ctx://account/users/user/memories"
    for ident in ("a", "b", "c"):
        archive = _archive(ident, _message(ident, claim))
        uri = f"ctx://account/sessions/{ident}/history/{ident}"
        archives[ident] = archive
        nodes[uri] = ContextNode(uri, "SESSION", "archive", 4, "user:user", "", "", json.dumps(archive.messages),
                                 {"version": 1, "status": "ACTIVE", "owner_user_id": "user"})
        node = _node("Alice", claim, category="event")
        node.uri = f"{root}/events/{ident}"
        _bind(node, archive, [_evidence(claim, "Alice", claim)])
        parents.append(node)
        nodes[node.uri] = node
    first, removed, remaining = parents
    fs, outbox = Mock(), Mock()
    outbox.get_event_status.return_value = "DONE"
    outbox.build_write_event.side_effect = lambda node: OutboxEvent("", "UPSERT_CONTEXT", node.uri, {}, "PENDING")
    fs.memory_identity_lock.side_effect = lambda *_: nullcontext()
    fs.find_dreaming_archive_uri.side_effect = lambda ident, *args, **kw: f"ctx://account/sessions/{ident}/history/{ident}"
    fs.list_identity_node_summaries_page.side_effect = (
        lambda directory, ctx, **kw: [node for node in nodes.values() if node.parent_uri == directory]
    )

    def metadata(uri, _):
        if uri not in nodes:
            raise NodeNotFoundError(uri)
        return nodes[uri].metadata

    def read(uri, ctx):
        if metadata(uri, ctx).get("status") == "ARCHIVED":
            raise NodeNotFoundError(uri)
        return nodes[uri]

    def write(node, _):
        node.metadata["version"] = node.metadata["_expected_version"] + 1
        nodes[node.uri] = node

    fs.read_node.side_effect = read
    fs.read_node_metadata_including_inactive.side_effect = metadata
    fs.archived_node_version.side_effect = (
        lambda uri, _: nodes[uri].metadata["version"]
        if uri in nodes and nodes[uri].metadata.get("status") == "ARCHIVED" else None
    )
    fs.write_node.side_effect = write
    fs.record_dreaming_review.side_effect = lambda node, _, result: node.metadata.update(dreaming_review=result)
    fs.demote_dreaming_node.side_effect = (
        lambda node, _, event_id: node.metadata.update(status="ARCHIVED", dreaming_demotion=event_id)
    )
    ctx = RequestContext("account", "user", "agent", "session", "request")
    manager = Dreaming(fs, outbox=outbox, enqueue_after_projection=True, read_source=lambda ident, _: archives[ident])
    for node in parents:
        manager.process(maintenance_event(node.uri, 1, day="2026-11-21"), ctx)
    derived, = [node for node in nodes.values() if node.category == "pattern"]
    assert {w["uri"] for w in derived.metadata[PROMOTION]["parents"]} == {first.uri, removed.uri}
    del nodes[removed.uri]
    manager.process(maintenance_event(derived.uri, 1, day="2026-11-22"), ctx)
    assert nodes[derived.uri].metadata["status"] == "ARCHIVED"

    manager.process(maintenance_event(remaining.uri, 1, day="2026-11-23"), ctx)

    restored = fs.read_node(derived.uri, ctx)
    assert restored.metadata["version"] == 2
    assert {w["uri"] for w in restored.metadata[PROMOTION]["parents"]} == {first.uri, remaining.uri}


@pytest.mark.parametrize("phrasing", ["Use English for the report", "Write the report in English"])
def test_duplicate_pass_preserves_separate_validity_windows(phrasing):
    earlier_claim = "Use English for the report"
    earlier = _node("Alice", earlier_claim, category="preference")
    current = _node("Alice", phrasing, category="preference")
    earlier.uri = "ctx://account/users/user/memories/preferences/september"
    current.uri = "ctx://account/users/user/memories/preferences/november"
    earlier.metadata.update(routing_key="report-language", created_at="2026-09-01",
                            provenance_ids=["previous-source"])
    current.metadata.update(routing_key="report-language", created_at="2026-11-01")
    earlier.metadata["dreaming_review"] = {
        "version": 1, "valid_from": "2026-09-01", "valid_to": "2026-10-01",
    }
    archive = _archive("november", _message("report", phrasing))
    _bind(current, archive, [_evidence(phrasing, "Alice", phrasing, valid_from="2026-11-01")])
    ctx = RequestContext("account", "user", "agent", "session", "request")
    fs, llm = Mock(), Mock()
    llm.complete_json.return_value = {"relationship": "equivalent"}
    nodes = {node.uri: node for node in (earlier, current)}
    fs.read_node.side_effect = lambda uri, _: nodes[uri]
    fs.dreaming_candidates.return_value = [earlier]
    fs.list_identity_node_summaries_page.return_value = []
    fs.record_dreaming_review.side_effect = lambda node, _, result: node.metadata.update(dreaming_review=result)
    fs.merge_dreaming_duplicate.side_effect = lambda _, source, *a, **kw: bool(nodes.pop(source.uri))

    Dreaming(fs, llm=llm, enqueue_after_projection=True, outbox=Mock(),
             read_source=lambda *_: archive).process(maintenance_event(current.uri, 1), ctx)

    assert set(nodes) == {earlier.uri, current.uri}
    assert recall_eligible(earlier.metadata, datetime(2026, 9, 15, tzinfo=UTC))
    assert recall_eligible(current.metadata, datetime(2026, 11, 15, tzinfo=UTC))
    assert current.metadata["dreaming_review"]["valid_from"] == "2026-11-01T00:00:00+00:00"


def test_projection_can_hide_again_after_reopening_at_same_expiry():
    node, _ = _reviewed_berth("A7")
    ctx = RequestContext("account", "user", "agent", "session", "request")
    keys = []
    outbox = Mock()
    outbox.build_write_event.side_effect = lambda target: OutboxEvent(
        "projection", "UPSERT_CONTEXT", target.uri, {"records": [{"id": "index"}]}, "PENDING",
    )
    outbox.register_once.side_effect = lambda event, _: keys.append((event.event_type, event.event_id))
    manager = Dreaming(Mock(), outbox=outbox)
    original = node.metadata["dreaming_review"]
    hidden = {**original, "lifecycle": {}, "valid_to": "2026-06-01T00:00:00+00:00"}

    with patch("commit.dreaming.datetime", wraps=datetime) as clock:
        clock.now.return_value = datetime(2026, 6, 20, tzinfo=UTC)
        for day, assessment in (("2026-06-17", hidden), ("2026-06-18", original), ("2026-06-20", hidden)):
            node.metadata["dreaming_review"] = assessment
            event = maintenance_event(node.uri, 1, day=day)
            manager._update_index_visibility(node, ctx, event, reopened=True)
        manager._update_index_visibility(node, ctx, event, reopened=True)

    assert [kind for kind, _ in keys] == ["ARCHIVE_CONTEXT", "UPSERT_CONTEXT", "ARCHIVE_CONTEXT", "ARCHIVE_CONTEXT"]
    assert len({identity for _, identity in keys}) == 3
    assert keys[-1] == keys[-2]


def test_agfs_saves_a_maintenance_checkpoint_through_outbox():
    node, _ = _reviewed_berth("A7")
    event = maintenance_event(node.uri, 1)
    store = OutboxStore(Mock(), Mock())
    with patch.object(store, "persist_event") as persist:
        store.save_payload(event)
    persist.assert_called_once_with(event)


@pytest.mark.parametrize("hidden", [False, True], ids=["reopen", "hide"])
def test_correction_retry_replays_index_after_review_was_saved(hidden):
    now = datetime(2026, 6, 20, tzinfo=UTC)
    prior, original = _reviewed_berth("A7")
    source, archive = _reviewed_berth("B6", "2026-06-10")
    prior.metadata["dreaming_review"] = review(prior, now, read_source=lambda _: original)
    source.metadata["dreaming_review"] = review(source, now, read_source=lambda _: archive)
    if not hidden:
        prior.metadata["dreaming_review"] = correction_plan(source, prior)
        source.metadata["routing_key"] = "other-harbor"
    source.metadata["version"] = 2
    assessment = review(source, now, read_source=lambda _: archive)
    ctx = RequestContext("account", "user", "agent", "summer", "review")
    fs, outbox = Mock(), Mock()
    fs.list_identity_node_summaries_page.return_value = [prior, source]
    nodes = {node.uri: node for node in (prior, source)}
    fs.read_node.side_effect = lambda uri, context: deepcopy(nodes[uri])

    def persist_correction(_, target, __, result):
        nodes[target.uri].metadata["dreaming_review"] = result
        return True

    fs.apply_dreaming_correction.side_effect = persist_correction
    checkpoint = {}
    outbox.save_payload.side_effect = lambda event: checkpoint.update(deepcopy(event.payload))
    outbox.build_write_event.side_effect = lambda target: OutboxEvent(
        "projection", "UPSERT_CONTEXT", target.uri, {"records": []}, "PENDING",
    )
    outbox.register_once.side_effect = [OSError("projection unavailable"), None]
    manager = Dreaming(fs, outbox=outbox)
    event = maintenance_event(source.uri, 2, day="2026-06-20")
    checkpoint.update(event.payload)
    with patch("commit.dreaming.datetime", wraps=datetime) as clock:
        clock.now.return_value = now
        with pytest.raises(OSError, match="projection unavailable"):
            manager._apply_review(source, ctx, assessment, None, event)
        source.metadata["version"] = 3
        manager.process(replace(event, payload=checkpoint), ctx)

    expiry = "2026-06-10" if hidden else "2026-07-01"
    assert prior.metadata["dreaming_review"]["valid_to"] == expiry + "T00:00:00+00:00"
    assert outbox.register_once.call_count == 2
    retried = outbox.register_once.call_args.args[0]
    assert (retried.uri, retried.payload["expected_version"]) == (prior.uri, 1)
    assert retried.event_type == ("ARCHIVE_CONTEXT" if hidden else "UPSERT_CONTEXT")


def test_promotion_reaches_reviewed_peers_beyond_the_first_four_pages():
    now = datetime(2026, 6, 20, tzinfo=UTC)
    node, _ = _reviewed_berth("A7")
    nodes = [replace(node, uri=f"{node.parent_uri.rstrip('/')}/{index:03d}") for index in range(161)]
    fs = Mock()
    fs.list_node_order_key.side_effect = lambda uri: uri
    fs.list_identity_node_summaries_page.side_effect = (
        lambda directory, ctx, limit, after_uri: [p for p in nodes if not after_uri or p.uri > after_uri][:limit]
    )
    ctx = RequestContext("account", "user", "agent", "summer", "review")
    assert list(_peers(fs, nodes[-1], ctx, now))[-1].uri == nodes[-2].uri


@pytest.mark.parametrize("inside_one_node", [False, True], ids=["peer", "same-node"])
def test_same_day_promotion_reaches_an_independent_archive_after_eight_proofs(inside_one_node):
    now = datetime(2026, 6, 20, tzinfo=UTC)
    claim = "Harbor mooring: A7"
    shared = _archive("shared", *(_message(f"m-{i}", f"{claim}, observation {i}.") for i in range(8)))
    independent = _archive("independent", _message("extra", f"{claim}, independently observed."))
    independent.session_id = "other-session"
    archives = {archive.archive_id: archive for archive in (shared, independent)}
    sources, nodes = {}, []
    for archive in archives.values():
        uri = f"ctx://account/sessions/{archive.session_id}/history/{archive.archive_id}"
        sources[archive.archive_id] = ContextNode(
            uri, "SESSION", "archive", 4, "agent:agent", "", "", json.dumps(archive.messages),
            {"version": 7, "status": "ACTIVE", "owner_agent_id": "agent"},
        )
    for index, inputs in enumerate(((shared,), (shared, independent))):
        node = _node("Alice", claim, category="case")
        node.uri, node.owner_space = f"ctx://account/agents/agent/memories/cases/{index}", "agent:agent"
        node.metadata["when"] = now.date().isoformat()
        for archive in inputs:
            _bind(node, archive, [_evidence(claim, "Alice", msg["content"]) for msg in archive.messages])
        node.metadata["dreaming_review"] = review(node, now, read_source=archives.get)
        nodes.append(node)
    ctx = RequestContext("account", "user", "agent", "summer", "review")
    fs, outbox = Mock(), Mock()
    current = nodes[1] if inside_one_node else nodes[0]
    fs.list_identity_node_summaries_page.return_value = [current] if inside_one_node else nodes
    fs.find_dreaming_archive_uri.side_effect = lambda ident, *_args, **_kw: sources[ident].uri
    by_uri = {node.uri: node for node in sources.values()}
    fs.read_node.side_effect = lambda uri, _: by_uri[uri]

    with (
        patch("commit.dreaming_promotion.datetime", wraps=datetime) as clock,
        patch("commit.dreaming_promotion._publish") as publish,
    ):
        clock.now.return_value = now
        reconcile_promotion(fs, outbox, current, ctx)

    publish.assert_called_once()
    witnesses = publish.call_args.args[4]
    assert {(w["source_uri"], w["source_version"]) for w in witnesses} == {
        (source.uri, 7) for source in sources.values()
    }
    assert fs.read_node.call_count == 2


@pytest.mark.parametrize("source_change", [
    "revised-archive", "deleted-correction", "recreated-correction", "legacy-recreated-correction",
])
def test_receiving_new_review_reopens_its_own_projection_after_retry(source_change):
    changed_correction = source_change != "revised-archive"
    now = datetime(2026, 6, 20, tzinfo=UTC) if changed_correction else datetime(2026, 7, 2, tzinfo=UTC)
    node, archive = _reviewed_berth("A7")
    recreated = None
    if changed_correction:
        source, _ = _reviewed_berth("B6", "2026-06-10")
        source.metadata["created_at"] = "2026-06-01T00:00:00+00:00"
        node.metadata["dreaming_review"] = correction_plan(source, node)
        if source_change == "legacy-recreated-correction":
            for fact in node.metadata["dreaming_review"]["facts"].values():
                if bound := fact.get("external_corrections", {}).get(source.uri):
                    del bound["source_created_at"]
        if source_change.endswith("recreated-correction"):
            recreated = replace(source, metadata={"version": 1, "created_at": "2026-06-18T00:00:00+00:00"})
    else:
        archive.messages[0]["content"] = "The original message has been revised."
    ctx = RequestContext("account", "user", "agent", "session", "request")
    fs, outbox = Mock(), Mock()

    def read_target(uri, _):
        if recreated and uri == recreated.uri:
            return deepcopy(recreated)
        if uri != node.uri:
            raise NodeNotFoundError(uri)
        return deepcopy(node)

    fs.read_node.side_effect = read_target
    fs.record_dreaming_review.side_effect = lambda _, __, value: node.metadata.update(dreaming_review=value)
    outbox.build_write_event.side_effect = lambda target: OutboxEvent(
        "projection", "UPSERT_CONTEXT", target.uri, {"records": []}, "PENDING",
    )
    event = maintenance_event(node.uri, 1, day=now.date().isoformat())
    payload = deepcopy(event.payload)
    outbox.save_payload.side_effect = lambda job: payload.update(deepcopy(job.payload))
    outbox.register_once.side_effect = [OSError("projection unavailable"), None]
    manager = Dreaming(fs, enqueue_after_projection=True, outbox=outbox, read_source=lambda *_: archive)

    with patch("commit.dreaming.datetime", wraps=datetime) as clock:
        clock.now.return_value = now
        with pytest.raises(OSError, match="projection unavailable"):
            manager.process(event, ctx)
        manager.process(replace(event, payload=payload), ctx)

    assert outbox.register_once.call_count == 2
    projection = outbox.register_once.call_args.args[0]
    assert (projection.uri, projection.event_type) == (node.uri, "UPSERT_CONTEXT")
    assert node.metadata["dreaming_review"]["evidence_state"] == ("verified" if changed_correction else "pending")


def test_recreated_version_one_gets_a_new_first_review_task():
    original, _ = _reviewed_berth("A7")
    original.metadata = {"version": 1, "created_at": "2026-06-01T00:00:00+00:00"}
    recreated = replace(original, metadata={**original.metadata, "created_at": "2026-06-18T00:00:00+00:00"})
    ctx = RequestContext("account", "user", "agent", "session", "request")
    fs, outbox = Mock(), Mock()
    completed = set()
    fs.read_node.side_effect = [original, recreated]
    outbox.get_event_status.side_effect = lambda _, event_id: "DONE" if event_id in completed else None
    outbox.persist_event.side_effect = lambda event, _: completed.add(event.event_id)
    manager = Dreaming(fs, enqueue_after_projection=True)
    manager.projection_ready(original.uri, ctx, outbox)
    manager.projection_ready(recreated.uri, ctx, outbox)
    assert len(completed) == 2
