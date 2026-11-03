from dataclasses import replace
from unittest.mock import Mock
import threading
import time
import pytest

from core.errors import AccessDeniedError, OperationDeadlineExceededError
from core.models import RequestContext
from extraction.card_format import validate_memory_card
from service.extraction_checkpoint import EXTENSION_KEY
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.server.test_extraction_batch_recovery import candidate, nodes, params, setup_service
from tests.unit.server.test_extraction_retry_task import ready
from tests.unit.extraction.test_node_processing import NodeLLM, operation


@pytest.mark.parametrize("error", [OperationDeadlineExceededError("expired"),
    AccessDeniedError("ctx://denied", "node-errors", "denied"), OSError("storage unavailable"),
    ValueError("provider configuration invalid")])
@pytest.mark.parametrize("background", [False, True])
def test_node_operational_failure_preserves_requested_recovery(error, background):
    ctx = RequestContext("node-errors", "reader", "agent", "source", "proof")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[card()])
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True
    writer._llm = Mock()
    writer._llm.complete_with_tools.side_effect = error
    request = {**params(ctx), "retryInBackground": background}
    response = service.after_turn(request)
    assert response["ok"] is False and response["status"] == "failed"
    writer._llm = NodeLLM(lambda _: [operation([0])])
    if background:
        assert response["recovery_task"]["status"] == "queued"
        ready(manager, ctx)
        service._extraction_retry.run_once(ctx, response["recovery_task"]["id"])
    else:
        assert service.after_turn(request)["status"] == "completed"
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    assert len(nodes(fs, "entity")) == 1


@pytest.mark.parametrize("wait", [False, True])
def test_node_recovery_assigns_identity_with_independent_configuration(wait):
    ctx = RequestContext("node-identity", "reader", "agent", "source", "proof")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[card()])
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True
    service._extraction_retry.wake = Mock()
    writer._llm = NodeLLM(lambda _: [operation([0], action="defer")])
    response = service.after_turn({**params(ctx, request=""), "wait": wait})
    if wait:
        assert response["status"] == "queued"
    else:
        assert response["status"] == "processing"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx, "timeoutSeconds": 0})
        if state.get("recovery_task"):
            break
        threading.Event().wait(.01)
    batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
    request_id = batch["client_request_id"]
    assert request_id.startswith("card-")
    writer._llm = NodeLLM(lambda _: [operation([0])])
    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, state["recovery_task"]["id"])
    assert manager.get_or_create(ctx.session_id).request_completed(request_id)
    assert writer._pipeline.extract.call_count == 1
    assert len(nodes(fs, "entity")) == 1


def card():
    return replace(candidate("entity"), routing_key="nora", object_name="Nora",
                   abstract="Nora is a radio engineer.", overview="- Occupation: radio engineer.",
                   content="# Nora\n\n## Identity\n- Occupation: radio engineer.")


def enable(service, writer):
    service._cfg.card_format_enabled = True
    writer._card_format_enabled = True
    writer._writer._archive_builder._card_format_enabled = True
    service._extraction_retry.wake = Mock()


@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("path,value", [
    (("operations",), "update"),
    (("operations", 0), None),
    (("operations", 0, "abstract"), None),
    (("operations", 0, "abstract"), "A" * 201),
    (("operations", 0, "overview"), "Blue"),
    (("operations", 0, "overview", 0), None),
    (("operations", 0, "overview"), ["A topic."] * 9),
    (("operations", 0, "overview", 0), "A" * 201),
    (("operations", 0, "sections"), None),
    (("operations", 0, "sections", 0), None),
    (("operations", 0, "sections", 0, "heading"), None),
    (("operations", 0, "sections", 0, "facts"), "Blue"),
    (("operations", 0, "sections", 0, "facts", 0), False),
    (("operations", 0, "routing_key"), None),
    (("operations", 0, "object_name"), {}),
    (("operations", 0, "target"), None),
    (("operations", 0, "action"), None),
    (("operations", 0, "edits"), None),
    (("operations", 0, "edits"), [{"search": None, "replacement": "Blue"}]),
    (("operations", 0, "edits"), [{"search": "Nora", "replacement": None}]),
    (("operations", 0, "draft_ids"), "0"),
    (("operations", 0, "draft_ids", 0), True),
])
def test_node_output_contract_recovers_saved_drafts(update, path, value):
    ctx = RequestContext("typed-card", "reader", "agent", "source", "proof")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[card()])
    target = writer.write_memory(card(), ctx)["target_uri"] if update else ""
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True
    payload = {"operations": [operation([0], target=target, action="update" if update else "create")]}
    parent = payload
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = value
    writer._llm = Mock()
    writer._llm.complete_with_tools.return_value = [{"tool": "process_memory_nodes", "input": payload}]
    response = service.after_turn(params(ctx))
    assert response["status"] == "queued"
    task = response["recovery_task"]
    assert task["failure_kind"] == "memory_card_format"
    assert ".".join(map(str, path)) in task["last_error"]
    saved = nodes(fs, "entity")
    assert len(saved) == int(update)
    if update:
        assert saved[0].metadata["version"] == 1
        assert saved[0].content == card().content
    writer._llm = NodeLLM(lambda _: [operation([0], target=target, action="update" if update else "create")])
    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, task["id"])
    saved = nodes(fs, "entity")
    assert len(saved) == 1
    validate_memory_card(saved[0])
    assert saved[0].metadata["version"] == (2 if update else 1)
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    assert writer._pipeline.extract.call_count == 1
    assert service.after_turn(params(ctx))["status"] == "duplicate"
    assert service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx})["idle"] is True


@pytest.mark.parametrize("category", ["entity", "preference"])
@pytest.mark.parametrize("field", ["abstract", "overview"])
@pytest.mark.parametrize("wait", [False, True])
def test_empty_create_summary_recovers_saved_drafts(category, field, wait):
    ctx = RequestContext("empty-card", "reader", "agent", "source", "proof")
    fs = MemoryFS()
    draft = replace(card(), category=category)
    service, writer, manager = setup_service(fs, ctx, entries=[draft])
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True
    op = operation([0])
    op[field] = "" if field == "abstract" else []
    writer._llm = NodeLLM(lambda _: [op])
    response = service.after_turn({**params(ctx, request=""), "wait": wait})
    assert response["status"] == ("queued" if wait else "processing")
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx, "timeoutSeconds": 0})
        if state.get("recovery_task"):
            break
        threading.Event().wait(.01)
    task = state["recovery_task"]
    assert task["failure_kind"] == "memory_card_format"
    assert field in task["last_error"]
    request_id = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]["client_request_id"]
    writer._llm = NodeLLM(lambda _: [operation([0])])
    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, task["id"])
    saved = nodes(fs, category)
    assert len(saved) == 1
    validate_memory_card(saved[0])
    assert manager.get_or_create(ctx.session_id).request_completed(request_id)
    assert writer._pipeline.extract.call_count == 1
    assert service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx})["idle"] is True


@pytest.mark.parametrize("automatic_first", [False, True])
def test_queued_old_batch_releases_successor_request(automatic_first):
    ctx = RequestContext("successor-card", "reader", "agent", "source", "proof")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[card()])
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True
    writer._llm = NodeLLM(lambda _: [operation([0], action="defer")])
    service.after_turn(params(ctx, request="" if automatic_first else "original"))
    buf = manager.get_or_create(ctx.session_id)
    batch = buf.runtime_extensions[EXTENSION_KEY]["batch"]
    original = batch["client_request_id"]
    successor = params(ctx, request="next", content="The receiver has a green carrying case.")
    queued = service.after_turn(successor)
    assert queued["status"] == "queued"
    assert buf.active_request_ids == set()
    assert buf.runtime_extensions[EXTENSION_KEY]["batch"]["client_request_id"] == original
    writer._llm = NodeLLM(lambda _: [operation([0])])
    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, queued["recovery_task"]["id"])
    assert buf.request_completed(original)
    assert [message.content for message in buf.messages] == [successor["messages"][0]["content"]]
    target = nodes(fs, "entity")[0].uri
    writer._llm = NodeLLM(lambda _: [operation([0], target=target, action="update", fact="Has a green carrying case.")])
    assert service.after_turn(successor)["status"] == "completed"
    assert service.after_turn(successor)["status"] == "duplicate"
    assert buf.request_completed("next")
    assert writer._pipeline.extract.call_count == 2
    assert len(nodes(fs, "entity")) == 1
    assert "green carrying case" in fs.read_node(target, ctx).content
    assert service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx})["idle"] is True


@pytest.mark.parametrize("category", ["entity", "preference"])
def test_duplicate_observation_upgrades_legacy_overview_and_completes(category):
    ctx = RequestContext("duplicate-card", "reader", "agent", "source", "proof")
    old = replace(card(), category=category, overview="## Occupation\nWorks on radios.")
    if category == "preference":
        old = replace(old, content="# Radio work\n\n- Prefers quiet rooms for radio work.")
    fs = MemoryFS()
    incoming = replace(old, overview="- Works on radios.")
    service, writer, manager = setup_service(fs, ctx, entries=[incoming])
    uri = writer.write_memory(old, ctx)["target_uri"]
    enable(service, writer)
    assert service.after_turn(params(ctx))["status"] == "completed"
    stored = fs.read_node(uri, ctx)
    validate_memory_card(stored)
    assert stored.content == old.content
    assert stored.overview == "- ## Occupation\n- Works on radios."
    assert stored.metadata["version"] == 2
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    assert service.after_turn(params(ctx))["status"] == "duplicate"


@pytest.mark.parametrize("fault", ["access", "storage", "deadline"])
def test_cas_retry_operational_failure_resumes_accepted_node(fault, monkeypatch):
    from commit.write_receipts import WriteDeadlineExceededError

    ctx = RequestContext("cas-operation", "reader", "agent", "source", "proof")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[card()])
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True
    uri = writer.write_memory(card(), ctx)["target_uri"]
    writer._llm = NodeLLM(lambda _: [operation([0], target=uri, action="update", fact="Uses a blue case.")])
    write, read = writer._writer.write_candidate, fs.read_node
    versions = []
    reading_retry = False

    def read_with_failure(target, *args, **kwargs):
        if reading_retry and target == uri:
            if fault == "access":
                raise AccessDeniedError(uri, ctx.account_id, "permission changed")
            raise OSError("storage read unavailable")
        return read(target, *args, **kwargs)

    def concurrent_write(candidate, *args, **kwargs):
        nonlocal reading_retry
        if candidate.target_uri == uri and candidate.write_operation_id:
            versions.append(candidate.expected_version)
            if len(versions) == 1:
                current = read(uri, ctx)
                current.content += "\n- Works in a quiet room."
                fs.write_node(current, ctx)
            elif len(versions) == 2:
                if fault == "deadline":
                    raise WriteDeadlineExceededError("write budget expired")
                reading_retry = True
        try:
            return write(candidate, *args, **kwargs)
        finally:
            reading_retry = False

    with monkeypatch.context() as patch:
        patch.setattr(fs, "read_node", read_with_failure)
        patch.setattr(writer._writer, "write_candidate", concurrent_write)
        result = service.after_turn(params(ctx))
    assert result["ok"] is False and result["status"] == "failed"
    assert versions == [1, 2]
    assert service.after_turn(params(ctx))["status"] == "completed"
    stored = fs.read_node(uri, ctx)
    assert stored.metadata["version"] == 3
    assert "quiet room" in stored.content and "blue case" in stored.content
    assert writer._pipeline.extract.call_count == 1
    assert len(writer._llm.calls) == 2
    assert manager.get_or_create(ctx.session_id).request_completed("original")


@pytest.mark.parametrize("category", ["entity", "preference"])
@pytest.mark.parametrize("decision", ["create", "other"])
def test_cas_revision_recovers_the_accepted_target(category, decision, monkeypatch):
    ctx = RequestContext("cas-identity", "reader", "agent", "source", "proof")
    fs = MemoryFS()
    original = replace(card(), category=category)
    if category == "preference":
        original = replace(original, content="# Radio work\n\n- Prefers a quiet room.")
    draft = replace(original, routing_key="radio_engineer", object_name="radio engineer")
    service, writer, manager = setup_service(fs, ctx, entries=[draft])
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True
    target = writer.write_memory(original, ctx)["target_uri"]
    other = writer.write_memory(replace(original, routing_key="mira", object_name="Mira",
                                       content=original.content.replace("quiet room", "music").replace(
                                           "radio engineer", "pilot")), ctx)["target_uri"]

    def decide(payload):
        op = operation([0], target=target, action="update", fact="Uses a blue case.")
        if len(writer._llm.calls) == 2:
            if decision == "other":
                op["target"] = other
            else:
                op.update(action="create", target="", routing_key="new_engineer", object_name="new engineer")
        return [op]

    writer._llm = NodeLLM(decide)
    write = writer._writer.write_candidate
    accepted = []

    def concurrent_write(candidate, *args, **kwargs):
        if candidate.target_uri == target and candidate.write_operation_id and not accepted:
            accepted.append(candidate)
            current = fs.read_node(target, ctx)
            current.content += "\n- Works in a quiet room."
            fs.write_node(current, ctx)
        return write(candidate, *args, **kwargs)

    monkeypatch.setattr(writer._writer, "write_candidate", concurrent_write)
    result = service.after_turn(params(ctx))
    assert result["status"] == "queued"
    assert target in result["last_error"]
    batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
    saved = [c for slot in batch["slots"].values() for c in slot["candidates"] if c["category"] == category]
    assert saved[0]["target_uri"] == target
    assert saved[0]["write_operation_id"] == accepted[0].write_operation_id
    assert fs.read_node(target, ctx).metadata["version"] == 2
    assert fs.read_node(other, ctx).metadata["version"] == 1
    writer._llm = NodeLLM(lambda _: [operation([0], target=target, action="update", fact="Uses a blue case.")])
    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, result["recovery_task"]["id"])
    stored = fs.read_node(target, ctx)
    assert stored.metadata["version"] == 3
    assert "quiet room" in stored.content and "blue case" in stored.content
    assert fs.read_node(other, ctx).metadata["version"] == 1
    assert len(nodes(fs, category)) == 2
    assert writer._pipeline.extract.call_count == 1
    assert manager.get_or_create(ctx.session_id).request_completed("original")


@pytest.mark.parametrize("restart", [False, True])
def test_identical_new_turns_keep_distinct_sources(restart):
    ctx = RequestContext("repeated-card", "reader", "agent", "source", "proof")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[card()])
    enable(service, writer)
    first = service.after_turn(params(ctx, request=""))
    if restart:
        service, writer, manager = setup_service(fs, ctx, entries=[card()])
        enable(service, writer)
    second = service.after_turn(params(ctx, request=""))
    assert first["status"] == second["status"] == "completed"
    assert first["client_request_id"] != second["client_request_id"]
    saved = nodes(fs, "entity")
    assert len(saved) == 1
    assert len(saved[0].metadata["provenance_ids"]) == 2
    for result in (first, second):
        assert manager.get_or_create(ctx.session_id).request_completed(result["client_request_id"])
    assert service.after_turn(params(ctx, request=second["client_request_id"]))["status"] == "duplicate"


def test_identical_accumulating_turns_keep_distinct_message_ids():
    ctx = RequestContext("accumulating-card", "reader", "agent", "source", "proof")
    service, writer, manager = setup_service(MemoryFS(), ctx, entries=[card()])
    enable(service, writer)
    service._cfg.after_turn_threshold = 10000
    first = service.after_turn(params(ctx, request=""))
    second = service.after_turn(params(ctx, request=""))
    assert first["status"] == second["status"] == "accumulating"
    messages = manager.get_or_create(ctx.session_id).messages
    assert len(messages) == 2
    assert messages[0].id != messages[1].id
    assert messages[0].content == messages[1].content


@pytest.mark.parametrize("invalid_field", ["overview", "content"])
@pytest.mark.parametrize("wait", [False, True])
def test_incremental_node_validation_queues_and_recovers_saved_drafts(invalid_field, wait):
    ctx = RequestContext("incremental-card", "reader", "agent", "source", "proof")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[card()])
    enable(service, writer)
    service._cfg.node_processing_enabled = writer._node_processing_enabled = True
    uri = writer.write_memory(card(), ctx)["target_uri"]
    original = fs.read_node(uri, ctx)
    op = operation([0], target=uri, action="update", fact="Nora works in a quiet room.")
    if invalid_field == "overview":
        op["overview"] = []
    else:
        op["sections"] = []
        op["edits"] = [{"search": original.content, "replacement": ""}]
    writer._llm = NodeLLM(lambda _: [op])
    response = service.after_turn({**params(ctx, request=""), "wait": wait})
    assert response["status"] == ("queued" if wait else "processing")
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx, "timeoutSeconds": 0})
        if state.get("recovery_task"):
            break
        threading.Event().wait(.01)
    task = state["recovery_task"]
    assert task["failure_kind"] == "memory_card_format"
    assert fs.read_node(uri, ctx).content == original.content
    assert len(writer._llm.calls) == 1
    writer._llm = NodeLLM(lambda _: [operation([0], target=uri, action="update", fact="Nora works in a quiet room.")])
    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, task["id"])
    stored = fs.read_node(uri, ctx)
    assert stored.metadata["version"] == 2
    assert "Occupation: radio engineer." in stored.content
    assert "Nora works in a quiet room." in stored.content
    assert writer._pipeline.extract.call_count == 1
    assert service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx})["idle"] is True


def test_card_recovers_after_restart_without_an_inline_model_retry():
    ctx = RequestContext(account_id="card-recovery", user_id="reader", agent_id="agent",
                         session_id="source", trace_id="proof")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[replace(card(), content="Nora is a radio engineer.")])
    enable(service, writer)
    response = service.after_turn(params(ctx))
    assert response["ok"] is True and response["status"] == "queued"
    assert response["reason"] == "memory_card_format"
    assert writer._pipeline.extract.call_count == 1
    task_id = response["recovery_task"]["id"]
    assert response["recovery_task"]["failure_kind"] == "memory_card_format"
    restarted, writer, manager = setup_service(fs, ctx, entries=[card()])
    enable(restarted, writer)
    assert restarted.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx})["recovery_task"]["id"] == task_id
    ready(manager, ctx)
    restarted._extraction_retry.run_once(ctx, task_id)
    saved = nodes(fs, "entity")
    assert len(saved) == 1
    validate_memory_card(saved[0])
    assert restarted.after_turn(params(ctx))["status"] == "duplicate"
    assert restarted.wait_until_idle({**params(ctx), "timeoutSeconds": 0})["idle"] is True


def test_format_recovery_assigns_a_durable_identity_to_legacy_turns():
    ctx = RequestContext(account_id="legacy-card", user_id="reader", agent_id="agent",
                         session_id="source", trace_id="proof")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx, entries=[replace(card(), overview="Nora is an engineer.")])
    enable(service, writer)
    response = service.after_turn(params(ctx, request=""))
    assert response["ok"] is True and response["status"] == "queued"
    batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
    assert batch["client_request_id"].startswith("card-")
    writer._pipeline.extract.side_effect = None
    writer._pipeline.extract.return_value = [card()]
    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, response["recovery_task"]["id"])
    assert len(nodes(fs, "entity")) == 1


def test_preference_card_keeps_the_holder_and_conditions():
    preference = replace(card(), category="preference", routing_key="nora_radio_work",
                         abstract="Nora prefers quiet rooms for radio work.",
                         overview="- Nora prefers quiet rooms during radio work.",
                         content="# Radio work\n\n- Nora prefers quiet rooms during radio work.")
    validate_memory_card(preference)


def test_async_turn_returns_while_extraction_runs_and_queues_format_recovery():
    ctx = RequestContext(account_id="async-card", user_id="reader", agent_id="agent",
                         session_id="source", trace_id="proof")
    fs = MemoryFS()
    service, writer, manager = setup_service(fs, ctx)
    enable(service, writer)
    release = threading.Event()
    started = threading.Event()

    def extract(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return [replace(card(), content="Nora is a radio engineer.")]

    writer._pipeline.extract.side_effect = extract
    try:
        response = service.after_turn({**params(ctx, request=""), "wait": False})
        assert response["ok"] is True and response["status"] == "processing"
        assert started.wait(2)
    finally:
        release.set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = service.wait_until_idle({"sessionId": ctx.session_id, "_ctx": ctx, "timeoutSeconds": 0})
        if status.get("recovery_task"):
            break
        threading.Event().wait(.01)
    assert status["recovery_task"]["status"] == "queued"
    assert writer._pipeline.extract.call_count == 1
    writer._pipeline.extract.side_effect = None
    writer._pipeline.extract.return_value = [card()]
    ready(manager, ctx)
    service._extraction_retry.run_once(ctx, status["recovery_task"]["id"])
    assert len(nodes(fs, "entity")) == 1
