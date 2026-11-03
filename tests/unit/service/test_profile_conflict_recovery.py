"""Profile conflicts refresh original source against the current stored profile."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Barrier, Lock

import pytest

from commit.write_receipts import WriteDeadlineExceededError
from service.extraction_checkpoint import EXTENSION_KEY
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.commit.test_profile_merge import profile, profile_context
from tests.unit.server.test_extraction_batch_recovery import params, setup_service


def setup():
    fs, ctx = MemoryFS(), profile_context()
    service, writer, manager = setup_service(fs, ctx)
    uri = writer.write_memory(profile("identity", "Name: Nora."), ctx)["target_uri"]
    return fs, ctx, service, writer, manager, uri


def snapshot(fs, uri, ctx, fact, archive_id):
    current = fs.read_node(uri, ctx)
    return replace(profile("identity", current.content + " " + fact), target_uri=uri,
                   expected_version=current.metadata["version"],
                   provenance_ids=[f"prov:1:archive:{archive_id}:source"])


def test_parallel_sessions_refresh_profile_and_complete_both_sources():
    fs, ctx, service, writer, manager, uri = setup()
    barrier, guard, calls = Barrier(2), Lock(), []

    def extract(messages, request, **kwargs):
        candidate = snapshot(fs, uri, request, messages[0]["content"], kwargs["archive_id"])
        with guard:
            first_read = request.session_id not in [entry[0] for entry in calls]
            calls.append((request.session_id, candidate.expected_version))
        if first_read:
            barrier.wait(timeout=5)
        return [candidate]

    writer._pipeline.extract = extract
    contexts = [replace(ctx, session_id=identity) for identity in ("first", "second")]
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(service.after_turn, params(request, content=f"{request.session_id} fact."))
                   for request in contexts]
        results = [future.result(timeout=10) for future in futures]

    assert all(result["status"] == "completed" for result in results)
    stored = fs.read_node(uri, ctx)
    assert stored.metadata["version"] == 3
    assert all(fact in stored.content for fact in ("Name: Nora.", "first fact.", "second fact."))
    assert sorted(version for _, version in calls) == [1, 1, 2]
    assert all(manager.get_or_create(request.session_id).request_completed("original") for request in contexts)


@pytest.mark.parametrize("resume", [False, True])
def test_external_profile_update_preserved_with_durable_refresh(resume):
    fs, ctx, service, writer, manager, uri = setup()
    calls, revisions = [], []

    def extract(messages, request, **kwargs):
        candidate = snapshot(fs, uri, request, messages[0]["content"], kwargs["archive_id"])
        calls.append(candidate.expected_version)
        if len(calls) == 1:
            external = snapshot(fs, uri, request, "Occupation: engineer.", "external")
            writer.write_memory(external, request)
        return [candidate]

    writer._pipeline.extract = extract
    write = writer._writer.write_candidate

    def durable_pause(candidate, request):
        if resume and candidate.category == "profile" and candidate.expected_version == 2 and not revisions:
            batch = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]
            revisions.append(deepcopy(next(slot for slot in batch["slots"].values() if slot.get("revisions"))))
            raise WriteDeadlineExceededError("resume after durable profile revision")
        return write(candidate, request)

    writer._writer.write_candidate = durable_pause
    request = params(ctx, content="Location: Utrecht.")
    result = service.after_turn(request)
    if resume:
        result = service.after_turn(request)
        saved = revisions[0]
        previous = saved["revision_history"][0]
        assert saved["revisions"] == {"0": 1}
        assert previous["operation"]["status"] == "rejected_before_write"
        assert saved["candidates"][0]["write_operation_id"] != previous["candidate"]["write_operation_id"]

    assert result["status"] == "completed"
    stored = fs.read_node(uri, ctx)
    assert stored.content == "Name: Nora. Occupation: engineer. Location: Utrecht."
    assert stored.metadata["version"] == 3
    assert calls == [1, 2]
    assert service.after_turn(request)["status"] == "duplicate"


def test_profile_recovery_resumes_after_bounded_concurrent_changes():
    fs, ctx, service, writer, manager, uri = setup()
    calls = []

    def extract(messages, request, **kwargs):
        candidate = snapshot(fs, uri, request, messages[0]["content"], kwargs["archive_id"])
        calls.append(candidate.expected_version)
        if len(calls) <= 3:
            external = snapshot(fs, uri, request, f"Concurrent fact {len(calls)}.", f"external-{len(calls)}")
            writer.write_memory(external, request)
        return [candidate]

    writer._pipeline.extract = extract
    request = params(ctx, content="Location: Utrecht.")
    service.after_turn(request)
    result = service.after_turn(request)

    assert result["status"] == "completed"
    stored = fs.read_node(uri, ctx)
    assert stored.metadata["version"] == 5
    assert stored.content == (
        "Name: Nora. Concurrent fact 1. Concurrent fact 2. Concurrent fact 3. Location: Utrecht.")
    assert calls == [1, 2, 3, 4]
    assert manager.get_or_create(ctx.session_id).request_completed("original")
