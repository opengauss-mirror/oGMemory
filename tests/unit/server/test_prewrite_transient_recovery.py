from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from core.errors import LLMTransientError
from core.models import RequestContext
from fs.sql_adapter.session_mutation import SessionMutationBusyError
from service.extraction_checkpoint import EXTENSION_KEY
from tests.unit.commit.test_memory_identity import MemoryFS
from tests.unit.commit.test_profile_merge import profile, result
from tests.unit.server.test_extraction_batch_recovery import nodes, params, setup_service
from tests.unit.server.test_extraction_retry_task import ready


class BusyArchiveFS(MemoryFS):
    busy_directory = ""

    @contextmanager
    def memory_identity_lock(self, directory, ctx):
        if directory == self.busy_directory:
            raise SessionMutationBusyError("archive directory is busy")
        yield


@pytest.mark.parametrize("category", ["session_archive", "profile"])
def test_prewrite_retry_completes_after_dependency_recovers(category):
    ctx = RequestContext("prewrite-retry", "reader", "agent", "source", "proof")
    fs = BusyArchiveFS()
    incoming = profile("location", "Location: Utrecht.")
    service, writer, manager = setup_service(
        fs, ctx, entries=[incoming] if category == "profile" else None)
    service._extraction_retry.wake = Mock()
    if category == "profile":
        writer.write_memory(profile("identity", "Name: Nora."), ctx)
        model = Mock()
        model.complete_json.side_effect = LLMTransientError("merge provider temporarily unavailable")
        writer._writer._archive_builder._llm = model
    else:
        fs.busy_directory = writer._writer._identity_resolver.uris.get_directory_uri(category, ctx).rstrip("/")

    response = service.after_turn({**params(ctx), "retryInBackground": True})
    identity = response["recovery_task"]["id"]
    for attempt in range(1, 3):
        ready(manager, ctx)
        assert service._extraction_retry.run_once(ctx, identity) > 0
        task = manager.get_or_create(ctx.session_id).runtime_extensions[EXTENSION_KEY]["batch"]["retry_task"]
        assert task["status"] == "queued"
        assert task["attempts"] == 0
        assert task.get("transient_attempts", 0) == (attempt if category == "profile" else 0)

    if category == "profile":
        model.complete_json.side_effect = None
        model.complete_json.return_value = result("Name: Nora. Location: Utrecht.")
    else:
        fs.busy_directory = ""
    ready(manager, ctx)
    assert service._extraction_retry.run_once(ctx, identity) is None
    assert manager.get_or_create(ctx.session_id).request_completed("original")
    assert writer._pipeline.extract.call_count == 1
    assert len(nodes(fs, "session_archive")) == 1
    if category == "profile":
        stored = nodes(fs, "profile")[0]
        assert stored.content == "Name: Nora. Location: Utrecht."
        assert stored.metadata["version"] == 2
    else:
        assert nodes(fs, "event")[0].content == "I repaired the receiver."
