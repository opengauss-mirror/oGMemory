"""Two ordinary Agent turns share a session while extraction runs on AGFS."""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

import server.app as app_module
import service.extraction_retry as retry_module
from core.models import CandidateMemory
from providers.unified_config import OgMemConfig
from providers.vector_index.in_memory_index import InMemoryVectorIndex
from server.memory_service import MemoryService
from service.extraction_checkpoint import EXTENSION_KEY
from session.session_state import SessionState

REQUEST_TIMEOUT = 20
BACKGROUND_TIMEOUT = 45


@pytest.mark.integration
def test_same_session_agent_turns_continue_during_background_extraction(monkeypatch):
    agfs_url = os.environ.get("OGMEM_AGFS_TEST_URL")
    if not agfs_url:
        pytest.skip("set OGMEM_AGFS_TEST_URL to an isolated AGFS server")

    service = MemoryService(config=OgMemConfig(
        agfs_transport="http",
        agfs_base_url=agfs_url,
        agfs_mount_prefix="/local",
        account_id=f"async-agent-{uuid4().hex}",
        user_id="nora",
        agent_id="assistant",
        role_control_enabled=False,
        after_turn_threshold=1024,
        directory_summary_enabled=False,
        rolling_compress_enabled=False,
        embedding_provider="mock",
    ))
    monkeypatch.setattr(app_module, "_service", service)
    monkeypatch.setattr(retry_module, "AGENT_LEASE_SECONDS", 9)
    monkeypatch.setattr(service, "_generate_summary_candidate", lambda *args, **kwargs: None)
    ctx = service.build_context({"sessionId": "repair-session"})
    extracting, resume_extraction = threading.Event(), threading.Event()
    projecting, resume_projection = threading.Event(), threading.Event()
    first_source = "I calibrated the receiver to 37 MHz this morning."
    second_source = "I replaced the backup battery with a lithium pack."
    first = [
        {"id": "receiver-user", "role": "user", "content": first_source},
        {"id": "receiver-agent", "role": "assistant", "content": "I recorded the calibration."},
    ]
    second = [
        {"id": "battery-user", "role": "user", "content": second_source},
        {"id": "battery-agent", "role": "assistant", "content": "I recorded the battery replacement."},
    ]
    extracted_sources = []
    original_lease_until = []

    class ProjectingIndex(InMemoryVectorIndex):
        def upsert(self, records):
            if any("/memories/events/" in record.uri and first_source in record.text for record in records):
                projecting.set()
                assert resume_projection.wait(BACKGROUND_TIMEOUT)
            super().upsert(records)

    service._vector_index = service._projection_index = ProjectingIndex()

    def extract(messages, context, **kwargs):
        text = next(message["content"] for message in messages if message["role"] == "user")
        extracted_sources.append(text)
        if text == first_source:
            original_lease_until.append(read_state()["runtime_extensions"][EXTENSION_KEY]["batch"]
                                        ["retry_task"]["lease_until"])
            extracting.set()
            assert resume_extraction.wait(BACKGROUND_TIMEOUT)
        key = "receiver_calibration" if text == first_source else "battery_replacement"
        return [CandidateMemory(
            category="event", owner_scope="user", routing_key=key, title=text,
            abstract=text, overview=text, content=text, who="Nora", confidence=.99,
        )]

    writer = service.get_write_api()
    writer._pipeline.extract = extract
    fs = service._get_context_fs()
    archive_store = service._get_archive_store()
    state_uri = SessionState.state_uri(ctx.account_id, ctx.session_id)
    headers = {"X-Account-ID": ctx.account_id, "X-User-ID": ctx.user_id, "X-Agent-ID": ctx.agent_id}

    def read_state():
        return json.loads(fs.read_node(state_uri, ctx).content)

    def wait_for_renewed_batch():
        deadline = time.monotonic() + REQUEST_TIMEOUT
        while time.monotonic() < deadline:
            batch = read_state()["runtime_extensions"][EXTENSION_KEY]["batch"]
            if batch["retry_task"]["lease_until"] > original_lease_until[0]:
                return batch
            time.sleep(.1)
        pytest.fail("expected the active Agent extraction lease to renew while the model is working")

    def post(path, payload):
        return app_module.app.test_client().post(
            f"/api/v1/{path}", headers=headers, json={"sessionId": ctx.session_id, **payload}
        )

    def after_turn(request_id, messages, previous=0, *, force=False):
        return post("after_turn", {"clientRequestId": request_id, "wait": False,
                                   "messages": messages, "prePromptMessageCount": previous,
                                   "forceExtract": force})

    def receipt(request_id):
        response = app_module.app.test_client().get(
            f"/api/v1/sessions/{ctx.session_id}/write-requests/{request_id}", headers=headers
        )
        assert response.status_code == 200
        return response.get_json()

    def wait_for_receipts():
        deadline = time.monotonic() + BACKGROUND_TIMEOUT
        while time.monotonic() < deadline:
            outcomes = [receipt(key) for key in ("receiver-turn", "battery-turn")]
            if all(outcome.get("terminal") for outcome in outcomes):
                return outcomes
            time.sleep(.05)
        pytest.fail(f"expected both Agent write receipts to settle: {outcomes}")

    try:
        with ThreadPoolExecutor(max_workers=2) as client_requests:
            try:
                accepted_first = client_requests.submit(after_turn, "receiver-turn", first, force=True).result(REQUEST_TIMEOUT)
                assert accepted_first.status_code == 202
                assert accepted_first.get_json()["request_id"] == "receiver-turn"
                assert accepted_first.get_json()["accepted"] is True
                assert extracting.wait(BACKGROUND_TIMEOUT)
                replay_first = client_requests.submit(after_turn, "receiver-turn", first, force=True).result(REQUEST_TIMEOUT)
                assert replay_first.status_code == 202
                assert replay_first.get_json()["request_id"] == "receiver-turn"
                assert replay_first.get_json()["accepted"] is True

                composed = client_requests.submit(
                    post, "compose", {"messages": first + second[:1], "prompt": second_source}
                ).result(REQUEST_TIMEOUT)
                assert composed.status_code == 200
                assert first_source in [message["content"] for message in composed.get_json()["messages"]]
                accepted_second = client_requests.submit(
                    after_turn, "battery-turn", first + second, len(first)
                ).result(REQUEST_TIMEOUT)
                assert accepted_second.status_code == 202
                assert accepted_second.get_json()["request_id"] == "battery-turn"
                assert accepted_second.get_json()["accepted"] is True

                replay = client_requests.submit(after_turn, "battery-turn", first + second, len(first)).result(REQUEST_TIMEOUT)
                assert replay.get_json()["request_id"] == "battery-turn"
                persisted = read_state()
                assert [message["content"] for message in persisted["buffer_messages"]] == [
                    message["content"] for message in first + second
                ]
                assert receipt("battery-turn")["source_status"] == "BUFFERED"
                assert wait_for_renewed_batch()["outbox_tracking_version"] == 1
                disposed = client_requests.submit(post, "dispose", {}).result(REQUEST_TIMEOUT)
                assert disposed.status_code == 200
                assert disposed.get_json() == {
                    "disposed": True, "flushed": False, "status": "background_flush_dispatched"
                }
                resume_extraction.set()
                assert projecting.wait(BACKGROUND_TIMEOUT)
                composed = client_requests.submit(
                    post, "compose", {"messages": first + second, "prompt": first_source}
                ).result(REQUEST_TIMEOUT)
                assert composed.status_code == 200
                assert second_source in [message["content"] for message in composed.get_json()["messages"]]
                assert extracted_sources == [first_source]
            finally:
                resume_extraction.set()
                resume_projection.set()

        outcomes = wait_for_receipts()
        assert extracted_sources == [first_source, second_source]
        assert [(receipt["request_status"], receipt["source_status"], receipt["projection_status"])
                for receipt in outcomes] == [("SUCCEEDED", "ARCHIVED", "DONE")] * 2
        assert all(receipt["outbox_tracking_complete"] for receipt in outcomes)
        assert [receipt["writes_completed"] for receipt in outcomes] == [1, 1]
        assert [archive_store.read_archive(ctx.session_id, receipt["archive_id"], ctx).messages[0]["content"]
                for receipt in outcomes] == [first_source, second_source]
    finally:
        service.shutdown(join_timeout=10)
