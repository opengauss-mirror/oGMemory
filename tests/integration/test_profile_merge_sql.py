"""Profile merges preserve content and sources across SQL connections."""

import os
from dataclasses import replace
from uuid import uuid4

import pytest

from commit.sql_outbox_store import SQLOutboxStore
from core.models import RequestContext
from fs.sql_adapter.sql_context_fs import SQLContextFS
from service.api import MemoryWriteAPI
from tests.unit.commit.test_profile_merge import ProfileLLM, profile, result


def test_sql_profile_merge_survives_reconnect():
    dsn = os.environ.get("OGMEM_IDENTITY_TEST_DSN")
    if not dsn:
        pytest.skip("set OGMEM_IDENTITY_TEST_DSN to a disposable PostgreSQL database")
    ctx = RequestContext(account_id="profile-" + uuid4().hex, user_id="reader",
                         agent_id="agent", session_id="session", trace_id="trace")
    original = "Name: Nora. Occupation: technician."
    located = original + " Location: Utrecht."
    promoted = "Name: Nora. Occupation: engineer. Location: Utrecht."
    llm = ProfileLLM(result(located), result(promoted))
    updates = [profile("identity", original), profile("identity", "Location: Utrecht."),
               profile("job", "Occupation: engineer.")]
    for index, candidate in enumerate(updates, start=1):
        fs = SQLContextFS(dsn)
        outbox = SQLOutboxStore(dsn, fs=fs)
        try:
            writer = MemoryWriteAPI(fs, llm, outbox, directory_summary_enabled=False)
            uri = writer.write_memory(candidate, ctx)["target_uri"]
            node = fs.read_node(uri, ctx)
            assert node.content == (original, located, promoted)[index - 1]
            assert node.overview == node.abstract == node.content
            assert node.metadata["version"] == index
            assert node.metadata["provenance_ids"] == list(dict.fromkeys(
                source for item in updates[:index] for source in item.provenance_ids
            ))
            assert node.metadata["routing_key"] == "identity"
        finally:
            for store in (fs, outbox):
                while store._pool:
                    store._pool.pop().close()
    assert len(llm.calls) == 2


def test_sql_complete_profile_update_survives_reconnect():
    dsn = os.environ.get("OGMEM_IDENTITY_TEST_DSN")
    if not dsn:
        pytest.skip("set OGMEM_IDENTITY_TEST_DSN to a disposable PostgreSQL database")
    ctx = RequestContext(account_id="snapshot-" + uuid4().hex, user_id="reader",
                         agent_id="agent", session_id="session", trace_id="trace")
    llm = ProfileLLM()
    original = "Name: Nora. Location: Utrecht. Occupation: technician."
    updated = "Name: Nora. Location: Utrecht. Occupation: engineer."
    uri = f"ctx://{ctx.account_id}/users/reader/memories/profile"
    candidates = [profile("identity", original),
                  replace(profile("occupation", updated), target_uri=uri, expected_version=1)]
    for version, candidate in enumerate(candidates, start=1):
        fs = SQLContextFS(dsn)
        outbox = SQLOutboxStore(dsn, fs=fs)
        try:
            write = MemoryWriteAPI(fs, llm, outbox, directory_summary_enabled=False).write_memory(candidate, ctx)
            node = fs.read_node(uri, ctx)
            assert write["target_uri"] == uri
            assert node.content == (original, updated)[version - 1]
            assert node.metadata["version"] == version
            assert node.metadata["routing_key"] == "identity"
        finally:
            for store in (fs, outbox):
                while store._pool:
                    store._pool.pop().close()
    assert len(llm.calls) == 0
