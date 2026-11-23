"""Evidence identities and offsets are independent of batching and storage."""

from dataclasses import replace

import pytest

from core.errors import ValidationError
from core.evidence import digest, evidence_uri, message_source, reassemble_messages
from core.models import RequestContext
from fs.access_control import check_uri_access
from fs.agfs_adapter.agfs_context_fs import parse_uri as agfs_parse
from fs.agfs_adapter.agfs_context_fs import uri_to_path
from fs.sql_adapter.sql_context_fs import parse_uri as sql_parse

CTX = RequestContext("account", "alice", "agent", "session", "trace")


def message(text="I completed 7 essays.", **kwargs):
    return {"id": "message", "sequence": 3, "role": "user", "content": text, **kwargs}


def test_identity_and_units_are_stable_and_scoped():
    source = message_source(message(), CTX)
    assert source == message_source(message(), replace(CTX, trace_id="another"))
    for ctx in [replace(CTX, account_id="other"), replace(CTX, user_id="other"),
                replace(CTX, session_id="other")]:
        assert source.source_key != message_source(message(), ctx).source_key
    assert source.source_key != message_source(message(id="another"), CTX).source_key
    changed = message_source(message("I completed 4 essays."), CTX)
    assert changed.source_key == source.source_key
    assert changed.content_hash != source.content_hash
    assert source.recorded_at is None and source.time_basis == "unknown"


def test_wrapper_keeps_parent_identity_and_character_offsets():
    text = "[Caroline]: 我昨天去了公园。\n[Melanie]: I played music.\n"
    source = message_source(message(text), CTX)
    assert [u.speaker for u in source.units] == ["Caroline", "Melanie"]
    for unit in source.units:
        start, end = unit.parent_span
        assert text[start:end] == unit.text
        assert unit.evidence_unit_id == digest(source.source_key, start, end)
    assert source.message_id == "message"
    assert len(message_source(message("Unrecognized preamble\n" + text), CTX).units) == 1


def test_fragments_reconstruct_identical_source_regardless_of_split():
    original = message("[Caroline]: 我昨天去了公园。\n[Melanie]: I played music.")
    length = len(original["content"])
    for split in range(1, length):
        pieces = [{**original, "content": original["content"][:split + 1],
                   "parent_offset": 0, "parent_length": length},
                  {**original, "content": original["content"][split:],
                   "parent_offset": split, "parent_length": length}]
        assert reassemble_messages(pieces[::-1]) == [original]
        assert message_source(reassemble_messages(pieces)[0], CTX) == message_source(original, CTX)


@pytest.mark.parametrize("pieces", [
    [message("ab", parent_offset=0, parent_length=3)],
    [message("ab", parent_offset=0, parent_length=3), message("cd", parent_offset=1, parent_length=3)],
    [message("ab", parent_offset=-1, parent_length=2)],
    [message("ab", parent_offset=0, parent_length=2, role="assistant"),
     message("ab", parent_offset=0, parent_length=2)],
    [message("a"), message("b")],
])
def test_invalid_fragments_fail_closed(pieces):
    with pytest.raises(ValidationError):
        reassemble_messages(pieces)


@pytest.mark.parametrize("changes", [{"id": ""}, {"sequence": None}, {"sequence": True},
                                     {"role": "tool"}, {"recorded_at": "yesterday"}])
def test_no_invented_source_identity(changes):
    with pytest.raises(ValidationError):
        message_source(message(**changes), CTX)


@pytest.mark.parametrize("timezone", ["Unknown/Place", "", "/etc/passwd", "../UTC", 1])
def test_invalid_timezone_is_rejected_before_source_creation(timezone):
    with pytest.raises(ValidationError, match="timezone"):
        message_source(message(timezone=timezone), CTX)


@pytest.mark.parametrize("timezone", [None, "UTC", "Asia/Shanghai", "America/New_York"])
def test_valid_timezone_is_preserved(timezone):
    assert message_source(message(timezone=timezone), CTX).timezone == timezone


@pytest.mark.parametrize("kind", ["sources", "windows", "sessions", "lifecycle"])
@pytest.mark.parametrize("key", ["", "a" * 64])
def test_storage_adapters_agree_and_preserve_user_scope(kind, key):
    uri = evidence_uri(CTX, kind, key)
    parts = agfs_parse(uri)
    assert parts == sql_parse(uri)
    assert parts["owner_type"] == "users" and parts["owner_id"] == "alice"
    assert uri_to_path(uri) == "/accounts/account/users/alice/evidence/" + kind + "/" + (key + "/" if key else "")
    assert check_uri_access(CTX, parts)
    assert not check_uri_access(replace(CTX, user_id="bob"), parts)
    assert not check_uri_access(replace(CTX, account_id="other"), parts)


@pytest.mark.parametrize("uri", [
    "ctx://account/users/alice/evidence/sources/short",
    "ctx://account/users/alice/evidence/sources/../secrets",
    "ctx://account/users/%2e%2e/evidence/sources",
    "ctx://account/users/../evidence/sources",
    "ctx://account/agents/alice/evidence/sources",
    "ctx://account/users/alice/evidence/unknown",
])
def test_malformed_evidence_uri_rejected_by_both_adapters(uri):
    for parser in [agfs_parse, sql_parse, uri_to_path]:
        with pytest.raises(ValueError):
            parser(uri)
