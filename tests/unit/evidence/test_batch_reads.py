"""Batching preserves ordering, validation, limits and final revocation checks."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from core.errors import AccessDeniedError, NodeNotFoundError, OperationDeadlineExceededError
from core.evidence import digest, evidence_uri
from service.evidence_service import EvidenceService
from tests.unit.evidence.test_extractor import extractor
from tests.unit.evidence.test_sources import CTX, message
from tests.unit.evidence.test_writer import write_candidates


def install_batch(fs, monkeypatch):
    read = fs.read_node
    calls = []

    def read_nodes(self, uris, ctx):
        calls.append(list(uris))
        values = {}
        for uri in uris:
            try:
                values[uri] = read(uri, ctx)
            except Exception as exc:
                values[uri] = exc
        return values

    monkeypatch.setattr(type(fs), "read_nodes", read_nodes, raising=False)
    return calls


def stable(result):
    result = deepcopy(result.to_dict())
    for key in ("elapsed_ms", "response_bytes"):
        result["diagnostics"].pop(key)
    return result


def test_batch_and_serial_results_match_with_facts_and_events(setup, monkeypatch):
    fs, _, _, _ = setup
    messages = [message(f"I visited Paris and completed {i} essays.", id=str(i), sequence=i) for i in range(35)]
    write_candidates(setup, messages)
    service = EvidenceService(fs)
    monkeypatch.setattr(type(fs), "read_nodes", None)
    expected = service.search("Paris essays", CTX, use_memories=True, use_events=True)
    calls = install_batch(fs, monkeypatch)
    actual = service.search("Paris essays", CTX, use_memories=True, use_events=True)
    assert stable(actual) == stable(expected)
    assert calls and all(len(uris) == 1 for uris in calls)


def test_read_limit_sentinel_does_not_fetch_or_raise_later_errors(setup, monkeypatch):
    fs, _, store, _ = setup
    source = store.register([message()], CTX, batch_id="batch")[0]
    uri = evidence_uri(CTX, "sources", source["source_key"])
    missing = evidence_uri(CTX, "sources", digest("missing"))
    calls = install_batch(fs, monkeypatch)
    pages = list(store.node_pages([uri, missing], CTX, read_limit=1))
    assert len(pages) == 2 and pages[-1] == [(missing, None)]
    assert calls == [[uri]]
    sources = store.iter_sources([uri, missing], CTX, read_limit=1)
    assert next(sources)[1]()["source_key"] == source["source_key"]
    assert next(sources)[1]() is None


def test_source_scan_never_replaces_final_authority_check(setup, monkeypatch):
    import service.evidence_service as search

    fs, _, store, _ = setup
    write_candidates(setup, [message("I visited Paris.")], categories={"fact"})
    install_batch(fs, monkeypatch)
    select_groups = search.select_groups

    def revoke(*args, **kwargs):
        selected = select_groups(*args, **kwargs)
        assert selected[0]
        store.revoke_session(CTX)
        return selected

    monkeypatch.setattr(search, "select_groups", revoke)
    result = EvidenceService(fs).search("Paris", CTX, use_memories=True)
    assert not result.evidence and not result.facts and result.diagnostics["partial"]


def test_batch_owner_errors_are_rechecked_in_order_and_missing_sources_are_empty(setup, monkeypatch):
    fs, _, store, _ = setup
    source = store.register([message()], CTX, batch_id="batch")[0]
    uri = evidence_uri(CTX, "sources", source["source_key"])
    original = fs.read_node(uri, CTX)
    foreign = deepcopy(original)
    foreign.owner_space = "user:other"
    missing = evidence_uri(CTX, "sources", digest("missing"))
    monkeypatch.setattr(type(fs), "read_nodes", lambda self, uris, ctx: {
        key: foreign if key == uri else NodeNotFoundError(key) for key in uris}, raising=False)
    values = store.iter_sources([missing, uri], CTX, read_limit=2)
    assert next(values)[1]() is None
    with pytest.raises(AccessDeniedError):
        next(values)[1]()


def test_record_verification_reads_current_source_once_per_reference(setup, monkeypatch):
    from extraction.deterministic.verification import verify_record

    store = setup[2]
    store.register([message()], CTX, batch_id="batch")
    record = extractor().extract([message()], CTX)[0].metadata["deterministic"]["fact_records"][0]
    read = Mock(wraps=store.read_source)
    monkeypatch.setattr(store, "read_source", read)
    assert verify_record(record, store, CTX)
    assert read.call_count == len(record["source_refs"])
    store.revoke_session(CTX)
    assert not verify_record(record, store, CTX)


@pytest.mark.parametrize("refs", [[], (), None, 1])
def test_verify_record_rejects_invalid_reference_collections_before_lookup(refs):
    from extraction.deterministic.verification import verify_record

    record = extractor().extract([message()], CTX)[0].metadata["deterministic"]["fact_records"][0]
    resolve = Mock()
    assert not verify_record({**record, "source_refs": refs}, None, CTX, resolve=resolve)
    resolve.assert_not_called()


def test_expired_deadline_does_not_issue_a_prefetch_query(setup, monkeypatch):
    fs, _, store, _ = setup
    uri = evidence_uri(CTX, "sources", digest("source"))
    calls = install_batch(fs, monkeypatch)
    with pytest.raises(OperationDeadlineExceededError):
        list(store.node_pages([uri], CTX, read_limit=1, deadline=0))
    assert calls == []
