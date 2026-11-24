"""Message-only retrieval, mixed memories, privacy and evidence/fact alignment."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from core.evidence import evidence_uri, message_source
from core.models import CandidateMemory
from core.operation_deadline import operation_deadline
from core.provenance_resolver import ProvenanceResolver
from service.evidence_service import EvidenceService
from tests.unit.evidence.test_sources import CTX, message
from tests.unit.evidence.test_writer import write_candidates


def track_body_reads(fs, monkeypatch):
    read = fs._client.read
    sizes = []

    def track(path, *args, **kwargs):
        content = read(path, *args, **kwargs)
        if path.endswith("/" + fs.FILE_CONTENT):
            sizes.append((path, len(content)))
        return content

    monkeypatch.setattr(fs._client, "read", track)
    return sizes


@pytest.fixture
def deadline_clock(monkeypatch):
    import core.operation_deadline as operation
    import service.evidence_service as search

    clock = SimpleNamespace(now=100.0, budget=1.0)
    clock.monotonic = lambda: clock.now
    monkeypatch.setattr(operation, "monotonic", clock.monotonic)
    monkeypatch.setattr(search, "time", clock)
    return clock


@pytest.fixture(params=["local", "inherited"])
def search_budget(request, monkeypatch, deadline_clock):
    import service.evidence_service as search

    if request.param == "inherited":
        return operation_deadline(deadline_clock.budget)
    monkeypatch.setattr(search, "_SEARCH_TIMEOUT_SECONDS", deadline_clock.budget)
    return operation_deadline()


def test_message_only_and_no_vector_dependency(setup):
    fs, _, store, _ = setup
    store.register([message("Caroline visited Paris.")], CTX, batch_id="batch")
    service = EvidenceService(fs)
    result = service.search("Caroline Paris", CTX, use_memories=True)
    assert len(result.evidence) == 1 and result.facts == []
    assert service.search("son", CTX).evidence == []
    assert service.search("Caroline", replace(CTX, user_id="other")).evidence == []


def test_punctuation_does_not_consume_a_supplementary_query_slot(setup):
    fs, _, store, _ = setup
    store.register([message("I visited Paris.")], CTX, batch_id="branches")
    result = EvidenceService(fs).search("Paris and ? and Rome", CTX)
    assert result.diagnostics["supplementary_queries"] == ["Paris", "Rome"]
    assert result.diagnostics["uncovered_branches"] == [1]


def test_verified_facts_align_with_final_sources_and_roles(setup):
    fs, _, store, _ = setup
    messages = [message("I visited Paris."), message("I recommend visiting Paris.", role="assistant", id="two", sequence=4)]
    write_candidates(setup, messages)
    service = EvidenceService(fs)
    result = service.search("Where did I visit Paris?", CTX, use_memories=True, top_k=1)
    assert len(result.evidence) <= 1
    assert result.facts and all(record["role"] == "user" for record in result.facts)
    ids = {item["evidence_unit_id"] for item in result.evidence}
    assert all(ref["evidence_unit_id"] in ids for record in result.facts for ref in record["source_refs"])
    recommendations = service.search("What did you recommend in Paris?", CTX, use_memories=True)
    assert recommendations.facts and all(record["role"] == "assistant" for record in recommendations.facts)
    store.revoke_session(CTX)
    assert service.search("Paris", CTX, use_memories=True).evidence == []


@pytest.mark.parametrize("category", ["entity", "profile"])
def test_legacy_llm_provenance_supplements_raw_source_without_inventing_facts(setup, category):
    fs, api, store, _ = setup
    store.register([message("I played it yesterday.")], CTX, batch_id="batch", archive_id="archive")
    api.write_memory(CandidateMemory(category=category, owner_scope="user", routing_key="piano",
        confidence=1.0, abstract="Piano practice", overview="Piano practice", content="Piano practice",
        provenance_ids=[ProvenanceResolver.build_id("archive", "archive", ["message"])]), CTX)
    service = EvidenceService(fs)
    assert service.search("piano", CTX).evidence == []
    result = service.search("piano", CTX, use_memories=True)
    assert len(result.evidence) == 1 and result.facts == []


@pytest.mark.parametrize("options", [{"top_k": True}, {"use_memories": "false"}, {"unknown": 1}, {"top_k": 51}])
def test_shared_validation_rejects_coercion_and_unknown_fields(setup, options):
    with pytest.raises(ValidationError):
        EvidenceService(setup[0]).search("Paris", CTX, **options)


def test_inflections_preserve_word_boundaries_and_places_are_not_subjects(setup):
    fs, _, _, _ = setup
    messages = [message("I visited Paris.")]
    write_candidates(setup, messages)
    service = EvidenceService(fs)
    assert service.search("visit", CTX).evidence
    assert service.search("Paris", CTX, use_memories=True).facts
    assert not service.search("Did Caroline visit Paris?", CTX, use_memories=True).facts


def test_question_and_pronoun_groups_respect_top_k_and_topic_boundaries(setup):
    fs, _, store, _ = setup
    store.register([message("Did you visit Paris?", sequence=0),
                    message("Yes, I visited Paris.", id="two", sequence=1),
                    message("It was my favorite visit.", id="three", sequence=2),
                    message("She repaired a piano.", id="four", sequence=3)], CTX, batch_id="group")
    service = EvidenceService(fs)
    result = service.search("visit Paris", CTX, top_k=3)
    assert {item["message_id"] for item in result.evidence} == {"message", "two", "three"}
    assert len(service.search("Paris", CTX, top_k=1).evidence) <= 1


def test_ordinary_pipeline_returns_both_engines_and_rejects_revoked_vector(setup):
    from core.models import RetrievalConfig
    from index.index_record_builder import build_index_records
    from providers.embedder.mock_embedder import MockEmbedder
    from providers.vector_index.in_memory_index import InMemoryVectorIndex
    from retrieval.context_reader import ContextReader
    from service.api import ReadAPI
    from tests.unit.retrieval.test_pipeline import _build_pipeline

    fs, writer, store, _ = setup
    old = writer.write_memory(CandidateMemory(category="entity", owner_scope="user", routing_key="paris",
        abstract="Paris visit", overview="Paris visit", content="I enjoy Paris visits.", confidence=1.0), CTX)
    new, = write_candidates(setup, [message("I visited Paris.")], categories={"fact"})
    cfg = RetrievalConfig(expand_hierarchy=False, hotness_alpha=0.0, default_top_k=10, global_search_topk=20)
    embedder, index = MockEmbedder(dimension=8), InMemoryVectorIndex(dimension=8)
    for uri in [old["target_uri"], new.target_uri]:
        records = build_index_records(fs.read_node(uri, CTX))
        for record in records:
            record.vector = embedder.embed_texts([record.text])[0]
        index.upsert(records)
    pipeline = _build_pipeline(index, embedder, cfg, with_expand=False)
    pipeline._context_reader = ContextReader(fs)
    api = ReadAPI(pipeline, pipeline._context_reader, cfg)
    result = api.search_memory("Paris", CTX, score_threshold=-1)
    assert {hit.category for hit in result.hits} == {"entity", "fact"}
    assert {hit.category for hit in api.search_memory("Paris", CTX, categories=["entity"], score_threshold=-1).hits} == {"entity"}
    store.revoke_session(CTX)
    assert {hit.category for hit in api.search_memory("Paris", CTX, score_threshold=-1).hits} == {"entity"}


@pytest.mark.parametrize("memories,events", [(False, False), (False, True), (True, False), (True, True)])
def test_event_and_memory_channels_are_independent_and_events_are_capped(setup, memories, events):
    fs, _, _, _ = setup
    messages = [message(f"I visited Paris on 2026-05-0{i + 1}.", id=str(i), sequence=i) for i in range(3)]
    write_candidates(setup, messages)
    result = EvidenceService(fs).search(
        "Paris visit", CTX, use_memories=memories, use_events=events)
    assert result.diagnostics["events_admitted"] == (2 if events else 0)
    assert bool(result.facts) == (memories or events)
    if not memories and events:
        assert len(result.facts) == 2


@pytest.mark.parametrize("limit", ["count", "shared_bytes"])
def test_source_scan_limits_respect_remaining_card_budget(setup, monkeypatch, limit):
    import service.evidence_service as search

    fs, _, _, _ = setup
    messages = sorted([message("Paris.", id=str(i), sequence=i * 2) for i in range(3)],
                      key=lambda item: message_source(item, CTX).source_key)
    for index, msg in enumerate(messages, start=1):
        msg["content"] = f"I visited Paris on 2026-05-0{index}."
    first = messages[0]
    write_candidates(setup, messages, categories={"fact", "event"})
    count_limit = limit == "count"
    if count_limit:
        monkeypatch.setattr(search, "_SOURCE_SCAN_LIMIT", 1)
    else:
        monkeypatch.setattr(search, "_SCAN_BYTE_LIMIT", len(first["content"].encode("utf-8")))

    paths = [fs._uri_to_agfs_path(evidence_uri(CTX, "sources", message_source(msg, CTX).source_key))
             + fs.FILE_CONTENT for msg in messages]
    expected_reads = paths[:1 if count_limit else 2] + paths[:1]
    expected_bytes = sum(len(fs._client.read(path)) for path in expected_reads)
    reads = track_body_reads(fs, monkeypatch)

    # The exact-date events are outside the source scan and would take both event slots.
    result = EvidenceService(fs).search("Paris visit 2026-05-02 2026-05-03", CTX, use_memories=True, use_events=True)

    assert result.diagnostics["scanned_sources"] == (1 if count_limit else 2)
    assert result.diagnostics["partial"] and result.diagnostics["truncated"]
    assert result.diagnostics["events_matched"] == int(count_limit)
    assert result.diagnostics["events_admitted"] == int(count_limit)
    assert [item["message_id"] for item in result.evidence] == [first["id"]]
    assert [fact["quote"] for fact in result.facts] == ([first["content"]] if count_limit else [])
    if not count_limit:
        assert result.diagnostics["scanned_cards"] == 0
    source_reads = [(path, size) for path, size in reads if path in paths]
    assert [path for path, _ in source_reads] == expected_reads
    assert sum(size for _, size in source_reads) == expected_bytes


def test_card_cannot_load_another_source_with_a_scanned_unit_id(setup, monkeypatch):
    fs, _, _, _ = setup
    messages = sorted([message(f"I visited Paris on 2026-05-0{i}.", id=str(i), sequence=i)
                       for i in range(1, 3)], key=lambda item: message_source(item, CTX).source_key)
    write_candidates(setup, messages, categories={"fact"})
    scanned, outside = (message_source(msg, CTX) for msg in messages)
    read_node = fs.read_node

    def corrupt_card(uri, ctx):
        node = read_node(uri, ctx)
        if node.category == "fact":
            for record in node.metadata["deterministic"]["fact_records"]:
                ref = record["source_refs"][0]
                if ref["source_key"] == outside.source_key:
                    ref["evidence_unit_id"] = scanned.units[0].evidence_unit_id
        return node

    monkeypatch.setattr(fs, "read_node", corrupt_card)
    monkeypatch.setattr("service.evidence_service._SOURCE_SCAN_LIMIT", 1)
    reads = track_body_reads(fs, monkeypatch)
    outside_path = fs._uri_to_agfs_path(evidence_uri(CTX, "sources", outside.source_key)) + fs.FILE_CONTENT

    result = EvidenceService(fs).search("Paris", CTX, use_memories=True)

    assert [fact["quote"] for fact in result.facts] == [scanned.text]
    assert all(path != outside_path for path, _ in reads)


@pytest.mark.parametrize("limit", ["card_bytes", "shared_bytes"])
def test_card_scan_reads_no_bodies_after_the_budget_stops_it(setup, monkeypatch, limit):
    fs, _, _, _ = setup
    messages = [message(f"I visited Paris and completed {i} essays.", id=str(i), sequence=i * 2)
                for i in range(1, 4)]
    cards = sorted(plan.target_uri for plan in write_candidates(setup, messages, categories={"fact"}))
    service = EvidenceService(fs)
    original = fs.read_node(cards[1], CTX)
    # Exercise the real storage read with multibyte bodies instead of only checking scan counters.
    content = "字" * (1024 * 1024 if limit == "card_bytes" else 200)
    fs.write_node(replace(original, content=content), CTX)
    paths = [fs._uri_to_agfs_path(uri) + fs.FILE_CONTENT for uri in cards]
    expected_bytes = sum(len(fs._client.read(path)) for path in paths[:2])
    if limit == "shared_bytes":
        monkeypatch.setattr("service.evidence_service._SCAN_BYTE_LIMIT",
                            sum(len(msg["content"].encode("utf-8")) for msg in messages) + expected_bytes - 1)
    reads = track_body_reads(fs, monkeypatch)

    result = service.search("Paris essays", CTX, use_memories=True)

    assert result.evidence and result.facts
    assert result.diagnostics["partial"] and result.diagnostics["truncated"]
    assert result.diagnostics["scanned_cards"] == 2
    card_reads = [(path, size) for path, size in reads if path in paths]
    assert [path for path, _ in card_reads] == paths[:2]
    assert sum(size for _, size in card_reads) == expected_bytes


@pytest.mark.parametrize("batch", [False, True])
def test_shared_dependencies_are_removed_when_their_source_fails_final_check(setup, monkeypatch, batch):
    import service.evidence_service as search

    fs, _, store, _ = setup
    first = message("I opened a studio.", id="first", sequence=0)
    following = message("That studio sounds wonderful.", id="following", sequence=1, role="assistant")
    store.register([first, following], CTX, batch_id="dependency")
    service = EvidenceService(fs)
    if not batch:
        monkeypatch.setattr(type(fs), "read_nodes", None)
    root = message_source(first, CTX).source_key
    original = service.store.read_source
    select_groups = search.select_groups

    def read_source(key, ctx, **kwargs):
        if key == root:
            return None
        return original(key, ctx, **kwargs)

    def invalidate_after_selection(*args, **kwargs):
        selected = select_groups(*args, **kwargs)
        monkeypatch.setattr(service.store, "read_source", read_source)
        return selected

    monkeypatch.setattr(search, "select_groups", invalidate_after_selection)
    result = service.search("studio", CTX, top_k=10)
    assert result.diagnostics["partial"]
    assert result.evidence == [] and result.diagnostics["evidence_groups"] == []


@pytest.mark.parametrize("state", ["active", "revoked", "new_version"])
def test_final_check_reads_one_body_and_manifest_for_all_units_of_a_source(setup, monkeypatch, state):
    import service.evidence_service as search

    fs, _, store, _ = setup
    msg = message("[Alex]: Alex visited Paris.\n[Blair]: Blair visited Paris.")
    write_candidates(setup, [msg], categories={"fact"})
    source_uri = evidence_uri(CTX, "sources", message_source(msg, CTX).source_key)
    node = fs.read_node(source_uri, CTX)
    body_path = fs._uri_to_agfs_path(source_uri) + fs.FILE_CONTENT
    manifest_path = fs._uri_to_agfs_path(store.session_uri(CTX)) + fs.FILE_CONTENT
    reads = track_body_reads(fs, monkeypatch)
    select_groups = search.select_groups

    def change_after_selection(*args, **kwargs):
        selected, groups = select_groups(*args, **kwargs)
        assert len(selected) == 2
        if state == "revoked":
            store.revoke_session(CTX)
        elif state == "new_version":
            metadata = {**node.metadata, "_expected_version": node.metadata["version"]}
            fs.write_node(replace(node, metadata=metadata), CTX)
        reads.clear()
        return selected, groups

    monkeypatch.setattr(search, "select_groups", change_after_selection)
    result = EvidenceService(fs).search("Paris", CTX, use_memories=True)

    assert [path for path, _ in reads] == [body_path, manifest_path]
    assert reads[0][1] == len(node.content.encode("utf-8"))
    assert result.diagnostics["partial"] == (state != "active")
    assert len(result.evidence) == len(result.facts) == (2 if state == "active" else 0)


@pytest.mark.parametrize("available_reads", [0, 2, 4])
def test_final_source_checks_stop_at_deadline_and_keep_only_complete_groups(
    setup, monkeypatch, deadline_clock, available_reads,
):
    import service.evidence_service as search

    fs, _, _, _ = setup
    write_candidates(setup, [
        message("[Alex]: Alex visited Paris.\n[Blair]: Blair visited Paris.", id="paris", sequence=0),
        message("I visited Rome.", id="rome", sequence=2),
        message("It was unforgettable.", id="followup", sequence=3, role="assistant"),
    ])
    service = EvidenceService(fs)
    checks = []
    read_source = service.store.read_source

    def check_source(key, ctx, **kwargs):
        source = read_source(key, ctx, **kwargs)
        checks.append(key)
        if len(checks) == available_reads:
            deadline_clock.now += deadline_clock.budget
        return source

    def select_then_expire(_query, _scores, units, *_args, **_kwargs):
        paris = [key for key, unit in units.items() if unit["message_id"] == "paris"]
        keys = {unit["message_id"]: key for key, unit in units.items() if unit["message_id"] != "paris"}
        monkeypatch.setattr(service.store, "read_source", check_source)
        if available_reads == 0:
            deadline_clock.now += deadline_clock.budget
        rome, followup = keys["rome"], keys["followup"]
        return [paris[0], rome, paris[1], followup], [paris, [rome, followup]]

    monkeypatch.setattr(search, "select_groups", select_then_expire)
    with operation_deadline(deadline_clock.budget):
        result = service.search("Where did someone travel to Paris and Rome?", CTX, use_memories=True)

    assert len(checks) == len(set(checks)) == min(available_reads, 3)
    paris_quotes = {"Alex visited Paris.", "Blair visited Paris."}
    assert result.diagnostics["partial"] == (available_reads < 3)
    assert [item["message_id"] for item in result.evidence] == {
        0: [], 2: ["paris", "paris"], 4: ["paris", "rome", "paris", "followup"],
    }[available_reads]
    assert {fact["quote"] for fact in result.facts} == {
        0: set(), 2: paris_quotes, 4: paris_quotes | {"I visited Rome."},
    }[available_reads]
    assert len(result.diagnostics["evidence_groups"]) == {0: 0, 2: 1, 4: 2}[available_reads]
    assert result.diagnostics["uncovered_branches"] == {0: [0, 1], 2: [1], 4: []}[available_reads]


def test_final_source_read_cannot_start_manifest_read_after_deadline(
    setup, monkeypatch, deadline_clock, search_budget,
):
    import service.evidence_service as search
    from core.evidence import evidence_uri, message_source

    fs, _, store, _ = setup
    msg = message("I visited Paris.")
    store.register([msg], CTX, batch_id="single-source")
    source_uri = evidence_uri(CTX, "sources", message_source(msg, CTX).source_key)
    select_groups, read_node = search.select_groups, fs.read_node
    reads = []

    def expire_after_source(uri, ctx):
        node = read_node(uri, ctx)
        reads.append(uri)
        if uri == source_uri:
            deadline_clock.now += deadline_clock.budget
        return node

    def track_final_reads(*args, **kwargs):
        selected = select_groups(*args, **kwargs)
        monkeypatch.setattr(fs, "read_node", expire_after_source)
        return selected

    monkeypatch.setattr(search, "select_groups", track_final_reads)
    with search_budget:
        result = EvidenceService(fs).search("Paris", CTX)

    assert reads == [source_uri]
    assert result.diagnostics["partial"]
    assert result.evidence == [] and result.diagnostics["evidence_groups"] == []


def test_time_channel_deadline_marks_partial_without_lexical_hits(setup, monkeypatch, deadline_clock):
    import retrieval.evidence.auxiliary as auxiliary

    fs, _, store, _ = setup
    store.register([message("I visited Paris.", recorded_at="2026-05-01T12:00:00Z")], CTX, batch_id="time")
    service = EvidenceService(fs)
    query = "2026-05-01"
    assert service.search(query, CTX, use_time=True).evidence
    fuse_auxiliary = auxiliary.fuse_auxiliary

    def expire_before_fusion(request, scores, *args, **kwargs):
        assert not scores
        deadline_clock.now += deadline_clock.budget
        return fuse_auxiliary(request, scores, *args, **kwargs)

    monkeypatch.setattr(auxiliary, "fuse_auxiliary", expire_before_fusion)
    with operation_deadline(deadline_clock.budget):
        result = service.search(query, CTX, use_time=True)

    assert not result.evidence
    assert result.diagnostics["auxiliary"]["reason"] == "deadline"
    assert result.diagnostics["partial"]


def test_serial_source_scan_deadline_stops_manifest_read(
    setup, monkeypatch, deadline_clock, search_budget,
):
    fs, _, _, _ = setup
    write_candidates(setup, [message("I visited Paris.")])
    read_node = fs.read_node
    reads = []

    def expire_on_single_source(uri, ctx):
        reads.append(uri.split("/")[-2])
        node = read_node(uri, ctx)
        if reads[-1] == "sources":
            deadline_clock.now += deadline_clock.budget
        return node

    monkeypatch.setattr(type(fs), "read_nodes", None)
    monkeypatch.setattr(fs, "read_node", expire_on_single_source)
    with search_budget:
        result = EvidenceService(fs).search("Paris", CTX, use_memories=True)

    assert reads == ["sources"]
    assert result.diagnostics["partial"] and not result.evidence and not result.facts


def test_agfs_batch_does_not_refill_workers_after_evidence_expiry(setup, monkeypatch, deadline_clock):
    from core.errors import OperationDeadlineExceededError

    fs, _, store, _ = setup
    sources = store.register([message("I visited Paris.", id=str(i), sequence=i) for i in range(8)],
                             CTX, batch_id="batch-deadline")
    first_page = sorted(evidence_uri(CTX, "sources", source["source_key"]) for source in sources)
    read_node = fs.read_node
    reads = []

    def expire_after_first(uri, ctx):
        reads.append(uri)
        node = read_node(uri, ctx)
        if uri == first_page[0]:
            deadline_clock.now += deadline_clock.budget
        return node

    monkeypatch.setattr(fs, "read_node", expire_after_first)
    with operation_deadline(deadline_clock.budget), pytest.raises(OperationDeadlineExceededError):
        list(store.node_pages(first_page, CTX, read_limit=len(first_page)))
    assert first_page[0] in reads and set(reads) <= set(first_page[:4])


def test_expired_query_does_not_start_card_listing_or_group_selection(setup, monkeypatch, deadline_clock):
    import service.evidence_service as search

    fs, _, store, _ = setup
    store.register([message("I visited Paris.")], CTX, batch_id="deadline")
    lexical_scores = search.lexical_scores

    def expire_after_lexical(*args):
        scores = lexical_scores(*args)
        deadline_clock.now += deadline_clock.budget
        return scores

    def unexpected_work(*args, **kwargs):
        pytest.fail("Started new work after the evidence deadline")

    pager = type(fs).list_node_summaries_page

    def list_page(self, uri, ctx, **kwargs):
        if "/memories/" in uri:
            unexpected_work()
        return pager(self, uri, ctx, **kwargs)

    monkeypatch.setattr(search, "lexical_scores", expire_after_lexical)
    monkeypatch.setattr(type(fs), "list_node_summaries_page", list_page)
    monkeypatch.setattr(search, "select_groups", unexpected_work)
    with operation_deadline(deadline_clock.budget):
        result = EvidenceService(fs).search("Paris", CTX, use_memories=True)
    assert result.diagnostics["partial"] and not result.evidence


@pytest.mark.parametrize("source_count", [1, 3])
def test_time_channel_stops_on_expiry_and_reports_partial_when_empty(
    setup, monkeypatch, deadline_clock, source_count,
):
    import retrieval.evidence.auxiliary as auxiliary

    fs, _, store, _ = setup
    store.register([message("I visited Paris.", id=str(i), sequence=i, recorded_at="2026-06-01T12:00:00Z")
                    for i in range(source_count)], CTX, batch_id="time-miss")
    resolve = auxiliary.resolve_event_time
    calls = []

    def expire_after_first(*args):
        calls.append(args[0])
        deadline_clock.now += deadline_clock.budget
        return resolve(*args)

    monkeypatch.setattr(auxiliary, "resolve_event_time", expire_after_first)
    with operation_deadline(deadline_clock.budget):
        result = EvidenceService(fs).search("2026-05-01", CTX, use_time=True)
    assert len(calls) == 1
    assert result.diagnostics["partial"] and not result.evidence


def test_reported_groups_only_reference_final_evidence_within_response_budget(setup):
    import json

    fs, _, store, _ = setup
    store.register([message("I completed studio repairs. " * 80, id=str(i), sequence=i * 10)
                    for i in range(50)], CTX, batch_id="budget")
    result = EvidenceService(fs).search("studio repairs", CTX, top_k=50)
    selected = {item["evidence_unit_id"] for item in result.evidence}
    assert selected and result.diagnostics["truncated"]
    assert {key for group in result.diagnostics["evidence_groups"] for key in group} == selected
    assert len(json.dumps(result.to_dict(), ensure_ascii=False).encode()) < 64 * 1024
