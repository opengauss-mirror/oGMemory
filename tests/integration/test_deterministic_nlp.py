"""Opt in with OGMEM_REAL_NLP_TESTS=1; selected runs REQUIRE both real models."""

import os
from types import SimpleNamespace

import pytest

from extraction.deterministic.extractor import DeterministicExtractor, fact_id
from extraction.deterministic.nlp import parser_language
from extraction.deterministic.verification import verify_record
from extraction.schemas.registry import SchemaRegistry
from fs.agfs_adapter import AGFSContextFS
from service.evidence_store import EvidenceStore
from tests.unit.evidence.test_sources import CTX, message
from tests.unit.fs.test_agfs_context_fs import _AtomicMemoryAGFSClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("OGMEM_REAL_NLP_TESTS") != "1", reason="opt-in real NLP suite"),
]


@pytest.fixture(scope="module")
def extractor():
    # No importorskip or mock: a missing or incompatible model fails this suite.
    return DeterministicExtractor(SchemaRegistry())


@pytest.mark.parametrize("text,subject,predicate,obj", [
    ("I completed 7 essays this year.", "user", "complete", "essays"),
    ("Oliver hid his bone under Melanie's slipper.", "Oliver", "hide", "bone"),
    ("我昨天参加了读书会。", "user", "参加", "读书会"),
    ("我完成了7篇文章。", "user", "完成", "文章"),
    ("我喜欢音乐。", "user", "喜欢", "音乐"),
])
def test_real_subject_predicate_object_and_source_verification(extractor, text, subject, predicate, obj):
    fs = AGFSContextFS(_AtomicMemoryAGFSClient(), mount_prefix="/local")
    store = EvidenceStore(fs)
    messages = [message(text, recorded_at="2026-05-08T10:00:00+08:00")]
    store.register(messages, CTX, batch_id="batch")
    records = [candidate.metadata["deterministic"]["fact_records"][0]
               for candidate in extractor.extract(messages, CTX) if candidate.category == "fact"]
    assert records
    assert (records[0]["subject"], records[0]["predicate"], records[0]["object"]) == (subject, predicate, obj)
    assert verify_record(records[0], store, CTX)
    if "昨天" in text:
        assert records[0]["event_time"] == "2026-05-07"
    assert records[0]["nlp_model"]["version"]


def test_real_models_preserve_transcript_speakers_and_independent_channels(extractor):
    messages = [message("[Caroline]: I visited Paris yesterday.\n[Melanie]: I completed 7 essays.",
                        recorded_at="2026-05-08T10:00:00+08:00"),
                message("[Caroline]: I attended a concert.", id="two", sequence=4)]
    candidates = extractor.extract(messages, CTX)
    assert {candidate.category for candidate in candidates} == {"fact", "entity", "event"}
    facts = [candidate.metadata["deterministic"]["fact_records"][0]
             for candidate in candidates if candidate.category == "fact"]
    assert {record["subject"] for record in facts} == {"Caroline", "Melanie"}
    assert not extractor.extract([message("Hello! I might visit Paris.")], CTX)



@pytest.mark.parametrize("text,role", [
    ("I have not completed 7 essays.", "user"),
    ("I never completed 7 essays.", "user"),
    ("I didn't complete 7 essays.", "user"),
    ("I do not recommend visiting Paris.", "assistant"),
    ("我没有完成7篇文章。", "user"),
    ("我没去北京。", "user"),
    ("我不喜欢音乐。", "user"),
])
def test_real_models_keep_negative_assertions_as_raw_evidence(extractor, text, role):
    assert extractor.extract([message(text, role=role)], CTX) == []



@pytest.mark.parametrize("text,direct", [
    ("I recommend visiting Paris.", True),
    ("I strongly recommend visiting Paris.", True),
    ("We suggest visiting Paris.", True),
    ("Try visiting Paris.", True),
    ("Consider visiting Paris.", True),
    ("我推荐去巴黎。", True),
    ("我们建议去巴黎。", True),
    ("可以试试去巴黎。", True),
    ("Alex and Morgan recommend visiting Paris.", False),
    ("I saw Alex recommend visiting Paris.", False),
    ("Alex建议去巴黎。", False),
])
def test_real_models_preserve_recommendation_actor_in_extraction_and_verification(extractor, text, direct):
    store = EvidenceStore(AGFSContextFS(_AtomicMemoryAGFSClient(), mount_prefix="/local"))
    messages = [message(text, role="assistant")]
    source = store.register(messages, CTX, batch_id="recommendation")[0]
    facts = [c.metadata["deterministic"]["fact_records"][0] for c in extractor.extract(messages, CTX)
             if c.category == "fact"]
    if direct:
        assert facts and all(record["subject"] == "assistant" and verify_record(record, store, CTX) for record in facts)
    else:
        assert facts == []
        record = extractor.extract([message("I recommend visiting Paris.", role="assistant")], CTX)[0].metadata[
            "deterministic"]["fact_records"][0]
        ref = {**record["source_refs"][0], "source_key": source["source_key"],
               "evidence_unit_id": source["units"][0]["evidence_unit_id"], "content_hash": source["content_hash"],
               "span": [0, len(text)], "quote": text}
        record.update(quote=text, object=text, source_refs=[ref], nlp_model=extractor.parser.signature[
            parser_language(text)])
        record["fact_id"] = fact_id(ref, record)
        assert not verify_record(record, store, CTX)


@pytest.mark.parametrize("text,quantity", [
    ("我在2026-05-08去了北京。", None),
    ("我在2026-05-08完成了7篇文章。", 7),
    ("I ran 3.5 miles on 2026-05-08.", 3.5),
    ("I visited Paris in 2026 accompanied by my sister.", None),
    ("I visited Paris after 2026 ended.", None),
    ("I visited Paris in March 2026 accompanied by my sister.", None),
    ("I visited Paris in 2026 autumn.", None),
    ("I visited Paris in late 2026 surrounded by 2 friends.", 2),
    ("I bought 2026 books.", 2026),
    ("I counted around 2026 books.", 2026),
    ("I participated in 2026 experiments.", 2026),
    ("I stayed in 2026 family homes.", 2026),
])
def test_real_models_keep_dates_separate_from_quantities(extractor, text, quantity):
    store = EvidenceStore(AGFSContextFS(_AtomicMemoryAGFSClient(), mount_prefix="/local"))
    messages = [message(text)]
    store.register(messages, CTX, batch_id="dated")
    facts = [c.metadata["deterministic"]["fact_records"][0] for c in extractor.extract(messages, CTX)
             if c.category == "fact"]
    assert facts and all(verify_record(record, store, CTX) for record in facts)
    assert facts[0]["quantity"] is None if quantity is None else facts[0]["quantity"]["value"] == quantity


@pytest.mark.parametrize("text,role,value,kind", [
    ("I measured -7 degrees.", "user", -7, "observation"),
    ("I measured −7.5 degrees.", "user", -7.5, "observation"),
    ("我测得了−7.5公里。", "user", -7.5, "observation"),
    ("I visited Paris while Alex completed 7 essays.", "user", None, None),
    ("While Alex completed 7 essays, I visited Paris.", "user", None, None),
    ("While Alex completed 9 essays, I read 7 books.", "user", 7, "observation"),
    ("I finished 3 and started 2 essays.", "user", None, None),
    ("I saw Alex complete 7 essays.", "user", None, None),
    ("我去了巴黎，而小明完成了7篇文章。", "user", None, None),
    ("I completed 7 essays, and Alex completed 10 in total.", "user", 7, "observation"),
    ("I recommend reading 2 books.", "assistant", 2, "observation"),
    ("我推荐读2本书。", "assistant", 2, "observation"),
    ("我推荐你读2本书。", "assistant", 2, "observation"),
])
def test_real_models_preserve_quantity_value_and_clause(extractor, text, role, value, kind):
    store = EvidenceStore(AGFSContextFS(_AtomicMemoryAGFSClient(), mount_prefix="/local"))
    messages = [message(text, role=role)]
    store.register(messages, CTX, batch_id="quantity")
    fact = next(c for c in extractor.extract(messages, CTX) if c.category == "fact")
    record = fact.metadata["deterministic"]["fact_records"][0]
    assert verify_record(record, store, CTX)
    quantity = record["quantity"]
    if value is None:
        assert quantity is None
    else:
        assert (quantity["value"], quantity["kind"]) == (value, kind)


def test_real_spacy_batch_candidates_and_documents_match_serial_execution(extractor):
    parser = extractor.parser
    texts = ["Caroline visited Paris yesterday.", "我完成了7篇文章。", "Caroline visited Paris yesterday."]
    assert [doc.to_json() for doc in parser.parse_many(texts)] == [parser.parse(text).to_json() for text in texts]
    serial = SimpleNamespace(parse=parser.parse, signature=parser.signature, degraded_languages=parser.degraded_languages)
    messages = [message("[Alex]: I visited Paris. I completed 7 essays.\n[Morgan]: Alex repaired a telescope."),
                message("我完成了7篇文章。", id="zh", sequence=10),
                message("I recommend visiting Paris.", role="assistant", id="assistant", sequence=11)]
    assert DeterministicExtractor(SchemaRegistry(), parser=parser).extract(messages, CTX) == (
        DeterministicExtractor(SchemaRegistry(), parser=serial).extract(messages, CTX))
