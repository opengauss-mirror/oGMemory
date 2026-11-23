"""Batch parsing changes execution only, preserving documents and candidates."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import core.operation_deadline as deadline
import extraction.deterministic.extractor as module
from core.errors import OperationDeadlineExceededError
from core.evidence import message_source
from extraction.deterministic.extractor import DeterministicExtractor
from extraction.deterministic.nlp import (
    DeterministicConfig,
    TextParser,
    reuse_documents,
    verified_document,
)
from extraction.deterministic.verification import verify_record
from extraction.schemas.registry import SchemaRegistry
from tests.unit.evidence.test_extractor import RulesParser, extractor
from tests.unit.evidence.test_sources import CTX, message


class BatchedRules(RulesParser):
    def __init__(self):
        self.batches = []

    def parse_many(self, texts):
        self.batches.append(texts)
        return [None] * len(texts)


def test_extraction_batches_across_units_without_changing_limits_or_candidates():
    messages = [message("I visited Paris. " * 8, id=str(i), sequence=i) for i in range(10)]
    messages += [message("I might visit Rome. Did I visit Rome?", id="questions", sequence=11)]
    expected = extractor().extract(messages, CTX)
    parser = BatchedRules()
    actual = DeterministicExtractor(SchemaRegistry(), parser=parser).extract(messages, CTX)
    assert actual == expected
    assert list(map(len, parser.batches)) == [32, 32, 16]


def test_language_routing_dedup_and_missing_model_behaviour_are_preserved():
    parser = object.__new__(TextParser)
    english, chinese = Mock(), Mock()
    english.pipe.side_effect = lambda texts, **kwargs: ["en:" + text for text in texts]
    chinese.pipe.side_effect = lambda texts, **kwargs: ["zh:" + text for text in texts]
    parser.models = {"en": english, "zh": chinese}
    texts = ["Paris", "我去了北京。", "Paris", "Rome"]
    assert parser.parse_many(texts) == ["en:Paris", "zh:我去了北京。", "en:Paris", "en:Rome"]
    english.pipe.assert_called_once_with(["Paris", "Rome"], batch_size=32)
    parser.models.pop("zh")
    assert parser.parse_many(["我去了北京。", "Paris"])[0] is None
    english.pipe.side_effect = RuntimeError("parser failed")
    with pytest.raises(RuntimeError, match="parser failed"):
        parser.parse_many(["Paris"])


def test_document_reuse_is_operation_local_model_scoped_and_bounded():
    model = Mock(side_effect=lambda text: text)
    signature = {"name": "model", "version": "1"}
    with reuse_documents():
        assert verified_document(model, signature, "quote") == "quote"
        verified_document(model, signature, "quote")
        assert model.call_count == 1
        verified_document(model, {**signature, "version": "2"}, "quote")
        assert model.call_count == 2
        for i in range(129):
            verified_document(model, signature, str(i))
        verified_document(model, signature, "quote")
        assert model.call_count == 132
    with reuse_documents():
        verified_document(model, signature, "quote")
    assert model.call_count == 133


def test_reusing_a_document_never_reuses_source_validity_or_model_version(setup, monkeypatch):
    store = setup[2]
    store.register([message()], CTX, batch_id="batch")
    record = extractor().extract([message()], CTX)[0].metadata["deterministic"]["fact_records"][0]
    record["nlp_model"] = {"name": "fake", "version": "1"}
    model = Mock(return_value=None, meta={"version": "1"})
    monkeypatch.setattr("extraction.deterministic.nlp.load_model", lambda name: model)
    with reuse_documents():
        assert verify_record(record, store, CTX) and verify_record(record, store, CTX)
        assert model.call_count == 1
        tampered = deepcopy(record)
        tampered["subject"] = "invented"
        assert not verify_record(tampered, store, CTX)
        model.meta["version"] = "2"
        assert not verify_record(record, store, CTX)
        model.meta["version"] = "1"
        store.revoke_session(CTX)
        assert not verify_record(record, store, CTX)
    assert model.call_count == 1


def test_real_spacy_batch_candidates_and_documents_match_serial_execution():
    pytest.importorskip("spacy")
    try:
        parser = TextParser(DeterministicConfig())
    except RuntimeError as exc:
        pytest.skip(str(exc))
    texts = ["Caroline visited Paris yesterday.", "我完成了7篇文章。", "Caroline visited Paris yesterday."]
    assert [doc.to_json() for doc in parser.parse_many(texts)] == [parser.parse(text).to_json() for text in texts]
    serial = SimpleNamespace(parse=parser.parse, signature=parser.signature, degraded_languages=parser.degraded_languages)
    messages = [message("[Alex]: I visited Paris. I completed 7 essays.\n[Morgan]: Alex repaired a telescope."),
                message("我完成了7篇文章。", id="zh", sequence=10),
                message("I recommend visiting Paris.", role="assistant", id="assistant", sequence=11)]
    assert DeterministicExtractor(SchemaRegistry(), parser=parser).extract(messages, CTX) == (
        DeterministicExtractor(SchemaRegistry(), parser=serial).extract(messages, CTX))


# Quotation filtering stays linear and observes the caller's deadline.


def test_quotation_scan_runs_once_per_unit_and_preserves_overlap_filtering(monkeypatch):
    messages = [message('[Alex]: I visited Rome.\n"I adopted a dog. I moved to Paris."\n'
                        'I completed 7 essays.\n[Morgan]: I joined a club.\n'
                        '“I won a prize. I visited Paris.”\nI played music.'),
                message("I repaired a telescope.\n'I visited Paris. I adopted a dog.'\n"
                        "I bought a book.", id="two", sequence=4)]
    sources = [message_source(item, CTX) for item in messages]
    scan = Mock(wraps=module._QUOTED.finditer)
    monkeypatch.setattr(module, "_QUOTED", SimpleNamespace(finditer=scan))
    engine = DeterministicExtractor(SchemaRegistry(), parser=BatchedRules())
    rows = list(engine._parsed_sentences(sources))
    assert [row[3] for row in rows] == ["I visited Rome.", "I completed 7 essays.",
                                       "I joined a club.", "I played music.",
                                       "I repaired a telescope.", "I bought a book."]
    assert scan.call_count == sum(len(source.units) for source in sources)


@pytest.mark.parametrize("text", ["Did I visit Paris? ", '"I visited Paris." '])
def test_filtered_sentences_stop_at_deadline_without_waiting_for_an_nlp_batch(monkeypatch, text):
    clock, visited = [100.0], []
    monkeypatch.setattr(deadline, "monotonic", lambda: clock[0])
    finditer = module._SENTENCE.finditer

    def sentences(value):
        for match in finditer(value):
            visited.append(match)
            clock[0] = 102.0
            yield match

    monkeypatch.setattr(module, "_SENTENCE", SimpleNamespace(finditer=sentences))
    parser = BatchedRules()
    engine = DeterministicExtractor(SchemaRegistry(), parser=parser)
    with deadline.operation_deadline(1), pytest.raises(OperationDeadlineExceededError):
        engine.extract([message(text * 320)], CTX)
    assert len(visited) <= 32
    assert parser.batches == []


def test_expired_nlp_batch_cannot_yield_assertions(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(deadline, "monotonic", lambda: clock[0])

    class SlowParser(BatchedRules):
        def parse_many(self, texts):
            clock[0] = 102.0
            return super().parse_many(texts)

    parser = SlowParser()
    engine = DeterministicExtractor(SchemaRegistry(), parser=parser)
    source = message_source(message("I visited Paris. " * 64), CTX)
    with deadline.operation_deadline(1), pytest.raises(OperationDeadlineExceededError):
        next(engine._parsed_sentences([source]))
    assert len(parser.batches) == 1
