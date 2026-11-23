"""Lazy, bounded model reuse; missing models never fall back to a generative LLM."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache

from core.evidence_timing import stage, timed

_documents = ContextVar("evidence_parsed_documents", default=None)


@contextmanager
def reuse_documents():
    """Cache only model computations, never source authority or verified facts."""
    token = _documents.set(OrderedDict())
    try:
        yield
    finally:
        _documents.reset(token)


def verified_document(model, signature, text):
    cache = _documents.get()
    key = (id(model), signature["name"], signature["version"], text)
    if cache is not None and key in cache:
        with stage("nlp_cache_hit"):
            cache.move_to_end(key)
            return cache[key]
    with stage("nlp"):
        doc = model(text)
    if cache is not None:
        cache[key] = doc
        if len(cache) > 128:
            cache.popitem(last=False)
    return doc


@dataclass(frozen=True)
class DeterministicConfig:
    spacy_model_en: str = "en_core_web_sm"
    spacy_model_zh: str = "zh_core_web_sm"
    on_model_missing: str = "error"

    def __post_init__(self):
        if self.on_model_missing not in {"error", "rules_only"}:
            raise ValueError("on_model_missing must be error or rules_only")
        for name in (self.spacy_model_en, self.spacy_model_zh):
            if not isinstance(name, str) or not name.strip():
                raise ValueError("spaCy model names must be non-empty")


@lru_cache(maxsize=8)
def load_model(name: str):
    import spacy

    return spacy.load(name)


class TextParser:
    def __init__(self, config: DeterministicConfig):
        self.config = config
        self.degraded_languages: set[str] = set()
        self.models = {}
        for language, name in (("en", config.spacy_model_en), ("zh", config.spacy_model_zh)):
            try:
                self.models[language] = load_model(name)
            except (ImportError, OSError) as exc:
                if config.on_model_missing == "error":
                    raise RuntimeError(f"deterministic extraction requires installed spaCy model {name}") from exc
                self.degraded_languages.add(language)

    @timed("nlp")
    def parse(self, text: str):
        language = "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"
        model = self.models.get(language)
        # Execution/parse failures are not equivalent to a normally empty result.
        return model(text) if model is not None else None

    @timed("nlp")
    def parse_many(self, texts):
        """Parse one bounded extraction batch, retaining language and input order."""
        groups = {"en": [], "zh": []}
        for text in dict.fromkeys(texts):
            language = "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"
            groups[language].append(text)
        documents = {}
        for language, group in groups.items():
            model = self.models.get(language)
            if not group:
                continue
            docs = list(model.pipe(group, batch_size=32)) if model is not None else [None] * len(group)
            if len(docs) != len(group):
                raise ValueError("NLP batch returned an unexpected document count")
            documents.update(zip(group, docs))
        return [documents[text] for text in texts]

    @property
    def signature(self):
        return {language: {"name": self.config.spacy_model_zh if language == "zh" else self.config.spacy_model_en,
                           "version": model.meta.get("version", "unknown")}
                for language, model in self.models.items()}
