"""Deterministic ingestion at the existing accepted-candidate checkpoint seam."""

from contextlib import nullcontext
from dataclasses import asdict

from commit.deterministic import deterministic_batch, validate_payload
from core.evidence import DETERMINISTIC_MEMORY_TYPES, digest
from core.operation_deadline import check_deadline
from core.validation import validate_candidate
from extraction.deterministic.extractor import RULE_VERSION, DeterministicExtractor
from extraction.deterministic.nlp import DeterministicConfig, reuse_documents
from extraction.deterministic.verification import record_verifier
from service.evidence_store import EvidenceStore
from service.extraction_checkpoint import RecoveryNeedsConfirmationError


def extraction_configuration(api):
    if api._extraction_engine == "llm":
        return {"engine": "llm", "mode": api._extraction_mode}
    extractor = get_extractor(api, asdict(api._deterministic_config))
    return {"engine": "deterministic", "rule_version": RULE_VERSION,
            "options": asdict(api._deterministic_config),
            "model_signature": extractor.parser.signature,
            "enabled_types": [kind for kind in DETERMINISTIC_MEMORY_TYPES
                              if api._schema_registry.get_compatible(kind) is not None
                              and api._schema_registry.get_compatible(kind).enabled]}


def get_extractor(api, options):
    key = digest(options)
    if key not in api._deterministic_extractors:
        api._deterministic_extractors[key] = DeterministicExtractor(
            api._schema_registry, DeterministicConfig(**options))
    return api._deterministic_extractors[key]


def commit_deterministic(api, messages, ctx, config, *, checkpoint=None, before_write=None,
                         confidence_threshold=.5, archive_id=None, publication_scope=None):
    store = EvidenceStore(api._fs)
    batch_id = (checkpoint.journal.batch["archive_id"] + ":" + checkpoint.slot if checkpoint is not None
                else archive_id or digest(messages, config))
    check_deadline()
    publish = publication_scope or nullcontext
    with publish():
        store.register(messages, ctx, batch_id=batch_id,
                       archive_id=checkpoint.journal.batch["archive_id"] if checkpoint is not None else archive_id)
    accepted = checkpoint.load() if checkpoint is not None else None
    diagnostics = {"replayed": accepted is not None}
    if accepted is None:
        if config.get("rule_version") != RULE_VERSION:
            raise RecoveryNeedsConfirmationError("frozen deterministic rules are unavailable")
        extractor = get_extractor(api, config["options"])
        if extractor.parser.signature != config["model_signature"]:
            raise RecoveryNeedsConfirmationError("frozen NLP model signature is unavailable")
        candidates = [validate_candidate(candidate) for candidate in extractor.extract(messages, ctx)]
        candidates = [candidate for candidate in candidates if candidate.category in config["enabled_types"]]
        extracted = len(candidates)
        candidates = [candidate for candidate in candidates if candidate.confidence >= confidence_threshold]
        filtered = extracted - len(candidates)
        diagnostics.update(extractor.diagnostics)
        resolve = store.reference_reader(ctx)
        verify = record_verifier(store, ctx, resolve)
        with reuse_documents():
            for candidate in candidates:
                validate_payload(candidate, store, ctx, resolve=resolve, verify=verify)
        if before_write is not None:
            before_write()
        if checkpoint is not None:
            with publish():
                candidates = checkpoint.accept(candidates, extracted, filtered)
    else:
        candidates, extracted, filtered = accepted
        if before_write is not None:
            before_write()
    # Newly validated or digest-checked durable candidates need source checks,
    # but must not re-run NLP while recovering an accepted operation.
    with deterministic_batch(candidates):
        if checkpoint is not None:
            plans = api._write_checkpointed_candidates(candidates, ctx, checkpoint,
                                                       publication_scope=publication_scope)
        else:
            with publish():
                plans = [api._writer.write_candidate(candidate, ctx) for candidate in candidates]
    return extracted, filtered, len(candidates), plans, diagnostics
