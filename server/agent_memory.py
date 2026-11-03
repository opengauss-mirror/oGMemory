"""Connect finalized traces to typed agent memory through the existing outbox."""

from dataclasses import asdict
from hashlib import sha256

from commit.agent_memory import (
    AgentMemoryArchiveBuilder,
    AgentMemoryConflictError,
    AgentMemoryPolicy,
)
from commit.context_writer import ContextWriter
from commit.identity_resolver import MemoryIdentityResolver
from commit.policy_router import PolicyRouter
from core.models import OutboxEvent, RequestContext
from core.uri_resolver import URIResolver
from extraction.agent_memory import (
    AGENT_CATEGORIES,
    AgentMemoryCandidate,
    AgentMemoryExtractor,
    agent_memory_registry,
)


def trace_memory_event(trace_uri: str, ctx: RequestContext) -> OutboxEvent:
    identity = sha256(
        f"agent-memory-v1:{trace_uri}:{ctx.user_id}:{ctx.agent_id}".encode()
    ).hexdigest()[:32]
    return OutboxEvent(
        event_id=identity,
        event_type="EXTRACT_AGENT_MEMORY",
        uri=f"ctx://{ctx.account_id}/agents/{ctx.agent_id}/memories/agent_extraction/{identity}",
        status="PENDING",
        payload={"context": asdict(ctx), "trace_uri": trace_uri},
    )


def process_trace_memory(event, *, trace_store, outbox, fs, llm,
                         directory_summary_enabled=False, vector_index=None, embedder=None):
    """Persist reviewed proposals before writing their versioned memory units."""
    ctx = RequestContext(**event.payload["context"])
    trace_uri = event.payload["trace_uri"]
    if not ctx.agent_id or not trace_uri.startswith(
        f"ctx://{ctx.account_id}/traces/{ctx.session_id}/turns/"
    ):
        raise ValueError("trace memory context does not match its finalized turn")
    if any("items" in candidate.get("structured_fields", {})
           for candidate in event.payload.get("candidates", [])):
        event.payload.pop("candidates")
    if "candidates" not in event.payload or event.payload.get("reextract"):
        events, cursor = [], 0
        while True:
            page = trace_store.read_skill_trace_event_page(
                account_id=ctx.account_id, evidence_uri=trace_uri,
                after_ordinal=cursor, limit=32,
            )
            events.extend(page["events"])
            if page["done"]:
                break
            if page["next_ordinal"] <= cursor:
                raise ValueError("trace paging did not advance")
            cursor = page["next_ordinal"]
        extractor = AgentMemoryExtractor(fs, llm, vector_index=vector_index, embedder=embedder)
        if event.payload.get("reextract"):
            index = event.payload.get("committed_count", 0)
            candidates = extractor.reextract_candidate(trace_uri, events, ctx,
                                                       event.payload["candidates"][index])
            event.payload["candidates"][index:index + 1] = [asdict(c) for c in candidates]
            event.payload.pop("reextract")
            event.payload.setdefault("value_reviews", []).extend(extractor.value_reviews)
        else:
            candidates = extractor.extract_candidates(trace_uri, events, ctx)
            event.payload.update(candidates=[asdict(c) for c in candidates],
                                 value_reviews=extractor.value_reviews,
                                 cursor=page["next_ordinal"], committed_count=0)
        outbox.save_payload(event)

    registry = agent_memory_registry()
    uris = URIResolver(registry)
    router = PolicyRouter(fs, registry, uris)
    for category in AGENT_CATEGORIES:
        router.register_policy(category, AgentMemoryPolicy(fs, uris))
    writer = ContextWriter(
        fs=fs, llm=llm, policy_router=router,
        archive_builder=AgentMemoryArchiveBuilder(llm), outbox_store=outbox,
        directory_summary_enabled=directory_summary_enabled,
        identity_resolver=MemoryIdentityResolver(fs, uris, registry, router),
        require_index_event=True,
    )
    for index in range(event.payload.get("committed_count", 0), len(event.payload["candidates"])):
        proposal = event.payload["candidates"][index]
        if proposal["category"] == "agent_skill":
            continue
        try:
            writer.write_candidate(AgentMemoryCandidate(**proposal), ctx)
        except Exception as exc:
            cause = exc
            while cause is not None:
                if isinstance(cause, AgentMemoryConflictError):
                    event.payload["reextract"] = True
                    outbox.save_payload(event)
                    break
                cause = cause.__cause__
            raise
        event.payload["committed_count"] = index + 1
        outbox.save_payload(event)
