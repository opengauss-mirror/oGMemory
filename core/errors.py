"""Domain error types for ContextEngine.

Phase 0 frozen — specific exceptions enable precise error handling.
All errors inherit from standard Exception for compatibility.
"""


class ContextEngineError(Exception):
    """Base exception for all ContextEngine errors."""
    pass


class AccessDeniedError(ContextEngineError):
    """Access control violation.

    Raised when RequestContext.account_id does not match URI account prefix.
    Also raised for unauthorized cross-agent or user-space access.
    """
    def __init__(self, uri: str, account_id: str, reason: str = ""):
        self.uri = uri
        self.account_id = account_id
        self.reason = reason
        message = f"Access denied to {uri} for account {account_id}"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class NodeNotFoundError(ContextEngineError):
    """Requested node does not exist or is not ACTIVE.

    Distinguished from HTTP 404 — node may exist but be PENDING/BROKEN.
    """
    def __init__(self, uri: str):
        self.uri = uri
        super().__init__(f"Node not found or not ACTIVE: {uri}")


class NodeBrokenError(ContextEngineError):
    """Node exists but is in BROKEN state.

    Indicates Repair Job detected corruption requiring manual intervention.
    """
    def __init__(self, uri: str):
        self.uri = uri
        super().__init__(f"Node is BROKEN: {uri}")


class OutboxError(ContextEngineError):
    """Outbox processing failure.

    Raised by OutboxWorker when event processing exceeds retry limit.
    Event is moved to DLQ (dead letter queue).
    """
    def __init__(self, event_id: str, reason: str):
        self.event_id = event_id
        self.reason = reason
        super().__init__(f"Outbox event {event_id} failed: {reason}")


class MergePolicyError(ContextEngineError):
    """Merge policy evaluation failure.

    Raised when MergePolicy cannot determine a valid WritePlan.
    """
    def __init__(self, candidate_uri: str, reason: str):
        self.candidate_uri = candidate_uri
        self.reason = reason
        super().__init__(f"Merge policy failed for {candidate_uri}: {reason}")


class EmbeddingError(ContextEngineError):
    """Embedding generation failure.

    Raised when Embedder fails to produce vectors.
    """
    def __init__(self, text_preview: str, reason: str):
        self.text_preview = text_preview[:100]
        self.reason = reason
        super().__init__(f"Embedding failed for '{text_preview[:50]}...': {reason}")


class VectorIndexError(ContextEngineError):
    """Vector index operation failure.

    Raised when VectorIndex upsert/delete/search fails.
    """
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Vector index {operation} failed: {reason}")


class ValidationError(ContextEngineError):
    """Schema or validation failure.

    Raised when input does not match expected schema.
    """
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Validation failed for {field}: {reason}")


class ExtractionIncompleteError(ContextEngineError, RuntimeError):
    """Extraction stopped with unresolved or incomplete draft operations."""


class LLMTransientError(ExtractionIncompleteError):
    """The model provider can be retried after a transport or service failure."""


class MemoryCardFormatError(ExtractionIncompleteError):
    """A generated preference or entity needs background format recovery."""


class OperationDeadlineExceededError(ExtractionIncompleteError, TimeoutError):
    """The caller's operation budget expired before further work could start."""


class LLMOutputTruncatedError(ExtractionIncompleteError):
    """The model reached its output token limit before completing the response."""


class ToolCallProtocolError(ContextEngineError, RuntimeError):
    """The model response did not satisfy the requested tool-call protocol.

    This is intentionally distinct from transport/auth/rate-limit failures so
    extraction can use a structured JSON fallback without hiding real API
    errors.
    """

    def __init__(self, reason: str, *, tool_choice: str = ""):
        self.reason = reason
        self.tool_choice = tool_choice
        message = f"Tool-call protocol failed: {reason}"
        if tool_choice:
            message += f" (tool_choice={tool_choice})"
        super().__init__(message)


class ConcurrentModificationError(ContextEngineError):
    """Optimistic lock failure during concurrent merge.

    Raised when a merge write detects that the node was modified
    between read and write (version mismatch).

    Indicates the operation should be retried with fresh data.
    """
    def __init__(self, uri: str, expected_version: int, actual_version: int):
        self.uri = uri
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Concurrent modification detected for {uri}: "
            f"expected version {expected_version}, found {actual_version}"
        )


# ---------------------------------------------------------------------------
# Retrieval pipeline errors
# ---------------------------------------------------------------------------

class RetrievalError(ContextEngineError):
    """Base exception for retrieval pipeline failures."""
    def __init__(self, stage: str, reason: str, *, retryable: bool = False):
        self.stage = stage
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"Retrieval [{stage}] failed: {reason}")


class RetrievalPlannerError(RetrievalError):
    """Query planner could not classify the query."""
    def __init__(self, reason: str):
        super().__init__("planner", reason)


class RetrievalSearchError(RetrievalError):
    """Vector search or hierarchical expansion failed."""
    def __init__(self, reason: str, *, retryable: bool = True):
        super().__init__("search", reason, retryable=retryable)


# ---------------------------------------------------------------------------
# LLM transport errors
# ---------------------------------------------------------------------------

class LLMCallError(ContextEngineError):
    """LLM API call failed at the transport level (400, 500, network, etc.).

    Distinct from "LLM returned an empty/unparsable response" — the loop
    can keep going on the latter (the LLM just didn't say anything useful),
    but the former is a hard stop: the LLM server rejected the request
    outright, so continuing to burn the iteration budget is exactly the
    silent-garbage behaviour this exception exists to surface.

    Drivers like :func:`m3.qa.run_react_qa` should catch this and set
    ``error="..."`` on the per-question JSONL record instead of letting
    the loop fall back to "no tools, continue, emit garbage".

    Attributes:
        status_code: HTTP status code if the failure came from the
            upstream API (e.g. 400 for ``openai.BadRequestError``,
            500 for ``APIStatusError``). ``None`` for transport-level
            failures where the request never reached the wire.
        cause: The underlying exception, preserved for stack traces.
    """
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.cause = cause
