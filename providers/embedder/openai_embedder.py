"""OpenAI Embedder implementation.

Production-ready embedder using OpenAI-compatible API.
Supports custom base_url for compatible services.

Under a storm of concurrent requests (parallel M3 drivers) the embedding
endpoint saturates and every query embed times out, emptying the working
set for hours. Two defenses are built in:

* a module-level semaphore (OGMEM_EMBED_MAX_CONCURRENCY, default 8)
  bounds in-flight embed requests;
* jittered exponential backoff retries (OGMEM_EMBED_RETRIES, default 2;
  OGMEM_EMBED_BASE_DELAY, default 1.0s) on **transient** failures only —
  timeouts / connection errors / rate limits / 5xx. Auth and other 4xx
  fail fast. The SDK's own retries are disabled so our budget controls
  the total attempt count.
"""

import logging
import math
import random
import threading
import time
import os
from contextlib import contextmanager
from typing import Final

import httpx

from core.errors import EmbeddingError
from core.interfaces import Embedder
from providers.token_tracker import (
    TokenTracker,
    UsagePersistenceError,
    embedding_input_scope,
    embedding_usage_tokens,
)

logger = logging.getLogger(__name__)


def _embed_max_concurrency() -> int:
    try:
        return max(1, int(os.environ.get("OGMEM_EMBED_MAX_CONCURRENCY") or 8))
    except ValueError:
        logger.warning(
            "Invalid OGMEM_EMBED_MAX_CONCURRENCY=%r — using 8",
            os.environ.get("OGMEM_EMBED_MAX_CONCURRENCY"),
        )
        return 8


# Bounded in-flight embed requests shared by ALL embedder instances in
# this process (module-level: the endpoint, not the instance, saturates).
# Frozen at import time (P3-1): the semaphore is constructed once when
# this module is first imported, so changing OGMEM_EMBED_MAX_CONCURRENCY
# later (e.g. in a long-lived process or via monkeypatch in tests) has
# no effect. Set it in the environment before process start.
_EMBED_SEMAPHORE = threading.BoundedSemaphore(_embed_max_concurrency())


def _embed_retries() -> int:
    try:
        return max(0, int(os.environ.get("OGMEM_EMBED_RETRIES") or 2))
    except ValueError:
        return 2


def _embed_base_delay() -> float:
    try:
        return max(0.0, float(os.environ.get("OGMEM_EMBED_BASE_DELAY") or 1.0))
    except ValueError:
        return 1.0


def _is_transient(exc: BaseException) -> bool:
    """True for transient embed failures worth retrying.

    Timeouts, connection errors, rate limits, and 5xx server errors.
    Auth failures and other 4xx fail fast (retrying cannot help).
    """
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )

        if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None)
            return isinstance(status, int) and status >= 500
    except ImportError:  # pragma: no cover - openai is a hard dep above
        pass
    # The multimodal path uses raw httpx.
    try:
        import httpx

        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = getattr(exc.response, "status_code", None)
            # 429 is rate limiting — transient, same as the SDK path's
            # RateLimitError (parallel load trips it; failing the request
            # here would bypass the retry budget entirely).
            if status == 429:
                return True
            return isinstance(status, int) and status >= 500
    except ImportError:  # pragma: no cover
        pass
    return False


def _remaining_seconds(deadline: float | None) -> float | None:
    """Remaining budget for an absolute monotonic deadline (None = unbounded).

    Raises TimeoutError when the deadline has already elapsed — callers must
    not start (or retry) an API call with no budget left (review item 6).
    """
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise TimeoutError("embedding request deadline elapsed")
    return remaining


@contextmanager
def _embed_slot(deadline: float | None):
    """Acquire a concurrency slot, bounded by the caller's deadline.

    The historical `with _EMBED_SEMAPHORE` waited unboundedly: a caller
    computed its request timeout BEFORE queueing for a slot, so a long
    queue wait silently exceeded the caller's deadline (review item 6).
    """
    if deadline is None:
        with _EMBED_SEMAPHORE:
            yield
        return
    wait = _remaining_seconds(deadline)
    if not _EMBED_SEMAPHORE.acquire(timeout=wait):
        raise TimeoutError(
            "embedding deadline elapsed waiting for a concurrency slot"
        )
    try:
        yield
    finally:
        _EMBED_SEMAPHORE.release()


def _run_with_retry(fn, what: str, deadline: float | None = None):
    """Run an embed call with jittered exponential backoff on transient errors.

    Attempts = 1 + OGMEM_EMBED_RETRIES (default 2). Non-transient errors
    (auth / 4xx) fail fast. The SDK's own retries are disabled, so this
    budget is the total attempt count.

    ``fn(remaining)`` receives the remaining deadline budget for THIS
    attempt (None when unbounded) so per-request timeouts never outlive
    the caller's deadline; backoff sleeps are truncated to the budget and
    a retry is never started once the deadline has elapsed (review item 6).
    """
    attempts = 1 + _embed_retries()
    base_delay = _embed_base_delay()
    for attempt in range(1, attempts + 1):
        remaining = _remaining_seconds(deadline)
        try:
            return fn(remaining)
        except Exception as exc:
            if attempt >= attempts or not _is_transient(exc):
                raise
            delay = base_delay * (2 ** (attempt - 1)) * random.uniform(0.7, 1.3)
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - time.monotonic()))
                if delay <= 0.0:
                    raise TimeoutError("embedding request deadline elapsed")
            logger.warning(
                "Transient embed failure (%s) on attempt %d/%d: %s — "
                "retrying in %.2fs",
                what, attempt, attempts, type(exc).__name__, delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover

# Set this env var to a truthy value to explicitly opt into running
# against a keyless OpenAI-compatible endpoint (e.g. a local
# text-embeddings-inference or vllm server that ignores the
# Authorization header). Without it the embedder only falls back to
# the placeholder key when ``base_url`` is also configured, which is
# the historical behaviour and remains the default for backwards
# compatibility. The explicit opt-in exists so a silent misconfig
# (empty key + public base_url pointing at api.openai.com) does not
# silently degrade to a keyless placeholder.
KEYLESS_ENV_VAR: Final[str] = "OGMEM_EMBED_KEYLESS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

try:
    from openai import APIError, OpenAI
except ImportError:
    raise ImportError(
        "OpenAI package is required for OpenAIEmbedder. "
        "Install it with: pip install openai"
    )

from core.interfaces import Embedder
from providers.token_tracker import TokenTracker

# Default embedding dimensions
ADA_002_DIM: Final[int] = 1536
EMBEDDING_3_SMALL_DIM: Final[int] = 1536
EMBEDDING_3_LARGE_DIM: Final[int] = 3072
DEFAULT_MAX_INPUT_CHARS: Final[int] = 30_000


class OpenAIEmbedder(Embedder):
    """OpenAI embedding model for production use.

    Supports OpenAI API and compatible services (via base_url).

    Models:
        - text-embedding-ada-002: 1536 dimensions
        - text-embedding-3-small: 1536 dimensions, faster/cheaper
        - text-embedding-3-large: 3072 dimensions, best quality

    Example:
        embedder = OpenAIEmbedder(
            api_key="sk-xxx",
            base_url="https://api.openai.com/v1",
            model="text-embedding-ada-002"
        )
        vectors = embedder.embed_texts(["hello", "world"])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "text-embedding-ada-002",
        dimension: int | None = None,
        multimodal: bool = False,
        max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
        batch_size: int = 2048,
    ):
        """Initialize OpenAI embedder.

        Args:
            api_key: OpenAI API key.
            base_url: Custom base URL for compatible APIs.
            model: Embedding model name.
            dimension: Expected dimension (for validation). If None, inferred from model.
            multimodal: Use the ``/embeddings/multimodal`` endpoint.
            max_input_chars: Maximum characters sent for each embedding input.
            batch_size: Maximum texts per request to the compatible endpoint.
        """
        if not api_key:
            if base_url:
                # Custom OpenAI-compatible endpoint (e.g. a locally deployed
                # embedding service) typically does not require a real key.
                # Use a placeholder so the underlying client is happy; the
                # server is expected to ignore or accept any value.
                #
                # Operators that prefer an explicit gate can also set the
                # ``OGMEM_EMBED_KEYLESS`` env var; otherwise this fallback
                # is purely a function of ``base_url`` being non-empty.
                if os.environ.get(KEYLESS_ENV_VAR, "").lower() in _TRUTHY:
                    logger.info(
                        "OGMEM_EMBED_KEYLESS enabled: using placeholder "
                        "API key for base_url=%s",
                        base_url,
                    )
                api_key = "no-key-required"
            else:
                raise ValueError(
                    "OpenAI API key is required. "
                    "Pass api_key parameter or set it via config / env."
                )
        if max_input_chars < 1:
            raise ValueError("max_input_chars must be positive")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 2048:
            raise ValueError("batch_size must be an integer between 1 and 2048")

        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._dimension = dimension or self._get_model_dimension(model)
        self._multimodal = multimodal
        self._max_input_chars = max_input_chars
        self._batch_size = batch_size

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        # Increase connection pool limits to avoid PoolTimeout when
        # background drain thread and main request thread compete for
        # the same embedding API connection pool.
        http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
            ),
            timeout=httpx.Timeout(120.0, pool=60.0),
        )
        # The SDK's own retries are disabled so _run_with_retry's budget
        # controls the total attempt count (no doubled backoff windows).
        self._client = OpenAI(
            http_client=http_client, max_retries=0, **client_kwargs
        )
        self.token_tracker = TokenTracker(model=self._model)

    @staticmethod
    def _bounded_timeout_seconds(timeout_seconds: float) -> float:
        """Validate one caller-owned absolute-deadline remainder."""

        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding timeout must be a positive finite number") from exc
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise TimeoutError("embedding request deadline elapsed")
        return timeout

    def _embedding_client(self, *, timeout_seconds: float | None):
        """Return the normal client or one bounded to one no-retry request."""

        if timeout_seconds is None:
            return self._client
        timeout = self._bounded_timeout_seconds(timeout_seconds)
        # OpenAI defaults to two retries. A candidate request owns a single
        # absolute deadline, so retries must not spend a second full timeout.
        return self._client.with_options(timeout=timeout, max_retries=0)

    def embed_texts(
        self,
        texts: list[str],
        *,
        timeout_seconds: float | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts using OpenAI API.

        Args:
            texts: Input texts, split into requests of at most batch_size texts.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            EmbeddingError: If the embedding provider request fails.
            ValueError: If the returned vectors have an unexpected shape.
        """
        if not texts:
            return []

        deadline = (None if timeout_seconds is None else
                    time.monotonic() + self._bounded_timeout_seconds(timeout_seconds))

        # Truncate texts exceeding token limit to avoid 429 errors
        truncated = []
        for t in texts:
            if len(t) > self._max_input_chars:
                logger.warning(
                    "Truncating text for embedding: %d -> %d chars",
                    len(t),
                    self._max_input_chars,
                )
                truncated.append(t[: self._max_input_chars])
            else:
                truncated.append(t)

        try:
            if self._multimodal:
                embeddings = self._embed_multimodal(
                    truncated,
                    timeout_seconds=None if deadline is None else deadline - time.monotonic(),
                )
            else:
                embeddings = []
                for start in range(0, len(truncated), self._batch_size):
                    batch = truncated[start:start + self._batch_size]
                    with embedding_input_scope(range(start, start + len(batch))):
                        try:
                            # Slot acquisition and every attempt are bounded
                            # by the caller's absolute deadline; the request
                            # timeout is recomputed per attempt (review item 6).
                            with _embed_slot(deadline):
                                client = self._embedding_client(
                                    timeout_seconds=_remaining_seconds(deadline))
                                response = _run_with_retry(
                                    lambda remaining: client.embeddings.create(
                                        input=batch, model=self._model,
                                        **({} if remaining is None
                                            else {"timeout": remaining}),
                                    ),
                                    "embeddings.create", deadline=deadline,
                                )
                        except Exception:
                            self.token_tracker.record_failure('embedding', deadline=deadline)
                            raise
                        usage = getattr(response, "usage", None)
                        total_tokens = embedding_usage_tokens(usage)
                        self.token_tracker.record_embed(total_tokens or 0,
                            usage_known=total_tokens is not None, deadline=deadline)
                        vectors = [item.embedding for item in response.data]
                        if len(vectors) != len(batch):
                            raise ValueError(
                                f"Embedding count mismatch: expected {len(batch)}, got {len(vectors)}"
                            )
                        embeddings.extend(vectors)
        except UsagePersistenceError:
            raise
        except Exception as e:
            # The candidate-local path uses this uncached provider directly.
            # Provider exception messages can contain upstream URL, request, or
            # implementation details, so retain only the safe exception type.
            logger.error("embed_texts FAILED: %s", type(e).__name__)
            if isinstance(e, (APIError, httpx.HTTPError)):
                raise EmbeddingError("", f"Embedding provider failed: {type(e).__name__}") from e
            raise
        for emb in embeddings:
            if len(emb) != self._dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self._dimension}, "
                    f"got {len(emb)}. Check model configuration."
                )

        return embeddings

    def embed_texts_with_timeout(
        self,
        texts: list[str],
        *,
        timeout_seconds: float,
    ) -> list[list[float]]:
        """Embed through a provider call bounded by the caller's remaining deadline."""

        return self.embed_texts(texts, timeout_seconds=timeout_seconds)

    def _embed_multimodal(
        self,
        texts: list[str],
        *,
        timeout_seconds: float | None = None,
    ) -> list[list[float]]:
        """Call the /embeddings/multimodal endpoint for vision embedding models.

        The multimodal endpoint treats `input` as parts of ONE embedding
        (e.g. text + image), so each text must be a separate request.
        """
        base = (self._base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base}/embeddings/multimodal"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        results: list[list[float]] = []
        call_deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + self._bounded_timeout_seconds(timeout_seconds)
        )
        with httpx.Client(timeout=120.0) as client:
            for index, text in enumerate(texts):
                payload = {
                    "model": self._model,
                    "input": [{"type": "text", "text": text}],
                    "dimensions": self._dimension,
                }
                try:
                    def _do_post(remaining):
                        resp = client.post(
                            url, json=payload, headers=headers,
                            # Per-attempt budget: never outlive the caller's
                            # deadline, even after a queue wait or a retry
                            # backoff (review item 6).
                            timeout=120.0 if remaining is None else remaining,
                        )
                        # Include status validation inside the retry so a
                        # 5xx response body is retried, not surfaced.
                        resp.raise_for_status()
                        return resp

                    with _embed_slot(call_deadline):
                        resp = _run_with_retry(
                            _do_post, "embeddings.multimodal",
                            deadline=call_deadline,
                        )
                except Exception:
                    with embedding_input_scope([index]):
                        self.token_tracker.record_failure('embedding', deadline=call_deadline)
                    raise
                data = resp.json()
                total_tokens = embedding_usage_tokens(data.get('usage'))
                with embedding_input_scope([index]):
                    self.token_tracker.record_embed(total_tokens or 0,
                        usage_known=total_tokens is not None, deadline=call_deadline)
                embedding = data.get("data", {}).get("embedding", [])
                if not embedding:
                    raise ValueError(f"No embedding from multimodal endpoint: {data}")
                results.append(embedding)
        return results

    def _get_model_dimension(self, model: str) -> int:
        """Get expected dimension for a model."""
        dimensions = {
            "text-embedding-ada-002": ADA_002_DIM,
            "text-embedding-3-small": EMBEDDING_3_SMALL_DIM,
            "text-embedding-3-large": EMBEDDING_3_LARGE_DIM,
        }
        return dimensions.get(model, ADA_002_DIM)  # Default to ada-002

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def max_input_chars(self) -> int:
        """Get the per-input outbound character limit."""
        return self._max_input_chars


class CachedOpenAIEmbedder(OpenAIEmbedder):
    """OpenAI embedder with simple in-memory caching.

    Caches embeddings for repeated texts to reduce API calls.
    Useful for repeated queries or indexing duplicate content.

    Note: Cache is in-memory and not persisted across restarts.
    Consider using Redis for distributed caching in production.
    """

    def __init__(self, *args, cache_max_size: int = 1000, **kwargs):
        """Initialize cached embedder.

        Args:
            cache_max_size: Maximum number of cached embeddings.
        """
        # Extract cache_max_size from kwargs before passing to parent
        self._cache_max_size = cache_max_size
        super().__init__(*args, **kwargs)
        self._cache: dict[str, list[float]] = {}

    def embed_texts(
        self,
        texts: list[str],
        *,
        timeout_seconds: float | None = None,
    ) -> list[list[float]]:
        """Embed texts with caching.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        results = []
        uncached_texts = []
        uncached_indices = []
        deadline = (None if timeout_seconds is None else
                    time.monotonic() + self._bounded_timeout_seconds(timeout_seconds))

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        self.token_tracker.record_embedding_cache_hits([i for i, _ in results], deadline=deadline)

        # Fetch uncached texts from API
        if uncached_texts:
            with embedding_input_scope(uncached_indices):
                new_embeddings = super().embed_texts(
                    uncached_texts,
                    timeout_seconds=None if deadline is None else deadline - time.monotonic(),
                )
            for text, embedding in zip(uncached_texts, new_embeddings):
                self._cache[text] = embedding

            # Evict oldest entries if cache exceeds max size
            while len(self._cache) > self._cache_max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            # Add to results in correct order
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results.append((idx, embedding))

        # Sort by original index and return
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self._cache)
