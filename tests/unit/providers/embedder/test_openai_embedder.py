"""Unit tests for the OpenAI embedder providers."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import providers.embedder as embedder_package
import providers.embedder.openai_embedder as openai_embedder_module

# Lazy import OpenAIEmbedder
OpenAIEmbedder, CachedOpenAIEmbedder = None, None


def _get_openai_embedder():
    global OpenAIEmbedder, CachedOpenAIEmbedder
    if OpenAIEmbedder is None:
        from providers.embedder import get_openai_embedder
        OpenAIEmbedder, CachedOpenAIEmbedder = get_openai_embedder()
    return OpenAIEmbedder, CachedOpenAIEmbedder


def _embedding_response(vectors: list[list[float]]) -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


@pytest.fixture
def fake_client() -> Mock:
    return Mock()


@pytest.fixture(autouse=True)
def fake_openai_ctor(monkeypatch: pytest.MonkeyPatch, fake_client: Mock) -> Mock:
    ctor = Mock(return_value=fake_client)
    monkeypatch.setattr(openai_embedder_module, "OpenAI", ctor)
    return ctor


class TestOpenAIEmbedder:
    def test_lazy_loader_returns_classes(self):
        openai_embedder, cached_embedder = embedder_package.get_openai_embedder()
        assert openai_embedder is openai_embedder_module.OpenAIEmbedder
        assert cached_embedder is openai_embedder_module.CachedOpenAIEmbedder

    def test_init_uses_explicit_credentials_and_dimension(self, fake_openai_ctor: Mock):
        embedder = openai_embedder_module.OpenAIEmbedder(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="text-embedding-3-large",
        )

        # Verify http_client is passed to OpenAI constructor
        assert fake_openai_ctor.call_count == 1
        call_kwargs = fake_openai_ctor.call_args.kwargs
        assert "http_client" in call_kwargs
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["base_url"] == "https://example.com/v1"
        assert embedder.dimension == openai_embedder_module.EMBEDDING_3_LARGE_DIM
        assert embedder.model == "text-embedding-3-large"
        assert embedder.max_input_chars == openai_embedder_module.DEFAULT_MAX_INPUT_CHARS

    def test_init_falls_back_to_environment_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_openai_ctor: Mock,
    ):
        # OpenAIEmbedder requires explicit api_key, no env fallback
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")

        with pytest.raises(ValueError, match="API key is required"):
            openai_embedder_module.OpenAIEmbedder()

    def test_init_without_api_key_raises_value_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OGMEM_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key"):
            openai_embedder_module.OpenAIEmbedder(api_key=None)

    def test_embed_texts_returns_empty_for_empty_input(self, fake_client: Mock):
        embedder = openai_embedder_module.OpenAIEmbedder(api_key="test-key")

        assert embedder.embed_texts([]) == []
        fake_client.embeddings.create.assert_not_called()

    def test_embed_texts_calls_client_and_returns_vectors(self, fake_client: Mock):
        fake_client.embeddings.create.return_value = _embedding_response([[0.1, 0.2], [0.3, 0.4]])
        embedder = openai_embedder_module.OpenAIEmbedder(
            api_key="test-key",
            model="custom-model",
            dimension=2,
        )

        result = embedder.embed_texts(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        fake_client.embeddings.create.assert_called_once_with(
            input=["hello", "world"],
            model="custom-model",
        )

    def test_deadline_aware_embedder_uses_remaining_timeout_and_disables_retries(
        self,
        fake_client: Mock,
    ):
        deadline_client = Mock()
        deadline_client.embeddings.create.return_value = _embedding_response([[0.1, 0.2]])
        fake_client.with_options.return_value = deadline_client
        embedder = openai_embedder_module.OpenAIEmbedder(
            api_key="test-key",
            model="custom-model",
            dimension=2,
        )

        result = embedder.embed_texts_with_timeout(["hello"], timeout_seconds=4.5)

        assert result == [[0.1, 0.2]]
        fake_client.with_options.assert_called_once()
        options = fake_client.with_options.call_args.kwargs
        assert 0 < options['timeout'] <= 4.5
        assert options['max_retries'] == 0
        deadline_client.embeddings.create.assert_called_once_with(
            input=["hello"],
            model="custom-model",
        )

    def test_deadline_aware_embedder_rejects_expired_budget_before_provider_call(
        self,
        fake_client: Mock,
    ):
        embedder = openai_embedder_module.OpenAIEmbedder(api_key="test-key")

        with pytest.raises(TimeoutError, match="deadline"):
            embedder.embed_texts_with_timeout(["hello"], timeout_seconds=0)

        fake_client.with_options.assert_not_called()

    def test_embed_texts_applies_configured_qwen_input_cap_before_provider_call(
        self,
        fake_client: Mock,
    ):
        fake_client.embeddings.create.return_value = _embedding_response([[0.1], [0.2]])
        # The formal Qwen endpoint is capped at 2,048 tokens. 8,000 chars is a
        # conservative provider-side estimate below 2,048 * 4 chars/token.
        qwen_max_input_chars = 8_000
        long_text = "x" * (qwen_max_input_chars + 500)
        normal_text = "normal input remains unchanged"
        texts = [long_text, normal_text]
        embedder = openai_embedder_module.OpenAIEmbedder(
            api_key="test-key",
            model="Qwen/Qwen3-Embedding-0.6B",
            dimension=1,
            max_input_chars=qwen_max_input_chars,
        )

        result = embedder.embed_texts(texts)

        assert result == [[0.1], [0.2]]
        assert texts == [long_text, normal_text]
        fake_client.embeddings.create.assert_called_once_with(
            input=[long_text[:qwen_max_input_chars], normal_text],
            model="Qwen/Qwen3-Embedding-0.6B",
        )

    def test_embed_texts_rejects_non_positive_input_cap(self):
        with pytest.raises(ValueError, match="max_input_chars must be positive"):
            openai_embedder_module.OpenAIEmbedder(
                api_key="test-key",
                max_input_chars=0,
            )

    def test_embed_texts_rejects_batches_larger_than_limit(self, fake_client: Mock):
        embedder = openai_embedder_module.OpenAIEmbedder(api_key="test-key")

        with pytest.raises(ValueError, match="Batch size too large"):
            embedder.embed_texts(["x"] * 2049)

        fake_client.embeddings.create.assert_not_called()

    def test_embed_texts_validates_embedding_dimension(self, fake_client: Mock):
        fake_client.embeddings.create.return_value = _embedding_response([[0.1]])
        embedder = openai_embedder_module.OpenAIEmbedder(api_key="test-key", dimension=2)

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            embedder.embed_texts(["hello"])

    def test_embed_failure_log_keeps_raw_provider_message_private(
        self,
        fake_client: Mock,
        caplog: pytest.LogCaptureFixture,
    ):
        secret_message = "upstream failed for /private/workspace with token=secret"
        fake_client.embeddings.create.side_effect = RuntimeError(secret_message)
        embedder = openai_embedder_module.OpenAIEmbedder(api_key="test-key")

        with caplog.at_level(logging.ERROR, logger="providers.embedder.openai_embedder"):
            with pytest.raises(RuntimeError, match="upstream failed"):
                embedder.embed_texts(["hello"])

        assert "RuntimeError" in caplog.text
        assert secret_message not in caplog.text
        assert "/private/workspace" not in caplog.text

    def test_unknown_model_dimension_defaults_to_ada_size(self):
        embedder = object.__new__(openai_embedder_module.OpenAIEmbedder)

        assert embedder._get_model_dimension("unknown") == openai_embedder_module.ADA_002_DIM

    def test_multimodal_deadline_budget_is_shared_across_subrequests(self, monkeypatch):
        class Clock:
            now = 0.0

            def monotonic(self):
                return self.now

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": {"embedding": [0.1, 0.2]}}

        class HttpClient:
            request_timeouts = []

            def __init__(self, *, timeout):
                assert timeout == 120.0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, *, json, headers, timeout):
                del url, json, headers
                self.request_timeouts.append(timeout)
                clock.now += 4.0
                return Response()

        import httpx

        clock = Clock()
        embedder = openai_embedder_module.OpenAIEmbedder(
            api_key="test-key",
            multimodal=True,
            dimension=2,
        )
        monkeypatch.setattr(openai_embedder_module.time, "monotonic", clock.monotonic)
        monkeypatch.setattr(httpx, "Client", HttpClient)

        result = embedder.embed_texts_with_timeout(["one", "two"], timeout_seconds=10.0)

        assert result == [[0.1, 0.2], [0.1, 0.2]]
        assert HttpClient.request_timeouts == [10.0, 6.0]


class TestCachedOpenAIEmbedder:
    def test_inherited_deadline_capability_dispatches_through_cached_override(
        self,
        fake_client: Mock,
    ):
        deadline_client = Mock()
        deadline_client.embeddings.create.return_value = _embedding_response([[1.0]])
        fake_client.with_options.return_value = deadline_client
        embedder = openai_embedder_module.CachedOpenAIEmbedder(
            api_key="test-key",
            dimension=1,
        )

        assert embedder.embed_texts_with_timeout(["value"], timeout_seconds=3.0) == [[1.0]]
        fake_client.with_options.assert_called_once()
        options = fake_client.with_options.call_args.kwargs
        assert 0 < options['timeout'] <= 3.0
        assert options['max_retries'] == 0

    def test_cached_embedder_reuses_cached_vectors_and_preserves_order(self, fake_client: Mock):
        fake_client.embeddings.create.side_effect = [
            _embedding_response([[1.0, 1.1], [2.0, 2.1]]),
            _embedding_response([[3.0, 3.1]]),
        ]
        embedder = openai_embedder_module.CachedOpenAIEmbedder(
            api_key="test-key",
            dimension=2,
            cache_max_size=3,
        )

        first = embedder.embed_texts(["alpha", "beta"])
        second = embedder.embed_texts(["beta", "gamma", "alpha"])

        assert first == [[1.0, 1.1], [2.0, 2.1]]
        assert second == [[2.0, 2.1], [3.0, 3.1], [1.0, 1.1]]
        assert fake_client.embeddings.create.call_count == 2
        assert embedder.cache_size == 3

    def test_cache_eviction_removes_oldest_entry(self, fake_client: Mock):
        fake_client.embeddings.create.side_effect = [
            _embedding_response([[1.0]]),
            _embedding_response([[2.0]]),
            _embedding_response([[3.0]]),
            _embedding_response([[4.0]]),
        ]
        embedder = openai_embedder_module.CachedOpenAIEmbedder(
            api_key="test-key",
            dimension=1,
            cache_max_size=2,
        )

        embedder.embed_texts(["first"])
        embedder.embed_texts(["second"])
        embedder.embed_texts(["third"])
        embedder.embed_texts(["first"])

        assert fake_client.embeddings.create.call_count == 4
        assert embedder.cache_size == 2

    def test_clear_cache_removes_all_entries(self, fake_client: Mock):
        fake_client.embeddings.create.return_value = _embedding_response([[1.0]])
        embedder = openai_embedder_module.CachedOpenAIEmbedder(
            api_key="test-key",
            dimension=1,
        )

        embedder.embed_texts(["value"])
        embedder.clear_cache()

        assert embedder.cache_size == 0


class TestEmbedRetryAndConcurrency:
    """WS 3.3: bounded in-flight requests + jittered transient retry."""

    def test_run_with_retry_retries_transient_then_succeeds(self, monkeypatch):
        import httpx

        monkeypatch.setenv("OGMEM_EMBED_RETRIES", "2")
        monkeypatch.setenv("OGMEM_EMBED_BASE_DELAY", "0")

        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise httpx.TimeoutException("transient")
            return "ok"

        assert openai_embedder_module._run_with_retry(flaky, "unit") == "ok"
        assert len(calls) == 2

    def test_run_with_retry_fails_fast_on_non_transient(self, monkeypatch):
        monkeypatch.setenv("OGMEM_EMBED_RETRIES", "3")
        monkeypatch.setenv("OGMEM_EMBED_BASE_DELAY", "0")

        calls = []

        def denied():
            calls.append(1)
            raise PermissionError("auth")

        with pytest.raises(PermissionError):
            openai_embedder_module._run_with_retry(denied, "unit")
        assert len(calls) == 1

    def test_run_with_retry_gives_up_after_budget(self, monkeypatch):
        import httpx

        monkeypatch.setenv("OGMEM_EMBED_RETRIES", "1")
        monkeypatch.setenv("OGMEM_EMBED_BASE_DELAY", "0")

        calls = []

        def always_transient():
            calls.append(1)
            raise httpx.TimeoutException("still down")

        with pytest.raises(httpx.TimeoutException):
            openai_embedder_module._run_with_retry(always_transient, "unit")
        assert len(calls) == 2  # 1 + 1 retry

    def test_is_transient_classifies_openai_errors(self):
        import httpx
        from openai import APIConnectionError, APITimeoutError

        req = httpx.Request("POST", "http://x")
        assert openai_embedder_module._is_transient(APITimeoutError(request=req))
        assert openai_embedder_module._is_transient(
            APIConnectionError(message="c", request=req)
        )

        resp_500 = httpx.Response(500, request=httpx.Request("POST", "http://x"))
        resp_401 = httpx.Response(401, request=httpx.Request("POST", "http://x"))
        assert openai_embedder_module._is_transient(
            httpx.HTTPStatusError("5xx", request=resp_500.request, response=resp_500)
        )
        assert not openai_embedder_module._is_transient(
            httpx.HTTPStatusError("401", request=resp_401.request, response=resp_401)
        )
        assert openai_embedder_module._is_transient(httpx.TimeoutException("t"))
        assert openai_embedder_module._is_transient(httpx.TransportError("t"))
        assert not openai_embedder_module._is_transient(ValueError("nope"))

    def test_embed_texts_semaphore_bounds_in_flight(self, fake_client, monkeypatch):
        import threading

        monkeypatch.setenv("OGMEM_EMBED_MAX_CONCURRENCY", "2")
        # The module-level semaphore was sized at import; replace it for
        # this test with a 2-slot one.
        monkeypatch.setattr(
            openai_embedder_module,
            "_EMBED_SEMAPHORE",
            threading.BoundedSemaphore(2),
        )
        fake_client.embeddings.create.return_value = _embedding_response([[0.1, 0.2]])

        embedder = openai_embedder_module.OpenAIEmbedder(
            api_key="test-key", model="custom-model", dimension=2,
        )

        in_flight = []
        peak = []
        lock = threading.Lock()

        original = fake_client.embeddings.create

        def tracking(*args, **kwargs):
            with lock:
                in_flight.append(1)
                peak.append(len(in_flight))
            import time as _t
            _t.sleep(0.05)
            try:
                return original(*args, **kwargs)
            finally:
                with lock:
                    in_flight.pop()

        fake_client.embeddings.create = tracking

        threads = [
            threading.Thread(target=lambda: embedder.embed_texts(["x"]))
            for _ in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max(peak) <= 2
