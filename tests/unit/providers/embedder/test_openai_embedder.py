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

    @pytest.mark.parametrize("count,batch_size", [(66, 8), (2049, 2048), (16, 8)])
    def test_embed_texts_splits_requests(self, fake_client, count, batch_size):
        fake_client.embeddings.create.side_effect = lambda **kw: _embedding_response(
            [[float(text)] for text in kw["input"]]
        )
        embedder = openai_embedder_module.OpenAIEmbedder(
            api_key="test-key", dimension=1, batch_size=batch_size,
        )
        assert embedder.embed_texts([str(i) for i in range(count)]) == [
            [float(i)] for i in range(count)
        ]
        assert [len(c.kwargs["input"]) for c in fake_client.embeddings.create.call_args_list] == [
            min(batch_size, count - start) for start in range(0, count, batch_size)
        ]

    @pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "8", 2049])
    def test_invalid_batch_size(self, batch_size):
        with pytest.raises(ValueError, match="batch_size"):
            openai_embedder_module.OpenAIEmbedder(api_key="test", batch_size=batch_size)

    def test_batch_response_count_is_validated(self, fake_client):
        fake_client.embeddings.create.return_value = _embedding_response([[0.1]])
        embedder = openai_embedder_module.OpenAIEmbedder(api_key="test", dimension=1)
        with pytest.raises(ValueError, match="Embedding count mismatch"):
            embedder.embed_texts(["a", "b"])

    def test_failed_batch_stops_without_returning_partial_results(self, fake_client):
        fake_client.embeddings.create.side_effect = [
            _embedding_response([[0.1], [0.2]]),
            RuntimeError("endpoint failed"),
        ]
        embedder = openai_embedder_module.OpenAIEmbedder(
            api_key="test", dimension=1, batch_size=2,
        )
        embedder.token_tracker = Mock()
        with pytest.raises(RuntimeError, match="endpoint failed"):
            embedder.embed_texts(["a", "b", "c", "d", "e"])
        assert fake_client.embeddings.create.call_count == 2
        embedder.token_tracker.record_embed.assert_called_once()
        embedder.token_tracker.record_failure.assert_called_once()

    def test_batches_share_deadline(self, fake_client, monkeypatch):
        clock = SimpleNamespace(now=0.0)
        monkeypatch.setattr(openai_embedder_module.time, "monotonic", lambda: clock.now)
        fake_client.with_options.return_value = fake_client

        def respond(**kwargs):
            clock.now += 3
            return _embedding_response([[0.1]])

        fake_client.embeddings.create.side_effect = respond
        embedder = openai_embedder_module.OpenAIEmbedder(
            api_key="test", dimension=1, batch_size=1,
        )
        with pytest.raises(TimeoutError, match="deadline"):
            embedder.embed_texts(["a", "b", "c"], timeout_seconds=5)
        assert fake_client.embeddings.create.call_count == 2
        assert [c.kwargs["timeout"] for c in fake_client.with_options.call_args_list] == [5, 2]

    def test_batch_usage_scopes_compose_with_cache(self, fake_client):
        from providers.token_tracker import distributed_usage_scope, _embedding_inputs

        fake_client.embeddings.create.side_effect = lambda **kw: _embedding_response(
            [[float(text)] for text in kw["input"]]
        )
        embedder = openai_embedder_module.CachedOpenAIEmbedder(
            api_key="test", dimension=1, batch_size=2,
        )
        embedder.embed_texts(["1"])
        scopes = []
        embedder.token_tracker.record_embed = Mock(
            side_effect=lambda *a, **kw: scopes.append(_embedding_inputs.get())
        )
        inputs = [(("run", str(i)), 1) for i in range(5)]
        with distributed_usage_scope({}, inputs=inputs):
            assert embedder.embed_texts(["0", "1", "2", "3", "4"]) == [
                [float(i)] for i in range(5)
            ]
        assert scopes == [[inputs[0], inputs[2]], [inputs[3], inputs[4]]]

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
