"""Tests for :class:`MiniMaxLLM`."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

import providers.llm.minimax_llm as minimax_llm_module


@pytest.fixture(autouse=True)
def _isolate_openai_ctor():
    """Patch the OpenAI ctor (in openai_llm, where MiniMaxLLM imports
    it from) so init doesn't try to hit the network."""
    from providers.llm import openai_llm as openai_llm_module
    with patch.object(openai_llm_module, "OpenAI") as ctor:
        ctor.return_value = None
        yield ctor


class TestMiniMaxLLM:
    def test_lazy_loader_returns_class(self):
        from providers.llm import get_minimax_llm
        cls = get_minimax_llm()
        assert cls is minimax_llm_module.MiniMaxLLM

    def test_default_base_url(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.delenv("OGMEM_MINIMAX_BASE_URL", raising=False)
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")
        llm = minimax_llm_module.MiniMaxLLM()
        assert llm._base_url == "https://api.minimaxi.com/v1"
        # OpenAI client constructed with the expected base_url.
        _isolate_openai_ctor.assert_called_once()
        kwargs = _isolate_openai_ctor.call_args.kwargs
        assert kwargs["base_url"] == "https://api.minimaxi.com/v1"
        assert kwargs["api_key"] == "test-key"

    def test_env_base_url_override(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")
        monkeypatch.setenv("OGMEM_MINIMAX_BASE_URL", "https://proxy.example/v1")
        llm = minimax_llm_module.MiniMaxLLM()
        assert llm._base_url == "https://proxy.example/v1"

    def test_explicit_args_override_env(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "env-key")
        llm = minimax_llm_module.MiniMaxLLM(
            api_key="arg-key", base_url="https://explicit.example/v1",
        )
        assert llm._base_url == "https://explicit.example/v1"
        kwargs = _isolate_openai_ctor.call_args.kwargs
        assert kwargs["api_key"] == "arg-key"

    def test_minimax_api_key_env_var(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.delenv("OGMEM_MINIMAX_API_KEY", raising=False)
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-only-key")
        minimax_llm_module.MiniMaxLLM()
        kwargs = _isolate_openai_ctor.call_args.kwargs
        assert kwargs["api_key"] == "minimax-only-key"

    def test_missing_api_key_raises(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.delenv("OGMEM_MINIMAX_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MiniMax API key is required"):
            minimax_llm_module.MiniMaxLLM()

    def test_default_model(self, _isolate_openai_ctor, monkeypatch):
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")
        monkeypatch.delenv("OGMEM_MINIMAX_MODEL", raising=False)
        llm = minimax_llm_module.MiniMaxLLM()
        assert llm._model == "MiniMax-M3"

    def test_inherits_complete_multimodal(self, _isolate_openai_ctor, monkeypatch):
        """MiniMaxLLM must satisfy the LLM Protocol including
        complete_multimodal — the multimodal path is the whole point."""
        from core.interfaces import LLM
        monkeypatch.setenv("OGMEM_MINIMAX_API_KEY", "test-key")
        llm = minimax_llm_module.MiniMaxLLM()
        assert isinstance(llm, LLM)
        assert hasattr(llm, "complete_multimodal")
        assert callable(llm.complete_multimodal)


def test_normalise_base_url_strips_chat_completions_endpoint():
    from providers.llm.minimax_llm import _normalise_base_url
    assert _normalise_base_url(
        "https://api.minimaxi.com/v1/chat/completions"
    ) == "https://api.minimaxi.com/v1"


@pytest.mark.parametrize("values,expected", [
    (["scissors"], {"query": "scissors"}),
    (["scissors", 8], {"query": "scissors", "top_k": 8}),
    ([8, "scissors"], {"query": "scissors", "top_k": 8}),
])
def test_positional_search_mapping_is_unique_and_does_not_add_defaults(values, expected):
    schema = {"type": "object", "properties": {
        "query": {"type": "string"}, "top_k": {"type": "integer", "default": 5}},
        "required": ["query"]}
    original = deepcopy((values, schema))
    assert minimax_llm_module._named_scalar_arguments(values, schema) == expected
    assert (values, schema) == original


def test_ambiguous_string_parameters_are_not_guessed():
    from core.errors import ToolCallProtocolError
    schema = {"type": "object", "properties": {
        "left": {"type": "string"}, "right": {"type": "string"}},
        "required": ["left", "right"]}
    with pytest.raises(ToolCallProtocolError, match="Ambiguous"):
        minimax_llm_module._named_scalar_arguments(["a", "b"], schema)


@pytest.mark.parametrize("values", [[], [3], [True], ["q", "not an integer"], ["q", 3, 4]])
def test_invalid_positional_arguments_are_rejected(values):
    from core.errors import ToolCallProtocolError
    schema = {"type": "object", "properties": {
        "query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]}
    with pytest.raises(ToolCallProtocolError):
        minimax_llm_module._named_scalar_arguments(values, schema)


def test_nested_or_constrained_schemas_require_named_arguments():
    from core.errors import ToolCallProtocolError
    for spec in [{"type": "object"}, {"type": "string", "pattern": "^safe$"}]:
        with pytest.raises(ToolCallProtocolError):
            minimax_llm_module._named_scalar_arguments(
                ["value"], {"type": "object", "properties": {"x": spec}, "required": ["x"]})


@pytest.mark.parametrize("arguments", ['["scissors",8]', '{"query": "scissors", "top_k": 8}'])
def test_provider_normalizes_wire_shape_without_changing_prompt_or_values(arguments):
    from providers.llm.openai_llm import OpenAILLM
    llm = minimax_llm_module.MiniMaxLLM(api_key="test-key")
    messages = [{"role": "system", "content": "Keep all speech facts."},
                {"role": "user", "content": "Find scissors."}]
    original = deepcopy(messages)
    tools = [{"type": "function", "function": {
        "name": "Search", "parameters": {"type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"]}}}]
    call = SimpleNamespace(id="real-call-id", function=SimpleNamespace(name="Search", arguments=arguments))
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls", message=SimpleNamespace(tool_calls=[call]))])
    with patch.object(OpenAILLM, "_create_completion", return_value=response) as create:
        assert llm._create_completion(messages=messages, tools=tools) is response
    llm._normalize_tool_response(response, tools)
    assert messages == original
    assert create.call_args.kwargs["messages"] is messages
    assert call.id == "real-call-id"
    assert json.loads(call.function.arguments) == {"query": "scissors", "top_k": 8}
    if arguments.startswith("{"):
        assert call.function.arguments == arguments


def test_non_tool_calls_are_not_changed():
    from providers.llm.openai_llm import OpenAILLM
    llm = minimax_llm_module.MiniMaxLLM(api_key="test-key")
    messages = [{"role": "system", "content": "Extract memory."}]
    with patch.object(OpenAILLM, "_create_completion") as create:
        response = llm._create_completion(messages=messages, tools=None, response_format={"type": "json_object"})
    llm._normalize_tool_response(response, None)
    assert create.call_args.kwargs["messages"] is messages


def test_truncated_response_is_left_for_shared_truncation_guard():
    from providers.llm.openai_llm import OpenAILLM
    llm = minimax_llm_module.MiniMaxLLM(api_key="test-key")
    call = SimpleNamespace(function=SimpleNamespace(name="unknown", arguments='["partial"]'))
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="length", message=SimpleNamespace(tool_calls=[call]))])
    with patch.object(OpenAILLM, "_create_completion", return_value=response):
        llm._create_completion(messages=[], tools=[{"type": "function", "function": {"name": "Answer"}}])
    llm._normalize_tool_response(response, [{"type": "function", "function": {"name": "Answer"}}])
    assert call.function.arguments == '["partial"]'


@pytest.mark.parametrize("mode", ["tools", "messages"])
@pytest.mark.parametrize("failure", ["ambiguous", "undeclared", "truncated"])
def test_failed_normalization_and_retry_each_record_usage_once(mode, failure):
    from core.errors import LLMOutputTruncatedError, ToolCallProtocolError

    llm = minimax_llm_module.MiniMaxLLM(api_key="test-key")
    llm._client = Mock()
    llm.token_tracker = Mock()
    llm.token_tracker.record_llm.side_effect = ["failed-round", "retry-round"]
    tools = [{"name": "Search", "description": "Search", "input_schema": {
        "type": "object", "properties": {
            "left": {"type": "string"}, "right": {"type": "string"}},
        "required": ["left", "right"],
    }}]

    def response(name, arguments, inp, out, finish_reason="tool_calls"):
        return SimpleNamespace(
            model="MiniMax-M3",
            usage=SimpleNamespace(prompt_tokens=inp, completion_tokens=out),
            choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(
                content=None, tool_calls=[SimpleNamespace(
                    id="search", function=SimpleNamespace(name=name, arguments=arguments),
                )],
            ))],
        )

    failed = response("Unknown" if failure == "undeclared" else "Search",
                      '["a", "b"]', 100, 20,
                      "length" if failure == "truncated" else "tool_calls")
    retried = response("Search", '{"left":"a","right":"b"}', 150, 30)
    llm._client.chat.completions.create.side_effect = [failed, retried]

    def complete():
        if mode == "tools":
            return llm.complete_with_tools("Search", tools)
        return llm.complete_with_tools_messages(
            [{"role": "user", "content": "Search"}], tools,
        )[0]

    error = LLMOutputTruncatedError if failure == "truncated" else ToolCallProtocolError
    with pytest.raises(error):
        complete()
    llm.token_tracker.record_llm.assert_called_once_with(
        100, 20, 0, 0, usage_known=True, model="MiniMax-M3")
    result = complete()
    assert result[0]["input"] == {"left": "a", "right": "b"}
    assert result[0]["_llm_usage"]["round_id"] == "retry-round"
    assert llm.token_tracker.record_llm.call_args_list == [
        call(100, 20, 0, 0, usage_known=True, model="MiniMax-M3"),
        call(150, 30, 0, 0, usage_known=True, model="MiniMax-M3"),
    ]


@pytest.mark.parametrize("mode", ["tools", "messages"])
@pytest.mark.parametrize("arguments", ['["scissors"]', '{"query":"scissors"}'])
def test_successful_tool_normalization_records_usage_once(mode, arguments):
    llm = minimax_llm_module.MiniMaxLLM(api_key="test-key")
    llm._client = Mock()
    llm.token_tracker = Mock()
    llm.token_tracker.record_llm.return_value = "round"
    llm._client.chat.completions.create.return_value = SimpleNamespace(
        model="MiniMax-M3", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
            content=None, tool_calls=[SimpleNamespace(
                id="search", function=SimpleNamespace(name="Search", arguments=arguments),
            )],
        ))],
    )
    tools = [{"name": "Search", "description": "Search", "input_schema": {
        "type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
    }}]
    if mode == "tools":
        result = llm.complete_with_tools("Search", tools)
    else:
        result, _ = llm.complete_with_tools_messages([{"role": "user", "content": "Search"}], tools)
    assert result[0]["input"] == {"query": "scissors"}
    assert result[0]["_llm_usage"]["round_id"] == "round"
    llm.token_tracker.record_llm.assert_called_once_with(
        100, 20, 0, 0, usage_known=True, model="MiniMax-M3")
