import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from providers.llm import openai_llm


@pytest.mark.parametrize("provider", [openai_llm.OpenAILLM, openai_llm.CachedOpenAILLM])
def test_message_stream_delivers_text_before_final_response_and_tracks_usage(monkeypatch, provider):
    events = []
    response = SimpleNamespace(
        model="stream-test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="青禾项目已完成", tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=31, completion_tokens=8),
    )

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            events.append("closed")

        def __iter__(self):
            for text in ("青禾", "项目", "已完成"):
                yield SimpleNamespace(type="content.delta", delta=text)

        def get_final_completion(self):
            events.append("final")
            return response

    client = Mock()
    client.beta.chat.completions.stream.return_value = Stream()
    monkeypatch.setattr(openai_llm, "OpenAI", Mock(return_value=client))
    llm = provider(api_key="integration-key")
    llm.token_tracker = Mock()
    calls, text = llm.complete_with_tools_messages(
        [{"role": "user", "content": "报告进展"}], on_text_delta=events.append
    )
    assert (calls, text) == ([], "青禾项目已完成")
    assert events == ["青禾", "项目", "已完成", "final", "closed"]
    kwargs = client.beta.chat.completions.stream.call_args.kwargs
    assert kwargs["stream_options"] == {"include_usage": True}
    llm.token_tracker.record_llm.assert_called_once_with(
        31, 8, 0, 0, usage_known=True, model="stream-test"
    )


def test_streamed_tool_round_retains_arguments_and_usage(monkeypatch):
    tool = {"name": "read_memory", "description": "Read", "input_schema": {"type": "object"}}
    response = SimpleNamespace(
        model="stream-test",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="read_memory", arguments='{"uri":"ctx://example"}'
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5),
    )

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def __iter__(self):
            yield SimpleNamespace(
                type="tool_calls.function.arguments.delta", delta='{"uri":"ctx://example"}'
            )

        def get_final_completion(self):
            return response

    client = Mock()
    client.beta.chat.completions.stream.return_value = Stream()
    monkeypatch.setattr(openai_llm, "OpenAI", Mock(return_value=client))
    llm = openai_llm.OpenAILLM(api_key="integration-key")
    llm.token_tracker = Mock()
    llm.token_tracker.record_llm.return_value = "usage-1"
    calls, text = llm.complete_with_tools_messages(
        [{"role": "user", "content": "查项目"}], tools=[tool], on_text_delta=lambda _: None
    )
    assert calls[0]["id"] == "call-1"
    assert calls[0]["input"] == {"uri": "ctx://example"}
    assert calls[0]["_llm_usage"]["output_tokens"] == 5
    assert text == ""


def test_sdk_stream_emits_body_deltas_and_final_usage():
    from openai import OpenAI

    emitted = []

    class Body(httpx.SyncByteStream):
        def __iter__(self):
            for content in ("资料", "已整理"):
                chunk = {
                    "id": "chat-stream", "object": "chat.completion.chunk", "created": 1,
                    "model": "stream-test", "choices": [{"index": 0,
                    "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
                }
                yield ("data: " + json.dumps(chunk) + "\n\n").encode()
                emitted.append("consumed:" + content)
            yield ("data: " + json.dumps({
                "id": "chat-stream", "object": "chat.completion.chunk", "created": 1,
                "model": "stream-test", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }) + "\n\n").encode()
            yield ("data: " + json.dumps({
                "id": "chat-stream", "object": "chat.completion.chunk", "created": 1,
                "model": "stream-test", "choices": [],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
            }) + "\n\ndata: [DONE]\n\n").encode()

    def respond(request):
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Body())

    llm = openai_llm.OpenAILLM(api_key="sdk-stream-test")
    llm._client.close()
    with OpenAI(api_key="sdk-stream-test", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        llm._client = client
        llm.token_tracker = Mock()
        calls, text = llm.complete_with_tools_messages(
            [{"role": "user", "content": "整理资料"}], on_text_delta=emitted.append
        )

    assert (calls, text) == ([], "资料已整理")
    assert emitted == ["资料", "consumed:资料", "已整理", "consumed:已整理"]
    llm.token_tracker.record_llm.assert_called_once_with(
        9, 3, 0, 0, usage_known=True, model="stream-test"
    )
