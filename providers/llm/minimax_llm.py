"""MiniMax-M3 LLM provider.

A thin wrapper around :class:`OpenAILLM` that points at MiniMax's
managed API endpoint (``https://api.minimaxi.com``) and defaults to
the ``MiniMax-M3`` model. Inherits all the multimodal support from
``OpenAILLM`` (text + image_url + input_audio content parts, JSON-mode
+ text-fallback), so callers that already pass OpenAI-style multimodal
messages get them through unchanged — only the routing changes.

This is the backend oG-Memory uses for ``episodic_memory_generator``
and other multimodal operators that need vision + audio joint input.
The user picked it over Qwen2.5-Omni-7B because it's reachable via
the existing ``api.minimaxi.com`` proxy and only requires a coding
plan; Cross-Modal Reasoning accuracy is expected to drift vs the M3-Agent
baseline and is treated as a follow-up rather than a blocker.

Configuration is via env or kwargs::

    export OGMEM_MINIMAX_API_KEY=...
    python -m m3.episodic_memory --llm-provider minimax

    # or
    from providers.llm import MiniMaxLLM
    llm = MiniMaxLLM(api_key="...", model="MiniMax-M3")
"""

from __future__ import annotations

import json
import logging
import os
from itertools import permutations

from core.errors import ToolCallProtocolError

from .openai_llm import OpenAILLM

DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_MODEL = "MiniMax-M3"
logger = logging.getLogger(__name__)


def _named_scalar_arguments(values: list, schema: dict) -> dict:
    """Losslessly name positional values only when the schema gives one answer.

    Do not rely on JSON property order or invent optional defaults. Deliberately
    limited to small, simple scalar schemas (including M3 Search/Answer);
    ambiguous, constrained, or nested schemas require native named arguments.
    """
    allowed = {"type", "properties", "required", "additionalProperties", "description", "title"}
    properties = schema.get("properties", {})
    if (schema.get("type") != "object" or set(schema) - allowed
            or not isinstance(properties, dict) or len(properties) > 6
            or len(values) > len(properties)):
        raise ToolCallProtocolError("Unsupported MiniMax positional argument schema")
    required = set(schema.get("required", []))

    def matches(value, spec):
        if not isinstance(spec, dict) or set(spec) - {"type", "description", "title", "default"}:
            return False
        kind = spec.get("type")
        return (
            (kind == "string" and isinstance(value, str))
            or (kind == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (kind == "boolean" and isinstance(value, bool))
            or (kind == "null" and value is None)
        )

    candidates = {}
    for names in permutations(properties, len(values)):
        if not required.issubset(names):
            continue
        if all(matches(value, properties[name]) for name, value in zip(names, values)):
            candidate = dict(zip(names, values))
            candidates[json.dumps(candidate, sort_keys=True)] = candidate
            if len(candidates) > 1:
                raise ToolCallProtocolError("Ambiguous MiniMax positional tool arguments")
    if len(candidates) != 1:
        raise ToolCallProtocolError("MiniMax positional arguments do not match required scalar parameters")
    return next(iter(candidates.values()))


def _normalise_base_url(base_url: str | None) -> str:
    """Return an OpenAI-client base URL, not a full endpoint URL.

    Users often copy the full MiniMax docs endpoint
    ``https://api.minimaxi.com/v1/chat/completions`` into env vars.
    The OpenAI client expects ``base_url`` to be only the API root and
    appends ``/chat/completions`` itself; if we pass the full endpoint,
    requests go to ``.../chat/completions/chat/completions`` and 404.
    """
    url = (base_url or DEFAULT_BASE_URL).rstrip("/")
    suffix = "/chat/completions"
    if url.endswith(suffix):
        url = url[: -len(suffix)]
    return url


class MiniMaxLLM(OpenAILLM):
    """OpenAI-compatible MiniMax-M3 client.

    Inherits ``complete_json``, ``complete_multimodal``, ``complete_with_tools``,
    and ``complete_with_tools_messages`` from :class:`OpenAILLM`. The
    defaults for ``base_url`` and ``model`` route requests to MiniMax's
    managed API. A provider-specific compatibility step also converts
    positional scalar tool arguments only when their schema mapping is unique.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = True,
        request_timeout_seconds: float = 120.0,
    ) -> None:
        # Read MiniMax-specific env first, then fall back to the
        # generic OpenAI env so a single OGMEM_API_KEY can power
        # either backend in dev.
        api_key = (
            api_key
            or os.environ.get("OGMEM_MINIMAX_API_KEY")
            or os.environ.get("MINIMAX_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "MiniMax API key is required. Set OGMEM_MINIMAX_API_KEY "
                "(or MINIMAX_API_KEY / OPENAI_API_KEY) or pass api_key=."
            )

        super().__init__(
            api_key=api_key,
            base_url=_normalise_base_url(
                base_url or os.environ.get("OGMEM_MINIMAX_BASE_URL"),
            ),
            model=os.environ.get("OGMEM_MINIMAX_MODEL", model),
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            request_timeout_seconds=request_timeout_seconds,
        )

    def _normalize_tool_response(self, response, tools) -> None:
        """Canonicalize wire arguments after the caller has recorded usage."""
        if not tools:
            return
        schemas = {t["function"]["name"]: t["function"].get("parameters", {})
                   for t in tools if t.get("type") == "function"}
        for choice in getattr(response, "choices", None) or []:
            if choice.finish_reason == "length":
                continue  # the shared truncation guard rejects this response
            for call in choice.message.tool_calls or []:
                try:
                    arguments = json.loads(call.function.arguments)
                except (ValueError, TypeError):
                    continue
                if not isinstance(arguments, list):
                    continue
                schema = schemas.get(call.function.name)
                if schema is None:
                    raise ToolCallProtocolError("MiniMax positional call names an undeclared function")
                named = _named_scalar_arguments(arguments, schema)
                call.function.arguments = json.dumps(named, ensure_ascii=False)
                logger.info("Normalized unambiguous MiniMax positional arguments for %s", call.function.name)


__all__ = ["MiniMaxLLM", "DEFAULT_BASE_URL", "DEFAULT_MODEL"]
