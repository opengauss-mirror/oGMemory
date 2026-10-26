"""OpenAI LLM implementation for production use.

Supports structured JSON output via response_format with automatic fallback
to text mode for models that don't support json_object format (e.g. Doubao Seed).
"""

import html
import json
import logging
import os
import re
from typing import Any

from core.llm_output import final_text
import threading

try:
    import json_repair
except ImportError:
    json_repair = None

logger = logging.getLogger(__name__)


def _sanitize_json_text(raw: str) -> str:
    """Fix common LLM JSON output issues BEFORE parsing."""
    text = html.unescape(raw)

    def _fix_string_contents(match: re.Match) -> str:
        s = match.group(0)
        inner = s[1:-1]
        inner = inner.replace("\n", "\\n")
        inner = inner.replace("\r", "\\r")
        inner = inner.replace("\t", "\\t")
        return f'"{inner}"'

    text = re.sub(r'"(?:[^"\\]|\\.)*"', _fix_string_contents, text)
    return text

def _loads_lenient_json_text(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    text = _sanitize_json_text(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            cleaned.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == ",":
            next_index = index + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if next_index < len(text) and text[next_index] in "}]":
                continue
        cleaned.append(char)
    return json.loads("".join(cleaned))

try:
    from openai import APIStatusError, BadRequestError, OpenAI
except ImportError:
    raise ImportError(
        "OpenAI package is required for OpenAILLM. "
        "Install it with: pip install openai"
    )

from core.errors import ToolCallProtocolError
from core.errors import LLMCallError  # noqa: E402
from core.interfaces import LLM
from providers.token_tracker import TokenTracker, UsagePersistenceError


# Default models
GPT_4O = "gpt-4o"
GPT_4O_MINI = "gpt-4o-mini"
GPT_4_TURBO = "gpt-4-turbo"


class OpenAILLM(LLM):
    """OpenAI LLM with JSON mode fallback and token tracking.

    When json_mode=True, first attempts response_format=json_object.
    If the model rejects it (BadRequestError), automatically falls back to
    text mode with manual JSON extraction for all subsequent calls.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = GPT_4O_MINI,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
        is_reasoning_model: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        request_timeout_seconds: float | None = None,
    ):
        # OGMEM_LLM_TIMEOUT (seconds): the default is too tight when many
        # video pipelines hit the endpoint concurrently (queue wait +
        # decode exceeds 120s -> non-retryable "Request timed out" ->
        # per-clip flash failures). Explicit config still wins.
        if request_timeout_seconds is None:
            request_timeout_seconds = float(
                os.environ.get("OGMEM_LLM_TIMEOUT", "120.0")
            )
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._json_mode = json_mode
        self._reasoning_effort = reasoning_effort
        self._is_reasoning_model_override = is_reasoning_model
        self._extra_body = dict(extra_body) if extra_body else None
        self._force_text_mode = False
        self._base_url = str(base_url or "")
        self._required_tool_choice_capability = "unknown"
        self._required_tool_choice_lock = threading.Lock()

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._client = OpenAI(timeout=request_timeout_seconds, **client_kwargs)
        self.token_tracker = TokenTracker(model=self._model)

    def _create_completion(self, **kwargs):
        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception:
            self.token_tracker.record_failure('llm')
            raise

    def _is_reasoning_model(self) -> bool:
        if self._is_reasoning_model_override is not None:
            return self._is_reasoning_model_override
        model = self._model.lower().rsplit("/", 1)[-1]
        return model.startswith(("gpt-5", "o1", "o3", "o4"))

    def _completion_parameters(self, temperature: float) -> dict[str, Any]:
        """Return parameters compatible with the configured model type."""
        if self._is_reasoning_model():
            parameters: dict[str, Any] = {"max_completion_tokens": self._max_tokens}
        else:
            parameters = {"temperature": temperature, "max_tokens": self._max_tokens}
        if self._is_reasoning_model() and self._reasoning_effort:
            parameters["reasoning_effort"] = self._reasoning_effort
        if self._extra_body:
            parameters["extra_body"] = self._extra_body
        return parameters

    def _raise_if_tool_truncated(self, response) -> None:
        """Do not repair partial tool arguments or retry an exhausted budget.

        A protocol retry, second-temperature run, and JSON fallback all use
        the same token budget. They cannot be assumed to fix truncation and
        previously hid the useful error behind many minutes of repeated calls.
        """
        if getattr(response.choices[0], "finish_reason", None) != "length":
            return
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        message = (
            f"LLM tool output truncated: model={self._model} finish_reason=length "
            f"max_tokens={self._max_tokens} "
            f"completion_tokens={getattr(usage, 'completion_tokens', None)} "
            f"reasoning_tokens={getattr(details, 'reasoning_tokens', None)}. "
            "Increase llm.max_tokens and, for slow reasoning models, "
            "llm.request_timeout_seconds; validate a full input before retrying."
        )
        logger.error(message)
        raise LLMCallError(message)

    def _track_usage(self, response) -> dict:
        usage = getattr(response, "usage", None)
        inp = getattr(usage, "prompt_tokens", 0) or 0
        out = getattr(usage, "completion_tokens", 0) or 0
        cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
        cache_write = getattr(usage, 'cache_creation_input_tokens', 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cache_read = getattr(details, "cached_tokens", None) or cache_read
        round_id = self.token_tracker.record_llm(
            inp, out, cache_read, cache_write,
            usage_known=(getattr(usage, 'prompt_tokens', None) is not None
                         and getattr(usage, 'completion_tokens', None) is not None),
            model=getattr(response, 'model', None) or self._model,
        )
        if usage is None:
            return {}
        return {
            "round_id": round_id,
            "model": getattr(response, "model", self._model) or self._model,
            "input_tokens": inp,
            "output_tokens": out,
            "cache_read": cache_read,
            "cache_write": cache_write,
        }

    def complete_json(self, prompt: str, schema: dict) -> dict:
        system_message = self._build_system_message(schema)
        json_type = schema.get("type", "object")
        use_json_format = (
            self._json_mode
            and json_type == "object"
            and not self._force_text_mode
        )

        if use_json_format:
            try:
                return self._complete_json_structured(system_message, prompt, schema)
            except BadRequestError as e:
                logger.warning(
                    "Model %s rejected json_object format: %s — switching to text mode",
                    self._model, e,
                )
                self._force_text_mode = True

        return self._complete_json_text_fallback(system_message, prompt, schema)

    def complete_multimodal(self, messages: list[dict], schema: dict) -> dict:
        """Complete a multimodal message list and return JSON matching schema.

        Mirrors ``complete_json``'s JSON-mode + text-fallback strategy
        but forwards the caller's full message list (each message may
        carry a multimodal ``content`` array with text + image_url +
        input_audio parts) to the underlying chat completions API.

        Content part shapes follow OpenAI's Chat Completions API::

            {"role": "user", "content": [
                {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                {"type": "input_audio", "input_audio": {"data": "<b64>", "format": "wav"}},
            ]}

        Any ``system`` message in ``messages`` is dropped — the system
        role is rebuilt from ``schema`` so JSON-mode guarantees hold.
        """
        system_message = self._build_system_message(schema)
        json_type = schema.get("type", "object")
        use_json_format = (
            self._json_mode
            and json_type == "object"
            and not self._force_text_mode
        )

        prepared = [{"role": "system", "content": system_message}]
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                # Schema-driven system_message is authoritative.
                continue
            prepared.append({"role": role, "content": m.get("content")})

        if use_json_format:
            try:
                return self._complete_multimodal_structured(prepared, schema)
            except BadRequestError as e:
                logger.warning(
                    "Model %s rejected json_object format on multimodal: %s — "
                    "switching to text mode",
                    self._model, e,
                )
                self._force_text_mode = True
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Model %s returned non-JSON / schema-invalid content "
                    "despite json_object on multimodal: %s — retrying in "
                    "text fallback mode",
                    self._model, e,
                )

        return self._complete_multimodal_text_fallback(prepared, schema)

    def _complete_multimodal_structured(
        self, prepared_messages: list[dict], schema: dict,
    ) -> dict:
        response = self._create_completion(
            model=self._model,
            messages=prepared_messages,
            response_format={"type": "json_object"},
            **self._completion_parameters(self._temperature),
        )
        self._track_usage(response)
        content = response.choices[0].message.content
        result = self._extract_json_from_text(content or "")
        return self._validate_result(result, schema)

    def _complete_multimodal_text_fallback(
        self, prepared_messages: list[dict], schema: dict,
    ) -> dict:
        """Multimodal text fallback — same shape as the prompt path.

        The system message is augmented with a "JSON-only" hint and we
        rely on free-text + JSON extraction rather than
        ``response_format``. Some multimodal models (early versions,
        vision-only) reject json_object even though they accept images.
        """
        # The first message is always the schema-driven system one;
        # augment in place rather than rebuilding.
        prepared_messages = list(prepared_messages)
        prepared_messages[0] = {
            **prepared_messages[0],
            "content": (
                prepared_messages[0]["content"]
                + "\n\nCRITICAL: Output ONLY the JSON object. "
                "No markdown fences, no explanation, no text before or after the JSON."
            ),
        }
        try:
            response = self._create_completion(
                model=self._model,
                messages=prepared_messages,
                **self._completion_parameters(self._temperature),
            )
            self._track_usage(response)
            content = response.choices[0].message.content or ""
            result = self._extract_json_from_text(content)
            return self._validate_result(result, schema)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse multimodal LLM response as JSON: {e}")
        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise RuntimeError(f"OpenAI API error on multimodal: {e}") from e


    def _complete_json_structured(self, system_message: str, prompt: str, schema: dict) -> dict:
        response = self._create_completion(
            model=self._model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            **self._completion_parameters(self._temperature),
        )
        self._track_usage(response)
        raw_content = response.choices[0].message.content or ""
        content = raw_content.strip()
        # Some endpoints (notably MiniMax-M3) emit a ``<think>…</think>``
        # reasoning prefix BEFORE the JSON object even though we asked
        # for ``response_format={"type": "json_object"}``. Direct
        # ``json.loads`` on the raw content then fails with the
        # "Expecting value" error since the JSON parser trips on the
        # ``<think>`` tag. Fall back to the same JSON extraction the
        # text-mode path uses so the reasoning prefix is stripped and
        # the trailing balanced ``{…}`` block is recovered.
        try:
            result = _loads_lenient_json_text(content)
            return self._validate_result(result, schema)
        except json.JSONDecodeError as e:
            logger.warning(
                "Model %s returned non-JSON content under json_object "
                "format (likely a reasoning prefix like <think>…</think>); "
                "extracting JSON from response text — %s",
                self._model, e,
            )
            result = self._extract_json_from_text(content)
            return self._validate_result(result, schema)

    def _complete_json_text_fallback(self, system_message: str, prompt: str, schema: dict) -> dict:
        enhanced_system = (
            system_message
            + "\n\nCRITICAL: Output ONLY the JSON object. "
            "No markdown fences, no explanation, no text before or after the JSON."
        )
        try:
            response = self._create_completion(
                model=self._model,
                messages=[
                    {"role": "system", "content": enhanced_system},
                    {"role": "user", "content": prompt},
                ],
                **self._completion_parameters(self._temperature),
            )
            self._track_usage(response)
            content = response.choices[0].message.content or ""
            values = self._extract_json_values_from_text(content)
            if not values:
                # Preserve response-shape evidence without logging conversation
                # text or reasoning. Empty/truncated output is a model failure,
                # not evidence that there are no memories to extract.
                import hashlib

                usage = getattr(response, "usage", None)
                details = getattr(usage, "completion_tokens_details", None)
                logger.warning(
                    "JSON fallback produced no usable JSON: model=%s "
                    "finish_reason=%s max_tokens=%d completion_tokens=%s "
                    "reasoning_tokens=%s content_len=%d content_sha=%s "
                    "visible_len=%d response_id=%s",
                    self._model,
                    getattr(response.choices[0], "finish_reason", None),
                    self._max_tokens,
                    getattr(usage, "completion_tokens", None),
                    getattr(details, "reasoning_tokens", None),
                    len(content),
                    hashlib.sha256(content.encode("utf-8")).hexdigest()[:12],
                    len(final_text(content)),
                    getattr(response, "id", ""),
                )
                raise json.JSONDecodeError("No valid JSON found in LLM output", content, 0)

            first_error = None
            for result in values:
                try:
                    return self._validate_result(result, schema)
                except (ValueError, TypeError) as exc:
                    if first_error is None:
                        first_error = exc
            assert first_error is not None
            raise first_error
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    @staticmethod
    def _extract_json_from_text(text: str) -> Any:
        values = OpenAILLM._extract_json_values_from_text(text)
        if values:
            return values[0]
        raise json.JSONDecodeError("No valid JSON found in LLM output", text, 0)

    @staticmethod
    def _extract_json_values_from_text(text: str) -> list[Any]:
        """Extract JSON value from LLM free-text output.

        Multi-layer parsing chain:
        1. Direct json.loads on sanitized text
        2. Extract from ```json``` fenced code blocks
        3. Decode the JSON object with a string-aware decoder
        4. Repair truncated JSON (close unclosed braces/brackets)
        5. json_repair.loads as last resort
        """
        # Valid JSON is data, including literal reasoning tags and HTML entities.
        try:
            return [json.loads(text)]
        except json.JSONDecodeError:
            pass
        # Hide JSON string tokens while removing *outer* reasoning. Applying
        # final_text to the complete response silently modifies quoted data.
        strings = []
        marker = "\x00JSON_STRING_"
        while marker in text:
            marker += "_"

        def protect(match):
            strings.append(match.group(0))
            return f"{marker}{len(strings) - 1}\x00"

        masked = re.sub(r'"(?:[^"\\]|\\.)*"', protect, text)
        stripped = re.sub(
            re.escape(marker) + r"(\d+)\x00",
            lambda match: strings[int(match.group(1))],
            final_text(masked),
        )
        if not stripped:
            return []
        try:
            return [_loads_lenient_json_text(stripped)]
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", stripped, re.DOTALL)
        if fence_match:
            try:
                return [_loads_lenient_json_text(fence_match.group(1).strip())]
            except json.JSONDecodeError:
                pass

        values = []
        start = OpenAILLM._first_json_value_start(stripped)
        while start >= 0:
            candidate = OpenAILLM._balanced_json_value(stripped, start)
            if candidate is not None:
                try:
                    values.append(_loads_lenient_json_text(candidate))
                except json.JSONDecodeError:
                    pass
                start = OpenAILLM._first_json_value_start(stripped, start + len(candidate))
            else:
                # A prose prefix may contain an unmatched bracket. Keep
                # searching rather than handing the whole prefix to repair,
                # which can hide a valid extraction object inside a bogus list.
                start = OpenAILLM._first_json_value_start(stripped, start + 1)
        if values:
            return values

        brace_start = OpenAILLM._first_json_value_start(stripped)
        if brace_start >= 0:
            # raw_decode respects braces inside strings and escaped quotes.
            try:
                result, _ = json.JSONDecoder().raw_decode(
                    _sanitize_json_text(stripped[brace_start:])
                )
                return [result]
            except json.JSONDecodeError:
                pass
            partial = stripped[brace_start:]
            try:
                return [_loads_lenient_json_text(OpenAILLM._repair_truncated_json(partial))]
            except json.JSONDecodeError:
                pass

        if json_repair is not None:
            try:
                result = json_repair.loads(stripped)
                if isinstance(result, (dict, list)):
                    logger.info("json_repair recovered JSON from malformed LLM output")
                    return [result]
            except Exception:
                pass

        return []

    @staticmethod
    def _first_json_value_start(text: str, start: int = 0) -> int:
        starts = [pos for pos in (text.find("{", start), text.find("[", start)) if pos >= 0]
        return min(starts) if starts else -1

    @staticmethod
    def _balanced_json_value(text: str, start: int) -> str | None:
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            ch = text[index]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return None

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """Attempt to fix truncated JSON by closing open brackets."""
        text = text.rstrip()
        if text.endswith(","):
            text = text[:-1]

        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        in_string = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string

        if in_string:
            text += '"'

        text += "]" * max(0, open_brackets)
        text += "}" * max(0, open_braces)
        return text

    @staticmethod
    def _parse_tool_args(args: str) -> dict:
        """Parse tool call arguments with multi-layer fallback."""
        if isinstance(args, dict):
            return args
        try:
            return json.loads(args)
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            return json.loads(_sanitize_json_text(args))
        except (json.JSONDecodeError, TypeError):
            pass
        if json_repair is not None:
            try:
                result = json_repair.loads(args)
                if isinstance(result, dict):
                    logger.info("json_repair recovered tool call arguments")
                    return result
            except Exception:
                pass
        logger.warning("Failed to parse tool call arguments, wrapping as raw: %.200s", args)
        return {"raw": args}

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float | None = None,
    ) -> list[dict]:
        import hashlib

        _temp = temperature if temperature is not None else self._temperature

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                }
            }
            for t in tools
        ]
        prompt_sha = hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()[:12]
        tool_names = [t.get("name", "") for t in tools]

        if (
            tool_choice == "required"
            and self._required_tool_choice_capability == "unknown"
        ):
            # Only the first unknown-capability request probes ``required``.
            # Concurrent dual-run extraction waits for that result so providers
            # that reject ``required`` are not probed twice.
            with self._required_tool_choice_lock:
                if self._required_tool_choice_capability == "unknown":
                    return self._complete_with_tools_impl(
                        prompt=prompt,
                        openai_tools=openai_tools,
                        tool_names=tool_names,
                        tool_choice=tool_choice,
                        temperature=_temp,
                        prompt_sha=prompt_sha,
                    )

        return self._complete_with_tools_impl(
            prompt=prompt,
            openai_tools=openai_tools,
            tool_names=tool_names,
            tool_choice=tool_choice,
            temperature=_temp,
            prompt_sha=prompt_sha,
        )

    def _complete_with_tools_impl(
        self,
        *,
        prompt: str,
        openai_tools: list[dict],
        tool_names: list[str],
        tool_choice: str,
        temperature: float,
        prompt_sha: str,
    ) -> list[dict]:
        import hashlib
        import time

        max_required_empty_attempts = 2
        required_empty_attempts = 0
        effective_tool_choice = (
            "auto"
            if tool_choice == "required"
            and self._required_tool_choice_capability == "unsupported"
            else tool_choice
        )

        while True:
            try:
                response = self._create_completion(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    tools=openai_tools,
                    tool_choice=effective_tool_choice,
                    **self._completion_parameters(temperature),
                )
                self._raise_if_tool_truncated(response)

                message = response.choices[0].message
                raw_tool_calls = message.tool_calls
                tool_calls = raw_tool_calls or []

                if not tool_calls:
                    text_content = message.content or ""
                    text_content_sha = (
                        hashlib.sha256(text_content.encode("utf-8", errors="replace")).hexdigest()[:12]
                        if text_content else ""
                    )
                    raw_choice = response.choices[0]
                    logger.warning(
                        "complete_with_tools returned 0 tool_calls: model=%s base_url=%s "
                        "attempt=%d/%d tool_choice=%s effective_tool_choice=%s "
                        "finish_reason=%s response_id=%s response_model=%s "
                        "raw_tool_calls_type=%s raw_tool_calls_is_none=%s "
                        "text_response_len=%d text_response_sha=%s prompt_len=%d prompt_sha=%s "
                        "tools=%s",
                        self._model,
                        self._base_url,
                        required_empty_attempts + 1,
                        max_required_empty_attempts,
                        tool_choice,
                        effective_tool_choice,
                        raw_choice.finish_reason,
                        getattr(response, "id", ""),
                        getattr(response, "model", ""),
                        type(raw_tool_calls).__name__,
                        raw_tool_calls is None,
                        len(text_content),
                        text_content_sha,
                        len(prompt),
                        prompt_sha,
                        tool_names,
                    )
                    if effective_tool_choice == "required":
                        required_empty_attempts += 1
                        if required_empty_attempts < max_required_empty_attempts:
                            delay = 0.5 * required_empty_attempts
                            logger.warning(
                                "complete_with_tools retrying empty required tool_calls: "
                                "model=%s prompt_sha=%s next_attempt=%d/%d delay=%.1fs",
                                self._model,
                                prompt_sha,
                                required_empty_attempts + 1,
                                max_required_empty_attempts,
                                delay,
                            )
                            time.sleep(delay)
                            continue
                        raise ToolCallProtocolError(
                            "Required tool call returned no tool_calls after "
                            f"{max_required_empty_attempts} attempts; "
                            f"model={self._model}, prompt_sha={prompt_sha}",
                            tool_choice="required",
                        )
                    # Accepted response (auto tool_choice legitimately
                    # returned text): track usage only now, not on the
                    # retried attempts above.
                    self._track_usage(response)
                    return []

                if effective_tool_choice == "required":
                    self._required_tool_choice_capability = "supported"
                # Usage belongs to the accepted response only: tracking on
                # every retry-loop attempt (and on the final raise) inflated
                # the token counters.
                usage = self._track_usage(response)
                parsed_tool_calls = [
                    {
                        "tool": call.function.name,
                        "input": self._parse_tool_args(call.function.arguments),
                        # Usage belongs to the whole LLM response. Multiple
                        # tool calls from the same response intentionally
                        # share this dict; downstream code must count it once
                        # per round, not once per tool call.
                        "_llm_usage": usage,
                    }
                    for call in tool_calls
                ]
                return parsed_tool_calls

            except (LLMCallError, ToolCallProtocolError, UsagePersistenceError):
                raise
            except BadRequestError as e:
                if (
                    tool_choice == "required"
                    and effective_tool_choice == "required"
                    and self._is_required_tool_choice_unsupported(e)
                ):
                    self._required_tool_choice_capability = "unsupported"
                    effective_tool_choice = "auto"
                    logger.warning(
                        "Model %s does not support tool_choice=required; "
                        "using auto for this and subsequent calls",
                        self._model,
                    )
                    continue
                raise LLMCallError(
                    f"OpenAI BadRequestError (400): {e}", status_code=400, cause=e,
                ) from e
            except APIStatusError as e:
                raise LLMCallError(
                    f"OpenAI API error ({getattr(e, 'status_code', '?')}): {e}",
                    status_code=getattr(e, "status_code", None),
                    cause=e,
                ) from e
            except Exception as e:
                raise LLMCallError(f"OpenAI API error: {e}", cause=e) from e

    @staticmethod
    def _is_required_tool_choice_unsupported(exc: BadRequestError) -> bool:
        """Conservatively recognize explicit rejection of ``required``.

        Empty tool calls and unrelated HTTP 400 responses must not poison the
        capability cache.
        """
        body = getattr(exc, "body", None)
        try:
            body_text = json.dumps(body, ensure_ascii=False) if body else ""
        except (TypeError, ValueError):
            body_text = str(body or "")
        text = f"{exc} {body_text}".lower().replace("-", "_")
        mentions_choice = "tool_choice" in text or "tool choice" in text
        mentions_required = "required" in text
        explicit_rejection = any(
            marker in text
            for marker in (
                "not support",
                "unsupported",
                "doesn't support",
                "does not allow",
                "not allowed",
                "only support",
            )
        )
        enumerates_allowed_values = (
            any(
                marker in text
                for marker in ("must be one of", "allowed values", "valid values")
            )
            and "auto" in text
        )
        return mentions_choice and mentions_required and (
            explicit_rejection or enumerates_allowed_values
        )

    @staticmethod
    def _is_required_tool_choice_unsupported(exc: BadRequestError) -> bool:
        """Conservatively recognize explicit rejection of ``required``.

        Empty tool calls and unrelated HTTP 400 responses must not poison the
        capability cache.
        """
        body = getattr(exc, "body", None)
        try:
            body_text = json.dumps(body, ensure_ascii=False) if body else ""
        except (TypeError, ValueError):
            body_text = str(body or "")
        text = f"{exc} {body_text}".lower().replace("-", "_")
        mentions_choice = "tool_choice" in text or "tool choice" in text
        mentions_required = "required" in text
        explicit_rejection = any(
            marker in text
            for marker in (
                "not support",
                "unsupported",
                "doesn't support",
                "does not allow",
                "not allowed",
                "only support",
            )
        )
        enumerates_allowed_values = (
            any(
                marker in text
                for marker in ("must be one of", "allowed values", "valid values")
            )
            and "auto" in text
        )
        return mentions_choice and mentions_required and (
            explicit_rejection or enumerates_allowed_values
        )

    def complete_with_tools_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[list[dict], str]:
        prepared_messages = []
        for msg in messages:
            msg_copy = dict(msg)
            if msg.get("role") == "system":
                msg_copy["cache_control"] = {"type": "ephemeral"}
            prepared_messages.append(msg_copy)

        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    }
                }
                for t in tools
            ]

        api_tool_choice = tool_choice
        if tools is None:
            # Don't pass tool_choice when tools is None (OpenAI API requirement)
            api_tool_choice = None

        try:
            response = self._create_completion(
                model=self._model,
                messages=prepared_messages,
                tools=openai_tools,
                tool_choice=api_tool_choice,
                **self._completion_parameters(self._temperature),
            )
            usage = self._track_usage(response)
            self._raise_if_tool_truncated(response)

            message = response.choices[0].message

            raw_tool_calls = message.tool_calls or []
            tool_calls = [
                {
                    "tool": call.function.name,
                    "input": self._parse_tool_args(call.function.arguments),
                    # Usage belongs to the whole LLM response. Multiple tool
                    # calls from the same response intentionally share it.
                    "_llm_usage": usage,
                }
                for call in raw_tool_calls
            ]

            content = message.content or ""

            return tool_calls, content

        except BadRequestError as e:
            raise LLMCallError(
                f"OpenAI BadRequestError (400): {e}",
                status_code=400,
                cause=e,
            ) from e
        except APIStatusError as e:
            raise LLMCallError(
                f"OpenAI API error ({getattr(e, 'status_code', '?')}): {e}",
                status_code=getattr(e, "status_code", None),
                cause=e,
            ) from e
        except Exception as e:
            raise LLMCallError(f"OpenAI API error: {e}", cause=e) from e

    def _build_system_message(self, schema: dict) -> str:
        """Build system message with schema instructions.

        Args:
            schema: JSON schema for response.

        Returns:
            System message content.
        """
        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)

        return f"""You are a helpful assistant that extracts structured information from text.

Respond with valid JSON that matches this schema:
{schema_str}

Important:
- Respond ONLY with valid JSON
- Do not include any text outside the JSON structure
- Use null for missing optional fields
- Follow the exact property names and types from the schema"""

    def _validate_result(self, result: dict, schema: dict) -> dict:
        """Validate result against schema (basic validation).

        Args:
            result: The parsed JSON result.
            schema: The expected schema.

        Returns:
            The validated result.
        """
        # Basic type validation
        expected_type = schema.get("type")

        if expected_type == "object":
            if not isinstance(result, dict):
                raise ValueError(f"Expected object, got {type(result).__name__}")
        elif expected_type == "array":
            if not isinstance(result, list):
                raise ValueError(f"Expected array, got {type(result).__name__}")

        # Check required properties
        required = schema.get("required", [])

        if isinstance(result, dict):
            for prop in required:
                if prop not in result:
                    raise ValueError(f"Missing required property: {prop}")

        return result

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._model


class CachedOpenAILLM(OpenAILLM):
    """OpenAI LLM with simple in-memory caching.

    Caches responses based on prompt hash to reduce API calls.
    Useful for repeated extraction tasks with identical inputs.

    Note: Cache is in-memory and not persisted across restarts.
    """

    def __init__(self, *args, cache_max_size: int = 1000, **kwargs):
        """Initialize cached LLM.

        Args:
            cache_max_size: Maximum number of cached responses.
        """
        super().__init__(*args, **kwargs)
        self._cache: dict[str, dict] = {}
        self._cache_max_size = cache_max_size
        self._cache_hits = 0
        self._cache_misses = 0

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Complete with caching.

        Args:
            prompt: Input prompt.
            schema: JSON schema.

        Returns:
            JSON response.
        """
        # Create cache key from prompt and schema
        import hashlib
        cache_key = hashlib.sha256(
            f"{prompt}:{json.dumps(schema, sort_keys=True)}".encode()
        ).hexdigest()

        # Check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.token_tracker.record_cache_hit('llm')
            self._cache_hits += 1
            return cached

        # Cache miss - call API
        self._cache_misses += 1
        result = super().complete_json(prompt, schema)

        # Store in cache
        self._cache[cache_key] = result

        # Evict oldest if cache is too large
        if len(self._cache) > self._cache_max_size:
            # Remove first (oldest) entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        return result

    def complete_with_tools_messages(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> tuple[list[dict], str]:
        """Cached version of messages-based tool use."""
        # For now, delegate to parent (caching messages is complex)
        return super().complete_with_tools_messages(messages, tools, tool_choice)

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": round(hit_rate * 100, 2),
        }
