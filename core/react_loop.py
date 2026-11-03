"""Generic ReAct loop abstraction.

Provides a template method for ReAct-style iterative tool use:
LLM think → call tool → observe result → repeat or produce.

Subclasses implement 4 abstract methods for scene-specific behavior:
- _get_tools(): return available tool definitions
- _build_messages(): build initial messages (system prompt + context)
- _execute_tool(): execute a single tool call
- _parse_output(): parse LLM output to structured result
"""
 
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any
from uuid import uuid4

from core.errors import ExtractionIncompleteError
from core.interfaces import LLM
from core.models import RequestContext
from core.operation_deadline import check_deadline
from core.errors import LLMCallError
from core.tools.operational_tool_spec import OperationalToolSpec, call_with_injection

logger = getLogger(__name__)


@dataclass
class ReactTaskContext:
    """Base class for ReAct loop task context.

    Subclass to carry scene-specific parameters.
    """
    pass


@dataclass
class ReactIterationTrace:
    """Trace data for a single ReAct iteration.

    Subclass to add scene-specific trace fields.
    """
    iteration: int
    latency_seconds: float
    tool_calls_count: int = 0
    tool_call_names: list[str] = field(default_factory=list)
    content_length: int = 0


@dataclass
class ReactTrace:
    """Full trace for a ReAct run."""
    total_iterations: int = 0
    total_latency_seconds: float = 0.0
    iterations: list[ReactIterationTrace] = field(default_factory=list)
    total_read_uris: int = 0


@dataclass
class ReactLoopResult:
    """Result of ReactLoop.run()."""
    output: list[Any] = field(default_factory=list)
    tools_used: list[dict] = field(default_factory=list)
    iterations: int = 0
    read_uris: set[str] = field(default_factory=set)
    trace: ReactTrace | None = None
    completed: bool = False
    finish_reason: str = ""  # "done" | "timeout" | "max_iterations" | "no_valid_output" | "llm_error"
    # Populated when ``finish_reason == "llm_error"`` — the original
    # :class:`LLMCallError` raised by ``_call_llm``. Drivers like
    # ``m3.qa.run_react_qa`` can read this to surface the upstream
    # status code / message instead of the generic "loop terminated"
    # text. None for non-LLM-error finish_reasons.
    llm_error: Exception | None = None


class ReactLoop(ABC):
    """Generic ReAct loop base class.

    Template method run() executes the loop:
    1. _build_messages() builds initial messages
    2. while iteration < max_iterations:
       - timeout check → break
       - call LLM with _get_tools() tools
       - three-way branch:
         - tool calls → _execute_tool() → continue
         - content → _parse_output() → _should_continue_after_output()
         - neither → disable tools → continue
    3. return ReactLoopResult
    """

    def __init__(
        self,
        llm: LLM,
        max_iterations: int = 5,
        timeout_seconds: float = 30.0,
        internal_tool_usage_tracker: Any | None = None,
        pipeline_name: str = "",
        operational_tools: list | None = None,
    ):
        self._llm = llm
        self._max_iterations = max_iterations
        self._timeout_seconds = timeout_seconds
        self._internal_tool_usage_tracker = internal_tool_usage_tracker
        self._pipeline_name = pipeline_name

        self._disable_tools_for_iteration: bool = False

        # Runtime state (reset per run())
        self._read_uris: set[str] = set()
        self._operational_tool_specs: list[OperationalToolSpec] = [
            fn._operational_tool_spec for fn in (operational_tools or [])
        ]
        self._operational_dispatch = {
            spec.name: spec for spec in self._operational_tool_specs
        }

    def run(
        self,
        ctx: RequestContext,
        task_ctx: ReactTaskContext,
    ) -> ReactLoopResult:
        """Execute the ReAct loop."""
        self._disable_tools_for_iteration = False
        self._read_uris = set()

        messages = self._build_messages(ctx, task_ctx)

        trace = ReactTrace()
        run_start = time.monotonic()

        iteration = 0
        max_iterations = self._max_iterations
        final_output: list[Any] = []
        tools_used: list[dict] = []
        read_uris: set[str] = set()
        finish_reason = "max_iterations"
        last_llm_error: Exception | None = None

        while iteration < max_iterations:
            check_deadline()
            iteration += 1
            iter_start = time.monotonic()
            logger.info(f"ReAct iteration {iteration}/{max_iterations}")

            elapsed = time.monotonic() - run_start
            if elapsed > self._timeout_seconds and iteration > 1:
                logger.warning(
                    f"ReAct timeout ({elapsed:.1f}s > {self._timeout_seconds}s) "
                    f"at iteration {iteration}, returning current results"
                )
                finish_reason = "timeout"
                break

            is_last_iteration = iteration >= max_iterations

            if is_last_iteration:
                messages.append({
                    "role": "user",
                    "content": self._build_final_instruction(),
                })

            try:
                tool_calls, content = self._call_llm(messages)
            except LLMCallError as e:
                # LLM transport failure (400, 500, network). Hard stop —
                # do not fall through to _on_invalid_response's "disable
                # tools and continue" path, because that turns a hard
                # server error into garbage answers. Drivers (e.g.
                # m3.qa.run_react_qa) detect ``finish_reason="llm_error"``
                # and read ``result.llm_error`` to set ``error="..."``
                # on the per-question record with the upstream's own
                # message / status code.
                iter_latency = time.monotonic() - iter_start
                logger.error(
                    f"LLM call failed (iteration {iteration}/{max_iterations}): {e}"
                )
                iter_trace = self._create_iter_trace(
                    iteration=iteration,
                    latency_seconds=iter_latency,
                    tool_calls_count=0,
                    tool_call_names=[],
                    content_length=0,
                )
                trace.iterations.append(iter_trace)
                finish_reason = "llm_error"
                last_llm_error = e
                break
            check_deadline()
            iter_latency = time.monotonic() - iter_start

            iter_trace = self._create_iter_trace(
                iteration=iteration,
                latency_seconds=iter_latency,
                tool_calls_count=len(tool_calls) if tool_calls else 0,
                tool_call_names=[tc.get("name", "") for tc in tool_calls] if tool_calls else [],
                content_length=len(content) if content else 0,
            )

            if tool_calls:
                round_id = self._record_round(tool_calls, ctx)
                exec_result = self._execute_and_append_tool_calls(
                    messages, tool_calls, tools_used, read_uris, ctx, round_id=round_id
                )
                if exec_result.get("has_unknown"):
                    self._disable_tools_for_iteration = True
                    logger.info("Unknown tool called, disabling tools for next iteration")

                if (iteration >= max_iterations and max_iterations == self._max_iterations
                        and self._allow_tool_iteration_extension()):
                    max_iterations += 1
                    self._disable_tools_for_iteration = True
                    logger.info(f"Extended max_iterations to {max_iterations} for tool call")

                self._post_tool_call_safety_check(
                    tool_calls, messages, tools_used, ctx, iter_trace
                )
                trace.iterations.append(iter_trace)
                tool_output = self._output_after_tools()
                if tool_output is not None:
                    final_output = tool_output
                    finish_reason = "done"
                    break
                continue

            if content:
                parsed = self._parse_output(content)
                if parsed is not None:
                    messages.append({"role": "assistant", "content": content})
                    should_continue = self._should_continue_after_output(
                        parsed, messages, ctx, iter_trace
                    )
                    if should_continue:
                        if is_last_iteration and max_iterations == self._max_iterations:
                            max_iterations += 1
                            self._disable_tools_for_iteration = True
                        trace.iterations.append(iter_trace)
                        continue

                    trace.iterations.append(iter_trace)
                    final_output = parsed
                    finish_reason = "done"
                    break
                else:
                    logger.warning(
                        f"_parse_output returned None on LLM content "
                        f"(iteration {iteration}/{max_iterations})"
                    )

            trace.iterations.append(iter_trace)
            if is_last_iteration:
                logger.warning("Last iteration: returning empty output")
                break

            should_end = self._on_invalid_response(content, iteration, max_iterations)
            if should_end:
                finish_reason = "no_valid_output"
                break

        trace.total_iterations = iteration
        trace.total_latency_seconds = time.monotonic() - run_start
        trace.total_read_uris = len(read_uris)

        return ReactLoopResult(
            output=final_output,
            tools_used=tools_used,
            iterations=iteration,
            read_uris=read_uris,
            trace=trace,
            completed=finish_reason == "done",
            finish_reason=finish_reason,
            llm_error=last_llm_error,
        )

    # ------------------------------------------------------------------
    # Tool list and execution — base class handles operational tools
    # ------------------------------------------------------------------

    def _output_after_tools(self) -> list[Any] | None:
        """Optional completion hook; default keeps the existing next-round flow."""
        return None

    def _allow_tool_iteration_extension(self) -> bool:
        """Preserve legacy grace rounds unless a subclass enforces a hard budget."""
        return True

    def _get_tools(self) -> list[dict]:
        """Compose operational and custom tools; legacy subclasses may override."""
        return [
            spec.tool_dict() for spec in self._operational_tool_specs
        ] + self._get_custom_tools()

    def _get_custom_tools(self) -> list[dict]:
        return []

    def _runtime_injection_values(self, ctx: RequestContext) -> dict:
        return {"ctx": ctx}

    def _execute_custom_tool(self, name: str, input: dict, ctx: RequestContext) -> dict | str:
        return {"error": f"Unknown tool: {name}"}

    def _build_final_instruction(self) -> str:
        return (
            "You have reached the maximum number of tool call iterations. "
            "Do not call any more tools - return your final result directly now."
        )

    @abstractmethod
    def _build_messages(
        self,
        ctx: RequestContext,
        task_ctx: ReactTaskContext,
    ) -> list[dict]:
        """Build initial messages for the LLM."""
        ...

    def _execute_tool(
        self,
        name: str,
        input: dict,
        ctx: RequestContext,
    ) -> dict | str:
        """Dispatch operational tools first, then scene-specific custom tools."""
        spec = self._operational_dispatch.get(name)
        if spec is not None:
            try:
                return call_with_injection(spec, input, **self._runtime_injection_values(ctx))
            except Exception as exc:
                logger.exception("Operational tool execution failed for %s", name)
                return {"error": str(exc)}
        return self._execute_custom_tool(name, input, ctx)

    @abstractmethod
    def _parse_output(self, content: str) -> list[Any] | None:
        """Parse LLM output to structured result."""
        ...

    def _should_continue_after_output(
        self,
        parsed: list[Any],
        messages: list[dict],
        ctx: RequestContext,
        iter_trace: ReactIterationTrace | None = None,
    ) -> bool:
        """Check if loop should continue after parsing output.

        Default returns False (stop after output).
        Subclasses can override for additional processing.
        """
        return False

    def _post_tool_call_safety_check(
        self,
        tool_calls: list[dict],
        messages: list[dict],
        tools_used: list[dict],
        ctx: RequestContext,
        iter_trace: ReactIterationTrace | None = None,
    ) -> bool:
        """Post-tool-call safety check hook.

        Called after all tool calls in the current round are executed.
        Subclasses can override to inject refetch results into messages
        when candidates target unread existing files.

        Returns True if a safety refetch was injected, False otherwise.
        """
        return False

    def _on_invalid_response(
        self,
        content: str | None,
        iteration: int,
        max_iterations: int,
    ) -> bool:
        """Handle an invalid LLM response (unparsable content or no output).

        Default behavior: disable tools and continue iterating.
        Subclasses can override to end the loop early instead.

        Return True to end the loop immediately, False to continue iterating.
        """
        if not content:
            logger.warning(
                f"LLM returned neither tool calls nor content "
                f"(iteration {iteration}/{max_iterations})"
            )
        self._disable_tools_for_iteration = True
        return False

    def _build_tool_descriptions(self, tools: list[dict] | None = None) -> list[str]:
        """Build human-readable tool descriptions for prompts."""
        tool_list = tools or self._get_tools()
        descriptions = []
        for tool in tool_list:
            desc = f"- {tool['name']}: {tool['description']}"
            if tool.get("input_schema", {}).get("properties"):
                params = ", ".join(tool["input_schema"]["properties"].keys())
                desc += f" (params: {params})"
            descriptions.append(desc)
        return descriptions

    def _create_iter_trace(
        self,
        iteration: int,
        latency_seconds: float,
        tool_calls_count: int = 0,
        tool_call_names: list[str] | None = None,
        content_length: int = 0,
    ) -> ReactIterationTrace:
        """Create an iteration trace. Override in subclasses to use a subclass type."""
        return ReactIterationTrace(
            iteration=iteration,
            latency_seconds=latency_seconds,
            tool_calls_count=tool_calls_count,
            tool_call_names=tool_call_names or [],
            content_length=content_length,
        )

    def _call_llm(
        self,
        messages: list[dict],
    ) -> tuple[list[dict] | None, str | None]:
        """Call LLM with tools."""
        check_deadline()
        tools = None
        tool_choice = "none"
        if not self._disable_tools_for_iteration:
            tools = self._get_tools()
            tool_choice = "auto"

        try:
            result = self._llm.complete_with_tools_messages(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )

            if isinstance(result, tuple) and len(result) == 2:
                return result

            if isinstance(result, list):
                return result, None

            if isinstance(result, str):
                return None, result

            return None, None

        except (ExtractionIncompleteError, LLMCallError):
            raise
        except Exception as e:
            # Unknown transport failure (e.g. socket error, JSON decode
            # in a response shape we didn't anticipate). Wrap so the
            # driver has a typed signal to surface as error="...".
            raise LLMCallError(f"LLM call failed: {e}", cause=e) from e

    def _execute_and_append_tool_calls(
        self,
        messages: list[dict],
        tool_calls: list[dict],
        tools_used: list[dict],
        read_uris: set[str],
        ctx: RequestContext,
        round_id: str = "",
    ) -> dict:
        """Execute tool calls and append results to messages."""
        has_unknown = False

        for tool_call in tool_calls:
            check_deadline()
            # Support both formats: {"tool": ...} from OpenAILLM and {"name": ...} from other sources
            tool_name = tool_call.get("name", "") or tool_call.get("tool", "")
            tool_input = tool_call.get("input", {})
            call_id = tool_call.get("id") or f"call_{uuid4().hex}"

            # Skip empty tool names to avoid API errors
            if not tool_name or not tool_name.strip():
                logger.warning(f"Skipping tool call with empty name: call_id={call_id} tool_call={tool_call}")
                has_unknown = True
                continue

            started = time.monotonic()
            result = self._execute_tool(tool_name, tool_input, ctx)
            duration_ms = (time.monotonic() - started) * 1000

            has_error = isinstance(result, dict) and bool(result.get("error"))
            if isinstance(result, dict) and result.get("error", "").startswith("Unknown tool"):
                has_unknown = True

            self._record_tool_call(
                ctx=ctx,
                round_id=round_id,
                tool_name=tool_name,
                status="error" if has_error else "success",
                duration_ms=duration_ms,
                error_type=str(result.get("error", "")) if isinstance(result, dict) else "",
            )

            self._track_tool_usage(tool_name, tool_input, result, tools_used, read_uris)

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_input),
                    },
                }],
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result) if isinstance(result, dict) else str(result),
            })

        return {"has_unknown": has_unknown}

    def _track_tool_usage(
        self,
        tool_name: str,
        tool_input: dict,
        result: dict | str,
        tools_used: list[dict],
        read_uris: set[str],
    ) -> None:
        """Track tool usage for result metadata."""
        tools_used.append({
            "tool_name": tool_name,
            "params": tool_input,
            "result": result,
        })

    def _record_round(self, tool_calls: list[dict], ctx: RequestContext) -> str:
        """Record a round of tool calls for usage tracking."""
        round_id = self._round_id_from_tool_calls(tool_calls)
        if not self._internal_tool_usage_tracker:
            return round_id
        usage = self._usage_from_tool_calls(tool_calls)
        self._internal_tool_usage_tracker.record_round(
            account_id=ctx.account_id,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            pipeline_name=self._pipeline_name,
            round_id=round_id,
            tool_names=[
                call.get("name") or call.get("tool", "")
                for call in tool_calls
                if call.get("name") or call.get("tool")
            ],
            model=usage.get("model", ""),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read=usage.get("cache_read", 0),
            cache_write=usage.get("cache_write", 0),
        )
        return round_id

    def _record_tool_call(
        self,
        *,
        ctx: RequestContext,
        round_id: str,
        tool_name: str,
        status: str,
        duration_ms: float,
        error_type: str = "",
    ) -> None:
        """Record a single tool call for usage tracking."""
        if not self._internal_tool_usage_tracker or not tool_name:
            return
        self._internal_tool_usage_tracker.record_tool_call(
            account_id=ctx.account_id,
            session_id=ctx.session_id,
            pipeline_name=self._pipeline_name,
            round_id=round_id,
            tool_name=tool_name,
            status=status,
            duration_ms=duration_ms,
            error_type=error_type,
        )

    @staticmethod
    def _usage_from_tool_calls(tool_calls: list[dict]) -> dict:
        for call in tool_calls or []:
            usage = call.get("_llm_usage")
            if isinstance(usage, dict):
                return usage
        return {}

    @staticmethod
    def _round_id_from_tool_calls(tool_calls: list[dict]) -> str:
        usage = ReactLoop._usage_from_tool_calls(tool_calls)
        if usage.get("round_id"):
            return str(usage["round_id"])
        for call in tool_calls or []:
            if call.get("id"):
                return str(call["id"])
        return f"llm_{uuid4().hex[:12]}"
