"""ReAct loop for memory extraction with tool use.

Implements iterative tool-assisted extraction with safety checks.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

from core.errors import (
    ExtractionIncompleteError,
    OperationDeadlineExceededError,
    ToolCallProtocolError,
    ValidationError,
)
from core.interfaces import LLM, ContextFS
from core.models import CandidateMemory, ContextNode, RequestContext
from core.operation_deadline import check_deadline
from core.react_loop import (
    ReactIterationTrace,
    ReactLoop,
    ReactLoopResult,
    ReactTaskContext,
    ReactTrace,
)
from core.uri_resolver import URIResolver
from core.validation import validate_candidate
from extraction.incremental_edit import (
    INCREMENTAL_TOOLS,
    coalesce_incremental_edits,
    incremental_read_key,
)
from extraction.prefetch import MemoryPrefetcher, PrefetchResult
from extraction.schemas.registry import SchemaRegistry
from extraction.tool_builder import (
    build_extraction_tools,
    build_update_instruction,
    parse_tool_call,
    validate_read_target,
)

logger = getLogger(__name__)


# Built-in tool definitions for reading memory nodes
_READ_TOOL = {
    "name": "read",
    "description": "Read a memory node by URI.",
    "input_schema": {
        "type": "object",
        "properties": {
            "uri": {
                "type": "string",
                "description": "URI of the memory node to read.",
            }
        },
        "required": ["uri"],
    },
}

_LIST_TOOL = {
    "name": "list",
    "description": "List child nodes under a directory URI.",
    "input_schema": {
        "type": "object",
        "properties": {
            "uri": {
                "type": "string",
                "description": "URI of the directory to list.",
            }
        },
        "required": ["uri"],
    },
}

_RELATIONS_TOOL = {
    "name": "get_relations",
    "description": "Get related nodes for a URI. Returns one-hop neighbors.",
    "input_schema": {
        "type": "object",
        "properties": {"uri": {"type": "string"}},
        "required": ["uri"],
    },
}

_ACCESS_STATS_TOOL = {
    "name": "get_access_stats",
    "description": "Get access stats for a node: last access time, 30d hit count.",
    "input_schema": {
        "type": "object",
        "properties": {"uri": {"type": "string"}},
        "required": ["uri"],
    },
}


@dataclass
class ExtractionIterationTrace(ReactIterationTrace):
    """Extraction-specific iteration trace with safety check fields."""
    safety_check_triggered: bool = False
    refetch_uris: list[str] = field(default_factory=list)


@dataclass
class ExtractionTaskContext(ReactTaskContext):
    """Task context for extraction ReAct loop."""
    conversation_text: str
    prefetch_result: PrefetchResult | None = None


@dataclass
class ExtractionTrace(ReactTrace):
    """Extraction-specific trace with candidate count."""
    final_candidate_count: int = 0


@dataclass
class ExtractionReActResult:
    """Result of a ReAct extraction loop."""
    candidates: list[CandidateMemory] = field(default_factory=list)
    tools_used: list[dict] = field(default_factory=list)
    iterations: int = 0
    read_uris: set[str] = field(default_factory=set)
    trace: ExtractionTrace | None = None
    read_versions: dict[str, int] = field(default_factory=dict)
    read_nodes: dict[str, ContextNode] = field(default_factory=dict)
    # Pass-through of the underlying loop's finish state so callers
    # (and tests) can detect transport failures (``finish_reason ==
    # "llm_error"``) without sniffing the trace. The base
    # :class:`core.react_loop.ReactLoopResult` carries these; the
    # extraction subclass previously dropped them on the floor.
    finish_reason: str = ""
    completed: bool = False
    # Original :class:`LLMCallError` if the underlying loop terminated
    # due to a transport failure. ``None`` for other finish reasons.
    llm_error: Exception | None = None


class ExtractionReActLoop(ReactLoop):
    """ReAct loop for memory extraction with tool use.

    Workflow:
    1. Start with system prompt + prefetch context + conversation
    2. While iteration < max_iterations:
       a. Call LLM with tools (read, list, extract_*)
       b. If tool calls: execute and continue
       c. If content: parse operations → safety check → return
       d. If neither: disable tools and continue
    3. Return extracted candidates with metadata

    Safety features:
    - Refetch check: re-read existing files before write operations
    - URI validation: ensure all URIs are within allowed namespace
    - Single safety re-read: prevent infinite loops
    """

    def __init__(
        self,
        llm: LLM,
        fs: ContextFS,
        registry: SchemaRegistry,
        uri_resolver: URIResolver,
        prompt_manager: Any,
        prefetcher: MemoryPrefetcher | None = None,
        max_iterations: int = 3,
        timeout_seconds: float = 30.0,
        internal_tool_usage_tracker: Any | None = None,
    ):
        super().__init__(
            llm, max_iterations, timeout_seconds,
            internal_tool_usage_tracker=internal_tool_usage_tracker,
            pipeline_name="extraction.lazy",
        )
        self._fs = fs
        self._registry = registry
        self._uri_resolver = uri_resolver
        self._prefetcher = prefetcher
        self._prompt_manager = prompt_manager

        self._extraction_tools = build_extraction_tools(registry)
        self._update_instruction = build_update_instruction(registry)

        # All available tools: read, list, relations, access_stats, and extraction tools
        self._all_tools = [_READ_TOOL, _LIST_TOOL, _RELATIONS_TOOL, _ACCESS_STATS_TOOL] + self._extraction_tools

        self._did_safety_reread: bool = False
        self._read_versions: dict[str, int] = {}
        self._read_nodes = {}
        self._edit_contents = {}
        self._materialized_edits = []
        self._edit_candidates = {}
        self._edit_errors = []
        self._extraction_errors = []
        self._ctx = None

    def run(
        self,
        conversation_text: str,
        ctx: RequestContext,
        prefetch_result: PrefetchResult | None = None,
    ) -> ExtractionReActResult:
        check_deadline()
        # Reset runtime state
        self._did_safety_reread = False
        self._read_versions = dict(prefetch_result.read_versions) if prefetch_result else {}
        self._read_nodes = deepcopy(prefetch_result.read_nodes) if prefetch_result else {}
        self._edit_contents = {}
        self._materialized_edits = []
        self._edit_candidates = {}
        self._edit_errors = []
        self._extraction_errors = []
        self._ctx = ctx

        task_ctx = ExtractionTaskContext(
            conversation_text=conversation_text,
            prefetch_result=prefetch_result,
        )

        loop_result = super().run(ctx, task_ctx)
        check_deadline()
        if not loop_result.completed:
            raise ExtractionIncompleteError(
                f"Extraction did not complete: {loop_result.finish_reason}"
            )
        return self._to_extraction_react_result(loop_result, ctx)

    def _get_tools(self) -> list[dict]:
        return self._all_tools

    def _create_iter_trace(
        self,
        iteration: int,
        latency_seconds: float,
        tool_calls_count: int = 0,
        tool_call_names: list[str] | None = None,
        content_length: int = 0,
    ) -> ExtractionIterationTrace:
        return ExtractionIterationTrace(
            iteration=iteration,
            latency_seconds=latency_seconds,
            tool_calls_count=tool_calls_count,
            tool_call_names=tool_call_names or [],
            content_length=content_length,
        )

    def _build_messages(
        self,
        ctx: RequestContext,
        task_ctx: ReactTaskContext,
    ) -> list[dict]:
        """Build messages for extraction using YAML template."""
        conversation_text = task_ctx.conversation_text
        prefetch_result = task_ctx.prefetch_result

        messages = []

        system_prompt = self._prompt_manager.render("extraction", "system_prompt")
        lazy_tools = self._prompt_manager.render("extraction", "lazy_tools_instruction")
        identity_block = self._prompt_manager.render("extraction", "identity_block", user_id=ctx.user_id)
        examples = self._prompt_manager.render("extraction", "examples")
        output_instruction = self._prompt_manager.render("extraction", "output_instruction")

        # Build system prompt: system_prompt + identity + lazy_tools + examples + output
        system_content = f"{system_prompt}\n\n{identity_block}\n\n{lazy_tools}\n\n{examples}\n\n{output_instruction}"
        system_content += "\n\n" + self._update_instruction

        messages = [{"role": "system", "content": system_content}]

        # Add prefetch context if available
        if prefetch_result:
            # Base run() resets read tracking before building messages. Seed it
            # here so prefetched memories survive that reset, without leaking
            # URIs from previous runs.
            self._read_uris.update(prefetch_result.read_uris)
            if prefetch_result.is_lazy and prefetch_result.uri_hints:
                # Lazy mode: add lightweight URI hints, LLM reads on demand
                hints_block = self._format_prefetch_hints(prefetch_result.uri_hints)
                messages.append({"role": "system", "content": hints_block})
            if prefetch_result.messages:
                # Full bodies or identity catalogs accompany lazy hints as well.
                for msg in prefetch_result.messages:
                    messages.append({"role": "system", "content": msg})

        # Add conversation text
        messages.append({
            "role": "user",
            "content": f"## Conversation\n\n{conversation_text}",
        })

        return messages

    def _format_prefetch_hints(self, uri_hints: list[dict]) -> str:
        """Format prefetch URI hints for Lazy mode."""
        if not uri_hints:
            return ""

        hints_block = "## Available Memory URIs (use `read` tool to explore)\n"
        for hint in uri_hints[:5]:
            abstract_preview = hint.get("abstract", "")[:100]
            score = hint.get("score", 0.0)
            if abstract_preview:
                hints_block += f"- URI: {hint['uri']} (similarity: {score:.2f})\n  Abstract: {abstract_preview}...\n"
            else:
                hints_block += f"- URI: {hint['uri']}\n"
        return hints_block

    def _execute_tool(
        self,
        name: str,
        input: dict,
        ctx: RequestContext,
    ) -> dict | str:
        check_deadline()
        try:
            if name == "read":
                uri = input.get("uri", "")
                if not uri:
                    return {"error": "Missing required parameter: uri"}

                node = self._fs.read_node(uri, ctx)
                self._read_versions[uri] = node.metadata.get("version", 0)
                self._read_nodes[uri] = deepcopy(node)
                return {
                    "uri": uri,
                    "abstract": node.abstract,
                    "overview": node.overview,
                    "content": node.content,
                    "metadata": node.metadata,
                }

            elif name == "list":
                uri = input.get("uri", "")
                if not uri:
                    return {"error": "Missing required parameter: uri"}

                children = self._fs.list_children(uri, ctx)
                return {
                    "uri": uri,
                    "children": children,
                }

            elif name == "get_relations":
                uri = input.get("uri", "")
                return self._execute_get_relations(uri, ctx)

            elif name == "get_access_stats":
                uri = input.get("uri", "")
                return self._execute_get_stats(uri, ctx)

            elif name in INCREMENTAL_TOOLS:
                candidate = self._to_candidate(name, input)
                operation_id = len(self._materialized_edits)
                self._materialized_edits.append(deepcopy(candidate))
                return {"status": "recorded", "operation_id": operation_id}

            elif name.startswith("extract_"):
                candidate = self._to_candidate(name, input)
                if candidate is None:
                    raise ToolCallProtocolError(f"{name} did not produce a schema-valid candidate")
                validate_candidate(candidate)
                validate_read_target(candidate, self._read_versions, read_nodes=self._read_nodes, ctx=ctx)
                return {"status": "recorded"}

            else:
                return {"error": f"Unknown tool: {name}"}

        except OperationDeadlineExceededError:
            raise
        except Exception as e:
            if name in INCREMENTAL_TOOLS:
                self._edit_errors.append(str(e))
            elif name.startswith("extract_"):
                if not isinstance(e, (ValidationError, ToolCallProtocolError)):
                    raise
                self._extraction_errors.append(e)
            logger.error(f"Tool execution failed for {name}: {e}")
            return {"error": str(e)}

    def _to_candidate(self, tool_name: str, tool_input: dict) -> CandidateMemory | None:
        """Convert an extraction or incremental tool call to CandidateMemory.

        Unified parsing used by both _parse_output and _to_extraction_react_result.
        """
        if tool_name in INCREMENTAL_TOOLS:
            key = incremental_read_key(tool_input)
            parsed = parse_tool_call(
                tool_name, tool_input, self._registry, read_nodes=self._read_nodes, ctx=self._ctx,
                current_content=self._edit_contents.get(key),
            )
            candidate = parsed[2]
            self._edit_contents[key] = candidate.content
            self._edit_candidates[id(candidate)] = candidate
            return candidate
        if not tool_name.startswith("extract_"):
            return None
        if "raw" in tool_input and len(tool_input) == 1:
            logger.warning(
                "Skipping tool call with unparsed raw arguments: tool=%s raw=%.200s",
                tool_name, tool_input["raw"],
            )
            return None
        parsed = parse_tool_call(tool_name, tool_input, self._registry)
        if parsed:
            memory_type, owner_scope, candidate = parsed
            logger.info(f"Parsed candidate from {tool_name}: {memory_type}/{candidate.routing_key}")
            return candidate
        return None

    def _build_final_instruction(self) -> str:
        return (
            "Do not call any more tools. Return ONLY a JSON array of remaining "
            "extraction operations, each with name and input. "
            "Operations already recorded through extract_*, edit_memory or append_memory "
            "tools need not be repeated. "
            "If all operations have been recorded or there are no memories, return []."
        )

    def _parse_output(self, content: str) -> list[Any] | None:
        # JSON output is tentative until safety checks accept it. Compose within
        # this output, retaining native edits but never committing discarded JSON.
        recorded_contents = self._edit_contents.copy()
        try:
            try:
                values = [json.loads(content)]
            except json.JSONDecodeError:
                from providers.llm.openai_llm import OpenAILLM

                values = OpenAILLM._extract_json_values_from_text(content)

            has_empty_output = False
            for data in values:
                if data == [] or (
                    isinstance(data, dict)
                    and data.get("tool_calls", data.get("operations")) == []
                ):
                    has_empty_output = True
                    continue
                candidates = self._candidates_from_json(data)
                if candidates:
                    return candidates
            return [] if has_empty_output else None

        except ToolCallProtocolError:
            raise
        except json.JSONDecodeError:
            logger.warning("Failed to parse content as JSON")
            return None
        except Exception as e:
            logger.error(f"Error parsing operations: {e}")
            return None
        finally:
            self._edit_contents = recorded_contents

    def _candidates_from_json(self, data: Any) -> list[Any]:
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if "tool_calls" in data:
                items = data["tool_calls"]
            elif "operations" in data:
                items = data["operations"]
            else:
                items = [data]
        else:
            logger.warning(f"Unexpected JSON structure: {type(data)}")
            return []

        candidates = []
        for item in items:
            if not isinstance(item, dict):
                continue

            tool_name = item.get("name", "") or item.get("tool", "")
            tool_input = item.get("input", item)
            if not tool_name.startswith("extract_") and tool_name not in INCREMENTAL_TOOLS:
                for key in item:
                    if key.startswith("extract_"):
                        tool_name = key
                        tool_input = item[key]
                        break

            candidate = self._to_candidate(tool_name, tool_input)
            if candidate:
                candidates.append(candidate)
        return candidates

    def _should_continue_after_output(
        self,
        parsed: list[Any],
        messages: list[dict],
        ctx: RequestContext,
        iter_trace: ReactIterationTrace | None = None,
    ) -> bool:
        candidates = [c for c in parsed if isinstance(c, CandidateMemory)]
        return self._safety_check_candidates(candidates, messages, ctx, iter_trace)

    def _post_tool_call_safety_check(
        self,
        tool_calls: list[dict],
        messages: list[dict],
        tools_used: list[dict],
        ctx: RequestContext,
        iter_trace: ReactIterationTrace | None = None,
    ) -> bool:
        # A failed native operation has no reliable correction ID. Restart the
        # entire span through the existing eager fallback instead of forgetting it.
        if self._extraction_errors:
            raise ToolCallProtocolError(
                "native extraction contains a failed operation: "
                + str(self._extraction_errors[0])[:1000]
            ) from self._extraction_errors[0]
        candidates = []
        extract_tool_names = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "") or tool_call.get("tool", "")
            tool_input = tool_call.get("input", {})
            if tool_name.startswith("extract_"):
                candidate = self._to_candidate(tool_name, tool_input)
                if candidate:
                    candidates.append(candidate)
                extract_tool_names.append(tool_name)

        triggered = self._safety_check_candidates(candidates, messages, ctx, iter_trace)

        if triggered:
            # Mark pre-refetch extract_* records as superseded so they
            # are excluded from final candidate collection.
            for record in tools_used:
                if record.get("tool_name", "") in extract_tool_names:
                    record["superseded"] = True

        return triggered

    def _safety_check_candidates(
        self,
        candidates: list[CandidateMemory],
        messages: list[dict],
        ctx: RequestContext,
        iter_trace: ReactIterationTrace | None = None,
    ) -> bool:
        """Unified safety check for unread existing files.

        Used by both content-output and tool-call paths.
        Returns True if refetch was needed (loop should continue).
        """
        if self._did_safety_reread:
            return False
        if not candidates:
            return False

        refetch_uris = self._check_unread_existing_files(candidates, ctx)
        if refetch_uris:
            logger.info(f"Found unread existing files: {refetch_uris}, refetching...")
            self._add_refetch_results(messages, refetch_uris, ctx)
            if iter_trace and isinstance(iter_trace, ExtractionIterationTrace):
                iter_trace.safety_check_triggered = True
                iter_trace.refetch_uris = refetch_uris
            return True

        return False

    def _on_invalid_response(
        self,
        content: str | None,
        iteration: int,
        max_iterations: int,
    ) -> bool:
        """End the loop on invalid response — extraction cannot proceed without tools."""
        if not content:
            logger.warning(
                f"LLM returned neither tool calls nor content, ending extraction loop "
                f"(iteration {iteration}/{max_iterations})"
            )
        else:
            logger.warning(
                f"LLM content could not be parsed as extraction operations, ending loop "
                f"(iteration {iteration}/{max_iterations})"
            )
        return True

    def _track_tool_usage(
        self,
        tool_name: str,
        tool_input: dict,
        result: dict | str,
        tools_used: list[dict],
        read_uris: set[str],
    ) -> None:
        super()._track_tool_usage(tool_name, tool_input, result, tools_used, read_uris)

        # Track read operations
        if tool_name == "read" and tool_input.get("uri"):
            uri = tool_input["uri"]
            self._read_uris.add(uri)
            read_uris.add(uri)

    def _check_unread_existing_files(
        self, candidates: list[CandidateMemory], ctx: RequestContext
    ) -> list[str]:
        if self._did_safety_reread:
            return []

        refetch_uris: list[str] = []
        seen_refetch_uris: set[str] = set()

        for candidate in candidates:
            schema = self._registry.get(candidate.category)
            if schema is None:
                continue

            # Skip add-only schemas (no conflict risk)
            if schema.is_add_only:
                continue

            fields = self._candidate_to_fields(candidate)
            try:
                uri = candidate.target_uri or self._uri_resolver.resolve(candidate.category, fields, ctx)
                if candidate.target_uri and not self._uri_resolver.validate_uri(uri, ctx):
                    raise ToolCallProtocolError("memory update target must be in the current namespace")
            except Exception as e:
                logger.warning(f"Failed to resolve URI for {candidate.category}: {e}")
                continue

            # Check if URI was already read
            if uri in self._read_uris:
                continue

            # Check if file exists
            if uri not in seen_refetch_uris and self._fs.exists(uri, ctx):
                refetch_uris.append(uri)
                seen_refetch_uris.add(uri)

        return refetch_uris

    def _add_refetch_results(
        self, messages: list[dict], refetch_uris: list[str], ctx: RequestContext
    ) -> None:
        self._did_safety_reread = True

        for uri in refetch_uris:
            try:
                node = self._fs.read_node(uri, ctx)
                self._read_versions[uri] = node.metadata.get("version", 0)
                self._read_nodes[uri] = deepcopy(node)
                self._read_uris.add(uri)
                call_id = f"refetch_{hashlib.sha256(uri.encode('utf-8')).hexdigest()[:16]}"

                result = {
                    "uri": uri,
                    "abstract": node.abstract,
                    "overview": node.overview,
                    "content": node.content,
                    "metadata": node.metadata,
                }

                # Add as read tool result
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps({"uri": uri}),
                        },
                    }],
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result),
                })

            except Exception as e:
                logger.warning(f"Failed to refetch {uri}: {e}")

        # Add safety reminder
        messages.append({
            "role": "user",
            "content": (
                "SAFETY REMINDER: The files above were automatically read because they "
                "exist and you didn't read them before deciding to write. Please consider "
                "the existing content when making write decisions. You can now output "
                "updated operations."
            ),
        })

    def _validate_operations_uris(
        self, candidates: list[CandidateMemory], ctx: RequestContext
    ) -> list[str]:
        errors = []

        for candidate in candidates:
            fields = self._candidate_to_fields(candidate)
            try:
                uri = candidate.target_uri or self._uri_resolver.resolve(candidate.category, fields, ctx)
                if not self._uri_resolver.validate_uri(uri, ctx):
                    errors.append(
                        f"Invalid URI for {candidate.category}/{candidate.routing_key}: {uri}"
                    )
            except Exception as e:
                errors.append(
                    f"Failed to validate URI for {candidate.category}/{candidate.routing_key}: {e}"
                )

        return errors

    def _candidate_to_fields(self, candidate: CandidateMemory) -> dict:
        # Get the routing key field name for this category
        fields = {
            "routing_key": candidate.routing_key,
            "abstract": candidate.abstract,
            "overview": candidate.overview,
            "content": candidate.content,
            "confidence": candidate.confidence,
        }

        if candidate.when:
            fields["when"] = candidate.when
        if candidate.who:
            fields["who"] = candidate.who
        if candidate.where:
            fields["where"] = candidate.where
        # Add optional fields if present
        if candidate.tool_stats:
            fields.update(candidate.tool_stats)

        return fields

    def _execute_get_relations(self, uri: str, ctx: RequestContext) -> dict:
        try:
            if hasattr(self._fs, 'get_relations'):
                edges = self._fs.get_relations(uri, ctx)
                return {"uri": uri, "relations": [
                    {"from": e.from_uri, "to": e.to_uri, "weight": e.weight, "reason": e.reason}
                    for e in edges
                ]}
            return {"uri": uri, "relations": [], "note": "Not available"}
        except Exception as e:
            return {"error": str(e)}

    def _execute_get_stats(self, uri: str, ctx: RequestContext) -> dict:
        try:
            node = self._fs.read_node(uri, ctx)
            metadata = node.metadata or {}
            return {
                "uri": uri,
                "last_accessed_at": metadata.get("last_accessed_at"),
                "hit_count_30d": metadata.get("hit_count_30d", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def _to_extraction_react_result(self, loop_result: ReactLoopResult, ctx: RequestContext) -> ExtractionReActResult:
        trace = None
        if loop_result.trace:
            trace = ExtractionTrace(
                total_iterations=loop_result.trace.total_iterations,
                total_latency_seconds=loop_result.trace.total_latency_seconds,
                iterations=loop_result.trace.iterations,
                total_read_uris=len(self._read_uris),
                final_candidate_count=len(loop_result.output),
            )

        if self._edit_errors:
            raise ToolCallProtocolError("incremental extraction contains failed operations")
        if self._extraction_errors:
            raise ToolCallProtocolError("native extraction contains failed operations")
        # Materialized edits have already bound their snapshot at execution time.
        entries = [(id(item) in self._edit_candidates, item) for item in loop_result.output
                   if isinstance(item, CandidateMemory)]

        # Also extract candidates from tools_used (extract_* tool calls)
        # This handles cases where LLM used tool calling instead of outputting JSON
        # Skip records superseded by a safety refetch (pre-refetch candidates
        # were produced without reading existing content).
        native_edits = []
        for tool_record in loop_result.tools_used:
            if tool_record.get("superseded"):
                continue
            tool_name = tool_record.get("tool_name", "")
            tool_params = tool_record.get("params", {})
            tool_result = tool_record.get("result", {})
            # Use params (original input) with unified _to_candidate method
            if tool_result.get("status") == "recorded":
                incremental = tool_name in INCREMENTAL_TOOLS
                if incremental:
                    candidate = self._materialized_edits[tool_result["operation_id"]]
                else:
                    candidate = self._to_candidate(tool_name, tool_params)
                if candidate:
                    if incremental:
                        native_edits.append((True, candidate))
                    else:
                        entries.append((False, candidate))

        # Native calls precede final JSON. Keep that chronology for composed edits,
        # while retaining the original relative order of complete extract operations.
        entries = native_edits + entries
        for incremental, candidate in entries:
            if not incremental:
                validate_read_target(candidate, self._read_versions, read_nodes=self._read_nodes, ctx=ctx)
        candidates = coalesce_incremental_edits(entries)

        # Validate URIs before returning
        errors = self._validate_operations_uris(candidates, ctx)
        if errors:
            logger.error(f"URI validation errors: {errors}")

        return ExtractionReActResult(
            candidates=candidates,
            tools_used=loop_result.tools_used,
            iterations=loop_result.iterations,
            read_uris=self._read_uris.copy(),
            read_versions=self._read_versions.copy(), read_nodes=deepcopy(self._read_nodes),
            trace=trace,
            finish_reason=loop_result.finish_reason,
            completed=loop_result.completed,
            llm_error=loop_result.llm_error,
        )
