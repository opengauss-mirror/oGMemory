"""Decision-driven control of the native extraction loop."""

import json
from copy import deepcopy
from math import isfinite

from core.errors import (
    ExtractionIncompleteError,
    ExtractionOutputError,
    LLMCallError,
    LLMOutputTruncatedError,
    LLMTransientError,
    OperationDeadlineExceededError,
    ToolCallProtocolError,
)
from core.model_stage import run_model_call
from core.operation_deadline import check_deadline
from extraction.assembly_contract import UnsupportedAssemblyTemplateError
from extraction.extraction_react_loop import ExtractionReActLoop
from extraction.incremental_edit import INCREMENTAL_TOOLS, incremental_read_key
from extraction.prefetch import IDENTITY_HINT_FIELDS
from extraction.prompt_assembly import PromptParts, validate_offered_calls
from extraction.tool_builder import build_extraction_tools, build_update_instruction
from providers.decision.client import DecisionError
from providers.llm_observation import observe_call, record_decision_result
from providers.token_tracker import UsagePersistenceError

_NEXT = {
    "EXPLORE_OLD_MEMORY": "A specific missing old memory body is needed to process the source.",
    "FORM_MEMORY_OPERATION": "Current evidence supports forming more memory operations.",
    "FINISH_EXTRACTION": "All useful facts and corrections in the source have been handled.",
}
_FORM = "FORM_MEMORY_OPERATION"
_FINISH = "FINISH_EXTRACTION"
_EXPLORE = "EXPLORE_OLD_MEMORY"
_GENERATION = "memory_generation"


class DecisionContextBuilder:
    """Render decision questions with the caller's extraction template manager."""

    def __init__(self, manager):
        self.manager = manager
        self.common = manager.render("extraction_decision", "system_prompt")
        self.next_step = manager.render("extraction_decision", "next_step")

    def next_question(self, allow_observation):
        return {"next_step": {
            "type": "choice",
            "instructions": self.common + "\n" + self.next_step,
            "criteria": {name: text for name, text in _NEXT.items()
                         if name != _EXPLORE or allow_observation},
        }}

    def read_questions(self, keys):
        return {
            key: {
                "type": "noul",
                "instructions": self.common + "\n" + self.manager.render(
                    "extraction_decision", "read_candidates", candidate_key=key,
                ),
                "criteria": {
                    "true": "Correct handling of the source requires this unread body.",
                    "false": "The body is unnecessary or current evidence is enough.",
                },
            }
            for key in keys
        }


class JevExtractionReActLoop(ExtractionReActLoop):
    def __init__(
        self, *, decision_client, generate_operations, read_probability_threshold, correlation=None, **kwargs,
    ):
        super().__init__(**kwargs)
        if (type(read_probability_threshold) not in (int, float)
                or not 0 < read_probability_threshold <= 1):
            raise ValueError("read_probability_threshold must be finite and in (0, 1]")
        self._threshold = read_probability_threshold
        self._decision_client = decision_client
        self._generate_operations = generate_operations
        self._correlation = dict(correlation or {})
        self._template = "extraction_graph" if self._activation_enabled else "extraction"
        try:
            self._prompt_manager.validate_purpose(self._template, _GENERATION)
            self._prompt_manager.load("extraction_decision")
            self._decisions = DecisionContextBuilder(self._prompt_manager)
        except (FileNotFoundError, ValueError) as exc:
            raise UnsupportedAssemblyTemplateError(f"Invalid JEV extraction template: {exc}") from exc

    def _build_messages(self, ctx, task_ctx):
        messages = super()._build_messages(ctx, task_ctx)
        self._source = task_ctx.conversation_text
        self._iteration = 0
        self._final_content = False
        self._finished_by_decision = False
        self._records = []
        self._read_selection = None
        self._index = self._memory_index(task_ctx.prefetch_result)
        return messages

    def _memory_index(self, prefetch):
        if prefetch is None:
            return {}
        directories = {
            schema.memory_type: self._uri_resolver.get_directory_uri(schema.memory_type, self._ctx).rstrip("/")
            for schema in self._registry.list_compatible_enabled()
            if self._assembly_categories is None or schema.memory_type in self._assembly_categories
        }
        hints = {hint["uri"]: dict(hint) for hint in prefetch.uri_hints if hint.get("uri")}
        for identity in prefetch.identity_hints:
            uri = identity["uri"]
            hint = hints.get(uri, {})
            hints[uri] = {**hint, **identity,
                          "abstract": identity.get("abstract") or hint.get("abstract", "")}
        for uri in sorted(prefetch.listed_uris | prefetch.read_uris):
            hints.setdefault(uri, {})
        index = {}
        for uri, hint in hints.items():
            if not isinstance(uri, str):
                continue
            matches = [category for category, directory in directories.items()
                       if uri.startswith(directory + "/")
                       or (uri == directory and self._registry.get_compatible(category).is_single_file)]
            if not matches:
                continue
            category = max(matches, key=lambda name: len(directories[name]))
            entry = {"uri": uri, "category": category, "abstract": hint.get("abstract", "")}
            entry.update({key: hint[key] for key in IDENTITY_HINT_FIELDS if key in hint})
            score = hint.get("score")
            if type(score) in (int, float) and isfinite(score):
                entry["score"] = score
            index[f"memory_{len(index)}"] = entry
        return index

    def _incomplete_error(self, result):
        return result.llm_error or super()._incomplete_error(result)

    def _build_final_instruction(self):
        self._final_content = True
        return super()._build_final_instruction()

    def _track_tool_usage(self, tool_name, tool_input, result, tools_used, read_uris):
        super()._track_tool_usage(tool_name, tool_input, result, tools_used, read_uris)
        self._records = tools_used
        self._read_selection = None

    def _evidence(self):
        reads = [
            {"uri": uri, "version": self._read_versions.get(uri),
             "category": node.category, "abstract": node.abstract, "overview": node.overview,
             "content": node.content, "metadata": node.metadata}
            for uri, node in self._read_nodes.items()
        ]
        operations, feedback = [], []
        for index, call in enumerate(self._records):
            result = call["result"]
            if not isinstance(result, dict):
                continue
            tool, inputs = call["tool_name"], call["params"]
            status = "superseded" if call.get("superseded") else result.get("status")
            event = {"call": index, "tool": tool, "status": status or ("error" if result.get("error") else "ok")}
            if tool == "read":
                event["uri"] = inputs["uri"]
            if result.get("error"):
                event["error"] = result["error"]
            if status == "recorded":
                operation = {"call": index, "tool": tool, "input": inputs}
                if tool in INCREMENTAL_TOOLS:
                    operation["current_content"] = self._edit_contents[incremental_read_key(inputs)]
                operations.append(operation)
            feedback.append(event)
        return reads, operations, feedback

    def _state(self, stage, *, allow_observation, tentative=()):
        reads, operations, feedback = self._evidence()
        return {
            "decision_goal": stage,
            "source_span": {"span_id": self._correlation.get("span_id", "current"),
                            "text": self._source, "session_time": self._correlation.get("session_time")},
            "request_identity": {field: getattr(self._ctx, field)
                                 for field in ("account_id", "user_id", "agent_id")},
            "allowed_categories": [tool["name"].removeprefix("extract_") for tool in self._memory_tools()
                                   if tool["name"].startswith("extract_") and tool["name"] != "extract_relation"],
            "native_policy": self._update_instruction,
            "memory_index": self._index,
            "read_results": reads,
            "recorded_operations": operations,
            "tool_feedback": feedback,
            "read_selection_feedback": self._read_selection,
            "tentative_operations": list(tentative),
            "execution_constraints": {
                "iteration": self._iteration,
                "max_iterations": self._max_iterations,
                "allow_observation": allow_observation,
                "final_content": self._text_mode,
            },
        }

    @property
    def _text_mode(self):
        return self._final_content or self._disable_tools_for_iteration

    def _evaluate(self, stage, questions, *, allow_observation, tentative=(), final_check=False):
        check_deadline()

        def invoke(state, questions):
            if stage == "read_candidates":
                batch = self._decision_client.evaluate(state, questions)
                selected = [key for key in questions if batch.answers[key] >= self._threshold]
                record_decision_result(
                    probabilities=batch.answers, threshold=self._threshold, selected=selected,
                    option_count=len(questions), elapsed_seconds=batch.seconds,
                )
                return batch.answers, selected
            batch = self._decision_client.evaluate_choice(state, questions)
            scores = batch.answers["next_step"]
            maximum = max(scores.values())
            action = next(name for name in _NEXT if name in scores and scores[name] >= maximum - 1e-12)
            record_decision_result(choices={"next_step": action}, option_count=len(scores),
                                   elapsed_seconds=batch.seconds)
            return action

        correlation = {**self._correlation, "iteration": self._iteration, "decision_stage": stage}
        if final_check:
            correlation["phase"] = "final_content_check"
        try:
            result = observe_call(
                "extraction", invoke,
                self._state(stage, allow_observation=allow_observation, tentative=tentative),
                questions, correlation=correlation,
            )
        except DecisionError as exc:
            if hasattr(exc, "error_type"):
                raise LLMCallError("Decision model call failed", cause=exc,
                                   status_code=getattr(exc, "status_code", None)) from exc
            raise ToolCallProtocolError(str(exc)) from exc
        check_deadline()
        return result

    def _memory_tools(self):
        return [tool for tool in self._get_tools()
                if tool["name"].startswith("extract_") or tool["name"] in INCREMENTAL_TOOLS]

    def _generation_tools(self):
        # URI hints and partial reads do not authorize version-bound operations.
        read_categories = set()
        for uri, node in self._read_nodes.items():
            version = self._read_versions.get(uri)
            if (type(version) is int and version > 0 and version == node.metadata.get("version")
                    and all(isinstance(text, str) and text.strip()
                            for text in (node.abstract, node.overview, node.content))):
                read_categories.add(node.category)
        tools = self._memory_tools()
        bound_names = {
            tool["name"] for tool in tools
            if tool["name"].removeprefix("extract_") in read_categories
            and "target_uri" in tool["input_schema"].get("properties", {})
        }
        source_only = {
            tool["name"]: tool for tool in build_extraction_tools(
                self._registry, include_updates=False, activation_enabled=self._activation_enabled,
            )
        }
        selected = []
        for tool in tools:
            name = tool["name"]
            if name in INCREMENTAL_TOOLS:
                if bound_names:
                    selected.append(tool)
            elif name in bound_names:
                bound_tool = deepcopy(tool)
                schema = bound_tool["input_schema"]
                schema["properties"]["target_uri"]["minLength"] = 1
                schema["dependentRequired"] = {
                    "target_uri": ["expected_version"], "expected_version": ["target_uri"],
                }
                selected.append(bound_tool)
            else:
                selected.append(source_only.get(name, tool))
        return selected

    def _generation_prompt(self, tools):
        manager = self._prompt_manager
        parts = self._prompt_parts
        variables = parts.template_variables if parts is not None else {}
        system = manager.render(self._template, "system_prompt", purpose=_GENERATION,
                                categories=self._assembly_categories, **variables)
        examples = ""
        if parts is None or parts.include_examples:
            examples = manager.render(self._template, "examples", purpose=_GENERATION,
                                      categories=self._assembly_categories)
        identity = manager.render(self._template, "identity_block", user_id=self._ctx.user_id)
        rules = manager.render(self._template, "lazy_tools_instruction", purpose=_GENERATION)
        output = (parts.output_instruction if parts is not None and parts.output_instruction
                  else manager.render(self._template, "output_instruction"))
        if self._text_mode:
            output += "\n" + manager.render(self._template, "final_content_instruction", purpose=_GENERATION)
            output += "\nAllowed operation schemas:\n" + json.dumps(tools, ensure_ascii=False)

        reads, operations, feedback = self._evidence()
        data = {
            "memory_index": self._index,
            "read_results": reads,
            "recorded_operations": operations,
            "tool_feedback": feedback,
        }
        categories = {tool["name"].removeprefix("extract_") for tool in tools
                      if tool["name"].startswith("extract_")}
        update_categories = {tool["name"].removeprefix("extract_") for tool in tools
                             if tool["name"].startswith("extract_")
                             and "target_uri" in tool["input_schema"].get("properties", {})}
        runtime_rules = [
            build_update_instruction(self._registry, update_categories),
            build_update_instruction(self._registry, categories - update_categories, include_updates=False),
        ]
        return PromptParts(
            system_prompt=system,
            prefix_data=f"\n\n{identity}\n\n{rules}\n\n",
            examples=examples,
            include_examples=bool(examples),
            suffix_data=f"{output}\n\n## Current evidence and progress\n"
                        f"{json.dumps(data, ensure_ascii=False)}\n\n## Current source\n{self._source}",
            runtime_instruction="\n\n".join(filter(None, runtime_rules)) + "\n\n",
            purpose=_GENERATION,
        ).render()

    def _form_memory(self):
        tools = self._generation_tools()
        prompt = self._generation_prompt(tools)
        correlation = {**self._correlation, "iteration": self._iteration, "purpose": _GENERATION}
        try:
            if self._text_mode:
                calls, content = run_model_call(
                    observe_call, "extraction", self._llm.complete_with_tools_messages,
                    messages=[{"role": "user", "content": prompt}], tools=None,
                    tool_choice="none", correlation=correlation,
                )
                if calls:
                    raise ToolCallProtocolError("Final extraction returned tool calls")
                check_deadline()
                return [], content
            calls = self._generate_operations(self._llm, prompt=prompt, tools=tools, correlation=correlation)
        except (LLMCallError, LLMOutputTruncatedError, LLMTransientError, OperationDeadlineExceededError,
                ToolCallProtocolError, UsagePersistenceError):
            raise
        except Exception as exc:
            raise LLMCallError("LLM call failed", cause=exc) from exc
        if not calls:
            raise ExtractionOutputError("Required tool call returned no memory operations", tool_choice="required")
        validate_offered_calls(calls, tools)
        return calls, None

    def _call_llm(self, messages):
        self._iteration += 1
        unread = [key for key, entry in self._index.items() if entry["uri"] not in self._read_nodes]
        can_observe = (
            bool(unread) and not self._text_mode and self._read_selection is None
            and any(tool["name"] == "read" for tool in self._get_tools())
        )
        action = self._evaluate(
            "next_step", self._decisions.next_question(can_observe), allow_observation=can_observe,
        )
        if action == _EXPLORE:
            probabilities, selected = self._evaluate(
                "read_candidates", self._decisions.read_questions(unread), allow_observation=True,
            )
            self._read_selection = {"probabilities": probabilities, "threshold": self._threshold,
                                    "selected": selected}
            if selected:
                return [{"tool": "read", "input": {"uri": self._index[key]["uri"]}} for key in selected], None
            action = self._evaluate("next_step", self._decisions.next_question(False), allow_observation=False)
        if action == _FINISH:
            self._finished_by_decision = True
            return [], "[]"
        return self._form_memory()

    def _should_continue_after_output(self, parsed, messages, ctx, iter_trace=None):
        if super()._should_continue_after_output(parsed, messages, ctx, iter_trace):
            self._read_selection = None
            return True
        if self._finished_by_decision:
            return False
        tentative = [
            {"category": candidate.category, "routing_key": candidate.routing_key,
             "target_uri": candidate.target_uri, "version": candidate.expected_version,
             "content": candidate.content, "evidence_quote": candidate.evidence_quote}
            for candidate in parsed
        ]
        action = self._evaluate(
            "next_step", self._decisions.next_question(False), allow_observation=False,
            tentative=tentative, final_check=True,
        )
        if action != _FINISH:
            raise ExtractionIncompleteError("Final extraction did not cover the complete source")
        return False
