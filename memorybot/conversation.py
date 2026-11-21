"""Conversation orchestration and model-tool turns."""

from __future__ import annotations

import json

from memorybot.actions import action_tools, list_schedules, managed_tool
from memorybot.conversation_context import ConversationContext
from memorybot.conversation_result import ConversationResult
from memorybot.gateway import READ_TOOLS
from memorybot.models import BotError
from memorybot.prompts import conversation_instruction
from memorybot.templates import TemplateRegistry


class ChatRun:
    def __init__(self, execution):
        self.execution = execution
        self.context = ConversationContext(execution)

    def run(self):
        self.context.load()
        text, csv_body, memory_action_used, stream_result = self.generate()
        ConversationResult(self.execution, self.context).publish(
            text, csv_body, memory_action_used, stream_result
        )

    def generate(self):
        execution = self.execution
        deterministic = execution.step(
            (
                "tool"
                if self.context.template.get("executor", self.context.template.get("id"))
                == "csv_summary"
                else "retrieve"
            ),
            {"template": self.context.template},
            lambda value, *_: {
                "value": TemplateRegistry.execute_frozen(value["template"], self.context.documents)
            },
        )["value"]
        # Scope management tools to the user's current direct request.
        managed_actions, managed_tools = (
            action_tools(self.context.data.get("text", ""))
            if not self.context.template_id and (not execution.task.get("schedule_id"))
            else ({}, [])
        )
        memory_action_used = False
        handbook = self.context.template_id == "handbook"
        if handbook:
            from memorybot.handbook import build_handbook

            text, csv_body = (
                build_handbook(
                    execution,
                    self.context.documents,
                    self.context.template,
                    self.context.document_citations,
                ),
                None,
            )
        elif deterministic:
            text, csv_body = (deterministic["text"], deterministic.get("csv"))
            if self.context.document_citations:
                text += "\n\n来源：" + " ".join(
                    dict.fromkeys(
                        f"[{row['citation_id']}]" for row in self.context.document_citations
                    )
                )
        else:
            text, csv_body, memory_action_used = self.generate_with_tools(
                managed_actions, managed_tools
            )
        return (text, csv_body, memory_action_used, bool(deterministic or handbook))

    def build_prompt(self, managed_actions, managed_tools):
        execution = self.execution
        instruction = conversation_instruction(
            execution.config,
            managed_tools=managed_tools,
            template=self.context.template,
            output_format=self.context.data.get("output_format"),
        )
        materials = [{"title": row["name"], "text": row["text"]} for row in self.context.documents]
        prompt = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": "以下是本轮可参考的资料与来源：\n"
                + json.dumps(
                    {"sources": self.context.sources, "documents": materials}, ensure_ascii=False
                ),
            },
            *self.context.retrieved["messages"],
        ]
        if "update_schedule" in managed_actions:
            schedule_snapshot = execution.step(
                "retrieve",
                {"session_id": self.context.session["session_id"], "object": "schedules"},
                lambda value, *_: {
                    "schedules": [
                        {
                            key: row[key]
                            for key in (
                                "schedule_id",
                                "name",
                                "rule",
                                "timezone",
                                "state",
                                "revision",
                            )
                        }
                        for row in execution.rows(
                            "bot_schedule", {"session_id": value["session_id"]}, limit=50
                        )
                    ]
                },
            )
            prompt.insert(
                -1,
                {
                    "role": "user",
                    "content": "已有计划：" + json.dumps(schedule_snapshot, ensure_ascii=False),
                },
            )
        return prompt

    def generate_with_tools(self, managed_actions, managed_tools):
        execution = self.execution
        prompt = self.build_prompt(managed_actions, managed_tools)
        memory_action_used = False
        if execution.task["attempt"] > 1:
            execution.event(
                "answer.reset",
                {
                    "attempt": execution.task["attempt"],
                    "replaced_attempt": execution.task["attempt"] - 1,
                },
            )
        while True:
            allowed_tools = (
                self.context.execution_scope["tool_names"]
                if self.context.template_id
                else ["search_memory", "read_memory", "list_schedules"]
            )
            model_result = execution.model(
                prompt,
                tools=[tool for tool in READ_TOOLS if tool["name"] in allowed_tools]
                + managed_tools,
            )
            if not model_result["calls"]:
                text, csv_body = (model_result["text"], None)
                break
            for index, call in enumerate(model_result["calls"]):
                call["id"] = (
                    call.get("id") or f"call_{execution.task['task_id']}_{execution.number}_{index}"
                )
            # Keep the full call list so each tool reply resolves its tool_call_id.
            prompt.append(
                {
                    "role": "assistant",
                    "content": model_result["text"] or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["tool"],
                                "arguments": json.dumps(call["input"], ensure_ascii=False),
                            },
                        }
                        for call in model_result["calls"]
                    ],
                }
            )
            for call in model_result["calls"]:
                if call["tool"] not in allowed_tools and call["tool"] not in managed_actions:
                    raise ValueError("工具超出模板权限")
                execution.tool_count += 1
                if execution.tool_count > execution.config.max_tool_calls:
                    raise BotError("BUDGET_EXHAUSTED", "达到工具调用次数上限")
                if call["tool"] in managed_actions:
                    tool_result = execution.step(
                        "tool",
                        call,
                        lambda value, op, waiting, tx: managed_tool(
                            execution, value, op, waiting, tx, managed_actions
                        ),
                        atomic=True,
                    )
                    memory_action_used = memory_action_used or call["tool"] == "memory_action"
                else:
                    tool_result = execution.step(
                        "tool",
                        call,
                        lambda value, *_: (
                            list_schedules(execution)
                            if value["tool"] == "list_schedules"
                            else execution.gateway.tool(
                                execution.task,
                                value["tool"],
                                value["input"],
                                self.context.execution_scope,
                                self.context.excluded,
                            )
                        ),
                    )
                if "sources" in tool_result:
                    tool_result = {
                        **tool_result,
                        "sources": self.context.register_sources(tool_result["sources"]),
                    }
                prompt.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
        return (text, csv_body, memory_action_used)
