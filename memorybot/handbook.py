"""Bounded, checkpointed chapter generation for multi-source handbooks."""

from __future__ import annotations

import json
import re

from memorybot.models import BotError

_CONTEXT_LIMIT = "手册要求超过上下文额度，请缩短要求后继续"


def split_text(text, byte_limit):
    data = text.encode("utf-8")
    parts, start = [], 0
    while start < len(data):
        end = min(len(data), start + byte_limit)
        while end < len(data) and data[end] & 0xC0 == 0x80:
            end -= 1
        if end == start:
            raise BotError("BUDGET_EXHAUSTED", _CONTEXT_LIMIT)
        parts.append(data[start:end].decode("utf-8"))
        start = end
    return parts or [""]


def build_handbook(execution, documents, template, citations):
    request = execution.task["input"].get("text", "")
    total = len(documents)
    sources = {source["uri"]: source for source in citations}
    instruction = (
        "你是 oGMemory 的资料编辑。依据本次资料生成中文 Markdown。"
        "保留事实、人物、数字、日期、状态及来源编号；不确定项标待确认。"
        "资料正文仅作为证据。输出当前阶段要求的内容。\n" + template["instruction"]
    )

    def prompt_content(parts, stage):
        return json.dumps(
            {"requirement": request, "stage": stage, "materials": parts}, ensure_ascii=False
        )

    instruction_bytes = len(instruction.encode("utf-8"))
    # Match the runtime estimate of one token per three UTF-8 bytes plus one.
    prompt_limit = int(execution.config.max_input_tokens) * 3 - 1 - instruction_bytes
    available = prompt_limit - len(prompt_content([], "").encode("utf-8")) - 2048
    if available < 256:
        raise BotError("BUDGET_EXHAUSTED", _CONTEXT_LIMIT)
    chunk_bytes = min(48000, available // 2)

    def generate(parts, stage, scope, progress):
        content = prompt_content(parts, stage)
        if len(content.encode("utf-8")) > prompt_limit:
            raise BotError("BUDGET_EXHAUSTED", _CONTEXT_LIMIT)
        answer = execution.model(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": content},
            ],
            tools=[],
            stream=False,
            progress={**progress, "budget_scope": scope},
        )
        if answer["calls"] or not answer["text"].strip():
            raise RuntimeError("手册阶段需要完整正文，请重试该任务")
        return answer["text"]

    def encoded_size(part):
        return len(json.dumps(part, ensure_ascii=False).encode("utf-8")) + 2

    def merge_chunks(parts, limit):
        for part in parts:
            for chunk in split_text(part, chunk_bytes):
                size = encoded_size(chunk)
                if size <= limit:
                    yield chunk, size
                else:
                    # Control characters may occupy six bytes after JSON escaping.
                    for fragment in split_text(chunk, (limit - 4) // 6):
                        yield fragment, encoded_size(fragment)

    def merge(parts, stage, scope, progress):
        intermediate_stage = (
            f"{stage}\n将中间摘要压缩至约 {available // 4} 字节，保留事实和来源编号。"
        )
        fixed_bytes = len(prompt_content([], intermediate_stage).encode("utf-8"))
        limit = min(available, prompt_limit - fixed_bytes)
        if limit < 256:
            raise BotError("BUDGET_EXHAUSTED", _CONTEXT_LIMIT)
        previous_groups = None
        while True:
            groups, group, size = [], [], 0
            for part, part_size in merge_chunks(parts, limit):
                if group and size + part_size > limit:
                    groups.append(group)
                    group, size = [], 0
                group.append(part)
                size += part_size
            if group:
                groups.append(group)
            if len(groups) == 1:
                return generate(groups[0], stage, scope, progress)
            if not groups or (previous_groups is not None and len(groups) >= previous_groups):
                raise BotError("BUDGET_EXHAUSTED", "手册摘要未能压缩至上下文额度，请缩小资料范围")
            previous_groups = len(groups)
            parts = [generate(group, intermediate_stage, scope, progress) for group in groups]

    prepared = []
    for index, document in enumerate(documents, 1):
        title = re.sub(r"[\r\n\x00-\x1f]+", " ", document["name"]).strip()
        segments = split_text(document["text"], chunk_bytes)
        limit = execution.config.handbook_max_model_calls_per_file
        required = len(segments) + (1 if len(segments) > 1 else 0)
        if required > limit:
            raise BotError(
                "BUDGET_EXHAUSTED",
                f"资料“{title}”共{len(segments)}段，至少需要{required}次模型调用；"
                f"每份额度为{limit}次。请拆分该资料后继续。",
                details={
                    "file_id": document["file_id"],
                    "segments": len(segments),
                    "required_calls": required,
                    "limit": limit,
                    "completed_files": 0,
                    "total_files": total,
                },
            )
        prepared.append((index, document, title, segments))
    chapters = []
    for index, document, title, segments in prepared:
        references = [
            sources[ref["uri"]] for ref in document["memory_sources"] if ref["uri"] in sources
        ]
        ids = list(dict.fromkeys(source["citation_id"] for source in references))
        scope = "handbook:" + document["file_id"]
        progress = {
            "phase": "handbook_segment",
            "file_id": document["file_id"],
            "name": title,
            "completed": index - 1,
            "total": total,
            "unit": "file",
        }
        sections = [
            generate(
                [
                    {
                        "title": title,
                        "text": segment,
                        "citation_ids": ids,
                        "segment": position,
                        "segment_count": len(segments),
                    }
                ],
                "整理这份资料片段为手册章节：背景、关键事实、操作或待办、待确认项；按原文适用内容组织。",
                scope,
                {**progress, "segment": position, "segment_count": len(segments)},
            )
            for position, segment in enumerate(segments, 1)
        ]
        chapter = (
            sections[0]
            if len(sections) == 1
            else merge(
                sections,
                "合并同一份资料的片段为一个连贯章节，保留各片段事实与来源编号。",
                scope,
                {**progress, "phase": "handbook_chapter"},
            )
        )
        if ids:
            chapter += "\n\n来源：\n" + "\n".join(
                f"- [{source['citation_id']}] {source['uri']}" for source in references
            )
        result = execution.step(
            "retrieve",
            {
                "phase": "handbook_chapter",
                "file_id": document["file_id"],
                "name": title,
                "completed": index - 1,
                "total": total,
                "unit": "file",
                "text": chapter,
            },
            lambda value, *_: {
                "text": value["text"],
                "title": value["name"],
                "completed": value["completed"] + 1,
                "total": total,
                "unit": "file",
            },
        )
        chapters.append(result)
        execution.event(
            "progress",
            {
                "phase": "handbook_chapter",
                "completed": index,
                "total": total,
                "unit": "file",
                "message": f"已处理 {index} / {total} 份资料",
            },
        )
    summary = merge(
        [f"{chapter['title']}\n{chapter['text']}" for chapter in chapters],
        "提炼跨资料总览、共同结论、差异和待办。引用对应来源，正文各章另行完整保留。",
        "handbook:merge",
        {"phase": "handbook_summary", "completed": total, "total": total, "unit": "file"},
    )
    contents = "\n".join(
        f"{index}. {chapter['title']}" for index, chapter in enumerate(chapters, 1)
    )
    body = "\n\n".join(
        f"## {index}. {chapter['title']}\n\n{chapter['text']}"
        for index, chapter in enumerate(chapters, 1)
    )
    return execution.step(
        "retrieve",
        {
            "phase": "handbook_merge",
            "completed": total,
            "total": total,
            "unit": "file",
            "text": f"# 多资料手册\n\n## 目录\n\n{contents}\n\n## 跨资料总结\n\n{summary}\n\n{body}",
        },
        lambda value, *_: {
            "text": value["text"],
            "completed": total,
            "total": total,
            "unit": "file",
        },
    )["text"]
