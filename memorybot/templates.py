"""Published task templates with bounded, deterministic CSV tools."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from decimal import Context, Decimal, Inexact, InvalidOperation

from core.errors import AccessDeniedError, NodeBrokenError, NodeNotFoundError
from memorybot.models import BotError
from retrieval.owner_scope import canonical_owner_space

_READ_TOOLS = frozenset({"search_memory", "read_memory"})
_PERMISSION_ACTIONS = frozenset({"memory.read", "resource.read", "file.write", "tool.execute"})
_CSV_MAX_ROWS = 200000
_CSV_MAX_DIGITS = 128
_CSV_MAX_OUTPUT_BYTES = 1024 * 1024

_DEFINITIONS = {
    "weekly_report": (
        "周报",
        "依据来源整理本周完成事项、待办、风险和下周计划，保留事实、负责人和日期。",
        0,
        ["markdown"],
    ),
    "handover": (
        "交接稿",
        "整理背景、已完成工作、待办、负责人、风险和资料入口，注明未知项。",
        0,
        ["markdown"],
    ),
    "handbook": (
        "多资料手册",
        "为每份来源保留一个有标题的章节，再给出目录与跨资料总结。逐份保留来源。",
        1,
        ["markdown"],
    ),
    "release_check": (
        "发布检查",
        "按提供的发布清单逐项核对资料。每项给出完成、待办或待确认及来源依据。",
        1,
        ["markdown", "csv"],
    ),
    "csv_summary": (
        "CSV 分组统计",
        "按第一列分组，对第二列数值精确求和，并输出总计。",
        1,
        ["csv", "markdown"],
    ),
}


class TemplateRegistry:
    def __init__(self, published=None):
        self._templates = {}
        for template_id, (name, instruction, minimum, formats) in _DEFINITIONS.items():
            self._templates[template_id] = {
                "id": template_id,
                "name": name,
                "instruction": instruction,
                "input": {
                    "text_required": False,
                    "file_min": minimum,
                    "file_max": 20,
                    "accepted_mime_types": (
                        ["text/csv"]
                        if template_id == "csv_summary"
                        else ["text/plain", "text/markdown", "text/csv"]
                    ),
                },
                "output": {
                    "formats": formats,
                    "mime_types": ["text/" + value for value in formats],
                },
                "permission_requirements": [
                    {"action": "memory.read", "scope": "user"},
                    {"action": "resource.read", "scope": "user"},
                    {"action": "file.write", "scope": "user"},
                ],
                "tools": ["search_memory", "read_memory"],
            }
        for template in published or []:
            value = dict(template)
            value["tools"] = list(value.get("tools", []))
            if value.get("executor", "model") not in {"model", "csv_summary"}:
                raise ValueError("模板执行器须为 model 或 csv_summary")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value["id"]):
                raise ValueError("模板标识无效")
            if not set(value.get("tools", [])) <= _READ_TOOLS:
                raise ValueError("模板工具须为已发布的只读工具")
            spec = value["input"]
            if not 0 <= spec["file_min"] <= spec["file_max"] <= 20:
                raise ValueError("模板文件数量须在 0 至 20 之间")
            if not set(spec["accepted_mime_types"]) <= {
                "text/plain",
                "text/markdown",
                "text/csv",
            }:
                raise ValueError("模板资料类型无效")
            if not value["output"]["formats"] or not set(value["output"]["formats"]) <= {
                "markdown",
                "csv",
            }:
                raise ValueError("模板输出格式无效")
            if any(
                row.get("action") not in _PERMISSION_ACTIONS
                or not isinstance(row.get("scope"), str)
                or not 1 <= len(row["scope"]) <= 2048
                for row in value["permission_requirements"]
            ):
                raise ValueError("模板权限声明无效")
            self._templates[value["id"]] = value

    @classmethod
    def from_service(cls, service, ctx):
        import yaml

        published, cursor = [], None
        while True:
            page = service.list_published_skills(ctx, limit=100, cursor=cursor)
            for skill in page["skills"]:
                node = service.read_published_skill(ctx, skill["name"], source=skill["source"])
                declaration = node.metadata.get("memorybot_template")
                instruction = node.content
                lines = node.content.splitlines()
                if lines and lines[0].strip() == "---":
                    end = next(
                        (index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"),
                        None,
                    )
                    if end is not None:
                        frontmatter = yaml.safe_load("\n".join(lines[1:end]))
                        if isinstance(frontmatter, dict):
                            declaration = frontmatter.get("memorybot_template") or declaration
                            instruction = "\n".join(lines[end + 1 :]).strip()
                if declaration:
                    published.append(
                        {
                            **declaration,
                            "instruction": instruction,
                            "source_uri": node.uri,
                            "source_version": node.metadata.get("version", 1),
                        }
                    )
            cursor = page.get("next_cursor")
            if not cursor:
                return cls(published)

    def list(self):
        return [
            {key: value for key, value in row.items() if key not in {"instruction", "executor"}}
            for row in self._templates.values()
        ]

    def visible(self, ctx, scope=None, resource_reader=None):
        rows = []
        for template in self._templates.values():
            try:
                self.authorize(template, ctx, scope, resource_reader)
            except BotError as exc:
                if exc.code != "PERMISSION_DENIED":
                    raise
                continue
            rows.append(
                {
                    key: template[key]
                    for key in ("id", "name", "input", "output", "permission_requirements")
                }
            )
        return rows

    @staticmethod
    def execution_scope(template, ctx, scope=None):
        """Intersect declared action ranges with native grants and the session range."""
        own = ctx.user_space_name()
        owners = {
            canonical_owner_space(value)
            for value in (
                ctx.visible_owner_spaces
                or (own, *([ctx.agent_space_name()] if ctx.agent_id else []))
            )
        }
        effective = dict(
            scope if scope is not None else {"memory_scope": "authorized", "resource_ids": []}
        )
        grouped = {}
        for requirement in template.get("permission_requirements", []):
            grouped.setdefault(requirement["action"], set()).add(requirement["scope"])

        def selected_owners(action):
            available = owners
            if (
                action == "memory.read"
                and scope is not None
                and scope.get("memory_scope", "user") == "user"
            ):
                available = owners & {own}
            allowed = set()
            for required in grouped.get(action, ()):
                if required == "authorized":
                    allowed.update(available)
                elif required == "user":
                    allowed.add(own)
                elif required.startswith(("user:", "agent:", "user_", "agent_")):
                    allowed.add(canonical_owner_space(required))
            return sorted(allowed & available)

        effective["memory_owners"] = selected_owners("memory.read")
        effective["resource_owners"] = selected_owners("resource.read")
        effective["resource_uris"] = sorted(
            required
            for required in grouped.get("resource.read", ())
            if required.startswith(f"ctx://{ctx.account_id}/")
            and (scope is None or required in scope.get("resource_ids", []))
        )
        tools = set(template.get("tools", [])) & _READ_TOOLS
        if "tool.execute" in grouped:
            tools &= grouped["tool.execute"]
        effective["tool_names"] = sorted(tools)
        return effective

    @staticmethod
    def authorize(template, ctx, scope=None, resource_reader=None):
        """Check current access to the effective declared ranges and skill source."""
        own = ctx.user_space_name()
        effective = TemplateRegistry.execution_scope(template, ctx, scope)
        native = TemplateRegistry.execution_scope(
            {"permission_requirements": [{"action": "memory.read", "scope": "authorized"}]},
            ctx,
        )["memory_owners"]

        def require(allowed, message):
            if not allowed:
                raise BotError("PERMISSION_DENIED", message, 403)

        def read(uri):
            require(resource_reader is not None, "模板需要校验当前资源访问权")
            try:
                return resource_reader(uri)
            except (AccessDeniedError, NodeNotFoundError, NodeBrokenError, PermissionError) as exc:
                raise BotError("PERMISSION_DENIED", "模板来源或资源当前不可访问", 403) from exc

        def resource_owner(uri, value):
            owner = (
                value.get("owner_space")
                if isinstance(value, dict)
                else getattr(value, "owner_space", None)
            )
            if owner:
                return canonical_owner_space(owner)
            for kind in ("user", "agent"):
                prefix = f"ctx://{ctx.account_id}/{kind}s/"
                if uri.startswith(prefix):
                    return kind + ":" + uri[len(prefix) :].split("/", 1)[0]
            return None

        if template.get("source_uri"):
            read(template["source_uri"])
        tools = set(template.get("tools", []))
        require(tools <= _READ_TOOLS, "模板工具需要已注册的执行能力")
        grouped = {}
        for requirement in template.get("permission_requirements", []):
            grouped.setdefault(requirement["action"], set()).add(requirement["scope"])
        for action, required in grouped.items():
            if action == "memory.read":
                require(effective["memory_owners"], "模板记忆范围需要当前身份和会话授权")
            elif action == "resource.read":
                readable = bool(effective["resource_owners"])
                for uri in effective["resource_uris"]:
                    value = read(uri)
                    readable |= resource_owner(uri, value) in native
                require(readable, "模板资料范围需要当前身份和会话授权")
            elif action == "file.write":
                require(
                    bool(required & {"user", own}) and own in native, "模板成果写入当前用户空间"
                )
            elif action == "tool.execute":
                require(effective["tool_names"], "模板需要声明对应的已注册工具")
            else:
                raise BotError("PERMISSION_DENIED", "模板需要已支持的权限声明", 403)
        return template

    def get(self, template_id):
        try:
            value = dict(self._templates[template_id])
        except KeyError:
            raise ValueError("模板尚未发布") from None
        value["revision"] = hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        return value

    def validate(
        self,
        template_id,
        text,
        documents,
        output_format,
        *,
        ctx=None,
        scope=None,
        resource_reader=None,
    ):
        template = self.get(template_id)
        if ctx is not None:
            self.authorize(template, ctx, scope, resource_reader)
        spec = template["input"]
        if not spec["file_min"] <= len(documents) <= spec["file_max"]:
            raise ValueError("资料数量超出模板范围")
        if spec["text_required"] and not text.strip():
            raise ValueError("模板需要文字要求")
        if output_format not in template["output"]["formats"]:
            raise ValueError("输出格式超出模板范围")
        if any(doc["mime_type"] not in spec["accepted_mime_types"] for doc in documents):
            raise ValueError("资料类型超出模板范围")
        return template

    def execute(self, template_id, text, documents):
        return self.execute_frozen(self.get(template_id), documents)

    @staticmethod
    def execute_frozen(template, documents):
        if template.get("executor", template.get("id")) == "csv_summary":
            return summarize_csv(documents)
        return None


def _safe_cell(value):
    value = str(value)
    return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def summarize_csv(documents):
    zero = Decimal(0)
    # Allow the full integer and fractional spans plus carries from every input row.
    arithmetic = Context(prec=2 * _CSV_MAX_DIGITS + len(str(_CSV_MAX_ROWS)), traps=[Inexact])
    totals = {}
    group_name, value_name = "分组", "合计"
    count = 0
    total = zero
    for document in documents:
        reader = csv.reader(io.StringIO(document["text"]))
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError("CSV 至少需要分组列和数值列")
        group_name, value_name = header[:2]
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            count += 1
            if count > _CSV_MAX_ROWS:
                raise ValueError(f"CSV 行数超过 {_CSV_MAX_ROWS}")
            if len(row) != len(header):
                raise ValueError(f"CSV 第 {reader.line_num} 行列数不一致")
            try:
                amount = Decimal(row[1].strip())
            except InvalidOperation:
                raise ValueError(f"CSV 第 {reader.line_num} 行数值格式无效") from None
            if not amount.is_finite():
                raise ValueError(f"CSV 第 {reader.line_num} 行须为有限数值")
            _, digits, exponent = amount.as_tuple()
            if (
                len(digits) > _CSV_MAX_DIGITS
                or not -_CSV_MAX_DIGITS <= exponent <= _CSV_MAX_DIGITS
                or amount.adjusted() >= _CSV_MAX_DIGITS
            ):
                raise ValueError(f"CSV 第 {reader.line_num} 行数值位数或指数超出支持范围")
            key = row[0].strip()
            totals[key] = arithmetic.add(totals.get(key, zero), amount)
            total = arithmetic.add(total, amount)
    output = io.StringIO(newline="")
    line = io.StringIO(newline="")
    writer = csv.writer(line, lineterminator="\n")
    remaining_bytes = _CSV_MAX_OUTPUT_BYTES

    def append_row(cells):
        nonlocal remaining_bytes
        writer.writerow(cells)
        rendered = line.getvalue()
        remaining_bytes -= len(rendered.encode("utf-8"))
        if remaining_bytes < 0:
            raise ValueError("CSV 汇总输出超过 1 MiB")
        output.write(rendered)
        line.seek(0)
        line.truncate()

    append_row([_safe_cell(group_name), _safe_cell(value_name)])
    for key in sorted(totals):
        append_row([_safe_cell(key), format(totals[key], "f")])
    append_row(["总计", format(total, "f")])
    body = output.getvalue()
    return {
        "text": f"已处理 {count} 行，{len(totals)} 个分组，总计 {total}。\n\n```csv\n{body}```",
        "csv": body,
        "rows": count,
        "total": str(total),
    }
