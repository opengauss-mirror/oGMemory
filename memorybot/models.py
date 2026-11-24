"""Public request contracts and response projections."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema


class BotError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, details: dict | None = None):
        super().__init__(message)
        self.code, self.status = code, status
        self.details = details or {}

    def payload(self) -> dict:
        return {"code": self.code, "message": str(self), "details": self.details}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


Identifier = Annotated[str, Field(min_length=1, max_length=64)]
URI = Annotated[str, Field(min_length=1, max_length=2048)]


class Scope(Contract):
    memory_scope: Literal["user", "authorized"] = "user"
    resource_ids: list[URI] = Field(default_factory=list, max_length=100)

    @field_validator("resource_ids")
    @classmethod
    def normalize_resources(cls, value):
        return sorted(set(value))


class Delivery(Contract):
    channel: Literal["web"] = "web"
    binding_id: SkipJsonSchema[None] = Field(default=None, exclude=True, repr=False)
    target_id: Annotated[str, Field(min_length=1, max_length=256)] | None = None


class ChatInput(Contract):
    text: str = ""
    file_ids: list[Identifier] = Field(default_factory=list, max_length=20)
    template_id: Identifier | None = None
    output_format: Literal["markdown", "csv"] = "markdown"
    delivery: Delivery = Field(default_factory=Delivery)

    @model_validator(mode="after")
    def check_input(self):
        if len(self.text.encode("utf-8")) > 32768:
            raise ValueError("text must be at most 32768 UTF-8 bytes")
        if not (self.text or self.file_ids or self.template_id):
            raise ValueError("provide text, file_ids or template_id")
        self.file_ids = list(dict.fromkeys(self.file_ids))
        return self


class SessionCreate(Contract):
    title: str = Field(min_length=1, max_length=128)
    scope: Scope = Field(default_factory=Scope)
    agent_id: str = Field(default="memorybot", min_length=1, max_length=128)


class SessionUpdate(Contract):
    expected_revision: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=128)
    archived: bool | None = None

    @field_validator("title")
    @classmethod
    def trim_title(cls, value):
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("title must contain text")
        return value

    @model_validator(mode="after")
    def has_change(self):
        if self.title is None and self.archived is None:
            raise ValueError("provide title or archived")
        return self


class FileCreate(Contract):
    source_kind: Literal["url"]
    source_url: str = Field(min_length=1, max_length=2048)
    scope: Scope = Field(default_factory=Scope)


class SourceRef(Contract):
    file_id: Identifier | None = None
    task_id: Identifier | None = None
    uri: URI | None = None
    quote: str = Field(default="", max_length=1024)

    @model_validator(mode="after")
    def one_source(self):
        if sum(v is not None for v in (self.file_id, self.task_id, self.uri)) != 1:
            raise ValueError("source requires exactly one file_id, task_id or uri")
        return self


class MemoryAction(Contract):
    action: Literal["correct", "forget", "remember"]
    memory_uri: URI | None = None
    new_text: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list, max_length=100)
    fact_scope: dict = Field(default_factory=lambda: {"mode": "entry"})

    @model_validator(mode="after")
    def check_action(self):
        if self.action in {"correct", "forget"} and not self.memory_uri:
            raise ValueError("memory_uri is required")
        if self.action in {"correct", "remember"} and not self.new_text:
            raise ValueError("new_text is required")
        if self.new_text and len(self.new_text.encode("utf-8")) > 32768:
            raise ValueError("new_text must be at most 32768 UTF-8 bytes")
        if self.fact_scope != {"mode": "entry"}:
            raise ValueError("fact_scope must be {mode: entry}")
        return self


class OnceRule(Contract):
    type: Literal["once"]
    at: datetime

    @field_validator("at")
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None:
            raise ValueError("at must contain a timezone")
        return value


class IntervalRule(Contract):
    type: Literal["interval"]
    interval_seconds: int = Field(ge=60, le=31536000, strict=True)


class WeeklyRule(Contract):
    type: Literal["weekly"]
    weekdays: list[Annotated[int, Field(ge=1, le=7, strict=True)]] = Field(
        min_length=1, max_length=7
    )
    local_time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d$")

    @field_validator("weekdays")
    @classmethod
    def unique_days(cls, value):
        return sorted(set(value))


Rule = Annotated[OnceRule | IntervalRule | WeeklyRule, Field(discriminator="type")]


class ScheduleCreate(Contract):
    session_id: Identifier
    name: str = Field(min_length=1, max_length=128)
    task_input: ChatInput
    rule: Rule
    timezone: str

    @field_validator("timezone")
    @classmethod
    def known_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (KeyError, ValueError) as exc:
            raise ValueError("timezone must be a valid IANA name") from exc
        return value


class SchedulePatch(Contract):
    expected_revision: int = Field(ge=1, strict=True)
    action: Literal["update", "pause", "resume", "cancel"]
    name: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    task_input: ChatInput | None = None
    rule: Rule | None = None
    timezone: str | None = None

    @model_validator(mode="after")
    def check_changes(self):
        changes = self.model_fields_set - {"expected_revision", "action"}
        if self.action == "update" and not changes:
            raise ValueError("update requires at least one changed field")
        if self.action != "update" and changes:
            raise ValueError("fields can only be changed by update")
        if self.timezone is not None:
            ScheduleCreate.known_timezone(self.timezone)
        if any(getattr(self, key) is None for key in changes):
            raise ValueError("changed fields must have values")
        return self


class Confirmation(Contract):
    operation_id: Identifier
    decision: Literal["recheck", "cancel"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def jsonable(value):
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


PUBLIC_FIELDS = {
    "session": "session_id title scope agent_id archived session_generation next_seq revision created_at updated_at",
    "task": "task_id request_id kind session_id parent_task_id seq input state result memory_status memory_operation_id schedule_id scheduled_at cancel_requested error created_at updated_at finished_at",
    "file": "file_id purpose source_kind task_id previous_file_id name mime_type size_bytes sha256 source_url scope state metadata created_at updated_at",
    "schedule": "schedule_id session_id name task_input rule timezone state next_run_at revision created_at updated_at",
    "delivery": "delivery_id task_id source_task_id channel target_id state provider_receipt created_at updated_at",
}


def public(kind: str, row: dict) -> dict:
    result = {key: row.get(key) for key in PUBLIC_FIELDS[kind].split()}
    if kind == "task":
        result["input"] = row.get("input") if row["kind"] == "chat" else None
        result["last_event_seq"] = row.get("next_event_seq", 1) - 1
        result["event_floor_seq"] = row.get("event_floor_seq", 1)
        result["confirmation"] = row.get("confirmation")
        result["pending_memory_confirmations"] = row.get("pending_memory_confirmations", [])
        result["steps"] = row.get("steps", [])
    elif kind == "schedule":
        result["delivery"] = row["task_input"].get("delivery", {})
    if kind in {"session", "file"}:
        scope = row.get("scope") or {}
        result["scope"] = {
            "memory_scope": scope.get("memory_scope", "user"),
            "resource_ids": scope.get("resource_ids", []),
        }
    return dict(jsonable(result))
