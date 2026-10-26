"""Typed inputs, outputs and recovery diagnostics for scene decisions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from scene.models import SceneSnapshot


class SceneEvidenceReader(Protocol):
    """Bounded read-only tools exposed to the preflight model."""

    def search_related_nodes(
        self, account_id: str, owner_space: str, query: str, limit: int
    ) -> Sequence[dict[str, Any]]:
        """Return bounded memory cards with their read-time node_fingerprint."""
        ...

    def search_scene_cards(
        self, account_id: str, owner_space: str, query: str, limit: int
    ) -> Sequence[dict[str, Any]]: ...

    def read_source_context(
        self, account_id: str, owner_space: str, source_refs: Sequence[str], limit: int
    ) -> Sequence[dict[str, Any]]:
        """Return bounded cards; memory cards include the read-time node_fingerprint."""
        ...


@dataclass(frozen=True)
class DecisionLimits:
    max_input_nodes: int = 128
    max_prompt_chars: int = 64_000
    max_node_chars: int = 800
    max_description_chars: int = 2_000

    def validate(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class PreflightOutput(_Strict):
    status: Literal["READY", "WAIT"]
    anchors: list[str] = Field(default_factory=list, max_length=4)
    evidence_uris: list[str] = Field(default_factory=list)
    candidate_scene_ids: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    reason: str = Field(default="", max_length=1_000)


_SCENE_TIME_DESCRIPTION = (
    "Full YYYY-MM-DD date or ISO 8601 timestamp grounded in the evidence. "
    "For year/month-only or uncertain dates use null and preserve the stated "
    "precision in boundary.time_scope."
)


class SceneDraft(_Strict):
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=2_000)
    member_uris: list[str] = Field(min_length=2)
    boundary: dict[str, Any] = Field(default_factory=dict)
    time_start: str | None = Field(default=None, description=_SCENE_TIME_DESCRIPTION)
    time_end: str | None = Field(default=None, description=_SCENE_TIME_DESCRIPTION)


class SceneDecisionOutput(_Strict):
    action: Literal["CREATE", "UPDATE", "MERGE", "WAIT"]
    target_scene_ids: list[str] = Field(default_factory=list)
    draft: SceneDraft | None = None
    evidence_uris: list[str] = Field(
        default_factory=list,
        description="Memory URIs from draft.member_uris supporting the scene; source_refs identify original sources.",
    )
    unassigned_member_uris: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    reason: str = Field(default="", max_length=1_000)


_QUALITY_REASON_MAX_CHARS = 1_000


class QualityGateOutput(_Strict):
    decision: Literal["ACCEPT", "ADJUST", "DEFER"]
    title: str | None = Field(default=None, max_length=120)
    description: str | None = Field(default=None, max_length=2_000)
    reason: str = Field(default="", max_length=_QUALITY_REASON_MAX_CHARS)

    @field_validator("reason", mode="before")
    @classmethod
    def bound_diagnostic_reason(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()[:_QUALITY_REASON_MAX_CHARS]
        return value


@dataclass(frozen=True)
class SceneDecisionResult:
    snapshot: SceneSnapshot
    action: str
    pending_member_uris: tuple[str, ...] = ()
    calls: int = 0
    reason: str = ""


class SceneDecisionError(ValueError):
    """Raised when a model proposal cannot satisfy the scene contract."""


class SceneRetryableDecisionError(SceneDecisionError):
    """Safe stage diagnostics for bounded task recovery."""

    code = "scene_decision_retryable"

    def __init__(self, stage: str, errors: Sequence[dict[str, str]]) -> None:
        self.stage = stage
        self.errors = tuple(errors)
        super().__init__(json.dumps({
            "code": self.code, "stage": stage, "errors": self.errors,
        }, separators=(",", ":"), sort_keys=True))


class SceneModelOutputError(SceneRetryableDecisionError):
    code = "scene_model_output_invalid"


class SceneCallBudgetError(SceneRetryableDecisionError):
    code = "scene_call_budget_exhausted"

    def __init__(self, stage: str) -> None:
        super().__init__(stage, ({"field": "max_llm_calls_per_batch", "type": "budget_exhausted"},))
