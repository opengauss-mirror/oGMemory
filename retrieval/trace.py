"""Lightweight structured trace for the retrieval pipeline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class StageTrace:
    stage: str
    elapsed_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "input_count": self.input_count,
            "output_count": self.output_count,
            "warnings": list(self.warnings),
            **({"metrics": self.metrics} if self.metrics else {}),
        }


@dataclass
class RetrievalTrace:
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    stages: list[StageTrace] = field(default_factory=list)
    total_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    level_histogram: dict[str, int] = field(default_factory=dict)
    graph_expansion: dict = field(default_factory=dict)

    def add_stage(self, stage: StageTrace) -> None:
        self.stages.append(stage)
        self.warnings.extend(stage.warnings)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "stages": [s.to_dict() for s in self.stages],
            "total_ms": round(self.total_ms, 2),
            "warnings": list(self.warnings),
            "level_histogram": self.level_histogram,
            **({"graph_expansion": self.graph_expansion} if self.graph_expansion else {}),
        }


class TraceTimer:
    def __init__(self, stage_name: str) -> None:
        self.trace = StageTrace(stage=stage_name)
        self._t0 = 0.0

    def __enter__(self) -> StageTrace:
        self._t0 = time.monotonic()
        return self.trace

    def __exit__(self, *_exc: object) -> None:
        self.trace.elapsed_ms = (time.monotonic() - self._t0) * 1000
