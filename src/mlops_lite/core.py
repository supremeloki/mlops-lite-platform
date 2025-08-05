from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence


class MlopsError(Exception):
    pass


class ExperimentNotFoundError(MlopsError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"run not found: {run_id!r}")


class InvalidStageError(MlopsError):
    pass


class Stage(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


VALID_TRANSITIONS = {
    Stage.CREATED: {Stage.RUNNING, Stage.FAILED},
    Stage.RUNNING: {Stage.FINISHED, Stage.FAILED},
    Stage.FINISHED: set(),
    Stage.FAILED: {Stage.RUNNING},
}


@dataclass(frozen=True)
class MetricPoint:
    key: str
    value: float
    step: int
    recorded_at: float

    def __str__(self) -> str:
        return f"{self.key}@{self.step}={self.value:.6f}"


@dataclass
class Run:
    run_id: str
    experiment: str
    stage: Stage = Stage.CREATED
    params: dict[str, Any] = field(default_factory=dict)
    metrics: list[MetricPoint] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return round(self.finished_at - self.started_at, 3)

    def best(self, metric_key: str, higher_is_better: bool = True) -> MetricPoint | None:
