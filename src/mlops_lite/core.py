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
        points = [m for m in self.metrics if m.key == metric_key]
        if not points:
            return None
        return max(points, key=lambda m: m.value) if higher_is_better else min(points, key=lambda m: m.value)


class RunStore:
    def __init__(self, storage_path: Path | None = None) -> None:
        self._runs: dict[str, Run] = {}
        self._path = storage_path
        if storage_path and storage_path.exists():
            self._load()

    def create(self, experiment: str, params: dict[str, Any]) -> Run:
        if not experiment.strip():
            raise MlopsError("experiment name required")
        run_id = uuid.uuid4().hex[:12]
        run = Run(run_id=run_id, experiment=experiment,
                  params=dict(params))
        self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> Run:
        run = self._runs.get(run_id)
        if run is None:
            raise ExperimentNotFoundError(run_id)
        return run

    def transition(self, run_id: str, target: Stage) -> Run:
        run = self.get(run_id)
        allowed = VALID_TRANSITIONS[run.stage]
        if target not in allowed:
            raise InvalidStageError(f"illegal transition {run.stage.value} → {target.value}")
        if target == Stage.RUNNING and run.started_at is None:
            run.started_at = time.time()
        if target in (Stage.FINISHED, Stage.FAILED):
            run.finished_at = time.time()
        run.stage = target
        return run

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> MetricPoint:
