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
        point = MetricPoint(key=key, value=value, step=step, recorded_at=time.time())
        self.get(run_id).metrics.append(point)
        return point

    def log_artifact(self, run_id: str, name: str, path: str) -> None:
        self.get(run_id).artifacts[name] = path

    def find_by_experiment(self, experiment: str) -> list[Run]:
        return [r for r in self._runs.values() if r.experiment == experiment]

    def best_run(self, experiment: str, metric_key: str,
                 higher_is_better: bool = True) -> Run | None:
        candidates = [
            r for r in self.find_by_experiment(experiment)
            if r.best(metric_key) is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: (r.best(metric_key).value * (1 if higher_is_better else -1)))

    def save(self) -> None:
        if not self._path:
            return
        payload = {}
        for run_id, run in self._runs.items():
            data = asdict(run)
            data["stage"] = run.stage.value
            data["metrics"] = [asdict(m) for m in run.metrics]
            payload[run_id] = data
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _load(self) -> None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MlopsError(f"corrupt store: {exc}") from exc
        for run_id, data in payload.items():
            metrics = [MetricPoint(**m) for m in data.pop("metrics", [])]
            data["stage"] = Stage(data["stage"])
            run = Run(**data)
            run.metrics = metrics
            self._runs[run_id] = run


class Trainer:
    def __init__(self, store: RunStore, experiment: str) -> None:
        self._store = store
        self._experiment = experiment

    def execute(
        self,
        params: dict[str, Any],
        train_fn: Callable[[dict[str, Any]], Callable[[int], float]],
        epochs: int,
    ) -> Run:
        run = self._store.create(self._experiment, params)
        self._store.transition(run.run_id, Stage.RUNNING)
        try:
            epoch_fn = train_fn(params)
            for epoch in range(epochs):
                value = epoch_fn(epoch)
                self._store.log_metric(run.run_id, "loss", value, step=epoch)
            self._store.transition(run.run_id, Stage.FINISHED)
        except Exception:
            self._store.transition(run.run_id, Stage.FAILED)
            raise
        finally:
            self._store.save()
        return run


def compare_runs(left: Run, right: Run, metric_key: str,
                 higher_is_better: bool = False) -> dict[str, Any]:
    left_best = left.best(metric_key, higher_is_better)
    right_best = right.best(metric_key, higher_is_better)
    if left_best is None or right_best is None:
        raise MlopsError("both runs need the metric for comparison")
    if higher_is_better:
        winner = left if left_best.value >= right_best.value else right
    else:
        winner = left if left_best.value <= right_best.value else right
    delta = round(abs(left_best.value - right_best.value), 6)
    return {
        "winner": winner.run_id,
        "delta": delta,
        "left": str(left_best),
        "right": str(right_best),
    }
