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
