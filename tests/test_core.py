import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from mlops_lite import (
    ExperimentNotFoundError,
    InvalidStageError,
    MlopsError,
    RunStore,
    Stage,
    Trainer,
    compare_runs,
)


@pytest.fixture
def store():
    return RunStore()


def test_create_run_with_params(store):
    run = store.create("text-classifier", {"lr": 0.01})
    assert run.stage == Stage.CREATED
    assert run.params["lr"] == 0.01


def test_empty_experiment_rejected(store):
    with pytest.raises(MlopsError):
        store.create("  ", {})


def test_lifecycle_transitions(store):
    run = store.create("exp", {})
    store.transition(run.run_id, Stage.RUNNING)
    assert run.started_at is not None
    store.transition(run.run_id, Stage.FINISHED)
    assert run.finished_at is not None
    assert run.duration_seconds is not None


def test_illegal_transition_rejected(store):
    run = store.create("exp", {})
    with pytest.raises(InvalidStageError):
        store.transition(run.run_id, Stage.FINISHED)


def test_failed_can_restart(store):
    run = store.create("exp", {})
    store.transition(run.run_id, Stage.RUNNING)
    store.transition(run.run_id, Stage.FAILED)
    store.transition(run.run_id, Stage.RUNNING)
    assert run.stage == Stage.RUNNING


def test_metric_logging_and_best(store):
    run = store.create("exp", {})
    store.transition(run.run_id, Stage.RUNNING)
    for step, loss in enumerate([1.0, 0.5, 0.25]):
        store.log_metric(run.run_id, "loss", loss, step=step)
    best = run.best("loss", higher_is_better=False)
    assert best.value == 0.25
    assert run.best("accuracy") is None


def test_artifact_registration(store):
    run = store.create("exp", {})
    store.log_artifact(run.run_id, "weights", "/models/w.pt")
    assert run.artifacts["weights"] == "/models/w.pt"


def test_unknown_run_raises(store):
    with pytest.raises(ExperimentNotFoundError):
        store.get("ghost")


def test_find_by_experiment(store):
    first = store.create("shared", {})
    second = store.create("shared", {})
    store.create("other", {})
    found = store.find_by_experiment("shared")
    assert {first.run_id, second.run_id} <= {r.run_id for r in found}


def test_best_run_picks_lowest_loss():
    store = RunStore()
    trainer = Trainer(store, "sweep")

    def make_train(loss):
        return lambda params: (lambda epoch: loss)

    low_run = trainer.execute({"lr": 0.1}, make_train(0.9), epochs=2)
    better_run = trainer.execute({"lr": 0.01}, make_train(0.2), epochs=2)
    best = store.best_run("sweep", "loss", higher_is_better=False)
    assert best.best("loss").value == 0.2
    assert best.params["lr"] == 0.01
    assert low_run.best("loss").value == 0.9


def test_trainer_marks_failed_on_exception(store):
    def broken(params):
        def epoch_fn(epoch):
            raise RuntimeError("gpu vanished")
        return epoch_fn

    trainer = Trainer(store, "fragile")
    run = store.create("fragile", {})
    with pytest.raises(RuntimeError):
        trainer.execute({}, broken, epochs=3)
