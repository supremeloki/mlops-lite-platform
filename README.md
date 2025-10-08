# mlops-lite-platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight MLOps platform: experiment tracking with typed run lifecycles, step-level metric logging, best-run selection, JSON persistence — MLflow's core ideas without the server.

## 🚀 Overview

Experiment tracking shouldn't require a tracking server. `mlops-lite-platform` gives every training run a lifecycle (`CREATED → RUNNING → FINISHED/FAILED` with illegal transitions rejected), step-level metric points, artifact registration, and a `Trainer` wrapper that records everything around your epoch function. `best_run()` picks winners per experiment and metric; the whole store persists as readable JSON.

## ✨ Features

- **Typed run state machine:** transitions validated against a transition table; failed runs may restart
- **Step metrics:** `MetricPoint(key, value, step, recorded_at)` history per run
- **Trainer wrapper:** wraps any `(params) → epoch_fn` factory; logs loss per epoch, marks FAILED on exception
- **Winner selection:** `store.best_run(experiment, metric)` respects direction (`higher_is_better`)
- **Run comparison:** `compare_runs(a, b, metric)` returns winner + delta, direction-aware
- **Artifacts:** attach name→path references to any run
- **JSON persistence:** survives restarts; corrupt stores raise typed errors

## 🚧 Structure

```
mlops-lite-platform/
├── src/mlops_lite/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/mlops-lite-platform.git
cd mlops-lite-platform
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies

## 🏃 Quick Start

```python
from pathlib import Path
from mlops_lite import RunStore, Trainer

store = RunStore(storage_path=Path("runs.json"))
trainer = Trainer(store, experiment="text-classifier")

def make_train(params):
    lr = params["lr"]
    def epoch_fn(epoch):
        return 1.0 / (epoch + 1) * (1 + lr)
    return epoch_fn

run = trainer.execute({"lr": 0.01}, make_train, epochs=5)
print(run.run_id, run.duration_seconds)
print(store.best_run("text-classifier", "loss"))
```

## 🔧 Error Handling

```text
MlopsError
├── ExperimentNotFoundError  # unknown run_id
├── InvalidStageError        # illegal lifecycle transition
└── corrupt-store detection on load
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen metric points
- Zero comments — names carry the meaning
- Lifecycle table drives validation, not scattered if-statements

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi** - [kooroushmasoumi@gmail.com](mailto:kooroushmasoumi@gmail.com)

---

⭐ Star this repo if you find it useful!
