"""Best-effort MLflow tracking for detector training (M5, Scope A only).

Scope A means the in-tool SAM2/detector pre-labeler helper - the model that
suggests boxes to annotators - never the pipeline team's own 8 production
defect-detection families. Q-C is explicit: this project stages data for
the pipeline team, it never writes into their MLflow. This module's
tracking server is a separate, low-stakes instance for a separate, low-
stakes model.

**Individually best-effort, not one wrapping context manager.** A context
manager around the whole training body risks catching (and mislabeling as
an MLflow problem) a real training failure raised from inside it, or - the
opposite mistake - letting an MLflow connectivity blip during `__exit__`
mark an otherwise-successful training run as failed. Instead, every call
here degrades on its own: if MLflow is unreachable, training proceeds
untracked rather than failing, matching the contract
app/services/object_store.py already established for snapshot publishing.
`detector_service._run_training` is responsible for deciding the *real*
success/failure of a run; this module only ever reports that decision to
MLflow, never makes it.

`mlflow` is imported lazily inside each function - a deployment with
MLFLOW_TRACKING_URI unset never needs to construct a client at all.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.config import Settings

logger = logging.getLogger(__name__)


def is_configured(settings: Settings) -> bool:
    return bool(settings.mlflow_tracking_uri)


def start(settings: Settings, run_name: str, tags: dict) -> bool:
    """Best-effort: start a run, return whether one is now active. Every
    other function in this module is a no-op if called without a run
    active having returned True here - callers don't need to check twice."""
    if not is_configured(settings):
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        mlflow.start_run(run_name=run_name, tags=tags)
        return True
    except Exception:
        logger.exception(
            "Could not start MLflow run '%s' at %s; training will proceed untracked",
            run_name, settings.mlflow_tracking_uri,
        )
        return False


def log_params(params: dict) -> None:
    _safe(lambda: __import__("mlflow").log_params(params))


def log_metrics(metrics: dict, step: int) -> None:
    _safe(lambda: __import__("mlflow").log_metrics(metrics, step=step))


def log_artifact(path: Path) -> None:
    """Logs one file. Called explicitly, before the caller's own
    `finally: shutil.rmtree(staging_dir)` - the bug this module exists
    partly to fix: training artifacts (results.csv, PR curve, confusion
    matrix) used to be silently deleted with nothing having ever read them,
    since only best.pt was copied out beforehand."""
    _safe(lambda: __import__("mlflow").log_artifact(str(path)))


def log_artifacts(directory: Path) -> None:
    """Logs an entire directory tree - the training run folder ultralytics
    writes (results.csv, confusion_matrix.png, PR_curve.png, args.yaml,
    weights/) - in one call, same before-the-rmtree timing as log_artifact."""
    _safe(lambda: __import__("mlflow").log_artifacts(str(directory)))


def set_tags(tags: dict) -> None:
    _safe(lambda: __import__("mlflow").set_tags(tags))


def current_run_id() -> Optional[str]:
    """The active run's id, if one is active - None on any failure
    (including "no run active"), same best-effort contract as everything
    else here. Used by model_registry_service (M7) to register this run's
    model under its own run_id, since MLflow's registry ties a model
    version to the run that produced it."""
    try:
        import mlflow

        run = mlflow.active_run()
        return run.info.run_id if run else None
    except Exception:
        logger.exception("Could not read the active MLflow run id")
        return None


def end(status: str) -> None:
    """status: one of MLflow's own run-status strings - "FINISHED" or
    "FAILED". Always called from detector_service's own except/success
    paths, explicitly, rather than relying on a context manager's __exit__
    to infer it - see module docstring for why that distinction matters."""
    _safe(lambda: __import__("mlflow").end_run(status=status))


def _safe(fn) -> None:
    try:
        fn()
    except Exception:
        logger.exception("MLflow logging call failed; ignored, training result is unaffected")
