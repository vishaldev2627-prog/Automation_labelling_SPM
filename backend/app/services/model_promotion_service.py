"""Promotion sync (M7.5): a second, human-gated step between "MLflow says
this version is Production" and "live annotators actually get suggestions
from it."

Two distinct decisions, deliberately kept separate:
  1. A `golden_curator`/anyone with MLflow access moves a version to
     `Production` in MLflow's own UI (M7) - that's a judgment about model
     quality against the golden set.
  2. A `model_reviewer` here approves or rejects actually swapping the
     live-serving file to match it - that's a judgment about operational
     safety, on the record in `model_promotions` (this app's own audit
     trail, not MLflow's).

**Detection is automatic** (`check_for_new_production_version`, called by a
background poller - see app.main). **Activation is not** - approval always
requires an explicit `model_reviewer` action, and even then the swap only
happens after a sanity-load-and-run-once check, so a broken or incompatible
checkpoint can never leave live serving with no model loaded at all.

**`detector_service.detect()` never talks to MLflow, at any point in this
flow.** Approval downloads the artifact and writes a normal local
`registry.json` update - the exact same file `_run_training` already
writes after a local training run. `_ensure_model_loaded` already re-reads
that file and swaps its in-memory model whenever the path on disk changes,
which is what makes the live-serving side of this "automatic" once
approved, with zero MLflow dependency on the request path.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.config import Settings
from app.models.db_models import Annotator, ModelPromotion
from app.services import mlflow_tracking
from app.services import model_promotion_repo as repo
from app.services.detector_service import DETECTORS_SUBDIR, _slug_for
from app.utils.file_utils import atomic_write_json, read_json

logger = logging.getLogger(__name__)


def _configure_mlflow(settings: Settings) -> bool:
    """Every MLflow-touching call in this module needs the tracking URI
    explicitly set first - `Settings` parses `.env` into its own object and
    never exports it to `os.environ`, so a bare `MlflowClient()` here would
    silently fall back to MLflow's own default (a local `./mlruns` dir) if
    nothing else in this process happened to set the env var first (e.g. a
    training run). Caught live: the background poller found nothing for an
    entire cycle before this was added, not because no Production version
    existed, but because it was never actually looking at the right server.
    Returns False (skip cleanly) if MLflow isn't configured at all."""
    if not mlflow_tracking.is_configured(settings):
        return False
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return True


class NotModelReviewerError(Exception):
    """Raised when a non-reviewer attempts to approve, reject, or roll back
    a model promotion."""


def _is_model_reviewer(annotator: Optional[Annotator]) -> bool:
    """Pure predicate of an already-fetched Annotator, same split as
    golden_service._is_golden_curator - testable without a database."""
    return annotator is not None and annotator.role == "model_reviewer"


def require_model_reviewer(db: Session, annotator_id: Optional[int]) -> None:
    if annotator_id is None:
        raise NotModelReviewerError("No identified annotator for this session")
    annotator = db.query(Annotator).filter(Annotator.id == annotator_id).one_or_none()
    if not _is_model_reviewer(annotator):
        raise NotModelReviewerError(f"Annotator {annotator_id} is not a model_reviewer")


def _view_dir(models_dir: Path, dataset_view: str) -> Path:
    return models_dir / DETECTORS_SUBDIR / _slug_for(dataset_view)


def check_for_new_production_version(
    db: Session, settings: Settings, dataset_view: str, mlflow_model_name: str
) -> Optional[ModelPromotion]:
    """Best-effort: compares MLflow's current Production version for
    `mlflow_model_name` against what's already been proposed for this view.
    Creates a new pending row only if this exact version has never been
    proposed before (approved, rejected, or still pending) - never
    duplicates, never re-asks about a version someone already decided on.

    Returns None (logged, not raised) on any MLflow connectivity problem, or
    if MLflow isn't configured at all - this is exactly the "detection"
    half of the design that must be independent of MLflow's availability; a
    down (or unconfigured) tracking server just means nothing new is found
    this cycle, not an error surfaced anywhere near an annotator.
    """
    try:
        if not _configure_mlflow(settings):
            return None

        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        production = client.get_latest_versions(mlflow_model_name, stages=["Production"])
    except Exception:
        logger.debug("Could not reach MLflow to check %s for a new Production version", mlflow_model_name)
        return None

    if not production:
        return None

    version = production[0]
    existing = repo.get_existing(
        db, dataset_view=dataset_view, mlflow_model_name=mlflow_model_name, mlflow_version=version.version
    )
    if existing is not None:
        return None  # already proposed (pending/approved/rejected) - nothing new

    tags = version.tags or {}
    promotion = repo.create_pending(
        db,
        dataset_view=dataset_view,
        mlflow_model_name=mlflow_model_name,
        mlflow_version=version.version,
        mlflow_run_id=version.run_id,
        promotion_recommendation=tags.get("promotion_recommendation", "unknown"),
        regressed_classes=tags.get("regressed_classes"),
    )
    logger.info(
        "New pending model promotion: %s v%s for %s (recommendation: %s)",
        mlflow_model_name, version.version, dataset_view, promotion.promotion_recommendation,
    )
    return promotion


def _sanity_check(weights_path: Path) -> None:
    """Load the checkpoint and run one inference on a synthetic probe image
    before anything live touches it. A checkpoint that fails this raises
    here, before the caller writes anything to registry.json - live serving
    keeps whatever was already active, untouched. Mirrors the "load,
    sanity-check on a known probe image, then flip the pointer, never end
    up with zero model loaded" principle from this project's own MLflow
    gap analysis."""
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    probe = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(probe, verbose=False)


def _activate_weights(view_dir: Path, weights_src: Path, source_metadata: dict) -> Path:
    """The exact same registry.json file _run_training's own local-training
    path writes - a new detector_v{N}.pt plus a version bump. Existing
    fields (classes, dataset_key) are preserved via dict.update, since a
    view only ever reaches this path after at least one local training run
    already populated them."""
    registry = read_json(view_dir / "registry.json", default={})
    new_version = registry.get("version", 0) + 1
    dest = view_dir / f"detector_v{new_version}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_src, dest)
    registry.update({"version": new_version, "trained_at": time.time(), "path": str(dest), **source_metadata})
    atomic_write_json(view_dir / "registry.json", registry)
    return dest


def approve(
    db: Session, settings: Settings, models_dir: Path, promotion_id: int, annotator_id: Optional[int]
) -> ModelPromotion:
    require_model_reviewer(db, annotator_id)
    promotion = repo.get(db, promotion_id)
    if promotion is None:
        raise LookupError(f"No model promotion with id {promotion_id}")
    if promotion.status != "pending":
        raise ValueError(f"Model promotion {promotion_id} is already '{promotion.status}', not pending")

    if not _configure_mlflow(settings):
        raise RuntimeError("MLflow is not configured (MLFLOW_TRACKING_URI unset) - cannot download this model")

    import mlflow

    downloaded = Path(
        mlflow.artifacts.download_artifacts(run_id=promotion.mlflow_run_id, artifact_path="weights/best.pt")
    )
    _sanity_check(downloaded)  # raises before anything live is touched

    dest = _activate_weights(
        _view_dir(models_dir, promotion.dataset_view),
        downloaded,
        {
            "source": "mlflow_promotion",
            "mlflow_model_name": promotion.mlflow_model_name,
            "mlflow_version": promotion.mlflow_version,
            "approved_by_annotator_id": annotator_id,
        },
    )
    return repo.decide(db, promotion_id, status="approved", decided_by_id=annotator_id, local_weights_path=str(dest))


def reject(db: Session, promotion_id: int, annotator_id: Optional[int]) -> ModelPromotion:
    require_model_reviewer(db, annotator_id)
    promotion = repo.get(db, promotion_id)
    if promotion is None:
        raise LookupError(f"No model promotion with id {promotion_id}")
    if promotion.status != "pending":
        raise ValueError(f"Model promotion {promotion_id} is already '{promotion.status}', not pending")
    return repo.decide(db, promotion_id, status="rejected", decided_by_id=annotator_id)


def rollback(db: Session, models_dir: Path, dataset_view: str, annotator_id: Optional[int]) -> Path:
    """Reactivate whatever was live *before* the most recent approved
    promotion - purely local, no MLflow round-trip needed, since the
    previous file was never deleted. Deliberately does not create a new
    `model_promotions` row (the unique constraint is on (view, model,
    version), and re-approving the same version would collide with it) -
    this is a simpler, direct file-level revert, logged rather than
    separately audited. A fuller rollback audit trail is a reasonable
    follow-up if this gets used often enough to want one.
    """
    require_model_reviewer(db, annotator_id)
    history = repo.list_history(db, dataset_view, limit=50)
    approved = [p for p in history if p.status == "approved" and p.local_weights_path]
    if len(approved) < 2:
        raise ValueError(f"Nothing to roll back to for {dataset_view} - fewer than 2 approved promotions on record")

    previous = approved[1]  # [0] is the current one; [1] is what was live before it
    previous_path = Path(previous.local_weights_path)
    if not previous_path.exists():
        raise FileNotFoundError(f"Previous weights file {previous_path} no longer exists on disk")

    registry = read_json(_view_dir(models_dir, dataset_view) / "registry.json", default={})
    registry.update(
        {
            "path": str(previous_path),
            "trained_at": time.time(),
            "source": "rollback",
            "rolled_back_to_mlflow_version": previous.mlflow_version,
            "rolled_back_by_annotator_id": annotator_id,
        }
    )
    atomic_write_json(_view_dir(models_dir, dataset_view) / "registry.json", registry)
    logger.warning(
        "Rolled back %s to mlflow version %s (promotion id %s) by annotator %s",
        dataset_view, previous.mlflow_version, previous.id, annotator_id,
    )
    return previous_path
