"""Postgres index over promotion-sync decisions (M7.5).

Mirrors golden_repo/snapshot_repo's split: this module is pure DB access;
app/services/model_promotion_service.py owns the actual workflow (detecting
a new Production version, downloading/validating on approval, the
permission gate) built on top of it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.db_models import ModelPromotion


def create_pending(
    db: Session,
    *,
    dataset_view: str,
    mlflow_model_name: str,
    mlflow_version: str,
    mlflow_run_id: str,
    promotion_recommendation: str,
    regressed_classes: Optional[str],
) -> ModelPromotion:
    promotion = ModelPromotion(
        dataset_view=dataset_view,
        mlflow_model_name=mlflow_model_name,
        mlflow_version=mlflow_version,
        mlflow_run_id=mlflow_run_id,
        promotion_recommendation=promotion_recommendation,
        regressed_classes=regressed_classes,
        status="pending",
    )
    db.add(promotion)
    db.commit()
    db.refresh(promotion)
    return promotion


def get_existing(
    db: Session, *, dataset_view: str, mlflow_model_name: str, mlflow_version: str
) -> Optional[ModelPromotion]:
    """Whether this exact (view, model, version) has already been proposed -
    pending, approved, or rejected - so the background check never creates
    a duplicate row for the same version it already asked about."""
    return db.execute(
        select(ModelPromotion).where(
            ModelPromotion.dataset_view == dataset_view,
            ModelPromotion.mlflow_model_name == mlflow_model_name,
            ModelPromotion.mlflow_version == mlflow_version,
        )
    ).scalar_one_or_none()


def get(db: Session, promotion_id: int) -> Optional[ModelPromotion]:
    return db.execute(select(ModelPromotion).where(ModelPromotion.id == promotion_id)).scalar_one_or_none()


def list_pending(db: Session, dataset_view: Optional[str] = None) -> list[ModelPromotion]:
    query = select(ModelPromotion).where(ModelPromotion.status == "pending").order_by(ModelPromotion.created_at)
    if dataset_view is not None:
        query = query.where(ModelPromotion.dataset_view == dataset_view)
    return list(db.execute(query).scalars())


def list_history(db: Session, dataset_view: Optional[str] = None, limit: int = 50) -> list[ModelPromotion]:
    query = select(ModelPromotion).order_by(ModelPromotion.created_at.desc()).limit(limit)
    if dataset_view is not None:
        query = query.where(ModelPromotion.dataset_view == dataset_view)
    return list(db.execute(query).scalars())


def get_last_approved(db: Session, dataset_view: str) -> Optional[ModelPromotion]:
    """The most recently approved promotion for this view - what a rollback
    reverts *from*. Not "the one before the latest", deliberately: if the
    latest approved promotion is itself the one being rolled back, this
    naturally returns it, and the service layer decides what "previous" means."""
    return db.execute(
        select(ModelPromotion)
        .where(ModelPromotion.dataset_view == dataset_view, ModelPromotion.status == "approved")
        .order_by(ModelPromotion.decided_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def decide(db: Session, promotion_id: int, *, status: str, decided_by_id, local_weights_path: Optional[str] = None) -> ModelPromotion:
    promotion = get(db, promotion_id)
    if promotion is None:
        raise LookupError(f"No model promotion with id {promotion_id}")
    promotion.status = status
    promotion.decided_at = datetime.now(timezone.utc)
    promotion.decided_by_id = decided_by_id
    if local_weights_path is not None:
        promotion.local_weights_path = local_weights_path
    db.commit()
    db.refresh(promotion)
    return promotion
