"""Model promotion-sync endpoints (M7.5).

`model_reviewer`-write-only, same permission-gate pattern as golden.py:
every approve/reject/rollback goes through
model_promotion_service.require_model_reviewer, which 403s anyone else.
Reads (listing pending/history) are open - there's nothing sensitive in
seeing that a promotion is pending, only in deciding it.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.db import SessionLocal
from app.models.schemas import ModelPromotionInfo
from app.services import model_promotion_repo as repo
from app.services import model_promotion_service
from app.session_context import get_current_annotator

router = APIRouter(prefix="/api/model-promotions", tags=["model-promotions"])


def _to_info(promotion) -> ModelPromotionInfo:
    return ModelPromotionInfo(
        id=promotion.id,
        dataset_view=promotion.dataset_view,
        mlflow_model_name=promotion.mlflow_model_name,
        mlflow_version=promotion.mlflow_version,
        mlflow_run_id=promotion.mlflow_run_id,
        promotion_recommendation=promotion.promotion_recommendation,
        regressed_classes=promotion.regressed_classes,
        status=promotion.status,
        local_weights_path=promotion.local_weights_path,
        created_at=promotion.created_at,
        decided_at=promotion.decided_at,
        decided_by=promotion.decided_by.name if promotion.decided_by else None,
    )


@router.get("/pending", response_model=list[ModelPromotionInfo])
def list_pending(dataset_view: str | None = None) -> list[ModelPromotionInfo]:
    db = SessionLocal()
    try:
        return [_to_info(p) for p in repo.list_pending(db, dataset_view)]
    finally:
        db.close()


@router.get("/history", response_model=list[ModelPromotionInfo])
def list_history(dataset_view: str | None = None, limit: int = 50) -> list[ModelPromotionInfo]:
    db = SessionLocal()
    try:
        return [_to_info(p) for p in repo.list_history(db, dataset_view, limit)]
    finally:
        db.close()


@router.post("/{promotion_id}/approve", response_model=ModelPromotionInfo)
def approve(promotion_id: int) -> ModelPromotionInfo:
    annotator_id, _ = get_current_annotator()
    settings = get_settings()
    db = SessionLocal()
    try:
        promotion = model_promotion_service.approve(db, settings, settings.models_dir, promotion_id, annotator_id)
        return _to_info(promotion)
    except model_promotion_service.NotModelReviewerError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - sanity-check/download failures surface as a clear 422
        raise HTTPException(status_code=422, detail=f"Could not activate this model: {exc}") from exc
    finally:
        db.close()


@router.post("/{promotion_id}/reject", response_model=ModelPromotionInfo)
def reject(promotion_id: int) -> ModelPromotionInfo:
    annotator_id, _ = get_current_annotator()
    db = SessionLocal()
    try:
        promotion = model_promotion_service.reject(db, promotion_id, annotator_id)
        return _to_info(promotion)
    except model_promotion_service.NotModelReviewerError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        db.close()


@router.post("/rollback")
def rollback(dataset_view: str) -> dict:
    annotator_id, _ = get_current_annotator()
    settings = get_settings()
    db = SessionLocal()
    try:
        path = model_promotion_service.rollback(db, settings.models_dir, dataset_view, annotator_id)
        return {"activated_weights_path": str(path)}
    except model_promotion_service.NotModelReviewerError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        db.close()
