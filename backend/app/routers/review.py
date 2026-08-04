"""Phase 4 QA/audit layer endpoints: second-reviewer sign-off and audit
sampling (see annotation_module_build_plan.md and app.services.review_service).
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from app.db import SessionLocal
from app.models.schemas import ClassAuditStats, ReviewRecord, SubmitReviewRequest, TriageItem
from app.services import review_service
from app.services.dataset_service import DatasetNotFoundError, get_dataset_service
from app.services.review_service import VALID_DECISIONS, VALID_REASONS
from app.session_context import get_current_annotator

router = APIRouter(prefix="/api/review", tags=["review"])


@router.get("/pending", response_model=list[TriageItem])
def get_pending() -> list[TriageItem]:
    ds = get_dataset_service()
    db = SessionLocal()
    try:
        return review_service.get_pending_second_review(db, ds)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        db.close()


@router.get("/audit-sample", response_model=list[TriageItem])
def get_audit_sample() -> list[TriageItem]:
    ds = get_dataset_service()
    db = SessionLocal()
    try:
        return review_service.get_audit_sample(db, ds)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        db.close()


@router.get("/class-stats", response_model=list[ClassAuditStats])
def get_class_stats() -> list[ClassAuditStats]:
    """Must be declared before GET /{image_id} - FastAPI matches routes in
    order, and {image_id} would otherwise greedily swallow this path."""
    ds = get_dataset_service()
    db = SessionLocal()
    try:
        stats = review_service.get_class_audit_stats(db, ds)
        return [
            ClassAuditStats(
                class_id=c.class_id,
                name=c.name,
                safety_critical=c.safety_critical,
                reviewed=stats.get(str(c.class_id), {}).get("reviewed", 0),
                approved=stats.get(str(c.class_id), {}).get("approved", 0),
                rejected=stats.get(str(c.class_id), {}).get("rejected", 0),
            )
            for c in ds.get_classes()
        ]
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        db.close()


@router.get("/{image_id}", response_model=Optional[ReviewRecord])
def get_review(image_id: str) -> Optional[ReviewRecord]:
    ds = get_dataset_service()
    db = SessionLocal()
    try:
        return review_service.get_latest_review(db, ds, image_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        db.close()


@router.post("/{image_id}", response_model=ReviewRecord)
def submit_review(image_id: str, payload: SubmitReviewRequest) -> ReviewRecord:
    if payload.decision not in VALID_DECISIONS:
        raise HTTPException(status_code=422, detail=f"decision must be one of {VALID_DECISIONS}")
    if payload.reason not in VALID_REASONS:
        raise HTTPException(status_code=422, detail=f"reason must be one of {VALID_REASONS}")

    reviewer_id, reviewer_name = get_current_annotator()
    if reviewer_id is None:
        raise HTTPException(
            status_code=400, detail="Identify yourself first (POST /api/annotator/identify) before reviewing"
        )

    ds = get_dataset_service()
    db = SessionLocal()
    try:
        review = review_service.submit_review(
            db, ds, image_id, reviewer_id, payload.decision, payload.reason, payload.notes
        )
        return ReviewRecord(
            id=review.id,
            image_id=review.image_id,
            reviewer_id=review.reviewer_id,
            reviewer_name=reviewer_name,
            decision=review.decision,
            reason=review.reason,
            notes=review.notes,
            created_at=review.created_at,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        db.close()
