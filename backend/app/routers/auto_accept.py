"""Confidence-based auto-accept endpoints (see app.services.auto_accept_service
and annotation_module_build_plan.md). Propose-then-confirm, like triage/
review: GET only ever previews, POST only ever acts on an explicit list a
caller chose after seeing that preview - nothing here runs automatically.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db import SessionLocal
from app.models.schemas import TriageItem
from app.services import auto_accept_service
from app.services.dataset_service import DatasetNotFoundError, get_dataset_service

router = APIRouter(prefix="/api/auto-accept", tags=["auto-accept"])


@router.get("/candidates", response_model=list[TriageItem])
def get_candidates() -> list[TriageItem]:
    ds = get_dataset_service()
    db = SessionLocal()
    try:
        return auto_accept_service.find_candidates(db, ds)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        db.close()


@router.post("/execute")
def execute(payload: dict) -> dict:
    image_ids = payload.get("image_ids")
    if not isinstance(image_ids, list) or not image_ids:
        raise HTTPException(status_code=422, detail="image_ids (non-empty list) is required")

    ds = get_dataset_service()
    db = SessionLocal()
    try:
        accepted = auto_accept_service.bulk_accept(db, ds, image_ids)
        return {"accepted": accepted}
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        db.close()
