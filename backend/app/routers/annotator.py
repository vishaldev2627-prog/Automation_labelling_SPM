"""Lightweight per-annotator identity - a name, not a login (Phase 1a, task
#3). See app.session_context for how the resolved identity is attached to
the current browser session."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db import SessionLocal
from app.models.schemas import AnnotatorIdentity, IdentifyRequest
from app.services.annotator_service import get_or_create_annotator
from app.session_context import get_current_annotator, set_current_annotator

router = APIRouter(prefix="/api/annotator", tags=["annotator"])


@router.get("/me", response_model=AnnotatorIdentity)
def get_me() -> AnnotatorIdentity:
    annotator_id, name = get_current_annotator()
    return AnnotatorIdentity(id=annotator_id, name=name)


@router.post("/identify", response_model=AnnotatorIdentity)
def identify(payload: IdentifyRequest) -> AnnotatorIdentity:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name must not be empty")

    db = SessionLocal()
    try:
        annotator = get_or_create_annotator(db, name)
    finally:
        db.close()

    set_current_annotator(annotator.id, annotator.name)
    return AnnotatorIdentity(id=annotator.id, name=annotator.name)
