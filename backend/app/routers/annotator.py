"""Lightweight per-annotator identity - a name, not a login (Phase 1a, task
#3). See app.session_context for how the resolved identity is attached to
the current browser session.

Also owns the "golden_curator" role (see app.models.db_models.Annotator) -
who's allowed to verify/update the frozen golden eval set once Phase 4
builds it. Role changes here aren't gated by anything (no real auth
exists), same caveat as the rest of this identity system - it's groundwork
for Phase 4's permission checks, not a security boundary today.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.db import SessionLocal
from app.models.schemas import AnnotatorIdentity, IdentifyRequest, SetRoleRequest
from app.services.annotator_service import get_or_create_annotator, list_annotators, set_role
from app.session_context import get_current_annotator, set_current_annotator

router = APIRouter(prefix="/api/annotator", tags=["annotator"])


@router.get("/me", response_model=AnnotatorIdentity)
def get_me() -> AnnotatorIdentity:
    annotator_id, name = get_current_annotator()
    if annotator_id is None:
        return AnnotatorIdentity(id=None, name=None, role=None)

    db = SessionLocal()
    try:
        from app.models.db_models import Annotator

        annotator = db.query(Annotator).filter(Annotator.id == annotator_id).one_or_none()
    finally:
        db.close()
    if annotator is None:
        return AnnotatorIdentity(id=annotator_id, name=name, role=None)
    return AnnotatorIdentity(id=annotator.id, name=annotator.name, role=annotator.role)


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
    return AnnotatorIdentity(id=annotator.id, name=annotator.name, role=annotator.role)


@router.get("", response_model=list[AnnotatorIdentity])
def list_all() -> list[AnnotatorIdentity]:
    db = SessionLocal()
    try:
        annotators = list_annotators(db)
        return [AnnotatorIdentity(id=a.id, name=a.name, role=a.role) for a in annotators]
    finally:
        db.close()


@router.put("/{annotator_id}/role", response_model=AnnotatorIdentity)
def update_role(annotator_id: int, payload: SetRoleRequest) -> AnnotatorIdentity:
    db = SessionLocal()
    try:
        annotator = set_role(db, annotator_id, payload.role)
        return AnnotatorIdentity(id=annotator.id, name=annotator.name, role=annotator.role)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        db.close()
