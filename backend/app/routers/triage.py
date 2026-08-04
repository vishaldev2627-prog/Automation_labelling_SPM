"""Phase 2 triage/prioritization endpoint (see annotation_module_build_plan.md)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import TriageQueue
from app.services.dataset_service import DatasetNotFoundError, get_dataset_service
from app.services.similarity_service import get_similarity_service
from app.services.triage_service import build_triage_queue

router = APIRouter(prefix="/api/triage", tags=["triage"])


@router.get("/queue", response_model=TriageQueue)
def get_triage_queue() -> TriageQueue:
    try:
        return build_triage_queue(get_dataset_service(), get_similarity_service())
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
