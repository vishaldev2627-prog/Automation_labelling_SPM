"""Visual-similarity index endpoints (Phase 1 annotation propagation)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import SimilarityIndexStatus, SimilarNeighbor
from app.services.dataset_service import DatasetNotFoundError, ImageNotFoundError, get_dataset_service
from app.services.similarity_service import get_similarity_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/similarity", tags=["similarity"])


@router.post("/reindex", response_model=SimilarityIndexStatus)
def start_reindex() -> SimilarityIndexStatus:
    ds = get_dataset_service()
    try:
        ds.require_loaded()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return get_similarity_service().start_reindex()


@router.get("/reindex/{job_id}", response_model=SimilarityIndexStatus)
def get_reindex_status(job_id: str) -> SimilarityIndexStatus:
    job = get_similarity_service().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.get("/{image_id}/neighbors", response_model=list[SimilarNeighbor])
def get_neighbors(image_id: str, k: int = Query(default=5, ge=1, le=50)) -> list[SimilarNeighbor]:
    try:
        return get_similarity_service().nearest_neighbors(image_id, k=k)
    except (DatasetNotFoundError, ImageNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
