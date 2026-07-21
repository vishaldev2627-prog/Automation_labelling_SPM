"""Batch processing endpoints for generating masks across the whole dataset."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import BatchJobStatus, BatchProcessRequest
from app.services.batch_service import get_batch_service
from app.services.dataset_service import DatasetNotFoundError, get_dataset_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/batch-process", tags=["batch"])


@router.post("", response_model=BatchJobStatus)
def start_batch_process(request: BatchProcessRequest) -> BatchJobStatus:
    ds = get_dataset_service()
    try:
        ds.require_loaded()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    image_ids = request.image_ids or ds.image_ids()
    return get_batch_service().start_job(image_ids, request.overwrite)


@router.get("/{job_id}", response_model=BatchJobStatus)
def get_batch_status(job_id: str) -> BatchJobStatus:
    job = get_batch_service().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.get("", response_model=list[BatchJobStatus])
def list_batch_jobs() -> list[BatchJobStatus]:
    return get_batch_service().list_jobs()
