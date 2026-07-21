"""Detector training endpoints: retrain YOLOv8 on reviewed annotations."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import DetectorInfo, DetectorTrainJobStatus
from app.services.dataset_service import DatasetNotFoundError, get_dataset_service
from app.services.detector_service import get_detector_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/detector", tags=["detector"])


@router.post("/train", response_model=DetectorTrainJobStatus)
def start_training() -> DetectorTrainJobStatus:
    try:
        get_dataset_service().require_loaded()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return get_detector_service().start_training()


@router.get("/train/{job_id}", response_model=DetectorTrainJobStatus)
def get_training_status(job_id: str) -> DetectorTrainJobStatus:
    job = get_detector_service().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.get("/active", response_model=DetectorInfo)
def get_active_detector() -> DetectorInfo:
    return get_detector_service().get_info()
