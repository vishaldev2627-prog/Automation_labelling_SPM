"""Dataset-wide progress endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import DatasetInfo
from app.services.dataset_service import DatasetNotFoundError, get_dataset_service

router = APIRouter(prefix="/api/progress", tags=["progress"])


@router.get("", response_model=DatasetInfo)
def get_progress() -> DatasetInfo:
    try:
        return get_dataset_service().get_dataset_info()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
