"""Export endpoint: writes final YOLO segmentation labels + copies images."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models.schemas import ExportRequest
from app.services.dataset_service import DatasetNotFoundError, get_dataset_service
from app.services.export_service import ExportService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/export", tags=["export"])


@router.post("")
def export_dataset(request: ExportRequest) -> dict:
    settings = get_settings()
    ds = get_dataset_service()
    try:
        ds.require_loaded()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    service = ExportService(ds, settings.exports_dir)
    return service.export(request)
