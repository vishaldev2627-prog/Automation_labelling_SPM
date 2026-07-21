"""Dataset loading and info endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import ClassInfo, DatasetInfo
from app.services.dataset_service import DatasetNotFoundError, get_dataset_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.post("/load", response_model=DatasetInfo)
def load_dataset(payload: dict) -> DatasetInfo:
    path = payload.get("dataset_path")
    if not path:
        raise HTTPException(status_code=422, detail="dataset_path is required")
    try:
        return get_dataset_service().load_dataset(path)
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/info", response_model=DatasetInfo)
def dataset_info() -> DatasetInfo:
    try:
        return get_dataset_service().get_dataset_info()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/classes", response_model=list[ClassInfo])
def get_classes() -> list[ClassInfo]:
    try:
        return get_dataset_service().get_classes()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/classes/{class_id}/color")
def set_class_color(class_id: int, payload: dict) -> dict:
    color = payload.get("color")
    if not color:
        raise HTTPException(status_code=422, detail="color is required")
    get_dataset_service().set_class_color(class_id, color)
    return {"class_id": class_id, "color": color}


@router.post("/classes", response_model=ClassInfo)
def add_class(payload: dict) -> ClassInfo:
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="name is required")
    try:
        return get_dataset_service().add_class(name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
