"""Dataset loading and info endpoints."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.models.schemas import ClassInfo, DatasetInfo, DatasetView
from app.services.dataset_service import DatasetNotFoundError, get_dataset_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dataset", tags=["dataset"])

# Fixed set of "dataset views" - independent dataset roots (own images/
# labels/.annotation_state/data.yaml each) living as sibling subfolders under
# the configured DATASET_PATH, split by camera angle. side_view holds the
# original, fully-annotated dataset; underbelly/wheel_shelling start empty
# and get populated as that footage becomes available.
DATASET_VIEWS = [
    DatasetView(key="side_view", label="Side View"),
    DatasetView(key="underbelly", label="Underbelly"),
    DatasetView(key="wheel_shelling", label="Wheel Shelling"),
]
_VIEW_KEYS = {v.key for v in DATASET_VIEWS}


@router.get("/views", response_model=list[DatasetView])
def list_dataset_views() -> list[DatasetView]:
    return DATASET_VIEWS


@router.post("/switch", response_model=DatasetInfo)
def switch_dataset_view(payload: dict) -> DatasetInfo:
    """Load one of the fixed DATASET_VIEWS for the current session only (see
    app.session_context) - other sessions' active dataset are unaffected."""
    view = payload.get("view")
    if view not in _VIEW_KEYS:
        raise HTTPException(status_code=422, detail=f"Unknown view '{view}'. Valid: {sorted(_VIEW_KEYS)}")

    base = Path(get_settings().dataset_path)
    try:
        info = get_dataset_service().load_dataset(str(base / view))
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    from app.services.similarity_service import get_similarity_service

    get_similarity_service().start_reindex()
    return info


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


@router.put("/classes/{class_id}/safety-critical")
def set_class_safety_critical(class_id: int, payload: dict) -> dict:
    safety_critical = payload.get("safety_critical")
    if not isinstance(safety_critical, bool):
        raise HTTPException(status_code=422, detail="safety_critical (boolean) is required")
    try:
        get_dataset_service().set_class_safety_critical(class_id, safety_critical)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"class_id": class_id, "safety_critical": safety_critical}


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
