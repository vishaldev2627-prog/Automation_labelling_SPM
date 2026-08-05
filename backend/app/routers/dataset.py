"""Dataset loading and info endpoints."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.db import SessionLocal
from app.models.schemas import ClassInfo, ClassMapVersionInfo, DatasetInfo, DatasetView
from app.services import class_map_service
from app.services.dataset_service import (
    DatasetNotFoundError,
    ExcludedClassError,
    get_dataset_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dataset", tags=["dataset"])

# Fixed set of "dataset views" - independent dataset roots (own images/
# labels/.annotation_state/data.yaml each) living as sibling subfolders under
# the configured DATASET_PATH, split by camera angle. side_view holds the
# original, fully-annotated dataset; underbelly/wheel_shelling start empty
# and get populated as that footage becomes available. buffer is a raw,
# unlabeled frame dump with no pre-existing class list - starts with zero
# classes, built up via the Classes panel's +Add as annotators encounter
# components (product decision, not a fallback default).
DATASET_VIEWS = [
    DatasetView(key="side_view", label="Side View"),
    DatasetView(key="underbelly", label="Underbelly"),
    DatasetView(key="wheel_shelling", label="Wheel Shelling"),
    DatasetView(key="buffer", label="Buffer"),
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


@router.get("/class-map", response_model=ClassMapVersionInfo)
def get_class_map() -> ClassMapVersionInfo:
    """The immutable class-map version the loaded view currently resolves to.

    This is what a dataset snapshot pins, so that "class 7" is answerable later.
    Declared before `/classes/...` routes only for readability - the paths don't
    collide."""
    ds = get_dataset_service()
    try:
        ds.require_loaded()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db = SessionLocal()
    try:
        current = class_map_service.get_current(db, ds.dataset_key)
        if current is None:
            raise HTTPException(
                status_code=503,
                detail="No class-map version recorded yet for this view - see the backend log; "
                "snapshots taken now cannot pin a class map.",
            )
        return _class_map_info(current)
    finally:
        db.close()


@router.get("/class-map/versions", response_model=list[ClassMapVersionInfo])
def list_class_map_versions() -> list[ClassMapVersionInfo]:
    """Full history, oldest first. Versions are immutable and never deleted, so
    this is the audit trail for class-map drift - the thing that was missing when
    this project went through its 27-class remap."""
    ds = get_dataset_service()
    try:
        ds.require_loaded()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db = SessionLocal()
    try:
        return [_class_map_info(v) for v in class_map_service.list_versions(db, ds.dataset_key)]
    finally:
        db.close()


def _class_map_info(version) -> ClassMapVersionInfo:
    return ClassMapVersionInfo(
        version=version.version,
        content_hash=version.content_hash,
        names={int(class_id): name for class_id, name in version.names},
        exclude_classes=list(version.exclude_classes or []),
        created_at=version.created_at,
        created_by=version.created_by.name if version.created_by else None,
    )


@router.put("/classes/{class_id}/fine-structure")
def set_class_fine_structure(class_id: int, payload: dict) -> dict:
    """Toggle whether this class's masks preserve thin/branching detail: all
    contours kept, no polygon simplification, and a binary mask raster written
    at export. For crack/corrosion/shelling-style defects, which the pipeline
    scores on Dice/IoU plus length-recall - see db_models.DatasetClass."""
    fine_structure = payload.get("fine_structure")
    if not isinstance(fine_structure, bool):
        raise HTTPException(status_code=422, detail="fine_structure (boolean) is required")
    try:
        get_dataset_service().set_class_fine_structure(class_id, fine_structure)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"class_id": class_id, "fine_structure": fine_structure}


@router.post("/classes", response_model=ClassInfo)
def add_class(payload: dict) -> ClassInfo:
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="name is required")
    try:
        return get_dataset_service().add_class(name)
    except ExcludedClassError as exc:
        # 422, not 409: the name isn't in conflict with anything, it's simply
        # not a permitted value.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
