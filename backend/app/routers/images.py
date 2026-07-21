"""Image listing, serving and annotation retrieval/save endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.models.schemas import ImageAnnotations, ImageListItem, SaveAnnotationRequest
from app.services.dataset_service import DatasetNotFoundError, ImageNotFoundError, get_dataset_service
from app.utils.image_utils import ImageLoadError, encode_jpeg, read_image_bgr

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/images", tags=["images"])


@router.get("", response_model=list[ImageListItem])
def list_images() -> list[ImageListItem]:
    try:
        return get_dataset_service().list_images()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{image_id}/file")
def get_image_file(image_id: str) -> Response:
    try:
        path = get_dataset_service().get_image_path(image_id)
        img = read_image_bgr(path)
        jpeg_bytes = encode_jpeg(img, quality=92)
        return Response(content=jpeg_bytes, media_type="image/jpeg")
    except ImageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ImageLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{image_id}/annotations", response_model=ImageAnnotations)
def get_annotations(image_id: str) -> ImageAnnotations:
    try:
        return get_dataset_service().get_annotations(image_id)
    except (DatasetNotFoundError, ImageNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/annotations/save", response_model=ImageAnnotations)
def save_annotations(request: SaveAnnotationRequest) -> ImageAnnotations:
    try:
        ds = get_dataset_service()
        annotations = ds.get_annotations(request.image_id)
        annotations.objects = request.objects
        annotations.completed = request.mark_completed or annotations.completed
        ds.save_annotations(annotations)
        return annotations
    except (DatasetNotFoundError, ImageNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
