"""Mask generation, refinement and mask-candidate selection endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import GenerateAllRequest, GenerateMaskRequest, GenerateMaskResponse, ImageAnnotations
from app.services.dataset_service import DatasetNotFoundError, ImageNotFoundError, get_dataset_service
from app.services.mask_generation_service import get_mask_generation_service
from app.services.sam_service import SAMNotAvailableError, get_sam_service
from app.utils.file_utils import new_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["masks"])


@router.get("/sam/status")
def sam_status() -> dict:
    return get_sam_service().status


@router.post("/generate-mask", response_model=GenerateMaskResponse)
def generate_mask(request: GenerateMaskRequest) -> GenerateMaskResponse:
    ds = get_dataset_service()
    mask_service = get_mask_generation_service()
    try:
        annotations = ds.get_annotations(request.image_id)

        if request.object_id:
            obj = next((o for o in annotations.objects if o.id == request.object_id), None)
            if obj is None:
                raise HTTPException(status_code=404, detail=f"Object '{request.object_id}' not found")
        elif request.bbox is not None:
            from app.models.schemas import AnnotationObject, ObjectStatus

            class_id = request.class_id if request.class_id is not None else 0
            class_name = next((c.name for c in ds.get_classes() if c.class_id == class_id), "")
            obj = AnnotationObject(
                id=new_id(), class_id=class_id, class_name=class_name, bbox=request.bbox, status=ObjectStatus.PENDING
            )
            annotations.objects.append(obj)
        else:
            raise HTTPException(status_code=422, detail="Either object_id or bbox must be provided")

        mask_service.generate_for_object(
            request.image_id, obj, request.positive_points, request.negative_points
        )
        ds.save_annotations(annotations)

        return GenerateMaskResponse(
            object_id=obj.id,
            polygon=obj.polygon,
            confidence=obj.confidence,
            all_scores=obj.all_mask_scores,
            selected_mask_index=obj.selected_mask_index,
        )
    except SAMNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (DatasetNotFoundError, ImageNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/generate-all", response_model=ImageAnnotations)
def generate_all(request: GenerateAllRequest) -> ImageAnnotations:
    ds = get_dataset_service()
    mask_service = get_mask_generation_service()
    try:
        annotations = ds.get_annotations(request.image_id)
        annotations = mask_service.generate_all_masks(annotations)
        ds.save_annotations(annotations)
        return annotations
    except SAMNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (DatasetNotFoundError, ImageNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/select-mask/{image_id}/{object_id}", response_model=GenerateMaskResponse)
def select_mask(image_id: str, object_id: str, payload: dict) -> GenerateMaskResponse:
    mask_index = int(payload.get("mask_index", 0))
    ds = get_dataset_service()
    mask_service = get_mask_generation_service()
    try:
        annotations = ds.get_annotations(image_id)
        obj = next((o for o in annotations.objects if o.id == object_id), None)
        if obj is None:
            raise HTTPException(status_code=404, detail=f"Object '{object_id}' not found")
        mask_service.select_alternate_mask(image_id, obj, mask_index)
        ds.save_annotations(annotations)
        return GenerateMaskResponse(
            object_id=obj.id,
            polygon=obj.polygon,
            confidence=obj.confidence,
            all_scores=obj.all_mask_scores,
            selected_mask_index=obj.selected_mask_index,
        )
    except SAMNotAvailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (DatasetNotFoundError, ImageNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
