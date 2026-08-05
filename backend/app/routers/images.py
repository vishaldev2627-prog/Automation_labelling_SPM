"""Image listing, serving and annotation retrieval/save endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.models.schemas import (
    CoachType,
    ImageAnnotations,
    ImageListItem,
    ObjectStatus,
    SaveAnnotationRequest,
    SetCoachTypeRequest,
)
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


@router.post("/coach-type")
def set_coach_type(request: SetCoachTypeRequest) -> dict:
    """Bulk-assign a coach type (see schemas.CoachType).

    With no `image_ids`, applies only to images that are still `unknown` -
    never overwriting an already-assigned type, since a bulk action shouldn't
    be able to silently undo someone's per-image correction. Pass explicit
    `image_ids` to overwrite deliberately.

    **Only images with already-saved annotation state are touched**, and it
    reads them through get_saved_states() rather than get_annotations(). Two
    reasons, both load-bearing:
      1. get_annotations() *initializes* state for an unopened image, and for
         one with no label file that means calling _try_auto_detect() - a
         detector inference. Over a 2000-image view a bulk apply would fire
         2000 YOLO runs.
      2. A coach type on an image nobody has annotated carries no information
         anyway: the per-coach-type label counts this field exists to produce
         only count annotated objects. Unopened images pick their coach type
         up when they are first saved.
    """
    ds = get_dataset_service()
    try:
        ds.require_loaded()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    explicit = bool(request.image_ids)
    states = ds.get_saved_states(request.image_ids or ds.image_ids())

    updated = 0
    skipped_no_state = 0
    for image_id, state in states.items():
        annotations = ImageAnnotations.model_validate(state)
        if not explicit and annotations.coach_type != CoachType.UNKNOWN:
            continue
        if annotations.coach_type == request.coach_type:
            continue
        annotations.coach_type = request.coach_type
        ds.save_annotations(annotations)
        updated += 1

    if explicit:
        skipped_no_state = len(request.image_ids) - len(states)

    return {
        "coach_type": request.coach_type.value,
        "updated": updated,
        "skipped_no_saved_state": skipped_no_state,
    }


@router.post("/annotations/save", response_model=ImageAnnotations)
def save_annotations(request: SaveAnnotationRequest) -> ImageAnnotations:
    try:
        ds = get_dataset_service()
        annotations = ds.get_annotations(request.image_id)
        was_completed = annotations.completed
        annotations.objects = request.objects
        annotations.completed = request.mark_completed or annotations.completed
        if request.coach_type is not None:
            annotations.coach_type = request.coach_type
        if request.no_objects_confirmed is not None:
            # Only touched when the client actually sends it (see
            # SaveAnnotationRequest) - an autosave omits the field and must
            # not clear an existing confirmation.
            annotations.no_objects_confirmed = request.no_objects_confirmed
        if any(o.status != ObjectStatus.REJECTED for o in annotations.objects):
            # A confirmation only describes a frame with nothing in it. If a
            # live object was added after one was recorded, the assertion is
            # no longer true, and keeping it would export a frame that has
            # objects as a negative sample.
            #
            # Rejected objects don't count as present: "SAM2/the detector
            # proposed these, I rejected all of them, there is genuinely
            # nothing here" is one of the ways a real negative gets made.
            annotations.no_objects_confirmed = False
        ds.save_annotations(annotations)

        # Only propagate the moment an image is *newly* finalized — not on
        # every autosave — so a similar image isn't repeatedly overwritten
        # while this one is still being worked on.
        if annotations.completed and not was_completed:
            from app.services.propagation_service import get_propagation_service

            get_propagation_service().propagate_from_async(annotations)

        return annotations
    except (DatasetNotFoundError, ImageNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
