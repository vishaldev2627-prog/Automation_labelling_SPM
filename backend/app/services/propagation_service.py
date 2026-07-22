"""Propagates finalized annotations onto visually similar, not-yet-reviewed images.

Runs entirely in the background off the back of a normal "save + mark
completed" request: once an image is finalized, its accepted objects are
carried onto its nearest neighbors (per SimilarityService) as SAM2
box-prompted seeds, replacing whatever those neighbors would otherwise have
shown. Because propagation always seeds from the *finalized* object list —
never merged with a neighbor's own independent detections — anything the
user rejected on the source image is simply absent from what gets proposed
on the neighbor too.

Only ever writes into images no human has reviewed yet (see
`_is_untouched`), so it can never clobber someone else's in-progress work.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.config import Settings
from app.models.schemas import AnnotationObject, ImageAnnotations, ObjectSource, ObjectStatus
from app.services.dataset_service import DatasetNotFoundError, DatasetService, ImageNotFoundError
from app.services.mask_generation_service import MaskGenerationService
from app.services.similarity_service import SimilarityService
from app.utils.file_utils import new_id

logger = logging.getLogger(__name__)


def _is_untouched(annotations: ImageAnnotations) -> bool:
    """True if no human has reviewed this image yet — i.e. it's safe for
    propagation to overwrite it wholesale (fresh, or only ever auto-filled)."""
    if annotations.completed:
        return False
    human_touched_statuses = {ObjectStatus.EDITED, ObjectStatus.CONFIRMED, ObjectStatus.REJECTED}
    return not any(obj.status in human_touched_statuses for obj in annotations.objects)


class PropagationService:
    def __init__(
        self,
        dataset_service: DatasetService,
        similarity_service: SimilarityService,
        mask_service: MaskGenerationService,
        settings: Settings,
    ) -> None:
        self._ds = dataset_service
        self._similarity = similarity_service
        self._mask_service = mask_service
        self._settings = settings
        self._executor = ThreadPoolExecutor(max_workers=1)

    def propagate_from_async(self, source_annotations: ImageAnnotations) -> None:
        if not self._settings.propagation_enabled:
            return
        self._executor.submit(self._propagate_from, source_annotations)

    def _propagate_from(self, source_annotations: ImageAnnotations) -> None:
        source_id = source_annotations.image_id
        seed_objects = [
            obj
            for obj in source_annotations.objects
            if obj.status != ObjectStatus.REJECTED and obj.visible
        ]
        if not seed_objects:
            logger.info("Propagation skipped for %s: no accepted objects to seed with", source_id)
            return

        try:
            self._similarity.ensure_indexed(source_id)
            neighbors = self._similarity.nearest_neighbors(
                source_id,
                k=self._settings.propagation_top_k,
                min_similarity=self._settings.similarity_threshold,
            )
        except (DatasetNotFoundError, ImageNotFoundError):
            return
        except Exception:
            logger.exception("Similarity lookup failed while propagating from %s", source_id)
            return

        if not neighbors:
            return

        logger.info(
            "Propagating %d object(s) from %s to %d candidate neighbor(s)",
            len(seed_objects), source_id, len(neighbors),
        )
        for neighbor in neighbors:
            try:
                self._propagate_to(source_id, seed_objects, neighbor.image_id)
            except (DatasetNotFoundError, ImageNotFoundError):
                continue
            except Exception:
                logger.exception("Propagation from %s to %s failed", source_id, neighbor.image_id)

    def _propagate_to(self, source_id: str, seed_objects: list[AnnotationObject], target_id: str) -> None:
        if target_id == source_id:
            return
        target = self._ds.get_annotations(target_id)
        if not _is_untouched(target):
            return

        new_objects: list[AnnotationObject] = []
        for src in seed_objects:
            new_obj = AnnotationObject(
                id=new_id(),
                class_id=src.class_id,
                class_name=src.class_name,
                bbox=src.bbox,
                status=ObjectStatus.PENDING,
                visible=True,
                source=ObjectSource.PROPAGATED,
                propagated_from_image_id=source_id,
            )
            try:
                self._mask_service.generate_for_object(target_id, new_obj)
            except Exception:
                logger.exception(
                    "Mask generation failed while propagating object %s from %s to %s",
                    src.id, source_id, target_id,
                )
                # Keep the box even if mask generation failed — better than losing the
                # propagated class/box entirely; the user can still regenerate manually.
            new_obj.source = ObjectSource.PROPAGATED
            new_obj.propagated_from_image_id = source_id
            new_objects.append(new_obj)

        target.objects = new_objects
        target.completed = False
        self._ds.save_annotations(target)
        logger.info("Propagated %d object(s) from %s onto %s", len(new_objects), source_id, target_id)


_propagation_service: Optional[PropagationService] = None
_lock = threading.Lock()


def get_propagation_service() -> PropagationService:
    global _propagation_service
    if _propagation_service is None:
        with _lock:
            if _propagation_service is None:
                from app.config import get_settings
                from app.services.dataset_service import get_dataset_service
                from app.services.mask_generation_service import get_mask_generation_service
                from app.services.similarity_service import get_similarity_service

                _propagation_service = PropagationService(
                    get_dataset_service(),
                    get_similarity_service(),
                    get_mask_generation_service(),
                    get_settings(),
                )
    return _propagation_service
