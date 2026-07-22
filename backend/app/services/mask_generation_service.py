"""Orchestrates SAM inference + polygon extraction for annotation objects."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import Settings
from app.models.schemas import AnnotationObject, ImageAnnotations, ObjectStatus, Point
from app.services.dataset_service import DatasetService
from app.services.polygon_service import mask_to_polygon
from app.services.sam_service import SAMService
from app.utils.image_utils import compute_file_hash, read_image_rgb

logger = logging.getLogger(__name__)


class MaskGenerationService:
    def __init__(self, dataset_service: DatasetService, sam_service: SAMService, settings: Settings) -> None:
        self._ds = dataset_service
        self._sam = sam_service
        self._settings = settings
        self._image_cache: dict[str, tuple[np.ndarray, str]] = {}
        self._cache_lock = threading.Lock()

    def _load_image_for_sam(self, image_id: str) -> tuple[np.ndarray, str]:
        path = self._ds.get_image_path(image_id)
        cache_key = f"{image_id}:{compute_file_hash(path)}"
        with self._cache_lock:
            cached = self._image_cache.get(image_id)
            if cached and cached[1] == cache_key:
                return cached[0], cache_key
        img_rgb = read_image_rgb(path)
        with self._cache_lock:
            self._image_cache[image_id] = (img_rgb, cache_key)
            if len(self._image_cache) > 8:
                oldest_key = next(iter(self._image_cache))
                if oldest_key != image_id:
                    self._image_cache.pop(oldest_key, None)
        return img_rgb, cache_key

    def generate_for_object(
        self,
        image_id: str,
        obj: AnnotationObject,
        positive_points: Optional[list[Point]] = None,
        negative_points: Optional[list[Point]] = None,
    ) -> AnnotationObject:
        """Run SAM on a single object's bbox (+ optional click refinements)."""
        img_rgb, cache_key = self._load_image_for_sam(image_id)
        h, w = img_rgb.shape[:2]

        pos_pts = [(p.x * w, p.y * h) for p in (positive_points or [])]
        neg_pts = [(p.x * w, p.y * h) for p in (negative_points or [])]

        if pos_pts:
            result = self._sam.predict_points(img_rgb, cache_key, pos_pts, neg_pts)
        else:
            box_xyxy = obj.bbox.to_xyxy(w, h)
            result = self._sam.predict_box(img_rgb, cache_key, box_xyxy, pos_pts, neg_pts)

        best_idx = result.best_index
        confidence = float(result.scores[best_idx])
        polygon = (
            mask_to_polygon(
                result.masks[best_idx],
                epsilon_ratio=self._settings.polygon_epsilon_ratio,
                min_points=self._settings.min_polygon_points,
            )
            if confidence > self._settings.mask_confidence_threshold
            else []
        )

        obj.polygon = polygon
        obj.confidence = confidence
        obj.all_mask_scores = [float(s) for s in result.scores]
        obj.selected_mask_index = best_idx
        obj.status = ObjectStatus.AUTO_GENERATED if polygon else ObjectStatus.PENDING
        return obj

    def select_alternate_mask(self, image_id: str, obj: AnnotationObject, mask_index: int) -> AnnotationObject:
        """Switch to a different one of SAM's returned mask candidates without re-running inference."""
        img_rgb, cache_key = self._load_image_for_sam(image_id)
        h, w = img_rgb.shape[:2]
        box_xyxy = obj.bbox.to_xyxy(w, h)
        result = self._sam.predict_box(img_rgb, cache_key, box_xyxy)
        mask_index = max(0, min(mask_index, len(result.scores) - 1))
        polygon = mask_to_polygon(
            result.masks[mask_index],
            epsilon_ratio=self._settings.polygon_epsilon_ratio,
            min_points=self._settings.min_polygon_points,
        )
        obj.polygon = polygon
        obj.confidence = float(result.scores[mask_index])
        obj.all_mask_scores = [float(s) for s in result.scores]
        obj.selected_mask_index = mask_index
        return obj

    def generate_all_masks(self, annotations: ImageAnnotations, overwrite: bool = False) -> ImageAnnotations:
        """Run SAM for every object in the image that doesn't already have a polygon."""
        for obj in annotations.objects:
            if not overwrite and obj.polygon:
                continue
            try:
                self.generate_for_object(annotations.image_id, obj)
            except Exception:
                logger.exception("Mask generation failed for object %s in image %s", obj.id, annotations.image_id)
        return annotations


_mask_service: Optional[MaskGenerationService] = None
_lock = threading.Lock()


def get_mask_generation_service() -> MaskGenerationService:
    global _mask_service
    if _mask_service is None:
        with _lock:
            if _mask_service is None:
                from app.config import get_settings
                from app.services.dataset_service import get_dataset_service
                from app.services.sam_service import get_sam_service

                _mask_service = MaskGenerationService(get_dataset_service(), get_sam_service(), get_settings())
    return _mask_service
