"""Exports reviewed annotations to YOLO segmentation label files."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.models.schemas import ExportRequest, ObjectStatus
from app.services.dataset_service import DatasetService
from app.utils.yolo_utils import write_segmentation_label_file

logger = logging.getLogger(__name__)


class ExportService:
    def __init__(self, dataset_service: DatasetService, exports_dir: Path) -> None:
        self._ds = dataset_service
        self._exports_dir = exports_dir

    def export(self, request: ExportRequest) -> dict:
        self._ds.require_loaded()
        image_ids = request.image_ids or self._ds.image_ids()

        out_images = self._exports_dir / "images"
        out_labels = self._exports_dir / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        exported, skipped = 0, 0
        for image_id in image_ids:
            annotations = self._ds.get_annotations(image_id)
            if request.only_completed and not annotations.completed:
                skipped += 1
                continue

            objects = [
                (obj.class_id, obj.polygon)
                for obj in annotations.objects
                if obj.status != ObjectStatus.REJECTED and len(obj.polygon) >= 3
            ]
            if not objects:
                skipped += 1
                continue

            label_path = out_labels / f"{image_id}.txt"
            write_segmentation_label_file(label_path, objects)

            src_image = self._ds.get_image_path(image_id)
            dst_image = out_images / src_image.name
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)

            exported += 1

        classes_file = self._exports_dir / "classes.txt"
        classes_file.write_text("\n".join(c.name for c in self._ds.get_classes()) + "\n", encoding="utf-8")

        logger.info("Export complete: %d exported, %d skipped", exported, skipped)
        return {"exported": exported, "skipped": skipped, "output_dir": str(self._exports_dir)}
