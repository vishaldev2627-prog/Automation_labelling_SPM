"""Exports reviewed annotations to a ready-to-train YOLO segmentation dataset."""
from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

import yaml

from app.models.schemas import ExportRequest, ObjectStatus
from app.services.dataset_service import DatasetService
from app.utils.yolo_utils import write_segmentation_label_file

logger = logging.getLogger(__name__)

VAL_SPLIT = 0.1


class ExportService:
    def __init__(self, dataset_service: DatasetService, exports_dir: Path) -> None:
        self._ds = dataset_service
        self._exports_dir = exports_dir

    @staticmethod
    def _split_for(image_id: str) -> str:
        """Deterministic per-image train/val assignment so an image always
        lands in the same split across repeated/incremental exports."""
        digest = hashlib.md5(image_id.encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % 100
        return "val" if bucket < VAL_SPLIT * 100 else "train"

    def export(self, request: ExportRequest) -> dict:
        self._ds.require_loaded()
        image_ids = request.image_ids or self._ds.image_ids()

        exportable: list[tuple[str, list[tuple[int, list]]]] = []
        skipped = 0
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

            exportable.append((image_id, objects))

        # Only one image available: put it in train so val isn't empty-vs-all.
        force_train = len(exportable) < 2

        exported = 0
        for image_id, objects in exportable:
            split = "train" if force_train else self._split_for(image_id)
            out_images = self._exports_dir / "images" / split
            out_labels = self._exports_dir / "labels" / split
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)

            label_path = out_labels / f"{image_id}.txt"
            write_segmentation_label_file(label_path, objects)

            src_image = self._ds.get_image_path(image_id)
            dst_image = out_images / src_image.name
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)

            exported += 1

        class_names = [c.name for c in self._ds.get_classes()]
        classes_file = self._exports_dir / "classes.txt"
        classes_file.write_text("\n".join(class_names) + "\n", encoding="utf-8")

        data_yaml = {
            "path": str(self._exports_dir.resolve()),
            "train": "images/train",
            "val": "images/val" if (self._exports_dir / "images" / "val").exists() else "images/train",
            "names": {i: name for i, name in enumerate(class_names)},
        }
        (self._exports_dir / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        logger.info("Export complete: %d exported, %d skipped", exported, skipped)
        return {"exported": exported, "skipped": skipped, "output_dir": str(self._exports_dir)}
