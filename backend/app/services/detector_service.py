"""Detector training and inference.

Fine-tunes a YOLOv8 detection model on whichever annotations you've reviewed
and marked complete (the same trust boundary export already uses), then runs
the most recently trained model on brand-new images that have no existing
labels at all - so boxes/classes it already learned show up automatically
instead of needing everything drawn by hand.
"""
from __future__ import annotations

import logging
import random
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from app.models.schemas import BoundingBox, DetectorInfo, DetectorTrainJobStatus, ObjectStatus
from app.services.dataset_service import DatasetService
from app.utils.file_utils import atomic_write_json, new_id, read_json

logger = logging.getLogger(__name__)

MODEL_WEIGHTS = "yolov8s.pt"
EPOCHS = 100
VAL_SPLIT = 0.1
MIN_TRAINING_IMAGES = 2


class DetectorService:
    """Trains a YOLOv8 detector on reviewed annotations and runs the active
    model on images that have no pre-existing detection labels."""

    def __init__(self, dataset_service: DatasetService, models_dir: Path) -> None:
        self._ds = dataset_service
        self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, DetectorTrainJobStatus] = {}
        self._lock = threading.Lock()
        self._loaded_model = None
        self._loaded_model_path: Optional[Path] = None

    # ------------------------------------------------------------- registry
    @property
    def _registry_path(self) -> Path:
        return self._models_dir / "detector_registry.json"

    def _load_registry(self) -> dict:
        return read_json(self._registry_path, default={})

    def _save_registry(self, data: dict) -> None:
        atomic_write_json(self._registry_path, data)

    def is_active(self) -> bool:
        return bool(self._load_registry().get("version"))

    def get_info(self) -> DetectorInfo:
        reg = self._load_registry()
        if not reg.get("version"):
            return DetectorInfo(active=False)
        return DetectorInfo(
            active=True,
            version=reg["version"],
            trained_at=reg.get("trained_at"),
            num_images=reg.get("num_images", 0),
            num_classes=len(reg.get("classes", [])),
            weights_size=MODEL_WEIGHTS,
        )

    # ------------------------------------------------------------- training
    def start_training(self) -> DetectorTrainJobStatus:
        self._ds.require_loaded()
        job_id = new_id()
        status = DetectorTrainJobStatus(
            job_id=job_id,
            status="running",
            stage="preparing",
            total_epochs=EPOCHS,
            started_at=time.time(),
            updated_at=time.time(),
        )
        with self._lock:
            self._jobs[job_id] = status
        threading.Thread(target=self._run_training, args=(job_id,), daemon=True).start()
        return status

    def get_job(self, job_id: str) -> Optional[DetectorTrainJobStatus]:
        return self._jobs.get(job_id)

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()

    def _run_training(self, job_id: str) -> None:
        staging_dir = self._models_dir / "training_runs" / job_id
        try:
            classes = [c.name for c in self._ds.get_classes()]
            data_yaml, num_images = self._assemble_dataset(staging_dir, classes)
            if num_images < MIN_TRAINING_IMAGES:
                raise RuntimeError(
                    f"Only {num_images} reviewed (saved + marked complete) image(s) with objects were found; "
                    f"need at least {MIN_TRAINING_IMAGES}. Review and save a few more images first."
                )
            self._update(job_id, num_images=num_images, stage="training")

            import torch
            from ultralytics import YOLO

            device = 0 if torch.cuda.is_available() else "cpu"
            model = YOLO(MODEL_WEIGHTS)

            def on_epoch_end(trainer) -> None:
                try:
                    self._update(job_id, current_epoch=int(trainer.epoch) + 1)
                except Exception:
                    logger.exception("Failed to record training epoch progress")

            model.add_callback("on_train_epoch_end", on_epoch_end)
            model.train(
                data=str(data_yaml),
                epochs=EPOCHS,
                batch=-1,
                device=device,
                project=str(staging_dir),
                name="run",
                exist_ok=True,
                verbose=False,
                patience=20,
            )

            self._update(job_id, stage="saving")
            best_weights = staging_dir / "run" / "weights" / "best.pt"
            if not best_weights.exists():
                raise RuntimeError("Training finished but no best.pt weights file was produced")

            registry = self._load_registry()
            new_version = registry.get("version", 0) + 1
            dest = self._models_dir / f"detector_v{new_version}.pt"
            shutil.copy2(best_weights, dest)
            registry.update(
                {
                    "version": new_version,
                    "trained_at": time.time(),
                    "num_images": num_images,
                    "classes": classes,
                    "path": str(dest),
                }
            )
            self._save_registry(registry)

            with self._lock:
                self._loaded_model = None
                self._loaded_model_path = None

            self._update(job_id, status="completed", stage="done", current_epoch=EPOCHS)
        except Exception as exc:
            logger.exception("Detector training failed")
            self._update(job_id, status="failed", error=str(exc))
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def _assemble_dataset(self, staging_dir: Path, classes: list[str]) -> tuple[Path, int]:
        """Write a fresh YOLO-detection dataset from images you've reviewed and
        marked complete - the same trust boundary export() already uses, so
        the detector only ever learns from annotations a human has approved."""
        image_ids = [
            image_id for image_id in self._ds.image_ids() if self._ds.get_annotations(image_id).completed
        ]
        random.Random(42).shuffle(image_ids)
        split_at = max(1, int(len(image_ids) * (1 - VAL_SPLIT))) if len(image_ids) > 1 else len(image_ids)
        splits = {"train": image_ids[:split_at], "val": image_ids[split_at:] or image_ids[:1]}

        total_images = 0
        for split, ids in splits.items():
            img_dir = staging_dir / split / "images"
            lbl_dir = staging_dir / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for image_id in ids:
                annotations = self._ds.get_annotations(image_id)
                objects = [o for o in annotations.objects if o.status != ObjectStatus.REJECTED]
                if not objects:
                    continue
                src_image = self._ds.get_image_path(image_id)
                shutil.copy2(src_image, img_dir / src_image.name)
                lines = [
                    f"{o.class_id} {o.bbox.x_center:.6f} {o.bbox.y_center:.6f} {o.bbox.width:.6f} {o.bbox.height:.6f}"
                    for o in objects
                ]
                (lbl_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                total_images += 1

        import yaml

        data_yaml = staging_dir / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(staging_dir),
                    "train": "train/images",
                    "val": "val/images",
                    "nc": len(classes),
                    "names": classes,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return data_yaml, total_images

    # ------------------------------------------------------------ inference
    def _ensure_model_loaded(self):
        registry = self._load_registry()
        path = Path(registry["path"])
        with self._lock:
            if self._loaded_model is not None and self._loaded_model_path == path:
                return self._loaded_model
            from ultralytics import YOLO

            self._loaded_model = YOLO(str(path))
            self._loaded_model_path = path
            return self._loaded_model

    def detect(self, image_path: Path, classes: list[str]) -> list[tuple[int, BoundingBox]]:
        """Run the most recently trained detector on an image with no
        pre-existing labels, returning (class_id, bbox) tuples in the same
        shape as parse_detection_label_file."""
        if not self.is_active():
            return []
        model = self._ensure_model_loaded()
        results = model.predict(str(image_path), conf=0.25, verbose=False)
        if not results:
            return []
        result = results[0]
        h, w = result.orig_shape
        detections: list[tuple[int, BoundingBox]] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id >= len(classes):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                (
                    class_id,
                    BoundingBox(
                        x_center=((x1 + x2) / 2) / w,
                        y_center=((y1 + y2) / 2) / h,
                        width=(x2 - x1) / w,
                        height=(y2 - y1) / h,
                    ),
                )
            )
        return detections


_detector_service: Optional[DetectorService] = None
_lock = threading.Lock()


def get_detector_service() -> DetectorService:
    global _detector_service
    if _detector_service is None:
        with _lock:
            if _detector_service is None:
                from app.config import get_settings
                from app.services.dataset_service import get_dataset_service

                settings = get_settings()
                _detector_service = DetectorService(get_dataset_service(), settings.models_dir)
    return _detector_service
