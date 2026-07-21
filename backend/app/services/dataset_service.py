"""Dataset scanning, per-image annotation state, autosave and progress tracking.

State is persisted as one JSON file per image inside `<dataset>/.annotation_state/`,
plus a `_meta.json` with class list/colors and dataset-level bookkeeping. This
gives crash recovery for free: any accepted edit is written to disk immediately
and reloaded on the next request for that image.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from app.config import Settings
from app.models.schemas import (
    AnnotationObject,
    BoundingBox,
    ClassInfo,
    DatasetInfo,
    ImageAnnotations,
    ImageListItem,
    ObjectStatus,
)
from app.utils.file_utils import atomic_write_json, image_stem_to_label_path, new_id, read_json
from app.utils.image_utils import get_image_dimensions, list_images
from app.utils.yolo_utils import load_class_names, parse_detection_label_file

logger = logging.getLogger(__name__)

DEFAULT_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#808080", "#ff4d4d", "#4dff4d", "#4d4dff", "#ffff4d",
    "#ff4dff", "#4dffff", "#c04d4d", "#4dc0c0",
]


class DatasetNotFoundError(Exception):
    pass


class ImageNotFoundError(Exception):
    pass


class DatasetService:
    """Manages dataset indexing and per-image annotation persistence."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._images_dir: Optional[Path] = None
        self._labels_dir: Optional[Path] = None
        self._state_dir: Optional[Path] = None
        self._index: dict[str, Path] = {}  # image_id -> image path
        self._classes: list[str] = []
        self._colors: dict[str, str] = {}
        self._loaded = False

    # ------------------------------------------------------------- loading
    def load_dataset(self, dataset_path: str) -> DatasetInfo:
        with self._lock:
            root = Path(dataset_path).expanduser().resolve()
            images_dir = root / "images"
            labels_dir = root / "labels"
            if not images_dir.exists():
                raise DatasetNotFoundError(f"'images' folder not found under {root}")
            if not labels_dir.exists():
                logger.warning("'labels' folder not found under %s; treating dataset as unlabeled", root)
                labels_dir.mkdir(parents=True, exist_ok=True)

            self._images_dir = images_dir
            self._labels_dir = labels_dir
            self._state_dir = root / self._settings.state_dir_name
            self._state_dir.mkdir(parents=True, exist_ok=True)

            images = list_images(images_dir)
            self._index = {self._image_id_for(p): p for p in images}

            raw_classes = load_class_names(root)
            max_class_id = self._scan_max_class_id(labels_dir)
            if len(raw_classes) <= max_class_id:
                raw_classes = raw_classes + [f"class_{i}" for i in range(len(raw_classes), max_class_id + 1)]
            self._classes = raw_classes

            self._load_class_colors()
            self._loaded = True
            logger.info("Loaded dataset at %s: %d images, %d classes", root, len(images), len(self._classes))
            return self.get_dataset_info()

    def _scan_max_class_id(self, labels_dir: Path) -> int:
        max_id = -1
        for label_file in labels_dir.glob("*.txt"):
            for class_id, _ in parse_detection_label_file(label_file):
                max_id = max(max_id, class_id)
        return max_id

    def _image_id_for(self, path: Path) -> str:
        return path.stem

    def require_loaded(self) -> None:
        if not self._loaded:
            raise DatasetNotFoundError("No dataset loaded. Call POST /api/dataset/load first.")

    # ------------------------------------------------------------ metadata
    @property
    def meta_path(self) -> Path:
        return self._state_dir / "_meta.json"

    def _load_class_colors(self) -> None:
        meta = read_json(self.meta_path, default={})
        colors = meta.get("colors", {}) if isinstance(meta, dict) else {}
        for i, name in enumerate(self._classes):
            key = str(i)
            if key not in colors:
                colors[key] = DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
        self._colors = colors
        self._save_meta()

    def _save_meta(self) -> None:
        atomic_write_json(self.meta_path, {"classes": self._classes, "colors": self._colors})

    def get_classes(self) -> list[ClassInfo]:
        return [
            ClassInfo(class_id=i, name=name, color=self._colors.get(str(i), DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]))
            for i, name in enumerate(self._classes)
        ]

    def set_class_color(self, class_id: int, color: str) -> None:
        with self._lock:
            self._colors[str(class_id)] = color
            self._save_meta()

    def add_class(self, name: str) -> ClassInfo:
        """Add a new class the detector never saw, persisting it back to disk
        (classes.txt or data.yaml) so it survives the next dataset reload."""
        self.require_loaded()
        name = name.strip()
        if not name:
            raise ValueError("Class name cannot be empty")
        with self._lock:
            if name in self._classes:
                raise ValueError(f"Class '{name}' already exists")
            class_id = len(self._classes)
            self._classes.append(name)
            color = DEFAULT_PALETTE[class_id % len(DEFAULT_PALETTE)]
            self._colors[str(class_id)] = color
            self._save_meta()
            self._persist_class_names()
            return ClassInfo(class_id=class_id, name=name, color=color)

    def _persist_class_names(self) -> None:
        """Write the current class list back to whichever file the dataset
        originally declared its classes in (data.yaml/.yml or classes.txt)."""
        if self._images_dir is None:
            return
        root = self._images_dir.parent
        for candidate in ("data.yaml", "data.yml", "dataset.yaml"):
            yaml_path = root / candidate
            if yaml_path.exists():
                try:
                    import yaml

                    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
                    data["names"] = list(self._classes)
                    data["nc"] = len(self._classes)
                    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
                except Exception:
                    logger.exception("Failed to persist class names to %s", yaml_path)
                return
        (root / "classes.txt").write_text("\n".join(self._classes) + "\n", encoding="utf-8")

    # --------------------------------------------------------------- index
    def list_images(self) -> list[ImageListItem]:
        self.require_loaded()
        items = []
        for image_id, path in self._index.items():
            state = self._read_state(image_id)
            items.append(
                ImageListItem(
                    image_id=image_id,
                    file_name=path.name,
                    completed=bool(state and state.get("completed")),
                    object_count=len(state.get("objects", [])) if state else 0,
                )
            )
        return items

    def get_image_path(self, image_id: str) -> Path:
        self.require_loaded()
        path = self._index.get(image_id)
        if path is None or not path.exists():
            raise ImageNotFoundError(f"Image '{image_id}' not found in loaded dataset")
        return path

    def image_ids(self) -> list[str]:
        return list(self._index.keys())

    # ------------------------------------------------------------ per-image
    def _state_path(self, image_id: str) -> Path:
        return self._state_dir / f"{image_id}.json"

    def _read_state(self, image_id: str) -> Optional[dict]:
        return read_json(self._state_path(image_id), default=None)

    def get_annotations(self, image_id: str) -> ImageAnnotations:
        """Get annotations for an image, loading from saved state or initializing
        fresh from the original YOLO detection labels on first access."""
        self.require_loaded()
        path = self.get_image_path(image_id)
        state = self._read_state(image_id)
        if state is not None:
            return ImageAnnotations.model_validate(state)

        width, height = get_image_dimensions(path)
        label_path = image_stem_to_label_path(self._labels_dir, path)
        detections = parse_detection_label_file(label_path)
        objects = [
            AnnotationObject(
                id=new_id(),
                class_id=class_id,
                class_name=self._classes[class_id] if class_id < len(self._classes) else f"class_{class_id}",
                bbox=bbox,
                status=ObjectStatus.PENDING,
            )
            for class_id, bbox in detections
        ]
        annotations = ImageAnnotations(
            image_id=image_id,
            file_name=path.name,
            width=width,
            height=height,
            objects=objects,
            completed=False,
        )
        return annotations

    def save_annotations(self, annotations: ImageAnnotations) -> None:
        """Persist annotation state for an image (autosave target)."""
        self.require_loaded()
        annotations.last_modified = time.time()
        with self._lock:
            atomic_write_json(self._state_path(annotations.image_id), annotations.model_dump(mode="json"))

    # -------------------------------------------------------------- progress
    def get_dataset_info(self) -> DatasetInfo:
        self.require_loaded()
        total = len(self._index)
        completed = 0
        durations: list[float] = []
        for image_id in self._index:
            state = self._read_state(image_id)
            if state and state.get("completed"):
                completed += 1

        percent = (completed / total * 100.0) if total else 0.0
        eta = self._estimate_eta(total, completed)
        return DatasetInfo(
            dataset_path=str(self._images_dir.parent) if self._images_dir else "",
            total_images=total,
            completed=completed,
            remaining=total - completed,
            percent_complete=round(percent, 2),
            classes=self._classes,
            estimated_seconds_remaining=eta,
        )

    def _estimate_eta(self, total: int, completed: int) -> Optional[float]:
        """Estimate remaining time from the average gap between completion timestamps."""
        if completed < 2:
            return None
        timestamps = []
        for image_id in self._index:
            state = self._read_state(image_id)
            if state and state.get("completed") and state.get("last_modified"):
                timestamps.append(state["last_modified"])
        if len(timestamps) < 2:
            return None
        timestamps.sort()
        deltas = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:]) if t2 > t1]
        if not deltas:
            return None
        avg = sum(deltas) / len(deltas)
        remaining = total - completed
        return round(avg * remaining, 1)

    @property
    def labels_dir(self) -> Path:
        self.require_loaded()
        return self._labels_dir

    @property
    def images_dir(self) -> Path:
        self.require_loaded()
        return self._images_dir


_dataset_service: Optional[DatasetService] = None
_ds_lock = threading.Lock()


def get_dataset_service() -> DatasetService:
    global _dataset_service
    if _dataset_service is None:
        with _ds_lock:
            if _dataset_service is None:
                from app.config import get_settings

                _dataset_service = DatasetService(get_settings())
    return _dataset_service
