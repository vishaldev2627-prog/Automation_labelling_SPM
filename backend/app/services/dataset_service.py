"""Dataset scanning, per-image annotation state, autosave and progress tracking.

Per-image annotation state and class colors are persisted in Postgres (see
app.services.annotation_state_repo), keyed by this dataset's resolved root
path - not per-dataset JSON files anymore (Phase 1a, task #4). `.annotation_state/`
still exists on disk as a directory: similarity_service.py caches its
embedding index there, which is unrelated to annotation state.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Optional

from app.config import Settings
from app.db import SessionLocal
from app.models.schemas import (
    AnnotationObject,
    BoundingBox,
    ClassInfo,
    DatasetInfo,
    ImageAnnotations,
    ImageListItem,
    ObjectStatus,
)
from app.services import annotation_state_repo as state_repo
from app.services import class_map_service
from app.session_context import get_current_annotator
from app.utils.file_utils import image_stem_to_label_path, new_id
from app.utils.image_utils import get_image_dimensions, list_images
from app.utils.yolo_utils import load_class_names, parse_detection_label_file

logger = logging.getLogger(__name__)

# Keyed by dataset root path rather than per-DatasetService-instance: with
# per-session DatasetService (see app.session_context), two teammates working
# on the same view each hold their own instance and their own self._lock, so
# an instance-level lock does nothing to serialize their concurrent
# add_class() calls against each other. This one is shared by anyone pointed
# at the same folder.
_class_list_locks: dict[str, threading.Lock] = {}
_class_list_locks_guard = threading.Lock()


def _get_class_list_lock(root: Path) -> threading.Lock:
    key = str(root.resolve())
    with _class_list_locks_guard:
        lock = _class_list_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _class_list_locks[key] = lock
        return lock

# Starting default for a brand-new class's safety_critical flag (never
# overwrites an existing class's curator-set value - see
# annotation_state_repo._upsert_class). An engineering guess based on the
# plan's own language (§4.4: "cracked wheel, wheel shelling, undercarriage
# crack/corrosion") and component families actually present in this
# dataset, not a certified list - a domain expert should review it.
SAFETY_KEYWORD_PATTERN = re.compile(r"(wheel|brake|axle|suspension|disc|spring|crack|corrosion|shelling)", re.IGNORECASE)

# Deliberately narrower than SAFETY_KEYWORD_PATTERN: fine_structure is about
# defect *morphology* (thin, branching, length-measured), not safety tier. A
# wheel component outline is a blob that simplifies fine; a crack across it is
# the fine-structure case. Matching 'wheel' here would flag every wheel
# component and write mask rasters nobody needs. Same caveat as above - a
# starting point for a domain expert, not a certified list.
FINE_STRUCTURE_KEYWORD_PATTERN = re.compile(
    r"(crack|corrosion|shelling|scratch|flaking|spalling|tread)", re.IGNORECASE
)

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


class ExcludedClassError(Exception):
    """A class the pipeline never serves, so labeling it is never useful.

    `exclude_classes: [leakage]` (docs/pipeline.md §12, FINAL_AIML §10) names an
    all-synthetic class excluded from serving. The build plan requires enforcing
    that at the tool level, not at export - refusing the class up front beats
    letting someone annotate for a week and then explaining why those labels
    were dropped.
    """


class DatasetService:
    """Manages dataset indexing and per-image annotation persistence."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._images_dir: Optional[Path] = None
        self._labels_dir: Optional[Path] = None
        self._state_dir: Optional[Path] = None
        self._dataset_key: Optional[str] = None  # resolved dataset root; DB key for annotation_state/dataset_classes
        self._index: dict[str, Path] = {}  # image_id -> image path
        self._classes: list[str] = []
        self._colors: dict[str, str] = {}
        self._safety: dict[str, bool] = {}
        self._fine_structure: dict[str, bool] = {}
        self._class_map_version: Optional[int] = None
        self._class_map_hash: Optional[str] = None
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
            self._dataset_key = str(root.resolve())

            images = list_images(images_dir)
            self._index = {self._image_id_for(p): p for p in images}

            raw_classes = load_class_names(root)
            max_class_id = self._scan_max_class_id(labels_dir)
            if len(raw_classes) <= max_class_id:
                raw_classes = raw_classes + [f"class_{i}" for i in range(len(raw_classes), max_class_id + 1)]
            self._classes = raw_classes

            self._load_class_colors()
            self._sync_class_map_version()
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
    def state_dir(self) -> Path:
        self.require_loaded()
        return self._state_dir

    def _load_class_colors(self) -> None:
        db = SessionLocal()
        try:
            colors = state_repo.get_colors(db, self._dataset_key)
            for i, _name in enumerate(self._classes):
                key = str(i)
                if key not in colors:
                    colors[key] = DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
            self._colors = colors
            default_safety = {
                str(i): bool(SAFETY_KEYWORD_PATTERN.search(name)) for i, name in enumerate(self._classes)
            }
            default_fine_structure = {
                str(i): bool(FINE_STRUCTURE_KEYWORD_PATTERN.search(name))
                for i, name in enumerate(self._classes)
            }
            state_repo.save_colors_bulk(
                db, self._dataset_key, self._classes, self._colors, default_safety, default_fine_structure
            )
            # Re-read: save_colors_bulk only applies the defaults to brand-new
            # rows, so this picks up the values that actually landed (curator's
            # existing choices, or the new defaults).
            self._safety = state_repo.get_safety_flags(db, self._dataset_key)
            self._fine_structure = state_repo.get_fine_structure_flags(db, self._dataset_key)
        finally:
            db.close()

    def _sync_class_map_version(self) -> None:
        """Resolve the on-disk class list to an immutable class-map version,
        minting one if this exact map has not been seen before.

        Called on every load, deliberately: the class list's source of truth is
        still `data.yaml`/`classes.txt`, which a person can edit by hand, so
        drift has to be detected rather than assumed away. Minting is idempotent
        by content hash (see class_map_service), so a normal load creates
        nothing.

        Never raises into the load path: a class-map version is an audit record,
        and failing to write one must not make the dataset unopenable. It is
        logged loudly instead, and the next load retries.
        """
        annotator_id, _ = get_current_annotator()
        db = SessionLocal()
        try:
            version = class_map_service.ensure_version(
                db,
                self._dataset_key,
                self._classes,
                self._settings.exclude_class_list,
                annotator_id,
            )
            self._class_map_version = version.version
            self._class_map_hash = version.content_hash
        except Exception:
            logger.exception(
                "Could not resolve a class-map version for %s; dataset still loaded, "
                "but snapshots taken now cannot pin a class map",
                self._dataset_key,
            )
            self._class_map_version = None
            self._class_map_hash = None
        finally:
            db.close()

    @property
    def class_map_version(self) -> Optional[int]:
        return self._class_map_version

    @property
    def class_map_hash(self) -> Optional[str]:
        return self._class_map_hash

    def get_classes(self) -> list[ClassInfo]:
        return [
            ClassInfo(
                class_id=i,
                name=name,
                color=self._colors.get(str(i), DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]),
                safety_critical=self._safety.get(str(i), False),
                fine_structure=self._fine_structure.get(str(i), False),
            )
            for i, name in enumerate(self._classes)
        ]

    def set_class_color(self, class_id: int, color: str) -> None:
        with self._lock:
            self._colors[str(class_id)] = color
            db = SessionLocal()
            try:
                name = self._classes[class_id] if class_id < len(self._classes) else f"class_{class_id}"
                state_repo.set_class_color(db, self._dataset_key, class_id, name, color)
            finally:
                db.close()

    def set_class_safety_critical(self, class_id: int, safety_critical: bool) -> None:
        with self._lock:
            db = SessionLocal()
            try:
                found = state_repo.set_class_safety_critical(db, self._dataset_key, class_id, safety_critical)
            finally:
                db.close()
            if not found:
                raise ValueError(f"No class {class_id} in this dataset view")
            self._safety[str(class_id)] = safety_critical

    def set_class_fine_structure(self, class_id: int, fine_structure: bool) -> None:
        """Toggle whether this class's masks preserve thin/branching detail -
        all contours kept, no simplification, mask raster exported. See
        DatasetClass.fine_structure for why it is separate from safety_critical."""
        with self._lock:
            db = SessionLocal()
            try:
                found = state_repo.set_class_fine_structure(db, self._dataset_key, class_id, fine_structure)
            finally:
                db.close()
            if not found:
                raise ValueError(f"No class {class_id} in this dataset view")
            self._fine_structure[str(class_id)] = fine_structure

    def is_fine_structure(self, class_id: int) -> bool:
        return self._fine_structure.get(str(class_id), False)

    def add_class(self, name: str) -> ClassInfo:
        """Add a new class the detector never saw, persisting it back to disk
        (classes.txt or data.yaml) so it survives the next dataset reload.

        Re-reads the class list from disk under a lock shared across every
        DatasetService instance pointed at this same folder (not just
        self._lock, which only protects this one instance) - otherwise two
        teammates in separate sessions each holding their own stale in-memory
        copy would have the second add_class() wholesale overwrite data.yaml
        with their own list, silently deleting whatever the first person just
        added.
        """
        self.require_loaded()
        name = name.strip()
        if not name:
            raise ValueError("Class name cannot be empty")
        if self._settings.is_excluded_class(name):
            # The build plan requires excluding these at the tool level rather
            # than filtering at export - a label that can never be shipped is
            # wasted annotator time, and it is cheaper to refuse the class than
            # to explain later why those labels vanished.
            raise ExcludedClassError(
                f"'{name}' is in exclude_classes and is never served by the pipeline, "
                f"so it cannot be labeled here. Excluded: {self._settings.exclude_class_list}"
            )
        root = self._images_dir.parent
        with _get_class_list_lock(root):
            current = load_class_names(root)
            if name in current:
                raise ValueError(f"Class '{name}' already exists")
            class_id = len(current)
            current.append(name)
            self._classes = current
            color = DEFAULT_PALETTE[class_id % len(DEFAULT_PALETTE)]
            self._colors[str(class_id)] = color
            safety_critical = bool(SAFETY_KEYWORD_PATTERN.search(name))
            fine_structure = bool(FINE_STRUCTURE_KEYWORD_PATTERN.search(name))
            db = SessionLocal()
            try:
                state_repo.save_colors_bulk(
                    db,
                    self._dataset_key,
                    self._classes,
                    self._colors,
                    {str(class_id): safety_critical},
                    {str(class_id): fine_structure},
                )
            finally:
                db.close()
            self._safety[str(class_id)] = safety_critical
            self._fine_structure[str(class_id)] = fine_structure
            self._persist_class_names()
            # The class map just changed, so it is a new map - mint the version
            # now rather than waiting for the next dataset load, so a snapshot
            # taken in between pins what is actually on disk.
            self._sync_class_map_version()
            return ClassInfo(
                class_id=class_id,
                name=name,
                color=color,
                safety_critical=safety_critical,
                fine_structure=fine_structure,
            )

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
        states = self._read_states_bulk(list(self._index.keys()))
        items = []
        for image_id, path in self._index.items():
            state = states.get(image_id)
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
    def _read_state(self, image_id: str) -> Optional[dict]:
        db = SessionLocal()
        try:
            return state_repo.get_state(db, self._dataset_key, image_id)
        finally:
            db.close()

    def _read_states_bulk(self, image_ids: list[str]) -> dict[str, dict]:
        db = SessionLocal()
        try:
            return state_repo.get_states_bulk(db, self._dataset_key, image_ids)
        finally:
            db.close()

    def get_saved_states(self, image_ids: list[str]) -> dict[str, dict]:
        """Public bulk read of already-saved annotation state, keyed by
        image_id. Used by triage_service to inspect saved object confidence
        without forcing a fresh detector run per image."""
        return self._read_states_bulk(image_ids)

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
        pre_existing = parse_detection_label_file(label_path)
        if pre_existing:
            # Pre-existing ground-truth-style boxes from labels/*.txt carry
            # no confidence value at all (plain YOLO label files never do) -
            # hence None, not 0.0: "no signal" and "low confidence" are
            # different facts and 0.0 conflated them (see AnnotationObject).
            detections = [(class_id, bbox, None) for class_id, bbox in pre_existing]
        else:
            detections = self._try_auto_detect(path)
        objects = [
            AnnotationObject(
                id=new_id(),
                class_id=class_id,
                class_name=self._classes[class_id] if class_id < len(self._classes) else f"class_{class_id}",
                bbox=bbox,
                detector_confidence=detector_confidence,
                status=ObjectStatus.PENDING,
            )
            for class_id, bbox, detector_confidence in detections
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

    def _try_auto_detect(self, path: Path) -> list[tuple[int, BoundingBox, Optional[float]]]:
        """Fall back to the most recently trained detector for images that
        have no pre-existing detection labels at all."""
        try:
            from app.services.detector_service import get_detector_service

            detector = get_detector_service()
            if not detector.is_active():
                return []
            return detector.detect(path, self._classes)
        except Exception:
            logger.exception("Auto-detect fallback failed for %s", path)
            return []

    def save_annotations(self, annotations: ImageAnnotations) -> None:
        """Persist annotation state for an image (autosave target)."""
        self.require_loaded()
        annotations.last_modified = time.time()
        annotator_id, _ = get_current_annotator()
        db = SessionLocal()
        try:
            state_repo.save_state(
                db,
                self._dataset_key,
                annotations.image_id,
                annotations.model_dump(mode="json"),
                annotations.completed,
                annotator_id,
            )
        finally:
            db.close()

    # -------------------------------------------------------------- progress
    def get_dataset_info(self) -> DatasetInfo:
        self.require_loaded()
        total = len(self._index)
        states = self._read_states_bulk(list(self._index.keys()))
        completed = sum(1 for state in states.values() if state.get("completed"))

        percent = (completed / total * 100.0) if total else 0.0
        eta = self._estimate_eta(total, completed, states)
        return DatasetInfo(
            dataset_path=str(self._images_dir.parent) if self._images_dir else "",
            total_images=total,
            completed=completed,
            remaining=total - completed,
            percent_complete=round(percent, 2),
            classes=self._classes,
            estimated_seconds_remaining=eta,
        )

    def _estimate_eta(self, total: int, completed: int, states: dict[str, dict]) -> Optional[float]:
        """Estimate remaining time from the average gap between completion timestamps."""
        if completed < 2:
            return None
        timestamps = [
            state["last_modified"]
            for state in states.values()
            if state.get("completed") and state.get("last_modified")
        ]
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
    def dataset_key(self) -> str:
        """The resolved dataset root path - the DB key annotation_state,
        annotation_reviews, and dataset_classes are all keyed by."""
        self.require_loaded()
        return self._dataset_key

    @property
    def labels_dir(self) -> Path:
        self.require_loaded()
        return self._labels_dir

    @property
    def images_dir(self) -> Path:
        self.require_loaded()
        return self._images_dir


def get_dataset_service() -> DatasetService:
    """Return the DatasetService for the current request's session (see
    app.session_context) - each session gets its own, lazily built on first use."""
    from app.session_context import get_session_bundle

    bundle = get_session_bundle()
    if bundle.dataset_service is None:
        with bundle.lock:
            if bundle.dataset_service is None:
                from app.config import get_settings

                bundle.dataset_service = DatasetService(get_settings())
    return bundle.dataset_service
