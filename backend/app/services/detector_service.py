"""Detector training and inference.

Fine-tunes a YOLOv8 detection model on whichever annotations you've reviewed
and marked complete (the same trust boundary export already uses), then runs
the most recently trained model on brand-new images that have no existing
labels at all - so boxes/classes it already learned show up automatically
instead of needing everything drawn by hand.

Registry and weights are scoped **per dataset view**, not per models_dir:
DetectorService is session-scoped (see app.session_context) and each session
can have a different view loaded, but models_dir is one shared host path. A
single global registry therefore let a detector trained while `buffer` was
loaded become the active auto-detect model for `side_view` annotators too -
with buffer's class list, so class ids meant different components in the two
views. Scoping by `dataset_service.dataset_key` makes that collision
structurally impossible instead of relying on nobody training two views in
the same afternoon.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.models.schemas import BoundingBox, DetectorInfo, DetectorTrainJobStatus, ObjectStatus
from app.services import mlflow_tracking
from app.services.dataset_service import DatasetNotFoundError, DatasetService
from app.utils.file_utils import atomic_write_json, new_id, read_json

logger = logging.getLogger(__name__)

MODEL_WEIGHTS = "yolov8s.pt"
EPOCHS = 100
VAL_SPLIT = 0.1
MIN_TRAINING_IMAGES = 2

# Per-view registries/weights live under this subdirectory of models_dir. The
# pre-existing global `detector_registry.json` + `detector_v*.pt` at the top
# level are deliberately left untouched (see _adopt_legacy_registry).
DETECTORS_SUBDIR = "detectors"
LEGACY_REGISTRY_NAME = "detector_registry.json"

_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug_for(dataset_key: str) -> str:
    """A filesystem-safe, collision-resistant directory name for a dataset key.

    `dataset_key` is currently a resolved absolute path (see
    annotation_state_repo's module docstring), so the basename alone isn't
    unique across two roots with the same last segment - hence the short
    hash suffix. Keeping the readable part first means an operator can still
    tell which directory belongs to which view by looking at it.
    """
    digest = hashlib.sha1(dataset_key.encode("utf-8")).hexdigest()[:10]
    readable = _SLUG_UNSAFE.sub("_", Path(dataset_key).name).strip("_") or "dataset"
    return f"{readable}-{digest}"


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
        # Slugs whose legacy-registry adoption was already evaluated and
        # declined - see _adopt_legacy_registry for why this is cached.
        self._legacy_adoption_declined: set[str] = set()

    # ------------------------------------------------------------- registry
    @property
    def _view_dir(self) -> Path:
        """Per-dataset-view directory holding this view's registry + weights.

        Raises DatasetNotFoundError (via dataset_key) if no dataset is
        loaded - there is no such thing as "the active detector" without a
        view to scope it to. Callers that must not fail on that
        (is_active/get_info) handle it explicitly.
        """
        return self._models_dir / DETECTORS_SUBDIR / _slug_for(self._ds.dataset_key)

    @property
    def _registry_path(self) -> Path:
        return self._view_dir / "registry.json"

    def _load_registry(self) -> dict:
        registry = read_json(self._registry_path, default={})
        if not registry.get("version"):
            registry = self._adopt_legacy_registry()
        return registry

    def _adopt_legacy_registry(self) -> dict:
        """Migrate the pre-per-view global registry into this view - but only
        when it provably belongs to this view.

        The old global `models_dir/detector_registry.json` recorded a
        `classes` list but not which view produced it, so adopting it blindly
        would recreate exactly the cross-view mix-up this scoping exists to
        prevent. Adopt only on an exact class-list match with this view's
        current class names; otherwise leave it alone and log, so an operator
        can see why their previously-active detector didn't carry over
        instead of silently getting no auto-detect.

        Never moves or deletes the legacy files - they stay where they are as
        a fallback if this adoption turns out to be wrong.

        The per-view "already decided not to adopt" set exists because
        is_active() runs on every image open (dataset_service._try_auto_detect);
        without it, a non-adoptable legacy registry would re-stat the file and
        re-log the same warning once per image.
        """
        try:
            slug = _slug_for(self._ds.dataset_key)
        except DatasetNotFoundError:
            return {}
        if slug in self._legacy_adoption_declined:
            return {}

        legacy_path = self._models_dir / LEGACY_REGISTRY_NAME
        legacy = read_json(legacy_path, default={})
        if not legacy.get("version"):
            self._legacy_adoption_declined.add(slug)
            return {}

        try:
            current_classes = [c.name for c in self._ds.get_classes()]
        except DatasetNotFoundError:
            return {}

        if list(legacy.get("classes") or []) != current_classes:
            logger.warning(
                "Legacy global detector registry at %s not adopted for view %s: its class list "
                "does not match this view's classes, so which view trained it can't be established. "
                "Retrain to get an active detector for this view.",
                legacy_path,
                self._ds.dataset_key,
            )
            self._legacy_adoption_declined.add(slug)
            return {}

        weights = Path(legacy.get("path", ""))
        if not weights.exists():
            logger.warning(
                "Legacy detector registry at %s points at missing weights %s; not adopted.",
                legacy_path,
                weights,
            )
            self._legacy_adoption_declined.add(slug)
            return {}

        adopted = dict(legacy)
        adopted["adopted_from_legacy_registry"] = str(legacy_path)
        self._save_registry(adopted)
        logger.info(
            "Adopted legacy global detector registry into view %s (class list matched exactly).",
            self._ds.dataset_key,
        )
        return adopted

    def _save_registry(self, data: dict) -> None:
        atomic_write_json(self._registry_path, data)

    def is_active(self) -> bool:
        try:
            return bool(self._load_registry().get("version"))
        except DatasetNotFoundError:
            return False

    def get_info(self) -> DetectorInfo:
        try:
            reg = self._load_registry()
        except DatasetNotFoundError:
            return DetectorInfo(active=False)
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
    def start_training(self, trigger: str = "manual") -> DetectorTrainJobStatus:
        """`trigger` is recorded as an MLflow tag only - "manual" (a person
        clicked the button) vs "export_handoff" (M9/W-auto: kicked off
        automatically when a snapshot finalizes, see export_service). Purely
        descriptive; doesn't change how training runs."""
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
        threading.Thread(target=self._run_training, args=(job_id, trigger), daemon=True).start()
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

    def _run_training(self, job_id: str, trigger: str = "manual") -> None:
        staging_dir = self._models_dir / "training_runs" / job_id
        tracked = False
        run_dir = staging_dir / "run"
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

            settings = get_settings()

            # ultralytics ships its own built-in MLflow integration, auto-
            # enabled the moment `mlflow` is importable (which it now always
            # is - mlflow-skinny is a hard requirement, see requirements.txt).
            # It reads MLFLOW_TRACKING_URI from os.environ directly, which
            # this app's Settings does NOT populate (pydantic-settings parses
            # .env into the Settings object only, never exports it to the
            # process environment) - so left alone, ultralytics' callback
            # can't see the same URI mlflow_tracking.start() below configures,
            # falls back to a local file-store path, and calls
            # mlflow.set_tracking_uri() with that fallback mid-training.
            # That's global module-level state: it silently redirects every
            # subsequent log_metrics call - ours included - away from the
            # real server.
            #
            # Fixed by exporting the same URI into os.environ so ultralytics'
            # callback resolves to the identical server we already configured
            # - not by touching ultralytics' own SETTINGS (a persisted,
            # per-*user* JSON file at ~/.config/Ultralytics/settings.json,
            # shared with any other ultralytics usage on this host outside
            # this app entirely; flipping that off here would be exactly the
            # kind of unrelated-system side effect this project avoids).
            # ultralytics then finds our run already active via
            # mlflow.active_run() and logs into the same one rather than
            # starting its own.
            if mlflow_tracking.is_configured(settings):
                os.environ["MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri
                os.environ["MLFLOW_EXPERIMENT_NAME"] = settings.mlflow_experiment_name

            device = 0 if torch.cuda.is_available() else "cpu"
            model = YOLO(MODEL_WEIGHTS)

            # M5 (Scope A only - see mlflow_tracking's module docstring):
            # unconditionally attempted, never blocks training if MLflow is
            # unreachable - see mlflow_tracking.start's own contract.
            tracked = mlflow_tracking.start(
                settings,
                run_name=f"detector-{job_id}",
                tags={
                    "dataset_key": self._ds.dataset_key,
                    "job_id": job_id,
                    "mode": "full_retrain",
                    "trigger": trigger,
                },
            )
            if tracked:
                mlflow_tracking.log_params(
                    {
                        "base_weights": MODEL_WEIGHTS,
                        "epochs": EPOCHS,
                        "patience": 20,
                        "device": device,
                        "num_images": num_images,
                        "num_classes": len(classes),
                    }
                )

            def on_epoch_end(trainer) -> None:
                try:
                    self._update(job_id, current_epoch=int(trainer.epoch) + 1)
                except Exception:
                    logger.exception("Failed to record training epoch progress")

            def on_fit_epoch_end(trainer) -> None:
                if tracked and trainer.metrics:
                    # ultralytics' own metric keys carry parentheses, e.g.
                    # "metrics/precision(B)" - MLflow's REST API rejects
                    # those outright ("Names may only contain alphanumerics,
                    # underscores, dashes, periods, spaces and slashes"),
                    # failing every single log_metrics call otherwise (caught
                    # live: 100/100 epochs errored before this was added,
                    # silently absorbed by mlflow_tracking's best-effort
                    # contract so training itself never noticed).
                    sanitized = {k.replace("(", "").replace(")", ""): v for k, v in trainer.metrics.items()}
                    mlflow_tracking.log_metrics(sanitized, step=int(trainer.epoch))

            model.add_callback("on_train_epoch_end", on_epoch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
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
            best_weights = run_dir / "weights" / "best.pt"
            if not best_weights.exists():
                raise RuntimeError("Training finished but no best.pt weights file was produced")

            registry = self._load_registry()
            new_version = registry.get("version", 0) + 1
            dest = self._view_dir / f"detector_v{new_version}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_weights, dest)
            registry.update(
                {
                    "version": new_version,
                    "trained_at": time.time(),
                    "num_images": num_images,
                    "classes": classes,
                    "path": str(dest),
                    # Recorded so a registry file is self-describing about
                    # which view's annotations produced it - the thing the
                    # old global registry could not answer.
                    "dataset_key": self._ds.dataset_key,
                }
            )
            self._save_registry(registry)

            with self._lock:
                self._loaded_model = None
                self._loaded_model_path = None

            if tracked:
                # Logged before the finally-block rmtree below deletes
                # run_dir entirely - results.csv, PR curve, confusion
                # matrix, args.yaml all used to be discarded unread here.
                mlflow_tracking.log_artifacts(run_dir)
                mlflow_tracking.set_tags({"detector_version": str(new_version)})
                mlflow_tracking.end(status="FINISHED")
                tracked = False

            self._update(job_id, status="completed", stage="done", current_epoch=EPOCHS)
        except Exception as exc:
            logger.exception("Detector training failed")
            self._update(job_id, status="failed", error=str(exc))
            if tracked:
                mlflow_tracking.set_tags({"error": str(exc)})
                mlflow_tracking.end(status="FAILED")
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
                # An empty frame is included only when a human confirmed it is
                # empty (see ImageAnnotations) - then it's a background/negative
                # sample, written as an empty label file, which is what
                # ultralytics expects and which teaches the pre-labeler where
                # *not* to propose boxes. An unconfirmed empty frame is still
                # skipped: it just means nobody has annotated it yet.
                if not objects and not annotations.no_objects_confirmed:
                    continue
                src_image = self._ds.get_image_path(image_id)
                shutil.copy2(src_image, img_dir / src_image.name)
                lines = [
                    f"{o.class_id} {o.bbox.x_center:.6f} {o.bbox.y_center:.6f} {o.bbox.width:.6f} {o.bbox.height:.6f}"
                    for o in objects
                ]
                # Truly empty, not a single blank line - a stray "\n" is a
                # malformed label line to some YOLO loaders.
                (lbl_dir / f"{image_id}.txt").write_text(
                    "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
                )
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

    def detect(self, image_path: Path, classes: list[str]) -> list[tuple[int, BoundingBox, float]]:
        """Run the most recently trained detector on an image with no
        pre-existing labels, returning (class_id, bbox, confidence) tuples -
        confidence is ultralytics' own box.conf, otherwise discarded here
        the same way it always was upstream of parse_detection_label_file
        (plain YOLO label files carry no confidence field at all). Used by
        the Phase 2 triage service as the only confidence signal available
        before the pipeline supplies its own (see Q-E, build plan §6)."""
        if not self.is_active():
            return []
        model = self._ensure_model_loaded()
        results = model.predict(str(image_path), conf=0.25, verbose=False)
        if not results:
            return []
        result = results[0]
        h, w = result.orig_shape
        detections: list[tuple[int, BoundingBox, float]] = []
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
                    float(box.conf.item()),
                )
            )
        return detections


def get_detector_service() -> DetectorService:
    """Session-scoped (see app.session_context) - a detector trained/active
    while one dataset view is loaded stays scoped to that session/view."""
    from app.session_context import get_session_bundle

    bundle = get_session_bundle()
    if bundle.detector_service is None:
        with bundle.lock:
            if bundle.detector_service is None:
                from app.config import get_settings
                from app.services.dataset_service import get_dataset_service

                settings = get_settings()
                bundle.detector_service = DetectorService(get_dataset_service(), settings.models_dir)
    return bundle.detector_service
