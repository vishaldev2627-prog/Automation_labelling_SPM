"""Detector training and inference.

Fine-tunes a YOLO11 **segmentation** model on whichever annotations you've
reviewed and marked complete (the same trust boundary export already uses),
then runs the most recently trained model on brand-new images that have no
existing labels at all - so boxes/classes it already learned show up
automatically instead of needing everything drawn by hand.

Trains on polygons, not boxes: `_assemble_dataset` writes YOLO-seg label
lines (one per polygon piece, same convention `export_service` already uses
for the handoff snapshot) rather than derived bounding boxes, and the base
checkpoint is a `*-seg.pt` weight. `detect()` returns both a bounding box
*and* the model's own predicted mask polygon per detection - `dataset_service`
uses the polygon to pre-fill an annotation directly, so a confident
detection skips a SAM2 mask-generation pass entirely (that object simply
already has a polygon by the time `generate_all_masks` would have run SAM2
on it). See `MaskSource` in app.models.schemas for how the rest of the
pipeline (auto_accept, in particular) tells that case apart from a SAM2-
produced mask.

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
from typing import Dict, List, Optional, Set, Tuple

from app.config import get_settings
from app.models.schemas import BoundingBox, DetectorInfo, DetectorTrainJobStatus, ObjectStatus, Point
from app.db import SessionLocal
from app.services import golden_eval_service, golden_repo, gpu_scheduler, mlflow_tracking, model_registry_service
from app.services.dataset_service import DatasetNotFoundError, DatasetService
from app.utils.file_utils import atomic_write_json, new_id, read_json
from app.utils.yolo_utils import write_segmentation_label_file

logger = logging.getLogger(__name__)

# YOLO11 segmentation checkpoint - was yolov8s.pt (detection). "s" scale kept
# for parity with the previous base weight; ultralytics infers task="segment"
# from the checkpoint itself, no separate task= flag needed anywhere below.
MODEL_WEIGHTS = "yolo11s-seg.pt"
TASK = "segment"
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
        self._jobs: Dict[str, DetectorTrainJobStatus] = {}
        self._lock = threading.Lock()
        self._loaded_model = None
        self._loaded_model_path: Optional[Path] = None
        # Slugs whose legacy-registry adoption was already evaluated and
        # declined - see _adopt_legacy_registry for why this is cached.
        self._legacy_adoption_declined: Set[str] = set()

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
    def start_training(self, trigger: str = "manual", dataset_snapshot_id: Optional[str] = None) -> DetectorTrainJobStatus:
        """`trigger` is recorded as an MLflow tag only - "manual" (a person
        clicked the button) vs "export_handoff" (M9/W-auto: kicked off
        automatically when a snapshot finalizes, see export_service). Purely
        descriptive; doesn't change how training runs.

        `dataset_snapshot_id` (Module 9 of the class-incremental promotion
        plan - docs/mlflow_class_incremental_architecture.md §G) is only
        ever real for an export_handoff trigger - a manual trigger trains
        directly off live annotation state, with no snapshot behind it at
        all, so this stays None there rather than fabricating a reference
        to something that doesn't exist."""
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
        threading.Thread(
            target=self._run_training, args=(job_id, trigger, dataset_snapshot_id), daemon=True
        ).start()
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

    def _run_training(self, job_id: str, trigger: str = "manual", dataset_snapshot_id: Optional[str] = None) -> None:
        staging_dir = self._models_dir / "training_runs" / job_id
        tracked = False
        run_dir = staging_dir / "run"
        try:
            all_classes = self._ds.get_classes()
            classes = [c.name for c in all_classes]
            trainable_class_ids = {c.class_id for c in all_classes if c.state in ("eligible", "active")}
            data_yaml, num_images = self._assemble_dataset(staging_dir, classes, trainable_class_ids)
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

            # M8 GPU-scheduling guard: "inference always wins" (see
            # gpu_scheduler.py's module docstring for the full reasoning).
            # Only meaningful with an actual GPU to contend over - a
            # CPU-only deployment has nothing to guard against. Checked
            # before the MLflow run even starts, so a skipped job never
            # creates a spurious started-then-abandoned run.
            if device == 0 and not self._wait_for_gpu_idle(job_id, settings):
                self._update(
                    job_id,
                    status="skipped",
                    stage="done",
                    error=(
                        f"Training deferred: GPU still busy serving SAM2 inference after "
                        f"{settings.gpu_wait_max_seconds}s (M8 guard, not a failure)"
                    ),
                )
                return

            # Reset explicitly: if _wait_for_gpu_idle ever moved this job to
            # "waiting_for_gpu", nothing else sets it back before model.train()
            # starts - leaving a stale "waiting" stage on a job that's
            # actually training (confirmed live: a job that had genuinely
            # waited, resumed, and completed all 100 epochs still reported
            # "waiting_for_gpu" the entire time because of this).
            self._update(job_id, stage="training")

            base_weights_path, parent_model_version = self._resolve_base_weights(settings)
            model = YOLO(base_weights_path)

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
                params = {
                    "base_weights": base_weights_path,
                    "task": TASK,
                    "epochs": EPOCHS,
                    "patience": 20,
                    "device": device,
                    "num_images": num_images,
                    "num_classes": len(classes),
                    "class_weight_power": settings.class_weight_power,
                }
                # Module 9 (docs/mlflow_class_incremental_architecture.md
                # §G): class_map_version is always real (every loaded
                # dataset has one); dataset_snapshot_id is only ever real
                # for an export_handoff trigger - a manual trigger trains
                # off live state directly, with no snapshot behind it, so
                # this stays absent rather than fabricated there.
                if self._ds.class_map_version is not None:
                    params["class_map_version"] = self._ds.class_map_version
                if dataset_snapshot_id:
                    params["dataset_snapshot_id"] = dataset_snapshot_id
                # Only set when this run genuinely warm-started from a real
                # Production version - never fabricated on a cold start
                # (docs/mlflow_class_incremental_architecture.md §G: this
                # param and model_registry_service's `baseline_version` tag
                # are usually, but not always, the same version - see that
                # module's own note on why).
                if parent_model_version:
                    params["parent_model_version"] = parent_model_version
                mlflow_tracking.log_params(params)

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
                batch=settings.detector_train_batch_size,
                device=device,
                project=str(staging_dir),
                name="run",
                exist_ok=True,
                verbose=False,
                patience=20,
                # workers=0 - Ultralytics' default (8) forks that many
                # DataLoader worker processes, each passing batches back
                # through /dev/shm. The backend container's shm is Docker's
                # 64MB default (docker-compose has no shm_size override),
                # so a handful of queued batches reliably hit "[Errno 28] No
                # space left on device" - confirmed live, right after the
                # batch-size/OOM fix let training reach this stage. Loading
                # data in the main process avoids the shared-memory
                # dependency entirely rather than growing shm_size, which
                # would compete with the same tight RAM the batch-size fix
                # was just protecting (see detector_train_batch_size).
                workers=0,
                # Module 8 (docs/mlflow_class_incremental_architecture.md
                # §F): Ultralytics' own built-in inverse-class-frequency
                # loss weighting (verified against the installed version's
                # source - DetectionTrainer.set_class_weights, gated by
                # this exact arg, disabled by default at cls_pw=0.0). Keeps
                # a newly-eligible class from being drowned out in early
                # batches purely because it's numerically minor relative to
                # long-shipping classes.
                cls_pw=settings.class_weight_power,
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
                    # Self-describing about what kind of checkpoint this is -
                    # a future registry consumer (or a manually dropped-in
                    # weights file) can be checked against this rather than
                    # assuming every registry.json ever written is a
                    # detection checkpoint.
                    "task": TASK,
                    "base_weights": MODEL_WEIGHTS,
                }
            )
            self._save_registry(registry)

            with self._lock:
                self._loaded_model = None
                self._loaded_model_path = None

            # M6: score the freshly trained model against the golden set, if
            # one exists for this view - closes the loop M5 opened. Without
            # this, a tracked run only shows train/val metrics from a
            # random hash-based split of whatever was completed, never
            # scored against the set nothing trains on. Best-effort and
            # non-fatal: a golden-eval failure must never turn an otherwise
            # successful training run into a reported failure.
            eval_result = None
            if settings.auto_eval_on_golden_set:
                try:
                    db = SessionLocal()
                    try:
                        golden_ids = golden_repo.get_golden_image_ids(db, self._ds.dataset_key)
                    finally:
                        db.close()
                    if golden_ids:
                        try:
                            eval_result = golden_eval_service.evaluate_on_golden_set(
                                self._ds, golden_ids, dest, classes, staging_dir,
                                candidate_class_ids=trainable_class_ids,
                            )
                        except golden_eval_service.InsufficientGoldenCoverageError as exc:
                            # Module 7 (docs/mlflow_class_incremental_
                            # architecture.md §J's flagged operational
                            # prerequisite): a candidate class with zero
                            # golden coverage must never silently produce a
                            # meaningless zero-instance metric. eval_result
                            # stays None, so M7's register_and_recommend
                            # below never runs for this run at all - no
                            # version gets registered, so nothing is even
                            # promotable, which is the safe direction of
                            # error. Tagged on the run so a curator sees
                            # exactly why, rather than a bare skipped eval.
                            logger.warning(
                                "Golden coverage gap for %s: %s", self._ds.dataset_key, exc
                            )
                            if tracked:
                                mlflow_tracking.set_tags(
                                    {"golden_coverage_gap": ",".join(map(str, exc.class_ids))}
                                )
                        if eval_result and tracked:
                            golden_metrics = {
                                f"golden/aggregate_{k}": v for k, v in eval_result["aggregate"].items()
                            }
                            for class_id, m in eval_result["per_class"].items():
                                golden_metrics.update(
                                    {f"golden/class{class_id}_{k}": v for k, v in m.items()}
                                )
                            mlflow_tracking.log_metrics(golden_metrics, step=EPOCHS)
                            mlflow_tracking.set_tags(
                                {
                                    "golden_eval_images": str(eval_result["num_golden_images"]),
                                    # Load-bearing for anyone reading historical
                                    # runs later: golden/class{N}_mAP50 means
                                    # *mask* AP from here on, not box AP as it
                                    # did for every yolov8s.pt run before this -
                                    # see golden_eval_service's module docstring.
                                    "eval_task": TASK,
                                }
                            )
                    else:
                        logger.info(
                            "No golden set yet for %s; skipping M6 eval for this run", self._ds.dataset_key
                        )
                except Exception:
                    logger.exception("Golden-set evaluation failed; training result is unaffected")

            if tracked:
                # Logged before the finally-block rmtree below deletes
                # run_dir entirely - results.csv, PR curve, confusion
                # matrix, args.yaml all used to be discarded unread here.
                mlflow_tracking.log_artifacts(run_dir)
                mlflow_tracking.set_tags({"detector_version": str(new_version)})

                # M7 (recommendation-only - see model_registry_service's own
                # module docstring for why nothing here auto-promotes, or
                # changes what registry.json-backed detect() actually
                # serves). Only meaningful once a golden-set eval exists to
                # base a recommendation on.
                if eval_result:
                    run_id = mlflow_tracking.current_run_id()
                    if run_id:
                        model_size_mb = dest.stat().st_size / (1024 * 1024)
                        latency_p95_ms = self._measure_inference_latency_ms(model)
                        model_registry_service.register_and_recommend(
                            _slug_for(self._ds.dataset_key),
                            run_id,
                            eval_result["per_class"],
                            self._ds.get_classes(),
                            model_size_mb=model_size_mb,
                            latency_p95_ms=latency_p95_ms,
                            dataset_snapshot_id=dataset_snapshot_id,
                            class_map_version=self._ds.class_map_version,
                        )

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

    def _wait_for_gpu_idle(self, job_id: str, settings) -> bool:
        """Block until SAM2 inference goes idle, polling every
        `gpu_wait_poll_seconds`, up to `gpu_wait_max_seconds` total. Returns
        False on timeout - the caller marks the job "skipped", not "failed":
        nothing about the data or the training setup was wrong, the GPU
        just stayed busy with live annotator traffic for longer than this
        job was willing to wait.
        """
        waited = 0
        while gpu_scheduler.is_gpu_busy():
            if waited >= settings.gpu_wait_max_seconds:
                logger.warning(
                    "Training job %s deferred: GPU still busy with SAM2 inference after %ds",
                    job_id, waited,
                )
                return False
            self._update(job_id, stage="waiting_for_gpu")
            time.sleep(settings.gpu_wait_poll_seconds)
            waited += settings.gpu_wait_poll_seconds
        return True

    def _resolve_base_weights(self, settings) -> Tuple[str, Optional[str]]:
        """Module 8 of the class-incremental promotion plan (docs/
        mlflow_class_incremental_architecture.md §F/§K item 7): warm-start
        from the current Production version's weights instead of always
        the stock checkpoint, so a retrain is a genuine continuation of
        what already ships, not a from-scratch relearn of the old classes
        alongside the new one.

        Falls back to the stock checkpoint on ANY failure - no Production
        version yet (the expected case for a view's first-ever run, logged
        at info not warning), MLflow unreachable, or a download failure -
        since this is an optimization, never a correctness requirement,
        and must never block training.

        Returns (weights_path_or_name, parent_model_version) - the second
        is None on a cold start, so callers never fabricate lineage where
        none exists.
        """
        if not mlflow_tracking.is_configured(settings):
            return MODEL_WEIGHTS, None
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            client = MlflowClient()
            name = model_registry_service.registered_model_name(_slug_for(self._ds.dataset_key))
            production = client.get_latest_versions(name, stages=["Production"])
            if not production:
                logger.info("No Production version yet for %s; training from stock %s", name, MODEL_WEIGHTS)
                return MODEL_WEIGHTS, None
            prod_version = production[0]
            downloaded = mlflow.artifacts.download_artifacts(
                run_id=prod_version.run_id, artifact_path="weights/best.pt"
            )
            logger.info("Warm-starting %s from Production v%s (%s)", name, prod_version.version, downloaded)
            return downloaded, prod_version.version
        except Exception:
            logger.exception(
                "Could not warm-start from Production weights for %s; training from stock %s instead",
                self._ds.dataset_key, MODEL_WEIGHTS,
            )
            return MODEL_WEIGHTS, None

    @staticmethod
    def _measure_inference_latency_ms(model) -> float:
        """A rough p95 stand-in for promotion_gate.py's Layer 4 operational
        check (docs/mlflow_class_incremental_architecture.md §G/§H) - a few
        timed predictions on the same kind of synthetic probe image
        `model_promotion_service._sanity_check` already uses before
        activation, reusing the just-trained `model` object already in
        memory rather than reloading from disk. Best-effort: any failure
        here returns 0.0 (never blocks registration) - see the caller's own
        try/except around the whole M7 block."""
        import numpy as np

        probe = np.zeros((640, 640, 3), dtype=np.uint8)
        samples = []
        for _ in range(5):
            t0 = time.time()
            model.predict(probe, verbose=False)
            samples.append((time.time() - t0) * 1000)
        samples.sort()
        # 5 samples: index 4 (the slowest) stands in for p95 at this sample
        # size - a real p95 needs far more than 5 draws, this is a cheap
        # per-training-run proxy, not a statistically rigorous measurement.
        return samples[-1]

    def _assemble_dataset(
        self, staging_dir: Path, classes: List[str], trainable_class_ids: Set[int]
    ) -> Tuple[Path, int]:
        """Write a fresh YOLO-**segmentation** dataset from images you've
        reviewed and marked complete - the same trust boundary export()
        already uses, so the detector only ever learns from annotations a
        human has approved.

        One label line per polygon piece, not one per object - the same
        convention export_service._write_dataset uses for the handoff
        snapshot, so a fine_structure class's disjoint pieces (a branching
        crack) all contribute rather than only the largest surviving.

        `trainable_class_ids` gates class eligibility (Module 3 of the
        class-incremental promotion plan - docs/mlflow_class_incremental_
        architecture.md §D/§E): only objects whose class_id is in this set
        (state in {eligible, active}) ever produce a label line. A
        discovered/collecting_data/deprecated class's objects are simply
        dropped from the label file for that image - the image itself is
        NOT excluded just because it also contains an untrainable class's
        object; whatever trainable content it has still trains normally.
        `classes`/`data.yaml`'s `names`/`nc` stay the FULL, real, unfiltered
        class map (Ultralytics requires len(names) == nc; verified this
        does not require every id to actually appear in a label file, so
        this needs no renumbering and no placeholder-name scheme - a
        detection's class_id in eval/inference output is therefore always
        the same real global id annotators see, never repositioned)."""
        excluded_class_ids = {i for i in range(len(classes)) if i not in trainable_class_ids}
        if excluded_class_ids:
            logger.info(
                "Training excludes class id(s) %s (not yet eligible/active) for %s",
                sorted(excluded_class_ids), self._ds.dataset_key,
            )
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
                live = [
                    o for o in annotations.objects
                    if o.status != ObjectStatus.REJECTED and o.class_id in trainable_class_ids
                ]
                objects = [
                    (o.class_id, piece)
                    for o in live
                    for piece in [o.polygon, *o.extra_polygons]
                    if len(piece) >= 3
                ]
                # An empty frame is included only when a human confirmed it is
                # empty (see ImageAnnotations) - then it's a background/negative
                # sample, written as an empty label file, which is what
                # ultralytics expects and which teaches the pre-labeler where
                # *not* to propose a mask. An unconfirmed empty frame is still
                # skipped: it just means nobody has annotated it yet. An object
                # with a status but no usable polygon (<3 points) contributes
                # nothing and is treated the same as "no objects" here.
                if not objects and not annotations.no_objects_confirmed:
                    continue
                src_image = self._ds.get_image_path(image_id)
                shutil.copy2(src_image, img_dir / src_image.name)
                write_segmentation_label_file(lbl_dir / f"{image_id}.txt", objects)
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

    def detect(
        self, image_path: Path, classes: List[str]
    ) -> List[Tuple[int, BoundingBox, float, List[Point]]]:
        """Run the most recently trained detector on an image with no
        pre-existing labels, returning (class_id, bbox, confidence, polygon)
        tuples - confidence is ultralytics' own box.conf, otherwise
        discarded here the same way it always was upstream of
        parse_detection_label_file (plain YOLO label files carry no
        confidence field at all). Used by the Phase 2 triage service as the
        only confidence signal available before the pipeline supplies its
        own (see Q-E, build plan §6).

        `polygon` is the segmentation model's own predicted mask contour
        (ultralytics' `result.masks.xyn`, already extracted and normalized
        0-1) - empty unless `confidence` clears `mask_confidence_threshold`,
        the same "never show a bad mask" bar `mask_generation_service`
        applies to SAM2's own output. `dataset_service.get_annotations` uses
        a non-empty polygon here to pre-fill the annotation directly
        (Option B) instead of leaving the mask for SAM2 to generate; an
        empty polygon (low confidence, or a checkpoint with no seg head)
        falls back to exactly the box-only behavior this had before."""
        if not self.is_active():
            return []
        model = self._ensure_model_loaded()
        results = model.predict(str(image_path), conf=0.25, verbose=False)
        if not results:
            return []
        result = results[0]
        h, w = result.orig_shape
        mask_threshold = get_settings().mask_confidence_threshold
        # None for a detection-only checkpoint (no seg head) - a leftover
        # yolov8s.pt-era registry.json is still loadable, just without polygons.
        mask_polys = result.masks.xyn if result.masks is not None else None
        detections: List[Tuple[int, BoundingBox, float, List[Point]]] = []
        for idx, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            if class_id >= len(classes):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf.item())
            polygon: List[Point] = []
            if mask_polys is not None and confidence > mask_threshold:
                xy = mask_polys[idx]
                if len(xy) >= 3:
                    polygon = [Point(x=float(px), y=float(py)) for px, py in xy]
            detections.append(
                (
                    class_id,
                    BoundingBox(
                        x_center=((x1 + x2) / 2) / w,
                        y_center=((y1 + y2) / 2) / h,
                        width=(x2 - x1) / w,
                        height=(y2 - y1) / h,
                    ),
                    confidence,
                    polygon,
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
