"""Exports reviewed annotations to a ready-to-train YOLO segmentation dataset."""
from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

import yaml

from app.config import get_settings
from app.db import SessionLocal
from app.models.schemas import (
    CoachType,
    ExportRequest,
    ImageAnnotations,
    ObjectSource,
    ObjectStatus,
)
from app.services import annotation_state_repo as state_repo
from app.services import golden_repo, object_store, review_service, snapshot_repo, snapshot_service
from app.services.dataset_service import DatasetService
from app.services.polygon_service import polygon_to_mask
from app.session_context import get_current_annotator
from app.utils.yolo_utils import write_segmentation_label_file

logger = logging.getLogger(__name__)

VAL_SPLIT = 0.1


class SplitIntegrityViolation(Exception):
    """Raised when a snapshot would ship pseudo/synthetic or auto-accepted
    labels outside train, despite M3's forcing pass. Should never trigger in
    practice - _write_dataset forces those images into train before this is
    ever checked - but a snapshot is an immutable, MLflow-tracked handoff, so
    this is a hard refusal rather than a logged warning if it somehow does."""


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

    @staticmethod
    def _enforce_split_integrity(
        natural_split: str, propagated_here: int, is_auto_accepted: bool
    ) -> tuple[str, bool]:
        """M3: pseudo/synthetic (propagated) and auto-accepted labels must
        never land in val/test (pipeline.md's retrain rule, via
        FINAL_AIML §12: "pseudo/synthetic -> train split only, never
        valid/test"). Returns the (possibly overridden) split and whether an
        override happened, so the caller can log it loudly - a re-split
        changes what a previously-exported split assignment meant for that
        image, and that must never happen silently.

        Pulled out as its own pure function - deliberately independent of
        DatasetService/Postgres - so the enforcement rule itself is directly
        testable without standing up either.
        """
        if natural_split == "train":
            return natural_split, False
        if propagated_here > 0 or is_auto_accepted:
            return "train", True
        return natural_split, False

    def _write_condition_crops(
        self, out_dir: Path, image_id: str, annotations: ImageAnnotations, margin_pct: float
    ) -> int:
        """Write one crop per condition-assessed object, foldered by condition.

        This is the training artifact for the `p1_side_damage` crop classifier
        (`docs/pipeline.md` §5.2: "Input: a component crop (from a detector box)
        + margin"). Folder-per-class is what YOLO11-cls and the usual
        EfficientNet pipelines expect, so the layout is:

            crops/<split>/<condition>/<image_id>__<object_id>.png

        Only objects with a condition actually set are written. An unassessed
        object (`condition is None`) is skipped rather than treated as `ok` -
        see Condition on why that direction matters.

        PNG, not JPEG: C3 mandates lossless throughout, and a condition like
        `fiba_red` or a hairline `securing_broken` is exactly the kind of signal
        JPEG chroma subsampling degrades.
        """
        assessed = [
            obj
            for obj in annotations.objects
            if obj.status != ObjectStatus.REJECTED and obj.condition is not None
        ]
        if not assessed:
            return 0

        # Note the split passed in is the image-level split, deliberately: two
        # crops from the same frame must never straddle train and val, or the
        # classifier is evaluated on near-copies of what it trained on.

        import cv2

        from app.utils.image_utils import read_image_bgr

        try:
            image = read_image_bgr(self._ds.get_image_path(image_id))
        except Exception:
            logger.exception("Could not read %s for condition crops; skipping its crops", image_id)
            return 0

        img_h, img_w = image.shape[:2]
        written = 0
        for obj in assessed:
            x1, y1, x2, y2 = obj.bbox.to_xyxy(img_w, img_h)
            # Expand by the margin, then clamp - a component at the frame edge
            # gets a smaller margin rather than a shifted or wrapped crop.
            margin_x = int(round((x2 - x1) * margin_pct / 100.0))
            margin_y = int(round((y2 - y1) * margin_pct / 100.0))
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(img_w, x2 + margin_x)
            y2 = min(img_h, y2 + margin_y)
            if x2 - x1 < 2 or y2 - y1 < 2:
                logger.warning(
                    "Degenerate crop for object %s in %s (%dx%d); skipped",
                    obj.id, image_id, x2 - x1, y2 - y1,
                )
                continue

            crop_dir = out_dir / obj.condition.value
            crop_dir.mkdir(parents=True, exist_ok=True)
            path = crop_dir / f"{image_id}__{obj.id}.png"
            if not cv2.imwrite(str(path), image[y1:y2, x1:x2]):
                logger.error("Failed to write condition crop %s", path)
                continue
            written += 1
        return written

    def _fine_structure_masks_by_class(self, annotations: ImageAnnotations) -> dict[int, np.ndarray]:
        """Rasterize every `fine_structure` class present in this image into
        one unioned 0/1 mask per class id, at native resolution. Shared by
        `_write_fine_structure_masks` (W-3) and `_write_wheel_unwraps` (W-5) -
        both need the same per-class raster, just written to different
        places (a raw-space PNG vs. the input to the log-polar transform).

        One mask per (image, class), not per object: the pipeline trains
        segmentation per class, and two cracks on one frame are the same
        class. Pieces are unioned via OR-accumulation rather than one
        fillPoly call with all pieces, since a single call would treat them
        as one path and overlapping pieces would cancel out under the
        even-odd rule.
        """
        import numpy as np

        by_class: dict[int, list[list]] = {}
        for obj in annotations.objects:
            if obj.status == ObjectStatus.REJECTED:
                continue
            if not self._ds.is_fine_structure(obj.class_id):
                continue
            for piece in [obj.polygon, *obj.extra_polygons]:
                if len(piece) >= 3:
                    by_class.setdefault(obj.class_id, []).append(piece)

        masks: dict[int, np.ndarray] = {}
        for class_id, pieces in by_class.items():
            mask = np.zeros((annotations.height, annotations.width), dtype=np.uint8)
            for piece in pieces:
                mask |= polygon_to_mask(piece, annotations.width, annotations.height).astype(np.uint8)
            masks[class_id] = mask
        return masks

    def _write_fine_structure_masks(
        self, out_dir: Path, image_id: str, annotations: ImageAnnotations
    ) -> int:
        """Write one binary mask PNG per `fine_structure` class present in this
        image, at the image's native resolution. Returns how many were written.

        Why rasters at all, when a polygon is already exported: crack/corrosion
        are scored on Dice/IoU **plus length-recall** and shelling on Dice plus
        length (docs/pipeline.md §5.4, §5.5). A polygon is a lossy
        representation of a thin branching shape even with simplification off -
        it cannot express a hole, and a closed ring around a hairline crack
        encodes its outline rather than its extent. The raster is the faithful
        artifact; the polygon stays for the tooling that expects YOLO-seg text.
        """
        by_class = self._fine_structure_masks_by_class(annotations)
        if not by_class:
            return 0

        import cv2

        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for class_id, mask in by_class.items():
            # 0/255 rather than 0/1 so the PNG is viewable by a human checking
            # it, and still loads as a binary mask after a >0 threshold.
            path = out_dir / f"{image_id}__class{class_id}.png"
            if not cv2.imwrite(str(path), mask * 255):
                logger.error("Failed to write mask raster %s", path)
                continue
            written += 1
        return written

    def _write_wheel_unwraps(self, out_dir: Path, image_id: str, annotations: ImageAnnotations) -> int:
        """Generate the log-polar unwrap for one wheel image, from raw-space
        masks + current (provisional) unwrap parameters (W-5, D-Q5).

        Writes `<out_dir>/images/<image_id>.png` (the unwrapped strip cut
        from the raw frame) and one `<out_dir>/masks/<image_id>__class{id}.png`
        per fine_structure class present - mirroring `_write_fine_structure_masks`'s
        raw-space layout, just unwrapped. Returns 1 if an unwrap was written,
        0 if skipped (no annotated wheel object, or no fine_structure content
        to unwrap - an unwrap with nothing on it isn't useful training data).

        Deliberately re-derives the circle from this image's own annotations
        every export rather than caching it anywhere: the whole point of W-5
        is that the transform is regenerated from raw ground truth each time,
        so a config change (or a corrected wheel-class annotation) takes
        effect on the very next export with nothing to invalidate.
        """
        import cv2
        import numpy as np

        from app.services import log_polar_service
        from app.utils.image_utils import read_image_bgr

        settings = get_settings()
        circle = log_polar_service.find_wheel_circle(
            annotations.objects,
            annotations.width,
            annotations.height,
            settings.wheel_class_name,
            settings.wheel_unwrap_radius_padding_pct,
        )
        if circle is None:
            return 0

        fine_structure_masks = self._fine_structure_masks_by_class(annotations)
        if not fine_structure_masks:
            return 0

        try:
            image = read_image_bgr(self._ds.get_image_path(image_id))
        except Exception:
            logger.exception("Could not read %s for wheel unwrap; skipping", image_id)
            return 0

        out_width = settings.wheel_unwrap_output_width
        out_height = settings.wheel_unwrap_output_height
        log_scale = settings.wheel_unwrap_log_scale

        images_dir = out_dir / "images"
        masks_dir = out_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_image = log_polar_service.unwrap_log_polar(image, circle, out_width, out_height, log_scale)
        cv2.imwrite(str(images_dir / f"{image_id}.png"), unwrapped_image)

        for class_id, mask in fine_structure_masks.items():
            unwrapped_mask = log_polar_service.unwrap_log_polar(
                mask.astype(np.uint8) * 255, circle, out_width, out_height, log_scale
            )
            cv2.imwrite(str(masks_dir / f"{image_id}__class{class_id}.png"), unwrapped_mask)

        return 1

    def export(self, request: ExportRequest) -> dict:
        """Write a dataset, as an immutable content-addressed snapshot by default.

        Snapshot mode builds into a staging directory, hashes the resulting file
        set, and moves it to `snapshots/<snapshot_id>/` - so a re-export of
        unchanged data resolves to the same snapshot instead of overwriting the
        previous one. `as_snapshot=false` keeps the old in-place write for a quick
        look at a mutable directory.
        """
        self._ds.require_loaded()

        if not request.as_snapshot:
            dest = (
                self._exports_dir / request.output_subdir if request.output_subdir else self._exports_dir
            )
            stats = self._write_dataset(request, dest)
            stats["output_dir"] = str(dest)
            stats["snapshot_id"] = None
            return stats

        snapshot_service.cleanup_stale_staging(self._exports_dir)
        staging = snapshot_service.new_staging_dir(self._exports_dir)
        try:
            stats = self._write_dataset(request, staging)
            return self._finalize_snapshot(request, staging, stats)
        except Exception:
            # A half-built staging dir is worse than none: it would be picked up
            # by nothing and cleaned up only by the age sweep.
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _finalize_snapshot(self, request: ExportRequest, staging: Path, stats: dict) -> dict:
        settings = get_settings()

        # M3: a snapshot is the immutable, MLflow-tracked handoff artifact - it
        # must never claim a split-integrity compliance it does not have.
        # _write_dataset already forces propagated/auto-accepted images into
        # train, so this should always be a no-op; it is a hard refusal rather
        # than the warning-only path M2 shipped with, in case that forcing pass
        # is ever bypassed or a future violation type isn't force-corrected.
        if not stats.get("split_integrity_ok", True):
            raise SplitIntegrityViolation(
                f"Refusing to finalize snapshot: split-integrity violations "
                f"{stats.get('split_integrity')}"
            )

        file_index = snapshot_service.build_file_index(staging)
        snapshot_id = snapshot_service.compute_snapshot_id(file_index, self._ds.class_map_hash)
        snapshot_service.write_checksums(staging, file_index)

        manifest = snapshot_service.build_manifest(
            snapshot_id=snapshot_id,
            dataset_key=self._ds.dataset_key,
            class_map_version=self._ds.class_map_version,
            class_map_hash=self._ds.class_map_hash,
            class_names=[c.name for c in self._ds.get_classes()],
            exclude_classes=settings.exclude_class_list,
            label_format={
                c.class_id: ("mask_raster+polygon" if c.fine_structure else "polygon")
                for c in self._ds.get_classes()
            },
            file_index=file_index,
            stats=stats,
        )
        snapshot_service.write_manifest(staging, manifest)

        snapshot_dir, created = snapshot_service.finalize(staging, self._exports_dir, snapshot_id)
        if not created:
            # Identical content already snapshotted. The manifest inside the
            # existing directory stays authoritative - it describes when that
            # snapshot was actually made, and rewriting it would make an
            # immutable artifact mutable.
            manifest = snapshot_service.read_manifest(snapshot_dir) or manifest

        annotator_id, _ = get_current_annotator()
        db = SessionLocal()
        try:
            record = snapshot_repo.upsert_snapshot(
                db,
                snapshot_id=snapshot_id,
                dataset_view=self._ds.dataset_key,
                class_map_version=self._ds.class_map_version,
                class_map_hash=self._ds.class_map_hash,
                manifest=manifest,
                local_path=str(snapshot_dir),
                file_count=len(file_index),
                total_bytes=sum(size for _r, _d, size in file_index),
                annotator_id=annotator_id,
            )
            publish_info = {"published": False, "error": None}
            if request.publish:
                result = object_store.publish_snapshot(settings, snapshot_dir, snapshot_id)
                publish_info = result.as_dict()
                if result.published:
                    snapshot_repo.mark_published(
                        db, snapshot_id, f"s3://{result.bucket}/{result.prefix}"
                    )
            elif record.published_at is not None:
                publish_info = {"published": True, "uri": record.published_uri, "error": None}
        finally:
            db.close()

        # M9/auto-trigger: a genuinely new snapshot is a handoff event - kick
        # off a Scope A training run for it automatically rather than
        # waiting for someone to click a button. Only on `created` (M2's
        # content-addressing already deduplicates a re-export of unchanged
        # data to the same snapshot_id) - nothing about the data changed, so
        # there is nothing new to train on. Never allowed to fail the export
        # itself: the snapshot is already safely on disk and recorded by the
        # time this runs, same "the primary action already succeeded"
        # reasoning as the publish call above.
        auto_train_job_id = None
        if created and settings.auto_train_on_handoff:
            try:
                from app.services.detector_service import get_detector_service

                auto_train_job_id = get_detector_service().start_training(trigger="export_handoff").job_id
            except Exception:
                logger.exception("Could not auto-trigger training after snapshot %s", snapshot_id)

        stats = dict(stats)
        stats.update(
            {
                "snapshot_id": snapshot_id,
                "snapshot_created": created,
                "output_dir": str(snapshot_dir),
                "manifest": snapshot_service.MANIFEST_NAME,
                "file_count": len(file_index),
                "publish": publish_info,
                "auto_train_job_id": auto_train_job_id,
                # Kept at the top level as well as inside the manifest: these are
                # the two fields anyone hands over or quotes in a ticket.
                "class_map_version": self._ds.class_map_version,
                "class_map_hash": self._ds.class_map_hash,
            }
        )
        return stats

    def _write_dataset(self, request: ExportRequest, exports_dir: Path) -> dict:
        image_ids = request.image_ids or self._ds.image_ids()
        settings = get_settings()

        # Belt-and-braces against exclude_classes. add_class() already refuses
        # these names, but a dataset annotated before that check existed - or one
        # whose data.yaml was edited by hand - can still contain them, and
        # `leakage` is all-synthetic and never served (docs/pipeline.md §12).
        # Filtering here means an excluded label cannot reach a snapshot even if
        # it reached the annotation store.
        excluded_class_ids = {
            c.class_id for c in self._ds.get_classes() if settings.is_excluded_class(c.name)
        }
        if excluded_class_ids:
            logger.warning(
                "Export will drop labels for excluded class id(s) %s (exclude_classes=%s)",
                sorted(excluded_class_ids),
                settings.exclude_class_list,
            )
        excluded_labels = 0

        # Second-reviewer export gate (Phase 4, product-owner decision - see
        # annotation_module_build_plan.md): an image completed before the
        # gate went live is grandfathered (ExportGateExemption); anything
        # completed after needs its *latest* review to be "approved."
        eligible_ids: set[str] = set()
        # Provenance sets, read once. These populate the manifest's `provenance`
        # block, which deliberately includes the uncomfortable numbers - notably
        # grandfathered_unreviewed_images, because if that is non-zero a model
        # trained on this snapshot cannot claim "all data second-reviewed".
        exempt_ids: set[str] = set()
        second_reviewed_ids: set[str] = set()
        audit_sampled_ids: set[str] = set()
        auto_accepted_ids: set[str] = set()
        db = SessionLocal()
        try:
            if request.only_completed:
                eligible_ids = review_service.get_export_eligible_ids(db, self._ds.dataset_key, image_ids)
            exempt_ids = review_service.get_exempt_image_ids(db, self._ds.dataset_key, image_ids)
            second_reviewed_ids = review_service.get_reviewed_image_ids(
                db, self._ds.dataset_key, "second_review"
            )
            audit_sampled_ids = review_service.get_reviewed_image_ids(
                db, self._ds.dataset_key, "audit_sample"
            )
            auto_accepted_ids = review_service.get_reviewed_image_ids(
                db, self._ds.dataset_key, "auto_accept"
            )
            annotator_ids = state_repo.get_updated_by_ids(db, self._ds.dataset_key, image_ids)
            # M4: every image_id ever frozen into a golden set for this view,
            # across all versions - the ruler nothing trains on. Excluded
            # entirely below, not just kept out of val/test like M3's
            # propagated/auto-accepted forcing: a golden image belongs in
            # neither train nor val of a training export.
            golden_ids = golden_repo.get_golden_image_ids(db, self._ds.dataset_key)
            golden_set_storage_exists = bool(golden_repo.list_versions(db, self._ds.dataset_key))
        finally:
            db.close()

        provenance = {
            "human_annotated_images": 0,
            "auto_accepted_images": 0,
            "second_reviewed_images": 0,
            "grandfathered_unreviewed_images": 0,
            "audit_sampled_images": 0,
            "propagated_labels": 0,
            "annotator_ids": annotator_ids,
        }
        # {coach_type: {class_id: label count}} - an aggregate per-class count
        # hides a class well covered on LHB and absent on ICF, which is what the
        # pipeline team's label-scarcity risk needs to be actionable.
        per_class_counts: dict[str, dict[int, int]] = {}
        split_counts: dict[str, dict[str, int]] = {}
        integrity = {
            # M3: enforced, not just measured - _split_for's natural assignment
            # is overridden below whenever it would land pseudo/synthetic
            # (propagated) or auto-accepted labels outside train, so these two
            # counts should always land at 0. Kept rather than removed: a
            # nonzero value here is now a bug signal, and _finalize_snapshot
            # refuses to finalize a snapshot that reports one.
            "pseudo_synthetic_in_val_or_test": 0,
            "auto_accepted_in_val_or_test": 0,
            # How many images the M3 forcing pass actually redirected into
            # train against their natural hash-based split - the visible
            # record that enforcement did something, as opposed to reporting
            # 0/0 above because there was nothing to enforce.
            "pseudo_synthetic_forced_to_train": 0,
            "auto_accepted_forced_to_train": 0,
            # M4: golden images are dropped from `exportable` entirely below
            # (see golden_images_excluded in the returned stats), so this
            # should always land at 0 too - kept as the same kind of bug
            # signal as the two counters above, not a measurement.
            "golden_ids_in_any_split": 0,
            "golden_set_storage_exists": golden_set_storage_exists,
        }
        spine = {
            "fields": [
                "coach_index",
                "coach_type",
                "axle_id",
                "side",
                "view",
                "longitudinal_position_mm",
                "zone",
            ],
            "frames_with_full_stamp": 0,
            "frames_with_coach_type_only": 0,
            "frames_missing_stamp": 0,
        }

        # (image_id, label lines, the annotations themselves) - the third element
        # is carried so the mask-raster pass below doesn't have to re-read state
        # per image just to get dimensions and per-object pieces.
        exportable: list[tuple[str, list[tuple[int, list]], ImageAnnotations]] = []
        skipped = 0
        needs_review = 0
        negatives = 0
        golden_images_excluded = 0
        for image_id in image_ids:
            # M4: a golden image never appears in a training export, in any
            # split, regardless of review/completion state - it is excluded
            # before any of those checks even run, not just kept out of
            # val/test the way M3's forced images are.
            if image_id in golden_ids:
                golden_images_excluded += 1
                continue

            annotations = self._ds.get_annotations(image_id)
            if request.only_completed and not annotations.completed:
                skipped += 1
                continue
            if request.only_completed and image_id not in eligible_ids:
                needs_review += 1
                continue

            # One label line per polygon piece. For a `fine_structure` class
            # (crack/corrosion/shelling) a single object can carry several
            # disjoint pieces - a branching or segmented crack - and each is
            # emitted separately rather than all but the largest being dropped,
            # which is what used to happen and what silently cost length-recall
            # (docs/pipeline.md §5.4).
            objects = [
                (obj.class_id, piece)
                for obj in annotations.objects
                if obj.status != ObjectStatus.REJECTED and obj.class_id not in excluded_class_ids
                for piece in [obj.polygon, *obj.extra_polygons]
                if len(piece) >= 3
            ]
            excluded_labels += sum(
                1
                for obj in annotations.objects
                if obj.status != ObjectStatus.REJECTED and obj.class_id in excluded_class_ids
            )
            if not objects:
                # An empty frame is only exportable when a human explicitly
                # asserted it is empty (see ImageAnnotations). Then it becomes
                # a negative/background sample: the image plus an empty label
                # file, which is the YOLO convention ultralytics consumes
                # natively. Without that assertion it is indistinguishable
                # from "not annotated yet" and is skipped, as before.
                #
                # This is what VB-BufferBoundary's `buffer_visible` needs -
                # frames with no buffer are half its training signal, and
                # they used to be discarded here.
                if annotations.no_objects_confirmed:
                    exportable.append((image_id, [], annotations))
                    negatives += 1
                else:
                    skipped += 1
                continue

            exportable.append((image_id, objects, annotations))

        # Only one image available: put it in train so val isn't empty-vs-all.
        force_train = len(exportable) < 2

        exported = 0
        mask_files = 0
        wheel_unwraps = 0
        condition_crops = 0
        condition_unassessed = 0
        crop_margin_pct = settings.condition_crop_margin_pct
        for image_id, objects, annotations in exportable:
            live = [o for o in annotations.objects if o.status != ObjectStatus.REJECTED]
            propagated_here = sum(1 for o in live if o.source == ObjectSource.PROPAGATED)
            is_auto_accepted = image_id in auto_accepted_ids

            natural_split = "train" if force_train else self._split_for(image_id)
            split, forced_to_train = self._enforce_split_integrity(
                natural_split, propagated_here, is_auto_accepted
            )
            if forced_to_train:
                logger.warning(
                    "M3 split-integrity: forcing %s into train (natural split was "
                    "%s) - %d propagated label(s), auto_accepted=%s",
                    image_id, natural_split, propagated_here, is_auto_accepted,
                )
                if propagated_here:
                    integrity["pseudo_synthetic_forced_to_train"] += propagated_here
                if is_auto_accepted:
                    integrity["auto_accepted_forced_to_train"] += 1

            out_images = exports_dir / "images" / split
            out_labels = exports_dir / "labels" / split
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)

            label_path = out_labels / f"{image_id}.txt"
            write_segmentation_label_file(label_path, objects)

            src_image = self._ds.get_image_path(image_id)
            dst_image = out_images / src_image.name
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)

            mask_files += self._write_fine_structure_masks(
                exports_dir / "masks" / split, image_id, annotations
            )
            wheel_unwraps += self._write_wheel_unwraps(
                exports_dir / "logpolar" / split, image_id, annotations
            )
            if request.include_condition_crops:
                condition_crops += self._write_condition_crops(
                    exports_dir / "crops" / split, image_id, annotations, crop_margin_pct
                )
            # Counted whether or not crops were requested, so "0 condition
            # crops" is legible as "nothing has been assessed yet" rather than
            # looking like a broken export.
            condition_unassessed += sum(
                1
                for obj in annotations.objects
                if obj.status != ObjectStatus.REJECTED and obj.condition is None
            )

            # ---- provenance / manifest accounting for this image
            bucket = split_counts.setdefault(split, {"images": 0, "labels": 0})
            bucket["images"] += 1
            bucket["labels"] += len(objects)

            provenance["propagated_labels"] += propagated_here

            if is_auto_accepted:
                provenance["auto_accepted_images"] += 1
            else:
                provenance["human_annotated_images"] += 1
            if image_id in second_reviewed_ids:
                provenance["second_reviewed_images"] += 1
            elif image_id in exempt_ids:
                # Exportable without a review only because it predates the gate.
                provenance["grandfathered_unreviewed_images"] += 1
            if image_id in audit_sampled_ids:
                provenance["audit_sampled_images"] += 1

            coach_type = annotations.coach_type.value
            per_coach = per_class_counts.setdefault(coach_type, {})
            for class_id, _piece in objects:
                per_coach[class_id] = per_coach.get(class_id, 0) + 1

            if annotations.coach_type != CoachType.UNKNOWN:
                spine["frames_with_coach_type_only"] += 1
            else:
                spine["frames_missing_stamp"] += 1

            # Should be unreachable: the forcing pass above already redirects
            # any propagated/auto-accepted image into train. Left as a
            # defensive check rather than removed - if it ever fires, that is
            # exactly the case _finalize_snapshot's refusal exists to catch.
            if split != "train":
                if propagated_here:
                    integrity["pseudo_synthetic_in_val_or_test"] += propagated_here
                if is_auto_accepted:
                    integrity["auto_accepted_in_val_or_test"] += 1

            exported += 1

        # exports_dir is otherwise only created inside the per-image loop
        # above (via out_images.mkdir/out_labels.mkdir) - if every candidate
        # was skipped/needs_review, that loop never ran and this directory
        # wouldn't exist yet, so classes.txt/data.yaml below would fail.
        exports_dir.mkdir(parents=True, exist_ok=True)

        class_names = [c.name for c in self._ds.get_classes()]
        classes_file = exports_dir / "classes.txt"
        classes_file.write_text("\n".join(class_names) + "\n", encoding="utf-8")

        data_yaml = {
            "path": str(exports_dir.resolve()),
            "train": "images/train",
            "val": "images/val" if (exports_dir / "images" / "val").exists() else "images/train",
            # Excluded classes stay in `names`, deliberately. Class ids are
            # positional, so dropping a name would renumber every class after it
            # - which is precisely the class-id drift this whole versioning
            # effort exists to prevent. The name is retained as a placeholder and
            # its *labels* are dropped (see excluded_class_ids above).
            "names": {i: name for i, name in enumerate(class_names)},
            # Extra keys YOLO ignores, so the export can answer "what did class 7
            # mean here" without needing the tool or its database. The hash is
            # what actually proves two exports share a map; the version number is
            # for humans.
            "class_map_version": self._ds.class_map_version,
            "class_map_hash": self._ds.class_map_hash,
            "exclude_classes": settings.exclude_class_list,
        }
        (exports_dir / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        logger.info(
            "Export complete: %d exported (%d confirmed negatives), %d skipped, %d needs_review, "
            "%d golden image(s) excluded, %d fine-structure mask raster(s), %d wheel unwrap(s), "
            "%d condition crop(s) (%d object(s) still unassessed), %d excluded label(s) dropped, "
            "class-map version %s",
            exported,
            negatives,
            skipped,
            needs_review,
            golden_images_excluded,
            mask_files,
            wheel_unwraps,
            condition_crops,
            condition_unassessed,
            excluded_labels,
            self._ds.class_map_version,
        )
        integrity_ok = (
            integrity["pseudo_synthetic_in_val_or_test"] == 0
            and integrity["auto_accepted_in_val_or_test"] == 0
            and integrity["golden_ids_in_any_split"] == 0
        )
        return {
            "exported": exported,
            "negatives": negatives,
            "skipped": skipped,
            "needs_review": needs_review,
            "mask_rasters": mask_files,
            "wheel_unwraps": wheel_unwraps,
            # Versioned per snapshot (D-Q5's Consequence C-4): a mask drawn in
            # raw space is meaningless under different unwrap parameters, so
            # the manifest must record exactly which ones produced this
            # export's logpolar/ files. `certified: False` is load-bearing,
            # not decoration - these are provisional engineering placeholders
            # ("wheel specs are to be considered in your own accord for now"),
            # never a claim that this diameter/geometry is correct.
            "wheel_unwrap_params": {
                "version": settings.wheel_unwrap_version,
                "certified": False,
                "wheel_class_name": settings.wheel_class_name,
                "radius_padding_pct": settings.wheel_unwrap_radius_padding_pct,
                "output_width": settings.wheel_unwrap_output_width,
                "output_height": settings.wheel_unwrap_output_height,
                "log_scale": settings.wheel_unwrap_log_scale,
            },
            "condition_crops": condition_crops,
            "condition_unassessed": condition_unassessed,
            "excluded_labels_dropped": excluded_labels,
            "golden_images_excluded": golden_images_excluded,
            # What this export's class ids actually mean. The number is for
            # humans; the hash is what proves two exports used the same map.
            "class_map_version": self._ds.class_map_version,
            "class_map_hash": self._ds.class_map_hash,
            "output_dir": str(exports_dir),
            # Manifest blocks (see snapshot_service.build_manifest). Sets are
            # converted here rather than in the manifest builder so the builder
            # stays JSON-only.
            "splits": split_counts,
            "split_integrity": integrity,
            "split_integrity_ok": integrity_ok,
            "provenance": {
                **{k: v for k, v in provenance.items() if k != "annotator_ids"},
                "annotator_ids": sorted(provenance["annotator_ids"]),
            },
            "per_class_counts": {
                coach_type: {str(class_id): count for class_id, count in sorted(counts.items())}
                for coach_type, counts in sorted(per_class_counts.items())
            },
            "spine_stamp_coverage": spine,
        }
