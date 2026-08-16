"""Evaluate a just-trained detector against the frozen golden set (M6).

Closes the loop M5 opened: training was tracked in MLflow, but nothing
scored a finished model against anything - a human had to eyeball the
train/val metrics ultralytics itself produces, which come from a random
hash-based split of whatever was completed, not the golden set nothing
trains on. This runs that same model against the golden set specifically,
and logs the result onto the *same* MLflow run alongside its training
metrics, so one run tells the whole story.

Scope A only, same boundary as everything else in this MLflow work: the
in-tool SAM2/detector pre-labeler helper, never the pipeline team's own
production families - Q-C again.

**Per-class, never aggregate-only** - the same principle
`export_service`'s `per_class_counts` and the promotion-gate design in
`MLFLOW_INTEGRATION_ANALYSIS.md` both insist on: an aggregate mAP can rise
while one safety-critical class quietly regresses, invisibly. Every class
present in the golden set gets its own logged precision/recall/mAP; the
aggregate is logged too, but never in place of the per-class numbers.

**Metric basis changed with the YOLO11-seg move.** `precision`/`recall`/
`mAP50`/`mAP50_95` in the returned per-class/aggregate dicts are **mask**
metrics (`results.seg`) as of this change, not box metrics - a mask AP is
what actually reflects polygon quality, which is the entire point of
training on polygons instead of boxes. Box metrics (`results.box`) are kept
alongside under `box_precision`/`box_recall`/`box_mAP50`/`box_mAP50_95` for
continuity, since a segmentation model still produces a box per detection.
This means `golden/class{N}_mAP50` in MLflow means something different for
runs before vs. after this change - `detector_service` tags every run with
`eval_task` precisely so that's discoverable rather than silently assumed.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

from app.models.schemas import ObjectStatus
from app.services.dataset_service import DatasetService
from app.services.polygon_service import mask_iou, polygon_to_mask
from app.utils.yolo_utils import write_segmentation_label_file

logger = logging.getLogger(__name__)

# The model's own serving confidence gate (detector_service.detect() uses
# this exact value) - FP/FN below are measured at THIS threshold precisely
# because that's the number that matters operationally, not an AP curve's
# implicit sweep across every possible threshold.
SERVING_CONFIDENCE_THRESHOLD = 0.25
# Standard mAP50 convention - a prediction counts as matching a ground-truth
# instance only above this mask IoU.
MATCH_IOU_THRESHOLD = 0.5


class NoGoldenSetError(Exception):
    """Raised when a dataset view has no golden images to evaluate against
    yet - not an error in the usual sense, just nothing to score against."""


class InsufficientGoldenCoverageError(Exception):
    """A class in the candidate's trainable class set has zero matching
    golden-set instances - its AP/precision/recall would be a meaningless
    zero-instance number, not a real absence-of-regression signal. Lists
    every uncovered class at once (not just the first), mirroring
    should_promote()'s own "collect every failing check" philosophy."""

    def __init__(self, class_ids: List[int]) -> None:
        self.class_ids = class_ids
        super().__init__(
            f"Class(es) {class_ids} have zero golden-set coverage - cannot evaluate them meaningfully. "
            f"A golden_curator must add representative images for these classes to the golden set first."
        )


def _assemble_golden_dataset(
    ds: DatasetService, golden_image_ids: Set[str], staging_dir: Path, classes: List[str]
) -> Tuple[Path, int, Dict[int, int]]:
    """Write a YOLO-**segmentation** "val-only" dataset from the golden set's
    *current* annotation state.

    Deliberately reads current state, not a frozen copy: the golden bucket
    freeze (M4's `object_store.freeze_golden_item`) is best-effort and often
    unconfigured in dev (no MinIO), so the only reliably-available source
    for evaluation right now is the live dataset - acceptable here because
    what matters structurally is that these image_ids never appear in a
    *training* split (enforced by `export_service`), not that this
    particular eval call reads through a separate frozen path.

    The third return value - per-class instance counts across the whole
    golden set - is what promotion_gate.py's new-class absolute-floor check
    needs (`n_val_instances`, docs/mlflow_class_incremental_architecture.md
    §E/§H): an AP number backed by 2 golden instances is not the same
    reliability claim as one backed by 30, even at an identical value.
    """
    img_dir = staging_dir / "val" / "images"
    lbl_dir = staging_dir / "val" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    instance_counts: Dict[int, int] = {}
    for image_id in sorted(golden_image_ids):
        try:
            annotations = ds.get_annotations(image_id)
        except Exception:
            logger.exception("Golden image %s could not be read; skipped from this eval", image_id)
            continue
        live = [o for o in annotations.objects if o.status != ObjectStatus.REJECTED]
        pieces = [
            (o.class_id, piece)
            for o in live
            for piece in [o.polygon, *o.extra_polygons]
            if len(piece) >= 3
        ]
        if not pieces and not annotations.no_objects_confirmed:
            continue

        src_image = ds.get_image_path(image_id)
        shutil.copy2(src_image, img_dir / src_image.name)
        write_segmentation_label_file(lbl_dir / f"{image_id}.txt", pieces)
        written += 1
        for class_id, _piece in pieces:
            instance_counts[class_id] = instance_counts.get(class_id, 0) + 1

    data_yaml = staging_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {"path": str(staging_dir), "train": "val/images", "val": "val/images", "nc": len(classes), "names": classes},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml, written, instance_counts


def _compute_fp_fn(
    ds: DatasetService,
    golden_image_ids: Set[str],
    model,
    confidence_threshold: float,
    iou_threshold: float,
) -> Dict[int, Dict[str, float]]:
    """Per-class false_positive_rate/false_negative_rate at a FIXED serving
    confidence threshold - deliberately not the AP curve's implicit sweep,
    since that isn't the number that matters operationally
    (docs/mlflow_class_incremental_architecture.md §G). A separate
    prediction pass from model.val() (which doesn't expose raw per-image
    matches), reusing this repo's own existing `polygon_to_mask`/`mask_iou`
    (polygon_service.py) for mask-IoU matching - consistent with this
    module's own "mask metrics are primary" stance, not box IoU.

    Greedy matching, standard mAP-style: predictions sorted by confidence
    descending, each claims the highest-IoU still-unmatched ground-truth
    instance of the same class if IoU clears `iou_threshold`; unmatched
    predictions are false positives, unmatched ground truth are false
    negatives.
    """
    from app.models.schemas import Point

    counts: Dict[int, Dict[str, int]] = {}

    def _bump(class_id: int, key: str) -> None:
        counts.setdefault(class_id, {"tp": 0, "fp": 0, "fn": 0})[key] += 1

    for image_id in sorted(golden_image_ids):
        try:
            annotations = ds.get_annotations(image_id)
        except Exception:
            continue
        width, height = annotations.width, annotations.height
        gt_by_class: Dict[int, list] = {}
        for obj in annotations.objects:
            if obj.status == ObjectStatus.REJECTED or len(obj.polygon) < 3:
                continue
            gt_by_class.setdefault(obj.class_id, []).append(polygon_to_mask(obj.polygon, width, height))

        try:
            image_path = ds.get_image_path(image_id)
            results = model.predict(str(image_path), conf=confidence_threshold, verbose=False)
        except Exception:
            logger.exception("FP/FN prediction failed for golden image %s; skipped from this measurement", image_id)
            continue

        if not results or not results[0].boxes:
            # No predictions at all above threshold - every real instance
            # in this image is a miss for its class.
            for class_id, masks in gt_by_class.items():
                for _ in masks:
                    _bump(class_id, "fn")
            continue

        result = results[0]
        mask_polys = result.masks.xyn if result.masks is not None else None
        pred_by_class: Dict[int, List[Tuple[float, int]]] = {}
        for idx, box in enumerate(result.boxes):
            pred_by_class.setdefault(int(box.cls.item()), []).append((float(box.conf.item()), idx))

        for class_id in set(gt_by_class) | set(pred_by_class):
            gt_masks = gt_by_class.get(class_id, [])
            matched_gt = [False] * len(gt_masks)
            for _conf, idx in sorted(pred_by_class.get(class_id, []), key=lambda t: -t[0]):
                if mask_polys is None or idx >= len(mask_polys) or len(mask_polys[idx]) < 3:
                    _bump(class_id, "fp")
                    continue
                pred_mask = polygon_to_mask(
                    [Point(x=float(px), y=float(py)) for px, py in mask_polys[idx]], width, height
                )
                best_iou, best_j = 0.0, -1
                for j, gt_mask in enumerate(gt_masks):
                    if matched_gt[j]:
                        continue
                    iou = mask_iou(pred_mask, gt_mask)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_threshold:
                    matched_gt[best_j] = True
                    _bump(class_id, "tp")
                else:
                    _bump(class_id, "fp")
            for matched in matched_gt:
                if not matched:
                    _bump(class_id, "fn")

    rates: Dict[int, Dict[str, float]] = {}
    for class_id, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        rates[class_id] = {
            "false_positive_rate": (fp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "false_negative_rate": (fn / (tp + fn)) if (tp + fn) > 0 else 0.0,
        }
    return rates


def evaluate_on_golden_set(
    ds: DatasetService,
    golden_image_ids: Set[str],
    model_path: Path,
    classes: List[str],
    staging_root: Path,
    candidate_class_ids: Optional[Set[int]] = None,
) -> Optional[dict]:
    """Run the trained model against the golden set and return per-class +
    aggregate metrics. Returns None (logged, not raised) if the golden set
    has nothing currently evaluable - e.g. every golden image happens to be
    unreachable right now - since that's a data-availability gap, not a
    training failure.

    Raises NoGoldenSetError if `golden_image_ids` is empty - the caller
    should treat this as "nothing to evaluate yet", not surface it as an
    error to the training job.

    Raises InsufficientGoldenCoverageError if `candidate_class_ids` (the
    classes this candidate actually trains on - Module 3's eligible/active
    set) includes any class with zero matching golden-set instances,
    checked BEFORE the expensive model.val() call - there is no point
    running a full evaluation pass that is guaranteed to produce a
    meaningless zero-instance number for that class. Pass None to skip
    this check (e.g. an ad-hoc eval where the caller doesn't have a
    candidate class set in hand).
    """
    if not golden_image_ids:
        raise NoGoldenSetError("No golden set exists for this dataset view yet")

    staging_dir = staging_root / "golden_eval"
    try:
        data_yaml, num_images, instance_counts = _assemble_golden_dataset(ds, golden_image_ids, staging_dir, classes)
        if num_images == 0:
            logger.warning("Golden set has %d image id(s) but none were evaluable right now", len(golden_image_ids))
            return None

        if candidate_class_ids:
            uncovered = sorted(c for c in candidate_class_ids if instance_counts.get(c, 0) == 0)
            if uncovered:
                raise InsufficientGoldenCoverageError(uncovered)

        from ultralytics import YOLO

        model = YOLO(str(model_path))
        results = model.val(data=str(data_yaml), split="val", verbose=False, plots=False)
        fp_fn_rates = _compute_fp_fn(ds, golden_image_ids, model, SERVING_CONFIDENCE_THRESHOLD, MATCH_IOU_THRESHOLD)

        # results.seg is the mask-quality metric - the one that actually
        # reflects polygon quality, which is why it's the primary key names
        # below. results.box (bbox around each predicted mask) is kept
        # alongside for continuity with dashboards built against pre-seg
        # runs, never as the primary number going forward - see this
        # module's docstring.
        per_class: Dict[int, Dict[str, float]] = {}
        for i, class_id in enumerate(results.ap_class_index):
            box_p, box_r, box_map50, box_map = results.box.class_result(i)
            seg_p, seg_r, seg_map50, seg_map = results.seg.class_result(i)
            rates = fp_fn_rates.get(
                int(class_id), {"false_positive_rate": 0.0, "false_negative_rate": 0.0}
            )
            per_class[int(class_id)] = {
                "precision": float(seg_p),
                "recall": float(seg_r),
                "mAP50": float(seg_map50),
                "mAP50_95": float(seg_map),
                "box_precision": float(box_p),
                "box_recall": float(box_r),
                "box_mAP50": float(box_map50),
                "box_mAP50_95": float(box_map),
                "n_val_instances": instance_counts.get(int(class_id), 0),
                "false_positive_rate": rates["false_positive_rate"],
                "false_negative_rate": rates["false_negative_rate"],
            }

        return {
            "num_golden_images": num_images,
            "num_golden_images_total": len(golden_image_ids),
            "per_class": per_class,
            "aggregate": {
                "precision": float(results.seg.mp),
                "recall": float(results.seg.mr),
                "mAP50": float(results.seg.map50),
                "mAP50_95": float(results.seg.map),
                "box_precision": float(results.box.mp),
                "box_recall": float(results.box.mr),
                "box_mAP50": float(results.box.map50),
                "box_mAP50_95": float(results.box.map),
            },
        }
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
