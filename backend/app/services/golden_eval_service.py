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
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

import yaml

from app.models.schemas import ObjectStatus
from app.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)


class NoGoldenSetError(Exception):
    """Raised when a dataset view has no golden images to evaluate against
    yet - not an error in the usual sense, just nothing to score against."""


def _assemble_golden_dataset(
    ds: DatasetService, golden_image_ids: set[str], staging_dir: Path, classes: list[str]
) -> tuple[Path, int]:
    """Write a YOLO-detection "val-only" dataset from the golden set's
    *current* annotation state.

    Deliberately reads current state, not a frozen copy: the golden bucket
    freeze (M4's `object_store.freeze_golden_item`) is best-effort and often
    unconfigured in dev (no MinIO), so the only reliably-available source
    for evaluation right now is the live dataset - acceptable here because
    what matters structurally is that these image_ids never appear in a
    *training* split (enforced by `export_service`), not that this
    particular eval call reads through a separate frozen path.
    """
    img_dir = staging_dir / "val" / "images"
    lbl_dir = staging_dir / "val" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_id in sorted(golden_image_ids):
        try:
            annotations = ds.get_annotations(image_id)
        except Exception:
            logger.exception("Golden image %s could not be read; skipped from this eval", image_id)
            continue
        objects = [o for o in annotations.objects if o.status != ObjectStatus.REJECTED]
        if not objects and not annotations.no_objects_confirmed:
            continue

        src_image = ds.get_image_path(image_id)
        shutil.copy2(src_image, img_dir / src_image.name)
        lines = [
            f"{o.class_id} {o.bbox.x_center:.6f} {o.bbox.y_center:.6f} {o.bbox.width:.6f} {o.bbox.height:.6f}"
            for o in objects
        ]
        (lbl_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
        written += 1

    data_yaml = staging_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {"path": str(staging_dir), "train": "val/images", "val": "val/images", "nc": len(classes), "names": classes},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml, written


def evaluate_on_golden_set(
    ds: DatasetService,
    golden_image_ids: set[str],
    model_path: Path,
    classes: list[str],
    staging_root: Path,
) -> Optional[dict]:
    """Run the trained model against the golden set and return per-class +
    aggregate metrics. Returns None (logged, not raised) if the golden set
    has nothing currently evaluable - e.g. every golden image happens to be
    unreachable right now - since that's a data-availability gap, not a
    training failure.

    Raises NoGoldenSetError if `golden_image_ids` is empty - the caller
    should treat this as "nothing to evaluate yet", not surface it as an
    error to the training job.
    """
    if not golden_image_ids:
        raise NoGoldenSetError("No golden set exists for this dataset view yet")

    staging_dir = staging_root / "golden_eval"
    try:
        data_yaml, num_images = _assemble_golden_dataset(ds, golden_image_ids, staging_dir, classes)
        if num_images == 0:
            logger.warning("Golden set has %d image id(s) but none were evaluable right now", len(golden_image_ids))
            return None

        from ultralytics import YOLO

        model = YOLO(str(model_path))
        results = model.val(data=str(data_yaml), split="val", verbose=False, plots=False)

        per_class: dict[int, dict[str, float]] = {}
        for i, class_id in enumerate(results.ap_class_index):
            precision, recall, map50, map50_95 = results.box.class_result(i)
            per_class[int(class_id)] = {
                "precision": float(precision),
                "recall": float(recall),
                "mAP50": float(map50),
                "mAP50_95": float(map50_95),
            }

        return {
            "num_golden_images": num_images,
            "num_golden_images_total": len(golden_image_ids),
            "per_class": per_class,
            "aggregate": {
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
                "mAP50": float(results.box.map50),
                "mAP50_95": float(results.box.map),
            },
        }
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
