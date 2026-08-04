"""One-time backfill: load every existing `.annotation_state/*.json` +
`_meta.json` into Postgres (Phase 1a, task #5 - see
annotation_module_build_plan.md).

Must run and be verified before dataset_service.py's DB-backed read/write
path (task #4) is what a live deployment actually uses - otherwise every
already-annotated image looks unannotated the moment the cutover lands.

Strictly additive and safe to re-run: it only reads the JSON files (never
writes, moves, or deletes them) and upserts into Postgres, so nothing about
the original per-image JSON state is at risk regardless of how many times
this is run or in what order relative to other work.

Usage (from backend/, with the venv/container's Python):
    python -m scripts.backfill_annotation_state
    python -m scripts.backfill_annotation_state /custom/dataset/root [...]

With no arguments, backfills the three fixed dataset views under
settings.dataset_path (see routers/dataset.py's DATASET_VIEWS) that exist on
disk. Extra positional args backfill additional arbitrary dataset roots
(matching /api/dataset/load's arbitrary-path support).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert

from app.config import get_settings
from app.db import SessionLocal
from app.models.db_models import AnnotationState, DatasetClass
from app.utils.yolo_utils import load_class_names

FIXED_VIEWS = ["side_view", "underbelly", "wheel_shelling"]
DEFAULT_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#808080", "#ff4d4d", "#4dff4d", "#4d4dff", "#ffff4d",
    "#ff4dff", "#4dffff", "#c04d4d", "#4dc0c0",
]


def backfill_root(root: Path) -> tuple[int, int, int]:
    """Returns (state_rows_written, state_rows_skipped, class_rows_written)."""
    state_dir = root / ".annotation_state"
    if not state_dir.is_dir():
        print(f"  skip: no .annotation_state dir under {root}")
        return 0, 0, 0

    dataset_key = str(root.resolve())
    db = SessionLocal()
    written = skipped = 0
    try:
        for json_path in sorted(state_dir.glob("*.json")):
            if json_path.name == "_meta.json":
                continue
            image_id = json_path.stem
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  WARNING: failed to read {json_path}: {exc}")
                skipped += 1
                continue

            stmt = insert(AnnotationState).values(
                dataset_view=dataset_key,
                image_id=image_id,
                payload=payload,
                completed=bool(payload.get("completed")),
                updated_by_id=None,
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_annotation_state_view_image",
                set_={
                    "payload": stmt.excluded.payload,
                    "completed": stmt.excluded.completed,
                },
            )
            db.execute(stmt)
            written += 1
        db.commit()

        class_rows = _backfill_classes(db, root, dataset_key)
    finally:
        db.close()

    return written, skipped, class_rows


def _backfill_classes(db, root: Path, dataset_key: str) -> int:
    """Best-effort: names come from the same source load_dataset() itself
    uses (classes.txt/data.yaml), colors from _meta.json if present, falling
    back to the same default palette dataset_service.py would generate."""
    classes = load_class_names(root)
    if not classes:
        return 0

    meta_path = root / ".annotation_state" / "_meta.json"
    colors: dict[str, str] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            colors = meta.get("colors", {}) if isinstance(meta, dict) else {}
        except (json.JSONDecodeError, OSError):
            pass

    written = 0
    for class_id, name in enumerate(classes):
        color = colors.get(str(class_id), DEFAULT_PALETTE[class_id % len(DEFAULT_PALETTE)])
        stmt = insert(DatasetClass).values(dataset_view=dataset_key, class_id=class_id, name=name, color=color)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_dataset_classes_view_class",
            set_={"name": stmt.excluded.name, "color": stmt.excluded.color},
        )
        db.execute(stmt)
        written += 1
    db.commit()
    return written


def main() -> None:
    extra_roots = [Path(p) for p in sys.argv[1:]]
    if extra_roots:
        roots = extra_roots
    else:
        base = Path(get_settings().dataset_path)
        roots = [base / v for v in FIXED_VIEWS if (base / v).exists()]

    if not roots:
        print("No dataset roots found to backfill.")
        return

    total_written = total_skipped = 0
    for root in roots:
        print(f"Backfilling {root} ...")
        written, skipped, class_rows = backfill_root(root)
        print(f"  {written} state rows written, {skipped} skipped, {class_rows} class rows written")
        total_written += written
        total_skipped += skipped

    print(f"Done. {total_written} total state rows written, {total_skipped} skipped.")


if __name__ == "__main__":
    main()
