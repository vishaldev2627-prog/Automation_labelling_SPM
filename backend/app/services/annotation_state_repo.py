"""Postgres-backed replacement for dataset_service.py's per-image JSON state
and `_meta.json` (Phase 1a, task #4 - see annotation_module_build_plan.md).

`dataset_key` is the resolved dataset root path (`str(root.resolve())`), the
same identity `_get_class_list_lock` already uses in dataset_service.py - not
one of the three fixed DATASET_VIEWS keys, since `/api/dataset/load` accepts
arbitrary paths too. It's stored in the `dataset_view` column (named for the
common case, but holds any resolved dataset root).

Upserts use Postgres' native ON CONFLICT rather than a query-then-write
pattern, so concurrent saves from different sessions/annotators never race -
same reasoning as annotator_service.get_or_create_annotator's IntegrityError
retry, just expressed atomically instead since these tables' conflict target
should always win-on-latest rather than "first write wins".
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.db_models import AnnotationHistory, AnnotationState, DatasetClass


def get_state(db: Session, dataset_key: str, image_id: str) -> Optional[dict]:
    row = db.execute(
        select(AnnotationState.payload).where(
            AnnotationState.dataset_view == dataset_key, AnnotationState.image_id == image_id
        )
    ).scalar_one_or_none()
    return row


def get_states_bulk(db: Session, dataset_key: str, image_ids: list[str]) -> dict[str, dict]:
    """Batch equivalent of get_state - one query instead of one per image,
    for list_images()/get_dataset_info() which need every image's state."""
    if not image_ids:
        return {}
    rows = db.execute(
        select(AnnotationState.image_id, AnnotationState.payload).where(
            AnnotationState.dataset_view == dataset_key, AnnotationState.image_id.in_(image_ids)
        )
    ).all()
    return {image_id: payload for image_id, payload in rows}


def save_state(
    db: Session,
    dataset_key: str,
    image_id: str,
    payload: dict,
    completed: bool,
    annotator_id: Optional[int],
) -> None:
    # updated_at has a server_default of now() for inserts, but that default
    # doesn't fire again on an ON CONFLICT UPDATE - set it explicitly so an
    # update actually refreshes the timestamp.
    stmt = insert(AnnotationState).values(
        dataset_view=dataset_key,
        image_id=image_id,
        payload=payload,
        completed=completed,
        updated_by_id=annotator_id,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_annotation_state_view_image",
        set_={
            "payload": stmt.excluded.payload,
            "completed": stmt.excluded.completed,
            "updated_by_id": stmt.excluded.updated_by_id,
            "updated_at": func.now(),
        },
    )
    db.execute(stmt)
    db.add(
        AnnotationHistory(
            dataset_view=dataset_key,
            image_id=image_id,
            payload=payload,
            action="mark_completed" if completed else "save",
            annotator_id=annotator_id,
        )
    )
    db.commit()


def get_colors(db: Session, dataset_key: str) -> dict[str, str]:
    rows = db.execute(
        select(DatasetClass.class_id, DatasetClass.color).where(DatasetClass.dataset_view == dataset_key)
    ).all()
    return {str(class_id): color for class_id, color in rows}


def save_colors_bulk(db: Session, dataset_key: str, classes: list[str], colors: dict[str, str]) -> None:
    """Upsert one row per (dataset_key, class_id) - mirrors the old
    `_save_meta()` full-rewrite, just as N upserts instead of one file write.
    N is the class count (tens, not thousands), so this is cheap."""
    for class_id, name in enumerate(classes):
        color = colors.get(str(class_id))
        if color is None:
            continue
        _upsert_class(db, dataset_key, class_id, name, color)
    db.commit()


def set_class_color(db: Session, dataset_key: str, class_id: int, name: str, color: str) -> None:
    _upsert_class(db, dataset_key, class_id, name, color)
    db.commit()


def _upsert_class(db: Session, dataset_key: str, class_id: int, name: str, color: str) -> None:
    stmt = insert(DatasetClass).values(dataset_view=dataset_key, class_id=class_id, name=name, color=color)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_dataset_classes_view_class",
        set_={"name": stmt.excluded.name, "color": stmt.excluded.color},
    )
    db.execute(stmt)
