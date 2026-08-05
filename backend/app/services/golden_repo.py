"""Postgres index over the golden eval set (M4).

Mirrors snapshot_repo's split: this module is the queryable index (which
versions exist, which images are in them); app/services/golden_service.py
owns the curation workflow (role check, freezing bytes to the separate
bucket) built on top of it.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.db_models import GoldenSet, GoldenSetItem


def create_version(
    db: Session, *, dataset_view: str, description: Optional[str], annotator_id: Optional[int]
) -> GoldenSet:
    """Mint the next version number for this view. Never reused, never
    edited after creation - a curation round always creates a new version
    rather than mutating an old one (D-Q4: "we curate and version")."""
    next_version = (
        db.execute(
            select(func.coalesce(func.max(GoldenSet.version), 0)).where(
                GoldenSet.dataset_view == dataset_view
            )
        ).scalar_one()
        + 1
    )
    golden_set = GoldenSet(
        dataset_view=dataset_view,
        version=next_version,
        description=description,
        created_by_id=annotator_id,
    )
    db.add(golden_set)
    db.commit()
    db.refresh(golden_set)
    return golden_set


def get_version(db: Session, golden_set_id: int) -> Optional[GoldenSet]:
    return db.execute(select(GoldenSet).where(GoldenSet.id == golden_set_id)).scalar_one_or_none()


def list_versions(db: Session, dataset_view: Optional[str] = None) -> list[GoldenSet]:
    query = select(GoldenSet).order_by(GoldenSet.dataset_view, GoldenSet.version.desc())
    if dataset_view is not None:
        query = query.where(GoldenSet.dataset_view == dataset_view)
    return list(db.execute(query).scalars())


def add_item(
    db: Session, *, golden_set_id: int, image_id: str, frozen_object_store_key: Optional[str]
) -> GoldenSetItem:
    item = GoldenSetItem(
        golden_set_id=golden_set_id, image_id=image_id, frozen_object_store_key=frozen_object_store_key
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def list_items(db: Session, golden_set_id: int) -> list[GoldenSetItem]:
    return list(
        db.execute(
            select(GoldenSetItem).where(GoldenSetItem.golden_set_id == golden_set_id)
        ).scalars()
    )


def get_golden_image_ids(db: Session, dataset_view: str) -> set[str]:
    """Every image_id ever frozen into a golden set for this view, across
    **all** versions - not just the latest. Once an image is golden it must
    stay disjoint from every export split permanently (D-Q4: "nothing that
    trains ever sees it"), not just until the next curation round."""
    rows = db.execute(
        select(GoldenSetItem.image_id)
        .join(GoldenSet, GoldenSetItem.golden_set_id == GoldenSet.id)
        .where(GoldenSet.dataset_view == dataset_view)
    ).scalars()
    return set(rows)


def is_golden_image(db: Session, dataset_view: str, image_id: str) -> bool:
    return (
        db.execute(
            select(GoldenSetItem.id)
            .join(GoldenSet, GoldenSetItem.golden_set_id == GoldenSet.id)
            .where(GoldenSet.dataset_view == dataset_view, GoldenSetItem.image_id == image_id)
            .limit(1)
        ).scalar_one_or_none()
        is not None
    )
