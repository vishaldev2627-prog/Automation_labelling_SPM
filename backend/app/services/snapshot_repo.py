"""Postgres index over the immutable snapshot directories (M2).

The snapshot data lives on disk, content-addressed; this is the queryable index
over it. Kept separate from snapshot_service so that module stays pure
filesystem/hashing and can be tested without a database.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models.db_models import DatasetSnapshot


def upsert_snapshot(
    db: Session,
    *,
    snapshot_id: str,
    dataset_view: str,
    class_map_version: Optional[int],
    class_map_hash: Optional[str],
    manifest: dict,
    local_path: str,
    file_count: int,
    total_bytes: int,
    annotator_id: Optional[int],
) -> DatasetSnapshot:
    """Record a snapshot, or touch the existing row if this content was already
    snapshotted.

    On conflict only `last_exported_at` and `local_path` move. The manifest,
    creation time and creator are **not** overwritten: the snapshot is immutable,
    so those describe when this content first came into existence, and rewriting
    them would quietly turn an immutable artifact into a mutable one. `local_path`
    is refreshed because the directory can legitimately be relocated.
    """
    stmt = insert(DatasetSnapshot).values(
        snapshot_id=snapshot_id,
        dataset_view=dataset_view,
        class_map_version=class_map_version,
        class_map_hash=class_map_hash,
        manifest=manifest,
        local_path=local_path,
        file_count=file_count,
        total_bytes=total_bytes,
        created_by_id=annotator_id,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_dataset_snapshots_snapshot_id",
        set_={"last_exported_at": func.now(), "local_path": stmt.excluded.local_path},
    )
    db.execute(stmt)
    db.commit()
    return get_snapshot(db, snapshot_id)


def get_snapshot(db: Session, snapshot_id: str) -> Optional[DatasetSnapshot]:
    return db.execute(
        select(DatasetSnapshot).where(DatasetSnapshot.snapshot_id == snapshot_id)
    ).scalar_one_or_none()


def list_snapshots(db: Session, dataset_view: Optional[str] = None, limit: int = 100) -> list[DatasetSnapshot]:
    query = select(DatasetSnapshot).order_by(DatasetSnapshot.last_exported_at.desc()).limit(limit)
    if dataset_view is not None:
        query = query.where(DatasetSnapshot.dataset_view == dataset_view)
    return list(db.execute(query).scalars())


def mark_published(db: Session, snapshot_id: str, uri: str) -> None:
    db.execute(
        update(DatasetSnapshot)
        .where(DatasetSnapshot.snapshot_id == snapshot_id)
        .values(published_at=datetime.now(timezone.utc), published_uri=uri)
    )
    db.commit()
