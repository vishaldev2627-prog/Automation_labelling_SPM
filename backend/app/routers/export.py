"""Export endpoints.

By default an export is an **immutable, content-addressed snapshot** (M2, build
plan Phase 5) rather than an overwrite of a mutable directory. Publishing that
snapshot to the staging object store is a separate, retryable step, because the
snapshot is already safely on disk by then - see app/services/object_store.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.db import SessionLocal
from app.models.schemas import ExportRequest, SnapshotInfo
from app.services import object_store, snapshot_repo
from app.services.dataset_service import DatasetNotFoundError, get_dataset_service
from app.services.export_service import ExportService, SplitIntegrityViolation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/export", tags=["export"])


@router.post("")
def export_dataset(request: ExportRequest) -> dict:
    settings = get_settings()
    ds = get_dataset_service()
    try:
        ds.require_loaded()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    service = ExportService(ds, settings.exports_dir)
    try:
        return service.export(request)
    except SplitIntegrityViolation as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/snapshots", response_model=list[SnapshotInfo])
def list_snapshots(
    all_views: bool = Query(default=False, description="Include snapshots from other dataset views"),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[SnapshotInfo]:
    """Snapshots, most-recently-exported first.

    Scoped to the loaded view by default. `all_views=true` lifts that - useful
    because `snapshot_id` is a content hash and therefore globally unique, so two
    views producing identical data genuinely share one snapshot.
    """
    ds = get_dataset_service()
    dataset_view = None
    if not all_views:
        try:
            ds.require_loaded()
            dataset_view = ds.dataset_key
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    db = SessionLocal()
    try:
        return [_snapshot_info(s) for s in snapshot_repo.list_snapshots(db, dataset_view, limit)]
    finally:
        db.close()


@router.get("/snapshots/{snapshot_id}", response_model=SnapshotInfo)
def get_snapshot(snapshot_id: str) -> SnapshotInfo:
    db = SessionLocal()
    try:
        record = snapshot_repo.get_snapshot(db, snapshot_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"No snapshot '{snapshot_id}'")
        return _snapshot_info(record)
    finally:
        db.close()


@router.post("/snapshots/{snapshot_id}/publish")
def publish_snapshot(snapshot_id: str) -> dict:
    """Upload an existing snapshot to the staging object store.

    Separate from export on purpose: the snapshot is immutable, so retrying after
    a network failure is safe and resumes rather than restarting. Returns the
    failure reason rather than raising, for the same reason - a publish problem is
    not a data problem, and the data is already written.
    """
    settings = get_settings()
    db = SessionLocal()
    try:
        record = snapshot_repo.get_snapshot(db, snapshot_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"No snapshot '{snapshot_id}'")

        snapshot_dir = Path(record.local_path)
        if not snapshot_dir.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Snapshot directory {snapshot_dir} is missing on disk; cannot publish",
            )

        result = object_store.publish_snapshot(settings, snapshot_dir, snapshot_id)
        if result.published:
            snapshot_repo.mark_published(db, snapshot_id, f"s3://{result.bucket}/{result.prefix}")
        return result.as_dict()
    finally:
        db.close()


def _snapshot_info(record) -> SnapshotInfo:
    return SnapshotInfo(
        snapshot_id=record.snapshot_id,
        dataset_view=record.dataset_view,
        class_map_version=record.class_map_version,
        class_map_hash=record.class_map_hash,
        local_path=record.local_path,
        file_count=record.file_count,
        total_bytes=record.total_bytes,
        published_at=record.published_at,
        published_uri=record.published_uri,
        created_at=record.created_at,
        last_exported_at=record.last_exported_at,
        created_by=record.created_by.name if record.created_by else None,
        manifest=record.manifest,
    )
