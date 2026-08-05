"""Golden eval set endpoints (M4).

`golden_curator`-write-only (D-Q4): every write here goes through
golden_service.require_golden_curator, which 403s anyone else. Reads (listing
versions/items) are open, same as the rest of this tool's identity system -
there is nothing sensitive in knowing a golden set exists, only in being able
to write to it.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.db import SessionLocal
from app.models.schemas import (
    AddGoldenItemsRequest,
    CreateGoldenSetRequest,
    GoldenItemResult,
    GoldenSetInfo,
)
from app.services import golden_repo, golden_service
from app.services.dataset_service import DatasetNotFoundError, get_dataset_service
from app.session_context import get_current_annotator

router = APIRouter(prefix="/api/golden", tags=["golden"])


def _to_info(db, golden_set, item_count: int | None = None) -> GoldenSetInfo:
    return GoldenSetInfo(
        id=golden_set.id,
        dataset_view=golden_set.dataset_view,
        version=golden_set.version,
        description=golden_set.description,
        created_at=golden_set.created_at,
        created_by=golden_set.created_by.name if golden_set.created_by else None,
        item_count=len(golden_repo.list_items(db, golden_set.id)) if item_count is None else item_count,
    )


@router.get("/propose")
def propose_candidates(
    target_count: int = Query(default=200, ge=1, le=5000),
    min_per_class: int = Query(default=5, ge=1, le=100),
    treat_all_labeled_as_reviewed: bool = Query(
        default=False,
        description=(
            "Only set this when a curator has out-of-band confidence in the whole dataset "
            "(e.g. it predates this tool's second-review gate entirely, so no review trail "
            "will ever exist for it). Default False requires images to have actually passed "
            "second review."
        ),
    ),
) -> list[str]:
    """Read-only: propose a golden-set candidate list from the currently
    loaded view's reviewed images (see golden_service.propose_candidates for
    what "reviewed" means and when the override applies). Does not write
    anything - a curator reviews this list, then commits it via `POST /sets`
    + `POST /sets/{id}/items` same as any other golden-set write.
    """
    ds = get_dataset_service()
    try:
        ds.require_loaded()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db = SessionLocal()
    try:
        return golden_service.propose_candidates(
            db,
            ds,
            target_count=target_count,
            min_per_class=min_per_class,
            treat_all_labeled_as_reviewed=treat_all_labeled_as_reviewed,
        )
    finally:
        db.close()


@router.post("/sets", response_model=GoldenSetInfo)
def create_set(request: CreateGoldenSetRequest) -> GoldenSetInfo:
    annotator_id, _ = get_current_annotator()
    ds = get_dataset_service()
    try:
        ds.require_loaded()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db = SessionLocal()
    try:
        golden_set = golden_service.create_version(
            db,
            dataset_view=ds.dataset_key,
            description=request.description,
            annotator_id=annotator_id,
        )
        return _to_info(db, golden_set, item_count=0)
    except golden_service.NotGoldenCuratorError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    finally:
        db.close()


@router.get("/sets", response_model=list[GoldenSetInfo])
def list_sets(dataset_view: str | None = None) -> list[GoldenSetInfo]:
    db = SessionLocal()
    try:
        return [_to_info(db, v) for v in golden_repo.list_versions(db, dataset_view)]
    finally:
        db.close()


@router.get("/sets/{golden_set_id}", response_model=GoldenSetInfo)
def get_set(golden_set_id: int) -> GoldenSetInfo:
    db = SessionLocal()
    try:
        golden_set = golden_repo.get_version(db, golden_set_id)
        if golden_set is None:
            raise HTTPException(status_code=404, detail=f"No golden set with id {golden_set_id}")
        return _to_info(db, golden_set)
    finally:
        db.close()


@router.get("/sets/{golden_set_id}/items")
def list_items(golden_set_id: int) -> list[str]:
    db = SessionLocal()
    try:
        return [item.image_id for item in golden_repo.list_items(db, golden_set_id)]
    finally:
        db.close()


@router.post("/sets/{golden_set_id}/items", response_model=list[GoldenItemResult])
def add_items(golden_set_id: int, request: AddGoldenItemsRequest) -> list[GoldenItemResult]:
    annotator_id, _ = get_current_annotator()
    settings = get_settings()
    ds = get_dataset_service()
    try:
        ds.require_loaded()
    except DatasetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db = SessionLocal()
    try:
        golden_set = golden_repo.get_version(db, golden_set_id)
        if golden_set is None:
            raise HTTPException(status_code=404, detail=f"No golden set with id {golden_set_id}")
        # A version's dataset_view is fixed at creation (see GoldenSet). If the
        # currently loaded dataset is a different view, image_ids would
        # resolve against the wrong images/annotations entirely - refuse
        # rather than silently freezing the wrong frames under this version's
        # recorded view.
        if golden_set.dataset_view != ds.dataset_key:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Golden set {golden_set_id} belongs to dataset view "
                    f"'{golden_set.dataset_view}', but '{ds.dataset_key}' is currently loaded - "
                    "switch to the matching view before adding items."
                ),
            )

        results = golden_service.add_items(
            db,
            settings,
            ds,
            golden_set_id=golden_set_id,
            image_ids=request.image_ids,
            annotator_id=annotator_id,
        )
        return [
            GoldenItemResult(image_id=r.image_id, added=r.added, frozen=r.frozen, error=r.error)
            for r in results
        ]
    except golden_service.NotGoldenCuratorError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        db.close()
