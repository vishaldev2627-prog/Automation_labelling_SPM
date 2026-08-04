"""Phase 4 QA/audit layer (safety-critical - see annotation_module_build_plan.md):
second-reviewer sign-off and mandatory audit sampling of propagated
annotations. Backed by app.models.db_models.AnnotationReview, append-only.

Wired into export gating via get_export_eligible_ids(): an image is
export-eligible if it's in ExportGateExemption (grandfathered - completed
before this gate existed) or its *latest* review decision is "approved".
Images completed after the gate went live need an approved review before
export_service.py will include them.
"""
from __future__ import annotations

import random
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.db_models import AnnotationReview, Annotator, ExportGateExemption
from app.models.schemas import ReviewRecord, TriageItem
from app.services.dataset_service import DatasetService

VALID_DECISIONS = ("approved", "rejected")
VALID_REASONS = ("second_review", "audit_sample")

AUDIT_SAMPLE_RATE = 0.075  # 5-10% per the plan's Phase 4 - middle of that range
AUDIT_SAMPLE_SEED = 42  # stable across calls, like the Phase 2 routine tier


def submit_review(
    db: Session,
    ds: DatasetService,
    image_id: str,
    reviewer_id: int,
    decision: str,
    reason: str,
    notes: Optional[str] = None,
) -> AnnotationReview:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {VALID_DECISIONS}")
    if reason not in VALID_REASONS:
        raise ValueError(f"reason must be one of {VALID_REASONS}")

    dataset_key = ds.dataset_key
    from app.models.db_models import AnnotationState

    row = db.execute(
        select(AnnotationState.id, AnnotationState.updated_by_id).where(
            AnnotationState.dataset_view == dataset_key, AnnotationState.image_id == image_id
        )
    ).first()
    if row is None:
        raise LookupError(f"No saved annotation state for '{image_id}' - it must be saved before it can be reviewed")
    _, submitted_by_id = row

    # Historical/backfilled rows have no known submitter (updated_by_id is
    # None) - can't enforce "different reviewer" against an unknown person,
    # so audit work on that data isn't blocked by this check.
    if reason == "second_review" and submitted_by_id is not None and submitted_by_id == reviewer_id:
        raise ValueError("Second reviewer must be different from the annotator who submitted this image")

    review = AnnotationReview(
        dataset_view=dataset_key,
        image_id=image_id,
        reviewer_id=reviewer_id,
        decision=decision,
        reason=reason,
        notes=notes,
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


def get_latest_review(db: Session, ds: DatasetService, image_id: str) -> Optional[ReviewRecord]:
    row = (
        db.query(AnnotationReview, Annotator.name)
        .join(Annotator, Annotator.id == AnnotationReview.reviewer_id)
        .filter(AnnotationReview.dataset_view == ds.dataset_key, AnnotationReview.image_id == image_id)
        .order_by(AnnotationReview.created_at.desc())
        .first()
    )
    if row is None:
        return None
    review, reviewer_name = row
    return ReviewRecord(
        id=review.id,
        image_id=review.image_id,
        reviewer_id=review.reviewer_id,
        reviewer_name=reviewer_name,
        decision=review.decision,
        reason=review.reason,
        notes=review.notes,
        created_at=review.created_at,
    )


def _reviewed_image_ids(db: Session, dataset_key: str, reason: str) -> set[str]:
    rows = db.execute(
        select(AnnotationReview.image_id)
        .where(AnnotationReview.dataset_view == dataset_key, AnnotationReview.reason == reason)
        .distinct()
    ).all()
    return {r[0] for r in rows}


def get_pending_second_review(db: Session, ds: DatasetService, limit: int = 100) -> list[TriageItem]:
    """Completed images that need a second_review decision to become
    export-eligible - excludes grandfathered (exempt) images, since those
    don't need review to export regardless of review history."""
    dataset_key = ds.dataset_key
    items = [i for i in ds.list_images() if i.completed]
    reviewed = _reviewed_image_ids(db, dataset_key, "second_review")
    exempt = _exempt_image_ids(db, dataset_key, [i.image_id for i in items])
    pending = [i for i in items if i.image_id not in reviewed and i.image_id not in exempt]
    return [
        TriageItem(image_id=i.image_id, file_name=i.file_name, tier="pending_review", score=0.0)
        for i in pending[:limit]
    ]


def _exempt_image_ids(db: Session, dataset_key: str, image_ids: list[str]) -> set[str]:
    if not image_ids:
        return set()
    rows = db.execute(
        select(ExportGateExemption.image_id).where(
            ExportGateExemption.dataset_view == dataset_key, ExportGateExemption.image_id.in_(image_ids)
        )
    ).all()
    return {r[0] for r in rows}


def get_export_eligible_ids(db: Session, dataset_key: str, image_ids: list[str]) -> set[str]:
    """Union of grandfathered-exempt images and images whose *latest*
    review decision is "approved" - a later "rejected" after an earlier
    "approved" correctly un-approves it, since this always takes the most
    recent decision per image, not "approved at least once."
    """
    if not image_ids:
        return set()
    exempt = _exempt_image_ids(db, dataset_key, image_ids)

    rows = db.execute(
        select(AnnotationReview.image_id, AnnotationReview.decision)
        .where(AnnotationReview.dataset_view == dataset_key, AnnotationReview.image_id.in_(image_ids))
        .order_by(AnnotationReview.created_at.asc())
    ).all()
    latest_decision: dict[str, str] = {}
    for image_id, decision in rows:
        latest_decision[image_id] = decision  # later rows overwrite earlier ones - ascending order
    approved = {image_id for image_id, decision in latest_decision.items() if decision == "approved"}

    return exempt | approved


def get_audit_sample(db: Session, ds: DatasetService, sample_rate: float = AUDIT_SAMPLE_RATE) -> list[TriageItem]:
    """A stable random sample of completed images containing propagated
    objects (source == "propagated"), excluding ones already audit-sampled.
    Sample size is `sample_rate` of the total propagated-completed pool -
    the plan's mandatory 5-10% audit of propagation, not a fixed count.
    """
    dataset_key = ds.dataset_key
    items = [i for i in ds.list_images() if i.completed]
    states = ds.get_saved_states([i.image_id for i in items])
    propagated = [
        i
        for i in items
        if any(o.get("source") == "propagated" for o in states.get(i.image_id, {}).get("objects", []))
    ]
    if not propagated:
        return []

    already_audited = _reviewed_image_ids(db, dataset_key, "audit_sample")
    pool = [i for i in propagated if i.image_id not in already_audited]

    target_n = max(1, round(len(propagated) * sample_rate))
    rng = random.Random(AUDIT_SAMPLE_SEED)
    sample = rng.sample(pool, min(target_n, len(pool)))
    return [TriageItem(image_id=i.image_id, file_name=i.file_name, tier="audit_sample", score=0.0) for i in sample]
