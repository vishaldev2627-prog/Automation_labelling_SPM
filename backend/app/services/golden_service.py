"""Curation workflow for the golden eval set (M4).

Owns the one thing golden_repo (pure DB) and object_store (pure upload)
deliberately don't: the permission gate. `golden_curator`-only, per D-Q4 -
"we curate and version; they run the eval" - and the build plan's explicit
risk that the only contamination mitigation that counts is that no
propagation, triage, or export path can write here.

Freezing to the separate object-store bucket is best-effort, same contract
as snapshot publishing (M2): the golden_set_items DB row is the source of
truth for export exclusion and is written regardless of whether the object
store copy succeeds, because a curator's decision that "this image is golden"
must not be lost to a transient MinIO outage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from app.config import Settings
from app.models.db_models import Annotator, GoldenSet
from app.models.schemas import ObjectStatus
from app.services import golden_repo, object_store, review_service
from app.services.dataset_service import DatasetService
from app.services.golden_selection import ImageLabelSummary, propose_golden_candidates
from app.utils.yolo_utils import format_segmentation_line


class NotGoldenCuratorError(Exception):
    """Raised when a non-curator attempts a golden-set write."""


@dataclass
class AddItemResult:
    image_id: str
    added: bool
    frozen: bool
    error: Optional[str]


def _is_golden_curator(annotator: Optional[Annotator]) -> bool:
    """The actual permission predicate, pulled out as a pure function of an
    already-fetched Annotator so it's testable without a database - mirrors
    export_service._enforce_split_integrity's split from its DB-touching
    caller."""
    return annotator is not None and annotator.role == "golden_curator"


def require_golden_curator(db: Session, annotator_id: Optional[int]) -> None:
    """The one real permission boundary this module enforces. Everything
    else in this codebase's annotator-identity system is a name, not a
    login (see app/routers/annotator.py) - this is deliberately the
    exception, because contamination of the golden set is the build plan's
    named [MEDIUM] risk and "assert it in tests" is explicit in the M4 spec.
    """
    if annotator_id is None:
        raise NotGoldenCuratorError("No identified annotator for this session")
    annotator = db.query(Annotator).filter(Annotator.id == annotator_id).one_or_none()
    if not _is_golden_curator(annotator):
        raise NotGoldenCuratorError(f"Annotator {annotator_id} is not a golden_curator")


def propose_candidates(
    db: Session,
    ds: DatasetService,
    *,
    target_count: int,
    min_per_class: int,
    treat_all_labeled_as_reviewed: bool = False,
) -> list[str]:
    """Read-only: assemble this view's real data and run it through
    `golden_selection.propose_golden_candidates`. Never writes to
    `golden_sets`/`golden_set_items` - a curator still has to call
    `create_version` + `add_items` with whatever list they approve, same
    permission gate as any other golden-set write.

    By default, "reviewed" means **second-reviewed** specifically
    (`review_service`'s `"second_review"` reason) - not the broader
    export-eligible set, which also includes grandfathered/exempt images
    that were never actually independently double-checked. Golden-set
    population stands on "verified and accurate"; a grandfathered image is
    exactly the case where that claim doesn't hold *by default*.

    `treat_all_labeled_as_reviewed=True` overrides that for a dataset a
    curator already has out-of-band confidence in. Concretely: the original
    ~2000 `side_view` images predate this tool's second-review gate
    entirely, so `annotation_reviews` has no rows for them at all - the gate
    can't retroactively certify data it never saw. A curator who already
    trusts that dataset (it was independently verified before this tool's
    review workflow existed) can say so explicitly with this flag rather
    than the selection silently returning nothing because a review trail
    that was never going to exist doesn't exist. Left `False` by default so
    a *future*, not-yet-verified dataset doesn't accidentally get the same
    pass.
    """
    image_ids = ds.image_ids()
    reviewed_ids = review_service.get_reviewed_image_ids(db, ds.dataset_key, "second_review")
    safety_critical_ids = frozenset(c.class_id for c in ds.get_classes() if c.safety_critical)

    summaries: list[ImageLabelSummary] = []
    for image_id in image_ids:
        annotations = ds.get_annotations(image_id)
        class_ids = frozenset(
            obj.class_id for obj in annotations.objects if obj.status != ObjectStatus.REJECTED
        )
        if not class_ids:
            continue
        reviewed = treat_all_labeled_as_reviewed or image_id in reviewed_ids
        summaries.append(ImageLabelSummary(image_id=image_id, class_ids=class_ids, reviewed=reviewed))

    return propose_golden_candidates(
        summaries,
        safety_critical_class_ids=safety_critical_ids,
        target_count=target_count,
        min_per_class=min_per_class,
    )


def create_version(
    db: Session, *, dataset_view: str, description: Optional[str], annotator_id: Optional[int]
) -> GoldenSet:
    require_golden_curator(db, annotator_id)
    return golden_repo.create_version(
        db, dataset_view=dataset_view, description=description, annotator_id=annotator_id
    )


def add_items(
    db: Session,
    settings: Settings,
    ds: DatasetService,
    *,
    golden_set_id: int,
    image_ids: list[str],
    annotator_id: Optional[int],
) -> list[AddItemResult]:
    """Freeze each image's current annotations into the golden set.

    A frozen label reflects this image's state *right now* - the same
    "curate and version" contract as class-map versions: if the image is
    re-annotated later, that change does not retroactively alter what this
    golden version means. A later curation round adding the same image_id to
    a *new* version is how a re-curation is expressed.
    """
    require_golden_curator(db, annotator_id)
    golden_set = golden_repo.get_version(db, golden_set_id)
    if golden_set is None:
        raise LookupError(f"No golden set version with id {golden_set_id}")

    results: list[AddItemResult] = []
    for image_id in image_ids:
        annotations = ds.get_annotations(image_id)
        objects = [
            (obj.class_id, obj.polygon)
            for obj in annotations.objects
            if obj.status != ObjectStatus.REJECTED and len(obj.polygon) >= 3
        ]
        label_text = "".join(format_segmentation_line(cid, poly) + "\n" for cid, poly in objects)

        image_path = ds.get_image_path(image_id)
        publish = object_store.freeze_golden_item(
            settings,
            dataset_view=golden_set.dataset_view,
            version=golden_set.version,
            image_id=image_id,
            image_path=image_path,
            label_text=label_text,
        )
        frozen_key = f"{publish.bucket}/{publish.prefix}" if publish.published else None
        golden_repo.add_item(
            db, golden_set_id=golden_set_id, image_id=image_id, frozen_object_store_key=frozen_key
        )
        results.append(
            AddItemResult(image_id=image_id, added=True, frozen=publish.published, error=publish.error)
        )
    return results
