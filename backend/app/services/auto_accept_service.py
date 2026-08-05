"""Confidence-based auto-accept for non-safety classes with a proven audit
track record - the "700-800 frames/coach shouldn't mean 700-800 manual
clicks" lever (plan §4.3/4.4, see annotation_module_build_plan.md).

Conservative by design, per product decision:
- Both confidence bars are high (0.95), not a "probably fine" bar - and there
  are now genuinely two of them. This module used to read a single
  `confidence` field which, after mask generation ran, held SAM2's *mask*
  score rather than the detector's class confidence (see AnnotationObject's
  docstring) - so it could skip human review because a polygon looked clean,
  not because the class was certain. An object now needs the detector to be
  sure about the class **and** SAM2 to be sure about the mask.
- An object with no detector confidence at all (`None` - a box read from a
  plain YOLO label file, or anything annotated before the two signals were
  split apart) is never eligible. Absence of signal is not evidence.
- A class is only eligible once it has a *proven* audit-sample track record
  (see review_service.get_class_audit_stats) - a class nobody has actually
  checked yet is never eligible, no matter how confident the detector is.
- safety_critical classes are never eligible, full stop - always a human,
  always a second reviewer, regardless of confidence or track record.
- Eligibility is evaluated per-image, all-or-nothing: an image with even
  one object outside the eligible set (low confidence, safety-critical,
  or an unproven class) is never a candidate, so a borderline object can't
  silently ride along with the rest of the frame past a human's eyes.

Never runs automatically or silently: find_candidates() only proposes;
bulk_accept() only acts on an explicit image_id list a caller chose after
seeing that list. Nothing in this module marks anything completed on a
timer, a schedule, or a dataset load.
"""
from __future__ import annotations

import time

from sqlalchemy.orm import Session

from app.models.schemas import ImageAnnotations, TriageItem
from app.services import annotation_state_repo as state_repo
from app.services import review_service
from app.services.annotator_service import SYSTEM_ANNOTATOR_NAME, get_or_create_annotator
from app.services.dataset_service import DatasetService

DETECTOR_CONFIDENCE_THRESHOLD = 0.95  # is the class right?
MASK_CONFIDENCE_THRESHOLD = 0.95  # is the polygon right?
MIN_AUDIT_SAMPLE = 10  # need at least this many audit_sample reviews of a class before trusting it at all
MIN_APPROVAL_RATE = 1.0  # zero tolerated rejections in that sample - conservative, not "mostly fine"

CANDIDATE_LIMIT = 200


def _object_is_acceptable(obj: dict, eligible_class_ids: set[int]) -> bool:
    """One object clears the bar: eligible class, detector sure of the class,
    SAM2 sure of the mask. Missing detector confidence fails closed."""
    if obj.get("class_id") not in eligible_class_ids:
        return False
    detector_confidence = obj.get("detector_confidence")
    if detector_confidence is None or detector_confidence < DETECTOR_CONFIDENCE_THRESHOLD:
        return False
    return obj.get("mask_confidence", 0.0) >= MASK_CONFIDENCE_THRESHOLD


def eligible_class_ids(db: Session, ds: DatasetService) -> set[int]:
    """Classes that clear the conservative bar: not safety-critical, and a
    proven zero-rejection audit track record over a minimum sample size."""
    stats = review_service.get_class_audit_stats(db, ds)
    classes_by_id = {c.class_id: c for c in ds.get_classes()}
    eligible: set[int] = set()
    for class_id_str, entry in stats.items():
        class_id = int(class_id_str)
        cls = classes_by_id.get(class_id)
        if cls is None or cls.safety_critical:
            continue
        if entry["reviewed"] < MIN_AUDIT_SAMPLE:
            continue
        approval_rate = entry["approved"] / entry["reviewed"]
        if approval_rate >= MIN_APPROVAL_RATE:
            eligible.add(class_id)
    return eligible


def find_candidates(db: Session, ds: DatasetService, limit: int = CANDIDATE_LIMIT) -> list[TriageItem]:
    """Not-yet-completed images where every object is a high-confidence
    instance of an eligible class. Preview only - does not mark anything
    completed; see bulk_accept()."""
    eligible = eligible_class_ids(db, ds)
    if not eligible:
        return []

    items = [i for i in ds.list_images() if not i.completed]
    states = ds.get_saved_states([i.image_id for i in items])

    candidates = []
    for item in items:
        state = states.get(item.image_id)
        if not state:
            continue  # never opened/saved - no confidence data to judge yet
        objects = state.get("objects", [])
        if not objects:
            continue  # nothing to auto-accept
        if all(_object_is_acceptable(o, eligible) for o in objects):
            candidates.append(TriageItem(image_id=item.image_id, file_name=item.file_name, tier="auto_accept", score=0.0))
        if len(candidates) >= limit:
            break
    return candidates


def bulk_accept(db: Session, ds: DatasetService, image_ids: list[str]) -> int:
    """Marks each image completed, attributed to the reserved system
    identity (never impersonating whoever's logged in), and records an
    approving review so it's immediately export-eligible - the whole point
    of the mechanism. Silently skips any id that's already completed or
    has no saved state, rather than erroring the whole batch over one
    stale id (the candidate list a caller acts on may be slightly stale by
    the time they submit it).

    **Every id is re-checked against the same bar find_candidates() used**,
    rather than trusted because it appeared in some earlier preview. The
    endpoint takes an arbitrary caller-supplied id list, so without this the
    "safety_critical classes never auto-accept, full stop" rule (plan §4.4)
    held only for callers that happened to ask politely - a stale preview, a
    hand-written POST, or a class flipped to safety-critical between preview
    and execute would all have walked straight through. A rejected id is
    skipped like any other stale one, not an error for the whole batch.
    """
    system = get_or_create_annotator(db, SYSTEM_ANNOTATOR_NAME)
    dataset_key = ds.dataset_key
    eligible = eligible_class_ids(db, ds)

    accepted = 0
    for image_id in image_ids:
        state = state_repo.get_state(db, dataset_key, image_id)
        if state is None:
            continue
        objects = state.get("objects", [])
        if not objects or not all(_object_is_acceptable(o, eligible) for o in objects):
            continue
        annotations = ImageAnnotations.model_validate(state)
        if annotations.completed:
            continue
        annotations.completed = True
        annotations.last_modified = time.time()
        state_repo.save_state(
            db, dataset_key, image_id, annotations.model_dump(mode="json"), True, system.id
        )
        review_service.submit_review(
            db,
            ds,
            image_id,
            system.id,
            "approved",
            "auto_accept",
            notes=(
                f"detector_confidence >= {DETECTOR_CONFIDENCE_THRESHOLD} and "
                f"mask_confidence >= {MASK_CONFIDENCE_THRESHOLD}, all classes audit-verified "
                f"(>= {MIN_AUDIT_SAMPLE} reviews, {MIN_APPROVAL_RATE:.0%} approval)"
            ),
        )
        accepted += 1
    return accepted
