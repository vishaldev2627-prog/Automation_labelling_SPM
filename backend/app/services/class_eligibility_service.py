"""Class eligibility (Module 2 of the class-incremental promotion plan -
docs/mlflow_class_incremental_architecture.md §D/§E).

Computes, per class, whether it has crossed the composite data-sufficiency
gate (absolute instance count + image count + coach-type diversity +
projected validation-set presence, all per-tier) and drives the
discovered -> collecting_data -> eligible lifecycle transition.

**ACTIVE is sticky, and this module never demotes it.** Once a class is
ACTIVE (or DEPRECATED), this module treats both as terminal - it only ever
moves a class discovered -> collecting_data -> eligible, never the reverse,
and never touches an already-shipped class just because its instance count
or relative share later shrinks. Promoting a class to ACTIVE (once an
ELIGIBLE candidate actually passes the promotion gate) and deprecating a
class are both separate, deliberate actions elsewhere - this module only
ever answers "does this class have enough data to be worth training on
next," never "should this class currently be in production."

Reads Postgres via `DatasetService.get_saved_states` (the same bulk,
per-view read `review_service.get_class_audit_stats` already uses) rather
than `get_annotations()` per image - deliberately, since `get_annotations()`
on an image with no saved state falls back to running the live detector,
which would be both slow and wrong for a pure aggregation pass over
already-annotated data.
"""
from __future__ import annotations
from typing import Dict, Set

import logging
from dataclasses import dataclass, field

from app.config import Settings
from app.services.dataset_service import DatasetService
from app.services.detector_service import VAL_SPLIT as _TRAIN_VAL_SPLIT

logger = logging.getLogger(__name__)

# A class in either of these states is terminal to determine_state() - see
# the module docstring's "ACTIVE is sticky" note.
_TERMINAL_STATES = ("active", "deprecated")


@dataclass
class ClassEligibility:
    class_id: int
    instance_count: int
    image_count: int
    coach_types: Set[str] = field(default_factory=set)
    # instance_count * the training pipeline's own val split ratio - an
    # ESTIMATE (the real split is a random per-image shuffle, not a
    # per-class one), documented as such; good enough to catch "this class
    # has so few instances that even a generous split would starve val."
    projected_val_instances: float = 0.0
    relative_share: float = 0.0


@dataclass
class EligibilityThresholds:
    min_instances: Dict[str, int]
    min_images: Dict[str, int]
    min_coach_types: Dict[str, int]
    min_val_instances: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "EligibilityThresholds":
        return cls(
            min_instances={
                "safety": settings.eligibility_min_instances_safety,
                "structural": settings.eligibility_min_instances_structural,
                "cosmetic": settings.eligibility_min_instances_cosmetic,
            },
            min_images={
                "safety": settings.eligibility_min_images_safety,
                "structural": settings.eligibility_min_images_structural,
                "cosmetic": settings.eligibility_min_images_cosmetic,
            },
            min_coach_types={
                "safety": settings.eligibility_min_coach_types_safety,
                "structural": settings.eligibility_min_coach_types_structural,
                "cosmetic": settings.eligibility_min_coach_types_cosmetic,
            },
            min_val_instances=settings.eligibility_min_val_instances,
        )


def compute_eligibility(ds: DatasetService) -> Dict[int, ClassEligibility]:
    """One pass over every currently-completed image's saved state,
    aggregating per class_id. Non-completed images and rejected objects are
    excluded - the same trust boundary `detector_service._assemble_dataset`
    already uses, since this is answering "would training see enough of
    this class," not "how much has ever been drawn.\""""
    image_ids = ds.image_ids()
    states = ds.get_saved_states(image_ids)

    per_class: Dict[int, ClassEligibility] = {}
    seen_images: Dict[int, Set[str]] = {}
    total_instances = 0

    for image_id, state in states.items():
        if not state or not state.get("completed"):
            continue
        coach_type = state.get("coach_type", "unknown")
        for obj in state.get("objects", []):
            if obj.get("status") == "rejected":
                continue
            class_id = obj.get("class_id")
            if class_id is None:
                continue
            elig = per_class.setdefault(class_id, ClassEligibility(class_id=class_id, instance_count=0, image_count=0))
            elig.instance_count += 1
            elig.coach_types.add(coach_type)
            seen_images.setdefault(class_id, set()).add(image_id)
            total_instances += 1

    for class_id, elig in per_class.items():
        elig.image_count = len(seen_images.get(class_id, ()))
        elig.projected_val_instances = elig.instance_count * _TRAIN_VAL_SPLIT
        elig.relative_share = (elig.instance_count / total_instances) if total_instances else 0.0

    return per_class


def determine_state(
    current_state: str, elig: ClassEligibility, tier: str, thresholds: EligibilityThresholds
) -> str:
    """Never demotes 'active'/'deprecated' - sticky, per the module
    docstring. Only ever moves discovered -> collecting_data -> eligible
    upward for anything else."""
    if current_state in _TERMINAL_STATES:
        return current_state

    meets_floor = (
        elig.instance_count >= thresholds.min_instances[tier]
        and elig.image_count >= thresholds.min_images[tier]
        and len(elig.coach_types) >= thresholds.min_coach_types[tier]
        and elig.projected_val_instances >= thresholds.min_val_instances
    )
    if meets_floor:
        return "eligible"
    if elig.instance_count > 0:
        return "collecting_data"
    return "discovered"


def recompute_and_apply(ds: DatasetService, settings: Settings) -> Dict[int, str]:
    """Computes eligibility for every class currently in this view's class
    map and applies any resulting state transition. Returns
    {class_id: new_state} for every class whose state actually changed
    (an empty dict means nothing moved this cycle - the common case).

    Best-effort per class: one class's failure to update must not block
    recomputing the rest, mirroring this codebase's existing best-effort
    conventions (golden_eval_service, mlflow_tracking).
    """
    thresholds = EligibilityThresholds.from_settings(settings)
    eligibility = compute_eligibility(ds)
    changed: Dict[int, str] = {}

    for class_info in ds.get_classes():
        elig = eligibility.get(
            class_info.class_id,
            ClassEligibility(class_id=class_info.class_id, instance_count=0, image_count=0),
        )
        new_state = determine_state(class_info.state, elig, class_info.tier, thresholds)
        if new_state == class_info.state:
            continue
        try:
            ds.set_class_state(class_info.class_id, new_state)
            changed[class_info.class_id] = new_state
            logger.info(
                "Class %s (%s) in %s: %s -> %s (instances=%d, images=%d, coach_types=%d)",
                class_info.class_id, class_info.name, ds.dataset_key,
                class_info.state, new_state, elig.instance_count, elig.image_count, len(elig.coach_types),
            )
        except Exception:
            logger.exception(
                "Could not apply eligibility state change for class %s in %s; will retry next cycle",
                class_info.class_id, ds.dataset_key,
            )

    return changed
