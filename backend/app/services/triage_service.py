"""Phase 2 triage/prioritization (see annotation_module_build_plan.md §2
Phase 2) - ranks not-yet-completed images into priority tiers so human
review time goes to the frames most worth it.

Only tiers buildable without pipeline data are implemented:
- "low_confidence": images whose already-saved auto-detected objects have low
  average **detector** confidence (see detector_service.detect(), which
  returns confidence per box - this module is why). Only covers images that
  have been through at least one save with detector-sourced objects; this is
  a local-detector proxy for the pipeline's own per-box `conf`, not the real
  thing the build plan describes (pipeline confidence doesn't exist here
  yet - Q-E, build plan §6).
  Note this tier reads `detector_confidence` specifically. It used to read a
  single `confidence` field that mask generation overwrote with SAM2's mask
  score (see AnnotationObject), so it was partly ranking by segmentation
  quality. Objects annotated before that split have no detector confidence
  and now land in "no_confidence_signal" instead of being ranked on a number
  that meant something else.
- "no_confidence_signal": images whose objects carry no detector confidence
  at all - boxes read straight from a YOLO label file, or pre-split data.
  Not one of the build plan's priority tiers; it exists so "we have no
  signal on these" is visible rather than an unexplained absence from
  low_confidence.
- "novel": frames far from everything else in the similarity index -
  inverted from propagation's normal use (finding near-duplicates to skip)
  per the build plan's tier 3. See similarity_service.novelty_scores().
- "routine": a small stable random sample of whatever's left, to catch
  silent drift the other tiers wouldn't surface.

"field_flagged" and "gate_recall_audit_miss" (tiers 1) always come back
empty - blocked on Q-E and an actual pipeline connection.
"""
from __future__ import annotations

import random

from app.models.schemas import TriageItem, TriageQueue
from app.services.dataset_service import DatasetService
from app.services.similarity_service import SimilarityService

LOW_CONFIDENCE_THRESHOLD = 0.5
LOW_CONFIDENCE_TOP_N = 50
NO_SIGNAL_TOP_N = 50
NOVEL_TOP_N = 50
ROUTINE_SAMPLE_N = 20
ROUTINE_SAMPLE_SEED = 42  # stable across calls, so repeated requests don't reshuffle the sample


def build_triage_queue(ds: DatasetService, similarity: SimilarityService) -> TriageQueue:
    items = [i for i in ds.list_images() if not i.completed]
    candidate_ids = {i.image_id for i in items}
    file_names = {i.image_id: i.file_name for i in items}

    states = ds.get_saved_states(list(candidate_ids))

    low_confidence = _low_confidence_tier(states, file_names)
    excluded = {t.image_id for t in low_confidence}

    no_confidence_signal = _no_signal_tier(states, file_names, excluded)
    excluded |= {t.image_id for t in no_confidence_signal}

    novel = _novel_tier(similarity, candidate_ids, file_names, excluded)
    excluded |= {t.image_id for t in novel}

    routine = _routine_tier(items, excluded)

    return TriageQueue(
        field_flagged=[],
        gate_recall_audit_miss=[],
        low_confidence=low_confidence,
        no_confidence_signal=no_confidence_signal,
        novel=novel,
        routine=routine,
    )


def _detector_confidences(state: dict) -> list[float]:
    """Detector confidences present on this image's saved objects. `None`
    (no signal - a plain YOLO label file carries no confidence field) is
    dropped rather than read as 0.0; see AnnotationObject on why those two
    are different facts."""
    return [
        o["detector_confidence"]
        for o in state.get("objects", [])
        if o.get("detector_confidence") is not None
    ]


def _low_confidence_tier(states: dict[str, dict], file_names: dict[str, str]) -> list[TriageItem]:
    scored: list[tuple[str, float]] = []
    for image_id, state in states.items():
        confidences = _detector_confidences(state)
        if not confidences:
            continue  # no detector signal - handled by _no_signal_tier
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence < LOW_CONFIDENCE_THRESHOLD:
            scored.append((image_id, avg_confidence))
    scored.sort(key=lambda pair: pair[1])  # lowest confidence first
    return [
        TriageItem(image_id=image_id, file_name=file_names[image_id], tier="low_confidence", score=score)
        for image_id, score in scored[:LOW_CONFIDENCE_TOP_N]
    ]


def _no_signal_tier(
    states: dict[str, dict], file_names: dict[str, str], excluded: set[str]
) -> list[TriageItem]:
    """Saved images that have objects but no detector confidence on any of
    them. Images with no saved state at all are left out - those have no
    objects to have a signal about, and the routine tier already samples
    from whatever is untouched."""
    return [
        TriageItem(image_id=image_id, file_name=file_names[image_id], tier="no_confidence_signal", score=0.0)
        for image_id, state in states.items()
        if image_id not in excluded and state.get("objects") and not _detector_confidences(state)
    ][:NO_SIGNAL_TOP_N]


def _novel_tier(
    similarity: SimilarityService, candidate_ids: set[str], file_names: dict[str, str], excluded: set[str]
) -> list[TriageItem]:
    scores = similarity.novelty_scores()
    scored = [
        (image_id, score) for image_id, score in scores.items() if image_id in candidate_ids and image_id not in excluded
    ]
    scored.sort(key=lambda pair: -pair[1])  # most novel first
    return [
        TriageItem(image_id=image_id, file_name=file_names[image_id], tier="novel", score=score)
        for image_id, score in scored[:NOVEL_TOP_N]
    ]


def _routine_tier(items, excluded: set[str]) -> list[TriageItem]:
    pool = [i for i in items if i.image_id not in excluded]
    rng = random.Random(ROUTINE_SAMPLE_SEED)
    sample = rng.sample(pool, min(ROUTINE_SAMPLE_N, len(pool)))
    return [TriageItem(image_id=i.image_id, file_name=i.file_name, tier="routine", score=0.0) for i in sample]
