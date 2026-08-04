"""Phase 2 triage/prioritization (see annotation_module_build_plan.md §2
Phase 2) - ranks not-yet-completed images into priority tiers so human
review time goes to the frames most worth it.

Only tiers buildable without pipeline data are implemented:
- "low_confidence": images whose already-saved auto-detected objects have
  low average confidence (see detector_service.detect(), which now returns
  confidence per box - this module is why). Only covers images that have
  been through at least one save with detector-sourced objects; this is a
  local-detector proxy for the pipeline's own per-box `conf`, not the real
  thing the build plan describes (pipeline confidence doesn't exist here
  yet - Q-E, build plan §6).
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
NOVEL_TOP_N = 50
ROUTINE_SAMPLE_N = 20
ROUTINE_SAMPLE_SEED = 42  # stable across calls, so repeated requests don't reshuffle the sample


def build_triage_queue(ds: DatasetService, similarity: SimilarityService) -> TriageQueue:
    items = [i for i in ds.list_images() if not i.completed]
    candidate_ids = {i.image_id for i in items}
    file_names = {i.image_id: i.file_name for i in items}

    low_confidence = _low_confidence_tier(ds, candidate_ids, file_names)
    excluded = {t.image_id for t in low_confidence}

    novel = _novel_tier(similarity, candidate_ids, file_names, excluded)
    excluded |= {t.image_id for t in novel}

    routine = _routine_tier(items, excluded)

    return TriageQueue(
        field_flagged=[],
        gate_recall_audit_miss=[],
        low_confidence=low_confidence,
        novel=novel,
        routine=routine,
    )


def _low_confidence_tier(ds: DatasetService, candidate_ids: set[str], file_names: dict[str, str]) -> list[TriageItem]:
    states = ds.get_saved_states(list(candidate_ids))
    scored: list[tuple[str, float]] = []
    for image_id, state in states.items():
        confidences = [o.get("confidence", 0.0) for o in state.get("objects", []) if o.get("confidence", 0.0) > 0]
        if not confidences:
            continue  # no confidence signal saved for this image yet
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence < LOW_CONFIDENCE_THRESHOLD:
            scored.append((image_id, avg_confidence))
    scored.sort(key=lambda pair: pair[1])  # lowest confidence first
    return [
        TriageItem(image_id=image_id, file_name=file_names[image_id], tier="low_confidence", score=score)
        for image_id, score in scored[:LOW_CONFIDENCE_TOP_N]
    ]


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
