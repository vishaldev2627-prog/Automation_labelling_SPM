"""Golden-set candidate selection (M4 population).

D-Q4 parked golden-set population on "named domain experts to curate" -
nobody had been assigned that job. The product call made instead: the
existing ~2000 `side_view` images are already second-reviewed (an
independent human re-checked every object), so that reviewed pool stands in
for fresh expert curation as the source of golden candidates, rather than
annotating a new dedicated batch from scratch.

This module only *proposes* a candidate list - it never writes to
`golden_sets`/`golden_set_items` itself. A `golden_curator` still calls the
existing `golden_service.create_version` + `add_items` (M4) with whatever
list they end up approving, so the actual permission gate and "who curated
this" record stay exactly as built - this just makes assembling that list
mechanical instead of a blind scroll through 2000 images.

Deliberately pure - no DatasetService/Postgres - same split as
export_service._enforce_split_integrity and golden_service._is_golden_curator:
the selection *rule* is what needs to be tested and trusted, and it should be
testable without standing up a database.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageLabelSummary:
    """The minimum this module needs to know about one candidate image -
    assembled by the caller from DatasetService + review_service, which do
    the actual DB/filesystem work."""

    image_id: str
    class_ids: frozenset[int]
    reviewed: bool


def propose_golden_candidates(
    images: list[ImageLabelSummary],
    safety_critical_class_ids: frozenset[int],
    target_count: int,
    min_per_class: int,
    seed: int = 42,
) -> list[str]:
    """Propose a stratified, deterministic golden-set candidate list.

    Only draws from `reviewed=True` images - unreviewed images are excluded
    from consideration entirely, not just deprioritized, since "verified and
    accurate" is the whole premise this selection stands on.

    Coverage comes first, target size second: every class present in the
    reviewed pool - safety_critical classes especially, since those are what
    the promotion gate (once built) checks per-class rather than on
    aggregate - gets up to `min_per_class` images greedily selected before
    anything else. Only after every class has its guaranteed minimum does
    the remainder get filled up to `target_count` via a seeded random sample,
    so re-running this with the same inputs and seed always proposes the
    same list - a curator's review of "does this list look right" isn't
    invalidated by re-running the script.

    Safety-critical classes are weighted higher in the coverage pass, so a
    tie between two images is broken in favor of the one covering a
    safety-relevant class - an early `target_count` cutoff (if the reviewed
    pool is small) should never leave a safety-relevant class at zero
    coverage while a cosmetic class already has its full share.

    The coverage pass is a greedy set-cover, not a fixed per-class loop: at
    each step it picks whichever remaining image covers the most
    still-outstanding class minimums (an image covering two needed classes
    at once is preferred over one covering only one), so reaching every
    class's minimum doesn't cost more images than it has to.

    May return more than `target_count` images if there are more classes
    than `target_count / min_per_class` allows - the per-class guarantee is
    never sacrificed to hit an exact count.
    """
    reviewed = [img for img in images if img.reviewed]
    all_class_ids = {c for img in reviewed for c in img.class_ids}

    rng = random.Random(seed)
    pool = list(reviewed)
    rng.shuffle(pool)  # tie-break order for the greedy pass and the fill pass alike

    needs = {class_id: min_per_class for class_id in all_class_ids}
    selected: set[str] = set()

    def _coverage_score(img: ImageLabelSummary) -> int:
        return sum(
            (2 if class_id in safety_critical_class_ids else 1)
            for class_id in img.class_ids
            if needs.get(class_id, 0) > 0
        )

    while any(v > 0 for v in needs.values()):
        candidates = [img for img in pool if img.image_id not in selected]
        if not candidates:
            break
        best = max(candidates, key=_coverage_score)
        if _coverage_score(best) == 0:
            break
        selected.add(best.image_id)
        for class_id in best.class_ids:
            if needs.get(class_id, 0) > 0:
                needs[class_id] -= 1

    fill_pool = [img.image_id for img in pool if img.image_id not in selected]
    shortfall = max(0, target_count - len(selected))
    selected.update(fill_pool[:shortfall])

    return sorted(selected)
