"""Tests for golden-set candidate selection (M4 population).

The product call this defends: the existing ~2000 `side_view` images are
already second-reviewed, so that reviewed pool stands in for a fresh
dedicated curation pass, rather than blocking population entirely on naming
a domain expert (D-Q4). `propose_golden_candidates` still only *proposes* -
see golden_selection.py's module docstring for why the actual write path
(golden_service.create_version/add_items, M4) is untouched by this.

Pure, no DatasetService/Postgres - mirrors test_split_integrity.py and
test_golden_set.py's reasoning for staying DB-free.

    cd backend && python -m pytest tests/test_golden_selection.py
    cd backend && python -m tests.test_golden_selection   # no pytest
"""
from __future__ import annotations

from app.services.golden_selection import ImageLabelSummary, propose_golden_candidates


def _img(image_id: str, class_ids: list[int], reviewed: bool = True) -> ImageLabelSummary:
    return ImageLabelSummary(image_id=image_id, class_ids=frozenset(class_ids), reviewed=reviewed)


def test_unreviewed_images_are_never_selected() -> None:
    """The whole premise is 'verified and accurate' - an unreviewed image
    cannot stand in for that, no matter how much coverage it would add."""
    images = [_img("a", [0], reviewed=False), _img("b", [0], reviewed=True)]
    result = propose_golden_candidates(images, frozenset(), target_count=10, min_per_class=1)
    assert result == ["b"]


def test_every_class_gets_its_minimum_when_pool_allows() -> None:
    images = [
        _img("a1", [0]), _img("a2", [0]), _img("a3", [0]),
        _img("b1", [1]), _img("b2", [1]), _img("b3", [1]),
    ]
    result = propose_golden_candidates(images, frozenset(), target_count=2, min_per_class=2)
    # Coverage wins over target_count: 2 classes * 2 min_per_class = 4 minimum,
    # even though target_count asked for only 2.
    assert len(result) == 4
    assert any(r.startswith("a") for r in result)
    assert any(r.startswith("b") for r in result)


def test_safety_critical_classes_are_covered_even_under_a_tight_target() -> None:
    """A small target_count must never leave a safety-relevant class at zero
    coverage while a cosmetic class gets its full share first."""
    images = [_img(f"safety{i}", [0]) for i in range(5)] + [_img(f"cosmetic{i}", [1]) for i in range(5)]
    result = propose_golden_candidates(
        images, safety_critical_class_ids=frozenset({0}), target_count=2, min_per_class=1
    )
    assert any(r.startswith("safety") for r in result)


def test_result_is_deterministic_for_the_same_seed() -> None:
    images = [_img(f"img{i}", [i % 3]) for i in range(30)]
    first = propose_golden_candidates(images, frozenset(), target_count=10, min_per_class=1, seed=7)
    second = propose_golden_candidates(images, frozenset(), target_count=10, min_per_class=1, seed=7)
    assert first == second


def test_different_seeds_can_produce_different_fill_choices() -> None:
    """Not a hard requirement of any single run, but confirms the fill step
    actually consults the seed rather than always picking the same order
    images happened to be listed in."""
    images = [_img(f"img{i}", [0]) for i in range(20)]
    a = propose_golden_candidates(images, frozenset(), target_count=5, min_per_class=1, seed=1)
    b = propose_golden_candidates(images, frozenset(), target_count=5, min_per_class=1, seed=2)
    assert a != b


def test_never_selects_more_than_the_reviewed_pool() -> None:
    images = [_img("a", [0]), _img("b", [0])]
    result = propose_golden_candidates(images, frozenset(), target_count=1000, min_per_class=1)
    assert set(result) <= {"a", "b"}


def test_image_with_multiple_classes_counts_toward_each() -> None:
    """An image labeled with both class 0 and class 1 can satisfy both
    classes' minimums at once - the algorithm shouldn't require a separate
    image per class when one image already covers two."""
    images = [_img("multi", [0, 1]), _img("only0", [0]), _img("only1", [1])]
    result = propose_golden_candidates(images, frozenset(), target_count=1, min_per_class=1)
    assert "multi" in result


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - hand-rolled runner wants everything
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
