"""Tests for promotion_gate.py (Module 4 of the class-incremental promotion
plan - docs/mlflow_class_incremental_architecture.md §H/§I).

Pure functions, no database/MLflow/GPU. Every scenario named after the
architecture doc's own worked examples so a future reader can map
test->doc 1:1.

    cd backend && python -m pytest tests/test_promotion_gate.py
    cd backend && python -m tests.test_promotion_gate   # no pytest
"""
from __future__ import annotations

from app.services.promotion_gate import (
    ClassMetrics,
    ClassRegistryEntry,
    ClassRegistryView,
    ModelVersionMetrics,
    OperationalMetrics,
    PromotionThresholds,
    UnexplainedClassRemoval,
    compare,
    should_promote,
)

THRESHOLDS = PromotionThresholds(
    regression_tolerance_ap50={"safety": 0.5, "structural": 2.0, "cosmetic": 3.0},
    new_class_floor_ap50={"safety": 85.0, "structural": 70.0, "cosmetic": 55.0},
    min_val_instances=15,
    max_latency_p95_ms=500.0,
    max_model_size_mb=200.0,
    max_false_positive_rate={"safety": 0.02, "structural": 0.05, "cosmetic": 0.10},
)

NO_OPERATIONAL_ISSUES = OperationalMetrics(latency_p95_ms=100.0, model_size_mb=20.0, false_positive_rate={})


def _registry(tiers: dict[int, str], states: dict[int, str] = None, ever_active: dict[int, bool] = None):
    states = states or {}
    ever_active = ever_active or {}
    return ClassRegistryView(
        {
            cid: ClassRegistryEntry(tier=tier, state=states.get(cid, "active"), ever_active=ever_active.get(cid, False))
            for cid, tier in tiers.items()
        }
    )


def _metrics(ap50s: dict[int, float], n_val: int = 20) -> ModelVersionMetrics:
    return ModelVersionMetrics(
        per_class={cid: ClassMetrics(class_id=cid, ap50=ap, n_val_instances=n_val) for cid, ap in ap50s.items()}
    )


# ------------------------------------------------------------- compare()

def test_common_class_regression_beyond_tolerance_rejects() -> None:
    """The doc's Example 9: A+1,B+1,C-2,D-6 - D's -6pt trips the check."""
    production = _metrics({0: 91, 1: 89, 2: 87, 3: 90})  # A B C D
    candidate = _metrics({0: 92, 1: 90, 2: 88, 3: 84, 4: 76})  # A B C D E
    registry = _registry({0: "structural", 1: "structural", 2: "structural", 3: "structural", 4: "structural"})
    comparison = compare(candidate, production, registry, THRESHOLDS)
    decision = should_promote(comparison, NO_OPERATIONAL_ISSUES, registry, THRESHOLDS)
    assert decision.decision == "REJECT"
    assert any("3" in r or "[3]" in r for r in decision.reasons)


def test_common_class_small_wobble_within_tolerance_passes() -> None:
    production = _metrics({0: 80.0})
    candidate = _metrics({0: 79.7})  # -0.3pt, within cosmetic's 3.0pt tolerance
    registry = _registry({0: "cosmetic"})
    comparison = compare(candidate, production, registry, THRESHOLDS)
    decision = should_promote(comparison, NO_OPERATIONAL_ISSUES, registry, THRESHOLDS)
    assert decision.decision == "PROMOTE"


def test_safety_tier_regression_is_hard_fail_regardless_of_everything_else() -> None:
    production = _metrics({0: 90.0})
    candidate = _metrics({0: 85.0, 1: 90.0})  # class 0 regressed 5pt (safety tolerance is 0.5), class 1 is a perfect new class
    registry = _registry({0: "safety", 1: "structural"})
    comparison = compare(candidate, production, registry, THRESHOLDS)
    decision = should_promote(comparison, NO_OPERATIONAL_ISSUES, registry, THRESHOLDS)
    assert decision.decision == "REJECT"
    assert decision.hard_fail is True
    assert any("HARD FAIL" in r for r in decision.reasons)


def test_new_class_with_no_baseline_checked_against_absolute_floor_not_skipped() -> None:
    """The actual bug being fixed: today's `continue` (model_registry_
    service.py, verified live) meant a new class contributed nothing to
    the verdict. Here it must be checked and can reject on its own."""
    candidate = _metrics({0: 40.0})  # well below structural's 70.0 floor
    registry = _registry({0: "structural"})
    comparison = compare(candidate, None, registry, THRESHOLDS)
    decision = should_promote(comparison, NO_OPERATIONAL_ISSUES, registry, THRESHOLDS)
    assert decision.decision == "REJECT"
    assert any("0" in r for r in decision.reasons)
    assert decision.hard_fail is False  # ordinary reject, not a hard fail


def test_new_class_meeting_floor_with_existing_classes_improved_promotes() -> None:
    """The doc's Example 10: V2's overall mAP is lower purely because E is
    hard, but every existing class improved and E clears its floor - must
    still promote."""
    production = _metrics({0: 90, 1: 85, 2: 80, 3: 88})
    candidate = _metrics({0: 92, 1: 88, 2: 81, 3: 90, 4: 70})  # E=70, exactly at structural's floor
    registry = _registry({0: "structural", 1: "structural", 2: "structural", 3: "structural", 4: "structural"})
    comparison = compare(candidate, production, registry, THRESHOLDS)
    decision = should_promote(comparison, NO_OPERATIONAL_ISSUES, registry, THRESHOLDS)
    assert decision.decision == "PROMOTE"
    # overall is lower purely because E drags the mean - reported, never gating
    assert comparison.overall["candidate_map50"] < comparison.overall["production_map50"]


def test_silently_dropped_class_without_deprecation_raises() -> None:
    production = _metrics({0: 90.0, 1: 85.0})
    candidate = _metrics({0: 90.0})  # class 1 silently absent
    registry = _registry({0: "structural", 1: "structural"}, states={1: "active"})  # NOT deprecated
    try:
        compare(candidate, production, registry, THRESHOLDS)
        raise AssertionError("expected UnexplainedClassRemoval")
    except UnexplainedClassRemoval as exc:
        assert exc.class_ids == [1]


def test_explicitly_deprecated_class_removal_is_accepted() -> None:
    production = _metrics({0: 90.0, 1: 85.0})
    candidate = _metrics({0: 90.0})
    registry = _registry({0: "structural", 1: "structural"}, states={1: "deprecated"})
    comparison = compare(candidate, production, registry, THRESHOLDS)
    assert comparison.removed_classes == [1]
    decision = should_promote(comparison, NO_OPERATIONAL_ISSUES, registry, THRESHOLDS)
    assert decision.decision == "PROMOTE"


def test_reintroduced_class_routes_through_new_class_layer_not_regression_layer() -> None:
    """A previously-active-then-deprecated class reappearing must be
    checked against the absolute floor (as new), never against its stale
    historical production score (which doesn't even exist here since it
    was removed from production before this candidate)."""
    production = _metrics({0: 90.0})  # class 1 not in production at all (it was deprecated before this baseline)
    candidate = _metrics({0: 90.0, 1: 40.0})  # class 1 reintroduced, weak
    registry = _registry({0: "structural", 1: "structural"}, ever_active={1: True})
    comparison = compare(candidate, production, registry, THRESHOLDS)
    assert 1 in comparison.reintroduced_classes
    assert 1 not in comparison.common_classes
    assert 1 in comparison.new_classes
    decision = should_promote(comparison, NO_OPERATIONAL_ISSUES, registry, THRESHOLDS)
    assert decision.decision == "REJECT"  # 40.0 < structural's 70.0 floor
    assert decision.hard_fail is False  # a failed new-class floor, not a regression


def test_no_production_baseline_at_all_treats_every_class_as_new() -> None:
    candidate = _metrics({0: 90.0, 1: 88.0})
    registry = _registry({0: "structural", 1: "structural"})
    comparison = compare(candidate, None, registry, THRESHOLDS)
    assert comparison.common_classes == {}
    assert set(comparison.new_classes.keys()) == {0, 1}
    assert comparison.overall["production_map50"] is None
    decision = should_promote(comparison, NO_OPERATIONAL_ISSUES, registry, THRESHOLDS)
    assert decision.decision == "PROMOTE"  # both clear structural's 70.0 floor


# ------------------------------------------------------------- should_promote() operational layer

def test_operational_latency_regression_rejects_even_with_perfect_accuracy() -> None:
    candidate = _metrics({0: 95.0})
    registry = _registry({0: "structural"})
    comparison = compare(candidate, None, registry, THRESHOLDS)
    operational = OperationalMetrics(latency_p95_ms=600.0, model_size_mb=20.0)
    decision = should_promote(comparison, operational, registry, THRESHOLDS)
    assert decision.decision == "REJECT"
    assert any("latency" in r for r in decision.reasons)


def test_operational_model_size_regression_rejects() -> None:
    candidate = _metrics({0: 95.0})
    registry = _registry({0: "structural"})
    comparison = compare(candidate, None, registry, THRESHOLDS)
    operational = OperationalMetrics(latency_p95_ms=100.0, model_size_mb=999.0)
    decision = should_promote(comparison, operational, registry, THRESHOLDS)
    assert decision.decision == "REJECT"
    assert any("model size" in r for r in decision.reasons)


def test_operational_false_positive_rate_exceeds_tier_budget_rejects() -> None:
    candidate = _metrics({0: 95.0})
    registry = _registry({0: "safety"})
    comparison = compare(candidate, None, registry, THRESHOLDS)
    operational = OperationalMetrics(latency_p95_ms=100.0, model_size_mb=20.0, false_positive_rate={0: 0.10})
    decision = should_promote(comparison, operational, registry, THRESHOLDS)
    assert decision.decision == "REJECT"
    assert any("false-positive" in r for r in decision.reasons)


def test_false_positive_rate_within_budget_passes() -> None:
    candidate = _metrics({0: 95.0})
    registry = _registry({0: "cosmetic"})
    comparison = compare(candidate, None, registry, THRESHOLDS)
    operational = OperationalMetrics(latency_p95_ms=100.0, model_size_mb=20.0, false_positive_rate={0: 0.05})
    decision = should_promote(comparison, operational, registry, THRESHOLDS)
    assert decision.decision == "PROMOTE"


def test_class_set_shrink_without_deprecation_is_defense_in_depth_hard_fail() -> None:
    """compare() already raises before this is reachable through normal
    use - this locks down should_promote()'s own belt-and-suspenders
    re-check directly, bypassing compare()."""
    from app.services.promotion_gate import ComparisonResult

    comparison = ComparisonResult(
        common_classes={}, new_classes={}, removed_classes=[], reintroduced_classes=[],
        overall={"candidate_map50": 90.0, "production_map50": 90.0},
        candidate_class_set={0}, production_class_set={0, 1},  # class 1 vanished
    )
    registry = _registry({0: "structural", 1: "structural"})
    decision = should_promote(comparison, NO_OPERATIONAL_ISSUES, registry, THRESHOLDS)
    assert decision.decision == "REJECT"
    assert decision.hard_fail is True


def test_reasons_list_contains_every_failing_check_not_just_the_first() -> None:
    production = _metrics({0: 90.0})
    candidate = _metrics({0: 80.0})  # -10pt regression on a structural class (tolerance 2.0)
    registry = _registry({0: "structural"})
    comparison = compare(candidate, production, registry, THRESHOLDS)
    operational = OperationalMetrics(latency_p95_ms=600.0, model_size_mb=20.0)  # also breaches latency
    decision = should_promote(comparison, operational, registry, THRESHOLDS)
    assert any("regression" in r for r in decision.reasons)
    assert any("latency" in r for r in decision.reasons)
    assert len(decision.reasons) >= 2


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
