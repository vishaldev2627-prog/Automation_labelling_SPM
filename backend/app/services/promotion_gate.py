"""The class-incremental promotion decision engine (Module 4 -
docs/mlflow_class_incremental_architecture.md §H/§I).

Pure functions only: `compare()` and `should_promote()` never touch MLflow,
Postgres, or the filesystem - they take plain dataclasses in, return plain
dataclasses out. `model_registry_service.py` is the only module that wires
this to real MLflow/DatasetService data (Module 5); this file must stay
importable and fully testable without either.

The central problem this closes (verified live against this repo's actual
code before this module existed): `model_registry_service._compare_
against_production` used to `continue` past any class missing from the
production run's tags - a brand-new class contributed nothing to the
verdict, in either direction, and a zero-tolerance regression check meant
either false-positive noise or a check nobody trusted enough to gate on.
Here, a new class is checked against an absolute per-tier floor (Layer 2),
a common class against a per-tier regression tolerance (Layer 1), overall
accuracy is reported but never gates by itself (Layer 3), and operational
constraints are checked independently (Layer 4) - `should_promote()`
combines all four into one decision with every failing reason listed, not
just the first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from app.config import Settings


class UnexplainedClassRemoval(Exception):
    """A class present in production's class set is absent from the
    candidate's, without having been explicitly marked DEPRECATED first.
    The only legitimate way a class disappears from what a model trains on
    is a prior, deliberate deprecation - anything else is treated as a bug
    in dataset assembly, and compare() refuses to produce a result rather
    than silently treating it as an intentional removal."""

    def __init__(self, class_ids: List[int]) -> None:
        self.class_ids = class_ids
        super().__init__(
            f"Class(es) {class_ids} are absent from the candidate's class set but were never "
            f"explicitly deprecated - refusing to treat this as an intentional removal"
        )


@dataclass
class ClassMetrics:
    """One class's golden-eval numbers for one model version. Defaults are
    zero/empty, not because that's a meaningful value, but so this dataclass
    stays constructible before every field has a real source wired up
    (n_val_instances/false_positive_rate/false_negative_rate specifically
    depend on golden_eval_service extensions - Module 7 - landing)."""

    class_id: int
    precision: float = 0.0
    recall: float = 0.0
    ap50: float = 0.0
    ap50_95: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    n_val_instances: int = 0


@dataclass
class ModelVersionMetrics:
    per_class: Dict[int, ClassMetrics]


@dataclass
class ClassRegistryEntry:
    tier: str
    state: str
    ever_active: bool


class ClassRegistryView:
    """Adapter over a plain {class_id: ClassRegistryEntry} dict, so this
    module never imports DatasetService/ClassInfo directly - the caller
    (model_registry_service.py) builds this from `ds.get_classes()`.
    Unknown class ids (shouldn't happen in practice - every class_id in a
    golden-eval result should exist in the current class map) default to
    the same conservative values the DB columns themselves default to."""

    def __init__(self, entries: Dict[int, ClassRegistryEntry]) -> None:
        self._entries = entries

    def tier(self, class_id: int) -> str:
        entry = self._entries.get(class_id)
        return entry.tier if entry else "structural"

    def state(self, class_id: int) -> str:
        entry = self._entries.get(class_id)
        return entry.state if entry else "discovered"

    def was_ever_active(self, class_id: int) -> bool:
        entry = self._entries.get(class_id)
        return entry.ever_active if entry else False

    @classmethod
    def from_class_infos(cls, class_infos) -> "ClassRegistryView":
        return cls(
            {
                c.class_id: ClassRegistryEntry(tier=c.tier, state=c.state, ever_active=c.ever_active)
                for c in class_infos
            }
        )


@dataclass
class ComparisonResult:
    common_classes: Dict[int, dict]  # class_id -> {candidate, production, delta_ap50, regressed, tier}
    new_classes: Dict[int, dict]  # class_id -> {candidate, meets_floor, floor_used}
    removed_classes: List[int]  # present in production, absent (explicitly deprecated) from candidate
    reintroduced_classes: List[int]  # present in candidate, was_ever_active - routed through new_classes, not common
    overall: dict  # {candidate_map50, production_map50} - reported only, never a should_promote() criterion
    candidate_class_set: Set[int]
    production_class_set: Set[int]


@dataclass
class PromotionThresholds:
    regression_tolerance_ap50: Dict[str, float]
    new_class_floor_ap50: Dict[str, float]
    min_val_instances: int
    max_latency_p95_ms: float
    max_model_size_mb: float
    max_false_positive_rate: Dict[str, float]

    @classmethod
    def from_settings(cls, settings: Settings) -> "PromotionThresholds":
        return cls(
            regression_tolerance_ap50={
                "safety": settings.regression_tolerance_ap50_safety,
                "structural": settings.regression_tolerance_ap50_structural,
                "cosmetic": settings.regression_tolerance_ap50_cosmetic,
            },
            new_class_floor_ap50={
                "safety": settings.new_class_floor_ap50_safety,
                "structural": settings.new_class_floor_ap50_structural,
                "cosmetic": settings.new_class_floor_ap50_cosmetic,
            },
            min_val_instances=settings.eligibility_min_val_instances,
            max_latency_p95_ms=settings.max_latency_p95_ms,
            max_model_size_mb=settings.max_model_size_mb,
            max_false_positive_rate={
                "safety": settings.max_false_positive_rate_safety,
                "structural": settings.max_false_positive_rate_structural,
                "cosmetic": settings.max_false_positive_rate_cosmetic,
            },
        )


@dataclass
class OperationalMetrics:
    latency_p95_ms: float = 0.0
    model_size_mb: float = 0.0
    # Per class_id - only classes with a real measurement (Module 7) should
    # appear here; a class simply absent from this dict is treated as "not
    # measured yet, skip this check for it," never as an implicit 0.0 pass.
    false_positive_rate: Dict[int, float] = field(default_factory=dict)


@dataclass
class PromotionDecision:
    decision: str  # "PROMOTE" | "REJECT"
    reasons: List[str] = field(default_factory=list)
    hard_fail: bool = False  # a safety-tier regression or unexplained class-set shrink - never overridable
    comparison: Optional[ComparisonResult] = None
    operational: Optional[OperationalMetrics] = None


def _weighted_mean_ap50(per_class: Dict[int, ClassMetrics]) -> Optional[float]:
    total_weight = sum(m.n_val_instances for m in per_class.values())
    if total_weight == 0:
        return None
    return sum(m.ap50 * m.n_val_instances for m in per_class.values()) / total_weight


def compare(
    candidate: ModelVersionMetrics,
    production: Optional[ModelVersionMetrics],
    registry: ClassRegistryView,
    thresholds: PromotionThresholds,
) -> ComparisonResult:
    candidate_classes = set(candidate.per_class.keys())
    production_classes = set(production.per_class.keys()) if production else set()

    common = candidate_classes & production_classes
    new = candidate_classes - production_classes
    removed = production_classes - candidate_classes

    silently_dropped = sorted(c for c in removed if registry.state(c) != "deprecated")
    if silently_dropped:
        raise UnexplainedClassRemoval(silently_dropped)

    reintroduced = sorted(c for c in new if registry.was_ever_active(c))

    common_result: Dict[int, dict] = {}
    for class_id in common:
        cand = candidate.per_class[class_id]
        prod = production.per_class[class_id]
        delta = cand.ap50 - prod.ap50
        tier = registry.tier(class_id)
        tolerance = thresholds.regression_tolerance_ap50[tier]
        common_result[class_id] = {
            "candidate": cand,
            "production": prod,
            "delta_ap50": delta,
            "regressed": delta < -tolerance,
            "tier": tier,
        }

    new_result: Dict[int, dict] = {}
    for class_id in new:
        cand = candidate.per_class[class_id]
        tier = registry.tier(class_id)
        floor = thresholds.new_class_floor_ap50[tier]
        meets_floor = cand.ap50 >= floor and cand.n_val_instances >= thresholds.min_val_instances
        new_result[class_id] = {"candidate": cand, "meets_floor": meets_floor, "floor_used": floor}

    overall = {
        "candidate_map50": _weighted_mean_ap50(candidate.per_class),
        "production_map50": _weighted_mean_ap50(production.per_class) if production else None,
    }

    return ComparisonResult(
        common_classes=common_result,
        new_classes=new_result,
        removed_classes=sorted(removed),
        reintroduced_classes=reintroduced,
        overall=overall,
        candidate_class_set=candidate_classes,
        production_class_set=production_classes,
    )


def should_promote(
    comparison: ComparisonResult,
    operational: OperationalMetrics,
    registry: ClassRegistryView,
    thresholds: PromotionThresholds,
) -> PromotionDecision:
    reasons: List[str] = []
    hard_fail = False

    # --- Layer 1: common-class regression -----------------------------
    regressed = sorted(c for c, r in comparison.common_classes.items() if r["regressed"])
    if regressed:
        worst = min(comparison.common_classes[c]["delta_ap50"] for c in regressed)
        reasons.append(f"regression on classes {regressed} (worst {worst:+.1f}pt AP50)")

    # A safety-tier regression is an automatic, UNCONDITIONAL reject - see
    # Module 6, which enforces this exact "hard_fail" flag as unoverridable
    # at the approval step, not just advisory here.
    safety_regressed = sorted(c for c in regressed if comparison.common_classes[c]["tier"] == "safety")
    if safety_regressed:
        reasons.append(f"HARD FAIL: safety-tier regression on {safety_regressed}")
        hard_fail = True

    # --- Layer 2: new classes must clear an absolute floor -------------
    failing_new = sorted(c for c, r in comparison.new_classes.items() if not r["meets_floor"])
    if failing_new:
        reasons.append(f"new class(es) {failing_new} below absolute quality floor")

    # --- Layer 3: overall quality -- REPORTED, NEVER a standalone gate -
    # Deliberately no check here at all: a lower overall mAP driven purely
    # by a hard new class must never block promotion if Layers 1/2/4 pass,
    # and a higher overall mAP must never rescue a promotion those layers
    # reject. comparison.overall is attached to the decision as context,
    # full stop.

    # --- Layer 4: operational constraints -------------------------------
    if operational.latency_p95_ms > thresholds.max_latency_p95_ms:
        reasons.append(
            f"latency regression: {operational.latency_p95_ms:.1f}ms > "
            f"{thresholds.max_latency_p95_ms:.1f}ms budget"
        )
    if operational.model_size_mb > thresholds.max_model_size_mb:
        reasons.append(
            f"model size {operational.model_size_mb:.1f}MB exceeds {thresholds.max_model_size_mb:.1f}MB budget"
        )
    for class_id, fp_rate in sorted(operational.false_positive_rate.items()):
        tier = registry.tier(class_id)
        budget = thresholds.max_false_positive_rate[tier]
        if fp_rate > budget:
            reasons.append(
                f"class {class_id} false-positive rate {fp_rate:.3f} exceeds {tier} tier budget {budget:.3f}"
            )

    # --- Class-count sanity: defense-in-depth re-check ------------------
    # compare() already raises UnexplainedClassRemoval before a
    # ComparisonResult can even exist with a silent shrink in it, so this
    # should be structurally unreachable - kept anyway, since "should be
    # unreachable" is exactly the class of assumption worth re-checking at
    # the one place a wrong assumption would matter most.
    unexplained_shrink = sorted(
        c
        for c in comparison.production_class_set - comparison.candidate_class_set
        if registry.state(c) != "deprecated"
    )
    if unexplained_shrink:
        reasons.append(f"HARD FAIL: class set shrank without recorded deprecation: {unexplained_shrink}")
        hard_fail = True

    decision = "REJECT" if reasons else "PROMOTE"
    return PromotionDecision(
        decision=decision, reasons=reasons, hard_fail=hard_fail, comparison=comparison, operational=operational
    )
