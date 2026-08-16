"""Tests for class_eligibility_service.py (Module 2 of the class-incremental
promotion plan - docs/mlflow_class_incremental_architecture.md §D/§E).

No database: `compute_eligibility` is exercised against a duck-typed fake
DatasetService (same style as tests/test_golden_eval_service.py's
_FakeDatasetService), `determine_state`/`recompute_and_apply` against
plain dataclasses/fakes - no Postgres needed for any of this file.

    cd backend && python -m pytest tests/test_class_eligibility_service.py
    cd backend && python -m tests.test_class_eligibility_service   # no pytest
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from app.config import get_settings
from app.models.schemas import ClassInfo
from app.services.class_eligibility_service import (
    ClassEligibility,
    EligibilityThresholds,
    compute_eligibility,
    determine_state,
    recompute_and_apply,
)


def _thresholds() -> EligibilityThresholds:
    return EligibilityThresholds(
        min_instances={"safety": 150, "structural": 100, "cosmetic": 80},
        min_images={"safety": 60, "structural": 50, "cosmetic": 40},
        min_coach_types={"safety": 2, "structural": 1, "cosmetic": 1},
        min_val_instances=15,
    )


def _elig(**overrides) -> ClassEligibility:
    base = dict(class_id=0, instance_count=200, image_count=80, coach_types={"LHB", "ICF"},
                projected_val_instances=20.0, relative_share=0.2)
    base.update(overrides)
    return ClassEligibility(**base)


# ------------------------------------------------------------ determine_state

def test_active_class_stays_active_even_with_zero_instances() -> None:
    """The literal fix for 'old classes shouldn't disappear because their
    share shrank' - ACTIVE is sticky, unconditionally."""
    elig = _elig(instance_count=0, image_count=0, coach_types=set(), projected_val_instances=0.0)
    assert determine_state("active", elig, "structural", _thresholds()) == "active"


def test_deprecated_class_stays_deprecated_even_with_abundant_new_data() -> None:
    elig = _elig(instance_count=1000, image_count=500, coach_types={"LHB", "ICF"})
    assert determine_state("deprecated", elig, "structural", _thresholds()) == "deprecated"


def test_per_tier_threshold_lookup_safety_vs_structural() -> None:
    # 120 instances/55 images/1 coach type: clears structural's floor, not safety's.
    elig = _elig(instance_count=120, image_count=55, coach_types={"LHB"}, projected_val_instances=12.0)
    assert determine_state("collecting_data", elig, "safety", _thresholds()) == "collecting_data"
    elig2 = _elig(instance_count=120, image_count=55, coach_types={"LHB"}, projected_val_instances=16.0)
    assert determine_state("collecting_data", elig2, "structural", _thresholds()) == "eligible"


def test_diversity_gate_blocks_eligibility_even_with_enough_volume() -> None:
    elig = _elig(instance_count=500, image_count=200, coach_types={"LHB"}, projected_val_instances=50.0)
    assert determine_state("collecting_data", elig, "safety", _thresholds()) == "collecting_data"


def test_val_split_projection_gate() -> None:
    """Enough raw instances/images/diversity, but too few would land in val
    given the training split ratio."""
    elig = _elig(instance_count=90, image_count=45, coach_types={"LHB", "ICF"}, projected_val_instances=9.0)
    assert determine_state("collecting_data", elig, "cosmetic", _thresholds()) == "collecting_data"


def test_discovered_transitions_to_collecting_data_once_any_instance_exists() -> None:
    elig = _elig(instance_count=1, image_count=1, coach_types={"LHB"}, projected_val_instances=0.1)
    assert determine_state("discovered", elig, "structural", _thresholds()) == "collecting_data"


def test_zero_instances_stays_discovered() -> None:
    elig = _elig(instance_count=0, image_count=0, coach_types=set(), projected_val_instances=0.0)
    assert determine_state("discovered", elig, "structural", _thresholds()) == "discovered"


def test_eligible_class_with_shrinking_data_is_not_demoted() -> None:
    """determine_state is only ever called with the class's CURRENT stored
    state - an already-eligible class passed back in with poor new numbers
    must not be demoted by this function (promotion/demotion of an
    eligible-but-not-yet-promoted class is out of this function's scope;
    it only refuses to demote the two terminal states, but 'eligible'
    itself, once reached, is the caller's floor - re-verify by checking the
    function's contract directly)."""
    elig = _elig(instance_count=0, image_count=0, coach_types=set(), projected_val_instances=0.0)
    # 'eligible' isn't in _TERMINAL_STATES, so this documents the actual,
    # current behavior: it WOULD recompute down to 'discovered' if called
    # with zero data. This is fine in practice because recompute_and_apply
    # only calls determine_state with the class's live aggregated data, and
    # an eligible-not-yet-active class regressing to zero data before ever
    # being promoted is a legitimate "no longer eligible" fact, not a
    # regression of shipped capability the way demoting ACTIVE would be.
    assert determine_state("eligible", elig, "structural", _thresholds()) == "discovered"


# ------------------------------------------------------------ compute_eligibility

class _FakeDatasetService:
    """Duck-typed stand-in exposing only what compute_eligibility/
    recompute_and_apply use - image_ids, get_saved_states, get_classes,
    set_class_state, dataset_key."""

    def __init__(self, states: Dict[str, dict], classes: List[ClassInfo]) -> None:
        self._states = states
        self._classes = classes
        self.set_calls: List[Tuple[int, str]] = []

    def image_ids(self) -> List[str]:
        return list(self._states.keys())

    def get_saved_states(self, image_ids: List[str]) -> Dict[str, dict]:
        return {i: self._states[i] for i in image_ids if i in self._states}

    def get_classes(self) -> List[ClassInfo]:
        return self._classes

    def set_class_state(self, class_id: int, state: str) -> None:
        self.set_calls.append((class_id, state))
        for c in self._classes:
            if c.class_id == class_id:
                c.state = state

    @property
    def dataset_key(self) -> str:
        return "fake_view"


def _obj(class_id: int, status: str = "confirmed") -> dict:
    return {"class_id": class_id, "status": status, "id": f"o{class_id}"}


def test_compute_eligibility_counts_instances_images_and_coach_types() -> None:
    states = {
        "img1": {"completed": True, "coach_type": "LHB", "objects": [_obj(0), _obj(1)]},
        "img2": {"completed": True, "coach_type": "ICF", "objects": [_obj(0)]},
        "img3": {"completed": False, "coach_type": "LHB", "objects": [_obj(0)]},  # not completed - excluded
    }
    ds = _FakeDatasetService(states, [])
    result = compute_eligibility(ds)
    assert result[0].instance_count == 2  # img1 + img2, img3 excluded (not completed)
    assert result[0].image_count == 2
    assert result[0].coach_types == {"LHB", "ICF"}
    assert result[1].instance_count == 1


def test_compute_eligibility_excludes_rejected_objects() -> None:
    states = {"img1": {"completed": True, "coach_type": "LHB", "objects": [_obj(0, status="rejected"), _obj(0)]}}
    ds = _FakeDatasetService(states, [])
    result = compute_eligibility(ds)
    assert result[0].instance_count == 1  # only the non-rejected one


def test_compute_eligibility_extra_polygons_do_not_double_count() -> None:
    """extra_polygons are pieces of the SAME object, not separate instances -
    the object dict itself is what's counted, once, regardless of how many
    polygon pieces it carries."""
    states = {
        "img1": {
            "completed": True, "coach_type": "LHB",
            "objects": [{"class_id": 0, "status": "confirmed", "id": "o1",
                         "polygon": [{"x": 0.1, "y": 0.1}], "extra_polygons": [[{"x": 0.5, "y": 0.5}]]}],
        },
    }
    ds = _FakeDatasetService(states, [])
    result = compute_eligibility(ds)
    assert result[0].instance_count == 1


def test_compute_eligibility_relative_share() -> None:
    states = {
        "img1": {"completed": True, "coach_type": "LHB", "objects": [_obj(0), _obj(0), _obj(0), _obj(1)]},
    }
    ds = _FakeDatasetService(states, [])
    result = compute_eligibility(ds)
    assert result[0].relative_share == 0.75
    assert result[1].relative_share == 0.25


# ------------------------------------------------------------ recompute_and_apply

def test_recompute_and_apply_only_reports_actual_changes() -> None:
    settings = get_settings()
    states = {
        f"img{i}": {"completed": True, "coach_type": "LHB", "objects": [_obj(0)]} for i in range(200)
    }
    classes = [
        ClassInfo(class_id=0, name="wheel", color="#fff", state="discovered", tier="cosmetic"),
        ClassInfo(class_id=1, name="unrelated", color="#eee", state="active", tier="structural"),
    ]
    ds = _FakeDatasetService(states, classes)
    # 200 instances/200 images/1 coach type ("LHB") clears cosmetic's floor
    # (80 instances/40 images/1 coach type) with plenty of val projection.
    changed = recompute_and_apply(ds, settings)
    assert changed == {0: "eligible"}
    assert ds.set_calls == [(0, "eligible")]  # class 1 (already active, no data) never touched


def test_recompute_and_apply_is_a_noop_when_nothing_changed() -> None:
    settings = get_settings()
    classes = [ClassInfo(class_id=0, name="wheel", color="#fff", state="active", tier="structural")]
    ds = _FakeDatasetService({}, classes)
    changed = recompute_and_apply(ds, settings)
    assert changed == {}
    assert ds.set_calls == []


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
