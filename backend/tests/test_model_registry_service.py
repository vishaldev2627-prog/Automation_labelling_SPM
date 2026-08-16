"""Tests for model_registry_service.py (M7 + Module 5 of the
class-incremental promotion plan - docs/mlflow_class_incremental_
architecture.md §G).

A hand-rolled fake MlflowClient stands in for the real one - the small
subset of methods `_decide`/`register_and_recommend` actually call
(`get_latest_versions`, `get_run`, `create_registered_model`,
`create_model_version`, `set_model_version_tag`,
`set_registered_model_alias`) need doubles; everything else in this module
talks to a real (or absent) MLflow registry, exercised live rather than
mocked here (same reasoning as test_golden_set.py staying DB-free).

    cd backend && python -m pytest tests/test_model_registry_service.py
    cd backend && python -m tests.test_model_registry_service   # no pytest
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from app.models.schemas import ClassInfo
from app.services.model_registry_service import _decide, registered_model_name
from app.services.promotion_gate import PromotionThresholds

THRESHOLDS = PromotionThresholds(
    regression_tolerance_ap50={"safety": 0.005, "structural": 0.02, "cosmetic": 0.03},
    new_class_floor_ap50={"safety": 0.85, "structural": 0.70, "cosmetic": 0.55},
    min_val_instances=15,
    max_latency_p95_ms=500.0,
    max_model_size_mb=200.0,
    max_false_positive_rate={"safety": 0.02, "structural": 0.05, "cosmetic": 0.10},
)


@dataclass
class _FakeModelVersion:
    version: str
    run_id: str


@dataclass
class _FakeRunData:
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class _FakeRun:
    data: _FakeRunData


class _FakeClient:
    def __init__(self, production_version: Optional["_FakeModelVersion"], run_metrics: Dict[str, float]) -> None:
        self._production_version = production_version
        self._run_metrics = run_metrics

    def get_latest_versions(self, name, stages):
        assert stages == ["Production"]
        return [self._production_version] if self._production_version else []

    def get_run(self, run_id):
        return _FakeRun(data=_FakeRunData(metrics=self._run_metrics))


def _class(class_id: int, tier: str = "structural", state: str = "active", ever_active: bool = False) -> ClassInfo:
    return ClassInfo(class_id=class_id, name=f"c{class_id}", color="#fff", tier=tier, state=state, ever_active=ever_active)


def test_no_production_version_treats_every_class_as_new_and_promotes_if_it_clears_the_floor() -> None:
    client = _FakeClient(production_version=None, run_metrics={})
    candidate = {0: {"mAP50": 0.80, "n_val_instances": 20}}
    decision, compared = _decide(client, "AnnotDetector-x", candidate, [_class(0)], 20.0, 100.0, THRESHOLDS)
    assert decision.decision == "PROMOTE"
    assert compared is None


def test_matching_or_beating_every_class_promotes() -> None:
    client = _FakeClient(
        production_version=_FakeModelVersion(version="3", run_id="r1"),
        run_metrics={"golden/class0_mAP50": 0.40, "golden/class1_mAP50": 0.60},
    )
    candidate = {0: {"mAP50": 0.40, "n_val_instances": 20}, 1: {"mAP50": 0.70, "n_val_instances": 20}}
    decision, compared = _decide(
        client, "AnnotDetector-x", candidate, [_class(0), _class(1)], 20.0, 100.0, THRESHOLDS
    )
    assert decision.decision == "PROMOTE"
    assert compared == "3"


def test_regressing_beyond_tolerance_rejects() -> None:
    """The build plan's own [MEDIUM] risk, as an explicit test: an
    aggregate-looking improvement must not hide one class quietly
    regressing beyond its tier's tolerance."""
    client = _FakeClient(
        production_version=_FakeModelVersion(version="3", run_id="r1"),
        run_metrics={"golden/class0_mAP50": 0.40, "golden/class1_mAP50": 0.60},
    )
    candidate = {0: {"mAP50": 0.90, "n_val_instances": 20}, 1: {"mAP50": 0.50, "n_val_instances": 20}}
    decision, _compared = _decide(
        client, "AnnotDetector-x", candidate, [_class(0), _class(1)], 20.0, 100.0, THRESHOLDS
    )
    assert decision.decision == "REJECT"
    assert any("1" in r for r in decision.reasons)


def test_new_class_with_no_stored_baseline_is_checked_against_absolute_floor() -> None:
    """The actual bug this module used to have (verified live before this
    module existed): a class the candidate covers that Production's own
    run never recorded used to `continue` past entirely, contributing
    nothing to the verdict either way. It must now be checked against an
    absolute floor."""
    client = _FakeClient(
        production_version=_FakeModelVersion(version="3", run_id="r1"),
        run_metrics={"golden/class0_mAP50": 0.40, "golden/class0_n_val_instances": 20},
    )
    weak_new_class = {0: {"mAP50": 0.40, "n_val_instances": 20}, 5: {"mAP50": 0.01, "n_val_instances": 20}}
    decision, _compared = _decide(
        client, "AnnotDetector-x", weak_new_class, [_class(0), _class(5)], 20.0, 100.0, THRESHOLDS
    )
    assert decision.decision == "REJECT"
    assert any("5" in r for r in decision.reasons)

    strong_new_class = {0: {"mAP50": 0.40, "n_val_instances": 20}, 5: {"mAP50": 0.90, "n_val_instances": 20}}
    decision2, _compared2 = _decide(
        client, "AnnotDetector-x", strong_new_class, [_class(0), _class(5)], 20.0, 100.0, THRESHOLDS
    )
    assert decision2.decision == "PROMOTE"


def test_unexplained_class_removal_becomes_reject_not_an_exception() -> None:
    client = _FakeClient(
        production_version=_FakeModelVersion(version="3", run_id="r1"),
        run_metrics={"golden/class0_mAP50": 0.40, "golden/class1_mAP50": 0.60},
    )
    candidate = {0: {"mAP50": 0.40, "n_val_instances": 20}}  # class 1 silently absent
    decision, _compared = _decide(
        client, "AnnotDetector-x", candidate, [_class(0), _class(1, state="active")], 20.0, 100.0, THRESHOLDS
    )
    assert decision.decision == "REJECT"
    assert decision.hard_fail is True
    assert any("1" in r for r in decision.reasons)


def test_deprecated_class_removal_does_not_reject() -> None:
    client = _FakeClient(
        production_version=_FakeModelVersion(version="3", run_id="r1"),
        run_metrics={"golden/class0_mAP50": 0.40, "golden/class1_mAP50": 0.60},
    )
    candidate = {0: {"mAP50": 0.40, "n_val_instances": 20}}
    decision, _compared = _decide(
        client, "AnnotDetector-x", candidate, [_class(0), _class(1, state="deprecated")], 20.0, 100.0, THRESHOLDS
    )
    assert decision.decision == "PROMOTE"


def test_operational_latency_breach_rejects() -> None:
    client = _FakeClient(production_version=None, run_metrics={})
    candidate = {0: {"mAP50": 0.90, "n_val_instances": 20}}
    decision, _compared = _decide(
        client, "AnnotDetector-x", candidate, [_class(0)], 20.0, latency_p95_ms=999.0, thresholds=THRESHOLDS
    )
    assert decision.decision == "REJECT"
    assert any("latency" in r for r in decision.reasons)


def test_registered_model_name_is_scoped_per_view() -> None:
    assert registered_model_name("side_view") == "AnnotDetector-side_view"
    assert registered_model_name("side_view") != registered_model_name("buffer")


# ------------------------------------------------- register_and_recommend (full path)

class _FullFakeClient(_FakeClient):
    """Adds the write-side methods register_and_recommend calls, recording
    every one so tests can assert on the exact tags/aliases written."""

    def __init__(self, production_version=None, run_metrics=None):
        super().__init__(production_version, run_metrics or {})
        self.tags: Dict[str, str] = {}
        self.aliases: Dict[str, str] = {}
        self.created_model_version = None

    def create_registered_model(self, name):
        pass

    def create_model_version(self, name, source, run_id):
        self.created_model_version = _FakeModelVersion(version="7", run_id=run_id)
        return self.created_model_version

    def set_model_version_tag(self, name, version, key, value):
        self.tags[key] = value

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version


def test_register_and_recommend_writes_the_full_tag_table() -> None:
    import app.services.model_registry_service as mrs

    fake_client = _FullFakeClient(production_version=None, run_metrics={})
    fake_mlflow_client_cls = lambda: fake_client
    orig_mlflow_client = None

    import mlflow.tracking as mlflow_tracking_module

    orig_mlflow_client = mlflow_tracking_module.MlflowClient
    mlflow_tracking_module.MlflowClient = fake_mlflow_client_cls
    try:
        decision = mrs.register_and_recommend(
            "side_view", "run1", {0: {"mAP50": 0.90, "n_val_instances": 20}}, [_class(0)],
            model_size_mb=20.0, latency_p95_ms=100.0, thresholds=THRESHOLDS,
        )
    finally:
        mlflow_tracking_module.MlflowClient = orig_mlflow_client

    assert decision is not None
    assert decision.decision == "PROMOTE"
    # Legacy tags (backward compat for model_promotion_service, pre-Module 6)
    assert fake_client.tags["promotion_recommendation"] == "eligible_no_baseline"
    # §G tag table
    assert fake_client.tags["decision"] == "PROMOTE"
    assert fake_client.tags["hard_fail"] == "false"
    assert fake_client.tags["class_set_diff"] == "+0"
    assert fake_client.tags["new_classes"] == "0"
    assert fake_client.tags["common_class_regression"] == "none"
    assert fake_client.tags["operational_result"] == "pass"
    # Additive alias, alongside (not instead of) stage-based logic
    assert fake_client.aliases["candidate"] == "7"


def test_model_version_tags_include_snapshot_and_class_map_ids() -> None:
    """Module 9: lineage tag completeness - dataset_snapshot_id/
    class_map_version land on the model VERSION, not just the training
    run."""
    import mlflow.tracking as mlflow_tracking_module

    import app.services.model_registry_service as mrs

    fake_client = _FullFakeClient(production_version=None, run_metrics={})
    orig_mlflow_client = mlflow_tracking_module.MlflowClient
    mlflow_tracking_module.MlflowClient = lambda: fake_client
    try:
        mrs.register_and_recommend(
            "side_view", "run1", {0: {"mAP50": 0.90, "n_val_instances": 20}}, [_class(0)],
            model_size_mb=20.0, latency_p95_ms=100.0, thresholds=THRESHOLDS,
            dataset_snapshot_id="abc123snapshot", class_map_version=7,
        )
    finally:
        mlflow_tracking_module.MlflowClient = orig_mlflow_client

    assert fake_client.tags["dataset_snapshot_id"] == "abc123snapshot"
    assert fake_client.tags["class_map_version"] == "7"


def test_baseline_and_parent_version_can_differ() -> None:
    """The doc's own described edge case: a candidate trained against V1's
    weights (parent_model_version=1, a training-time RUN param -
    detector_service's own concern) can still end up compared against a
    NEWER production version (baseline_version=2, this module's tag) if
    something else got promoted to Production before this candidate's eval
    finished. The two are genuinely different things and must not be
    conflated - this module only ever produces/reads baseline_version;
    parent_model_version is asserted here only to document that distinction,
    not because this module sets it."""
    client = _FakeClient(
        production_version=_FakeModelVersion(version="2", run_id="r2"),  # V2 is Production now, not V1
        run_metrics={"golden/class0_mAP50": 0.85, "golden/class0_n_val_instances": 20},
    )
    # This candidate's run (not modeled here - that's detector_service's
    # parent_model_version param, a separate concern) was trained starting
    # from V1's weights, but by the time its eval finished, V2 was already
    # Production - so it's compared against V2, not V1.
    candidate = {0: {"mAP50": 0.90, "n_val_instances": 20}}
    decision, compared_against_version = _decide(
        client, "AnnotDetector-x", candidate, [_class(0)], 20.0, 100.0, THRESHOLDS
    )
    assert compared_against_version == "2"  # baseline_version tag would be "2", not "1"


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
