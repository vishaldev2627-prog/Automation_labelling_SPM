"""Tests for the M7 promotion-recommendation logic (recommendation-only).

Per the build plan's own risk note for this milestone - "this is where a
wrong design silently promotes a bad model" - the one thing genuinely worth
pinning down precisely is the comparison rule itself:
`_compare_against_production`. Everything else in model_registry_service.py
talks to a real (or absent) MLflow registry, exercised live rather than
mocked here (same reasoning as test_golden_set.py staying DB-free).

A hand-rolled fake MlflowClient stands in for the real one - only the two
methods `_compare_against_production` actually calls
(`get_latest_versions`, `get_run`) need doubles.

    cd backend && python -m pytest tests/test_model_registry_service.py
    cd backend && python -m tests.test_model_registry_service   # no pytest
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.model_registry_service import _compare_against_production, registered_model_name


@dataclass
class _FakeModelVersion:
    version: str
    run_id: str


@dataclass
class _FakeRunData:
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class _FakeRun:
    data: _FakeRunData


class _FakeClient:
    def __init__(self, production_version: _FakeModelVersion | None, run_metrics: dict[str, float]) -> None:
        self._production_version = production_version
        self._run_metrics = run_metrics

    def get_latest_versions(self, name, stages):
        assert stages == ["Production"]
        return [self._production_version] if self._production_version else []

    def get_run(self, run_id):
        return _FakeRun(data=_FakeRunData(metrics=self._run_metrics))


def test_no_production_version_is_eligible_no_baseline() -> None:
    client = _FakeClient(production_version=None, run_metrics={})
    result = _compare_against_production(client, "AnnotDetector-x", {0: {"mAP50": 0.5}})
    assert result.verdict == "eligible_no_baseline"


def test_matching_or_beating_every_class_is_eligible() -> None:
    client = _FakeClient(
        production_version=_FakeModelVersion(version="3", run_id="r1"),
        run_metrics={"golden/class0_mAP50": 0.4, "golden/class1_mAP50": 0.6},
    )
    candidate = {0: {"mAP50": 0.4}, 1: {"mAP50": 0.7}}  # equal on 0, better on 1
    result = _compare_against_production(client, "AnnotDetector-x", candidate)
    assert result.verdict == "eligible"
    assert result.regressed_classes == []
    assert result.compared_against_version == "3"


def test_regressing_even_one_class_is_flagged() -> None:
    """The build plan's own [MEDIUM] risk, as an explicit test: an
    aggregate-looking improvement must not hide one class quietly
    regressing."""
    client = _FakeClient(
        production_version=_FakeModelVersion(version="3", run_id="r1"),
        run_metrics={"golden/class0_mAP50": 0.4, "golden/class1_mAP50": 0.6},
    )
    candidate = {0: {"mAP50": 0.9}, 1: {"mAP50": 0.5}}  # much better on 0, worse on 1
    result = _compare_against_production(client, "AnnotDetector-x", candidate)
    assert result.verdict == "regressed"
    assert result.regressed_classes == [1]


def test_a_class_with_no_stored_baseline_cannot_regress() -> None:
    """A class the candidate covers that Production's own run never
    recorded (e.g. added since Production was trained) has nothing to
    regress against - it must not count against the candidate."""
    client = _FakeClient(
        production_version=_FakeModelVersion(version="3", run_id="r1"),
        run_metrics={"golden/class0_mAP50": 0.4},
    )
    candidate = {0: {"mAP50": 0.4}, 5: {"mAP50": 0.01}}  # class 5 has no baseline
    result = _compare_against_production(client, "AnnotDetector-x", candidate)
    assert result.verdict == "eligible"


def test_registered_model_name_is_scoped_per_view() -> None:
    assert registered_model_name("side_view") == "AnnotDetector-side_view"
    assert registered_model_name("side_view") != registered_model_name("buffer")


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
