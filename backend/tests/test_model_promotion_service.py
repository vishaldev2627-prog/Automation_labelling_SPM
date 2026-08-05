"""Tests for the M7.5 permission gate and pure logic.

The DB/MLflow-touching pieces (check_for_new_production_version, approve,
reject, rollback) are exercised live against the running dev stack rather
than mocked here, same reasoning as test_golden_set.py and
test_model_registry_service.py: what's worth pinning down precisely with a
unit test is the permission predicate itself.

    cd backend && python -m pytest tests/test_model_promotion_service.py
    cd backend && python -m tests.test_model_promotion_service   # no pytest
"""
from __future__ import annotations

from app.models.db_models import Annotator
from app.services.model_promotion_service import _is_model_reviewer


def _annotator(role: str) -> Annotator:
    return Annotator(name="x", role=role)


def test_no_annotator_is_not_a_reviewer() -> None:
    assert _is_model_reviewer(None) is False


def test_plain_annotator_is_not_a_reviewer() -> None:
    assert _is_model_reviewer(_annotator("annotator")) is False


def test_golden_curator_is_not_automatically_a_model_reviewer() -> None:
    """The two roles are deliberately distinct - curating what's eligible
    and deciding what live annotators actually get suggestions from are
    different responsibilities, per the design discussion this milestone
    was built from."""
    assert _is_model_reviewer(_annotator("golden_curator")) is False


def test_model_reviewer_role_passes() -> None:
    assert _is_model_reviewer(_annotator("model_reviewer")) is True


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
