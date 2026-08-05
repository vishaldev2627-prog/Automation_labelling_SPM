"""Tests for the golden eval set's permission gate (M4).

D-Q4: "we curate and version; they run the eval. Nothing that trains ever
sees it." The build plan's own risk note for this milestone: "the *only*
mitigation that counts is that no propagation, triage, or export path can
write here. Assert it in tests."

This file covers the one piece of that gate that's a pure function of already
-fetched data: `golden_service._is_golden_curator`, the actual permission
predicate `require_golden_curator` is built on. The DB-backed pieces
(golden_repo's cumulative-across-versions query, export_service's exclusion,
propagation_service's refusal) were exercised live against a running dev
instance rather than mocked here, for the same reason the rest of this test
suite stays DB-free: SQLAlchemy Session behavior is not what these tests are
meant to pin down, the permission and exclusion *rules* are.

    cd backend && python -m pytest tests/test_golden_set.py
    cd backend && python -m tests.test_golden_set   # no pytest
"""
from __future__ import annotations

from app.models.db_models import Annotator
from app.services.golden_service import _is_golden_curator


def _annotator(role: str) -> Annotator:
    a = Annotator(name="x", role=role)
    return a


def test_no_annotator_is_not_a_curator() -> None:
    """The unidentified-session case - annotator_id is None before any DB
    lookup even happens, but the predicate itself must also reject a bare
    None safely rather than raising."""
    assert _is_golden_curator(None) is False


def test_plain_annotator_role_is_not_a_curator() -> None:
    assert _is_golden_curator(_annotator("annotator")) is False


def test_system_role_is_not_a_curator() -> None:
    """The auto-accept system identity (see annotator_service.SYSTEM_ANNOTATOR_NAME)
    must not incidentally satisfy this - auto-accept writes reviews, never
    golden-set curation, and the two must stay unrelated permissions."""
    assert _is_golden_curator(_annotator("system")) is False


def test_golden_curator_role_passes() -> None:
    assert _is_golden_curator(_annotator("golden_curator")) is True


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
