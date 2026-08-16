"""Tests for the class lifecycle schema (Module 1 of the class-incremental
promotion plan - docs/mlflow_class_incremental_architecture.md §D).

No database: `DatasetService`'s new state/tier methods are exercised
against the in-memory caches directly by monkeypatching the module-level
`state_repo`/`SessionLocal` names `dataset_service.py` calls through -
plain attribute swap + restore, no pytest fixture needed, so this runs
identically under pytest or the hand-rolled standalone runner below. The
migration's backfill logic is tested as the same pure predicate the
migration file's SQL encodes (safety_critical -> tier), locking down the
mapping independently of Alembic actually running.

    cd backend && python -m pytest tests/test_class_lifecycle_schema.py
    cd backend && python -m tests.test_class_lifecycle_schema   # no pytest
"""
from __future__ import annotations
from typing import Dict

from contextlib import contextmanager

from app.models.schemas import ClassInfo
from app.services import dataset_service as ds_module
from app.services.dataset_service import DatasetService


def _backfill_tier(safety_critical: bool) -> str:
    """Mirrors the migration's own UPDATE predicate exactly (see
    migrations/versions/e2f4a7b91c30_add_class_lifecycle_state_tier.py):
    safety_critical=true -> tier='safety', else the column's own
    server_default of 'structural' (never 'cosmetic' by default)."""
    return "safety" if safety_critical else "structural"


def test_backfill_maps_safety_critical_true_to_safety_tier() -> None:
    assert _backfill_tier(True) == "safety"


def test_backfill_maps_safety_critical_false_to_structural_never_cosmetic() -> None:
    assert _backfill_tier(False) == "structural"


def test_class_info_defaults_match_column_server_defaults() -> None:
    """A ClassInfo built with no explicit state/tier (e.g. an old caller
    that doesn't know about this field yet) must default identically to
    what the DB column itself defaults to - so a pre-migration code path
    and a post-migration one agree."""
    info = ClassInfo(class_id=0, name="coupler", color="#fff")
    assert info.state == "active"
    assert info.tier == "structural"


# --------------------------------------------------- DatasetService methods

def _bare_ds(states: Dict[str, str]) -> DatasetService:
    """A DatasetService instance with only the attributes these methods
    touch populated - bypasses __init__ (which requires a real Settings +
    a loaded dataset) since none of that matters for testing the
    validation/bookkeeping logic in isolation."""
    ds = DatasetService.__new__(DatasetService)
    ds._lock = __import__("threading").RLock()
    ds._dataset_key = "fake_dataset_key"
    ds._states = dict(states)
    ds._tiers = {}
    ds._ever_active = {}
    return ds


@contextmanager
def _patched_repo(*, set_state_returns=True, set_tier_returns=True, deprecate_returns=True):
    """Swaps out the exact module-level names dataset_service.py calls
    through (`SessionLocal`, `state_repo.set_class_state`, etc.) for the
    duration of the block, restoring them afterward - no pytest fixture
    dependency, works under the standalone runner too."""
    class _NoopSession:
        def close(self):
            pass

    originals = {
        "SessionLocal": ds_module.SessionLocal,
        "set_class_state": ds_module.state_repo.set_class_state,
        "set_class_tier": ds_module.state_repo.set_class_tier,
        "mark_class_deprecated": ds_module.state_repo.mark_class_deprecated,
    }
    ds_module.SessionLocal = lambda: _NoopSession()
    ds_module.state_repo.set_class_state = lambda db, key, cid, state: set_state_returns
    ds_module.state_repo.set_class_tier = lambda db, key, cid, tier: set_tier_returns
    ds_module.state_repo.mark_class_deprecated = lambda db, key, cid, aid: deprecate_returns
    try:
        yield
    finally:
        ds_module.SessionLocal = originals["SessionLocal"]
        ds_module.state_repo.set_class_state = originals["set_class_state"]
        ds_module.state_repo.set_class_tier = originals["set_class_tier"]
        ds_module.state_repo.mark_class_deprecated = originals["mark_class_deprecated"]


def test_set_class_state_refuses_to_move_a_class_out_of_active() -> None:
    """Only mark_class_deprecated() may move a class out of 'active' -
    set_class_state() must never be usable to silently demote a shipped
    class."""
    ds = _bare_ds({"0": "active"})
    with _patched_repo():
        try:
            ds.set_class_state(0, "collecting_data")
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "active" in str(exc)
    assert ds._states["0"] == "active"  # untouched


def test_set_class_state_refuses_to_set_deprecated_directly() -> None:
    """Deprecating must go through mark_class_deprecated() so
    deprecated_at/deprecated_by_id are actually recorded."""
    ds = _bare_ds({"0": "eligible"})
    with _patched_repo():
        try:
            ds.set_class_state(0, "deprecated")
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "mark_class_deprecated" in str(exc)


def test_set_class_state_still_allows_promoting_eligible_to_active() -> None:
    ds = _bare_ds({"0": "eligible"})
    with _patched_repo():
        ds.set_class_state(0, "active")
    assert ds._states["0"] == "active"


def test_set_class_state_rejects_invalid_value_before_touching_the_db() -> None:
    ds = _bare_ds({"0": "active"})
    with _patched_repo():
        try:
            ds.set_class_state(0, "not_a_real_state")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    assert ds._states["0"] == "active"  # untouched


def test_set_class_state_succeeds_and_updates_the_in_memory_cache() -> None:
    ds = _bare_ds({"0": "collecting_data"})
    with _patched_repo():
        ds.set_class_state(0, "eligible")
    assert ds._states["0"] == "eligible"


def test_set_class_state_raises_when_repo_reports_not_found() -> None:
    ds = _bare_ds({"0": "active"})
    with _patched_repo(set_state_returns=False):
        try:
            ds.set_class_state(5, "eligible")
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "No class 5" in str(exc)


def test_set_class_tier_rejects_invalid_value() -> None:
    ds = _bare_ds({"0": "active"})
    with _patched_repo():
        try:
            ds.set_class_tier(0, "not_a_real_tier")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_set_class_tier_succeeds() -> None:
    ds = _bare_ds({"0": "active"})
    with _patched_repo():
        ds.set_class_tier(0, "safety")
    assert ds._tiers["0"] == "safety"


def test_mark_class_deprecated_raises_for_discovered_class() -> None:
    ds = _bare_ds({"0": "discovered"})
    with _patched_repo():
        try:
            ds.mark_class_deprecated(0, annotator_id=1)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "discovered" in str(exc)


def test_mark_class_deprecated_raises_for_collecting_data_class() -> None:
    ds = _bare_ds({"0": "collecting_data"})
    with _patched_repo():
        try:
            ds.mark_class_deprecated(0, annotator_id=1)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_mark_class_deprecated_raises_for_already_deprecated_class() -> None:
    ds = _bare_ds({"0": "deprecated"})
    with _patched_repo():
        try:
            ds.mark_class_deprecated(0, annotator_id=1)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_mark_class_deprecated_raises_for_missing_class() -> None:
    ds = _bare_ds({})
    with _patched_repo():
        try:
            ds.mark_class_deprecated(99, annotator_id=1)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "No class 99" in str(exc)


def test_mark_class_deprecated_succeeds_for_active_class() -> None:
    ds = _bare_ds({"0": "active"})
    with _patched_repo():
        ds.mark_class_deprecated(0, annotator_id=7)
    assert ds._states["0"] == "deprecated"
    assert ds._ever_active["0"] is True


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
