"""Tests for enforced split integrity (M3).

The problem: `export_service` already *measured* split-integrity violations
into the manifest (M2) - `pseudo_synthetic_in_val_or_test`,
`auto_accepted_in_val_or_test` - but did nothing about them. A propagated
(pseudo/synthetic) or auto-accepted image whose deterministic hash-based split
landed it in val/test shipped there anyway, with the manifest honestly
recording the violation instead of preventing it. `pipeline.md`'s retrain rule
(via `FINAL_AIML_ARCHITECTURE` §12) is explicit: pseudo/synthetic labels go to
train only, never valid/test.

M3 turns the measurement into enforcement in two places:

1. `ExportService._enforce_split_integrity` - the per-image decision, forcing
   a propagated or auto-accepted image's split to `train` regardless of what
   its deterministic hash bucket says.
2. `ExportService._finalize_snapshot` - a hard refusal (`SplitIntegrityViolation`)
   if a snapshot's measured stats ever report a violation despite (1), since a
   snapshot is the immutable, MLflow-tracked handoff artifact and must never
   claim a compliance it does not have.

Both are pure/isolated from Postgres and SAM2 deliberately, so they're tested
directly rather than through the full DB-backed export path.

    cd backend && python -m pytest tests/test_split_integrity.py
    cd backend && python -m tests.test_split_integrity   # no pytest
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from app.services.export_service import ExportService, SplitIntegrityViolation

# ------------------------------------------- the per-image forcing decision

def test_train_split_is_left_alone_regardless_of_provenance() -> None:
    """An image already headed to train needs no override - nothing to force."""
    split, forced = ExportService._enforce_split_integrity("train", propagated_here=3, is_auto_accepted=True)
    assert split == "train"
    assert forced is False


def test_val_split_with_no_propagation_or_auto_accept_is_untouched() -> None:
    """The common case: an ordinary human-annotated image's val assignment
    must not be perturbed by a rule that doesn't apply to it."""
    split, forced = ExportService._enforce_split_integrity("val", propagated_here=0, is_auto_accepted=False)
    assert split == "val"
    assert forced is False


def test_propagated_image_is_forced_out_of_val() -> None:
    split, forced = ExportService._enforce_split_integrity("val", propagated_here=1, is_auto_accepted=False)
    assert split == "train"
    assert forced is True


def test_auto_accepted_image_is_forced_out_of_val() -> None:
    split, forced = ExportService._enforce_split_integrity("val", propagated_here=0, is_auto_accepted=True)
    assert split == "train"
    assert forced is True


def test_both_propagated_and_auto_accepted_still_forces_once() -> None:
    """Belt-and-braces: an image can be both at once (auto-accepted from a
    propagated label). The result is still just "train", not a special case."""
    split, forced = ExportService._enforce_split_integrity("val", propagated_here=2, is_auto_accepted=True)
    assert split == "train"
    assert forced is True


# --------------------------------------------- the snapshot-finalize refusal

def _stub_service() -> ExportService:
    # _finalize_snapshot's split-integrity check runs before self._ds is ever
    # touched, so a real DatasetService isn't needed to exercise it.
    return ExportService(dataset_service=None, exports_dir=Path("/unused"))


def test_finalize_refuses_a_snapshot_reporting_a_violation() -> None:
    service = _stub_service()
    stats = {
        "split_integrity_ok": False,
        "split_integrity": {"pseudo_synthetic_in_val_or_test": 2, "auto_accepted_in_val_or_test": 0},
    }
    with tempfile.TemporaryDirectory() as tmp:
        try:
            service._finalize_snapshot(request=None, staging=Path(tmp), stats=stats)
        except SplitIntegrityViolation:
            pass
        else:
            raise AssertionError("expected SplitIntegrityViolation, none was raised")


def test_finalize_does_not_raise_when_integrity_is_clean() -> None:
    """Confirms the guard is specific to the failure case - it must not reject
    every snapshot. Runs far enough to prove no SplitIntegrityViolation was
    raised; a real snapshot finalize needs a loaded DatasetService, which is
    out of scope for this guard-only test."""
    service = _stub_service()
    stats = {"split_integrity_ok": True, "split_integrity": {}}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            service._finalize_snapshot(request=None, staging=Path(tmp), stats=stats)
        except SplitIntegrityViolation:
            raise
        except Exception:
            pass  # expected: fails later needing a real self._ds - not this guard


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
