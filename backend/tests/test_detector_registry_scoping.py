"""Regression test for the per-dataset-view detector registry (M0.1 / P-1).

The bug this locks down: DetectorService is session-scoped (see
app.session_context) but models_dir is one shared host path, so a single
global `detector_registry.json` meant training while `buffer` was loaded
made that model the active auto-detect model for `side_view` annotators too,
with buffer's class list attached - so class ids silently meant different
components in the two views.

Runs with no database and no GPU: DetectorService only touches
dataset_service for `dataset_key` and `get_classes()`, both faked here.
Importing app.services.detector_service builds a SQLAlchemy engine at import
time but never connects (pool_pre_ping is lazy), so no Postgres is needed.

    cd backend && python -m pytest tests/test_detector_registry_scoping.py
    cd backend && python -m tests.test_detector_registry_scoping   # no pytest
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from app.models.schemas import ClassInfo
from app.services.dataset_service import DatasetNotFoundError
from app.services.detector_service import (
    LEGACY_REGISTRY_NAME,
    DetectorService,
    _slug_for,
)
from app.utils.file_utils import atomic_write_json, read_json


@dataclass
class FakeDatasetService:
    """Minimal stand-in for the two DatasetService members DetectorService uses."""

    key: str | None
    classes: list[str] = field(default_factory=list)

    @property
    def dataset_key(self) -> str:
        if self.key is None:
            raise DatasetNotFoundError("No dataset loaded.")
        return self.key

    def get_classes(self) -> list[ClassInfo]:
        if self.key is None:
            raise DatasetNotFoundError("No dataset loaded.")
        return [
            ClassInfo(class_id=i, name=name, color="#000000", safety_critical=False)
            for i, name in enumerate(self.classes)
        ]


def _svc(models_dir: Path, key: str | None, classes: list[str]) -> DetectorService:
    return DetectorService(FakeDatasetService(key, classes), models_dir)


def test_two_views_get_separate_registries() -> None:
    """The actual P-1 scenario: train under one view, confirm the other view's
    active detector is untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp)
        side = _svc(models_dir, "/data/dataset/side_view", ["coupler", "wheel"])
        buffer = _svc(models_dir, "/data/dataset/buffer", ["buffer_plate"])

        assert side._registry_path != buffer._registry_path

        # Stand in for a completed training run under `buffer`.
        weights = buffer._view_dir / "detector_v1.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"not-real-weights")
        buffer._save_registry(
            {
                "version": 1,
                "classes": ["buffer_plate"],
                "path": str(weights),
                "dataset_key": "/data/dataset/buffer",
            }
        )

        assert buffer.is_active() is True
        # Before the fix this was also True, with buffer's single class.
        assert side.is_active() is False
        assert side.get_info().active is False


def test_slug_is_unique_per_root_and_stable() -> None:
    """Two roots sharing a basename must not collide, and the slug must not
    drift between calls (it names a directory holding real weights)."""
    a = _slug_for("/data/one/images_root")
    b = _slug_for("/data/two/images_root")
    assert a != b
    assert a == _slug_for("/data/one/images_root")
    # Filesystem-safe: no separators or characters needing escaping.
    assert "/" not in a and "\\" not in a and " " not in a


def test_no_dataset_loaded_reports_inactive_instead_of_raising() -> None:
    """GET /api/detector/active does not require_loaded(), so is_active() and
    get_info() must degrade rather than raise DatasetNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp:
        svc = _svc(Path(tmp), None, [])
        assert svc.is_active() is False
        assert svc.get_info().active is False


def test_legacy_registry_adopted_only_on_exact_class_match() -> None:
    """The old global registry recorded no view, so it may only be adopted
    when its class list proves which view it belongs to."""
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp)
        legacy_weights = models_dir / "detector_v7.pt"
        legacy_weights.write_bytes(b"not-real-weights")
        atomic_write_json(
            models_dir / LEGACY_REGISTRY_NAME,
            {"version": 7, "classes": ["coupler", "wheel"], "path": str(legacy_weights)},
        )

        matching = _svc(models_dir, "/data/dataset/side_view", ["coupler", "wheel"])
        assert matching.is_active() is True
        assert matching.get_info().version == 7
        # Adoption copies into the per-view registry; legacy file stays put.
        assert read_json(matching._registry_path)["adopted_from_legacy_registry"]
        assert (models_dir / LEGACY_REGISTRY_NAME).exists()

        mismatched = _svc(models_dir, "/data/dataset/buffer", ["buffer_plate"])
        assert mismatched.is_active() is False

        # Next trained version continues from the adopted number, not from 1,
        # so weights files can't collide with the adopted ones.
        assert matching._load_registry().get("version") == 7


def test_legacy_registry_with_missing_weights_not_adopted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp)
        atomic_write_json(
            models_dir / LEGACY_REGISTRY_NAME,
            {"version": 3, "classes": ["coupler"], "path": str(models_dir / "gone.pt")},
        )
        svc = _svc(models_dir, "/data/dataset/side_view", ["coupler"])
        assert svc.is_active() is False


def test_declined_adoption_is_cached() -> None:
    """is_active() runs on every image open (dataset_service._try_auto_detect),
    so a non-adoptable legacy registry must not be re-read every time."""
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp)
        atomic_write_json(
            models_dir / LEGACY_REGISTRY_NAME,
            {"version": 2, "classes": ["other"], "path": str(models_dir / "x.pt")},
        )
        svc = _svc(models_dir, "/data/dataset/side_view", ["coupler"])
        assert svc.is_active() is False
        assert _slug_for("/data/dataset/side_view") in svc._legacy_adoption_declined
        assert svc.is_active() is False


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a hand-rolled runner wants everything
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
