"""Tests for DetectorService._resolve_base_weights (Module 8 of the
class-incremental promotion plan - docs/mlflow_class_incremental_
architecture.md §F/§K item 7).

No real MLflow server or GPU: `mlflow_tracking.is_configured` and the
lazily-imported `mlflow`/`MlflowClient` are monkeypatched at the module
level (plain attribute swap + restore, same pattern as
tests/test_class_lifecycle_schema.py) so this stays a fast unit test.

    cd backend && python -m pytest tests/test_detector_service_warm_start.py
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path

import app.services.detector_service as det_svc
from app.services.detector_service import MODEL_WEIGHTS, DetectorService


class _FakeDS:
    dataset_key = "fake_view"


def _detector() -> DetectorService:
    return DetectorService(dataset_service=_FakeDS(), models_dir=Path("/tmp/unused_models_dir_warm_start_test"))


class _FakeSettings:
    mlflow_tracking_uri = "http://fake-mlflow:5000"


class _FakeVersion:
    def __init__(self, version: str, run_id: str) -> None:
        self.version = version
        self.run_id = run_id


@contextmanager
def _fake_mlflow_module(production_versions: list, download_side_effect=None):
    """Installs a fake `mlflow` package (and `mlflow.tracking`) into
    sys.modules for the duration of the block, so detector_service.py's
    lazy `import mlflow` / `from mlflow.tracking import MlflowClient`
    resolve to controlled fakes - restores whatever was there afterward.
    `production_versions` is a plain list of _FakeVersion (empty = no
    Production version exists yet)."""

    class _FakeClient:
        def get_latest_versions(self, name, stages):
            assert stages == ["Production"]
            return production_versions

    def _download_artifacts(run_id, artifact_path):
        if download_side_effect is not None:
            raise download_side_effect
        return f"/fake/downloaded/{run_id}/{artifact_path}"

    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda uri: None
    fake_mlflow.artifacts = types.SimpleNamespace(download_artifacts=_download_artifacts)
    fake_tracking = types.ModuleType("mlflow.tracking")
    fake_tracking.MlflowClient = _FakeClient
    fake_mlflow.tracking = fake_tracking

    originals = {k: sys.modules.get(k) for k in ("mlflow", "mlflow.tracking")}
    sys.modules["mlflow"] = fake_mlflow
    sys.modules["mlflow.tracking"] = fake_tracking
    try:
        yield
    finally:
        for k, v in originals.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_mlflow_not_configured_falls_back_to_stock_checkpoint_cold() -> None:
    det = _detector()
    settings = _FakeSettings()
    settings.mlflow_tracking_uri = ""  # not configured
    weights, parent = det._resolve_base_weights(settings)
    assert weights == MODEL_WEIGHTS
    assert parent is None


def test_no_production_version_falls_back_cleanly() -> None:
    det = _detector()
    with _fake_mlflow_module(production_versions=[]):
        weights, parent = det._resolve_base_weights(_FakeSettings())
    assert weights == MODEL_WEIGHTS
    assert parent is None


def test_warm_start_downloads_production_weights_when_available() -> None:
    det = _detector()
    with _fake_mlflow_module(production_versions=[_FakeVersion(version="3", run_id="run-abc")]):
        weights, parent = det._resolve_base_weights(_FakeSettings())
    assert weights == "/fake/downloaded/run-abc/weights/best.pt"
    assert parent == "3"


def test_download_failure_falls_back_to_stock_checkpoint_not_an_exception() -> None:
    det = _detector()
    with _fake_mlflow_module(
        production_versions=[_FakeVersion(version="3", run_id="run-abc")],
        download_side_effect=RuntimeError("network down"),
    ):
        weights, parent = det._resolve_base_weights(_FakeSettings())
    assert weights == MODEL_WEIGHTS
    assert parent is None


def test_mlflow_lookup_failure_falls_back_cleanly() -> None:
    det = _detector()

    class _RaisingClient:
        def get_latest_versions(self, name, stages):
            raise RuntimeError("mlflow server unreachable")

    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda uri: None
    fake_tracking = types.ModuleType("mlflow.tracking")
    fake_tracking.MlflowClient = _RaisingClient
    fake_mlflow.tracking = fake_tracking

    originals = {k: sys.modules.get(k) for k in ("mlflow", "mlflow.tracking")}
    sys.modules["mlflow"] = fake_mlflow
    sys.modules["mlflow.tracking"] = fake_tracking
    try:
        weights, parent = det._resolve_base_weights(_FakeSettings())
    finally:
        for k, v in originals.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert weights == MODEL_WEIGHTS
    assert parent is None


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
