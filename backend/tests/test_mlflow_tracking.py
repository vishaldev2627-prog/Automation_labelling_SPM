"""Tests for the MLflow tracking wrapper (M5, Scope A only).

Only `is_configured` is meaningfully pure - everything else in
mlflow_tracking.py talks to a real (or absent) MLflow server, which is
exercised live against a running instance rather than mocked here (same
reasoning as test_golden_set.py staying DB-free: the *rule* worth pinning is
"unconfigured means no-op", not SDK call plumbing).

    cd backend && python -m pytest tests/test_mlflow_tracking.py
    cd backend && python -m tests.test_mlflow_tracking   # no pytest
"""
from __future__ import annotations

from app.config import Settings
from app.services.mlflow_tracking import is_configured, start


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_unconfigured_by_default() -> None:
    """A deployment that never sets MLFLOW_TRACKING_URI must not need a
    reachable MLflow server just to train - training stays fully functional,
    only untracked."""
    assert is_configured(_settings()) is False


def test_configured_when_uri_is_set() -> None:
    assert is_configured(_settings(mlflow_tracking_uri="http://mlflow:5000")) is True


def test_start_is_a_no_op_when_unconfigured() -> None:
    """start() must never attempt to import/contact mlflow at all when
    unconfigured - not just fail gracefully if it did."""
    assert start(_settings(), run_name="x", tags={}) is False


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
