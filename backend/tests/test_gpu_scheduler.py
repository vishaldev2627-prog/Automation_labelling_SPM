"""Tests for the M8 GPU-scheduling guard ("inference always wins").

The risk this defends against, from the build plan: an automated training
trigger on a shared GPU must never compete with SAM2 serving live
annotators. `is_gpu_busy` is the check `detector_service._wait_for_gpu_idle`
polls before letting a training run's GPU-heavy `model.train()` call start -
see gpu_scheduler.py's module docstring for why this is check-before-start
rather than runtime preemption of an already-running job.

Pure/in-process - no GPU, no SAM2, no Postgres. `is_gpu_busy`'s `now`
parameter is injectable specifically so the recent-activity window can be
tested without a real `time.sleep`.

    cd backend && python -m pytest tests/test_gpu_scheduler.py
    cd backend && python -m tests.test_gpu_scheduler   # no pytest
"""
from __future__ import annotations

import time

from app.services import gpu_scheduler


def _reset() -> None:
    """Module-level state is process-wide by design (see the module
    docstring - SAM2 is one shared instance across all sessions) - tests
    must reset it explicitly so one test's activity doesn't leak into the
    next."""
    gpu_scheduler._active_inference_count = 0
    gpu_scheduler._last_activity_at = 0.0


def test_idle_at_process_start() -> None:
    _reset()
    assert gpu_scheduler.is_gpu_busy() is False


def test_busy_while_inference_is_in_flight() -> None:
    _reset()
    with gpu_scheduler.track_inference():
        assert gpu_scheduler.is_gpu_busy() is True


def test_idle_again_once_inference_completes_and_the_window_passes() -> None:
    _reset()
    with gpu_scheduler.track_inference():
        pass
    # Simulate "well past the grace window" without a real sleep.
    later = time.monotonic() + gpu_scheduler.RECENT_ACTIVITY_WINDOW_SECONDS + 1
    assert gpu_scheduler.is_gpu_busy(now=later) is False


def test_still_busy_within_the_grace_window_after_completion() -> None:
    """A training-start check landing in the gap between two of the same
    annotator's clicks must still see 'busy', not slip through."""
    _reset()
    with gpu_scheduler.track_inference():
        pass
    soon = time.monotonic() + (gpu_scheduler.RECENT_ACTIVITY_WINDOW_SECONDS / 2)
    assert gpu_scheduler.is_gpu_busy(now=soon) is True


def test_nested_or_concurrent_inference_only_clears_when_all_have_finished() -> None:
    """Two callers overlapping (SAMService's own lock serializes the actual
    GPU work, but the busy-tracking counter must still reflect "at least one
    in flight" correctly across overlapping enter/exit)."""
    _reset()
    cm1 = gpu_scheduler.track_inference()
    cm2 = gpu_scheduler.track_inference()
    cm1.__enter__()
    cm2.__enter__()
    assert gpu_scheduler.is_gpu_busy() is True
    cm1.__exit__(None, None, None)
    assert gpu_scheduler.is_gpu_busy() is True  # cm2 still active
    cm2.__exit__(None, None, None)
    assert gpu_scheduler.is_gpu_busy() is True  # just finished, inside the grace window


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
