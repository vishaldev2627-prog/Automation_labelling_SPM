"""GPU scheduling guard (M8's "inference always wins" rule, guard-only slice).

The risk this exists for, from the build plan's own words: an automated
training trigger on a shared GPU is a [HIGH] hazard - training must never
compete with SAM2 serving live annotators, and a run must never be forced
through just because a snapshot happened to finalize. M8's full scope also
covers a scheduler, data watcher, and monitoring; this module is only the
one piece that was missing before `auto_train_on_handoff` (M5/M9) could be
called safe to enable in a live multi-annotator deployment: the actual
GPU-busy check.

Process-wide, not per-session: SAM2 (`sam_service.py`) is deliberately one
shared instance across all sessions - one GPU serving everyone - so
"is the GPU busy" has to mean "is any session's inference in flight right
now", not scoped to whichever session happens to be training.

**Check-before-start, not runtime preemption.** Once training's own
`model.train()` call is running, this module does not pause or interrupt
it mid-epoch - true GPU preemption of an already-running training loop is a
different, much larger problem than the one being solved here. What this
guards against is training *starting* while SAM2 is actively serving, and
it makes that check block-and-retry rather than a one-shot refusal, so a
training run doesn't get abandoned just because an annotator happened to
click at the wrong moment.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

_lock = threading.Lock()
_active_inference_count = 0
_last_activity_at: float = 0.0

# Inference is bursty - one predict() call takes well under a second, with
# gaps between an annotator's clicks. A busy check that only looked at "is a
# call in flight right now" would happily slip training in between two
# clicks from the same annotator, then immediately contend with the very
# next one. This grace window after the last call closes that gap without
# meaningfully delaying training once an annotator has genuinely gone idle.
RECENT_ACTIVITY_WINDOW_SECONDS = 5.0


@contextmanager
def track_inference():
    """Wrap around any SAM2 GPU call (set_image/predict_box/predict_points).
    Reentrant-safe via the counter, not the lock - SAMService's own lock
    already serializes concurrent calls into one at a time; this only needs
    to know whether *any* are in flight or were, recently."""
    global _active_inference_count, _last_activity_at
    with _lock:
        _active_inference_count += 1
    try:
        yield
    finally:
        with _lock:
            _active_inference_count -= 1
            _last_activity_at = time.monotonic()


def is_gpu_busy(now: float | None = None) -> bool:
    """True if inference is in flight right now, or finished within the
    last `RECENT_ACTIVITY_WINDOW_SECONDS`. `now` is injectable for tests -
    real callers never need to pass it."""
    now = time.monotonic() if now is None else now
    with _lock:
        if _active_inference_count > 0:
            return True
        if _last_activity_at == 0.0:
            return False
        return (now - _last_activity_at) < RECENT_ACTIVITY_WINDOW_SECONDS
