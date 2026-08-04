"""Per-session backend service registry.

Every browser tab sends an X-Session-Id header (see main.py's middleware),
so each can have its own currently-loaded dataset view (DatasetService) and
the higher-level services built on top of it - mask generation, batch
processing, propagation, detector training, similarity search - without
stepping on other sessions' state. Without this, all of those services were
module-level singletons that captured one fixed DatasetService reference the
first time they were built and held it forever, so switching the "active
dataset" for one browser tab silently switched it for every other tab too.

SAM2 itself (app.services.sam_service) stays a single shared instance across
all sessions - one GPU serving everyone - which is fine since set_image()+
predict() are now one atomic locked section (see sam_service.py), safe for
concurrent callers regardless of which session/dataset they belong to.
"""
from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from typing import Any, Optional

DEFAULT_SESSION_ID = "default"

# Sessions idle longer than this are evicted on the next new-session lookup,
# so a growing number of abandoned browser tabs doesn't leak memory forever.
SESSION_IDLE_TTL_SECONDS = 8 * 60 * 60

_current_session_id: ContextVar[str] = ContextVar("current_session_id", default=DEFAULT_SESSION_ID)


def set_current_session_id(session_id: Optional[str]) -> None:
    _current_session_id.set(session_id or DEFAULT_SESSION_ID)


def get_current_session_id() -> str:
    return _current_session_id.get()


class SessionBundle:
    """Lazily-constructed, session-scoped instances of every dataset-bound service.

    Each attribute is built on first access by that service's own
    get_X_service() function (see dataset_service.py etc.) and cached here,
    mirroring the module-level singleton pattern those functions used before -
    just keyed per session instead of once per process.
    """

    def __init__(self) -> None:
        # RLock, not Lock: get_propagation_service() acquires this lock, then
        # - while still holding it - calls get_similarity_service() and
        # get_mask_generation_service(), which each acquire it again for
        # their own first-time construction (see their get_X_service()
        # functions). With a plain Lock that's a same-thread self-deadlock
        # whenever propagation is triggered before those two have been built
        # for this session - which a first-ever "save + mark completed"
        # always is. Reproduced live: the underlying DB write succeeded (it
        # happens earlier in the request, unrelated to this lock) but the
        # HTTP response never returned because the request thread deadlocked
        # here afterward. RLock keeps the same cross-thread mutual-exclusion
        # guarantee while allowing the thread that already holds it to
        # re-enter.
        self.lock = threading.RLock()
        self.last_used = time.monotonic()
        self.dataset_service: Any = None
        self.mask_generation_service: Any = None
        self.batch_service: Any = None
        self.propagation_service: Any = None
        self.detector_service: Any = None
        self.similarity_service: Any = None
        # Lightweight per-annotator identity (Phase 1a, task #3) - a name,
        # not a login. Set via POST /api/annotator/identify, read by
        # dataset_service.py when it records who made a save (task #4).
        self.annotator_id: Optional[int] = None
        self.annotator_name: Optional[str] = None

    def touch(self) -> None:
        self.last_used = time.monotonic()


_bundles: dict[str, SessionBundle] = {}
_registry_lock = threading.Lock()


def get_session_bundle() -> SessionBundle:
    """Get or create the SessionBundle for the current request's session id."""
    session_id = get_current_session_id()
    bundle = _bundles.get(session_id)
    if bundle is None:
        with _registry_lock:
            bundle = _bundles.get(session_id)
            if bundle is None:
                bundle = SessionBundle()
                _bundles[session_id] = bundle
                _evict_idle_sessions_locked()
    bundle.touch()
    return bundle


def set_current_annotator(annotator_id: int, name: str) -> None:
    """Attach a resolved annotator identity to the current request's session
    bundle, so it survives across requests from the same browser tab."""
    bundle = get_session_bundle()
    with bundle.lock:
        bundle.annotator_id = annotator_id
        bundle.annotator_name = name


def get_current_annotator() -> tuple[Optional[int], Optional[str]]:
    bundle = get_session_bundle()
    return bundle.annotator_id, bundle.annotator_name


def _evict_idle_sessions_locked() -> None:
    now = time.monotonic()
    stale = [
        sid
        for sid, b in _bundles.items()
        if sid != DEFAULT_SESSION_ID and now - b.last_used > SESSION_IDLE_TTL_SECONDS
    ]
    for sid in stale:
        del _bundles[sid]
