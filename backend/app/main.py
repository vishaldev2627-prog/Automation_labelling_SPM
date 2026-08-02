"""FastAPI application entrypoint."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import batch, dataset, detector, export, images, masks, progress, similarity
from app.utils.logging_config import setup_logging

settings = get_settings()
setup_logging(settings.log_level, settings.log_file)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Railway Segmentation Annotation Tool",
    description="Local AI-assisted tool for converting YOLO detection boxes to YOLO segmentation polygons using SAM2.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def session_context_middleware(request: Request, call_next):
    """Tag this request with its session id (see app.session_context) so
    get_dataset_service() and friends resolve to the right session's
    currently-loaded dataset view instead of one shared global instance.

    Read from the session_id cookie, not just the X-Session-Id header: plain
    <img src="..."> requests (how the browser actually fetches image bytes)
    never carry custom headers, only whatever axios/fetch calls set - but a
    cookie is sent automatically on every same-origin request regardless of
    how it was initiated, so it's the only mechanism that covers both.
    """
    from app.session_context import set_current_session_id

    session_id = request.cookies.get("session_id") or request.headers.get("x-session-id")
    set_current_session_id(session_id)
    return await call_next(request)

app.include_router(dataset.router)
app.include_router(images.router)
app.include_router(masks.router)
app.include_router(batch.router)
app.include_router(export.router)
app.include_router(progress.router)
app.include_router(detector.router)
app.include_router(similarity.router)


@app.on_event("startup")
def auto_load_dataset() -> None:
    """Load the "side_view" dataset view on boot for the default session (any
    client that hasn't sent an X-Session-Id header yet), so there's no
    manual "load dataset" step required before the multi-view dropdown
    lands in the frontend. DATASET_PATH is the parent dir containing the
    side_view/underbelly/wheel_shelling sibling folders - see
    routers/dataset.py's DATASET_VIEWS."""
    from pathlib import Path

    from app.services.dataset_service import DatasetNotFoundError, get_dataset_service

    default_path = Path(settings.dataset_path) / "side_view"
    try:
        info = get_dataset_service().load_dataset(str(default_path))
        logger.info("Auto-loaded dataset at %s (%d images)", info.dataset_path, info.total_images)
    except DatasetNotFoundError as exc:
        logger.warning("Skipping dataset auto-load: %s", exc)
        return

    if settings.propagation_enabled:
        from app.services.similarity_service import get_similarity_service

        get_similarity_service().start_reindex()
        logger.info("Started background similarity indexing for annotation propagation")


@app.get("/api/health")
def health() -> dict:
    from app.services.sam_service import get_sam_service

    return {"status": "ok", "sam": get_sam_service().status}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error while processing %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
