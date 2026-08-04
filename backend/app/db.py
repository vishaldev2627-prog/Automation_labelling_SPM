"""SQLAlchemy engine/session setup for Postgres-backed annotation state.

This is the durable storage layer underneath app.session_context's
SessionBundle - SessionBundle stays the in-memory, per-request cache of
service instances; this module is what those services persist to instead of
per-image JSON files (see dataset_service.py, migrated in Phase 1a of
annotation_module_build_plan.md).
"""
from __future__ import annotations

from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

_engine = create_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yields a request-scoped DB session, closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
