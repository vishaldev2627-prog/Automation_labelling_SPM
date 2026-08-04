"""Get-or-create for the lightweight per-annotator identity (Phase 1a, task
#3) - a name, not a login. See app.session_context for how a name gets
attached to the current session, and app.models.db_models.Annotator for the
table this reads/writes.
"""
from __future__ import annotations

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.db_models import Annotator


def get_or_create_annotator(db: Session, name: str) -> Annotator:
    """Look up an annotator by name, creating it on first use.

    Races on the same new name from two requests are resolved by the table's
    unique constraint on `name`: the losing insert rolls back and re-selects
    rather than erroring, so callers never see the race.
    """
    name = name.strip()
    existing = db.query(Annotator).filter(Annotator.name == name).one_or_none()
    if existing is not None:
        return existing

    annotator = Annotator(name=name)
    db.add(annotator)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        annotator = db.query(Annotator).filter(Annotator.name == name).one()
    else:
        db.refresh(annotator)
    return annotator
