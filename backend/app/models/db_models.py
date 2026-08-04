"""SQLAlchemy ORM models for Postgres-backed annotation state.

Phase 1a of annotation_module_build_plan.md: replaces per-image JSON files
(`<dataset>/.annotation_state/*.json` + `_meta.json`) with durable, queryable
storage, keyed by (dataset_view, image_id) rather than a spine stamp - the
spine stamp (coach_index, axle_id, side, view, longitudinal_position_mm)
only exists once Phase 1b's pipeline ingestion lands, so those columns are
deliberately not here yet. Adding them later is a nullable-column migration,
not a breaking one.

annotation_state.payload / annotation_history.payload store the same JSON
shape dataset_service.py already produces via
`ImageAnnotations.model_dump(mode="json")` (see app/models/schemas.py) - kept
as JSONB rather than fully normalized into object/polygon rows so the
DatasetService's existing public interface doesn't have to change (task #4).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Annotator(Base):
    """A named human identity (see task #3) - not full auth, just a name
    every save/review can be attributed to.

    `role` is plain "annotator" by default. "golden_curator" marks who is
    allowed to verify or update the frozen golden eval set (Phase 4, plan
    §5 risk "Golden-set contamination") - the golden set itself (a separate
    storage table) isn't built yet, so this role doesn't gate anything on
    its own yet either. It exists now so Phase 4's golden-set write paths
    can check it from day one instead of retrofitting a permission concept
    onto data that's already live. Enforced at the application layer, not a
    DB enum/check constraint, to keep adding roles a one-line change."""

    __tablename__ = "annotators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    role: Mapped[str] = mapped_column(String, nullable=False, server_default="annotator")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AnnotationState(Base):
    """Current/latest annotation payload for one image in one dataset view.

    One row per (dataset_view, image_id) - replaces
    `.annotation_state/<image_id>.json`.
    """

    __tablename__ = "annotation_state"
    __table_args__ = (UniqueConstraint("dataset_view", "image_id", name="uq_annotation_state_view_image"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_view: Mapped[str] = mapped_column(String, nullable=False, index=True)
    image_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Denormalized from payload["completed"] so progress/ETA queries
    # (get_dataset_info, list_images) don't need to parse JSON per row.
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    updated_by_id: Mapped[int | None] = mapped_column(ForeignKey("annotators.id"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    updated_by: Mapped["Annotator | None"] = relationship()


class AnnotationHistory(Base):
    """Append-only audit trail - every save creates a row here, the
    annotation_state row above only ever holds the latest snapshot.

    Not FK'd to annotation_state.id: history must survive independent of
    whatever the current row looks like, and must never be edited or
    deleted (Phase 4 audit sampling reads this).
    """

    __tablename__ = "annotation_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    dataset_view: Mapped[str] = mapped_column(String, nullable=False, index=True)
    image_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)  # e.g. "save", "mark_completed"
    annotator_id: Mapped[int | None] = mapped_column(ForeignKey("annotators.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    annotator: Mapped["Annotator | None"] = relationship()


class DatasetClass(Base):
    """One class definition (id/name/color) for one dataset view - replaces
    `_meta.json`'s {"classes": [...], "colors": {...}} and is the DB source
    of truth `add_class()` writes through to classes.txt/data.yaml for YOLO
    export compatibility (dataset_service._persist_class_names, unchanged)."""

    __tablename__ = "dataset_classes"
    __table_args__ = (UniqueConstraint("dataset_view", "class_id", name="uq_dataset_classes_view_class"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_view: Mapped[str] = mapped_column(String, nullable=False, index=True)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    color: Mapped[str] = mapped_column(String, nullable=False)
