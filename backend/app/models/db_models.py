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


class AnnotationReview(Base):
    """Second-reviewer sign-off (Phase 4, safety-critical QA layer - see
    annotation_module_build_plan.md). Append-only, like AnnotationHistory,
    but specifically the second-reviewer's decision, not every save - the
    plan's "one save = done" gap this tool used to have.

    Wired into export gating (export_service.py) alongside ExportGateExemption
    below: an image is export-eligible if it's exempt (grandfathered) OR its
    *latest* review decision is "approved". A later "rejected" review after
    an "approved" one correctly un-approves it - see review_service.py's
    get_export_eligible_ids, which always takes the latest decision, not
    "any approval ever."
    """

    __tablename__ = "annotation_reviews"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    dataset_view: Mapped[str] = mapped_column(String, nullable=False, index=True)
    image_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("annotators.id"), nullable=False)
    decision: Mapped[str] = mapped_column(String, nullable=False)  # "approved" | "rejected"
    reason: Mapped[str] = mapped_column(String, nullable=False)  # "second_review" | "audit_sample"
    notes: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    reviewer: Mapped["Annotator"] = relationship()


class ExportGateExemption(Base):
    """Grandfather list for the second-reviewer export gate: images that
    were already `completed` before this gate went live don't need a
    retroactive review to stay exportable (product-owner decision, see
    annotation_module_build_plan.md Phase 4/§5) - only images that become
    `completed` from here on need an approved AnnotationReview.

    Populated once, by the migration that introduces this table, as a
    snapshot of every completed image at that moment - not maintained
    afterward. A row existing here means "exempt," nothing more; there is
    no code path that inserts into this table after the migration.
    """

    __tablename__ = "export_gate_exemptions"

    dataset_view: Mapped[str] = mapped_column(String, primary_key=True)
    image_id: Mapped[str] = mapped_column(String, primary_key=True)


class DatasetSnapshot(Base):
    """A record of one immutable dataset snapshot (M2).

    The snapshot itself is a content-addressed directory on disk (see
    snapshot_service); this table is the index over them, so "which snapshots
    exist, from which class map, published where" is a query rather than a
    directory listing.

    `snapshot_id` is unique **globally, not per view**: it is a content hash, so
    two views producing the identical file set and class map genuinely are the
    same dataset. Re-exporting unchanged data therefore updates this row's
    `last_exported_at` rather than inserting a duplicate.

    Rows are append-only in spirit - a snapshot's identity and manifest never
    change. Only the publishing columns are updated, since publishing is a
    separate, retryable step against already-immutable content.
    """

    __tablename__ = "dataset_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    dataset_view: Mapped[str] = mapped_column(String, nullable=False, index=True)
    class_map_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    class_map_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    manifest: Mapped[dict] = mapped_column(JSONB, nullable=False)
    local_path: Mapped[str] = mapped_column(String, nullable=False)
    file_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    published_uri: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_exported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("annotators.id"), nullable=True)

    created_by: Mapped["Annotator | None"] = relationship()


class ClassMapVersion(Base):
    """An immutable snapshot of one dataset view's class map (M1).

    Never updated, never deleted - a change to the id->name mapping or to
    `exclude_classes` mints a new row (see class_map_service). This is what a
    dataset snapshot pins so that "class 7" is answerable years later, and it is
    the fix for the build plan's `[HIGH]` class-map-drift risk, which already
    materialized once on this project as a 27-class remap.

    Two unique constraints, doing different jobs:
      - `(dataset_view, version)` gives per-view monotonic numbering that is
        never reused.
      - `(dataset_view, content_hash)` makes minting idempotent, so calling
        ensure_version() on every dataset load records genuine drift without
        creating a version per load.

    `names` is `[[class_id, name], ...]` ordered by id, not an object keyed by
    stringified ints - see class_map_service.canonical_payload for why.

    Deliberately excluded from the hashed content: colour, `safety_critical`,
    `fine_structure`. Those live on DatasetClass and are tool-side metadata; a
    curator fixing a safety flag must not invalidate the class map that previous
    snapshots pinned.
    """

    __tablename__ = "class_map_versions"
    __table_args__ = (
        UniqueConstraint("dataset_view", "version", name="uq_class_map_versions_view_version"),
        UniqueConstraint("dataset_view", "content_hash", name="uq_class_map_versions_view_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_view: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    names: Mapped[list] = mapped_column(JSONB, nullable=False)
    exclude_classes: Mapped[list] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("annotators.id"), nullable=True)

    created_by: Mapped["Annotator | None"] = relationship()


class DatasetClass(Base):
    """One class definition (id/name/color) for one dataset view - replaces
    `_meta.json`'s {"classes": [...], "colors": {...}} and is the DB source
    of truth `add_class()` writes through to classes.txt/data.yaml for YOLO
    export compatibility (dataset_service._persist_class_names, unchanged).

    `safety_critical` gates auto_accept_service.py: a safety-critical class
    never auto-accepts regardless of confidence or audit track record -
    always a human, always a second reviewer (plan §4.4). Seeded with a
    keyword-based default (wheel/brake/axle/suspension/disc/spring/crack/
    corrosion/shelling) by the migration that adds this column - an
    engineering starting point for a domain expert to review and correct,
    not a certified list. Editable per-class via the API/UI.

    `fine_structure` marks classes whose value lives in thin, branching,
    length-measured detail - crack, corrosion, wheel shelling. The pipeline
    scores these on Dice/IoU **plus length-recall** (docs/pipeline.md §5.4)
    and shelling length (§5.5), all of which our default polygon path
    degraded: largest-contour-only dropped every piece of a branching crack
    but one, and Douglas-Peucker smoothed away the thin structure being
    measured. A flagged class keeps all contours, skips simplification, and
    exports a binary mask raster alongside the polygon.

    Deliberately a separate flag from `safety_critical`, not a reuse of it:
    they answer different questions. A wheel is safety-critical and its
    outline is a blob that simplifies fine; a crack across that wheel is the
    fine-structure case. Seeded by its own, narrower keyword set.
    """

    __tablename__ = "dataset_classes"
    __table_args__ = (UniqueConstraint("dataset_view", "class_id", name="uq_dataset_classes_view_class"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_view: Mapped[str] = mapped_column(String, nullable=False, index=True)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    color: Mapped[str] = mapped_column(String, nullable=False)
    safety_critical: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    fine_structure: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")


class GoldenSet(Base):
    """One curated, versioned golden eval set (M4, D-Q4: "frozen per-class
    golden set, gating before the shadow canary").

    Append-only, like ClassMapVersion: a curation round never edits an
    existing version's items, it mints a new one. "We curate and version;
    they run the eval" (D-Q4) means a version must stay byte-identical once
    created, or a re-run of the pipeline team's gate against "the same"
    version would silently be scoring something else.

    `dataset_view` + `version` gives per-view monotonic numbering, mirroring
    class_map_versions. There is deliberately no `frozen_at`/"draft" state -
    a version's items are written by `golden_service.add_items` at creation
    and the row exists once they're recorded; nothing later mutates it.
    """

    __tablename__ = "golden_sets"
    __table_args__ = (UniqueConstraint("dataset_view", "version", name="uq_golden_sets_view_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_view: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("annotators.id"), nullable=True)

    created_by: Mapped["Annotator | None"] = relationship()


class GoldenSetItem(Base):
    """One image frozen into a golden set version (M4).

    `image_id` is intentionally **not** unique per dataset_view across
    versions - the same image can appear in more than one curation round (a
    later version re-curating the same frame is a legitimate, if unusual,
    edit path since versions are immutable once written). What every export
    path actually needs is "has this image_id ever been made golden for this
    view, in any version" - see golden_repo.get_golden_image_ids, which reads
    across all versions on purpose so a golden image excluded under an old
    version doesn't quietly become exportable again just because a newer
    version exists.

    `frozen_object_store_key` records where this item's image+label bytes
    landed in the separate golden bucket (see app/services/object_store.py's
    golden functions) - null if that upload failed, since freezing to object
    storage is best-effort/retryable like snapshot publishing (M2) and must
    not block the DB record, which is the source of truth for export
    exclusion regardless of whether the bytes made it to the bucket yet.
    """

    __tablename__ = "golden_set_items"
    __table_args__ = (
        UniqueConstraint("golden_set_id", "image_id", name="uq_golden_set_items_set_image"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    golden_set_id: Mapped[int] = mapped_column(ForeignKey("golden_sets.id"), nullable=False, index=True)
    image_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    frozen_object_store_key: Mapped[str | None] = mapped_column(String, nullable=True)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    golden_set: Mapped["GoldenSet"] = relationship()
