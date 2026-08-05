"""Pydantic models shared across the API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ObjectStatus(str, Enum):
    PENDING = "pending"
    AUTO_GENERATED = "auto_generated"
    EDITED = "edited"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class Condition(str, Enum):
    """Component condition - the label axis the `p1_side_damage` crop
    classifier trains on.

    Vocabulary taken verbatim from `docs/pipeline.md` §5.2:
      "Output: condition in {ok, broken, missing, hanging, displaced,
       dislocated, leaking, damaged, securing_broken, securing_hanging,
       fiba_red, sparking, binding}"

    **This is orthogonal to `class_id`.** `class_id` says *what the component
    is* (coupler, brake cylinder, axle box); this says *what state it is in*.
    They were previously the same field's worth of information, which is to say
    the condition axis did not exist at all - so `side_view` supplied the crops
    that classifier trains on but none of its labels (see DECISIONS_LOG.md,
    C-1).

    Absent (`None`) means **unassessed**, and that is deliberately not `OK`.
    Roughly 2000 images were annotated before this field existed; nobody
    asserted those components were in good condition, and on a defect-detection
    system silently promoting "not looked at" to "fine" is the wrong direction
    of error. Unassessed objects stay fully valid for the detection families and
    are simply absent from the condition-crop export.

    Note `LEAKING` is a legitimate condition and is **not** what
    `exclude_classes: [leakage]` refers to - that excludes a synthetic
    *component class* from serving (`pipeline.md` §5.2, §12). Two different
    things with similar names; do not filter this out.
    """

    OK = "ok"
    BROKEN = "broken"
    MISSING = "missing"
    HANGING = "hanging"
    DISPLACED = "displaced"
    DISLOCATED = "dislocated"
    LEAKING = "leaking"
    DAMAGED = "damaged"
    SECURING_BROKEN = "securing_broken"
    SECURING_HANGING = "securing_hanging"
    FIBA_RED = "fiba_red"
    SPARKING = "sparking"
    BINDING = "binding"


class CoachType(str, Enum):
    """Coach types in scope for labeling: **LHB and ICF only** (confirmed
    2026-08-05).

    This matches the architecture docs - `pipeline.md` §5.7 specifies the
    coach-type classifier's output as `{LHB, ICF}` and
    `FINAL_AIML_ARCHITECTURE` §9's detection record uses the same pair - so
    there is no longer a discrepancy to carry. An earlier reply had listed
    four types (adding Vande Bharat and hybrid); those are deliberately **not**
    labelable values, so the tool cannot produce a label the pipeline's
    2-class classifier and per-type manifests have no slot for.

    `UNKNOWN` is not a third coach type - it is the absence of an answer, and
    it is the default. It exists rather than defaulting to LHB because LHB is
    commonest: `pipeline.md` §8 refuses to select a manifest for an unknown
    coach type instead of assuming one, and inventing a type here would be
    the labeling-side version of exactly the mistake R14 describes
    ("manifest/coach-type error -> false-missing storm").
    """

    LHB = "LHB"
    ICF = "ICF"
    UNKNOWN = "unknown"


class ObjectSource(str, Enum):
    """Where an object's initial box/class came from — independent of its
    review status (ObjectStatus), which tracks whether a human has vetted it."""

    DETECTION_BOX = "detection_box"
    MANUAL = "manual"
    PROPAGATED = "propagated"


class Point(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    """Normalized YOLO-style bounding box (0-1 range), center format."""

    x_center: float
    y_center: float
    width: float
    height: float

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        cx, cy, w, h = self.x_center * img_w, self.y_center * img_h, self.width * img_w, self.height * img_h
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = min(img_w, int(cx + w / 2))
        y2 = min(img_h, int(cy + h / 2))
        return x1, y1, x2, y2


class AnnotationObject(BaseModel):
    """A single annotated object instance within an image.

    **Two independent confidence signals, deliberately separate fields.**
    They used to share one `confidence` field, which meant three different
    things depending on what had run last: 0.0 for a box read from a YOLO
    label file (no signal at all), ultralytics' `box.conf` for a box the
    local detector produced, and - overwriting either of those the moment
    `mask_generation_service.generate_for_object()` ran - the SAM2 *mask*
    score. Both `triage_service`'s low-confidence tier and, more seriously,
    `auto_accept_service`'s 0.95 gate read that field as a detector
    confidence, so after mask generation they were actually deciding whether
    to skip human review based on how cleanly SAM2 segmented the box rather
    than how sure anything was about the class.

    - `detector_confidence`: how sure the *detector* is about this class/box.
      `None` means no signal exists - a plain YOLO label file carries no
      confidence field, so absence is genuinely different from "low", and
      encoding it as 0.0 conflated the two.
    - `mask_confidence`: SAM2's score for the currently selected mask. Says
      nothing about whether the class is right.
    """

    id: str
    class_id: int
    class_name: str = ""
    # What state the component is in, as opposed to what it is. None means
    # unassessed - see Condition, and note that is not the same as "ok".
    condition: Optional[Condition] = None
    bbox: BoundingBox
    polygon: list[Point] = Field(default_factory=list)
    # Extra disjoint pieces of the same object, for `fine_structure` classes
    # only (crack, corrosion, shelling). `polygon` stays the largest piece so
    # every existing consumer - canvas rendering, export, mask rasterization -
    # keeps working unchanged; this holds the rest. A branching or segmented
    # crack used to lose these entirely (polygon_service kept only the largest
    # contour), which silently destroyed the length-recall the pipeline scores
    # these classes on (docs/pipeline.md §5.4).
    extra_polygons: list[list[Point]] = Field(default_factory=list)
    detector_confidence: Optional[float] = None
    mask_confidence: float = 0.0
    all_mask_scores: list[float] = Field(default_factory=list)
    selected_mask_index: int = 0
    status: ObjectStatus = ObjectStatus.PENDING
    visible: bool = True
    source: ObjectSource = ObjectSource.DETECTION_BOX
    propagated_from_image_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_confidence(cls, data):
        """Read the pre-split `confidence` key out of already-stored payloads.

        Every `annotation_state.payload` / `annotation_history.payload` row
        written before this split holds a single `confidence`. It is mapped to
        `mask_confidence` and `detector_confidence` is left `None`, rather
        than trying to guess which of the three meanings a given historical
        value carried - that guess is not recoverable from the payload (a
        below-threshold mask score and a detector confidence both appear as
        `0 < value < 1` with an empty polygon and `PENDING` status).

        Consequence, stated plainly: historical objects have no detector
        confidence, so they are **not** auto-accept eligible until the frame
        is processed again. That is the conservative direction of error for a
        gate whose job is skipping human review, and matches
        `review_service.get_class_audit_stats`' existing habit of
        under-trusting rather than over-trusting.

        No DB migration is needed - this runs on read, and every subsequent
        save writes the new shape.
        """
        if isinstance(data, dict) and "confidence" in data and "mask_confidence" not in data:
            data = dict(data)
            data["mask_confidence"] = data.pop("confidence") or 0.0
        return data


class ImageAnnotations(BaseModel):
    """`no_objects_confirmed` is an explicit "a human looked and there is
    nothing here" assertion, deliberately distinct from an empty `objects`
    list.

    An image can have no objects for three different reasons, and only one of
    them is a usable training sample:
      1. a human confirmed nothing is present  -> a labeled **negative**,
      2. nothing has been annotated yet        -> not done,
      3. every proposed object was rejected    -> also (1) if the human then
         confirms it, otherwise still (2).

    Without this flag all three are indistinguishable, and `export_service`
    drops them all (`if not objects: skipped`). That matters because a
    presence classifier trains on both classes - `VB-BufferBoundary`'s
    `buffer_visible` (FINAL_AIML_ARCHITECTURE §2.2) needs frames where the
    buffer is *absent* as much as frames where it is present, and those were
    being silently discarded.

    On export a confirmed negative writes the image plus an **empty label
    file**, which is the YOLO convention for a background/negative sample.
    """

    image_id: str
    file_name: str
    width: int
    height: int
    objects: list[AnnotationObject] = Field(default_factory=list)
    completed: bool = False
    no_objects_confirmed: bool = False
    # Set per image, applied in bulk from the UI (there is no batch entity in
    # this tool). Needed because the pipeline's component manifest is
    # coach-type-conditional, and because an aggregate per-class label count
    # hides a class that is well covered on LHB and absent on ICF - which is
    # what their R10 "label scarcity - needs label counts" actually needs to
    # be actionable. Unlike longitudinal_position_mm, this does not require
    # encoder calibration constants we don't have, so it can be captured now.
    coach_type: CoachType = CoachType.UNKNOWN
    last_modified: Optional[float] = None


class GenerateMaskRequest(BaseModel):
    image_id: str
    object_id: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    class_id: Optional[int] = None
    positive_points: list[Point] = Field(default_factory=list)
    negative_points: list[Point] = Field(default_factory=list)


class GenerateMaskResponse(BaseModel):
    """`confidence` here is unambiguously the SAM2 mask score - this endpoint
    only ever produces a mask, never a class decision. It maps to
    `AnnotationObject.mask_confidence`; see that model for why the two
    confidences are separate fields."""

    object_id: str
    polygon: list[Point]
    # Must be returned, not just persisted server-side: the frontend holds the
    # object list in memory and posts it back on the next autosave, so anything
    # missing from this response is silently dropped from a fine-structure
    # object on the following save.
    extra_polygons: list[list[Point]] = Field(default_factory=list)
    confidence: float
    all_scores: list[float]
    selected_mask_index: int


class GenerateAllRequest(BaseModel):
    image_id: str


class SaveAnnotationRequest(BaseModel):
    image_id: str
    objects: list[AnnotationObject]
    mark_completed: bool = False
    # Tri-state on purpose: None means "leave whatever is stored alone" so a
    # plain autosave never clears a confirmation, while False is an explicit
    # retraction by a human who changed their mind. A plain bool would make
    # every autosave silently un-confirm the negative.
    no_objects_confirmed: Optional[bool] = None
    # Tri-state for the same reason as the flag above: omitted on a routine
    # save so an autosave can't reset a coach type someone set.
    coach_type: Optional[CoachType] = None


class SetCoachTypeRequest(BaseModel):
    """Bulk coach-type assignment. `image_ids` empty means "every image in the
    loaded view that doesn't have one set yet" - deliberately not "every
    image", so a bulk apply can never silently overwrite work already done."""

    coach_type: CoachType
    image_ids: list[str] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    dataset_path: str
    total_images: int
    completed: int
    remaining: int
    percent_complete: float
    classes: list[str]
    estimated_seconds_remaining: Optional[float] = None


class DatasetView(BaseModel):
    key: str
    label: str


class ClassMapVersionInfo(BaseModel):
    """One immutable class-map version (see db_models.ClassMapVersion).

    `content_hash` is what a dataset snapshot should record alongside `version`:
    the number is convenient for humans, the hash is what actually proves two
    snapshots were built against the same map."""

    version: int
    content_hash: str
    names: dict[int, str]
    exclude_classes: list[str]
    created_at: datetime
    created_by: Optional[str] = None


class ImageListItem(BaseModel):
    image_id: str
    file_name: str
    completed: bool
    object_count: int


class ClassInfo(BaseModel):
    class_id: int
    name: str
    color: str
    safety_critical: bool = False
    # See db_models.DatasetClass.fine_structure - thin/branching/length-measured
    # defects keep every contour, skip simplification, and export a mask raster.
    fine_structure: bool = False


class BatchProcessRequest(BaseModel):
    image_ids: list[str] = Field(default_factory=list)
    overwrite: bool = False


class BatchJobStatus(BaseModel):
    job_id: str
    total: int
    processed: int
    failed: int
    status: str
    current_image: Optional[str] = None
    started_at: float
    updated_at: float


class ExportRequest(BaseModel):
    image_ids: list[str] = Field(default_factory=list)
    only_completed: bool = True
    output_subdir: Optional[str] = None
    # Writes crops/<split>/<condition>/... for the p1_side_damage crop
    # classifier. Defaults on but is a no-op until objects actually carry a
    # condition, so it costs nothing on datasets annotated before the field
    # existed.
    include_condition_crops: bool = True
    # Immutable content-addressed snapshot (M2) is the default: build plan
    # Phase 5 makes every export a named snapshot rather than an overwrite of a
    # mutable directory. `false` keeps the old in-place write for a quick look.
    as_snapshot: bool = True
    # Opt-in upload to the staging object store. Never fails the export - the
    # local snapshot is already written and recorded by the time this runs, so a
    # publish failure is reported, not raised (see object_store).
    publish: bool = False


class SnapshotInfo(BaseModel):
    """One immutable dataset snapshot (see db_models.DatasetSnapshot).

    `snapshot_id` is a content hash over the file set plus the class map, so it
    is globally unique and stable: re-exporting unchanged data resolves to the
    same id rather than creating another snapshot."""

    snapshot_id: str
    dataset_view: str
    class_map_version: Optional[int] = None
    class_map_hash: Optional[str] = None
    local_path: str
    file_count: int
    total_bytes: int
    published_at: Optional[datetime] = None
    published_uri: Optional[str] = None
    created_at: datetime
    last_exported_at: datetime
    created_by: Optional[str] = None
    manifest: dict


class DetectorTrainJobStatus(BaseModel):
    job_id: str
    status: str  # "running" | "completed" | "failed" | "skipped" (M8 GPU-busy deferral, not a failure)
    stage: str = "preparing"  # "preparing" | "waiting_for_gpu" | "training" | "saving" | "done"
    current_epoch: int = 0
    total_epochs: int = 0
    num_images: int = 0
    error: Optional[str] = None
    started_at: float
    updated_at: float


class DetectorInfo(BaseModel):
    active: bool
    version: int = 0
    trained_at: Optional[float] = None
    num_images: int = 0
    num_classes: int = 0
    weights_size: str = ""


class SimilarNeighbor(BaseModel):
    image_id: str
    file_name: str
    similarity: float


class IdentifyRequest(BaseModel):
    name: str


class AnnotatorIdentity(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    role: Optional[str] = None


class SetRoleRequest(BaseModel):
    role: str


class TriageItem(BaseModel):
    image_id: str
    file_name: str
    tier: str
    score: float = 0.0


class TriageQueue(BaseModel):
    """Phase 2 triage (see annotation_module_build_plan.md §2 Phase 2).
    field_flagged/gate_recall_audit_miss are always empty here - blocked on
    Q-E (pipeline return-path format) and an actual pipeline connection,
    neither of which exist yet. Kept in the response shape (not omitted) so
    a future pipeline integration only has to populate them, not add a
    field the frontend doesn't already know about."""

    field_flagged: list[TriageItem] = Field(default_factory=list)
    gate_recall_audit_miss: list[TriageItem] = Field(default_factory=list)
    low_confidence: list[TriageItem] = Field(default_factory=list)
    # Frames whose objects carry no detector confidence at all - boxes read
    # straight from a YOLO label file, or anything annotated before
    # detector/mask confidence were split apart (see AnnotationObject). Not a
    # priority tier from the build plan; a visibility tier, so "we have no
    # signal on these" reads as a stated fact instead of silent absence from
    # the low_confidence tier.
    no_confidence_signal: list[TriageItem] = Field(default_factory=list)
    novel: list[TriageItem] = Field(default_factory=list)
    routine: list[TriageItem] = Field(default_factory=list)


class SubmitReviewRequest(BaseModel):
    decision: str  # "approved" | "rejected"
    reason: str = "second_review"  # "second_review" | "audit_sample"
    notes: Optional[str] = None


class ClassAuditStats(BaseModel):
    class_id: int
    name: str
    safety_critical: bool
    reviewed: int
    approved: int
    rejected: int


class ReviewRecord(BaseModel):
    id: int
    image_id: str
    reviewer_id: int
    reviewer_name: str
    decision: str
    reason: str
    notes: Optional[str] = None
    created_at: datetime


class SimilarityIndexStatus(BaseModel):
    job_id: str
    total: int
    processed: int
    status: str  # "running" | "completed" | "failed"
    started_at: float
    updated_at: float
    indexed_images: int = 0


class CreateGoldenSetRequest(BaseModel):
    # No dataset_view field: unlike ExportRequest, this always targets
    # whichever dataset the caller currently has loaded, the same way
    # export.py derives self._ds.dataset_key server-side rather than
    # trusting a client-supplied identifier. dataset_key is a resolved
    # absolute filesystem path (see DatasetService), not the short view name
    # ("side_view") a client would naturally type here - accepting it as
    # free text invites exactly the silent-mismatch bug this avoids.
    description: Optional[str] = None


class AddGoldenItemsRequest(BaseModel):
    image_ids: list[str]


class GoldenSetInfo(BaseModel):
    id: int
    dataset_view: str
    version: int
    description: Optional[str] = None
    created_at: datetime
    created_by: Optional[str] = None
    item_count: int = 0


class GoldenItemResult(BaseModel):
    image_id: str
    added: bool
    frozen: bool
    error: Optional[str] = None


class ModelPromotionInfo(BaseModel):
    id: int
    dataset_view: str
    mlflow_model_name: str
    mlflow_version: str
    mlflow_run_id: str
    promotion_recommendation: str
    regressed_classes: Optional[str] = None
    status: str
    local_weights_path: Optional[str] = None
    created_at: datetime
    decided_at: Optional[datetime] = None
    decided_by: Optional[str] = None
