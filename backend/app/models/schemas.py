"""Pydantic models shared across the API."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ObjectStatus(str, Enum):
    PENDING = "pending"
    AUTO_GENERATED = "auto_generated"
    EDITED = "edited"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


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
    """A single annotated object instance within an image."""

    id: str
    class_id: int
    class_name: str = ""
    bbox: BoundingBox
    polygon: list[Point] = Field(default_factory=list)
    confidence: float = 0.0
    all_mask_scores: list[float] = Field(default_factory=list)
    selected_mask_index: int = 0
    status: ObjectStatus = ObjectStatus.PENDING
    visible: bool = True


class ImageAnnotations(BaseModel):
    image_id: str
    file_name: str
    width: int
    height: int
    objects: list[AnnotationObject] = Field(default_factory=list)
    completed: bool = False
    last_modified: Optional[float] = None


class GenerateMaskRequest(BaseModel):
    image_id: str
    object_id: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    positive_points: list[Point] = Field(default_factory=list)
    negative_points: list[Point] = Field(default_factory=list)


class GenerateMaskResponse(BaseModel):
    object_id: str
    polygon: list[Point]
    confidence: float
    all_scores: list[float]
    selected_mask_index: int


class GenerateAllRequest(BaseModel):
    image_id: str


class SaveAnnotationRequest(BaseModel):
    image_id: str
    objects: list[AnnotationObject]
    mark_completed: bool = False


class DatasetInfo(BaseModel):
    dataset_path: str
    total_images: int
    completed: int
    remaining: int
    percent_complete: float
    classes: list[str]
    estimated_seconds_remaining: Optional[float] = None


class ImageListItem(BaseModel):
    image_id: str
    file_name: str
    completed: bool
    object_count: int


class ClassInfo(BaseModel):
    class_id: int
    name: str
    color: str


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
