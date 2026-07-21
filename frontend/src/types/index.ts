export interface Point {
  x: number;
  y: number;
}

export interface BoundingBox {
  x_center: number;
  y_center: number;
  width: number;
  height: number;
}

export type ObjectStatus = "pending" | "auto_generated" | "edited" | "confirmed" | "rejected";

export interface AnnotationObject {
  id: string;
  class_id: number;
  class_name: string;
  bbox: BoundingBox;
  polygon: Point[];
  confidence: number;
  all_mask_scores: number[];
  selected_mask_index: number;
  status: ObjectStatus;
  visible: boolean;
}

export interface ImageAnnotations {
  image_id: string;
  file_name: string;
  width: number;
  height: number;
  objects: AnnotationObject[];
  completed: boolean;
  last_modified: number | null;
}

export interface ImageListItem {
  image_id: string;
  file_name: string;
  completed: boolean;
  object_count: number;
}

export interface DatasetInfo {
  dataset_path: string;
  total_images: number;
  completed: number;
  remaining: number;
  percent_complete: number;
  classes: string[];
  estimated_seconds_remaining: number | null;
}

export interface ClassInfo {
  class_id: number;
  name: string;
  color: string;
}

export interface BatchJobStatus {
  job_id: string;
  total: number;
  processed: number;
  failed: number;
  status: string;
  current_image: string | null;
  started_at: number;
  updated_at: number;
}

export interface GenerateMaskResponse {
  object_id: string;
  polygon: Point[];
  confidence: number;
  all_scores: number[];
  selected_mask_index: number;
}

export interface DetectorTrainJobStatus {
  job_id: string;
  status: string;
  stage: string;
  current_epoch: number;
  total_epochs: number;
  num_images: number;
  error: string | null;
  started_at: number;
  updated_at: number;
}

export interface DetectorInfo {
  active: boolean;
  version: number;
  trained_at: number | null;
  num_images: number;
  num_classes: number;
  weights_size: string;
}

export type ToolMode = "select" | "edit-vertex" | "add-point" | "positive-click" | "negative-click" | "draw-box" | "pan";
