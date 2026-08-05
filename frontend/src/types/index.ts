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
export type ObjectSource = "detection_box" | "manual" | "propagated";

/** Component condition — verbatim from pipeline.md §5.2. Orthogonal to
 * class_id: class_id is *what the component is*, this is *what state it's in*.
 * null means unassessed, deliberately NOT "ok" — see the backend's Condition.
 * Note "leaking" is a valid condition and is unrelated to the excluded
 * synthetic `leakage` component class. */
export type Condition =
  | "ok"
  | "broken"
  | "missing"
  | "hanging"
  | "displaced"
  | "dislocated"
  | "leaking"
  | "damaged"
  | "securing_broken"
  | "securing_hanging"
  | "fiba_red"
  | "sparking"
  | "binding";

export const CONDITIONS: Condition[] = [
  "ok",
  "broken",
  "missing",
  "hanging",
  "displaced",
  "dislocated",
  "leaking",
  "damaged",
  "securing_broken",
  "securing_hanging",
  "fiba_red",
  "sparking",
  "binding",
];

export interface AnnotationObject {
  id: string;
  class_id: number;
  class_name: string;
  /** null = unassessed. Not "ok". */
  condition: Condition | null;
  bbox: BoundingBox;
  polygon: Point[];
  /** Extra disjoint pieces of the same object, for fine_structure classes only
   * (crack/corrosion/shelling). `polygon` remains the largest piece so every
   * existing consumer keeps working; this holds the rest. See the backend's
   * AnnotationObject. */
  extra_polygons: Point[][];
  // Two independent signals, deliberately separate (see the backend's
  // AnnotationObject docstring). detector_confidence answers "is the class
  // right" and is null when there is no signal at all - a plain YOLO label
  // file carries no confidence field. mask_confidence answers "is the
  // polygon right" and says nothing about the class.
  detector_confidence: number | null;
  mask_confidence: number;
  all_mask_scores: number[];
  selected_mask_index: number;
  status: ObjectStatus;
  visible: boolean;
  source: ObjectSource;
  propagated_from_image_id: string | null;
}

export interface ImageAnnotations {
  image_id: string;
  file_name: string;
  width: number;
  height: number;
  objects: AnnotationObject[];
  completed: boolean;
  /** A human explicitly asserted this frame is empty, as opposed to it merely
   * having no objects yet. Only a confirmed-empty frame is exported, as a
   * negative/background sample with an empty label file. */
  no_objects_confirmed: boolean;
  coach_type: CoachType;
  last_modified: number | null;
}

/** LHB and ICF only (confirmed 2026-08-05), matching pipeline.md §5.7 and
 * FINAL_AIML §9. "unknown" is not a third type — it's the absence of an
 * answer, and the default, because pipeline.md §8 refuses to pick a manifest
 * for an unknown coach type rather than assuming one. */
export type CoachType = "LHB" | "ICF" | "unknown";

export const COACH_TYPES: { value: CoachType; label: string }[] = [
  { value: "unknown", label: "Unknown" },
  { value: "LHB", label: "LHB" },
  { value: "ICF", label: "ICF" },
];

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
  safety_critical: boolean;
  /** Thin/branching/length-measured defect: all contours kept, no polygon
   * simplification, and a binary mask raster written at export. */
  fine_structure: boolean;
}

/** One immutable class-map version. `content_hash` is what proves two exports
 * used the same map; the version number is for humans. */
export interface ClassMapVersionInfo {
  version: number;
  content_hash: string;
  names: Record<number, string>;
  exclude_classes: string[];
  created_at: string;
  created_by: string | null;
}

export interface DatasetView {
  key: string;
  label: string;
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
  /** Extra disjoint pieces, for fine_structure classes. Returned rather than
   * only persisted server-side: the store holds objects in memory and posts
   * them back on the next autosave, so anything absent here would be dropped. */
  extra_polygons: Point[][];
  /** SAM2 mask score - maps to AnnotationObject.mask_confidence, never to
   * detector_confidence. This endpoint only produces masks. */
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

export interface TriageItem {
  image_id: string;
  file_name: string;
  tier: string;
  score: number;
}

export interface TriageQueue {
  field_flagged: TriageItem[];
  gate_recall_audit_miss: TriageItem[];
  low_confidence: TriageItem[];
  no_confidence_signal: TriageItem[];
  novel: TriageItem[];
  routine: TriageItem[];
}

export type ReviewDecision = "approved" | "rejected";
export type ReviewReason = "second_review" | "audit_sample";

export interface ReviewRecord {
  id: number;
  image_id: string;
  reviewer_id: number;
  reviewer_name: string;
  decision: ReviewDecision;
  reason: ReviewReason;
  notes: string | null;
  created_at: string;
}
