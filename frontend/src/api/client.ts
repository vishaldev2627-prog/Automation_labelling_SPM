import axios from "axios";
import type {
  AnnotationObject,
  BatchJobStatus,
  ClassInfo,
  DatasetInfo,
  DatasetView,
  DetectorInfo,
  DetectorTrainJobStatus,
  GenerateMaskResponse,
  ImageAnnotations,
  ImageListItem,
  Point,
} from "../types";

const SESSION_ID_KEY = "railway-annotator:session-id";

function randomUUID(): string {
  // crypto.randomUUID() only exists in secure contexts (https, or plain
  // http on "localhost") - it throws when the tool is reached over plain
  // HTTP by IP/hostname (e.g. an internal deployment address), which would
  // otherwise crash the app before it renders anything. Fall back to a
  // non-cryptographic UUID v4 in that case; this id only needs to be
  // unique per browser tab, not unguessable.
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

function getOrCreateSessionId(): string {
  let id = localStorage.getItem(SESSION_ID_KEY);
  if (!id) {
    id = randomUUID();
    localStorage.setItem(SESSION_ID_KEY, id);
  }
  return id;
}

const sessionId = getOrCreateSessionId();

// Set as a cookie, not just a header: plain <img src="..."> requests (how
// AnnotationCanvas/ImagesAPI.fileUrl actually fetches image bytes) never
// carry axios's default headers, only whatever the browser sends
// automatically - a cookie is sent on every same-origin request regardless
// of how it was initiated, so it's what makes the backend's per-session
// dataset view (see app/session_context.py) apply consistently to both the
// JSON API calls and the raw image loads.
document.cookie = `session_id=${sessionId}; path=/; SameSite=Lax; max-age=31536000`;

const api = axios.create({ baseURL: "/api", timeout: 60_000 });
// Kept alongside the cookie for any non-browser API consumers (scripts,
// curl) that don't carry cookies but can set a header.
api.defaults.headers.common["X-Session-Id"] = sessionId;

type AnnotatorIdentity = { id: number | null; name: string | null; role: string | null };

export const AnnotatorAPI = {
  me: () => api.get<AnnotatorIdentity>("/annotator/me").then((r) => r.data),
  identify: (name: string) => api.post<AnnotatorIdentity>("/annotator/identify", { name }).then((r) => r.data),
  list: () => api.get<AnnotatorIdentity[]>("/annotator").then((r) => r.data),
  setRole: (annotatorId: number, role: string) =>
    api.put<AnnotatorIdentity>(`/annotator/${annotatorId}/role`, { role }).then((r) => r.data),
};

export const DatasetAPI = {
  load: (datasetPath: string) => api.post<DatasetInfo>("/dataset/load", { dataset_path: datasetPath }).then((r) => r.data),
  info: () => api.get<DatasetInfo>("/dataset/info").then((r) => r.data),
  classes: () => api.get<ClassInfo[]>("/dataset/classes").then((r) => r.data),
  setClassColor: (classId: number, color: string) =>
    api.put(`/dataset/classes/${classId}/color`, { color }).then((r) => r.data),
  addClass: (name: string) => api.post<ClassInfo>("/dataset/classes", { name }).then((r) => r.data),
  views: () => api.get<DatasetView[]>("/dataset/views").then((r) => r.data),
  switchView: (view: string) => api.post<DatasetInfo>("/dataset/switch", { view }).then((r) => r.data),
};

export const ImagesAPI = {
  list: () => api.get<ImageListItem[]>("/images").then((r) => r.data),
  fileUrl: (imageId: string) => `/api/images/${imageId}/file`,
  getAnnotations: (imageId: string) =>
    api.get<ImageAnnotations>(`/images/${imageId}/annotations`).then((r) => r.data),
  saveAnnotations: (imageId: string, objects: AnnotationObject[], markCompleted: boolean) =>
    api
      .post<ImageAnnotations>("/images/annotations/save", {
        image_id: imageId,
        objects,
        mark_completed: markCompleted,
      })
      .then((r) => r.data),
};

export const MaskAPI = {
  samStatus: () => api.get("/sam/status").then((r) => r.data),
  generateMask: (params: {
    imageId: string;
    objectId?: string;
    bbox?: AnnotationObject["bbox"];
    classId?: number;
    positivePoints?: Point[];
    negativePoints?: Point[];
  }) =>
    api
      .post<GenerateMaskResponse>("/generate-mask", {
        image_id: params.imageId,
        object_id: params.objectId,
        bbox: params.bbox,
        class_id: params.classId,
        positive_points: params.positivePoints ?? [],
        negative_points: params.negativePoints ?? [],
      })
      .then((r) => r.data),
  generateAll: (imageId: string) =>
    api.post<ImageAnnotations>("/generate-all", { image_id: imageId }).then((r) => r.data),
  selectMask: (imageId: string, objectId: string, maskIndex: number) =>
    api
      .post<GenerateMaskResponse>(`/select-mask/${imageId}/${objectId}`, { mask_index: maskIndex })
      .then((r) => r.data),
};

export const BatchAPI = {
  start: (imageIds: string[], overwrite: boolean) =>
    api.post<BatchJobStatus>("/batch-process", { image_ids: imageIds, overwrite }).then((r) => r.data),
  status: (jobId: string) => api.get<BatchJobStatus>(`/batch-process/${jobId}`).then((r) => r.data),
};

export const ExportAPI = {
  export: (imageIds: string[], onlyCompleted: boolean) =>
    api.post("/export", { image_ids: imageIds, only_completed: onlyCompleted }).then((r) => r.data),
};

export const ProgressAPI = {
  get: () => api.get<DatasetInfo>("/progress").then((r) => r.data),
};

export const DetectorAPI = {
  train: () => api.post<DetectorTrainJobStatus>("/detector/train").then((r) => r.data),
  status: (jobId: string) => api.get<DetectorTrainJobStatus>(`/detector/train/${jobId}`).then((r) => r.data),
  active: () => api.get<DetectorInfo>("/detector/active").then((r) => r.data),
};

export default api;
