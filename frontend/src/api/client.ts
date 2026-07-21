import axios from "axios";
import type {
  AnnotationObject,
  BatchJobStatus,
  ClassInfo,
  DatasetInfo,
  GenerateMaskResponse,
  ImageAnnotations,
  ImageListItem,
  Point,
} from "../types";

const api = axios.create({ baseURL: "/api", timeout: 60_000 });

export const DatasetAPI = {
  load: (datasetPath: string) => api.post<DatasetInfo>("/dataset/load", { dataset_path: datasetPath }).then((r) => r.data),
  info: () => api.get<DatasetInfo>("/dataset/info").then((r) => r.data),
  classes: () => api.get<ClassInfo[]>("/dataset/classes").then((r) => r.data),
  setClassColor: (classId: number, color: string) =>
    api.put(`/dataset/classes/${classId}/color`, { color }).then((r) => r.data),
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
    positivePoints?: Point[];
    negativePoints?: Point[];
  }) =>
    api
      .post<GenerateMaskResponse>("/generate-mask", {
        image_id: params.imageId,
        object_id: params.objectId,
        bbox: params.bbox,
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

export default api;
