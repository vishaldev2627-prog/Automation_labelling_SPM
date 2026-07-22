import { create } from "zustand";
import { ImagesAPI, MaskAPI } from "../api/client";
import type { AnnotationObject, BoundingBox, ImageAnnotations, Point } from "../types";

const MAX_HISTORY = 50;
const AUTOSAVE_DEBOUNCE_MS = 800;

interface AnnotationState {
  imageId: string | null;
  imageWidth: number;
  imageHeight: number;
  objects: AnnotationObject[];
  completed: boolean;
  selectedObjectId: string | null;
  loading: boolean;
  saving: boolean;
  saveError: string | null;
  generatingAll: boolean;

  past: AnnotationObject[][];
  future: AnnotationObject[][];

  pendingPositivePoints: Point[];
  pendingNegativePoints: Point[];
  addPendingPoint: (point: Point, positive: boolean) => void;
  clearPendingPoints: () => void;

  loadImage: (imageId: string) => Promise<void>;
  generateAllMasks: () => Promise<void>;
  regenerateObject: (objectId: string) => Promise<void>;
  refineWithPoints: (objectId: string, positive: Point[], negative: Point[]) => Promise<void>;
  selectMaskCandidate: (objectId: string, maskIndex: number) => Promise<void>;
  createObjectFromBox: (bbox: BoundingBox, classId: number, className: string) => Promise<void>;

  setObjects: (objects: AnnotationObject[], pushHistory?: boolean) => void;
  updateObject: (objectId: string, updater: (obj: AnnotationObject) => AnnotationObject) => void;
  addObject: (obj: AnnotationObject) => void;
  deleteObject: (objectId: string) => void;
  selectObject: (objectId: string | null) => void;
  toggleVisibility: (objectId: string) => void;

  undo: () => void;
  redo: () => void;

  saveNow: (markCompleted?: boolean) => Promise<void>;
  scheduleAutosave: () => void;
}

let autosaveTimer: ReturnType<typeof setTimeout> | null = null;

export const useAnnotationStore = create<AnnotationState>((set, get) => ({
  imageId: null,
  imageWidth: 0,
  imageHeight: 0,
  objects: [],
  completed: false,
  selectedObjectId: null,
  loading: false,
  saving: false,
  saveError: null,
  generatingAll: false,
  past: [],
  future: [],
  pendingPositivePoints: [],
  pendingNegativePoints: [],

  addPendingPoint: (point: Point, positive: boolean) =>
    set((s) =>
      positive
        ? { pendingPositivePoints: [...s.pendingPositivePoints, point] }
        : { pendingNegativePoints: [...s.pendingNegativePoints, point] },
    ),
  clearPendingPoints: () => set({ pendingPositivePoints: [], pendingNegativePoints: [] }),

  loadImage: async (imageId: string) => {
    set({ loading: true, selectedObjectId: null, past: [], future: [] });
    try {
      const data: ImageAnnotations = await ImagesAPI.getAnnotations(imageId);
      set({
        imageId: data.image_id,
        imageWidth: data.width,
        imageHeight: data.height,
        objects: data.objects,
        completed: data.completed,
        loading: false,
      });

      const needsGeneration = data.objects.some((o) => o.polygon.length === 0);
      if (needsGeneration) {
        await get().generateAllMasks();
      }
    } catch (err) {
      set({ loading: false });
      throw err;
    }
  },

  generateAllMasks: async () => {
    const { imageId } = get();
    if (!imageId) return;
    set({ generatingAll: true });
    try {
      const data = await MaskAPI.generateAll(imageId);
      set({ objects: data.objects, generatingAll: false });
    } catch (err) {
      set({ generatingAll: false });
      throw err;
    }
  },

  regenerateObject: async (objectId: string) => {
    const { imageId, objects } = get();
    if (!imageId) return;
    const obj = objects.find((o) => o.id === objectId);
    if (!obj) return;
    const result = await MaskAPI.generateMask({ imageId, objectId, bbox: obj.bbox });
    get().updateObject(objectId, (o) => ({
      ...o,
      polygon: result.polygon,
      confidence: result.confidence,
      all_mask_scores: result.all_scores,
      selected_mask_index: result.selected_mask_index,
      status: "auto_generated",
    }));
  },

  refineWithPoints: async (objectId: string, positive: Point[], negative: Point[]) => {
    const { imageId, objects } = get();
    if (!imageId) return;
    const obj = objects.find((o) => o.id === objectId);
    if (!obj) return;
    const result = await MaskAPI.generateMask({
      imageId,
      objectId,
      bbox: obj.bbox,
      positivePoints: positive,
      negativePoints: negative,
    });
    get().updateObject(objectId, (o) => ({
      ...o,
      polygon: result.polygon,
      confidence: result.confidence,
      all_mask_scores: result.all_scores,
      selected_mask_index: result.selected_mask_index,
      status: "edited",
    }));
  },

  createObjectFromBox: async (bbox: BoundingBox, classId: number, className: string) => {
    const { imageId } = get();
    if (!imageId) return;
    const result = await MaskAPI.generateMask({ imageId, bbox, classId });
    get().addObject({
      id: result.object_id,
      class_id: classId,
      class_name: className,
      bbox,
      polygon: result.polygon,
      confidence: result.confidence,
      all_mask_scores: result.all_scores,
      selected_mask_index: result.selected_mask_index,
      status: "auto_generated",
      visible: true,
      source: "manual",
      propagated_from_image_id: null,
    });
  },

  selectMaskCandidate: async (objectId: string, maskIndex: number) => {
    const { imageId } = get();
    if (!imageId) return;
    const result = await MaskAPI.selectMask(imageId, objectId, maskIndex);
    get().updateObject(objectId, (o) => ({
      ...o,
      polygon: result.polygon,
      confidence: result.confidence,
      selected_mask_index: result.selected_mask_index,
    }));
  },

  setObjects: (objects: AnnotationObject[], pushHistory = true) => {
    set((state) => {
      const past = pushHistory ? [...state.past, state.objects].slice(-MAX_HISTORY) : state.past;
      return { objects, past, future: pushHistory ? [] : state.future };
    });
    get().scheduleAutosave();
  },

  updateObject: (objectId: string, updater) => {
    const { objects } = get();
    const next = objects.map((o) => (o.id === objectId ? updater(o) : o));
    get().setObjects(next, true);
  },

  addObject: (obj: AnnotationObject) => {
    const { objects } = get();
    get().setObjects([...objects, obj], true);
    set({ selectedObjectId: obj.id });
  },

  deleteObject: (objectId: string) => {
    const { objects, selectedObjectId } = get();
    get().setObjects(
      objects.map((o) => (o.id === objectId ? { ...o, status: "rejected" as const, visible: false } : o)),
      true,
    );
    if (selectedObjectId === objectId) set({ selectedObjectId: null });
  },

  selectObject: (objectId: string | null) => set({ selectedObjectId: objectId }),

  toggleVisibility: (objectId: string) => {
    const { objects } = get();
    get().setObjects(
      objects.map((o) => (o.id === objectId ? { ...o, visible: !o.visible } : o)),
      false,
    );
  },

  undo: () => {
    const { past, objects, future } = get();
    if (past.length === 0) return;
    const previous = past[past.length - 1];
    set({
      objects: previous,
      past: past.slice(0, -1),
      future: [objects, ...future].slice(0, MAX_HISTORY),
    });
    get().scheduleAutosave();
  },

  redo: () => {
    const { future, objects, past } = get();
    if (future.length === 0) return;
    const nextState = future[0];
    set({
      objects: nextState,
      future: future.slice(1),
      past: [...past, objects].slice(-MAX_HISTORY),
    });
    get().scheduleAutosave();
  },

  saveNow: async (markCompleted?: boolean) => {
    const { imageId, objects, completed } = get();
    if (!imageId) return;
    set({ saving: true, saveError: null });
    try {
      const result = await ImagesAPI.saveAnnotations(imageId, objects, markCompleted ?? completed);
      set({ saving: false, completed: result.completed });
    } catch (err: any) {
      set({ saving: false, saveError: err?.message ?? "Save failed" });
      throw err;
    }
  },

  scheduleAutosave: () => {
    if (autosaveTimer) clearTimeout(autosaveTimer);
    autosaveTimer = setTimeout(() => {
      get().saveNow().catch(() => {
        /* surfaced via saveError in state */
      });
    }, AUTOSAVE_DEBOUNCE_MS);
  },
}));
