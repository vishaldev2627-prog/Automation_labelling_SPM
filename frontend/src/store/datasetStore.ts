import { create } from "zustand";
import { DatasetAPI, ImagesAPI } from "../api/client";
import type { ClassInfo, DatasetInfo, DatasetView, ImageListItem } from "../types";

interface DatasetState {
  datasetPath: string;
  info: DatasetInfo | null;
  images: ImageListItem[];
  classes: ClassInfo[];
  currentIndex: number;
  loading: boolean;
  error: string | null;
  views: DatasetView[];
  currentView: string | null;

  loadDataset: (path: string) => Promise<DatasetInfo>;
  loadCurrent: () => Promise<DatasetInfo | null>;
  loadViews: () => Promise<DatasetView[]>;
  switchView: (view: string) => Promise<DatasetInfo>;
  reloadCurrent: () => Promise<DatasetInfo>;
  refreshInfo: () => Promise<void>;
  refreshImages: () => Promise<void>;
  setCurrentIndex: (index: number) => void;
  next: () => void;
  prev: () => void;
  jumpTo: (imageId: string) => void;
  setClassColor: (classId: number, color: string) => Promise<void>;
  setClassSafetyCritical: (classId: number, safetyCritical: boolean) => Promise<void>;
  setClassFineStructure: (classId: number, fineStructure: boolean) => Promise<void>;
  addClass: (name: string) => Promise<ClassInfo>;
  markImageCompleted: (imageId: string, completed: boolean) => void;
}

// Dataset views live as sibling subfolders of the dataset root (see backend's
// routers/dataset.py DATASET_VIEWS), so the view key is just the last path
// segment of whatever dataset_path the backend reports.
function viewKeyFromPath(path: string): string | null {
  const segments = path.split("/").filter(Boolean);
  return segments.length ? segments[segments.length - 1] : null;
}

export const useDatasetStore = create<DatasetState>((set, get) => ({
  datasetPath: "",
  info: null,
  images: [],
  classes: [],
  currentIndex: 0,
  loading: false,
  error: null,
  views: [],
  currentView: null,

  loadDataset: async (path: string) => {
    set({ loading: true, error: null });
    try {
      const info = await DatasetAPI.load(path);
      const [images, classes] = await Promise.all([ImagesAPI.list(), DatasetAPI.classes()]);
      set({
        datasetPath: path,
        info,
        images,
        classes,
        currentIndex: 0,
        loading: false,
        currentView: viewKeyFromPath(info.dataset_path),
      });
      return info;
    } catch (err: any) {
      set({ loading: false, error: err?.response?.data?.detail ?? err.message ?? "Failed to load dataset" });
      throw err;
    }
  },

  loadCurrent: async () => {
    try {
      const info = await DatasetAPI.info();
      const [images, classes] = await Promise.all([ImagesAPI.list(), DatasetAPI.classes()]);
      set({
        datasetPath: info.dataset_path,
        info,
        images,
        classes,
        currentIndex: 0,
        currentView: viewKeyFromPath(info.dataset_path),
      });
      return info;
    } catch {
      return null;
    }
  },

  loadViews: async () => {
    const views = await DatasetAPI.views();
    set({ views });
    return views;
  },

  switchView: async (view: string) => {
    set({ loading: true, error: null });
    try {
      const info = await DatasetAPI.switchView(view);
      const [images, classes] = await Promise.all([ImagesAPI.list(), DatasetAPI.classes()]);
      set({
        datasetPath: info.dataset_path,
        info,
        images,
        classes,
        currentIndex: 0,
        loading: false,
        currentView: view,
      });
      return info;
    } catch (err: any) {
      set({ loading: false, error: err?.response?.data?.detail ?? err.message ?? "Failed to switch dataset view" });
      throw err;
    }
  },

  // Re-scans the current view's images/labels folders from disk - the
  // backend only does this on an actual load/switch call, never on its own,
  // so newly added/removed image files won't show up until this (or a view
  // switch) runs. Unlike switchView, this tries to keep you on the same
  // image afterward instead of always jumping back to frame 1.
  reloadCurrent: async () => {
    const { images, currentIndex, currentView, datasetPath } = get();
    const currentImageId = images[currentIndex]?.image_id;
    set({ loading: true, error: null });
    try {
      const info = currentView ? await DatasetAPI.switchView(currentView) : await DatasetAPI.load(datasetPath);
      const [newImages, classes] = await Promise.all([ImagesAPI.list(), DatasetAPI.classes()]);
      const restoredIndex = currentImageId ? newImages.findIndex((i) => i.image_id === currentImageId) : -1;
      set({
        datasetPath: info.dataset_path,
        info,
        images: newImages,
        classes,
        currentIndex: restoredIndex >= 0 ? restoredIndex : 0,
        loading: false,
        currentView: currentView ?? viewKeyFromPath(info.dataset_path),
      });
      return info;
    } catch (err: any) {
      set({ loading: false, error: err?.response?.data?.detail ?? err.message ?? "Failed to reload dataset" });
      throw err;
    }
  },

  refreshInfo: async () => {
    try {
      const info = await DatasetAPI.info();
      set({ info });
    } catch {
      /* dataset not loaded yet; ignore */
    }
  },

  refreshImages: async () => {
    const images = await ImagesAPI.list();
    set({ images });
  },

  setCurrentIndex: (index: number) => {
    const { images } = get();
    if (index < 0 || index >= images.length) return;
    set({ currentIndex: index });
  },

  next: () => {
    const { currentIndex, images } = get();
    if (currentIndex < images.length - 1) set({ currentIndex: currentIndex + 1 });
  },

  prev: () => {
    const { currentIndex } = get();
    if (currentIndex > 0) set({ currentIndex: currentIndex - 1 });
  },

  jumpTo: (imageId: string) => {
    const { images } = get();
    const idx = images.findIndex((i) => i.image_id === imageId);
    if (idx >= 0) set({ currentIndex: idx });
  },

  setClassColor: async (classId: number, color: string) => {
    await DatasetAPI.setClassColor(classId, color);
    set((state) => ({
      classes: state.classes.map((c) => (c.class_id === classId ? { ...c, color } : c)),
    }));
  },

  setClassSafetyCritical: async (classId: number, safetyCritical: boolean) => {
    await DatasetAPI.setClassSafetyCritical(classId, safetyCritical);
    set((state) => ({
      classes: state.classes.map((c) => (c.class_id === classId ? { ...c, safety_critical: safetyCritical } : c)),
    }));
  },

  setClassFineStructure: async (classId: number, fineStructure: boolean) => {
    await DatasetAPI.setClassFineStructure(classId, fineStructure);
    set((state) => ({
      classes: state.classes.map((c) => (c.class_id === classId ? { ...c, fine_structure: fineStructure } : c)),
    }));
  },

  addClass: async (name: string) => {
    const newClass = await DatasetAPI.addClass(name);
    set((state) => ({ classes: [...state.classes, newClass] }));
    return newClass;
  },

  markImageCompleted: (imageId: string, completed: boolean) => {
    set((state) => ({
      images: state.images.map((img) => (img.image_id === imageId ? { ...img, completed } : img)),
    }));
  },
}));
