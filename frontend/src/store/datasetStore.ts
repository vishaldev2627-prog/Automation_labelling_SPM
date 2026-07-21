import { create } from "zustand";
import { DatasetAPI, ImagesAPI } from "../api/client";
import type { ClassInfo, DatasetInfo, ImageListItem } from "../types";

interface DatasetState {
  datasetPath: string;
  info: DatasetInfo | null;
  images: ImageListItem[];
  classes: ClassInfo[];
  currentIndex: number;
  loading: boolean;
  error: string | null;

  loadDataset: (path: string) => Promise<DatasetInfo>;
  refreshInfo: () => Promise<void>;
  refreshImages: () => Promise<void>;
  setCurrentIndex: (index: number) => void;
  next: () => void;
  prev: () => void;
  jumpTo: (imageId: string) => void;
  setClassColor: (classId: number, color: string) => Promise<void>;
  addClass: (name: string) => Promise<ClassInfo>;
  markImageCompleted: (imageId: string, completed: boolean) => void;
}

export const useDatasetStore = create<DatasetState>((set, get) => ({
  datasetPath: "",
  info: null,
  images: [],
  classes: [],
  currentIndex: 0,
  loading: false,
  error: null,

  loadDataset: async (path: string) => {
    set({ loading: true, error: null });
    try {
      const info = await DatasetAPI.load(path);
      const [images, classes] = await Promise.all([ImagesAPI.list(), DatasetAPI.classes()]);
      set({ datasetPath: path, info, images, classes, currentIndex: 0, loading: false });
      return info;
    } catch (err: any) {
      set({ loading: false, error: err?.response?.data?.detail ?? err.message ?? "Failed to load dataset" });
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
