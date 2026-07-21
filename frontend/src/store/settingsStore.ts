import { create } from "zustand";
import type { ToolMode } from "../types";

interface SettingsState {
  showBoundingBox: boolean;
  showMask: boolean;
  showPolygon: boolean;
  showImage: boolean;
  maskOpacity: number;
  darkMode: boolean;
  toolMode: ToolMode;
  hiddenClassIds: Set<number>;
  activeClassId: number | null;
  zoom: number;
  panX: number;
  panY: number;

  toggleBoundingBox: () => void;
  toggleMask: () => void;
  togglePolygon: () => void;
  toggleImage: () => void;
  setMaskOpacity: (v: number) => void;
  toggleDarkMode: () => void;
  setToolMode: (mode: ToolMode) => void;
  toggleClassVisibility: (classId: number) => void;
  setActiveClassId: (classId: number | null) => void;
  setZoom: (zoom: number) => void;
  setPan: (x: number, y: number) => void;
  resetView: () => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  showBoundingBox: true,
  showMask: true,
  showPolygon: true,
  showImage: true,
  maskOpacity: 0.45,
  darkMode: true,
  toolMode: "select",
  hiddenClassIds: new Set(),
  activeClassId: null,
  zoom: 1,
  panX: 0,
  panY: 0,

  toggleBoundingBox: () => set((s) => ({ showBoundingBox: !s.showBoundingBox })),
  toggleMask: () => set((s) => ({ showMask: !s.showMask })),
  togglePolygon: () => set((s) => ({ showPolygon: !s.showPolygon })),
  toggleImage: () => set((s) => ({ showImage: !s.showImage })),
  setMaskOpacity: (v: number) => set({ maskOpacity: v }),
  toggleDarkMode: () =>
    set((s) => {
      const next = !s.darkMode;
      document.documentElement.classList.toggle("dark", next);
      return { darkMode: next };
    }),
  setToolMode: (mode: ToolMode) => set({ toolMode: mode }),
  toggleClassVisibility: (classId: number) =>
    set((s) => {
      const next = new Set(s.hiddenClassIds);
      if (next.has(classId)) next.delete(classId);
      else next.add(classId);
      return { hiddenClassIds: next };
    }),
  setActiveClassId: (classId: number | null) => set({ activeClassId: classId }),
  setZoom: (zoom: number) => set({ zoom: Math.min(8, Math.max(0.1, zoom)) }),
  setPan: (x: number, y: number) => set({ panX: x, panY: y }),
  resetView: () => set({ zoom: 1, panX: 0, panY: 0 }),
}));
