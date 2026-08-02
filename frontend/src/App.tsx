import { useEffect } from "react";
import toast from "react-hot-toast";
import AnnotationCanvas from "./components/Canvas/AnnotationCanvas";
import DatasetBrowser from "./components/DatasetBrowser/DatasetBrowser";
import Sidebar from "./components/Sidebar/Sidebar";
import StatusBar from "./components/StatusBar/StatusBar";
import Toolbar from "./components/Toolbar/Toolbar";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { useAnnotationStore } from "./store/annotationStore";
import { useDatasetStore } from "./store/datasetStore";

const LAST_DATASET_KEY = "railway-annotator:last-dataset-path";
const LAST_VIEW_KEY = "railway-annotator:last-dataset-view";

export default function App() {
  const { info, images, currentIndex, currentView, loadDataset, loadCurrent, loadViews, switchView } =
    useDatasetStore();
  const loadImage = useAnnotationStore((s) => s.loadImage);
  const imageLoading = useAnnotationStore((s) => s.loading);
  const generatingAll = useAnnotationStore((s) => s.generatingAll);

  useKeyboardShortcuts();

  // Each browser tab/session has its own currently-loaded dataset view on
  // the backend (see api/client.ts's X-Session-Id header), so a brand-new
  // session never has anything loaded yet - unlike before, when the
  // backend's single auto-loaded dataset was shared by everyone. Restore,
  // in order: this session's own last-used view, this browser's last raw
  // dataset path (legacy manual-path flow), or default to the "legacy"
  // view; only fall to the manual picker if all of those fail.
  useEffect(() => {
    loadViews().catch(() => {
      /* view list unavailable; dropdown just won't show options */
    });
    loadCurrent().then((current) => {
      if (current) return;
      const lastView = localStorage.getItem(LAST_VIEW_KEY);
      const lastPath = localStorage.getItem(LAST_DATASET_KEY);
      if (lastPath) {
        loadDataset(lastPath).catch(() => {
          /* dataset may have moved; user can load manually */
        });
        return;
      }
      switchView(lastView || "legacy").catch(() => {
        /* no views configured yet; user can load manually */
      });
    });
  }, [loadDataset, loadCurrent, loadViews, switchView]);

  useEffect(() => {
    if (info) localStorage.setItem(LAST_DATASET_KEY, info.dataset_path);
    if (currentView) localStorage.setItem(LAST_VIEW_KEY, currentView);
  }, [info, currentView]);

  useEffect(() => {
    const current = images[currentIndex];
    if (current) {
      loadImage(current.image_id).catch(() => toast.error(`Failed to load ${current.file_name}`));
    }
  }, [images, currentIndex, loadImage]);

  if (!info) {
    return <DatasetBrowser />;
  }

  return (
    <div className="flex h-screen w-screen flex-col bg-surface-950 text-gray-100">
      <Toolbar />
      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1">
          <AnnotationCanvas />
          {(imageLoading || generatingAll) && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/40">
              <span className="rounded-md bg-surface-800 px-4 py-2 text-sm text-gray-200">
                {generatingAll ? "Running SAM2 on all boxes..." : "Loading image..."}
              </span>
            </div>
          )}
        </div>
        <Sidebar />
      </div>
      <StatusBar />
    </div>
  );
}
