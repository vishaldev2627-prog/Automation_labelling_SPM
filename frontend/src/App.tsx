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

export default function App() {
  const { info, images, currentIndex, loadDataset } = useDatasetStore();
  const loadImage = useAnnotationStore((s) => s.loadImage);
  const imageLoading = useAnnotationStore((s) => s.loading);
  const generatingAll = useAnnotationStore((s) => s.generatingAll);

  useKeyboardShortcuts();

  // Crash recovery: auto-reload the last opened dataset on startup.
  useEffect(() => {
    const lastPath = localStorage.getItem(LAST_DATASET_KEY);
    if (lastPath) {
      loadDataset(lastPath).catch(() => {
        /* dataset may have moved; user can load manually */
      });
    }
  }, [loadDataset]);

  useEffect(() => {
    if (info) localStorage.setItem(LAST_DATASET_KEY, info.dataset_path);
  }, [info]);

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
