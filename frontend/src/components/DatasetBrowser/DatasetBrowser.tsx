import { useState } from "react";
import toast from "react-hot-toast";
import { useDatasetStore } from "../../store/datasetStore";

export default function DatasetBrowser() {
  const [path, setPath] = useState("");
  const { loadDataset, loading, info } = useDatasetStore();

  const handleLoad = async () => {
    if (!path.trim()) {
      toast.error("Enter a dataset folder path");
      return;
    }
    try {
      await toast.promise(loadDataset(path.trim()), {
        loading: "Scanning dataset...",
        success: (i) => `Loaded ${i.total_images} images, ${i.classes.length} classes`,
        error: (err) => err?.response?.data?.detail ?? "Failed to load dataset",
      });
    } catch {
      /* toast already shown */
    }
  };

  if (info) return null;

  return (
    <div className="flex h-full w-full items-center justify-center bg-surface-950">
      <div className="w-full max-w-lg rounded-lg border border-surface-700 bg-surface-900 p-8 shadow-xl">
        <h1 className="mb-1 text-xl font-semibold text-gray-100">Railway Segmentation Annotator</h1>
        <p className="mb-6 text-sm text-gray-400">
          Point to a dataset folder containing <code className="text-accent-500">images/</code> and{" "}
          <code className="text-accent-500">labels/</code> subfolders with YOLO detection labels.
        </p>
        <div className="flex gap-2">
          <input
            className="flex-1 rounded-md border border-surface-600 bg-surface-800 px-3 py-2 text-sm text-gray-100 outline-none focus:border-accent-500"
            placeholder="/path/to/dataset"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleLoad()}
          />
          <button
            className="rounded-md bg-accent-600 px-4 py-2 text-sm font-medium text-white hover:bg-accent-500 disabled:opacity-50"
            onClick={handleLoad}
            disabled={loading}
          >
            {loading ? "Loading..." : "Load"}
          </button>
        </div>
      </div>
    </div>
  );
}
