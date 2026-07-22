import { useEffect, useState } from "react";
import toast from "react-hot-toast";
import { BatchAPI, DetectorAPI, ExportAPI } from "../../api/client";
import { useAnnotationStore } from "../../store/annotationStore";
import { useDatasetStore } from "../../store/datasetStore";
import { useSettingsStore } from "../../store/settingsStore";
import type { DetectorInfo, ToolMode } from "../../types";

const TOOLS: { mode: ToolMode; label: string; icon: string }[] = [
  { mode: "select", label: "Select / Edit", icon: "🖱" },
  { mode: "draw-box", label: "Draw new box (B) — label something the detector missed", icon: "▭" },
  { mode: "positive-click", label: "Positive point", icon: "➕" },
  { mode: "negative-click", label: "Negative point", icon: "➖" },
  { mode: "pan", label: "Pan (space)", icon: "✋" },
];

export default function Toolbar() {
  const { images, currentIndex, next, prev, setCurrentIndex, info, refreshInfo, refreshImages, markImageCompleted } =
    useDatasetStore();
  const [frameInput, setFrameInput] = useState("");

  const goToFrame = () => {
    const n = parseInt(frameInput, 10);
    if (!Number.isNaN(n) && n >= 1 && n <= images.length) {
      setCurrentIndex(n - 1);
    } else {
      toast.error(`Enter a frame number between 1 and ${images.length}`);
    }
    setFrameInput("");
  };
  const {
    objects,
    imageId,
    undo,
    redo,
    saveNow,
    generateAllMasks,
    generatingAll,
    needsGeneration,
    saving,
    completed,
    pendingPositivePoints,
    pendingNegativePoints,
    clearPendingPoints,
    refineWithPoints,
    selectedObjectId,
  } = useAnnotationStore();
  const { toolMode, setToolMode, showBoundingBox, showMask, showPolygon, showImage, maskOpacity, setMaskOpacity, toggleBoundingBox, toggleMask, togglePolygon, toggleImage, darkMode, toggleDarkMode, activeClassId } =
    useSettingsStore();
  const classes = useDatasetStore((s) => s.classes);
  const [batchRunning, setBatchRunning] = useState(false);
  const [training, setTraining] = useState(false);
  const [detectorInfo, setDetectorInfo] = useState<DetectorInfo | null>(null);

  const refreshDetectorInfo = () => {
    DetectorAPI.active()
      .then(setDetectorInfo)
      .catch(() => {
        /* dataset not loaded yet; ignore */
      });
  };

  useEffect(() => {
    refreshDetectorInfo();
  }, [info?.dataset_path]);

  const handleSave = async () => {
    try {
      await saveNow(true);
      if (imageId) markImageCompleted(imageId, true);
      await refreshInfo();
      toast.success("Saved");
    } catch {
      toast.error("Save failed");
    }
  };

  const handleApplyRefinement = async () => {
    if (!selectedObjectId) {
      toast.error("Select an object first");
      return;
    }
    if (pendingPositivePoints.length === 0 && pendingNegativePoints.length === 0) {
      toast.error("Click on the image to add positive/negative points first");
      return;
    }
    try {
      await refineWithPoints(selectedObjectId, pendingPositivePoints, pendingNegativePoints);
      clearPendingPoints();
      toast.success("Mask refined");
    } catch {
      toast.error("Refinement failed");
    }
  };

  const handleBatchProcess = async () => {
    setBatchRunning(true);
    try {
      const job = await BatchAPI.start([], false);
      toast.loading(`Batch processing started (job ${job.job_id.slice(0, 6)})`, { id: "batch" });
      const poll = async () => {
        const status = await BatchAPI.status(job.job_id);
        toast.loading(`Batch: ${status.processed}/${status.total} (${status.failed} failed)`, { id: "batch" });
        if (status.status === "completed") {
          toast.success(`Batch complete: ${status.processed}/${status.total}`, { id: "batch" });
          setBatchRunning(false);
          refreshImages();
          refreshInfo();
        } else {
          setTimeout(poll, 2000);
        }
      };
      setTimeout(poll, 2000);
    } catch {
      toast.error("Failed to start batch job");
      setBatchRunning(false);
    }
  };

  const handleTrainDetector = async () => {
    setTraining(true);
    try {
      const job = await DetectorAPI.train();
      toast.loading("Retraining detector (preparing dataset)...", { id: "train" });
      const poll = async () => {
        const status = await DetectorAPI.status(job.job_id);
        if (status.status === "running") {
          toast.loading(
            status.stage === "training"
              ? `Retraining: epoch ${status.current_epoch}/${status.total_epochs}`
              : `Retraining: ${status.stage}...`,
            { id: "train" },
          );
          setTimeout(poll, 3000);
        } else if (status.status === "completed") {
          toast.success(`Detector retrained on ${status.num_images} images`, { id: "train" });
          setTraining(false);
          refreshDetectorInfo();
        } else {
          toast.error(status.error ?? "Training failed", { id: "train" });
          setTraining(false);
        }
      };
      setTimeout(poll, 1500);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Failed to start training");
      setTraining(false);
    }
  };

  const handleExport = async () => {
    try {
      const result = await toast.promise(ExportAPI.export([], true), {
        loading: "Exporting...",
        success: (r: any) => `Exported ${r.exported} images to ${r.output_dir}`,
        error: "Export failed",
      });
      return result;
    } catch {
      /* handled by toast */
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-surface-700 bg-surface-900 px-3 py-2">
      <div className="flex items-center gap-1">
        <button className="toolbar-btn" onClick={prev} disabled={currentIndex <= 0} title="Previous (←)">
          ◀
        </button>
        <span className="min-w-[80px] text-center text-sm text-gray-300">
          {images.length > 0 ? `${currentIndex + 1} / ${images.length}` : "-"}
        </span>
        <button className="toolbar-btn" onClick={next} disabled={currentIndex >= images.length - 1} title="Next (→)">
          ▶
        </button>
      </div>

      <div className="flex items-center gap-1">
        <input
          type="number"
          min={1}
          max={images.length || 1}
          value={frameInput}
          onChange={(e) => setFrameInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") goToFrame();
          }}
          placeholder="Go to #"
          className="w-20 rounded border border-surface-600 bg-surface-800 px-2 py-1 text-sm text-gray-200 placeholder:text-gray-500"
          title="Jump to frame number"
        />
        <button className="toolbar-btn" onClick={goToFrame} disabled={images.length === 0} title="Jump to frame">
          Go
        </button>
      </div>

      <div className="mx-2 h-6 w-px bg-surface-600" />

      {TOOLS.map((t) => (
        <button
          key={t.mode}
          className={`toolbar-btn ${toolMode === t.mode ? "bg-accent-600 text-white" : ""}`}
          onClick={() => setToolMode(t.mode)}
          title={t.label}
        >
          {t.icon}
        </button>
      ))}
      {toolMode === "draw-box" && (
        <span className="text-xs text-accent-400">
          {activeClassId != null
            ? `Drawing: ${classes.find((c) => c.class_id === activeClassId)?.name ?? "class_" + activeClassId}`
            : "Pick a class in the sidebar first, then drag on the image"}
        </span>
      )}
      {(pendingPositivePoints.length > 0 || pendingNegativePoints.length > 0) && (
        <>
          <button className="toolbar-btn bg-accent-600 text-white" onClick={handleApplyRefinement}>
            Apply refine ({pendingPositivePoints.length}+/{pendingNegativePoints.length}-)
          </button>
          <button className="toolbar-btn" onClick={clearPendingPoints}>
            Clear
          </button>
        </>
      )}

      <div className="mx-2 h-6 w-px bg-surface-600" />

      <button className="toolbar-btn" onClick={undo} title="Undo (Ctrl+Z)">
        ↶
      </button>
      <button className="toolbar-btn" onClick={redo} title="Redo (Ctrl+Y)">
        ↷
      </button>
      <button
        className={`toolbar-btn ${needsGeneration && !generatingAll ? "animate-pulse bg-amber-600 hover:bg-amber-500" : ""}`}
        onClick={() => generateAllMasks()}
        disabled={generatingAll}
        title={needsGeneration ? "This image has objects without masks yet — click to generate" : "Regenerate all masks for this image"}
      >
        {generatingAll ? "Generating..." : needsGeneration ? "🪄 Generate masks" : "🪄 Generate all"}
      </button>

      <div className="mx-2 h-6 w-px bg-surface-600" />

      <label className="flex items-center gap-1 text-xs text-gray-400">
        <input type="checkbox" checked={showBoundingBox} onChange={toggleBoundingBox} /> BBox
      </label>
      <label className="flex items-center gap-1 text-xs text-gray-400">
        <input type="checkbox" checked={showMask} onChange={toggleMask} /> Mask
      </label>
      <label className="flex items-center gap-1 text-xs text-gray-400">
        <input type="checkbox" checked={showPolygon} onChange={togglePolygon} /> Polygon
      </label>
      <label className="flex items-center gap-1 text-xs text-gray-400">
        <input type="checkbox" checked={showImage} onChange={toggleImage} /> Image
      </label>
      <div className="flex items-center gap-1 text-xs text-gray-400">
        Opacity
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={maskOpacity}
          onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
          className="w-20"
        />
      </div>

      <div className="ml-auto flex items-center gap-2">
        <span className="text-xs text-gray-500">{saving ? "Saving..." : completed ? "Saved ✓" : ""}</span>
        {detectorInfo?.active && (
          <span className="text-xs text-gray-500" title={`Trained on ${detectorInfo.num_images} reviewed images`}>
            Detector v{detectorInfo.version}
          </span>
        )}
        <button
          className="toolbar-btn"
          onClick={handleTrainDetector}
          disabled={training}
          title="Fine-tune a YOLOv8 detector on every image you've reviewed and marked complete, so it auto-boxes new images correctly next time"
        >
          {training ? "Retraining..." : "🎯 Retrain detector"}
        </button>
        <button className="toolbar-btn" onClick={handleBatchProcess} disabled={batchRunning} title="Batch process entire dataset">
          {batchRunning ? "Processing..." : "⚙ Batch process"}
        </button>
        <button className="toolbar-btn" onClick={handleExport} title="Export YOLO segmentation labels">
          ⬇ Export
        </button>
        <button className="toolbar-btn bg-accent-600 text-white" onClick={handleSave} title="Save (Ctrl+S)">
          💾 Save
        </button>
        <button className="toolbar-btn" onClick={toggleDarkMode} title="Toggle theme">
          {darkMode ? "🌙" : "☀"}
        </button>
      </div>
      {objects.length === 0 && info && info.total_images === 0 && (
        <span className="text-xs text-gray-500">No images found in dataset</span>
      )}
    </div>
  );
}
