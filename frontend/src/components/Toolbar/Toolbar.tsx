import { useState } from "react";
import toast from "react-hot-toast";
import { BatchAPI, ExportAPI } from "../../api/client";
import { useAnnotationStore } from "../../store/annotationStore";
import { useDatasetStore } from "../../store/datasetStore";
import { useSettingsStore } from "../../store/settingsStore";
import type { ToolMode } from "../../types";

const TOOLS: { mode: ToolMode; label: string; icon: string }[] = [
  { mode: "select", label: "Select / Edit", icon: "🖱" },
  { mode: "draw-box", label: "Draw new box (B) — label something the detector missed", icon: "▭" },
  { mode: "positive-click", label: "Positive point", icon: "➕" },
  { mode: "negative-click", label: "Negative point", icon: "➖" },
  { mode: "pan", label: "Pan (space)", icon: "✋" },
];

export default function Toolbar() {
  const { images, currentIndex, next, prev, info, refreshInfo, refreshImages, markImageCompleted } = useDatasetStore();
  const {
    objects,
    imageId,
    undo,
    redo,
    saveNow,
    generateAllMasks,
    generatingAll,
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
      <button className="toolbar-btn" onClick={() => generateAllMasks()} disabled={generatingAll} title="Regenerate all masks for this image">
        {generatingAll ? "Generating..." : "🪄 Generate all"}
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
