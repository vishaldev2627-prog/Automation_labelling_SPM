import { useEffect, useState } from "react";
import toast from "react-hot-toast";
import { DatasetAPI } from "../../api/client";
import { useDatasetStore } from "../../store/datasetStore";
import { useSettingsStore } from "../../store/settingsStore";
import type { ClassMapVersionInfo } from "../../types";

export default function ClassPanel() {
  const classes = useDatasetStore((s) => s.classes);
  const setClassColor = useDatasetStore((s) => s.setClassColor);
  const setClassSafetyCritical = useDatasetStore((s) => s.setClassSafetyCritical);
  const setClassFineStructure = useDatasetStore((s) => s.setClassFineStructure);
  const addClass = useDatasetStore((s) => s.addClass);
  const { hiddenClassIds, toggleClassVisibility, activeClassId, setActiveClassId } = useSettingsStore();
  const [newClassName, setNewClassName] = useState("");
  const [adding, setAdding] = useState(false);
  const [classMap, setClassMap] = useState<ClassMapVersionInfo | null>(null);

  // Keyed on the class list rather than fetched once: adding a class mints a new
  // version, and switching dataset view changes both the classes and the map, so
  // this stays correct without threading the map through the store's five
  // load/switch/reload paths.
  useEffect(() => {
    let cancelled = false;
    DatasetAPI.classMap()
      .then((m) => {
        if (!cancelled) setClassMap(m);
      })
      .catch(() => {
        // No dataset loaded, or no version recorded yet (the backend logs why).
        // The badge simply doesn't render - it's provenance, not a control.
        if (!cancelled) setClassMap(null);
      });
    return () => {
      cancelled = true;
    };
  }, [classes]);

  const handleAddClass = async () => {
    const name = newClassName.trim();
    if (!name) return;
    setAdding(true);
    try {
      const created = await addClass(name);
      setNewClassName("");
      setActiveClassId(created.class_id);
      toast.success(`Added class "${created.name}"`);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Failed to add class");
    } finally {
      setAdding(false);
    }
  };

  return (
    <div className="max-h-64 overflow-y-auto border-t border-surface-700 p-2">
      <div className="mb-1 flex items-baseline justify-between px-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          Classes ({classes.length})
        </h3>
        {/* The class map is versioned and immutable — adding or renaming a class
            mints a new version rather than editing in place. Surfaced here so
            an annotator can tell at a glance which map they're labelling
            against, since a mismatch between the map a snapshot pins and the
            map a model trained on is a corrupted run that looks normal. */}
        {classMap && (
          <span
            className="cursor-help text-[10px] font-medium text-gray-500"
            title={`Class-map version ${classMap.version}\n${classMap.content_hash}\nExcluded (never labelled or shipped): ${
              classMap.exclude_classes.join(", ") || "none"
            }`}
          >
            map v{classMap.version}
          </span>
        )}
      </div>
      <p className="mb-2 px-2 text-[11px] text-gray-500">
        Click a class, then drag a box on the image (tool: ▭) to label something the detector missed.
      </p>
      <ul className="space-y-0.5">
        {classes.map((c) => (
          <li
            key={c.class_id}
            className={`flex cursor-pointer items-center gap-2 rounded px-2 py-1 text-sm hover:bg-surface-800 ${
              activeClassId === c.class_id ? "bg-accent-600/20 ring-1 ring-accent-500" : ""
            }`}
            onClick={() => setActiveClassId(c.class_id)}
            title="Set as active class for new boxes"
          >
            <input
              type="color"
              value={c.color}
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => setClassColor(c.class_id, e.target.value)}
              className="h-4 w-4 cursor-pointer rounded border-none bg-transparent p-0"
              title="Change class color"
            />
            <span className={`flex-1 truncate ${hiddenClassIds.has(c.class_id) ? "text-gray-600 line-through" : "text-gray-300"}`}>
              {c.name}
            </span>
            <button
              // The ⚠ emoji renders with its own built-in color via the
              // emoji font on most platforms, ignoring CSS `color` entirely
              // (it's a multi-color glyph, not a monochrome one that
              // respects currentColor) - conditionally rendering the emoji
              // itself, not just its color class, is what actually toggles
              // the visual state.
              className={c.safety_critical ? "opacity-100" : "opacity-20 hover:opacity-60"}
              onClick={(e) => {
                e.stopPropagation();
                setClassSafetyCritical(c.class_id, !c.safety_critical);
              }}
              title={
                c.safety_critical
                  ? "Safety-critical: never auto-accepted, always a human, always a second reviewer. Click to unmark."
                  : "Not marked safety-critical. Click to mark - this class will never be eligible for auto-accept."
              }
            >
              ⚠
            </button>
            <button
              // Same conditional-opacity trick as ⚠ above: multi-color emoji
              // glyphs ignore CSS `color`.
              className={c.fine_structure ? "opacity-100" : "opacity-20 hover:opacity-60"}
              onClick={(e) => {
                e.stopPropagation();
                setClassFineStructure(c.class_id, !c.fine_structure);
              }}
              title={
                c.fine_structure
                  ? "Fine structure: every mask contour kept, no polygon simplification, and a binary mask raster exported. For thin/branching defects scored on length (crack, corrosion, shelling). Click to unmark."
                  : "Not marked fine structure. Click to mark — use for thin/branching defects where simplification and dropped contours would cost measured length."
              }
            >
              🧵
            </button>
            <button
              className="text-gray-500 hover:text-gray-200"
              onClick={(e) => {
                e.stopPropagation();
                toggleClassVisibility(c.class_id);
              }}
              title="Toggle class visibility"
            >
              {hiddenClassIds.has(c.class_id) ? "🚫" : "👁"}
            </button>
          </li>
        ))}
      </ul>

      <div className="mt-2 flex items-center gap-1 px-2">
        <input
          type="text"
          value={newClassName}
          onChange={(e) => setNewClassName(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleAddClass();
          }}
          placeholder="New class name..."
          disabled={adding}
          className="min-w-0 flex-1 rounded border border-surface-600 bg-surface-800 px-2 py-1 text-sm text-gray-200 placeholder:text-gray-600"
        />
        <button
          className="toolbar-btn shrink-0"
          onClick={handleAddClass}
          disabled={adding || !newClassName.trim()}
          title="Add a new class missing from this dataset"
        >
          + Add
        </button>
      </div>
    </div>
  );
}
