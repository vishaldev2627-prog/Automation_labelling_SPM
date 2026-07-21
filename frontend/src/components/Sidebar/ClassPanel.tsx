import { useState } from "react";
import toast from "react-hot-toast";
import { useDatasetStore } from "../../store/datasetStore";
import { useSettingsStore } from "../../store/settingsStore";

export default function ClassPanel() {
  const classes = useDatasetStore((s) => s.classes);
  const setClassColor = useDatasetStore((s) => s.setClassColor);
  const addClass = useDatasetStore((s) => s.addClass);
  const { hiddenClassIds, toggleClassVisibility, activeClassId, setActiveClassId } = useSettingsStore();
  const [newClassName, setNewClassName] = useState("");
  const [adding, setAdding] = useState(false);

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
      <h3 className="mb-1 px-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
        Classes ({classes.length})
      </h3>
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
