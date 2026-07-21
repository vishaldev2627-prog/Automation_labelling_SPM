import { useDatasetStore } from "../../store/datasetStore";
import { useSettingsStore } from "../../store/settingsStore";

export default function ClassPanel() {
  const classes = useDatasetStore((s) => s.classes);
  const setClassColor = useDatasetStore((s) => s.setClassColor);
  const { hiddenClassIds, toggleClassVisibility } = useSettingsStore();

  return (
    <div className="max-h-64 overflow-y-auto border-t border-surface-700 p-2">
      <h3 className="mb-2 px-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
        Classes ({classes.length})
      </h3>
      <ul className="space-y-0.5">
        {classes.map((c) => (
          <li key={c.class_id} className="flex items-center gap-2 rounded px-2 py-1 text-sm hover:bg-surface-800">
            <input
              type="color"
              value={c.color}
              onChange={(e) => setClassColor(c.class_id, e.target.value)}
              className="h-4 w-4 cursor-pointer rounded border-none bg-transparent p-0"
              title="Change class color"
            />
            <span className={`flex-1 truncate ${hiddenClassIds.has(c.class_id) ? "text-gray-600 line-through" : "text-gray-300"}`}>
              {c.name}
            </span>
            <button
              className="text-gray-500 hover:text-gray-200"
              onClick={() => toggleClassVisibility(c.class_id)}
              title="Toggle class visibility"
            >
              {hiddenClassIds.has(c.class_id) ? "🚫" : "👁"}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
