import { useAnnotationStore } from "../../store/annotationStore";
import { useDatasetStore } from "../../store/datasetStore";

export default function ObjectList() {
  const { objects, selectedObjectId, selectObject, toggleVisibility, deleteObject, regenerateObject, updateObject } =
    useAnnotationStore();
  const classes = useDatasetStore((s) => s.classes);
  const colorFor = (classId: number) => classes.find((c) => c.class_id === classId)?.color ?? "#9ca3af";

  const handleReclassify = (objectId: string, newClassId: number) => {
    const newClass = classes.find((c) => c.class_id === newClassId);
    updateObject(objectId, (o) => ({
      ...o,
      class_id: newClassId,
      class_name: newClass?.name ?? o.class_name,
      status: "edited",
    }));
  };

  const active = objects.filter((o) => o.status !== "rejected");

  return (
    <div className="flex-1 overflow-y-auto p-2">
      <h3 className="mb-2 px-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
        Objects ({active.length})
      </h3>
      {active.length === 0 && <p className="px-2 text-sm text-gray-500">No objects detected in this image.</p>}
      <ul className="space-y-1">
        {active.map((o) => (
          <li
            key={o.id}
            className={`group flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm ${
              selectedObjectId === o.id ? "bg-surface-700" : "hover:bg-surface-800"
            }`}
            onClick={() => selectObject(o.id)}
          >
            <span className="h-3 w-3 flex-shrink-0 rounded-sm" style={{ backgroundColor: colorFor(o.class_id) }} />
            <select
              className="min-w-0 flex-1 truncate rounded border-none bg-transparent py-0.5 text-gray-200 hover:bg-surface-700 focus:bg-surface-700"
              value={o.class_id}
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => handleReclassify(o.id, parseInt(e.target.value, 10))}
              title="Wrong label? Pick the correct class here."
            >
              {classes.map((c) => (
                <option key={c.class_id} value={c.class_id} className="bg-surface-800 text-gray-200">
                  {c.name}
                </option>
              ))}
            </select>
            <span className="text-xs text-gray-500">{(o.confidence * 100).toFixed(0)}%</span>
            {o.source === "propagated" && (
              <span
                className="rounded-sm bg-purple-900/60 px-1 text-[10px] font-medium uppercase tracking-wide text-purple-300"
                title="Carried over from a similar, already-annotated image — please review"
              >
                Propagated
              </span>
            )}
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                o.status === "confirmed"
                  ? "bg-green-500"
                  : o.status === "edited"
                    ? "bg-yellow-500"
                    : o.status === "auto_generated"
                      ? "bg-blue-500"
                      : "bg-gray-600"
              }`}
              title={o.status}
            />
            <button
              className="hidden text-gray-500 hover:text-gray-200 group-hover:block"
              title="Toggle visibility"
              onClick={(e) => {
                e.stopPropagation();
                toggleVisibility(o.id);
              }}
            >
              {o.visible ? "👁" : "🚫"}
            </button>
            <button
              className="hidden text-gray-500 hover:text-accent-500 group-hover:block"
              title="Regenerate mask"
              onClick={(e) => {
                e.stopPropagation();
                regenerateObject(o.id);
              }}
            >
              ↻
            </button>
            <button
              className="hidden text-gray-500 hover:text-red-500 group-hover:block"
              title="Delete object"
              onClick={(e) => {
                e.stopPropagation();
                deleteObject(o.id);
              }}
            >
              ✕
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
