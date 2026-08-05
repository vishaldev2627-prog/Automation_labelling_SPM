import toast from "react-hot-toast";
import { useAnnotationStore } from "../../store/annotationStore";
import { useDatasetStore } from "../../store/datasetStore";
import { CONDITIONS } from "../../types";
import type { Condition } from "../../types";

export default function ObjectList() {
  const { objects, selectedObjectId, selectObject, toggleVisibility, deleteObject, regenerateObject, updateObject } =
    useAnnotationStore();
  const noObjectsConfirmed = useAnnotationStore((s) => s.noObjectsConfirmed);
  const saving = useAnnotationStore((s) => s.saving);
  const setNoObjectsConfirmed = useAnnotationStore((s) => s.setNoObjectsConfirmed);
  const refreshImages = useDatasetStore((s) => s.refreshImages);
  const refreshInfo = useDatasetStore((s) => s.refreshInfo);
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

  const handleSetCondition = (objectId: string, value: string) => {
    // "" is the unassessed sentinel from the select; it maps back to null, not
    // to a condition string, so clearing an assessment stays distinguishable
    // from asserting "ok".
    const condition = value === "" ? null : (value as Condition);
    updateObject(objectId, (o) => ({ ...o, condition, status: "edited" }));
  };

  const handleToggleEmpty = async () => {
    const next = !noObjectsConfirmed;
    try {
      await setNoObjectsConfirmed(next);
      // Confirming completes the image, so the image list and progress counts
      // both move - refresh them rather than leaving a stale "remaining" count.
      await Promise.all([refreshImages(), refreshInfo()]);
      toast.success(next ? "Confirmed empty — exports as a negative sample" : "Empty confirmation retracted");
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Failed to update empty confirmation");
    }
  };

  const handleRegenerate = async (objectId: string) => {
    try {
      await regenerateObject(objectId);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Failed to regenerate mask");
    }
  };

  const active = objects.filter((o) => o.status !== "rejected");

  return (
    <div className="flex-1 overflow-y-auto p-2">
      <h3 className="mb-2 px-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
        Objects ({active.length})
      </h3>
      {active.length === 0 && (
        <div className="px-2">
          <p className="text-sm text-gray-500">No objects detected in this image.</p>
          {/* An empty frame is only training data once a human says it's
              genuinely empty - otherwise it's indistinguishable from "not
              annotated yet" and export skips it. Confirming turns it into a
              negative/background sample (empty label file). */}
          <button
            className={`mt-2 w-full rounded-md px-2 py-1.5 text-xs font-medium ${
              noObjectsConfirmed
                ? "bg-emerald-900/60 text-emerald-300 hover:bg-emerald-900"
                : "bg-surface-700 text-gray-300 hover:bg-surface-600"
            }`}
            disabled={saving}
            onClick={handleToggleEmpty}
            title={
              noObjectsConfirmed
                ? "Confirmed empty — exports as a negative sample. Click to retract."
                : "Confirm this frame is genuinely empty. It then exports as a negative/background sample instead of being skipped."
            }
          >
            {noObjectsConfirmed ? "✓ Confirmed empty" : "Confirm empty (no objects here)"}
          </button>
          {noObjectsConfirmed && (
            <p className="mt-1 text-[11px] leading-snug text-gray-500">
              Exports as a negative sample. Still needs a second-reviewer sign-off like any other
              completed image.
            </p>
          )}
        </div>
      )}
      <ul className="space-y-1">
        {active.map((o) => (
          <li
            key={o.id}
            className={`group cursor-pointer rounded-md px-2 py-1.5 text-sm ${
              selectedObjectId === o.id ? "bg-surface-700" : "hover:bg-surface-800"
            }`}
            onClick={() => selectObject(o.id)}
          >
            <div className="flex items-center gap-2">
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
            {/* Two separate signals now (see AnnotationObject): "M" is SAM2's
                mask score, "D" the detector's class confidence. This used to
                be one number that silently switched meaning once a mask was
                generated. "D —" means no detector signal exists, which is not
                the same as a low one. */}
            <span
              className="whitespace-nowrap text-xs text-gray-500"
              title={`Mask confidence (SAM2): ${(o.mask_confidence * 100).toFixed(0)}%\nDetector confidence (class): ${
                o.detector_confidence === null ? "no signal" : `${(o.detector_confidence * 100).toFixed(0)}%`
              }`}
            >
              M {(o.mask_confidence * 100).toFixed(0)}% · D{" "}
              {o.detector_confidence === null ? "—" : `${(o.detector_confidence * 100).toFixed(0)}%`}
            </span>
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
                handleRegenerate(o.id);
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
            </div>
            {/* Condition — what state the component is in, as opposed to what
                it is (that's the class above). This is the label the
                p1_side_damage crop classifier trains on; without it side_view
                supplies that model's crops but none of its labels.
                "Unassessed" is a real state, not a missing value: an object
                only exports as training data once a human has actually judged
                it. Shown on the selected row only, to keep the list compact. */}
            {selectedObjectId === o.id && (
              <div className="mt-1 flex items-center gap-2 pl-5">
                <span className="text-[11px] uppercase tracking-wide text-gray-500">Condition</span>
                <select
                  className={`min-w-0 flex-1 truncate rounded border border-surface-600 bg-surface-800 px-1 py-0.5 text-xs ${
                    o.condition === null ? "text-gray-500" : "text-gray-200"
                  }`}
                  value={o.condition ?? ""}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => handleSetCondition(o.id, e.target.value)}
                  title="Component condition — the p1_side_damage classifier's label. 'Unassessed' means nobody has judged this component yet; it is not the same as 'ok'."
                >
                  <option value="" className="bg-surface-800 text-gray-500">
                    — unassessed —
                  </option>
                  {CONDITIONS.map((c) => (
                    <option key={c} value={c} className="bg-surface-800 text-gray-200">
                      {c.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
