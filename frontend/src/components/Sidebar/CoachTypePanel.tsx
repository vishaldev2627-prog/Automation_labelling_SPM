import { useState } from "react";
import toast from "react-hot-toast";
import { ImagesAPI } from "../../api/client";
import { useAnnotationStore } from "../../store/annotationStore";
import { useDatasetStore } from "../../store/datasetStore";
import { COACH_TYPES } from "../../types";
import type { CoachType } from "../../types";

/** Coach type for the open image, plus a bulk apply.
 *
 * The pipeline's component manifest is coach-type-conditional, and an
 * aggregate per-class label count hides a class that is well covered on LHB
 * and absent on ICF — which is what the pipeline team's label-scarcity risk
 * actually needs to be actionable. LHB and ICF only; see DECISIONS_LOG.md C-5.
 *
 * Bulk apply deliberately fills only images still marked "unknown", so it can
 * never silently overwrite a per-image correction, and it only affects images
 * that already have saved annotation state (see the backend endpoint for why).
 */
export default function CoachTypePanel() {
  const coachType = useAnnotationStore((s) => s.coachType);
  const setCoachType = useAnnotationStore((s) => s.setCoachType);
  const saving = useAnnotationStore((s) => s.saving);
  const imageId = useAnnotationStore((s) => s.imageId);
  const refreshImages = useDatasetStore((s) => s.refreshImages);
  const [applying, setApplying] = useState(false);

  const handleChange = async (next: CoachType) => {
    try {
      await setCoachType(next);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Failed to set coach type");
    }
  };

  const handleBulkApply = async () => {
    if (coachType === "unknown") {
      toast.error("Pick a coach type first — 'Unknown' is what bulk apply fills in.");
      return;
    }
    setApplying(true);
    try {
      const result = await ImagesAPI.setCoachType(coachType);
      await refreshImages();
      toast.success(
        result.updated > 0
          ? `Set ${result.updated} image(s) still marked Unknown to ${coachType}`
          : "Nothing to fill — no saved images were marked Unknown",
      );
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Bulk apply failed");
    } finally {
      setApplying(false);
    }
  };

  return (
    <div className="border-t border-surface-700 p-2">
      <h3 className="mb-2 px-1 text-xs font-semibold uppercase tracking-wide text-gray-500">Coach type</h3>
      <select
        className="w-full rounded-md border border-surface-700 bg-surface-800 px-2 py-1.5 text-sm text-gray-200 disabled:opacity-50"
        value={coachType}
        disabled={!imageId || saving}
        onChange={(e) => handleChange(e.target.value as CoachType)}
        title="Coach type for the currently open image"
      >
        {COACH_TYPES.map((c) => (
          <option key={c.value} value={c.value}>
            {c.label}
          </option>
        ))}
      </select>
      <button
        className="mt-2 w-full rounded-md bg-surface-700 px-2 py-1.5 text-xs font-medium text-gray-300 hover:bg-surface-600 disabled:opacity-50"
        disabled={applying || coachType === "unknown"}
        onClick={handleBulkApply}
        title="Apply this coach type to every already-annotated image still marked Unknown. Never overwrites an image that already has a type set."
      >
        {applying ? "Applying…" : "Apply to all Unknown"}
      </button>
    </div>
  );
}
