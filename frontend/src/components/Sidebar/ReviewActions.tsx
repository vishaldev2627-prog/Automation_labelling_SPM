import { useEffect, useState } from "react";
import toast from "react-hot-toast";
import { ReviewAPI } from "../../api/client";
import { useAnnotationStore } from "../../store/annotationStore";
import type { ReviewReason, ReviewRecord } from "../../types";

/** Second-reviewer sign-off + audit sampling for the currently-open image
 * (Phase 4, safety-critical QA layer - see annotation_module_build_plan.md).
 * Tied to whichever image is actually open, not a queue list, since
 * approving/rejecting requires having looked at the image. */
export default function ReviewActions() {
  const imageId = useAnnotationStore((s) => s.imageId);
  const completed = useAnnotationStore((s) => s.completed);
  const [review, setReview] = useState<ReviewRecord | null>(null);
  const [reason, setReason] = useState<ReviewReason>("second_review");
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!imageId) {
      setReview(null);
      return;
    }
    setLoading(true);
    ReviewAPI.get(imageId)
      .then(setReview)
      .catch(() => setReview(null))
      .finally(() => setLoading(false));
  }, [imageId]);

  const submit = async (decision: "approved" | "rejected") => {
    if (!imageId) return;
    setSubmitting(true);
    try {
      const record = await ReviewAPI.submit(imageId, decision, reason, notes.trim() || undefined);
      setReview(record);
      setNotes("");
      toast.success(decision === "approved" ? "Approved" : "Rejected");
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Review failed");
    } finally {
      setSubmitting(false);
    }
  };

  if (!imageId) return null;

  return (
    <div className="border-t border-surface-700 p-2">
      <h3 className="mb-2 px-2 text-xs font-semibold uppercase tracking-wide text-gray-500">Review</h3>
      {!completed && (
        <p className="px-2 text-[11px] text-amber-500">This image isn't marked completed yet.</p>
      )}
      {loading ? (
        <p className="px-2 text-sm text-gray-500">Loading review status...</p>
      ) : review ? (
        <div className="mx-2 mb-2 rounded border border-surface-700 bg-surface-800 p-2 text-xs">
          <span className={review.decision === "approved" ? "text-green-400" : "text-red-400"}>
            {review.decision === "approved" ? "✓ Approved" : "✗ Rejected"}
          </span>{" "}
          <span className="text-gray-500">
            by {review.reviewer_name} ({review.reason === "audit_sample" ? "audit sample" : "second review"})
          </span>
          {review.notes && <p className="mt-1 text-gray-400">"{review.notes}"</p>}
        </div>
      ) : (
        <p className="mb-2 px-2 text-[11px] text-gray-500">Not yet reviewed.</p>
      )}

      <div className="px-2">
        <select
          value={reason}
          onChange={(e) => setReason(e.target.value as ReviewReason)}
          className="mb-1 w-full rounded border border-surface-600 bg-surface-800 px-2 py-1 text-xs text-gray-200"
        >
          <option value="second_review">Second review</option>
          <option value="audit_sample">Audit sample (propagated)</option>
        </select>
        <input
          type="text"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Notes (optional)"
          className="mb-1 w-full rounded border border-surface-600 bg-surface-800 px-2 py-1 text-xs text-gray-200 placeholder:text-gray-600"
        />
        <div className="flex gap-1">
          <button
            className="toolbar-btn flex-1 bg-green-700/40 hover:bg-green-700/60"
            onClick={() => submit("approved")}
            disabled={submitting}
          >
            Approve
          </button>
          <button
            className="toolbar-btn flex-1 bg-red-700/40 hover:bg-red-700/60"
            onClick={() => submit("rejected")}
            disabled={submitting}
          >
            Reject
          </button>
        </div>
      </div>
    </div>
  );
}
