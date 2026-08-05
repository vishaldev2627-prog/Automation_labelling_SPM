import { useState } from "react";
import toast from "react-hot-toast";
import { AutoAcceptAPI, ReviewAPI, TriageAPI } from "../../api/client";
import { useDatasetStore } from "../../store/datasetStore";
import type { TriageQueue } from "../../types";

type SimpleItem = { image_id: string; file_name: string };
type TabKey =
  | "low_confidence"
  | "no_confidence_signal"
  | "novel"
  | "routine"
  | "pending_review"
  | "audit_sample"
  | "auto_accept";

const TABS: { key: TabKey; label: string; title: string }[] = [
  {
    key: "low_confidence",
    label: "Low conf.",
    title: "Low average detector confidence on the class (Phase 2 tier 2)",
  },
  {
    key: "no_confidence_signal",
    label: "No signal",
    title:
      "Objects with no detector confidence at all - boxes read from a YOLO label file, or annotated before detector and mask confidence were separated. Unranked rather than silently absent.",
  },
  { key: "novel", label: "Novel", title: "Frames far from everything else in the similarity index (Phase 2 tier 3)" },
  { key: "routine", label: "Routine", title: "A stable random sample, to catch silent drift (Phase 2 tier 4)" },
  { key: "pending_review", label: "Pending", title: "Completed images that still need a second-reviewer sign-off" },
  { key: "audit_sample", label: "Audit", title: "Mandatory 5-10% sample of propagated annotations" },
  {
    key: "auto_accept",
    label: "Auto-accept",
    title: "High-confidence frames of classes with a proven audit track record - never safety-critical classes",
  },
];

/** Phase 2 triage + Phase 4 review queues - jump-to-image lists, not
 * inline actions (approving/reviewing happens in ReviewActions once
 * you've actually opened the image). The one exception is "auto_accept",
 * which has its own explicit bulk-accept button - see
 * app.services.auto_accept_service for why that's safe (conservative,
 * propose-then-confirm, never automatic). See
 * annotation_module_build_plan.md. */
export default function QueuePanel() {
  const jumpTo = useDatasetStore((s) => s.jumpTo);
  const refreshImages = useDatasetStore((s) => s.refreshImages);
  const refreshInfo = useDatasetStore((s) => s.refreshInfo);
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<TabKey>("low_confidence");
  const [items, setItems] = useState<Record<TabKey, SimpleItem[]>>({
    low_confidence: [],
    no_confidence_signal: [],
    novel: [],
    routine: [],
    pending_review: [],
    audit_sample: [],
    auto_accept: [],
  });
  const [loading, setLoading] = useState(false);
  const [accepting, setAccepting] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      const [triage, pending, audit, autoAccept] = await Promise.all([
        TriageAPI.queue(),
        ReviewAPI.pending(),
        ReviewAPI.auditSample(),
        AutoAcceptAPI.candidates(),
      ]);
      const q = triage as TriageQueue;
      setItems({
        low_confidence: q.low_confidence,
        no_confidence_signal: q.no_confidence_signal,
        novel: q.novel,
        routine: q.routine,
        pending_review: pending,
        audit_sample: audit,
        auto_accept: autoAccept,
      });
    } catch {
      toast.error("Failed to load queues");
    } finally {
      setLoading(false);
    }
  };

  const handleOpen = () => {
    const next = !open;
    setOpen(next);
    if (next) refresh();
  };

  const acceptAll = async () => {
    const candidates = items.auto_accept;
    if (candidates.length === 0) return;
    setAccepting(true);
    try {
      const result = await AutoAcceptAPI.execute(candidates.map((c) => c.image_id));
      toast.success(`Auto-accepted ${result.accepted} frame${result.accepted === 1 ? "" : "s"}`);
      await Promise.all([refresh(), refreshImages(), refreshInfo()]);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail ?? "Auto-accept failed");
    } finally {
      setAccepting(false);
    }
  };

  const current = items[tab];

  return (
    <div className="border-t border-surface-700">
      <button
        className="flex w-full items-center justify-between px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-500 hover:bg-surface-800"
        onClick={handleOpen}
      >
        <span>Queues</span>
        <span>{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="max-h-64 overflow-y-auto px-2 pb-2">
          <div className="mb-1 flex flex-wrap gap-1">
            {TABS.map((t) => (
              <button
                key={t.key}
                title={t.title}
                onClick={() => setTab(t.key)}
                className={`rounded px-1.5 py-0.5 text-[11px] ${
                  tab === t.key ? "bg-accent-600 text-white" : "bg-surface-800 text-gray-400 hover:bg-surface-700"
                }`}
              >
                {t.label} ({items[t.key].length})
              </button>
            ))}
            <button className="ml-auto text-[11px] text-gray-500 hover:text-gray-300" onClick={refresh} disabled={loading}>
              {loading ? "..." : "↻"}
            </button>
          </div>
          {tab === "auto_accept" && current.length > 0 && (
            <button
              className="toolbar-btn mb-1 w-full bg-green-700/40 text-[11px] hover:bg-green-700/60"
              onClick={acceptAll}
              disabled={accepting}
              title="Marks every frame below completed, attributed to System (auto-accept), with an approving review"
            >
              {accepting ? "Accepting..." : `Accept all ${current.length} frame${current.length === 1 ? "" : "s"}`}
            </button>
          )}
          {current.length === 0 ? (
            <p className="px-1 text-[11px] text-gray-500">{loading ? "Loading..." : "Nothing in this queue."}</p>
          ) : (
            <ul className="space-y-0.5">
              {current.map((item) => (
                <li key={item.image_id}>
                  <button
                    className="w-full truncate rounded px-1.5 py-0.5 text-left text-[11px] text-gray-300 hover:bg-surface-800"
                    onClick={() => jumpTo(item.image_id)}
                    title={item.file_name}
                  >
                    {item.file_name}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
