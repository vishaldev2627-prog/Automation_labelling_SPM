import { useState } from "react";
import toast from "react-hot-toast";
import { ReviewAPI, TriageAPI } from "../../api/client";
import { useDatasetStore } from "../../store/datasetStore";
import type { TriageQueue } from "../../types";

type SimpleItem = { image_id: string; file_name: string };
type TabKey = "low_confidence" | "novel" | "routine" | "pending_review" | "audit_sample";

const TABS: { key: TabKey; label: string; title: string }[] = [
  { key: "low_confidence", label: "Low conf.", title: "Auto-detected objects with low confidence (Phase 2 tier 2)" },
  { key: "novel", label: "Novel", title: "Frames far from everything else in the similarity index (Phase 2 tier 3)" },
  { key: "routine", label: "Routine", title: "A stable random sample, to catch silent drift (Phase 2 tier 4)" },
  { key: "pending_review", label: "Pending", title: "Completed images that still need a second-reviewer sign-off" },
  { key: "audit_sample", label: "Audit", title: "Mandatory 5-10% sample of propagated annotations" },
];

/** Phase 2 triage + Phase 4 review queues - jump-to-image lists, not
 * inline actions (approving/reviewing happens in ReviewActions once
 * you've actually opened the image). See annotation_module_build_plan.md. */
export default function QueuePanel() {
  const jumpTo = useDatasetStore((s) => s.jumpTo);
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<TabKey>("low_confidence");
  const [items, setItems] = useState<Record<TabKey, SimpleItem[]>>({
    low_confidence: [],
    novel: [],
    routine: [],
    pending_review: [],
    audit_sample: [],
  });
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      const [triage, pending, audit] = await Promise.all([
        TriageAPI.queue(),
        ReviewAPI.pending(),
        ReviewAPI.auditSample(),
      ]);
      const q = triage as TriageQueue;
      setItems({
        low_confidence: q.low_confidence,
        novel: q.novel,
        routine: q.routine,
        pending_review: pending,
        audit_sample: audit,
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
