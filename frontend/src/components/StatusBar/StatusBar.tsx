import { useDatasetStore } from "../../store/datasetStore";
import { useAnnotationStore } from "../../store/annotationStore";

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

export default function StatusBar() {
  const info = useDatasetStore((s) => s.info);
  const images = useDatasetStore((s) => s.images);
  const currentIndex = useDatasetStore((s) => s.currentIndex);
  const saveError = useAnnotationStore((s) => s.saveError);

  if (!info) return null;
  const current = images[currentIndex];

  return (
    <div className="flex items-center gap-4 border-t border-surface-700 bg-surface-900 px-3 py-1.5 text-xs text-gray-400">
      <span className="truncate">{current?.file_name ?? "-"}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-700">
        <div
          className="h-full rounded-full bg-accent-600 transition-all"
          style={{ width: `${info.percent_complete}%` }}
        />
      </div>
      <span>
        {info.completed}/{info.total_images} complete ({info.percent_complete.toFixed(1)}%)
      </span>
      {info.estimated_seconds_remaining != null && (
        <span>ETA {formatDuration(info.estimated_seconds_remaining)}</span>
      )}
      {saveError && <span className="text-red-400">Save error: {saveError}</span>}
    </div>
  );
}
