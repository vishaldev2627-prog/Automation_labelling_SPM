import ClassPanel from "./ClassPanel";
import ObjectList from "./ObjectList";
import QueuePanel from "./QueuePanel";
import ReviewActions from "./ReviewActions";

export default function Sidebar() {
  return (
    <div className="flex h-full w-72 flex-shrink-0 flex-col border-l border-surface-700 bg-surface-900">
      <ObjectList />
      <ReviewActions />
      <ClassPanel />
      <QueuePanel />
    </div>
  );
}
