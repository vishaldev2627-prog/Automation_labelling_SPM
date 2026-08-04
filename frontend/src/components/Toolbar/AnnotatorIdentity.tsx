import { useEffect, useState } from "react";
import toast from "react-hot-toast";
import { AnnotatorAPI } from "../../api/client";

const ANNOTATOR_NAME_KEY = "railway-annotator:annotator-name";

/** Lightweight per-annotator identity - a name, not a login (Phase 1a, task
 * #3). Every annotation save needs to know *who* made it (see
 * app.session_context.py / annotation_state.updated_by), so this prompts
 * once per browser and re-identifies silently on every later load. */
export default function AnnotatorIdentity() {
  const [name, setName] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  useEffect(() => {
    const stored = localStorage.getItem(ANNOTATOR_NAME_KEY);
    if (stored) {
      AnnotatorAPI.identify(stored)
        .then((r) => setName(r.name))
        .catch(() => setEditing(true));
    } else {
      setEditing(true);
    }
  }, []);

  const submit = async () => {
    const trimmed = draft.trim();
    if (!trimmed) return;
    try {
      const r = await AnnotatorAPI.identify(trimmed);
      localStorage.setItem(ANNOTATOR_NAME_KEY, trimmed);
      setName(r.name);
      setEditing(false);
    } catch {
      toast.error("Couldn't set annotator name");
    }
  };

  if (editing) {
    return (
      <div className="flex items-center gap-1">
        <input
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Your name"
          className="w-28 rounded border border-surface-600 bg-surface-800 px-2 py-1 text-sm text-gray-200 placeholder:text-gray-500"
        />
        <button className="toolbar-btn" onClick={submit} title="Set annotator name">
          Set
        </button>
      </div>
    );
  }

  return (
    <button
      className="toolbar-btn"
      onClick={() => {
        setDraft(name ?? "");
        setEditing(true);
      }}
      title="Click to change annotator name"
    >
      👤 {name ?? "…"}
    </button>
  );
}
