import { useEffect } from "react";
import toast from "react-hot-toast";
import { useAnnotationStore } from "../store/annotationStore";
import { useDatasetStore } from "../store/datasetStore";
import { useSettingsStore } from "../store/settingsStore";

/** Wires global keyboard shortcuts: navigation, save, undo/redo, delete, escape, pan-hold. */
export function useKeyboardShortcuts() {
  const next = useDatasetStore((s) => s.next);
  const prev = useDatasetStore((s) => s.prev);
  const markImageCompleted = useDatasetStore((s) => s.markImageCompleted);

  const undo = useAnnotationStore((s) => s.undo);
  const redo = useAnnotationStore((s) => s.redo);
  const saveNow = useAnnotationStore((s) => s.saveNow);
  const deleteObject = useAnnotationStore((s) => s.deleteObject);
  const selectObject = useAnnotationStore((s) => s.selectObject);

  const toolMode = useSettingsStore((s) => s.toolMode);
  const setToolMode = useSettingsStore((s) => s.setToolMode);

  useEffect(() => {
    let previousTool = toolMode;

    const isEditableTarget = (target: EventTarget | null) => {
      const el = target as HTMLElement;
      return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable);
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (isEditableTarget(e.target)) return;

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        const imageId = useAnnotationStore.getState().imageId;
        saveNow(true)
          .then(() => {
            if (imageId) markImageCompleted(imageId, true);
            toast.success("Saved");
          })
          .catch(() => toast.error("Save failed"));
        return;
      }

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
        e.preventDefault();
        redo();
        return;
      }

      switch (e.key) {
        case "ArrowLeft":
          prev();
          break;
        case "ArrowRight":
          next();
          break;
        case "Delete":
        case "Backspace": {
          const selected = useAnnotationStore.getState().selectedObjectId;
          if (selected) deleteObject(selected);
          break;
        }
        case "Escape":
          selectObject(null);
          setToolMode("select");
          break;
        case " ":
          if (toolMode !== "pan") {
            previousTool = toolMode;
            e.preventDefault();
            setToolMode("pan");
          }
          break;
        default:
          break;
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === " " && toolMode === "pan") {
        setToolMode(previousTool);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [toolMode, next, prev, undo, redo, saveNow, deleteObject, selectObject, setToolMode, markImageCompleted]);
}
