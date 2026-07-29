"""Review dashboard for exports/class_folders/<class>/*.

Browse each class folder's highlighted images, approve or disapprove them.
Decisions are appended to per-folder approve.txt / deleted.txt files.
Disapproving only logs the filename to deleted.txt - it does NOT delete
the image from disk. Nothing here performs the actual purge.
"""
from __future__ import annotations

import hmac
import os
from functools import wraps
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

CLASS_FOLDERS_ROOT = Path(
    os.environ.get(
        "CLASS_FOLDERS_ROOT",
        "/home/omronix/Component/annotation_tool/exports/class_folders",
    )
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

DASHBOARD_USERNAME = os.environ.get("DASHBOARD_USERNAME")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD")

app = Flask(__name__)


def _auth_ok(auth) -> bool:
    if not auth:
        return False
    return hmac.compare_digest(auth.username, DASHBOARD_USERNAME) and hmac.compare_digest(
        auth.password, DASHBOARD_PASSWORD
    )


@app.before_request
def require_basic_auth():
    if not (DASHBOARD_USERNAME and DASHBOARD_PASSWORD):
        # No credentials configured (e.g. local dev run) - leave open.
        return None
    if not _auth_ok(request.authorization):
        return Response(
            "Authentication required", 401, {"WWW-Authenticate": 'Basic realm="Review Dashboard"'}
        )
    return None


def list_class_folders() -> list[str]:
    if not CLASS_FOLDERS_ROOT.exists():
        return []
    return sorted(
        p.name for p in CLASS_FOLDERS_ROOT.iterdir()
        if p.is_dir()
    )


def folder_path(class_name: str) -> Path:
    safe = Path(class_name).name  # strip any path traversal
    folder = CLASS_FOLDERS_ROOT / safe
    if not folder.is_dir():
        abort(404, description=f"Unknown class folder: {class_name}")
    return folder


def read_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_line(path: Path, file_name: str) -> None:
    existing = read_lines(path)
    if file_name in existing:
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(file_name + "\n")


def remove_line(path: Path, file_name: str) -> None:
    if not path.exists():
        return
    lines = [ln for ln in read_lines(path) if ln != file_name]
    path.write_text("\n".join(sorted(lines)) + ("\n" if lines else ""), encoding="utf-8")


@app.get("/api/folders")
def api_folders():
    folders = []
    for name in list_class_folders():
        folder = CLASS_FOLDERS_ROOT / name
        images = [p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        approved = read_lines(folder / "approve.txt")
        disapproved = read_lines(folder / "deleted.txt")
        folders.append(
            {
                "name": name,
                "total": len(images),
                "approved": len(approved & set(images)),
                "disapproved": len(disapproved & set(images)),
                "pending": len([i for i in images if i not in approved and i not in disapproved]),
            }
        )
    return jsonify(folders)


@app.get("/api/folders/<class_name>/images")
def api_images(class_name: str):
    folder = folder_path(class_name)
    approved = read_lines(folder / "approve.txt")
    disapproved = read_lines(folder / "deleted.txt")
    images = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    result = []
    for name in images:
        if name in approved:
            status = "approved"
        elif name in disapproved:
            status = "disapproved"
        else:
            status = "pending"
        result.append({"file_name": name, "status": status})
    return jsonify(result)


@app.get("/images/<class_name>/<file_name>")
def serve_image(class_name: str, file_name: str):
    folder = folder_path(class_name)
    safe_file = Path(file_name).name
    if not (folder / safe_file).exists():
        abort(404)
    return send_from_directory(folder, safe_file)


@app.post("/api/folders/<class_name>/decision")
def api_decision(class_name: str):
    folder = folder_path(class_name)
    payload = request.get_json(force=True, silent=True) or {}
    file_name = Path(str(payload.get("file_name", ""))).name
    decision = payload.get("decision")
    if not file_name or not (folder / file_name).exists():
        abort(404, description="Unknown image")
    if decision not in ("approve", "disapprove", "reset"):
        abort(422, description="decision must be approve, disapprove, or reset")

    approve_path = folder / "approve.txt"
    deleted_path = folder / "deleted.txt"

    if decision == "approve":
        remove_line(deleted_path, file_name)
        append_line(approve_path, file_name)
    elif decision == "disapprove":
        remove_line(approve_path, file_name)
        append_line(deleted_path, file_name)
    else:  # reset
        remove_line(approve_path, file_name)
        remove_line(deleted_path, file_name)

    return jsonify({"file_name": file_name, "status": decision})


@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent / "static", "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8600, debug=False)
