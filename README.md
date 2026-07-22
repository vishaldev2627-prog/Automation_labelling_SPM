# Railway Segmentation Annotation Tool

A local, AI-assisted tool for converting YOLO **detection** bounding boxes into YOLO
**segmentation** polygons using Meta's Segment Anything Model 2 (SAM2), purpose-built
for annotating railway component datasets (couplers, brake cylinders, wheels, axle
boxes, springs, bearing housings, etc.) where bounding boxes are typically much
larger than the actual object silhouette.

```
React (Vite + TS + Tailwind + Konva)
        │  REST (axios)
        ▼
FastAPI backend
        │
        ├── SAM2 inference service   (box/point prompted mask generation)
        ├── Polygon service          (OpenCV contour extraction + simplification)
        ├── Dataset service          (indexing, per-image JSON state, autosave)
        ├── Batch service            (background mask generation across the dataset)
        └── Export service           (writes final YOLO segmentation labels)
```

## Features

- Load any dataset with `images/` + `labels/` (YOLO detection `.txt`) subfolders.
- Automatic SAM2 box-prompted mask generation for every box the moment an image opens.
- OpenCV contour extraction → simplified, valid polygons (Douglas-Peucker).
- Full polygon editing: drag/insert/delete vertices, drag whole polygon, undo/redo.
- Multi-object support per image with per-class colors, visibility toggles.
- Keyboard shortcuts: `←`/`→` navigate, `Ctrl+S` save, `Delete` remove selected object,
  `Esc` deselect, `Space` (hold) pan, `Ctrl+Z`/`Ctrl+Shift+Z` undo/redo.
- Autosave (debounced) with crash recovery — every accepted edit is written to disk
  immediately as JSON, and the last opened dataset path is restored on next launch.
- Batch/background processing: generate masks for the whole dataset unattended, then
  just review and correct.
- Multiple mask candidates with confidence scores; switch between them per object.
- Positive/negative point clicks to refine a mask ("magic wand" style refinement).
- Overlay toggles (bbox / mask / polygon / image) with an opacity slider.
- Dark-mode UI in the CVAT/Roboflow/Supervisely style.
- Dataset progress tracking with ETA based on observed completion rate.

## Project layout

```
annotation_tool/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app + routers
│   │   ├── config.py          Settings (env-driven)
│   │   ├── models/schemas.py  Pydantic API models
│   │   ├── services/          sam_service, polygon_service, dataset_service,
│   │   │                      mask_generation_service, batch_service, export_service
│   │   ├── routers/           dataset, images, masks, batch, export, progress
│   │   └── utils/              image_utils, yolo_utils, file_utils, logging_config
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── api/client.ts       Axios wrapper for every backend endpoint
│   │   ├── store/               Zustand stores: dataset, annotation (undo/redo,
│   │   │                        autosave), settings (overlays, tool mode, zoom/pan)
│   │   ├── components/          Canvas (Konva), Toolbar, Sidebar, DatasetBrowser,
│   │   │                        StatusBar, ClassPanel/ObjectList
│   │   └── hooks/useKeyboardShortcuts.ts
│   ├── package.json
│   └── Dockerfile / nginx.conf
├── docker-compose.yml
├── setup.sh / setup.bat
└── README.md   (this file)
```

Runtime artifacts live outside git-tracked source:
- `<dataset>/.annotation_state/` — per-image JSON annotation state (autosave target).
- `exports/` — final YOLO segmentation dataset (`images/`, `labels/`, `classes.txt`).
- `models/` — SAM2 checkpoint(s).
- `logs/` — rotating backend log file.

## 1. Prerequisites

- Python 3.10+ (3.11 recommended)
- Node.js 18+
- An NVIDIA GPU + CUDA-capable PyTorch build is **strongly** recommended for SAM2;
  CPU inference works but is slow (seconds per box instead of tens of milliseconds).

## 2. Quick start (local, no Docker)

```bash
cd annotation_tool
./setup.sh        # Linux/macOS — see setup.bat for Windows
```

`setup.sh` / `setup.bat` will:
1. Create a Python virtualenv and install `backend/requirements.txt`.
2. Clone Meta's `sam2` repo and `pip install -e` it.
3. Download the SAM2.1 "hiera_large" checkpoint into `models/`.
4. Create `backend/.env` from the template.
5. `npm install` the frontend.

Then, in two terminals:

```bash
# Terminal 1 — backend
cd annotation_tool/backend
source .venv/bin/activate          # .venv\Scripts\activate on Windows
python run.py                      # http://localhost:8000

# Terminal 2 — frontend
cd annotation_tool/frontend
npm run dev                        # http://localhost:5173
```

Open **http://localhost:5173**, enter the absolute path to your dataset folder
(the one containing `images/` and `labels/`), and click **Load**.

### GPU acceleration

By default `SAM_DEVICE=auto` in `.env` picks CUDA if `torch.cuda.is_available()`,
then MPS (Apple Silicon), then falls back to CPU. To confirm PyTorch sees your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If it prints `False`, install the CUDA-matched PyTorch wheel for your driver from
https://pytorch.org/get-started/locally/ before re-running `pip install -r requirements.txt`.

Check `GET http://localhost:8000/api/sam/status` (or the small SAM status indicator
network call the frontend makes) to confirm the backend picked up the GPU:
`{"available": true, "backend": "pytorch", "device": "cuda", "error": null}`.

### Using an ONNX backend instead

If you've exported SAM2's encoder/decoder to ONNX (e.g. for a lighter CPU-only
deployment), set in `.env`:

```
SAM_USE_ONNX=true
SAM_ONNX_ENCODER_PATH=../models/sam2_encoder.onnx
SAM_ONNX_DECODER_PATH=../models/sam2_decoder.onnx
```

## 3. Quick start (Docker)

```bash
cd annotation_tool
cp .env.example .env
docker compose up --build
```

Requires the NVIDIA Container Toolkit for GPU passthrough
(`nvidia-ctk runtime configure --runtime=docker`). Frontend at
http://localhost:5173, backend at http://localhost:8000.

Dataset, exports, models, and logs live in Docker-managed named volumes
(`dataset_data`, `exports_data`, `models_data`, `logs_data`) — nothing needs to
exist on the host beforehand. The SAM2 repo is cloned during the image build
(`backend/Dockerfile`), and the SAM2.1 checkpoint (~900MB) is downloaded once,
automatically, into `models_data` the first time the backend container starts
(`backend/entrypoint.sh`) — it's skipped on every later restart/redeploy since
it's already on the volume.

### Deploying on Coolify

This repo deploys as-is as a Coolify **Docker Compose** resource:

1. In Coolify: **New Resource → Docker Compose**, point it at this Git repo
   (`docker-compose.yml` at the root), branch `main`.
2. Deploy. Coolify builds both images (GPU build for `backend` takes longer
   the first time), creates the four named volumes, and starts both
   containers. The backend downloads the SAM2 checkpoint on its first boot —
   watch the deployment logs for `[entrypoint] ... downloading (~900MB)`.
3. Confirm the Coolify host has the NVIDIA Container Toolkit installed so the
   `deploy.resources.reservations.devices` GPU passthrough in the compose file
   actually attaches a GPU to the `backend` container.
4. In the resource's **Domains** tab, attach your domain(s) to the `frontend`
   service (port 80) — the `backend` service doesn't need a public domain
   since nginx already proxies `/api/*` to it over the internal Docker
   network (`frontend/nginx.conf`).
5. The only manual step left is getting your actual image dataset onto the
   `dataset_data` volume (it's your private data, not app setup) — use
   Coolify's file manager for that volume, or `docker cp`/`docker compose cp`
   into the running `backend` container's `/data/dataset`.
6. Optional overrides (CORS origins, SAM device, log level, etc.) can be set
   as environment variables in Coolify's **Environment Variables** tab — see
   `backend/.env.example` for the full list; Coolify writes them into the
   `.env` the compose file already reads via `env_file`.

## 4. Workflow

1. **Load dataset** — the app scans `images/` + `labels/`, infers class names from
   `data.yaml` / `classes.txt` (or generates `class_0`, `class_1`, ...), and shows
   total/completed/remaining/progress.
2. **Open an image** — every YOLO box is automatically sent to SAM2 as a box prompt;
   returned masks are contoured into polygons and drawn as overlays.
3. **Review & edit** — drag vertices, double-click an edge to insert a vertex,
   right-click a vertex to delete it, drag the polygon body to move the whole shape.
   Use positive/negative point clicks + "Apply refine" for a magic-wand-style fix,
   or "↻ Regenerate mask" to re-run SAM2 from the original box.
4. **Save** — `Ctrl+S` or the Save button persists to
   `<dataset>/.annotation_state/<image_id>.json` and marks the image complete.
5. **Batch process** (optional) — "⚙ Batch process" runs SAM2 over every remaining
   image in the background so you only need to review, not wait per-image.
6. **Export** — "⬇ Export" writes final YOLO segmentation labels (`class x1 y1 x2 y2 ...`,
   normalized) plus copied images into `exports/`.

## 5. API reference

| Method | Path                                   | Purpose                                   |
|--------|-----------------------------------------|--------------------------------------------|
| POST   | `/api/dataset/load`                    | Load a dataset by folder path              |
| GET    | `/api/dataset/info`                    | Progress/summary stats                     |
| GET    | `/api/dataset/classes`                 | Class list + colors                        |
| PUT    | `/api/dataset/classes/{id}/color`      | Persist a class color                      |
| GET    | `/api/images`                          | List images + completion state             |
| GET    | `/api/images/{id}/file`                | Serve the image (JPEG)                     |
| GET    | `/api/images/{id}/annotations`         | Get (or lazily initialize) annotations     |
| POST   | `/api/images/annotations/save`         | Autosave / accept edits                    |
| POST   | `/api/generate-mask`                   | SAM2 for one object (box and/or points)    |
| POST   | `/api/generate-all`                    | SAM2 for every box in an image             |
| POST   | `/api/select-mask/{image_id}/{obj_id}` | Switch to another SAM mask candidate       |
| POST   | `/api/batch-process`                   | Start background batch mask generation     |
| GET    | `/api/batch-process/{job_id}`          | Poll batch job status                      |
| POST   | `/api/export`                          | Write final YOLO segmentation dataset      |
| GET    | `/api/progress`                        | Same as dataset/info, dedicated endpoint   |
| GET    | `/api/sam/status`                      | SAM2 backend/device/availability           |
| GET    | `/api/health`                          | Liveness check                             |

## 6. Error handling notes

- Missing `labels/` folder: treated as an unlabeled dataset (empty dir created), not fatal.
- Malformed detection label lines: skipped with a logged warning, not aborted.
- Corrupted/unreadable images: return HTTP 422 with a clear message instead of crashing.
- SAM2 checkpoint/package missing or GPU unavailable: `/api/sam/status` and any mask
  endpoint report a clear `503` with the exact missing dependency/file, instead of a
  silent failure or crash.
- Large images are served as re-encoded JPEGs; embedding computation is cached per
  image (keyed by file hash) so repeated edits don't re-run the SAM encoder.

## 7. Configuration (`backend/.env`)

See `backend/.env.example` for the full list: dataset/export paths, SAM checkpoint
path & device, CORS origins, polygon simplification epsilon, batch worker count,
and log level/file.
# Automation_labelling_SPM
