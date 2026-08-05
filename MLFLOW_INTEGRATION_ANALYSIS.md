# MLflow Integration — Project Understanding, Gap Analysis & Design

**Status:** Phase 1–5 analysis report. **No code was modified.** Nothing here is implemented.
**Repo analysed:** `D:\Automation_labelling_SPM` @ branch `automation_pipeline`, commit `36933ee`.
**Reference doc read in full:** `MLflow (1).md`.
**Also read in full:** `annotation_module_build_plan.md`, `README.md`, all 102 tracked files' relevant paths.

---

# PART 0 — Executive answer up front

Three findings dominate everything below.

**1. There is no MLflow in this repository. Zero.**
`grep -rniE "mlflow|dvc|boto3" --include=*.py --include=*.txt` returns **no matches in any source
file**. `mlflow` is not in `backend/requirements.txt`. There is no tracking server, no registry, no
`mlflow.start_run()`, no artifact logging, no promotion gate, no rollback path. Phase 3 of your brief
("verify whether the current implementation follows the document") has a short answer: **0 of the
document's MLflow features are implemented.** This is a greenfield MLflow build, not a correction of
an existing one.

**2. This repo is the *annotation module*, not the VB inspection pipeline.**
The MLflow doc (`MLflow (1).md`) is written against a *different* system — the VB detection pipeline
with its **8 registered model families** (`VB-SharedDetector`, `VB-P3-WheelSeg`, …), `pipeline.md`,
and `FINAL_AIML_ARCHITECTURE.md`. **None of those three documents exist in this repo**, and none of
those 8 model families are trained here. What *is* trained here is one thing: a **YOLOv8 pre-labeler**
(`backend/app/services/detector_service.py`) that speeds up human annotation. The MLflow doc's §2.1a
table does not describe any model this codebase produces.

**3. Your own locked build plan says this module must NOT write to MLflow directly.**
`annotation_module_build_plan.md` §6, **Q-C — RESOLVED**:
> "The annotation module does **not** get write access to the pipeline's MLflow. It stages versioned
> dataset snapshots (e.g. S3/MinIO) that the pipeline team imports into MLflow themselves. Phase 5
> export design should target a staging store + a defined snapshot manifest format, **not** direct
> MLflow API calls from this module."

Your request ("integrate the annotation tool into my MLflow pipeline") is in **direct tension with
Q-C**. I have not assumed which one wins. This is the single blocking decision — see
**Part 6 / Decision D1**. Everything downstream (which components get built, where they live, who
owns the tracking server) changes depending on the answer, so I have designed both branches rather
than silently picking one.

---

# PART 1 — PROJECT UNDERSTANDING

## Overall Architecture

Five runtime processes, one host, defined in `docker-compose.yml`:

```
                            ┌──────────────────────────────────┐
  Browser ────HTTP/80──────►│ frontend  (nginx + React SPA)    │
                            │ nginx proxies /api/* → backend   │
                            └───────────────┬──────────────────┘
                                            │ REST (axios), session_id cookie
                                            ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ backend  (FastAPI, uvicorn, GPU passthrough)   :8000              │
   │                                                                    │
   │  app/main.py ─ 12 routers ─ session_context.SessionBundle          │
   │       │                                                            │
   │       ├── sam_service          SAM2.1 hiera_large — PROCESS-WIDE    │
   │       │                        singleton, one lock, one GPU         │
   │       ├── mask_generation_svc  SAM2 → mask → polygon                │
   │       ├── polygon_service      OpenCV contour + Douglas-Peucker     │
   │       ├── dataset_service      index images/labels, class map       │
   │       ├── annotation_state_repo Postgres upsert + history           │
   │       ├── similarity_service   MobileNetV3-Small embeddings, .npz    │
   │       ├── propagation_service  copy labels onto near-duplicates     │
   │       ├── batch_service        background whole-dataset mask gen     │
   │       ├── detector_service     ★ YOLOv8 TRAIN + INFER (the only     │
   │       │                          training in the whole repo)        │
   │       ├── triage_service       priority tiers for human queue       │
   │       ├── review_service       2nd-reviewer sign-off, audit sample   │
   │       ├── auto_accept_service  confidence-gated bulk accept          │
   │       └── export_service       → YOLO-seg dataset on disk            │
   └───────┬──────────────────────┬─────────────────────┬───────────────┘
           │                      │                     │
      ┌────▼─────┐        ┌───────▼────────┐   ┌────────▼────────────┐
      │ postgres │        │ host filesystem │   │ minio (RUNNING BUT  │
      │  :5432   │        │ /data/dataset   │   │ UNUSED — no code    │
      │ 6 tables │        │ /data/exports   │   │ references it)      │
      └──────────┘        │ /data/models    │   └─────────────────────┘
                          └────────┬─────────┘
                                   │ exports/class_folders
                          ┌────────▼──────────────────────┐
                          │ review-dashboard (Flask)       │
                          │ approve.txt / deleted.txt      │
                          │ separate app, separate storage │
                          └────────────────────────────────┘
```

Key architectural properties, verified in code:

- **Session-scoped services.** `app/session_context.py` keys a `SessionBundle` per browser
  `session_id` (cookie, set in `frontend/src/api/client.ts`). Each session has its *own*
  `DatasetService`, `DetectorService`, `SimilarityService`, `BatchService`, `PropagationService`.
  In-memory only, `SESSION_IDLE_TTL_SECONDS = 8h`, evicted on next new-session lookup.
- **SAM2 is the one exception** — a single process-wide instance behind a lock (`sam_service.py`),
  one GPU shared by all sessions.
- **Postgres is the durable state layer** (Phase 1a shipped); `SessionBundle` is only a
  service-instance cache above it.
- **No auth anywhere.** `POST /api/annotator/identify` is a name, explicitly "not a login"
  (`routers/annotator.py` docstring). Role changes are ungated. The Flask dashboard has optional
  HTTP Basic. This matters for MLflow promotion sign-off (Part 5, R-4).

## Current Workflow

End-to-end, as the code actually runs it:

1. **Boot** — `main.py` `@app.on_event("startup")` auto-loads `DATASET_PATH/side_view` for the
   `default` session, then kicks off background similarity indexing if `propagation_enabled`.
   `entrypoint.sh` runs `alembic upgrade head` before uvicorn starts, and downloads the ~900 MB SAM2
   checkpoint on first boot only.
2. **Identify** — annotator types a name → `annotators` row get-or-created → attached to the session
   bundle. Every save records `updated_by_id`.
3. **Pick a view** — `POST /api/dataset/switch` with one of 4 fixed keys
   (`side_view` / `underbelly` / `wheel_shelling` / `buffer`, `routers/dataset.py`). Per-session
   only; other tabs unaffected.
4. **Queue** — `GET /api/triage/queue` returns 5 tiers; frontend `QueuePanel` renders them as
   jump-to lists. `field_flagged` and `gate_recall_audit_miss` are **always empty** by design.
5. **Open an image** — `GET /api/images/{id}/annotations`. On first access, state is initialised from
   `labels/<stem>.txt` YOLO detection boxes (`confidence=0.0`, meaning *no signal*, explicitly not
   "low confidence" — see `dataset_service.get_annotations` comment). If no label file exists at all,
   falls back to `detector_service.detect()` using the most recently trained local YOLOv8.
6. **Mask** — `POST /api/generate-mask` / `generate-all`: SAM2 box prompt (+ optional pos/neg point
   clicks) → best-scoring mask → `mask_to_polygon()`. If `confidence <= mask_confidence_threshold`
   (0.5), polygon is **empty** and status stays `PENDING` (`mask_generation_service.py`).
7. **Edit** — Konva canvas: drag/insert/delete vertices, drag polygon, undo/redo, per-class colors,
   visibility toggles.
8. **Save** — `POST /api/images/annotations/save`. Upsert into `annotation_state` + append row to
   `annotation_history` (one transaction, `annotation_state_repo.save_state`).
9. **Propagate** — only on the **transition** to `completed` (`routers/images.py`: `if
   annotations.completed and not was_completed`). Background thread copies accepted objects onto up
   to `propagation_top_k=5` neighbors above `similarity_threshold=0.85`, re-running SAM2 per object,
   and **only** onto images `_is_untouched()` — not completed, no `EDITED`/`CONFIRMED`/`REJECTED`
   object. Propagated objects get `source=PROPAGATED` + `propagated_from_image_id`.
10. **Review** — `POST /api/review/{image_id}` with `decision` ∈ {approved, rejected} and `reason` ∈
    {second_review, audit_sample, auto_accept}. Reviewer must differ from `updated_by_id` for
    `second_review` (unless submitter is unknown/backfilled).
11. **Audit sample** — `GET /api/review/audit-sample`: stable seeded 7.5 % sample of completed images
    that contain at least one `source == "propagated"` object, excluding already-audited ones.
12. **Auto-accept** — `GET /api/auto-accept/candidates` (preview) → `POST /api/auto-accept/execute`
    (explicit ids). Gate: class not `safety_critical`, ≥ 10 `audit_sample` reviews at **100 %**
    approval, every object in the frame ≥ 0.95 confidence, all-or-nothing per image. Completion
    attributed to reserved identity `System (auto-accept)` with its own approving review.
13. **Retrain the pre-labeler** — `POST /api/detector/train`: assembles a YOLO *detection* dataset
    from completed images (boxes only, not polygons), fine-tunes `yolov8s.pt` for 100 epochs
    (`patience=20`, `batch=-1`), copies `best.pt` → `models/detector_v{N}.pt`, bumps
    `models/detector_registry.json`.
14. **Export** — `POST /api/export`: for each completed + export-eligible image, writes YOLO-seg
    labels (`class x1 y1 …` normalised) and copies the image into `exports/{images,labels}/{train,val}`
    with a deterministic md5-based 90/10 split, plus `classes.txt` and `data.yaml`.
15. **Dashboard (separate)** — Flask app over `exports/class_folders/<class>/*`, appends filenames to
    `approve.txt` / `deleted.txt`. Never deletes files. Not connected to the Postgres review tables.

## Folder Structure

```
D:\Automation_labelling_SPM
├─ MLflow (1).md                  ← the spec you gave me (describes the *pipeline*, not this repo)
├─ annotation_module_build_plan.md ← THIS repo's own plan; Q-C forbids direct MLflow writes
├─ README.md                       ← tool docs, API table, Coolify deploy notes
├─ docker-compose.yml              ← 5 services: db, minio, backend, frontend, review-dashboard
├─ .env.example                    ← dashboard basic-auth, POSTGRES_*, MINIO_ROOT_*
├─ setup.sh / setup.bat / start.bat
├─ backend/
│  ├─ Dockerfile                   ← CUDA 12.8 runtime, torch 2.7.1+cu128, aarch64/GB10 targeted,
│  │                                  SAM2 cloned + torch.jit.script patch, MobileNetV3 pre-pull
│  ├─ entrypoint.sh                ← SAM2 ckpt download + `alembic upgrade head` + exec uvicorn
│  ├─ requirements.txt             ← NO mlflow, NO dvc, NO boto3/minio client
│  ├─ alembic.ini
│  ├─ run.py
│  ├─ migrations/versions/         ← 5 revisions (see Database)
│  ├─ scripts/backfill_annotation_state.py  ← one-time JSON→Postgres backfill, re-runnable
│  └─ app/
│     ├─ main.py, config.py, db.py, session_context.py
│     ├─ models/{db_models.py, schemas.py}
│     ├─ routers/  annotator auto_accept batch dataset detector export images masks
│     │            progress review similarity triage      (12)
│     ├─ services/ annotation_state_repo annotator auto_accept batch dataset detector
│     │            export mask_generation polygon propagation review sam similarity triage (14)
│     └─ utils/    file_utils image_utils logging_config yolo_utils
├─ frontend/  React 18 + Vite + TS + Tailwind + Konva + Zustand
│  └─ src/ api/client.ts · store/{annotation,dataset,settings} · components/{Canvas,Sidebar,
│          Toolbar,DatasetBrowser,StatusBar} · hooks/useKeyboardShortcuts.ts
├─ review_dashboard/  Flask app.py + static/index.html  (class-folder approve/reject)
├─ exports/  models/  logs/   ← all empty locally (.gitkeep only); populated on the deploy host
```

**Absent but referenced:** `pipeline.md`, `FINAL_AIML_ARCHITECTURE.md`,
`component_defect_taxonomy.yaml`, `COACH_BOUNDARY_BUFFER_DETECTOR.md`, `DefectReviewLog`.
Every one of these is cited as authoritative by `MLflow (1).md`. I cannot verify any threshold, tier
assignment, metric floor, or class-map claim that depends on them.

## Existing MLflow Usage

**None.** Verified exhaustively:

| Check | Result |
|---|---|
| `mlflow` in `backend/requirements.txt` | absent |
| `import mlflow` anywhere | 0 occurrences |
| `mlflow.start_run` / `log_metric` / `log_artifact` / `register_model` | 0 occurrences |
| Tracking server in `docker-compose.yml` | absent |
| `MLFLOW_TRACKING_URI` env var | absent from `.env.example`, `backend/.env.example`, compose |
| DVC / content-addressed dataset snapshots | absent |
| `boto3` / `minio` / S3 client | absent — the `minio` **container runs but no code talks to it** |

The only version tracking that exists is **hand-rolled**: `models/detector_registry.json`, written
by `detector_service._save_registry()`, holding a single monotonically-increasing integer:

```json
{"version": N, "trained_at": <epoch>, "num_images": N, "classes": [...], "path": "…/detector_vN.pt"}
```

Properties of that registry, which are exactly what MLflow would replace:
- One active version, no stage concept, no `Staging`/`Production`/`Archived`.
- No metrics stored at all — not even final mAP. Training metrics are computed by ultralytics inside
  the staging dir and then **deleted** (`shutil.rmtree(staging_dir)` in the `finally` block).
- No dataset SHA, no class-map version, no git commit, no hyperparameters beyond the module constants.
- No comparison against the previous version. **Every trained model is promoted unconditionally** by
  overwriting `version`, and `is_active()` immediately makes it the auto-detect model for the whole
  session. There is no gate of any kind.
- No rollback: `detector_v{N-1}.pt` files remain on disk, but nothing reads them and there is no
  endpoint or code path to revert.
- The registry is per-`models_dir` (host path shared by all sessions and **all four dataset views**)
  while `DetectorService` is *session*-scoped — so a detector trained on `buffer` overwrites the
  version a colleague trained on `side_view`, and both then auto-detect with whichever landed last.
  **This is a real, present bug independent of MLflow** (Part 3, P-1).

## Dataset Pipeline

**Storage is a plain filesystem tree, one root per view.** No object storage, no content addressing,
no versioning.

```
DATASET_PATH/                     (host: /home/omronix/Component/annotation_dataset)
├─ side_view/                     ← auto-loaded on boot; the fully-annotated one (~2013 images
│  ├─ images/*.{jpg,png,…}           per build plan Phase 2 test note)
│  ├─ labels/*.txt                ← YOLO detection: `class cx cy w h`, normalised
│  ├─ data.yaml | classes.txt     ← class-map source of truth on disk
│  └─ .annotation_state/          ← now ONLY holds _similarity_index.npz
│                                    (annotation JSON moved to Postgres in Phase 1a)
├─ underbelly/     (starts empty)
├─ wheel_shelling/ (starts empty)
└─ buffer/         (raw dump, zero classes, built up via UI +Add)
```

- `dataset_key` = `str(Path(dataset_path).resolve())` — the resolved absolute root path. Stored in a
  column literally named `dataset_view` (`annotation_state_repo.py` docstring explains the naming).
  **Consequence: the DB key is a host filesystem path.** Move or rename the dataset directory and
  every annotation, review, and class row is orphaned. This is a serious portability problem for any
  MLflow lineage claim (Part 3, P-2).
- Class map: `load_class_names(root)` reads `data.yaml`/`data.yml`/`dataset.yaml` `names:`, else
  `classes.txt`, else synthesises `class_0…class_N` from the max class id found by scanning every
  label file. `add_class()` appends and **rewrites the whole file** under a
  path-keyed cross-session lock (`_get_class_list_lock`).
- **There is no class-map version anywhere** — not in the DB, not in the export, not in `data.yaml`.
  The build plan calls unversioned class maps a `[HIGH]` risk that *already bit this project* (the
  27-class remap, evidenced by `*_before_27class_remap_*` backup suffixes). Still unfixed.
- **No spine stamp.** `coach_index`, `coach_type`, `axle_id`, `side`, `view`,
  `longitudinal_position_mm` — none of these columns exist. `db_models.py` says so explicitly:
  deferred to Phase 1b, "adding them later is a nullable-column migration, not a breaking one."
- Export target is a **single mutable directory** (`exports/`, or `exports/<output_subdir>` if
  supplied). Re-exporting **overwrites in place**: labels are rewritten unconditionally, images are
  skipped if the destination already exists (`if not dst_image.exists()`). There is no snapshot id,
  no manifest, no immutability, no content hash. Build plan Phase 5 describes exactly this gap and
  the hand-rolled `*_before_synthetic_removal_*` backups that resulted.
- Split: `_split_for(image_id)` = `md5(image_id) % 100 < 10 → val`. Deterministic and stable across
  incremental exports — good. But **there is no pseudo/synthetic flag anywhere**, so the plan's
  mandate "pseudo/synthetic labels constrained to train split only" is **not enforceable today**;
  `source=PROPAGATED` objects flow into `val` exactly like human ones.

## Annotation Pipeline

*(Phase 2 of your brief, answered against code.)*

**How annotation works** — Detection box (from `labels/*.txt`, or the local YOLOv8, or drawn by hand)
becomes a SAM2 box prompt. `sam_service.predict_box()` returns N candidate masks + scores;
`mask_generation_service` picks `argmax(score)`, converts via `polygon_service.mask_to_polygon()`
(OpenCV `findContours` + Douglas-Peucker at `polygon_epsilon_ratio=0.002`, min 3 points). Below
`mask_confidence_threshold=0.5` the polygon is dropped and status stays `PENDING` — a deliberate
"never show a bad mask" rule. Human then edits vertices, or refines with positive/negative point
clicks, or switches to another candidate (`POST /api/select-mask/...`, no re-inference of the encoder
thanks to the per-image-hash embedding cache).

**How labels are stored** — Postgres `annotation_state.payload` as **JSONB**, holding the exact
`ImageAnnotations.model_dump(mode="json")` shape. Deliberately not normalised into object/polygon
rows, so `DatasetService`'s public interface didn't change during the Phase 1a migration. One row per
`(dataset_view, image_id)`, unique-constrained, upserted with Postgres `ON CONFLICT` (win-on-latest).
`completed` is denormalised into its own indexed boolean column so progress queries never parse JSON.

**How projects are stored** — There is no "project" entity. The unit is a **dataset view**: 4 hardcoded
keys in `routers/dataset.py`, plus arbitrary paths via `POST /api/dataset/load`. No project table, no
ownership, no per-project settings.

**How images are stored** — Filesystem only. Never in the DB. Served re-encoded as JPEG q=92 through
`GET /api/images/{id}/file`. `image_id` = filename stem (`path.stem`) — **not a hash, not a UUID**.
Two files with the same stem in different views are distinct only because `dataset_view` differs.

**How masks are stored** — Masks are **never persisted as rasters**. Only the extracted polygon
(list of normalised `{x, y}` points) inside the JSONB payload, plus `all_mask_scores`,
`selected_mask_index`, and `confidence` (the SAM2 mask score for mask objects; the ultralytics
`box.conf` for detector-sourced boxes; `0.0` for boxes read from label files). *Note the semantic
overload of one `confidence` field across three different meanings — relevant to Part 3, P-3.*

**How metadata is stored** — Split across four places: `dataset_classes` table (id/name/color/
`safety_critical` per view), `data.yaml`/`classes.txt` on disk (mirrored for YOLO compatibility),
`annotators` table, and the JSONB payload (`width`, `height`, `file_name`, `last_modified`).

**How completed annotations are detected** — `completed` boolean, set by `mark_completed=true` on
save (or by `auto_accept_service.bulk_accept`). It is monotonic within a save:
`annotations.completed = request.mark_completed or annotations.completed` — **once true, a normal save
can never set it back to false.** Export additionally requires *export eligibility*:
`review_service.get_export_eligible_ids()` = grandfathered exempt ∪ images whose **latest** review is
`approved`.

**How annotation versions are managed** — `annotation_history` is append-only: every save writes a
full JSONB snapshot with `action` ∈ {`save`, `mark_completed`} and `annotator_id`. Not FK'd to
`annotation_state.id` so history survives independently. **But:** there is no version *number*, no
diff, no rollback endpoint, and nothing reads `annotation_history` back — it is written and never
queried by any current code path. It is raw material for lineage, not yet lineage.

## Training Pipeline

**One trainer exists, and it trains the pre-labeler, not any deliverable model.**
`backend/app/services/detector_service.py`:

| Aspect | Value |
|---|---|
| Framework | ultralytics YOLOv8 (`from ultralytics import YOLO`) |
| Base weights | `yolov8s.pt`, module constant `MODEL_WEIGHTS` |
| Task | **Detection** (boxes). Uses `obj.bbox`, never `obj.polygon`. |
| Epochs | 100, `patience=20`, `batch=-1` (auto), `device = 0 if cuda else "cpu"` |
| Training data | every image where `get_annotations(id).completed` is true, minus `REJECTED` objects |
| Split | `random.Random(42).shuffle` then 90/10 — **different logic from export's md5 split** |
| Min data | `MIN_TRAINING_IMAGES = 2` |
| Concurrency | `threading.Thread(daemon=True)`, no queue, no GPU-contention check |
| Progress | in-memory `self._jobs` dict + `on_train_epoch_end` callback |
| Output | `models/detector_v{N}.pt` + `detector_registry.json` bump |
| Metrics captured | **none** — staging dir with all ultralytics results is `rmtree`'d in `finally` |
| Gate before activation | **none** — new version is instantly the active model |
| Rollback | **none** |

Two other models run inference but are never trained here: **SAM2.1 hiera_large** (frozen
checkpoint, downloaded by `entrypoint.sh`) and **MobileNetV3-Small** (frozen ImageNet weights, used
only for similarity/novelty embeddings).

**None of the MLflow doc's 8 families is trained by this repo.** `VB-SharedDetector`,
`VB-DefectState`, `VB-P2-Anomaly` (PatchCore), `VB-P2-CrackSeg`, `VB-P3-WheelSeg`, `VB-P3-Fastener`,
`VB-CoachType`, `VB-BufferBoundary` — all pipeline-side. There is no PatchCore code, no crack-seg
training, no log-polar unwrap, no fastener slot logic, no coach classifier. The build plan Phase 3
"Deferred" table already says so.

## Deployment Pipeline

- **Mechanism:** `docker compose` (or Coolify **Docker Compose** resource pointed at this repo).
  Two images built from source (`backend/Dockerfile`, `frontend/Dockerfile`), three pulled.
- **Migrations** run automatically in `entrypoint.sh` (`alembic upgrade head`) before uvicorn.
  There is no down-migration or backup step guarding that.
- **Host paths are hardcoded in `docker-compose.yml`**, with an explicit comment explaining why:
  Coolify injects UI env vars into the container but *not* into compose's own `${VAR}` substitution
  pass, so interpolated bind-mount paths silently fell back to defaults. Current bindings:
  `/home/omronix/Component/annotation_dataset`, `…/annotation_tool/exports`, `…/annotation_tool/models`.
- **GPU:** `deploy.resources.reservations.devices` → nvidia, `count: all`. The image is built for
  **aarch64 Grace Blackwell GB10, sm_121, torch 2.7.1+cu128** (Dockerfile comments are explicit), and
  patches out `torch.jit.script` in SAM2's transforms because it segfaults on that aarch64 build.
  `onnxruntime-gpu` is pinned `; platform_machine == "x86_64"` — i.e. **unavailable on the current
  build target**.
- **No CI/CD.** No `.github/`, no test suite, no linter in the build, no smoke test, no health gate.
  `GET /api/health` exists but nothing polls it (no compose `healthcheck` on backend or frontend).
- **No model deployment concept at all.** "Deploy" here means redeploying the whole app. Model
  "deployment" is `detector_registry.json`'s `path` field changing.

## Environment Variables

Full set, from `backend/.env.example` + `config.py` + compose:

| Group | Vars |
|---|---|
| Paths | `DATASET_PATH`, `EXPORTS_PATH`, `MODELS_PATH` (+ `state_dir_name` = `.annotation_state`, not env-exposed) |
| SAM2 | `SAM_CHECKPOINT`, `SAM_MODEL_CFG`, `SAM_DEVICE` (auto/cuda/cpu/mps), `SAM_USE_ONNX`, `SAM_ONNX_ENCODER_PATH`, `SAM_ONNX_DECODER_PATH` |
| Server | `HOST`, `PORT`, `CORS_ORIGINS` |
| Processing | `MAX_IMAGE_DIMENSION`, `THUMBNAIL_MAX_DIMENSION`, `POLYGON_EPSILON_RATIO`, `MIN_POLYGON_POINTS`, `MASK_CONFIDENCE_THRESHOLD`, `BATCH_MAX_WORKERS` |
| Propagation | `PROPAGATION_ENABLED`, `SIMILARITY_THRESHOLD`, `PROPAGATION_TOP_K` |
| Logging | `LOG_LEVEL`, `LOG_FILE` |
| Postgres | `POSTGRES_HOST/PORT/USER/PASSWORD/DB` |
| Dashboard | `DASHBOARD_USERNAME`, `DASHBOARD_PASSWORD`, `CLASS_FOLDERS_ROOT` |
| MinIO | `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` (**server-side only; no client uses them**) |

**Not present, and needed for MLflow:** `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`,
`AWS_ACCESS_KEY_ID`/`SECRET_ACCESS_KEY` (MinIO artifact store), `MLFLOW_EXPERIMENT_NAME`,
`GIT_COMMIT`/build stamp.

**Config hygiene note:** hardcoded fallbacks include `postgres_password: "change-me"` in
`config.py` and `MINIO_ROOT_PASSWORD=change-me-8plus` in `.env.example`. Fine for local dev, must not
survive to the H200 host.

## Configuration Files

`backend/.env` (from `.env.example`) → `pydantic-settings` `Settings`, `@lru_cache`'d singleton.
`.env` at root → compose `env_file` for db/minio/review-dashboard. `alembic.ini`. `data.yaml` /
`classes.txt` per dataset view (class map). `models/detector_registry.json` (the hand-rolled model
registry). Frontend: `vite.config.ts`, `tailwind.config.js`, `tsconfig.json`, `nginx.conf`,
`.eslintrc.cjs`. **No YAML config for thresholds** — every ML-relevant constant is a Python module
constant, not configuration:

```
mask_confidence_threshold  0.5     config.py (env-driven)
similarity_threshold       0.85    config.py (env-driven)
propagation_top_k          5       config.py (env-driven)
CONFIDENCE_THRESHOLD       0.95    auto_accept_service.py     ← hardcoded
MIN_AUDIT_SAMPLE           10      auto_accept_service.py     ← hardcoded
MIN_APPROVAL_RATE          1.0     auto_accept_service.py     ← hardcoded
AUDIT_SAMPLE_RATE          0.075   review_service.py          ← hardcoded
LOW_CONFIDENCE_THRESHOLD   0.5     triage_service.py          ← hardcoded
EPOCHS / VAL_SPLIT         100/0.1 detector_service.py        ← hardcoded
VAL_SPLIT                  0.1     export_service.py          ← hardcoded (duplicated constant)
SAFETY_KEYWORD_PATTERN     regex   dataset_service.py         ← hardcoded seed heuristic
```

Every one of these is a value MLflow would want logged as a param or tag. Today none is recorded
with the artifacts it produced.

## Database

Postgres 16-alpine. 5 alembic revisions, 6 tables (`backend/app/models/db_models.py`):

| Table | Purpose | Key columns |
|---|---|---|
| `annotators` | named identity, not auth | `id`, `name` UNIQUE, `role` (`annotator`/`golden_curator`/`system`), `created_at` |
| `annotation_state` | latest payload per image | UNIQUE(`dataset_view`, `image_id`), `payload` JSONB, `completed` (indexed), `updated_by_id` FK, `updated_at` |
| `annotation_history` | append-only audit trail | BigInt pk, `dataset_view`, `image_id`, `payload` JSONB, `action`, `annotator_id`, `created_at` — no FK to state, never deleted |
| `annotation_reviews` | 2nd-reviewer / audit decisions | BigInt pk, `reviewer_id` FK NOT NULL, `decision`, `reason`, `notes`, `created_at` |
| `export_gate_exemptions` | grandfather list | PK(`dataset_view`, `image_id`) — populated **once** by its migration, never written again |
| `dataset_classes` | per-view class map | UNIQUE(`dataset_view`, `class_id`), `name`, `color`, `safety_critical` |

Migration chain: `3dd523697175` (Phase 1a schema) → `c1b7fbb8045a` (annotator role) →
`18d19aa85c3c` (reviews) → `9537304fbc09` (export-gate exemptions snapshot) → `dbc0c570a39a`
(`safety_critical` + keyword seeding).

**Missing, relative to any MLflow-integrated design:** dataset-snapshot table, class-map version
table, golden-set table, training-run table, model-version table, spine-stamp columns, pseudo/
synthetic flag, promotion/approval table.

## Current Automation

Honest inventory of what runs without a human pressing a button:

| Automated | Trigger | Notes |
|---|---|---|
| Dataset auto-load (`side_view`) | app startup | default session only |
| Similarity reindex | app startup + `POST /api/dataset/switch` | background `ThreadPoolExecutor(1)` |
| SAM2 checkpoint download | first boot | `entrypoint.sh` |
| Alembic migrations | every boot | `entrypoint.sh` |
| Propagation onto neighbors | save that *newly* completes an image | background, `_is_untouched` guarded |
| Idle session eviction | next new-session lookup | 8 h TTL |
| Batch mask generation | operator starts a job | `POST /api/batch-process` |

| **NOT automated** | |
|---|---|
| Detector retraining | manual `POST /api/detector/train` — no schedule, no data-count trigger, no cron |
| Export / snapshot | manual `POST /api/export` |
| Auto-accept execution | deliberately manual two-step (preview → explicit ids) |
| Model promotion / rollback | does not exist |
| Anything MLflow | does not exist |
| Alerting / notification | does not exist (no Slack, email, PagerDuty, webhook — anywhere) |
| Drift monitoring | does not exist |
| GPU contention control | does not exist — training and SAM2 inference share the GPU with no coordination |

There is **no cron, no systemd timer, no scheduler, no task queue** (no Celery/RQ/APScheduler/
Prefect) in the entire repo. All background work is bare `threading.Thread` / `ThreadPoolExecutor`
inside the uvicorn process, so **every in-flight job dies on restart** and job status
(`self._jobs` dicts in `batch_service`, `detector_service`, `similarity_service`) is lost with it.

---

# PART 2 — MISSING COMPONENTS

Mapped against `MLflow (1).md` §3's own gap table plus this repo's reality.

## A. MLflow itself (nothing exists)

| # | Component | Doc ref | Present? |
|---|---|---|---|
| A1 | Tracking server + backend store | §1 | ✗ |
| A2 | Artifact store (S3/MinIO/local) wired to MLflow | §1 | ✗ (MinIO container idle) |
| A3 | `mlflow.start_run()` around training | §2.1 | ✗ |
| A4 | Param logging (epochs, lr, batch, imgsz, base weights) | §2.1 | ✗ |
| A5 | Per-epoch + final metric logging | §2.1 | ✗ (metrics deleted with staging dir) |
| A6 | Artifact logging (weights, config, PR/confusion plots) | §2.1 | ✗ |
| A7 | Tags: dataset SHA, manifest version, taxonomy version, coach coverage, machine, job id, git commit | §2.1, §2.5 | ✗ (none of these values even exist) |
| A8 | Model Registry, one registered model per family | §2.2 | ✗ (JSON int counter instead) |
| A9 | Stage lifecycle None→Staging→Production→Archived | §2.2 | ✗ |
| A10 | Promotion-gate script (metric floor, beat-production, probe set, tier strictness) | §2.2 | ✗ |
| A11 | Fine-tune vs full-retrain decision logic | §2.2a | ✗ (always full retrain from `yolov8s.pt`) |
| A12 | Shadow test + comparison run | §2.2b step 4, §6.7 | ✗ |
| A13 | Human verification / sign-off gate on promotion | §2.2b step 5 | ✗ (image-level review exists; model-level does not) |
| A14 | Promote + hot-swap | §2.2b step 6 | ✗ (activation is instant and unguarded) |
| A15 | Post-promotion canary monitoring | §2.2c | ✗ |
| A16 | Automatic rollback + `vb-rollback-events` | §2.2c, §2.3 | ✗ |
| A17 | `vb-production-monitoring` experiment | §2.4 | ✗ |
| A18 | `vb-retrain-cycles` experiment | §5.2 step 8 | ✗ |
| A19 | Reproducibility chain (data + code + params ⇒ any past version) | §2.5 | ✗ |

## B. External automation the doc says you must build (§6)

| # | Component | Doc ref | Present? | Note for this repo |
|---|---|---|---|---|
| B1 | Retrain scheduler (cron + `retrain_schedule.yaml`) | §6.1 | ✗ | no scheduler of any kind exists |
| B2 | "New data arrived" detector, per family, `MIN_RETRAIN_THRESHOLD` | §6.2 | ✗ | the count *is* queryable today: `count(annotation_state where completed and export-eligible)` |
| B3 | Resident multi-model serving loader w/ hot-swap | §6.3 | ✗ | out of this module's scope per build plan Phase 6/7 |
| B4 | FP8/TensorRT engine export | §6.4 | ✗ | pipeline-side; irrelevant to a YOLOv8 pre-labeler |
| B5 | Domain metric eval scripts (mAP50, Dice+length-recall, AUROC, recall@FP) | §6.5 | ✗ | **nothing computes any metric today** |
| B6 | PatchCore purity gate | §6.6 | ✗ | no PatchCore, no confirmed-normal set, no `DefectReviewLog` here |
| B7 | Shadow-mode replay + comparison | §6.7 | ✗ | replay corpus would have to be built |
| B8 | GPU/resource scheduler (inference always wins) | §6.8 | ✗ | **acute here**: `detector.train()` and SAM2 share one GPU with zero coordination |
| B9 | Alerting dispatcher w/ durable-log fallback | §6.9 | ✗ | no notification path exists anywhere |

## C. Missing from this repo's *own* plan (prerequisites for any MLflow lineage)

| # | Gap | Plan ref | Consequence if skipped |
|---|---|---|---|
| C1 | **Class-map versioning** | Phase 5, risk §5 `[HIGH]` | Every MLflow run tag claiming a class map is unverifiable. Already caused the 27-class incident. |
| C2 | **Immutable dataset snapshots + manifest** | Phase 5, Q-C | `dataset_sha` tag has nothing to point at; `exports/` is mutable, so a re-export silently changes what a logged run trained on. |
| C3 | **Frozen golden eval set storage** | Phase 4, Q-D | No probe set ⇒ no metric floor, no beat-production check. The promotion gate is unimplementable without it. Role (`golden_curator`) exists; **table does not**. |
| C4 | **Pseudo/synthetic split flag** | Phase 5 | Cannot assert "propagated labels train-split only". `source=PROPAGATED` currently leaks into `val`. |
| C5 | **Spine stamp** | Phase 1b, risk §5 `[HIGH]` | No coach-type coverage tag; no way to key labels back to physical wheels/coaches. |
| C6 | **Pipeline ingestion endpoint + return path** | Phase 1b, Q-E | Triage tiers 1 stay permanently empty; the loop is one-way. Payload schema still open with the pipeline team. |
| C7 | **Per-family label-count report** | §1 data contract | Nothing feeds label-scarcity planning. |
| C8 | **Golden-set write isolation** | risk §5 `[MEDIUM]` | With no golden table, contamination is not *prevented*, merely *not yet possible*. |

---

# PART 3 — POTENTIAL PROBLEMS (found in existing code, independent of MLflow)

Ordered by severity. All are read-only findings; nothing was changed.

**P-1 `[HIGH]` — Detector registry is global while `DetectorService` is session-scoped.**
`detector_service.py`: `_registry_path = models_dir / "detector_registry.json"`, and `models_dir`
comes from `get_settings().models_dir` — one host path for **all sessions and all four dataset
views**. But `get_detector_service()` returns a per-session instance bound to that session's
`DatasetService`. So training while `buffer` is loaded writes `version: N+1` and a
`classes` list from *buffer*'s class map into the same registry the `side_view` annotators' auto-detect
reads from. `_try_auto_detect()` then runs a buffer-trained model with side_view class names
(`detector.detect(path, self._classes)` filters on `class_id >= len(classes)` only). **Silent
cross-view class-id corruption.** Any MLflow registry design must key model versions by view/family,
and this bug should be fixed before it becomes lineage that says something false.

**P-2 `[HIGH]` — `dataset_key` is an absolute host filesystem path.**
`str(root.resolve())` is the primary key component for `annotation_state`, `annotation_history`,
`annotation_reviews`, `export_gate_exemptions`, `dataset_classes`. Moving from the current aarch64
host to the H200 host, or changing the bind-mount path, **orphans every row** — annotations,
reviews, audit history, grandfather exemptions. There is no rename/migrate path. Migrating hosts is
explicitly on your roadmap, so this will fire.

**P-3 `[MEDIUM]` — `confidence` means three different things in one field.**
`AnnotationObject.confidence` holds: (a) `0.0` for boxes read from `labels/*.txt` — documented as
"no signal, not low confidence"; (b) ultralytics `box.conf` for detector-sourced boxes; (c) the SAM2
**mask** score after `generate_for_object()` **overwrites it**. Two consumers read it as a detector
confidence: `triage_service._low_confidence_tier` (filters `> 0`, averages) and
`auto_accept_service.find_candidates` (`>= 0.95`). After any mask generation runs, both are actually
reading SAM2 mask quality. **Auto-accept can therefore skip human review based on how cleanly SAM2
segmented, not on how sure the detector was about the class.** For a safety-adjacent gate this is the
most consequential correctness issue I found. (Mitigated, not removed, by the `safety_critical`
never-eligible rule and the 10/10 audit requirement.)

**P-4 `[MEDIUM]` — Two different train/val splits.**
`export_service.VAL_SPLIT=0.1` with an md5-of-image_id split; `detector_service.VAL_SPLIT=0.1` with
`random.Random(42).shuffle`. An image can be `train` for the exported dataset and `val` for the local
detector. If both ever feed MLflow-logged runs, "which split did this metric come from" has two
answers.

**P-5 `[MEDIUM]` — Training is unguarded against GPU contention.**
`start_training()` spawns a daemon thread immediately; a 100-epoch YOLOv8 run with `batch=-1`
(auto-batch, which probes for maximum memory) competes with the process-wide SAM2 singleton serving
live annotators. No busy-check, no memory reservation, no queue, and multiple sessions can each start
a training run concurrently. This is exactly `MLflow (1).md` §6.8's problem, present today.

**P-6 `[MEDIUM]` — All job state is in-process and lost on restart.**
`batch_service._jobs`, `detector_service._jobs`, `similarity_service._jobs` are plain dicts. A
redeploy mid-training loses the job, the progress, and (via the `finally: rmtree`) the partially
trained staging dir. `_jobs` also grows unbounded — no eviction.

**P-7 `[LOW-MEDIUM]` — Metrics are computed then deliberately destroyed.**
`_run_training`'s `finally: shutil.rmtree(staging_dir, ignore_errors=True)` deletes the entire
ultralytics run directory: `results.csv`, PR curve, confusion matrix, `args.yaml`, `last.pt`. Only
`best.pt` survives (copied just before). Every artifact MLflow §2.1 asks you to log is produced and
thrown away. **The cheapest possible first win is to log them instead.**

**P-8 `[LOW-MEDIUM]` — Export gate has an unbounded grandfather hole.**
`export_gate_exemptions` was snapshotted once by its migration. `review_service.get_export_eligible_ids`
returns `exempt ∪ latest-approved`. Exempt images bypass review **forever**, and
`get_pending_second_review` filters them out, so they are invisible in the pending list. Defensible
product decision (documented), but any MLflow run trained on a snapshot containing exempt images
cannot claim "all data second-reviewed."

**P-9 `[LOW]` — `completed` is effectively one-way.**
`annotations.completed = request.mark_completed or annotations.completed`. There is no un-complete
endpoint. A rejected review does *not* clear `completed`; it only removes export eligibility. So
`get_dataset_info()`'s progress percentage counts rejected work as done.

**P-10 `[LOW]` — Audit sample can never grow past its own exclusion.**
`get_audit_sample` computes `target_n = max(1, round(len(propagated) * 0.075))` but samples from
`pool = propagated - already_audited` with a **fixed seed 42**. As `already_audited` grows the pool
shrinks while `target_n` grows, and the fixed seed means the sequence isn't independent across calls.
Reaching a true 7.5 % coverage over time works, but the sampling is not statistically clean.

**P-11 `[LOW]` — `_estimate_eta` uses `last_modified` as a completion timestamp.**
Any later edit to a completed image moves that timestamp, so the ETA is derived from edit recency,
not completion cadence.

**P-12 `[LOW]` — No tests, no CI, no linting in the build.**
Zero test files in the repo. For code that gates safety-critical annotation, and about to gate model
promotion, this is a structural risk more than a style complaint.

**P-13 `[LOW]` — Global exception handler returns opaque 500s.**
`main.py`'s `@app.exception_handler(Exception)` returns `{"detail": "Internal server error"}`. Good
hygiene, but the recently fixed "silent failure on Generate masks" bug (commit `36933ee`) is the kind
this masks. An MLflow-driven automation layer needs machine-readable failure reasons, not opaque 500s.

---

# PART 4 — RISKS

## Risks from the ask itself

**R-1 `[BLOCKING]` — Q-C contradiction.** Your build plan locked "no direct MLflow writes from this
module"; your request is to integrate this module into MLflow. Building the wrong branch wastes the
work and, worse, could create a **second registry** — which the plan explicitly forbids ("Do not
stand up a second registry"). **Do not start until Decision D1 is made.**

**R-2 `[HIGH]` — The MLflow doc describes a system that is not in this repo.**
`pipeline.md`, `FINAL_AIML_ARCHITECTURE.md`, `component_defect_taxonomy.yaml` are absent. Every metric
floor, `gate_tiers` assignment, taxonomy version, and the entire 8-family table is **unverifiable
from here**. If I implement the doc's promotion gate against the local YOLOv8 pre-labeler, I am
applying pipeline-model policy to a different model class. I will not silently do that.

**R-3 `[HIGH]` — Hardware target mismatch, two ways.**
(a) The doc is written for **DGX Spark, 128 GB unified memory, 8 resident models**. You specified
**H200, 141 GB VRAM, 24 vCPU, 240 GB RAM, 5 TB scratch**. Different memory model (discrete VRAM vs
unified), different contention math — every §6.8 conclusion needs re-deriving.
(b) The current image is built for **aarch64 GB10 / sm_121** with `torch 2.7.1+cu128`, a SAM2
`torch.jit.script` patch for an aarch64 bug, and `onnxruntime-gpu` excluded on non-x86_64. An H200
host is almost certainly **x86_64**. The Dockerfile will need revisiting (likely: keep cu12x, drop
the aarch64-specific patch guard, re-enable onnxruntime). Not hard, but it *is* a migration, and P-2
(path-keyed DB rows) fires at the same time.

**R-4 `[HIGH]` — "Mandatory human sign-off" has no identity system to hang on.**
`MLflow (1).md` §2.2b step 5 requires human approval before Production for safety and structural
tiers. This repo's identity is a **self-declared name with no authentication** and **ungated role
changes** (`routers/annotator.py`: "Role changes here aren't gated by anything (no real auth
exists)"). A promotion approval recorded against that identity is not an audit trail that survives an
incident review. Either real auth arrives, or sign-off lives in the pipeline's MLflow with the
pipeline team's identity system.

**R-5 `[MEDIUM]` — Single-host SPOF, restated for the H200.**
Postgres, MinIO, backend, GPU, and (if D1 chooses local) the MLflow tracking server + artifact store
all on one box. `MLflow (1).md` §6.3 flags "single device = SPOF"; the annotation build plan
`[HIGH]`-flags GPU sharing. Adding an MLflow server plus training jobs plus 141 GB of resident models
to the same host needs a deliberate memory/backup plan, not just more services in the compose file.

**R-6 `[MEDIUM]` — Alembic runs on every boot with no backup.**
`entrypoint.sh` → `alembic upgrade head` → uvicorn. Adding snapshot/class-map/golden-set tables means
new migrations against live annotation data with no pre-migration dump step. The `export_gate_exemptions`
migration already proves migrations here *write data*, not just schema.

**R-7 `[MEDIUM]` — Auto-accept + propagation is a compounding-error path into MLflow lineage.**
Propagation copies labels onto ≤5 neighbors ≥0.85 similarity; auto-accept can bulk-complete frames
using a confidence field that may hold SAM2 mask score (P-3); both then feed export → snapshot →
(future) MLflow run. The plan's `[HIGH]` "propagation-driven error compounding" risk is real and the
7.5 % audit is the only brake. Any MLflow snapshot manifest **must** carry propagated/auto-accepted
counts so a bad run is diagnosable after the fact.

**R-8 `[LOW-MEDIUM]` — Class-map drift will invalidate old runs.**
`add_class()` appends and rewrites `data.yaml`. Old exports and old `detector_registry.json` entries
hold stale class lists. With no version, a future MLflow run tagged with "the class map" is tagged
with a moving target. This has already happened once (27-class remap).

---

# PART 5 — SUGGESTIONS & RECOMMENDED ARCHITECTURE

## Design principles I recommend locking first

1. **MLflow stays passive.** Exactly as the doc says: record-keeper + gate API. No scheduling, no
   watching, no serving inside MLflow. Everything active lives in a new `automation/` package.
2. **Prerequisites before tracking.** Logging runs whose tags point at a *mutable* `exports/` dir and
   an *unversioned* class map produces authoritative-looking lies. **Snapshot + class-map version +
   golden set come before `mlflow.start_run()`.** This is the biggest sequencing recommendation in
   this document.
3. **Two model scopes, kept apart.**
   - **Scope A — the annotation pre-labeler** (`detector_v{N}.pt`). Local, this repo's, low blast
     radius, cosmetic-tier by nature. This is what MLflow can track *here*, today, cheaply.
   - **Scope B — the 8 VB families.** Pipeline-owned. This module supplies **datasets + golden set**
     only, per Q-C and Phase 6/7.
   Conflating them is how a second registry gets built by accident.
4. **Datasets are first-class MLflow citizens.** Snapshot id, class-map version, split integrity,
   propagated/auto-accept counts, annotator+reviewer ids, golden-set version — the manifest is the
   handoff contract, whether the pipeline team imports it or this module logs it.
5. **Nothing auto-promotes.** Preserve the existing repo instinct (auto-accept is preview→explicit;
   propagation only touches untouched images). Promotion gets the same treatment.
6. **Fail loud.** Per §6.5/§6.9: an eval crash holds the candidate at `None`, never defaults to pass
   or fail; alerting has a durable local-file fallback.

## Recommended target architecture

```
┌─ ANNOTATION TOOL (this repo, mostly unchanged) ──────────────────────────┐
│ SAM2 assist · triage · propagation · 2nd review · audit · auto-accept    │
│ Postgres: annotation_state / history / reviews / classes                  │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ POST /api/export  (existing, extended)
                                ▼
┌─ NEW: dataset versioning layer ─────────────────────────────────────────┐
│ class_map_versions      immutable rows, content-hashed                   │
│ dataset_snapshots       snapshot_id = sha256(sorted manifest)            │
│ golden_sets             SEPARATE table + SEPARATE bucket, curator-only   │
│ manifest.json           snapshot_id · class_map_version · counts ·       │
│                         split integrity flags · lineage · golden ref     │
│              writes to ──► MinIO  (automation-minio-1, already running)  │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                ▼                                 ▼
┌─ validation gate ─────────┐      ┌─ HANDOFF (Q-C branch) ───────────────┐
│ class-map pinned?         │      │ pipeline team imports snapshot into   │
│ pseudo/synth ∉ val/test?  │      │ THEIR MLflow as a dataset artifact    │
│ golden ∩ train = ∅?       │      │ → they run the 8 families' retrain    │
│ per-class counts ≥ floor? │      └───────────────────────────────────────┘
│ FAIL ⇒ refuse, alert      │
└──────────┬────────────────┘
           │ PASS
           ▼
┌─ NEW: local MLflow (Scope A only — the pre-labeler) ────────────────────┐
│ experiments: annot-detector-training · annot-dataset-snapshots ·        │
│              annot-prelabeler-monitoring · annot-retrain-cycles         │
│ registry:    AnnotDetector-<view>   None→Staging→Production→Archived    │
│ artifacts:   MinIO bucket via MLFLOW_S3_ENDPOINT_URL                    │
└──────────┬──────────────────────────────────────────────────────────────┘
           │
┌─ NEW: automation/ (all the active parts MLflow will never do) ──────────┐
│ data_watcher.py     ready? new completed+eligible count per view        │
│ gpu_scheduler.py    inference always wins; no train while SAM2 busy     │
│ snapshot_service.py build + hash + upload + manifest                   │
│ validator.py        the gate above                                      │
│ train_runner.py     wrap detector training in mlflow.start_run()        │
│ eval_runner.py      mAP50/precision/recall on the frozen golden set     │
│ promotion_gate.py   floor → beat-Production → human approve → stage move│
│ prelabeler_monitor.py  correction rate, auto-accept rate, audit error   │
│ rollback.py         stage-move back + alert + vb-style rollback event    │
│ alerting.py         Slack/email + DURABLE LOCAL FILE fallback           │
│ retrain_trigger.py  cron entry; reads retrain_schedule.yaml             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Phase 4 of your brief — every stage of the automation pipeline, explained

**Stage 1 — Annotation Tool.** Unchanged. Human + SAM2 produce `annotation_state` rows;
`annotation_history` records every save; `annotation_reviews` records sign-off. Trust boundary stays
where it is: only `completed` **and** export-eligible images ever leave. *Change needed:* record
per-object provenance well enough that a snapshot can report propagated/auto-accepted counts (the
data exists — `source`, `propagated_from_image_id` — it just isn't aggregated anywhere).

**Stage 2 — Dataset Versioning.** `POST /api/export` stops writing into a mutable `exports/`. It
builds a snapshot: content-hash every image + label, sort, `snapshot_id = sha256(manifest)`, write
under `snapshots/<snapshot_id>/` in MinIO, and insert an immutable `dataset_snapshots` row. The
manifest pins `class_map_version` (new immutable `class_map_versions` row, created on any class
change — never edited in place), the split map, per-class counts, annotator/reviewer ids,
propagated/auto-accepted counts, and the golden-set version it must be evaluated against. This is
the fix for C1, C2, R-8, and the plan's two `[HIGH]` risks.

**Stage 3 — Dataset Validation.** Refuse to proceed unless: class map pinned and matching the tool's
current map; **no pseudo/synthetic (propagated) or auto-accepted label in `val`/`test`** (C4 — needs
the split logic to become provenance-aware); golden-set ids **disjoint** from every split (C8);
per-class counts above a floor; every included image export-eligible. Failure = refuse + alert with
the offending ids. Never auto-drop data (same reasoning as §6.6's "suspect list, not automatic
removal").

**Stage 4 — Training Queue.** Persistent queue (Postgres table, not an in-process dict — fixes P-6).
Entries are `(view, snapshot_id, mode)`. `gpu_scheduler.py` gates dequeue: **inference always wins**
(§6.8). Concretely for the H200: check whether SAM2 is serving / a batch job is active before
starting; preempt training, never inference. With 141 GB VRAM a memory-budget-slice model becomes
viable *once measured*, but the busy-check/preempt model is the conservative default until it is.
Serialise per GPU so two sessions can't both train (fixes P-5).

**Stage 5 — Training Pipeline.** `detector_service` keeps its logic; a wrapper adds: `mlflow.start_run()`,
param logging (epochs, patience, batch, imgsz, base weights, split logic), per-epoch metric logging
via the existing `on_train_epoch_end` callback, and — **crucially — artifact logging *before* the
`finally: rmtree`** (fixes P-7: `results.csv`, PR curve, confusion matrix, `args.yaml`, `best.pt`).
Tags: `snapshot_id`, `class_map_version`, `dataset_view`, `golden_set_version`, `git_commit`,
`trigger_job_id`, `machine`, `mode` (`full_retrain`/`fine_tune`). Fine-tune vs full-retrain per
§2.2a, adapted: class-map change ⇒ FULL_RETRAIN (non-negotiable — it *is* the label-space change the
doc warns about); else ratio/interval rules with thresholds left **unset until real numbers exist**,
exactly as the doc insists.

**Stage 6 — MLflow Tracking.** Passive. Every candidate is logged **unconditionally** — a rejected
candidate is an audited data point, not a wasted run (§2.2b step 1). Backend store: Postgres (reuse
the existing db service, separate database). Artifact store: MinIO (already running, currently idle).

**Stage 7 — Evaluation.** `eval_runner.py` scores the candidate on the **frozen golden set** —
which must be built first (C3). Metrics for the pre-labeler: per-class precision/recall/mAP50.
Standardised JSON + provenance (`golden_set_version`, `snapshot_id`, `eval_timestamp`) so the
composite comparison's recency tie-break has something concrete. **Per-class, never aggregate** —
the plan's `[MEDIUM]` risk "aggregate-mAP hides per-class regression" and the doc's §2.2 both demand
this. An eval crash holds the candidate at `None` and alerts; it never defaults either way (§6.5).

**Stage 8 — Model Registry.** One registered model per `(view)` for Scope A:
`AnnotDetector-side_view`, etc. Stage lifecycle `None→Staging→Production→Archived`. **Registering
per-view fixes P-1's cross-view collision** by construction. Archived versions never deleted.

**Stage 9 — Approval.** Metric floor → beat current Production per-class → human sign-off. Given
R-4 (no real auth), I recommend sign-off happen through MLflow's own UI/API with a real account, and
the tool merely link to it — rather than pretending the name-only identity is an approval record.
Tier strictness follows `safety_critical`: a snapshot containing safety-critical classes gets the
strictest path, no auto-approve, ever.

**Stage 10 — Deployment (Scope A only).** "Deploy" = `detector_registry.json` (or its replacement)
pointing at the Production version, plus dropping the in-memory `_loaded_model` so the next
`detect()` reloads. Keep the previous version resident briefly for instant rollback, per §6.3's
never-swap-blind rule: load, sanity-check on a known probe image, *then* flip the pointer, and never
end up with zero model loaded. **Scope B deployment (TensorRT/engine/hot-swap) stays pipeline-side —
build plan Phase 6/7, not this module.**

**Stage 11 — Monitoring.** New experiment logging the plan's own §4.6 metrics per class per cycle:
auto-accept rate ↑, correction rate ↓, audit error rate ≤ target, human-touch rate ↓ (flat for safety
classes), golden-set per-class scores. **The plan's alarm condition — "auto-accept rate rises while
golden safety scores fall ⇒ STOP" — should be an implemented check, not a sentence in a document.**
All of these are computable *today* from `annotation_history` + `annotation_reviews`, which is why
this stage is cheaper than it looks.

**Stage 12 — Rollback.** Stage-move Production→Archived, previous→Production, reload the pointer,
log a rollback event with the triggering metric, magnitude, and failing samples, and alert on the
urgent channel with a durable-file fallback (§6.9 — the one place "fail silently" is categorically
unacceptable). For Scope A a rollback is cheap and safe; wire it early rather than as a phase-N
afterthought.

---

# PART 6 — IMPLEMENTATION PLAN (Phase 5 of your brief)

**Nothing below is built. Nothing will be built until you approve.**

## Decisions needed before Milestone 1

**D1 `[BLOCKING]` — MLflow ownership.** Which is true?
 (a) **Q-C stands** — this module stages snapshots to MinIO; pipeline team imports into their MLflow.
     Build M1–M4 + M9 only. No MLflow dependency in this repo at all.
 (b) **Q-C is superseded** — this module gets read+write to the pipeline's MLflow. Needs the URI,
     credentials, network reachability, and an agreed experiment/model naming convention **from the
     pipeline team** before any code.
 (c) **Local MLflow for Scope A only** (my recommendation as an interim) — stand up MLflow in this
     compose stack to track *the annotation pre-labeler*, while datasets still hand off per Q-C.
     Gives the full tracking/registry/gate/rollback loop, on a low-blast-radius model, with no
     dependency on the pipeline team, and no second registry for the 8 families.

**D2** — Confirm the 8 VB families are **out of scope** for this repo (I believe they are; both docs
say so). If they are in scope, this is a much larger programme and needs `pipeline.md` +
`FINAL_AIML_ARCHITECTURE.md` + `component_defect_taxonomy.yaml` supplied first.

**D3** — H200 host: x86_64 confirmed? (Dockerfile rebuild + P-2 dataset-path migration land together.)

**D4** — Who curates the golden set, by name? Still open as Q-D in your own plan. Golden set is on
the critical path for every gate; without it, M5–M8 cannot start.

## Milestones

Sequenced so each is independently valuable and nothing depends on an unmade decision.

---

### M0 — Fix the three pre-existing correctness bugs *(no MLflow, do first)*
**Purpose:** don't build lineage on top of known-wrong data.
**Files to modify:** `detector_service.py` (P-1: key registry per dataset view),
`annotation_state_repo.py` + a migration (P-2: introduce a stable `dataset_view` key, path→key map),
`schemas.py` + `mask_generation_service.py` + `auto_accept_service.py` + `triage_service.py` (P-3:
split `detector_confidence` from `mask_confidence`).
**Files to create:** one alembic revision; a backfill for the confidence split.
**Dependencies:** none. **Risks:** P-3 touches the auto-accept gate — highest-care change in the repo;
P-2 is a data migration and needs a DB dump first (R-6).
**Testing:** reproduce P-1 by training under `buffer` then auto-detecting under `side_view`; assert
per-view isolation. For P-3, assert an auto-accept candidate set computed before/after mask
generation is identical (it currently is not).
**Outcome:** the data MLflow will describe is actually what it claims.

---

### M1 — Class-map versioning
**Purpose:** C1. Nothing else can be honestly tagged without it.
**Create:** `class_map_versions` table + migration; `automation/class_map.py` (hash, create version,
resolve current). **Modify:** `dataset_service.add_class` / `_persist_class_names` to mint a new
version instead of editing in place; `routers/dataset.py`.
**Dependencies:** M0's migration ordering. **Risks:** `add_class` is already the site of a
concurrency bug the team patched (`_get_class_list_lock`) — re-touching it needs care.
**Testing:** concurrent `add_class` from two sessions ⇒ two versions, no lost class; old exports
still resolve their historical version.
**Outcome:** every dataset and model can name an immutable class map.

---

### M2 — Immutable dataset snapshots + manifest, to MinIO
**Purpose:** C2, and the `dataset_sha` MLflow tag becomes meaningful.
**Create:** `dataset_snapshots` table + migration; `automation/snapshot_service.py`;
`automation/manifest.py` (schema + validation); MinIO client wiring (add `boto3`/`minio` to
requirements — **the first new dependency**); `MLFLOW_S3_*`/`MINIO_*` client env vars.
**Modify:** `export_service.py` (write to a content-addressed path, emit a manifest, stop overwriting),
`routers/export.py`, `.env.example`, `docker-compose.yml` (backend needs MinIO creds).
**Dependencies:** M1. **Risks:** `exports/` is consumed by the Flask review dashboard
(`CLASS_FOLDERS_ROOT`) — do not break it; keep a compatibility path.
**Testing:** export twice with no annotation change ⇒ identical `snapshot_id`; change one label ⇒
different id; old snapshot bytes unchanged; manifest round-trips.
**Outcome:** the Phase 5 handoff artifact your own plan specified.

---

### M3 — Provenance-aware split + validation gate
**Purpose:** C4, C8, and the plan's `[MEDIUM]` split-integrity mandate.
**Create:** `automation/validator.py`. **Modify:** `export_service._split_for` to force
propagated/auto-accepted images into `train` only; manifest to carry the flags and counts.
**Dependencies:** M2. **Risks:** changes existing split assignments for some images — must be logged
loudly, not silently (a re-split changes what a previously exported dataset meant).
**Testing:** assert no `source=PROPAGATED` and no auto-accepted image in `val`; assert validator
refuses a golden/train overlap.
**Outcome:** "pseudo/synthetic never in valid/test" becomes enforced, not trusted.

---

### M4 — Golden eval set storage (curator-gated, structurally separate)
**Purpose:** C3. Blocker for every gate downstream. **Needs D4 answered.**
**Create:** `golden_sets` + `golden_set_items` tables + migration; `automation/golden_set.py`;
`routers/golden.py` with `golden_curator` role enforcement; **separate MinIO bucket**, not a flag on
the shared one (the plan is explicit: "structurally separate storage, not a flag on shared storage").
**Modify:** `review_service`/`export_service` to exclude golden ids from every export split.
**Dependencies:** M2, M3. **Risks:** the plan's `[MEDIUM]` contamination risk — the *only* mitigation
that counts is that no propagation, triage, or export path can write here. Assert it in tests.
**Testing:** attempt a golden write as a plain annotator ⇒ 403; attempt propagation into a golden
image ⇒ refused; frozen snapshot is byte-identical after a curation round on a *new* version.
**Outcome:** a ruler that nothing trains on.

---

### M5 — MLflow tracking server + first tracked training run *(needs D1)*
**Purpose:** A1–A7. The first actual MLflow code.
**Create:** `mlflow` service in `docker-compose.yml` (Postgres backend store, MinIO artifact store);
`automation/train_runner.py`; `MLFLOW_TRACKING_URI` + `MLFLOW_S3_ENDPOINT_URL` env.
**Modify:** `requirements.txt` (+`mlflow`), `detector_service._run_training` — **log artifacts before
`finally: rmtree`** (P-7), log params/per-epoch metrics/tags.
**Dependencies:** D1, M1, M2. If D1=(a), skip this milestone entirely.
**Risks:** adding a service + a heavy dependency to a single-host stack (R-5); `mlflow` pulls a large
dependency tree — pin it and check for conflicts with the `torch 2.7.1+cu128` / ultralytics pins,
which are already delicate on this build.
**Testing:** one training run appears with all params/metrics/artifacts/tags; kill the run mid-way
and confirm it lands as `FAILED`, not silently absent.
**Outcome:** every pre-labeler training run is reproducible from data + code + params.

---

### M6 — Eval on the golden set + metric logging
**Purpose:** A5, B5. **Create:** `automation/eval_runner.py`, `eval/` per-metric modules.
**Dependencies:** M4, M5. **Risks:** must fail loud, never default to pass/fail (§6.5).
**Testing:** known-good/known-bad checkpoints produce expected per-class numbers; a deliberate crash
leaves the candidate at `None` + alert fired.
**Outcome:** numbers a gate can act on.

---

### M7 — Model registry + promotion gate + rollback
**Purpose:** A8–A10, A13–A16, A19.
**Create:** `automation/promotion_gate.py`, `automation/rollback.py`, `automation/alerting.py`
(with durable-file fallback per §6.9). **Modify:** `detector_service` model activation to read
Production stage from the registry instead of `detector_registry.json`; keep the JSON as a
read-through cache during transition.
**Dependencies:** M5, M6, and R-4 resolved (who signs off, with what identity).
**Risks:** this is where a wrong design silently promotes a bad model. Recommend **no auto-approve
for any class in the first iteration**, regardless of tier — tighten later, never loosen first.
**Testing:** candidate below floor ⇒ stays `Staging`, reason logged; candidate that regresses one
class while improving aggregate ⇒ **rejected** (the `[MEDIUM]` risk, as an explicit test); rollback
restores the previous Production and fires the urgent alert; alerting with the transport down still
writes the durable log.
**Outcome:** the doc's promotion state machine, running on Scope A.

---

### M8 — Scheduler, data watcher, GPU scheduler, monitoring
**Purpose:** B1, B2, B8, A17, A18; fixes P-5, P-6.
**Create:** `automation/retrain_trigger.py`, `retrain_schedule.yaml`, `automation/data_watcher.py`,
`automation/gpu_scheduler.py`, `automation/prelabeler_monitor.py`, a persistent
`training_jobs` table + migration, cron/systemd unit.
**Dependencies:** M7. **Risks:** an automated trigger on a shared GPU is exactly §6.8's hazard —
inference must always win; do not force a run through. Leave `TAU_DRIFT`,
`FINE_TUNE_RATIO_CEILING`, `FULL_RETRAIN_MAX_INTERVAL_DAYS`, `MIN_RETRAIN_THRESHOLD` **unset** until
real numbers exist, per the doc's own insistence — but log the skip reason every cycle so a silent
multi-day gap is visible.
**Testing:** trigger fires on interval and no-ops otherwise; below-threshold data ⇒ skip + logged
reason; training refuses to start while SAM2 is serving; job survives a backend restart.
**Outcome:** the loop runs without a human remembering to start it.

---

### M9 — Handoff contract + feedback closure *(joint with pipeline team)*
**Purpose:** build plan Phase 6/7; C6, C7.
**Create:** documented snapshot-manifest schema for the pipeline team; per-family label-count report;
ingestion endpoint **only once Q-E's payload schema is answered** (still open with them).
**Dependencies:** M2–M4, plus the pipeline team. **Risks:** designing against a guessed contract —
your plan already flags this as why Phase 1b is blocked. Don't build it on assumption.
**Outcome:** the loop closes; triage tiers 1 stop being permanently empty.

---

## Suggested order

```
M0 ──► M1 ──► M2 ──► M3 ──► M4 ──┬──► M5 ──► M6 ──► M7 ──► M8
 (bugs) (classmap)(snapshot)(split)(golden) │   (needs D1)
                                            └──► M9 (joint, parallel from M4)
```

M0–M4 are **valuable regardless of D1** — they are your own build plan's Phase 4/5 deliverables and
they are the prerequisites that make any MLflow tag truthful. **My recommendation: approve M0–M4 now,
decide D1 in parallel, and start M5 only once D1 is settled.**

---

# Uncertainties I am not resolving by assumption

1. **D1 (Q-C vs your request).** Stated, not decided. Blocks M5+.
2. **`pipeline.md` / `FINAL_AIML_ARCHITECTURE.md` / `component_defect_taxonomy.yaml` are absent.**
   Every tier assignment, metric floor, and the 8-family table in `MLflow (1).md` is unverifiable
   from this repo. I have not invented values for them.
3. **DGX Spark (doc) vs H200 (your brief).** Different memory architecture; §6.8's conclusions need
   re-deriving. I flagged it rather than silently re-specifying.
4. **aarch64 GB10 image vs presumed x86_64 H200.** Dockerfile/wheel/ONNX implications, plus P-2's
   path-keyed rows, fire together on migration.
5. **Who approves a promotion, with what identity** (R-4). No auth exists here today.
6. **Q-D: named golden-set curators.** Open in your own plan; blocks M4.
7. **Q-E payload schema.** Open with the pipeline team; blocks the ingestion half of M9.
8. **Whether `exports/class_folders/` (the Flask dashboard's tree) is still an active workflow.**
   Nothing in the FastAPI backend writes it; I could not determine what populates it. If it is live,
   M2 must not break it.

---

*Report produced by static analysis of every relevant file in the repository. No file was modified.
No code was written. Awaiting approval of the implementation plan and answers to D1–D4.*
