# Annotation Module — Build Plan

**A separate upstream SAM2 labeler that feeds the Vande Bharat (VB) retraining loop**

This module is **not** the inspection pipeline and does **not** run inference, retraining,
or model deployment. Those are owned by the VB AI/ML unit specified in
`FINAL_AIML_ARCHITECTURE.md` (the "pipeline" below). This module is the **human-in-the-loop
data supply line**: it turns pipeline-flagged frames into versioned, QA'd, class-map-locked
datasets and hands them to the pipeline's MLflow-backed retrain loop (pipeline §12).

- **Consumer:** the VB inspection pipeline (RunPod L4 / Triton) — via MLflow dataset artifacts
- **Source:** pipeline-emitted frames (zones P1 side / P2 undercarriage / P3 bogie-wheel)
- **Loop:** capture → **annotate (this module)** → dataset snapshot → retrain (pipeline) → shadow/A-B promote (pipeline) → recapture
- **Boundary:** this module ends at *"registered, versioned dataset + golden eval set in MLflow."* Everything downstream of that is the pipeline's.

> **Build scope decisions (locked with product owner):**
> 1. **Separate upstream labeler** — distinct from the pipeline's post-inference *review console* / `DefectReviewLog` (pipeline §9, §13). Two systems, one defined handoff.
> 2. **Base codebase = the existing E:\ SAM2 tool.** Harden it; do not rebuild in the pipeline repo.
> 3. **Label scope THIS build = detection boxes + SAM2 polygon segmentation only.** Confirmed-normal curation (PatchCore), crop-classify, fastener slot-occupancy, and wheel log-polar seg are **architecture-required but explicitly deferred** — see Phase 3 "Deferred".
> 4. **Promotion-gate design = engineer recommendation** — see Phase 6.
>
> **Unverified at time of writing:** the E:\ tool's source files referenced below
> (`session_context.py`, `sam_service.py`, `similarity_service.py`, `propagation_service.py`,
> `export_service.py`, `detector_service.py`) were **not locatable on the E:/D: drives during
> this pass** — only this plan, sample videos, and `frame.py` were present. Their behaviour is
> taken from this plan's own descriptions. **Confirm the tool's actual location/repo before
> Phase 1** so hardening targets real code, not a description. (Open question Q-A, §6.)

---

## 0. The loop this module serves

The existing tool is a single-user, single-machine SAM2-assisted labeler. The ask is to
turn it into the **data engine** for the VB pipeline's retrain loop (pipeline §12:
*review-approved → dataset → retrain → MLflow → OTA → shadow canary → A/B promote*).
Every decision below is judged against one question: does it make that loop faster, safer,
and self-improving — or just make labeling more convenient in isolation?

```
VB PIPELINE (RunPod L4 / Triton inference, cloud)
   │  low-confidence detections  +  gate-recall-audit misses (pipeline R6)
   │  +  field-flagged frames     +  routine sampled frames
   ▼                                          [spine-stamped, per pipeline §9]
Ingestion & triage  ──dedup + prioritize──▶  Annotation queue
   (THIS MODULE)                                   │  SAM2-assisted box+mask
                                                   ▼
                                              Human review + 2nd-reviewer QA
                                                   │  QA sign-off (completed-only)
                                                   ▼
                              Versioned dataset snapshot  +  frozen per-class golden set
                                   (class-map locked, exclude_classes:[leakage])
                                                   │  registered as MLflow dataset artifact
                          ══════════════ HANDOFF ══════════════
                                                   ▼
VB PIPELINE:  retrain (shared backbone + zone heads / crack-seg)  →  MLflow registry
                                                   │  offline eval vs golden set (per-class)
                                                   ▼
                              shadow-mode canary  →  A/B promote  →  TensorRT engine on L4
                                                   │
                                                   └────── per-frame confidence + field flags ──▶ back to triage
```

> **What is NOT in this module** (owned by the pipeline, do not reimplement): ROI gate cascade,
> gated SAHI, TensorRT/Triton serving, temporal k-of-n voting, the ≤10-min alert path, and the
> retrain/shadow/A-B promotion mechanism itself. This module *produces the data those consume
> and the golden set that gates them.*

---

## 1. Data contract with the VB pipeline

Lock what crosses the boundary each way **before** touching annotation infra. Field names
below bind directly to the pipeline's upload envelope and detection record (pipeline §9) —
do not invent parallel names.

### Pipeline → Annotation (frames in)

| Field | Source in pipeline §9 | Why it's needed |
|---|---|---|
| `coach_index`, `coach_type`, `axle_id`, `side`, `view` (entry/exit) | detection record / spine (pipeline §4) | The **coordinate spine stamp**. Must ride with every annotation or entry/exit fusion & voting (pipeline §7) cannot key back. Not `frame_id` alone. |
| `longitudinal_position_mm` | encoder counts (pipeline §4) | Deterministic position, not wall-clock. Dedup and traceability. |
| `zone` (P1/P2/P3) | detection record | Selects which model family the label feeds (pipeline §5). Existing per-view split already keys on this. |
| Model version + per-box `conf` | detection record | Drives triage priority (Phase 2) — the single highest-leverage field in the loop. |
| Existing detection boxes | detection record | Reused as SAM2 box prompts (existing detection→segmentation flow). |
| `dropped_regions` | upload envelope | QA-invalid areas — never label inside them. |
| Field-flagged frames + **gate-recall-audit misses** (pipeline R6, `audit_full_sahi_every_n_trains`) | review console / audit | Hard negatives + defects the gate missed — highest-priority tier. |

### Annotation → Pipeline (datasets out — **not models**)

> The plan's earlier "export weights to edge hardware" direction was wrong for this
> architecture: the pipeline runs a **single TensorRT engine on a cloud L4 via Triton**,
> with **no inference on the edge node** (pipeline C1, §2, §8). This module ships **data**,
> never weights or engines.

| Artifact | Why it's needed |
|---|---|
| Immutable **dataset snapshot ID** + **class-map version** | Class-ID drift already bit this project (27-class remap; and pipeline's `exclude_classes:[leakage]` synthetic rule). Must be versioned in MLflow, never silently mutated (Phase 5, risk §5). |
| **Frozen per-class golden eval set** (separate storage) | The offline pre-canary promotion gate (Phase 6). Curated by domain experts, never touched by propagation/pseudo-label. |
| Split manifest — **pseudo/synthetic labels flagged train-split-only** | Pipeline §12 mandates pseudo/synthetic never enter valid/test. Enforced at export, not on trust. |
| Per-family label-count report | Feeds pipeline R10 (label scarcity) planning. |

---

## 2. Build phases

### Phase 1 — Ingestion & multi-tenant hardening `Foundation`

The tool today is a local single-session app (`session_context.py` scopes SAM2, similarity
index, and batch jobs to one in-memory bundle). Right shape for one annotator; wrong shape
for a continuous pipeline-fed queue and a multi-annotator team.

Recent commits (`e97da04`, `b9d07f8`, `a862eac`) have been reactively patching this exact
gap — per-session bleed, image loads breaking, SAM2 races — one bug at a time via
`SessionBundle`. That's useful raw material but not a design. Split into two sub-phases so the
half with a real dependency doesn't block the half that doesn't:

**Phase 1a — Tool hardening (build now, no external dependency):**
- Move per-image annotation state out of per-dataset JSON (`.annotation_state/*.json`,
  `dataset_service.py`) into **Postgres**. Doesn't need to wait for spine-stamp ingestion —
  key by `(dataset_view, image_id)` now, add spine-stamp columns when Phase 1b lands. This is
  what actually fixes the class of bug the last 3 commits patched piecemeal (concurrent
  `add_class`, state races) — real transactions instead of file locks + atomic-write JSON.
- Add **lightweight per-annotator identity** — not a full auth system (this is a small internal
  team on one host, matching the review-dashboard's existing basic-auth-optional pattern) — a
  named identity attached to `session_context`, recorded on every save. Phase 4's second-reviewer
  sign-off has no meaning without knowing *who* did the first pass.
- Add an **append-only audit/history table** (who changed what, when) — currently a save
  just overwrites the JSON file with no history. Needed for Phase 4 audit sampling and for any
  "everything coach 13 touched last week" query once spine stamps exist.
- `SessionBundle` (in-memory, per-process) stays as the *request-scoped* service cache it already
  is — Postgres becomes the durable state layer underneath it, not a replacement for it.

**Phase 1b — Pipeline-facing ingestion (blocked, do not build yet):**
- Ingestion endpoint accepting pipeline frame batches + §1 spine metadata, writing to
  **object storage (S3/MinIO)** instead of the current `images/` folder. Sized around **one
  bogie's batch** per intake (Q-F, §6), not unbounded streaming.
- Extend Postgres schema with the full spine stamp (`coach_index`, `axle_id`, `side`, `view`,
  `longitudinal_position_mm`) once ingestion is real, rather than the interim key above.
- Keep the SAM2 service singleton (`sam_service.py`) but move it behind a proper inference
  server (Triton/TorchServe) **on the annotation module's own GPU** — only worth doing once
  concurrent-annotator load actually contends on the current single Python lock; premature today.
  > **Hard constraint (Q-B answered, §6):** this SAM2 server must be **completely separate**
  > from the pipeline's single RunPod L4 (pipeline C1/C3 — serial SPOF, ≤10-min inference SLA).
- **Why blocked:** the endpoint's request/response contract depends on Q-E (return-path format,
  §6) and an actual live connection to the pipeline, neither of which exist yet. Designing it now
  means guessing at a contract we'd likely have to rebuild.

### Phase 2 — Triage & prioritization `Foundation` — partially shipped

Not every pipeline frame is worth a human's time. Highest-leverage phase for throughput.

**Priority tiers (highest first)**
1. **Field-flagged + gate-recall-audit misses** — operator/inspector reports, and defects the
   pipeline's anomaly gate never tiled (pipeline R6). Always reviewed, senior annotator.
   **Not built — blocked on Q-E (§6) and an actual pipeline connection**, neither of which exist
   yet. `GET /api/triage/queue` always returns these tiers empty (not omitted), so wiring in real
   pipeline data later is additive, not a response-shape change.
2. **Low-confidence pipeline detections** — the model's own uncertainty (`conf` from the
   detection record) is the cheapest active-learning signal; already in the pipeline output.
   **Shipped as a local-detector proxy, not the real thing.** `detector_service.detect()` now
   returns per-box confidence instead of discarding it (it was computed via ultralytics'
   `box.conf` and thrown away before reaching `AnnotationObject`, which already had an unused
   `confidence` field). `triage_service.py`'s `low_confidence` tier ranks not-yet-completed
   images by their saved objects' average confidence — only covers images that have been through
   at least one save with detector-sourced objects, since confidence isn't computed synchronously
   per triage request. Swap for the pipeline's real `conf` once Phase 1b lands.
3. **Novel / out-of-distribution frames — shipped.** `similarity_service.novelty_scores()`
   inverts the existing near-duplicate index — one vectorized matmul over whatever's already
   indexed, scoring every image by `1 - (max similarity to any other image)`. Exactly tier 3 as
   designed; doesn't touch pipeline data at all.
4. **Routine confirmations — shipped as a simplification.** A small seeded-random sample of
   whatever's left after tiers 2/3 claim their images (not filtered to "high-`conf`" specifically,
   since that filter needs tier 2's real signal — see above). Stable across repeated calls so it
   doesn't reshuffle on every request.

Near-duplicate frames of the same pass stay deduped via existing similarity/propagation
(`propagation_service.py`) rather than entering the human queue.

> Tested against the real mounted dataset (2013 images, side_view): tiers are mutually exclusive,
> completed images never leak into any tier, an empty dataset degrades to all-empty tiers instead
> of erroring, and the Phase 1a migration's progress counts were unchanged after this change
> (no regression). No frontend surface yet — `GET /api/triage/queue` only, for now.

> **Deferred triage input:** the pipeline's PatchCore anomaly gate (P2) needs a curated
> *confirmed-normal* pool (pipeline §5, §12). Curating that pool is out of this build's label
> scope — noted in Phase 3 "Deferred."

### Phase 3 — Annotation workflow (SAM2-assisted) `Keep as-is`

The core UX — box prompt → SAM2 mask → polygon → human correction — is sound; only
re-point it at the new queue and storage. The confidence-gated auto-accept
(`mask_confidence_threshold`) and the never-trust-unreviewed boundary (only `completed`
annotations feed export) are exactly right and must survive every later phase.

**Bind to the pipeline's class-map**
- Labeling uses the pipeline's **versioned class-map**; `leakage` is **excluded at the tool
  level** (`exclude_classes:[leakage]`, pipeline §10) — all-synthetic, never shippable.
- Every saved label carries its spine stamp (§1) and the class-map version it was made against.

**Deferred label types (architecture-required, NOT in this build — track as debt):**
| Type | Pipeline family | Why deferred / what it needs |
|---|---|---|
| Confirmed-normal curation | `p2_under_anomaly` (PatchCore) | Not a polygon task — a keep/reject normal-set + purity gate. Separate UX. |
| Crop-classify labels | `p1_side_damage`, `p3_comp_defect` | Class-only, no mask. Different UX. |
| Fastener slot-occupancy | `p3_fastener` | Binary occupied? per known bracket slot — not SAM2 polygons. |
| Wheel log-polar seg | `p3_wheel_shelling` | Masks live in **log-polar unwrapped** space (pipeline §5/§6), not raw frame. Needs geometry-seeded unwrap in the tool first. |

> These four are on the pipeline's critical path (esp. anomaly-first ship, pipeline R10) but
> are **explicitly out of scope now**. Do not silently drop them — carry as a named backlog so
> the annotation module can extend to them without a rebuild.

### Phase 4 — QA / audit layer `New — safety-critical`

The phase the current tool lacks, and the one that matters most given the target: undercarriage
and wheel defects on operating trains. One annotator's mistake propagated onto near-duplicate
frames, then into training data, is a **safety** risk here, not a quality nuisance.

- **Second-reviewer sign-off** required before annotations count as `completed` for training
  (today, one save = done).
- **Mandatory audit sample (5–10%) of *propagated* annotations** re-reviewed independently —
  propagation trades review depth for speed, so it needs its own spot-check.
- **Curate + freeze the per-class golden eval set** here (feeds Phase 6). Structurally separate
  storage; **no propagation/pseudo-label/pipeline path may ever write to it** (risk §5).
- Golden set must have **enough examples per safety-relevant class** (cracked wheel, shelling,
  crack/corrosion) to make a per-class promotion check statistically meaningful (pipeline R10).

### Phase 5 — Dataset versioning & export → staged snapshot for MLflow `New — this is the versioning plan`

`export_service.py` already writes clean YOLO-seg datasets with a deterministic train/val
split — keep that logic, but every export becomes an **immutable, named dataset snapshot**,
not an overwrite of `exports/`. The hand-rolled backup suffixes already in the repo
(`*_before_27class_remap_*`, `*_before_synthetic_removal_*`) are evidence this need
exists — formalize it.

> **Handoff mechanics (Q-C answered, §6): stage, don't write MLflow directly.** The annotation
> module does **not** get write access to the pipeline's MLflow. This module owns a **staging
> store** (S3/MinIO) that holds versioned, content-addressed dataset snapshots + a manifest
> (snapshot ID, class-map version, lineage tags, split integrity flags — see below); the
> pipeline team imports from there into their own MLflow. Design the export target as an
> object-storage snapshot + manifest format, **not** direct `mlflow.data` / registry API calls
> from this module — that dependency doesn't exist.

**Versioning plan (annotation side owns the *dataset* half; pipeline owns the *model* half
and the MLflow registry itself):**
- **Tracking + registry** live on the pipeline's MLflow (persistent volume, pipeline §2/§10).
  This module does not write to it directly (Q-C) — it stages snapshots the pipeline team pulls
  in, so dataset→model lineage still ends up as one graph, just via an import step rather than
  a shared write path. Do not stand up a second registry.
- **Dataset versioning:** DVC (or an equivalent content-addressed snapshot layout) per export,
  written to the staging store. Snapshot ID + **class-map version** are immutable and recorded
  in the snapshot's manifest (mirrored into MLflow as tags once the pipeline team imports it).
- **Class-map as a versioned artifact** (not a loose YAML): every snapshot pins the exact
  class-map + `exclude_classes` it was built against. A class-map change = a new version, never
  an in-place edit (risk §5).
- **Split integrity:** pseudo/synthetic labels tagged and constrained to **train split only**
  (pipeline §12) — enforced and asserted at export time.
- **Families this build feeds** (detection + seg scope): `shared_backbone_v1` +
  detection heads `p1_side` / `p2_under` / `p3_bogie` (boxes) and `p2_under_crackseg` (masks).
  SAM2 emits the polygon; boxes derive from it. `p3_wheel_shelling` and all classify/anomaly/
  fastener families are **deferred** (Phase 3).
- **Lineage tags** on every snapshot: source frames' spine stamps, annotator + reviewer IDs,
  class-map version, golden-set version — so any future model in the registry can be traced to
  the exact data + people that produced it.

### Phase 6 — Handoff to pipeline retrain + promotion gate `New — recommendation`

**This module does not retrain, serve, or deploy.** Retraining the shared backbone/heads/
crack-seg, TensorRT export, and serving on the L4 are the pipeline's (pipeline §8, §12). This
phase defines the **handoff contract** and the **promotion gate this module's golden set backs.**

The pipeline's stated promotion mechanism (pipeline §12) is **shadow-mode canary → A/B
promote**; the plan's earlier proposal was a **frozen golden set**. The architecture has no
explicit golden set. **Recommended design (layered — belt-and-suspenders, justified by the
safety-critical target):**

1. **Offline per-class golden-set gate (pre-canary) — annotation module supplies the data,
   pipeline runs the eval.** Candidate model is scored on the frozen golden set (Phase 4):
   per-class precision/recall/mAP50 for detectors, **Dice/IoU + length-recall** for crack-seg
   (box mAP is the wrong metric for cracks — pipeline §5). **Gate rule: a candidate must meet
   or beat the current model on *every safety-relevant class*, not on aggregate** — an
   aggregate-mAP win that hides a "cracked wheel" regression is a **fail** (plan MEDIUM risk;
   pipeline R6 spirit). A cheap offline filter that stops a bad model *before* any live exposure.
2. **Shadow-mode canary (pipeline-owned).** Only golden-set passers run shadow on live trains
   — **score-only, no alert authority** — compared against prod on next-stop-alert agreement +
   gate recall.
3. **A/B promote (pipeline-owned).** Shadow winners promoted via **MLflow registry stage
   transition**; the deployed TensorRT engine is rebuilt from the promoted checkpoint.

**Why both, in this order:** shadow/A-B alone catches serving regressions but is slow and
exposes an under-vetted model to the live safety path; the offline golden gate is a cheap,
fast pre-filter that never reaches live traffic if it regresses a safety class. For
wheel/undercarriage safety the redundancy is warranted. **Ownership is clean:** the golden
*set* is an annotation deliverable; the promotion *mechanism* stays with the pipeline.

**Lineage on every promoted model** (pipeline records, we supply the inputs): which dataset
snapshot, which class-map version, which golden-set version + per-class report, who approved.

### Phase 7 — Feedback closure `Closes the loop`

- The pipeline returns, per cycle: per-frame `conf`, gate-recall-audit misses (pipeline R6),
  and field-flagged misses — these feed Phase 2 triage on the next cycle. Without this return
  path the "loop" is a one-way pipeline that runs twice.
- **No edge-package / weight-copy step here** — deployment to the L4 is entirely pipeline-side
  (Triton, TensorRT, OTA shadow/A-B, pipeline §8/§12). This module's loop closes at "next batch
  of triaged frames arrives," not at "model shipped."

---

## 3. Compute & timeline

| Workload | Cost driver | Scaling note |
|---|---|---|
| SAM2 mask assist (per frame) | ~50–150 ms encoder, once/frame | Stateless; horizontally scalable behind an inference server **on separate annotation GPU — never the pipeline L4** (pipeline C1 SPOF). |
| Similarity/triage embedding | ~5–10 ms/frame | Not a bottleneck; move brute-force cosine → FAISS/HNSW past ~50–100k frames. |
| Retraining | — | **Pipeline-owned** (pipeline §12). Not this module's compute. Triggered on schedule / QA'd-frame-count threshold, decoupled from the annotation UI. |
| Human review throughput | The actual bottleneck | Triage (Phase 2) + QA (Phase 4) move this number, not GPU. |

| Phase | Depends on | Rough order |
|---|---|---|
| 1 — Ingestion & hardening | Storage/infra + **annotation GPU** decision; confirmed tool location (Q-A) | First — everything sits on it |
| 2 — Triage | Phase 1 + pipeline `conf` field available | Parallel with Phase 1 tail |
| 3 — Annotation workflow | Phase 1 storage swap + class-map version | Re-pointing existing code, not a rebuild |
| 4 — QA/audit + golden set | Phase 1 auth/identity; **domain experts for golden curation** | Before any output is trusted; golden-set lead time is on the critical path |
| 5 — Versioned export → MLflow | Phase 4 + access to pipeline MLflow (persistent vol) | Short — wraps existing export logic |
| 6 — Handoff + promotion gate | Phase 5 + pipeline retrain team | Joint with pipeline team — not annotation-side alone |
| 7 — Feedback closure | Phase 6 + pipeline return path | Joint with pipeline team |

---

## 4. Continuous improvement & diminishing human load

The goal: **human verification shrinks every cycle, but never reaches zero on safety classes.**
This works because the loop already retrains the *pre-labeler*, not because the tool
online-learns. Keep the two ideas apart — conflating them is how a defect system silently rots.

### 4.1 What learns, and what must never learn

| Component | Learns over time? | Mechanism | Never trained on |
|---|---|---|---|
| **Pre-label detector** (shared backbone + zone heads) — the main lever | Yes, every cycle | Batch retrain on **corrected + hard-negative** data (Phase 5 → pipeline §12) | the golden set |
| **SAM2 mask assist** | Optional, periodic | Offline fine-tune on accepted domain masks — later build, not now | the golden set |
| **Auto-accept / quality gate** | Optional | Learns which auto-labels are trustworthy → widens auto-accept coverage | the golden set |
| **Golden eval set** | **No — frozen forever** | It is the ruler; a ruler you stretch measures nothing (risk §5) | — (it is the thing nothing trains on) |

> **The golden set does not teach the tool.** It is the frozen promotion gate (Phase 6). If any
> learner above ever ingests it, the gate stops measuring anything. This is the single most
> important boundary in the whole loop.

### 4.2 Discarded / corrected annotations are the highest-value signal — never delete them

Every human rejection produces **two** training products; route both, discard neither:

1. **Human's corrected label** → **train split** of the next snapshot (Phase 5). Next retrain,
   the detector stops making that specific error → fewer proposals to correct next cycle. This
   is the mechanism that actually shrinks human load.
2. **Rejected raw prediction** → **hard negative** → feed Phase 2 triage: "find frames like
   this, prioritize them." Active learning — the model's own mistakes steer what humans see next.

Both go to retrain data (train split only, pseudo/synth flagged). **Neither ever touches the
golden set.**

### 4.3 The auto-accept curve — how human touches drop

Reuse the existing confidence-gated auto-accept (`mask_confidence_threshold`), driven by
*measured* accuracy, not hope:

```
per class c, per cycle:
    a_c = auto_accept_rate observed at current threshold τ_c
    e_c = post-hoc error rate of auto-accepted labels (from Phase 4 audit sample)
    if e_c < TARGET_ERR and c not in SAFETY_CLASSES:
        raise τ_c coverage        # more frames auto-accepted, human skips them
    else:
        hold / lower τ_c          # detector not good enough yet, keep human in
```

As the detector improves each retrain, more frames clear high-confidence → auto-accepted →
humans only see the **low-conf + novel/OOD + field-flagged** tiers (Phase 2). The human queue
shrinks to the genuinely hard/new frames.

### 4.4 The safety floor — non-negotiable

- **Safety-relevant classes** (cracked wheel, wheel shelling, undercarriage crack/corrosion —
  the pipeline's `tier: safety`) **never auto-accept.** Always a human, always a second reviewer
  (Phase 4). Auto-accept applies only to non-safety classes at proven-low error.
- "Less and less human" is a property of the *cosmetic/structural* classes; the safety tier
  floor is fixed by the QA boundary (Phase 4) and does not move with model confidence.

### 4.5 No live online learning

Do **not** update any model mid-session from each accept/reject. It causes feedback poisoning
(model reinforces its own mistakes), drift, catastrophic forgetting — with no gate in the loop.
Improvement is **batch, scheduled, gated**: QA'd data → snapshot → retrain → golden gate →
shadow canary → A/B promote (pipeline §12). Slow-and-gated beats fast-and-self-poisoning on a
safety system.

### 4.6 Metrics that prove the loop is working (track per class, per cycle)

- **Auto-accept rate** ↑ (coverage growing) — the headline "less human" number.
- **Correction rate** ↓ (fraction of proposals humans change) — detector getting it right.
- **Audit error rate** on auto-accepted labels ≤ target — the safety check on 4.3.
- **Human-touch rate** ↓ overall, **flat/at-floor for safety classes** — proves the floor holds.
- **Golden-set per-class scores** ↑ or flat, never regressing on a safety class (Phase 6 gate).

If auto-accept rate rises while golden-set safety scores fall, **stop** — the tool is getting
lazier, not better. That divergence is the alarm.

---

## 5. Risks worth naming now

**[HIGH] Propagation-driven error compounding**
Propagating one accepted label onto near-duplicates is a time-saver, but a wrong class or
missed defect silently becomes many wrong labels. On a defect-detection system this is
safety-relevant. The Phase 4 audit sampling is **not optional**.

**[HIGH] Class-map drift across exports**
Already happened (27-class remap). With a shared MLflow registry, an unversioned class map —
or one out of sync with the pipeline's `exclude_classes:[leakage]` — is a corrupted training
run waiting to happen. Class-map is a versioned artifact (Phase 5), never edited in place.

**[HIGH] Annotation SAM2 contends with the pipeline L4**
The pipeline runs on a **single L4 that is a serial SPOF with a 10-min SLA** (pipeline C1/C3,
R3). If annotation SAM2 is ever scheduled onto that GPU it can starve the live inference path.
Annotation compute **must be physically separate**. (Q-B.)

**[HIGH] Spine-stamp loss breaks pipeline fusion**
If ingestion drops or mangles `coach_index`/`axle_id`/`side`/`view`/`longitudinal_position_mm`,
the pipeline's entry/exit fusion and k-of-n voting (pipeline §7) cannot key labels back to
physical wheels/coaches. Treat the spine stamp as required, validated on ingest.

**[MEDIUM] Aggregate-mAP promotion hides per-class regression**
A retrain that improves overall mAP but regresses one defect class passes a naive gate. The
Phase 6 golden gate must be **per-class**, and the golden set must have enough per-class examples
(pipeline R10) to make that check meaningful.

**[MEDIUM] Golden-set contamination**
If triage, propagation, or any pipeline path can write into the golden set, the promotion gate
stops measuring anything. Structurally separate storage, not a flag on shared storage.

**[MEDIUM] Deferred label types are on the pipeline's critical path**
Confirmed-normal curation (PatchCore, pipeline's zero-label coverage + ROI gate), fastener
slot-occupancy (the "SAHI-killer"), wheel log-polar seg, and crop-classify are **out of this
build** but **required by the architecture**. Tracked as named backlog (Phase 3), not dropped.

**[MEDIUM] Existing-tool source unverified**
The E:\ tool's referenced modules were not locatable this pass. Hardening estimates assume the
plan's descriptions are accurate; confirm the real code first (Q-A) or Phase 1 scope may shift.

---

## 6. Open questions before build (do not assume)

- **✅ Q-A — Tool location — RESOLVED.** Found: `backend/app/` in this repo
  (`session_context.py`, `sam_service.py`, `export_service.py`, `detector_service.py`,
  `similarity_service.py`, `propagation_service.py` all present, matching this plan's
  descriptions). Phase 1 hardening targets are real code, confirmed.
- **✅ Q-B — Annotation compute — RESOLVED: completely separate from the pipeline L4.**
  Own GPU/pod, never shared with the pipeline's RunPod L4. Today SAM2 runs as a single shared
  instance on the annotation tool's own host GPU (via `docker-compose.yml` GPU passthrough) —
  keep that isolation as a hard constraint when this scales beyond one host.
- **✅ Q-C — MLflow access — RESOLVED: stage, don't write directly.** The annotation module
  does **not** get write access to the pipeline's MLflow. It stages versioned dataset snapshots
  (e.g. S3/MinIO) that the pipeline team imports into MLflow themselves. Phase 5 export design
  should target a staging store + a defined snapshot manifest format, not direct MLflow API
  calls from this module.
- **✅ Q-D — Golden-set ownership — RESOLVED (partially): threshold + timeline.**
  Per-class example thresholds are **not a fixed number** — they're derived from each class's
  actual data balance (rarer safety classes need relatively more curation attention to reach
  statistical meaningfulness, not an equal count across classes). No fixed calendar target:
  golden-set curation is **continuous and incremental**, improving in statistical power as more
  data accumulates, not a one-time freeze-by-date deliverable. Note this doesn't relax the
  Phase 4 rule that *each* golden-set snapshot used to gate a given promotion is itself frozen —
  it means new snapshots can supersede old ones over time as curation matures, each one still
  versioned and immutable once in use (§4.1, §5 risk).
  **✅ Curation permission mechanism now exists:** annotators can be assigned a `golden_curator`
  role (`PUT /api/annotator/{id}/role`), so once the golden-set storage itself is built (Phase 4)
  its write paths can check this role from day one. **Still open:** who (named domain experts,
  by name) actually does the curation — that's a people/process answer, not something a role
  field resolves on its own; still waiting on the pipeline team.
- **Q-E — Return-path format — mostly scoped.** In what form does the pipeline emit low-conf /
  field-flagged / gate-recall-audit misses back for triage (pipeline §7/§12, R6)? Defines the
  Phase 2 / Phase 7 contract. **Position fields (`longitudinal_position_mm` etc.) explicitly not
  calculated locally** — those come from physical encoder counts (pipeline §4), not camera
  optics, and no real calibration constants exist in this repo to compute them from safely (see
  §5 risk "Spine-stamp loss breaks pipeline fusion" — fabricating this would be worse than
  leaving it blank).
  **✅ Wire format decided: same channel as Q-F, not a separate path.** Field-flagged and
  gate-recall-audit-miss frames ride the same bogie-batched ingestion intake as routine frames
  (Q-F), just tagged within the batch (e.g. a priority/reason field per frame) rather than a
  second polling endpoint or push channel. One ingestion contract to build and keep in sync, not
  two. Phase 1b's ingestion endpoint design should carry this tag through to Phase 2's tier-1
  triage bucket. **Still open, sent to pipeline team:** the actual per-frame payload shape/schema
  for the flag itself (reason code, source, timestamp format, etc.) — the *channel* is settled,
  the *payload* isn't.
- **✅ Q-F — Frame arrival model — RESOLVED: discrete batches, one per bogie.** Not a continuous
  stream — each bogie's frames arrive as one complete batch. This bounds the ingestion
  endpoint's back-pressure design: size the intake/queue around "one bogie's worth of frames,"
  not an unbounded stream, and the tool is explicitly meant not to be burdened by full-train
  volume at once.
- **✅ Q-G — Deferred-type priority — RECONFIRMED, unchanged.** Confirmed-normal curation for
  PatchCore remains the highest-priority deferred label type (Phase 3) after the current
  detection+segmentation scope — the pipeline's zero-label unblocker (pipeline build-order step
  3), ahead of crop-classify, fastener slot-occupancy, and wheel log-polar seg.

---

*Annotation module build plan · a separate upstream SAM2 labeler feeding the VB pipeline retrain loop (`FINAL_AIML_ARCHITECTURE.md` §12). Detection + segmentation scope, this build.*
