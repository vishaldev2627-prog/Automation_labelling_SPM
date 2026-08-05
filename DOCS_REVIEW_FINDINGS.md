# Review of `docs/pipeline.md` + `docs/FINAL_AIML_ARCHITECTURE (2).md`

**Read in full.** Both are dated/consistent with each other; `FINAL_AIML_ARCHITECTURE` (2026-08-04)
explicitly supersedes the RunPod-era cloud/edge topology. This changes several conclusions in
`MLFLOW_INTEGRATION_ANALYSIS.md` and shrinks the ask list in
`HANDOFF_REQUEST_TO_PIPELINE_TEAM.md` considerably.

**No code changed as a result of this read.** Two new blockers found (N-1, N-2) that need answers
before they can be, and one label-quality defect (N-3) that is ours to fix once a decision is made.

---

## 1. Resolved — stop asking for these

| Previously open | Answer from the docs | Source |
|---|---|---|
| `exclude_classes` complete? | **Yes — `[leakage]`, nothing else.** "Synthetic `leakage` class excluded from serving." | `pipeline.md` §5.2, §12; `FINAL` §10 |
| Spine-stamp field list + exact names | **Fully specified, with a literal example record.** `coach_index`, `coach_type`, `zone`, `axle_id`, `side`, `view`, `longitudinal_position_mm`, `class`, `conf`, `mask_ref`, `votes`, `tier` | `FINAL` §9; `pipeline.md` §11 |
| Field value vocabularies | `side` = `"L"`/`"R"` (not left/right) · `view` ∈ {`entry`,`exit`} · `coach_type` ∈ {`LHB`,`ICF`} · `zone` ∈ {`P1`,`P2`,`P3`} | `FINAL` §9 |
| Metric floors for a promotion gate | **All present.** `component: 0.35`, `p1_damage: 0.40`, `comp_defect: 0.40`, `crack_dice_min: 0.30`, `shelling_dice_min: 0.30`, `fastener_recall_conf: 0.20`, `p2_anomaly_percentile: 99.0`, `k_of_n: 3`, `cell_size_mm: 100.0`, `missing_min_views: 2` | `pipeline.md` §12; `FINAL` §10 |
| Per-model metrics | mAP50 (detector) · accuracy+macro-F1 (defect-state) · AUROC (anomaly) · Dice/IoU **+ length-recall** (crack-seg) · Dice/IoU (wheel-seg) · recall@FP (fastener) | `pipeline.md` §5.1–5.7 |
| Tier assignments (per model) | `gate_tiers: {crack_seg: structural, wheel_seg: safety, fastener: safety, detect: structural}` | `pipeline.md` §12 |
| Why 8 families, not 7 | `pipeline.md` specs 7; the **buffer-boundary detector** is the new 8th | `FINAL` §2.2 |
| MLflow ownership | **Theirs, confirmed.** §13: "Registry — `model_manager.py`, MLflow — add backbone + head + specialist families." Q-C stands. | `FINAL` §13 |
| Hardware | **DGX Spark (GB10), 128 GB unified memory, on-site.** Stated as a locked constraint in both docs. | `pipeline.md` C1; `FINAL` C1 |

### Two things I previously wrote that are now stale — corrections

- **"A/B promote"** — wrong, drop it. `FINAL` §12 is explicit: "**shadow-mode canary before promote**
  (no A/B-across-region complexity — one device, sequential canary window instead)." Our build plan
  Phase 6 inherited "A/B promote" from the superseded cloud doc. There is no A/B in this architecture.
- **`dropped_regions`** — no longer exists. It belonged to the old **upload envelope**, and there is
  no upload step at all now: the preprocessing→AI handoff is a local file read (`frames.jsonl` +
  `stitched.jsonl` + PNG on shared storage). QA-invalid areas are now expressed as
  `coverage = degraded` / `data_unavailable`. Our build plan §1 data contract is partly written
  against the dead topology and should be re-baselined.

### Hardware — the H200 question, reframed

The docs lock the inference device as **DGX Spark GB10, 128 GB unified**. You told me the deployment
machine is an **H200, 141 GB VRAM, 24 vCPU, 240 GB RAM, 5 TB scratch**. These are different machines,
but they may not be in conflict — and the most likely reading is actually the *desirable* one:

- Our annotation backend image is **already built for aarch64 GB10 / sm_121** with `torch 2.7.1+cu128`
  and an aarch64-specific SAM2 `torch.jit.script` patch. That is a DGX Spark build, not an H200 build.
- Build plan Q-B (locked) requires annotation compute to be **physically separate** from the inference
  device. `pipeline.md` C1/C3 and `FINAL` §0.1.3 call the single DGX Spark a serial SPOF.
- **So: DGX Spark = on-site inference; H200 = annotation + training box** would satisfy Q-B exactly,
  and it is what the 5 TB scratch disk suggests (dataset snapshots, training runs).

If that reading is right, my earlier concern about an aarch64→x86 Docker rebuild applies to the
**annotation module moving to the H200**, not to the pipeline. Still real, still needs doing, but it is
a much cleaner story than "the docs and the hardware disagree." **Needs one sentence of confirmation.**

---

## 2. New blockers — found by cross-reading the docs against our code

These are the reason it was worth reading the docs before sending the team draft.

### N-1 `[HIGH]` P1 (side) labels may be in the wrong coordinate space

The docs are unambiguous and repeat it three times:

> C10: "**Stitched panorama for component localization**, raw frame for defect/metrology pixels"
> §5 P1 table: "Shared backbone + P1 head (detect + keypoints) | **stitched panorama**, per side, tiled 1280px | boxes+class+conf, keypoints — **panorama coords**"
> Build order step 4: "P1 head **trained/run against stitched panoramas, not raw frames**."

**Our `side_view` dataset is a flat folder of images with per-image YOLO label files.** If those are
raw per-camera frames — which is what "side_view" and a flat `images/` + `labels/` layout imply — then:

1. Labels are in **raw camera pixel space**; the P1 head consumes **panorama pixel space**. Not
   interchangeable. A transform exists (`H_cam`), so conversion is *possible* — but see (2).
2. **The defect is baked into the labels, not just the coordinates.** Stitching exists precisely
   because a single camera frame cuts tall components (door, window pillar, tall bracket) at its
   top/bottom edge (§2.1a). An annotator working on a raw frame sees a truncated component and
   annotates it truncated — or as two separate objects across two frames. Transforming those
   coordinates into panorama space yields *correctly positioned wrong boxes*. The full-height context
   that the panorama provides cannot be recovered after the fact.
3. Seam-spanning components (§2.1a step 5, risk R-SEAM) are the exact case where this matters most.

**Three possible resolutions, and this is theirs to choose, not ours:**
- **(a)** Preprocessing stitches first; we annotate **panoramas** in a new dataset view. Cleanest,
  matches the architecture, but needs the stitcher + homography built first (build-order step 3, whose
  own prerequisites — stitcher library, calibration method — are listed as undecided in §16).
- **(b)** We keep annotating raw frames; they transform labels via `H_cam` at training time — accepting
  that any component cut at a raw frame edge is mislabeled. Cheap, and wrong in a known way.
- **(c)** `side_view` is serving some other purpose (P1 defect-state crops, which *do* come from raw
  frames per §2.1a step 4) and the panorama-space detection labels are a separate future dataset.

**(c) is worth taking seriously**, and would be good news: §2.1a step 4 says the defect-state
classifier and metrology run on **native raw crops**, so raw-frame labels are exactly right *for that
model*. In that case `side_view` feeds `p1_side_damage`, not the P1 detection head, and we need a
second view for panoramas.

**Until this is answered, we do not know which pipeline model our largest annotated dataset feeds.**
That is the highest-value question in this document.

### N-2 `[HIGH]` `buffer_visible` needs a label type our tool cannot produce

The docs flag the training data as missing, twice:

> §2.2 open items: "**training-data availability for the new `buffer_visible` class** (not covered by
> any existing dataset referenced elsewhere in this repo's docs)"
> R-BUFFER: "Buffer-boundary detector is new and untrained … **OPEN — new component, no training data yet**"

Our tool has a **`buffer` dataset view** — added recently, described in `routers/dataset.py` as "a raw,
unlabeled frame dump with no pre-existing class list — starts with zero classes, built up via the
Classes panel's +Add as annotators encounter components (product decision, not a fallback default)."

**That looks like it was created to be this training data.** If so, there is a format mismatch:

| Needed | What our tool produces |
|---|---|
| `buffer_visible` yes/no **per frame**, plus confidence — a single-class **classifier** label (§2.2: "lightweight single-class classifier … the triggering use case only needs presence, not localization") | Per-object **bounding boxes + SAM2 polygons** |

A frame-level binary flag is not expressible in our current schema at all. It is the same family as the
**crop-classify** label type our build plan Phase 3 explicitly deferred. So annotating `buffer` with
boxes and polygons produces something real, but **not what `VB-BufferBoundary` needs**, unless the
intent is "a box drawn around the buffer implies `buffer_visible=true`, absence implies false" — which
would work as a convention but needs to be stated, because it silently makes *empty frames* meaningful
data rather than skipped data, and our tool currently treats an unannotated frame as un-*done*, not as
a labeled negative.

**Also worth flagging back to them:** §2.2 leaves the architecture choice open (classifier vs
detector). If they choose the detector variant "later, only if a bounding box proves useful for QA
visualization," then boxes are the right label after all — and annotating boxes now covers both cases,
since a box trivially reduces to a presence flag. **That is the cheap answer, and I'd recommend it**:
annotate boxes, derive the binary label at export. It costs nothing extra now and preserves both
options. But negatives still need an explicit representation.

### N-3 `[MEDIUM]` Our polygon pipeline is lossy in exactly the dimension crack-seg is scored on

`pipeline.md` §5.4 sets the crack/corrosion metric as **"Dice/IoU + length-recall (not box mAP)"**, and
§5.5 sets wheel-shelling as Dice/IoU with **shelling length** as an output. Three properties of our
export work against both:

1. **`mask_to_polygon()` keeps only the largest external contour.**
   `largest = max(contours, key=cv2.contourArea)` — everything else is discarded. A crack that
   branches, or breaks into several collinear segments (which is what real cracks do), loses every
   piece but the biggest one. Length-recall is destroyed by this directly, and it is silent.
2. **Douglas-Peucker simplification at `epsilon = 0.002 × perimeter`.** Fine for a coupler outline;
   for a hairline crack, the perimeter is long relative to the feature width, so epsilon is large
   relative to the detail being preserved. Thin structure gets smoothed away.
3. **Export writes polygons only** (`write_segmentation_label_file`), never mask rasters. So even a
   perfect in-tool mask is downgraded to a simplified single-contour polygon on the way out.

None of this matters for the component-detection classes, which is what the tool was built for. It
matters a lot for `p2_under_crackseg` and `p3_wheel_shelling`.

**Recommendation (ours to implement once they confirm the format they want):** for crack/corrosion and
shelling classes specifically, export **binary mask PNGs alongside the polygons**, and/or retain
**multi-contour** polygons with `epsilon_ratio` at or near 0. Both are contained changes to
`polygon_service` + `export_service`. This is a natural fit inside milestone M2 (snapshots), since the
manifest is where per-class label format would be declared anyway.

### N-4 `[MEDIUM]` No `coach_type` in our data, so nothing coach-type-conditional can be checked

The manifest is **coach-type-conditional** (LHB vs ICF expect different components), unknown coach-type
is a hard refusal (`pipeline.md` §8: "refuse to guess … never assume a manifest"), and R14 names
"manifest/coach-type error → false-missing storm" as a designed-in risk.

Our labels carry no `coach_type`. Consequences:
- We cannot produce **per-coach-type** label counts, which is what R10 ("label scarcity — OPEN, needs
  label counts") actually needs to be actionable. An aggregate count per class hides that a class is
  well covered on LHB and absent on ICF.
- We cannot assert manifest coverage per coach type from our side at all.

This is a subset of the spine-stamp gap (Phase 1b), but worth separating: `coach_type` is a **single
low-cardinality field** ({LHB, ICF}) that a human annotator could set per batch today, without waiting
for encoder-derived ingestion. Unlike `longitudinal_position_mm`, it does not require calibration
constants we don't have. **Cheapest high-value spine field to add early.**

### N-5 `[LOW]` C3 (no resize, lossless) — verified satisfied, with dead config to keep an eye on

Checked because C3 ("lossless PNG, native resolution … no resize") would be violated by any silent
downscale:

- `downscale_if_needed()` exists in `image_utils.py` but is **never called anywhere**. Dead code.
- `max_image_dimension: 4096` and `thumbnail_max_dimension: 1024` in `config.py` are **never read**.
  Dead config, exposed in `.env.example` as if live.
- Export uses `shutil.copy2` → images are **byte-identical**, so lossless native resolution survives.
- Image *display* re-encodes to JPEG q=92 (`GET /api/images/{id}/file`), but annotation coordinates are
  normalized 0–1 against the true dimensions, so labels are unaffected. Display-only.

**Verdict: compliant.** Flagging only because those three dead items look live — enabling
`MAX_IMAGE_DIMENSION` on the assumption that it does something would silently violate C3.

---

## 3. Now-answerable: what a promotion gate would actually check

Previously I said the doc's metric floors were unverifiable from this repo. They are now fully
specified, so the gate the MLflow doc describes is implementable **on the pipeline side** (not ours).
Recording the mapping here because it is the concrete content of build plan Phase 6, and it is what our
golden set would have to be sized to support:

| Family | Metric | Floor | Tier |
|---|---|---|---|
| Shared detector (+keypoints) | mAP50 | `confidence.component 0.35` | structural (`detect`) |
| Defect-state classifier | accuracy + macro-F1 | `p1_damage 0.40`, `comp_defect 0.40` | structural |
| P2 anomaly (PatchCore) | AUROC | `p2_anomaly_percentile 99.0` | gate (safety-adjacent) |
| P2 crack/corrosion seg | **Dice/IoU + length-recall** | `crack_dice_min 0.30` | structural |
| P3 wheel-shelling seg | Dice/IoU | `shelling_dice_min 0.30` | **safety** |
| P3 fastener slot-occupancy | recall@FP | `fastener_recall_conf 0.20` | **safety** |
| Coach-type classifier | (unstated) | — | structural |
| Buffer-boundary | (unstated) | `TAU_BUFFER` **not yet set** | structural |

Two gaps in that table are theirs to close: no metric/floor is stated for the **coach-type classifier**
or the **buffer-boundary detector**, and `TAU_BUFFER` is explicitly "needs labeled data to tune" (§2.2)
— which is our data to supply (N-2).

**Note the tier granularity mismatch that remains:** `gate_tiers` assigns tiers **per model**. Our
`safety_critical` flag is **per class**. `component_defect_taxonomy.yaml` is described as the
authoritative "component × defect × coach-type × zone × **tier**" schema (`pipeline.md` §11), so the
per-class tiers do exist — in a file we still don't have. This is still ask #2, now with an exact
filename and a quote to point at.

---

## 4. The golden set is *not* in either document

Worth being direct about, because it changes the nature of the ask.

`FINAL` §12's retrain loop is: review console → approved → dataset (pseudo/synthetic train-split only)
→ retrain → MLflow → **shadow-mode canary before promote**. `pipeline.md` has no promotion section at
all. **Neither document contains a frozen golden eval set, or any per-class offline gate.**

So our build plan Phase 6's "offline per-class golden-set gate, pre-canary" is an **annotation-team
recommendation that the pipeline architecture has not adopted.** I had been treating the named-curators
question as a missing input from them; it is actually a **proposal awaiting their decision**.

The argument for it is unchanged and, I think, strong given what the docs now confirm:
- `wheel_seg` and `fastener` are **safety** tier (`pipeline.md` §12), and C6 says safety checks are
  never skipped.
- C4 is confirmed **post-departure, alert-only** — the verdict does not gate dispatch (R1). A missed
  safety defect has already left the depot. That makes a cheap pre-canary filter *more* valuable, not
  less: shadow canary exposes an under-vetted model to the live safety path, and the aggregate-mAP
  problem (a win overall that regresses "cracked wheel") is invisible to a shadow-agreement metric.

But it is their call, and it needs to be asked as a question, not as a request for curator names.

---

## 5. Net effect on our plan

**No change to M0–M4.** Everything in flight stays valid: class-map versioning, immutable snapshots,
split integrity, golden-set storage. The docs make the *content* of those artifacts concrete rather
than guessed.

**Changes to what goes in the manifest** (M2), now that field names are known:
- `spine_stamp_coverage` should enumerate the real field names from `FINAL` §9, and use `L`/`R`,
  `entry`/`exit`, `LHB`/`ICF` as value vocabularies rather than inventing any.
- Add **`coach_type` breakdown to `per_class_counts`** (N-4) so it answers R10 properly.
- Add a **per-class label format** declaration (polygon vs mask raster), needed by N-3.
- Drop `dropped_regions` from anything we were going to carry — it does not exist any more.

**New work items not previously scoped** (not started, pending answers):
- N-1 resolution may require a **new dataset view for P1 panoramas**, or a documented decision that
  `side_view` feeds `p1_side_damage` instead of the detection head.
- N-2 requires a way to express **frame-level negatives** (`buffer_visible = false`) — currently
  an unannotated frame means "not done," not "labeled negative."
- N-3 requires **mask-raster export** for crack/corrosion/shelling classes.
- N-4 requires a **`coach_type` field** on annotations, settable per batch.
