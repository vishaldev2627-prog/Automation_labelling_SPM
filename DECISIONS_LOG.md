# Decisions log — annotation module

Answers received from the pipeline team, what each one settles, and what it changes on our side.
Recorded here so nothing is re-litigated and so the consequences are traceable to a decision.

**Date received:** 2026-08-05
**Source:** pipeline team reply to `HANDOFF_REQUEST_TO_PIPELINE_TEAM.md`

---

## D-Q1 — P1 labels: **option (b)**

> `side_view` feeds **`p1_side_damage`** (the defect-state classifier), not the P1 detection head.

**Settled:** our ~2000 raw-frame `side_view` labels are in the right coordinate space.
`FINAL` §2.1a step 4 confirms the defect-state classifier and metrology run on **native raw crops**,
never panorama pixels. No `H_cam` transform needed for this dataset. N-1 resolved.

### Consequence C-1 `[HIGH]` — the defect-state classifier needs a label axis we don't have

`pipeline.md` §5.2 specifies `p1_side_damage` as a **crop classifier**:

> **Input:** a component crop (from a detector box) + margin.
> **Output:** condition ∈ {ok, broken, missing, hanging, displaced, dislocated, leaking, damaged,
> securing_broken, securing_hanging, fiba_red, sparking, binding}.

Our labels carry **one `class_id` per object = the component's identity** (coupler, brake cylinder,
axle box, spring, bearing housing — per `README.md`). There is **no condition field anywhere** in
`AnnotationObject`.

So `side_view` currently supplies the **crops** that classifier trains on, but not its **labels**.
Feeding `p1_side_damage` needs a second, orthogonal label axis: *component identity* × *condition*.

This is the **crop-classify** label type our build plan Phase 3 deferred. Answer (b) moves it from
"deferred backlog" to "required for the family this dataset feeds."

**Options, cheapest first:**
- **(i)** Add a `condition` field to `AnnotationObject`, defaulting to `ok`, with the 13-value
  vocabulary from §5.2. Annotators set it per object. Existing 2000 images become `condition: ok` by
  default — **which is a claim we should not make silently**, since nobody actually asserted those
  components were in good condition. Better: default `null` = unassessed, and treat unassessed as
  not-exportable-for-`p1_side_damage` while remaining fully valid for component detection.
- **(ii)** Treat condition as a separate annotation pass over exported crops — a different UX, more
  work, cleaner separation.

**Recommendation: (i) with `null` default.** One nullable field, no invented data, and the existing
component labels keep their full value for the detection families.

### Consequence C-2 `[MEDIUM]` — the P1 detection head now has no training data from us

If `side_view` feeds `p1_side_damage`, then the **shared backbone + P1 head** — which trains on
stitched panoramas in panorama coords (C10, build-order step 4) — has **no dataset from this module**.

Not blocking us, but it should be on someone's list: a panorama dataset view can only exist after
build-order step 3 (stitcher + homography), whose own prerequisites are still open in `FINAL` §16
(stitcher library, calibration method). **Flagging, not asking.**

---

## D-Q2 — `buffer_visible`: **annotate a box around the buffer**

**Settled:** our recommendation accepted. Annotators draw a box around the buffer; `buffer_visible =
true` is derived at export from box presence. Covers `FINAL` §2.2's still-open classifier-vs-detector
choice either way — a box reduces to a presence flag, a flag does not reduce to a box.

### Consequence C-3 `[HIGH]` — negatives are currently discarded at export

Not answered in the reply, and it is a real gap we have to close for this data to be usable.

A presence classifier trains on **both** classes. Frames where the buffer is *not* visible are
training data — but today:

- Our tool has no way to say "I looked, there is nothing here." An unannotated image reads as
  **not done**, not as a labeled negative.
- `export_service.py` actively drops them: `if not objects: skipped += 1; continue`. So even an image
  a human completed with zero objects is **excluded from the export**.

**Implementing:** an explicit `no_objects_confirmed` flag on the annotation, distinct from "empty".
The distinction matters — an image can be empty because (a) a human confirmed nothing is present, or
(b) nothing has been annotated yet, or (c) every object was rejected. Only (a) is a negative sample.
On export, (a) writes the image plus an **empty `.txt`** label file — the YOLO convention for a
background/negative sample, which ultralytics consumes natively.

---

## D-Q3 — crack/corrosion/shelling label format: **our suggestion**

**Settled:**
- Export **binary mask PNGs alongside polygons** for crack/corrosion/shelling classes.
- Keep **multi-contour** polygons with simplification effectively off for those classes.
- **Full frames + masks**, not pre-tiled 320 px crops — they tile at training time, since gated SAHI
  depends on the anomaly ROI, which is a runtime artifact we don't have.

**Implementing:** a per-class `fine_structure` flag on `dataset_classes` (same pattern as the existing
`safety_critical` — keyword-seeded default, curator-editable in the UI). For a flagged class:
`mask_to_polygon` retains **all** external contours instead of only the largest, `epsilon_ratio` drops
to 0, and export additionally writes a mask raster.

This directly fixes N-3, where our largest-contour-only + Douglas-Peucker path was silently degrading
the two metrics those families are actually scored on (Dice/IoU **+ length-recall**, `pipeline.md` §5.4).

---

## D-Q4 — golden eval set: **accepted**

**Settled:** the frozen per-class golden set, gating **before** the shadow canary, is adopted. A
candidate must meet or beat current production on **every safety-relevant class**, not on aggregate.
We curate and version; they run the eval. Nothing that trains ever sees it.

Milestone M4 proceeds as planned: separate storage, separate bucket, `golden_curator`-write-only, and
golden ids asserted disjoint from every split at export.

**Still open — not blocking the build, blocking the population:** named domain experts to curate. The
storage and permission gate can be built now; the set cannot be filled without them.

---

## D-Q5 — zone coverage

### Underbelly: **line-scan footage completely excluded — but our view is area-cam**

**Clarified.** My first reading was wrong and is corrected here: the excluded line-scan feed is a
*separate* feed. Our `underbelly` dataset view holds **area-camera (cams 5/6)** footage, which is
exactly what P2 consumes.

**So `underbelly` is in scope**, and it feeds:
- `p2_under_anomaly` (PatchCore gate) — confirmed-normal curation, still a deferred label type.
- `p2_under_crackseg` — crack/corrosion masks.

**Two consequences:**
1. **W-3 (mask rasters + multi-contour) applies to `underbelly`, not just wheels.** `p2_under_crackseg`
   is scored on Dice/IoU **+ length-recall** (`pipeline.md` §5.4) — the exact metrics our
   largest-contour-only + Douglas-Peucker path was degrading. This raises W-3's priority: it is now
   load-bearing for two families, not one.
2. The deferred **confirmed-normal curation** for PatchCore is against *this* footage. `FINAL` build
   order puts P2 anomaly at step 6 as "the ROI gate that fits P2 in budget", so this remains the
   nearest deferred item to their critical path.

No work item to mark anything out-of-scope. W-6 deleted.

### Wheel: **"wheel specs are to be considered in your own accord for now"**

**Settled:** we choose the wheel geometry constants for the log-polar unwrap ourselves, provisionally.

### Consequence C-4 `[HIGH]` — log-polar masks are only valid relative to their unwrap parameters

This is the part that needs care, and it changes *how* we should annotate rather than just *what*.

`pipeline.md` §5.5 seeds the unwrap circle from "fixed geometry + known diameter — not blind Hough".
A log-polar unwrap is a **coordinate transform parameterised by centre, radius range, and output
size**. A mask drawn in unwrapped space is meaningless under different parameters — change the assumed
diameter and every previously-drawn mask silently misaligns. Since we are choosing those constants
**provisionally**, they will almost certainly change when real specs arrive.

**So: do not annotate directly in log-polar space.** Instead:

1. Annotate in **raw frame space** (where we already work, and where the data is ground truth).
2. Store the unwrap parameters used, **versioned per snapshot**, in the manifest.
3. Generate the log-polar view **at export**, from raw masks + current parameters.

A spec change then becomes a **re-export**, not a re-annotation of every wheel. If we annotated
directly in unwrapped space, the first corrected diameter figure would invalidate the entire dataset.

Provisional constants will be recorded in config with an explicit "not certified, engineering
placeholder" marker — the same treatment the `safety_critical` keyword seed already carries. **We are
not going to state a wheel diameter as authoritative.** Someone with the rolling-stock spec should
replace them, and the versioned parameters make that a cheap swap.

---

## D-Q6 — manifest and coach types

### Field names: **ours to define**

**Settled.** Manifest proceeds as drafted in `HANDOFF_REQUEST_TO_PIPELINE_TEAM.md` §7.

### Coach types: **LHB and ICF only** — resolved

An initial reply listed four types (LHB, ICF, Vande Bharat, hybrid). Both documents specify **two**:

> `pipeline.md` §5.7 — Coach-type classifier: "**Output:** {LHB, ICF} → selects the correct component manifest."
> `FINAL` §9 — detection record: `"coach_type":"LHB"`, with `{LHB, ICF}` throughout.

**Confirmed 2026-08-05: LHB and ICF only.** C-5 is closed — there is no discrepancy to carry, no
4-manifest implication, and no need for a definition of "hybrid". The follow-up to the pipeline team on
this is withdrawn.

**Implemented** as `{LHB, ICF, unknown}`. `unknown` is not a third coach type — it is the absence of an
answer, and it is the default. Deliberately kept rather than defaulting to LHB because LHB is
commonest: `pipeline.md` §8 refuses to select a manifest for an unknown coach type instead of assuming
one, and inventing a type at labeling time would be the upstream version of the mistake R14 describes
("manifest/coach-type error → false-missing storm"). Set per image, with a bulk apply that fills only
images still `unknown`, and `per_class_counts` breaks down by it — which is what their R10 ("label
scarcity — needs label counts") actually needs, since an aggregate count hides a class well-covered on
LHB and absent on ICF.

---

## Summary of new work items created by these answers

| # | Item | From | Size |
|---|---|---|---|
| **W-1** | `condition` field (13-value enum, `null` default) on annotation objects | C-1 (D-Q1) | small + migration |
| **W-2** | `no_objects_confirmed` explicit-negative state; export as empty `.txt` | C-3 (D-Q2) | small + export change |
| **W-3** | `fine_structure` per-class flag → multi-contour polygons, epsilon 0, mask-raster export | D-Q3 | medium + migration |
| **W-4** | `coach_type` per-batch field, enum of 4 + unknown; `per_class_counts` broken down by it | C-5 (D-Q6) | small + migration |
| **W-5** | Log-polar generated **at export** from raw-space masks, with versioned unwrap parameters | C-4 (D-Q5) | medium, later |

### Status, 2026-08-05

| Item | State |
|---|---|
| W-1 condition field + condition-crop export | **done** |
| W-2 explicit negative state | **done** |
| W-3 fine-structure masks | **done** (migration `a4f1c9e207b3` unapplied on the deploy host) |
| W-4 coach_type | **done** |
| M1 class-map versioning + exclude_classes enforcement | **done** (migration `b7e3d81a45c2`) |
| M2 content-addressed dataset snapshots + manifest | **done** (migration `c92a5f14d8e0`, adds `boto3`) |
| W-5 log-polar at export | not started — belongs after M2 |
| M0.3 host-path `dataset_key` | parked, needs a `pg_dump` on the deploy host first |

**M1 as built.** `class_map_versions` is immutable and content-addressed: a
sha256 of a canonical `[[id, name], ...]` plus sorted `exclude_classes`. Minting is
idempotent by hash, so `ensure_version()` runs on every dataset load and records
genuine drift — including a hand-edited `data.yaml` — without creating a version
per load. `add_class()` mints immediately rather than waiting for the next load, so
a snapshot taken in between pins what is actually on disk. The seeding migration
mints version 1 per view from current `dataset_classes` contents.

Deliberately **not** part of the hash: colour, `safety_critical`, `fine_structure`.
Those are tool-side metadata — a curator correcting a safety flag must not
invalidate the class map every previous snapshot pinned. `fine_structure` does
affect exported label format, but that belongs in the snapshot manifest's
per-class `label_format`, which describes one export rather than the taxonomy.

`exclude_classes` (`[leakage]`) is now enforced at the tool level as build plan
Phase 3 requires, not just filtered at export: `add_class()` refuses the name
(422), and export additionally drops any that already exist in a dataset and
reports `excluded_labels_dropped`. Excluded names stay in the exported `data.yaml`
`names` map — class ids are positional, so removing one would renumber every class
after it, which is the exact drift this work exists to prevent. `data.yaml` also
now carries `class_map_version`, `class_map_hash` and `exclude_classes`, so an
export can answer "what did class 7 mean" without the tool or its database.

Also done, from the earlier bug pass: M0.1 (per-view detector registry) and M0.2 (detector/mask
confidence split). Test suite went from 0 to 63.

**W-1 as built.** `condition` is a nullable 13-value enum on `AnnotationObject`, vocabulary verbatim
from `pipeline.md` §5.2. Export writes `crops/<split>/<condition>/<image_id>__<object_id>.png` -
folder-per-class, which is what YOLO11-cls and the usual EfficientNet pipelines consume - with a 7.5%
margin matching the pipeline's own `homography.crop_margin_pct`, so training crops are cut the way the
serving path cuts them. PNG not JPEG, per C3. The crop split is the **image-level** split, so two crops
from one frame can never straddle train and val. The export response reports
`condition_unassessed` alongside `condition_crops`, so "0 crops" reads as "nothing assessed yet" rather
than a broken export.

`condition` is deliberately **not** propagated onto near-duplicate frames: a component's *class* is a
property of the component, but its *condition* is a judgement about that frame, and copying a defect
state onto a frame no human has seen is the build plan's `[HIGH]` propagation-error-compounding risk.

**M2 as built.** Three decisions taken 2026-08-05:

1. **`snapshot_id` hashes data only** — file paths + content hashes, plus the class-map hash. Not
   `created_at`, annotator ids, review counts or per-class counts. So re-exporting unchanged data
   resolves to the *same* snapshot (finalize is idempotent and never overwrites), one extra audit
   review does not mint a new one, and renaming a class *does* — identical label bytes mean different
   things under a different map.
2. **Snapshot is the default** for `POST /api/export`; `as_snapshot: false` keeps the old in-place
   write. Nothing in this codebase writes `exports/class_folders`, so the Flask review dashboard is
   unaffected.
3. **Build local, publish opt-in.** The snapshot is written, hashed and recorded before any upload.
   Publishing is a separate retryable call that reports `published: false` with a reason rather than
   failing an export whose data is already safely on disk. Retry resumes, because the content is
   immutable and already-present objects are skipped.

Layout: built into `exports/.staging/<id>/` (same filesystem, so finalize is a rename not a
multi-gigabyte copy), then moved to `exports/snapshots/sha256-<hex>/`. Carries `manifest.json` and a
`sha256sum`-compatible `checksums.sha256` so the pipeline team can verify a transfer with no tooling
from us. `dataset_snapshots` indexes them; `snapshot_id` is unique **globally**, not per view, because
it is a content hash.

Manifest reports the uncomfortable numbers deliberately:
`provenance.grandfathered_unreviewed_images` (non-zero ⇒ a model trained on this snapshot cannot claim
all data was second-reviewed), `split_integrity` counts, and `spine_stamp_coverage` publishing its
zeros. **Split integrity is measured, not yet enforced** — M3 turns a violation into a refusal and
fixes the split; M2 records it so the manifest cannot claim a compliance it does not have.

W-1 through W-4 slot into milestones M1–M2. W-5 belongs with the wheel work, after M2. None of them
change the M0–M4 shape already approved.

**Agreed build order:** W-2 → W-4 → then W-3 / W-1. W-2 first because a confirmed-empty frame is
currently dropped at export, so negative samples are being lost *while annotators work*.

## Decisions taken on our side

- **C-1 condition default = `null` (unassessed).** Not `ok`. Nobody asserted those 2000 components
  were in good condition, and this is a defect-detection system. Unassessed objects stay fully valid
  for the detection families and are excluded from `p1_side_damage` export until a human sets one.
- **C-5 coach types = LHB and ICF only.** Closed, and it matches both docs. Enum is
  `{LHB, ICF, unknown}` where `unknown` is the absence of an answer, not a type.

## Follow-ups back to the pipeline team

One item remains, and it is a consequence of their own answer rather than a new ask:

1. **C-1** — `p1_side_damage` needs condition labels, which do not exist in our schema yet. Confirm
   the 13-value vocabulary in `pipeline.md` §5.2 is current. We are defaulting existing data to
   `null`/unassessed rather than `ok`, so **the existing ~2000 `side_view` images will not appear as
   `p1_side_damage` training data** until someone assesses them — flagging so that is not a surprise
   when they go looking for it.

*C-5 (coach types) withdrawn — resolved to LHB/ICF. D-Q5 (underbelly) resolved — area-cam, in scope.*
