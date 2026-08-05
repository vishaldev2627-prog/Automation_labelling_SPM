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
| M3 split-integrity enforcement | **done** (no migration — `export_service.py` only) |
| M4 golden eval set storage + permission gate | **done**, structure only (migration `d5a7c0f3e8b1`) — population blocked on named domain experts to curate |
| W-5 log-polar at export | **done** (no migration — export-time transform only) |
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
zeros. Split integrity was *measured but not enforced* at the time M2 shipped — M3, below, closes
that gap.

**M3 as built.** `pipeline.md`'s retrain rule (via `FINAL_AIML_ARCHITECTURE` §12): pseudo/synthetic
(propagated) and auto-accepted labels go to train only, never valid/test. M2 recorded a violation
honestly; M3 makes one structurally impossible instead of recording it.

`ExportService._enforce_split_integrity` is the per-image decision: an image's deterministic
hash-based split (`_split_for`) is overridden to `train` whenever it carries a propagated label or is
auto-accepted, regardless of which bucket its hash landed in. Pulled out as its own pure
staticmethod — no Postgres, no filesystem — specifically so the enforcement rule is unit-testable in
isolation rather than only reachable through the full DB-backed export path. Every override is logged
at `warning`, per the plan's own risk note: forcing a re-split changes what a previously-exported
split assignment meant for that image, and that must never happen silently.

The pre-existing `pseudo_synthetic_in_val_or_test` / `auto_accepted_in_val_or_test` manifest counters
are kept rather than removed — with enforcement in place they should always read 0, so a nonzero value
is now a bug signal rather than an expected measurement. Two new counters,
`pseudo_synthetic_forced_to_train` / `auto_accepted_forced_to_train`, record how many images the
forcing pass actually redirected, so a manifest reading all-zeros is legible as "nothing needed
enforcing" rather than "enforcement didn't run".

`ExportService._finalize_snapshot` additionally refuses to finalize — raising `SplitIntegrityViolation`,
surfaced as HTTP 422 — if a snapshot's stats ever report a violation despite the forcing pass. This
should be unreachable in practice; it exists because a snapshot is the immutable, MLflow-tracked
handoff artifact, so on this one specific check "should never happen" gets a hard refusal rather than
a logged warning. The check runs before any snapshot file is written, so a violation costs nothing to
recover from. Scoped to snapshot mode only (`as_snapshot=true`) — `as_snapshot=false`'s mutable
in-place write was already the lower-stakes path M2 chose not to gate.

Verified against the real API, not just unit tests: an image engineered to hash into `val` was given a
`source: propagated` object, exported, and landed in `images/train/` on disk with
`pseudo_synthetic_forced_to_train: 1` in both the API response and the written manifest — no
`images/val/` directory was created at all for that export.

No migration — this milestone is entirely `export_service.py`/`routers/export.py` logic, nothing
persisted differently.

**M4 as built — structure and permission gate only, per D-Q4: "the storage and permission gate can be
built now; the set cannot be filled without [named domain experts to curate]."** Nothing here
populates a golden set with real images; it builds the mechanism so curation can start the moment
those experts are named, instead of retrofitting a permission concept onto data that already exists.

Two new tables, migration `d5a7c0f3e8b1`: `golden_sets` (one row per curated version, append-only —
mirrors `class_map_versions`' "a curation round mints a new version, never edits an old one's items")
and `golden_set_items` (which `image_id`s are in which version). `golden_repo.get_golden_image_ids`
reads **across all versions**, not just the latest, deliberately: once an image is golden it stays
disjoint from every export split permanently, not just until the next curation round.

`golden_curator` was already a recognized role — `app/services/annotator_service.VALID_ROLES` and the
`Annotator.role` column both predate this work, added during Phase 1a specifically so this day would
not need a role system retrofitted onto live data. M4 is the first thing that actually checks it:
`golden_service.require_golden_curator` 403s any write from a non-curator, and the actual predicate
(`_is_golden_curator`) is a pure function of an already-fetched `Annotator` — pulled out the same way
M3's `_enforce_split_integrity` was, so the permission rule is unit-testable without a database. Reads
(listing sets/items) stay open, consistent with the rest of this tool's identity system having no real
login — there's nothing sensitive in knowing a golden set exists, only in writing to one.

Storage is a **separate MinIO bucket** (`vb-golden-eval-set`, not a prefix under the snapshot store's
`vb-dataset-snapshots`), per the build plan's explicit requirement: "structurally separate storage, not
a flag on shared storage," because the only contamination mitigation that counts is that no
export/propagation/triage path can reach it at all. A prefix under a shared bucket is still one IAM
boundary an export path could accidentally cross; a separate bucket cannot be, short of explicitly
being handed its credentials. Freezing an item's image+label bytes to that bucket
(`object_store.freeze_golden_item`) is best-effort and non-fatal, same contract as M2's snapshot
publish — a curator's "this image is golden" decision is recorded in the DB regardless of whether the
object-store copy succeeds, because that decision must not be lost to a transient MinIO outage.

Two enforcement points, both load-bearing (the build plan: "assert it in tests"):

1. **Export.** `export_service._write_dataset` now fetches `golden_repo.get_golden_image_ids` in the
   same DB round-trip as the existing provenance queries and drops any matching `image_id` from
   `exportable` entirely, before the completion/review-gate checks even run — a golden image belongs
   in neither train nor val of a training export, which is a harder exclusion than M3's "push into
   train only." The pre-existing `golden_ids_in_any_split` manifest counter (hardcoded `0` since M2,
   with a comment that no golden storage existed yet) is now a real invariant check: it should always
   read `0` by construction, same "nonzero is a bug signal" pattern M3 established for its two
   counters, and `_finalize_snapshot`'s existing refusal already covers it. `golden_set_storage_exists`
   flips from a hardcoded `False` to reflecting whether this view actually has a golden set.
2. **Propagation.** `propagation_service._propagate_to` checks `golden_repo.is_golden_image` before
   `_is_untouched`, so a golden image that happens to also look untouched is still refused, not just
   deprioritized. Propagation is an automation writing into an image no human necessarily reviewed
   yet — exactly the path that must never be allowed near the ruler nothing trains on.

Verified against the real API and the real dev-DB propagation log, not just unit tests, after finding
and fixing a real bug this way: `dataset_view` is not the short view name ("side_view") anywhere else
in this codebase, it's `DatasetService.dataset_key` — the resolved absolute dataset path. Every other
write path (`export.py`, `review.py`) derives it server-side from whichever dataset is currently
loaded rather than trusting a client-supplied string. The first `golden.py` draft took `dataset_view`
as a request field, and a live test against it silently created a golden set keyed by the string
`"side_view"` while the export path was querying by the resolved path — so `golden_images_excluded`
read `0` even with a real golden item on disk. Fixed by removing the field entirely and deriving it
from `get_dataset_service().dataset_key`, matching `export.py`'s own pattern; `add_items` additionally
refuses (409) if the currently loaded view doesn't match the golden set's own recorded view, so the
same class of mismatch can't silently freeze the wrong images under the wrong version. Re-verified
after the fix: a 403 for a plain annotator, a successful curator write, an export that produced
`golden_images_excluded: 1` with the golden image genuinely absent from the exported bytes on disk,
and a propagation attempt that logged `"Propagation into <id> refused: image is in the golden eval
set"` and left the image's original objects untouched.

**W-5 as built.** `pipeline.md` §5.5's wheel-shelling segmentation runs on a log-polar unwrapped wheel
crop - a coordinate transform parameterised by circle center, radius, and output size. D-Q5's
Consequence C-4 is the reason this is a milestone and not just a helper function: a mask drawn
*directly* in unwrapped space is meaningless the moment those parameters change, and our wheel
geometry constants are explicitly provisional ("wheel specs are to be considered in your own accord
for now"). So annotation stays in raw frame space, and the unwrap is generated fresh at every export
from current parameters - a spec change becomes a re-export, not a re-annotation of every wheel.

Two new pieces, both pure/filesystem-only, no migration:

- `log_polar_service.find_wheel_circle` seeds the circle from the annotated `wheel`-class object's own
  bbox in *that* frame, not a fixed pixel constant - deliberately different from `pipeline.md` §5.5's
  own production method ("circle seeded from fixed geometry + known diameter, not blind Hough"), which
  describes seeding for raw camera frames with no human in the loop. This tool has a human-drawn wheel
  outline already, and a fixed pixel radius would be wrong the instant camera distance/zoom varies
  between frames - there is no camera calibration yet to convert a real-world diameter into a per-frame
  pixel radius (`FINAL_AIML_ARCHITECTURE` §16, still open). `config.py`'s new `wheel_unwrap_*` settings
  are marked "not certified, engineering placeholder" - the same treatment `safety_critical`'s keyword
  seed already carries - and control only padding/output resolution, never a wheel diameter in mm, so
  there is no false claim of an authoritative wheel spec to walk back later.
- `log_polar_service.unwrap_log_polar` wraps `cv2.warpPolar`. Its axis convention turned out to be
  undocumented and counter-intuitive: `dsize=(w, h)` maps `w` to the *radius* axis and `h` to the
  *angle* axis of the output - confirmed empirically with a single known point (radius=50, angle=0)
  landing exactly where the math predicts, only after an initial synthetic-wedge test gave misleading
  full-range results (the wedge straddled the 0°/360° wrap boundary, which looks like "no angular
  structure" if you don't already know to check for that). Getting the swap+transpose wrong would have
  silently produced a tall narrow image instead of `pipeline.md`'s "flat strip" - pinned by a unit test
  asserting a constant-radius ring lands in a narrow *horizontal* band, not vertical.

`export_service._write_wheel_unwraps` runs per image, skipping (not guessing) when there's no annotated
wheel object or nothing fine_structure to unwrap - an unwrap with nothing on it isn't useful training
data. Writes `logpolar/<split>/images/<image_id>.png` (the unwrapped strip) and one
`logpolar/<split>/masks/<image_id>__class{id}.png` per fine_structure class present, mirroring W-3's
raw-space `masks/` layout. The manifest's new `wheel_unwrap` block records exactly which version and
parameters produced those files - `certified: false` is load-bearing, not decoration, matching how
W-3's per-class rasterization shares logic with this via the new `_fine_structure_masks_by_class` helper
(pulled out of `_write_fine_structure_masks` so both consumers rasterize identically rather than risking
drift between two copies of the same fill-polygon loop).

Verified against the real API on a synthetic wheel image (no real wheel-shelling footage exists yet -
the view starts empty, per README.md): a `wheel` class and a `shelling` class both classified correctly
by the existing safety_critical/fine_structure keyword seeding without any code change, an export that
produced `wheel_unwraps: 1` and the full versioned params block, and the unwrapped mask file's nonzero
pixels landing at the *exact* row predicted by OpenCV's own log-polar formula
(`row = (out_height / ln(maxRadius)) * ln(r)`) for the shelling polygon's true distance from the wheel
center - not just "a mask exists", but its position is mathematically exactly where the transform should
put it. An initial linear-radius sanity check flagged what looked like a discrepancy; recomputing with
the log-scale formula (since `wheel_unwrap_log_scale` defaults `true`) matched the observed pixels to
within a fraction of a row, closing out the check rather than leaving it as an unexplained gap.

W-1 through W-4 slot into milestones M1–M2. W-5 belongs with the wheel work, after M2. None of them
change the M0–M4 shape already approved.

**Agreed build order:** W-2 → W-4 → then W-3 / W-1. W-2 first because a confirmed-empty frame is
currently dropped at export, so negative samples are being lost *while annotators work*.

## M4 population and M5 — as built (2026-08-05)

**Product decision:** the golden set is populated from the existing, already-verified ~2000-image
`side_view` dataset, rather than waiting on named domain experts to curate a fresh batch from scratch
(D-Q4's original blocker). This dataset predates the tool's second-review gate entirely, so
`annotation_reviews` has no rows for it - the gate can't retroactively certify data it never saw. A
curator invoking selection now explicitly opts into treating it as pre-verified
(`treat_all_labeled_as_reviewed=true`); the default stays `false` so a future, not-yet-verified
dataset doesn't silently get the same pass.

**`golden_selection.py`** (new, pure): `propose_golden_candidates` is a greedy set-cover, not a fixed
per-class quota loop - at each step it picks whichever remaining image covers the most still-
outstanding class minimums, so reaching every class's coverage floor doesn't cost more images than it
has to. Safety-critical classes are weighted 2x in that scoring, so a tie is broken in their favor -
an early `target_count` cutoff should never leave a safety-relevant class at zero coverage while a
cosmetic class already has its full share. `golden_service.propose_candidates` is the thin DB-facing
wrapper (image list + per-class safety flags + second-review status); it only *proposes* - a curator
still commits via the existing `create_version`/`add_items` (M4), same permission gate as any other
write.

Executed for real, not just tested: proposed 200 images from the real 2013-image `side_view` dataset
(min 5 per class, all 30 classes covered), created golden set version 1 for that view, and added all
200 items via the real curator API. Verified against the real data at this scale, not a
5-image sample: a subsequent export reported `golden_images_excluded: 200`, and none of the 200
golden image ids appear anywhere in the exported files on disk. This population currently lives in
local dev only - it needs migration `d5a7c0f3e8b1` applied on the deploy host before the same
population can happen against production (same prerequisite M0.3 already needs).

**M5 — MLflow tracking for the in-tool pre-labeler (Scope A only).** Per Q-C, this is **never** the
pipeline team's own 8 production families - a separate, low-stakes tracking server for the SAM2/YOLO
helper that suggests boxes to annotators, wired into the *existing* `DetectorService._run_training`.

`mlflow_tracking.py` (new) is individually-best-effort per call, deliberately not one context manager
wrapping the whole training body - a single wrapping `with mlflow.start_run():` risks either
mislabeling a real training failure as an MLflow problem, or the opposite mistake: an MLflow
connectivity blip at `__exit__` marking an otherwise-successful run as failed. Every call
(`start`/`log_params`/`log_metrics`/`log_artifact(s)`/`end`) degrades on its own instead, matching the
"never fail the primary action" contract `object_store.py` already established for snapshot
publishing - a deployment that never sets `MLFLOW_TRACKING_URI` needs no reachable server at all,
training just runs untracked.

Fixes the P-7 bug the gap-analysis flagged: training artifacts (`results.csv`, PR curve, confusion
matrix) were previously silently deleted by `_run_training`'s own `finally: shutil.rmtree(staging_dir)`
with nothing ever having read them - only `best.pt` was copied out first. `mlflow_tracking.log_artifacts`
now runs on the success path before that rmtree.

**Auto-trigger (M9-adjacent):** `export_service._finalize_snapshot` now kicks off
`DetectorService.start_training(trigger="export_handoff")` automatically whenever a snapshot
**genuinely new** (`created=True` - M2's content-addressing already dedupes a re-export of unchanged
data, so nothing new to train on) finalizes. Gated by `settings.auto_train_on_handoff` (default
`true`), and wrapped so a failure to start training can never fail the export response - the snapshot
is already safely on disk and recorded by the time this runs.

**Update: the M8 GPU-scheduling guard now exists**, added specifically so `auto_train_on_handoff`
could be called safe for a live multi-annotator deployment (it originally shipped without one - see
below for what that gap looked like). Scope is deliberately the one piece the build plan's own [HIGH]
risk actually names - "inference always wins" - not the rest of M8 (scheduler, data watcher,
monitoring), which remain unbuilt and out of scope here.

`gpu_scheduler.py` is a process-wide, in-memory tracker: `track_inference()` wraps every SAM2
GPU-touching call (`sam_service.py`'s `set_image`/`predict_box`/`predict_points`), and `is_gpu_busy()`
reports true while any call is in flight or finished within the last 5 seconds - a short grace window
so a training-start check landing between two clicks from the same annotator doesn't slip through.
Process-wide rather than per-session because SAM2 itself is deliberately one shared instance across
all sessions (`sam_service.py`'s own docstring) - "is the GPU busy" has to mean *any* session's
inference, not whichever session happens to be training.

**Check-before-start, not runtime preemption.** `detector_service._wait_for_gpu_idle` polls
`is_gpu_busy()` (every `gpu_wait_poll_seconds`, default 5s) right before the GPU-heavy
`model.train()` call - after dataset assembly, before the MLflow run even starts, so a deferred job
never creates a spurious started-then-abandoned run. Bounded by `gpu_wait_max_seconds` (default 600s);
past that, the job is marked **`status: "skipped"`** (a new, distinct status - not `"failed"`, since
nothing about the data or setup was wrong, the GPU just stayed busy longer than this job was willing
to wait). Only guards the *start*; true preemption of an already-running training loop is a
deliberately different, larger problem this doesn't attempt.

Verified live, not just unit-tested: real continuous SAM2 traffic (via the actual
`/api/generate-mask` endpoint) held a real training job at `stage: "waiting_for_gpu"` for the full
duration of the traffic, then let it proceed within 9 seconds of the traffic stopping, and the job
completed all 100 epochs normally. Caught and fixed a real bug in the process: `stage` had nothing
resetting it from `"waiting_for_gpu"` back to `"training"` once the guard cleared, so a job that had
genuinely waited, resumed, and finished all 100 epochs still *reported* stuck on "waiting_for_gpu" the
whole time - purely a status-display bug, training itself was correct throughout, but exactly the
kind of thing that would have looked like a hang to anyone watching the job.

**What the gap looked like before this existed**, for the record: there was no "inference always
wins" scheduling. `MIN_TRAINING_IMAGES` is only 2, so almost any handoff was enough to trigger a real,
`EPOCHS=100` GPU training run with nothing stopping it from competing with SAM2 serving live
annotators on the same GPU. `auto_train_on_handoff` defaulted on regardless, but was flagged as not
safe for a live multi-annotator deployment until this guard existed.

**Verified live end-to-end** against a standalone dev MLflow instance (not the docker-compose service,
which needs the full stack up): a real export against the 2013-image dataset with genuinely
completed, labeled images auto-triggered training (`auto_train_job_id` returned in the export
response), and the resulting run in MLflow shows `status: FINISHED`, all 13 per-epoch metrics, 119
params, the `trigger`/`dataset_key`/`mode`/`detector_version` tags, and all 19 training artifacts
(`results.csv`, both confusion matrices, PR/F1/P/R curves, `args.yaml`, weights/) - confirming the P-7
fix actually works, not just that it compiles.

Two real bugs found and fixed only because this was run for real rather than left as "should work":

1. **ultralytics ships its own built-in MLflow integration**, auto-enabled the instant `mlflow` is
   importable (true unconditionally now that `mlflow-skinny` is a hard dependency). It reads
   `MLFLOW_TRACKING_URI` from `os.environ` directly; this app's `Settings` parses `.env` into its own
   object without exporting it to the process environment, so ultralytics' callback couldn't see the
   URI this module's own `mlflow_tracking.start()` had just configured, fell back to a local file-store
   path, and called `mlflow.set_tracking_uri()` with that fallback *mid-training* - global module state,
   so it silently redirected every subsequent `log_metrics` call, ours included, off the real server.
   Fixed by exporting the same URI into `os.environ` in `_run_training` so both this module's calls and
   ultralytics' own resolve to the identical server (ultralytics then finds our run already active via
   `mlflow.active_run()` and logs into it rather than starting a competing one). Deliberately **not**
   fixed by disabling ultralytics' integration via its own `SETTINGS['mlflow']` - that's a *persisted,
   per-user* JSON file at `~/.config/Ultralytics/settings.json`, shared with any other ultralytics usage
   on this host outside this app entirely (this host runs the user's own unrelated training scripts) -
   flipping that off here would have been exactly the kind of unrelated-system side effect this project
   works to avoid.
2. **ultralytics' own metric keys carry parentheses** (`metrics/precision(B)`), which MLflow's REST API
   rejects outright ("Names may only contain alphanumerics, underscores, dashes, periods, spaces and
   slashes") - every one of this module's own `log_metrics` calls failed on every single epoch, 100/100,
   silently absorbed by `mlflow_tracking`'s best-effort contract so training itself never noticed or
   surfaced it. Only caught by actually reading the exception tracebacks the "ignored" log line
   deliberately still records. Fixed by sanitizing keys (stripping `(`/`)`) before logging, matching
   ultralytics' own `sanitize_dict` convention for the same values.

Both bugs would have shipped invisibly if this had stopped at "the code runs without raising" - training
completed successfully both times regardless, precisely because the best-effort contract is designed to
never let a tracking failure surface as a training failure. The only way to know logging was actually
working was to check MLflow itself.

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
