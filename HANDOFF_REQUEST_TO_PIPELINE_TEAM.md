# Annotation Module → VB Pipeline: 6 questions, after reading the architecture docs

**From:** Annotation module team
**To:** AI/ML team, VB pipeline team
**Re:** Two label-format blockers, one proposal, and confirmation on the handoff manifest
**Reply format:** answer inline under each numbered item — Q1 and Q2 are the ones that block us

---

## 0. Read receipt, so you know what we're not asking

We've read `pipeline.md` and `FINAL_AIML_ARCHITECTURE` end to end. They answered most of what we were
about to ask you, so this is a much shorter list than it would have been:

**Taken as settled, from your docs — no reply needed:**

- `exclude_classes: [leakage]`, complete. Excluded at the tool level on our side.
- Spine-stamp fields and their exact value vocabularies — `side` = `L`/`R`, `view` = `entry`/`exit`,
  `coach_type` = `LHB`/`ICF`, `zone` = `P1`/`P2`/`P3`. We'll use the detection-record shape from
  `FINAL` §9 verbatim rather than inventing parallel names.
- All confidence/metric thresholds (`pipeline.md` §12, `FINAL` §10) and the per-model metrics
  (mAP50 / acc+macro-F1 / AUROC / Dice+length-recall / recall@FP).
- Per-model `gate_tiers` — `wheel_seg` and `fastener` are safety; `crack_seg` and `detect` structural.
- Inference hardware = on-site DGX Spark GB10, 128 GB unified, all models resident.
- MLflow stays yours (`FINAL` §13: `model_manager.py` + MLflow). We stage snapshots to our own object
  store; you import. We are not asking for write access and are not standing up a second registry.
- **Retrain promotion is shadow-mode canary, sequential, no A/B.** We had "A/B promote" in our own
  plan, inherited from the superseded cloud doc. Corrected on our side.
- **No upload envelope / `dropped_regions`.** We'd been carrying that from the old topology. The
  handoff is a local file read now, and QA-invalid areas are `coverage = degraded` /
  `data_unavailable`. Corrected on our side too.

---

## Q1. Which coordinate space should P1 (side) labels be in? — **blocking, highest priority**

Your docs say the P1 detection head trains and runs on **stitched panoramas**, in panorama coordinates:

> C10: "Stitched panorama for component localization, raw frame for defect/metrology pixels"
> §5 P1 table: "**stitched panorama**, per side, tiled 1280px → boxes+class+conf, keypoints — **panorama coords**"
> Build order step 4: "P1 head **trained/run against stitched panoramas, not raw frames**"

**Our largest annotated dataset (`side_view`, ~2000 images) is annotated on individual raw frames.**

We can transform coordinates through `H_cam` — that part is mechanical. The problem is not
coordinates, it's that **the labels themselves were made without full-height context**:

> §2.1a, the reason stitching exists: "A single camera's frame can cut a tall component (door, window
> pillar, tall bracket) at its top/bottom edge."

An annotator looking at a raw frame annotates that component **as it appears — truncated**, or as two
separate objects in two frames. Transforming those into panorama space produces correctly *positioned*
but wrongly *shaped* boxes. The context is not recoverable after the fact. Seam-spanning components
(§2.1a step 5, risk R-SEAM) are the worst case.

**Which is it:**

- **(a)** `side_view` is meant for the **P1 detection head** → then we need panoramas to annotate,
  which means build-order step 3 (stitcher + homography) has to land first, and its own prerequisites
  are still open in your §16 (stitcher library, calibration method). Please tell us the ETA, and
  whether the existing raw-frame labels are salvageable or should be treated as a different dataset.
- **(b)** `side_view` feeds **`p1_side_damage`** (the defect-state classifier), not the detection head.
  Then raw-frame labels are exactly right — §2.1a step 4 says the classifier and metrology run on
  **native raw crops**, never panorama pixels — and we additionally need a **new panorama dataset view**
  later for the detection head.
- **(c)** Something else.

**We think (b) is likely and would be the good outcome**, but we are not going to assume it — the
difference decides whether ~2000 annotated frames feed the model we thought they did.

## Q2. `buffer_visible` — is our `buffer` view the training data, and what label shape do you want? — **blocking**

Your docs flag this data as missing, twice:

> §2.2 open items: "training-data availability for the new `buffer_visible` class (**not covered by any
> existing dataset** referenced elsewhere in this repo's docs)"
> R-BUFFER: "new component, **no training data yet**"

We have a **`buffer` dataset view** — a raw, unlabeled frame dump, no pre-existing class list, classes
added as annotators encounter components. It looks like it was created for exactly this. Confirm?

If yes, there's a format question, because §2.2 leaves the architecture open:

> "**lightweight single-class classifier** — `buffer_visible` (yes/no) + confidence. **Open choice:** a
> classifier is proposed over a full detector (bounding box) because the triggering use case only needs
> presence, not localization … Swap to a lightweight detector later only if a bounding box proves
> useful for QA visualization. **Confirm before building.**"

A per-frame yes/no flag is a label type our tool does not currently produce — we produce boxes and
polygons. **Our recommendation, which costs nothing and preserves both of your options:**

> **We annotate a box around the buffer.** `buffer_visible = true` is then derived at export (box
> present), and you get a bounding box for free if you later take the detector variant. A box reduces
> to a presence flag; a presence flag does not reduce to a box.

One thing we need from you either way: **how do you want negatives represented?** For a presence
classifier, frames where the buffer is *not* visible are training data, not skipped data — but our tool
currently treats an unannotated frame as "not done," not as "labeled negative." We need an explicit
"reviewed, nothing here" state, and we'd like to build it to match how you want to consume it.

Two smaller things we can help with once we have data, flagged because your doc lists them as open:
`TAU_BUFFER` ("needs labeled data to tune") and the debounce window ("needs a value derived from
typical buffer dwell-time in frame at line speed"). Both are measurable from labeled frames plus
`encoder_mm` — tell us if you want us to produce those numbers rather than just the labels.

## Q3. We want to change how we export crack/corrosion and shelling labels — confirm the format

`pipeline.md` §5.4 scores crack/corrosion on **"Dice/IoU + length-recall (not box mAP)"**, and §5.5
outputs shelling **length**. Three properties of our current export work against precisely those:

1. Our mask→polygon step keeps **only the largest external contour**. A crack that branches or breaks
   into segments loses every piece but the biggest. **Length-recall is hit directly, and silently.**
2. We simplify with Douglas-Peucker at `epsilon = 0.002 × contour perimeter`. Fine for a coupler
   outline; for a hairline crack that smooths away the thin structure being measured.
3. We export **polygons only**, no mask rasters — so even a good in-tool mask is downgraded on the way
   out.

This is ours to fix, not a request. We propose, for crack/corrosion and shelling classes specifically:
**export binary mask PNGs alongside the polygons, and keep multi-contour polygons with simplification
effectively off.** Please confirm which you'd rather consume — mask rasters, un-simplified
multi-contour polygons, or both — before we build it, so we're not converting formats twice.

Related, and cheaper to answer than to guess: your crack-seg trains on **gated SAHI tiles, 320 px,
0.20 overlap, inside the anomaly ROI**. Do you want us to ship **full frames + masks** and you tile, or
should the snapshot contain **pre-tiled 320 px crops**? We assume the former (tiling depends on the
anomaly ROI, which is a runtime artifact we don't have) — confirming because it's a one-word answer.

## Q4. The frozen golden eval set — a proposal, not a request

Correcting our earlier framing: we had been asking you for **named golden-set curators**, as though the
golden set were an agreed component and only the staffing was missing. Having read both docs, it isn't
— **neither document contains a frozen golden eval set or any per-class offline gate.** `FINAL` §12 is
review → dataset → retrain → MLflow → shadow-mode canary → promote.

So this is a proposal for you to accept or reject:

> **Add a per-class offline gate on a frozen golden set, before the shadow canary.** A candidate must
> meet or beat current production on **every safety-relevant class**, not on aggregate. We curate and
> version the set; you run the eval. Nothing that trains ever sees it.

Our argument, using what your docs now confirm:

- `wheel_seg` and `fastener` are **safety** tier (§12), and C6 says safety checks are never skipped.
- C4/R1 confirm the verdict is **post-departure, alert-only** — it does not gate dispatch. A missed
  safety defect has already left the depot. That makes a cheap filter *before* live exposure worth
  more, not less.
- A shadow-agreement metric cannot see the failure mode we're most worried about: a retrain that
  improves aggregate mAP while regressing one defect class. Agreement with production looks fine;
  "cracked wheel" recall quietly dropped.
- Cost is low and entirely on our side (curation) plus one eval script on yours.

**If you accept:** we need named domain experts to curate. Per-class example counts derive from each
class's actual data balance rather than a fixed count, and curation is continuous — but each golden
snapshot used to gate a given promotion is frozen and immutable once in use.

**If you reject:** say so and we'll drop it from our plan and stop building the storage for it. That's
a legitimate answer and we'd rather have it now than half-build it.

## Q5. Which zones does our existing footage actually cover?

Two of our four dataset views may not correspond to anything you consume, and we'd rather find out now:

- **`underbelly`** — your P2 is **cams 5 and 6 (area cameras)**, and `pipeline.md` §0 puts the
  **underbelly line-scan feed explicitly out of scope**. If our underbelly footage came from the
  line-scan camera, it feeds nothing in this pipeline. Which is it?
- **`wheel_shelling`** — your wheel-seg masks live in **log-polar unwrapped space**, from a crop
  "seeded from fixed geometry + known diameter — not blind Hough" (§5.5). Our tool annotates raw frame
  space and has no unwrap. So this view cannot currently produce `p3_wheel_shelling` labels at all.
  We'd need the geometry priors (wheel diameter, expected circle position per camera) to build the
  unwrap into the tool. Is that wanted, and do those constants exist?

## Q6. Manifest confirmation, and `coach_type`

Draft snapshot manifest below (§7). Two specific asks:

**(a) Field names** — tell us whether your import step expects particular names, or whether we define
them and you adapt. Either is fine; changing it later is rework both sides.

**(b) `coach_type` on annotations — we'd like to add it now.** Your manifest is coach-type-conditional,
unknown coach-type is a hard refusal (§8: "refuse to guess … never assume a manifest"), and R14 names
"manifest/coach-type error → false-missing storm."

Our labels currently carry **no `coach_type`**. That means our per-class label counts can't answer R10
("label scarcity — needs label counts") properly: an aggregate count hides a class that's well covered
on LHB and absent on ICF.

Unlike `longitudinal_position_mm` — which we will **not** compute locally, since it comes from encoder
counts and we have no calibration constants to derive it from safely — `coach_type` is two values and a
human can set it per batch today, without waiting for pipeline ingestion. **We plan to add it as a
per-batch field unless you object**, and to break `per_class_counts` down by it.

---

## 7. Draft snapshot manifest — see Q6(a)

Every export becomes an immutable, content-addressed snapshot. `snapshot_id` hashes the sorted
manifest, so an identical dataset yields an identical id and any change yields a different one.
Nothing is overwritten in place. Field names and value vocabularies taken from `FINAL` §9.

```json
{
  "snapshot_id": "sha256:<hex>",
  "created_at": "2026-08-05T11:42:07Z",
  "dataset_view": "side_view",
  "target_families": ["shared_backbone_v1", "p1_side"],
  "class_map": {
    "version": "<your class-map version>",
    "taxonomy_version": "<component_defect_taxonomy.yaml version>",
    "names": { "0": "coupler", "1": "brake_cylinder" },
    "exclude_classes": ["leakage"]
  },
  "label_format": {
    "0": "polygon",
    "1": "polygon",
    "7": "mask_raster+polygon"
  },
  "golden_set_version": "<null if Q4 rejected>",
  "splits": { "train": {"images": 1834}, "val": {"images": 203} },
  "split_integrity": {
    "pseudo_synthetic_in_val_or_test": 0,
    "auto_accepted_in_val_or_test": 0,
    "golden_ids_in_any_split": 0
  },
  "provenance": {
    "propagated_labels": 412,
    "auto_accepted_images": 87,
    "human_annotated_images": 1950,
    "second_reviewed_images": 1901,
    "grandfathered_unreviewed_images": 49,
    "audit_sampled_images": 143,
    "annotator_ids": [1, 3, 4],
    "reviewer_ids": [2, 5]
  },
  "per_class_counts": {
    "LHB": { "0": 812, "1": 640 },
    "ICF": { "0": 392, "1": 243 },
    "unknown": { "0": 0, "1": 0 }
  },
  "spine_stamp_coverage": {
    "fields": ["coach_index","coach_type","axle_id","side","view","longitudinal_position_mm","zone"],
    "frames_with_full_stamp": 0,
    "frames_with_coach_type_only": 0,
    "frames_missing_stamp": 2037
  },
  "files": { "images": "images/", "labels": "labels/", "masks": "masks/", "checksums": "checksums.sha256" }
}
```

Three fields we specifically want a reaction to:

- **`split_integrity`** — asserted at export, not trusted. Non-zero on any count and we refuse to
  publish the snapshot. Tell us whether your import re-verifies or treats a published snapshot as
  already passed.
- **`provenance`** — deliberately includes the uncomfortable number.
  `grandfathered_unreviewed_images` counts images completed before our second-reviewer gate went live;
  by product decision they stay exportable without retroactive review. If that number is non-zero, a
  model trained on the snapshot **cannot** claim "all data second-reviewed." Better you see it in a
  field than discover it later.
- **`per_class_counts` broken down by `coach_type`** — this is the shape R10 actually needs (Q6b).
  Currently `unknown` would hold everything.

---

## 8. Scope boundaries, restated

**We do not** train, serve, or deploy any of the 8 families; ship weights or engines; run FP8/TensorRT
export; run the promotion gate; or compute drift. Our loop closes at "next batch of triaged frames
arrives," not "model shipped."

**We do** train one thing, so it isn't mistaken for one of yours: a local YOLOv8 **pre-labeler** that
proposes boxes to speed up human annotation. Never report-authoritative, not one of the 8 families.

**Still deferred, and on your critical path more than ours** — flagging because your build order needs
them and we can't deliver them under current scope:

| Label type | Your family | Blocked on |
|---|---|---|
| **Confirmed-normal curation** (highest priority) | `p2_under_anomaly` (PatchCore) | Not a polygon task — a keep/reject normal-set + purity gate. Your build order puts P2 anomaly at step 6 as "the ROI gate that fits P2 in budget", so this is the nearest deferred item to your critical path. |
| Crop-classify (13 conditions per §5.2) | `p1_side_damage`, `p3_comp_defect` | Class-only, no mask. Same missing capability as Q2's presence label. |
| Fastener slot-occupancy | `p3_fastener` | Binary occupied? per known bracket slot. Needs the slot position priors. |
| Wheel log-polar seg | `p3_wheel_shelling` | Needs geometry-seeded unwrap in the tool — see Q5. |

On PatchCore specifically, since it's the one we'd pull forward first: one mislabeled "normal" sample
corrupts the memory bank directly, with no gradient step to absorb it. Whoever builds that curation UX
needs a purity gate cross-checking every confirmed-normal sample against known defects, refusing to run
rather than silently dropping suspects. `FINAL` §13 lists `DefectReviewLog` as an existing asset — we'd
want that as the cross-check source. **If anomaly-first shipping is still the plan, tell us and we'll
re-plan to bring this forward.**

---

## 9. Summary

| # | Question | Blocks | Type |
|---|---|---|---|
| **Q1** | P1 labels: panorama space or raw space? | ~2000 annotated frames' destination | **decision needed** |
| **Q2** | Is `buffer` our `buffer_visible` data, and what label shape? Plus: how to represent negatives? | your R-BUFFER / no-training-data item | **decision needed** |
| Q3 | Mask rasters vs multi-contour polygons for crack/shelling? Full frames or pre-tiled? | our export rework | confirm format |
| Q4 | Accept or reject the frozen golden-set pre-canary gate? | whether we build the storage at all | accept/reject |
| Q5 | Is `underbelly` area-cam or line-scan? Do wheel geometry priors exist? | whether 2 of our 4 views feed anything | clarify |
| Q6 | Manifest field names; and we plan to add `coach_type` per batch | snapshot format, R10 label counts | confirm |

Q1 and Q2 are the ones we'd like this week. Q3–Q6 can follow.

Two things worth 20 minutes on a call rather than a thread: **Q1** (because the answer may change what
we annotate next) and the **deferred-label table in §8** (because confirmed-normal curation looks like
it's closer to your critical path than to ours).
