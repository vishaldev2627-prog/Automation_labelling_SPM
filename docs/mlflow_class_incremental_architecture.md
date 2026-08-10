# MLflow Class-Incremental Promotion Architecture
### Annotation → Training → Promotion, for a growing class set

*Design analysis for `Automation_labelling_SPM`, branch `automation_pipeline_mlflow_prep`. Read-only analysis — no code changed. Prepared 2026-08-10. Updated 2026-08-10 — see status note below.*

**How to read this document:** §0 below is the whole argument in plain language — read only that and you'll understand the problem, the fix, and why it's designed this way. Everything after §0 is the same argument again, but rigorous: exact file/line references, thresholds with a calibration method instead of a guessed number, full pseudocode, a worked example, and a file-by-file implementation plan. A glossary of every term used is at the very end (§L) — jump there any time a word doesn't parse.

---

## 0. The whole thing, in plain language

**The problem you actually have.** Right now, when a new model is trained and it looks better on paper (a higher overall accuracy number), the system has no way to tell *why* it looks better. It could be better because it genuinely got better at everything. It could *also* look better purely because it learned a brand-new class, while quietly getting *worse* at classes it already knew — and one global number can't tell those two situations apart. That's a real risk: if the system (or a person skimming one dashboard number) treats "higher overall score" as "promote it," a model that got worse at detecting an existing, already-trusted defect could silently go live, just because it also happened to learn something new.

**The fix, in one sentence.** Instead of asking "is the new model's overall score higher?", ask three separate questions and answer each one on its own terms: *(1) Did anything the model already knew get worse? (2) Is anything it newly learned actually good enough to trust? (3) Does it still run fast enough and stay small enough to deploy?* Only if the answer to all three is "no problem" does it get promoted — and if the answer to *any* of them is "yes, there's a problem," it's rejected, no matter how good the overall number looks.

**Why this is harder than it sounds — the "growing class list" problem.** A defect-detection model's job is to recognize a list of component/defect types. That list isn't fixed forever — annotators keep finding new things worth labeling (a new defect type, a new component variant), so the list of classes the model needs to know grows over time. The moment the class list itself changes between the old (production) model and the new (candidate) model, a plain side-by-side score comparison stops making sense: the old model was *never trained* on the new class, so it has no score for that class to compare against — there is nothing to be "better" or "worse" than. The new class needs a completely different kind of check: not "is it better than before" (there is no "before"), but "is it good enough, on its own absolute terms, to ship."

**The three-question fix in more detail (this is exactly §H/§I below):**
1. *Classes both models already know* — compare candidate vs. production, class by class, and flag any that got meaningfully worse. "Meaningfully" is deliberately not zero-tolerance (raw scores wiggle a little between any two training runs, that's noise, not a regression) and it's stricter for a safety-relevant class than a cosmetic one.
2. *Classes only the candidate knows (the new ones)* — instead of comparing to a production score that doesn't exist, check the new class against an absolute quality bar — a bar calibrated from real data, not a guessed number (§E explains exactly how to calibrate it).
3. *Operational reality* — even a candidate that's accuracy-perfect on both of the above can still be the wrong model to ship if it's slower, bigger, or throws more false alarms than the budget allows. That's checked too, separately.

The overall/global score is still computed and shown to a human — it's useful context — but it is **never**, by itself, the reason a model gets promoted or rejected. That's the one-sentence fix restated as a rule: *no single number decides; three independent checks do, and any one of them can veto the other two.*

**Where this plugs into what already exists.** The good news: this repo already has almost everything needed to *run* this decision — a system that tracks every training attempt (MLflow), a place to register "here is a candidate, here is the currently-live version" (MLflow's Model Registry), a frozen, never-trained-on set of images used only to grade a finished model (the "golden set"), and — critically — a human has to explicitly approve a promotion; nothing here ever auto-deploys anything. What's missing is *just the decision logic itself* — today, the code technically produces a verdict, but that verdict quietly ignores brand-new classes (see §A/§B for exactly why) and has no separate "is this class new" awareness at all. This document proposes filling in that one missing piece — a `compare()` function and a `should_promote()` function — without touching or replacing anything that already works.

**One more piece: when is a new class even "ready"?** Before a brand-new class can go through the three-question check above, it needs *enough labeled examples* to be trainable at all. The original idea on the table was "once a class reaches 8% of the dataset, it's ready." §E walks through why that's the wrong primary rule (a percentage says nothing about whether there's *enough absolute data*, or whether that data is *varied enough* to generalize) and proposes a better, multi-part check — while keeping the 8% idea alive as a cheap early-warning trigger, just not as the actual gate.

**What you'll find below, section by section:**
- **§A–§B**: what the code does today, and exactly where today's logic falls short (with file/line citations).
- **§C**: the one new piece being added, and where it sits relative to everything that already exists.
- **§D**: the life story of a single class, from "someone labeled it once" to "it's a trusted part of production" to (if it ever needs it) "retired."
- **§E**: the data-sufficiency question — how much labeled data is actually "enough," and how to find that number empirically instead of guessing.
- **§F**: how training itself should behave once the class list can grow.
- **§G**: exactly what to record in MLflow (which fields, which tags) so every decision is explainable after the fact.
- **§H–§I**: the actual decision logic, as runnable-shaped pseudocode.
- **§J**: a full worked example, stage by stage, showing a rejection and then a later promotion.
- **§K**: the literal list of files to touch, in the order to touch them.
- **§L**: a glossary — every acronym and term used above, defined in one place.

---

## Status update — what's shipped since this was first written

**This document's core subject — §C–§K's class-lifecycle/eligibility/promotion-gate design — is now built, Modules 1–9 of the implementation plan.** Every piece described below as a proposal (`class_eligibility_service.py`, `promotion_gate.py`, the `state`/`tier`/`deprecated_at` columns, the layered `compare()`/`should_promote()` logic, the full §G tag table, warm-starting, per-class FP/FN) is real, running code — verified against real Postgres, a real (sqlite-backed) MLflow registry, and real GPU training runs, not just unit tests against fakes. What follows in §A–§K is kept exactly as originally written (it's still the accurate design reference), but read every "not yet implemented" caveat below as historical unless a note says otherwise. Three things worth knowing specifically:

1. **Enforcement, not just tagging.** §C's diagram shows the gate feeding a tag onto the MLflow model version — that part was always the easy half. The harder half, added specifically because a live verification pass in this session caught it as a real gap, is in `model_promotion_service.approve()`: a `REJECT` decision now blocks the actual live-model swap by default (overridable only with an explicit, audited reason), and a safety-tier `hard_fail` blocks it **unconditionally**, no override accepted at all. Before this, the gate's verdict was purely advisory — a human could approve a flagged-regressed version with nothing in the code stopping them.
2. **A real scale bug was caught before it shipped.** The regression-tolerance and new-class-floor thresholds in §H/§I's pseudocode are written in the doc's own "91% AP" percentage language. The actual `PromotionThresholds` config defaults had to be expressed as 0–1 fractions instead, matching what Ultralytics' `mAP50` genuinely returns (e.g. `0.671`, never `67.1`) — mixing the two scales would have made the new-class floor reject every class unconditionally and made the regression tolerance nearly meaningless against real deltas. Caught by testing against real Ultralytics output before wiring real data through it, not left as a latent bug.
3. **Metric basis (from the earlier update, still true)**: `golden/class{N}_mAP50` means mask AP, not box AP, since the YOLO11-seg move - see point 2 of the original note below.

The original note this replaces, for the specific YOLO11-seg/polygon/`MaskSource` changes (a separate track, landed before Modules 1-9 above):

1. **Base checkpoint**: every `yolov8s.pt` reference below is `yolo11s-seg.pt` — `MODEL_WEIGHTS`/`TASK` in `detector_service.py`.
2. **What the per-class metric actually measures**: `golden/class{N}_mAP50` is **mask AP**, not box AP — `golden_eval_service.evaluate_on_golden_set` reports seg metrics under the primary `mAP50`/`precision`/`recall` keys, with `box_mAP50`/`box_precision`/`box_recall` kept alongside for continuity. `detector_service` tags every run with `eval_task` so that's discoverable rather than assumed.
3. **A safety-relevant field worth reusing**: `AnnotationObject.mask_source` (`sam2` | `detector` | `null`) - a signal produced by a different model than the one a threshold was calibrated against must never be treated as equivalent just because the number looks the same. `auto_accept_service` enforces this for masks; §E/§H's new-class absolute-floor logic (now built, per point 1 above) got the analogous per-tier calibration treatment, not a borrowed number.

---

## Scope note — please read before the rest

This repo builds and owns **one** trainable model: a YOLO11-seg "pre-labeler" that gives annotators a head-start polygon (not just a box) before SAM2 (`backend/app/services/detector_service.py`). Everything under §A–§K below — the class-lifecycle, the promotion gate, the comparison algorithm — is written generically, but it is grounded in and cross-referenced against **this repo's actual code**, not an abstraction. It applies directly to the pre-labeler today. If a separate "8-model VB detection pipeline" is ever brought under the same MLflow instance, the same policy applies unchanged — the gate doesn't know or care how many model families call it, only that each one is compared against *its own* production baseline.

Everything described as "already built" below is real, present, running code in this branch — not a proposal. M6 (golden eval), M7 (registry + recommendation), M7.5 (promotion sync + human gate), M8 (GPU guard) landed in the last four commits; the YOLO11-seg/polygon/MaskSource work landed since, on top of the same M6/M7/M7.5 machinery, which is why the metric-basis note above matters. The gap this document closes is specifically the **dynamic-class problem**: none of M6/M7/M7.5 currently knows that a candidate's class *set* can differ from production's, so a class-set change today gets silently treated as noise in a per-class metric dict rather than as a structural event with its own evaluation layer.

---

## A. Current architecture

```
Annotator (SAM2-assisted)                                    class_map_service
    │  save / complete / review                                     │ mints an immutable,
    ▼                                                                │ content-hashed version
Postgres: annotation_state, annotation_history,                     │ whenever the on-disk
          annotation_reviews, dataset_classes  ◄────────────────────┘ class list changes
    │
    │  POST /api/export
    ▼
export_service  →  content-hashed dataset_snapshots (immutable)  →  MinIO staging bucket
    │                                                                   (pipeline-team pull, Q-C)
    │  auto_train_on_handoff=true, on a genuinely-new snapshot only
    ▼
detector_service.start_training()
    │  gpu_scheduler.is_gpu_busy() gate — "inference always wins" (M8)
    ▼
YOLO11s-seg.train(), 100 epochs, on polygon labels  →  mlflow.start_run() (annot-detector-training)
    │  params, per-epoch metrics, tags: dataset snapshot, class-map version, view, task=segment
    ▼
golden_eval_service.evaluate_on_golden_set()  →  per-class mask P/R/mAP50/mAP50-95
    │                                              (+ box_ metrics kept for continuity), on the
    │                                              SAME run (M6)
    ▼
model_registry_service.register_and_recommend()  (M7)
    │  MLflow Model Registry: creates AnnotDetector-<view> v{N}
    │  compares candidate's per-class golden mAP50 against whichever version
    │  is currently tagged Production, tags verdict: eligible / regressed / no_baseline
    ▼
Human (golden_curator) moves the MLflow stage to Production, by hand, in MLflow's UI
    ▼
model_promotion_service (M7.5)
    │  poller detects "MLflow Production changed" → inserts a `pending` ModelPromotion row
    │  a SECOND human (role=model_reviewer) reviews it, approve() or reject()
    │  approve(): downloads weights, sanity-check inference on a probe image,
    │             THEN atomically swaps the per-view registry.json
    ▼
detector_service.detect()  reads registry.json only — never talks to MLflow at request time
    ▼
Live annotators get suggestions from the swapped-in model
```

**What this gets right, already:** two independent human gates (accuracy judgment vs. deploy-safety judgment) that structurally cannot be the same person; sanity-check-before-swap so a corrupt download can never leave nothing loaded; atomic registry swap with instant, dependency-free rollback; per-view isolation so training on one dataset view can't silently overwrite another's active model (this was a real bug, since fixed via `_slug_for`); content-hashed, immutable class-map versions and dataset snapshots so a run's tags point at something that provably can't have drifted under it; a golden set structurally isolated from every write path (propagation, auto-accept, export) that could otherwise contaminate it.

**What it does not yet handle — the actual subject of this document:** `_compare_against_production` (`model_registry_service.py:53-89`) walks `candidate_per_class` and, for every class id, looks up `golden/class{id}_mAP50` on the production run. If that key is missing on the baseline run — which is exactly what happens for a brand-new class — the loop's `continue` at line 82 means **the new class contributes nothing to the verdict, in either direction.** It cannot regress (nothing to compare to) but it also cannot be *required* to clear any bar. A candidate that adds one garbage class and holds every existing class flat gets `verdict="eligible"` — same as a candidate that adds one genuinely useful class. The regression check itself is also currently unconditional-per-class with no tolerance band: `metrics["mAP50"] < prod_metrics[prod_key]` flags a regression on a *0.1-point* dip exactly as hard as a 6-point collapse, which in practice means either the check is too noisy to trust or (more likely) nobody is actually gating on `regressed_classes` today — it's a tag a human reads, not a promotion blocker. And there is no mechanism anywhere that decides *when a class is even allowed into a training run* — `detector_service._assemble_dataset` trains on every class in the current map with zero regard for how many labeled examples a given class has. A class discovered yesterday with 3 boxes enters training on equal footing with a class that has 900.

---

## B. Architecture gaps

| Gap | Where it lives today | Consequence if unaddressed |
|---|---|---|
| No per-class data-sufficiency gate | `detector_service._assemble_dataset` trains on the full current class map | A class with 3 labeled boxes destabilizes shared-backbone training and produces a meaningless per-class AP for everyone reading the golden eval |
| No class lifecycle / state | `dataset_classes` table has no state column at all | "Is class E ready to train on" is answered by eyeballing counts, not by the system |
| Comparison silently skips new classes | `model_registry_service._compare_against_production:82` | A candidate can add a useless class and register as `eligible`, identical to adding a useful one — the exact failure mode in the user's Example 9 |
| No regression tolerance band | same function, line 83 | Either false-positive noisy regressions, or (today) a check nobody actually trusts enough to gate on |
| No "common-class vs. new-class vs. overall" split in the verdict | `PromotionRecommendation` has one `verdict` field | A −6pt regression on class D and a −0.1pt wobble on class A both just say `regressed`; a curator can't tell severity or scope without opening MLflow and reading raw numbers |
| No operational gate (latency, model size, false-positive rate) | not computed anywhere | A candidate that regresses inference latency or explodes false positives can still be marked `eligible` |
| No stable global class-ID space across versions | `class_map_service` mints a new version on any change, but doesn't guarantee ID 4 always means the same thing forever | A deprecate-then-reintroduce sequence could collide IDs; nothing currently prevents it, it just hasn't happened yet |
| No explicit "class removed / deprecated" handling in comparison | comparison code only iterates `candidate_per_class` | A candidate that silently drops a class (in the config it trains, not in `dataset_classes`) would just not be checked against its production score at all — a false "no problem" |
| Golden set has no per-class *floor*, only a same-model-vs-itself relative check | `golden_eval_service` computes numbers; nothing consumes them against an absolute bar | A brand-new class can pass "no regression found" purely because there's nothing to regress against — exactly the case the user identifies as needing an absolute threshold instead |

None of this requires new infrastructure — MLflow, the registry, the golden set, and the two-gate human workflow are already there and correctly shaped. This is a `compare()` / `should_promote()` logic problem plus one new lifecycle concept, not an infrastructure problem.

---

## C. Recommended architecture

The **only** structural addition to the diagram in §A is inserting a real decision engine between "golden eval finishes" and "register + tag a verdict" — everything upstream and downstream stays exactly as built.

```
golden_eval_service.evaluate_on_golden_set()
        │  per-class {precision, recall, mAP50, mAP50-95, FP, FN} — candidate
        ▼
┌───────────────────────────── NEW: promotion_gate.py ─────────────────────────────┐
│                                                                                    │
│  class_diff(candidate_classes, production_classes)                               │
│        │                                                                          │
│        ├─► COMMON classes  ──► Layer 1: regression check (relative, tolerance)   │
│        ├─► NEW classes     ──► Layer 2: absolute-floor check (no baseline exists)│
│        └─► REMOVED classes ──► must be explicit + DEPRECATED, never silent       │
│                                                                                    │
│  Layer 3: overall candidate quality — reported, NEVER sole promotion criterion   │
│  Layer 4: operational constraints — latency, model size, FP rate, critical-class │
│                                                                                    │
│  should_promote() combines all four layers → PROMOTE / REJECT + a REASON, not    │
│  a bare boolean                                                                   │
└───────────────────────────────────────┬────────────────────────────────────────────┘
        │  structured PromotionDecision (this replaces PromotionRecommendation)
        ▼
model_registry_service: create_model_version + tag with the FULL decision
   (verdict, per-layer results, thresholds used, comparison baseline version)
        ▼
Human #1 (golden_curator) reads the decision, moves MLflow stage → Production
        ▼
Human #2 (model_reviewer) — unchanged M7.5 — approves the live swap
```

The gate is a pure function: `(candidate_metrics, production_metrics, candidate_classes, production_classes, thresholds) → PromotionDecision`. It never talks to MLflow, Postgres, or the filesystem — `model_registry_service` remains the only thing that touches the MLflow client, exactly as it does today. This keeps the gate unit-testable without a running MLflow server, which matters given the repo currently has zero tests (Part 3, P-12 of the prior analysis).

---

## D. Class lifecycle

```
DISCOVERED ──► COLLECTING_DATA ──► ELIGIBLE ──► ACTIVE ──► DEPRECATED
                                                    ▲            │
                                                    └── REINTRODUCED
```

| State | Meaning | Entry condition | What it's allowed to do |
|---|---|---|---|
| **DISCOVERED** | An annotator has labeled at least one instance. It exists in `dataset_classes` but is far below the training floor. | First object with this class id saved. | Visible in the UI for continued annotation. **Never** enters `_assemble_dataset`. Not evaluated in golden eval (no golden images have it yet, structurally). |
| **COLLECTING_DATA** | Same as DISCOVERED, just a duration/count checkpoint — exists mainly for the dashboard ("E: 34/80 images, still collecting"), not a behavioral gate distinct from DISCOVERED. | Optional — can be merged with DISCOVERED if a two-state distinction adds no operational value. | Same restrictions as DISCOVERED. |
| **ELIGIBLE** | Crosses the data-sufficiency bar (§E) on the **current, live annotation state** — not frozen at a point in time, since the bar can be crossed by tomorrow's annotation session with no dataset export. | Meets absolute-count + diversity floor, computed live off `annotation_state`. | May be included in the **next** training run's dataset assembly. Not yet in any exported snapshot or trained model. |
| **ACTIVE** | Included in the current **Production** model's class set. | A candidate containing this class as ELIGIBLE passed the promotion gate and was promoted. | Served in production. Subject to regression monitoring on every future candidate via Layer 1. |
| **DEPRECATED** | Explicitly retired — never silently. | A human marks it deprecated (e.g., component redesigned, class merged into another, coach type retired). | Excluded from the *next* training run's class set by explicit config, not by falling below a threshold. Existing production model keeps detecting it until the next promoted candidate — deprecation is a training-time decision, not an instant undeploy. |
| **REINTRODUCED** | A DEPRECATED class gets new annotation activity again. | New object saved with a deprecated class id. | Re-enters at COLLECTING_DATA — it does **not** resume ACTIVE and does **not** get a new class id (see §E of the user's ask / class mapping below). Must re-earn eligibility and re-pass the promotion gate like any other class, even though it was in production before — its old metrics are stale and its old training data may be as well. |

**Why this lifecycle is right, and one thing to watch:** the state should live on `dataset_classes` as a computed/derived column (not manually toggled, except DEPRECATED/REINTRODUCED which are inherently human decisions), recomputed whenever annotation counts change or a promotion happens — never itself a promotion input read stale. The one gap in the user's proposed 5-state list: it has no explicit way to represent "ACTIVE, but this specific promoted version's data has since fallen back below the floor" (their edge case 5/6). The fix is definitional, not a new state: **ACTIVE is sticky.** A class that reaches ACTIVE stays ACTIVE (and stays in every future training run's class set) regardless of what its percentage share does afterward, *unless* explicitly DEPRECATED by a human. Percentage share is an *eligibility* gate for new classes, never a *retention* gate for classes already in production — this directly answers edge case 6.

---

## E. Dataset eligibility — critiquing the 8% rule

**The 8% share rule, on its own, is the wrong primary gate for object detection**, for three independent reasons, each with a concrete failure the user already anticipated:

1. **A relative share is not a sample-size guarantee.** If eligible-dataset size is 60 images total, 8% is 5 images — not enough to learn a detector head for anything, let alone something visually complex. If it's 6,000 images, 8% is 480 images, likely already generous. The floor that actually matters to a CNN/YOLO head is an **absolute minimum instance count**, not a share. Object-detection literature and practice (COCO-style few-shot baselines, Ultralytics' own per-class guidance) converge on the low hundreds of *labeled object instances* (not images — one image can hold zero or many instances of a class) as a floor below which per-class AP is dominated by variance, not signal.
2. **Images vs. objects is the wrong unit if left ambiguous.** A class that appears once per image (e.g., "buffer") and a class that appears many times per image cluster in busy scenes (e.g., "fastener") reach the same *image* percentage on wildly different amounts of true training signal (bounding boxes). The gate should be defined on **object/instance count**, with image count as a secondary diversity check, not the primary one.
3. **A pure count/share floor says nothing about diversity or balance.** 200 boxes of Class E all cropped from the same three photos under the same lighting will not generalize; 60 boxes across 60 different coaches, angles, and lighting conditions will generalize far better despite failing an absolute-count-only rule. The rule needs a diversity dimension, not just a count.

**Recommended composite gate — a class is ELIGIBLE when *all* of the following hold**, computed live off current annotation state (never off a frozen snapshot, since eligibility should track today's reality):

| Dimension | Metric | Recommended floor (starting point — see calibration below) |
|---|---|---|
| Absolute instances | count of labeled, non-rejected object instances for the class | ≥ 80–150 instances (calibrate; see below) |
| Absolute images | distinct images containing ≥1 instance of the class | ≥ 40–60 images (guards against instance-count being satisfied by a handful of dense photos) |
| Diversity | distinct `coach_type` values represented, distinct source batches/dates represented | ≥ 2 coach types if the class can plausibly appear on both LHB and ICF; otherwise flagged for manual diversity review rather than auto-blocked |
| Relative share (secondary, not primary) | class instances ÷ total instances in the training-eligible pool | used as a **early-warning signal** ("E is 8% of the pool, worth checking absolute floors now"), never as the sole gate |
| Validation-set presence | after split, ≥ some minimum count lands in `val` too | ≥ 10–15 instances in val, or the class's own AP number is statistically meaningless regardless of train-side sufficiency |

**Should 8% be a hard threshold?** No — treat it as a *trigger to check*, not a *pass/fail gate itself*. It's genuinely useful as the signal that fires the "is E worth evaluating for eligibility now" check (cheap to compute, correlates with "there's now enough presence to matter"), but the actual eligibility decision should be the absolute-count + diversity table above. This also directly resolves edge cases 1 and 2 from the user's list ("reaches 8% but poor diversity" / "reaches 8% but too few absolute images") — under a share-only rule those are false positives; under the composite gate they fail cleanly on the diversity or absolute-count dimension while still showing 8%.

**Should different thresholds apply to different class types?** Yes, along one axis that already exists in this codebase: `safety_critical` (`dataset_classes.safety_critical`, already a column, already read by `auto_accept_service`). A safety-critical class (e.g., a specific defect state) should have a **higher** absolute floor and mandatory manual diversity sign-off before ELIGIBLE, not a lower one — the cost of a false "detects fine" on a safety-critical class is categorically worse than on a cosmetic one. This is the same tier concept `pipeline.md` already uses (`cosmetic`/`structural`/`safety` degrade order) — reuse it rather than inventing a parallel one.

**Oversampling/augmentation:** appropriate as a *training-time* technique once a class is ELIGIBLE (e.g., class-weighted loss, or repeat-sampling under-represented classes within an epoch) — it compensates for class imbalance during learning. It is **not** a substitute for the eligibility floor itself: augmenting 8 real images of Class E into 200 augmented ones does not create 200 instances of real-world diversity, and a per-class AP computed against a golden set built from *real, distinct* images would expose that immediately. Use it to improve the loss landscape for a class that already cleared the floor, never to manufacture eligibility.

**Should rare classes be retained but excluded from training?** Yes — this is exactly the DISCOVERED/COLLECTING_DATA states in §D. The class stays visible in the annotation UI (so people keep labeling it) and stays in `dataset_classes` (so the class-map version is stable and doesn't need to mint a new version just because a class crossed the eligibility line — the map already contains it, unassigned to any trained model yet), but is filtered out of `_assemble_dataset`'s class list until ELIGIBLE.

**Calibrating the actual numbers, not guessing them:** run a **learning-curve probe** once real annotation data exists for a candidate new class — train the shared backbone with that class at N=25, 50, 100, 150 instances (cheap: same backbone, just varying how much of the class's data is included) and plot per-class AP vs. N on the golden/held-out set. The floor is wherever that curve visibly flattens for classes structurally similar to the new one (similar visual complexity, similar bounding-box scale) — not a number picked in the abstract. Until enough historical classes exist to calibrate this empirically, start conservative (the higher end of the ranges above) and loosen only with evidence, never the reverse.

---

## F. Training strategy for a growing class set

- **Always full-retrain the shared model on the full current class set (fine-tuned from the previous production weights as the starting checkpoint, not from scratch).** This is already `detector_service`'s behavior (`MODEL_WEIGHTS = "yolo11s-seg.pt"` is the *base* checkpoint, not a from-scratch init flag — previously `yolov8s.pt`, migrated to a segmentation checkpoint trained on polygons rather than derived boxes) and is the right call for a single-backbone, multi-class detector: there is no "just add a head for E" option with a shared YOLO backbone the way there might be with a modular architecture, and full joint retraining is exactly what makes catastrophic forgetting *checkable* — you can't detect forgetting on classes A–D if E was trained as a bolt-on side model that never touches the shared backbone. Warm-starting from the *previous production version's* weights specifically (rather than always the stock `yolo11s-seg.pt` checkpoint) is still the item 7 gap in §K below — not yet implemented.
- **This is where catastrophic forgetting risk actually lives**, and it's a training-time risk, not just an eval-time one: fine-tuning a shared backbone on a class distribution that's now, say, 15% new-class E can measurably shift feature representations for A–D even before any promotion decision is made. Mitigate with (a) class-weighted sampling so E doesn't dominate batches just because it's newly emphasized in labeling effort, (b) keeping the previous production weights as the fine-tune starting point (warm start regularizes toward not-forgetting far better than from-scratch), (c) treating Layer 1 (§H) as the actual forgetting detector — don't try to prevent forgetting architecturally beyond warm-starting; catch it empirically, every time, at the gate.
- **Training dataset composition on class-set change:** include every ELIGIBLE and ACTIVE class; never include DISCOVERED/COLLECTING_DATA classes; never silently include a DEPRECATED class's leftover-labeled images unless a human explicitly re-included it (this is what makes deprecation training-time and reversible, not destructive).
- **Validate against two things, not one:** the standard held-out `val` split (used for early stopping / training diagnostics, as today) **and** the frozen golden set (used for the promotion decision). Never let the promotion gate read metrics computed against `val`, since `val` is drawn from the same annotation stream as `train` and shares its biases (propagation, auto-accept) — this is exactly the golden set's reason for existing, and it's already correctly separated in this codebase (`golden_eval_service` reads a genuinely separate id set). Keep it that way for the class-set-change case specifically, since that's exactly when the temptation to trust a good-looking `val` number on a shiny new class is highest.

---

## G. MLflow design — experiments, models, aliases, tags, lineage

**Experiments** (already correctly scoped, keep as-is): one experiment per model family/view — `annot-detector-training` today; if a second family is ever tracked here, give it its own experiment, never share one experiment across families.

**Registered models:** one per view, as today — `AnnotDetector-<view>`. Do **not** create a second registered model when the class set changes (e.g., no `AnnotDetector-side_view-v2-with-E`) — the class set is a **property of a version**, not a reason to fork the model identity. Versions of the same registered model are exactly MLflow's mechanism for "same model lineage, changing capability over time," which is precisely what §15/§16 of the ask needs.

**Aliases over raw stage strings, going forward:** MLflow's stage model (`None`/`Staging`/`Production`/`Archived`, which this repo already uses via `get_latest_versions(stages=["Production"])`) is fine and already wired up, but MLflow itself has been moving toward **aliases** (`@champion`, `@candidate`) as the more flexible mechanism — multiple aliases can point at different versions simultaneously, and an alias can be moved without the semantic baggage of "stage." Recommendation: keep reading `Production` stage for backward compatibility with the current M7.5 poller, but additionally set `@champion` on whatever version stage-Production currently holds, and set `@candidate` on every newly-registered version. This gives a stable, renamed-away-from-"stage" query surface (`get_model_version_by_alias(name, "champion")`) if MLflow's stage feature is ever deprecated, without breaking anything live today.

**Tags — what to add to `set_model_version_tag` beyond the current three** (`promotion_recommendation`, `regressed_classes`, `compared_against_version`):

| Tag | Value | Why |
|---|---|---|
| `baseline_version` | the specific production version this candidate was compared against | Makes lineage queryable without re-deriving it from `Production` stage history, which mutates |
| `class_set_diff` | `+E` / `-F` / `none` | Cheapest possible "did the class set change" signal, readable without opening per-class metrics |
| `new_classes` | `E` | Layer 2 target list |
| `removed_classes` | `` (empty unless explicit deprecation) | Distinguishes "silently absent from this run" (should never happen, see §H) from "explicitly deprecated" |
| `common_class_regression` | `none` / `D:-6.0pts` | Human-readable Layer 1 summary, so a curator doesn't have to open the run's raw metrics to see the headline |
| `new_class_floor_result` | `E:pass(76.0>=70.0)` | Layer 2 summary |
| `operational_result` | `latency:pass,size:pass,fp_rate:fail` | Layer 4 summary |
| `decision` | `PROMOTE` / `REJECT` | The gate's actual output, distinct from the softer `promotion_recommendation` verdict string — this is what should drive any future auto-behavior, kept separate from the advisory field so tightening the gate later doesn't require a schema change |
| `dataset_snapshot_id` | the content hash from `export_service` | Already computable, not yet tagged onto the *model* version (only onto the training run) — put it on both, since the version is what a promotion decision references |
| `class_map_version` | int, from `class_map_service` | Same reasoning |

**Params to log on the run** (via `mlflow_tracking.log_params`, already wired — `base_weights` and `task` land there today; the rest is still to add): `epochs`, `patience`, `batch`, `imgsz`, `base_weights` (the actual starting checkpoint path — currently always `yolo11s-seg.pt`, critical once fine-tuning from a previous production version instead, per §K item 7), `dataset_snapshot_id`, `class_map_version`, `class_count`, `new_classes`, `removed_classes`, `parent_model_version` (the version whose weights seeded this fine-tune).

**Metrics beyond current per-class golden metrics:** `false_positive_rate` and `false_negative_rate` per class (not just precision/recall, which the user is right to point out don't map 1:1 onto operational FP/FN cost — precision/recall are computed at a specific confidence threshold sweep; FP/FN counts at the *actual serving threshold* are the number that matters operationally); `inference_latency_ms_p50` / `p95` (measured on the same probe/sanity-check image `model_promotion_service._sanity_check` already runs before activation — reuse that exact call site to also time it); `model_size_mb`.

**Lineage:** MLflow doesn't have a first-class "parent model version" graph edge, so represent it as **tags + params on both ends**: the candidate's run gets `parent_model_version` (param) and `baseline_model_version` (tag) pointing at the production version it forked from and was compared against, respectively — usually the same version, but not always (a candidate could be re-evaluated against a *newer* production version than the one it was originally trained against, if something else got promoted first). Given both, the full chain V1→V2→V3 is reconstructable by walking `baseline_model_version` backward from any version — exactly the lineage picture in §15 of the ask, without needing anything beyond tags MLflow already supports.

---

## H. `compare(candidate, production)` — pseudocode

```python
@dataclass
class ClassMetrics:
    class_id: int
    precision: float
    recall: float
    ap50: float
    ap50_95: float
    false_positive_rate: float
    false_negative_rate: float
    n_val_instances: int          # how many golden-set instances back this number

@dataclass
class ComparisonResult:
    common_classes: dict[int, dict]      # class_id -> {candidate, production, delta, regressed}
    new_classes: dict[int, dict]         # class_id -> {candidate, meets_floor}
    removed_classes: list[int]           # present in production, absent from candidate
    reintroduced_classes: list[int]      # present in candidate, previously DEPRECATED
    overall: dict                        # macro/weighted mAP over the union of classes — reported only
    candidate_class_set: set[int]
    production_class_set: set[int]


def compare(candidate: ModelVersionMetrics, production: Optional[ModelVersionMetrics],
            thresholds: PromotionThresholds) -> ComparisonResult:

    candidate_classes = set(candidate.per_class.keys())
    production_classes = set(production.per_class.keys()) if production else set()

    common = candidate_classes & production_classes
    new = candidate_classes - production_classes
    removed = production_classes - candidate_classes   # present before, absent now

    # A class only counts as "removed" if it was explicitly DEPRECATED.
    # If it's missing from the candidate's class set WITHOUT a deprecation
    # record, that's a data/config bug, not a design decision -> hard fail,
    # never silently treated as "removed" in the lineage sense.
    silently_dropped = [
        c for c in removed
        if class_registry.state(c) != ClassState.DEPRECATED
    ]
    if silently_dropped:
        raise UnexplainedClassRemoval(silently_dropped)

    reintroduced = [
        c for c in new
        if class_registry.was_ever_active(c)   # DEPRECATED -> REINTRODUCED path
    ]
    # Reintroduced classes are evaluated as NEW classes (Layer 2), never as
    # COMMON -- their old production metrics are stale and are never read.

    common_result = {}
    for class_id in common:
        cand = candidate.per_class[class_id]
        prod = production.per_class[class_id]
        delta = cand.ap50 - prod.ap50
        tier = class_registry.tier(class_id)   # cosmetic | structural | safety
        tolerance = thresholds.regression_tolerance[tier]   # e.g. safety=0.5pt, cosmetic=3.0pt
        common_result[class_id] = {
            "candidate": cand, "production": prod, "delta_ap50": delta,
            "regressed": delta < -tolerance,
            "tier": tier,
        }

    new_result = {}
    for class_id in new:
        cand = candidate.per_class[class_id]
        tier = class_registry.tier(class_id)
        floor = thresholds.new_class_floor[tier]   # absolute AP floor, not relative
        new_result[class_id] = {
            "candidate": cand,
            "meets_floor": cand.ap50 >= floor and cand.n_val_instances >= thresholds.min_val_instances,
            "floor_used": floor,
        }

    overall = {
        "candidate_map50": weighted_mean(candidate.per_class, by="n_val_instances"),
        "production_map50": weighted_mean(production.per_class, by="n_val_instances") if production else None,
        # explicitly NOT used by should_promote() as a standalone criterion -- see Layer 3 note below
    }

    return ComparisonResult(common, new_result, list(removed), reintroduced, overall,
                             candidate_classes, production_classes)
```

**Design notes baked into the pseudocode above, stated explicitly because they answer the user's harder questions directly:**

- **Regression tolerance is per-tier, not one global number.** A safety-critical class gets almost zero slack; a cosmetic class gets real slack. This directly answers "maximum allowed regression per existing class" (§8 of the ask) — it isn't one number, it's a lookup on the tier this codebase already has (`dataset_classes.safety_critical`, extended to a 3-way tier to match `pipeline.md`'s existing vocabulary rather than inventing a new one).
- **New-class floor is absolute, per-tier, never relative to production** — because production has no score for that class, a relative comparison is mathematically meaningless (this is the user's own correct observation in §7, restated as code).
- **Silent class removal is a hard error, never a soft "removed" classification.** This is the fix for the "removed classes handled correctly" requirement in §H of the ask — the only legitimate way a class disappears from a candidate's class set is a prior, explicit DEPRECATED action; anything else is treated as a bug in dataset assembly, and `compare()` refuses to produce a result rather than guessing.
- **Reintroduced classes are always routed through Layer 2 (new-class), never Layer 1.** This is the direct answer to "what happens if deprecated and later reintroduced" (§13 of the ask) — their historical production score is explicitly never read for comparison purposes, because it was earned under different data and possibly a different class definition.

---

## I. `should_promote(candidate, production)` — pseudocode

```python
@dataclass
class PromotionDecision:
    decision: str                 # "PROMOTE" | "REJECT"
    reasons: list[str]             # every failing check, not just the first
    comparison: ComparisonResult
    operational: dict


def should_promote(comparison: ComparisonResult,
                    operational: OperationalMetrics,
                    thresholds: PromotionThresholds) -> PromotionDecision:

    reasons = []

    # --- Layer 1: common-class regression -----------------------------
    regressed = [c for c, r in comparison.common_classes.items() if r["regressed"]]
    if regressed:
        worst = min(comparison.common_classes[c]["delta_ap50"] for c in regressed)
        reasons.append(f"regression on classes {regressed} (worst {worst:+.1f}pt AP50)")

    # A single safety-tier regression is an automatic reject regardless of
    # every other layer -- this is what makes "critical-class requirements"
    # (ask §8) actually binding rather than advisory.
    safety_regressed = [c for c in regressed
                         if comparison.common_classes[c]["tier"] == "safety"]
    if safety_regressed:
        reasons.append(f"HARD FAIL: safety-tier regression on {safety_regressed}")

    # --- Layer 2: new classes must clear an absolute floor -------------
    failing_new = [c for c, r in comparison.new_classes.items() if not r["meets_floor"]]
    if failing_new:
        reasons.append(f"new class(es) {failing_new} below absolute quality floor")

    # --- Layer 3: overall quality -- REPORTED, NEVER a standalone gate -
    # A higher overall mAP never overrides a Layer-1/2/4 failure, and a
    # LOWER overall mAP never blocks promotion by itself if Layers 1/2/4
    # all pass -- this is the direct fix for ask example #10 (V2's global
    # mAP is lower purely because E is hard, but V2 is still the better
    # model). Overall is diagnostic context attached to the decision, full
    # stop.

    # --- Layer 4: operational constraints -------------------------------
    if operational.latency_p95_ms > thresholds.max_latency_p95_ms:
        reasons.append(f"latency regression: {operational.latency_p95_ms}ms > "
                        f"{thresholds.max_latency_p95_ms}ms budget")
    if operational.model_size_mb > thresholds.max_model_size_mb:
        reasons.append(f"model size {operational.model_size_mb}MB exceeds budget")
    for class_id, fp_rate in operational.false_positive_rate.items():
        if fp_rate > thresholds.max_false_positive_rate[class_registry.tier(class_id)]:
            reasons.append(f"class {class_id} false-positive rate {fp_rate:.3f} "
                            f"exceeds tier budget")

    # --- Class-count sanity: never promote a candidate with a SMALLER --
    # class set than production unless every removal is an explicit,
    # already-recorded deprecation (compare() already enforces this by
    # raising rather than returning; this is a defense-in-depth re-check).
    unexplained_shrink = comparison.production_class_set - comparison.candidate_class_set
    if unexplained_shrink:
        reasons.append(f"HARD FAIL: class set shrank without recorded deprecation: "
                        f"{unexplained_shrink}")

    decision = "REJECT" if reasons else "PROMOTE"
    return PromotionDecision(decision=decision, reasons=reasons,
                              comparison=comparison, operational=operational)
```

This is what plugs into `model_registry_service.register_and_recommend` in place of `_compare_against_production` — same call site, same "advisory, never raises, logged not thrown" wrapping behavior the module already has, richer output. **`PROMOTE`/`REJECT` from this function is still not a live deploy** — it only changes what tag gets written on the MLflow model version. Both existing human gates (curator moves the stage; a *different* model_reviewer approves the swap) stay exactly as-is. This function makes their job "read one clear reason list" instead of "reverse-engineer a `regressed_classes` tag by hand."

---

## J. Example walkthrough — V1 (A,B,C,D) → E discovered → V2 (A,B,C,D,E)

| Stage | What happens |
|---|---|
| **Dataset** | Annotators keep labeling `side_view`. A new object type gets labeled for the first time → `dataset_classes` gains a row for E (`add_class`), state=DISCOVERED. `class_map_service.ensure_version` mints class-map version N+1 (content hash changed) — but E is not yet in any trained model. |
| **Ongoing** | E's counts grow. A scheduled/on-save check (§K) recomputes E's eligibility metrics live: instances, images, coach-type diversity, val-split projection. State flips DISCOVERED→COLLECTING_DATA→**ELIGIBLE** once the composite gate in §E clears (not merely 8% share). |
| **Trigger** | Next `POST /api/export` produces a genuinely-new content-hashed snapshot (E's new labels change the hash) → `auto_train_on_handoff` fires `detector_service.start_training()`. `_assemble_dataset` now includes E because it's ELIGIBLE, and includes A–D because they're ACTIVE (sticky, §D). |
| **Training** | GPU guard waits for SAM2 idle. Fine-tune starts from **V1's production weights**, not `yolo11s-seg.pt` cold — logged as `base_weights=runs:/<v1_run_id>/weights`, `parent_model_version=1`. Class-weighted sampling prevents E (now present but still numerically minor) from being starved in early batches. (Today, absent §K item 7, this step actually starts from the stock `yolo11s-seg.pt` checkpoint every time — the warm-start-from-production behavior described here is still proposed, not shipped.) |
| **MLflow run** | `mlflow.start_run()` in `annot-detector-training`. Params: `class_count=5`, `new_classes=[E]`, `dataset_snapshot_id=<hash>`, `class_map_version=N+1`. Per-epoch metrics as today. |
| **Golden eval** | `golden_eval_service` scores against the frozen golden set. **Assumes the golden set itself has been extended with a few E examples by the curator** — if it hasn't, Layer 2 has literally nothing to score E against, and the gate must REJECT with reason "no golden coverage for new class E" rather than silently passing it. (This is a real operational prerequisite this document is flagging, not glossing over: extending the golden set for a new class is a manual curator step that has to happen before E can ever be promoted, mirroring §5.9's existing curator-only gate.) |
| **compare()** | Common = {A,B,C,D}: per the user's Example 9 numbers (A+1, B+1, C−2, D−6), D's −6pt trips the regression check (D is likely `structural` or `safety` tier given it's a real component — tolerance is tight). New = {E}: 76% AP50 checked against E's tier floor, say 70% for a newly-introduced structural class → passes. Removed = {} (nothing deprecated). Reintroduced = {}. |
| **should_promote()** | Layer 1 fails on D. Regardless of E passing Layer 2 and overall mAP looking fine, `decision=REJECT`, `reasons=["regression on classes [D] (worst -6.0pt AP50)"]`. This is the exact outcome the user demands in §9 — the system does not say "V2 has more classes, therefore better." |
| **Registry** | `model_registry_service` still creates `AnnotDetector-side_view` v6 (registration is unconditional — a rejected candidate is still recorded, for lineage and so the next retrain doesn't repeat the exact same mistake invisibly), tags `decision=REJECT`, `common_class_regression=D:-6.0pts`, `new_class_floor_result=E:pass(76.0>=70.0)`. |
| **Human #1** | Curator opens MLflow, sees the tag, does **not** move the stage. Production stays V1 (A,B,C,D). Root-causes the D regression (likely a labeling or sampling issue introduced alongside E) and re-trains. |
| **Second attempt (hypothetical, matching the ask's Example 10 instead)** | Suppose the retrain fixes whatever caused D's collapse: A+2, B+3, C+1, D+2, E=70%. Layer 1: all deltas positive → passes. Layer 2: E at 70% clears its floor → passes. Layer 3 (overall): possibly *lower* than V1's raw mAP because E is hard and now drags the mean — reported, not gating. Layer 4: latency/size/FP within budget → passes. `decision=PROMOTE`. |
| **Registry/Production** | Curator moves this new version's stage to Production, sets `@champion` alias. `baseline_model_version` for the *next* candidate is now this version, not V1 — the lineage chain advances. |
| **M7.5** | Poller detects the Production change, inserts a `pending` row. A different-identity `model_reviewer` reviews the tagged decision, approves. `approve()` downloads, sanity-checks on a probe image, atomically swaps `registry.json`. |
| **Detection machine** | `detector_service.detect()` immediately starts serving A,B,C,D,E — with zero MLflow round-trip at request time, exactly as today. Rollback, if ever needed, is the existing instant local-file swap back to V1's weights, no MLflow dependency. |

---

## Required manual step: golden-set coverage for a newly-eligible class

**This is process, not code (Module 10 of the implementation plan) — read this before a class ever reaches ELIGIBLE for the first time.**

The moment a class crosses the composite gate in §E and becomes trainable, it can appear in a candidate model — but that candidate can **never be promoted** until the golden set itself has at least one representative image of that class. This isn't a bug to route around; it's the deliberate, correct consequence of §7's own requirement ("an absolute threshold, not a fabricated comparison"): a class with zero golden coverage has nothing for `evaluate_on_golden_set` to score it against, and `InsufficientGoldenCoverageError` refuses to let that produce a meaningless zero-instance number that could accidentally look like a pass.

**What this means in practice:** when a `golden_curator` sees a class newly reach ELIGIBLE (or REINTRODUCED), they need to add a handful of representative images containing that class to the golden set — the same curator-only path that already exists (`GoldenSet`/`GoldenSetItem`, `db_models.py`) — *before* the next training cycle's candidate can ever be promoted for it. Until they do, every candidate touching that class will train fine, register fine, and then hit the coverage-gap check and simply not produce a registered version at all (the training job itself still reports "completed" — this is a registration skip, not a training failure; see `detector_service._run_training`'s handling of `InsufficientGoldenCoverageError`).

**Why this is worth writing down explicitly:** a curator who doesn't know this will be confused the first time a new class's training run produces no MLflow model version to review, and will (reasonably) go looking for a bug. There isn't one — the system is doing exactly what §7 asked for. The fix is always the same: add golden coverage for that class, then retrain.

---

## K. Implementation plan — exact files

No code should change yet; this is the concrete map for when it does, in dependency order.

1. **`backend/app/models/db_models.py`** — add a `state` column (enum: discovered/collecting_data/eligible/active/deprecated) to `DatasetClass` (currently `L235-273`), plus a `tier` column (cosmetic/structural/safety) generalizing the existing boolean `safety_critical` into the three-way tier `pipeline.md` already uses. Add a migration (new Alembic revision after `dbc0c570a39a`). Backfill: existing `safety_critical=True` → `tier=safety`; everything else → `tier=structural` as a conservative default, reviewable by a human afterward, never `cosmetic` by default.
2. **New `backend/app/services/class_eligibility_service.py`** — implements the composite gate from §E (instance count, image count, coach-type diversity, val-split projection) as a pure function over `annotation_state` + `dataset_classes`, called on a schedule (reuse the pattern already established by `model_promotion_service`'s poller in `main.py`) or on every `POST /api/dataset/switch` / save. Writes the computed `state` back onto `DatasetClass`. This is the piece that makes ELIGIBLE a live, recomputed fact rather than a one-time flag.
3. **`backend/app/services/detector_service._assemble_dataset`** (`L488-545`) — filter the class list to `state in {eligible, active}` instead of the full current class map. This is the one-line-in-spirit, but load-bearing, change that makes eligibility actually gate training.
4. **New `backend/app/services/promotion_gate.py`** — implements `compare()` (§H) and `should_promote()` (§I) as pure functions, unit-testable with fixture metrics dicts and no MLflow/DB dependency. Add `PromotionThresholds` as a small dataclass with per-tier tolerance/floor tables, loaded from config (new `Settings` fields, alongside the existing hardcoded-constant pattern noted in the prior gap analysis — but these specifically should be *config*, not module constants, since they're the numbers most likely to need calibration per §E's learning-curve process).
5. **`backend/app/services/model_registry_service.py`** — replace `_compare_against_production` (`L53-89`) with a call into `promotion_gate.compare()` + `should_promote()`; extend `PromotionRecommendation` (or replace with `PromotionDecision` from step 4) with the richer field set; extend `register_and_recommend`'s tag-writing (`L118-126`) with the full tag table from §G. Also set the `@champion`/`@candidate` aliases here (`client.set_registered_model_alias`) alongside the existing stage-based logic, additively.
6. **`backend/app/services/detector_service._run_training`** — extend the params dict passed to `mlflow_tracking.log_params` with `base_weights` (already logged today; the fine-tune *source* becomes meaningful once step 7 lands) plus `dataset_snapshot_id`, `class_map_version`, `new_classes`, `removed_classes`, `parent_model_version`. Extend the golden-eval metrics logging to also compute and log `false_positive_rate`/`false_negative_rate` per class (extend `golden_eval_service.evaluate_on_golden_set`, which as of the YOLO11-seg migration already reports per-class precision/recall/AP for **both** mask (primary) and box (continuity) bases — FP/FN counts are a natural further addition to that same per-class dict, still not done).
7. **`backend/app/services/detector_service`** — change training to warm-start from the current Production version's weights (downloaded via the same `mlflow.artifacts.download_artifacts` pattern `model_promotion_service.approve` already uses, `model_promotion_service.py:175-205`) instead of always the stock `yolo11s-seg.pt` checkpoint, whenever a Production version already exists for the view. Add class-weighted sampling to the `model.train()` call — Ultralytics supports this natively via dataset config, no custom trainer needed. (Line numbers throughout this plan were accurate against the pre-YOLO11-seg version of this file; the functions named are unchanged, but exact line numbers have shifted with that migration — re-check before citing them literally.)
8. **`backend/app/services/golden_eval_service.py`** — no structural change needed, but flag (raise a specific, catchable `InsufficientGoldenCoverageError`) when a class present in the candidate's class set has zero golden-set representation, so `promotion_gate` can turn that into an explicit REJECT reason (§J's "no golden coverage for new class E" case) instead of a silent zero-instance metric.
9. **`backend/app/services/model_promotion_service.py`** — no logic change required; it already reads whatever the registry tags say and gates on a separate human identity. Optionally surface the richer `reasons` list in the `ModelPromotion` row / `routers/model_promotions.py` response so the `model_reviewer`'s UI shows the actual failing checks, not just a verdict string.
10. **Golden set process (people, not code):** document that extending the golden set for a newly-eligible class is a required manual curator step *before* that class can ever pass Layer 2 — this is a process gap to close operationally, matching the existing curator-only write path (`GoldenSet`/`GoldenSetItem`, `db_models.py:276-337`), not a new code path.

**Suggested sequencing:** 1→2→3 (eligibility, independently valuable and low-risk on its own) can ship and be observed for a cycle before 4→5→6 (the gate logic) lands, since the gate is what actually changes promotion outcomes and deserves to be reviewed against real historical run data first — register-and-tag-only (no behavior change to what's already unconditional-register) means steps 4-6 are safe to deploy dark (compute and tag the decision, but keep both human gates exactly as they are) before anyone trusts `decision=REJECT` enough to treat it as more than a strong hint.

---

## L. Glossary

Every term used above, in one place — most are defined the first time they appear too, but this is for jumping straight to a word that didn't parse.

| Term | Plain-language meaning |
|---|---|
| **mAP / mAP50 / mAP50-95** | "Mean Average Precision" — the standard single-number accuracy score for an object detector, on a 0–100% scale (higher = better). `mAP50` grades a detection as "correct" if its outline overlaps the true object by at least 50%; `mAP50-95` is a stricter average across several overlap thresholds (50% through 95%). This document mostly talks about the **per-class** version of this number — one score per component/defect type, not one score for the whole model — because that's the only version that can catch "class D got worse while everything else looks fine." |
| **Precision / Recall** | Precision = "of everything the model flagged, how much was actually real" (low precision = too many false alarms). Recall = "of everything that was actually there, how much did the model catch" (low recall = it's missing real cases). A model can have a great mAP and still be operationally wrong for a safety use case if its recall on one critical class is bad — this is part of why a single accuracy number isn't the whole story. |
| **False positive / False negative** | False positive = the model flagged something that wasn't actually there (a false alarm). False negative = the model missed something that *was* actually there (a miss). For a defect-detection system, false negatives are usually the more dangerous failure mode — a missed crack is worse than an extra alert a human dismisses in two seconds. |
| **Golden set** | A small, fixed, hand-picked set of images that is used **only** to grade a finished model, and that a model is **never** trained on. Think of it as an exam the model never saw the answer key to. It exists specifically so a model's reported score can't be inflated by accidentally training on the same data it's later graded against. |
| **Class-map version** | A permanent, numbered snapshot of "what the class list looked like at this exact moment" — so that if the class list changes later (a class renamed, a class added), you can always answer "what did class ID 7 mean in this specific export/model," even years later, instead of that meaning silently drifting. |
| **Dataset snapshot** | An immutable, fingerprinted export of the training data at one point in time — re-exporting identical data gives back the same snapshot; changing even one label produces a genuinely new one. This is what lets a trained model's MLflow tags honestly say "this exact data produced this exact model," rather than pointing at a folder that might have silently changed since. |
| **MLflow** | The tool this whole document is about integrating with — think of it as an automatic lab notebook for machine learning. Every time a model is trained, MLflow can record what data/settings went in, how it scored, and which version is considered "the one currently in production." It does **not**, on its own, know that a 5-class model is "better" than a 4-class one — that judgment is exactly the gap this document fills in. |
| **Model Registry (MLflow)** | The part of MLflow that keeps a numbered history of every trained version of a given model (v1, v2, v3, …) and lets you mark one of them as the currently "live"/production one. |
| **Stage vs. Alias (MLflow)** | Two different ways MLflow lets you mark "this version is special." **Stage** is the older mechanism — a version is `None` / `Staging` / `Production` / `Archived`, and only one version can hold a given stage at a time. **Alias** is the newer, more flexible mechanism — you can attach a label like `@champion` to any version, and multiple aliases can point at different versions simultaneously. This document recommends using both for a while (stage for backward compatibility with what's already built, alias as the more future-proof one to lean on). |
| **Registered model** | The permanent "name" a family of model versions lives under in the registry — e.g. `AnnotDetector-side_view`. Every retrain produces a new *version* of the same registered model; the class list changing does **not** mean a new registered model gets created — same identity, evolving capability, which is exactly the "lineage" this document cares about. |
| **Run (MLflow)** | One single recorded training attempt — its settings (params), its results over time (metrics, e.g. accuracy per epoch), and its output files (artifacts, e.g. the trained weights). |
| **Tag vs. Param vs. Metric (MLflow)** | Three different kinds of information MLflow lets you attach to a run or model version. **Param** = a setting you chose before training started (e.g. how many epochs). **Metric** = a number that came out of training/evaluation (e.g. accuracy). **Tag** = free-form labeling metadata, often computed *after* the fact by something like this document's promotion gate (e.g. `decision=REJECT`) — tags are what a human or dashboard reads to understand a version without re-computing anything. |
| **Lineage** | The traceable chain of "which model came from which" — V2 was a retrain of V1, V3 was a retrain of V2, and so on. This document represents lineage using ordinary MLflow tags (`parent_model_version`, `baseline_model_version`) rather than anything exotic, since MLflow has no built-in "parent version" concept of its own. |
| **Production / Candidate / Baseline** | **Production** = whatever model version is currently live, serving real users. **Candidate** = a newly trained version being considered as a possible replacement. **Baseline** = the specific production version a given candidate is being measured against (usually the current production version, but see §G for why it's not *always* the same one). |
| **Promotion** | The act of officially marking a candidate as the new production version. In this design, promotion is always a deliberate human action (never automatic), and it happens in two separate steps by two separate people — see the next two rows. |
| **golden_curator** | The role/person who looks at a candidate's scores (including the new promotion-gate verdict this document proposes) and makes the *accuracy* judgment call: "yes, this model is genuinely good." |
| **model_reviewer** | A **different** person from the golden_curator, who makes a *separate*, deliberately distinct judgment call: "yes, it's safe to actually swap this into what's live right now." Requiring two different people for these two different questions is a real, already-built safety mechanism in this codebase (not something this document is proposing) — it means one person's optimism about accuracy can't also be their own sign-off on deploy safety. |
| **Rollback** | Undoing a promotion — instantly reverting to the previous model version. Already built to be dependency-free (it doesn't need MLflow or the network to be reachable to work), which matters if the reason you're rolling back is that something else is broken. |
| **Class lifecycle / state (DISCOVERED, COLLECTING_DATA, ELIGIBLE, ACTIVE, DEPRECATED, REINTRODUCED)** | The life story of one class, from "an annotator labeled it for the first time" through "it has enough data to train on" through "it's actually shipped in production," and — if it's ever retired — back out again. Fully defined with entry conditions in §D. |
| **Tier (cosmetic / structural / safety)** | A classification of *how much it matters* if a given component/defect class is wrong. Safety-tier classes get the strictest regression tolerance and the highest new-class quality bar; cosmetic-tier classes get the most slack. This isn't a new idea invented for this document — it reuses a tiering concept `pipeline.md` already defines for a related purpose (deciding what to skip first if the system is overloaded). |
| **Regression (in this context)** | A class the model already knew getting measurably *worse* in the new candidate version compared to the current production version. The word specifically does **not** apply to a brand-new class (there's no "before" for it to regress from) — see §H for why new classes are deliberately routed through a different check entirely. |
| **Catastrophic forgetting** | The general risk, in any model that learns multiple things with one shared "brain" (one shared network), that teaching it something new can quietly damage what it already knew — because the same internal weights are being adjusted for everything at once. §F explains why this is specifically a training-time risk here (not just something to check for afterward) and how it's mitigated. |
| **Warm start / fine-tune** | Starting a new training run from an *already-trained* model's weights (e.g. the current production version) rather than from a generic pretrained starting point. This tends to make catastrophic forgetting less severe, since the model isn't relearning everything from scratch — it's adjusting from a state that already knows the old classes well. |
| **`MaskSource`** | A field added to this codebase (separately from the class-promotion work this document is about) that records *which model actually produced a given predicted outline* — the SAM2 assistant, or the YOLO detector's own prediction. It matters here only as a pattern to imitate: a number produced by one model must never be silently treated as equivalent to the same-looking number from a different model that a threshold was actually calibrated against. The new-class absolute-floor check in §E/§H needs the same discipline. |
| **YOLO / YOLO11-seg** | The specific object-detection model architecture this repo trains as its pre-labeling assistant. "YOLO11-seg" specifically means the *segmentation* variant, which predicts a pixel-accurate outline (a polygon) rather than just a bounding box — the detail of *which* architecture is trained is not the subject of this document, but it's mentioned in the status update up top because it changed what the per-class accuracy numbers actually measure (outline quality, not just box quality). |
| **Detection machine / pre-labeler / annotation-assist model** | This repo's own trainable model, whose whole job is to give a human annotator a head-start outline to correct, rather than making a final, unreviewed decision about anything. This is a different, smaller-stakes system than the separate "production defect-detection pipeline" this document's Scope Note distinguishes it from — everything here applies to whichever of the two is plugged into this MLflow instance, but concretely, today, that's this pre-labeler. |
