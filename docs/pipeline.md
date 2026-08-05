# AI Detection Pipeline — Developer Specification

**Scope:** the AI detection pipeline that consumes the **10 area-camera** preprocessed feed and produces a per-coach inspection report. This document is the developer reference — every model, term, threshold, constraint, and control rule is stated.

**Out of scope (excluded here):** the underbelly line-scan feed, the OCR (coach-number) model, and the gap-detection model. OCR and gap detection are already deployed and working in the final system; this pipeline does not re-implement them. Coach/axle identity here is derived from the **axle-count** obtained from the wheel cameras + the fixed rake formation (see §4), so the pipeline is fully self-contained on the 10-camera feed.

---

## 1. Locked constraints

| # | Constraint | Value |
|---|---|---|
| C1 | Inference device | NVIDIA DGX Spark (GB10), 128 GB unified memory, on-site |
| C2 | Model residency | All models resident in memory simultaneously — no load/unload during a run |
| C3 | Input format | Lossless PNG, native resolution (from the preprocessing unit) |
| C4 | Overall SLA | Input → detection conclusions ≤ 5 min (preprocessing consumes ≤ 3 min; detection ≤ ~2 min; measured detection ~30–70 s) |
| C5 | Scale | Up to 24 coaches per train |
| C6 | Safety rule | Safety-critical checks are never skipped; un-inspected regions are reported `data_unavailable`, never "clean" |
| C7 | Precision | Detectors/classifiers FP8 (INT8 fallback); segmentation FP16; anomaly FP16 |

**Term — resident:** every model's weights/engine stay loaded in the 128 GB unified memory for the whole session, so there is no per-model swap-in latency.

---

## 2. Input contract (from the preprocessing unit)

The pipeline consumes the preprocessing handoff (`preprocessed_frame.schema.json`): one lossless PNG per kept frame + a metadata record. Relevant fields:

| Field | Use |
|---|---|
| `camera`, `zone` (P1/P2/P3), `colour` | routes the frame to the correct zone models |
| `seq`, `timestamp_ms` | ordering / provenance |
| `encoder_mm` | longitudinal position — the coordinate reference for voting (never wall-clock) |
| `width`, `height` | native resolution |
| `png_ref` | the lossless PNG payload |

**Zones:** P1 = Side (4 cams, RGB), P2 = Underbelly (2 area cams, RGB), P3 = Wheel/bogie (4 cams, greyscale).

---

## 3. Data flow

```
preprocessed PNG frames (10 cams)
        │
        ▼
  Coordinate spine  ── axle-count (from P3 wheel passes) + fixed formation
        │  stamps each detection: (coach_index, axle_id, side, view, longitudinal_position_mm)
        ▼
  Shared detector (backbone + P1/P2/P3 heads + metrology keypoints)
        │
        ├─► Gate cascade ──► gated specialists:
        │        P2: anomaly gate → crack/corrosion seg (only inside ROI)
        │        P3: wheel-seg (log-polar) + fastener slot-occupancy
        │        P1: defect-state classifier + metrology (buffer/coupler mm)
        │
        ▼
  Temporal voting + entry/exit fusion (k-of-n)
        ▼
  Completeness — manifest diff (coach-type-conditional)
        ▼
  Report assembly + coverage  ─►  InspectionReport (per coach)

  Degrade + bounded FIFO  ── wraps the whole path (load shedding, safety never dropped)
```

---

## 4. Coordinate spine

**Purpose:** give every detection a deterministic coordinate stamp so findings land on the correct coach/axle and votes align.

**Logic:**
- **Ground truth = axle-pass count.** The P3 wheel head detects each wheel/axle pass; the count is the truth. VB rakes have a **fixed formation** (known coach order and axles-per-coach), so the axle sequence maps deterministically to `coach_index`.
- Each detection inherits `(coach_index, coach_type, axle_id, side, view, longitudinal_position_mm)`.
- `longitudinal_position_mm` comes from the frame's `encoder_mm` (encoder-derived), **never wall-clock**.
- `view` ∈ {entry, exit} distinguishes the two passes of the same physical wheel.

**Invariants (hard):**
- `axle_count` must equal the formation's expected axle count → mismatch is a **hard fail** (never silently renumber).
- Encoder positions must be monotonic non-decreasing → non-monotonic input rejected.

**Note:** the coach-number *label* (OCR) is out of scope here; identity does not depend on it. Gap/coach boundaries are derived from the axle sequence + formation, not from the gap-detection model.

---

## 5. Models (7)

All model families; OCR and gap-detection excluded. Metrology is folded into the shared detector as an auxiliary keypoint output (not a separate model).

### 5.1 Shared detector (+ metrology keypoints)
- **Arch:** YOLO11m backbone + three lightweight detection **heads** (P1/P2/P3) sharing the backbone, + a **pose (keypoint)** output for buffer-face/rail and coupler/reference points.
- **Input:** preprocessed frames, per zone (P1 panorama tiles at 1280 px, P2/P3 whole-frame downscaled for the gate).
- **Output:** component bounding boxes + class + confidence; wheel/axle detections (feed the spine); keypoints (buffer, coupler).
- **Precision:** FP8 (INT8 fallback). **Metric:** mAP50. **Threshold:** `confidence.component = 0.35`.
- **Metrology:** keypoints → **pixels-to-millimetre calibration** (per-site rail scale, arithmetic in the serve wrapper, not a model) → buffer height (spec 1030–1105 mm) and coupler sag.
- **Term — head:** a small task-specific network attached to the shared backbone; one retrain, one engine.

### 5.2 Defect-state classifier
- **Arch:** EfficientNet-B0 / YOLO11-cls (crop classifier). Includes FIBA red-state and brake-sparking as classes (RGB crops).
- **Input:** a component crop (from a detector box) + margin.
- **Output:** condition ∈ {ok, broken, missing, hanging, displaced, dislocated, leaking, damaged, securing_broken, securing_hanging, fiba_red, sparking, binding}.
- **Precision:** FP8. **Metric:** accuracy + macro-F1. Synthetic `leakage` class excluded from serving.

### 5.3 Anomaly gate (PatchCore) — the gate
- **Arch:** Anomalib PatchCore (WideResNet50 features), **coreset ratio 0.01** (1 % memory bank — bounds the nearest-neighbour latency).
- **Input:** the downsampled belly strip (P2).
- **Output:** per-region anomaly score → a **ROI mask** (which belly regions need detailed crack analysis).
- **Training:** memory-bank fit on **confirmed-normal only** (no backprop); one defect in the normal set poisons it (purity gate).
- **Precision:** FP16. **Metric:** AUROC. **Threshold:** `p2_anomaly_percentile = 99.0` (conservative → higher gate recall).
- **Term — gate:** a cheap full-field pass that decides where the expensive stage runs, so the expensive stage does not run everywhere.

### 5.4 Crack / corrosion segmentation
- **Arch:** YOLO11-seg (or U-Net).
- **Input:** **gated SAHI** tiles — 320 px, 0.20 overlap — cut **only inside** the anomaly ROI mask.
- **Output:** crack / corrosion pixel masks + crack length.
- **Precision:** FP16. **Metric:** Dice/IoU **+ length-recall** (not box mAP). **Threshold:** `crack_dice_min = 0.30`.
- **Term — SAHI:** slicing a large image into overlapping tiles so small/thin defects are detectable; here it runs **gated** (only in the ROI), not full-frame.

### 5.5 Wheel-shelling segmentation
- **Arch:** YOLO11-seg on a **log-polar unwrapped** wheel crop (circle seeded from fixed geometry + known diameter — not blind Hough).
- **Output:** shelling %, shelling length, flat count, tread-crack mask.
- **Precision:** FP16. **Metric:** Dice/IoU. **Threshold:** `shelling_dice_min = 0.30`.
- **Scope:** length only — shelling **depth** is a physical/separate-system measurement, never output here.
- **Term — log-polar unwrap:** re-mapping the round wheel into a flat strip so ring defects become a standard segmentation problem.

### 5.6 Fastener slot-occupancy
- **Arch:** per-known-slot classifier + a verifier CNN (not full-frame SAHI).
- **Input:** a crop at each known bracket-slot position.
- **Output:** occupied? (fastener_present / fastener_missing) + confidence.
- **Precision:** FP8. **Metric:** recall@FP. **Threshold:** `fastener_recall_conf = 0.20` (low threshold + verifier → high recall without false-positive flood).

### 5.7 Coach-type classifier
- **Arch:** small CNN on the coach panorama.
- **Output:** {LHB, ICF} → selects the correct component manifest.
- **Precision:** FP8. Wrong coach-type → false-missing storm, so the manifest is versioned per type and validated.

---

## 6. Gate cascade

**Problem:** full-frame SAHI over belly + bogie ≈ hundreds of thousands of tiles/train → infeasible.
**Solution:** gate, then tile only the ROI.

**Rules:**
- **P2:** `anomaly ROI mask → SAHI tiles (320 px, 0.20 overlap) only inside the mask → crack/corrosion seg`. Cold strip (no anomaly) → heavy stage skipped.
- **P3:** `detect wheels → per-wheel log-polar crop → wheel-seg`; `expected bracket slots → fastener check`. **No full-frame SAHI in P3.**
- **P1:** detect direct (parts ≥ 15 mm) → defect-state classifier + metrology. No SAHI.

The cascade emits a **work plan** (the exact tiles/crops to run) and reports gated-vs-ungated tile counts. Empty ROI = zero tiles.

---

## 7. Temporal voting + entry/exit fusion

**Purpose:** a real defect appears across several frames; a false positive usually does not. Nothing is flagged on a single detection.

**Logic:**
- Quantize each detection into a **vote cell** = `(coach_index, longitudinal band, side)`, band = `longitudinal_position_mm // cell_size_mm` (`cell_size_mm = 100`).
- Accumulate per (cell, class, view). A class in a cell is flagged only when **≥ k confirmations** carry confidence ≥ the class threshold. **`k_of_n = 3`.**
- **Entry/exit fusion (wheels):** the entry and exit views of the same physical wheel are keyed on `(coach_index, side, class)` — **axle/side, never appearance**. Either view flagging → recall; both agreeing → precision (max confidence).
- **Idempotency:** flag key `(coach_index, band, side, class)` dedupes re-ingest so the console never double-lists.

**Term — k-of-n:** require k confirming looks out of n frames before flagging.

---

## 8. Completeness (manifest diff)

**Purpose:** is every expected component present per coach.

**Logic:**
- Manifest is **coach-type-conditional** (LHB vs ICF expect different components).
- Runs on the **voted present-set** (not raw detections) → transient misses/false-positives don't fake a result.
- A component is `missing` only when it had **0 hits across ≥ `missing_min_views` opportunities** (`missing_min_views = 2`). Fewer views → `data_unavailable` (not enough coverage to judge), never `missing`, never `present`.
- Components with no camera coverage → `data_unavailable` (never "clean").
- Unknown coach-type → refuse to guess (raises), never assume a manifest.

---

## 9. Degrade + bounded FIFO

**Purpose:** the single device is serial; bursty back-to-back trains can push past the SLA.

**Logic:**
- **Bounded per-train FIFO:** a full queue rejects a new train → the edge holds it on spill (nothing lost).
- **Degrade order under pressure:** shed **cosmetic → structural**. **Safety is never dropped** — if safety alone would exceed the budget, it still runs and the overrun is flagged (`over_budget_safety`).
- Triggers: work cost exceeds the per-train budget, or queue depth crosses the high-watermark (`queue_high_watermark = 0.8`).
- Dropped items are recorded → their zones are marked `coverage = degraded`, never "clean".

**Term — tier:** each check is tagged `cosmetic` | `structural` | `safety`; the tier decides degrade order.

---

## 10. Report + coverage

Per coach, the pipeline emits an **InspectionReport** (`inspection_report.schema.json`):
- `manifest_status` — present / missing / displaced / data_unavailable per expected component,
- `defects` — voted defect records (class, location, confidence, tier, source_model),
- measured values (buffer height, coupler sag) where applicable,
- `coverage` — per-zone inspected / degraded / data_unavailable,
- `worst_tier` and `review_status` (needs_review if any safety flag or any missing).

**Honest coverage:** a zone whose model did not run (or a dropped region) is `data_unavailable`/`degraded`, never reported as passed.

---

## 11. Internal contracts

- **Detection record** (`detection_record.schema.json`): one per voted flag — coach_index, coach_type, zone, axle_id, side, view, longitudinal_position_mm, class, conf, votes, tier, source_model.
- **Inspection report** (`inspection_report.schema.json`): the per-coach output above; validated before handoff to the report generator.
- **Component/defect taxonomy** (`component_defect_taxonomy.yaml`): the authoritative component × defect × coach-type × zone × tier schema; detectors train against it and manifests check against it.

---

## 12. Config keys (single source of truth)

```yaml
confidence: { component: 0.35 }
gate:       { p2_anomaly_percentile: 99.0 }
tiling:
  p2: { tile: 320, overlap: 0.20, gated: true }
  p3: { sahi: false }                       # wheel crops + slot checks only
voting:     { k_of_n: 3, cell_size_mm: 100.0, fuse_entry_exit: true }
completeness: { missing_min_views: 2 }
gate_tiers: { crack_seg: structural, wheel_seg: safety, fastener: safety, detect: structural }
degrade:    { order: [cosmetic, structural], queue_high_watermark: 0.8, fifo_capacity: 4 }
precision:  { detectors: fp8, seg: fp16, anomaly: fp16 }
exclude_classes: [leakage]
```

---

## 13. Glossary (developer terms)

| Term | Meaning |
|---|---|
| Backbone / head | shared feature network + small task-specific attachments |
| Gate | cheap full-field pass that decides where the expensive stage runs |
| ROI | region of interest — the belly area the gate flags |
| SAHI | slicing an image into overlapping tiles to catch small defects |
| Log-polar unwrap | flattening a round wheel into a strip for segmentation |
| k-of-n | require k confirming frames out of n before flagging |
| Vote cell | `(coach_index, longitudinal band, side)` bucket for accumulating votes |
| Entry/exit fusion | combining the two views of one wheel, keyed on axle/side |
| Manifest | the expected-component checklist for a coach type |
| Coach-type-conditional | the check set depends on LHB vs ICF |
| Tier | cosmetic / structural / safety — decides degrade priority |
| Coreset | the subsampled PatchCore memory bank (latency bound) |
| FP8 / FP16 | 8-bit / 16-bit inference precision on the Blackwell GPU |
| Resident | all model engines kept loaded in unified memory, no swap |
| Dice / mAP50 / AUROC | segmentation overlap / detection accuracy / anomaly separability metrics |
| data_unavailable | a region that was not inspected — never reported as clean |
