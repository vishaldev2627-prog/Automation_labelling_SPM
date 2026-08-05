# MLflow — Duties, Possibilities, and Limits for Vande Bharat Detection Pipeline

**Scope:** MLflow's exact role against the locked pipeline spec (`pipeline.md` + `FINAL_AIML_ARCHITECTURE.md` — DGX Spark, 128GB unified memory, 8 resident models, on-site, no cloud/edge split). This document supersedes MLflow sections in earlier RunPod-era planning.

**One-line answer to "can MLflow retrain models for me":** No. MLflow tracks and gates what a training run produces; it does not decide *when* to run one. Scheduling/triggering is a separate system you must build (§5).

---

## 1. What MLflow actually is (mechanically)

Three independent components, used together:

| Component | What it stores | What it does NOT do |
|---|---|---|
| **Tracking Server** | Run params, metrics (time-series), tags, artifact pointers | Does not run training. Just records what you tell it, when you tell it. |
| **Model Registry** | Named model versions, stage (`None→Staging→Production→Archived`), version metadata | Does not decide if a model is "good" — your script decides, registry just stores the verdict. |
| **Artifact Store** (S3/local/NFS) | Actual weight files, plots, configs | Not a serving engine. Doesn't convert `.pt`→engine format, doesn't load into inference. |

MLflow is a **passive record-keeper + gatekeeper API**, not an orchestrator, not a scheduler, not a serving runtime.

---

## 2. What MLflow DOES for this system

### 2.1 Experiment tracking (per training run)
For each of the **8** model families (§2.1a — 7 from `pipeline.md`'s detection
stack plus the new `VB-BufferBoundary`):

- Logs hyperparams (epochs, lr, batch size, img size, base weights).
- Logs metrics per-epoch (loss, mAP/Dice/AUROC as applicable) and final metrics.
- Logs artifacts (weights, config, confusion matrix / PR curve, calibration plots for anomaly).
- Logs tags: dataset SHA, **manifest version**, **taxonomy_version** (`component_defect_taxonomy.yaml` version — required, per pipeline.md §11), coach-type coverage, training machine, triggering job ID.

### 2.2 Model Registry — versioning + promotion gate
- One registered model per family (**8 total** — see §2.1a; no separate
  backbone/head entries; §5.1 of pipeline.md is one artifact, "one retrain,
  one engine").
- Stage-based lifecycle: `None → Staging → Production → Archived`.
- Promotion gate logic (a script you write, calling MLflow's client API) enforces:
  - Metric floor per family (mAP50 / Dice+length-recall / AUROC / recall@FP / acc+F1 — per pipeline.md §5).
  - Must beat current Production on composite score.
  - Probe-set validation (held-out, never in training).
  - **Tier-driven strictness** (from `gate_tiers` in pipeline.md §12; **revised**
    — see §2.2b for the full state machine this table summarizes):
    - `safety` (`VB-P3-WheelSeg`, `VB-P3-Fastener`): hardest floor + mandatory shadow test + **mandatory human sign-off** before Production.
    - `structural` (`VB-SharedDetector`, `VB-P2-CrackSeg`, `VB-CoachType`, most `VB-DefectState`, `VB-BufferBoundary`): threshold + beat-production + **mandatory shadow test** + **mandatory human sign-off**.
    - `cosmetic`: threshold + **mandatory shadow test** (cheap safety net, always run) + **auto-approve** if the shadow-test margin clears a high-confidence bar (`cosmetic_auto_approve_margin`, §2.2b) — otherwise falls through to human sign-off like the other tiers.
  - **Every promoted model, regardless of tier, is now covered by mandatory
    post-promotion automated rollback monitoring** (§2.2c) — this is new
    relative to the prior version of this document, which only implied it
    for safety tier.

### 2.1a Registered model families (8)

| # | MLflow registered name | pipeline.md ref | Tier | Notes |
|---|---|---|---|---|
| 1 | `VB-SharedDetector` | §5.1 | structural | Shared backbone + P1/P2/P3 heads + keypoints — one artifact. High blast radius: a bad version affects all three zones simultaneously. Treat with safety-tier *caution* in practice even though its formal tier is structural (§2.2a). |
| 2 | `VB-DefectState` | §5.2 | structural | P1 crop classifier |
| 3 | `VB-P2-Anomaly` | §5.3 | safety-adjacent (gate) | PatchCore — **not gradient-trained**; "retraining" means rebuilding the memory bank, not fine-tuning weights. See §2.2a special case. |
| 4 | `VB-P2-CrackSeg` | §5.4 | structural | |
| 5 | `VB-P3-WheelSeg` | §5.5 | **safety** | |
| 6 | `VB-P3-Fastener` | §5.6 | **safety** | |
| 7 | `VB-CoachType` | §5.7 | structural | Data-efficient per pipeline.md — tolerates a slower cadence |
| 8 | `VB-BufferBoundary` | `FINAL_AIML_ARCHITECTURE.md` §2.2, `COACH_BOUNDARY_BUFFER_DETECTOR.md` | structural | **New.** Its output (`coach_hint`) is a preprocessing-time storage convenience, never report-authoritative — but the model itself still goes through the same registry/promotion/rollback machinery as every other family. First training cycle is a **cold start** (no existing labeled `buffer_visible` data — flagged open in the companion doc), not a retrain; do not schedule it on the standard retrain cadence until an initial dataset and baseline Production version exist. |

### 2.2a Fine-tune vs. full retrain — the decision your trigger script must make

MLflow does not make this call — it only logs whichever one your script runs
(`base_weights` tag records which happened, for audit). Decision logic, run
per family at each trigger (§5):

```
on retrain_trigger(family):
    if taxonomy_version_changed(family) or manifest_version_changed(family):
        mode = FULL_RETRAIN        # label/class space changed — fine-tuning
                                    # an old checkpoint onto new classes risks
                                    # catastrophic forgetting / misalignment
    elif drift_score(family) > TAU_DRIFT:         # from vb-production-monitoring
        mode = FULL_RETRAIN        # candidate would be fine-tuning on top of
                                    # a baseline already known to be stale
    elif (new_samples(family) / total_dataset(family)) >= FINE_TUNE_RATIO_CEILING \
         or days_since_last_full_retrain(family) >= FULL_RETRAIN_MAX_INTERVAL_DAYS:
        mode = FULL_RETRAIN        # periodic baseline refresh — prevents an
                                    # indefinite chain of fine-tunes drifting
                                    # away from a from-scratch baseline
    elif new_samples(family) < MIN_RETRAIN_THRESHOLD(family):
        mode = SKIP                # not enough new data — log why, no run
    else:
        mode = FINE_TUNE           # small incremental batch, no taxonomy
                                    # change, no drift, recent full retrain —
                                    # cheap, fast, matches the 2-day/weekly cadence
```

- **`TAU_DRIFT`, `FINE_TUNE_RATIO_CEILING`, `FULL_RETRAIN_MAX_INTERVAL_DAYS`
  are not yet set** — tune once real drift-metric and data-growth numbers
  exist; treat the values above as placeholders in the pseudocode, not
  defaults to hardcode blind.
- **`VB-P2-Anomaly` special case:** PatchCore has no fine-tune/full-retrain
  distinction in the gradient-training sense — "retraining" is always a
  **full memory-bank rebuild** from the current confirmed-normal set, always
  behind the purity gate (§3 gap, §5.2 step 2). Route this family straight to
  `FULL_RETRAIN` semantics (rebuild) whenever `new_samples` (confirmed-normal
  additions) clears its threshold; the `mode` branch above simplifies to a
  binary rebuild-or-skip for this one family.
- **`VB-SharedDetector` caution:** formally `structural` tier, but because
  every zone head depends on this one backbone artifact, prefer `FULL_RETRAIN`
  over `FINE_TUNE` more readily than the ratio/interval rule alone would
  suggest — a bad fine-tune here has a wider blast radius than a bad fine-tune
  on a single-zone specialist. Policy call, not an MLflow mechanism; encode it
  as a lower `FINE_TUNE_RATIO_CEILING` for this family specifically if the
  generic rule feels too permissive once real data exists.

### 2.2b Promotion flow — comparison → shadow test → human verification

The full state machine from "candidate produced" to "either Production or
shelved." Every step logs to MLflow regardless of outcome — a candidate that
never gets promoted is not a wasted run, it's an audited data point.

```
CANDIDATE PRODUCED (fine-tune or full retrain, §2.2a)
    │
    ▼
[1] mlflow.start_run() — log params/metrics/artifacts/tags as usual (§2.1)
    Every version is stored here, unconditionally — this step never gates.
    │
    ▼
[2] METRIC-FLOOR CHECK (tier-driven floor, pipeline.md §5 thresholds)
    fail ──► stage stays None/Staging, reason logged, CYCLE ENDS
    pass │
    ▼
[3] COMPOSITE-SCORE COMPARISON vs. current Production
    Compares on: the family's domain metric(s) (§2.2) AND training-data
    recency (each run is date/dataset-SHA tagged, §2.1) — if two candidates'
    metrics are within a small epsilon of each other, prefer the one trained
    on more recent data as a tie-break, logged explicitly as the reason.
    does NOT beat Production ──► stays Staging/Archived, reason logged, CYCLE ENDS
                                  ("MLflow stores it as usual" — tracked, not promoted)
    beats Production │
    ▼
[4] SHADOW TEST (mandatory, all tiers — see reconciliation note above §2.2)
    Candidate runs in parallel on live (or recently-mirrored) traffic,
    receiving the same inputs as current Production, but its outputs are
    NEVER served — comparison only. Your orchestrator mirrors traffic and
    computes agreement/performance stats (MLflow doesn't do this — §3 gap);
    log the comparison as its own MLflow run, linked to the candidate version.
    underperforms in shadow (even though it beat Production offline —
    the offline eval set may not be representative) ──►
        stays Staging/Archived, shadow-test failure reason logged, CYCLE ENDS
    surpasses Production in shadow │
    ▼
[5] HUMAN VERIFICATION (tier-scoped — see reconciliation above §2.2)
    `safety` / `structural`: mandatory human sign-off. Notification (Slack/
    email) with the MLflow run + shadow-comparison report link; a human
    reviews and clicks approve/reject.
    `cosmetic`: auto-approved if shadow-test margin ≥ `cosmetic_auto_approve_margin`
    (not yet set — needs real shadow-test variance data to calibrate);
    otherwise falls through to the same manual gate as the other tiers.
    human/policy REJECTS ──► stays Staging/Archived, rejection reason
                              logged (feeds back into future trigger tuning), CYCLE ENDS
    APPROVED │
    ▼
[6] PROMOTE: MLflow stage-move Staging → Production. Previous Production
    auto-demoted to Archived (never deleted — rollback-ready, §2.3).
    Export FP8/FP16 engine (§3 gap — your export script, not MLflow).
    Hot-swap into the resident DGX session (your loader, §8 of
    FINAL_AIML_ARCHITECTURE.md) — a pointer swap, not a pipeline restart.
    │
    ▼
[7] POST-PROMOTION AUTOMATED ROLLBACK MONITORING (§2.2c) — mandatory, all tiers
```

### 2.2c Post-promotion automated rollback monitoring

**Purpose:** the promotion gate (steps 2–5) is necessarily evaluated on
held-out/shadow data, which is never a perfect proxy for full live behavior
across every coach type and edge case. This is the safety net that catches
what the gate missed, **without ever stopping the pipeline** — the pipeline
keeps serving the just-promoted model while this monitor watches it; a
rollback is a hot-swap back to the previous engine, not downtime.

- **Canary window:** a defined time-bound or train-count-bound observation
  period starting the moment a version is promoted (exact size not yet
  set — needs to be long enough to see representative coach-type/defect-class
  coverage, short enough that a bad model isn't live for long; a starting
  point to tune from is "first N trains or M hours, whichever is longer").
- **What it watches** (reusing the existing `vb-production-monitoring`
  metrics, §2.4, compared against the pre-promotion baseline established by
  the previous Production version's own historical stats):
  - Per-zone mean confidence and zero-detection rate — a sudden shift
    suggests the new model is behaving differently on live data than it did
    in shadow.
  - Gate-recall (periodic full-SAHI audit, pipeline.md §6) — a drop here is
    a safety-relevant regression, watched especially closely for `safety`-tier
    families.
  - Per-tier flag-rate — a spike can mean false-positive flood; a drop can
    mean missed defects.
  - Pipeline latency impact — does the new version risk breaching the ≤5 min
    SLA (`FINAL_AIML_ARCHITECTURE.md` C4)? A slower model is a regression
    even if its detection metrics are fine.
  - Any hard-fail/exception rate increase.
- **Trigger for automatic rollback:** any watched metric crosses a
  regression threshold relative to the pre-promotion baseline (thresholds
  not yet set per metric — needs real baseline variance data; do not pick
  arbitrary numbers before that data exists, but do not skip setting them
  once it does — an unmonitored promotion defeats the point of this section).
- **Rollback action (automatic, no human step in the loop for this specific
  action — speed matters here):** MLflow stage-move back to the previous
  Production version (<2 min, §2.3 — no retrain, no file copy) + the
  DGX-resident loader hot-swaps the engine pointer back. The pipeline never
  stops serving through this transition.
- **Flagging (mandatory, not optional):** the rollback event, the metric(s)
  that triggered it, the magnitude of the regression, and a sample of the
  failing cases are logged to MLflow (new experiment, e.g.
  `vb-rollback-events`) and alerted (Slack/PagerDuty — MLflow has no
  built-in alerting, §3 gap) so the next retrain cycle has concrete diagnostic
  input rather than a bare "it didn't work." This is what keeps the
  improvement loop converging instead of oscillating blindly between
  versions.
- **If the canary window completes with no regression:** the promotion is
  considered confirmed-stable; the version continues under standard
  `vb-production-monitoring` (§2.4) with no special canary scrutiny beyond
  that point.

### 2.3 Rollback
- Archived versions never deleted → revert to previous Production in <2 min via a stage-move call, no retrain, no file copy.
- Promotion log (who/when/what composite score) — for audit, safety-critical traceability.

### 2.4 Production monitoring
- Separate experiment (`vb-production-monitoring`) logs post-batch stats: per-zone mean confidence, zero-detection rate, gate-recall (from periodic full-SAHI audit), per-tier flag rates.
- These logged metrics are what your drift-detection script reads to decide "retrain needed early" — MLflow stores the numbers, doesn't compute or act on drift itself.

### 2.5 Reproducibility / audit
- Any past model version traces back to: exact dataset SHA, exact manifest/taxonomy version, exact hyperparams, exact code (if you tag Git commit).
- Critical for a safety-tiered system — when `VB-P3-WheelSeg` is challenged after an incident, you can answer "what data, what code, what metrics" in minutes.

---

## 3. What MLflow does NOT do (must build separately)

| Gap | Why it matters here | What you build instead |
|---|---|---|
| **Scheduling / triggering retraining** | Your 2-day→weekly cadence requires *something* to fire on a schedule | Cron / systemd timer / GitHub Actions schedule / Prefect — see §5 |
| **"New data arrived" detection** | You want retrain "as data arrives," not blind schedule | A small watcher script (data-drop counter, file-count threshold, or manifest of new labelled batches) — MLflow has no hook for this |
| **Serving / inference runtime** | 8 resident engines on DGX Spark, no swap | Your own resident-loader / inference server. MLflow Registry is a pointer, not a server. See §6.3. |
| **FP8/TensorRT engine export** | pipeline.md C7: FP8 detectors, FP16 seg/anomaly | Your export script (post-training, pre-registration or post-promotion) — MLflow stores the source weights, not the compiled engine, unless you explicitly log the engine file as an artifact too |
| **Domain metric computation** | Dice/IoU, length-recall, shelling%, gate-recall, recall@FP | Your eval scripts compute these; MLflow just stores the numbers you hand it |
| **PatchCore purity gate** | one mislabeled "normal" sample poisons the whole anomaly model | A pre-training data-hygiene script; log pass/fail as a tag, refuse `mlflow.start_run()` if it fails |
| **Shadow-mode comparison** | safety-tier models need live-traffic comparison before promote | Your orchestrator mirrors traffic, computes agreement stats, logs them to MLflow as a run — MLflow doesn't mirror traffic |
| **Backbone-head coupling checks** | N/A now — pipeline.md §5.1 confirms shared detector is one engine, not split heads. No coupling problem in this design. | — (resolved by pipeline.md's architecture, not by MLflow) |
| **GPU/resource scheduling on DGX Spark** | training run and 8 resident inference models both want the same box's memory/compute | Your job scheduler must ensure training runs don't starve the resident inference session — MLflow has zero awareness of GPU contention. See §6.8. |
| **Alerting on drift** | pipeline.md's ≤5min SLA has no room for a silently-degrading model | Your monitoring script reads MLflow-logged drift metrics and calls out (Slack/email/PagerDuty) — MLflow has no built-in alerting. See §6.9. |

**Bottom line on your question:** you must automate the retraining logic yourself. MLflow will happily log and gate whatever that automation produces, but it will never decide on its own to start a run. **Full build spec for every row above: §6.**

---

## 4. Division of responsibility (explicit)

```
YOUR AUTOMATION (build this)              MLFLOW (already does this)
─────────────────────────────             ──────────────────────────
Watch for new labelled data      ──┐
Decide "enough data, go" (§5)      │
Kick off training job (per family) ├──►    Log params/metrics/artifacts
Run eval script                    │       Store run history
Compute domain metrics             │
                                    ├──►    Register model version
Call promotion-gate script         │
                                    ├──►    Enforce stage transitions
                                    │       (Staging/Production/Archived)
Export to FP8/engine format        │
Deploy to DGX resident session   ◄─┘       Serve as source-of-truth
                                            "what's in Production now"
Monitor production (drift calc)  ──►       Log monitoring metrics
Decide "retrain early" trigger   ◄──       (you read the metrics back)
```

---

## 5. Retrain automation design for your cadence (2-day, then weekly)

MLflow plays no role in this section except as the thing being *called by* the automation below.

### 5.1 Trigger mechanism
Use a scheduler external to MLflow — recommend **cron on the DGX host (or a companion VM) + a small Python trigger script**, not GitHub Actions (no cloud round-trip needed now that everything's on-site; GitHub Actions made sense in the old RunPod plan, not here).

```cron
# First month: every 2 days at 02:00
0 2 */2 * *  /opt/vb/scripts/retrain_trigger.sh

# After month 1: switch to weekly (edit crontab, or make retrain_trigger.sh
# read a "phase" config so you don't have to remember to edit crontab)
0 2 * * MON  /opt/vb/scripts/retrain_trigger.sh
```

Better: make the schedule config-driven, not two separate crontab edits:

```yaml
# retrain_schedule.yaml
phase_1:
  starts: 2026-08-03
  ends:   2026-09-02
  interval_days: 2
phase_2:
  starts: 2026-09-03
  interval_days: 7
```
One daily cron checks this config and no-ops unless the interval has elapsed since last run (store `last_run_ts` in a small state file or an MLflow tag on the last triggering run — reuse MLflow as the state store here, cheap and audit-friendly).

### 5.2 Trigger script responsibilities (this is the part you build)
1. **Data-readiness check** — count new labelled samples since last run per model family (query your label store / dataset manifest). If below a minimum threshold, skip this cycle and log why (avoid retraining on 12 new images).
2. **Purity gate** (anomaly model only) — verify no defect leaked into the confirmed-normal set.
3. **Sequential per-family training** — DGX Spark is one device; even with 128GB unified memory, training and the 8 resident inference engines share it (concrete scheduling design: §6.8). Either:
   - run training in an isolated time window with inference paused/degraded, or
   - reserve a memory budget slice and confirm it doesn't evict resident inference models.
   This is a resourcing decision your script must enforce — MLflow has no concept of this.
4. **Call your existing training scripts**, each wrapped in `mlflow.start_run()` (§2.1).
5. **Call eval script** → domain metrics → log to MLflow.
6. **Call promotion-gate script** (§2.2) → tier-driven checks → stage transition if passed.
7. **If promoted:** export FP8/FP16 engine, hot-swap into the resident DGX session (your loader, not MLflow) with a rollback path ready (§2.3) if the new version misbehaves live.
8. **Log a monitoring run** summarizing the cycle (data volume, pass/fail per family, promoted or skipped) — separate from `vb-production-monitoring`, e.g. `vb-retrain-cycles`.

### 5.3 Per-family independence
Not all 7 families need retraining on the same cadence just because you check every 2 days. Structure the trigger script per-family:

```
for family in [SharedDetector, DefectState, P2Anomaly, P2CrackSeg, P3WheelSeg, P3Fastener, CoachType]:
    if new_data_count(family) >= min_threshold(family):
        run_training(family)
    else:
        skip_and_log(family, reason="insufficient new data")
```
Safety-tier families (`P3WheelSeg`, `P3Fastener`) may warrant a lower `min_threshold` (retrain sooner on less data, given C6) than cosmetic-tier ones — a policy call, not an MLflow setting.

### 5.4 Human-in-the-loop and shadow-test/rollback wiring
**Superseded/expanded by §2.2b–§2.2c** — the full promotion flow (metric
floor → composite comparison → mandatory shadow test, all tiers → tier-scoped
human verification → promote → mandatory post-promotion rollback monitoring,
all tiers) is now the standard path for every family, not just safety tier.
What step 4 of §5.2 ("call promotion-gate script") actually invokes is that
whole state machine, not a single threshold check. Given C6
("safety-critical checks never skipped"), `VB-P3-WheelSeg` / `VB-P3-Fastener`
never skip the human sign-off step (§2.2b step 5) — that part is unchanged
from the original design. What's new: even `cosmetic`-tier auto-approvals
and `structural`-tier promotions now go through the mandatory shadow test
(§2.2b step 4) and mandatory post-promotion rollback monitoring (§2.2c) —
previously only implied for safety tier, now explicit and universal, because
the rollback safety net is cheap to apply everywhere and the pipeline never
stops serving during a rollback (§2.2c).

---

## 6. Automation components — full build spec

Every gap in §3's table, specced out so the development team can build each
one without needing to re-derive purpose or design from a one-line table
cell. Same structure per component: **Purpose** (why it exists) · **Design**
(how it works) · **Inputs/Outputs** · **Where it lives** (repo location) ·
**Trigger** (what invokes it, how often) · **Failure handling** (what
happens when it breaks — never silent).

### 6.1 Retrain scheduler / trigger

- **Purpose:** fire a retrain cycle on the configured cadence (2-day →
  weekly, §5.1) without a human remembering to start one.
- **Design:** already specced in full at §5.1 — cron (or systemd timer) on
  the DGX host, reading a config-driven schedule file
  (`retrain_schedule.yaml`), no-op unless the interval has elapsed since the
  last run (state tracked via an MLflow tag on the last triggering run, or a
  local state file).
- **Inputs:** `retrain_schedule.yaml` (phase/interval config), last-run
  timestamp.
- **Outputs:** invokes §6.2 (data watcher) then, per family, the training
  job.
- **Where it lives:** `/opt/vb/scripts/retrain_trigger.sh` + companion
  `retrain_schedule.yaml`.
- **Trigger:** daily cron check; no-ops most days, fires on interval.
- **Failure handling:** if the trigger script itself crashes, the next
  day's cron invocation retries — no manual intervention needed, but a
  missed cycle should be logged (write to `vb-retrain-cycles`, §5.2 step 8)
  so a silent multi-day gap is visible, not just inferred from absence.

### 6.2 "New data arrived" detector

- **Purpose:** answer "is there enough new labelled data to justify a
  retrain cycle for this family" — without this, the scheduler (§6.1) would
  either retrain blind on a fixed schedule regardless of data volume, or
  never retrain at all.
- **Design:** a small watcher that, per family (§2.1a, 8 families), counts
  labelled samples added since that family's last training run. Source of
  the count is whatever labelled-data store/manifest this repo already uses
  (dataset manifest with per-sample family/class tags, or a direct query
  against the label store) — **not yet named/confirmed which one**, flag
  before building if more than one candidate source exists. Compares the
  count against `MIN_RETRAIN_THRESHOLD(family)` (§5.3 — safety-tier families
  get a lower threshold, deliberately, §12.1 of `FINAL_AIML_ARCHITECTURE.md`).
- **Inputs:** labelled-data store/manifest, per-family last-training
  timestamp (read from MLflow run history — the most recent run's tag, not a
  separate state store, so there's one source of truth).
- **Outputs:** per-family `{ready: bool, new_sample_count: int}`, logged as
  a tag on the eventual run (or logged even when `ready=False`, as a
  skipped-cycle record — visibility matters here as much as the decision).
- **Where it lives:** `automation/data_watcher.py`, called as step 1 of the
  per-family loop inside `retrain_trigger.sh` (§5.2 step 1).
- **Trigger:** invoked synchronously by §6.1 on every scheduled check, once
  per family.
- **Failure handling:** if the label store/manifest is unreachable, that
  family's cycle is skipped (not crashed) — log the reason, alert (§6.9) if
  this happens repeatedly (a data-pipeline problem worth knowing about
  independent of any single retrain cycle), and do not let one family's
  data-source outage block the other 7 families' checks.

### 6.3 Serving / inference runtime (resident loader)

- **Purpose:** keep all 8 model engines resident in the DGX Spark's 128GB
  unified memory (C2) and serve inference to the AI pipeline, while
  supporting **hot-swap** — replacing the active engine for a family without
  restarting the pipeline — for both promotion (§2.2b step 6) and rollback
  (§2.2c).
- **Design:** extend the existing `GPU/yolo/server.py`
  (`FINAL_AIML_ARCHITECTURE.md` §8, §13) into a resident multi-model server.
  Holds one active engine per family; exposes an internal swap operation
  that (1) loads the new engine into memory alongside the currently-active
  one, (2) runs a cheap sanity check (engine loads, produces output on a
  known probe input, shape/dtype as expected) — **never swap blind**, (3)
  atomically flips the "active" pointer for that family only, (4) keeps the
  previous engine warm for a short grace period in case an immediate
  rollback is needed, then unloads it. Framework choice (Triton vs. a
  simpler custom server) is explicitly left open in
  `FINAL_AIML_ARCHITECTURE.md` §8 — re-evaluate now that VRAM scarcity isn't
  the driving constraint it was on the old L4 plan.
- **Inputs:** MLflow Registry query ("what's in Production now" per family),
  the exported engine file (§6.4) for the version being swapped in.
- **Outputs:** inference results to the AI pipeline (§5–§7 of
  `pipeline.md`/`FINAL_AIML_ARCHITECTURE.md`); swap success/failure status
  back to the promotion/rollback caller.
- **Where it lives:** `GPU/yolo/server.py` (extended).
- **Trigger:** started once per session (long-running); swap operation
  triggered by §2.2b step 6 (promotion) or §2.2c (automatic rollback) via a
  direct call or internal message, not on any schedule of its own.
- **Failure handling:** if the new engine fails the sanity check or fails to
  load, **abort the swap, keep the previous engine active**, and surface the
  failure to whichever flow triggered it (promotion attempt fails cleanly;
  rollback attempt is the more severe case — if a rollback itself can't
  load, this must page a human immediately, since the pipeline is now stuck
  on a known-bad model with no automatic recovery path). The pipeline must
  never end up with **zero** engine loaded for a family — the old engine
  stays active until a new one is verified good.

### 6.4 FP8/TensorRT engine export

- **Purpose:** convert trained source weights into the deployable precision
  format this system actually serves (C7: FP8 detectors/classifiers with
  INT8 fallback, FP16 segmentation/anomaly) — MLflow stores whatever you log
  it, but never compiles anything itself.
- **Design:** extend the existing `export.py --format engine --half` with
  FP8 calibration (a calibration dataset pass, per pipeline.md C7). **Runs
  right after the metric-floor check passes (§2.2b step 2), before the
  shadow test (step 4)** — deliberately, so the shadow test evaluates the
  actual artifact that would be served, not the uncompiled source weights;
  precision-loss regressions from FP8 quantization need to show up in shadow
  results, not slip through because shadow tested a different artifact than
  what gets deployed.
- **Inputs:** trained source weights (`.pt`), a calibration dataset
  (representative sample for FP8 range calibration — not the training set
  itself, to avoid calibration overfitting).
- **Outputs:** compiled engine file, logged to MLflow **as an artifact
  alongside the source weights** (§3 gap note — this doesn't happen
  automatically, must be an explicit `mlflow.log_artifact()` call).
- **Where it lives:** `export.py` (extended).
- **Trigger:** invoked by the trigger script immediately after a candidate
  clears the metric floor (§2.2b step 2→3 boundary).
- **Failure handling:** export/calibration failure fails the candidate at
  this stage — same as any other gate failure (§2.2b): logged, cycle ends
  for that candidate, no promotion. This is a hard stop, not a fallback to
  serving uncompiled weights (which would violate C7's precision policy).

### 6.5 Domain metric computation (eval scripts)

- **Purpose:** compute the actual per-family metrics (mAP50 / Dice+
  length-recall / AUROC / recall@FP / acc+F1 — `pipeline.md` §5) that the
  promotion gate and the composite-score comparison (§2.2b step 3) run on.
  MLflow stores whatever number you hand it; it has no idea what a Dice
  score even means.
- **Design:** one eval script per family (or a shared harness with a
  per-family metric plugin — implementation detail, not fixed here), run
  against a **held-out probe set that is never in the training split**
  (§2.2, `pipeline.md`'s train/valid/test split discipline). Output is a
  standardized JSON — `{metric_name: value, ...}` — plus provenance fields
  (`dataset_sha`, `eval_timestamp`) so the composite comparison's
  date-recency tie-break (§2.2b step 3) has something concrete to compare.
- **Inputs:** the trained (or exported, §6.4) model, the held-out probe set.
- **Outputs:** standardized metrics JSON, logged to MLflow (§2.1).
- **Where it lives:** `eval/` (one module/script per family).
- **Trigger:** invoked immediately after training/export succeeds, before
  the registry stage decision (§2.2b step 2).
- **Failure handling:** an eval crash must **not** silently default to
  pass or fail — hold the candidate at `None` stage, flag it for developer
  attention (§6.9), and do not let the promotion gate proceed on missing
  numbers.

### 6.6 PatchCore purity gate

- **Purpose:** stop one mislabeled "normal" sample from poisoning the
  `VB-P2-Anomaly` memory bank — PatchCore has no backprop-based error
  correction to absorb a bad label the way a gradient-trained model might
  partially tolerate one; a single contaminated sample directly corrupts the
  bank it's stored in.
- **Design:** a pre-training script that cross-checks every sample in the
  confirmed-normal dataset against known-defect records (the existing
  `DefectReviewLog`, `FINAL_AIML_ARCHITECTURE.md` §13) — any sample that
  overlaps a logged defect at that position/timeframe is flagged as suspect
  and the run is refused until resolved. Outputs a pass/fail plus the
  suspect list (for manual cleanup, not automatic removal — removing data
  silently is its own risk).
- **Inputs:** confirmed-normal dataset manifest, `DefectReviewLog`.
- **Outputs:** pass/fail tag; if fail, a suspect-sample list for a human to
  clean up.
- **Where it lives:** `automation/purity_gate.py`.
- **Trigger:** mandatory pre-check before every `VB-P2-Anomaly` cycle
  (memory-bank rebuild, §2.2a special case) — refuse `mlflow.start_run()` if
  it fails (§3's original framing, unchanged).
- **Failure handling:** fail = **no training run happens at all** for this
  cycle. Alert (§6.9) with the suspect list attached so the data-hygiene
  issue gets fixed before the next scheduled attempt, rather than silently
  retrying against the same contaminated set.

### 6.7 Shadow-mode traffic mirroring + comparison

- **Purpose:** the mandatory shadow test every candidate goes through
  (§2.2b step 4) before human verification — running the candidate against
  real-world-shaped input without ever serving its output, so promotion
  isn't decided on held-out-eval numbers alone.
- **Design — and a real constraint to flag, not gloss over:** true
  live-parallel shadow inference (running the candidate on every train as it
  arrives, alongside Production) doubles GPU compute during the exact window
  the ≤5 min SLA (`FINAL_AIML_ARCHITECTURE.md` C4) is measured against — on
  a single-device, no-cloud-burst system (C1, C3 "single DGX Spark = SPOF"),
  that risk needs to be taken seriously rather than assumed away. **Proposed
  design (open item, confirm before building):** shadow test runs as a
  **replay** against recently-processed trains' already-stored raw +
  preprocessed frames (preprocessing's raw retention window — not yet
  explicitly re-specified in this architecture after the RunPod-era
  retention config was dropped in the rewrite; needs a decision, not an
  assumption, on how many days/trains of replay data to retain for this
  purpose specifically), run **off the live SLA-critical path**, rather than
  true live-parallel mirroring. This trades "instant" shadow results for
  protecting the SLA.
- **Inputs:** candidate's exported engine (§6.4), a replay window of
  recently-processed trains (raw + preprocessed, with their known-good
  Production-model outputs already on record for comparison).
- **Outputs:** agreement/performance stats (candidate vs. Production on the
  same replayed inputs) — logged as its own MLflow run, linked to the
  candidate version (§2.2b step 4).
- **Where it lives:** `automation/shadow_test.py` (or an orchestrator
  module, if traffic-replay logic already exists elsewhere in this repo —
  check before building a second one).
- **Trigger:** invoked automatically once a candidate clears the
  composite-score comparison (§2.2b step 3→4 boundary).
- **Failure handling:** if the replay window doesn't have enough volume/
  coverage (e.g., too few of a given coach-type or defect-class in the
  window) to draw a reliable conclusion, that is itself a failure mode —
  flag "insufficient coverage," do not silently pass the candidate through
  on a thin sample.

### 6.8 GPU / resource scheduling on the DGX Spark

- **Purpose:** training (§6.1–§6.2's triggered jobs) and the 8 resident
  inference engines (§6.3) share one device (C1) — without explicit
  scheduling, a training run can starve or evict resident inference and
  break the live pipeline's ≤5 min SLA (C4) or, worse, its safety-tier
  checks (C6).
- **Design:** **inference always wins** — this follows directly from design
  principle 8 (`FINAL_AIML_ARCHITECTURE.md` §1: "degrade tiers, never drop
  capture... safety never dropped"). Concretely: a job scheduler/queue that
  only starts a training run when the pipeline is not actively mid-inference
  on a train (checks the pipeline's busy/idle state, not a blind time
  window — a depot's train arrivals aren't necessarily on a predictable
  schedule, so a fixed off-peak window is a weaker guarantee than an
  explicit busy-check). If a train arrives while training is running,
  training **pauses/preempts** — never inference. This is a stricter
  reservation model than the memory-budget-slice alternative (reserve a
  fixed VRAM slice for training, let both run concurrently) — the
  budget-slice approach is only safe once R-VRAM (`FINAL_AIML_ARCHITECTURE.md`
  §14) is actually measured; until then, the busy-check/preempt model is the
  conservative default.
- **Inputs:** pipeline busy/idle state (from the AI detection stage,
  `FINAL_AIML_ARCHITECTURE.md` §3), the pending training job queue.
- **Outputs:** a go/no-go/pause decision per training job attempt.
- **Where it lives:** `automation/gpu_scheduler.py`, consulted by
  `retrain_trigger.sh` (§5.2 step 3) before starting any training run.
- **Trigger:** checked before every training-job start, and monitored
  continuously while a training job runs (to catch a mid-run train
  arrival).
- **Failure handling:** if a safe window can't be found within a maximum
  wait time, skip the cycle, log why, and let the next scheduled trigger
  (§6.1) try again — never force a training run through at the cost of
  inference availability.

### 6.9 Alerting (drift, rollback, human-verification-needed, cycle summaries)

- **Purpose:** MLflow logs numbers; it never tells anyone. Every event in
  this system that needs a human's attention — a rollback firing (§2.2c), a
  promotion waiting on sign-off (§2.2b step 5), drift crossing a threshold
  (§2.4), a data-source outage (§6.2), a purity-gate failure (§6.6) — needs
  an explicit push, not a hope that someone checks a dashboard.
- **Design:** a notification dispatcher that polls the MLflow tracking API
  (MLflow has no push/webhook mechanism of its own to rely on here — polling
  is the honest mechanism, not a placeholder for something better) on a
  short interval, watching for new runs/tags in the relevant experiments
  (`vb-production-monitoring`, `vb-rollback-events`, `vb-retrain-cycles`,
  pending-`Staging` runs awaiting human sign-off). Routes by severity:
  **rollback events → urgent channel** (PagerDuty or high-priority Slack —
  this is the one that most directly means "a live model just got pulled");
  **human-verification-needed → Slack/email with the run link** (matches the
  approve/reject flow already described in §2.2b step 5, §5.4); **drift /
  data-outage / purity-gate failures → standard alert channel**; **routine
  cycle summaries → daily digest**, not urgent.
- **Inputs:** MLflow tracking API (polled), the event type/severity from
  whichever component logged it.
- **Outputs:** Slack/email/PagerDuty notifications (channel/service choice
  not fixed here — match whatever this team already uses operationally).
- **Where it lives:** `automation/alerting.py`.
- **Trigger:** continuous poll loop (interval not yet chosen — balance
  "rollback alerts arrive fast" against "don't hammer the MLflow tracking
  API"; a short interval, e.g. under a minute, for the urgent categories is
  a reasonable starting point, longer for the daily-digest category).
- **Failure handling:** if the alerting dispatcher itself can't reach
  Slack/PagerDuty, it must **not** silently swallow the event — write to a
  local durable log file as a fallback channel, so a rollback event is never
  lost even if the notification transport is down. This is the one
  component in the whole automation layer where "fail silently" is
  categorically unacceptable, given what it's alerting on.

### 6.10 Backbone-head coupling checks — resolved, no automation needed

Carried forward unchanged: `pipeline.md` §5.1 confirms the shared detector is
one artifact (backbone + heads together), not separately-versioned pieces
that could drift out of sync with each other. There is no coupling problem
in this design to automate a check for.

---

## 7. Summary

- MLflow: tracking, registry, promotion-gate enforcement, rollback, audit trail. Excellent at all of it.
- MLflow: zero scheduling, zero data-watching, zero serving, zero drift-computation, zero GPU-contention awareness.
- Your 2-day/weekly cadence needs an external trigger (cron + config-driven schedule, §5.1), a data-readiness check (§5.2.1), and resourcing discipline on the shared DGX device (§5.2.3) — none of which MLflow provides. Build once, it calls MLflow at each step; MLflow never calls itself.
- **The full promotion path, end to end, all 8 families:** trigger →
  fine-tune-or-full-retrain decision (§2.2a, and a memory-bank rebuild for
  `VB-P2-Anomaly` specifically) → metric floor → composite comparison vs.
  current Production (performance + date-tagged tie-break) → **mandatory
  shadow test, every tier** (§2.2b step 4) → tier-scoped human verification
  (§2.2b step 5) → promote + hot-swap (§2.2b step 6) → **mandatory
  post-promotion automated rollback monitoring, every tier** (§2.2c) → either
  confirmed-stable or automatic rollback + flagged diagnostics feeding the
  next cycle. The pipeline never stops serving at any point in this loop —
  every transition is a pointer/stage swap, never a restart.
