"""Application configuration loaded from environment variables / .env file."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the annotation tool backend."""

    # protected_namespaces=(): pydantic reserves the "model_" prefix for its
    # own internal fields by default, which would otherwise warn on this
    # class's genuine model_promotion_* settings (M7.5) - none of which are
    # pydantic internals, so the protection isn't needed here.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=()
    )

    # Dataset
    dataset_path: str = "../dataset"
    exports_path: str = "../exports"
    models_path: str = "../models"
    state_dir_name: str = ".annotation_state"

    # SAM2
    sam_checkpoint: str = "../models/sam2.1_hiera_large.pt"
    sam_model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_device: Literal["cuda", "cpu", "mps", "auto"] = "auto"
    sam_use_onnx: bool = False
    sam_onnx_encoder_path: str = "../models/sam2_encoder.onnx"
    sam_onnx_decoder_path: str = "../models/sam2_decoder.onnx"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Processing
    max_image_dimension: int = 4096
    thumbnail_max_dimension: int = 1024
    polygon_epsilon_ratio: float = 0.002
    min_polygon_points: int = 3
    # Margin added around a component's bbox when cutting the crop the
    # p1_side_damage condition classifier trains on. 7.5% matches the
    # pipeline's own P1 crop margin (`homography.crop_margin_pct` in
    # FINAL_AIML_ARCHITECTURE §10) so training crops are cut the same way the
    # serving path cuts them - a different margin at train time than at
    # inference time is a silent domain shift.
    condition_crop_margin_pct: float = 7.5
    mask_confidence_threshold: float = 0.5
    batch_max_workers: int = 2

    # Similarity-based propagation (Phase 1): when an image is finalized,
    # carry its accepted objects onto near-duplicate images so they don't
    # need to be annotated from scratch.
    propagation_enabled: bool = True
    similarity_threshold: float = 0.85
    propagation_top_k: int = 5

    # Classes the pipeline never serves, so we must never ship labels for them.
    # `leakage` is all-synthetic (docs/pipeline.md §5.2, §12 and
    # FINAL_AIML_ARCHITECTURE §10: `exclude_classes: [leakage]`), and the build
    # plan requires excluding it at the tool level rather than trusting export
    # to filter it. Comma-separated; matched case-insensitively on class name.
    # Part of every class-map version's hashed content, so changing this mints a
    # new version rather than silently altering what past snapshots meant.
    exclude_classes: str = "leakage"

    # Wheel log-polar unwrap (W-5, D-Q5). PROVISIONAL / ENGINEERING PLACEHOLDER,
    # NOT CERTIFIED - "wheel specs are to be considered in your own accord for
    # now" (pipeline team). Deliberately NOT a wheel diameter in mm: converting
    # a real-world diameter to a pixel radius needs camera calibration, which
    # does not exist yet (stitcher/homography prerequisites are still open,
    # FINAL_AIML_ARCHITECTURE §16). Unlike pipeline.md §5.5's own production
    # method ("circle seeded from fixed geometry + known diameter, not blind
    # Hough" - for raw camera frames with no human in the loop), this export
    # tool derives the circle per-image from the annotated `wheel_class_name`
    # object's own bbox instead - grounded in that frame's actual annotation
    # rather than a fixed pixel constant that would be wrong the moment camera
    # distance/zoom varies between frames. These settings only control padding
    # and output resolution, not wheel geometry itself - see D-Q5's Consequence
    # C-4 for why they must stay swappable without invalidating annotations:
    # they are never baked into a raw-space mask, only applied at export time.
    wheel_unwrap_version: int = 1
    wheel_class_name: str = "wheel"
    wheel_unwrap_radius_padding_pct: float = 5.0
    wheel_unwrap_output_width: int = 512
    wheel_unwrap_output_height: int = 128
    wheel_unwrap_log_scale: bool = True

    # Object store for publishing dataset snapshots (M2). Points at the
    # `automation-minio-1` service in docker-compose by default. Publishing is
    # opt-in per export and never fails an export - the local snapshot is the
    # source of truth, the bucket is the staging copy the pipeline team pulls
    # from (build plan Q-C: we stage, they import into their own MLflow).
    # boto3 talks S3, so pointing this at real S3 later is a config change.
    s3_endpoint_url: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "vb-dataset-snapshots"
    s3_prefix: str = "snapshots"
    s3_region: str = "us-east-1"

    # Golden eval set storage (M4). A **separate bucket** from s3_bucket above,
    # not a prefix under it - D-Q4 / annotation_module_build_plan.md Q-C are
    # explicit that this needs to be structurally separate storage, not a flag
    # on shared storage, because the only mitigation that counts against
    # contamination is that no propagation, triage, or export path can write
    # here. Same endpoint/credentials as the snapshot store (same MinIO
    # instance) - it's the bucket boundary that matters, not a separate
    # connection.
    golden_s3_bucket: str = "vb-golden-eval-set"
    golden_s3_prefix: str = "golden"

    # MLflow tracking (M5, Scope A only - the in-tool SAM2/detector
    # pre-labeler helper, never the pipeline team's own 8 production
    # families; Q-C: we stage data for them, we never write into their
    # MLflow). Empty by default so a deployment that never sets this up
    # doesn't need the mlflow-skinny dependency to be reachable - training
    # still runs, it just isn't tracked. See detector_service._run_training.
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "annot-detector-training"
    # Auto-kick a Scope A training run whenever a *genuinely new* snapshot
    # finalizes (export_service._finalize_snapshot) - a re-export that
    # resolves to an already-existing snapshot (M2's content-addressing)
    # does not retrigger, since nothing about the data actually changed.
    # Runs in the background; never blocks or fails the export response if
    # training can't start (see the try/except around it).
    auto_train_on_handoff: bool = True

    # M8 GPU-scheduling guard: "inference always wins" - see
    # gpu_scheduler.py. Training waits (polling every gpu_wait_poll_seconds)
    # for SAM2 to go idle before its GPU-heavy model.train() call starts,
    # rather than barging in on live annotators. gpu_wait_max_seconds bounds
    # that wait - past it, the job is marked "skipped" (not "failed": no
    # data or code was wrong, the GPU just stayed busy) rather than parking
    # a background thread indefinitely. This is what makes
    # auto_train_on_handoff safe to enable in a live multi-annotator
    # deployment - without it, almost any handoff could trigger a real,
    # long-running training run competing with SAM2 on the same GPU.
    gpu_wait_max_seconds: int = 600
    gpu_wait_poll_seconds: int = 5

    # M6: score a just-trained detector against the golden set (if one
    # exists for the view) and log the result onto the same MLflow run -
    # per-class, never aggregate-only, see golden_eval_service.py. Skips
    # cleanly (not an error) for a view with no golden set yet. Best-effort:
    # an eval failure never turns a successful training run into a
    # reported failure.
    auto_eval_on_golden_set: bool = True

    # M7.5 promotion sync: a background poller checks MLflow's current
    # Production version against what's already been proposed, on this
    # interval - detection only, see model_promotion_service.py. Never
    # touches what's actually live; that always needs an explicit
    # model_reviewer approval. Safe to leave enabled even with no MLflow
    # configured - the check degrades to a no-op (logged at debug, not an
    # error) rather than failing.
    model_promotion_poll_enabled: bool = True
    model_promotion_poll_interval_seconds: int = 1800

    # Class-incremental promotion plan, Module 2: class eligibility
    # (docs/mlflow_class_incremental_architecture.md §E). Per-tier floors a
    # class must clear before class_eligibility_service ever moves it out
    # of discovered/collecting_data into eligible (and therefore into
    # _assemble_dataset's training set, Module 3). These are STARTING
    # POINTS, not calibrated numbers - the doc's own §E instruction is to
    # run a learning-curve probe (per-class AP vs. N) once real data for a
    # given class exists, and adjust from measured evidence, never loosen
    # without it.
    eligibility_min_instances_safety: int = 150
    eligibility_min_instances_structural: int = 100
    eligibility_min_instances_cosmetic: int = 80
    eligibility_min_images_safety: int = 60
    eligibility_min_images_structural: int = 50
    eligibility_min_images_cosmetic: int = 40
    eligibility_min_coach_types_safety: int = 2
    eligibility_min_coach_types_structural: int = 1
    eligibility_min_coach_types_cosmetic: int = 1
    eligibility_min_val_instances: int = 15
    # The old "8% of the dataset" idea - kept only as a cheap secondary
    # signal (worth checking absolute floors now), never the actual gate.
    # Not read by class_eligibility_service's determine_state() at all
    # today; reserved for a future dashboard/early-warning surface.
    eligibility_relative_share_trigger: float = 0.08
    eligibility_poll_enabled: bool = True
    eligibility_poll_interval_seconds: int = 1800

    # Class-incremental promotion plan, Module 4: the compare()/
    # should_promote() thresholds (docs/mlflow_class_incremental_
    # architecture.md §H/§I). Every value below is a STARTING POINT, not a
    # calibrated number - per the doc's own §E instruction, these should
    # come from measured run-to-run noise floor (regression tolerance) and
    # real learning-curve data (new-class floor), not stay as guesses.
    #
    # SCALE: fractional 0-1, matching Ultralytics' own mAP50 output
    # (golden_eval_service reports e.g. 0.671, never "67.1") - NOT the
    # doc's human-readable "91% AP" percentage language. Mixing scales here
    # would make Layer 2 reject every class unconditionally and make
    # Layer 1's tolerance nearly meaningless against real deltas - verified
    # this distinction explicitly before wiring real data through it.
    regression_tolerance_ap50_safety: float = 0.005
    regression_tolerance_ap50_structural: float = 0.02
    regression_tolerance_ap50_cosmetic: float = 0.03
    new_class_floor_ap50_safety: float = 0.85
    new_class_floor_ap50_structural: float = 0.70
    new_class_floor_ap50_cosmetic: float = 0.55
    # PLACEHOLDER - yolo11s-seg.pt inference is typically well under this;
    # replace with a real measured p95 + margin before trusting this gate.
    max_latency_p95_ms: float = 500.0
    # PLACEHOLDER - yolo11s-seg.pt is ~20MB; generous headroom until a real
    # deployment-size budget exists.
    max_model_size_mb: float = 200.0
    max_false_positive_rate_safety: float = 0.02
    max_false_positive_rate_structural: float = 0.05
    max_false_positive_rate_cosmetic: float = 0.10

    # Module 8: Ultralytics' own built-in inverse-class-frequency loss
    # weighting (DetectionTrainer.set_class_weights, verified against the
    # installed version's source) - exponent applied to inverse class
    # frequency, range [0, 1], 0 disables it entirely (Ultralytics' own
    # default). 0.5 is a deliberately dampened starting point - full (1.0)
    # inverse-frequency weighting can overcorrect when a brand-new class
    # is still numerically tiny, swinging the loss too far the other way.
    class_weight_power: float = 0.5

    # Logging
    log_level: str = "INFO"
    log_file: str = "../logs/backend.log"

    # Postgres (annotation state / audit history - see app/db.py)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "annotator"
    postgres_password: str = "change-me"
    postgres_db: str = "annotator"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def dataset_dir(self) -> Path:
        return Path(self.dataset_path).resolve()

    @property
    def exports_dir(self) -> Path:
        return Path(self.exports_path).resolve()

    @property
    def models_dir(self) -> Path:
        return Path(self.models_path).resolve()

    @property
    def state_dir(self) -> Path:
        return self.dataset_dir / self.state_dir_name

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def exclude_class_list(self) -> list[str]:
        return [c.strip() for c in self.exclude_classes.split(",") if c.strip()]

    def is_excluded_class(self, name: str) -> bool:
        return name.strip().lower() in {c.lower() for c in self.exclude_class_list}


@lru_cache
def get_settings() -> Settings:
    return Settings()
