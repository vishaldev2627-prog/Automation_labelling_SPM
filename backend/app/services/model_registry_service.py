"""Model registry + promotion decision (M7 + Module 5 of the
class-incremental promotion plan - docs/mlflow_class_incremental_
architecture.md §G).

**Deliberately does not auto-promote anything.** The build plan's own
recommendation for this milestone is blunt: no auto-approve for any class
in the first iteration, regardless of tier - this is where a wrong design
silently promotes a bad model. What this module does is register each
successfully-trained candidate as a version of `AnnotDetector-<view>` in
MLflow's own Model Registry, run it through `promotion_gate.compare()`/
`should_promote()` against whatever is currently `Production`, and tag
the full decision onto the version - a human still moves the stage
themselves, through MLflow's own UI, with a real account. This tool's
identity system is a name, not a login (see app/routers/annotator.py) -
not something to pretend is an approval record for a decision this
consequential.

**Recommendation-only, not live-serving, in this pass.**
`detector_service.detect()` keeps reading the active model from
`registry.json` exactly as before - a promotion decision made in MLflow's
UI has no effect on what annotators are served yet. Making promotion
actually swap the live model is `model_promotion_service.py`'s job
(M7.5 + Module 6), which reads the tags this module writes and enforces
the ones that matter (REJECT blocks by default, a safety-tier hard_fail
blocks unconditionally).

**Comparison is against stored metrics, not a re-run.** The candidate's
golden-set scores were just computed fresh (M6). Rather than re-downloading
and re-evaluating whatever is currently in Production - extra complexity
for a number a human is going to look at anyway - this reads the *stored*
`golden/classN_*` metrics from the run that originally produced the
current Production version. A curator who suspects those are stale (the
golden set can grow over time) can always re-run eval by hand; this is a
recommendation, not a re-verified guarantee.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Set, Tuple

from app.config import get_settings
from app.services.promotion_gate import (
    ClassMetrics,
    ClassRegistryView,
    ModelVersionMetrics,
    OperationalMetrics,
    PromotionDecision,
    PromotionThresholds,
    UnexplainedClassRemoval,
    compare,
    should_promote,
)

logger = logging.getLogger(__name__)

REGISTERED_MODEL_PREFIX = "AnnotDetector"

# The metric keys detector_service._run_training logs per class, and the
# ClassMetrics field each one maps to - see golden_eval_service.py's
# per-class dict, which this list must stay in lockstep with.
_METRIC_KEY_TO_FIELD = {
    "precision": "precision",
    "recall": "recall",
    "mAP50": "ap50",
    "mAP50_95": "ap50_95",
    "n_val_instances": "n_val_instances",
}


def registered_model_name(slug: str) -> str:
    return f"{REGISTERED_MODEL_PREFIX}-{slug}"


def _candidate_metrics(golden_per_class: Dict[int, Dict[str, float]]) -> ModelVersionMetrics:
    per_class = {}
    for class_id, m in golden_per_class.items():
        kwargs = {"class_id": class_id}
        for metric_key, field_name in _METRIC_KEY_TO_FIELD.items():
            if metric_key in m:
                kwargs[field_name] = m[metric_key]
        per_class[class_id] = ClassMetrics(**kwargs)
    return ModelVersionMetrics(per_class=per_class)


def _production_metrics_from_run(prod_metrics: dict, candidate_class_ids: Set[int]) -> ModelVersionMetrics:
    """Reads back the same golden/class{N}_* metric keys detector_service
    logs, for whatever classes the PRODUCTION run happens to have logged -
    not just the candidate's classes, since compare() needs production's
    full class set to detect an unexplained removal."""
    per_class: dict = {}
    seen_class_ids: Set[int] = set()
    for key in prod_metrics:
        if not key.startswith("golden/class"):
            continue
        rest = key[len("golden/class"):]
        class_id_str, _, metric_key = rest.partition("_")
        try:
            class_id = int(class_id_str)
        except ValueError:
            continue
        seen_class_ids.add(class_id)

    for class_id in seen_class_ids | candidate_class_ids:
        kwargs = {"class_id": class_id}
        for metric_key, field_name in _METRIC_KEY_TO_FIELD.items():
            prod_key = f"golden/class{class_id}_{metric_key}"
            if prod_key in prod_metrics:
                kwargs[field_name] = prod_metrics[prod_key]
        if len(kwargs) > 1:  # at least one real metric was found for this class
            per_class[class_id] = ClassMetrics(**kwargs)
    return ModelVersionMetrics(per_class=per_class)


def _decide(
    client,
    name: str,
    golden_per_class: Dict[int, Dict[str, float]],
    class_infos,
    model_size_mb: float,
    latency_p95_ms: float,
    thresholds: Optional[PromotionThresholds] = None,
) -> Tuple[PromotionDecision, Optional[str]]:
    """Never raises - a comparison failure degrades to a REJECT with a
    clear reason rather than blocking registration, since this is advisory
    only (register_and_recommend's own contract). Returns
    (decision, compared_against_version).

    `thresholds` defaults to real config - injectable so tests don't have
    to depend on (or mutate) global Settings to exercise specific
    boundaries."""
    registry = ClassRegistryView.from_class_infos(class_infos)
    thresholds = thresholds or PromotionThresholds.from_settings(get_settings())
    candidate = _candidate_metrics(golden_per_class)
    operational = OperationalMetrics(latency_p95_ms=latency_p95_ms, model_size_mb=model_size_mb)

    try:
        production_versions = client.get_latest_versions(name, stages=["Production"])
    except Exception:
        logger.exception("Could not look up current Production version for %s; treating as no baseline", name)
        production_versions = []

    production = None
    compared_against_version = None
    if production_versions:
        prod_version = production_versions[0]
        compared_against_version = prod_version.version
        try:
            prod_run = client.get_run(prod_version.run_id)
            production = _production_metrics_from_run(prod_run.data.metrics, set(candidate.per_class.keys()))
        except Exception:
            logger.exception(
                "Could not read metrics for Production version %s of %s; treating as no baseline",
                prod_version.version, name,
            )

    try:
        comparison = compare(candidate, production, registry, thresholds)
    except UnexplainedClassRemoval as exc:
        decision = PromotionDecision(
            decision="REJECT",
            reasons=[f"HARD FAIL: {exc}"],
            hard_fail=True,
        )
        return decision, compared_against_version

    decision = should_promote(comparison, operational, registry, thresholds)
    return decision, compared_against_version


def register_and_recommend(
    dataset_slug: str,
    run_id: str,
    golden_per_class: Dict[int, Dict[str, float]],
    class_infos,
    model_size_mb: float = 0.0,
    latency_p95_ms: float = 0.0,
    thresholds: Optional[PromotionThresholds] = None,
    dataset_snapshot_id: Optional[str] = None,
    class_map_version: Optional[int] = None,
) -> Optional[PromotionDecision]:
    """Register this run's model as a new version of `AnnotDetector-<slug>`,
    run it through the promotion gate against whatever is currently in
    Production for this view, and tag the full decision onto the version.

    Returns None (logged, not raised) on any registry failure - registering
    is advisory, not a prerequisite for training to be considered
    successful. The caller (`detector_service`) must not let a registry
    problem affect the training job's own reported outcome.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        name = registered_model_name(dataset_slug)
        try:
            client.create_registered_model(name)
        except mlflow.exceptions.MlflowException:
            pass  # already exists - fine, this is idempotent by design

        model_version = client.create_model_version(name=name, source=f"runs:/{run_id}/weights", run_id=run_id)

        decision, compared_against_version = _decide(
            client, name, golden_per_class, class_infos, model_size_mb, latency_p95_ms, thresholds
        )

        regressed_classes = sorted(
            c for c, r in (decision.comparison.common_classes.items() if decision.comparison else []) if r["regressed"]
        )
        new_classes = sorted(decision.comparison.new_classes.keys()) if decision.comparison else []
        removed_classes = decision.comparison.removed_classes if decision.comparison else []

        # Legacy tags (M7) - kept verbatim for model_promotion_service's
        # current tag-reading code, which this module (Module 5) does not
        # change; Module 6 migrates that reader to the richer tags below.
        legacy_verdict = "eligible_no_baseline" if compared_against_version is None else (
            "regressed" if regressed_classes else "eligible"
        )
        client.set_model_version_tag(name, model_version.version, "promotion_recommendation", legacy_verdict)
        if regressed_classes:
            client.set_model_version_tag(
                name, model_version.version, "regressed_classes", ",".join(map(str, regressed_classes))
            )
        if compared_against_version:
            client.set_model_version_tag(
                name, model_version.version, "compared_against_version", compared_against_version
            )

        # §G tag table - the actual decision, in a form a curator can read
        # without opening raw per-class metrics.
        client.set_model_version_tag(name, model_version.version, "decision", decision.decision)
        client.set_model_version_tag(name, model_version.version, "hard_fail", str(decision.hard_fail).lower())
        if compared_against_version:
            client.set_model_version_tag(name, model_version.version, "baseline_version", compared_against_version)
        class_set_diff_parts = [f"+{c}" for c in new_classes] + [f"-{c}" for c in removed_classes]
        client.set_model_version_tag(
            name, model_version.version, "class_set_diff", ",".join(class_set_diff_parts) or "none"
        )
        if new_classes:
            client.set_model_version_tag(name, model_version.version, "new_classes", ",".join(map(str, new_classes)))
        if removed_classes:
            client.set_model_version_tag(
                name, model_version.version, "removed_classes", ",".join(map(str, removed_classes))
            )
        common_summary = (
            ",".join(f"{c}:{decision.comparison.common_classes[c]['delta_ap50']:+.1f}" for c in regressed_classes)
            if regressed_classes else "none"
        )
        client.set_model_version_tag(name, model_version.version, "common_class_regression", common_summary)
        if decision.comparison:
            new_floor_summary = ",".join(
                f"{c}:{'pass' if r['meets_floor'] else 'fail'}"
                f"({r['candidate'].ap50:.1f}{'>=' if r['meets_floor'] else '<'}{r['floor_used']:.1f})"
                for c, r in sorted(decision.comparison.new_classes.items())
            ) or "none"
            client.set_model_version_tag(name, model_version.version, "new_class_floor_result", new_floor_summary)
        operational_summary = ",".join(
            r for r in decision.reasons if any(k in r for k in ("latency", "model size", "false-positive"))
        ) or "pass"
        client.set_model_version_tag(name, model_version.version, "operational_result", operational_summary)

        # Module 9: already logged onto the training RUN (detector_service's
        # own params dict) - also on the model VERSION here, since a
        # promotion decision references the version, not the run, and a
        # curator reading tags shouldn't have to cross-reference back to
        # the run to answer "what data/class-map produced this."
        if dataset_snapshot_id:
            client.set_model_version_tag(name, model_version.version, "dataset_snapshot_id", dataset_snapshot_id)
        if class_map_version is not None:
            client.set_model_version_tag(name, model_version.version, "class_map_version", str(class_map_version))

        # Additive: aliases alongside the existing stage-based logic - keep
        # reading `Production` stage for backward compatibility with the
        # current M7.5 poller, but give a stage-independent query surface
        # too, per the doc's own recommendation.
        try:
            client.set_registered_model_alias(name, "candidate", model_version.version)
        except Exception:
            logger.debug("Could not set @candidate alias for %s v%s (older MLflow?)", name, model_version.version)

        logger.info(
            "Registered %s v%s - decision: %s%s",
            name, model_version.version, decision.decision,
            f" (reasons: {decision.reasons})" if decision.reasons else "",
        )
        return decision
    except Exception:
        logger.exception("Model registration/recommendation failed; training result is unaffected")
        return None
