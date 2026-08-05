"""Model registry + promotion recommendation (M7, recommendation-only slice).

**Deliberately does not auto-promote anything.** The build plan's own
recommendation for this milestone is blunt: no auto-approve for any class
in the first iteration, regardless of tier - this is where a wrong design
silently promotes a bad model. What this module does is register each
successfully-trained candidate as a version of `AnnotDetector-<view>` in
MLflow's own Model Registry, compute whether it matches-or-beats whatever
is currently `Production` on *every* class the golden set covers, and tag
that verdict onto the version - a human still moves the stage themselves,
through MLflow's own UI, with a real account. This tool's identity system
is a name, not a login (see app/routers/annotator.py) - not something to
pretend is an approval record for a decision this consequential.

**Recommendation-only, not live-serving, in this pass.**
`detector_service.detect()` keeps reading the active model from
`registry.json` exactly as before - a promotion decision made in MLflow's
UI has no effect on what annotators are served yet. Making promotion
actually swap the live model is a deliberate, separate follow-up once this
recommendation logic has been trusted for a while, not bundled in here.

**Comparison is against stored metrics, not a re-run.** The candidate's
golden-set scores were just computed fresh (M6). Rather than re-downloading
and re-evaluating whatever is currently in Production - extra complexity
for a number a human is going to look at anyway - this reads the
*stored* `golden/classN_mAP50` metrics from the run that originally
produced the current Production version. A curator who suspects those are
stale (the golden set can grow over time) can always re-run eval by hand;
this is a recommendation, not a gate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

REGISTERED_MODEL_PREFIX = "AnnotDetector"


@dataclass
class PromotionRecommendation:
    verdict: str  # "eligible_no_baseline" | "eligible" | "regressed"
    regressed_classes: list[int] = field(default_factory=list)
    compared_against_version: Optional[str] = None


def registered_model_name(slug: str) -> str:
    return f"{REGISTERED_MODEL_PREFIX}-{slug}"


def _compare_against_production(
    client, name: str, candidate_per_class: dict[int, dict[str, float]]
) -> PromotionRecommendation:
    """Never raises - a comparison failure degrades to "no baseline" rather
    than blocking registration, since this is advisory only."""
    try:
        production = client.get_latest_versions(name, stages=["Production"])
    except Exception:
        logger.exception("Could not look up current Production version for %s; treating as no baseline", name)
        return PromotionRecommendation(verdict="eligible_no_baseline")

    if not production:
        return PromotionRecommendation(verdict="eligible_no_baseline")

    prod_version = production[0]
    try:
        prod_run = client.get_run(prod_version.run_id)
        prod_metrics = prod_run.data.metrics
    except Exception:
        logger.exception(
            "Could not read metrics for Production version %s of %s; treating as no baseline",
            prod_version.version, name,
        )
        return PromotionRecommendation(verdict="eligible_no_baseline", compared_against_version=prod_version.version)

    regressed: list[int] = []
    for class_id, metrics in candidate_per_class.items():
        prod_key = f"golden/class{class_id}_mAP50"
        if prod_key not in prod_metrics:
            continue  # no baseline for this specific class - can't have regressed
        if metrics["mAP50"] < prod_metrics[prod_key]:
            regressed.append(class_id)

    verdict = "regressed" if regressed else "eligible"
    return PromotionRecommendation(
        verdict=verdict, regressed_classes=sorted(regressed), compared_against_version=prod_version.version
    )


def register_and_recommend(
    dataset_slug: str, run_id: str, golden_per_class: dict[int, dict[str, float]]
) -> Optional[PromotionRecommendation]:
    """Register this run's model as a new version of `AnnotDetector-<slug>`,
    compute a promotion recommendation against whatever is currently in
    Production for this view, and tag both onto the new version.

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

        recommendation = _compare_against_production(client, name, golden_per_class)
        client.set_model_version_tag(name, model_version.version, "promotion_recommendation", recommendation.verdict)
        if recommendation.regressed_classes:
            client.set_model_version_tag(
                name, model_version.version, "regressed_classes", ",".join(map(str, recommendation.regressed_classes))
            )
        if recommendation.compared_against_version:
            client.set_model_version_tag(
                name, model_version.version, "compared_against_version", recommendation.compared_against_version
            )

        logger.info(
            "Registered %s v%s - promotion recommendation: %s%s",
            name, model_version.version, recommendation.verdict,
            f" (regressed: {recommendation.regressed_classes})" if recommendation.regressed_classes else "",
        )
        return recommendation
    except Exception:
        logger.exception("Model registration/recommendation failed; training result is unaffected")
        return None
