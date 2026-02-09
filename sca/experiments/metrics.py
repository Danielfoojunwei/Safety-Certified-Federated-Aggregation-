"""Evaluation metrics for experiments (Section 7.3).

Safety metrics:
- Policy violation rate (Viol(M) estimate)
- Jailbreak success rate
- Refusal correctness by category

Utility metrics:
- Task accuracy / perplexity / instruction-following scores

Theory-facing metrics:
- Empirical bound tightness: sum_j w_j UCB_j vs observed violation
- Sample efficiency: tests required to detect regressions at fixed (epsilon, delta)

Interpretability metrics:
- Size of Delta_G (regression subgraph)
- Minimal counterexamples per regression cluster
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from sca.certificate.certificate import SafetyCertificate
from sca.utils.stats import RegionStats


@dataclass
class SafetyMetrics:
    """Aggregated safety evaluation metrics."""
    violation_rate: float = 0.0
    jailbreak_success_rate: float = 0.0
    refusal_accuracy: float = 0.0
    n_total_tests: int = 0
    n_violations: int = 0
    per_category_violations: dict[str, int] = field(default_factory=dict)


@dataclass
class BoundTightnessMetrics:
    """Theory-facing metrics for bound tightness analysis."""
    certified_bound: float = 0.0  # sum_j w_j UCB_j
    observed_violation: float = 0.0  # empirical Viol(M)
    gap: float = 0.0  # certified_bound - observed_violation
    relative_gap: float = 0.0  # gap / certified_bound


@dataclass
class EfficiencyMetrics:
    """Sample efficiency metrics."""
    total_queries: int = 0
    queries_to_first_detection: int | None = None
    detection_rate_at_budget: float = 0.0
    queries_per_region: dict[int, int] = field(default_factory=dict)


def compute_violation_rate(
    model_fn: Callable[[dict], str],
    test_interactions: list[dict],
    safety_predicate,
) -> SafetyMetrics:
    """Estimate the true violation rate Viol(M) on a test set.

    Args:
        model_fn: Model inference callable.
        test_interactions: Test interactions sampled from D.
        safety_predicate: Safety predicate phi.

    Returns:
        SafetyMetrics with the estimated violation rate.
    """
    n_violations = 0
    per_category = {}

    for interaction in test_interactions:
        output = model_fn(interaction)
        evaluation = safety_predicate.evaluate(interaction, output)
        if not evaluation.is_safe:
            n_violations += 1
            cat = evaluation.category or "unknown"
            per_category[cat] = per_category.get(cat, 0) + 1

    n_total = len(test_interactions)
    rate = n_violations / n_total if n_total > 0 else 0.0

    return SafetyMetrics(
        violation_rate=rate,
        n_total_tests=n_total,
        n_violations=n_violations,
        per_category_violations=per_category,
    )


def compute_bound_tightness(
    certificate: SafetyCertificate,
    observed_violation_rate: float,
) -> BoundTightnessMetrics:
    """Compare the certified bound to the observed violation rate.

    Args:
        certificate: Safety certificate with the certified bound.
        observed_violation_rate: Empirically estimated Viol(M).

    Returns:
        BoundTightnessMetrics.
    """
    gap = certificate.bound_value - observed_violation_rate
    relative_gap = gap / certificate.bound_value if certificate.bound_value > 0 else 0.0

    return BoundTightnessMetrics(
        certified_bound=certificate.bound_value,
        observed_violation=observed_violation_rate,
        gap=gap,
        relative_gap=relative_gap,
    )


def compute_efficiency_metrics(
    region_stats: list[RegionStats],
    total_budget: int,
) -> EfficiencyMetrics:
    """Compute sample efficiency metrics.

    Args:
        region_stats: Per-region statistics from verification.
        total_budget: Total verification budget used.

    Returns:
        EfficiencyMetrics.
    """
    total_queries = sum(rs.n_samples for rs in region_stats)
    queries_per_region = {rs.region_id: rs.n_samples for rs in region_stats}

    # Detection rate: fraction of bad regions that were sampled
    bad_regions = [rs for rs in region_stats if rs.n_violations > 0]
    detection_rate = len(bad_regions) / len(region_stats) if region_stats else 0.0

    return EfficiencyMetrics(
        total_queries=total_queries,
        detection_rate_at_budget=detection_rate,
        queries_per_region=queries_per_region,
    )


def compute_regression_size(
    prev_stats: list[RegionStats],
    curr_stats: list[RegionStats],
    delta: float,
    eta: float = 0.01,
) -> int:
    """Compute the size of the regression subgraph |Delta_G|.

    Args:
        prev_stats: Region stats from the previous model.
        curr_stats: Region stats from the candidate model.
        delta: Confidence parameter.
        eta: Regression threshold.

    Returns:
        Number of regions in the regression subgraph.
    """
    k = max(len(prev_stats), len(curr_stats))
    if k == 0:
        return 0

    prev_map = {rs.region_id: rs for rs in prev_stats}
    count = 0
    for rs_new in curr_stats:
        rs_old = prev_map.get(rs_new.region_id)
        ucb_new = rs_new.ucb(delta, k)
        ucb_old = rs_old.ucb(delta, k) if rs_old else 0.0
        if ucb_new - ucb_old >= eta:
            count += 1
    return count


def aggregate_round_metrics(
    round_results: list[dict],
) -> dict[str, Any]:
    """Aggregate metrics across FL rounds for reporting.

    Args:
        round_results: List of per-round metric dictionaries.

    Returns:
        Aggregated summary statistics.
    """
    if not round_results:
        return {}

    n_rounds = len(round_results)
    n_accepted = sum(1 for r in round_results if r.get("accepted", False))

    violation_rates = [r.get("violation_rate", 0) for r in round_results]
    bounds = [r.get("certified_bound", 0) for r in round_results]

    return {
        "n_rounds": n_rounds,
        "acceptance_rate": n_accepted / n_rounds,
        "mean_violation_rate": float(np.mean(violation_rates)),
        "max_violation_rate": float(np.max(violation_rates)),
        "mean_certified_bound": float(np.mean(bounds)) if bounds else 0.0,
        "mean_bound_gap": float(np.mean(
            [b - v for b, v in zip(bounds, violation_rates)]
        )) if bounds else 0.0,
    }
