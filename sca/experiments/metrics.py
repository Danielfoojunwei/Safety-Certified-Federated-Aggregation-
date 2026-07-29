"""Evaluation metrics and the honest-statistics layer.

Audit findings addressed here:

* **F14 (statistics).**  ``paired_significance_test`` used to fall back to a
  fabricated ``tanh``-based "p-value" when scipy was absent, and was called
  with ``n = 5`` seeds, where the smallest attainable two-sided p under any
  exact paired resampling scheme is ``2 / 2**5 = 0.0625`` -- so its own
  ``p < 0.05`` criterion was unreachable.  It is now a **paired sign-flip
  permutation test**, exact by full enumeration when ``2**n`` is small enough
  and Monte-Carlo otherwise, needing no scipy.  :data:`MIN_SEEDS_FOR_ALPHA_05`
  records the smallest ``n`` at which ``p < 0.05`` is even attainable, and
  :func:`min_attainable_p` lets a test assert it.
* **F14 (parameter count).**  :func:`count_parameters` counts programmatically.
  Nothing in this repo may hardcode a parameter count.
* **C5.**  :func:`bootstrap_ci` is the standard reporting path; every headline
  metric is reported as ``mean [lo, hi]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Sequence

import numpy as np

from sca.certificate.certificate import SafetyCertificate
from sca.utils.seeding import stable_rng
from sca.utils.stats import RegionStat, compute_ucb

# ---------------------------------------------------------------------------
# Metric containers
# ---------------------------------------------------------------------------


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
    observed_violation: float = 0.0  # empirical Viol(M) on a held-out set
    gap: float = 0.0
    relative_gap: float = 0.0


@dataclass
class EfficiencyMetrics:
    """Sample efficiency metrics."""

    total_queries: int = 0
    queries_to_first_detection: int | None = None
    detection_rate_at_budget: float = 0.0
    queries_per_region: dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Basic safety metrics
# ---------------------------------------------------------------------------


def compute_violation_rate(
    model_fn: Callable[[dict], str],
    test_interactions: list[dict],
    safety_predicate,
) -> SafetyMetrics:
    """Estimate the true violation rate ``Viol(M)`` on a set of interactions.

    The caller is responsible for the provenance of ``test_interactions``.  For
    a *reported* number they must come from the held-out test split; see
    :func:`sca.experiments.evaluation.evaluate`, which enforces that.
    """
    n_violations = 0
    per_category: dict[str, int] = {}

    for interaction in test_interactions:
        output = model_fn(interaction)
        evaluation = safety_predicate.evaluate(interaction, output)
        if not evaluation.is_safe:
            n_violations += 1
            cat = evaluation.category or "unknown"
            per_category[cat] = per_category.get(cat, 0) + 1

    n_total = len(test_interactions)
    return SafetyMetrics(
        violation_rate=n_violations / n_total if n_total else 0.0,
        n_total_tests=n_total,
        n_violations=n_violations,
        per_category_violations=per_category,
    )


def compute_bound_tightness(
    certificate: SafetyCertificate,
    observed_violation_rate: float,
) -> BoundTightnessMetrics:
    """Compare a certified bound to an independently observed violation rate."""
    gap = certificate.bound_value - observed_violation_rate
    return BoundTightnessMetrics(
        certified_bound=certificate.bound_value,
        observed_violation=observed_violation_rate,
        gap=gap,
        relative_gap=gap / certificate.bound_value
        if certificate.bound_value > 0 else 0.0,
    )


def compute_efficiency_metrics(
    region_stats: Sequence[RegionStat],
    total_budget: int,
) -> EfficiencyMetrics:
    """Sample-efficiency summary over a list of :class:`RegionStat`."""
    total_queries = sum(rs.n_samples for rs in region_stats)
    queries_per_region = {rs.region_id: rs.n_samples for rs in region_stats}
    bad = [rs for rs in region_stats if rs.n_violations > 0]
    return EfficiencyMetrics(
        total_queries=total_queries,
        detection_rate_at_budget=len(bad) / len(region_stats) if region_stats else 0.0,
        queries_per_region=queries_per_region,
    )


def compute_regression_size(
    prev_stats: Sequence[RegionStat],
    curr_stats: Sequence[RegionStat],
    delta: float,
    eta: float = 0.01,
    *,
    k_total: int | None = None,
    budget_cap: int = 1,
) -> int:
    """Size of the regression subgraph ``|Delta_G|``.

    ``k_total`` and ``budget_cap`` are the a-priori union-bound parameters and
    should be passed explicitly (finding F10: deriving ``K`` from the data is
    what let a degenerate run receive the smallest penalty).  When ``k_total``
    is omitted it falls back to ``max(len(prev), len(curr))`` and this function
    emits no certificate -- it is a descriptive statistic only.

    Note the caveat from finding **F11**: a difference of UCBs is dominated by
    the difference of Hoeffding widths, so a region the baseline barely sampled
    can have ``ucb_old = 1`` and never be flagged however badly it degrades.
    That asymmetry is the MKG owner's to fix; this function only reports.
    """
    k = k_total if k_total is not None else max(len(prev_stats), len(curr_stats))
    if k == 0:
        return 0
    cap = max(int(budget_cap), 1)
    prev_map = {rs.region_id: rs for rs in prev_stats}
    count = 0
    for new in curr_stats:
        old = prev_map.get(new.region_id)
        ucb_new = compute_ucb(new.n_violations, new.n_samples, k, cap, delta)
        ucb_old = (
            compute_ucb(old.n_violations, old.n_samples, k, cap, delta)
            if old is not None else 0.0
        )
        if ucb_new - ucb_old >= eta:
            count += 1
    return count


def aggregate_round_metrics(round_results: list[dict]) -> dict[str, Any]:
    """Aggregate per-round FL metrics for reporting."""
    if not round_results:
        return {}
    n_rounds = len(round_results)
    n_accepted = sum(1 for r in round_results if r.get("accepted", False))
    violation_rates = [r.get("violation_rate", 0.0) for r in round_results]
    bounds = [r.get("certified_bound", 0.0) for r in round_results]
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


# ---------------------------------------------------------------------------
# Parameter counting (F14: README claimed 3.3M for a 6,466,690-parameter model)
# ---------------------------------------------------------------------------


def count_parameters(model) -> dict[str, int]:
    """Count a torch module's parameters programmatically.

    Returns ``{"total", "trainable", "frozen", "buffers"}``.  Nothing in this
    repository may hardcode a parameter count; every reported figure must come
    from this function so it cannot drift from the model.
    """
    params = list(model.parameters())
    total = sum(p.numel() for p in params)
    trainable = sum(p.numel() for p in params if p.requires_grad)
    buffers = sum(b.numel() for b in getattr(model, "buffers", lambda: [])())
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
        "buffers": int(buffers),
    }


def format_parameter_count(model) -> str:
    c = count_parameters(model)
    return f"{c['total']:,} parameters ({c['trainable']:,} trainable)"


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals -- the standard reporting path (C5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapCI:
    """A bootstrap percentile confidence interval.

    ``point`` is the statistic on the observed sample (not the bootstrap mean),
    which is what should be printed as the headline number.
    """

    point: float
    lo: float
    hi: float
    n: int
    n_boot: int
    alpha: float

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    def as_dict(self) -> dict[str, float | int]:
        return {"point": self.point, "ci_lo": self.lo, "ci_hi": self.hi,
                "n": self.n, "n_boot": self.n_boot, "alpha": self.alpha}


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for ``statistic`` (default: the mean).

    Args:
        values: Observed values, one per seed / replicate.
        n_boot: Number of bootstrap resamples.
        alpha: ``1 - alpha`` is the coverage, so ``alpha=0.05`` -> 95% CI.
        statistic: Applied to each resample.
        seed: Deterministic across processes via
            :func:`sca.utils.seeding.stable_rng` -- no ``PYTHONHASHSEED``
            dependence (F9).

    With ``n <= 1`` the interval degenerates to the point estimate; that is
    reported honestly rather than papered over with a normal approximation.
    """
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), 0, 0, alpha)
    point = float(statistic(arr))
    if n == 1:
        return BootstrapCI(point, point, point, 1, 0, alpha)
    rng = stable_rng(seed, "bootstrap_ci", str(n), str(n_boot))
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.apply_along_axis(statistic, 1, arr[idx]) if statistic is not np.mean \
        else arr[idx].mean(axis=1)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return BootstrapCI(point, lo, hi, n, n_boot, alpha)


def paired_bootstrap_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    *,
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap CI on the **paired** effect ``mean(a - b)``.

    Resamples seed indices, not the two arms independently, so the pairing is
    preserved.
    """
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    return bootstrap_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed)


# ---------------------------------------------------------------------------
# Paired permutation test (replaces the fabricated tanh p-value; F14)
# ---------------------------------------------------------------------------

#: Largest ``n`` for which the sign-flip null is enumerated exactly.
#: ``2**18 = 262144`` sign vectors, which is a few hundred ms in numpy.
EXACT_PERMUTATION_MAX_N = 18

#: Smallest number of paired seeds at which an exact two-sided sign-flip test
#: can even return ``p < 0.05``: ``2 / 2**n < 0.05`` first holds at ``n = 6``.
MIN_SEEDS_FOR_ALPHA_05 = 6

#: What the rebuilt experiments actually use (criterion C5).
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]


def min_attainable_p(n: int) -> float:
    """Smallest two-sided p-value an exact sign-flip test on ``n`` pairs can give.

    Under the sign-flip null there are ``2**n`` equally likely sign vectors.
    For any non-degenerate sample the maximal ``|mean|`` is attained by at
    least the identity assignment and its global flip, so ``p >= 2 / 2**n``.

    This is the function that makes finding F14 non-recurrable: a test asserts
    ``min_attainable_p(n_seeds) < alpha`` for the ``n_seeds`` actually used.
    """
    if n <= 0:
        return 1.0
    return min(1.0, 2.0 / (2.0 ** n))


@dataclass(frozen=True)
class PairedTestResult:
    """Outcome of :func:`paired_permutation_test`."""

    statistic: float          # observed mean(a - b)
    p_value: float
    n_pairs: int
    exact: bool
    n_resamples: int
    min_attainable_p: float
    effect_ci: BootstrapCI
    alternative: str = "two-sided"
    #: True when ``p < alpha`` is unreachable at this ``n`` no matter the data.
    underpowered_by_construction: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_pairs": self.n_pairs,
            "exact": self.exact,
            "n_resamples": self.n_resamples,
            "min_attainable_p": self.min_attainable_p,
            "effect": self.effect_ci.as_dict(),
            "alternative": self.alternative,
            "underpowered_by_construction": self.underpowered_by_construction,
        }

    def __str__(self) -> str:
        kind = "exact" if self.exact else f"MC({self.n_resamples})"
        return (f"effect {self.effect_ci}, p={self.p_value:.4g} "
                f"[{kind} paired sign-flip, n={self.n_pairs}, "
                f"p_min={self.min_attainable_p:.4g}]")


def paired_permutation_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    n_resamples: int = 10000,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    seed: int = 0,
    n_boot: int = 10000,
) -> PairedTestResult:
    """Paired sign-flip permutation test on ``a - b``.

    The null is that the per-pair differences are symmetric about 0, so
    flipping the sign of any subset is equidistributed with the observation.
    No distributional assumption, no scipy.

    * ``n <= EXACT_PERMUTATION_MAX_N``: all ``2**n`` sign vectors are
      enumerated, giving an exact p-value ``count / 2**n``.
    * otherwise: ``n_resamples`` Monte-Carlo sign vectors with the
      ``(1 + count) / (1 + n_resamples)`` estimator, which is valid (never 0)
      and slightly conservative.

    A bootstrap CI on the effect is **always** returned, because with a handful
    of seeds the interval is far more informative than the p-value.
    """
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"unknown alternative {alternative!r}")
    n = a.size
    d = a - b
    obs = float(d.mean()) if n else 0.0
    ci = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
    p_min = min_attainable_p(n)

    if n == 0:
        return PairedTestResult(0.0, 1.0, 0, True, 0, 1.0, ci, alternative, True)
    if np.allclose(d, 0.0):
        return PairedTestResult(obs, 1.0, n, True, 0, p_min, ci, alternative,
                                p_min >= alpha)

    if n <= EXACT_PERMUTATION_MAX_N:
        signs = np.array(list(product((1.0, -1.0), repeat=n)), dtype=float)
        exact = True
        n_res = signs.shape[0]
    else:
        rng = stable_rng(seed, "paired_permutation", str(n), str(n_resamples))
        signs = rng.choice((1.0, -1.0), size=(n_resamples, n))
        exact = False
        n_res = n_resamples

    null = (signs * d).mean(axis=1)
    if alternative == "two-sided":
        count = int(np.sum(np.abs(null) >= abs(obs) - 1e-15))
    elif alternative == "greater":
        count = int(np.sum(null >= obs - 1e-15))
    else:
        count = int(np.sum(null <= obs + 1e-15))

    p = count / n_res if exact else (1 + count) / (1 + n_res)
    p = float(min(1.0, p))
    return PairedTestResult(
        statistic=obs,
        p_value=p,
        n_pairs=n,
        exact=exact,
        n_resamples=n_res,
        min_attainable_p=p_min,
        effect_ci=ci,
        alternative=alternative,
        underpowered_by_construction=bool(p_min >= alpha),
    )


def paired_significance_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Backwards-compatible wrapper returning ``(effect, p_value)``.

    Prefer :func:`paired_permutation_test`, which also returns the bootstrap CI
    and tells you whether ``p < alpha`` was attainable at all.  The old
    implementation of this function fabricated a p-value from ``tanh`` when
    scipy was missing; that code is gone.
    """
    res = paired_permutation_test(values_a, values_b, alpha=alpha, seed=seed)
    return res.statistic, res.p_value


# ---------------------------------------------------------------------------
# Multi-seed reporting
# ---------------------------------------------------------------------------


def compute_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    *,
    seed: int = 0,
    n_boot: int = 10000,
) -> tuple[float, float, float]:
    """``(mean, lo, hi)`` via the bootstrap.

    Kept for callers that want the plain tuple.  This used to be a
    t-distribution interval with a hard-coded ``1.96`` fallback; the bootstrap
    makes no normality assumption and is the standard path per C5.
    """
    vals = list(values)
    if not vals:
        return 0.0, 0.0, 0.0
    ci = bootstrap_ci(vals, n_boot=n_boot, alpha=1.0 - confidence, seed=seed)
    return ci.point, ci.lo, ci.hi


def format_with_ci(
    values: Sequence[float], confidence: float = 0.95, *, seed: int = 0
) -> str:
    """Format as ``mean [lo, hi] (n=k)`` -- the mandated headline format."""
    vals = list(values)
    if not vals:
        return "N/A"
    ci = bootstrap_ci(vals, alpha=1.0 - confidence, seed=seed)
    return f"{ci.point:.4f} [{ci.lo:.4f}, {ci.hi:.4f}] (n={ci.n})"


@dataclass
class MultiSeedResults:
    """Per-seed values for one metric, with bootstrap reporting."""

    metric_name: str
    values: list[float]
    seeds: list[int]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.seeds):
            raise ValueError(
                f"{self.metric_name}: {len(self.values)} values but "
                f"{len(self.seeds)} seeds"
            )

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.values, ddof=1)) if len(self.values) > 1 else 0.0

    def ci(self, alpha: float = 0.05, seed: int = 0) -> BootstrapCI:
        return bootstrap_ci(self.values, alpha=alpha, seed=seed)

    @property
    def ci_95(self) -> tuple[float, float, float]:
        return compute_confidence_interval(self.values, 0.95)

    def meets_seed_requirement(self, required: int = 10) -> bool:
        """C5: every headline number needs at least ``required`` seeds."""
        return self.n >= required

    def as_dict(self) -> dict[str, Any]:
        c = self.ci()
        return {"metric": self.metric_name, "n_seeds": self.n,
                "seeds": list(self.seeds), "values": list(self.values),
                "mean": self.mean, "std": self.std,
                "ci95_lo": c.lo, "ci95_hi": c.hi}

    def __repr__(self) -> str:
        return f"{self.metric_name}: {format_with_ci(self.values)}"


__all__ = [
    "SafetyMetrics", "BoundTightnessMetrics", "EfficiencyMetrics",
    "compute_violation_rate", "compute_bound_tightness",
    "compute_efficiency_metrics", "compute_regression_size",
    "aggregate_round_metrics",
    "count_parameters", "format_parameter_count",
    "BootstrapCI", "bootstrap_ci", "paired_bootstrap_ci",
    "PairedTestResult", "paired_permutation_test", "paired_significance_test",
    "min_attainable_p", "MIN_SEEDS_FOR_ALPHA_05", "EXACT_PERMUTATION_MAX_N",
    "DEFAULT_SEEDS",
    "compute_confidence_interval", "format_with_ci", "MultiSeedResults",
]
