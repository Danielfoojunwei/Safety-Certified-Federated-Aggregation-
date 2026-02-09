"""Statistical utilities for concentration inequalities and confidence bounds.

Implements the Hoeffding-based UCB construction from Section 4.3:
    UCB_j = p_hat_j + sqrt(ln(2K / delta) / (2 * m_j))

and the acceptance rule:
    sum_j w_j * UCB_j <= epsilon
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def hoeffding_bound(m: int, delta: float, k: int = 1) -> float:
    """Hoeffding confidence width for a Bernoulli mean estimate.

    For m i.i.d. Bernoulli samples with a union-bound correction over K regions,
    the one-sided confidence width at level delta is:

        sqrt(ln(2K / delta) / (2m))

    Args:
        m: Number of samples in the region.
        delta: Global failure probability.
        k: Number of regions (for union-bound correction).

    Returns:
        The confidence width (additive term for UCB).
    """
    if m <= 0:
        return 1.0  # No samples -> maximum uncertainty
    return math.sqrt(math.log(2 * k / delta) / (2 * m))


def compute_ucb(p_hat: float, m: int, delta: float, k: int = 1) -> float:
    """Upper confidence bound on regional violation probability.

    UCB_j = p_hat_j + sqrt(ln(2K / delta) / (2 * m_j)), clipped to [0, 1].

    Args:
        p_hat: Empirical violation rate in the region.
        m: Number of samples in the region.
        delta: Global failure probability.
        k: Number of regions (for union-bound correction).

    Returns:
        The UCB estimate, clipped to [0, 1].
    """
    width = hoeffding_bound(m, delta, k)
    return min(1.0, p_hat + width)


@dataclass
class RegionStats:
    """Per-region empirical statistics collected by the verifier.

    Attributes:
        region_id: Index j of the region R_j.
        n_samples: Number of samples m_j drawn from this region.
        n_violations: Number of violations observed.
        weight: Prior weight w_j = Pr[x in R_j] under evaluation distribution D.
    """
    region_id: int
    n_samples: int
    n_violations: int
    weight: float

    @property
    def p_hat(self) -> float:
        """Empirical violation rate p_hat_j = n_violations / m_j."""
        if self.n_samples == 0:
            return 0.0
        return self.n_violations / self.n_samples

    def ucb(self, delta: float, k: int) -> float:
        """UCB_j for this region."""
        return compute_ucb(self.p_hat, self.n_samples, delta, k)

    def weighted_ucb(self, delta: float, k: int) -> float:
        """w_j * UCB_j contribution to the global bound."""
        return self.weight * self.ucb(delta, k)


def check_acceptance(
    region_stats: list[RegionStats],
    epsilon: float,
    delta: float,
) -> tuple[bool, float]:
    """Evaluate the SCA acceptance rule (Section 4.3).

    Accept iff:  sum_j w_j * UCB_j <= epsilon.

    Args:
        region_stats: Per-region statistics from the verifier.
        epsilon: Target violation bound.
        delta: Confidence parameter.

    Returns:
        (accepted, bound_value) where bound_value = sum_j w_j * UCB_j.
    """
    k = len(region_stats)
    if k == 0:
        return True, 0.0

    bound = sum(rs.weighted_ucb(delta, k) for rs in region_stats)
    return bound <= epsilon, bound


def optimal_allocation(
    weights: np.ndarray,
    p_hats: np.ndarray,
    total_budget: int,
    delta: float,
) -> np.ndarray:
    """Neyman-style allocation of sampling budget across regions.

    Allocates more samples to regions with higher weight and higher
    estimated violation rate to minimize the overall bound width.

    Under Hoeffding bounds the bound is:
        sum_j w_j * (p_hat_j + sqrt(C / (2 m_j)))

    Minimizing the width term sum_j w_j * sqrt(C / (2 m_j)) subject to
    sum_j m_j = M gives the allocation m_j proportional to w_j.

    Args:
        weights: Region weights w_j (must sum to 1).
        p_hats: Current empirical violation rates.
        total_budget: Total number of samples M to allocate.
        delta: Confidence parameter (used for the log term).

    Returns:
        Integer array of per-region sample allocations.
    """
    k = len(weights)
    # Proportional allocation: m_j ~ w_j (minimizes sum_j w_j / sqrt(m_j))
    raw = weights * total_budget
    # Ensure at least 1 sample per region
    alloc = np.maximum(np.floor(raw), 1).astype(int)
    # Distribute remaining budget to highest-weight regions
    remaining = total_budget - alloc.sum()
    if remaining > 0:
        fractional = raw - alloc
        top_indices = np.argsort(-fractional)[:remaining]
        alloc[top_indices] += 1
    return alloc


def clopper_pearson_upper(n_violations: int, n_samples: int, alpha: float) -> float:
    """Clopper-Pearson exact upper confidence bound for a binomial proportion.

    An alternative to the Hoeffding-based UCB that can be tighter for
    small sample sizes.

    Args:
        n_violations: Number of violations (successes).
        n_samples: Total number of samples.
        alpha: Significance level (one-sided).

    Returns:
        Upper bound on the true violation probability.
    """
    if n_samples == 0:
        return 1.0
    if n_violations == n_samples:
        return 1.0
    from scipy.stats import beta as beta_dist
    return beta_dist.ppf(1 - alpha, n_violations + 1, n_samples - n_violations)
