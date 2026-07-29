"""Anytime-valid stratified concentration bounds for the SCA certificate.

This module is the mathematical core of the project.  Everything downstream
(the acceptance gate, the certificate, the federated experiments) is only as
sound as what is written here, so the central theorem is proved in full below
and Monte-Carlo verified in ``tests/test_stats.py``.

--------------------------------------------------------------------------
WHY THE PREVIOUS VERSION WAS UNSOUND
--------------------------------------------------------------------------
The previous implementation computed, for each region ``j``,

    UCB_j = p_hat_j + sqrt( ln(2K / delta) / (2 m_j) )

with three fatal problems:

(P1) ``p_hat_j`` was computed from samples produced by an *adaptive search*
     procedure (a mutation recursion that recursed only on already-violating
     nodes).  Those samples are not i.i.d. draws from ``D | R_j``; they are
     drawn from a distribution that is deliberately biased toward violations
     and whose bias depends on the outcomes already observed.  Hoeffding's
     inequality does not apply and the resulting "certificate" was not a
     certificate of anything.

(P2) The number of samples ``m_j`` was chosen adaptively (more budget was
     spent where the search found trouble).  Even with genuinely i.i.d.
     draws, a *fixed-m* Hoeffding bound is not valid at a data-dependent
     stopping time: an adversary who is allowed to stop when the running
     mean looks favourable can drive the coverage arbitrarily far below
     ``1 - delta``.

(P3) ``K`` was set to the number of regions that happened to contain data
     (``k = len(region_stats)`` after empty buckets were dropped), and the
     weights were renormalised over that subset.  So a run whose sampler
     degenerated onto a few easy regions received the *smallest* union-bound
     penalty and had the un-sampled (possibly catastrophic) regions silently
     removed from the weighted sum.  In the limit ``region_stats == []`` the
     old code returned ``(accepted=True, bound=0.0)``: zero evidence
     certified a violation rate of exactly zero.

The rebuild fixes (P1) by physically separating the SEARCH counters from the
ESTIMATION counters (see ``sca/knowledge_graph/regions.py``), fixes (P2) with
the anytime-valid bound proved below, and fixes (P3) in
:func:`check_acceptance`, which takes ``k_total`` and ``budget_cap`` as
a-priori arguments and charges ``UCB = 1`` to every declared region that was
not estimated.

--------------------------------------------------------------------------
THEOREM 1 (anytime-valid stratified upper bound, adaptive allocation)
--------------------------------------------------------------------------
SETUP.  Before any data is observed, fix

  * a measurable partition of the input space into ``K`` regions
    ``R_1, ..., R_K`` (``K = k_total``);
  * region weights ``w_j >= 0`` with ``sum_j w_j = 1``, where ``w_j`` is the
    mass that the evaluation distribution ``D`` places on ``R_j``.  The
    weights are computed from the estimation pool alone and are frozen
    before verification begins;
  * a per-region budget cap ``M >= 1`` (``M = budget_cap``);
  * a confidence parameter ``delta in (0, 1)``.

For every region ``j`` let ``X_{j,1}, X_{j,2}, ...`` be an infinite i.i.d.
sequence of Bernoulli(``p_j``) random variables, where
``p_j = Pr_{x ~ D|R_j}[ x is a violation ]``, and let

    p_hat_j^(m) = (1/m) * sum_{i=1}^{m} X_{j,i},      1 <= m <= M.

Define the width

    W(m) = sqrt( ln(2 K M / delta) / (2 m) ),     W(0) = +infinity.

CLAIM.  Let ``m_1, ..., m_K`` be *any* random variables taking values in
``{0, 1, ..., M}`` -- in particular they may be chosen by an adversary with
full knowledge of every ``X_{j,i}``, may depend on the observed violations,
and may differ across regions.  Set

    UCB_j = min( 1, p_hat_j^(m_j) + W(m_j) )        (so UCB_j = 1 if m_j = 0).

Then

    Pr[ for all j in {1..K}:  p_j <= UCB_j ]  >=  1 - delta,

and on that same event, since ``w_j >= 0`` and ``sum_j w_j = 1``,

    sum_j w_j p_j  <=  sum_j w_j UCB_j.

PROOF.
Step 1 (fixed pair).  Fix ``j in {1..K}`` and ``m in {1..M}``.  The variables
``X_{j,1}, ..., X_{j,m}`` are i.i.d. and bounded in ``[0, 1]``, so Hoeffding's
inequality gives, for any ``t > 0``,

    Pr[ p_j - p_hat_j^(m) >= t ]  <=  exp(-2 m t^2).

Choose ``t = W(m) = sqrt( ln(2KM/delta) / (2m) )``.  Then
``2 m t^2 = ln(2KM/delta)`` and

    Pr[ p_j - p_hat_j^(m) >= W(m) ]  <=  delta / (2 K M).

(The factor ``2`` inside the logarithm is *not* needed for this one-sided
statement -- ``ln(KM/delta)`` would already suffice.  We keep it deliberately.
With the factor ``2`` the same width simultaneously controls the lower
deviation ``p_hat_j^(m) - p_j >= W(m)``, so the identical constant also
certifies the two-sided interval used for diagnostics and for the regression
subgraph.  Keeping one conservative constant everywhere removes any chance of
a caller accidentally applying the one-sided constant two-sidedly.  The price
is a factor ``sqrt(ln(2KM/delta) / ln(KM/delta))``, typically under 1.05.)

Step 2 (union bound over a finite, a-priori index set).  Define for each pair
the bad event

    B_{j,m} = { p_j - p_hat_j^(m) >= W(m) },     j in [K], m in [M].

There are exactly ``K * M`` such pairs and -- this is the crux -- the index
set ``[K] x [M]`` is fixed *before* any data is seen, because both ``K`` and
``M`` are declared a priori.  By the union bound,

    Pr[ union_{j,m} B_{j,m} ]  <=  K * M * delta / (2 K M)  =  delta / 2  <=  delta.

Let ``E`` be the complement, ``E = intersection_{j,m} B_{j,m}^c``.  Then
``Pr[E] >= 1 - delta/2 >= 1 - delta``.

Step 3 (transfer to data-dependent sample sizes).  The event ``E`` is a
statement quantified over *all* ``m in [M]`` simultaneously; its definition
does not mention ``m_j`` at all.  Therefore, deterministically, on every
sample path ``omega in E`` and for every ``j``,

    p_j  <  p_hat_j^{(m)}(omega) + W(m)     for every m in [M],

and in particular for ``m = m_j(omega)`` whenever ``m_j(omega) >= 1``.  No
measurability or optional-stopping argument is required: the adversary
selecting ``m_j`` cannot land on a pair ``(j, m)`` that is not already
covered, because every pair is covered.  If ``m_j(omega) = 0`` then
``UCB_j = 1 >= p_j`` trivially.  Clipping ``UCB_j`` at ``1`` is likewise
harmless since ``p_j <= 1`` always.  Hence ``E`` implies
``p_j <= UCB_j`` for all ``j``, giving ``Pr[for all j: p_j <= UCB_j] >= 1 - delta``.

Step 4 (weighted aggregate).  On ``E``, multiplying ``p_j <= UCB_j`` by
``w_j >= 0`` and summing over ``j`` gives
``sum_j w_j p_j <= sum_j w_j UCB_j``.  The left-hand side is exactly the
violation probability under ``D``, since ``D`` decomposes over the partition
as ``Pr_D[violation] = sum_j w_j p_j``.                                  QED

COST.  Relative to a fixed-``m`` bound with an a-priori ``K``, the anytime
version pays only ``sqrt( ln(2KM/delta) / ln(2K/delta) )`` -- i.e. a
``sqrt(ln M)`` inflation.  For ``K = 8``, ``M = 512``, ``delta = 0.05`` this
is a factor of 1.42.  That factor is the entire price of making the adaptive,
search-guided allocation legal, and it is why the rebuilt protocol can spend
its estimation budget wherever Stage A says it should.

--------------------------------------------------------------------------
WHAT THEOREM 1 DOES *NOT* SAY -- read this before quoting a number
--------------------------------------------------------------------------
1. It bounds the violation rate under the distribution the estimation samples
   are actually drawn from.  In the implementation, region samples are drawn
   i.i.d. *with replacement from the estimation stratum*, so ``p_j`` is the
   violation rate of the model on the empirical distribution of that stratum,
   not on the underlying population.  Generalising from stratum to population
   is a second, separate inferential step that this module does not perform
   and that no number produced here should be read as covering.
2. It assumes the samples fed to the estimation counters really are i.i.d.
   draws from ``D | R_j`` and are *not* selected by the search.  The
   accounting separation that enforces this lives in ``regions.py``; if a
   mutant ever reaches an estimation counter, this theorem is void.
3. It assumes ``K``, ``M``, ``delta`` and the partition are fixed before the
   data.  :func:`check_acceptance` raises ``ValueError`` rather than silently
   accepting data-derived values, but it cannot detect a caller that
   re-declares ``K`` after peeking.  That discipline is the caller's.
4. It is a statement about a single certificate.  Running the gate for ``T``
   federated rounds and reporting the minimum bound requires a further union
   bound over ``T``; ``delta`` must be divided by ``T`` by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "RegionStat",
    "RegionUCB",
    "AcceptanceResult",
    "anytime_hoeffding_width",
    "compute_ucb",
    "check_acceptance",
    "optimal_allocation",
    "clopper_pearson_upper",
    "empirical_bernstein_width",
    "legacy_fixed_m_width",
]

# Tolerance used when comparing float weight sums to 1.
_WEIGHT_TOL = 1e-9


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionStat:
    """Raw per-region counts from the ESTIMATION stage.

    These are the only quantities a certificate is allowed to see.  Search /
    mutation counters live in a parallel structure and never appear here.

    Attributes:
        region_id: Index ``j`` of the region, in ``[0, k_total)``.
        weight: ``w_j``, the mass of ``R_j`` under the evaluation
            distribution, computed from the estimation pool and frozen before
            verification.
        n_samples: ``m_j``, the number of i.i.d. estimation draws taken.
        n_violations: Number of those draws flagged as violations.
    """

    region_id: int
    weight: float
    n_samples: int
    n_violations: int

    def __post_init__(self) -> None:
        if self.n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {self.n_samples}")
        if self.n_violations < 0:
            raise ValueError(
                f"n_violations must be >= 0, got {self.n_violations}"
            )
        if self.n_violations > self.n_samples:
            raise ValueError(
                f"n_violations ({self.n_violations}) > n_samples "
                f"({self.n_samples}) in region {self.region_id}"
            )
        if self.weight < 0 or not math.isfinite(self.weight):
            raise ValueError(f"weight must be finite and >= 0, got {self.weight}")
        if self.region_id < 0:
            raise ValueError(f"region_id must be >= 0, got {self.region_id}")

    @property
    def p_hat(self) -> float:
        """Empirical violation rate; ``0.0`` when there are no samples.

        Note that ``p_hat == 0.0`` for an unsampled region is a placeholder,
        not evidence: :func:`check_acceptance` charges such a region
        ``UCB = 1``.
        """
        if self.n_samples == 0:
            return 0.0
        return self.n_violations / self.n_samples


@dataclass(frozen=True)
class RegionUCB:
    """A per-region upper confidence bound and the inputs that produced it."""

    region_id: int
    weight: float
    n_samples: int
    p_hat: float
    width: float
    ucb: float


@dataclass(frozen=True)
class AcceptanceResult:
    """Outcome of the acceptance rule, with everything needed to re-derive it.

    Attributes:
        accepted: ``True`` iff the certificate is non-vacuous and
            ``bound <= epsilon``.
        bound: ``sum_j w_j UCB_j`` over the full declared partition, including
            a charge of ``1.0`` for any declared weight that was not
            estimated.
        per_region: One :class:`RegionUCB` per supplied region, in the order
            supplied.
        vacuous: ``True`` when the result carries no information -- no
            regions, zero total weight, or ``bound >= 1``.
        k_total: The a-priori number of regions used in the union bound.
        budget_cap: The a-priori per-region budget cap used in the union
            bound.
        epsilon: Threshold the bound was compared against.
        delta: Global failure probability.
        unaccounted_weight: Declared probability mass not covered by any
            supplied region; charged at ``UCB = 1``.
    """

    accepted: bool
    bound: float
    per_region: list[RegionUCB]
    vacuous: bool
    k_total: int
    budget_cap: int
    epsilon: float
    delta: float
    unaccounted_weight: float = 0.0


# ---------------------------------------------------------------------------
# Widths and UCBs
# ---------------------------------------------------------------------------


def _validate_partition_params(k_total: int, budget_cap: int, delta: float) -> None:
    if not isinstance(k_total, (int, np.integer)) or k_total < 1:
        raise ValueError(f"k_total must be an integer >= 1, got {k_total!r}")
    if not isinstance(budget_cap, (int, np.integer)) or budget_cap < 1:
        raise ValueError(f"budget_cap must be an integer >= 1, got {budget_cap!r}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must lie in (0, 1), got {delta!r}")


def anytime_hoeffding_width(
    m: int, k_total: int, budget_cap: int, delta: float
) -> float:
    """Anytime-valid Hoeffding half-width ``W(m)`` from Theorem 1.

    ``W(m) = sqrt( ln(2 * k_total * budget_cap / delta) / (2 m) )``

    The ``k_total * budget_cap`` inside the log is the union bound over the
    a-priori grid of ``(region, sample-count)`` pairs; it is what makes the
    width valid simultaneously for every ``m in [1, budget_cap]`` and hence
    valid at a data-dependent ``m``.  See the module docstring for the proof.

    Args:
        m: Realised number of i.i.d. estimation samples in the region.
        k_total: A-priori number of regions ``K``.
        budget_cap: A-priori per-region budget cap ``M``.
        delta: Global failure probability.

    Returns:
        The additive width, or ``math.inf`` when ``m == 0`` (no evidence).

    Raises:
        ValueError: If the partition parameters are invalid, ``m`` is
            negative, or ``m > budget_cap`` (a sample count beyond the
            declared cap is outside the union bound, so the width would not
            be valid).
    """
    _validate_partition_params(k_total, budget_cap, delta)
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")
    if m > budget_cap:
        raise ValueError(
            f"m={m} exceeds the a-priori budget_cap={budget_cap}. Theorem 1 "
            "only union-bounds pairs (j, m) with m <= budget_cap, so this "
            "width would not be valid. Declare a larger budget_cap BEFORE "
            "collecting data."
        )
    if m == 0:
        return math.inf
    return math.sqrt(math.log(2.0 * k_total * budget_cap / delta) / (2.0 * m))


def compute_ucb(
    n_violations: int,
    n_samples: int,
    k_total: int,
    budget_cap: int,
    delta: float,
) -> float:
    """Anytime-valid upper confidence bound on the regional violation rate.

    ``UCB_j = min(1, p_hat_j + W(m_j))``.  Returns ``1.0`` when
    ``n_samples == 0``: zero evidence gives the trivial bound, never ``0``.
    """
    if n_samples == 0:
        return 1.0
    if n_violations > n_samples:
        raise ValueError(
            f"n_violations ({n_violations}) > n_samples ({n_samples})"
        )
    p_hat = n_violations / n_samples
    width = anytime_hoeffding_width(n_samples, k_total, budget_cap, delta)
    return float(min(1.0, max(0.0, p_hat + width)))


def legacy_fixed_m_width(m: int, k_nonempty: int, delta: float) -> float:
    """The OLD, unsound width: ``sqrt(ln(2 k / delta) / (2 m))``.

    Retained *only* so that ``tests/test_stats.py`` can use it as a negative
    control and demonstrate that the Monte-Carlo coverage test has the power
    to detect an invalid bound.  It is not anytime-valid, and in the old code
    ``k_nonempty`` was derived from the data (non-empty buckets only), which
    compounds the error.  Never use this to certify anything.
    """
    if m <= 0:
        return 1.0
    return math.sqrt(math.log(2.0 * max(k_nonempty, 1) / delta) / (2.0 * m))


# ---------------------------------------------------------------------------
# Acceptance rule
# ---------------------------------------------------------------------------


def check_acceptance(
    region_stats: Sequence[RegionStat],
    epsilon: float,
    delta: float,
    k_total: int,
    budget_cap: int,
) -> AcceptanceResult:
    """Evaluate the SCA acceptance rule with the anytime-valid bound.

    Accept iff ``sum_j w_j UCB_j <= epsilon`` **and** the certificate is not
    vacuous.

    Semantics (this is the fix for finding F10):

    * ``k_total`` and ``budget_cap`` are declared A PRIORI by the caller.
      They must not be derived from the data (e.g. from the number of
      non-empty buckets).  ``ValueError`` is raised if more regions are
      supplied than were declared.
    * A region with ``weight > 0`` and ``n_samples == 0`` contributes
      ``UCB = 1.0``.  It is never dropped.
    * Weights are interpreted against the FULL declared partition.  If the
      supplied weights sum to ``W < 1``, the missing mass ``1 - W`` belongs to
      declared regions that were not reported at all and is charged at
      ``UCB = 1.0``.  If they sum to ``W > 1`` (e.g. the caller passed stratum
      counts rather than normalised weights) all weights are divided by ``W``.
      In either case the effective weights are a probability vector over the
      declared partition and the bound is conservative.
    * Empty input, or total weight ``0``, yields
      ``accepted=False, bound=1.0, vacuous=True``.  Zero evidence never
      certifies anything.

    Args:
        region_stats: ESTIMATION-stage counts only.  Passing search/mutation
            counts here voids Theorem 1.
        epsilon: Target violation bound.
        delta: Global failure probability.
        k_total: A-priori number of regions in the partition.
        budget_cap: A-priori per-region estimation budget cap.

    Returns:
        An :class:`AcceptanceResult`.

    Raises:
        ValueError: On invalid parameters, duplicate region ids, region ids
            outside ``[0, k_total)``, more regions than ``k_total``, or a
            region whose ``n_samples`` exceeds ``budget_cap``.
    """
    _validate_partition_params(k_total, budget_cap, delta)
    if not math.isfinite(epsilon):
        raise ValueError(f"epsilon must be finite, got {epsilon!r}")

    stats = list(region_stats)

    if len(stats) > k_total:
        raise ValueError(
            f"received {len(stats)} regions but k_total={k_total} was declared "
            "a priori. k_total must be fixed before data collection and must "
            "cover the whole partition."
        )

    seen: set[int] = set()
    for rs in stats:
        if rs.region_id in seen:
            raise ValueError(f"duplicate region_id {rs.region_id} in region_stats")
        seen.add(rs.region_id)
        if rs.region_id >= k_total:
            raise ValueError(
                f"region_id {rs.region_id} is outside the declared partition "
                f"[0, {k_total})"
            )
        if rs.n_samples > budget_cap:
            raise ValueError(
                f"region {rs.region_id} used {rs.n_samples} samples but "
                f"budget_cap={budget_cap} was declared a priori; the union "
                "bound in Theorem 1 does not cover this sample count."
            )

    if not stats:
        return AcceptanceResult(
            accepted=False,
            bound=1.0,
            per_region=[],
            vacuous=True,
            k_total=k_total,
            budget_cap=budget_cap,
            epsilon=epsilon,
            delta=delta,
            unaccounted_weight=1.0,
        )

    total_weight = sum(rs.weight for rs in stats)
    if total_weight <= 0.0:
        return AcceptanceResult(
            accepted=False,
            bound=1.0,
            per_region=[],
            vacuous=True,
            k_total=k_total,
            budget_cap=budget_cap,
            epsilon=epsilon,
            delta=delta,
            unaccounted_weight=1.0,
        )

    # Renormalise against the full declared partition (never against the
    # subset that happens to carry data).
    scale = 1.0 / total_weight if total_weight > 1.0 + _WEIGHT_TOL else 1.0
    unaccounted = max(0.0, 1.0 - total_weight * scale)

    per_region: list[RegionUCB] = []
    bound = unaccounted * 1.0  # missing declared mass is charged UCB = 1
    for rs in stats:
        w = rs.weight * scale
        if rs.n_samples == 0:
            width = math.inf
            ucb = 1.0
        else:
            width = anytime_hoeffding_width(
                rs.n_samples, k_total, budget_cap, delta
            )
            ucb = float(min(1.0, max(0.0, rs.p_hat + width)))
        per_region.append(
            RegionUCB(
                region_id=rs.region_id,
                weight=w,
                n_samples=rs.n_samples,
                p_hat=rs.p_hat,
                width=width,
                ucb=ucb,
            )
        )
        bound += w * ucb

    bound = float(min(1.0, max(0.0, bound)))
    vacuous = bound >= 1.0 - _WEIGHT_TOL
    accepted = bool((not vacuous) and bound <= epsilon)

    return AcceptanceResult(
        accepted=accepted,
        bound=bound,
        per_region=per_region,
        vacuous=vacuous,
        k_total=k_total,
        budget_cap=budget_cap,
        epsilon=epsilon,
        delta=delta,
        unaccounted_weight=unaccounted,
    )


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------


def optimal_allocation(
    weights: Sequence[float] | np.ndarray,
    total_budget: int,
    k_total: int,
) -> list[int]:
    r"""Width-minimising allocation of the estimation budget: ``m_j ~ w_j^(2/3)``.

    DERIVATION (this is the fix for finding F14; the old code used
    ``m_j ~ w_j``, which is the answer to a different optimisation problem).

    Holding the empirical rates fixed, the part of the certified bound that
    the allocation controls is the total width

        F(m_1, ..., m_K) = sum_j w_j * sqrt( C / (2 m_j) )
                         = c * sum_j w_j * m_j^{-1/2},
          with c = sqrt(C / 2),  C = ln(2 K M / delta) > 0,

    to be minimised subject to ``sum_j m_j = B`` and ``m_j > 0``.  Form the
    Lagrangian

        L(m, lambda) = c * sum_j w_j m_j^{-1/2} + lambda * ( sum_j m_j - B ).

    Stationarity in ``m_j``:

        dL/dm_j = -(c/2) * w_j * m_j^{-3/2} + lambda = 0
              =>  m_j^{3/2} = (c / (2 lambda)) * w_j
              =>  m_j       = (c / (2 lambda))^{2/3} * w_j^{2/3}.

    So ``m_j`` is proportional to ``w_j^{2/3}``.  Imposing the budget
    constraint fixes the constant:

        m_j = B * w_j^{2/3} / sum_i w_i^{2/3}.

    ``F`` is convex on the positive orthant (each term ``w_j m_j^{-1/2}`` is
    convex in ``m_j`` for ``w_j >= 0``) and the constraint set is affine, so
    this stationary point is the global minimum.  Substituting back gives the
    optimal value ``F* = c * B^{-1/2} * ( sum_j w_j^{2/3} )^{3/2}``, whose
    ``( sum w^{2/3} )^{3/2}`` factor is the 2/3-norm of the weight vector --
    a standard Neyman-allocation-style result.

    Sanity check against the old code: proportional allocation ``m_j = B w_j``
    gives ``F_prop = c B^{-1/2} sum_j sqrt(w_j)`` and uniform gives
    ``F_unif = c B^{-1/2} sqrt(K) * 1`` (using ``sum w_j = 1``).  By the power
    mean / Holder inequality ``( sum w^{2/3} )^{3/2} <= sum sqrt(w)`` and
    ``<= sqrt(K)``, with equality only when all weights are equal, so the
    ``2/3`` rule is never worse and is strictly better whenever the weights
    are non-uniform.  ``tests/test_stats.py`` verifies this numerically.

    PRACTICAL DEVIATION.  A region with ``m_j = 0`` contributes ``UCB = 1`` and
    would destroy the bound, so every region with ``w_j > 0`` is guaranteed at
    least one sample when the budget allows.  The remaining budget is then
    distributed by the ``2/3`` rule using the largest-remainder method, which
    makes the returned allocation sum to ``total_budget`` exactly.

    Args:
        weights: Non-negative region weights, one per region.  Need not be
            normalised.
        total_budget: Total number of estimation samples ``B`` to distribute.
            The returned list sums to exactly this value.
        k_total: A-priori number of regions; ``len(weights)`` must not exceed
            it.

    Returns:
        A list of non-negative integers of length ``len(weights)`` summing to
        exactly ``total_budget``.

    Raises:
        ValueError: On negative weights, negative budget, or
            ``len(weights) > k_total``.

    Note:
        This function does not know ``budget_cap``; a caller that declared a
        per-region cap ``M`` must check ``max(alloc) <= M`` itself, since
        Theorem 1 only covers ``m_j <= M``.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError(f"weights must be 1-D, got shape {w.shape}")
    if len(w) > k_total:
        raise ValueError(
            f"len(weights)={len(w)} exceeds the a-priori k_total={k_total}"
        )
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    if total_budget < 0:
        raise ValueError(f"total_budget must be >= 0, got {total_budget}")

    n = len(w)
    alloc = [0] * n
    if n == 0 or total_budget == 0:
        return alloc

    positive = [j for j in range(n) if w[j] > 0]
    if not positive:
        # No positive weights: nothing is informative. Spread uniformly so the
        # budget is still exactly spent.
        base, rem = divmod(total_budget, n)
        for j in range(n):
            alloc[j] = base + (1 if j < rem else 0)
        return alloc

    q = np.zeros(n, dtype=float)
    q[positive] = w[positive] ** (2.0 / 3.0)
    q = q / q.sum()

    if total_budget <= len(positive):
        # Cannot give everyone one sample; hand them out to the largest q.
        order = sorted(positive, key=lambda j: (-q[j], j))
        for j in order[:total_budget]:
            alloc[j] = 1
        return alloc

    # Guarantee one sample per positive-weight region, then distribute the
    # rest by the 2/3 rule using largest remainders.
    for j in positive:
        alloc[j] = 1
    remaining = total_budget - len(positive)
    raw = q * remaining
    floors = np.floor(raw).astype(int)
    for j in range(n):
        alloc[j] += int(floors[j])
    leftover = remaining - int(floors.sum())
    if leftover > 0:
        frac = raw - floors
        order = sorted(range(n), key=lambda j: (-frac[j], -q[j], j))
        for j in order[:leftover]:
            alloc[j] += 1

    assert sum(alloc) == total_budget, (sum(alloc), total_budget)
    return alloc


# ---------------------------------------------------------------------------
# Optional tighter variant
# ---------------------------------------------------------------------------


def empirical_bernstein_width(
    n_violations: int,
    m: int,
    k_total: int,
    budget_cap: int,
    delta: float,
) -> float:
    """Anytime-valid empirical-Bernstein half-width (Maurer & Pontil, 2009).

    For i.i.d. ``X_1..X_m`` in ``[0, 1]`` with unbiased sample variance
    ``V_m``, Maurer-Pontil Theorem 4 gives, with probability ``1 - d``,

        E[X] <= mean + sqrt( 2 V_m ln(2/d) / m ) + 7 ln(2/d) / (3 (m - 1)).

    Applying it at ``d = delta / (k_total * budget_cap)`` and union-bounding
    over the a-priori grid of ``(region, m)`` pairs -- exactly the argument of
    Theorem 1, Steps 2-3 -- makes it anytime-valid too.  Writing
    ``L = ln(2 * k_total * budget_cap / delta)``, the width is

        sqrt( 2 V_m L / m ) + 7 L / (3 (m - 1)),

    with ``V_m = m / (m - 1) * p_hat (1 - p_hat)`` for Bernoulli data.

    It is tighter than Hoeffding when violations are rare or near-certain and
    ``m`` is not tiny; it is *looser* for small ``m`` because of the
    ``7L / (3(m-1))`` remainder.  Returns ``math.inf`` for ``m < 2``, where the
    inequality does not apply.

    STATUS: Monte-Carlo verified for coverage under the same adversarial
    adaptive allocator used for Theorem 1 (see
    ``tests/test_stats.py::TestEmpiricalBernsteinCoverage``).  It is exported,
    but nothing in the certificate path uses it -- the headline numbers are
    all Hoeffding, which is the more conservative choice.
    """
    _validate_partition_params(k_total, budget_cap, delta)
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")
    if m > budget_cap:
        raise ValueError(
            f"m={m} exceeds the a-priori budget_cap={budget_cap}"
        )
    if n_violations > m:
        raise ValueError(f"n_violations ({n_violations}) > m ({m})")
    if m < 2:
        return math.inf
    log_term = math.log(2.0 * k_total * budget_cap / delta)
    p_hat = n_violations / m
    var_unbiased = (m / (m - 1.0)) * p_hat * (1.0 - p_hat)
    return math.sqrt(2.0 * var_unbiased * log_term / m) + 7.0 * log_term / (
        3.0 * (m - 1.0)
    )


def clopper_pearson_upper(n_violations: int, n_samples: int, alpha: float) -> float:
    """Exact (Clopper-Pearson) one-sided upper bound for a binomial proportion.

    Valid for a FIXED, pre-specified ``n_samples``.  It is *not* anytime-valid
    and must not be substituted into the certificate path in place of
    :func:`compute_ucb` without its own union bound over ``(j, m)`` pairs.

    Args:
        n_violations: Number of violations observed.
        n_samples: Number of samples.
        alpha: One-sided significance level.

    Returns:
        Upper bound on the true rate; ``1.0`` when there are no samples or all
        samples were violations.
    """
    if n_samples == 0:
        return 1.0
    if n_violations > n_samples:
        raise ValueError(f"n_violations ({n_violations}) > n_samples ({n_samples})")
    if n_violations == n_samples:
        return 1.0
    from scipy.stats import beta as beta_dist

    return float(
        beta_dist.ppf(1 - alpha, n_violations + 1, n_samples - n_violations)
    )
