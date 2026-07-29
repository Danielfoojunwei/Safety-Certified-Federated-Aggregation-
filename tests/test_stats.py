"""Tests for the anytime-valid stratified bound (Theorem 1).

The centrepiece is :class:`TestTheorem1Coverage`, a real Monte-Carlo coverage
experiment with an adversarial, data-dependent allocator.  The old repository
had a ``test_soundness_simulation`` that accumulated a ``false_accepts``
counter over 500 trials and then never read it, so it could not fail.  This
one reads its counters, asserts on them, and -- crucially -- carries negative
controls that must FAIL, proving the test has the power to detect an invalid
bound.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sca.utils.stats import (
    AcceptanceResult,
    RegionStat,
    RegionUCB,
    anytime_hoeffding_width,
    check_acceptance,
    clopper_pearson_upper,
    compute_ucb,
    empirical_bernstein_width,
    legacy_fixed_m_width,
    optimal_allocation,
)

# ---------------------------------------------------------------------------
# Monte-Carlo configuration.  Chosen so that:
#   * the new anytime bound covers comfortably,
#   * the OLD fixed-m bound provably does not (negative control),
#   * the whole thing runs in a few seconds on CPU.
# ---------------------------------------------------------------------------
MC_K = 12          # a-priori number of regions
MC_M = 4000        # a-priori per-region budget cap
MC_DELTA = 0.10    # global failure probability
MC_TRIALS = 3000   # >= 2000 as required
MC_PILOT = 32      # pilot size for the reallocating adversary
MC_KEEP = 3        # regions the "drop empties" adversary keeps


def _mc_widths(k_for_log: int, budget_for_log: int | None) -> np.ndarray:
    """Width table over m = 1..MC_M.

    ``budget_for_log=None`` reproduces the OLD, non-anytime width
    ``sqrt(ln(2k/delta) / (2m))``.
    """
    ms = np.arange(1, MC_M + 1, dtype=np.float64)
    if budget_for_log is None:
        c = math.log(2.0 * k_for_log / MC_DELTA)
    else:
        c = math.log(2.0 * k_for_log * budget_for_log / MC_DELTA)
    return np.sqrt(c / (2.0 * ms)).astype(np.float32)


@pytest.fixture(scope="module")
def mc_draws():
    """Run the Monte-Carlo experiment once and share it across tests.

    For each of ``MC_TRIALS`` trials and each of ``MC_K`` regions we draw an
    i.i.d. Bernoulli(``p_j``) stream of length ``MC_M`` and then let several
    adversaries choose a data-dependent ``m_j``.

    Returned dict keys hold ``(MC_TRIALS, MC_K)`` arrays of realised UCBs.
    """
    p_true = np.full(MC_K, 0.5)  # maximal-variance regime
    weights = np.full(MC_K, 1.0 / MC_K)
    rng = np.random.default_rng(20240729)

    ms = np.arange(1, MC_M + 1, dtype=np.float32)
    w_new = _mc_widths(MC_K, MC_M)              # Theorem 1
    w_old = _mc_widths(MC_K, None)              # old fixed-m, correct K
    w_old_drop = _mc_widths(MC_KEEP, None)      # old fixed-m, K from the data
    log_term = math.log(2.0 * MC_K * MC_M / MC_DELTA)
    eb_rem = np.full(MC_M, np.inf, dtype=np.float64)
    eb_rem[1:] = 7.0 * log_term / (3.0 * (ms[1:].astype(np.float64) - 1.0))
    eb_rem = eb_rem.astype(np.float32)

    shape = (MC_TRIALS, MC_K)
    out = {
        "new_stop": np.empty(shape, np.float32),
        "new_pilot": np.empty(shape, np.float32),
        "old_stop": np.empty(shape, np.float32),
        "old_drop_stop": np.empty(shape, np.float32),
        "eb_stop": np.empty(shape, np.float32),
        "pilot_phat": np.empty(shape, np.float32),
    }

    for j in range(MC_K):
        x = rng.random((MC_TRIALS, MC_M), dtype=np.float32) < p_true[j]
        p_hat = np.cumsum(x, axis=1, dtype=np.float32) / ms[None, :]

        # Adversary 1: optimal stopping. Pick the m in [1, M] that minimises
        # the UCB. This is the strongest possible early-stopping attack --
        # it stops exactly on the most favourable draw, in hindsight.
        out["new_stop"][:, j] = np.minimum(1.0, p_hat + w_new[None, :]).min(axis=1)
        out["old_stop"][:, j] = np.minimum(1.0, p_hat + w_old[None, :]).min(axis=1)
        out["old_drop_stop"][:, j] = np.minimum(
            1.0, p_hat + w_old_drop[None, :]
        ).min(axis=1)

        # Empirical Bernstein under the same optimal-stopping adversary.
        var = np.zeros_like(p_hat)
        var[:, 1:] = (ms[1:] / (ms[1:] - 1.0))[None, :] * p_hat[:, 1:] * (
            1.0 - p_hat[:, 1:]
        )
        eb_w = np.sqrt(2.0 * var * log_term / ms[None, :]) + eb_rem[None, :]
        out["eb_stop"][:, j] = np.minimum(1.0, p_hat + eb_w).min(axis=1)

        # Adversary 2 inputs: a pilot look, then reallocation.
        out["pilot_phat"][:, j] = p_hat[:, MC_PILOT - 1]
        out["new_pilot"][:, j] = np.minimum(
            1.0, p_hat[:, :MC_PILOT] + w_new[None, :MC_PILOT]
        ).min(axis=1)

        del x, p_hat, var, eb_w

    out["p_true"] = p_true
    out["weights"] = weights
    return out


def _simultaneous_coverage(ucb: np.ndarray, p_true: np.ndarray) -> float:
    """Fraction of trials where ``p_j <= UCB_j`` holds for EVERY region."""
    failed = (p_true[None, :] > ucb + 1e-6).any(axis=1)
    return 1.0 - float(failed.mean())


class TestTheorem1Coverage:
    """Monte-Carlo verification of Theorem 1 under adversarial allocation."""

    def test_coverage_under_optimal_stopping_adversary(self, mc_draws):
        """H1 / criterion C3: the anytime bound survives optimal stopping.

        The adversary sees the whole sample path and picks, per region, the
        ``m`` minimising the UCB.  No allocator can do better than this at
        breaking the bound, so coverage here lower-bounds coverage under any
        real allocation rule.
        """
        n_failed_trials = 0
        ucb = mc_draws["new_stop"]
        p_true = mc_draws["p_true"]
        for t in range(MC_TRIALS):
            if bool((p_true > ucb[t] + 1e-6).any()):
                n_failed_trials += 1

        coverage = 1.0 - n_failed_trials / MC_TRIALS
        # The counter is READ and asserted on. This is the assertion the old
        # test_soundness_simulation was missing.
        assert coverage >= 1.0 - MC_DELTA, (
            f"Theorem 1 coverage {coverage:.4f} fell below 1-delta="
            f"{1 - MC_DELTA:.4f} over {MC_TRIALS} trials "
            f"({n_failed_trials} failures)"
        )

    def test_coverage_under_adaptive_reallocation_adversary(self, mc_draws):
        """Data-dependent reallocation: spend budget where p_hat looks lowest.

        Protocol per trial: take a pilot of ``MC_PILOT`` samples in every
        region, rank regions by pilot ``p_hat``, give the full budget ``M`` to
        the half whose pilot rate is LOWEST (so their favourable-looking
        estimate gets the narrowest width) and leave the rest at the pilot
        size; then apply optimal stopping within whatever each region got.
        This is exactly the "re-allocate based on observed violations, stop
        early on favourable draws" attack the rebuild has to survive.
        """
        pilot = mc_draws["pilot_phat"]
        order = np.argsort(pilot, axis=1)  # ascending pilot p_hat
        half = MC_K // 2
        give_full = np.zeros((MC_TRIALS, MC_K), dtype=bool)
        rows = np.repeat(np.arange(MC_TRIALS), half)
        give_full[rows, order[:, :half].ravel()] = True

        ucb = np.where(give_full, mc_draws["new_stop"], mc_draws["new_pilot"])
        p_true = mc_draws["p_true"]

        n_failed_trials = 0
        for t in range(MC_TRIALS):
            if bool((p_true > ucb[t] + 1e-6).any()):
                n_failed_trials += 1

        coverage = 1.0 - n_failed_trials / MC_TRIALS
        assert coverage >= 1.0 - MC_DELTA, (
            f"coverage {coverage:.4f} under the reallocating adversary fell "
            f"below {1 - MC_DELTA:.4f} ({n_failed_trials}/{MC_TRIALS} failures)"
        )

    def test_aggregate_weighted_bound_covers(self, mc_draws):
        """The weighted aggregate ``sum_j w_j p_j <= sum_j w_j UCB_j``."""
        w = mc_draws["weights"]
        p_true = mc_draws["p_true"]
        bound = (w[None, :] * mc_draws["new_stop"]).sum(axis=1)
        true_agg = float((w * p_true).sum())
        n_failed = int((true_agg > bound + 1e-6).sum())
        coverage = 1.0 - n_failed / MC_TRIALS
        assert coverage >= 1.0 - MC_DELTA, (
            f"aggregate coverage {coverage:.4f} < {1 - MC_DELTA:.4f}"
        )

    # -- NEGATIVE CONTROLS --------------------------------------------------

    def test_negative_control_old_fixed_m_bound_fails(self, mc_draws):
        """NEGATIVE CONTROL 1: the old fixed-m width must NOT cover.

        Same adversary, same data, same a-priori ``K`` -- the only change is
        dropping the ``budget_cap`` factor from the log, i.e. reverting to the
        non-anytime width the old repo used.  If this passed, the coverage
        test above would be vacuous.
        """
        coverage = _simultaneous_coverage(
            mc_draws["old_stop"], mc_draws["p_true"]
        )
        assert coverage < 1.0 - MC_DELTA, (
            "NEGATIVE CONTROL FAILED TO FAIL: the old fixed-m Hoeffding width "
            f"achieved coverage {coverage:.4f} >= {1 - MC_DELTA:.4f} under the "
            "adversarial allocator, so this Monte-Carlo test has no power to "
            "detect an invalid bound. Strengthen the adversary (larger "
            "budget_cap / more regions) before trusting any positive result."
        )

    def test_negative_control_data_derived_k_fails(self, mc_draws):
        """NEGATIVE CONTROL 2: old rule with ``k`` from non-empty regions.

        This reproduces finding F10 exactly: the adversary reports only the
        ``MC_KEEP`` best-looking regions, the union-bound ``k`` shrinks to the
        number of surviving regions, and the weighted sum silently loses the
        dropped mass.  The certified aggregate then sits far below the truth.
        """
        w = mc_draws["weights"]
        p_true = mc_draws["p_true"]
        ucb = mc_draws["old_drop_stop"]

        order = np.argsort(ucb, axis=1)[:, :MC_KEEP]
        kept_ucb = np.take_along_axis(ucb, order, axis=1)
        kept_w = np.take_along_axis(np.tile(w, (MC_TRIALS, 1)), order, axis=1)
        bound = (kept_w * kept_ucb).sum(axis=1)

        true_agg = float((w * p_true).sum())
        coverage = float((true_agg <= bound + 1e-6).mean())
        assert coverage < 1.0 - MC_DELTA, (
            "NEGATIVE CONTROL FAILED TO FAIL: dropping empty regions and "
            f"deriving k from the data still covered ({coverage:.4f})."
        )

    def test_new_bound_strictly_improves_on_the_negative_control(self, mc_draws):
        """Sanity: the anytime correction is what buys the coverage."""
        cov_new = _simultaneous_coverage(
            mc_draws["new_stop"], mc_draws["p_true"]
        )
        cov_old = _simultaneous_coverage(
            mc_draws["old_stop"], mc_draws["p_true"]
        )
        assert cov_new > cov_old


class TestEmpiricalBernsteinCoverage:
    """Coverage check for the optional tighter variant."""

    def test_coverage_under_optimal_stopping(self, mc_draws):
        coverage = _simultaneous_coverage(
            mc_draws["eb_stop"], mc_draws["p_true"]
        )
        assert coverage >= 1.0 - MC_DELTA, (
            f"empirical Bernstein coverage {coverage:.4f} < {1 - MC_DELTA:.4f}; "
            "it must be un-exported and unused until this passes."
        )

    def test_matches_the_vectorised_table(self):
        """The scalar function agrees with the MC table implementation."""
        m, v = 500, 37
        log_term = math.log(2.0 * MC_K * MC_M / MC_DELTA)
        p_hat = v / m
        var = (m / (m - 1.0)) * p_hat * (1 - p_hat)
        expected = math.sqrt(2 * var * log_term / m) + 7 * log_term / (
            3 * (m - 1.0)
        )
        got = empirical_bernstein_width(v, m, MC_K, MC_M, MC_DELTA)
        assert abs(got - expected) < 1e-12

    def test_infinite_below_two_samples(self):
        assert empirical_bernstein_width(0, 0, 4, 100, 0.05) == math.inf
        assert empirical_bernstein_width(0, 1, 4, 100, 0.05) == math.inf

    def test_tighter_than_hoeffding_when_violations_are_rare(self):
        eb = empirical_bernstein_width(2, 2000, 8, 2000, 0.05)
        ho = anytime_hoeffding_width(2000, 8, 2000, 0.05)
        assert eb < ho

    def test_looser_than_hoeffding_at_small_m(self):
        eb = empirical_bernstein_width(2, 10, 8, 2000, 0.05)
        ho = anytime_hoeffding_width(10, 8, 2000, 0.05)
        assert eb > ho


# ---------------------------------------------------------------------------
# Width / UCB unit tests
# ---------------------------------------------------------------------------


class TestAnytimeWidth:
    def test_formula(self):
        m, k, cap, delta = 50, 8, 256, 0.05
        expected = math.sqrt(math.log(2 * k * cap / delta) / (2 * m))
        assert abs(anytime_hoeffding_width(m, k, cap, delta) - expected) < 1e-12

    def test_infinite_with_no_samples(self):
        assert anytime_hoeffding_width(0, 4, 64, 0.05) == math.inf

    def test_decreasing_in_m(self):
        a = anytime_hoeffding_width(10, 4, 1000, 0.05)
        b = anytime_hoeffding_width(1000, 4, 1000, 0.05)
        assert a > b

    def test_increasing_in_k_and_cap(self):
        base = anytime_hoeffding_width(64, 4, 64, 0.05)
        assert anytime_hoeffding_width(64, 40, 64, 0.05) > base
        assert anytime_hoeffding_width(64, 4, 640, 0.05) > base

    def test_wider_than_the_old_bound(self):
        """The anytime guarantee costs a sqrt(ln M) inflation and must pay it."""
        for cap in (16, 256, 4096):
            new = anytime_hoeffding_width(10, 8, cap, 0.05)
            old = legacy_fixed_m_width(10, 8, 0.05)
            assert new > old

    def test_rejects_m_above_budget_cap(self):
        with pytest.raises(ValueError, match="budget_cap"):
            anytime_hoeffding_width(101, 4, 100, 0.05)

    def test_rejects_bad_parameters(self):
        with pytest.raises(ValueError):
            anytime_hoeffding_width(10, 0, 100, 0.05)
        with pytest.raises(ValueError):
            anytime_hoeffding_width(10, 4, 0, 0.05)
        with pytest.raises(ValueError):
            anytime_hoeffding_width(10, 4, 100, 1.5)


class TestComputeUCB:
    def test_zero_samples_gives_one_not_zero(self):
        assert compute_ucb(0, 0, 8, 256, 0.05) == 1.0

    def test_clipped_to_unit_interval(self):
        assert compute_ucb(1, 1, 8, 256, 0.05) == 1.0
        assert 0.0 <= compute_ucb(0, 10000, 8, 10000, 0.05) <= 1.0

    def test_equals_width_when_no_violations(self):
        w = anytime_hoeffding_width(200, 8, 256, 0.05)
        assert abs(compute_ucb(0, 200, 8, 256, 0.05) - w) < 1e-12

    def test_monotone_in_violations(self):
        lo = compute_ucb(5, 200, 8, 256, 0.05)
        hi = compute_ucb(50, 200, 8, 256, 0.05)
        assert hi > lo

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError):
            compute_ucb(11, 10, 8, 256, 0.05)


class TestRegionStat:
    def test_p_hat(self):
        assert RegionStat(0, 0.5, 100, 10).p_hat == 0.1

    def test_p_hat_zero_samples(self):
        assert RegionStat(0, 0.5, 0, 0).p_hat == 0.0

    def test_rejects_more_violations_than_samples(self):
        with pytest.raises(ValueError):
            RegionStat(0, 0.5, 10, 11)

    def test_rejects_negative_weight(self):
        with pytest.raises(ValueError):
            RegionStat(0, -0.1, 10, 1)

    def test_frozen(self):
        rs = RegionStat(0, 0.5, 10, 1)
        with pytest.raises(Exception):
            rs.n_samples = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# F10: check_acceptance must never certify from zero evidence
# ---------------------------------------------------------------------------


class TestCheckAcceptanceF10:
    def test_empty_input_is_vacuous_and_rejected(self):
        """The old code returned (True, 0.0) here. That was finding F10."""
        res = check_acceptance([], epsilon=0.1, delta=0.05, k_total=8, budget_cap=256)
        assert res.accepted is False
        assert res.bound == 1.0
        assert res.vacuous is True

    def test_zero_total_weight_is_vacuous(self):
        stats = [RegionStat(0, 0.0, 100, 0), RegionStat(1, 0.0, 100, 0)]
        res = check_acceptance(stats, 0.1, 0.05, k_total=8, budget_cap=256)
        assert res.accepted is False
        assert res.bound == 1.0
        assert res.vacuous is True

    def test_unsampled_region_contributes_ucb_one_and_is_kept(self):
        stats = [
            RegionStat(0, 0.5, 400, 0),
            RegionStat(1, 0.5, 0, 0),  # weight > 0, no samples
        ]
        res = check_acceptance(stats, 0.1, 0.05, k_total=2, budget_cap=400)
        assert len(res.per_region) == 2
        unsampled = [r for r in res.per_region if r.region_id == 1][0]
        assert unsampled.ucb == 1.0
        assert res.bound >= 0.5
        assert res.accepted is False

    def test_missing_declared_mass_is_charged_ucb_one(self):
        """A region simply not reported cannot be silently dropped."""
        stats = [RegionStat(0, 0.7, 400, 0)]
        res = check_acceptance(stats, 0.5, 0.05, k_total=4, budget_cap=400)
        assert abs(res.unaccounted_weight - 0.3) < 1e-9
        assert res.bound >= 0.3

    def test_more_regions_than_declared_raises(self):
        stats = [RegionStat(i, 0.25, 10, 0) for i in range(4)]
        with pytest.raises(ValueError, match="k_total"):
            check_acceptance(stats, 0.1, 0.05, k_total=3, budget_cap=100)

    def test_region_id_outside_partition_raises(self):
        with pytest.raises(ValueError, match="outside the declared partition"):
            check_acceptance(
                [RegionStat(9, 1.0, 10, 0)], 0.1, 0.05, k_total=3, budget_cap=100
            )

    def test_duplicate_region_id_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            check_acceptance(
                [RegionStat(0, 0.5, 10, 0), RegionStat(0, 0.5, 10, 0)],
                0.1, 0.05, k_total=3, budget_cap=100,
            )

    def test_samples_beyond_budget_cap_raises(self):
        with pytest.raises(ValueError, match="budget_cap"):
            check_acceptance(
                [RegionStat(0, 1.0, 101, 0)], 0.1, 0.05, k_total=1, budget_cap=100
            )

    def test_k_total_is_not_derived_from_data(self):
        """Declaring a larger partition must never make the bound tighter."""
        stats = [RegionStat(0, 0.5, 500, 5), RegionStat(1, 0.5, 500, 5)]
        small = check_acceptance(stats, 0.5, 0.05, k_total=2, budget_cap=500)
        big = check_acceptance(stats, 0.5, 0.05, k_total=64, budget_cap=500)
        assert big.bound > small.bound

    def test_accept_when_clean_and_well_sampled(self):
        stats = [RegionStat(0, 0.5, 5000, 0), RegionStat(1, 0.5, 5000, 0)]
        res = check_acceptance(
            stats, epsilon=0.1, delta=0.05, k_total=2, budget_cap=5000
        )
        assert res.accepted is True
        assert 0.0 < res.bound <= 0.1
        assert res.vacuous is False

    def test_reject_when_violations_are_common(self):
        stats = [RegionStat(0, 0.5, 500, 250), RegionStat(1, 0.5, 500, 250)]
        res = check_acceptance(
            stats, epsilon=0.1, delta=0.05, k_total=2, budget_cap=500
        )
        assert res.accepted is False
        assert res.bound > 0.5

    def test_bound_never_below_weighted_p_hat(self):
        stats = [RegionStat(0, 0.4, 300, 30), RegionStat(1, 0.6, 300, 90)]
        res = check_acceptance(stats, 0.9, 0.05, k_total=2, budget_cap=300)
        weighted_p_hat = 0.4 * 0.1 + 0.6 * 0.3
        assert res.bound >= weighted_p_hat

    def test_weights_given_as_counts_are_renormalised(self):
        stats = [RegionStat(0, 300.0, 200, 0), RegionStat(1, 700.0, 200, 0)]
        res = check_acceptance(stats, 0.9, 0.05, k_total=2, budget_cap=200)
        assert abs(sum(r.weight for r in res.per_region) - 1.0) < 1e-9
        assert res.unaccounted_weight == 0.0

    def test_result_carries_the_declared_parameters(self):
        res = check_acceptance(
            [RegionStat(0, 1.0, 100, 1)], 0.5, 0.05, k_total=1, budget_cap=100
        )
        assert isinstance(res, AcceptanceResult)
        assert isinstance(res.per_region[0], RegionUCB)
        assert res.k_total == 1 and res.budget_cap == 100
        assert res.epsilon == 0.5 and res.delta == 0.05


# ---------------------------------------------------------------------------
# F14: optimal_allocation must use w^(2/3)
# ---------------------------------------------------------------------------


def _total_width(weights: np.ndarray, alloc: list[int]) -> float:
    """``sum_j w_j / sqrt(m_j)``, the quantity the allocation minimises."""
    total = 0.0
    for w, m in zip(weights, alloc):
        if w <= 0:
            continue
        total += w / math.sqrt(m) if m > 0 else float("inf")
    return total


def _proportional(weights: np.ndarray, budget: int) -> list[int]:
    """The OLD rule from finding F14: ``m_j ~ w_j``."""
    q = weights / weights.sum()
    raw = q * budget
    alloc = np.floor(raw).astype(int)
    leftover = budget - int(alloc.sum())
    if leftover:
        for j in np.argsort(-(raw - alloc))[:leftover]:
            alloc[j] += 1
    return [int(a) for a in alloc]


def _uniform(n: int, budget: int) -> list[int]:
    base, rem = divmod(budget, n)
    return [base + (1 if j < rem else 0) for j in range(n)]


class TestOptimalAllocation:
    WEIGHT_VECTORS = [
        np.array([0.7, 0.2, 0.05, 0.05]),
        np.array([0.4, 0.3, 0.2, 0.1]),
        np.array([0.9, 0.04, 0.03, 0.02, 0.01]),
        np.array([0.5, 0.25, 0.125, 0.0625, 0.0625]),
        np.array([0.34, 0.33, 0.33]),
    ]

    def test_sums_to_exactly_the_budget(self):
        for w in self.WEIGHT_VECTORS:
            for budget in (7, 10, 101, 1000, 4097):
                alloc = optimal_allocation(w, budget, k_total=len(w))
                assert sum(alloc) == budget, (w, budget, alloc)

    def test_two_thirds_rule_shape(self):
        """Allocation is proportional to w^(2/3) once the +1 floor washes out."""
        w = np.array([0.64, 0.216, 0.08, 0.064])
        budget = 100_000
        alloc = np.array(optimal_allocation(w, budget, k_total=4), dtype=float)
        target = w ** (2 / 3)
        target = target / target.sum()
        got = alloc / alloc.sum()
        assert np.allclose(got, target, atol=2e-4), (got, target)

    def test_beats_proportional_and_uniform_on_bound_width(self):
        """F14: proportional (the OLD rule) is never better and usually worse."""
        budget = 2000
        for w in self.WEIGHT_VECTORS:
            ours = optimal_allocation(w, budget, k_total=len(w))
            f_ours = _total_width(w, ours)
            f_prop = _total_width(w, _proportional(w, budget))
            f_unif = _total_width(w, _uniform(len(w), budget))
            assert f_ours <= f_prop + 1e-9, (w, f_ours, f_prop)
            assert f_ours <= f_unif + 1e-9, (w, f_ours, f_unif)

    def test_strictly_better_for_non_uniform_weights(self):
        w = np.array([0.7, 0.2, 0.05, 0.05])
        budget = 2000
        f_ours = _total_width(w, optimal_allocation(w, budget, k_total=4))
        assert f_ours < _total_width(w, _proportional(w, budget))
        assert f_ours < _total_width(w, _uniform(4, budget))

    def test_matches_uniform_for_uniform_weights(self):
        w = np.full(5, 0.2)
        assert optimal_allocation(w, 1000, k_total=5) == [200] * 5

    def test_translates_into_a_tighter_certified_bound(self):
        """End-to-end: the 2/3 rule yields a smaller ``sum_j w_j UCB_j``."""
        w = np.array([0.7, 0.2, 0.05, 0.05])
        budget, cap, delta, k = 2000, 2000, 0.05, 4
        p = np.array([0.02, 0.05, 0.10, 0.20])

        def bound_for(alloc):
            rng = np.random.default_rng(3)
            stats = []
            for j, m in enumerate(alloc):
                v = int(rng.binomial(m, p[j])) if m > 0 else 0
                stats.append(RegionStat(j, float(w[j]), int(m), v))
            return check_acceptance(stats, 1.0, delta, k, cap).bound

        b_ours = bound_for(optimal_allocation(w, budget, k_total=k))
        b_prop = bound_for(_proportional(w, budget))
        b_unif = bound_for(_uniform(4, budget))
        assert b_ours < b_prop
        assert b_ours < b_unif

    def test_every_positive_weight_region_gets_at_least_one_sample(self):
        w = np.array([0.97, 0.01, 0.01, 0.01])
        assert all(a >= 1 for a in optimal_allocation(w, 100, k_total=4))

    def test_zero_weight_regions_get_nothing(self):
        w = np.array([0.5, 0.5, 0.0])
        alloc = optimal_allocation(w, 100, k_total=3)
        assert alloc[2] == 0
        assert sum(alloc) == 100

    def test_budget_smaller_than_region_count(self):
        w = np.array([0.5, 0.3, 0.15, 0.05])
        alloc = optimal_allocation(w, 2, k_total=4)
        assert sum(alloc) == 2
        assert alloc[0] == 1 and alloc[1] == 1

    def test_rejects_more_weights_than_k_total(self):
        with pytest.raises(ValueError, match="k_total"):
            optimal_allocation(np.array([0.5, 0.5]), 10, k_total=1)

    def test_rejects_negative_weights(self):
        with pytest.raises(ValueError):
            optimal_allocation(np.array([0.5, -0.5]), 10, k_total=2)

    def test_zero_budget(self):
        assert optimal_allocation(np.array([0.5, 0.5]), 0, k_total=2) == [0, 0]


class TestClopperPearson:
    def test_no_samples_is_one(self):
        assert clopper_pearson_upper(0, 0, 0.05) == 1.0

    def test_all_violations_is_one(self):
        assert clopper_pearson_upper(10, 10, 0.05) == 1.0

    def test_upper_bound_exceeds_p_hat(self):
        assert clopper_pearson_upper(10, 100, 0.05) > 0.1

    def test_tightens_with_more_samples(self):
        a = clopper_pearson_upper(10, 100, 0.05)
        b = clopper_pearson_upper(100, 1000, 0.05)
        assert b < a

    def test_coverage_monte_carlo(self):
        """CP is exact at fixed m; verify it here so it is not taken on faith."""
        rng = np.random.default_rng(11)
        p, m, alpha, trials = 0.08, 120, 0.05, 4000
        draws = rng.binomial(m, p, size=trials)
        covered = sum(
            1 for v in draws if clopper_pearson_upper(int(v), m, alpha) >= p
        )
        assert covered / trials >= 1 - alpha


class TestProcessIndependence:
    """C4 support: nothing in this module depends on ``PYTHONHASHSEED``."""

    def test_no_builtin_hash_of_strings(self):
        import inspect

        import sca.utils.stats as stats_mod

        src = inspect.getsource(stats_mod)
        cleaned = src.replace("p_hat", "").replace("stable_hash", "")
        assert "hash(" not in cleaned

    def test_repeated_calls_are_bit_identical(self):
        stats = [RegionStat(0, 0.6, 500, 13), RegionStat(1, 0.4, 300, 7)]
        a = check_acceptance(stats, 0.2, 0.05, 4, 500)
        b = check_acceptance(stats, 0.2, 0.05, 4, 500)
        assert a.bound == b.bound
        assert [r.ucb for r in a.per_region] == [r.ucb for r in b.per_region]
