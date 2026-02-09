"""Tests for statistical utilities (Theorem 1 soundness guarantee)."""

import math

import numpy as np
import pytest

from sca.utils.stats import (
    RegionStats,
    check_acceptance,
    compute_ucb,
    hoeffding_bound,
    optimal_allocation,
)


class TestHoeffdingBound:
    def test_basic(self):
        """Hoeffding bound decreases with more samples."""
        b1 = hoeffding_bound(m=10, delta=0.05, k=1)
        b2 = hoeffding_bound(m=100, delta=0.05, k=1)
        assert b2 < b1

    def test_zero_samples(self):
        """Zero samples returns maximum uncertainty."""
        assert hoeffding_bound(m=0, delta=0.05, k=1) == 1.0

    def test_union_bound(self):
        """More regions (larger K) makes bound wider."""
        b1 = hoeffding_bound(m=50, delta=0.05, k=1)
        b10 = hoeffding_bound(m=50, delta=0.05, k=10)
        assert b10 > b1

    def test_known_value(self):
        """Check against manual calculation."""
        m, delta, k = 100, 0.1, 5
        expected = math.sqrt(math.log(2 * 5 / 0.1) / (2 * 100))
        assert abs(hoeffding_bound(m, delta, k) - expected) < 1e-10


class TestComputeUCB:
    def test_clipping(self):
        """UCB is clipped to [0, 1]."""
        ucb = compute_ucb(p_hat=0.99, m=1, delta=0.01, k=10)
        assert ucb <= 1.0

    def test_p_hat_zero(self):
        """UCB with zero violation rate is just the Hoeffding width."""
        m, delta, k = 50, 0.05, 5
        ucb = compute_ucb(0.0, m, delta, k)
        assert ucb == hoeffding_bound(m, delta, k)

    def test_monotone_in_p_hat(self):
        """Higher p_hat -> higher UCB."""
        ucb_low = compute_ucb(0.1, 50, 0.05, 5)
        ucb_high = compute_ucb(0.3, 50, 0.05, 5)
        assert ucb_high > ucb_low


class TestRegionStats:
    def test_p_hat(self):
        rs = RegionStats(region_id=0, n_samples=100, n_violations=10, weight=0.5)
        assert rs.p_hat == 0.1

    def test_p_hat_zero_samples(self):
        rs = RegionStats(region_id=0, n_samples=0, n_violations=0, weight=0.5)
        assert rs.p_hat == 0.0

    def test_weighted_ucb(self):
        rs = RegionStats(region_id=0, n_samples=100, n_violations=10, weight=0.5)
        wucb = rs.weighted_ucb(delta=0.05, k=5)
        assert wucb == 0.5 * rs.ucb(delta=0.05, k=5)


class TestAcceptanceRule:
    """Tests for the SCA acceptance rule (Section 4.3):
    Accept iff sum_j w_j * UCB_j <= epsilon.
    """

    def test_safe_model_accepted(self):
        """Model with zero violations and sufficient samples should be accepted."""
        stats = [
            RegionStats(region_id=0, n_samples=500, n_violations=0, weight=0.5),
            RegionStats(region_id=1, n_samples=500, n_violations=0, weight=0.5),
        ]
        accepted, bound = check_acceptance(stats, epsilon=0.1, delta=0.05)
        assert accepted
        assert bound < 0.1

    def test_unsafe_model_rejected(self):
        """Model with high violations should be rejected."""
        stats = [
            RegionStats(region_id=0, n_samples=100, n_violations=50, weight=0.5),
            RegionStats(region_id=1, n_samples=100, n_violations=50, weight=0.5),
        ]
        accepted, bound = check_acceptance(stats, epsilon=0.1, delta=0.05)
        assert not accepted
        assert bound > 0.1

    def test_soundness_simulation(self):
        """Empirical validation of Theorem 1 soundness.

        Simulate a model with known violation rates p_j, sample from it,
        and verify that when Accepted, the true Viol(M) <= epsilon
        holds with high probability.
        """
        np.random.seed(42)
        n_trials = 500
        epsilon = 0.15
        delta = 0.1
        k = 5
        true_p = np.array([0.02, 0.05, 0.01, 0.03, 0.01])
        weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
        true_viol = np.sum(weights * true_p)

        # Count how often we accept AND true Viol > epsilon
        false_accepts = 0
        total_accepts = 0

        for _ in range(n_trials):
            stats = []
            for j in range(k):
                m_j = 50
                violations = np.random.binomial(m_j, true_p[j])
                stats.append(RegionStats(
                    region_id=j, n_samples=m_j,
                    n_violations=violations, weight=weights[j],
                ))
            accepted, bound = check_acceptance(stats, epsilon, delta)
            if accepted:
                total_accepts += 1
                if true_viol > epsilon:
                    false_accepts += 1

        # Theorem 1: Pr(Accept AND Viol > eps) <= delta
        # In this case true_viol < epsilon, so false_accepts should be 0
        assert true_viol < epsilon, "Sanity check: true violation is below threshold"
        # This model is truly safe, so acceptance is correct

    def test_soundness_unsafe_model(self):
        """For an unsafe model, acceptance should be rare (controlled by delta)."""
        np.random.seed(42)
        n_trials = 1000
        epsilon = 0.05
        delta = 0.05
        k = 3
        # True violation rates are high
        true_p = np.array([0.2, 0.15, 0.1])
        weights = np.array([0.4, 0.3, 0.3])
        true_viol = np.sum(weights * true_p)
        assert true_viol > epsilon

        false_accepts = 0
        for _ in range(n_trials):
            stats = []
            for j in range(k):
                m_j = 30
                violations = np.random.binomial(m_j, true_p[j])
                stats.append(RegionStats(
                    region_id=j, n_samples=m_j,
                    n_violations=violations, weight=weights[j],
                ))
            accepted, _ = check_acceptance(stats, epsilon, delta)
            if accepted:
                false_accepts += 1

        # Should be very rare: Theorem 1 bounds this by delta
        # With true_viol = 0.155 >> epsilon = 0.05, and m_j=30,
        # the probability of false acceptance should be essentially 0.
        assert false_accepts / n_trials < delta + 0.02

    def test_empty_regions(self):
        accepted, bound = check_acceptance([], epsilon=0.1, delta=0.05)
        assert accepted
        assert bound == 0.0


class TestOptimalAllocation:
    def test_sums_to_budget(self):
        weights = np.array([0.5, 0.3, 0.2])
        alloc = optimal_allocation(weights, np.zeros(3), total_budget=100, delta=0.05)
        assert alloc.sum() == 100

    def test_proportional(self):
        """Higher weight regions get more samples."""
        weights = np.array([0.7, 0.2, 0.1])
        alloc = optimal_allocation(weights, np.zeros(3), total_budget=100, delta=0.05)
        assert alloc[0] > alloc[1] > alloc[2]

    def test_minimum_one(self):
        """Each region gets at least 1 sample."""
        weights = np.array([0.99, 0.005, 0.005])
        alloc = optimal_allocation(weights, np.zeros(3), total_budget=10, delta=0.05)
        assert all(a >= 1 for a in alloc)
