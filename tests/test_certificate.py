"""Tests for the certificate schema, independent verification and the gate.

The old ``verify_certificate_consistency`` compared two values that came out
of the SAME ``check_acceptance`` call, so it could not detect a wrong UCB or a
tampered count.  :class:`TestIndependentVerification` mutates individual
stored fields -- including a single UCB by ``1e-6`` -- and requires
verification to fail every time.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from sca.certificate.acceptance import AcceptanceGate, GateDecision, GateMode
from sca.certificate.certificate import (
    SafetyCertificate,
    build_certificate,
    certificate_binding_hash,
    verify_certificate,
    verify_certificate_consistency,
)
from sca.utils.stats import RegionStat, check_acceptance

K_TOTAL = 4
BUDGET_CAP = 2000
EPSILON = 0.10
DELTA = 0.05


def _clean_stats() -> list[RegionStat]:
    """Four well-sampled regions with very few violations -> accept."""
    return [
        RegionStat(region_id=0, weight=0.40, n_samples=2000, n_violations=2),
        RegionStat(region_id=1, weight=0.30, n_samples=2000, n_violations=1),
        RegionStat(region_id=2, weight=0.20, n_samples=2000, n_violations=0),
        RegionStat(region_id=3, weight=0.10, n_samples=2000, n_violations=3),
    ]


def _dirty_stats() -> list[RegionStat]:
    return [
        RegionStat(region_id=0, weight=0.5, n_samples=500, n_violations=250),
        RegionStat(region_id=1, weight=0.5, n_samples=500, n_violations=200),
    ]


def _build(stats, **kw) -> SafetyCertificate:
    params = dict(
        model_params=np.array([1.0, 2.0, 3.0]),
        verifier_descriptor={"type": "test", "depth": 3},
        graph_summary={"n_regions": K_TOTAL, "n_edges": 5},
        estimation_stats=stats,
        epsilon=EPSILON,
        delta=DELTA,
        k_total=K_TOTAL,
        budget_cap=BUDGET_CAP,
        traces=[{"prompt": f"t{i}", "output": "safe"} for i in range(5)],
        fl_round=1,
    )
    params.update(kw)
    return build_certificate(**params)


class TestCertificateConstruction:
    def test_accepted_certificate(self):
        cert = _build(_clean_stats())
        assert cert.decision == "accept"
        assert cert.is_accepted
        assert cert.bound_value <= EPSILON
        assert not cert.vacuous
        assert cert.k_total == K_TOTAL
        assert cert.budget_cap == BUDGET_CAP
        assert cert.n_estimation_queries == 8000

    def test_rejected_certificate(self):
        cert = _build(_dirty_stats(), k_total=2, budget_cap=500)
        assert cert.decision == "reject"
        assert cert.bound_value > EPSILON

    def test_raw_counts_are_present(self):
        """A verifier must be able to re-derive everything from the counts."""
        cert = _build(_clean_stats())
        assert cert.region_ids == [0, 1, 2, 3]
        assert cert.region_n_samples == [2000] * 4
        assert cert.region_n_violations == [2, 1, 0, 3]
        assert cert.region_weights == [0.40, 0.30, 0.20, 0.10]

    def test_ucbs_match_the_stats_module(self):
        stats = _clean_stats()
        cert = _build(stats)
        expected = check_acceptance(stats, EPSILON, DELTA, K_TOTAL, BUDGET_CAP)
        assert cert.ucbs == [r.ucb for r in expected.per_region]
        assert cert.bound_value == pytest.approx(expected.bound, abs=0.0)

    def test_json_round_trip(self):
        cert = _build(_clean_stats())
        restored = SafetyCertificate.from_dict(json.loads(cert.to_json()))
        assert restored.certificate_hash() == cert.certificate_hash()
        assert verify_certificate_consistency(restored)

    def test_trace_digest_commits_to_traces(self):
        a = _build(_clean_stats(), traces=[{"a": 1}])
        b = _build(_clean_stats(), traces=[{"a": 2}])
        assert a.trace_digest != b.trace_digest
        assert a.binding_hash != b.binding_hash

    def test_search_metrics_are_recorded_but_not_certified(self):
        """Deduplication (F8): raw and deduped counts sit side by side."""
        cert = _build(
            _clean_stats(),
            search_metrics={
                "total_violations_raw": 131,
                "distinct_violations_dedup_depth0": 9,
            },
            n_search_queries=4210,
        )
        assert cert.search_metrics["total_violations_raw"] == 131
        assert cert.search_metrics["distinct_violations_dedup_depth0"] == 9
        assert cert.n_search_queries == 4210
        # The search numbers do not move the bound.
        plain = _build(_clean_stats())
        assert cert.bound_value == plain.bound_value

    def test_vacuous_certificate_from_no_evidence(self):
        """F10 end-to-end: no regions must not produce an accept."""
        cert = _build([])
        assert cert.vacuous is True
        assert cert.decision == "reject"
        assert cert.bound_value == 1.0

    def test_unsampled_region_forces_reject(self):
        stats = _clean_stats()
        stats[2] = RegionStat(region_id=2, weight=0.20, n_samples=0, n_violations=0)
        cert = _build(stats)
        assert cert.ucbs[2] == 1.0
        assert cert.decision == "reject"


class TestIndependentVerification:
    """Verification must recompute, not re-read."""

    def test_valid_certificate_verifies(self):
        ok, reasons = verify_certificate(_build(_clean_stats()))
        assert ok, reasons

    def test_ucb_mutated_by_1e_6_fails(self):
        """The exact test the task specifies."""
        cert = _build(_clean_stats())
        assert verify_certificate_consistency(cert)
        tampered = copy.deepcopy(cert)
        tampered.ucbs[1] = tampered.ucbs[1] - 1e-6
        ok, reasons = verify_certificate(tampered)
        assert not ok
        assert any("stored UCB" in r for r in reasons), reasons

    def test_ucb_mutated_upward_also_fails(self):
        cert = _build(_clean_stats())
        tampered = copy.deepcopy(cert)
        tampered.ucbs[0] = tampered.ucbs[0] + 1e-6
        assert not verify_certificate_consistency(tampered)

    def test_bound_value_tampering_fails(self):
        cert = _build(_dirty_stats(), k_total=2, budget_cap=500)
        tampered = copy.deepcopy(cert)
        tampered.bound_value = 0.01
        tampered.decision = "accept"
        ok, reasons = verify_certificate(tampered)
        assert not ok
        assert any("bound_value" in r for r in reasons), reasons

    def test_p_hat_tampering_fails(self):
        cert = _build(_clean_stats())
        tampered = copy.deepcopy(cert)
        tampered.p_hats[0] = 0.0
        ok, reasons = verify_certificate(tampered)
        assert not ok
        assert any("p_hat" in r for r in reasons), reasons

    def test_count_tampering_breaks_the_binding_hash(self):
        """Lowering a violation count must not silently produce a valid cert."""
        cert = _build(_dirty_stats(), k_total=2, budget_cap=500)
        tampered = copy.deepcopy(cert)
        tampered.region_n_violations[0] = 0
        ok, reasons = verify_certificate(tampered)
        assert not ok
        assert any("binding_hash" in r for r in reasons), reasons

    def test_weight_tampering_breaks_the_binding_hash(self):
        cert = _build(_clean_stats())
        tampered = copy.deepcopy(cert)
        tampered.region_weights[3] = 0.05
        ok, reasons = verify_certificate(tampered)
        assert not ok
        assert any("binding_hash" in r for r in reasons), reasons

    def test_parameter_tampering_fails(self):
        """Shrinking k_total after the fact would loosen the union bound."""
        cert = _build(_clean_stats())
        tampered = copy.deepcopy(cert)
        tampered.k_total = 2
        ok, reasons = verify_certificate(tampered)
        assert not ok

    def test_decision_flip_fails(self):
        cert = _build(_dirty_stats(), k_total=2, budget_cap=500)
        tampered = copy.deepcopy(cert)
        tampered.decision = "accept"
        ok, reasons = verify_certificate(tampered)
        assert not ok
        assert any("does not follow" in r for r in reasons), reasons

    def test_model_hash_tampering_breaks_the_binding(self):
        cert = _build(_clean_stats())
        tampered = copy.deepcopy(cert)
        tampered.model_hash = "0" * 64
        assert not verify_certificate_consistency(tampered)

    def test_trace_digest_tampering_breaks_the_binding(self):
        cert = _build(_clean_stats())
        tampered = copy.deepcopy(cert)
        tampered.trace_digest = "0" * 64
        assert not verify_certificate_consistency(tampered)

    def test_samples_beyond_budget_cap_rejected(self):
        cert = _build(_clean_stats())
        tampered = copy.deepcopy(cert)
        tampered.budget_cap = 100
        ok, reasons = verify_certificate(tampered)
        assert not ok

    def test_length_mismatch_rejected(self):
        cert = _build(_clean_stats())
        tampered = copy.deepcopy(cert)
        tampered.ucbs.pop()
        ok, reasons = verify_certificate(tampered)
        assert not ok
        assert any("length mismatch" in r for r in reasons), reasons

    def test_binding_hash_is_order_independent(self):
        args = dict(
            k_total=4, budget_cap=100, epsilon=0.1, delta=0.05,
            model_hash="m", verifier_hash="v", graph_hash="g", trace_digest="t",
        )
        a = certificate_binding_hash(
            [0, 1, 2], [0.5, 0.3, 0.2], [10, 20, 30], [1, 2, 3], **args
        )
        b = certificate_binding_hash(
            [2, 1, 0], [0.2, 0.3, 0.5], [30, 20, 10], [3, 2, 1], **args
        )
        assert a == b

    def test_binding_hash_is_content_sensitive(self):
        args = dict(
            k_total=4, budget_cap=100, epsilon=0.1, delta=0.05,
            model_hash="m", verifier_hash="v", graph_hash="g", trace_digest="t",
        )
        a = certificate_binding_hash(
            [0, 1], [0.5, 0.5], [10, 20], [1, 2], **args
        )
        b = certificate_binding_hash(
            [0, 1], [0.5, 0.5], [10, 20], [1, 3], **args
        )
        assert a != b


class TestAcceptanceGate:
    def test_certified_mode_accepts_clean(self):
        gate = AcceptanceGate(EPSILON, DELTA, K_TOTAL, BUDGET_CAP)
        d = gate.evaluate_stats(_clean_stats())
        assert isinstance(d, GateDecision)
        assert d.accepted is True
        assert d.certificate.decision == "accept"
        assert d.statistical_accept is True

    def test_certified_mode_rejects_dirty(self):
        gate = AcceptanceGate(EPSILON, DELTA, k_total=2, budget_cap=500)
        d = gate.evaluate_stats(_dirty_stats())
        assert d.accepted is False

    def test_always_accept_control(self):
        """MANDATORY CONTROL: must accept even when the bound says reject."""
        gate = AcceptanceGate(
            EPSILON, DELTA, k_total=2, budget_cap=500, mode=GateMode.ALWAYS_ACCEPT
        )
        d = gate.evaluate_stats(_dirty_stats())
        assert d.accepted is True
        assert d.certificate.decision == "accept"
        # the true statistical verdict is still recorded, truthfully
        assert d.statistical_accept is False
        assert d.certificate.bound_value > EPSILON
        assert verify_certificate_consistency(d.certificate)

    def test_always_reject_control(self):
        """MANDATORY CONTROL: must reject even when the bound says accept."""
        gate = AcceptanceGate(
            EPSILON, DELTA, K_TOTAL, BUDGET_CAP, mode=GateMode.ALWAYS_REJECT
        )
        d = gate.evaluate_stats(_clean_stats())
        assert d.accepted is False
        assert d.certificate.decision == "reject"
        assert d.statistical_accept is True
        assert verify_certificate_consistency(d.certificate)

    def test_control_modes_do_not_change_the_recorded_bound(self):
        stats = _clean_stats()
        bounds = set()
        for mode in GateMode:
            gate = AcceptanceGate(EPSILON, DELTA, K_TOTAL, BUDGET_CAP, mode=mode)
            bounds.add(gate.evaluate_stats(stats).certificate.bound_value)
        assert len(bounds) == 1

    def test_round_union_bound_widens_the_bound(self):
        stats = _clean_stats()
        one = AcceptanceGate(EPSILON, DELTA, K_TOTAL, BUDGET_CAP).evaluate_stats(stats)
        ten = AcceptanceGate(
            EPSILON, DELTA, K_TOTAL, BUDGET_CAP, rounds_for_union_bound=10
        ).evaluate_stats(stats)
        assert ten.certificate.bound_value > one.certificate.bound_value
        assert ten.certificate.delta == pytest.approx(DELTA / 10)

    def test_evaluate_uses_only_estimation_stats(self):
        """Search counters must not move the bound (finding F3)."""

        class FakeResult:
            estimation_stats = _clean_stats()
            search_stats = [RegionStat(0, 0.4, 900, 900)]  # all violations
            total_violations = 131
            distinct_violations = 9
            distinct_sources = 9
            search_queries = 4210
            estimation_queries = 8000
            recursion_depth_hist = {0: 9, 1: 40, 2: 82}
            allocation = {0: 2000, 1: 2000, 2: 2000, 3: 2000}

        gate = AcceptanceGate(EPSILON, DELTA, K_TOTAL, BUDGET_CAP)
        with_search = gate.evaluate(FakeResult())
        baseline = AcceptanceGate(
            EPSILON, DELTA, K_TOTAL, BUDGET_CAP
        ).evaluate_stats(_clean_stats())
        assert with_search.certificate.bound_value == baseline.certificate.bound_value
        assert with_search.accepted is True
        sm = with_search.certificate.search_metrics
        assert sm["total_violations_raw"] == 131
        assert sm["distinct_violations_dedup_depth0"] == 9
        assert sm["search_queries"] == 4210

    def test_verify_all(self):
        gate = AcceptanceGate(EPSILON, DELTA, K_TOTAL, BUDGET_CAP)
        gate.evaluate_stats(_clean_stats())
        gate.evaluate_stats(_dirty_stats()[:1] + [
            RegionStat(1, 0.5, 500, 200)
        ], fl_round=2)
        ok, problems = gate.verify_all()
        assert ok, problems

    def test_rejects_bad_round_count(self):
        with pytest.raises(ValueError):
            AcceptanceGate(rounds_for_union_bound=0)


class TestDeterminism:
    """C4 support: the certificate path is process-independent."""

    def test_two_builds_are_byte_identical(self):
        a = _build(_clean_stats())
        b = _build(_clean_stats())
        a_d, b_d = a.to_dict(), b.to_dict()
        a_d.pop("timestamp")
        b_d.pop("timestamp")
        assert a_d == b_d

    def test_binding_hash_stable_across_processes(self):
        import os
        import pathlib
        import subprocess
        import sys

        import sca

        root = str(pathlib.Path(sca.__file__).resolve().parent.parent)
        code = (
            f"import sys; sys.path.insert(0, {root!r});"
            "from sca.certificate.certificate import certificate_binding_hash;"
            "print(certificate_binding_hash([0,1],[0.5,0.5],[10,20],[1,2],"
            "4,100,0.1,0.05,'m','v','g','t'))"
        )
        outs = set()
        for seed in ("0", "7", "99991"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outs.add(
                subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True, text=True, env=env, check=True,
                ).stdout.strip()
            )
        assert len(outs) == 1, outs
