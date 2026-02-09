"""Tests for certificate schema and acceptance gate."""

import numpy as np
import pytest

from sca.certificate.certificate import (
    SafetyCertificate,
    build_certificate,
    verify_certificate_consistency,
)
from sca.utils.stats import RegionStats


class TestCertificateConstruction:
    def test_build_accepted(self):
        """Build a certificate for an accepted model."""
        stats = [
            RegionStats(region_id=0, n_samples=500, n_violations=2, weight=0.5),
            RegionStats(region_id=1, n_samples=500, n_violations=1, weight=0.5),
        ]
        cert = build_certificate(
            model_params=np.array([1.0, 2.0, 3.0]),
            verifier_descriptor={"type": "test", "depth": 3},
            graph_summary={"n_regions": 2, "n_edges": 1},
            region_stats=stats,
            epsilon=0.1,
            delta=0.05,
            traces=[{"prompt": "test", "output": "safe"}],
            fl_round=1,
        )
        assert cert.is_accepted
        assert cert.decision == "accept"
        assert cert.n_regions == 2
        assert len(cert.p_hats) == 2
        assert len(cert.ucbs) == 2

    def test_build_rejected(self):
        """Build a certificate for a rejected model."""
        stats = [
            RegionStats(region_id=0, n_samples=100, n_violations=50, weight=0.5),
            RegionStats(region_id=1, n_samples=100, n_violations=40, weight=0.5),
        ]
        cert = build_certificate(
            model_params=np.array([1.0, 2.0]),
            verifier_descriptor={"type": "test"},
            graph_summary={"n_regions": 2},
            region_stats=stats,
            epsilon=0.05,
            delta=0.05,
            traces=[],
        )
        assert not cert.is_accepted
        assert cert.decision == "reject"

    def test_serialization(self):
        """Certificate can be serialized and deserialized."""
        stats = [
            RegionStats(region_id=0, n_samples=50, n_violations=5, weight=1.0),
        ]
        cert = build_certificate(
            model_params="precomputed_hash",
            verifier_descriptor={"type": "test"},
            graph_summary={},
            region_stats=stats,
            epsilon=0.2,
            delta=0.1,
            traces=[],
        )
        json_str = cert.to_json()
        assert "accept" in json_str or "reject" in json_str

        d = cert.to_dict()
        cert2 = SafetyCertificate.from_dict(d)
        assert cert2.epsilon == cert.epsilon
        assert cert2.model_hash == cert.model_hash


class TestCertificateVerification:
    def test_consistent_accept(self):
        cert = SafetyCertificate(
            model_hash="abc",
            verifier_hash="def",
            graph_hash="ghi",
            p_hats=[0.05],
            ucbs=[0.08],
            region_weights=[1.0],
            region_sample_counts=[100],
            epsilon=0.1,
            delta=0.05,
            decision="accept",
            bound_value=0.08,
            trace_digest="merkle_root",
            n_total_queries=100,
            n_regions=1,
        )
        assert verify_certificate_consistency(cert)

    def test_inconsistent_decision(self):
        """Reject should not have bound <= epsilon."""
        cert = SafetyCertificate(
            model_hash="abc",
            verifier_hash="def",
            graph_hash="ghi",
            p_hats=[0.05],
            ucbs=[0.08],
            region_weights=[1.0],
            region_sample_counts=[100],
            epsilon=0.1,
            delta=0.05,
            decision="reject",  # Inconsistent: bound 0.08 <= eps 0.1
            bound_value=0.08,
            trace_digest="merkle_root",
            n_total_queries=100,
            n_regions=1,
        )
        assert not verify_certificate_consistency(cert)

    def test_invalid_p_hat(self):
        cert = SafetyCertificate(
            model_hash="abc",
            verifier_hash="def",
            graph_hash="ghi",
            p_hats=[-0.1],  # Invalid
            ucbs=[0.08],
            region_weights=[1.0],
            region_sample_counts=[100],
            epsilon=0.1,
            delta=0.05,
            decision="accept",
            bound_value=0.08,
            trace_digest="merkle_root",
            n_total_queries=100,
            n_regions=1,
        )
        assert not verify_certificate_consistency(cert)

    def test_certificate_hash_deterministic(self):
        cert = SafetyCertificate(
            model_hash="abc",
            verifier_hash="def",
            graph_hash="ghi",
            p_hats=[0.05],
            ucbs=[0.08],
            region_weights=[1.0],
            region_sample_counts=[100],
            epsilon=0.1,
            delta=0.05,
            decision="accept",
            bound_value=0.08,
            trace_digest="merkle_root",
            n_total_queries=100,
            n_regions=1,
            timestamp=1000.0,
        )
        h1 = cert.certificate_hash()
        h2 = cert.certificate_hash()
        assert h1 == h2
