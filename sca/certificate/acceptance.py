"""Acceptance rule implementation (Section 4.3).

The SCA acceptance rule:
    Accept iff sum_j w_j * UCB_j <= epsilon

where UCB_j = p_hat_j + sqrt(ln(2K / delta) / (2 * m_j)).

This module provides the high-level interface that combines verification
results into an accept/reject decision with certificate generation.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from sca.certificate.certificate import SafetyCertificate, build_certificate
from sca.knowledge_graph.mkg import ModelKnowledgeGraph
from sca.utils.stats import RegionStats, check_acceptance
from sca.verifier.rlm_verifier import RLMVerifier, VerifierState


class AcceptanceGate:
    """High-level acceptance gate combining verification and certification.

    Implements the commit-gated FL protocol (Section 4.1):
    1. Run verifier on candidate model.
    2. Compute acceptance decision.
    3. Build certificate.
    4. Commit or rollback.
    """

    def __init__(
        self,
        verifier: RLMVerifier,
        epsilon: float = 0.05,
        delta: float = 0.01,
    ) -> None:
        """
        Args:
            verifier: The RLM verifier instance.
            epsilon: Target violation bound.
            delta: Confidence parameter.
        """
        self.verifier = verifier
        self.epsilon = epsilon
        self.delta = delta
        self.certificates: list[SafetyCertificate] = []
        self.prev_stats: list[RegionStats] | None = None

    def evaluate(
        self,
        model_fn: Callable[[dict], str],
        model_params: np.ndarray | Any,
        seed_interactions: list[dict],
        fl_round: int = 0,
    ) -> tuple[bool, SafetyCertificate]:
        """Evaluate a candidate model through the acceptance gate.

        Args:
            model_fn: Callable for model inference.
            model_params: Model parameters (for hashing/commitment).
            seed_interactions: Seed interactions for verification.
            fl_round: Current FL round number.

        Returns:
            (accepted, certificate): Whether the model was accepted and
            the generated safety certificate.
        """
        # Run verification
        accepted, state, region_stats = self.verifier.verify(
            model_fn=model_fn,
            seed_interactions=seed_interactions,
            epsilon=self.epsilon,
            delta=self.delta,
        )

        # Build trace dicts for Merkle tree
        trace_dicts = [
            {
                "interaction": t.interaction,
                "output": t.output,
                "violation": t.violation,
                "region_id": t.region_id,
                "depth": t.depth,
                "mutation_type": t.mutation_type,
            }
            for t in state.traces
        ]

        # Build certificate
        cert = build_certificate(
            model_params=model_params,
            verifier_descriptor=self.verifier.get_verifier_descriptor(),
            graph_summary=self.verifier.mkg.summary(),
            region_stats=region_stats,
            epsilon=self.epsilon,
            delta=self.delta,
            traces=trace_dicts,
            fl_round=fl_round,
        )

        self.certificates.append(cert)

        # Compute regression subgraph if we have previous stats
        if self.prev_stats is not None:
            regression = self.verifier.mkg.compute_regression_subgraph(
                self.prev_stats, region_stats, self.delta,
            )
            if regression:
                cert_dict = cert.to_dict()
                cert_dict["regression_regions"] = regression

        # Save current stats for next round comparison
        self.prev_stats = region_stats

        return accepted, cert
