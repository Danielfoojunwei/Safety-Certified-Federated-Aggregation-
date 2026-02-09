"""Safety certificate schema and construction (Section 4.2).

Certificate:
    C = (h(theta), h(V), h(G), {p_hat_j}, {UCB_j}, epsilon, delta, Decision, TraceDigest)

- h(theta): hash of model parameters
- h(V): hash of verifier policy
- h(G): hash of MKG state
- {p_hat_j}: empirical violation rates per region
- {UCB_j}: upper confidence bounds per region
- epsilon, delta: safety parameters
- Decision: Accept/Reject
- TraceDigest: Merkle root of the recursion tree
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from sca.utils.crypto import MerkleTree, hash_object, hash_tensor
from sca.utils.stats import RegionStats, check_acceptance


@dataclass
class SafetyCertificate:
    """Cryptographic safety certificate (Section 4.2).

    All fields are serializable and hashable for commitment.
    """
    # Commitments
    model_hash: str
    verifier_hash: str
    graph_hash: str

    # Per-region statistics
    p_hats: list[float]
    ucbs: list[float]
    region_weights: list[float]
    region_sample_counts: list[int]

    # Parameters
    epsilon: float
    delta: float

    # Decision
    decision: str  # "accept" or "reject"
    bound_value: float  # sum_j w_j * UCB_j

    # Trace
    trace_digest: str  # Merkle root of the verification trace
    n_total_queries: int
    n_regions: int

    # Metadata
    timestamp: float = field(default_factory=time.time)
    fl_round: int = 0

    def to_dict(self) -> dict:
        """Serialize certificate to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize certificate to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> SafetyCertificate:
        """Deserialize from dictionary."""
        return cls(**d)

    @property
    def is_accepted(self) -> bool:
        return self.decision == "accept"

    def certificate_hash(self) -> str:
        """Compute the hash of the certificate itself."""
        return hash_object(self.to_dict())


def build_certificate(
    model_params: np.ndarray | Any,
    verifier_descriptor: dict,
    graph_summary: dict,
    region_stats: list[RegionStats],
    epsilon: float,
    delta: float,
    traces: list[dict],
    fl_round: int = 0,
) -> SafetyCertificate:
    """Construct a safety certificate from verification results.

    This implements the full certificate schema from Section 4.2.

    Args:
        model_params: Model parameters theta (or a hash thereof).
        verifier_descriptor: Serializable descriptor of the verifier V.
        graph_summary: Summary dict of the MKG state.
        region_stats: Per-region statistics from verification.
        epsilon: Target violation bound.
        delta: Confidence parameter.
        traces: List of trace entry dicts for Merkle construction.
        fl_round: Current FL round number.

    Returns:
        A SafetyCertificate.
    """
    k = len(region_stats)

    # Compute hashes
    if isinstance(model_params, str):
        model_hash = model_params  # Already a hash
    elif isinstance(model_params, np.ndarray):
        model_hash = hash_tensor(model_params)
    else:
        model_hash = hash_object(model_params)

    verifier_hash = hash_object(verifier_descriptor)
    graph_hash = hash_object(graph_summary)

    # Per-region statistics
    p_hats = [rs.p_hat for rs in region_stats]
    ucbs = [rs.ucb(delta, k) for rs in region_stats]
    weights = [rs.weight for rs in region_stats]
    sample_counts = [rs.n_samples for rs in region_stats]

    # Acceptance decision
    accepted, bound_value = check_acceptance(region_stats, epsilon, delta)
    decision = "accept" if accepted else "reject"

    # Trace digest via Merkle tree
    merkle = MerkleTree(traces)
    trace_digest = merkle.root_hash

    total_queries = sum(rs.n_samples for rs in region_stats)

    return SafetyCertificate(
        model_hash=model_hash,
        verifier_hash=verifier_hash,
        graph_hash=graph_hash,
        p_hats=p_hats,
        ucbs=ucbs,
        region_weights=weights,
        region_sample_counts=sample_counts,
        epsilon=epsilon,
        delta=delta,
        decision=decision,
        bound_value=bound_value,
        trace_digest=trace_digest,
        n_total_queries=total_queries,
        n_regions=k,
        fl_round=fl_round,
    )


def verify_certificate_consistency(cert: SafetyCertificate) -> bool:
    """Check internal consistency of a certificate.

    Validates that the decision matches the bound_value vs epsilon,
    and that statistics are well-formed.

    Args:
        cert: The certificate to verify.

    Returns:
        True if certificate is internally consistent.
    """
    # Check decision matches bound
    if cert.decision == "accept" and cert.bound_value > cert.epsilon:
        return False
    if cert.decision == "reject" and cert.bound_value <= cert.epsilon:
        return False

    # Check lengths match
    if not (len(cert.p_hats) == len(cert.ucbs) == len(cert.region_weights)
            == len(cert.region_sample_counts) == cert.n_regions):
        return False

    # Check p_hats in [0, 1]
    if any(p < 0 or p > 1 for p in cert.p_hats):
        return False

    # Check UCBs >= p_hats
    if any(u < p - 1e-10 for u, p in zip(cert.ucbs, cert.p_hats)):
        return False

    # Check weights sum to ~1
    weight_sum = sum(cert.region_weights)
    if abs(weight_sum - 1.0) > 0.01 and weight_sum > 0:
        return False

    return True
