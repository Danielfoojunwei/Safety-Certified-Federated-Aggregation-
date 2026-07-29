"""Safety certificate schema, construction and independent verification.

A certificate is a self-contained record of one acceptance decision.  Its
purpose is that a third party who trusts nothing about the run can re-derive
the decision from the raw counts.  That means the certificate must carry the
raw counts (``n_samples``, ``n_violations``, ``weight`` per region) and the
a-priori partition parameters (``k_total``, ``budget_cap``, ``delta``), not
just the derived ``UCB`` values.

    C = ( h(theta), h(V), h(G),
          {region_id_j, w_j, m_j, v_j, p_hat_j, UCB_j},
          k_total, budget_cap, epsilon, delta,
          Decision, bound, vacuous, TraceDigest, BindingHash )

WHAT CHANGED FROM THE OLD VERSION (finding F14, last clause):
``verify_certificate_consistency`` used to compare two values that came out
of the *same* ``check_acceptance`` call, so it could not detect a wrong UCB,
a wrong bound, or tampering with the counts.  It now recomputes every UCB and
the aggregate bound from the raw counts using the declared parameters, and
checks a binding hash over the sorted per-region counts plus the partition
parameters.  See :func:`verify_certificate`.

WHAT THE CERTIFICATE IS ALLOWED TO SEE: estimation statistics only.  Search /
mutation counters are discovery metrics and are recorded in the certificate
as opaque metadata (never as evidence) so that a reader can see how much
searching happened without that searching entering the bound.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from sca.utils.crypto import MerkleTree, hash_object, hash_tensor
from sca.utils.stats import (
    AcceptanceResult,
    RegionStat,
    check_acceptance,
    compute_ucb,
)

__all__ = [
    "SafetyCertificate",
    "build_certificate",
    "certificate_binding_hash",
    "verify_certificate",
    "verify_certificate_consistency",
]

# Certified quantities are compared at this tolerance.  It must be well below
# the 1e-6 perturbation that ``tests/test_certificate.py`` injects.
_RECOMPUTE_TOL = 1e-12


@dataclass
class SafetyCertificate:
    """Cryptographic safety certificate.

    Every field needed to re-derive ``decision`` is present, so verification
    never has to trust the producer.
    """

    # Commitments
    model_hash: str
    verifier_hash: str
    graph_hash: str

    # Raw per-region ESTIMATION counts (the evidence)
    region_ids: list[int]
    region_weights: list[float]
    region_n_samples: list[int]
    region_n_violations: list[int]

    # Derived quantities (recomputable from the above + parameters)
    p_hats: list[float]
    ucbs: list[float]

    # A-priori partition parameters
    k_total: int
    budget_cap: int
    epsilon: float
    delta: float

    # Decision
    decision: str  # "accept" | "reject"
    bound_value: float  # sum_j w_j * UCB_j over the full declared partition
    vacuous: bool
    unaccounted_weight: float

    # Trace / provenance
    trace_digest: str
    n_estimation_queries: int
    n_search_queries: int

    # Binding commitment over counts + partition parameters
    binding_hash: str

    # Discovery metrics: reported, never certified
    search_metrics: dict = field(default_factory=dict)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    fl_round: int = 0
    gate_mode: str = "certified"

    def to_dict(self) -> dict:
        """Serialize certificate to a plain dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize certificate to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "SafetyCertificate":
        """Deserialize from a dictionary."""
        return cls(**d)

    @property
    def is_accepted(self) -> bool:
        return self.decision == "accept"

    @property
    def n_regions_reported(self) -> int:
        return len(self.region_ids)

    def region_stats(self) -> list[RegionStat]:
        """Reconstruct the :class:`RegionStat` list from the raw counts."""
        return [
            RegionStat(
                region_id=int(rid),
                weight=float(w),
                n_samples=int(m),
                n_violations=int(v),
            )
            for rid, w, m, v in zip(
                self.region_ids,
                self.region_weights,
                self.region_n_samples,
                self.region_n_violations,
            )
        ]

    def certificate_hash(self) -> str:
        """Hash of the whole certificate (including the binding hash)."""
        return hash_object(self.to_dict())


def certificate_binding_hash(
    region_ids: Sequence[int],
    region_weights: Sequence[float],
    region_n_samples: Sequence[int],
    region_n_violations: Sequence[int],
    k_total: int,
    budget_cap: int,
    epsilon: float,
    delta: float,
    model_hash: str,
    verifier_hash: str,
    graph_hash: str,
    trace_digest: str,
) -> str:
    """Commitment binding the decision to exactly what was probed.

    The per-region tuples are sorted by ``region_id`` before hashing so that
    the commitment does not depend on the order the verifier happened to emit
    regions in.  Weights are rounded to 12 decimal places so that the
    commitment survives JSON round-tripping but still changes under any
    perturbation large enough to move a bound.
    """
    rows = sorted(
        (
            int(rid),
            round(float(w), 12),
            int(m),
            int(v),
        )
        for rid, w, m, v in zip(
            region_ids, region_weights, region_n_samples, region_n_violations
        )
    )
    payload = {
        "schema": "sca-certificate-binding-v2",
        "regions": rows,
        "k_total": int(k_total),
        "budget_cap": int(budget_cap),
        "epsilon": round(float(epsilon), 12),
        "delta": round(float(delta), 12),
        "model_hash": model_hash,
        "verifier_hash": verifier_hash,
        "graph_hash": graph_hash,
        "trace_digest": trace_digest,
    }
    return hash_object(payload)


def build_certificate(
    model_params: np.ndarray | Any,
    verifier_descriptor: dict,
    graph_summary: dict,
    estimation_stats: Sequence[RegionStat],
    epsilon: float,
    delta: float,
    k_total: int,
    budget_cap: int,
    traces: Sequence[dict] | None = None,
    fl_round: int = 0,
    search_metrics: dict | None = None,
    n_search_queries: int = 0,
    gate_mode: str = "certified",
    forced_decision: str | None = None,
) -> SafetyCertificate:
    """Construct a safety certificate from Stage-C estimation statistics.

    Args:
        model_params: Model parameters (array, state dict, or a hash string).
        verifier_descriptor: Serializable descriptor of the verifier ``V``.
        graph_summary: Serializable summary of the MKG state ``G``.
        estimation_stats: Per-region ESTIMATION counts.  Passing search or
            mutation counts here voids Theorem 1 and is the bug this rebuild
            exists to remove.
        epsilon: Target violation bound.
        delta: Global failure probability.
        k_total: A-priori number of regions.
        budget_cap: A-priori per-region estimation budget cap.
        traces: Trace entry dicts for the Merkle commitment.
        fl_round: Federated round index.
        search_metrics: Discovery metrics (raw and deduped violation counts,
            recursion depth histogram, ...).  Recorded, never certified.
        n_search_queries: Number of Stage-A model queries.
        gate_mode: ``"certified"``, ``"always_accept"`` or ``"always_reject"``.
            Recorded so the ablation controls are visible in the artifact.
        forced_decision: If given (``"accept"``/``"reject"``), overrides the
            statistical decision.  Only the ablation gate modes use this; the
            statistical ``bound_value`` is still recorded truthfully.

    Returns:
        A :class:`SafetyCertificate`.
    """
    stats = list(estimation_stats)

    if isinstance(model_params, str):
        model_hash = model_params
    elif isinstance(model_params, np.ndarray):
        model_hash = hash_tensor(model_params)
    elif hasattr(model_params, "detach"):
        model_hash = hash_tensor(model_params)
    elif isinstance(model_params, dict):
        model_hash = hash_object(
            {k: hash_tensor(v) if hasattr(v, "shape") else str(v)
             for k, v in sorted(model_params.items())}
        )
    else:
        model_hash = hash_object(model_params)

    verifier_hash = hash_object(verifier_descriptor)
    graph_hash = hash_object(graph_summary)

    result: AcceptanceResult = check_acceptance(
        stats, epsilon, delta, k_total, budget_cap
    )

    # per_region is in the order supplied, and is empty when the result is
    # vacuous by way of zero weight; fall back to recomputing per-stat.
    if result.per_region:
        ucbs = [r.ucb for r in result.per_region]
        weights_out = [r.weight for r in result.per_region]
    else:
        ucbs = [
            compute_ucb(rs.n_violations, rs.n_samples, k_total, budget_cap, delta)
            for rs in stats
        ]
        weights_out = [rs.weight for rs in stats]

    region_ids = [rs.region_id for rs in stats]
    n_samples = [rs.n_samples for rs in stats]
    n_violations = [rs.n_violations for rs in stats]
    p_hats = [rs.p_hat for rs in stats]

    merkle = MerkleTree(list(traces) if traces else [])
    trace_digest = merkle.root_hash

    if forced_decision is not None:
        if forced_decision not in ("accept", "reject"):
            raise ValueError(f"forced_decision must be accept/reject, got {forced_decision!r}")
        decision = forced_decision
    else:
        decision = "accept" if result.accepted else "reject"

    binding = certificate_binding_hash(
        region_ids=region_ids,
        region_weights=weights_out,
        region_n_samples=n_samples,
        region_n_violations=n_violations,
        k_total=k_total,
        budget_cap=budget_cap,
        epsilon=epsilon,
        delta=delta,
        model_hash=model_hash,
        verifier_hash=verifier_hash,
        graph_hash=graph_hash,
        trace_digest=trace_digest,
    )

    return SafetyCertificate(
        model_hash=model_hash,
        verifier_hash=verifier_hash,
        graph_hash=graph_hash,
        region_ids=region_ids,
        region_weights=weights_out,
        region_n_samples=n_samples,
        region_n_violations=n_violations,
        p_hats=p_hats,
        ucbs=ucbs,
        k_total=int(k_total),
        budget_cap=int(budget_cap),
        epsilon=float(epsilon),
        delta=float(delta),
        decision=decision,
        bound_value=float(result.bound),
        vacuous=bool(result.vacuous),
        unaccounted_weight=float(result.unaccounted_weight),
        trace_digest=trace_digest,
        n_estimation_queries=int(sum(n_samples)),
        n_search_queries=int(n_search_queries),
        binding_hash=binding,
        search_metrics=dict(search_metrics or {}),
        fl_round=int(fl_round),
        gate_mode=gate_mode,
    )


def verify_certificate(cert: SafetyCertificate) -> tuple[bool, list[str]]:
    """Independently re-derive every certified quantity from the raw counts.

    This function deliberately does NOT call anything that produced the
    certificate's stored values in the first place beyond the primitive
    :func:`sca.utils.stats.compute_ucb` / :func:`check_acceptance` -- it
    starts from ``(region_id, weight, n_samples, n_violations)`` plus the
    declared ``(k_total, budget_cap, epsilon, delta)`` and rebuilds
    ``p_hat_j``, ``UCB_j``, the aggregate bound, the decision and the binding
    hash.  Any single stored value that does not match is a failure.

    Returns:
        ``(ok, reasons)`` where ``reasons`` is empty iff ``ok``.
    """
    reasons: list[str] = []

    n = len(cert.region_ids)
    lengths = {
        "region_weights": len(cert.region_weights),
        "region_n_samples": len(cert.region_n_samples),
        "region_n_violations": len(cert.region_n_violations),
        "p_hats": len(cert.p_hats),
        "ucbs": len(cert.ucbs),
    }
    for name, length in lengths.items():
        if length != n:
            reasons.append(
                f"length mismatch: {name} has {length} entries, "
                f"region_ids has {n}"
            )
    if reasons:
        return False, reasons

    if cert.decision not in ("accept", "reject"):
        reasons.append(f"decision must be accept/reject, got {cert.decision!r}")

    if len(set(cert.region_ids)) != n:
        reasons.append("duplicate region_id in certificate")
    if any(rid < 0 or rid >= cert.k_total for rid in cert.region_ids):
        reasons.append("region_id outside the declared partition [0, k_total)")
    if n > cert.k_total:
        reasons.append(
            f"{n} regions reported but k_total={cert.k_total} was declared"
        )
    if any(m > cert.budget_cap for m in cert.region_n_samples):
        reasons.append(
            "a region used more samples than the declared budget_cap; the "
            "anytime union bound does not cover it"
        )
    if reasons:
        return False, reasons

    # ---- independent recomputation of p_hat and UCB, region by region -----
    try:
        stats = cert.region_stats()
    except ValueError as exc:  # malformed counts
        return False, [f"malformed region counts: {exc}"]

    for i, rs in enumerate(stats):
        expected_p = rs.p_hat
        if abs(expected_p - float(cert.p_hats[i])) > _RECOMPUTE_TOL:
            reasons.append(
                f"region {rs.region_id}: stored p_hat={cert.p_hats[i]!r} but "
                f"{rs.n_violations}/{rs.n_samples} = {expected_p!r}"
            )
        try:
            expected_ucb = compute_ucb(
                rs.n_violations,
                rs.n_samples,
                cert.k_total,
                cert.budget_cap,
                cert.delta,
            )
        except ValueError as exc:
            reasons.append(f"region {rs.region_id}: cannot recompute UCB: {exc}")
            continue
        if abs(expected_ucb - float(cert.ucbs[i])) > _RECOMPUTE_TOL:
            reasons.append(
                f"region {rs.region_id}: stored UCB={cert.ucbs[i]!r} but "
                f"recomputed {expected_ucb!r} from "
                f"(v={rs.n_violations}, m={rs.n_samples}, K={cert.k_total}, "
                f"M={cert.budget_cap}, delta={cert.delta})"
            )

    # ---- independent recomputation of the aggregate bound and decision ----
    try:
        recomputed = check_acceptance(
            stats, cert.epsilon, cert.delta, cert.k_total, cert.budget_cap
        )
    except ValueError as exc:
        return False, reasons + [f"cannot recompute acceptance: {exc}"]

    if abs(recomputed.bound - float(cert.bound_value)) > _RECOMPUTE_TOL:
        reasons.append(
            f"stored bound_value={cert.bound_value!r} but recomputed "
            f"{recomputed.bound!r}"
        )
    if bool(recomputed.vacuous) != bool(cert.vacuous):
        reasons.append(
            f"stored vacuous={cert.vacuous} but recomputed "
            f"{recomputed.vacuous}"
        )
    if abs(recomputed.unaccounted_weight - float(cert.unaccounted_weight)) > 1e-9:
        reasons.append(
            f"stored unaccounted_weight={cert.unaccounted_weight!r} but "
            f"recomputed {recomputed.unaccounted_weight!r}"
        )

    statistical_decision = "accept" if recomputed.accepted else "reject"
    if cert.gate_mode == "certified":
        if cert.decision != statistical_decision:
            reasons.append(
                f"decision {cert.decision!r} does not follow from the "
                f"recomputed bound {recomputed.bound!r} vs epsilon "
                f"{cert.epsilon!r} (expected {statistical_decision!r})"
            )
    elif cert.gate_mode == "always_accept":
        if cert.decision != "accept":
            reasons.append("gate_mode=always_accept but decision is not accept")
    elif cert.gate_mode == "always_reject":
        if cert.decision != "reject":
            reasons.append("gate_mode=always_reject but decision is not reject")
    else:
        reasons.append(f"unknown gate_mode {cert.gate_mode!r}")

    # ---- binding commitment ------------------------------------------------
    expected_binding = certificate_binding_hash(
        region_ids=cert.region_ids,
        region_weights=cert.region_weights,
        region_n_samples=cert.region_n_samples,
        region_n_violations=cert.region_n_violations,
        k_total=cert.k_total,
        budget_cap=cert.budget_cap,
        epsilon=cert.epsilon,
        delta=cert.delta,
        model_hash=cert.model_hash,
        verifier_hash=cert.verifier_hash,
        graph_hash=cert.graph_hash,
        trace_digest=cert.trace_digest,
    )
    if expected_binding != cert.binding_hash:
        reasons.append(
            "binding_hash does not commit to the stored counts / parameters "
            "(certificate was tampered with or rebuilt inconsistently)"
        )

    if int(sum(cert.region_n_samples)) != int(cert.n_estimation_queries):
        reasons.append(
            f"n_estimation_queries={cert.n_estimation_queries} does not equal "
            f"sum of per-region samples {sum(cert.region_n_samples)}"
        )

    return (not reasons), reasons


def verify_certificate_consistency(cert: SafetyCertificate) -> bool:
    """Boolean wrapper around :func:`verify_certificate`."""
    ok, _ = verify_certificate(cert)
    return ok
