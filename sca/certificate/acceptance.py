"""The acceptance gate: Stage C of the rebuilt three-stage protocol.

    Stage A  SEARCH      -- mutation recursion / MKG frontier exploration over
                            the search pool.  Produces per-region suspicion
                            scores.  Recorded to ``search_*`` counters.  NEVER
                            touches the bound.
    Stage B  ESTIMATION  -- allocate the estimation budget across the K
                            regions using the Stage-A scores (adaptive and
                            data-dependent, which Theorem 1 makes legal), then
                            draw m_j FRESH i.i.d. samples per region from a
                            DISJOINT estimation pool.  Recorded to
                            ``estimation_*`` counters.
    Stage C  CERTIFICATE -- THIS MODULE.  Computes the bound from
                            ``estimation_stats`` only and emits a certificate.

Design note on coupling: this module deliberately does not import the
verifier or the knowledge graph.  It consumes a *result object* that merely
has to expose ``estimation_stats`` (and optionally the discovery fields of
``VerificationResult``).  That keeps Stage C independent of how Stages A and B
were implemented, and makes the ablation gates below trivially honest -- they
run the same code path and differ only in the final decision.

MANDATORY CONTROLS.  ``GateMode`` provides the two controls whose absence made
the old results uninterpretable:

  * ``always_reject`` -- the gate rejects unconditionally.  In the FL harness
    this MUST reproduce the frozen pretrained checkpoint exactly.
  * ``always_accept`` -- the gate accepts unconditionally.  In the FL harness
    this MUST reproduce the no-gate aggregator exactly.

Both still run verification and still emit a certificate carrying the true
statistical bound, so the artifact records what the certified gate *would*
have done.  ``forced_decision`` in the certificate's ``gate_mode`` field makes
the override visible to any reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

from sca.certificate.certificate import (
    SafetyCertificate,
    build_certificate,
    verify_certificate,
)
from sca.utils.stats import AcceptanceResult, RegionStat, check_acceptance

__all__ = [
    "GateMode",
    "GateDecision",
    "AcceptanceGate",
    "HasEstimationStats",
]


class GateMode(str, Enum):
    """How the gate turns a certificate into an accept/reject decision."""

    CERTIFIED = "certified"
    ALWAYS_ACCEPT = "always_accept"
    ALWAYS_REJECT = "always_reject"


@runtime_checkable
class HasEstimationStats(Protocol):
    """Structural type of what Stage C consumes.

    ``sca.verifier.rlm_verifier.VerificationResult`` satisfies this.
    """

    estimation_stats: Sequence[RegionStat]


@dataclass
class GateDecision:
    """Result of one pass through the gate."""

    accepted: bool
    certificate: SafetyCertificate
    acceptance: AcceptanceResult
    mode: GateMode

    @property
    def statistical_accept(self) -> bool:
        """What the certified rule would have decided, ignoring the mode."""
        return self.acceptance.accepted


def _get(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default)


class AcceptanceGate:
    """Stage C: turn estimation statistics into a decision plus certificate.

    Args:
        epsilon: Target violation bound.
        delta: Global failure probability for ONE certificate.  If the caller
            runs the gate for ``T`` rounds and wants a simultaneous guarantee,
            it must pass ``delta / T`` -- Theorem 1 covers a single
            certificate.  ``rounds_for_union_bound`` does this for you.
        k_total: A-priori number of regions.  Must be fixed before any data.
        budget_cap: A-priori per-region estimation budget cap ``M``.
        mode: :class:`GateMode`.
        rounds_for_union_bound: If ``> 1``, the effective delta used is
            ``delta / rounds_for_union_bound``, giving a guarantee that holds
            simultaneously across that many rounds.
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        delta: float = 0.05,
        k_total: int = 8,
        budget_cap: int = 512,
        mode: GateMode | str = GateMode.CERTIFIED,
        rounds_for_union_bound: int = 1,
    ) -> None:
        if rounds_for_union_bound < 1:
            raise ValueError("rounds_for_union_bound must be >= 1")
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.rounds_for_union_bound = int(rounds_for_union_bound)
        self.effective_delta = self.delta / self.rounds_for_union_bound
        self.k_total = int(k_total)
        self.budget_cap = int(budget_cap)
        self.mode = GateMode(mode)
        self.certificates: list[SafetyCertificate] = []
        self.decisions: list[GateDecision] = []

    # ------------------------------------------------------------------
    def evaluate_stats(
        self,
        estimation_stats: Sequence[RegionStat],
        *,
        model_params: Any = None,
        verifier_descriptor: dict | None = None,
        graph_summary: dict | None = None,
        traces: Sequence[dict] | None = None,
        search_metrics: dict | None = None,
        n_search_queries: int = 0,
        fl_round: int = 0,
    ) -> GateDecision:
        """Certify a set of estimation statistics.

        This is the primitive entry point; :meth:`evaluate` wraps it for
        ``VerificationResult``-shaped inputs.
        """
        result = check_acceptance(
            estimation_stats,
            self.epsilon,
            self.effective_delta,
            self.k_total,
            self.budget_cap,
        )

        if self.mode is GateMode.CERTIFIED:
            accepted = result.accepted
            forced = None
        elif self.mode is GateMode.ALWAYS_ACCEPT:
            accepted = True
            forced = "accept"
        else:
            accepted = False
            forced = "reject"

        cert = build_certificate(
            model_params=model_params if model_params is not None else "unspecified",
            verifier_descriptor=verifier_descriptor or {},
            graph_summary=graph_summary or {},
            estimation_stats=estimation_stats,
            epsilon=self.epsilon,
            delta=self.effective_delta,
            k_total=self.k_total,
            budget_cap=self.budget_cap,
            traces=traces,
            fl_round=fl_round,
            search_metrics=search_metrics,
            n_search_queries=n_search_queries,
            gate_mode=self.mode.value,
            forced_decision=forced,
        )

        decision = GateDecision(
            accepted=bool(accepted),
            certificate=cert,
            acceptance=result,
            mode=self.mode,
        )
        self.certificates.append(cert)
        self.decisions.append(decision)
        return decision

    # ------------------------------------------------------------------
    def evaluate(
        self,
        verification_result: HasEstimationStats,
        *,
        model_params: Any = None,
        verifier_descriptor: dict | None = None,
        graph_summary: dict | None = None,
        traces: Sequence[dict] | None = None,
        fl_round: int = 0,
    ) -> GateDecision:
        """Certify a ``VerificationResult`` from Stages A and B.

        Only ``verification_result.estimation_stats`` reaches the bound.  The
        discovery fields (``search_stats``, ``total_violations``,
        ``distinct_violations``, ``distinct_sources``, ``search_queries``,
        ``recursion_depth_hist``) are copied into the certificate's
        ``search_metrics`` as reported-but-uncertified metadata, with both the
        raw and deduplicated violation counts side by side.
        """
        estimation_stats = list(verification_result.estimation_stats)

        search_stats = _get(verification_result, "search_stats", []) or []
        search_metrics = {
            "total_violations_raw": int(
                _get(verification_result, "total_violations", 0)
            ),
            "distinct_violations_dedup_depth0": int(
                _get(verification_result, "distinct_violations", 0)
            ),
            "distinct_sources": int(
                _get(verification_result, "distinct_sources", 0)
            ),
            "search_queries": int(_get(verification_result, "search_queries", 0)),
            "estimation_queries": int(
                _get(verification_result, "estimation_queries", 0)
            ),
            "recursion_depth_hist": {
                str(k): int(v)
                for k, v in dict(
                    _get(verification_result, "recursion_depth_hist", {}) or {}
                ).items()
            },
            "allocation": {
                str(k): int(v)
                for k, v in dict(
                    _get(verification_result, "allocation", {}) or {}
                ).items()
            },
            "search_stats": [
                {
                    "region_id": int(s.region_id),
                    "weight": float(s.weight),
                    "n_samples": int(s.n_samples),
                    "n_violations": int(s.n_violations),
                }
                for s in search_stats
            ],
            "_note": (
                "Discovery metrics only. None of these values enters the "
                "certified bound; see Theorem 1 in sca/utils/stats.py."
            ),
        }

        return self.evaluate_stats(
            estimation_stats,
            model_params=model_params,
            verifier_descriptor=verifier_descriptor,
            graph_summary=graph_summary,
            traces=traces,
            search_metrics=search_metrics,
            n_search_queries=int(_get(verification_result, "search_queries", 0)),
            fl_round=fl_round,
        )

    # ------------------------------------------------------------------
    def verify_all(self) -> tuple[bool, list[str]]:
        """Independently re-verify every certificate this gate has issued."""
        problems: list[str] = []
        for i, cert in enumerate(self.certificates):
            ok, reasons = verify_certificate(cert)
            if not ok:
                problems.extend(f"certificate[{i}]: {r}" for r in reasons)
        return (not problems), problems
