"""Federated server with an auditable accept / rollback protocol.

Each round:

1. Honest clients compute ``Delta_i`` from the CURRENT COMMITTED model.
2. Byzantine clients are then handed those honest updates and craft theirs
   (omniscient adversary; see :mod:`sca.experiments.attacks`).
3. The aggregator proposes a candidate ``theta_tilde_{t+1}``.
4. A gate accepts or rejects the candidate.
5. Commit on accept, roll back on reject.

WHY THE BOOKKEEPING IS SO EXPLICIT (audit finding F5)
-----------------------------------------------------
In the old harness, aggregation wrote into a ``deepcopy`` and "rollback"
restored a state that had never been touched.  Rejection was therefore a
strict no-op -- which is *correct* behaviour, but nothing in the code or the
results made it visible, so "93.75 % accuracy under 50 % Byzantine clients"
could be reported when the true content of that number was "the gate rejected
8 of 8 rounds and the model is still the frozen pretrained checkpoint".

So this server:

* hashes the committed parameters before and after every round and records
  ``rollback_was_noop`` per round;
* exposes :meth:`assert_rejections_were_noops`, which fails loudly if a
  rejected round ever changed a committed byte;
* records ``rounds_accepted`` / ``acceptance_rate`` and, per round, the
  metric of the CANDIDATE model -- the "would-be value if this round had been
  accepted" -- next to the metric of the committed model;
* keeps the gate strictly separate from evaluation.  ``eval_fn`` is for
  REPORTING ONLY and must be given the held-out test set the gate never sees.

An always-accept run and an always-reject run are ordinary configurations of
this same class (see :mod:`sca.experiments.baselines`), and the sanity
identities "always-reject == frozen pretrained" and "always-accept == no gate"
are asserted in code.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn

from sca.federated.aggregation import Aggregator, FedAvg, apply_delta_
from sca.federated.client import ClientUpdate, FLClient, delta_norm

logger = logging.getLogger(__name__)

__all__ = [
    "GateDecision",
    "SafetyGate",
    "AlwaysAcceptGate",
    "AlwaysRejectGate",
    "FunctionGate",
    "RoundResult",
    "FederatedServer",
    "model_state_hash",
]


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


@dataclass
class GateDecision:
    """A gate's verdict on one candidate model.

    Attributes:
        accepted: Commit the candidate?
        bound: The certified bound, when the gate produces one.
        epsilon: The threshold the bound was compared with.
        certificate: Opaque certificate object, stored for the audit trail.
        info: Free-form diagnostics.
    """

    accepted: bool
    bound: float | None = None
    epsilon: float | None = None
    certificate: Any | None = None
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SafetyGate(Protocol):
    """Anything that can accept or reject a candidate model.

    Deliberately structural: the server does not import the certificate stack,
    so a gate can be a full RLM verifier, a static suite, or one of the two
    trivial controls below.
    """

    name: str

    def evaluate(
        self, candidate_model: nn.Module, round_num: int
    ) -> GateDecision: ...


class AlwaysAcceptGate:
    """Control gate: commit every round.

    A run with this gate MUST produce numerically identical results to a run
    with no gate at all.  :func:`sca.experiments.baselines.assert_always_accept_equals_no_gate`
    checks it.
    """

    name = "always_accept"

    def evaluate(self, candidate_model: nn.Module, round_num: int) -> GateDecision:
        return GateDecision(accepted=True, info={"control": "always_accept"})


class AlwaysRejectGate:
    """Control gate: never commit.

    A run with this gate MUST equal the frozen initial checkpoint exactly.
    This control exists because the old paper's headline Byzantine number was
    exactly this row, unlabelled (finding F5).
    """

    name = "always_reject"

    def evaluate(self, candidate_model: nn.Module, round_num: int) -> GateDecision:
        return GateDecision(accepted=False, info={"control": "always_reject"})


class FunctionGate:
    """Adapter turning a plain callable into a :class:`SafetyGate`."""

    def __init__(
        self,
        fn: Callable[[nn.Module, int], Any],
        name: str = "function_gate",
    ) -> None:
        self.fn = fn
        self.name = name

    def evaluate(self, candidate_model: nn.Module, round_num: int) -> GateDecision:
        out = self.fn(candidate_model, round_num)
        if isinstance(out, GateDecision):
            return out
        if isinstance(out, tuple):
            accepted, bound = out[0], (out[1] if len(out) > 1 else None)
            return GateDecision(accepted=bool(accepted), bound=bound)
        return GateDecision(accepted=bool(out))


# ---------------------------------------------------------------------------
# State hashing
# ---------------------------------------------------------------------------


def model_state_hash(model: nn.Module) -> str:
    """Stable content hash of a model's parameters.

    blake2b over the raw parameter bytes in ``named_parameters`` order.  No
    Python ``hash()`` and no float formatting, so it is byte-exact and
    process-independent (finding F9).
    """
    h = hashlib.blake2b(digest_size=16)
    for name, param in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Round bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class RoundResult:
    """Everything an auditor needs to reconstruct one round."""

    round_num: int
    accepted: bool
    n_clients: int
    n_byzantine: int
    aggregated_delta_norm: float
    hash_before: str
    hash_candidate: str
    hash_after: str
    rollback_was_noop: bool
    gate_name: str = "none"
    bound: float | None = None
    epsilon: float | None = None
    certificate: Any | None = None
    gate_info: dict[str, Any] = field(default_factory=dict)
    aggregator_assumption: dict[str, Any] | None = None
    aggregator_info: dict[str, Any] = field(default_factory=dict)
    attack_diagnostics: dict[str, Any] | None = None
    committed_metric: Any = None
    candidate_metric: Any = None

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("certificate", None)
        return d


class FederatedServer:
    """Federated server implementing the accept / rollback protocol."""

    def __init__(
        self,
        global_model: nn.Module,
        aggregator: Aggregator | None = None,
        gate: SafetyGate | None = None,
        *,
        eval_fn: Callable[[nn.Module], Any] | None = None,
        server_update_fn: Callable[[nn.Module, int], Mapping[str, torch.Tensor]]
        | None = None,
        evaluate_candidate: bool = True,
        record_attack_diagnostics: bool = True,
    ) -> None:
        """
        Args:
            global_model: Initial model ``M(theta_0)``.  Deep-copied, so the
                caller's frozen checkpoint stays untouched and can be used as
                the "frozen pretrained" baseline.
            aggregator: Aggregation rule (defaults to :class:`FedAvg`).
            gate: Safety gate, or ``None`` for an ungated run.  ``None`` and
                :class:`AlwaysAcceptGate` must give identical results.
            eval_fn: REPORTING ONLY.  Called on the committed (and, if
                ``evaluate_candidate``, the candidate) model each round.  Give
                it the held-out test set; the gate must never see it.
            server_update_fn: Produces FLTrust's server reference update from
                the server verification pool.  Called once per round before
                aggregation and installed on the aggregator when it accepts
                one.
            evaluate_candidate: Also evaluate the candidate model, giving the
                per-round "would-be metric if accepted".
            record_attack_diagnostics: Compute cosine/norm diagnostics when
                any client is flagged Byzantine.
        """
        self.global_model = copy.deepcopy(global_model)
        self.initial_hash = model_state_hash(self.global_model)
        self.aggregator = aggregator if aggregator is not None else FedAvg()
        self.gate = gate
        self.eval_fn = eval_fn
        self.server_update_fn = server_update_fn
        self.evaluate_candidate = evaluate_candidate
        self.record_attack_diagnostics = record_attack_diagnostics
        self.round_num = 0
        self.history: list[RoundResult] = []

    # -- update collection --------------------------------------------------

    def collect_updates(
        self, clients: Sequence[FLClient]
    ) -> tuple[list[ClientUpdate], list[ClientUpdate]]:
        """Collect honest updates first, then adversarial ones.

        The ordering implements the omniscient threat model: Byzantine clients
        see the round's honest updates before choosing theirs.  ``benign_deltas``
        is passed explicitly and is a required argument of every attack.

        Returns:
            ``(benign_updates, byzantine_updates)``.
        """
        benign = [c for c in clients if not getattr(c, "is_byzantine", False)]
        byz = [c for c in clients if getattr(c, "is_byzantine", False)]

        benign_updates = [
            c.compute_update(self.global_model, self.round_num) for c in benign
        ]
        benign_deltas = [u.delta for u in benign_updates]

        benign_counts = [u.n_samples for u in benign_updates]

        byz_updates: list[ClientUpdate] = []
        for k, c in enumerate(byz):
            byz_updates.append(
                c.compute_update(
                    self.global_model,
                    self.round_num,
                    benign_deltas=benign_deltas,
                    attacker_index=k,
                    n_attackers=len(byz),
                    benign_n_samples=benign_counts,
                )
            )
        return benign_updates, byz_updates

    # -- one round ----------------------------------------------------------

    def run_round(self, clients: Sequence[FLClient]) -> RoundResult:
        """Execute one FL round."""
        self.round_num += 1
        hash_before = model_state_hash(self.global_model)

        benign_updates, byz_updates = self.collect_updates(clients)
        updates = benign_updates + byz_updates

        diagnostics = None
        if self.record_attack_diagnostics and byz_updates and benign_updates:
            from sca.experiments.attacks import attack_diagnostics

            diagnostics = attack_diagnostics(
                [u.delta for u in benign_updates],
                [u.delta for u in byz_updates],
            ).as_dict()

        if self.server_update_fn is not None and hasattr(
            self.aggregator, "set_server_update"
        ):
            self.aggregator.set_server_update(  # type: ignore[attr-defined]
                self.server_update_fn(self.global_model, self.round_num)
            )

        aggregated = self.aggregator.aggregate(self.global_model, updates)

        candidate = copy.deepcopy(self.global_model)
        apply_delta_(candidate, aggregated)
        hash_candidate = model_state_hash(candidate)

        if self.gate is None:
            decision = GateDecision(accepted=True, info={"gate": "none"})
            gate_name = "none"
        else:
            decision = self.gate.evaluate(candidate, self.round_num)
            gate_name = getattr(self.gate, "name", type(self.gate).__name__)

        candidate_metric = (
            self.eval_fn(candidate)
            if (self.eval_fn is not None and self.evaluate_candidate)
            else None
        )

        if decision.accepted:
            self.global_model = candidate
        # else: ROLLBACK. Nothing to undo -- the candidate was a copy and the
        # committed model was never written to. The hash check below is what
        # turns that claim into evidence.

        hash_after = model_state_hash(self.global_model)
        rollback_noop = decision.accepted or (hash_after == hash_before)

        committed_metric = (
            self.eval_fn(self.global_model) if self.eval_fn is not None else None
        )

        assumption = getattr(self.aggregator, "last_assumption", None)
        result = RoundResult(
            round_num=self.round_num,
            accepted=bool(decision.accepted),
            n_clients=len(updates),
            n_byzantine=len(byz_updates),
            aggregated_delta_norm=delta_norm(aggregated) if aggregated else 0.0,
            hash_before=hash_before,
            hash_candidate=hash_candidate,
            hash_after=hash_after,
            rollback_was_noop=bool(rollback_noop),
            gate_name=gate_name,
            bound=decision.bound,
            epsilon=decision.epsilon,
            certificate=decision.certificate,
            gate_info=dict(decision.info),
            aggregator_assumption=(
                assumption.as_dict() if assumption is not None else None
            ),
            aggregator_info=dict(getattr(self.aggregator, "last_info", {}) or {}),
            attack_diagnostics=diagnostics,
            committed_metric=committed_metric,
            candidate_metric=candidate_metric,
        )
        self.history.append(result)

        logger.info(
            "round %d: %s (gate=%s, |delta|=%.4g, n=%d, f=%d)",
            self.round_num,
            "ACCEPT" if result.accepted else "REJECT",
            gate_name,
            result.aggregated_delta_norm,
            result.n_clients,
            result.n_byzantine,
        )
        return result

    def run(
        self, clients: Sequence[FLClient], n_rounds: int
    ) -> list[RoundResult]:
        """Run ``n_rounds`` rounds and return the per-round results."""
        return [self.run_round(clients) for _ in range(n_rounds)]

    # -- reporting hooks ----------------------------------------------------

    def get_model(self) -> nn.Module:
        return self.global_model

    @property
    def rounds_accepted(self) -> int:
        return sum(1 for r in self.history if r.accepted)

    @property
    def rounds_rejected(self) -> int:
        return sum(1 for r in self.history if not r.accepted)

    def get_acceptance_rate(self) -> float:
        if not self.history:
            return 0.0
        return self.rounds_accepted / len(self.history)

    def model_is_unchanged(self) -> bool:
        """True iff no round has ever modified the initial checkpoint."""
        return model_state_hash(self.global_model) == self.initial_hash

    def assert_rejections_were_noops(self) -> None:
        """Raise unless every rejected round left the committed model intact.

        This is the executable form of the rollback claim.  If it ever fails,
        the accept/rollback protocol is broken and no gated number in the paper
        means anything.
        """
        for r in self.history:
            if not r.accepted and r.hash_after != r.hash_before:
                raise AssertionError(
                    f"round {r.round_num} was REJECTED but the committed model "
                    f"changed: {r.hash_before} -> {r.hash_after}. Rollback is "
                    "not a no-op."
                )

    def summary(self) -> dict[str, Any]:
        """Compact per-run summary for the results file."""
        self.assert_rejections_were_noops()
        return {
            "aggregator": getattr(self.aggregator, "name", "unknown"),
            "gate": (
                getattr(self.gate, "name", type(self.gate).__name__)
                if self.gate is not None
                else "none"
            ),
            "n_rounds": len(self.history),
            "rounds_accepted": self.rounds_accepted,
            "rounds_rejected": self.rounds_rejected,
            "acceptance_rate": self.get_acceptance_rate(),
            "initial_hash": self.initial_hash,
            "final_hash": model_state_hash(self.global_model),
            "model_is_frozen_pretrained": self.model_is_unchanged(),
            "final_metric": (
                self.history[-1].committed_metric if self.history else None
            ),
            "would_be_metric_if_last_round_accepted": (
                self.history[-1].candidate_metric if self.history else None
            ),
            "aggregator_assumption": (
                self.history[-1].aggregator_assumption if self.history else None
            ),
        }
