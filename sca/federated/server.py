"""Federated learning server with Safety-Certified Aggregation (Section 4.1).

Protocol overview at each FL round:
1. Propose M_tilde_{t+1} by aggregation.
2. Run verifier V on M_tilde_{t+1} to obtain certificate C_{t+1} and Accept.
3. Commit or rollback:
    theta_{t+1} = theta_tilde_{t+1}  if Accept
                 = theta_t             otherwise
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from sca.certificate.acceptance import AcceptanceGate
from sca.certificate.certificate import SafetyCertificate
from sca.federated.aggregation import Aggregator, FedAvg
from sca.federated.client import ClientUpdate, FLClient

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Result of a single FL round.

    Attributes:
        round_num: FL round number t.
        accepted: Whether the candidate model was accepted.
        certificate: The safety certificate (None if gating disabled).
        candidate_params: Parameters of the candidate model.
        committed_params: Parameters of the committed model.
        n_clients: Number of clients participating.
    """
    round_num: int
    accepted: bool
    certificate: SafetyCertificate | None
    candidate_params: dict[str, torch.Tensor] | None = None
    committed_params: dict[str, torch.Tensor] | None = None
    n_clients: int = 0


class FederatedServer:
    """Federated learning server implementing the SCA protocol.

    Manages the global model, collects client updates, applies
    aggregation, and gates commitment via the acceptance gate.
    """

    def __init__(
        self,
        global_model: nn.Module,
        aggregator: Aggregator | None = None,
        acceptance_gate: AcceptanceGate | None = None,
        seed_interactions: list[dict] | None = None,
    ) -> None:
        """
        Args:
            global_model: Initial global model M(theta_0).
            aggregator: Aggregation rule A (defaults to FedAvg).
            acceptance_gate: Safety acceptance gate (None = no gating).
            seed_interactions: Seed interactions for the verifier.
        """
        self.global_model = global_model
        self.aggregator = aggregator or FedAvg()
        self.acceptance_gate = acceptance_gate
        self.seed_interactions = seed_interactions or []
        self.round_num = 0
        self.history: list[RoundResult] = []

    def run_round(
        self,
        clients: list[FLClient],
        model_fn_factory: Callable[[nn.Module], Callable[[dict], str]] | None = None,
    ) -> RoundResult:
        """Execute one FL round with optional safety gating.

        Args:
            clients: Participating clients for this round.
            model_fn_factory: Factory that creates a model inference function
                from an nn.Module. Required if acceptance_gate is set.

        Returns:
            RoundResult with the outcome of this round.
        """
        self.round_num += 1
        logger.info(f"FL Round {self.round_num}: {len(clients)} clients")

        # Step 1: Collect client updates
        updates = []
        for client in clients:
            update = client.compute_update(self.global_model, self.round_num)
            updates.append(update)

        # Step 2: Aggregate -> propose candidate model
        aggregated_delta = self.aggregator.aggregate(self.global_model, updates)

        candidate_model = copy.deepcopy(self.global_model)
        with torch.no_grad():
            for name, param in candidate_model.named_parameters():
                if name in aggregated_delta:
                    param.add_(aggregated_delta[name])

        # Step 3: Safety verification (if gate is set)
        accepted = True
        certificate = None

        if self.acceptance_gate is not None:
            if model_fn_factory is None:
                raise ValueError(
                    "model_fn_factory required when acceptance_gate is set"
                )

            model_fn = model_fn_factory(candidate_model)

            # Get candidate params as numpy for hashing
            candidate_params_np = np.concatenate([
                p.detach().cpu().numpy().flatten()
                for p in candidate_model.parameters()
            ])

            accepted, certificate = self.acceptance_gate.evaluate(
                model_fn=model_fn,
                model_params=candidate_params_np,
                seed_interactions=self.seed_interactions,
                fl_round=self.round_num,
            )

            if accepted:
                logger.info(
                    f"Round {self.round_num}: ACCEPTED "
                    f"(bound={certificate.bound_value:.4f} <= eps={certificate.epsilon})"
                )
            else:
                logger.info(
                    f"Round {self.round_num}: REJECTED "
                    f"(bound={certificate.bound_value:.4f} > eps={certificate.epsilon})"
                )

        # Step 4: Commit or rollback
        if accepted:
            self.global_model = candidate_model

        result = RoundResult(
            round_num=self.round_num,
            accepted=accepted,
            certificate=certificate,
            n_clients=len(clients),
        )
        self.history.append(result)
        return result

    def get_model(self) -> nn.Module:
        """Return the current global model."""
        return self.global_model

    def get_certificates(self) -> list[SafetyCertificate]:
        """Return all certificates generated so far."""
        return [r.certificate for r in self.history if r.certificate is not None]

    def get_acceptance_rate(self) -> float:
        """Fraction of rounds where the candidate was accepted."""
        if not self.history:
            return 0.0
        return sum(1 for r in self.history if r.accepted) / len(self.history)
