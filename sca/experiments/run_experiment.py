"""Main experiment runner (Section 7).

Orchestrates federated learning experiments with different attack
scenarios, aggregation methods, and safety gating configurations.
Collects metrics aligned with the theory (Section 7.3).
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from sca.certificate.acceptance import AcceptanceGate
from sca.experiments.attacks import AttackConfig, create_attack_scenario
from sca.experiments.baselines import BaselineConfig, build_verifier_for_config, create_baseline_configs
from sca.experiments.metrics import (
    SafetyMetrics,
    aggregate_round_metrics,
    compute_bound_tightness,
    compute_violation_rate,
)
from sca.federated.client import BenignClient, FLClient
from sca.federated.server import FederatedServer
from sca.knowledge_graph.embedding import RandomProjectionEmbedder
from sca.verifier.safety_predicate import SafetyPredicate

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a full experiment.

    Attributes:
        n_rounds: Number of FL rounds.
        n_clients: Total number of clients.
        attack_config: Byzantine attack configuration.
        baseline_configs: List of baseline configurations to evaluate.
        epsilon: Target violation bound.
        delta: Confidence parameter.
        verification_budget: Total verification queries per round.
        seed: Random seed.
    """
    n_rounds: int = 10
    n_clients: int = 10
    attack_config: AttackConfig = field(
        default_factory=lambda: AttackConfig(n_byzantine=2)
    )
    baseline_configs: list[BaselineConfig] | None = None
    epsilon: float = 0.05
    delta: float = 0.01
    verification_budget: int = 200
    seed: int = 42


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config_name: str
    round_results: list[dict] = field(default_factory=list)
    final_safety: SafetyMetrics | None = None
    summary: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """Runs federated learning experiments with safety gating.

    Supports running multiple baselines against the same attack scenario
    for fair comparison.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        client_data_factory: Callable[[int], Any],
        safety_predicate: SafetyPredicate,
        model_fn_factory: Callable[[nn.Module], Callable[[dict], str]],
        seed_interactions: list[dict],
        test_interactions: list[dict],
    ) -> None:
        """
        Args:
            model_factory: Creates a fresh model instance.
            client_data_factory: Creates a dataset for client i.
            safety_predicate: Safety predicate phi.
            model_fn_factory: Creates model inference fn from nn.Module.
            seed_interactions: Seed interactions for verification.
            test_interactions: Held-out test set for measuring true Viol(M).
        """
        self.model_factory = model_factory
        self.client_data_factory = client_data_factory
        self.safety_predicate = safety_predicate
        self.model_fn_factory = model_fn_factory
        self.seed_interactions = seed_interactions
        self.test_interactions = test_interactions

    def run(self, config: ExperimentConfig) -> list[ExperimentResult]:
        """Run all baseline configurations under the given attack scenario.

        Args:
            config: Experiment configuration.

        Returns:
            List of ExperimentResult, one per baseline.
        """
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        baselines = config.baseline_configs or create_baseline_configs(
            n_byzantine=config.attack_config.n_byzantine,
            verification_budget=config.verification_budget,
        )

        # Create Byzantine clients (shared across baselines for fairness)
        byzantine_clients = create_attack_scenario(
            n_total_clients=config.n_clients,
            config=config.attack_config,
        )

        results = []
        for baseline in baselines:
            logger.info(f"Running baseline: {baseline.name}")
            result = self._run_single(config, baseline, byzantine_clients)
            results.append(result)

        return results

    def _run_single(
        self,
        config: ExperimentConfig,
        baseline: BaselineConfig,
        byzantine_clients: list[FLClient],
    ) -> ExperimentResult:
        """Run a single baseline configuration."""
        # Fresh model for each baseline
        model = self.model_factory()

        # Build verifier if needed
        verifier = build_verifier_for_config(
            config=baseline,
            safety_predicate=self.safety_predicate,
        )

        acceptance_gate = None
        if verifier is not None:
            acceptance_gate = AcceptanceGate(
                verifier=verifier,
                epsilon=config.epsilon,
                delta=config.delta,
            )

        # Create server
        server = FederatedServer(
            global_model=model,
            aggregator=baseline.aggregator,
            acceptance_gate=acceptance_gate,
            seed_interactions=self.seed_interactions,
        )

        # Create benign clients
        n_benign = config.n_clients - config.attack_config.n_byzantine
        benign_clients = [
            BenignClient(
                client_id=i,
                dataset=self.client_data_factory(i),
            )
            for i in range(n_benign)
        ]

        # Combine benign + Byzantine
        all_clients: list[FLClient] = benign_clients + byzantine_clients

        # Run FL rounds
        round_results = []
        for t in range(config.n_rounds):
            round_result = server.run_round(
                clients=all_clients,
                model_fn_factory=self.model_fn_factory,
            )

            # Measure true violation rate
            model_fn = self.model_fn_factory(server.get_model())
            safety_metrics = compute_violation_rate(
                model_fn, self.test_interactions, self.safety_predicate,
            )

            round_dict = {
                "round": t + 1,
                "accepted": round_result.accepted,
                "violation_rate": safety_metrics.violation_rate,
                "n_violations": safety_metrics.n_violations,
            }

            if round_result.certificate is not None:
                round_dict["certified_bound"] = round_result.certificate.bound_value
                tightness = compute_bound_tightness(
                    round_result.certificate,
                    safety_metrics.violation_rate,
                )
                round_dict["bound_gap"] = tightness.gap

            round_results.append(round_dict)

        # Final evaluation
        final_model_fn = self.model_fn_factory(server.get_model())
        final_safety = compute_violation_rate(
            final_model_fn, self.test_interactions, self.safety_predicate,
        )

        summary = aggregate_round_metrics(round_results)
        summary["final_violation_rate"] = final_safety.violation_rate
        summary["acceptance_rate"] = server.get_acceptance_rate()

        return ExperimentResult(
            config_name=baseline.name,
            round_results=round_results,
            final_safety=final_safety,
            summary=summary,
        )
