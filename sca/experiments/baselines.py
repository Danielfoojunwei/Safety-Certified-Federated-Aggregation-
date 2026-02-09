"""Baseline configurations for experiments (Section 7.2).

Baselines:
1. FedAvg / FedAdam without gating
2. Robust aggregation only (median / trimmed mean / Krum)
3. Static safety suite gating (fixed prompts, no recursion)
4. SCA without MKG (adaptive recursion but no graph coverage)
5. Full SCA (RLM + MKG + certificates)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from sca.certificate.acceptance import AcceptanceGate
from sca.federated.aggregation import (
    Aggregator,
    CoordinateMedian,
    FedAdam,
    FedAvg,
    Krum,
    TrimmedMean,
)
from sca.knowledge_graph.embedding import InteractionEmbedder, RandomProjectionEmbedder
from sca.knowledge_graph.mkg import ModelKnowledgeGraph
from sca.knowledge_graph.regions import RegionPartition
from sca.utils.stats import RegionStats, check_acceptance
from sca.verifier.mutations import CompositeMutator
from sca.verifier.rlm_verifier import RLMVerifier
from sca.verifier.safety_predicate import SafetyPredicate


@dataclass
class BaselineConfig:
    """Configuration for a baseline experiment.

    Attributes:
        name: Baseline name.
        aggregator: Aggregation rule.
        use_safety_gate: Whether safety gating is enabled.
        use_recursion: Whether RLM recursion is enabled.
        use_mkg: Whether MKG-guided sampling is enabled.
        verification_budget: Total number of verification queries.
    """
    name: str
    aggregator: Aggregator
    use_safety_gate: bool = False
    use_recursion: bool = False
    use_mkg: bool = False
    verification_budget: int = 200


class StaticSuiteGate:
    """Static safety suite gating (Baseline 3).

    Evaluates a fixed set of prompts without recursion or graph guidance.
    This is the simplest form of safety gating.
    """

    def __init__(
        self,
        safety_predicate: SafetyPredicate,
        test_suite: list[dict],
        epsilon: float = 0.05,
        delta: float = 0.01,
    ) -> None:
        self.safety_predicate = safety_predicate
        self.test_suite = test_suite
        self.epsilon = epsilon
        self.delta = delta

    def evaluate(
        self,
        model_fn: Callable[[dict], str],
    ) -> tuple[bool, float]:
        """Run static test suite and check violation rate.

        Returns:
            (accepted, violation_rate)
        """
        violations = 0
        for interaction in self.test_suite:
            output = model_fn(interaction)
            result = self.safety_predicate.evaluate(interaction, output)
            if not result.is_safe:
                violations += 1

        n = len(self.test_suite)
        if n == 0:
            return True, 0.0

        # Create a single-region stats for acceptance check
        rs = RegionStats(
            region_id=0,
            n_samples=n,
            n_violations=violations,
            weight=1.0,
        )
        accepted, bound = check_acceptance([rs], self.epsilon, self.delta)
        return accepted, bound


def create_baseline_configs(
    n_byzantine: int = 0,
    verification_budget: int = 200,
) -> list[BaselineConfig]:
    """Create all baseline configurations from Section 7.2.

    Args:
        n_byzantine: Upper bound on Byzantine clients (for Krum).
        verification_budget: Total verification query budget.

    Returns:
        List of BaselineConfig for each baseline.
    """
    return [
        # Baseline 1a: FedAvg without gating
        BaselineConfig(
            name="FedAvg (no gating)",
            aggregator=FedAvg(),
            use_safety_gate=False,
        ),
        # Baseline 1b: FedAdam without gating
        BaselineConfig(
            name="FedAdam (no gating)",
            aggregator=FedAdam(),
            use_safety_gate=False,
        ),
        # Baseline 2a: Coordinate-wise median
        BaselineConfig(
            name="Median aggregation",
            aggregator=CoordinateMedian(),
            use_safety_gate=False,
        ),
        # Baseline 2b: Trimmed mean
        BaselineConfig(
            name="Trimmed mean aggregation",
            aggregator=TrimmedMean(beta=0.1),
            use_safety_gate=False,
        ),
        # Baseline 2c: Krum
        BaselineConfig(
            name="Krum aggregation",
            aggregator=Krum(n_byzantine=n_byzantine),
            use_safety_gate=False,
        ),
        # Baseline 3: Static safety suite
        BaselineConfig(
            name="Static safety suite",
            aggregator=FedAvg(),
            use_safety_gate=True,
            use_recursion=False,
            use_mkg=False,
            verification_budget=verification_budget,
        ),
        # Baseline 4: SCA without MKG
        BaselineConfig(
            name="SCA (no MKG)",
            aggregator=FedAvg(),
            use_safety_gate=True,
            use_recursion=True,
            use_mkg=False,
            verification_budget=verification_budget,
        ),
        # Baseline 5: Full SCA
        BaselineConfig(
            name="Full SCA (RLM + MKG)",
            aggregator=FedAvg(),
            use_safety_gate=True,
            use_recursion=True,
            use_mkg=True,
            verification_budget=verification_budget,
        ),
    ]


def build_verifier_for_config(
    config: BaselineConfig,
    safety_predicate: SafetyPredicate,
    embedder: InteractionEmbedder | None = None,
    n_regions: int = 10,
    reference_embeddings: np.ndarray | None = None,
) -> RLMVerifier | None:
    """Build an RLM verifier matching the baseline configuration.

    Args:
        config: Baseline config specifying recursion/MKG settings.
        safety_predicate: Safety predicate phi.
        embedder: Interaction embedder (defaults to RandomProjectionEmbedder).
        n_regions: Number of initial regions K.
        reference_embeddings: Reference data for initializing regions.

    Returns:
        RLMVerifier instance, or None if gating is disabled.
    """
    if not config.use_safety_gate:
        return None

    if embedder is None:
        embedder = RandomProjectionEmbedder()

    partition = RegionPartition(tau_new=2.0)
    if reference_embeddings is not None:
        partition.initialize_from_embeddings(reference_embeddings, n_regions)
    else:
        # Create dummy regions
        for j in range(n_regions):
            from sca.knowledge_graph.regions import InteractionRegion
            partition.regions.append(InteractionRegion(
                region_id=j,
                centroid=np.random.randn(64),
                weight=1.0 / n_regions,
            ))

    mkg = ModelKnowledgeGraph(partition=partition)

    max_depth = 3 if config.use_recursion else 0
    branching_factor = 4 if config.use_recursion else 0
    neighborhood_hops = 2 if config.use_mkg else 0

    return RLMVerifier(
        safety_predicate=safety_predicate,
        embedder=embedder,
        mkg=mkg,
        max_depth=max_depth,
        branching_factor=branching_factor,
        total_budget=config.verification_budget,
        neighborhood_hops=neighborhood_hops,
    )
