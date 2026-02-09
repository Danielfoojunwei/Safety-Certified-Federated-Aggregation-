"""Main experiment runner (Sections 5 and 7).

Orchestrates federated learning experiments with different attack
scenarios, aggregation methods, and safety gating configurations.
Collects metrics aligned with the theory (Section 7.3) and the
empirical benchmarking plan (Section 5.4).

Supports:
- Multi-baseline comparison under shared attack scenarios.
- Integration with safety benchmark suites (SafetyBench, JailbreakBench, etc.).
- Holistic Evaluation Metrics (HEM) aggregation.
- Ablation studies over key parameters.
- Comprehensive per-round and final evaluation.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from sca.certificate.acceptance import AcceptanceGate
from sca.experiments.attacks import (
    AttackConfig,
    AttackScenario,
    create_attack_scenario,
    create_standard_attack_scenarios,
)
from sca.experiments.baselines import BaselineConfig, build_verifier_for_config, create_baseline_configs
from sca.experiments.benchmarks import Benchmark, BenchmarkResult, get_all_benchmarks, run_benchmark_suite
from sca.experiments.evaluation import (
    ComprehensiveMetrics,
    EvaluationProtocol,
    EvaluationReport,
    HEMWeights,
    compute_comprehensive_metrics,
    compute_hem_score,
    generate_ablation_configs,
)
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


# ---------------------------------------------------------------------------
# Full Benchmarking Experiment Runner (Section 5 of benchmarking plan)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkingConfig:
    """Configuration for the full benchmarking evaluation pipeline.

    Extends ExperimentConfig with benchmark suite settings, HEM weights,
    ablation parameters, and attack scenario sweeps.

    Attributes:
        base_config: Base experiment configuration.
        attack_scenarios: List of attack scenarios to evaluate.
        benchmarks: Safety benchmark suites to run.
        hem_weights: HEM importance weights.
        run_ablations: Whether to run ablation studies.
        run_benchmarks: Whether to run benchmark suite evaluation.
        output_dir: Directory for saving results.
    """
    base_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    attack_scenarios: list[AttackScenario] | None = None
    benchmarks: list[Benchmark] | None = None
    hem_weights: HEMWeights | None = None
    run_ablations: bool = False
    run_benchmarks: bool = True
    output_dir: str | Path = "results"


@dataclass
class BenchmarkingResult:
    """Results from the full benchmarking pipeline.

    Attributes:
        scenario_results: Per-scenario, per-baseline experiment results.
        benchmark_results: Per-baseline benchmark suite results.
        ablation_results: Ablation study results.
        report: Evaluation report with comparisons.
    """
    scenario_results: dict[str, list[ExperimentResult]] = field(default_factory=dict)
    benchmark_results: dict[str, dict[str, BenchmarkResult]] = field(default_factory=dict)
    ablation_results: dict[str, list[dict]] = field(default_factory=dict)
    report: EvaluationReport | None = None


class BenchmarkingRunner:
    """Full benchmarking pipeline runner (Section 5 of evaluation plan).

    Orchestrates:
    1. Multi-scenario attack experiments.
    2. Benchmark suite evaluation (SafetyBench, JailbreakBench, etc.).
    3. HEM score computation.
    4. Ablation studies.
    5. Report generation.
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
        self.model_factory = model_factory
        self.client_data_factory = client_data_factory
        self.safety_predicate = safety_predicate
        self.model_fn_factory = model_fn_factory
        self.seed_interactions = seed_interactions
        self.test_interactions = test_interactions

        self.experiment_runner = ExperimentRunner(
            model_factory=model_factory,
            client_data_factory=client_data_factory,
            safety_predicate=safety_predicate,
            model_fn_factory=model_fn_factory,
            seed_interactions=seed_interactions,
            test_interactions=test_interactions,
        )

    def run(self, config: BenchmarkingConfig) -> BenchmarkingResult:
        """Run the full benchmarking pipeline.

        Args:
            config: Benchmarking configuration.

        Returns:
            BenchmarkingResult with all results.
        """
        result = BenchmarkingResult()

        # 1. Attack scenario experiments
        scenarios = config.attack_scenarios or create_standard_attack_scenarios(
            n_total_clients=config.base_config.n_clients,
        )

        for scenario in scenarios:
            logger.info(f"Running attack scenario: {scenario.name}")
            for attack_config in scenario.configs:
                exp_config = copy.copy(config.base_config)
                exp_config.attack_config = attack_config
                exp_results = self.experiment_runner.run(exp_config)
                result.scenario_results[scenario.name] = exp_results

        # 2. Benchmark suite evaluation
        if config.run_benchmarks:
            benchmarks = config.benchmarks or get_all_benchmarks(n_synthetic=50)

            # Run benchmarks on the final model from each baseline in the
            # first scenario
            first_scenario = list(result.scenario_results.values())
            if first_scenario:
                for exp_result in first_scenario[0]:
                    model = self.model_factory()
                    model_fn = self.model_fn_factory(model)
                    bench_results = run_benchmark_suite(model_fn, benchmarks)
                    result.benchmark_results[exp_result.config_name] = bench_results

        # 3. Ablation studies
        if config.run_ablations:
            ablation_configs = generate_ablation_configs()
            for sweep in ablation_configs:
                sweep_name = sweep[0].parameter if sweep else "unknown"
                sweep_results = []

                for abl in sweep:
                    logger.info(f"Ablation: {abl.name}")
                    exp_config = copy.copy(config.base_config)

                    # Apply ablation parameter
                    if abl.parameter == "epsilon":
                        exp_config.epsilon = abl.value
                    elif abl.parameter == "delta":
                        exp_config.delta = abl.value
                    elif abl.parameter == "total_budget":
                        exp_config.verification_budget = abl.value

                    # Only run the full SCA baseline for ablations
                    from sca.federated.aggregation import FedAvg
                    exp_config.baseline_configs = [
                        BaselineConfig(
                            name=f"SCA_{abl.name}",
                            aggregator=FedAvg(),
                            use_safety_gate=True,
                            use_recursion=True,
                            use_mkg=True,
                            verification_budget=exp_config.verification_budget,
                        ),
                    ]

                    exp_results = self.experiment_runner.run(exp_config)
                    if exp_results:
                        sweep_results.append({
                            "config": abl.name,
                            "parameter": abl.parameter,
                            "value": abl.value,
                            "summary": exp_results[0].summary,
                        })

                result.ablation_results[sweep_name] = sweep_results

        return result

    def save_results(
        self,
        result: BenchmarkingResult,
        output_dir: str | Path,
    ) -> None:
        """Save benchmarking results to disk.

        Args:
            result: Benchmarking results.
            output_dir: Output directory.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save scenario results
        for scenario_name, exp_results in result.scenario_results.items():
            scenario_data = []
            for er in exp_results:
                scenario_data.append({
                    "config_name": er.config_name,
                    "summary": er.summary,
                    "round_results": er.round_results,
                })
            with open(out / f"scenario_{scenario_name}.json", "w") as f:
                json.dump(scenario_data, f, indent=2, default=str)

        # Save benchmark results
        for baseline_name, bench_results in result.benchmark_results.items():
            bench_data = {
                name: {
                    "metrics": br.metrics,
                    "per_category": br.per_category_metrics,
                    "n_samples": br.n_samples,
                }
                for name, br in bench_results.items()
            }
            safe_name = baseline_name.replace(" ", "_").replace("/", "_")
            with open(out / f"benchmarks_{safe_name}.json", "w") as f:
                json.dump(bench_data, f, indent=2, default=str)

        # Save ablation results
        if result.ablation_results:
            with open(out / "ablation_results.json", "w") as f:
                json.dump(result.ablation_results, f, indent=2, default=str)

        logger.info(f"Results saved to {out}")
