"""Tests for evaluation protocol and HEM scoring."""

import numpy as np
import pytest

from sca.certificate.certificate import SafetyCertificate
from sca.experiments.benchmarks import get_all_benchmarks
from sca.experiments.evaluation import (
    AblationConfig,
    ComprehensiveMetrics,
    EvaluationProtocol,
    HEMScore,
    HEMWeights,
    compute_comprehensive_metrics,
    compute_hem_score,
    generate_ablation_configs,
)
from sca.experiments.metrics import SafetyMetrics
from sca.knowledge_graph.mkg import ModelKnowledgeGraph
from sca.knowledge_graph.regions import InteractionRegion, RegionPartition
from sca.utils.stats import RegionStats
from sca.verifier.safety_predicate import KeywordSafetyPredicate


class TestHEMWeights:
    def test_normalize(self):
        w = HEMWeights(accuracy=1.0, safety=1.0, convergence=1.0,
                        efficiency=1.0, fairness=1.0, privacy=1.0)
        n = w.normalize()
        total = n.accuracy + n.safety + n.convergence + n.efficiency + n.fairness + n.privacy
        assert abs(total - 1.0) < 1e-10

    def test_custom_weights(self):
        w = HEMWeights(accuracy=0.1, safety=0.5)  # Safety-focused
        n = w.normalize()
        assert n.safety > n.accuracy


class TestHEMScore:
    def test_perfect_score(self):
        score = compute_hem_score(
            task_accuracy=1.0,
            safety_score=1.0,
            convergence_score=1.0,
            efficiency_score=1.0,
            fairness_score=1.0,
            privacy_score=1.0,
        )
        assert abs(score.aggregate - 1.0) < 1e-10

    def test_zero_score(self):
        score = compute_hem_score(
            task_accuracy=0.0,
            safety_score=0.0,
            convergence_score=0.0,
            efficiency_score=0.0,
            fairness_score=0.0,
            privacy_score=0.0,
        )
        assert abs(score.aggregate) < 1e-10

    def test_safety_dominant(self):
        """When safety weight is high, safety score dominates aggregate."""
        w = HEMWeights(accuracy=0.1, safety=0.8, convergence=0.025,
                        efficiency=0.025, fairness=0.025, privacy=0.025)
        score_safe = compute_hem_score(
            task_accuracy=0.5, safety_score=1.0,
            convergence_score=0.5, efficiency_score=0.5,
            fairness_score=0.5, privacy_score=0.5,
            weights=w,
        )
        score_unsafe = compute_hem_score(
            task_accuracy=1.0, safety_score=0.0,
            convergence_score=1.0, efficiency_score=1.0,
            fairness_score=1.0, privacy_score=1.0,
            weights=w,
        )
        assert score_safe.aggregate > score_unsafe.aggregate

    def test_components_preserved(self):
        score = compute_hem_score(
            task_accuracy=0.8, safety_score=0.9,
            convergence_score=0.7, efficiency_score=0.6,
            fairness_score=0.5, privacy_score=0.4,
        )
        assert score.components["accuracy"] == 0.8
        assert score.components["safety"] == 0.9
        assert score.components["fairness"] == 0.5


class TestComprehensiveMetrics:
    def test_compute_basic(self):
        predicate = KeywordSafetyPredicate(
            forbidden_keywords=["unsafe"],
            refusal_phrases=["I cannot"],
        )
        test_interactions = [
            {"prompt": f"test {i}", "requires_refusal": False}
            for i in range(10)
        ]

        def model_fn(interaction):
            return "This is a safe response."

        metrics = compute_comprehensive_metrics(
            model_fn=model_fn,
            test_interactions=test_interactions,
            safety_predicate=predicate,
            task_accuracy=0.85,
        )

        assert metrics.safety is not None
        assert metrics.safety.violation_rate == 0.0
        assert metrics.task_accuracy == 0.85
        assert metrics.hem_score is not None
        assert metrics.hem_score.aggregate > 0.0

    def test_with_region_stats(self):
        predicate = KeywordSafetyPredicate()
        test_interactions = [{"prompt": "test"}]

        def model_fn(interaction):
            return "Safe."

        region_stats = [
            RegionStats(region_id=0, n_samples=50, n_violations=2, weight=0.5),
            RegionStats(region_id=1, n_samples=50, n_violations=1, weight=0.5),
        ]

        metrics = compute_comprehensive_metrics(
            model_fn=model_fn,
            test_interactions=test_interactions,
            safety_predicate=predicate,
            region_stats=region_stats,
        )

        assert metrics.efficiency is not None
        assert metrics.efficiency.total_queries == 100

    def test_fairness_gap(self):
        predicate = KeywordSafetyPredicate()

        def model_fn(interaction):
            return "Safe."

        metrics = compute_comprehensive_metrics(
            model_fn=model_fn,
            test_interactions=[{"prompt": "test"}],
            safety_predicate=predicate,
            client_accuracies=[0.9, 0.8, 0.7, 0.6],
        )

        assert metrics.fairness_gap == pytest.approx(0.3, abs=1e-10)
        assert len(metrics.client_accuracies) == 4


class TestAblationConfigs:
    def test_generate(self):
        sweeps = generate_ablation_configs()
        assert len(sweeps) >= 5

        # Check that each sweep has multiple values
        for sweep in sweeps:
            assert len(sweep) >= 3
            assert all(isinstance(a, AblationConfig) for a in sweep)

    def test_depth_sweep(self):
        sweeps = generate_ablation_configs()
        depth_sweep = [s for s in sweeps if s[0].parameter == "max_depth"]
        assert len(depth_sweep) == 1
        depths = [a.value for a in depth_sweep[0]]
        assert 0 in depths  # Static (no recursion)
        assert 3 in depths  # Standard depth

    def test_epsilon_sweep(self):
        sweeps = generate_ablation_configs()
        eps_sweep = [s for s in sweeps if s[0].parameter == "epsilon"]
        assert len(eps_sweep) == 1
        epsilons = [a.value for a in eps_sweep[0]]
        assert all(0 < e < 1 for e in epsilons)


class TestEvaluationProtocol:
    def test_evaluate_over_refusal(self):
        predicate = KeywordSafetyPredicate(
            forbidden_keywords=["unsafe"],
            refusal_phrases=["I cannot"],
        )

        def model_fn_factory(model):
            def fn(interaction):
                return "Here is a helpful response."
            return fn

        protocol = EvaluationProtocol(
            model_fn_factory=model_fn_factory,
            safety_predicate=predicate,
            test_interactions=[{"prompt": "test"}],
            benchmarks=get_all_benchmarks(n_synthetic=5),
        )

        result = protocol.evaluate_over_refusal(
            model=None,  # model_fn_factory ignores model in this test
            benign_interactions=[
                {"prompt": "Write a poem"},
                {"prompt": "Help with homework"},
                {"prompt": "Explain science"},
            ],
        )

        assert result["over_refusal_rate"] == 0.0
        assert result["n_benign_tested"] == 3

    def test_compare_baselines(self):
        predicate = KeywordSafetyPredicate()

        def model_fn_factory(model):
            def fn(interaction):
                return "Response."
            return fn

        protocol = EvaluationProtocol(
            model_fn_factory=model_fn_factory,
            safety_predicate=predicate,
            test_interactions=[{"prompt": "test"}],
        )

        m1 = ComprehensiveMetrics(
            safety=SafetyMetrics(violation_rate=0.01),
            task_accuracy=0.9,
            hem_score=HEMScore(aggregate=0.85, components={}),
        )
        m2 = ComprehensiveMetrics(
            safety=SafetyMetrics(violation_rate=0.1),
            task_accuracy=0.95,
            hem_score=HEMScore(aggregate=0.75, components={}),
        )

        comparison = protocol.compare_baselines({
            "SCA": m1,
            "FedAvg": m2,
        })

        assert comparison["best_safety"] == "SCA"
        assert comparison["best_accuracy"] == "FedAvg"
        assert comparison["best_overall"] == "SCA"  # Higher HEM

    def test_interpretability_analysis(self):
        predicate = KeywordSafetyPredicate()

        def model_fn_factory(model):
            def fn(interaction):
                return "Response."
            return fn

        protocol = EvaluationProtocol(
            model_fn_factory=model_fn_factory,
            safety_predicate=predicate,
            test_interactions=[{"prompt": "test"}],
        )

        partition = RegionPartition(tau_new=5.0)
        for j in range(3):
            centroid = np.zeros(4)
            centroid[0] = j
            partition.regions.append(InteractionRegion(
                region_id=j, centroid=centroid, weight=1/3,
            ))
        mkg = ModelKnowledgeGraph(partition, tau=1.5)

        prev_stats = [
            RegionStats(region_id=0, n_samples=100, n_violations=2, weight=1/3),
            RegionStats(region_id=1, n_samples=100, n_violations=3, weight=1/3),
            RegionStats(region_id=2, n_samples=100, n_violations=1, weight=1/3),
        ]
        curr_stats = [
            RegionStats(region_id=0, n_samples=100, n_violations=20, weight=1/3),
            RegionStats(region_id=1, n_samples=100, n_violations=3, weight=1/3),
            RegionStats(region_id=2, n_samples=100, n_violations=1, weight=1/3),
        ]

        analysis = protocol.run_interpretability_analysis(
            mkg, prev_stats, curr_stats,
        )

        assert "regression_subgraph" in analysis
        assert "minimal_explanation" in analysis
        assert "region_deltas" in analysis
        assert 0 in analysis["regression_subgraph"]
        assert analysis["regression_size"] >= 1
