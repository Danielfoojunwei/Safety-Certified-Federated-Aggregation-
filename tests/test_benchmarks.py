"""Tests for benchmark suite integration."""

import numpy as np
import pytest

from sca.experiments.benchmarks import (
    BenchmarkResult,
    BenchmarkSample,
    BenchmarkType,
    CASEBenchSuite,
    HHHAlignmentSuite,
    JailbreakBenchSuite,
    SafetyBenchSuite,
    SafetyCategory,
    ToxiGenSuite,
    TruthfulQASuite,
    get_all_benchmarks,
    run_benchmark_suite,
)


class TestBenchmarkSamples:
    def test_sample_creation(self):
        s = BenchmarkSample(
            id="test_1",
            interaction={"prompt": "Hello"},
            ground_truth="greeting",
            category="general",
        )
        assert s.id == "test_1"
        assert s.interaction["prompt"] == "Hello"

    def test_safety_categories(self):
        assert SafetyCategory.VIOLENCE.value == "violence"
        assert len(SafetyCategory) >= 6


class TestSafetyBench:
    def test_load_synthetic(self):
        bench = SafetyBenchSuite(n_synthetic=20)
        samples = bench.load()
        assert len(samples) == 20
        assert all(isinstance(s, BenchmarkSample) for s in samples)
        assert bench.benchmark_type == BenchmarkType.SAFETY

    def test_evaluate_safe_model(self):
        """Safe model that refuses unsafe prompts and answers safe ones."""
        bench = SafetyBenchSuite(n_synthetic=20)
        samples = bench.load()

        def model_fn(interaction):
            if interaction.get("requires_refusal"):
                return "I cannot help with that request."
            return "Here is a helpful answer to your question."

        result = bench.evaluate(model_fn, samples)
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "SafetyBench"
        assert result.n_samples == 20
        assert "safety_understanding_accuracy" in result.metrics
        assert result.metrics["safety_understanding_accuracy"] > 0.5

    def test_evaluate_unsafe_model(self):
        """Model that never refuses -- should have low accuracy on unsafe items."""
        bench = SafetyBenchSuite(n_synthetic=20)
        samples = bench.load()

        def model_fn(interaction):
            return "Here is the information you requested."

        result = bench.evaluate(model_fn, samples)
        # Some samples require refusal, so accuracy should be < 1.0
        assert result.metrics["safety_understanding_accuracy"] < 1.0

    def test_to_verifier_interactions(self):
        bench = SafetyBenchSuite(n_synthetic=10)
        samples = bench.load()
        interactions = bench.to_verifier_interactions(samples)
        assert len(interactions) == 10
        assert all(isinstance(x, dict) for x in interactions)


class TestJailbreakBench:
    def test_load_synthetic(self):
        bench = JailbreakBenchSuite(n_synthetic=30)
        samples = bench.load()
        assert len(samples) == 30
        misuse = [s for s in samples if s.category == "misuse"]
        benign = [s for s in samples if s.category == "benign"]
        assert len(misuse) > 0
        assert len(benign) > 0

    def test_evaluate_resistant_model(self):
        """Model that always refuses jailbreaks."""
        bench = JailbreakBenchSuite(n_synthetic=20)
        samples = bench.load()

        def model_fn(interaction):
            if interaction.get("requires_refusal"):
                return "I refuse to help with that."
            return "Here is a helpful response."

        result = bench.evaluate(model_fn, samples)
        assert result.metrics["jailbreak_success_rate"] == 0.0
        assert result.metrics["refusal_accuracy"] == 1.0
        assert result.metrics["over_refusal_rate"] == 0.0

    def test_evaluate_vulnerable_model(self):
        """Model that complies with everything -- vulnerable to jailbreaks."""
        bench = JailbreakBenchSuite(n_synthetic=20)
        samples = bench.load()

        def model_fn(interaction):
            return "Sure, here is what you asked for."

        result = bench.evaluate(model_fn, samples)
        # Should have high jailbreak success rate
        assert result.metrics["jailbreak_success_rate"] > 0.0
        assert result.metrics["refusal_accuracy"] < 1.0


class TestTruthfulQA:
    def test_load_synthetic(self):
        bench = TruthfulQASuite(n_synthetic=15)
        samples = bench.load()
        assert len(samples) == 15

    def test_evaluate(self):
        bench = TruthfulQASuite(n_synthetic=10)
        samples = bench.load()

        def model_fn(interaction):
            # Generic response that may or may not be truthful
            return "No, that is a common misconception."

        result = bench.evaluate(model_fn, samples)
        assert "truthful_rate" in result.metrics
        assert "informative_rate" in result.metrics


class TestToxiGen:
    def test_load_synthetic(self):
        bench = ToxiGenSuite(n_synthetic=20)
        samples = bench.load()
        assert len(samples) == 20
        toxic = [s for s in samples if s.category == "toxic"]
        benign = [s for s in samples if s.category == "benign"]
        assert len(toxic) > 0
        assert len(benign) > 0

    def test_evaluate(self):
        bench = ToxiGenSuite(n_synthetic=20)
        samples = bench.load()

        def model_fn(interaction):
            if "trust" in interaction.get("prompt", "").lower():
                return "I cannot continue that harmful statement."
            return "This is a positive continuation."

        result = bench.evaluate(model_fn, samples)
        assert "toxicity_detection_rate" in result.metrics
        assert "benign_pass_rate" in result.metrics
        assert "balanced_accuracy" in result.metrics


class TestCASEBench:
    def test_load_synthetic(self):
        bench = CASEBenchSuite(n_synthetic=25)
        samples = bench.load()
        assert len(samples) == 25
        contexts = set(s.category for s in samples)
        assert len(contexts) > 1  # Multiple contexts

    def test_evaluate_context_aware(self):
        bench = CASEBenchSuite(n_synthetic=20)
        samples = bench.load()

        def model_fn(interaction):
            if interaction.get("requires_refusal"):
                return "I cannot provide that information."
            return "Here is the context-appropriate response."

        result = bench.evaluate(model_fn, samples)
        assert "context_aware_accuracy" in result.metrics
        assert "over_refusal_rate" in result.metrics
        assert "under_refusal_rate" in result.metrics
        assert len(result.per_category_metrics) > 0


class TestHHHAlignment:
    def test_load_synthetic(self):
        bench = HHHAlignmentSuite(n_synthetic=15)
        samples = bench.load()
        assert len(samples) == 15

    def test_evaluate(self):
        bench = HHHAlignmentSuite(n_synthetic=20)
        samples = bench.load()

        def model_fn(interaction):
            if interaction.get("requires_refusal"):
                return "I cannot help with harmful requests."
            return "Here is a helpful and detailed response to your question."

        result = bench.evaluate(model_fn, samples)
        assert "helpfulness" in result.metrics
        assert "honesty" in result.metrics
        assert "harmlessness" in result.metrics
        assert "hhh_aggregate" in result.metrics


class TestBenchmarkRegistry:
    def test_get_all_benchmarks(self):
        benchmarks = get_all_benchmarks(n_synthetic=10)
        assert len(benchmarks) >= 6
        names = {b.name for b in benchmarks}
        assert "SafetyBench" in names
        assert "JailbreakBench" in names
        assert "TruthfulQA" in names

    def test_run_benchmark_suite(self):
        def model_fn(interaction):
            if interaction.get("requires_refusal"):
                return "I cannot assist with that."
            return "Here is a helpful response."

        benchmarks = get_all_benchmarks(n_synthetic=5)
        results = run_benchmark_suite(model_fn, benchmarks)
        assert len(results) == len(benchmarks)
        for name, result in results.items():
            assert isinstance(result, BenchmarkResult)
            assert result.n_samples > 0
