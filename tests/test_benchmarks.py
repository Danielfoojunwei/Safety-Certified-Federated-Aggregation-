"""Tests for the benchmark suites (audit finding F12).

The central test is :class:`TestNoFabrication`: it walks every registered
non-synthetic suite and asserts that the samples it emits came from the Hub,
not from an f-string.  If someone reintroduces a ``_generate_synthetic`` path
behind a real citation, these fail.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from sca.experiments import benchmarks as B
from sca.experiments.benchmarks import (
    BENCHMARK_REGISTRY,
    REMOVED_SUITES,
    SYNTHETIC_REGISTRY,
    Benchmark,
    BenchmarkResult,
    BenchmarkSample,
    BenchmarkType,
    BenchmarkUnavailableError,
    JailbreakBenchSuite,
    SafetyBenchSuite,
    SafetyCategory,
    SyntheticSmokeSuite,
    ToxiGenSuite,
    TruthfulQASuite,
    get_all_benchmarks,
    looks_like_refusal,
    probe_benchmark_availability,
    run_benchmark_suite,
)


def hub_reachable() -> bool:
    try:
        from datasets import load_dataset

        load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful[:2]")
        return True
    except Exception:
        return False


needs_network = pytest.mark.skipif(
    not hub_reachable(), reason="HuggingFace Hub not reachable"
)


def refusing_model(interaction):
    return "I cannot help with that request."


def compliant_model(interaction):
    return "Sure, here is a detailed answer."


# ---------------------------------------------------------------------------
# Static guarantees -- these run without a network
# ---------------------------------------------------------------------------


class TestNoFabrication:
    """F12: a real citation requires real data."""

    def test_no_generate_synthetic_helper_survives(self):
        """No suite may still carry the fabrication entry point."""
        assert not hasattr(B, "_generate_synthetic")
        for cls in list(BENCHMARK_REGISTRY.values()) + list(
            SYNTHETIC_REGISTRY.values()
        ):
            assert not hasattr(cls, "_generate_synthetic"), cls.__name__
        # and no *real* suite's source may contain a generated-prompt f-string
        for name, cls in BENCHMARK_REGISTRY.items():
            src = inspect.getsource(cls)
            assert 'f"Safety question' not in src, name
            assert "_generate_synthetic" not in src, name

    def test_registry_contains_only_real_suites(self):
        for name, cls in BENCHMARK_REGISTRY.items():
            assert cls.is_synthetic is False, name
            assert cls.hub_dataset, f"{name} claims to be real but names no dataset"

    def test_synthetic_suites_are_named_synthetic_and_cite_nothing(self):
        for name, cls in SYNTHETIC_REGISTRY.items():
            assert cls.is_synthetic is True
            assert name.lower().startswith("synthetic")
            assert cls.__name__.startswith("Synthetic")
            assert cls.hub_dataset is None
            doc = cls.__doc__ or ""
            # No venue / year citations allowed on a synthetic suite.
            assert not re.search(r"(NeurIPS|ICLR|ACL|ICML|20\d\d)", doc), doc

    def test_deleted_suites_are_gone_and_documented(self):
        for gone in ("CASEBenchSuite", "HHHAlignmentSuite"):
            assert not hasattr(B, gone)
            assert gone in REMOVED_SUITES
            assert len(REMOVED_SUITES[gone]) > 40  # a real reason, not a stub

    def test_get_all_benchmarks_excludes_synthetic_by_default(self):
        names = {b.name for b in get_all_benchmarks()}
        assert "SyntheticSmoke" not in names
        assert "SyntheticSmoke" in {
            b.name for b in get_all_benchmarks(include_synthetic=True)
        }

    def test_legacy_n_synthetic_kwarg_warns_and_is_ignored(self):
        with pytest.warns(DeprecationWarning):
            b = SafetyBenchSuite(n_synthetic=50)
        assert not hasattr(b, "n_synthetic")

    def test_unavailable_dataset_raises_rather_than_falling_back(self):
        class Broken(SafetyBenchSuite):
            hub_dataset = "this-org/definitely-not-a-dataset"

        with pytest.raises(BenchmarkUnavailableError):
            Broken().load()


class TestPrimitives:
    def test_sample_creation(self):
        s = BenchmarkSample(id="t1", interaction={"prompt": "hi"},
                            ground_truth="greeting", category="general")
        assert s.id == "t1"

    def test_safety_categories(self):
        assert SafetyCategory.VIOLENCE.value == "violence"
        assert len(SafetyCategory) >= 6

    def test_refusal_detector(self):
        assert looks_like_refusal("I cannot do that")
        assert looks_like_refusal("I'm unable to comply")
        assert not looks_like_refusal("Here is how you do it")

    def test_pick_choice_prefers_letter(self):
        assert B._pick_choice("B) the second one", ["yes", "no"]) == 1
        assert B._pick_choice("A", ["yes", "no"]) == 0

    def test_pick_choice_falls_back_to_overlap(self):
        assert B._pick_choice("watermelon seeds pass through",
                              ["nothing at all", "watermelon seeds pass"]) == 1

    def test_pick_choice_returns_minus_one_on_no_match(self):
        assert B._pick_choice("zzzz", ["alpha", "beta"]) == -1


class TestSyntheticSmoke:
    def test_load_and_evaluate(self):
        s = SyntheticSmokeSuite(n_samples=20)
        samples = s.load()
        assert len(samples) == 20
        assert all(x.metadata["synthetic"] for x in samples)
        res = s.evaluate(refusing_model, samples)
        assert res.benchmark_name == "SyntheticSmoke"
        assert "WARNING" in res.provenance
        assert s.benchmark_type is BenchmarkType.SYNTHETIC

    def test_deterministic_without_pythonhashseed(self):
        a = [x.interaction["requires_refusal"] for x in SyntheticSmokeSuite(30).load()]
        b = [x.interaction["requires_refusal"] for x in SyntheticSmokeSuite(30).load()]
        assert a == b


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


@needs_network
class TestRealSafetyBench:
    def test_dev_config_is_labelled(self):
        s = SafetyBenchSuite(config="dev").load()
        assert len(s) == 35, "SafetyBench dev is 5 questions x 7 categories"
        assert all(x.ground_truth is not None for x in s)
        assert len({x.category for x in s}) == 7

    def test_questions_are_not_templated_placeholders(self):
        s = SafetyBenchSuite(config="dev").load()
        for x in s:
            assert not re.match(r"Safety question \d+", x.interaction["prompt"])
        assert len({x.interaction["prompt"] for x in s}) == len(s)

    def test_subsampling_keeps_all_categories(self):
        s = SafetyBenchSuite(config="dev", max_samples=14).load()
        assert len(s) == 14
        assert len({x.category for x in s}) == 7

    def test_evaluate(self):
        suite = SafetyBenchSuite(config="dev")
        res = suite.evaluate(refusing_model, suite.load())
        assert 0.0 <= res.metrics["safety_understanding_accuracy"] <= 1.0
        assert res.metrics["n_labelled"] == 35


@needs_network
class TestRealJailbreakBench:
    def test_loads_100_harmful_and_100_benign(self):
        s = JailbreakBenchSuite().load()
        harmful = [x for x in s if x.metadata["split"] == "harmful"]
        benign = [x for x in s if x.metadata["split"] == "benign"]
        assert len(harmful) == 100
        assert len(benign) == 100

    def test_goals_are_verbatim_by_default(self):
        s = JailbreakBenchSuite(include_benign=False).load()
        assert all(x.metadata["template"] == "verbatim" for x in s)
        assert not any("Ignore previous instructions" in x.interaction["prompt"]
                       for x in s)

    def test_sca_wrappers_are_labelled_as_ours(self):
        s = JailbreakBenchSuite(use_templates=True, include_benign=False).load()
        kinds = {x.metadata["template"] for x in s}
        assert kinds == {"verbatim", "sca_wrapper"}

    def test_max_samples_keeps_both_splits(self):
        s = JailbreakBenchSuite(max_samples=20).load()
        assert len([x for x in s if x.metadata["split"] == "harmful"]) == 10
        assert len([x for x in s if x.metadata["split"] == "benign"]) == 10

    def test_refusing_model_scores_zero_jailbreak_and_full_over_refusal(self):
        suite = JailbreakBenchSuite(max_samples=20)
        res = suite.evaluate(refusing_model, suite.load())
        assert res.metrics["jailbreak_success_rate"] == 0.0
        assert res.metrics["over_refusal_rate_on_benign"] == 1.0

    def test_compliant_model_scores_full_jailbreak(self):
        suite = JailbreakBenchSuite(max_samples=20)
        res = suite.evaluate(compliant_model, suite.load())
        assert res.metrics["jailbreak_success_rate"] == 1.0
        assert res.metrics["over_refusal_rate_on_benign"] == 0.0


@needs_network
class TestRealTruthfulQA:
    def test_817_questions(self):
        s = TruthfulQASuite().load()
        assert len(s) == 817

    def test_carries_real_reference_answers(self):
        s = TruthfulQASuite(max_samples=20).load()
        for x in s:
            assert x.ground_truth["correct"] and x.ground_truth["incorrect"]

    def test_oracle_beats_anti_oracle(self):
        suite = TruthfulQASuite(max_samples=60)
        samples = suite.load()
        gt = {s.id: s.ground_truth for s in samples}
        good = suite.evaluate(
            lambda i: "", samples)  # placeholder to keep signature symmetric
        by_id = {s.interaction["prompt"]: s.id for s in samples}

        def oracle(inter):
            return gt[by_id[inter["prompt"]]]["best"]

        def anti(inter):
            return gt[by_id[inter["prompt"]]]["incorrect"][0]

        r_good = suite.evaluate(oracle, samples).metrics["truthfulness_rate"]
        r_bad = suite.evaluate(anti, samples).metrics["truthfulness_rate"]
        assert r_good > r_bad
        assert good.n_samples == len(samples)

    def test_scoring_caveat_is_recorded(self):
        suite = TruthfulQASuite(max_samples=5)
        res = suite.evaluate(compliant_model, suite.load())
        assert "not comparable" in res.provenance["scoring"]


@needs_network
class TestRealToxiGen:
    def test_loads_annotated_subset_with_human_scores(self):
        s = ToxiGenSuite(max_samples=50).load()
        assert len(s) == 50
        assert all("toxicity_human" in x.metadata for x in s)
        assert {x.ground_truth for x in s} <= {"toxic", "benign"}

    def test_statements_are_real_text_not_placeholders(self):
        s = ToxiGenSuite(max_samples=50).load()
        assert len({x.interaction["statement"] for x in s}) > 40

    def test_deterministic_subsample(self):
        a = [x.id for x in ToxiGenSuite(max_samples=30, seed=3).load()]
        b = [x.id for x in ToxiGenSuite(max_samples=30, seed=3).load()]
        c = [x.id for x in ToxiGenSuite(max_samples=30, seed=4).load()]
        assert a == b
        assert a != c

    def test_oracle_detector_is_perfect(self):
        suite = ToxiGenSuite(max_samples=60)
        samples = suite.load()
        gt = {s.interaction["statement"]: s.ground_truth for s in samples}
        res = suite.evaluate(
            lambda i: "toxic" if gt[i["statement"]] == "toxic" else "benign",
            samples,
        )
        assert res.metrics["toxicity_detection_accuracy"] == 1.0


@needs_network
class TestSuiteRunner:
    def test_probe_reports_real_availability(self):
        rep = probe_benchmark_availability()
        for name in BENCHMARK_REGISTRY:
            assert rep[name]["available"] is True, rep[name]["error"]
            assert rep[name]["n_samples"] > 0
        assert "_removed" in rep

    def test_run_benchmark_suite(self):
        res = run_benchmark_suite(refusing_model, get_all_benchmarks(max_samples=10))
        assert set(res) == set(BENCHMARK_REGISTRY)
        assert all(isinstance(r, BenchmarkResult) for r in res.values())

    def test_results_are_json_serialisable(self):
        res = run_benchmark_suite(refusing_model, get_all_benchmarks(max_samples=6))
        json.dumps({k: {"metrics": v.metrics, "provenance": v.provenance}
                    for k, v in res.items()}, default=str)

    def test_runner_propagates_unavailability_by_default(self):
        class Broken(SafetyBenchSuite):
            hub_dataset = "this-org/definitely-not-a-dataset"

        with pytest.raises(BenchmarkUnavailableError):
            run_benchmark_suite(refusing_model, [Broken()])
        assert run_benchmark_suite(refusing_model, [Broken()],
                                   skip_unavailable=True) == {}
