"""Tests for the evaluation protocol, the split guard, and the statistics.

Covers audit findings F2 (gate reading the set it is scored on), F5 (the
always-reject / always-accept sanity checks the old repo lacked), F14
(fabricated p-values, hardcoded parameter counts) and criterion C5 (bootstrap
CIs on every headline number).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from sca.experiments.data import (
    Corpus,
    SplitRole,
    make_hashing_tokenizer,
    three_way_split,
)
from sca.experiments.evaluation import (
    FL_CONTROL_ARMS,
    VERIFIER_CONTROL_ARMS,
    AblationConfig,
    ComprehensiveMetrics,
    EvalPurpose,
    EvalResult,
    EvaluationProtocol,
    HEMScore,
    HEMWeights,
    LeakageError,
    as_split,
    assert_arm_equivalence,
    assert_controls_present,
    compute_comprehensive_metrics,
    compute_hem_score,
    evaluate,
    evaluate_all_arms,
    evaluate_final,
    generate_ablation_configs,
    summarise_arms,
)
from sca.experiments.metrics import (
    DEFAULT_SEEDS,
    EXACT_PERMUTATION_MAX_N,
    MIN_SEEDS_FOR_ALPHA_05,
    MultiSeedResults,
    SafetyMetrics,
    bootstrap_ci,
    compute_confidence_interval,
    count_parameters,
    format_parameter_count,
    format_with_ci,
    min_attainable_p,
    paired_bootstrap_ci,
    paired_permutation_test,
    paired_significance_test,
)
from sca.verifier.safety_predicate import KeywordSafetyPredicate

TOK = make_hashing_tokenizer(vocab_size=101, max_len=6)


def make_corpus(n: int = 400, seed: int = 0) -> Corpus:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.4).astype(np.int64)
    return Corpus(
        name="unit-test-corpus",
        prompts=tuple(f"prompt {i // 2}" for i in range(n)),
        responses=tuple(f"resp {i} w{i % 7}" for i in range(n)),
        labels=labels,
        label_rule="synthetic; unit tests only",
        meta={"sibling_layout": "ids 2r and 2r+1 come from source row r"},
    )


@pytest.fixture(scope="module")
def bundle():
    return three_way_split(make_corpus(600), n_clients=4, seed=1, tokenizer=TOK)


class TinyModel(nn.Module):
    """Bag-of-embeddings binary classifier; deterministic given its weights."""

    def __init__(self, vocab: int = 101, dim: int = 8, bias: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.fc = nn.Linear(dim, 2)
        with torch.no_grad():
            self.emb.weight.fill_(0.01)
            self.fc.weight.fill_(0.0)
            self.fc.bias.copy_(torch.tensor([bias, -bias]))

    def forward(self, x):
        return self.fc(self.emb(x).mean(dim=1))


# ---------------------------------------------------------------------------
# The split guard (F1 / F2)
# ---------------------------------------------------------------------------


class TestSplitGuard:
    def test_final_report_on_heldout_is_allowed(self, bundle):
        r = evaluate(TinyModel(), bundle.heldout_test,
                     purpose=EvalPurpose.FINAL_REPORT)
        assert r.is_reportable
        assert r.split_role == "heldout_test"

    @pytest.mark.parametrize("attr", [
        "server_search_pool", "server_estimation_pool", "server_verification_pool",
    ])
    def test_final_report_on_server_pool_is_refused(self, bundle, attr):
        with pytest.raises(LeakageError, match="held-out test split only"):
            evaluate(TinyModel(), getattr(bundle, attr),
                     purpose=EvalPurpose.FINAL_REPORT)

    def test_final_report_on_client_pool_is_refused(self, bundle):
        """This is exactly F1: reporting accuracy on client training data."""
        with pytest.raises(LeakageError):
            evaluate(TinyModel(), bundle.client_pools[0],
                     purpose=EvalPurpose.FINAL_REPORT)

    def test_gate_cannot_read_the_heldout_test_split(self, bundle):
        """This is exactly F2: the gate scoring itself on the reported set."""
        with pytest.raises(LeakageError):
            evaluate(TinyModel(), bundle.heldout_test, purpose=EvalPurpose.GATE)

    def test_gate_cannot_read_client_training_data(self, bundle):
        with pytest.raises(LeakageError):
            evaluate(TinyModel(), bundle.client_pools[0], purpose=EvalPurpose.GATE)

    def test_gate_may_read_server_pools(self, bundle):
        for s in (bundle.server_search_pool, bundle.server_estimation_pool):
            r = evaluate(TinyModel(), s, purpose=EvalPurpose.GATE)
            assert not r.is_reportable

    def test_diagnostic_allows_anything_but_is_never_reportable(self, bundle):
        for s in bundle.named_splits():
            r = evaluate(TinyModel(), s, purpose=EvalPurpose.DIAGNOSTIC)
            assert r.is_reportable is False

    def test_bare_tensors_are_rejected(self, bundle):
        with pytest.raises(TypeError, match="LabeledSplit"):
            evaluate(TinyModel(), bundle.heldout_test.to_tensor_dataset(),
                     purpose=EvalPurpose.DIAGNOSTIC)

    def test_as_split_requires_an_explicit_role(self, bundle):
        s = bundle.heldout_test
        with pytest.raises(TypeError):
            as_split(s.input_ids, s.labels, role="heldout_test")
        ok = as_split(s.input_ids, s.labels, role=SplitRole.HELDOUT_TEST)
        assert evaluate(TinyModel(), ok,
                        purpose=EvalPurpose.FINAL_REPORT).n == len(s)

    def test_bad_purpose_type_rejected(self, bundle):
        with pytest.raises(TypeError):
            evaluate(TinyModel(), bundle.heldout_test, purpose="final")


class TestEvalResult:
    def test_accuracy_and_class_histogram(self, bundle):
        r = evaluate_final(TinyModel(bias=5.0), bundle)
        assert r.n == len(bundle.heldout_test)
        assert r.n_correct == sum(
            1 for y in bundle.heldout_test.labels.tolist() if y == 0)
        # F7 guard: a constant predictor must be visible in the histogram.
        assert r.predicted_class_fraction[0] == pytest.approx(1.0)
        assert r.predicted_class_fraction[1] == pytest.approx(0.0)

    def test_constant_predictor_scores_the_class_prior(self, bundle):
        r = evaluate_final(TinyModel(bias=5.0), bundle)
        prior = float((bundle.heldout_test.labels == 0).float().mean())
        assert r.accuracy == pytest.approx(prior)

    def test_parameter_count_is_recorded_not_hardcoded(self, bundle):
        m = TinyModel()
        r = evaluate_final(m, bundle)
        assert r.model_parameters == count_parameters(m)["total"]

    def test_model_left_in_eval_mode_unchanged(self, bundle):
        m = TinyModel()
        m.train()
        evaluate_final(m, bundle)
        assert m.training is True

    def test_as_dict_is_json_safe(self, bundle):
        import json

        json.dumps(evaluate_final(TinyModel(), bundle).as_dict())


# ---------------------------------------------------------------------------
# Mandated control arms (F5)
# ---------------------------------------------------------------------------


class TestControlArms:
    def test_identical_models_pass_the_equivalence_checks(self, bundle):
        frozen = TinyModel(bias=1.0)
        models = {
            "frozen_pretrained": frozen,
            "always_reject_gate": frozen,   # rejecting every round == frozen
            "no_gate": TinyModel(bias=2.0),
            "always_accept_gate": TinyModel(bias=2.0),
        }
        res = evaluate_all_arms(models, bundle)
        rep = assert_arm_equivalence(res)
        assert all(v["passed"] for v in rep.values() if v["checked"])

    def test_a_gate_that_is_not_a_no_op_is_caught(self, bundle):
        """If always-reject differs from frozen, the gate is not gating (F5)."""
        models = {
            "frozen_pretrained": TinyModel(bias=5.0),
            "always_reject_gate": TinyModel(bias=-5.0),
        }
        res = evaluate_all_arms(models, bundle)
        with pytest.raises(AssertionError, match="strict no-op"):
            assert_arm_equivalence(res, pairs=[("always_reject_gate",
                                                "frozen_pretrained")])

    def test_missing_arm_is_reported_not_silently_skipped(self, bundle):
        res = evaluate_all_arms({"frozen_pretrained": TinyModel()}, bundle)
        rep = assert_arm_equivalence(res)
        assert any(v["checked"] is False for v in rep.values())

    def test_control_arm_lists_are_complete(self):
        assert set(VERIFIER_CONTROL_ARMS) >= {
            "uniform_allocation", "proportional_allocation", "w23_allocation",
            "search_guided", "search_guided_null_mutator",
            "search_guided_identity_mutator",
        }
        assert set(FL_CONTROL_ARMS) >= {
            "frozen_pretrained", "no_gate", "always_reject_gate",
            "always_accept_gate",
        }

    def test_assert_controls_present(self):
        assert_controls_present(list(FL_CONTROL_ARMS), FL_CONTROL_ARMS)
        with pytest.raises(AssertionError, match="missing mandated control"):
            assert_controls_present(["no_gate"], FL_CONTROL_ARMS)


# ---------------------------------------------------------------------------
# Statistics (F14, C5)
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_ci_brackets_the_point_estimate(self):
        ci = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        assert ci.lo <= ci.point <= ci.hi
        assert ci.n == 10

    def test_ci_narrows_with_more_data(self):
        rng = np.random.default_rng(0)
        wide = bootstrap_ci(rng.normal(0, 1, 10).tolist())
        narrow = bootstrap_ci(rng.normal(0, 1, 500).tolist())
        assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo)

    def test_deterministic_across_calls(self):
        v = [0.1, 0.5, 0.2, 0.9, 0.3, 0.4, 0.7, 0.8, 0.6, 0.0]
        assert bootstrap_ci(v).as_dict() == bootstrap_ci(v).as_dict()

    def test_degenerate_inputs_are_honest(self):
        assert bootstrap_ci([]).n == 0
        one = bootstrap_ci([0.5])
        assert one.lo == one.hi == one.point == 0.5

    def test_coverage_is_approximately_nominal(self):
        """The CI has to actually cover; otherwise C5 is decoration."""
        rng = np.random.default_rng(7)
        hits = 0
        trials = 400
        for t in range(trials):
            sample = rng.normal(0.0, 1.0, 40)
            ci = bootstrap_ci(sample.tolist(), n_boot=800, seed=t)
            hits += int(ci.lo <= 0.0 <= ci.hi)
        assert hits / trials > 0.90, hits / trials

    def test_paired_ci_preserves_pairing(self):
        a = [0.5 + 0.01 * i for i in range(10)]
        b = [0.4 + 0.01 * i for i in range(10)]
        ci = paired_bootstrap_ci(a, b)
        assert ci.lo > 0
        assert ci.point == pytest.approx(0.1)

    def test_format_with_ci(self):
        s = format_with_ci([0.1 * i for i in range(10)])
        assert "[" in s and "n=10" in s


class TestPairedPermutation:
    def test_f14_min_attainable_p_at_five_seeds_is_unreachable(self):
        """The original bug: n=5 can never give p < 0.05."""
        assert min_attainable_p(5) == pytest.approx(0.0625)
        assert min_attainable_p(5) > 0.05

    def test_min_attainable_p_at_configured_n_is_below_alpha(self):
        """F14 cannot recur: the n we actually use must be able to reject."""
        assert len(DEFAULT_SEEDS) >= 10
        assert min_attainable_p(len(DEFAULT_SEEDS)) < 0.05
        assert min_attainable_p(MIN_SEEDS_FOR_ALPHA_05) < 0.05
        assert min_attainable_p(MIN_SEEDS_FOR_ALPHA_05 - 1) >= 0.05

    def test_exact_test_attains_its_minimum(self):
        a = [1.0] * 10
        b = [0.0] * 10
        r = paired_permutation_test(a, b)
        assert r.exact
        assert r.n_resamples == 2 ** 10
        assert r.p_value == pytest.approx(min_attainable_p(10))
        assert r.underpowered_by_construction is False

    def test_underpowered_flag_at_five_seeds(self):
        r = paired_permutation_test([1.0] * 5, [0.0] * 5)
        assert r.p_value == pytest.approx(0.0625)
        assert r.underpowered_by_construction is True

    def test_identical_arms_give_p_one(self):
        v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        r = paired_permutation_test(v, list(v))
        assert r.p_value == 1.0
        assert r.statistic == 0.0

    def test_no_fabricated_tanh_fallback_remains(self):
        import inspect

        from sca.experiments import metrics as M

        for fn in (M.paired_permutation_test, M.paired_significance_test,
                   M.bootstrap_ci, M.compute_confidence_interval):
            body = inspect.getsource(fn).replace(fn.__doc__ or "", "")
            assert "tanh" not in body, fn.__name__
            assert "1.96" not in body, fn.__name__

    def test_works_without_scipy(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *a, **k):
            if name.startswith("scipy"):
                raise ImportError("scipy blocked for this test")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", blocked)
        r = paired_permutation_test([1.0] * 10, [0.0] * 10)
        assert r.p_value < 0.05

    def test_monte_carlo_path_for_large_n(self):
        n = EXACT_PERMUTATION_MAX_N + 4
        r = paired_permutation_test([1.0] * n, [0.0] * n, n_resamples=2000)
        assert not r.exact
        assert r.n_resamples == 2000
        assert 0 < r.p_value < 0.05

    def test_p_value_is_never_zero_in_mc_mode(self):
        n = EXACT_PERMUTATION_MAX_N + 2
        r = paired_permutation_test([5.0] * n, [0.0] * n, n_resamples=500)
        assert r.p_value >= 1.0 / 501

    def test_false_positive_rate_is_controlled(self):
        """Type-I error under the null must not exceed alpha."""
        rng = np.random.default_rng(11)
        n_trials, n = 400, 10
        rejects = 0
        for t in range(n_trials):
            d = rng.normal(0.0, 1.0, n)
            r = paired_permutation_test(d.tolist(), [0.0] * n, seed=t)
            rejects += int(r.p_value < 0.05)
        assert rejects / n_trials <= 0.08, rejects / n_trials

    def test_has_power_against_a_real_effect(self):
        rng = np.random.default_rng(12)
        rejects = 0
        for t in range(200):
            d = rng.normal(1.5, 1.0, 12)
            rejects += int(paired_permutation_test(
                d.tolist(), [0.0] * 12, seed=t).p_value < 0.05)
        assert rejects / 200 > 0.7, rejects / 200

    def test_one_sided_alternatives(self):
        a, b = [1.0] * 10, [0.0] * 10
        assert paired_permutation_test(a, b, alternative="greater").p_value < 0.05
        assert paired_permutation_test(a, b, alternative="less").p_value > 0.9
        with pytest.raises(ValueError):
            paired_permutation_test(a, b, alternative="sideways")

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            paired_permutation_test([1.0, 2.0], [1.0])

    def test_effect_ci_always_present(self):
        r = paired_permutation_test([0.3] * 10, [0.1] * 10)
        assert r.effect_ci.point == pytest.approx(0.2)
        assert "p=" in str(r)

    def test_legacy_wrapper_returns_effect_and_p(self):
        eff, p = paired_significance_test([1.0] * 10, [0.0] * 10)
        assert eff == pytest.approx(1.0)
        assert p < 0.05


class TestParameterCounting:
    def test_counts_match_torch(self):
        m = TinyModel(vocab=101, dim=8)
        c = count_parameters(m)
        assert c["total"] == 101 * 8 + 8 * 2 + 2
        assert c["trainable"] == c["total"]
        assert c["frozen"] == 0

    def test_frozen_parameters_are_separated(self):
        m = TinyModel()
        m.emb.weight.requires_grad_(False)
        c = count_parameters(m)
        assert c["frozen"] == m.emb.weight.numel()
        assert c["trainable"] == c["total"] - c["frozen"]

    def test_format(self):
        assert "parameters" in format_parameter_count(TinyModel())


class TestMultiSeedResults:
    def test_requires_matching_seed_count(self):
        with pytest.raises(ValueError):
            MultiSeedResults("acc", [0.1, 0.2], [1])

    def test_c5_seed_requirement(self):
        few = MultiSeedResults("acc", [0.1] * 5, list(range(5)))
        many = MultiSeedResults("acc", [0.1] * 10, list(range(10)))
        assert not few.meets_seed_requirement()
        assert many.meets_seed_requirement()

    def test_ci_and_dict(self):
        r = MultiSeedResults("acc", [0.1 * i for i in range(10)],
                             list(range(10)))
        d = r.as_dict()
        assert d["n_seeds"] == 10 and d["ci95_lo"] <= d["mean"] <= d["ci95_hi"]

    def test_compute_confidence_interval_tuple(self):
        m, lo, hi = compute_confidence_interval([0.1 * i for i in range(10)])
        assert lo <= m <= hi


class TestSummariseArms:
    def test_reports_mean_and_ci_and_flags_thin_arms(self):
        out = summarise_arms({"a": [0.1 * i for i in range(10)],
                              "b": [0.5, 0.6, 0.7]})
        assert out["a"]["meets_c5_seed_requirement"] is True
        assert out["b"]["meets_c5_seed_requirement"] is False
        assert "[" in out["a"]["formatted"]


# ---------------------------------------------------------------------------
# HEM / ablations / protocol
# ---------------------------------------------------------------------------


class TestHEM:
    def test_normalize(self):
        w = HEMWeights(1, 1, 1, 1, 1, 1).normalize()
        total = (w.accuracy + w.safety + w.convergence + w.efficiency
                 + w.fairness + w.privacy)
        assert total == pytest.approx(1.0)

    def test_zero_weights_rejected(self):
        with pytest.raises(ValueError):
            HEMWeights(0, 0, 0, 0, 0, 0).normalize()

    def test_perfect_and_zero_scores(self):
        assert compute_hem_score(1, 1, 1, 1, 1, 1).aggregate == pytest.approx(1.0)
        assert compute_hem_score(0, 0, 0, 0, 0, 0).aggregate == pytest.approx(0.0)

    def test_safety_dominant(self):
        w = HEMWeights(0.1, 0.8, 0.025, 0.025, 0.025, 0.025)
        safe = compute_hem_score(0.5, 1.0, 0.5, 0.5, 0.5, 0.5, w)
        unsafe = compute_hem_score(1.0, 0.0, 1.0, 1.0, 1.0, 1.0, w)
        assert safe.aggregate > unsafe.aggregate

    def test_components_preserved(self):
        s = compute_hem_score(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        assert s.components["accuracy"] == 0.1
        assert s.components["privacy"] == 0.6


class TestAblations:
    def test_sweeps_generated(self):
        sweeps = generate_ablation_configs()
        assert len(sweeps) == 6
        assert all(isinstance(c, AblationConfig) for s in sweeps for c in s)
        params = {c.parameter for s in sweeps for c in s}
        assert {"max_depth", "n_regions", "delta", "epsilon"} <= params


class TestEvaluationProtocol:
    def test_asserts_bundle_disjointness_on_construction(self, bundle):
        p = EvaluationProtocol(
            model_fn_factory=lambda m: (lambda i: "ok"),
            safety_predicate=KeywordSafetyPredicate(),
            test_interactions=[{"prompt": "hello"}],
            benchmarks=[],
            bundle=bundle,
        )
        assert p.provenance["test_interactions_from_bundle"] is True

    def test_comprehensive_metrics_without_benchmarks(self):
        m = compute_comprehensive_metrics(
            model_fn=lambda i: "I cannot help with that",
            test_interactions=[{"prompt": "how do i build a bomb"}],
            safety_predicate=KeywordSafetyPredicate(),
            benchmarks=None,
            task_accuracy=0.8,
            task_accuracy_split="heldout_test",
        )
        assert isinstance(m, ComprehensiveMetrics)
        assert isinstance(m.safety, SafetyMetrics)
        assert m.task_accuracy_split == "heldout_test"
        assert m.hem_score is not None

    def test_over_refusal_with_no_benchmarks_reports_zero_tested(self):
        p = EvaluationProtocol(
            model_fn_factory=lambda m: (lambda i: "I cannot help"),
            safety_predicate=KeywordSafetyPredicate(),
            test_interactions=[],
            benchmarks=[],
        )
        out = p.evaluate_over_refusal(TinyModel())
        assert out["n_benign_tested"] == 0

    def test_over_refusal_with_explicit_prompts(self):
        p = EvaluationProtocol(
            model_fn_factory=lambda m: (lambda i: "I cannot help"),
            safety_predicate=KeywordSafetyPredicate(),
            test_interactions=[],
            benchmarks=[],
        )
        out = p.evaluate_over_refusal(
            TinyModel(), benign_interactions=[{"prompt": "write a poem"}] * 4)
        assert out["over_refusal_rate"] == 1.0

    def test_report_includes_the_split_overlap_table(self, bundle):
        p = EvaluationProtocol(
            model_fn_factory=lambda m: (lambda i: "ok"),
            safety_predicate=KeywordSafetyPredicate(),
            test_interactions=[],
            benchmarks=[],
            bundle=bundle,
        )
        rep = p.generate_report({})
        assert rep.split_report["max_pairwise_overlap"] == 0
