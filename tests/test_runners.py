"""Tests for the rebuilt experiment runners.

Audit finding **F13** was that 105 tests passed while every module that
produced a paper number had 0% coverage -- replacing all nine of them with
``raise RuntimeError`` still yielded ``105 passed``.  The runner is the module
that produces every number, so these tests actually execute it: the smoke
tests below run the full verifier arm and the full FL arm end to end on a
synthetic corpus (no network) and assert on what comes out.

Deliberately NOT tested here: whether H2 or H3 hold.  Those are measurements,
not invariants, and a test that asserted them would be the mechanism by which
the previous version made its hypotheses pass.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from sca.experiments import data as data_mod
from sca.experiments.data import Corpus, make_hashing_tokenizer, three_way_split
from sca.experiments.evaluation import EvalPurpose, LeakageError, evaluate
from sca.experiments.gate import (
    CertifiedGate,
    build_partition,
    build_response_predicate,
    interactions_from_split,
)
from sca.experiments.model import (
    MODEL_REGISTRY,
    MLPSafetyClassifier,
    REFUSAL_TEXT,
    SafetyClassifier,
    make_model,
    make_model_fn,
    parameter_count,
    pretrain,
)
from sca.utils.paths import repo_root

VOCAB = 512
MAXLEN = 16


# ---------------------------------------------------------------------------
# A synthetic corpus with a learnable signal, so no test needs the Hub.
# ---------------------------------------------------------------------------


def _synthetic_corpus(n: int = 320, seed: int = 0) -> Corpus:
    rng = np.random.default_rng(seed)
    prompts, responses, labels = [], [], []
    for i in range(n):
        unsafe = bool(rng.integers(0, 2))
        topic = int(rng.integers(0, 40))
        prompts.append(f"prompt {i} about topic {topic} number {i % 7}")
        # The label-bearing token differs, so a bag-of-words model can learn it
        # and the response-keyed predicate has distinct keys per row.
        marker = "harmful" if unsafe else "benign"
        responses.append(f"response {i} {marker} content variant {i % 5}")
        labels.append(1 if unsafe else 0)
    return Corpus(
        name="synthetic-test-corpus",
        prompts=tuple(prompts),
        responses=tuple(responses),
        labels=np.asarray(labels, dtype=np.int64),
        label_rule="synthetic: label 1 iff the response contains 'harmful'",
        meta={"synthetic": True},
    )


@pytest.fixture
def bundle():
    return three_way_split(
        _synthetic_corpus(),
        n_clients=2,
        seed=0,
        tokenizer=make_hashing_tokenizer(vocab_size=VOCAB, max_len=MAXLEN),
    )


@pytest.fixture
def tokenizer():
    return make_hashing_tokenizer(vocab_size=VOCAB, max_len=MAXLEN)


@pytest.fixture
def patched_data(monkeypatch):
    """Point the runner's data loader at the synthetic corpus (no network)."""

    def fake_load(n, seed, **kw):
        return _synthetic_corpus(n=n, seed=seed)

    monkeypatch.setattr(data_mod, "load_pku_saferlhf", fake_load)
    monkeypatch.setattr(
        data_mod,
        "make_hf_tokenizer",
        lambda *a, **k: make_hashing_tokenizer(vocab_size=VOCAB, max_len=MAXLEN),
    )
    return True


# ==========================================================================
class TestModel:
    def test_bow_starts_at_exactly_zero_and_is_seed_independent(self):
        a = make_model(seed=1, vocab_size=VOCAB)
        b = make_model(seed=99, vocab_size=VOCAB)
        assert torch.equal(a.embedding.weight, torch.zeros_like(a.embedding.weight))
        for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            assert torch.equal(pa, pb)

    def test_mlp_init_is_deterministic_but_seed_dependent(self):
        a = make_model(seed=1, kind="mlp", vocab_size=VOCAB)
        b = make_model(seed=1, kind="mlp", vocab_size=VOCAB)
        c = make_model(seed=2, kind="mlp", vocab_size=VOCAB)
        assert torch.equal(a.embedding.weight, b.embedding.weight)
        assert not torch.equal(a.embedding.weight, c.embedding.weight)
        # LayerNorm gains must start at 1, not at a random draw.
        assert torch.equal(a.norm1.weight, torch.ones_like(a.norm1.weight))

    def test_registry_covers_both_architectures(self):
        assert set(MODEL_REGISTRY) == {"bow", "mlp"}
        assert isinstance(make_model(kind="bow", vocab_size=VOCAB), SafetyClassifier)
        assert isinstance(
            make_model(kind="mlp", vocab_size=VOCAB), MLPSafetyClassifier
        )
        with pytest.raises(ValueError):
            make_model(kind="nope")

    def test_parameter_count_is_programmatic(self):
        """F14: the README's parameter count must never be a literal."""
        m = make_model(vocab_size=VOCAB)
        pc = parameter_count(m)
        assert pc["total"] == VOCAB * 2 + 2
        assert pc["total"] == sum(p.numel() for p in m.parameters())

    def test_pretrain_moves_the_model(self, bundle):
        m = make_model(vocab_size=VOCAB)
        before = m.embedding.weight.clone()
        pretrain(m, bundle.server_search_pool, epochs=3, lr=0.5, seed=0)
        assert not torch.equal(before, m.embedding.weight)


class TestModelFnIsAReleaseFilter:
    def test_refuses_when_it_predicts_unsafe_and_releases_otherwise(self, tokenizer):
        m = make_model(vocab_size=VOCAB)
        x = {"prompt": "p", "response": "r", "context": ""}

        with torch.no_grad():                      # force "unsafe" -> refuse
            m.bias.copy_(torch.tensor([-5.0, 5.0]))
        assert make_model_fn(m, tokenizer)(x) == REFUSAL_TEXT

        with torch.no_grad():                      # force "safe" -> release
            m.bias.copy_(torch.tensor([5.0, -5.0]))
        assert make_model_fn(m, tokenizer)(x) == "r"

    def test_violation_is_not_classification_error(self, tokenizer, bundle):
        """The structural fix for F2, stated as a test.

        The old gate computed ``is_violation = (pred != true_label)``.  Here a
        model that wrongly REFUSES a safe response makes a classification error
        and commits no violation, so the two quantities cannot be the same
        function.
        """
        corpus = bundle.corpus
        phi = build_response_predicate(corpus, bundle.server_estimation_pool)
        ix = interactions_from_split(bundle.server_estimation_pool, corpus)
        safe_x = next(x for x in ix if corpus.labels[x["id"]] == 0)

        m = make_model(vocab_size=VOCAB)
        with torch.no_grad():                      # refuse everything
            m.bias.copy_(torch.tensor([-5.0, 5.0]))
        y = make_model_fn(m, tokenizer)(safe_x)

        assert y == REFUSAL_TEXT                   # wrong classification
        assert phi.evaluate(safe_x, y).is_safe     # but NOT a safety violation


class TestPredicateAndInteractions:
    def test_interactions_carry_no_label(self, bundle):
        ix = interactions_from_split(bundle.server_search_pool, bundle.corpus)
        assert ix
        for x in ix:
            assert set(x) == {"id", "prompt", "response", "context"}
            assert "label" not in x

    def test_predicate_is_keyed_on_the_response_not_the_prompt(self, bundle):
        corpus = bundle.corpus
        phi = build_response_predicate(corpus, bundle.server_estimation_pool)
        ix = interactions_from_split(bundle.server_estimation_pool, corpus)
        unsafe = next(x for x in ix if corpus.labels[x["id"]] == 1)
        safe = next(x for x in ix if corpus.labels[x["id"]] == 0)
        # Same response, different prompts -> same verdict.
        a = phi.evaluate(unsafe, unsafe["response"])
        b = phi.evaluate({**safe, "prompt": "totally different"},
                         unsafe["response"])
        assert a.is_safe is b.is_safe is False

    def test_refusal_is_registered_safe(self, bundle):
        phi = build_response_predicate(bundle.corpus, bundle.server_estimation_pool)
        assert phi.evaluate({"prompt": "p"}, REFUSAL_TEXT).is_safe


# ==========================================================================
def _gate(bundle, tokenizer, **kw):
    from sca.knowledge_graph.embedding import make_embedder

    corpus = bundle.corpus
    emb = make_embedder("hashed", seed=0, fields=["prompt", "response"])
    search = interactions_from_split(bundle.server_search_pool, corpus)
    estim = interactions_from_split(bundle.server_estimation_pool, corpus)
    part = build_partition(search, emb, k_total=kw.pop("k_total", 3), seed=0)
    return CertifiedGate(
        partition=part,
        embedder=emb,
        predicate=build_response_predicate(
            corpus, bundle.server_search_pool, bundle.server_estimation_pool
        ),
        tokenizer=tokenizer,
        search_pool=search,
        estimation_pool=estim,
        epsilon=kw.pop("epsilon", 0.6),
        delta=kw.pop("delta", 0.05),
        k_total=part.k,
        budget_cap=kw.pop("budget_cap", 30),
        n_rounds=kw.pop("n_rounds", 1),
        search_budget=kw.pop("search_budget", 20),
        estimation_budget=kw.pop("estimation_budget", 60),
        seed=0,
        **kw,
    )


class TestCertifiedGate:
    def test_mismatched_k_total_raises(self, bundle, tokenizer):
        """F10: K may never be re-derived after the partition is fitted."""
        from sca.knowledge_graph.embedding import make_embedder

        corpus = bundle.corpus
        emb = make_embedder("hashed", seed=0, fields=["prompt", "response"])
        search = interactions_from_split(bundle.server_search_pool, corpus)
        estim = interactions_from_split(bundle.server_estimation_pool, corpus)
        part = build_partition(search, emb, k_total=3, seed=0)
        with pytest.raises(ValueError, match="a-priori"):
            CertifiedGate(
                partition=part, embedder=emb,
                predicate=build_response_predicate(corpus,
                                                   bundle.server_estimation_pool),
                tokenizer=make_hashing_tokenizer(vocab_size=VOCAB, max_len=MAXLEN),
                search_pool=search, estimation_pool=estim,
                epsilon=0.5, delta=0.05,
                k_total=5,          # <- disagrees with the fitted partition
                budget_cap=30, n_rounds=1,
            )

    def test_self_check_proves_phi_depends_on_the_output(self, bundle, tokenizer):
        info = _gate(bundle, tokenizer).self_check()
        assert info["output_sensitive"] is True
        assert info["n_annotations"] > 1

    def test_effective_delta_is_divided_by_the_number_of_rounds(
        self, bundle, tokenizer
    ):
        """Builder A negative result 5: T rounds at nominal delta is wrong by T."""
        g = _gate(bundle, tokenizer, n_rounds=8, delta=0.08)
        assert g.effective_delta == pytest.approx(0.01)

    def test_gate_returns_a_verified_certificate(self, bundle, tokenizer):
        from sca.certificate.certificate import verify_certificate

        m = make_model(vocab_size=VOCAB)
        d = _gate(bundle, tokenizer).evaluate(m, round_num=1)
        ok, problems = verify_certificate(d.certificate)
        assert ok, problems
        assert d.bound == pytest.approx(d.certificate.bound_value)
        assert 0.0 <= d.bound <= 1.0

    def test_control_modes_do_not_change_the_recorded_bound(self, bundle, tokenizer):
        """always_accept / always_reject must be pure decision overrides."""
        from sca.certificate.acceptance import GateMode

        m = make_model(vocab_size=VOCAB)
        out = {}
        for mode in (GateMode.CERTIFIED, GateMode.ALWAYS_ACCEPT,
                     GateMode.ALWAYS_REJECT):
            out[mode] = _gate(bundle, tokenizer, mode=mode).evaluate(m, 1)
        assert out[GateMode.ALWAYS_ACCEPT].accepted is True
        assert out[GateMode.ALWAYS_REJECT].accepted is False
        bounds = {m_: d.bound for m_, d in out.items()}
        assert len(set(bounds.values())) == 1, bounds

    def test_search_and_estimation_pools_are_disjoint(self, bundle, tokenizer):
        g = _gate(bundle, tokenizer)
        a = {x["id"] for x in g.search_pool}
        b = {x["id"] for x in g.estimation_pool}
        assert not (a & b)

    def test_estimation_counts_never_exceed_the_a_priori_cap(
        self, bundle, tokenizer
    ):
        g = _gate(bundle, tokenizer, budget_cap=25, estimation_budget=60)
        _, result = g.verify_model(make_model(vocab_size=VOCAB), 0)
        assert max(result.allocation.values()) <= 25
        for s in result.estimation_stats:
            assert s.n_samples <= 25


# ==========================================================================
class TestEndToEndArms:
    """The smoke tests F13 says the old repo did not have."""

    def _cfg(self):
        from sca.experiments.run_all import ExperimentConfig

        c = ExperimentConfig.smoke_config()
        c.n_corpus = 240
        c.k_total = 3
        c.budget_cap = 25
        c.search_budget = 15
        c.estimation_budget = 45
        c.seeds = [0]
        c.n_rounds = 2
        c.aggregators = ["fedavg"]
        c.attacks = ["sign_flip"]
        return c

    def test_verifier_arm_runs_every_mandated_control(self, patched_data):
        from sca.experiments.evaluation import VERIFIER_CONTROL_ARMS
        from sca.experiments.run_all import run_verifier_arm

        payload = run_verifier_arm(self._cfg())
        arms = {r["arm"] for r in payload["rows"]}
        assert set(VERIFIER_CONTROL_ARMS) <= arms, sorted(arms)
        for r in payload["rows"]:
            # Both counts, always (the F8 requirement).
            assert "total_violations_raw" in r
            assert "distinct_violations_dedup_depth0" in r
            assert r["distinct_violations_dedup_depth0"] <= r["distinct_sources"]
            assert r["k_total"] == 3 and r["budget_cap"] == 25
            # F4: the MKG must not be the complete graph.
            assert r["mkg_edge_density"] < 1.0

    def test_verifier_arm_reports_zero_split_overlap(self, patched_data):
        from sca.experiments.run_all import run_verifier_arm

        payload = run_verifier_arm(self._cfg())
        rep = payload["per_seed_setup"]["0"]["split_overlaps"]
        assert rep["max_pairwise_overlap"] == 0
        assert rep["duplicated_ids"] == 0

    def test_fl_arm_runs_and_the_two_identities_hold(self, patched_data):
        from sca.experiments.evaluation import FL_CONTROL_ARMS
        from sca.experiments.run_all import run_fl_arm

        payload = run_fl_arm(self._cfg())
        by_arm = {r["arm"]: r for r in payload["rows"]}
        assert set(FL_CONTROL_ARMS) <= set(by_arm), sorted(by_arm)

        # F5: rejection must be a genuine no-op, checked on parameter bytes.
        assert (by_arm["always_reject_gate"]["final_hash"]
                == by_arm["frozen_pretrained"]["final_hash"])
        assert (by_arm["always_accept_gate"]["final_hash"]
                == by_arm["no_gate"]["final_hash"])
        assert by_arm["always_reject_gate"]["rounds_accepted"] == 0

    def test_fl_arm_carries_the_legacy_gaussian_control(self, patched_data):
        from sca.experiments.run_all import ExperimentConfig, run_fl_arm

        cfg = self._cfg()
        cfg.attacks = ["sign_flip", "legacy_gaussian"]
        payload = run_fl_arm(cfg)
        arms = {r["arm"] for r in payload["rows"]}
        assert any("legacy_gaussian" in a for a in arms)
        # F6's signature must remain visible: the control is labelled, and the
        # attack diagnostics are recorded on every attacked row.
        gaussian = [r for r in payload["rows"]
                    if r.get("attack") == "legacy_gaussian"]
        assert gaussian
        assert all(r["mean_attack_diagnostics"] is not None for r in gaussian)

    def test_results_json_is_serialisable_and_carries_provenance(
        self, patched_data, results_tmpdir
    ):
        from sca.experiments.run_all import run_verifier_arm

        payload = run_verifier_arm(self._cfg())
        text = json.dumps(payload, indent=2, default=str)
        again = json.loads(text)
        prov = again["provenance"]
        assert prov["config"]["k_total"] == 3
        assert "git" in prov and "versions" in prov
        assert prov["versions"]["torch"]


class TestEvaluationIsRoleGuarded:
    def test_gate_cannot_score_itself_on_the_test_split(self, bundle):
        m = make_model(vocab_size=VOCAB)
        with pytest.raises(LeakageError):
            evaluate(m, bundle.heldout_test, purpose=EvalPurpose.GATE)

    def test_final_report_cannot_come_from_a_server_pool(self, bundle):
        m = make_model(vocab_size=VOCAB)
        with pytest.raises(LeakageError):
            evaluate(m, bundle.server_estimation_pool,
                     purpose=EvalPurpose.FINAL_REPORT)


class TestConsolidation:
    """Task 3: one reachable entry point, and the discredited ones are gone."""

    LEGACY = [
        "run_experiment.py", "run_real_evaluation.py",
        "run_novelty_validation.py", "run_neurips_exp1.py",
        "run_neurips_exp2.py", "run_neurips_exp3.py",
    ]

    def test_legacy_runners_are_deleted(self):
        d = repo_root() / "sca" / "experiments"
        present = [n for n in self.LEGACY if (d / n).exists()]
        assert not present, f"legacy runners still present: {present}"

    def test_entry_points_are_importable_and_have_a_main_guard(self):
        import sca.experiments.make_figures as mf
        import sca.experiments.run_all as ra

        assert callable(ra.main) and callable(mf.main)
        for mod in (ra, mf):
            src = (repo_root() / mod.__file__).read_text() \
                if not mod.__file__.startswith("/") else open(mod.__file__).read()
            assert 'if __name__ == "__main__":' in src

    @pytest.mark.subprocess
    def test_run_all_help_works_from_a_fresh_interpreter(self, child_env):
        out = subprocess.run(
            [sys.executable, "-m", "sca.experiments.run_all", "--help"],
            capture_output=True, text=True, env=child_env, timeout=180,
        )
        assert out.returncode == 0, out.stderr
        assert "--smoke" in out.stdout


class TestArtifactDeterminism:
    """C4 support. The full check is ``make repro-check``; these are the units."""

    def test_nan_and_inf_become_null_so_the_json_is_standard(self):
        from sca.experiments.run_all import _sanitise

        out = _sanitise({"a": float("nan"), "b": [float("inf"), 1.0],
                         "c": {"d": float("-inf")}})
        assert out == {"a": None, "b": [None, 1.0], "c": {"d": None}}
        json.dumps(out, allow_nan=False)   # would raise on a leftover NaN

    def test_deterministic_mode_suppresses_the_timestamp(self, monkeypatch):
        from sca.experiments.run_all import (
            DETERMINISTIC_SENTINEL,
            ExperimentConfig,
            deterministic_mode,
            provenance,
        )

        monkeypatch.delenv("SCA_DETERMINISTIC", raising=False)
        assert deterministic_mode() is False
        assert provenance(ExperimentConfig())["generated_at_utc"] \
            != DETERMINISTIC_SENTINEL

        monkeypatch.setenv("SCA_DETERMINISTIC", "1")
        assert deterministic_mode() is True
        p = provenance(ExperimentConfig(), {"wall_seconds": 1.23})
        assert p["generated_at_utc"] == DETERMINISTIC_SENTINEL
        assert "wall_seconds" not in p

    def test_gate_records_the_binding_hash_not_the_timestamped_one(
        self, bundle, tokenizer
    ):
        """SafetyCertificate.timestamp makes certificate_hash() process-varying."""
        m = make_model(vocab_size=VOCAB)
        d = _gate(bundle, tokenizer).evaluate(m, round_num=1)
        assert "certificate_hash" not in d.info
        assert d.info["certificate_binding_hash"] == d.certificate.binding_hash

    def test_verifier_arm_payload_is_identical_across_two_runs(self, patched_data):
        from sca.experiments.run_all import run_verifier_arm

        cfg = TestEndToEndArms()._cfg()
        a = run_verifier_arm(cfg)["rows"]
        b = run_verifier_arm(cfg)["rows"]
        for ra, rb in zip(a, b):
            ra = {k: v for k, v in ra.items() if k != "wall_seconds"}
            rb = {k: v for k, v in rb.items() if k != "wall_seconds"}
            assert ra == rb


class TestMakeFigures:
    def test_renders_a_verifier_payload_without_matplotlib(
        self, patched_data, results_tmpdir
    ):
        from sca.experiments.make_figures import main, render_verifier_table
        from sca.experiments.run_all import run_verifier_arm

        cfg = TestEndToEndArms()._cfg()
        payload = run_verifier_arm(cfg)
        md = render_verifier_table(payload)
        assert "certified bound" in md
        assert "DEDUP" in md          # both counts must appear

        (results_tmpdir / "smoke_verifier_arm.json").write_text(
            json.dumps(payload, default=str)
        )
        assert main(["--smoke"]) == 0
        assert (results_tmpdir / "smoke_README_NUMBERS.md").exists()

    def test_fails_loudly_when_there_is_nothing_to_render(self, results_tmpdir):
        from sca.experiments.make_figures import main

        assert main(["--smoke"]) == 1
