"""Tests for the leakage-proof split protocol (audit findings F1, F7, F9).

Network-dependent tests are marked ``network`` and skipped automatically when
the HuggingFace Hub is unreachable, so the structural tests always run.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch

from sca.experiments.data import (
    Corpus,
    LabeledSplit,
    SplitBundle,
    SplitRole,
    build_splits,
    dirichlet_client_partition,
    load_pku_saferlhf,
    make_hashing_tokenizer,
    three_way_split,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOK = make_hashing_tokenizer(vocab_size=997, max_len=8)


def make_corpus(n: int = 400, frac_unsafe: float = 0.4, seed: int = 0,
                siblings: bool = True) -> Corpus:
    """A structurally faithful stand-in used only to test the split logic.

    With ``siblings=True`` rows ``2r`` and ``2r+1`` share a prompt, mirroring
    PKU-SafeRLHF's two-responses-per-prompt layout.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < frac_unsafe).astype(np.int64)
    prompts = tuple(f"prompt {i // 2}" if siblings else f"prompt {i}"
                    for i in range(n))
    responses = tuple(f"response {i} word{i % 11} word{i % 13}" for i in range(n))
    return Corpus(
        name="unit-test-corpus",
        prompts=prompts,
        responses=responses,
        labels=labels,
        label_rule="synthetic; unit tests only",
        meta={"layout": "siblings" if siblings else "unique-prompts"},
    )


def hub_reachable() -> bool:
    try:
        from datasets import load_dataset

        load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train[:2]")
        return True
    except Exception:
        return False


needs_network = pytest.mark.skipif(
    not hub_reachable(), reason="HuggingFace Hub not reachable"
)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


class TestCorpus:
    def test_class_balance_is_measured(self):
        c = make_corpus(n=100, frac_unsafe=0.4, seed=1)
        bal = c.class_balance()
        assert bal["n"] == 100
        assert bal["n_safe"] + bal["n_unsafe"] == 100
        assert bal["frac_unsafe"] == bal["n_unsafe"] / 100

    def test_rejects_non_binary_labels(self):
        with pytest.raises(ValueError):
            Corpus(name="x", prompts=("a",), responses=("b",),
                   labels=np.array([2]), label_rule="")

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            Corpus(name="x", prompts=("a", "b"), responses=("b",),
                   labels=np.array([0, 1]), label_rule="")


# ---------------------------------------------------------------------------
# F1: leakage
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_all_pairwise_overlaps_are_zero(self):
        b = three_way_split(make_corpus(600), n_clients=5, seed=3, tokenizer=TOK)
        rep = b.overlap_report()
        assert rep["max_pairwise_overlap"] == 0
        assert rep["duplicated_ids"] == 0
        assert all(v == 0 for v in rep["pairwise_overlaps"].values())

    def test_assert_disjoint_passes_on_a_real_bundle(self):
        b = three_way_split(make_corpus(600), n_clients=5, seed=3, tokenizer=TOK)
        b.assert_disjoint()  # must not raise

    def test_assert_disjoint_actually_catches_the_f1_bug(self):
        """The exact old bug: test drawn from the client index range."""
        b = three_way_split(make_corpus(600), n_clients=4, seed=3, tokenizer=TOK)
        client_ids = b.client_pools[0].ids
        leaked = LabeledSplit(
            name="heldout_test",
            role=SplitRole.HELDOUT_TEST,
            ids=client_ids[:20].copy(),
            input_ids=b.client_pools[0].input_ids[:20],
            labels=b.client_pools[0].labels[:20],
        )
        b.heldout_test = leaked
        with pytest.raises(ValueError, match="F1"):
            b.assert_disjoint()

    def test_three_way_split_cannot_return_a_leaking_bundle(self):
        """assert_disjoint is called inside the constructor path."""
        for seed in range(6):
            b = three_way_split(make_corpus(500), n_clients=6, seed=seed,
                                tokenizer=TOK)
            assert b.overlap_report()["max_pairwise_overlap"] == 0

    def test_search_and_estimation_pools_are_disjoint(self):
        b = three_way_split(make_corpus(600), n_clients=4, seed=1, tokenizer=TOK)
        s = b.server_search_pool.id_set()
        e = b.server_estimation_pool.id_set()
        assert s and e
        assert not (s & e)
        assert s | e == b.server_verification_pool.id_set()

    def test_client_pools_are_disjoint_from_each_other(self):
        b = three_way_split(make_corpus(800), n_clients=8, seed=2, tokenizer=TOK)
        seen: set[int] = set()
        for p in b.client_pools:
            ids = p.id_set()
            assert not (ids & seen)
            seen |= ids

    def test_heldout_test_never_touches_client_data(self):
        b = three_way_split(make_corpus(700), n_clients=5, seed=4, tokenizer=TOK)
        test_ids = b.heldout_test.id_set()
        for p in b.client_pools:
            assert not (test_ids & p.id_set())

    def test_sibling_grouping_keeps_prompts_within_one_split(self):
        """A prompt may be shared between two client pools, and nowhere else."""
        b = three_way_split(make_corpus(600), n_clients=4, seed=5, tokenizer=TOK)
        assert b.config["grouped_by_prompt"] is True
        grep = b.group_overlap_report()
        assert grep["max_group_overlap"] == 0, grep["pairwise_group_overlaps"]

    def test_prompt_level_leakage_is_caught(self):
        """Row-level disjointness is not enough: a sibling row carries the
        same prompt, so it must be caught too."""
        b = three_way_split(make_corpus(600), n_clients=4, seed=5, tokenizer=TOK)
        # Take a test id and hand its SIBLING (different row, same prompt)
        # to the estimation pool.
        t = int(b.heldout_test.ids[0])
        sib = t + 1 if t % 2 == 0 else t - 1
        # Drop the sibling from the test split so that ROW-level disjointness
        # still holds -- only the prompt-level check can catch this.
        h = b.heldout_test
        keep = h.ids != sib
        b.heldout_test = LabeledSplit(
            name=h.name, role=h.role, ids=h.ids[keep],
            input_ids=h.input_ids[torch.as_tensor(keep)],
            labels=h.labels[torch.as_tensor(keep)],
        )
        e = b.server_estimation_pool
        b.server_estimation_pool = LabeledSplit(
            name=e.name, role=e.role,
            ids=np.concatenate([e.ids, np.array([sib])]),
            input_ids=torch.cat([e.input_ids, e.input_ids[:1]]),
            labels=torch.cat([e.labels, e.labels[:1]]),
        )
        v = b.server_verification_pool
        b.server_verification_pool = LabeledSplit(
            name=v.name, role=v.role,
            ids=np.concatenate([v.ids, np.array([sib])]),
            input_ids=torch.cat([v.input_ids, v.input_ids[:1]]),
            labels=torch.cat([v.labels, v.labels[:1]]),
        )
        with pytest.raises(ValueError, match="PROMPT-LEVEL LEAKAGE"):
            b.assert_disjoint()


# ---------------------------------------------------------------------------
# F7: non-IID via Dirichlet, not sort-and-chunk
# ---------------------------------------------------------------------------


class TestDirichletPartition:
    def test_partition_is_exact_and_disjoint(self):
        rng = np.random.default_rng(0)
        labels = (rng.random(400) < 0.4).astype(np.int64)
        ids = np.arange(400)
        blocks = dirichlet_client_partition(labels, ids, 6, 0.5, rng)
        assert sum(len(b) for b in blocks) == 400
        assert set(np.concatenate(blocks).tolist()) == set(ids.tolist())

    def test_no_empty_clients(self):
        rng = np.random.default_rng(1)
        labels = (rng.random(300) < 0.5).astype(np.int64)
        blocks = dirichlet_client_partition(labels, np.arange(300), 8, 0.3, rng)
        assert all(len(b) >= 2 for b in blocks)

    def test_high_alpha_is_more_iid_than_low_alpha(self):
        """The knob has to actually do something."""
        def skew(alpha, seed):
            rng = np.random.default_rng(seed)
            labels = (rng.random(2000) < 0.5).astype(np.int64)
            blocks = dirichlet_client_partition(labels, np.arange(2000), 10,
                                                alpha, rng)
            fr = [float((labels[b] == 1).mean()) for b in blocks]
            return float(np.std(fr))

        low = np.mean([skew(0.1, s) for s in range(5)])
        high = np.mean([skew(100.0, s) for s in range(5)])
        assert low > high

    def test_not_single_class_unlike_the_old_sort_and_chunk(self):
        """F7: sorted-and-chunked clients were single-class. Dirichlet(0.5)
        clients should be overwhelmingly multi-class."""
        b = three_way_split(make_corpus(2000, frac_unsafe=0.45), n_clients=8,
                            dirichlet_alpha=0.5, seed=7, tokenizer=TOK)
        multi = sum(1 for p in b.client_pools
                    if min(p.class_counts().values()) > 0)
        assert multi >= 6, [p.class_counts() for p in b.client_pools]

    def test_min_classes_constraint_can_force_multi_class_clients(self):
        b = three_way_split(make_corpus(2000, frac_unsafe=0.45), n_clients=8,
                            dirichlet_alpha=0.5, seed=7, tokenizer=TOK,
                            min_classes_per_client=2)
        assert b.summary()["n_single_class_clients"] == 0

    def test_single_class_client_count_is_reported(self):
        b = three_way_split(make_corpus(2000, frac_unsafe=0.45), n_clients=8,
                            dirichlet_alpha=0.5, seed=7, tokenizer=TOK)
        s = b.summary()
        assert "n_single_class_clients" in s
        assert s["n_single_class_clients"] <= 2

    def test_sorted_chunking_would_have_failed_that_test(self):
        """Demonstrates the contrast rather than asserting it in prose."""
        c = make_corpus(800, frac_unsafe=0.45)
        order = np.argsort(c.labels, kind="stable")
        chunks = np.array_split(order, 8)
        single = sum(1 for ch in chunks if len(set(c.labels[ch].tolist())) == 1)
        assert single >= 6


# ---------------------------------------------------------------------------
# Determinism (F9 / C4)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_ids(self):
        a = three_way_split(make_corpus(500), n_clients=5, seed=11, tokenizer=TOK)
        b = three_way_split(make_corpus(500), n_clients=5, seed=11, tokenizer=TOK)
        for x, y in zip(a.named_splits(), b.named_splits()):
            assert np.array_equal(x.ids, y.ids)

    def test_different_seed_different_ids(self):
        a = three_way_split(make_corpus(500), n_clients=5, seed=11, tokenizer=TOK)
        b = three_way_split(make_corpus(500), n_clients=5, seed=12, tokenizer=TOK)
        assert not np.array_equal(a.heldout_test.ids, b.heldout_test.ids)

    def test_byte_identical_across_processes_without_pythonhashseed(self):
        """C4: three separate interpreters, no PYTHONHASHSEED set."""
        code = (
            "import numpy as np;"
            "import sys; sys.path.insert(0, '.');"
            "from tests.test_data import make_corpus, TOK;"
            "from sca.experiments.data import three_way_split;"
            "b = three_way_split(make_corpus(400), n_clients=4, seed=9, tokenizer=TOK);"
            "print('|'.join(','.join(map(str, s.ids.tolist())) "
            "for s in b.named_splits()));"
            "print(int(b.heldout_test.input_ids.sum()))"
        )
        import os
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
        outs = [
            subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=repo, env=env, timeout=300)
            for _ in range(3)
        ]
        for o in outs:
            assert o.returncode == 0, o.stderr[-3000:]
        assert outs[0].stdout == outs[1].stdout == outs[2].stdout


# ---------------------------------------------------------------------------
# Structure / API
# ---------------------------------------------------------------------------


class TestSplitBundleAPI:
    def test_roles_are_tagged(self):
        b = three_way_split(make_corpus(400), n_clients=3, seed=1, tokenizer=TOK)
        assert all(p.role is SplitRole.CLIENT_TRAIN for p in b.client_pools)
        assert b.server_search_pool.role is SplitRole.SERVER_SEARCH
        assert b.server_estimation_pool.role is SplitRole.SERVER_ESTIMATION
        assert b.heldout_test.role is SplitRole.HELDOUT_TEST

    def test_tensors_match_ids(self):
        b = three_way_split(make_corpus(400), n_clients=3, seed=1, tokenizer=TOK)
        for s in b.named_splits():
            assert s.input_ids.shape[0] == len(s)
            assert s.labels.shape[0] == len(s)
            assert torch.equal(
                s.labels, torch.as_tensor(b.corpus.labels[s.ids], dtype=torch.long)
            )

    def test_to_tensor_dataset(self):
        b = three_way_split(make_corpus(400), n_clients=3, seed=1, tokenizer=TOK)
        ds = b.heldout_test.to_tensor_dataset()
        assert len(ds) == len(b.heldout_test)

    def test_summary_is_json_serialisable(self):
        import json

        b = three_way_split(make_corpus(400), n_clients=3, seed=1, tokenizer=TOK)
        json.dumps(b.summary(), default=str)

    def test_bad_fractions_rejected(self):
        with pytest.raises(ValueError):
            three_way_split(make_corpus(200), client_frac=0.8, server_frac=0.3,
                            test_frac=0.3, tokenizer=TOK)
        with pytest.raises(ValueError):
            three_way_split(make_corpus(200), search_frac=1.0, tokenizer=TOK)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


@needs_network
class TestRealPKU:
    def test_loads_real_rows(self):
        c = load_pku_saferlhf(n=60, seed=0)
        assert len(c) == 60
        assert c.name == "PKU-Alignment/PKU-SafeRLHF"
        assert all(isinstance(p, str) and p for p in c.prompts)
        assert "is_response_k_safe" in c.label_rule

    def test_sibling_layout_shares_prompts(self):
        c = load_pku_saferlhf(n=40, seed=0)
        for r in range(0, 40, 2):
            assert c.prompts[r] == c.prompts[r + 1]

    def test_class_balance_is_nontrivial(self):
        c = load_pku_saferlhf(n=400, seed=0)
        bal = c.class_balance()
        assert 0.05 < bal["frac_unsafe"] < 0.95, bal

    def test_deterministic_across_calls(self):
        a = load_pku_saferlhf(n=40, seed=5)
        b = load_pku_saferlhf(n=40, seed=5)
        assert a.prompts == b.prompts
        assert np.array_equal(a.labels, b.labels)

    def test_one_response_per_prompt_gives_iid_rows(self):
        c = load_pku_saferlhf(n=200, seed=1, one_response_per_prompt=True)
        assert len(c) == 200
        assert "sibling_layout" not in c.meta
        assert len(set(c.prompts)) == len(c.prompts)

    def test_iid_mode_splits_at_row_granularity(self):
        c = load_pku_saferlhf(n=400, seed=1, one_response_per_prompt=True)
        b = three_way_split(c, n_clients=4, seed=1, tokenizer=TOK)
        assert b.config["grouped_by_prompt"] is True
        b.assert_disjoint()
        prompts_by_split = {
            s.name: {c.prompts[i] for i in s.ids} for s in b.named_splits()
        }
        assert not (prompts_by_split["server_estimation"]
                    & prompts_by_split["server_search"])
        assert not (prompts_by_split["heldout_test"]
                    & prompts_by_split["server_estimation"])

    def test_build_splits_end_to_end(self):
        b = build_splits(n=200, seed=0, n_clients=4, max_len=32)
        b.assert_disjoint()
        assert b.overlap_report()["max_pairwise_overlap"] == 0
        assert len(b.heldout_test) > 0

    def test_no_synthetic_fallback(self):
        with pytest.raises(RuntimeError):
            load_pku_saferlhf(n=10, seed=0,
                              dataset_id="this-org/definitely-not-a-dataset")
