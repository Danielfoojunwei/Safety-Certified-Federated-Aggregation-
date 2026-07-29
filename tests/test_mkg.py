"""Tests for the region partition, the embedders, and the Model Knowledge Graph.

Three defects are pinned here:

* Cross-process embedding reproducibility.  ``TestEmbeddingReproducibility``
  spawns real subprocesses, because within a single process ``hash()`` is
  perfectly consistent and an in-process test cannot see the bug.
* Graph degeneracy.  ``TestGraphDensity`` asserts the MKG is neither complete
  nor edgeless, and that ``neighborhood_hops`` actually changes the focus set.
* Regression-subgraph blindness.  ``TestRegressionSubgraph`` contains the exact
  failure that the UCB-difference rule missed: a region going 0.000 -> 1.000.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from sca.knowledge_graph.embedding import (
    HashedBagOfWordsEmbedder,
    PrecomputedEmbedder,
    make_embedder,
)
from sca.knowledge_graph.mkg import (
    DegenerateGraphError,
    ModelKnowledgeGraph,
    RegressionReport,
)
from sca.knowledge_graph.regions import Region, RegionPartition
from sca.utils.stats import RegionStat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_child(code: str) -> str:
    """Run ``code`` in a fresh interpreter with PYTHONHASHSEED explicitly unset."""
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    env["PYTHONPATH"] = REPO_ROOT
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=300,
    )
    if out.returncode != 0:
        raise RuntimeError(f"child failed:\nSTDOUT{out.stdout}\nSTDERR{out.stderr}")
    return out.stdout.strip()


# ======================================================================
class TestEmbeddingReproducibility:
    """Cross-process determinism.  This is the fix for the hash() defect."""

    def test_builtin_hash_really_is_unstable_across_processes(self):
        """Control: prove this test methodology can detect the bug at all.

        If CPython ever stopped randomising string hashing, the reproducibility
        test below would pass vacuously.  This asserts the hazard is real in
        this interpreter.
        """
        vals = {
            _run_child("print(hash('safety violation token'))") for _ in range(5)
        }
        assert len(vals) > 1, (
            "builtin hash() was stable across 5 processes; the subprocess "
            "reproducibility test can no longer detect the original defect"
        )

    def test_hashed_bow_embedding_is_byte_identical_across_processes(self):
        code = """
            import hashlib
            import numpy as np
            from sca.knowledge_graph.embedding import HashedBagOfWordsEmbedder
            e = HashedBagOfWordsEmbedder(embed_dim=32, seed=42)
            v = e.embed({"prompt": "how do I bypass the safety filter", "context": "ctx"})
            print(hashlib.sha256(np.ascontiguousarray(v, dtype=np.float64).tobytes()).hexdigest())
        """
        digests = {_run_child(code) for _ in range(3)}
        assert len(digests) == 1, f"embeddings differ across processes: {digests}"

    def test_stable_hash_is_identical_across_processes(self):
        code = """
            from sca.utils.seeding import stable_hash
            print(stable_hash("safety violation token"))
        """
        vals = {_run_child(code) for _ in range(3)}
        assert len(vals) == 1

    def test_whole_partition_is_identical_across_processes(self):
        """Determinism must survive embed -> k-means -> stratify -> weights."""
        code = """
            import hashlib, json
            import numpy as np
            from sca.knowledge_graph.embedding import HashedBagOfWordsEmbedder
            from sca.knowledge_graph.regions import RegionPartition
            from sca.knowledge_graph.mkg import ModelKnowledgeGraph
            texts = [{"prompt": f"request {i} regarding subject {i%9}"} for i in range(120)]
            e = HashedBagOfWordsEmbedder(embed_dim=32, seed=7)
            V = e.embed_batch(texts)
            p = RegionPartition(k=6, seed=7).fit(V)
            p.stratify(V)
            g = ModelKnowledgeGraph(p, tau=None, target_density=0.2, strict=False)
            payload = json.dumps({
                "w": p.empirical_weights(),
                "tau": round(g.tau, 12),
                "edges": sorted(map(sorted, g.graph.edges())),
            }, sort_keys=True)
            print(hashlib.sha256(payload.encode()).hexdigest())
        """
        digests = {_run_child(code) for _ in range(3)}
        assert len(digests) == 1, f"partitions differ across processes: {digests}"


# ======================================================================
class TestRegionPartition:
    def make(self, k=4, n=200, dim=16, seed=0):
        rng = np.random.default_rng(seed)
        centres = rng.standard_normal((k, dim)) * 6.0
        V = np.concatenate(
            [centres[j] + rng.standard_normal((n // k, dim)) for j in range(k)]
        )
        return RegionPartition(k=k, seed=seed).fit(V), V

    def test_fit_creates_exactly_k_regions(self):
        part, _ = self.make(k=4)
        assert part.k == 4 == part.k_declared
        assert part.is_fitted

    def test_assign_never_creates_a_region(self):
        """K is part of the a-priori union bound; the partition cannot grow."""
        part, _ = self.make(k=4, dim=16)
        far = np.full(16, 1e6)
        j = part.assign(far)
        assert 0 <= j < 4
        assert part.k == 4

    def test_assign_matches_assign_batch(self):
        part, V = self.make(k=5)
        batch = part.assign_batch(V)
        for i in range(0, len(V), 17):
            assert part.assign(V[i]) == batch[i]

    def test_stratify_partitions_the_pool_exactly(self):
        part, V = self.make(k=5, n=250)
        strata = part.stratify(V)
        assert set(strata) == set(range(5))
        flat = sorted(i for ids in strata.values() for i in ids)
        assert flat == list(range(len(V)))

    def test_weights_are_empirical_and_sum_to_one(self):
        part, V = self.make(k=5, n=250)
        part.stratify(V)
        w = part.empirical_weights()
        assert len(w) == 5
        assert abs(sum(w.values()) - 1.0) < 1e-12
        for j, ws in w.items():
            assert abs(ws - len(part.strata[j]) / len(V)) < 1e-12

    def test_empty_stratum_gets_weight_zero_and_is_not_dropped(self):
        part = RegionPartition(k=3, seed=0).set_centroids(
            np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0]])
        )
        pool = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.1, 0.0]])
        part.stratify(pool)
        w = part.empirical_weights()
        assert set(w) == {0, 1, 2}
        assert w[2] == 0.0
        assert part.sample_from_region(2, 5, np.random.default_rng(0)) == []

    def test_sample_from_region_is_iid_with_replacement_inside_the_stratum(self):
        part, V = self.make(k=4, n=400)
        strata = part.stratify(V)
        rng = np.random.default_rng(0)
        j = max(strata, key=lambda r: len(strata[r]))
        draws = part.sample_from_region(j, 500, rng)
        assert len(draws) == 500
        assert set(draws) <= set(strata[j])
        # With replacement: 500 draws from a stratum smaller than that must repeat.
        assert len(set(draws)) < 500

    def test_sample_from_region_is_reproducible_for_a_given_rng(self):
        part, V = self.make(k=4)
        part.stratify(V)
        a = part.sample_from_region(0, 30, np.random.default_rng(5))
        b = part.sample_from_region(0, 30, np.random.default_rng(5))
        assert a == b

    def test_stratify_required_before_sampling(self):
        part, _ = self.make(k=3)
        with pytest.raises(RuntimeError):
            part.sample_from_region(0, 5, np.random.default_rng(0))
        with pytest.raises(RuntimeError):
            part.empirical_weights()

    def test_fit_rejects_more_regions_than_points(self):
        with pytest.raises(ValueError):
            RegionPartition(k=10, seed=0).fit(np.zeros((3, 4)))


class TestRegionCounters:
    def test_counters_are_independent(self):
        r = Region(region_id=0, centroid=np.zeros(3))
        r.record_search_sample(True)
        r.record_search_sample(False)
        r.record_estimation_sample(True)
        assert (r.n_search_samples, r.n_search_violations) == (2, 1)
        assert (r.n_estimation_samples, r.n_estimation_violations) == (1, 1)
        assert r.p_hat_search == 0.5
        assert r.p_hat_estimation == 1.0

    def test_resets_are_independent(self):
        r = Region(region_id=0, centroid=np.zeros(3))
        r.record_search_sample(True)
        r.record_estimation_sample(True)
        r.reset_search_stats()
        assert r.n_search_samples == 0
        assert r.n_estimation_samples == 1
        r.reset_estimation_stats()
        assert r.n_estimation_samples == 0

    def test_p_hat_of_an_unsampled_region_is_zero_not_an_error(self):
        r = Region(region_id=0, centroid=np.zeros(3))
        assert r.p_hat_estimation == 0.0
        assert r.p_hat_search == 0.0


# ======================================================================
class TestGraphDensity:
    """The MKG must never ship as the complete graph."""

    @staticmethod
    def spread_partition(k=12, dim=8, seed=0):
        rng = np.random.default_rng(seed)
        return RegionPartition(k=k, seed=seed).set_centroids(
            rng.standard_normal((k, dim)) * 3.0
        )

    def test_tau_is_calibrated_by_default(self):
        part = self.spread_partition()
        mkg = ModelKnowledgeGraph(part, tau=None, target_density=0.2)
        assert mkg.tau_was_calibrated
        assert 0.01 < mkg.get_edge_density() < 0.5

    def test_complete_graph_raises(self):
        """The exact configuration that shipped: tau above every distance."""
        part = self.spread_partition(k=12)
        with pytest.raises(DegenerateGraphError, match="exceeds max_density"):
            ModelKnowledgeGraph(part, tau=1e9, strict=True)

    def test_edgeless_graph_raises(self):
        part = self.spread_partition(k=12)
        with pytest.raises(DegenerateGraphError, match="below min_density"):
            ModelKnowledgeGraph(part, tau=0.0, strict=True)

    def test_non_strict_mode_records_the_flag_instead_of_raising(self):
        part = self.spread_partition(k=12)
        with pytest.warns(RuntimeWarning):
            mkg = ModelKnowledgeGraph(part, tau=1e9, strict=False)
        assert mkg.density_flag is not None
        assert "exceeds max_density" in mkg.density_flag

    def test_calibration_hits_the_target_density_approximately(self):
        part = self.spread_partition(k=20, dim=10, seed=3)
        for target in (0.1, 0.2, 0.35):
            mkg = ModelKnowledgeGraph(part, tau=None, target_density=target)
            assert abs(mkg.get_edge_density() - target) < 0.08

    def test_neighborhood_hops_actually_changes_the_focus_set(self):
        """The old complete graph made every r-hop neighbourhood equal to V.

        On a calibrated sparse graph the focus set must strictly grow with r
        for at least one seed region, otherwise 'graph-guided' is a no-op.
        """
        part = self.spread_partition(k=20, dim=10, seed=1)
        mkg = ModelKnowledgeGraph(part, tau=None, target_density=0.12)
        grew = False
        for j in mkg.graph.nodes:
            sizes = [len(mkg.get_focus_regions({j}, h)) for h in (1, 2, 3)]
            assert sizes == sorted(sizes)
            if sizes[0] < sizes[-1]:
                grew = True
        assert grew, "focus set was invariant to neighborhood_hops"
        assert len(mkg.get_focus_regions(set(mkg.graph.nodes), 1)) <= mkg.graph.number_of_nodes()

    def test_frontier_excludes_explored_regions(self):
        part = self.spread_partition(k=15, dim=8, seed=2)
        mkg = ModelKnowledgeGraph(part, tau=None, target_density=0.2)
        explored = {0, 1}
        frontier = mkg.get_frontier_regions(explored)
        assert not (frontier & explored)

    def test_mutation_edges_are_added_and_labelled(self):
        part = self.spread_partition(k=10, dim=6, seed=4)
        mkg = ModelKnowledgeGraph(part, tau=None, target_density=0.15)
        u, v = next(
            (a, b)
            for a in range(10)
            for b in range(a + 1, 10)
            if not mkg.graph.has_edge(a, b)
        )
        mkg.add_mutation_edge(u, v)
        assert mkg.graph[u][v]["edge_type"] == "mutation"
        mkg.add_mutation_edge(u, u)  # self-loop ignored
        assert not mkg.graph.has_edge(u, u)

    def test_summary_reports_the_density_evidence(self):
        part = self.spread_partition(k=12, dim=8, seed=5)
        mkg = ModelKnowledgeGraph(part, tau=None, target_density=0.2)
        s = mkg.summary()
        assert s["n_edges"] < s["max_possible_edges"]
        assert s["tau_was_calibrated"] is True
        assert s["density_flag"] is None


class TestGraphDensityOnRealData:
    """Density must be sane on real prompts, not just synthetic Gaussians."""

    @staticmethod
    def load_real_prompts(n=300):
        try:
            import datasets

            ds = datasets.load_dataset(
                "PKU-Alignment/PKU-SafeRLHF", split=f"train[:{n}]"
            )
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"PKU-SafeRLHF unavailable: {type(exc).__name__}: {exc}")
        return [{"prompt": r["prompt"]} for r in ds]

    def test_density_is_strictly_between_bounds_on_pku_saferlhf(self):
        prompts = self.load_real_prompts(300)
        emb = HashedBagOfWordsEmbedder(embed_dim=64, seed=0)
        V = emb.embed_batch(prompts)
        part = RegionPartition(k=12, seed=0).fit(V)
        mkg = ModelKnowledgeGraph(part, tau=None, target_density=0.2)
        d = mkg.get_edge_density()
        assert 0.0 < d < 1.0, d
        assert mkg.min_density <= d <= mkg.max_density, d
        assert mkg.graph.number_of_edges() < 12 * 11 // 2
        assert mkg.density_flag is None


# ======================================================================
class TestRegressionSubgraph:
    """Rank by the change in the POINT ESTIMATE, not by the change in the UCB."""

    @staticmethod
    def stat(rid, n, v, w=0.25):
        return RegionStat(region_id=rid, weight=w, n_samples=n, n_violations=v)

    @staticmethod
    def mkg2(k=4):
        # The regression maths does not read the graph at all, so the density
        # guard is relaxed here rather than distorting the fixture's centroids.
        part = RegionPartition(k=k, seed=0).set_centroids(
            np.array([[float(i), 0.0] for i in range(k)])
        )
        return ModelKnowledgeGraph(
            part, tau=1.5, strict=False, min_density=0.0, max_density=1.0
        )

    # The exact numbers from the audited failure: K = 40 regions, delta = 0.05,
    # a baseline that sampled the region 3 times, and a candidate that violates
    # every time.  Under the UCB-difference rule ucb_old saturates at 1.0, so
    # delta_j = ucb_new - ucb_old <= 0 and the region cannot be flagged at any
    # eta > 0 no matter how badly it regressed.
    F11_K = 40
    F11_BASELINE_N = 3

    @staticmethod
    def old_ucb(n, v, k=40, delta=0.05):
        """The rule that shipped: p_hat + Hoeffding width, clipped to 1."""
        import math

        if n == 0:
            return 1.0
        p = v / n
        return min(1.0, p + math.sqrt(math.log(2 * k / delta) / (2 * n)))

    def test_old_ucb_difference_rule_could_not_flag_a_thin_baseline(self):
        """Establishes the defect, using the audited configuration."""
        ucb_old = self.old_ucb(self.F11_BASELINE_N, 0, k=self.F11_K)
        ucb_new = self.old_ucb(30, 30, k=self.F11_K)
        assert ucb_old == 1.0, ucb_old
        assert ucb_new - ucb_old <= 0.0, "old rule was blind to 0.000 -> 1.000"

    def test_zero_to_one_region_is_flagged(self):
        """Same numbers, new rule: the point estimate moved by +1.0, so flag it."""
        mkg = self.mkg2()
        prev = [self.stat(0, self.F11_BASELINE_N, 0), self.stat(1, 50, 5)]
        curr = [self.stat(0, 30, 30), self.stat(1, 50, 5)]
        report = mkg.compute_regression_subgraph(
            prev, curr, eta=0.05, min_samples=self.F11_BASELINE_N
        )
        assert 0 in report.flagged
        assert report.deltas[0] == pytest.approx(1.0)
        assert 1 not in report.flagged
        assert report.insufficient_evidence == {}

    def test_zero_to_one_region_is_flagged_with_ample_samples(self):
        mkg = self.mkg2()
        prev = [self.stat(0, 40, 0), self.stat(1, 50, 5)]
        curr = [self.stat(0, 40, 40), self.stat(1, 50, 5)]
        report = mkg.compute_regression_subgraph(prev, curr, eta=0.05, min_samples=5)
        assert report.flagged == [0]
        assert report.deltas[0] == pytest.approx(1.0)

    def test_three_zero_to_one_regions_are_all_flagged(self):
        """The audit found three such regions in one run; all three must surface."""
        mkg = self.mkg2(k=4)
        prev = [self.stat(j, 10, 0) for j in range(3)] + [self.stat(3, 60, 6)]
        curr = [self.stat(j, 10, 10) for j in range(3)] + [self.stat(3, 60, 6)]
        report = mkg.compute_regression_subgraph(prev, curr, eta=0.05, min_samples=5)
        assert set(report.flagged) == {0, 1, 2}
        assert 3 not in report.flagged

    def test_thin_evidence_is_reported_not_silently_dropped(self):
        mkg = self.mkg2()
        prev = [self.stat(0, 2, 0), self.stat(1, 40, 2)]
        curr = [self.stat(0, 30, 30), self.stat(1, 3, 3)]
        report = mkg.compute_regression_subgraph(prev, curr, eta=0.05, min_samples=5)
        assert report.flagged == []
        assert set(report.insufficient_evidence) == {0, 1}
        assert "n_baseline=2" in report.insufficient_evidence[0]
        assert "n_candidate=3" in report.insufficient_evidence[1]

    def test_improvement_is_not_flagged(self):
        mkg = self.mkg2()
        prev = [self.stat(0, 50, 40)]
        curr = [self.stat(0, 50, 2)]
        report = mkg.compute_regression_subgraph(prev, curr, eta=0.05, min_samples=5)
        assert report.flagged == []
        assert report.deltas[0] < 0
        assert report.weighted_deltas[0] == 0.0

    def test_flagged_are_ranked_by_weighted_severity(self):
        mkg = self.mkg2(k=3)
        prev = [self.stat(j, 40, 0, w=w) for j, w in enumerate([0.1, 0.6, 0.3])]
        curr = [
            RegionStat(region_id=0, weight=0.1, n_samples=40, n_violations=40),
            RegionStat(region_id=1, weight=0.6, n_samples=40, n_violations=20),
            RegionStat(region_id=2, weight=0.3, n_samples=40, n_violations=8),
        ]
        report = mkg.compute_regression_subgraph(prev, curr, eta=0.05, min_samples=5)
        assert report.flagged == [1, 0, 2]  # 0.30, 0.10, 0.06

    def test_report_behaves_like_the_old_list_return(self):
        mkg = self.mkg2()
        prev = [self.stat(0, 40, 0)]
        curr = [self.stat(0, 40, 40)]
        report = mkg.compute_regression_subgraph(prev, curr, eta=0.05, min_samples=5)
        assert isinstance(report, RegressionReport)
        assert 0 in report
        assert len(report) == 1
        assert list(report) == [0]

    def test_minimal_explanation_set(self):
        mkg = self.mkg2(k=3)
        prev = [self.stat(j, 60, 2, w=w) for j, w in enumerate([0.4, 0.3, 0.3])]
        curr = [
            RegionStat(region_id=0, weight=0.4, n_samples=60, n_violations=42),
            RegionStat(region_id=1, weight=0.3, n_samples=60, n_violations=36),
            RegionStat(region_id=2, weight=0.3, n_samples=60, n_violations=2),
        ]
        s = mkg.minimal_explanation_set(prev, curr, gamma=0.2, min_samples=5)
        assert 0 in s
        assert 2 not in s


# ======================================================================
class TestEmbedders:
    def test_hashed_bow_shape_and_determinism(self):
        e = HashedBagOfWordsEmbedder(embed_dim=16, seed=1)
        v1 = e.embed({"prompt": "hello world"})
        v2 = e.embed({"prompt": "hello world"})
        assert v1.shape == (16,)
        np.testing.assert_array_equal(v1, v2)

    def test_hashed_bow_distinguishes_different_text(self):
        e = HashedBagOfWordsEmbedder(embed_dim=16, seed=1)
        assert not np.allclose(
            e.embed({"prompt": "alpha beta"}), e.embed({"prompt": "gamma delta"})
        )

    def test_empty_interaction_does_not_crash(self):
        e = HashedBagOfWordsEmbedder(embed_dim=8, seed=1)
        assert e.embed({}).shape == (8,)

    def test_batch_matches_singles(self):
        e = HashedBagOfWordsEmbedder(embed_dim=8, seed=1)
        xs = [{"prompt": f"p {i}"} for i in range(5)]
        np.testing.assert_allclose(e.embed_batch(xs), np.stack([e.embed(x) for x in xs]))

    def test_precomputed_raises_on_miss_rather_than_guessing(self):
        e = PrecomputedEmbedder({"a": np.ones(3)}, dim=3)
        np.testing.assert_array_equal(e.embed({"id": "a"}), np.ones(3))
        with pytest.raises(KeyError):
            e.embed({"id": "missing"})

    def test_make_embedder_factory(self):
        assert isinstance(make_embedder("hashed", embed_dim=8), HashedBagOfWordsEmbedder)
        with pytest.raises(ValueError):
            make_embedder("nonsense")

    def test_descriptor_is_recorded(self):
        d = HashedBagOfWordsEmbedder(embed_dim=8, seed=3).descriptor
        assert d["type"] == "HashedBagOfWordsEmbedder"
        assert d["seed"] == 3
