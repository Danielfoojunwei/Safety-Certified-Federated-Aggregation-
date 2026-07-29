"""Interaction regions R_j, the estimation/search counter separation, and the
stratified i.i.d. sampler that Stage B of the verification protocol draws from.

The partition X = bigsqcup_{j=1..K} R_j is induced by k-means in the embedding
space. K is declared A PRIORI and is *never* grown at test time: the anytime
Hoeffding bound (Theorem 1) unions over K * M fixed (region, sample-count)
pairs, and a partition that grows in response to the data invalidates that
union.  ``RegionPartition.assign`` therefore always returns an existing region
id.  The previous version of this file created new regions on the fly whenever
an embedding fell farther than ``tau_new`` from every centroid; that behaviour
is deliberately removed and the parameter is gone.

THE CENTRAL INVARIANT OF THIS MODULE
------------------------------------
Every region carries two disjoint pairs of counters:

    n_search_samples      / n_search_violations       (Stage A: adaptive,
                                                       mutation-generated,
                                                       NOT i.i.d.)
    n_estimation_samples  / n_estimation_violations   (Stage B: fresh i.i.d.
                                                       draws from D|R_j)

Only the estimation counters may ever reach the certificate.  The search
counters exist so that discovery metrics ("we found N distinct failures") can
be reported without contaminating the bound.  This separation is the fix for
the false-theorem defect: the old code recursed only on already-violating
nodes and then fed every mutant into the same counter that became p_hat_j, so
the samples backing the Hoeffding bound were neither independent nor
identically distributed.

The separation is enforced structurally, not by convention.  Stage A code is
handed a :class:`SearchRecorder` and Stage B code is handed an
:class:`EstimationRecorder`; neither recorder exposes the other's method, and
the stage functions never receive the partition itself.  ``tests/test_verifier.py``
additionally monkeypatches ``Region.record_estimation_sample`` to raise and
runs a complete search stage to prove no path exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class Region:
    """A region R_j of the interaction-space partition.

    Attributes:
        region_id: Index j.
        centroid: Cluster centroid c_j in embedding space.
        weight: w_j = Pr[x in R_j], estimated from the ESTIMATION pool only
            (see :meth:`RegionPartition.empirical_weights`).
        n_search_samples: Stage-A queries charged to this region.
        n_search_violations: Stage-A violations.  Discovery metric only.
        n_estimation_samples: m_j, the number of fresh i.i.d. Stage-B draws.
        n_estimation_violations: Stage-B violations.  This is the ONLY
            violation count the certificate is allowed to read.
    """

    region_id: int
    centroid: np.ndarray
    weight: float = 0.0

    n_search_samples: int = 0
    n_search_violations: int = 0
    n_estimation_samples: int = 0
    n_estimation_violations: int = 0

    search_member_ids: list[str] = field(default_factory=list)
    estimation_member_ids: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Recording.  Two methods, two counters, no shared path.
    # ------------------------------------------------------------------
    def record_estimation_sample(
        self, is_violation: bool, sample_id: str | None = None
    ) -> None:
        """Record ONE fresh i.i.d. draw from D|R_j.

        Call this only from Stage B, only for samples drawn by
        :meth:`RegionPartition.sample_from_region` out of the estimation pool.
        Anything recorded here feeds the certificate.
        """
        self.n_estimation_samples += 1
        if is_violation:
            self.n_estimation_violations += 1
        if sample_id is not None:
            self.estimation_member_ids.append(sample_id)

    def record_search_sample(
        self, is_violation: bool, sample_id: str | None = None
    ) -> None:
        """Record ONE Stage-A probe (seed, mutant, or frontier probe).

        These draws are adaptive and mutation-generated.  They are NOT i.i.d.
        under D|R_j and they never touch the bound.
        """
        self.n_search_samples += 1
        if is_violation:
            self.n_search_violations += 1
        if sample_id is not None:
            self.search_member_ids.append(sample_id)

    # ------------------------------------------------------------------
    # Read-only views.
    # ------------------------------------------------------------------
    @property
    def p_hat_estimation(self) -> float:
        """Empirical violation rate from estimation samples (0.0 if m_j = 0)."""
        if self.n_estimation_samples == 0:
            return 0.0
        return self.n_estimation_violations / self.n_estimation_samples

    @property
    def p_hat_search(self) -> float:
        """Empirical violation rate among Stage-A probes.

        Biased upward by construction (the search deliberately hunts for
        failures).  Usable as a *suspicion score* for allocation; never as an
        estimate of p_j.
        """
        if self.n_search_samples == 0:
            return 0.0
        return self.n_search_violations / self.n_search_samples

    def reset_search_stats(self) -> None:
        self.n_search_samples = 0
        self.n_search_violations = 0
        self.search_member_ids.clear()

    def reset_estimation_stats(self) -> None:
        self.n_estimation_samples = 0
        self.n_estimation_violations = 0
        self.estimation_member_ids.clear()

    def reset_stats(self) -> None:
        self.reset_search_stats()
        self.reset_estimation_stats()


#: Backwards-compatible alias.  The old name lacked the counter split.
InteractionRegion = Region


class SearchRecorder:
    """Capability object handed to Stage A.  Can ONLY write search counters."""

    __slots__ = ("_partition", "n_recorded")

    def __init__(self, partition: "RegionPartition") -> None:
        self._partition = partition
        self.n_recorded = 0

    def record(
        self, region_id: int, is_violation: bool, sample_id: str | None = None
    ) -> None:
        self._partition.regions[region_id].record_search_sample(
            is_violation, sample_id
        )
        self.n_recorded += 1


class EstimationRecorder:
    """Capability object handed to Stage B.  Can ONLY write estimation counters."""

    __slots__ = ("_partition", "n_recorded")

    def __init__(self, partition: "RegionPartition") -> None:
        self._partition = partition
        self.n_recorded = 0

    def record(
        self, region_id: int, is_violation: bool, sample_id: str | None = None
    ) -> None:
        self._partition.regions[region_id].record_estimation_sample(
            is_violation, sample_id
        )
        self.n_recorded += 1


class RegionPartition:
    """Fixed-K partition of the interaction space plus a stratified sampler.

    Typical lifecycle::

        part = RegionPartition(k=12, seed=0)
        part.fit(fit_vectors)                    # declare the partition
        part.stratify(estimation_pool_vectors)   # freeze strata + weights
        idx = part.sample_from_region(j, m_j, rng)   # Stage B draws
    """

    def __init__(self, k: int, seed: int = 0) -> None:
        """
        Args:
            k: Number of regions K.  Declared a priori; never changes.
            seed: Seed for k-means initialisation.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k_declared = int(k)
        self.seed = int(seed)
        self.regions: list[Region] = []
        self._strata: dict[int, list[int]] = {}
        self._pool_size: int = 0
        self._fitted = False

    # ------------------------------------------------------------------
    @property
    def k(self) -> int:
        """Number of regions actually materialised (== ``k_declared`` once fit)."""
        return len(self.regions)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, vectors: np.ndarray) -> "RegionPartition":
        """Fit the partition by k-means on ``vectors``.

        Args:
            vectors: (N, p) embedding matrix.  N must be >= k.

        Returns:
            self, for chaining.
        """
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if len(vectors) < self.k_declared:
            raise ValueError(
                f"cannot fit K={self.k_declared} regions from {len(vectors)} "
                "vectors; declare a smaller K or supply more fit data"
            )

        km = KMeans(
            n_clusters=self.k_declared, n_init=10, random_state=self.seed
        )
        km.fit(vectors)

        self.regions = [
            Region(
                region_id=j,
                centroid=km.cluster_centers_[j].copy(),
                weight=0.0,
            )
            for j in range(self.k_declared)
        ]
        self._strata = {}
        self._pool_size = 0
        self._fitted = True
        return self

    def set_centroids(self, centroids: np.ndarray) -> "RegionPartition":
        """Install centroids directly (testing / reproducing a saved partition)."""
        centroids = np.asarray(centroids, dtype=float)
        if len(centroids) != self.k_declared:
            raise ValueError(
                f"expected {self.k_declared} centroids, got {len(centroids)}"
            )
        self.regions = [
            Region(region_id=j, centroid=centroids[j].copy(), weight=0.0)
            for j in range(self.k_declared)
        ]
        self._strata = {}
        self._pool_size = 0
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def assign(self, vector: np.ndarray) -> int:
        """Return j = argmin_l ||psi(x) - c_l||.

        Always an EXISTING region id.  The partition is never grown at test
        time: K is part of the a-priori union bound.
        """
        if not self._fitted:
            raise RuntimeError("RegionPartition.assign called before fit()")
        vector = np.asarray(vector, dtype=float).reshape(-1)
        centroids = np.stack([r.centroid for r in self.regions])
        dists = np.linalg.norm(centroids - vector, axis=1)
        return int(np.argmin(dists))

    def assign_batch(self, vectors: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`assign`.  Returns an int array of region ids."""
        if not self._fitted:
            raise RuntimeError("RegionPartition.assign_batch called before fit()")
        vectors = np.asarray(vectors, dtype=float)
        centroids = np.stack([r.centroid for r in self.regions])
        d2 = (
            (vectors ** 2).sum(1)[:, None]
            - 2.0 * vectors @ centroids.T
            + (centroids ** 2).sum(1)[None, :]
        )
        return np.argmin(d2, axis=1).astype(int)

    # ------------------------------------------------------------------
    def stratify(self, pool_vectors: np.ndarray) -> dict[int, list[int]]:
        """Bucket an estimation pool into per-region strata.

        This also FREEZES the region weights: w_j = |stratum_j| / |pool|.
        Call it exactly once, on the estimation pool, before verification
        starts.  Weights derived from the search pool or from the verifier's
        own adaptive draws would be data-dependent in the wrong way.

        Args:
            pool_vectors: (N, p) embeddings of the estimation pool, in pool order.

        Returns:
            ``{region_id: [pool indices]}`` for every region 0..K-1 (empty
            lists included, so callers see the zero-weight regions).
        """
        pool_vectors = np.asarray(pool_vectors, dtype=float)
        labels = self.assign_batch(pool_vectors)
        strata: dict[int, list[int]] = {j: [] for j in range(self.k)}
        for idx, j in enumerate(labels):
            strata[int(j)].append(int(idx))

        self._strata = strata
        self._pool_size = len(pool_vectors)
        n = max(1, self._pool_size)
        for region in self.regions:
            region.weight = len(strata[region.region_id]) / n
        return strata

    @property
    def strata(self) -> dict[int, list[int]]:
        return self._strata

    def sample_from_region(
        self, region_id: int, n: int, rng: np.random.Generator
    ) -> list[int]:
        """Draw ``n`` i.i.d. samples from D|R_j, WITH replacement.

        With-replacement sampling from the stratum is what makes the draws
        genuinely i.i.d. conditional on the region, which is exactly the
        hypothesis Theorem 1 needs.  Without replacement they would be
        exchangeable but not independent, and m_j would be capped by the
        stratum size.

        Returns:
            A list of ``n`` indices into the pool passed to :meth:`stratify`.
            Empty list if the stratum is empty (a zero-weight region).
        """
        if not self._strata:
            raise RuntimeError("sample_from_region called before stratify()")
        stratum = self._strata.get(int(region_id), [])
        if n <= 0 or not stratum:
            return []
        picks = rng.integers(0, len(stratum), size=int(n))
        return [stratum[int(i)] for i in picks]

    def empirical_weights(self) -> dict[int, float]:
        """w_j = |stratum_j| / |estimation pool|, for every region.

        Sums to 1 (up to float error) whenever :meth:`stratify` has been
        called.  Regions with an empty stratum get weight 0.0 and are still
        present in the dict -- a caller must never silently drop them.
        """
        if not self._strata:
            raise RuntimeError("empirical_weights called before stratify()")
        n = max(1, self._pool_size)
        return {
            j: len(self._strata.get(j, [])) / n for j in range(self.k)
        }

    def get_weights(self) -> np.ndarray:
        """Weights as a (K,) array in region-id order."""
        return np.array([r.weight for r in self.regions], dtype=float)

    # ------------------------------------------------------------------
    def search_recorder(self) -> SearchRecorder:
        return SearchRecorder(self)

    def estimation_recorder(self) -> EstimationRecorder:
        return EstimationRecorder(self)

    def reset_all_stats(self) -> None:
        for region in self.regions:
            region.reset_stats()

    def reset_search_stats(self) -> None:
        for region in self.regions:
            region.reset_search_stats()

    def reset_estimation_stats(self) -> None:
        for region in self.regions:
            region.reset_estimation_stats()
