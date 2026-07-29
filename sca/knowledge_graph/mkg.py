"""Model Knowledge Graph (MKG): a sparse adjacency structure over regions.

G = (V, E) with V = {0..K-1} the interaction regions and

    (j, l) in E   iff   ||c_j - c_l|| <= tau   or   R_l was reached from R_j
                                                    by a mutation during search.

WHY tau IS NOW CALIBRATED AND GUARDED
-------------------------------------
The previous version shipped with ``tau=1.0`` hardcoded in every experiment
driver.  On the embeddings those drivers produced, every pairwise centroid
distance fell below 1.0, so the "graph" was the complete graph K_40: 780 edges
on 40 nodes.  In a complete graph the r-hop neighbourhood of every node is the
whole vertex set for every r >= 1, the frontier equals the unexplored set, and
"graph-guided exploration" degenerates to "explore everything".  The
budget-matched MKG-guided and blind arms were then bit-identical, and the
reported gap came entirely from an unrelated constant.

Two changes prevent that from recurring:

1. ``tau=None`` (the default) triggers :meth:`auto_calibrate_tau`, which sets
   tau to the ``target_density`` quantile of the pairwise centroid distances.
2. :meth:`check_density` raises ``DegenerateGraphError`` when the realised edge
   density leaves ``[min_density, max_density]``.  A near-complete graph and a
   fully disconnected graph are both structurally meaningless, and the failure
   must be loud.  Callers that would rather record the problem than crash pass
   ``strict=False``; the flag then lands in :attr:`density_flag` and must be
   propagated into the results file.

REGRESSION SUBGRAPH
-------------------
:meth:`compute_regression_subgraph` ranks by the change in the POINT ESTIMATE
p_hat, not by the change in the UCB.  Ranking by UCB difference -- what the old
code did -- is dominated by the difference of Hoeffding widths, which is large
and negative for any region the baseline barely sampled.  Concretely, a region
the baseline sampled 3 times has ucb_old = 1.0, so delta_j = ucb_new - 1.0 <= 0
and the region cannot be flagged no matter how badly it regressed.  Three
regions that went from a 0.000 to a 1.000 violation rate were invisible to that
rule.  Regions with too little evidence on either side are now reported as
"insufficient evidence" rather than silently scoring <= 0.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from sca.knowledge_graph.regions import Region, RegionPartition
from sca.utils.stats import RegionStat

logger = logging.getLogger(__name__)


class DegenerateGraphError(RuntimeError):
    """Raised when the MKG is complete (or empty) and therefore uninformative."""


@dataclass
class MKGEdge:
    """Edge metadata in the MKG."""

    source: int
    target: int
    edge_type: str  # "proximity" | "mutation"
    distance: float = 0.0


@dataclass(frozen=True)
class RegressionReport:
    """Result of a regression-subgraph computation (Section 5.3).

    Attributes:
        flagged: Region ids whose point-estimate violation rate rose by at
            least ``eta`` and that had enough samples on BOTH sides.
        deltas: ``{region_id: p_hat_new - p_hat_old}`` for eligible regions.
        weighted_deltas: ``{region_id: w_j * max(0, delta_j)}``.
        insufficient_evidence: ``{region_id: reason}``.  These regions were
            NOT tested.  They are neither "clean" nor "regressed"; the run
            simply cannot say.  Report this dict, do not drop it.
        min_samples: The per-side sample floor that was applied.
        eta: The flagging threshold.
    """

    flagged: list[int]
    deltas: dict[int, float]
    weighted_deltas: dict[int, float]
    insufficient_evidence: dict[int, str]
    min_samples: int
    eta: float

    def __contains__(self, region_id: object) -> bool:
        return region_id in self.flagged

    def __iter__(self):
        return iter(self.flagged)

    def __len__(self) -> int:
        return len(self.flagged)


class ModelKnowledgeGraph:
    """Sparse region-adjacency graph used to steer Stage-A search only.

    The MKG never touches the certificate.  It decides where the search stage
    spends its probes; the bound is computed from Stage-B i.i.d. draws whose
    allocation the MKG may influence but whose validity it cannot affect
    (Theorem 1 holds for ANY data-dependent m_j in [1, M]).
    """

    def __init__(
        self,
        partition: RegionPartition,
        tau: float | None = None,
        target_density: float = 0.15,
        min_density: float = 0.01,
        max_density: float = 0.50,
        strict: bool = True,
    ) -> None:
        """
        Args:
            partition: A fitted :class:`RegionPartition`.
            tau: Proximity threshold.  ``None`` (recommended) calibrates it to
                ``target_density``.  An explicit value is honoured but still
                density-checked.
            target_density: Fraction of node pairs to connect when calibrating.
            min_density / max_density: Acceptable band for the realised density.
            strict: Raise :class:`DegenerateGraphError` outside the band.  When
                False, warn and set :attr:`density_flag` instead.
        """
        self.partition = partition
        self.target_density = float(target_density)
        self.min_density = float(min_density)
        self.max_density = float(max_density)
        self.strict = bool(strict)
        self.density_flag: str | None = None
        self.tau_was_calibrated = tau is None
        self.graph = nx.Graph()

        if tau is None:
            self.tau = float("nan")
            self.auto_calibrate_tau(self.target_density)
        else:
            self.tau = float(tau)
            self._build_graph()
        self.check_density()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _pairwise_centroid_distances(self) -> np.ndarray:
        regions = self.partition.regions
        if len(regions) < 2:
            return np.zeros(0)
        c = np.stack([r.centroid for r in regions])
        d2 = (
            (c ** 2).sum(1)[:, None]
            - 2.0 * c @ c.T
            + (c ** 2).sum(1)[None, :]
        )
        np.maximum(d2, 0.0, out=d2)
        d = np.sqrt(d2)
        iu = np.triu_indices(len(regions), k=1)
        return d[iu]

    def _build_graph(self) -> None:
        self.graph.clear()
        regions = self.partition.regions
        for region in regions:
            self.graph.add_node(
                region.region_id,
                centroid=region.centroid,
                weight=region.weight,
            )
        n = len(regions)
        if n < 2:
            return
        c = np.stack([r.centroid for r in regions])
        d2 = (
            (c ** 2).sum(1)[:, None] - 2.0 * c @ c.T + (c ** 2).sum(1)[None, :]
        )
        np.maximum(d2, 0.0, out=d2)
        d = np.sqrt(d2)
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] <= self.tau:
                    self.graph.add_edge(
                        regions[i].region_id,
                        regions[j].region_id,
                        edge_type="proximity",
                        distance=float(d[i, j]),
                    )

    def auto_calibrate_tau(self, target_density: float | None = None) -> float:
        """Set tau to the ``target_density`` quantile of pairwise distances.

        This is the call the old code defined and then never made, which is how
        the complete graph shipped.  It is now invoked from ``__init__``
        whenever ``tau`` is not given explicitly.
        """
        if target_density is not None:
            self.target_density = float(target_density)
        dists = self._pairwise_centroid_distances()
        if dists.size == 0:
            self.tau = 0.0
            self._build_graph()
            return self.tau
        # Nudge just below the quantile so ties at the quantile do not blow the
        # density past the target on degenerate (many-equal-distance) inputs.
        q = float(np.quantile(dists, self.target_density))
        self.tau = q
        self._build_graph()
        if self.get_edge_density() > self.max_density:
            # Ties pushed us over: fall back to the largest distance that keeps
            # density within budget.
            ordered = np.sort(dists)
            budget = int(self.max_density * len(ordered))
            budget = max(1, min(budget, len(ordered)))
            self.tau = float(np.nextafter(ordered[budget - 1], -np.inf))
            self._build_graph()
        return self.tau

    def check_density(self) -> float:
        """Validate the realised edge density; raise or flag if degenerate."""
        density = self.get_edge_density()
        n = self.graph.number_of_nodes()
        if n < 3:
            return density
        problem = None
        if density > self.max_density:
            problem = (
                f"MKG edge density {density:.3f} exceeds max_density "
                f"{self.max_density:.3f} on K={n} regions "
                f"({self.graph.number_of_edges()} of {n * (n - 1) // 2} pairs). "
                "A near-complete graph makes every r-hop neighbourhood the whole "
                "vertex set, so graph-guided search is indistinguishable from "
                "blind search."
            )
        elif density < self.min_density:
            problem = (
                f"MKG edge density {density:.3f} is below min_density "
                f"{self.min_density:.3f} on K={n} regions. An edgeless graph "
                "has an empty frontier, so graph-guided search never moves."
            )
        if problem is not None:
            self.density_flag = problem
            if self.strict:
                raise DegenerateGraphError(problem)
            warnings.warn(problem, RuntimeWarning, stacklevel=2)
        return density

    def get_edge_density(self) -> float:
        n = self.graph.number_of_nodes()
        if n < 2:
            return 0.0
        return self.graph.number_of_edges() / (n * (n - 1) / 2)

    def add_mutation_edge(self, source_id: int, target_id: int) -> None:
        """Record that a mutation carried a probe from R_source into R_target."""
        if source_id == target_id:
            return
        for rid in (source_id, target_id):
            if not self.graph.has_node(rid):
                self.graph.add_node(rid)
        dist = 0.0
        src, tgt = self._get_region(source_id), self._get_region(target_id)
        if src is not None and tgt is not None:
            dist = float(np.linalg.norm(src.centroid - tgt.centroid))
        if self.graph.has_edge(source_id, target_id):
            return  # keep the proximity label; the edge already exists
        self.graph.add_edge(
            source_id, target_id, edge_type="mutation", distance=dist
        )

    def _get_region(self, region_id: int) -> Region | None:
        for r in self.partition.regions:
            if r.region_id == region_id:
                return r
        return None

    # ------------------------------------------------------------------
    # Search guidance
    # ------------------------------------------------------------------
    def get_neighborhood(self, region_id: int, hops: int = 1) -> set[int]:
        """r-hop neighbourhood N_r(j), inclusive of j."""
        if region_id not in self.graph:
            return {region_id}
        visited = {region_id}
        frontier = {region_id}
        for _ in range(max(0, hops)):
            nxt = set()
            for node in frontier:
                for nb in self.graph.neighbors(node):
                    if nb not in visited:
                        visited.add(nb)
                        nxt.add(nb)
            frontier = nxt
            if not frontier:
                break
        return visited

    def get_frontier_regions(self, explored: set[int]) -> set[int]:
        """Unexplored regions adjacent to an explored one."""
        frontier = set()
        for node in explored:
            if node in self.graph:
                for nb in self.graph.neighbors(node):
                    if nb not in explored:
                        frontier.add(nb)
        return frontier

    def get_focus_regions(
        self, failure_regions: set[int], hops: int
    ) -> set[int]:
        """Union of the r-hop neighbourhoods of the failure regions.

        On a properly sparse graph this is a strict subset of V and it GROWS
        with ``hops``; on the old complete graph it was V for every hops >= 1,
        which is why the old ``neighborhood_hops`` parameter had no effect.
        """
        focus: set[int] = set()
        for j in failure_regions:
            focus |= self.get_neighborhood(j, hops)
        return focus

    # ------------------------------------------------------------------
    # Stats extraction
    # ------------------------------------------------------------------
    def get_estimation_stats(self) -> list[RegionStat]:
        """Per-region Stage-B statistics.  The ONLY input to the certificate."""
        return [
            RegionStat(
                region_id=r.region_id,
                weight=float(r.weight),
                n_samples=int(r.n_estimation_samples),
                n_violations=int(r.n_estimation_violations),
            )
            for r in self.partition.regions
        ]

    def get_search_stats(self) -> list[RegionStat]:
        """Per-region Stage-A statistics.  Discovery metrics; never certified."""
        return [
            RegionStat(
                region_id=r.region_id,
                weight=float(r.weight),
                n_samples=int(r.n_search_samples),
                n_violations=int(r.n_search_violations),
            )
            for r in self.partition.regions
        ]

    # ------------------------------------------------------------------
    # Regression analysis (Section 5.3)
    # ------------------------------------------------------------------
    def compute_regression_subgraph(
        self,
        prev_stats: list[RegionStat],
        curr_stats: list[RegionStat],
        eta: float = 0.05,
        min_samples: int = 5,
    ) -> RegressionReport:
        """Regions whose violation POINT ESTIMATE rose by at least ``eta``.

        Args:
            prev_stats: Baseline model's per-region estimation stats.
            curr_stats: Candidate model's per-region estimation stats.
            eta: Minimum increase in p_hat to flag.
            min_samples: Required sample count on BOTH sides.  A region below
                this on either side is reported under
                ``insufficient_evidence`` and is not flagged -- it is explicitly
                untested, which is different from tested-and-clean.

        Returns:
            A :class:`RegressionReport`.  It supports ``in``, ``iter`` and
            ``len`` over the flagged ids so it can stand in for the old
            ``list[int]`` return at the call sites that only needed membership.
        """
        prev_map = {rs.region_id: rs for rs in prev_stats}
        curr_map = {rs.region_id: rs for rs in curr_stats}

        flagged: list[int] = []
        deltas: dict[int, float] = {}
        weighted: dict[int, float] = {}
        insufficient: dict[int, str] = {}

        for rid in sorted(set(prev_map) | set(curr_map)):
            rs_old = prev_map.get(rid)
            rs_new = curr_map.get(rid)
            if rs_new is None:
                insufficient[rid] = "absent from candidate stats"
                continue
            if rs_old is None:
                insufficient[rid] = "absent from baseline stats"
                continue
            if rs_old.n_samples < min_samples and rs_new.n_samples < min_samples:
                insufficient[rid] = (
                    f"n_baseline={rs_old.n_samples} and "
                    f"n_candidate={rs_new.n_samples} both < min_samples={min_samples}"
                )
                continue
            if rs_old.n_samples < min_samples:
                insufficient[rid] = (
                    f"n_baseline={rs_old.n_samples} < min_samples={min_samples}"
                )
                continue
            if rs_new.n_samples < min_samples:
                insufficient[rid] = (
                    f"n_candidate={rs_new.n_samples} < min_samples={min_samples}"
                )
                continue

            p_old = rs_old.n_violations / rs_old.n_samples
            p_new = rs_new.n_violations / rs_new.n_samples
            delta_j = p_new - p_old
            deltas[rid] = delta_j
            weighted[rid] = rs_new.weight * max(0.0, delta_j)
            if delta_j >= eta:
                flagged.append(rid)

        # Rank by weighted severity, most severe first.
        flagged.sort(key=lambda r: (-weighted[r], -deltas[r], r))
        return RegressionReport(
            flagged=flagged,
            deltas=deltas,
            weighted_deltas=weighted,
            insufficient_evidence=insufficient,
            min_samples=int(min_samples),
            eta=float(eta),
        )

    def minimal_explanation_set(
        self,
        prev_stats: list[RegionStat],
        curr_stats: list[RegionStat],
        gamma: float = 0.05,
        min_samples: int = 5,
        eta: float = 0.0,
    ) -> list[int]:
        """Smallest set S with sum_{j in S} w_j * (p_hat_new - p_hat_old)_+ >= gamma.

        Greedy is optimal here: the objective is a sum of nonnegative constants
        over the chosen set, so taking the largest terms first is exact.

        Returns the ids sorted ascending.  Returns the greedily-chosen prefix
        even if the total never reaches ``gamma`` (the caller can compare the
        achieved mass against gamma via
        :meth:`compute_regression_subgraph`).
        """
        report = self.compute_regression_subgraph(
            prev_stats, curr_stats, eta=eta, min_samples=min_samples
        )
        ranked = sorted(
            (r for r in report.weighted_deltas if report.weighted_deltas[r] > 0),
            key=lambda r: -report.weighted_deltas[r],
        )
        chosen: list[int] = []
        total = 0.0
        for rid in ranked:
            chosen.append(rid)
            total += report.weighted_deltas[rid]
            if total >= gamma:
                break
        return sorted(chosen)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        n = self.graph.number_of_nodes()
        return {
            "n_regions": self.partition.k,
            "n_nodes": n,
            "n_edges": self.graph.number_of_edges(),
            "max_possible_edges": n * (n - 1) // 2,
            "edge_density": self.get_edge_density(),
            "tau": self.tau,
            "tau_was_calibrated": self.tau_was_calibrated,
            "target_density": self.target_density,
            "density_flag": self.density_flag,
            "n_proximity_edges": sum(
                1
                for _, _, d in self.graph.edges(data=True)
                if d.get("edge_type") == "proximity"
            ),
            "n_mutation_edges": sum(
                1
                for _, _, d in self.graph.edges(data=True)
                if d.get("edge_type") == "mutation"
            ),
        }
