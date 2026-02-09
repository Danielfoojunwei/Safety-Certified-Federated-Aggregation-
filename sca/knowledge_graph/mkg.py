"""Model Knowledge Graph (MKG) -- formal coverage structure.

Section 3.2: G = (V, E) where V = {1, ..., K} are interaction regions
and edges encode adjacency or mutation lineage:
    (j, l) in E iff dist(c_j, c_l) <= tau  or  R_l derived from R_j.

The MKG serves as a coverage map and a frontier for recursive expansion
by the RLM verifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from sca.knowledge_graph.regions import InteractionRegion, RegionPartition
from sca.utils.stats import RegionStats


@dataclass
class MKGEdge:
    """Edge metadata in the MKG.

    Attributes:
        source: Source region index.
        target: Target region index.
        edge_type: Either "proximity" or "mutation" (lineage).
        distance: Embedding-space distance between centroids.
    """
    source: int
    target: int
    edge_type: str  # "proximity" | "mutation"
    distance: float = 0.0


class ModelKnowledgeGraph:
    """Model Knowledge Graph (MKG) implementation.

    Maintains a graph over interaction regions with:
    - Proximity edges (centroid distance <= tau).
    - Mutation edges (R_l derived from R_j via verifier recursion).
    - Per-region violation statistics.
    - Frontier tracking for adaptive sampling.

    The graph is used for:
    1. Coverage tracking (Section 3.2).
    2. Graph-guided adaptive sampling (Section 5.2, Proposition 1).
    3. Regression subgraph identification (Section 5.3).
    """

    def __init__(
        self,
        partition: RegionPartition,
        tau: float = 1.5,
    ) -> None:
        """
        Args:
            partition: The interaction region partition.
            tau: Distance threshold for proximity edges.
        """
        self.partition = partition
        self.tau = tau
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the MKG from current regions."""
        self.graph.clear()

        # Add nodes
        for region in self.partition.regions:
            self.graph.add_node(
                region.region_id,
                centroid=region.centroid,
                weight=region.weight,
            )

        # Add proximity edges
        regions = self.partition.regions
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                dist = float(np.linalg.norm(
                    regions[i].centroid - regions[j].centroid
                ))
                if dist <= self.tau:
                    self.graph.add_edge(
                        regions[i].region_id,
                        regions[j].region_id,
                        edge_type="proximity",
                        distance=dist,
                    )

    def add_mutation_edge(self, source_id: int, target_id: int) -> None:
        """Record a mutation lineage edge (R_target derived from R_source).

        Args:
            source_id: Source region index.
            target_id: Target region index.
        """
        if not self.graph.has_node(source_id):
            self.graph.add_node(source_id)
        if not self.graph.has_node(target_id):
            self.graph.add_node(target_id)

        dist = 0.0
        src_region = self._get_region(source_id)
        tgt_region = self._get_region(target_id)
        if src_region is not None and tgt_region is not None:
            dist = float(np.linalg.norm(
                src_region.centroid - tgt_region.centroid
            ))

        self.graph.add_edge(
            source_id, target_id,
            edge_type="mutation",
            distance=dist,
        )

    def _get_region(self, region_id: int) -> InteractionRegion | None:
        """Lookup a region by ID."""
        for r in self.partition.regions:
            if r.region_id == region_id:
                return r
        return None

    def get_neighborhood(self, region_id: int, hops: int = 1) -> set[int]:
        """Return the r-hop neighborhood N_r of a region in the MKG.

        Used for graph-guided adaptive sampling (Section 5.2):
        after the first hit in a bad region, sample within the
        r-hop neighborhood to concentrate testing on the failure cluster.

        Args:
            region_id: Center region index.
            hops: Radius of the neighborhood.

        Returns:
            Set of region IDs within r hops.
        """
        if region_id not in self.graph:
            return {region_id}

        visited = {region_id}
        frontier = {region_id}
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        return visited

    def get_frontier_regions(self, explored: set[int]) -> set[int]:
        """Return unexplored regions adjacent to explored ones.

        The frontier guides where the RLM verifier should probe next.

        Args:
            explored: Set of already-explored region IDs.

        Returns:
            Set of unexplored region IDs adjacent to explored regions.
        """
        frontier = set()
        for node in explored:
            if node in self.graph:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in explored:
                        frontier.add(neighbor)
        return frontier

    def get_region_stats(self, delta: float) -> list[RegionStats]:
        """Collect per-region statistics for the acceptance rule.

        Args:
            delta: Confidence parameter.

        Returns:
            List of RegionStats for all regions.
        """
        stats = []
        for region in self.partition.regions:
            stats.append(RegionStats(
                region_id=region.region_id,
                n_samples=region.n_samples,
                n_violations=region.n_violations,
                weight=region.weight,
            ))
        return stats

    def compute_regression_subgraph(
        self,
        prev_stats: list[RegionStats],
        curr_stats: list[RegionStats],
        delta: float,
        eta: float = 0.01,
    ) -> list[int]:
        """Identify the regression subgraph Delta_G (Section 5.3).

        Delta_G = {j : Delta_j >= eta} where
        Delta_j = UCB_j(M_new) - UCB_j(M_old).

        Args:
            prev_stats: Region stats from the previous model.
            curr_stats: Region stats from the candidate model.
            delta: Confidence parameter.
            eta: Threshold for regression detection.

        Returns:
            List of region IDs in the regression subgraph.
        """
        k = max(len(prev_stats), len(curr_stats))
        if k == 0:
            return []

        prev_map = {rs.region_id: rs for rs in prev_stats}
        regression_regions = []

        for rs_new in curr_stats:
            rs_old = prev_map.get(rs_new.region_id)
            ucb_new = rs_new.ucb(delta, k)
            ucb_old = rs_old.ucb(delta, k) if rs_old else 0.0
            delta_j = ucb_new - ucb_old
            if delta_j >= eta:
                regression_regions.append(rs_new.region_id)

        return regression_regions

    def minimal_explanation_set(
        self,
        prev_stats: list[RegionStats],
        curr_stats: list[RegionStats],
        delta: float,
        gamma: float = 0.05,
    ) -> list[int]:
        """Compute a minimal regression explanation set (Section 5.3).

        Solve: min |S| s.t. sum_{j in S} w_j * Delta_j >= gamma.
        Greedy solution (nonneg weights make greedy optimal for coverage).

        Args:
            prev_stats: Region stats from the previous model.
            curr_stats: Region stats from the candidate model.
            delta: Confidence parameter.
            gamma: Minimum weighted delta to explain.

        Returns:
            Sorted list of region IDs forming the minimal explanation set.
        """
        k = max(len(prev_stats), len(curr_stats))
        if k == 0:
            return []

        prev_map = {rs.region_id: rs for rs in prev_stats}

        # Compute weighted deltas
        deltas = []
        for rs_new in curr_stats:
            rs_old = prev_map.get(rs_new.region_id)
            ucb_new = rs_new.ucb(delta, k)
            ucb_old = rs_old.ucb(delta, k) if rs_old else 0.0
            delta_j = ucb_new - ucb_old
            if delta_j > 0:
                deltas.append((rs_new.region_id, rs_new.weight * delta_j))

        # Greedy: sort by weighted delta descending
        deltas.sort(key=lambda x: -x[1])

        result = []
        total = 0.0
        for region_id, w_delta in deltas:
            result.append(region_id)
            total += w_delta
            if total >= gamma:
                break

        return sorted(result)

    def refresh_edges(self) -> None:
        """Rebuild proximity edges (call after adding new regions)."""
        self._build_graph()

    def summary(self) -> dict:
        """Return a summary of the MKG state."""
        return {
            "n_regions": self.partition.k,
            "n_edges": self.graph.number_of_edges(),
            "n_proximity_edges": sum(
                1 for _, _, d in self.graph.edges(data=True)
                if d.get("edge_type") == "proximity"
            ),
            "n_mutation_edges": sum(
                1 for _, _, d in self.graph.edges(data=True)
                if d.get("edge_type") == "mutation"
            ),
        }
