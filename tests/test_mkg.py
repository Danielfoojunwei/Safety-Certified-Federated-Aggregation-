"""Tests for Model Knowledge Graph (MKG)."""

import numpy as np
import pytest

from sca.knowledge_graph.mkg import ModelKnowledgeGraph
from sca.knowledge_graph.regions import InteractionRegion, RegionPartition
from sca.utils.stats import RegionStats


def make_partition(n_regions=5, dim=4, spacing=1.0):
    """Helper to create a test partition with evenly spaced centroids."""
    partition = RegionPartition(tau_new=2.0)
    for j in range(n_regions):
        centroid = np.zeros(dim)
        centroid[0] = j * spacing
        partition.regions.append(InteractionRegion(
            region_id=j,
            centroid=centroid,
            weight=1.0 / n_regions,
        ))
    return partition


class TestRegionPartition:
    def test_assign_nearest(self):
        partition = make_partition(n_regions=3, spacing=2.0)
        # Point near region 1
        embedding = np.array([2.1, 0.0, 0.0, 0.0])
        j = partition.assign(embedding)
        assert j == 1

    def test_create_new_region(self):
        partition = make_partition(n_regions=2, spacing=1.0)
        partition.tau_new = 1.5
        # Far from both existing centroids
        embedding = np.array([10.0, 0.0, 0.0, 0.0])
        j = partition.assign(embedding)
        assert j == 2  # New region created
        assert partition.k == 3

    def test_record_sample(self):
        partition = make_partition(n_regions=2)
        region = partition.regions[0]
        region.record_sample(violation=True, interaction_id="test1")
        region.record_sample(violation=False, interaction_id="test2")
        assert region.n_samples == 2
        assert region.n_violations == 1
        assert region.p_hat == 0.5


class TestMKG:
    def test_proximity_edges(self):
        """Regions within tau should be connected."""
        partition = make_partition(n_regions=4, spacing=1.0)
        mkg = ModelKnowledgeGraph(partition, tau=1.5)
        # Adjacent regions (distance 1.0 < tau=1.5) should be connected
        assert mkg.graph.has_edge(0, 1)
        assert mkg.graph.has_edge(1, 2)
        # Distant regions should not
        assert not mkg.graph.has_edge(0, 3)

    def test_neighborhood(self):
        partition = make_partition(n_regions=5, spacing=1.0)
        mkg = ModelKnowledgeGraph(partition, tau=1.5)
        # 1-hop neighborhood of region 2
        n1 = mkg.get_neighborhood(2, hops=1)
        assert 2 in n1
        assert 1 in n1
        assert 3 in n1
        # 2-hop neighborhood
        n2 = mkg.get_neighborhood(2, hops=2)
        assert 0 in n2
        assert 4 in n2

    def test_frontier(self):
        partition = make_partition(n_regions=5, spacing=1.0)
        mkg = ModelKnowledgeGraph(partition, tau=1.5)
        explored = {0, 1}
        frontier = mkg.get_frontier_regions(explored)
        assert 2 in frontier
        assert 0 not in frontier
        assert 1 not in frontier

    def test_mutation_edge(self):
        partition = make_partition(n_regions=3)
        mkg = ModelKnowledgeGraph(partition, tau=0.5)
        mkg.add_mutation_edge(0, 2)
        assert mkg.graph.has_edge(0, 2)

    def test_regression_subgraph(self):
        """Test regression detection (Section 5.3)."""
        prev_stats = [
            RegionStats(region_id=0, n_samples=100, n_violations=2, weight=0.5),
            RegionStats(region_id=1, n_samples=100, n_violations=3, weight=0.5),
        ]
        curr_stats = [
            RegionStats(region_id=0, n_samples=100, n_violations=20, weight=0.5),
            RegionStats(region_id=1, n_samples=100, n_violations=3, weight=0.5),
        ]
        partition = make_partition(n_regions=2)
        mkg = ModelKnowledgeGraph(partition, tau=1.5)

        regression = mkg.compute_regression_subgraph(
            prev_stats, curr_stats, delta=0.05, eta=0.01,
        )
        # Region 0 had a large increase in violations
        assert 0 in regression
        # Region 1 did not change
        assert 1 not in regression

    def test_minimal_explanation(self):
        prev_stats = [
            RegionStats(region_id=0, n_samples=100, n_violations=2, weight=0.4),
            RegionStats(region_id=1, n_samples=100, n_violations=2, weight=0.3),
            RegionStats(region_id=2, n_samples=100, n_violations=2, weight=0.3),
        ]
        curr_stats = [
            RegionStats(region_id=0, n_samples=100, n_violations=30, weight=0.4),
            RegionStats(region_id=1, n_samples=100, n_violations=25, weight=0.3),
            RegionStats(region_id=2, n_samples=100, n_violations=2, weight=0.3),
        ]
        partition = make_partition(n_regions=3)
        mkg = ModelKnowledgeGraph(partition, tau=1.5)

        explanation = mkg.minimal_explanation_set(
            prev_stats, curr_stats, delta=0.05, gamma=0.01,
        )
        # Should include the regions with the biggest regressions
        assert len(explanation) >= 1
        assert 0 in explanation  # Largest regression
