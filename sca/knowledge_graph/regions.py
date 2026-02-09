"""Interaction regions R_j and their management.

Section 3.1: The interaction space X is partitioned into K regions R_j
via clustering in the embedding space. Each region has a centroid c_j,
weight w_j = Pr[x in R_j], and tracks sample statistics.

Section 6.2 (MKG update): New regions are created dynamically when
a tested interaction is far from all existing centroids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class InteractionRegion:
    """A region R_j in the interaction space partition.

    Attributes:
        region_id: Index j.
        centroid: Cluster centroid c_j in embedding space.
        weight: Prior weight w_j = Pr[x in R_j].
        n_samples: Number of verifier samples drawn from this region.
        n_violations: Number of safety violations observed.
        member_ids: IDs of interactions assigned to this region.
    """
    region_id: int
    centroid: np.ndarray
    weight: float = 0.0
    n_samples: int = 0
    n_violations: int = 0
    member_ids: list[str] = field(default_factory=list)

    @property
    def p_hat(self) -> float:
        """Empirical violation rate."""
        if self.n_samples == 0:
            return 0.0
        return self.n_violations / self.n_samples

    def record_sample(self, violation: bool, interaction_id: str | None = None) -> None:
        """Record a new sample observation in this region."""
        self.n_samples += 1
        if violation:
            self.n_violations += 1
        if interaction_id is not None:
            self.member_ids.append(interaction_id)

    def reset_stats(self) -> None:
        """Reset sample statistics (for a new verification round)."""
        self.n_samples = 0
        self.n_violations = 0
        self.member_ids.clear()


class RegionPartition:
    """Manages the partition X = bigsqcup R_j.

    Supports:
    - Initial clustering from a reference dataset.
    - Dynamic region creation when new interactions are far from
      existing centroids (Section 6.2).
    - Assignment of new interactions to regions.
    """

    def __init__(
        self,
        tau_new: float = 2.0,
    ) -> None:
        """
        Args:
            tau_new: Distance threshold for creating new regions.
                     If min_l dist(psi(x), c_l) > tau_new, create new region.
        """
        self.tau_new = tau_new
        self.regions: list[InteractionRegion] = []

    @property
    def k(self) -> int:
        """Number of regions K."""
        return len(self.regions)

    def initialize_from_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        weights: np.ndarray | None = None,
    ) -> None:
        """Initialize regions via K-means clustering.

        Args:
            embeddings: (N, p) embedding matrix of reference interactions.
            n_clusters: Target number of clusters K.
            weights: Optional per-cluster weights. If None, estimated from
                     cluster sizes.
        """
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)

        self.regions = []
        for j in range(n_clusters):
            mask = labels == j
            w = weights[j] if weights is not None else mask.sum() / len(labels)
            region = InteractionRegion(
                region_id=j,
                centroid=km.cluster_centers_[j].copy(),
                weight=float(w),
            )
            self.regions.append(region)

    def assign(self, embedding: np.ndarray) -> int:
        """Assign an embedding to the nearest region, creating a new one if needed.

        Section 6.2: j = argmin_l dist(psi(x), c_l).
        If min_l dist > tau_new, create new region.

        Args:
            embedding: Embedding vector of shape (p,).

        Returns:
            Region index j.
        """
        if not self.regions:
            # First region
            region = InteractionRegion(
                region_id=0,
                centroid=embedding.copy(),
                weight=1.0,
            )
            self.regions.append(region)
            return 0

        # Find nearest centroid
        centroids = np.stack([r.centroid for r in self.regions])
        dists = np.linalg.norm(centroids - embedding, axis=1)
        j = int(np.argmin(dists))
        min_dist = dists[j]

        if min_dist > self.tau_new:
            # Create new region (Section 6.2)
            new_id = len(self.regions)
            region = InteractionRegion(
                region_id=new_id,
                centroid=embedding.copy(),
                weight=0.0,  # Weight will be updated
            )
            self.regions.append(region)
            return new_id

        return j

    def get_weights(self) -> np.ndarray:
        """Return region weights as a numpy array."""
        w = np.array([r.weight for r in self.regions])
        total = w.sum()
        if total > 0:
            w /= total  # Normalize
        return w

    def reset_all_stats(self) -> None:
        """Reset all region sample statistics for a new verification round."""
        for region in self.regions:
            region.reset_stats()
