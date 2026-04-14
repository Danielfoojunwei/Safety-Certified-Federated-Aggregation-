"""Aggregation rules for federated learning (Section 1.1).

Server aggregator A proposes:
    theta_tilde_{t+1} = A(theta_t; Delta_{t,1}, ..., Delta_{t,n})

Implements:
- FedAvg: simple averaging of client updates.
- FedAdam: adaptive server optimizer.
- Robust aggregation: coordinate-wise median, trimmed mean, Krum.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from sca.federated.client import ClientUpdate


class Aggregator(ABC):
    """Abstract base class for federated aggregation rules."""

    @abstractmethod
    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        """Aggregate client updates into a global parameter update.

        Args:
            global_model: Current global model M(theta_t).
            updates: List of client updates Delta_{t,i}.

        Returns:
            Aggregated update to apply to global_model parameters.
        """


class FedAvg(Aggregator):
    """Federated Averaging (McMahan et al., 2017).

    Weighted average of client updates:
        Delta = sum_i (n_i / N) * Delta_i
    """

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}

        total_samples = sum(max(u.n_samples, 1) for u in updates)
        aggregated = {}

        for name, _ in global_model.named_parameters():
            weighted_sum = torch.zeros_like(updates[0].delta[name])
            for u in updates:
                weight = max(u.n_samples, 1) / total_samples
                weighted_sum += weight * u.delta[name]
            aggregated[name] = weighted_sum

        return aggregated


class FedAdam(Aggregator):
    """Federated Adam optimizer (Reddi et al., 2021).

    Applies Adam-style adaptive learning rate on the server side
    to the averaged client updates.
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-3,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict[str, torch.Tensor] = {}
        self.v: dict[str, torch.Tensor] = {}
        self.t = 0

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}

        self.t += 1

        # First compute FedAvg-style mean
        total_samples = sum(max(u.n_samples, 1) for u in updates)
        pseudo_gradient = {}
        for name, _ in global_model.named_parameters():
            weighted_sum = torch.zeros_like(updates[0].delta[name])
            for u in updates:
                weight = max(u.n_samples, 1) / total_samples
                weighted_sum += weight * u.delta[name]
            pseudo_gradient[name] = -weighted_sum  # Negate: delta is update, not gradient

        # Apply Adam
        aggregated = {}
        for name, param in global_model.named_parameters():
            g = pseudo_gradient[name]

            if name not in self.m:
                self.m[name] = torch.zeros_like(g)
                self.v[name] = torch.zeros_like(g)

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Return update (negative gradient step)
            aggregated[name] = -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

        return aggregated


class CoordinateMedian(Aggregator):
    """Coordinate-wise median aggregation (Byzantine-robust).

    For each parameter coordinate, takes the median across client updates.
    """

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}

        aggregated = {}
        for name, _ in global_model.named_parameters():
            stacked = torch.stack([u.delta[name] for u in updates])
            aggregated[name] = torch.median(stacked, dim=0).values

        return aggregated


class TrimmedMean(Aggregator):
    """Trimmed mean aggregation (Byzantine-robust).

    Removes the top and bottom beta fraction of values at each coordinate
    before averaging.
    """

    def __init__(self, beta: float = 0.1) -> None:
        """
        Args:
            beta: Fraction to trim from each end (0 < beta < 0.5).
        """
        self.beta = beta

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}

        n = len(updates)
        trim_count = max(1, int(n * self.beta))

        aggregated = {}
        for name, _ in global_model.named_parameters():
            stacked = torch.stack([u.delta[name] for u in updates])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            # Trim top and bottom
            trimmed = sorted_vals[trim_count:n - trim_count]
            aggregated[name] = trimmed.mean(dim=0)

        return aggregated


class Krum(Aggregator):
    """Krum aggregation (Blanchard et al., 2017).

    Selects the update whose sum of distances to its nearest n-f-2
    neighbors is smallest, where f is the number of Byzantine clients.

    IMPORTANT: n_byzantine must be set correctly for each experiment.
    When n_byzantine=0, Krum degenerates (n_nearest = n-2) and may
    select a Byzantine update. Always set n_byzantine >= actual count.
    """

    def __init__(self, n_byzantine: int = 0) -> None:
        """
        Args:
            n_byzantine: Upper bound on number of Byzantine clients f.
                        Must satisfy f < (n - 2) / 2 for Krum's guarantee.
        """
        self.n_byzantine = n_byzantine

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}

        n = len(updates)
        f = self.n_byzantine

        # Krum requires n >= 2f + 3 for its theoretical guarantee
        if n < 2 * f + 3:
            import logging
            logging.getLogger(__name__).warning(
                "Krum: n=%d < 2*f+3=%d. Theoretical guarantee does not hold. "
                "Falling back to FedAvg-style selection of closest-to-mean update.",
                n, 2 * f + 3,
            )

        n_nearest = max(1, n - f - 2)

        # Flatten each update into a single vector
        flat_updates = []
        for u in updates:
            flat = torch.cat([u.delta[name].flatten() for name in sorted(u.delta)])
            flat_updates.append(flat)

        flat_stack = torch.stack(flat_updates)

        # Compute pairwise distances
        scores = torch.zeros(n)
        for i in range(n):
            dists = torch.norm(flat_stack - flat_stack[i], dim=1)
            dists[i] = float("inf")  # Exclude self
            sorted_dists, _ = torch.sort(dists)
            scores[i] = sorted_dists[:n_nearest].sum()

        # Select the update with the smallest score
        best_idx = torch.argmin(scores).item()
        return updates[best_idx].delta


class MultiKrum(Aggregator):
    """Multi-Krum aggregation (Blanchard et al., 2017).

    Selects the top-k closest updates (by Krum score) and averages
    them. Generally more robust than single-Krum.
    """

    def __init__(self, n_byzantine: int = 0, top_k: int = 3) -> None:
        """
        Args:
            n_byzantine: Upper bound on number of Byzantine clients f.
            top_k: Number of top-scoring updates to average.
        """
        self.n_byzantine = n_byzantine
        self.top_k = top_k

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}

        n = len(updates)
        f = self.n_byzantine
        n_nearest = max(1, n - f - 2)
        k = min(self.top_k, n)

        flat_updates = []
        for u in updates:
            flat = torch.cat([u.delta[name].flatten() for name in sorted(u.delta)])
            flat_updates.append(flat)

        flat_stack = torch.stack(flat_updates)

        scores = torch.zeros(n)
        for i in range(n):
            dists = torch.norm(flat_stack - flat_stack[i], dim=1)
            dists[i] = float("inf")
            sorted_dists, _ = torch.sort(dists)
            scores[i] = sorted_dists[:n_nearest].sum()

        # Select top-k updates (lowest scores)
        _, top_indices = torch.topk(scores, k, largest=False)

        aggregated = {}
        for name, _ in global_model.named_parameters():
            selected = torch.stack([updates[i].delta[name] for i in top_indices])
            aggregated[name] = selected.mean(dim=0)

        return aggregated


class FLTrust(Aggregator):
    """FLTrust aggregation (Cao et al., NDSS 2021).

    The server maintains a small clean root dataset and computes its
    own reference update. Client updates are weighted by their cosine
    similarity to the server update (negative similarities clipped to 0).

    This is a strong modern baseline for Byzantine-robust FL.
    """

    def __init__(self, server_update: dict[str, torch.Tensor] | None = None) -> None:
        """
        Args:
            server_update: The server's own update computed on trusted data.
                          Can be set after construction via set_server_update().
        """
        self.server_update = server_update

    def set_server_update(self, update: dict[str, torch.Tensor]) -> None:
        """Set the server's reference update for this round."""
        self.server_update = update

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}

        if self.server_update is None:
            # Fallback to FedAvg if no server update
            return FedAvg().aggregate(global_model, updates)

        # Flatten server update
        server_flat = torch.cat([
            self.server_update[name].flatten()
            for name in sorted(self.server_update)
        ])
        server_norm = torch.norm(server_flat)
        if server_norm < 1e-10:
            return FedAvg().aggregate(global_model, updates)

        # Compute trust scores (ReLU cosine similarity)
        trust_scores = []
        normalized_updates = []
        for u in updates:
            client_flat = torch.cat([
                u.delta[name].flatten() for name in sorted(u.delta)
            ])
            client_norm = torch.norm(client_flat)
            if client_norm < 1e-10:
                trust_scores.append(0.0)
                normalized_updates.append(u.delta)
                continue

            cosine_sim = torch.dot(server_flat, client_flat) / (server_norm * client_norm)
            trust = max(0.0, cosine_sim.item())  # ReLU
            trust_scores.append(trust)

            # Normalize client update to server update magnitude
            scale = server_norm / client_norm
            normed = {name: d * scale for name, d in u.delta.items()}
            normalized_updates.append(normed)

        total_trust = sum(trust_scores)
        if total_trust < 1e-10:
            return FedAvg().aggregate(global_model, updates)

        # Weighted average by trust scores
        aggregated = {}
        for name, _ in global_model.named_parameters():
            weighted_sum = torch.zeros_like(updates[0].delta[name])
            for i, normed in enumerate(normalized_updates):
                weight = trust_scores[i] / total_trust
                weighted_sum += weight * normed[name]
            aggregated[name] = weighted_sum

        return aggregated
