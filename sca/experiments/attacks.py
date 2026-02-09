"""Byzantine attack strategies for experiments (Section 7.1).

Implements safety-degrading federated attacks:
- Targeted updates that reduce refusal / increase jailbreak success
  while keeping update norms similar to benign.
- Stealthy adversaries that evade vector-based anomaly detection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from sca.federated.client import ByzantineClient, ClientUpdate, FLClient


@dataclass
class AttackConfig:
    """Configuration for a Byzantine attack scenario.

    Attributes:
        n_byzantine: Number of Byzantine clients.
        attack_type: Attack strategy name.
        scale: Scaling factor for the attack.
        norm_bound: Optional norm bound to match benign updates.
    """
    n_byzantine: int
    attack_type: str = "sign_flip"
    scale: float = 1.0
    norm_bound: float | None = None


class StealthyAttacker(FLClient):
    """Stealthy Byzantine client that matches benign update statistics.

    Projects adversarial updates to match the norm distribution of
    benign updates, making them harder to detect via anomaly detection.
    """

    def __init__(
        self,
        client_id: int,
        attack_direction: dict[str, torch.Tensor] | None = None,
        norm_bound: float = 1.0,
    ) -> None:
        super().__init__(client_id)
        self.attack_direction = attack_direction
        self.norm_bound = norm_bound

    def compute_update(
        self,
        global_model: nn.Module,
        round_num: int,
    ) -> ClientUpdate:
        """Generate an adversarial update constrained to the benign norm ball."""
        delta = {}

        if self.attack_direction is not None:
            # Use provided attack direction
            for name, param in global_model.named_parameters():
                if name in self.attack_direction:
                    delta[name] = self.attack_direction[name].clone()
                else:
                    delta[name] = torch.zeros_like(param)
        else:
            # Random direction within norm ball
            for name, param in global_model.named_parameters():
                delta[name] = torch.randn_like(param)

        # Project to norm ball
        flat = torch.cat([d.flatten() for d in delta.values()])
        norm = flat.norm()
        if norm > self.norm_bound:
            scale = self.norm_bound / norm
            delta = {name: d * scale for name, d in delta.items()}

        return ClientUpdate(
            client_id=self.client_id,
            delta=delta,
            metadata={"attack_type": "stealthy", "norm": float(flat.norm())},
        )


class SafetyDegradationAttacker(FLClient):
    """Attacker that targets safety-relevant parameters.

    Identifies parameters most relevant to safety behavior (e.g.,
    output layer biases, attention heads associated with refusal)
    and perturbs them to reduce safety compliance.
    """

    def __init__(
        self,
        client_id: int,
        target_layer_patterns: list[str] | None = None,
        perturbation_scale: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Args:
            client_id: Client identifier.
            target_layer_patterns: Substrings to match parameter names
                that should be targeted (e.g., ["output", "head", "lm_head"]).
            perturbation_scale: Magnitude of perturbation.
            seed: Random seed.
        """
        super().__init__(client_id)
        self.target_patterns = target_layer_patterns or ["output", "head"]
        self.perturbation_scale = perturbation_scale
        self.rng = torch.Generator().manual_seed(seed)

    def compute_update(
        self,
        global_model: nn.Module,
        round_num: int,
    ) -> ClientUpdate:
        delta = {}

        for name, param in global_model.named_parameters():
            is_target = any(p in name for p in self.target_patterns)

            if is_target:
                # Large perturbation on safety-relevant parameters
                delta[name] = torch.randn(
                    param.shape, generator=self.rng,
                ) * self.perturbation_scale
            else:
                # Small benign-looking noise on other parameters
                delta[name] = torch.randn(
                    param.shape, generator=self.rng,
                ) * self.perturbation_scale * 0.01

        return ClientUpdate(
            client_id=self.client_id,
            delta=delta,
            metadata={"attack_type": "safety_degradation"},
        )


def create_attack_scenario(
    n_total_clients: int,
    config: AttackConfig,
    **kwargs,
) -> list[FLClient]:
    """Create a mixed set of benign and Byzantine clients.

    Returns a list where the first (n_total - n_byzantine) are None
    placeholders (to be replaced with benign clients) and the last
    n_byzantine are Byzantine clients.

    Args:
        n_total_clients: Total number of clients.
        config: Attack configuration.
        **kwargs: Additional args passed to Byzantine clients.

    Returns:
        List of Byzantine FLClient instances (benign slots are None).
    """
    byzantine_clients = []
    for i in range(config.n_byzantine):
        client_id = n_total_clients - config.n_byzantine + i
        if config.attack_type == "stealthy":
            client = StealthyAttacker(
                client_id=client_id,
                norm_bound=config.norm_bound or 1.0,
            )
        elif config.attack_type == "safety_degradation":
            client = SafetyDegradationAttacker(
                client_id=client_id,
                perturbation_scale=config.scale,
            )
        else:
            client = ByzantineClient(
                client_id=client_id,
                attack_type=config.attack_type,
                scale=config.scale,
            )
        byzantine_clients.append(client)

    return byzantine_clients
