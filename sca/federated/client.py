"""Federated learning clients (Section 1.1).

Clients i = 1, ..., n produce local updates Delta_{t,i}.
Byzantine clients B subset {1,...,n} may submit arbitrary Delta_{t,i}.

This module provides:
- BenignClient: honest client performing local SGD.
- ByzantineClient: adversarial client injecting safety-degrading updates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class ClientUpdate:
    """A client's local update Delta_{t,i}.

    Attributes:
        client_id: Client identifier.
        delta: Parameter update (same shape as model params).
        n_samples: Number of local training samples used.
        metadata: Optional metadata (for auditing).
    """
    client_id: int
    delta: dict[str, torch.Tensor]
    n_samples: int = 0
    metadata: dict[str, Any] | None = None


class FLClient(ABC):
    """Abstract base class for federated learning clients."""

    def __init__(self, client_id: int) -> None:
        self.client_id = client_id

    @abstractmethod
    def compute_update(
        self,
        global_model: nn.Module,
        round_num: int,
    ) -> ClientUpdate:
        """Compute a local update from the current global model.

        Args:
            global_model: Current global model M(theta_t).
            round_num: Current FL round t.

        Returns:
            The local update Delta_{t,i}.
        """


class BenignClient(FLClient):
    """Honest client performing local SGD on private data."""

    def __init__(
        self,
        client_id: int,
        dataset: Any,
        local_epochs: int = 1,
        lr: float = 0.01,
        batch_size: int = 32,
    ) -> None:
        super().__init__(client_id)
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size

    def compute_update(
        self,
        global_model: nn.Module,
        round_num: int,
    ) -> ClientUpdate:
        """Perform local SGD and return the parameter delta."""
        import copy
        local_model = copy.deepcopy(global_model)
        local_model.train()

        # Save initial parameters
        init_params = {
            name: param.clone()
            for name, param in local_model.named_parameters()
        }

        optimizer = torch.optim.SGD(local_model.parameters(), lr=self.lr)

        if isinstance(self.dataset, DataLoader):
            loader = self.dataset
        else:
            loader = DataLoader(
                self.dataset, batch_size=self.batch_size, shuffle=True,
            )

        n_samples = 0
        for _ in range(self.local_epochs):
            for batch in loader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, batch
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                n_samples += inputs.size(0)

        # Compute delta = theta_local - theta_global
        delta = {}
        for name, param in local_model.named_parameters():
            delta[name] = param.data - init_params[name]

        return ClientUpdate(
            client_id=self.client_id,
            delta=delta,
            n_samples=n_samples,
        )


class ByzantineClient(FLClient):
    """Byzantine client submitting safety-degrading updates.

    Implements attack strategies from Section 7.1:
    targeted updates that reduce refusal / increase jailbreak success
    while keeping update norms similar to benign.
    """

    def __init__(
        self,
        client_id: int,
        attack_type: str = "sign_flip",
        scale: float = 1.0,
        target_params: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Args:
            client_id: Client identifier.
            attack_type: Attack strategy ("sign_flip", "noise", "targeted", "scaling").
            scale: Scaling factor for the attack.
            target_params: Target parameters for targeted attacks.
        """
        super().__init__(client_id)
        self.attack_type = attack_type
        self.scale = scale
        self.target_params = target_params

    def compute_update(
        self,
        global_model: nn.Module,
        round_num: int,
    ) -> ClientUpdate:
        """Generate an adversarial update."""
        delta = {}

        if self.attack_type == "sign_flip":
            # Flip the sign of gradients (direction attack)
            for name, param in global_model.named_parameters():
                # Simulate a benign-looking gradient then flip
                noise = torch.randn_like(param) * 0.01
                delta[name] = -noise * self.scale

        elif self.attack_type == "noise":
            # Add large noise scaled to match benign norms
            for name, param in global_model.named_parameters():
                delta[name] = torch.randn_like(param) * self.scale * 0.01

        elif self.attack_type == "targeted":
            # Push parameters toward a target (e.g., a less-safe checkpoint)
            if self.target_params is None:
                raise ValueError("target_params required for targeted attack")
            for name, param in global_model.named_parameters():
                if name in self.target_params:
                    delta[name] = (
                        self.target_params[name] - param.data
                    ) * self.scale

        elif self.attack_type == "scaling":
            # Scale up a benign-looking update
            for name, param in global_model.named_parameters():
                noise = torch.randn_like(param) * 0.01
                delta[name] = noise * self.scale

        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

        return ClientUpdate(
            client_id=self.client_id,
            delta=delta,
            metadata={"attack_type": self.attack_type, "scale": self.scale},
        )
