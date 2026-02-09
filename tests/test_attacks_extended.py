"""Tests for extended attack strategies."""

import torch
import torch.nn as nn
import pytest

from sca.experiments.attacks import (
    AttackConfig,
    AttackScenario,
    InnerProductManipulationAttacker,
    StealthyAttacker,
    SafetyDegradationAttacker,
    create_attack_scenario,
    create_standard_attack_scenarios,
)
from sca.federated.client import ClientUpdate


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        # Name one layer 'output' to test SafetyDegradationAttacker
        self.output_layer = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.output_layer(self.linear(x))


class TestIPMAttacker:
    def test_creates_update(self):
        model = SimpleModel()
        attacker = InnerProductManipulationAttacker(client_id=0, epsilon=0.1)
        update = attacker.compute_update(model, round_num=1)
        assert isinstance(update, ClientUpdate)
        assert "linear.weight" in update.delta

    def test_negative_direction(self):
        """IPM update should oppose the sign of parameters."""
        model = SimpleModel()
        # Set known parameters
        with torch.no_grad():
            model.linear.weight.fill_(1.0)

        attacker = InnerProductManipulationAttacker(client_id=0, epsilon=0.5)
        update = attacker.compute_update(model, round_num=1)
        # Delta should be negative (opposing positive params)
        assert (update.delta["linear.weight"] < 0).all()


class TestStealthyAttacker:
    def test_norm_bound(self):
        """Stealthy updates should respect the norm bound."""
        model = SimpleModel()
        attacker = StealthyAttacker(client_id=0, norm_bound=0.5)
        update = attacker.compute_update(model, round_num=1)

        flat = torch.cat([d.flatten() for d in update.delta.values()])
        assert flat.norm().item() <= 0.5 + 1e-6


class TestSafetyDegradationAttacker:
    def test_targets_output_layers(self):
        """Should apply larger perturbations to 'output' named layers."""
        model = SimpleModel()
        attacker = SafetyDegradationAttacker(
            client_id=0,
            target_layer_patterns=["output"],
            perturbation_scale=1.0,
        )
        update = attacker.compute_update(model, round_num=1)

        # Output layer should have larger perturbation than linear
        output_norm = update.delta["output_layer.weight"].norm().item()
        linear_norm = update.delta["linear.weight"].norm().item()
        assert output_norm > linear_norm * 5  # 100x scale difference


class TestAttackScenarios:
    def test_create_standard_scenarios(self):
        scenarios = create_standard_attack_scenarios(n_total_clients=10)
        assert len(scenarios) >= 4

        names = {s.name for s in scenarios}
        assert "no_attack" in names
        assert "sign_flip" in names
        assert "stealthy" in names
        assert "safety_degradation" in names

    def test_no_attack_scenario(self):
        scenarios = create_standard_attack_scenarios(n_total_clients=10)
        no_attack = [s for s in scenarios if s.name == "no_attack"][0]
        assert no_attack.configs[0].n_byzantine == 0

    def test_create_ipm_scenario(self):
        clients = create_attack_scenario(
            n_total_clients=10,
            config=AttackConfig(
                n_byzantine=2,
                attack_type="ipm",
                scale=0.1,
            ),
        )
        assert len(clients) == 2
        assert all(isinstance(c, InnerProductManipulationAttacker) for c in clients)
