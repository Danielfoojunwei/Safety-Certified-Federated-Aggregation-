"""Tests for federated aggregation rules."""

import torch
import torch.nn as nn
import pytest

from sca.federated.aggregation import (
    CoordinateMedian,
    FedAdam,
    FedAvg,
    Krum,
    TrimmedMean,
)
from sca.federated.client import ClientUpdate


class SimpleModel(nn.Module):
    """Minimal model for testing aggregation."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.linear(x)


def make_updates(model, n=5, scale=0.01, seed=42):
    """Create synthetic client updates."""
    torch.manual_seed(seed)
    updates = []
    for i in range(n):
        delta = {}
        for name, param in model.named_parameters():
            delta[name] = torch.randn_like(param) * scale
        updates.append(ClientUpdate(
            client_id=i,
            delta=delta,
            n_samples=100,
        ))
    return updates


class TestFedAvg:
    def test_basic(self):
        model = SimpleModel()
        updates = make_updates(model, n=3)
        agg = FedAvg()
        result = agg.aggregate(model, updates)
        assert "linear.weight" in result
        assert result["linear.weight"].shape == model.linear.weight.shape

    def test_single_client(self):
        model = SimpleModel()
        updates = make_updates(model, n=1)
        agg = FedAvg()
        result = agg.aggregate(model, updates)
        # Single client: aggregate should equal the client's delta
        torch.testing.assert_close(
            result["linear.weight"],
            updates[0].delta["linear.weight"],
        )

    def test_empty(self):
        model = SimpleModel()
        agg = FedAvg()
        result = agg.aggregate(model, [])
        assert result == {}


class TestFedAdam:
    def test_basic(self):
        model = SimpleModel()
        updates = make_updates(model, n=3)
        agg = FedAdam(lr=0.01)
        result = agg.aggregate(model, updates)
        assert "linear.weight" in result

    def test_multiple_rounds(self):
        """FedAdam should accumulate momentum across rounds."""
        model = SimpleModel()
        agg = FedAdam(lr=0.01)
        for _ in range(3):
            updates = make_updates(model, n=3)
            result = agg.aggregate(model, updates)
        assert agg.t == 3


class TestRobustAggregation:
    def test_median(self):
        model = SimpleModel()
        updates = make_updates(model, n=5)
        agg = CoordinateMedian()
        result = agg.aggregate(model, updates)
        assert "linear.weight" in result

    def test_trimmed_mean(self):
        model = SimpleModel()
        updates = make_updates(model, n=5)
        agg = TrimmedMean(beta=0.2)
        result = agg.aggregate(model, updates)
        assert "linear.weight" in result

    def test_trimmed_mean_robust_to_outlier(self):
        """Trimmed mean should be robust to extreme updates."""
        model = SimpleModel()
        updates = make_updates(model, n=5, scale=0.01)
        # Add an outlier
        outlier_delta = {}
        for name, param in model.named_parameters():
            outlier_delta[name] = torch.ones_like(param) * 100.0
        updates.append(ClientUpdate(
            client_id=99, delta=outlier_delta, n_samples=100,
        ))

        agg_avg = FedAvg()
        agg_trim = TrimmedMean(beta=0.2)

        result_avg = agg_avg.aggregate(model, updates)
        result_trim = agg_trim.aggregate(model, updates)

        # Trimmed mean should be closer to zero than FedAvg
        avg_norm = result_avg["linear.weight"].norm()
        trim_norm = result_trim["linear.weight"].norm()
        assert trim_norm < avg_norm

    def test_krum(self):
        model = SimpleModel()
        updates = make_updates(model, n=5)
        agg = Krum(n_byzantine=1)
        result = agg.aggregate(model, updates)
        assert "linear.weight" in result

    def test_krum_rejects_outlier(self):
        """Krum should not select the outlier update."""
        model = SimpleModel()
        updates = make_updates(model, n=5, scale=0.01)
        # Make the last update an obvious outlier
        outlier_delta = {}
        for name, param in model.named_parameters():
            outlier_delta[name] = torch.ones_like(param) * 100.0
        updates.append(ClientUpdate(
            client_id=99, delta=outlier_delta, n_samples=100,
        ))

        agg = Krum(n_byzantine=1)
        result = agg.aggregate(model, updates)

        # Result should NOT be the outlier
        assert result["linear.weight"].norm() < 10.0
