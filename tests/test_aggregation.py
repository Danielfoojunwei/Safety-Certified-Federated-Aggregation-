"""Tests for aggregation rules, the accept/rollback protocol, and the FL controls.

These tests target audit findings F5 (rollback was invisible) and F7 (Krum
collapsed with no adversary because the client split was degenerate, and that
collapse was reported as a property of Krum).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from sca.experiments.attacks import AttackConfig
from sca.experiments.baselines import (
    FLBaselineConfig,
    assert_always_accept_equals_no_gate,
    assert_always_reject_equals_frozen,
    run_fl_baseline,
    run_gate_control_suite,
)
from sca.federated.aggregation import (
    AGGREGATOR_REGISTRY,
    CoordinateMedian,
    FedAdam,
    FedAvg,
    FLTrust,
    Krum,
    MultiKrum,
    TrimmedMean,
    build_aggregator,
    krum_scores,
)
from sca.federated.client import BenignClient, ClientUpdate
from sca.federated.server import (
    AlwaysAcceptGate,
    AlwaysRejectGate,
    FederatedServer,
    FunctionGate,
    GateDecision,
    RoundResult,
    model_state_hash,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

N_CLASSES = 4
N_CLIENTS = 8
DIM = 10


class SimpleModel(nn.Module):
    def __init__(self, d: int = 4, k: int = 2):
        super().__init__()
        self.linear = nn.Linear(d, k, bias=False)

    def forward(self, x):
        return self.linear(x)


def make_updates(model, n=5, scale=0.01, seed=42, client_ids=None):
    g = torch.Generator().manual_seed(seed)
    updates = []
    for i in range(n):
        delta = {
            name: torch.randn(param.shape, generator=g) * scale
            for name, param in model.named_parameters()
        }
        updates.append(
            ClientUpdate(
                client_id=(client_ids[i] if client_ids else i),
                delta=delta,
                n_samples=100,
            )
        )
    return updates


def constant_update(model, value, client_id=99, n_samples=100, byz=False):
    return ClientUpdate(
        client_id=client_id,
        delta={
            name: torch.full_like(param, float(value))
            for name, param in model.named_parameters()
        },
        n_samples=n_samples,
        is_byzantine=byz,
    )


# -- a small but genuinely non-trivial classification task -------------------


def make_task(seed: int = 0, n_train: int = 480, n_test: int = 240):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(N_CLASSES, DIM)) * 0.8

    def sample(n):
        y = rng.integers(0, N_CLASSES, size=n)
        X = centers[y] + rng.normal(size=(n, DIM))
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )

    return sample(n_train), sample(n_test)


def dirichlet_split(y: np.ndarray, n_clients: int, alpha: float, seed: int):
    """Non-IID split by class proportions with balanced client sizes.

    This is the split a federated benchmark is supposed to use.  Contrast
    :func:`label_sorted_split`.
    """
    rng = np.random.default_rng(seed)
    pools = {c: list(rng.permutation(np.where(y == c)[0])) for c in range(N_CLASSES)}
    per_client = len(y) // n_clients
    parts = []
    for _ in range(n_clients):
        p = rng.dirichlet([alpha] * N_CLASSES)
        chosen = []
        for c in rng.choice(N_CLASSES, size=per_client, p=p):
            for cc in [c] + [k for k in range(N_CLASSES) if k != c]:
                if pools[cc]:
                    chosen.append(pools[cc].pop())
                    break
        parts.append(np.array(sorted(chosen)))
    return parts


def label_sorted_split(y: np.ndarray, n_clients: int):
    """The F7 split: sort by label and chunk contiguously.

    Every client ends up holding (almost) a single class.  This is what
    ``prepare_fl_data`` did in the old harness.
    """
    order = np.argsort(y, kind="stable")
    return [np.array(sorted(c)) for c in np.array_split(order, n_clients)]


def zero_init_model(k: int = N_CLASSES) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Linear(DIM, k)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


def accuracy_fn(X, y):
    def _eval(model: nn.Module):
        model.eval()
        with torch.no_grad():
            pred = model(X).argmax(dim=1)
            return {"accuracy": float((pred == y).float().mean())}

    return _eval


def run_fl(parts, X, y, eval_fn, agg_name, n_rounds=20, lr=0.5, n_byzantine=0):
    datasets = [TensorDataset(X[p], y[p]) for p in parts]
    clients = [
        BenignClient(i, ds, lr=lr, local_epochs=1, batch_size=16, seed=0)
        for i, ds in enumerate(datasets)
    ]
    server = FederatedServer(
        zero_init_model(),
        aggregator=build_aggregator(agg_name, n_byzantine=n_byzantine),
        gate=None,
        eval_fn=eval_fn,
    )
    server.run(clients, n_rounds)
    return server.history[-1].committed_metric["accuracy"], server


# ---------------------------------------------------------------------------
# Plain aggregation behaviour
# ---------------------------------------------------------------------------


class TestFedAvg:
    def test_basic(self):
        model = SimpleModel()
        result = FedAvg().aggregate(model, make_updates(model, n=3))
        assert result["linear.weight"].shape == model.linear.weight.shape

    def test_single_client(self):
        model = SimpleModel()
        updates = make_updates(model, n=1)
        result = FedAvg().aggregate(model, updates)
        torch.testing.assert_close(
            result["linear.weight"], updates[0].delta["linear.weight"]
        )

    def test_empty(self):
        assert FedAvg().aggregate(SimpleModel(), []) == {}

    def test_sample_weighting_vs_uniform(self):
        model = SimpleModel()
        a = constant_update(model, 1.0, client_id=0, n_samples=900)
        b = constant_update(model, 0.0, client_id=1, n_samples=100)
        weighted = FedAvg().aggregate(model, [a, b])["linear.weight"]
        uniform = FedAvg(uniform=True).aggregate(model, [a, b])["linear.weight"]
        assert weighted.mean().item() == pytest.approx(0.9, abs=1e-6)
        assert uniform.mean().item() == pytest.approx(0.5, abs=1e-6)


class TestFedAdam:
    def test_accumulates_state(self):
        model = SimpleModel()
        agg = FedAdam(lr=0.01)
        for _ in range(3):
            agg.aggregate(model, make_updates(model, n=3))
        assert agg.t == 3


class TestMedianAndTrimmedMean:
    def test_median_ignores_single_outlier(self):
        model = SimpleModel()
        updates = [constant_update(model, v, client_id=i) for i, v in enumerate([1, 1, 1, 1, 1000])]
        out = CoordinateMedian().aggregate(model, updates)["linear.weight"]
        assert out.mean().item() == pytest.approx(1.0)

    def test_trimmed_mean_more_robust_than_mean(self):
        model = SimpleModel()
        updates = make_updates(model, n=5, scale=0.01)
        updates.append(constant_update(model, 100.0))
        avg = FedAvg().aggregate(model, updates)["linear.weight"].norm()
        trimmed = TrimmedMean(beta=0.2).aggregate(model, updates)["linear.weight"].norm()
        assert trimmed < avg

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_never_produces_nan_for_small_n(self, n):
        """The old implementation trimmed the list empty and returned NaN."""
        model = SimpleModel()
        out = TrimmedMean(beta=0.4).aggregate(model, make_updates(model, n=n))
        assert torch.isfinite(out["linear.weight"]).all()


# ---------------------------------------------------------------------------
# Krum
# ---------------------------------------------------------------------------


class TestKrum:
    def test_scores_match_hand_computation(self):
        """Krum score = sum of the n-f-2 smallest SQUARED distances."""
        model = SimpleModel(d=1, k=1)
        vals = [0.0, 1.0, 2.0, 10.0, 11.0]
        updates = [constant_update(model, v, client_id=i) for i, v in enumerate(vals)]
        scores = krum_scores(updates, n_byzantine=1)  # n=5, f=1 -> 2 nearest
        expected = []
        for i, v in enumerate(vals):
            d2 = sorted((v - w) ** 2 for j, w in enumerate(vals) if j != i)
            expected.append(sum(d2[:2]))
        torch.testing.assert_close(
            scores, torch.tensor(expected, dtype=scores.dtype), rtol=1e-5, atol=1e-6
        )

    def test_returns_a_single_clients_update_verbatim(self):
        model = SimpleModel()
        updates = make_updates(model, n=6, scale=0.01)
        out = Krum(n_byzantine=1).aggregate(model, updates)
        idx = Krum(n_byzantine=1)._select(updates)
        torch.testing.assert_close(out["linear.weight"], updates[idx].delta["linear.weight"])

    def test_rejects_obvious_outlier(self):
        model = SimpleModel()
        updates = make_updates(model, n=5, scale=0.01)
        updates.append(constant_update(model, 100.0, byz=True))
        agg = Krum(n_byzantine=1)
        out = agg.aggregate(model, updates)
        assert out["linear.weight"].norm() < 1.0
        assert agg.last_info["selected_is_byzantine"] is False

    def test_reports_assumption_satisfied(self):
        agg = Krum(n_byzantine=1)
        status = agg.check_assumptions(n_clients=6)
        assert status.satisfied
        assert status.requirement == "n > 2f + 2"

    def test_reports_assumption_violated_at_50_percent_byzantine(self):
        """f = n/2 is OUTSIDE Krum's stated assumption and must be annotated.

        Reporting the resulting degradation as a defeat of Krum is finding F7.
        """
        agg = Krum(n_byzantine=5)
        status = agg.check_assumptions(n_clients=10)
        assert not status.satisfied
        assert "OUTSIDE STATED ASSUMPTION" in status.detail
        assert status.n_byzantine_assumed == 5

    def test_records_assumption_on_every_aggregate_call(self):
        model = SimpleModel()
        agg = Krum(n_byzantine=4)
        agg.aggregate(model, make_updates(model, n=6))
        assert agg.last_assumption is not None
        assert agg.last_assumption.satisfied is False
        assert agg.last_info["assumption_satisfied"] is False

    def test_strict_mode_refuses_to_run_outside_assumption(self):
        model = SimpleModel()
        with pytest.raises(ValueError, match="outside its stated assumption"):
            Krum(n_byzantine=4, strict=True).aggregate(model, make_updates(model, n=6))


class TestMultiKrum:
    def test_averages_selected_updates(self):
        model = SimpleModel(d=1, k=1)
        vals = [1.0, 1.1, 0.9, 50.0]
        updates = [constant_update(model, v, client_id=i) for i, v in enumerate(vals)]
        agg = MultiKrum(n_byzantine=1)
        out = agg.aggregate(model, updates)
        assert agg.last_info["n_selected"] == 3
        assert 50.0 not in [vals[i] for i in agg.last_info["selected_indices"]]
        assert out["linear.weight"].mean().item() == pytest.approx(1.0, abs=0.15)

    def test_shares_krum_assumption(self):
        status = MultiKrum(n_byzantine=5).check_assumptions(n_clients=10)
        assert not status.satisfied
        assert status.aggregator == "multi_krum"


# ---------------------------------------------------------------------------
# FLTrust -- the closest prior work
# ---------------------------------------------------------------------------


class TestFLTrust:
    def test_zero_trust_for_opposing_updates(self):
        model = SimpleModel()
        server_update = {n: torch.ones_like(p) for n, p in model.named_parameters()}
        good = constant_update(model, 1.0, client_id=0)
        bad = constant_update(model, -1.0, client_id=1, byz=True)
        agg = FLTrust(server_update=server_update)
        out = agg.aggregate(model, [good, bad])
        assert agg.last_info["trust_scores"][1] == pytest.approx(0.0)
        # Only the aligned client contributes, rescaled to the server norm.
        assert (out["linear.weight"] > 0).all()

    def test_normalizes_scaled_updates(self):
        """A 100x scaled-up update must not get 100x influence."""
        model = SimpleModel()
        server_update = {n: torch.ones_like(p) for n, p in model.named_parameters()}
        normal = constant_update(model, 1.0, client_id=0)
        scaled = constant_update(model, 100.0, client_id=1, byz=True)
        out = FLTrust(server_update=server_update).aggregate(model, [normal, scaled])
        server_norm = torch.cat(
            [v.flatten() for v in server_update.values()]
        ).norm()
        out_norm = torch.cat([v.flatten() for v in out.values()]).norm()
        assert out_norm.item() == pytest.approx(server_norm.item(), rel=1e-4)

    def test_all_opposing_yields_zero_update(self):
        model = SimpleModel()
        server_update = {n: torch.ones_like(p) for n, p in model.named_parameters()}
        bad = [constant_update(model, -1.0, client_id=i, byz=True) for i in range(3)]
        agg = FLTrust(server_update=server_update)
        out = agg.aggregate(model, bad)
        assert agg.last_info["all_rejected"] is True
        assert out["linear.weight"].abs().max().item() == 0.0

    def test_output_depends_on_server_update(self):
        model = SimpleModel()
        updates = make_updates(model, n=4, scale=0.1)
        a = FLTrust(
            server_update={n: torch.ones_like(p) for n, p in model.named_parameters()}
        ).aggregate(model, updates)
        b = FLTrust(
            server_update={n: -torch.ones_like(p) for n, p in model.named_parameters()}
        ).aggregate(model, updates)
        assert not torch.allclose(a["linear.weight"], b["linear.weight"])

    def test_missing_server_update_is_flagged_not_silent(self):
        model = SimpleModel()
        agg = FLTrust()
        agg.aggregate(model, make_updates(model, n=3))
        assert agg.last_info["fallback"] == "fedavg_no_server_update"
        assert agg.check_assumptions(3).satisfied is False


class TestFLTrustEndToEnd:
    """FLTrust is the closest prior work; it has to be run, not cited.

    Under the IPM attack that destroys FedAvg, FLTrust must stay near its
    clean accuracy.  Any novelty claim for a server-side safety gate has to
    clear this bar, so the number belongs in the results table.
    """

    @staticmethod
    def _run(task, attack, aggregator):
        X, y, eval_fn = task
        parts = dirichlet_split(y.numpy(), N_CLIENTS, alpha=0.5, seed=1)
        datasets = [TensorDataset(X[p], y[p]) for p in parts]
        rng = np.random.default_rng(99)
        idx = rng.choice(len(y), 60, replace=False)
        root = TensorDataset(X[idx], y[idx])
        return run_fl_baseline(
            FLBaselineConfig(
                name="x", aggregator=aggregator, gate="none",
                attack=attack, n_rounds=12,
            ),
            zero_init_model(),
            datasets,
            eval_fn,
            lr=0.5,
            batch_size=16,
            seed=0,
            server_root_dataset=root,
        )

    def test_requires_a_server_root_dataset(self, task):
        X, y, eval_fn = task
        parts = dirichlet_split(y.numpy(), N_CLIENTS, alpha=0.5, seed=1)
        with pytest.raises(ValueError, match="server_root_dataset"):
            run_fl_baseline(
                FLBaselineConfig(name="x", aggregator="fltrust", n_rounds=1),
                zero_init_model(),
                [TensorDataset(X[p], y[p]) for p in parts],
                eval_fn,
            )

    def test_survives_an_attack_that_destroys_fedavg(self, task):
        attack = AttackConfig(3, "ipm", {"epsilon": 2.0})
        clean = self._run(task, None, "fltrust").final_metric["accuracy"]
        attacked = self._run(task, attack, "fltrust").final_metric["accuracy"]
        fedavg_attacked = self._run(task, attack, "fedavg").final_metric["accuracy"]
        assert clean - attacked < 0.10, (
            f"FLTrust clean={clean:.3f} attacked={attacked:.3f}"
        )
        assert attacked - fedavg_attacked > 0.30, (
            "the attack must actually break FedAvg for this comparison to "
            f"mean anything: fedavg={fedavg_attacked:.3f}"
        )


class TestRegistry:
    def test_all_registered_aggregators_build(self):
        for name in AGGREGATOR_REGISTRY:
            agg = build_aggregator(name, n_byzantine=1)
            assert agg.name == name

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown aggregator"):
            build_aggregator("nope")

    def test_irrelevant_kwargs_are_dropped(self):
        assert isinstance(build_aggregator("fedavg", n_byzantine=3), FedAvg)


# ---------------------------------------------------------------------------
# F7: zero-Byzantine parity, and what actually breaks Krum
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def task():
    (X, y), (Xte, yte) = make_task(seed=0)
    return X, y, accuracy_fn(Xte, yte)


class TestZeroByzantineParity:
    """With NO adversary, every aggregator must match FedAvg.

    If it does not, the experimental setup is broken, not the aggregator --
    unless the aggregator has a documented adversary-free cost, which is the
    case for Krum and only for Krum (see the tolerance table below).
    """

    #: Per-aggregator adversary-free tolerance against FedAvg.
    #:
    #: The averaging rules (median, trimmed mean, multi-Krum) use every
    #: client's data every round and should be indistinguishable from FedAvg.
    #: Single-selection Krum commits ONE client's update per round and so
    #: throws away (n-1)/n of the data; under a genuinely non-IID split that
    #: costs real accuracy with zero adversaries present.  Measured at ~0.07
    #: here; see ``test_krum_pays_a_documented_no_adversary_penalty``, which
    #: reports the number rather than hiding it inside a loose tolerance.
    TOL = {
        "fedavg": 0.001,
        "median": 0.03,
        "trimmed_mean": 0.03,
        "multi_krum": 0.03,
        "krum": 0.10,
    }

    @pytest.mark.parametrize(
        "agg_name", ["fedavg", "median", "trimmed_mean", "krum", "multi_krum"]
    )
    def test_matches_fedavg_on_a_sane_non_iid_split(self, task, agg_name):
        X, y, eval_fn = task
        parts = dirichlet_split(y.numpy(), N_CLIENTS, alpha=0.5, seed=1)
        base, _ = run_fl(parts, X, y, eval_fn, "fedavg")
        acc, server = run_fl(parts, X, y, eval_fn, agg_name)
        assert server.history[-1].aggregator_assumption["satisfied"] is True
        assert base - acc < self.TOL[agg_name], (
            f"{agg_name} lost {base - acc:.3f} accuracy against FedAvg with "
            f"ZERO adversaries (FedAvg={base:.3f}, {agg_name}={acc:.3f})"
        )

    def test_krum_pays_a_documented_no_adversary_penalty(self, task):
        """Krum's adversary-free cost is real and must be the baseline.

        Any "Krum degrades under attack" claim has to be measured against
        Krum's own zero-adversary accuracy, not against FedAvg's.
        """
        X, y, eval_fn = task
        parts = dirichlet_split(y.numpy(), N_CLIENTS, alpha=0.5, seed=1)
        fedavg, _ = run_fl(parts, X, y, eval_fn, "fedavg")
        krum, _ = run_fl(parts, X, y, eval_fn, "krum")
        multi, _ = run_fl(parts, X, y, eval_fn, "multi_krum")
        penalty = fedavg - krum
        assert 0.0 < penalty < 0.10, f"unexpected Krum penalty {penalty:.3f}"
        # Averaging the m best scores removes the penalty entirely.
        assert fedavg - multi < 0.03

    def test_label_sorted_split_collapses_krum_with_no_adversary(self, task):
        """Finding F7, exhibited.

        Sorting by label and chunking gives single-class clients.  Krum then
        returns one client's update, i.e. trains the global model on one class.
        The collapse is caused by the split; the same split leaves FedAvg
        untouched, and the same Krum is fine on a Dirichlet split.
        """
        X, y, eval_fn = task
        sorted_parts = label_sorted_split(y.numpy(), N_CLIENTS)
        dirichlet_parts = dirichlet_split(y.numpy(), N_CLIENTS, alpha=0.5, seed=1)

        fedavg_sorted, _ = run_fl(sorted_parts, X, y, eval_fn, "fedavg")
        krum_sorted, _ = run_fl(sorted_parts, X, y, eval_fn, "krum")
        krum_dirichlet, _ = run_fl(dirichlet_parts, X, y, eval_fn, "krum")

        assert fedavg_sorted - krum_sorted > 0.20, (
            "expected Krum to collapse on the single-class split; got "
            f"FedAvg={fedavg_sorted:.3f}, Krum={krum_sorted:.3f}"
        )
        assert krum_dirichlet - krum_sorted > 0.15, (
            "the collapse must be attributable to the split, not to Krum: "
            f"Krum(dirichlet)={krum_dirichlet:.3f}, Krum(sorted)={krum_sorted:.3f}"
        )


# ---------------------------------------------------------------------------
# F5: rollback semantics
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup():
    (X, y), (Xte, yte) = make_task(seed=0)
    parts = dirichlet_split(y.numpy(), 4, alpha=0.5, seed=1)
    datasets = [TensorDataset(X[p], y[p]) for p in parts]
    clients = [
        BenignClient(i, ds, lr=0.5, local_epochs=1, batch_size=16, seed=0)
        for i, ds in enumerate(datasets)
    ]
    return clients, accuracy_fn(Xte, yte), datasets


class TestRollbackSemantics:
    def test_rejected_round_changes_nothing(self, setup):
        clients, eval_fn, _ = setup
        server = FederatedServer(
            zero_init_model(), gate=AlwaysRejectGate(), eval_fn=eval_fn
        )
        server.run(clients, 4)
        server.assert_rejections_were_noops()
        assert server.rounds_accepted == 0
        assert server.model_is_unchanged()
        for r in server.history:
            assert r.hash_before == r.hash_after
            assert r.rollback_was_noop
            # The candidate really was different -- rejection is doing work.
            assert r.hash_candidate != r.hash_before

    def test_candidate_metric_is_recorded_for_every_round(self, setup):
        """The 'would-be value if this round had been accepted' hook (F5)."""
        clients, eval_fn, _ = setup
        server = FederatedServer(
            zero_init_model(), gate=AlwaysRejectGate(), eval_fn=eval_fn
        )
        server.run(clients, 3)
        for r in server.history:
            assert r.candidate_metric is not None
            assert r.committed_metric is not None
        summary = server.summary()
        assert summary["model_is_frozen_pretrained"] is True
        assert summary["would_be_metric_if_last_round_accepted"] is not None

    def test_accepted_rounds_do_change_the_model(self, setup):
        clients, eval_fn, _ = setup
        server = FederatedServer(
            zero_init_model(), gate=AlwaysAcceptGate(), eval_fn=eval_fn
        )
        server.run(clients, 3)
        assert server.rounds_accepted == 3
        assert not server.model_is_unchanged()
        assert server.get_acceptance_rate() == 1.0

    def test_mixed_gate_bookkeeping(self, setup):
        clients, eval_fn, _ = setup
        gate = FunctionGate(lambda m, t: GateDecision(accepted=(t % 2 == 1)), "odd")
        server = FederatedServer(zero_init_model(), gate=gate, eval_fn=eval_fn)
        server.run(clients, 4)
        assert server.rounds_accepted == 2
        assert server.get_acceptance_rate() == 0.5
        server.assert_rejections_were_noops()

    def test_the_noop_check_has_teeth(self):
        """A rejected round that changed the model MUST be detected."""
        server = FederatedServer(zero_init_model())
        server.history.append(
            RoundResult(
                round_num=1,
                accepted=False,
                n_clients=1,
                n_byzantine=0,
                aggregated_delta_norm=0.0,
                hash_before="aaaa",
                hash_candidate="bbbb",
                hash_after="cccc",
                rollback_was_noop=False,
            )
        )
        with pytest.raises(AssertionError, match="Rollback is not a no-op"):
            server.assert_rejections_were_noops()

    def test_no_gate_and_always_accept_are_identical(self, setup):
        clients, eval_fn, datasets = setup

        def fresh():
            return [
                BenignClient(i, ds, lr=0.5, local_epochs=1, batch_size=16, seed=0)
                for i, ds in enumerate(datasets)
            ]

        a = FederatedServer(zero_init_model(), gate=None, eval_fn=eval_fn)
        b = FederatedServer(zero_init_model(), gate=AlwaysAcceptGate(), eval_fn=eval_fn)
        a.run(fresh(), 3)
        b.run(fresh(), 3)
        assert model_state_hash(a.global_model) == model_state_hash(b.global_model)


# ---------------------------------------------------------------------------
# The mandatory FL control suite
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def suite():
    (X, y), (Xte, yte) = make_task(seed=0)
    parts = dirichlet_split(y.numpy(), N_CLIENTS, alpha=0.5, seed=1)
    datasets = [TensorDataset(X[p], y[p]) for p in parts]
    return run_gate_control_suite(
        zero_init_model(),
        datasets,
        accuracy_fn(Xte, yte),
        attack=AttackConfig(2, "ipm", {"epsilon": 2.0}),
        n_rounds=6,
        lr=0.5,
        batch_size=16,
        seed=0,
    )


class TestGateControlSuite:
    def test_identities_hold(self, suite):
        assert suite["identity_always_accept_equals_no_gate"]
        assert suite["identity_always_reject_equals_frozen"]

    def test_always_reject_is_the_frozen_checkpoint(self, suite):
        assert (
            suite["always_reject"]["final_hash"]
            == suite["frozen_pretrained"]["final_hash"]
        )
        assert suite["always_reject"]["summary"]["rounds_accepted"] == 0

    def test_c1_is_reported(self, suite):
        c1 = suite["C1"]
        assert c1["criterion"] == "C1"
        assert c1["clean_fedavg"] is not None
        # In this harness federated training helps, so C1 passes here.
        assert c1["passed"] is True, (
            "clean FedAvg failed to beat the frozen checkpoint: "
            f"{c1['clean_fedavg']} vs {c1['frozen_pretrained']}"
        )

    def test_determinism_across_processes_with_different_hash_seeds(self):
        """Acceptance criterion C4, for the federated subsystem.

        Two fresh interpreters with DIFFERENT ``PYTHONHASHSEED`` values must
        produce byte-identical model hashes.  Finding F9 was caused by
        ``embedding.py`` calling the builtin ``hash()`` on strings; nothing in
        the FL stack may do that, and this test is what keeps it that way.
        """
        import os
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import numpy as np, torch
            from torch.utils.data import TensorDataset
            from tests.test_aggregation import (
                make_task, dirichlet_split, accuracy_fn, zero_init_model,
            )
            from sca.experiments.attacks import AttackConfig
            from sca.experiments.baselines import FLBaselineConfig, run_fl_baseline

            (X, y), (Xte, yte) = make_task(0)
            parts = dirichlet_split(y.numpy(), 4, 0.5, 1)
            ds = [TensorDataset(X[p], y[p]) for p in parts]
            out = []
            for atk in [None, AttackConfig(1, "sign_flip"),
                        AttackConfig(1, "targeted_safety")]:
                r = run_fl_baseline(
                    FLBaselineConfig("x", "fedavg", gate="none", attack=atk,
                                     n_rounds=3),
                    zero_init_model(), ds, accuracy_fn(Xte, yte),
                    lr=0.5, batch_size=16, seed=0,
                )
                out.append(r.final_hash)
            print(";".join(out))
            """
        )
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        outputs = []
        for hash_seed in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=repo_root)
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, env=env, cwd=repo_root,
                timeout=600,
            )
            assert proc.returncode == 0, proc.stderr[-2000:]
            outputs.append(proc.stdout.strip().splitlines()[-1])
        assert outputs[0] == outputs[1], (
            f"results depend on PYTHONHASHSEED: {outputs[0]} != {outputs[1]}"
        )

    def test_assertion_helpers_fail_when_they_should(self, suite):
        from sca.experiments.baselines import FLRunResult

        frozen = FLRunResult(**{k: v for k, v in suite["frozen_pretrained"].items()
                                if k != "rounds"}, rounds=[])
        tampered = FLRunResult(
            name="tampered",
            summary={"rounds_accepted": 0, "n_rounds": 3},
            rounds=[],
            final_metric=frozen.final_metric,
            final_hash="deadbeef",
            initial_hash=frozen.initial_hash,
        )
        with pytest.raises(AssertionError):
            assert_always_reject_equals_frozen(tampered, frozen)
        with pytest.raises(AssertionError):
            assert_always_accept_equals_no_gate(tampered, frozen)
