"""Tests for the rebuilt Byzantine attacks (audit finding F6).

The old "attacks" were fresh Gaussian noise: independent of the honest
updates, mutually orthogonal across colluding attackers, and net-beneficial to
accuracy.  The tests here are written so that the old implementation would
fail every one of them:

* :class:`TestDependsOnBenignDeltas` -- an attack must change when the honest
  updates change.
* :class:`TestCoordination` -- colluding attackers must be positively
  correlated; the legacy control's near-zero correlation is asserted too, so
  the F6 signature stays visible in the results.
* :class:`TestEndToEndDamage` -- the real attacks must damage the model more
  than the legacy control does.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from sca.experiments.attacks import (
    ALIEAttack,
    ATTACK_REGISTRY,
    REAL_ATTACKS,
    AttackConfig,
    AttackContext,
    IPMAttack,
    LegacyGaussianNoiseAttack,
    NoAttack,
    SignFlipAttack,
    TargetedSafetyDegradationAttack,
    attack_diagnostics,
    build_attack,
    cosine,
    create_attack_scenario,
    create_byzantine_clients,
    create_standard_attack_scenarios,
    mean_delta,
    project_into_norm_envelope,
    std_delta,
)
from sca.experiments.baselines import FLBaselineConfig, run_fl_baseline
from sca.federated.client import (
    ByzantineClient,
    MissingBenignDeltasError,
    delta_norm,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 10
N_CLIENTS = 8
UNSAFE = 1


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.linear(x)


def deltas(values, key="w", size=6):
    """Build parameter dicts from a list of scalars (constant tensors)."""
    return [{key: torch.full((size,), float(v))} for v in values]


def random_deltas(n, seed=0, size=6, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return [{"w": torch.randn(size, generator=g) * scale} for _ in range(n)]


def ctx(**kwargs):
    return AttackContext(**kwargs)


# -- an end-to-end binary safety task ---------------------------------------


def make_safety_task(seed: int = 0, n_train: int = 480, n_test: int = 240):
    """Binary task: class 0 = safe, class 1 = unsafe."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(2, DIM)) * 0.6

    def sample(n):
        y = rng.integers(0, 2, size=n)
        X = centers[y] + rng.normal(size=(n, DIM))
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )

    return sample(n_train), sample(n_test)


def dirichlet_split(y: np.ndarray, n_clients: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    pools = {c: list(rng.permutation(np.where(y == c)[0])) for c in range(2)}
    per_client = len(y) // n_clients
    parts = []
    for _ in range(n_clients):
        p = rng.dirichlet([alpha] * 2)
        chosen = []
        for c in rng.choice(2, size=per_client, p=p):
            for cc in [c, 1 - c]:
                if pools[cc]:
                    chosen.append(pools[cc].pop())
                    break
        parts.append(np.array(sorted(chosen)))
    return parts


@pytest.fixture(scope="module")
def safety_fl():
    (X, y), (Xte, yte) = make_safety_task()
    parts = dirichlet_split(y.numpy(), N_CLIENTS, alpha=0.5, seed=1)
    datasets = [TensorDataset(X[p], y[p]) for p in parts]

    torch.manual_seed(0)
    model = nn.Linear(DIM, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    def eval_fn(m: nn.Module):
        m.eval()
        with torch.no_grad():
            pred = m(Xte).argmax(dim=1)
            return {
                "accuracy": float((pred == yte).float().mean()),
                "recall_unsafe": float((pred[yte == UNSAFE] == UNSAFE).float().mean()),
                "recall_safe": float((pred[yte == 0] == 0).float().mean()),
            }

    return model, datasets, eval_fn


def run_attack(safety_fl, attack_cfg, n_rounds=15, aggregator="fedavg"):
    model, datasets, eval_fn = safety_fl
    return run_fl_baseline(
        FLBaselineConfig(
            name="t", aggregator=aggregator, gate="none",
            attack=attack_cfg, n_rounds=n_rounds,
        ),
        model,
        datasets,
        eval_fn,
        lr=0.5,
        batch_size=16,
        seed=0,
    )


# ---------------------------------------------------------------------------
# The API rule
# ---------------------------------------------------------------------------


class TestBenignDeltasAreRequired:
    def test_byzantine_client_refuses_without_benign_deltas(self):
        client = ByzantineClient(client_id=0, attack=IPMAttack())
        with pytest.raises(MissingBenignDeltasError):
            client.compute_update(SimpleModel(), round_num=1)

    def test_byzantine_client_refuses_with_empty_benign_deltas(self):
        client = ByzantineClient(client_id=0, attack=IPMAttack())
        with pytest.raises(MissingBenignDeltasError):
            client.compute_update(SimpleModel(), 1, benign_deltas=[])

    def test_attacks_that_train_require_a_dataset(self):
        client = ByzantineClient(client_id=0, attack=SignFlipAttack())
        with pytest.raises(ValueError, match="honest update"):
            client.compute_update(
                SimpleModel(), 1, benign_deltas=random_deltas(2)
            )

    def test_create_byzantine_clients_requires_datasets_for_training_attacks(self):
        with pytest.raises(ValueError, match="requires datasets"):
            create_byzantine_clients(AttackConfig(2, "sign_flip"))


class TestDependsOnBenignDeltas:
    """The single test the old implementation could never have passed."""

    @pytest.mark.parametrize("name", REAL_ATTACKS)
    def test_real_attacks_change_with_the_honest_updates(self, name):
        attack = build_attack(name)
        honest = {"w": torch.full((6,), 3.0)}
        poison = {"w": torch.full((6,), -2.0)}
        c = ctx(honest_delta=honest, poisoned_delta=poison, n_attackers=2)

        a = attack.craft(deltas([1.0, 1.0, 1.0]), c)
        a_again = attack.craft(deltas([1.0, 1.0, 1.0]), c)
        b = attack.craft(deltas([-4.0, 0.5, 2.0]), c)

        # Determinism first: without it, "the output changed" would also be
        # satisfied by a fresh noise generator, which is exactly the defect
        # (F6) this test exists to rule out.
        torch.testing.assert_close(a["w"], a_again["w"])
        assert not torch.allclose(a["w"], b["w"]), (
            f"{name} produced the same update for different honest updates; "
            "it is not a function of benign_deltas"
        )

    def test_legacy_control_does_not_depend_on_them(self):
        """Finding F6, preserved as an explicit, labelled control."""
        attack = LegacyGaussianNoiseAttack()
        assert attack.depends_on_benign_deltas is False
        assert attack.is_degenerate_control is True
        c = ctx(seed=0, client_id=0, round_num=1)
        a = attack.craft(deltas([1.0, 1.0, 1.0]), c)
        b = attack.craft(deltas([-4.0, 0.5, 2.0]), c)
        torch.testing.assert_close(a["w"], b["w"])

    def test_every_registered_attack_declares_its_dependence(self):
        for name, cls in ATTACK_REGISTRY.items():
            assert isinstance(cls.depends_on_benign_deltas, bool)
            assert (name in REAL_ATTACKS) == (not cls.is_degenerate_control)


# ---------------------------------------------------------------------------
# Individual attack semantics
# ---------------------------------------------------------------------------


class TestSignFlip:
    def test_is_exactly_the_negated_honest_update(self):
        honest = {"w": torch.tensor([1.0, -2.0, 3.0, 0.0, 1.0, 1.0])}
        attack = SignFlipAttack(scale=1.0, norm_match=False)
        out = attack.craft(random_deltas(3, seed=1), ctx(honest_delta=honest))
        torch.testing.assert_close(out["w"], -honest["w"])
        assert cosine(out, honest) == pytest.approx(-1.0, abs=1e-6)

    def test_is_not_noise(self):
        """Deterministic in its inputs, and different for different inputs."""
        attack = SignFlipAttack(norm_match=False)
        benign = random_deltas(3, seed=1)
        a = attack.craft(benign, ctx(honest_delta={"w": torch.ones(6)}))
        a2 = attack.craft(benign, ctx(honest_delta={"w": torch.ones(6)}))
        b = attack.craft(benign, ctx(honest_delta={"w": torch.ones(6) * 2}))
        torch.testing.assert_close(a["w"], a2["w"])
        assert not torch.allclose(a["w"], b["w"])

    def test_norm_matching_respects_the_benign_envelope(self):
        honest = {"w": torch.full((6,), 100.0)}
        benign = deltas([1.0, 1.5, 0.5])
        out = SignFlipAttack(norm_match=True).craft(benign, ctx(honest_delta=honest))
        assert delta_norm(out) <= max(delta_norm(d) for d in benign) * (1 + 1e-6)
        # Direction is preserved.
        assert cosine(out, honest) == pytest.approx(-1.0, abs=1e-6)

    def test_shared_direction_variant_needs_no_dataset(self):
        attack = SignFlipAttack(share_direction=True, norm_match=False)
        assert attack.requires_honest_delta is False
        benign = deltas([1.0, 3.0])
        out = attack.craft(benign, ctx())
        torch.testing.assert_close(out["w"], -mean_delta(benign)["w"])


class TestIPM:
    def test_matches_the_paper_formula(self):
        benign = deltas([1.0, 2.0, 3.0])  # mean = 2
        out = IPMAttack(epsilon=0.5).craft(benign, ctx())
        torch.testing.assert_close(out["w"], torch.full((6,), -1.0))

    def test_flips_the_fedavg_mean_when_epsilon_exceeds_the_threshold(self):
        """With f attackers among n, FedAvg flips sign iff eps > (n-f)/f."""
        n, f = 8, 3
        benign = deltas([1.0] * (n - f))
        eps = 2.0  # > 5/3
        attack_delta = IPMAttack(epsilon=eps).craft(benign, ctx(n_attackers=f))
        agg = (
            sum(d["w"] for d in benign) + f * attack_delta["w"]
        ) / n
        assert (agg < 0).all()

    def test_all_attackers_send_the_same_vector(self):
        benign = random_deltas(4, seed=2)
        a = IPMAttack().craft(benign, ctx(attacker_index=0, n_attackers=3))
        b = IPMAttack().craft(benign, ctx(attacker_index=2, n_attackers=3))
        torch.testing.assert_close(a["w"], b["w"])
        assert cosine(a, b) == pytest.approx(1.0, abs=1e-6)


class TestALIE:
    def test_matches_the_paper_formula(self):
        benign = deltas([1.0, 2.0, 3.0])
        mu = mean_delta(benign)["w"]
        sigma = std_delta(benign)["w"]
        out = ALIEAttack(z=1.5).craft(benign, ctx())
        torch.testing.assert_close(out["w"], mu - 1.5 * sigma)

    def test_stays_within_the_honest_spread(self):
        benign = random_deltas(6, seed=3)
        out = ALIEAttack(z=1.0).craft(benign, ctx())
        mu = mean_delta(benign)["w"]
        sigma = std_delta(benign)["w"]
        assert torch.all((out["w"] - mu).abs() <= sigma + 1e-6)

    def test_baruch_z_max_rule(self):
        z = ALIEAttack.baruch_z_max(n=50, f=12)
        assert 0.0 < z < 1.0
        # Degenerate configurations must give 0, not NaN or a crash.
        assert ALIEAttack.baruch_z_max(n=4, f=4) == 0.0

    def test_z_none_uses_the_paper_rule(self):
        attack = ALIEAttack(z=None, n_clients=50)
        assert attack.effective_z(38, 12) == pytest.approx(
            ALIEAttack.baruch_z_max(50, 12)
        )


class TestTargetedSafetyDegradation:
    def test_label_map_only_touches_the_unsafe_class(self):
        attack = TargetedSafetyDegradationAttack(unsafe_label=1, safe_label=0)
        targets = torch.tensor([0, 1, 0, 1, 1])
        mapped = attack.default_label_map(targets)
        torch.testing.assert_close(mapped, torch.tensor([0, 0, 0, 0, 0]))

    def test_uses_both_honest_and_poisoned_updates(self):
        attack = TargetedSafetyDegradationAttack(boost=2.0, enforce_envelope=False)
        honest = {"w": torch.ones(6)}
        poison = {"w": torch.full((6,), 4.0)}
        out = attack.craft(deltas([1.0]), ctx(honest_delta=honest, poisoned_delta=poison))
        # honest + 2*(poison - honest) = 1 + 2*3 = 7
        torch.testing.assert_close(out["w"], torch.full((6,), 7.0))

    def test_missing_poison_update_raises(self):
        attack = TargetedSafetyDegradationAttack()
        with pytest.raises(ValueError, match="honest and the poisoned"):
            attack.craft(deltas([1.0]), ctx(honest_delta={"w": torch.ones(6)}))

    def test_envelope_enforcement_is_real(self):
        attack = TargetedSafetyDegradationAttack(boost=10.0, enforce_envelope=True)
        benign = deltas([1.0, 1.2, 0.9])
        out = attack.craft(
            benign,
            ctx(
                honest_delta={"w": torch.ones(6)},
                poisoned_delta={"w": torch.full((6,), 20.0)},
            ),
        )
        cap = max(delta_norm(d) for d in benign)
        assert delta_norm(out) <= cap * (1 + 1e-6)
        diag = attack_diagnostics(benign, [out])
        assert diag.is_stealthy


class TestControls:
    def test_no_attack_returns_the_honest_update(self):
        honest = {"w": torch.arange(6, dtype=torch.float32)}
        out = NoAttack().craft(deltas([1.0]), ctx(honest_delta=honest))
        torch.testing.assert_close(out["w"], honest["w"])

    def test_legacy_control_is_reproducible(self):
        a = LegacyGaussianNoiseAttack().craft(
            deltas([1.0]), ctx(seed=7, client_id=3, round_num=2)
        )
        b = LegacyGaussianNoiseAttack().craft(
            deltas([1.0]), ctx(seed=7, client_id=3, round_num=2)
        )
        torch.testing.assert_close(a["w"], b["w"])


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestCoordination:
    """Coordinated attackers must be POSITIVELY correlated (finding F6)."""

    def _attacker_deltas(self, attack, benign, f=3, honest=None, poison=None):
        return [
            attack.craft(
                benign,
                ctx(
                    honest_delta=honest,
                    poisoned_delta=poison,
                    attacker_index=k,
                    n_attackers=f,
                    client_id=100 + k,
                    seed=0,
                ),
            )
            for k in range(f)
        ]

    @pytest.mark.parametrize(
        "attack", [IPMAttack(epsilon=1.0), ALIEAttack(z=1.5),
                   SignFlipAttack(share_direction=True)]
    )
    def test_coordinated_attacks_have_high_pairwise_cosine(self, attack):
        benign = random_deltas(5, seed=4)
        att = self._attacker_deltas(attack, benign)
        diag = attack_diagnostics(benign, att)
        assert diag.mean_pairwise_cosine_attackers > 0.9
        assert diag.is_coordinated

    def test_legacy_control_attackers_are_uncorrelated(self):
        """The F6 signature: independent noise, cosine ~= 0."""
        benign = random_deltas(5, seed=4, size=400)
        att = self._attacker_deltas(LegacyGaussianNoiseAttack(), benign)
        diag = attack_diagnostics(benign, att)
        assert abs(diag.mean_pairwise_cosine_attackers) < 0.2
        assert not diag.is_coordinated

    def test_direction_attacks_oppose_the_honest_mean(self):
        benign = random_deltas(5, seed=5)
        att = self._attacker_deltas(IPMAttack(epsilon=1.0), benign)
        diag = attack_diagnostics(benign, att)
        assert diag.cosine_attacker_mean_vs_benign_mean < -0.9


class TestDiagnosticsFields:
    def test_norm_ratio_and_envelope(self):
        benign = deltas([1.0, 1.0, 1.0])
        loud = deltas([10.0, 10.0])
        diag = attack_diagnostics(benign, loud)
        assert diag.norm_ratio == pytest.approx(10.0, rel=1e-5)
        assert diag.frac_within_benign_envelope == 0.0
        assert not diag.is_stealthy

    def test_projection_makes_the_attack_stealthy(self):
        benign = random_deltas(5, seed=6)
        loud = [{"w": d["w"] * 50} for d in random_deltas(3, seed=7)]
        projected = [project_into_norm_envelope(d, benign) for d in loud]
        assert attack_diagnostics(benign, loud).frac_within_benign_envelope == 0.0
        assert attack_diagnostics(benign, projected).is_stealthy

    def test_percentile_is_reported(self):
        benign = deltas([1.0, 2.0, 3.0, 4.0])
        diag = attack_diagnostics(benign, deltas([2.5]))
        assert 0.0 <= diag.attacker_norm_percentile_in_benign <= 1.0

    def test_single_attacker_gives_nan_not_a_crash(self):
        diag = attack_diagnostics(deltas([1.0, 2.0]), deltas([1.0]))
        assert np.isnan(diag.mean_pairwise_cosine_attackers)
        assert not diag.is_coordinated


# ---------------------------------------------------------------------------
# End to end: the attacks must actually hurt
# ---------------------------------------------------------------------------


class TestEndToEndDamage:
    def test_real_attacks_hurt_more_than_the_legacy_control(self, safety_fl):
        """F6 in one assertion.

        The old headline was produced against ``legacy_gaussian``.  A real
        direction attack has to do materially more damage than that, or the
        threat model is decorative.
        """
        clean_acc = run_attack(safety_fl, None).final_metric["accuracy"]
        legacy_acc = run_attack(
            safety_fl, AttackConfig(3, "legacy_gaussian")
        ).final_metric["accuracy"]
        signflip_acc = run_attack(
            safety_fl, AttackConfig(3, "sign_flip", {"scale": 1.0})
        ).final_metric["accuracy"]
        ipm_acc = run_attack(
            safety_fl, AttackConfig(3, "ipm", {"epsilon": 2.0})
        ).final_metric["accuracy"]

        legacy_damage = clean_acc - legacy_acc
        assert clean_acc - signflip_acc > 1.5 * legacy_damage, (
            f"clean={clean_acc:.3f} legacy={legacy_acc:.3f} "
            f"sign_flip={signflip_acc:.3f}"
        )
        assert clean_acc - ipm_acc > 1.5 * legacy_damage, (
            f"clean={clean_acc:.3f} legacy={legacy_acc:.3f} ipm={ipm_acc:.3f}"
        )

    def test_targeted_attack_is_selective_for_the_unsafe_class(self, safety_fl):
        """The point of a *safety* threat model.

        The attack must raise the unsafe-class error much more than it raises
        the safe-class error; an attack that just breaks the model is not a
        safety attack and would be caught by ordinary accuracy monitoring.
        """
        clean = run_attack(safety_fl, None).final_metric
        attacked = run_attack(
            safety_fl,
            AttackConfig(
                3, "targeted_safety", {"boost": 4.0, "enforce_envelope": False}
            ),
        ).final_metric

        unsafe_damage = clean["recall_unsafe"] - attacked["recall_unsafe"]
        safe_damage = clean["recall_safe"] - attacked["recall_safe"]
        assert unsafe_damage > 0.3, f"unsafe recall barely moved: {attacked}"
        assert unsafe_damage > 3 * max(safe_damage, 0.0) + 0.2, (
            f"attack was not selective: clean={clean}, attacked={attacked}"
        )

    def test_stealth_costs_the_attacker_effectiveness(self, safety_fl):
        """An honest negative-ish result, reported rather than hidden.

        Constraining the targeted attack to the benign norm envelope makes it
        undetectable by norm screening AND much weaker at f = 3/8.  Both halves
        of that trade-off belong in the paper.
        """
        loud = run_attack(
            safety_fl,
            AttackConfig(3, "targeted_safety",
                         {"boost": 4.0, "enforce_envelope": False}),
        )
        stealthy = run_attack(
            safety_fl,
            AttackConfig(3, "targeted_safety",
                         {"boost": 4.0, "enforce_envelope": True}),
        )
        assert (
            loud.final_metric["recall_unsafe"]
            < stealthy.final_metric["recall_unsafe"]
        )
        assert (
            stealthy.mean_attack_diagnostics["norm_ratio"]
            < loud.mean_attack_diagnostics["norm_ratio"]
        )

    def test_diagnostics_are_recorded_in_every_attacked_run(self, safety_fl):
        res = run_attack(safety_fl, AttackConfig(3, "ipm", {"epsilon": 1.0}),
                         n_rounds=3)
        diag = res.mean_attack_diagnostics
        assert diag is not None
        assert diag["mean_pairwise_cosine_attackers"] > 0.9
        assert diag["cosine_attacker_mean_vs_benign_mean"] < -0.9
        assert "norm_ratio" in diag


# ---------------------------------------------------------------------------
# Scenario plumbing
# ---------------------------------------------------------------------------


class TestScenarios:
    def test_standard_scenarios_include_the_controls(self):
        names = {s.name for s in create_standard_attack_scenarios(10)}
        assert {"none", "sign_flip", "ipm", "alie", "targeted_safety"} <= names
        assert "legacy_gaussian_CONTROL" in names
        assert "no_attack_CONTROL" in names

    def test_no_attack_scenario_has_zero_byzantine(self):
        s = [x for x in create_standard_attack_scenarios(10) if x.name == "none"][0]
        assert s.config.n_byzantine == 0
        assert s.config.build() is None

    def test_create_attack_scenario_uses_the_last_client_ids(self):
        clients = create_attack_scenario(
            10, AttackConfig(2, "ipm", {"epsilon": 0.5})
        )
        assert [c.client_id for c in clients] == [8, 9]
        assert all(isinstance(c, ByzantineClient) for c in clients)

    def test_build_attack_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="unknown attack"):
            build_attack("definitely_not_an_attack")
