"""FL-arm baselines and the controls that make a Byzantine claim readable.

MANDATORY CONTROLS (the absence of these is what made the old results
fiction):

===========================  =================================================
``frozen_pretrained``        No federated training at all.  The checkpoint,
                             evaluated on held-out test data.
``clean_fedavg_no_gate``     FedAvg, zero attackers, no gate.  Acceptance
                             criterion C1: this MUST exceed
                             ``frozen_pretrained``.  If federated training is
                             net-harmful then "reject everything" is
                             accuracy-maximising by construction and every
                             Byzantine number is an artefact.
``always_reject``            Gate that rejects every round.  MUST be
                             numerically identical to ``frozen_pretrained``.
``always_accept``            Gate that accepts every round.  MUST be
                             numerically identical to the same run with no
                             gate.
===========================  =================================================

The last two identities are asserted in code by
:func:`assert_always_reject_equals_frozen` and
:func:`assert_always_accept_equals_no_gate`.  They are cheap, and their
absence is exactly how the old repo shipped "93.75 % under 50 % Byzantine
clients" when the real content of the number was "the frozen checkpoint".

This module owns only the FEDERATED arm.  The verifier arm (uniform /
proportional / w^(2/3) / search-guided allocation, plus the null and identity
mutator controls) belongs to the verifier package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch.nn as nn

from sca.experiments.attacks import (
    AttackConfig,
    attack_diagnostics,
    create_byzantine_clients,
)
from sca.federated.aggregation import (
    Aggregator,
    FLTrust,
    build_aggregator,
    fltrust_server_update,
)
from sca.federated.client import BenignClient, FLClient
from sca.federated.server import (
    AlwaysAcceptGate,
    AlwaysRejectGate,
    FederatedServer,
    GateDecision,
    RoundResult,
    SafetyGate,
    model_state_hash,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FLBaselineConfig",
    "FLRunResult",
    "build_clients",
    "run_fl_baseline",
    "frozen_pretrained_baseline",
    "create_fl_arm_configs",
    "run_gate_control_suite",
    "assert_always_reject_equals_frozen",
    "assert_always_accept_equals_no_gate",
    "check_c1_fedavg_beats_frozen",
    "StaticSuiteGate",
    "GATE_FACTORIES",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


GATE_FACTORIES: dict[str, Callable[[], SafetyGate | None]] = {
    "none": lambda: None,
    "always_accept": AlwaysAcceptGate,
    "always_reject": AlwaysRejectGate,
}


@dataclass
class FLBaselineConfig:
    """One cell of the FL results table.

    Attributes:
        name: Row label.
        aggregator: Registry key, e.g. ``"fedavg"`` / ``"krum"`` / ``"fltrust"``.
        aggregator_kwargs: Extra constructor arguments (``n_byzantine=...``).
        gate: Key into :data:`GATE_FACTORIES`, or a ready
            :class:`~sca.federated.server.SafetyGate`.
        attack: Attack configuration; ``None`` means no adversaries.
        n_rounds: FL rounds.
        description: Free text for the results file.
    """

    name: str
    aggregator: str = "fedavg"
    aggregator_kwargs: dict[str, Any] = field(default_factory=dict)
    gate: Any = "none"
    attack: AttackConfig | None = None
    n_rounds: int = 10
    description: str = ""

    @property
    def n_byzantine(self) -> int:
        return 0 if self.attack is None else int(self.attack.n_byzantine)

    def build_gate(self) -> SafetyGate | None:
        if isinstance(self.gate, str):
            return GATE_FACTORIES[self.gate]()
        return self.gate

    def build_aggregator(self) -> Aggregator:
        kwargs = dict(self.aggregator_kwargs)
        kwargs.setdefault("n_byzantine", self.n_byzantine)
        return build_aggregator(self.aggregator, **kwargs)


@dataclass
class FLRunResult:
    """Result of one FL configuration."""

    name: str
    summary: dict[str, Any]
    rounds: list[dict[str, Any]]
    final_metric: Any
    final_hash: str
    initial_hash: str
    mean_attack_diagnostics: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "final_metric": self.final_metric,
            "final_hash": self.final_hash,
            "initial_hash": self.initial_hash,
            "mean_attack_diagnostics": self.mean_attack_diagnostics,
            "rounds": self.rounds,
        }


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


def build_clients(
    client_datasets: Sequence[Any],
    attack: AttackConfig | None,
    *,
    lr: float = 0.05,
    local_epochs: int = 1,
    batch_size: int = 32,
    seed: int = 0,
) -> list[FLClient]:
    """Build ``n`` clients, the last ``f`` of which are Byzantine.

    Adversaries keep their own local data, because the real attacks need it:
    ``sign_flip`` negates the attacker's own honest gradient and
    ``targeted_safety`` trains on relabelled local data.
    """
    n = len(client_datasets)
    f = 0 if attack is None else int(attack.n_byzantine)
    if f > n:
        raise ValueError(f"n_byzantine={f} exceeds n_clients={n}")

    clients: list[FLClient] = [
        BenignClient(
            client_id=i,
            dataset=client_datasets[i],
            lr=lr,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
        )
        for i in range(n - f)
    ]
    if f > 0 and attack is not None:
        clients.extend(
            create_byzantine_clients(
                attack,
                datasets=list(client_datasets[n - f :]),
                client_ids=list(range(n - f, n)),
                lr=lr,
                local_epochs=local_epochs,
                batch_size=batch_size,
                seed=seed,
            )
        )
    return clients


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def frozen_pretrained_baseline(
    initial_model: nn.Module,
    eval_fn: Callable[[nn.Module], Any],
    name: str = "frozen_pretrained",
) -> FLRunResult:
    """The no-FL row: evaluate the checkpoint and change nothing.

    Every gated row that rejects all rounds must reproduce this exactly.
    """
    h = model_state_hash(initial_model)
    metric = eval_fn(initial_model)
    return FLRunResult(
        name=name,
        summary={
            "aggregator": "none",
            "gate": "none",
            "n_rounds": 0,
            "rounds_accepted": 0,
            "rounds_rejected": 0,
            "acceptance_rate": 0.0,
            "initial_hash": h,
            "final_hash": h,
            "model_is_frozen_pretrained": True,
        },
        rounds=[],
        final_metric=metric,
        final_hash=h,
        initial_hash=h,
    )


def _mean_diagnostics(rounds: Sequence[RoundResult]) -> dict[str, float] | None:
    diags = [r.attack_diagnostics for r in rounds if r.attack_diagnostics]
    if not diags:
        return None
    keys = [
        k
        for k, v in diags[0].items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    out = {}
    for k in keys:
        vals = [float(d[k]) for d in diags if d.get(k) is not None]
        vals = [v for v in vals if v == v]  # drop NaN
        out[k] = float(np.mean(vals)) if vals else float("nan")
    out["frac_rounds_stealthy"] = float(
        np.mean([1.0 if d.get("is_stealthy") else 0.0 for d in diags])
    )
    out["frac_rounds_coordinated"] = float(
        np.mean([1.0 if d.get("is_coordinated") else 0.0 for d in diags])
    )
    return out


def run_fl_baseline(
    config: FLBaselineConfig,
    initial_model: nn.Module,
    client_datasets: Sequence[Any],
    eval_fn: Callable[[nn.Module], Any],
    *,
    lr: float = 0.05,
    local_epochs: int = 1,
    batch_size: int = 32,
    seed: int = 0,
    server_root_dataset: Any | None = None,
    clients: Sequence[FLClient] | None = None,
) -> FLRunResult:
    """Run one FL configuration end to end.

    Args:
        config: The row to run.
        initial_model: Frozen pretrained checkpoint.  Deep-copied.
        client_datasets: One local dataset per client, disjoint.  Build them
            with the three-way split so the gate never sees test data.
        eval_fn: REPORTING ONLY -- evaluated on held-out test data.
        server_root_dataset: The SERVER VERIFICATION POOL.  Required for
            FLTrust, which computes its reference update on it.
        clients: Pre-built clients, overriding ``client_datasets``.  Used by
            the always-accept / no-gate identity check, which must run the two
            configurations against byte-identical client behaviour.
    """
    aggregator = config.build_aggregator()
    gate = config.build_gate()

    server_update_fn = None
    if isinstance(aggregator, FLTrust):
        if server_root_dataset is None:
            raise ValueError(
                "FLTrust requires server_root_dataset (the server verification "
                "pool). Running it without one silently degenerates to FedAvg."
            )

        def server_update_fn(model: nn.Module, round_num: int):  # noqa: F811
            return fltrust_server_update(
                model,
                server_root_dataset,
                lr=lr,
                local_epochs=local_epochs,
                batch_size=batch_size,
                seed=seed + 991 * round_num,
            )

    if clients is None:
        clients = build_clients(
            client_datasets,
            config.attack,
            lr=lr,
            local_epochs=local_epochs,
            batch_size=batch_size,
            seed=seed,
        )

    server = FederatedServer(
        initial_model,
        aggregator=aggregator,
        gate=gate,
        eval_fn=eval_fn,
        server_update_fn=server_update_fn,
    )
    rounds = server.run(clients, config.n_rounds)
    server.assert_rejections_were_noops()

    return FLRunResult(
        name=config.name,
        summary=server.summary(),
        rounds=[r.as_dict() for r in rounds],
        final_metric=(rounds[-1].committed_metric if rounds else eval_fn(initial_model)),
        final_hash=model_state_hash(server.global_model),
        initial_hash=server.initial_hash,
        mean_attack_diagnostics=_mean_diagnostics(rounds),
    )


# ---------------------------------------------------------------------------
# The control suite and its assertions
# ---------------------------------------------------------------------------


def create_fl_arm_configs(
    n_rounds: int = 10,
    attack: AttackConfig | None = None,
    aggregators: Sequence[str] = ("fedavg", "median", "trimmed_mean", "krum",
                                  "multi_krum", "fltrust"),
    gates: Sequence[str] = ("none", "always_accept", "always_reject"),
) -> list[FLBaselineConfig]:
    """The full FL grid: controls first, then aggregator x gate.

    ``attack=None`` gives the clean grid.  Every reported table must contain
    the clean grid as well as the attacked one, because a defence that also
    destroys the clean run has not defended anything.
    """
    configs = [
        FLBaselineConfig(
            name="clean_fedavg_no_gate",
            aggregator="fedavg",
            gate="none",
            attack=None,
            n_rounds=n_rounds,
            description=(
                "Acceptance criterion C1 reference: must beat frozen_pretrained"
            ),
        ),
        FLBaselineConfig(
            name="always_reject",
            aggregator="fedavg",
            gate="always_reject",
            attack=attack,
            n_rounds=n_rounds,
            description="CONTROL: must equal frozen_pretrained exactly",
        ),
        FLBaselineConfig(
            name="always_accept",
            aggregator="fedavg",
            gate="always_accept",
            attack=attack,
            n_rounds=n_rounds,
            description="CONTROL: must equal the same run with gate='none'",
        ),
    ]
    for agg in aggregators:
        for gate in gates:
            configs.append(
                FLBaselineConfig(
                    name=f"{agg}__gate={gate}",
                    aggregator=agg,
                    gate=gate,
                    attack=attack,
                    n_rounds=n_rounds,
                )
            )
    return configs


def assert_always_reject_equals_frozen(
    reject: FLRunResult, frozen: FLRunResult, tol: float = 0.0
) -> None:
    """Assert the always-reject row IS the frozen checkpoint.

    Checked at the level of parameter bytes (hash), not just the metric, so a
    coincidental metric match cannot hide a real change.
    """
    if reject.final_hash != frozen.final_hash:
        raise AssertionError(
            "always_reject did not reproduce frozen_pretrained: "
            f"{reject.final_hash} != {frozen.final_hash}. Rollback is not a "
            "no-op (finding F5)."
        )
    if reject.summary.get("rounds_accepted", 0) != 0:
        raise AssertionError(
            "always_reject accepted "
            f"{reject.summary['rounds_accepted']} rounds"
        )
    _assert_metric_close(reject.final_metric, frozen.final_metric, tol,
                         "always_reject vs frozen_pretrained")


def assert_always_accept_equals_no_gate(
    accept: FLRunResult, no_gate: FLRunResult, tol: float = 0.0
) -> None:
    """Assert the always-accept gate is a no-op relative to running ungated."""
    if accept.final_hash != no_gate.final_hash:
        raise AssertionError(
            "always_accept did not reproduce the ungated run: "
            f"{accept.final_hash} != {no_gate.final_hash}"
        )
    if accept.summary.get("rounds_accepted") != accept.summary.get("n_rounds"):
        raise AssertionError("always_accept rejected at least one round")
    _assert_metric_close(accept.final_metric, no_gate.final_metric, tol,
                         "always_accept vs no gate")


def _assert_metric_close(a: Any, b: Any, tol: float, label: str) -> None:
    if a is None or b is None:
        return
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for k in set(a) & set(b):
            if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float)):
                if abs(float(a[k]) - float(b[k])) > tol:
                    raise AssertionError(
                        f"{label}: metric {k!r} differs by "
                        f"{abs(float(a[k]) - float(b[k])):.6g} > tol={tol}"
                    )
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(float(a) - float(b)) > tol:
            raise AssertionError(
                f"{label}: metric differs by {abs(float(a) - float(b)):.6g} "
                f"> tol={tol}"
            )


def check_c1_fedavg_beats_frozen(
    clean_fedavg: FLRunResult,
    frozen: FLRunResult,
    metric_key: str | None = "accuracy",
) -> dict[str, Any]:
    """Acceptance criterion C1, as a reported number rather than an assertion.

    C1: clean FedAvg with zero attackers and no gate must EXCEED the frozen
    pretrained checkpoint on held-out test data.  We return pass/fail and both
    values instead of raising, because "C1 cannot be made to hold at this
    scale" is a legitimate -- and reportable -- outcome; silently tuning until
    it passes is not.
    """

    def _get(m: Any) -> float | None:
        if m is None:
            return None
        if isinstance(m, Mapping):
            return float(m[metric_key]) if metric_key in m else None
        return float(m)

    a = _get(clean_fedavg.final_metric)
    b = _get(frozen.final_metric)
    passed = a is not None and b is not None and a > b
    return {
        "criterion": "C1",
        "statement": (
            "clean FedAvg (0 attackers, no gate) > frozen pretrained on "
            "held-out test data"
        ),
        "clean_fedavg": a,
        "frozen_pretrained": b,
        "delta": (None if a is None or b is None else a - b),
        "passed": bool(passed),
    }


def run_gate_control_suite(
    initial_model: nn.Module,
    client_datasets: Sequence[Any],
    eval_fn: Callable[[nn.Module], Any],
    *,
    attack: AttackConfig | None = None,
    n_rounds: int = 3,
    aggregator: str = "fedavg",
    lr: float = 0.05,
    local_epochs: int = 1,
    batch_size: int = 32,
    seed: int = 0,
    metric_key: str | None = "accuracy",
) -> dict[str, Any]:
    """Run the four FL controls and verify both identities.

    Returns a dict with the four :class:`FLRunResult` rows, the C1 verdict,
    and the two identity checks -- everything the results file needs.  Raises
    :class:`AssertionError` if either identity fails, because a failure there
    invalidates every gated number.
    """
    common = dict(
        lr=lr,
        local_epochs=local_epochs,
        batch_size=batch_size,
        seed=seed,
    )

    frozen = frozen_pretrained_baseline(initial_model, eval_fn)

    clean_cfg = FLBaselineConfig(
        name="clean_fedavg_no_gate",
        aggregator=aggregator,
        gate="none",
        attack=None,
        n_rounds=n_rounds,
    )
    clean = run_fl_baseline(
        clean_cfg, initial_model, client_datasets, eval_fn, **common
    )

    # The two identity checks must compare runs whose CLIENTS behave
    # identically, so build the client list once and share it.
    no_gate_cfg = FLBaselineConfig(
        name="no_gate", aggregator=aggregator, gate="none",
        attack=attack, n_rounds=n_rounds,
    )
    accept_cfg = FLBaselineConfig(
        name="always_accept", aggregator=aggregator, gate="always_accept",
        attack=attack, n_rounds=n_rounds,
    )
    reject_cfg = FLBaselineConfig(
        name="always_reject", aggregator=aggregator, gate="always_reject",
        attack=attack, n_rounds=n_rounds,
    )

    def _fresh_clients() -> list[FLClient]:
        return build_clients(client_datasets, attack, **common)

    no_gate = run_fl_baseline(
        no_gate_cfg, initial_model, client_datasets, eval_fn,
        clients=_fresh_clients(), **common,
    )
    accept = run_fl_baseline(
        accept_cfg, initial_model, client_datasets, eval_fn,
        clients=_fresh_clients(), **common,
    )
    reject = run_fl_baseline(
        reject_cfg, initial_model, client_datasets, eval_fn,
        clients=_fresh_clients(), **common,
    )

    assert_always_accept_equals_no_gate(accept, no_gate)
    assert_always_reject_equals_frozen(reject, frozen)

    return {
        "frozen_pretrained": frozen.as_dict(),
        "clean_fedavg_no_gate": clean.as_dict(),
        "no_gate": no_gate.as_dict(),
        "always_accept": accept.as_dict(),
        "always_reject": reject.as_dict(),
        "identity_always_accept_equals_no_gate": True,
        "identity_always_reject_equals_frozen": True,
        "C1": check_c1_fedavg_beats_frozen(clean, frozen, metric_key),
    }


# ---------------------------------------------------------------------------
# Static-suite gate (Baseline 3)
# ---------------------------------------------------------------------------


class StaticSuiteGate:
    """Fixed-prompt safety gate: no recursion, no graph guidance.

    Kept as a baseline because it is the honest floor a "recursive,
    graph-guided" verifier has to beat.  The bound is computed with the same
    anytime-valid machinery as the real gate, with ``k_total = 1`` and the
    suite size as the a-priori budget cap -- both declared here, before any
    data is seen, as Theorem 1 requires.
    """

    name = "static_suite"

    def __init__(
        self,
        score_fn: Callable[[nn.Module], tuple[int, int]],
        epsilon: float = 0.1,
        delta: float = 0.05,
        budget_cap: int | None = None,
    ) -> None:
        """
        Args:
            score_fn: ``model -> (n_violations, n_samples)`` on the FIXED
                suite.  It must draw from the server verification pool, never
                from the held-out test set.
            epsilon: Acceptance threshold.
            delta: Global failure probability.
            budget_cap: A-priori per-region cap ``M``.  Defaults to the suite
                size observed on the first call, which is legitimate only
                because the suite is fixed in advance.
        """
        self.score_fn = score_fn
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.budget_cap = budget_cap

    def evaluate(self, candidate_model: nn.Module, round_num: int) -> GateDecision:
        from sca.utils.stats import RegionStat, check_acceptance

        n_violations, n_samples = self.score_fn(candidate_model)
        if self.budget_cap is None:
            self.budget_cap = max(1, int(n_samples))

        result = check_acceptance(
            [
                RegionStat(
                    region_id=0,
                    weight=1.0,
                    n_samples=int(n_samples),
                    n_violations=int(n_violations),
                )
            ],
            epsilon=self.epsilon,
            delta=self.delta,
            k_total=1,
            budget_cap=int(self.budget_cap),
        )
        return GateDecision(
            accepted=bool(result.accepted),
            bound=float(result.bound),
            epsilon=self.epsilon,
            certificate=result,
            info={
                "n_violations": int(n_violations),
                "n_samples": int(n_samples),
                "vacuous": bool(result.vacuous),
            },
        )
