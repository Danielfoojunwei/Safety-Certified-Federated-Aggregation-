"""Server-side aggregation rules ``A(theta_t; Delta_{t,1}, ..., Delta_{t,n})``.

This module is the SINGLE implementation of every aggregation rule in the
repository.  ``sca/experiments/run_real_evaluation.py`` used to carry its own
inline copies (``aggregate_fedavg``, ``aggregate_trimmed_mean``,
``aggregate_krum``) that silently disagreed with these -- the inline Krum used
un-squared distances and had no assumption check at all.  Those duplicates must
be deleted and replaced by :func:`build_aggregator` (see the integrator note in
the builder report).

Implemented
-----------
``fedavg``        McMahan et al. 2017, sample-weighted mean.
``fedadam``       Reddi et al. 2021, server-side Adam.
``median``        Coordinate-wise median (Yin et al. 2018).
``trimmed_mean``  Coordinate-wise beta-trimmed mean (Yin et al. 2018).
``krum``          Blanchard et al. 2017.
``multi_krum``    Blanchard et al. 2017, average of the best ``m`` scores.
``fltrust``       Cao et al., NDSS 2021 -- server-side validation-gated
                  aggregation.  This is the closest prior work to this
                  project and MUST appear in every comparison table.

Stated assumptions are enforced, not assumed
--------------------------------------------
Krum and Multi-Krum are only defined when ``n > 2f + 2``.  Running them at
``f = n/2`` and reporting the resulting collapse as a defeat is
strawmanning -- that is finding F7.  Every aggregator therefore exposes
:meth:`Aggregator.check_assumptions`, and every ``aggregate`` call records an
:class:`AssumptionStatus` on ``self.last_assumption`` that the experiment
harness is expected to write into the results file.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from sca.federated.client import ClientUpdate, flatten_delta, local_sgd

logger = logging.getLogger(__name__)

__all__ = [
    "AssumptionStatus",
    "Aggregator",
    "FedAvg",
    "FedAdam",
    "CoordinateMedian",
    "TrimmedMean",
    "Krum",
    "MultiKrum",
    "FLTrust",
    "AGGREGATOR_REGISTRY",
    "build_aggregator",
    "fltrust_server_update",
    "apply_delta_",
    "pairwise_sq_distances",
    "krum_scores",
]


# ---------------------------------------------------------------------------
# Assumption bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssumptionStatus:
    """Whether an aggregator was run inside its stated assumption.

    Attributes:
        aggregator: Aggregator name.
        requirement: The condition in symbols, e.g. ``"n > 2f + 2"``.
        satisfied: Whether it held for this configuration.
        detail: Human-readable instantiation of the requirement.
        n_clients: ``n`` at the time of the check.
        n_byzantine_assumed: The ``f`` the aggregator was configured with.
    """

    aggregator: str
    requirement: str
    satisfied: bool
    detail: str
    n_clients: int = 0
    n_byzantine_assumed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "aggregator": self.aggregator,
            "requirement": self.requirement,
            "satisfied": self.satisfied,
            "detail": self.detail,
            "n_clients": self.n_clients,
            "n_byzantine_assumed": self.n_byzantine_assumed,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def apply_delta_(model: nn.Module, delta: Mapping[str, torch.Tensor]) -> None:
    """Apply ``delta`` to ``model`` in place (no-op for missing keys)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in delta:
                param.add_(delta[name].to(param.dtype))


def _param_keys(model: nn.Module, updates: Sequence[ClientUpdate]) -> list[str]:
    """Parameter names present in both the model and every update."""
    names = [n for n, _ in model.named_parameters()]
    return [n for n in names if all(n in u.delta for u in updates)]


def pairwise_sq_distances(updates: Sequence[ClientUpdate]) -> torch.Tensor:
    """``n x n`` matrix of squared Euclidean distances between flattened updates."""
    flat = torch.stack([flatten_delta(u.delta) for u in updates])
    return torch.cdist(flat, flat, p=2) ** 2


def krum_scores(
    updates: Sequence[ClientUpdate], n_byzantine: int
) -> torch.Tensor:
    """Krum scores: sum of the ``n - f - 2`` smallest squared distances.

    Blanchard et al. (2017), Section 3.  The score of client ``i`` is

    .. math::  s(i) = \\sum_{i \\to j} \\| \\Delta_i - \\Delta_j \\|^2

    where ``i -> j`` denotes the ``n - f - 2`` closest other clients.
    """
    n = len(updates)
    f = int(n_byzantine)
    n_nearest = max(1, n - f - 2)
    d2 = pairwise_sq_distances(updates)
    d2 = d2.clone()
    d2.fill_diagonal_(float("inf"))
    sorted_d2, _ = torch.sort(d2, dim=1)
    return sorted_d2[:, :n_nearest].sum(dim=1)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Aggregator(ABC):
    """Abstract base class for federated aggregation rules."""

    name: str = "aggregator"
    #: Populated by every ``aggregate`` call.
    last_assumption: AssumptionStatus | None = None
    #: Diagnostics from the most recent call (e.g. Krum's selected index).
    last_info: dict[str, Any]

    def __init__(self) -> None:
        self.last_assumption = None
        self.last_info = {}

    @abstractmethod
    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        """Aggregate client updates into a single parameter update."""

    def check_assumptions(
        self, n_clients: int, n_byzantine_actual: int | None = None
    ) -> AssumptionStatus:
        """Report whether this configuration lies inside the stated assumption.

        The default rule has no Byzantine-tolerance assumption to violate --
        FedAvg is simply not robust, which is the honest statement.
        """
        return AssumptionStatus(
            aggregator=self.name,
            requirement="none",
            satisfied=True,
            detail="no Byzantine-robustness assumption is claimed",
            n_clients=n_clients,
            n_byzantine_assumed=0,
        )

    def _record(self, n_clients: int) -> None:
        self.last_assumption = self.check_assumptions(n_clients)


# ---------------------------------------------------------------------------
# Non-robust rules
# ---------------------------------------------------------------------------


class FedAvg(Aggregator):
    """Federated Averaging (McMahan et al., AISTATS 2017).

    ``Delta = sum_i (n_i / N) Delta_i``.  Set ``uniform=True`` to ignore the
    client-reported sample counts, which is the right thing to do when
    adversaries can inflate ``n_i``.
    """

    name = "fedavg"

    def __init__(self, uniform: bool = False) -> None:
        super().__init__()
        self.uniform = bool(uniform)

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}
        self._record(len(updates))

        if self.uniform:
            weights = [1.0 / len(updates)] * len(updates)
        else:
            total = sum(max(u.n_samples, 1) for u in updates)
            weights = [max(u.n_samples, 1) / total for u in updates]

        aggregated: dict[str, torch.Tensor] = {}
        for name in _param_keys(global_model, updates):
            acc = torch.zeros_like(updates[0].delta[name])
            for w, u in zip(weights, updates):
                acc += w * u.delta[name]
            aggregated[name] = acc
        self.last_info = {"weights": weights}
        return aggregated


class FedAdam(Aggregator):
    """Server-side Adam on the averaged pseudo-gradient (Reddi et al., 2021)."""

    name = "fedadam"

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
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
        self._record(len(updates))
        self.t += 1

        mean = FedAvg().aggregate(global_model, updates)

        aggregated: dict[str, torch.Tensor] = {}
        for name, g in mean.items():
            # delta is an update, Adam expects a gradient: negate.
            grad = -g
            if name not in self.m:
                self.m[name] = torch.zeros_like(grad)
                self.v[name] = torch.zeros_like(grad)
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2
            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)
            aggregated[name] = -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        return aggregated


# ---------------------------------------------------------------------------
# Robust rules
# ---------------------------------------------------------------------------


class CoordinateMedian(Aggregator):
    """Coordinate-wise median (Yin et al., ICML 2018).

    Breakdown point ``f < n/2``.
    """

    name = "median"

    def __init__(self, n_byzantine: int = 0) -> None:
        super().__init__()
        self.n_byzantine = int(n_byzantine)

    def check_assumptions(
        self, n_clients: int, n_byzantine_actual: int | None = None
    ) -> AssumptionStatus:
        f = self.n_byzantine if n_byzantine_actual is None else n_byzantine_actual
        ok = f < n_clients / 2
        return AssumptionStatus(
            aggregator=self.name,
            requirement="f < n/2",
            satisfied=bool(ok),
            detail=f"f={f}, n={n_clients}, n/2={n_clients / 2:g}",
            n_clients=n_clients,
            n_byzantine_assumed=f,
        )

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}
        self._record(len(updates))
        return {
            name: torch.median(
                torch.stack([u.delta[name] for u in updates]), dim=0
            ).values
            for name in _param_keys(global_model, updates)
        }


class TrimmedMean(Aggregator):
    """Coordinate-wise trimmed mean (Yin et al., ICML 2018).

    Removes the ``k`` largest and ``k`` smallest values at each coordinate
    before averaging.  ``k`` is ``ceil(beta * n)`` by default, or exactly
    ``n_byzantine`` when that is given -- the latter is the setting under
    which the paper's robustness guarantee is stated (``f < n/2``).
    """

    name = "trimmed_mean"

    def __init__(self, beta: float = 0.2, n_byzantine: int | None = None) -> None:
        super().__init__()
        if not 0.0 <= beta < 0.5:
            raise ValueError(f"beta must lie in [0, 0.5), got {beta}")
        self.beta = float(beta)
        self.n_byzantine = n_byzantine

    def _trim_count(self, n: int) -> int:
        if self.n_byzantine is not None:
            k = int(self.n_byzantine)
        else:
            k = int(round(self.beta * n))
        # Always leave at least one value at every coordinate.
        return max(0, min(k, (n - 1) // 2))

    def check_assumptions(
        self, n_clients: int, n_byzantine_actual: int | None = None
    ) -> AssumptionStatus:
        f = (
            (self.n_byzantine or 0)
            if n_byzantine_actual is None
            else n_byzantine_actual
        )
        k = self._trim_count(n_clients)
        ok = f <= k and f < n_clients / 2
        return AssumptionStatus(
            aggregator=self.name,
            requirement="f <= trim_count and f < n/2",
            satisfied=bool(ok),
            detail=f"f={f}, trim_count={k}, n={n_clients}",
            n_clients=n_clients,
            n_byzantine_assumed=f,
        )

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}
        n = len(updates)
        self._record(n)
        k = self._trim_count(n)
        self.last_info = {"trim_count": k, "n": n}

        aggregated: dict[str, torch.Tensor] = {}
        for name in _param_keys(global_model, updates):
            stacked = torch.stack([u.delta[name] for u in updates])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            kept = sorted_vals[k : n - k] if k > 0 else sorted_vals
            if kept.shape[0] == 0:  # pragma: no cover - guarded by _trim_count
                kept = sorted_vals
            aggregated[name] = kept.mean(dim=0)
        return aggregated


class Krum(Aggregator):
    r"""Krum (Blanchard et al., NeurIPS 2017).

    Selects the single update minimising the sum of squared distances to its
    ``n - f - 2`` nearest neighbours, and returns it unchanged.

    STATED ASSUMPTION: ``n > 2f + 2`` (equivalently ``n >= 2f + 3``).  Outside
    it, ``n - f - 2`` is not a majority of the honest population and the
    guarantee of Blanchard et al. simply does not apply.  We still *run* the
    rule -- refusing to run would hide the behaviour -- but
    :meth:`check_assumptions` reports ``satisfied=False`` and the harness must
    print that annotation next to the number.  Reporting a 50 %-Byzantine
    result against Krum without it is strawmanning (finding F7).

    SECOND FAILURE MODE, ALSO NOT KRUM'S FAULT: Krum returns *one client's*
    update.  If the client partition is degenerate -- e.g. sorted by label and
    chunked, so each client holds a single class -- then the selected update
    trains the global model on one class and the model collapses to a constant
    predictor even with zero adversaries.  That is a broken data split, not a
    broken aggregator; see ``tests/test_aggregation.py::TestZeroByzantineParity``,
    which exhibits both splits side by side.
    """

    name = "krum"

    def __init__(self, n_byzantine: int = 0, strict: bool = False) -> None:
        """
        Args:
            n_byzantine: Assumed upper bound ``f`` on the number of adversaries.
            strict: Raise instead of warning when ``n <= 2f + 2``.
        """
        super().__init__()
        self.n_byzantine = int(n_byzantine)
        self.strict = bool(strict)

    def check_assumptions(
        self, n_clients: int, n_byzantine_actual: int | None = None
    ) -> AssumptionStatus:
        f = self.n_byzantine if n_byzantine_actual is None else n_byzantine_actual
        ok = n_clients > 2 * f + 2
        return AssumptionStatus(
            aggregator=self.name,
            requirement="n > 2f + 2",
            satisfied=bool(ok),
            detail=(
                f"n={n_clients}, f={f}, need n >= {2 * f + 3}"
                + ("" if ok else "  [OUTSIDE STATED ASSUMPTION]")
            ),
            n_clients=n_clients,
            n_byzantine_assumed=f,
        )

    def _select(self, updates: Sequence[ClientUpdate]) -> int:
        scores = krum_scores(updates, self.n_byzantine)
        return int(torch.argmin(scores).item())

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}
        n = len(updates)
        status = self.check_assumptions(n)
        self.last_assumption = status
        if not status.satisfied:
            msg = (
                f"Krum run outside its stated assumption: {status.detail}. "
                "Any resulting degradation must be reported as 'outside "
                "assumption', not as a defeat of Krum."
            )
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)

        idx = self._select(updates)
        self.last_info = {
            "selected_index": idx,
            "selected_client_id": updates[idx].client_id,
            "selected_is_byzantine": bool(updates[idx].is_byzantine),
            "n_nearest": max(1, n - self.n_byzantine - 2),
            "assumption_satisfied": status.satisfied,
        }
        keys = _param_keys(global_model, updates)
        return {name: updates[idx].delta[name].clone() for name in keys}


class MultiKrum(Aggregator):
    """Multi-Krum (Blanchard et al., NeurIPS 2017).

    Averages the ``m`` updates with the smallest Krum scores.  ``m`` defaults
    to ``n - f``, the value used in the original paper.  Shares Krum's
    ``n > 2f + 2`` assumption.
    """

    name = "multi_krum"

    def __init__(
        self,
        n_byzantine: int = 0,
        m: int | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.n_byzantine = int(n_byzantine)
        self.m = m
        self.strict = bool(strict)

    def check_assumptions(
        self, n_clients: int, n_byzantine_actual: int | None = None
    ) -> AssumptionStatus:
        base = Krum(self.n_byzantine).check_assumptions(
            n_clients, n_byzantine_actual
        )
        return AssumptionStatus(
            aggregator=self.name,
            requirement=base.requirement,
            satisfied=base.satisfied,
            detail=base.detail,
            n_clients=base.n_clients,
            n_byzantine_assumed=base.n_byzantine_assumed,
        )

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}
        n = len(updates)
        status = self.check_assumptions(n)
        self.last_assumption = status
        if not status.satisfied:
            msg = (
                f"Multi-Krum run outside its stated assumption: "
                f"{status.detail}."
            )
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)

        m = self.m if self.m is not None else max(1, n - self.n_byzantine)
        m = max(1, min(m, n))
        scores = krum_scores(updates, self.n_byzantine)
        _, order = torch.topk(scores, m, largest=False)
        selected = [int(i) for i in order.tolist()]
        self.last_info = {
            "selected_indices": selected,
            "selected_client_ids": [updates[i].client_id for i in selected],
            "n_selected": m,
            "assumption_satisfied": status.satisfied,
        }

        keys = _param_keys(global_model, updates)
        return {
            name: torch.stack([updates[i].delta[name] for i in selected]).mean(
                dim=0
            )
            for name in keys
        }


class FLTrust(Aggregator):
    r"""FLTrust (Cao, Fang, Liu, Gong -- NDSS 2021).

    THE CLOSEST PRIOR WORK TO THIS PROJECT.  FLTrust is already a
    server-side, validation-set-gated aggregator: the server holds a small
    clean *root dataset*, computes its own update ``g_0`` on it, and then

    .. math::
        TS_i = \operatorname{ReLU}\left(
            \cos(g_0, \Delta_i) \right), \qquad
        \Delta = \frac{1}{\sum_i TS_i} \sum_i TS_i
                 \frac{\|g_0\|}{\|\Delta_i\|} \Delta_i .

    Two mechanisms: the ReLU-cosine trust score zeroes out any update pointing
    away from the server's own descent direction, and the norm rescaling
    removes the scaling attack entirely.

    Any novelty claim for a "safety-certified" gate must be made *relative to
    this*, which is why it is a first-class aggregator here rather than a
    footnote.  Use :func:`fltrust_server_update` to produce ``g_0`` from the
    SERVER VERIFICATION POOL each round.
    """

    name = "fltrust"

    def __init__(
        self,
        server_update: Mapping[str, torch.Tensor] | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.server_update = dict(server_update) if server_update else None
        self.normalize = bool(normalize)

    def set_server_update(self, update: Mapping[str, torch.Tensor]) -> None:
        """Install the server's reference update for the current round."""
        self.server_update = dict(update)

    def check_assumptions(
        self, n_clients: int, n_byzantine_actual: int | None = None
    ) -> AssumptionStatus:
        ok = self.server_update is not None
        return AssumptionStatus(
            aggregator=self.name,
            requirement="server holds a clean root dataset",
            satisfied=bool(ok),
            detail=(
                "server reference update present"
                if ok
                else "NO server update installed -- FLTrust degenerates to FedAvg"
            ),
            n_clients=n_clients,
            n_byzantine_assumed=(n_byzantine_actual or 0),
        )

    def aggregate(
        self,
        global_model: nn.Module,
        updates: list[ClientUpdate],
    ) -> dict[str, torch.Tensor]:
        if not updates:
            return {}
        n = len(updates)
        self.last_assumption = self.check_assumptions(n)

        if self.server_update is None:
            logger.warning(
                "FLTrust has no server update; falling back to FedAvg. This "
                "is NOT FLTrust and must not be reported as such."
            )
            self.last_info = {"fallback": "fedavg_no_server_update"}
            return FedAvg().aggregate(global_model, updates)

        g0 = flatten_delta(self.server_update)
        g0_norm = float(g0.norm().item())
        if g0_norm < 1e-12:
            logger.warning("FLTrust server update is ~zero; falling back to FedAvg.")
            self.last_info = {"fallback": "fedavg_zero_server_update"}
            return FedAvg().aggregate(global_model, updates)

        keys = _param_keys(global_model, updates)
        trust: list[float] = []
        scales: list[float] = []
        for u in updates:
            gi = flatten_delta(u.delta)
            gi_norm = float(gi.norm().item())
            if gi_norm < 1e-12:
                trust.append(0.0)
                scales.append(0.0)
                continue
            cos = float((torch.dot(g0, gi) / (g0_norm * gi_norm)).item())
            trust.append(max(0.0, cos))
            scales.append(g0_norm / gi_norm if self.normalize else 1.0)

        total = float(sum(trust))
        self.last_info = {
            "trust_scores": trust,
            "n_zero_trust": int(sum(1 for t in trust if t <= 0.0)),
            "total_trust": total,
        }

        if total < 1e-12:
            # Every client points away from the server. Committing nothing is
            # the correct FLTrust behaviour here (all weights are zero).
            self.last_info["all_rejected"] = True
            return {name: torch.zeros_like(updates[0].delta[name]) for name in keys}

        aggregated: dict[str, torch.Tensor] = {}
        for name in keys:
            acc = torch.zeros_like(updates[0].delta[name])
            for t, s, u in zip(trust, scales, updates):
                if t <= 0.0:
                    continue
                acc += (t / total) * s * u.delta[name]
            aggregated[name] = acc
        return aggregated


def fltrust_server_update(
    global_model: nn.Module,
    root_dataset: Any,
    *,
    lr: float = 0.05,
    local_epochs: int = 1,
    batch_size: int = 32,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Compute FLTrust's server reference update ``g_0`` on the root dataset.

    The root dataset must be the SERVER VERIFICATION POOL -- data the server is
    allowed to see.  It must never be the held-out test set.
    """
    delta, _ = local_sgd(
        global_model,
        root_dataset,
        lr=lr,
        local_epochs=local_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    return delta


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


AGGREGATOR_REGISTRY: dict[str, type[Aggregator]] = {
    FedAvg.name: FedAvg,
    FedAdam.name: FedAdam,
    CoordinateMedian.name: CoordinateMedian,
    TrimmedMean.name: TrimmedMean,
    Krum.name: Krum,
    MultiKrum.name: MultiKrum,
    FLTrust.name: FLTrust,
}


def build_aggregator(name: str, **kwargs: Any) -> Aggregator:
    """Instantiate a registered aggregator by name.

    This is the single entry point the experiment runners must use; the inline
    ``aggregate_*`` helpers in ``run_real_evaluation.py`` are to be deleted.
    Unknown keyword arguments are dropped with a warning so that a shared
    config dict (``n_byzantine=...``) can be passed to every aggregator.
    """
    if name not in AGGREGATOR_REGISTRY:
        raise ValueError(
            f"unknown aggregator {name!r}; known: {sorted(AGGREGATOR_REGISTRY)}"
        )
    cls = AGGREGATOR_REGISTRY[name]
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    used = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = sorted(set(kwargs) - set(used))
    if dropped:
        logger.debug("build_aggregator(%s): ignoring %s", name, dropped)
    return cls(**used)
