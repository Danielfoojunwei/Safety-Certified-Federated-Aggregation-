"""Byzantine attacks against federated aggregation.

WHY THIS FILE WAS REBUILT (audit finding F6)
--------------------------------------------
The previous implementation's "attacks" were fresh zero-mean Gaussian noise::

    delta[name] = -torch.randn_like(param) * 0.01 * scale     # "sign_flip"
    delta[name] =  torch.randn_like(param) * 0.05 * scale     # "noise"
    delta[name] = -scale * 0.1 * torch.sign(param.data)       # "ipm"

None of the first two depends on the honest updates or on the model; two
colluding attackers had pairwise cosine similarity ``-0.007`` (i.e. they were
independent), and in the shipped logs the "attack" *raised* FedAvg accuracy
from 0.84 to 0.9225.  Averaging independent zero-mean noise into an average of
n updates shrinks it by 1/n and acts as a mild regulariser.  A defence that
"survives" such an attack has been tested against nothing.

WHAT AN ATTACK MUST BE HERE
---------------------------
Every attack in this module:

1. takes ``benign_deltas`` -- the honest clients' updates for the *current*
   round -- as a REQUIRED first positional argument (omniscient adversary, the
   standard model in Xie et al. 2020 and Baruch et al. 2019);
2. declares whether its output actually depends on them
   (``depends_on_benign_deltas``), so the test-suite can assert dependence for
   real attacks and assert *in*dependence for the deliberately-degenerate
   control that reproduces F6;
3. is deterministic given ``(benign_deltas, context)`` -- no ambient RNG.

Implemented attacks
-------------------
``sign_flip``   Negate the attacker's OWN honest update (Blanchard et al. 2017
                style direction attack), optionally rescaled to sit inside the
                benign norm envelope.
``ipm``         Inner Product Manipulation, Xie et al. (UAI 2020):
                ``-epsilon * mean(benign_deltas)``.
``alie``        A Little Is Enough, Baruch et al. (NeurIPS 2019):
                ``mean(benign) - z * std(benign)``, coordinate-wise.
``targeted_safety``  Targeted safety degradation: train on data whose unsafe
                examples are relabelled safe, amplify the resulting direction,
                then project into the benign norm envelope so the update is
                genuinely stealthy under norm-based screening.

Controls (must be reported alongside the real attacks)
------------------------------------------------------
``legacy_gaussian``  The F6 attack, kept verbatim so the paper can show that
                the old "50% Byzantine" headline was obtained against noise.
``no_attack``   The attacker behaves honestly.  Any defence must be a no-op
                here.

References:
    Blanchard, El Mhamdi, Guerraoui, Stainer. "Machine Learning with
        Adversaries: Byzantine Tolerant Gradient Descent." NeurIPS 2017.
    Baruch, Baruch, Goldberg. "A Little Is Enough: Circumventing Defenses for
        Distributed Learning." NeurIPS 2019.
    Xie, Koyejo, Gupta. "Fall of Empires: Breaking Byzantine-tolerant SGD by
        Inner Product Manipulation." UAI 2020.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from sca.federated.client import (
    ByzantineClient,
    delta_norm,
    flatten_delta,
    scale_delta,
)

__all__ = [
    "AttackContext",
    "ByzantineAttack",
    "SignFlipAttack",
    "IPMAttack",
    "ALIEAttack",
    "TargetedSafetyDegradationAttack",
    "LegacyGaussianNoiseAttack",
    "NoAttack",
    "ATTACK_REGISTRY",
    "REAL_ATTACKS",
    "build_attack",
    "AttackDiagnostics",
    "attack_diagnostics",
    "cosine",
    "mean_delta",
    "std_delta",
    "project_into_norm_envelope",
    "AttackConfig",
    "AttackScenario",
    "create_byzantine_clients",
    "create_standard_attack_scenarios",
    "create_attack_scenario",
]


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------


def _keys(deltas: Sequence[Mapping[str, torch.Tensor]]) -> list[str]:
    if not deltas:
        raise ValueError("empty delta sequence")
    keys = sorted(deltas[0])
    for d in deltas[1:]:
        if sorted(d) != keys:
            raise ValueError("client updates have mismatched parameter keys")
    return keys


def mean_delta(
    deltas: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Coordinate-wise mean of a list of parameter dicts."""
    keys = _keys(deltas)
    return {
        k: torch.stack([d[k] for d in deltas]).mean(dim=0) for k in keys
    }


def std_delta(
    deltas: Sequence[Mapping[str, torch.Tensor]],
    unbiased: bool = False,
) -> dict[str, torch.Tensor]:
    """Coordinate-wise standard deviation of a list of parameter dicts."""
    keys = _keys(deltas)
    if len(deltas) < 2:
        return {k: torch.zeros_like(deltas[0][k]) for k in keys}
    return {
        k: torch.stack([d[k] for d in deltas]).std(dim=0, unbiased=unbiased)
        for k in keys
    }


def cosine(
    a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]
) -> float:
    """Cosine similarity between two flattened parameter dicts."""
    fa = flatten_delta(a)
    fb = flatten_delta(b)
    na = float(fa.norm().item())
    nb = float(fb.norm().item())
    if na < 1e-30 or nb < 1e-30:
        return float("nan")
    return float((torch.dot(fa, fb) / (na * nb)).item())


def project_into_norm_envelope(
    delta: Mapping[str, torch.Tensor],
    benign_deltas: Sequence[Mapping[str, torch.Tensor]],
    quantile: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Rescale ``delta`` so its norm does not exceed the benign norm envelope.

    "Stealthy" in this repo means a testable property, not an adjective: the
    attacker's update norm lies at or below the ``quantile``-th quantile of the
    benign update norms, so a norm-threshold screen cannot separate it.

    Only shrinks; never inflates a small update.
    """
    norms = np.array([delta_norm(d) for d in benign_deltas], dtype=float)
    if norms.size == 0:
        return dict(delta)
    cap = float(np.quantile(norms, quantile))
    own = delta_norm(delta)
    if own <= cap or own < 1e-30 or cap <= 0.0:
        return dict(delta)
    return scale_delta(delta, cap / own)


# ---------------------------------------------------------------------------
# Attack interface
# ---------------------------------------------------------------------------


@dataclass
class AttackContext:
    """Everything an attack may use besides ``benign_deltas``.

    Attributes:
        honest_delta: The attacker's own honest local update, when the attack
            declares ``requires_honest_delta``.
        poisoned_delta: The attacker's update from training on relabelled
            data, when the attack declares ``requires_poisoned_delta``.
        global_model: The current global model (read-only).
        round_num: 1-based FL round index.
        attacker_index: Index of this attacker inside the colluding group.
        n_attackers: Size of the colluding group ``f``.
        client_id: Client id, for reproducible per-client streams.
        seed: Base seed for any (deliberately rare) randomness.
    """

    honest_delta: dict[str, torch.Tensor] | None = None
    poisoned_delta: dict[str, torch.Tensor] | None = None
    global_model: nn.Module | None = None
    round_num: int = 1
    attacker_index: int = 0
    n_attackers: int = 1
    client_id: int = 0
    seed: int = 0

    def torch_generator(self, stream: int = 0) -> torch.Generator:
        """A reproducible torch generator derived from the context.

        Uses integer arithmetic only -- no Python ``hash()`` (finding F9).
        """
        g = torch.Generator()
        raw = (
            self.seed * 1_000_003
            + self.round_num * 7919
            + self.client_id * 104_729
            + stream * 15_485_863
        )
        g.manual_seed(int(raw) % (2**63 - 1))
        return g


class ByzantineAttack(ABC):
    """Base class for Byzantine attacks.

    Subclasses implement :meth:`craft`, whose FIRST positional argument is the
    round's honest updates.  That is a hard API rule, not a convention: the
    old code's ability to produce an "attack" without ever looking at an
    honest update is finding F6.
    """

    #: Short registry name.
    name: str = "attack"
    #: Does ``craft``'s output actually change when ``benign_deltas`` change?
    #: False only for the deliberately-degenerate F6 control.
    depends_on_benign_deltas: bool = True
    #: Is this a control rather than a real attack?
    is_degenerate_control: bool = False
    #: Does the attack need the attacker's own honest update?
    requires_honest_delta: bool = False
    #: Does the attack need the attacker's poisoned-data update?
    requires_poisoned_delta: bool = False
    #: Do colluding attackers send identical updates?
    is_coordinated: bool = True
    #: Optional relabelling used to construct ``poisoned_delta``.
    default_label_map: Callable[[torch.Tensor], torch.Tensor] | None = None

    @abstractmethod
    def craft(
        self,
        benign_deltas: Sequence[Mapping[str, torch.Tensor]],
        context: AttackContext,
    ) -> dict[str, torch.Tensor]:
        """Return the adversarial update for one attacker.

        Args:
            benign_deltas: REQUIRED. The honest clients' updates this round.
            context: Auxiliary information (see :class:`AttackContext`).
        """

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "depends_on_benign_deltas": self.depends_on_benign_deltas,
            "is_degenerate_control": self.is_degenerate_control,
            "requires_honest_delta": self.requires_honest_delta,
            "requires_poisoned_delta": self.requires_poisoned_delta,
            "is_coordinated": self.is_coordinated,
            "params": {
                k: v
                for k, v in vars(self).items()
                if isinstance(v, (int, float, str, bool))
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{type(self).__name__}({self.describe()['params']})"


# ---------------------------------------------------------------------------
# Real attacks
# ---------------------------------------------------------------------------


class SignFlipAttack(ByzantineAttack):
    r"""Negate the attacker's own honest update.

    ``delta = -scale * honest_delta``, optionally rescaled so that
    ``||delta||`` sits inside the benign norm envelope.

    This is the fix for F6's "sign_flip": the old version negated *fresh
    Gaussian noise*, which is distributionally identical to fresh Gaussian
    noise.  Here the attacker actually computes the gradient it would have
    contributed and sends its negation, so the attack removes exactly the
    signal the honest client would have added and pushes the model backwards
    along the true descent direction.

    Colluding attackers do NOT send identical vectors (each negates its own
    local gradient), but on non-IID data those gradients are positively
    correlated, so the group is still positively correlated -- unlike
    independent noise.  Set ``share_direction=True`` to make the group send
    the negation of the *mean honest* update instead, which is the fully
    coordinated variant.
    """

    name = "sign_flip"
    requires_honest_delta = True

    def __init__(
        self,
        scale: float = 1.0,
        norm_match: bool = True,
        envelope_quantile: float = 1.0,
        share_direction: bool = False,
    ) -> None:
        """
        Args:
            scale: Multiplier on the negated update.
            norm_match: Project into the benign norm envelope afterwards.
            envelope_quantile: Which benign-norm quantile to match.
            share_direction: If True, all attackers negate the mean of the
                honest updates (perfect collusion, cosine = 1).
        """
        self.scale = float(scale)
        self.norm_match = bool(norm_match)
        self.envelope_quantile = float(envelope_quantile)
        self.share_direction = bool(share_direction)
        self.requires_honest_delta = not share_direction

    def craft(
        self,
        benign_deltas: Sequence[Mapping[str, torch.Tensor]],
        context: AttackContext,
    ) -> dict[str, torch.Tensor]:
        if self.share_direction:
            base = mean_delta(benign_deltas)
        else:
            if context.honest_delta is None:
                raise ValueError(
                    "sign_flip requires the attacker's own honest update; "
                    "give the ByzantineClient a dataset."
                )
            base = context.honest_delta

        flipped = scale_delta(base, -self.scale)
        if self.norm_match:
            flipped = project_into_norm_envelope(
                flipped, benign_deltas, self.envelope_quantile
            )
        return flipped


class IPMAttack(ByzantineAttack):
    r"""Inner Product Manipulation (Xie, Koyejo, Gupta, UAI 2020).

    Every colluding attacker sends

    .. math::  \Delta_{\text{byz}} = -\varepsilon \cdot
               \operatorname{mean}(\{\Delta_i : i \text{ benign}\}).

    With ``f`` attackers among ``n`` clients, the FedAvg mean becomes
    ``((n - f) - f\varepsilon)/n`` times the honest mean, so for
    ``epsilon > (n-f)/f`` the aggregate points *against* the descent
    direction while every individual update is small.  Small ``epsilon``
    (< 1) makes the attack invisible to norm-based screening and to Krum,
    which is the point of the paper.
    """

    name = "ipm"

    def __init__(self, epsilon: float = 0.5) -> None:
        self.epsilon = float(epsilon)

    def craft(
        self,
        benign_deltas: Sequence[Mapping[str, torch.Tensor]],
        context: AttackContext,
    ) -> dict[str, torch.Tensor]:
        mu = mean_delta(benign_deltas)
        return scale_delta(mu, -self.epsilon)


class ALIEAttack(ByzantineAttack):
    r"""A Little Is Enough (Baruch, Baruch, Goldberg, NeurIPS 2019).

    Coordinate-wise,

    .. math::  \Delta_{\text{byz}} = \mu - z \sigma,

    with ``mu`` and ``sigma`` the mean and standard deviation of the honest
    updates.  Because the perturbation is measured in units of the honest
    *variance*, the attack stays inside the empirical support of the honest
    population and therefore defeats distance-based defences (Krum,
    trimmed-mean) while still shifting the mean.

    ``z`` may be given explicitly (1.5 is the value used by most follow-up
    work) or computed with the paper's own rule ``z = Phi^{-1}(
    (n - f - s)/(n - f) )`` with ``s = floor(n/2 + 1) - f`` by passing
    ``z=None``.  The paper's rule is the *maximally stealthy* choice and can
    be near zero for large ``f``; we report which one was used.
    """

    name = "alie"

    def __init__(self, z: float | None = 1.5, n_clients: int | None = None) -> None:
        """
        Args:
            z: Standard-deviation multiplier.  ``None`` selects Baruch's
                ``z^max`` rule, which additionally needs ``n_clients``.
            n_clients: Total client count ``n`` (only used when ``z is None``).
        """
        self.z = None if z is None else float(z)
        self.n_clients = n_clients

    @staticmethod
    def baruch_z_max(n: int, f: int) -> float:
        """Baruch et al.'s ``z^max``; returns 0.0 when the rule degenerates."""
        from scipy.stats import norm as _norm

        if n <= f:
            return 0.0
        s = math.floor(n / 2 + 1) - f
        num = (n - f - s) / (n - f)
        if not (0.0 < num < 1.0):
            return 0.0
        return max(0.0, float(_norm.ppf(num)))

    def effective_z(self, benign_deltas_len: int, n_attackers: int) -> float:
        if self.z is not None:
            return self.z
        n = self.n_clients or (benign_deltas_len + n_attackers)
        return self.baruch_z_max(n, n_attackers)

    def craft(
        self,
        benign_deltas: Sequence[Mapping[str, torch.Tensor]],
        context: AttackContext,
    ) -> dict[str, torch.Tensor]:
        mu = mean_delta(benign_deltas)
        sigma = std_delta(benign_deltas)
        z = self.effective_z(len(benign_deltas), context.n_attackers)
        return {k: mu[k] - z * sigma[k] for k in mu}


class TargetedSafetyDegradationAttack(ByzantineAttack):
    r"""Targeted safety degradation with an explicit stealth constraint.

    The attacker trains on its own local data with the *unsafe* class
    relabelled *safe* (``default_label_map``).  The resulting update
    ``poisoned_delta`` is a legitimate SGD step for the wrong objective: it
    specifically raises the error rate on the unsafe class while leaving the
    safe class largely intact -- exactly the failure mode a safety gate is
    supposed to catch, and the one an accuracy-only metric hides.

    The direction is amplified around the honest update,

    .. math::  \Delta = \Delta_{\text{honest}} + b\,
               (\Delta_{\text{poison}} - \Delta_{\text{honest}}),

    and then projected into the benign norm envelope, so the update is
    *provably* within the honest norm distribution (assert it with
    :func:`attack_diagnostics`).  Boosting before clipping is what makes the
    attack effective at a norm a screen cannot flag.
    """

    name = "targeted_safety"
    requires_honest_delta = True
    requires_poisoned_delta = True

    def __init__(
        self,
        boost: float = 4.0,
        unsafe_label: int = 1,
        safe_label: int = 0,
        envelope_quantile: float = 1.0,
        enforce_envelope: bool = True,
    ) -> None:
        """
        Args:
            boost: Amplification of the poison direction before clipping.
            unsafe_label: Label of the class the attacker wants misclassified.
            safe_label: Label it should be flipped to.
            envelope_quantile: Benign-norm quantile used as the norm cap.
            enforce_envelope: Set False to study the non-stealthy variant.
        """
        self.boost = float(boost)
        self.unsafe_label = int(unsafe_label)
        self.safe_label = int(safe_label)
        self.envelope_quantile = float(envelope_quantile)
        self.enforce_envelope = bool(enforce_envelope)

    @property
    def default_label_map(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Relabel every ``unsafe_label`` example as ``safe_label``."""
        unsafe, safe = self.unsafe_label, self.safe_label

        def _map(targets: torch.Tensor) -> torch.Tensor:
            return torch.where(
                targets == unsafe,
                torch.full_like(targets, safe),
                targets,
            )

        return _map

    def craft(
        self,
        benign_deltas: Sequence[Mapping[str, torch.Tensor]],
        context: AttackContext,
    ) -> dict[str, torch.Tensor]:
        if context.poisoned_delta is None or context.honest_delta is None:
            raise ValueError(
                "targeted_safety needs both the honest and the poisoned local "
                "update; give the ByzantineClient a dataset."
            )
        honest = context.honest_delta
        poison = context.poisoned_delta
        raw = {
            k: honest[k] + self.boost * (poison[k] - honest[k]) for k in honest
        }
        if self.enforce_envelope:
            return project_into_norm_envelope(
                raw, benign_deltas, self.envelope_quantile
            )
        return raw


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


class LegacyGaussianNoiseAttack(ByzantineAttack):
    """The old repo's "attack", kept verbatim as a NEGATIVE CONTROL (F6).

    ``delta = -randn_like(param) * 0.01 * scale``.  It ignores
    ``benign_deltas`` entirely, colluding copies are mutually orthogonal, and
    under FedAvg it typically *improves* test accuracy.  Every table that
    reports a defence result must also report this row, so a reader can see
    what the old headline was measured against.
    """

    name = "legacy_gaussian"
    depends_on_benign_deltas = False
    is_degenerate_control = True
    is_coordinated = False

    def __init__(self, scale: float = 1.0, sigma: float = 0.01) -> None:
        self.scale = float(scale)
        self.sigma = float(sigma)

    def craft(
        self,
        benign_deltas: Sequence[Mapping[str, torch.Tensor]],
        context: AttackContext,
    ) -> dict[str, torch.Tensor]:
        # benign_deltas is accepted and deliberately unused -- that is the
        # documented defect this control exists to exhibit.
        g = context.torch_generator(stream=3)
        keys = _keys(benign_deltas)
        template = benign_deltas[0]
        return {
            k: -torch.randn(
                template[k].shape, generator=g, dtype=template[k].dtype
            )
            * self.sigma
            * self.scale
            for k in keys
        }


class NoAttack(ByzantineAttack):
    """Control: the "attacker" submits its honest update."""

    name = "no_attack"
    depends_on_benign_deltas = False
    is_degenerate_control = True
    requires_honest_delta = True
    is_coordinated = False

    def craft(
        self,
        benign_deltas: Sequence[Mapping[str, torch.Tensor]],
        context: AttackContext,
    ) -> dict[str, torch.Tensor]:
        if context.honest_delta is None:
            raise ValueError("no_attack control requires a dataset")
        return dict(context.honest_delta)


ATTACK_REGISTRY: dict[str, type[ByzantineAttack]] = {
    SignFlipAttack.name: SignFlipAttack,
    IPMAttack.name: IPMAttack,
    ALIEAttack.name: ALIEAttack,
    TargetedSafetyDegradationAttack.name: TargetedSafetyDegradationAttack,
    LegacyGaussianNoiseAttack.name: LegacyGaussianNoiseAttack,
    NoAttack.name: NoAttack,
}

#: Attacks that are real attacks, i.e. everything except the controls.
REAL_ATTACKS: tuple[str, ...] = (
    "sign_flip",
    "ipm",
    "alie",
    "targeted_safety",
)


def build_attack(name: str, **kwargs: Any) -> ByzantineAttack:
    """Instantiate a registered attack by name."""
    if name not in ATTACK_REGISTRY:
        raise ValueError(
            f"unknown attack {name!r}; known: {sorted(ATTACK_REGISTRY)}"
        )
    return ATTACK_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Diagnostics -- these numbers MUST appear in the paper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttackDiagnostics:
    """Measured properties of one round's attack.

    The three questions a reader needs answered about any Byzantine result:

    * Are the attackers actually *colluding*?  ``mean_pairwise_cosine_attackers``
      must be clearly positive; the F6 attack scored ``-0.007``.
    * Are they pointing *against* the honest signal?
      ``cosine_attacker_mean_vs_benign_mean`` should be negative for a
      direction attack.
    * Are they *stealthy*?  ``frac_within_benign_envelope`` is the fraction of
      attacker norms lying within ``[min, max]`` of the benign norms; a
      norm-threshold screen cannot flag those.
    """

    n_attackers: int
    n_benign: int
    mean_pairwise_cosine_attackers: float
    min_pairwise_cosine_attackers: float
    mean_pairwise_cosine_benign: float
    cosine_attacker_mean_vs_benign_mean: float
    mean_cosine_attacker_vs_benign: float
    attacker_norm_mean: float
    benign_norm_mean: float
    benign_norm_min: float
    benign_norm_max: float
    norm_ratio: float
    frac_within_benign_envelope: float
    #: Fraction of benign norms below the mean attacker norm.  0.5 means the
    #: attackers sit exactly at the middle of the honest norm distribution;
    #: 1.0 means they are at or above every honest update.
    attacker_norm_percentile_in_benign: float = float("nan")

    @property
    def is_stealthy(self) -> bool:
        """True iff every attacker norm lies inside the benign norm range."""
        return self.frac_within_benign_envelope >= 1.0 - 1e-12

    @property
    def is_coordinated(self) -> bool:
        """True iff colluding attackers are clearly positively correlated."""
        c = self.mean_pairwise_cosine_attackers
        return bool(c == c and c > 0.5)  # NaN-safe

    def as_dict(self) -> dict[str, Any]:
        d = {
            "n_attackers": self.n_attackers,
            "n_benign": self.n_benign,
            "mean_pairwise_cosine_attackers": self.mean_pairwise_cosine_attackers,
            "min_pairwise_cosine_attackers": self.min_pairwise_cosine_attackers,
            "mean_pairwise_cosine_benign": self.mean_pairwise_cosine_benign,
            "cosine_attacker_mean_vs_benign_mean": self.cosine_attacker_mean_vs_benign_mean,
            "mean_cosine_attacker_vs_benign": self.mean_cosine_attacker_vs_benign,
            "attacker_norm_mean": self.attacker_norm_mean,
            "benign_norm_mean": self.benign_norm_mean,
            "benign_norm_min": self.benign_norm_min,
            "benign_norm_max": self.benign_norm_max,
            "norm_ratio": self.norm_ratio,
            "frac_within_benign_envelope": self.frac_within_benign_envelope,
            "attacker_norm_percentile_in_benign": self.attacker_norm_percentile_in_benign,
            "is_stealthy": self.is_stealthy,
            "is_coordinated": self.is_coordinated,
        }
        return d


def _mean_pairwise_cosine(
    deltas: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[float, float]:
    if len(deltas) < 2:
        return float("nan"), float("nan")
    vals = []
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            c = cosine(deltas[i], deltas[j])
            if c == c:  # not NaN
                vals.append(c)
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.min(vals))


def attack_diagnostics(
    benign_deltas: Sequence[Mapping[str, torch.Tensor]],
    attacker_deltas: Sequence[Mapping[str, torch.Tensor]],
) -> AttackDiagnostics:
    """Compute the attack diagnostics for one round.

    Args:
        benign_deltas: Honest clients' updates.
        attacker_deltas: Byzantine clients' updates.
    """
    b_norms = np.array([delta_norm(d) for d in benign_deltas], dtype=float)
    a_norms = np.array([delta_norm(d) for d in attacker_deltas], dtype=float)

    att_mean_cos, att_min_cos = _mean_pairwise_cosine(attacker_deltas)
    ben_mean_cos, _ = _mean_pairwise_cosine(benign_deltas)

    if benign_deltas and attacker_deltas:
        cross = [
            cosine(a, b)
            for a in attacker_deltas
            for b in benign_deltas
        ]
        cross = [c for c in cross if c == c]
        mean_cross = float(np.mean(cross)) if cross else float("nan")
        agg_cos = cosine(mean_delta(attacker_deltas), mean_delta(benign_deltas))
    else:
        mean_cross = float("nan")
        agg_cos = float("nan")

    lo = float(b_norms.min()) if b_norms.size else float("nan")
    hi = float(b_norms.max()) if b_norms.size else float("nan")
    if a_norms.size and b_norms.size:
        # Relative tolerance: an update clipped to exactly the benign max
        # differs from it in the last float32 digit and must still count as
        # inside the envelope.
        rtol = 1e-6
        within = float(
            np.mean(
                (a_norms >= lo * (1 - rtol) - 1e-12)
                & (a_norms <= hi * (1 + rtol) + 1e-12)
            )
        )
        pct = float(np.mean(b_norms < float(a_norms.mean())))
    else:
        within = float("nan")
        pct = float("nan")

    b_mean = float(b_norms.mean()) if b_norms.size else float("nan")
    a_mean = float(a_norms.mean()) if a_norms.size else float("nan")
    ratio = a_mean / b_mean if b_mean and b_mean > 1e-30 else float("inf")

    return AttackDiagnostics(
        n_attackers=len(attacker_deltas),
        n_benign=len(benign_deltas),
        mean_pairwise_cosine_attackers=att_mean_cos,
        min_pairwise_cosine_attackers=att_min_cos,
        mean_pairwise_cosine_benign=ben_mean_cos,
        cosine_attacker_mean_vs_benign_mean=agg_cos,
        mean_cosine_attacker_vs_benign=mean_cross,
        attacker_norm_mean=a_mean,
        benign_norm_mean=b_mean,
        benign_norm_min=lo,
        benign_norm_max=hi,
        norm_ratio=ratio,
        frac_within_benign_envelope=within,
        attacker_norm_percentile_in_benign=pct,
    )


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------


@dataclass
class AttackConfig:
    """Configuration for a Byzantine attack scenario.

    Attributes:
        n_byzantine: Number of Byzantine clients ``f``.
        attack_type: Registry key.
        params: Keyword arguments forwarded to the attack constructor.
    """

    n_byzantine: int
    attack_type: str = "sign_flip"
    params: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ByzantineAttack | None:
        if self.n_byzantine <= 0:
            return None
        return build_attack(self.attack_type, **self.params)


@dataclass
class AttackScenario:
    """A named attack configuration for the results table."""

    name: str
    config: AttackConfig
    description: str = ""


def create_byzantine_clients(
    config: AttackConfig,
    datasets: Sequence[Any] | None = None,
    client_ids: Sequence[int] | None = None,
    *,
    lr: float = 0.05,
    local_epochs: int = 1,
    batch_size: int = 32,
    seed: int = 0,
) -> list[ByzantineClient]:
    """Instantiate the Byzantine clients described by ``config``.

    Args:
        config: Attack configuration.
        datasets: One local dataset per attacker.  Required for attacks with
            ``requires_honest_delta`` or ``requires_poisoned_delta``.
        client_ids: Explicit client ids (defaults to ``0..f-1``).
        lr, local_epochs, batch_size, seed: Local SGD settings for attackers
            that train.
    """
    attack = config.build()
    if attack is None:
        return []

    f = config.n_byzantine
    ids = list(client_ids) if client_ids is not None else list(range(f))
    if len(ids) != f:
        raise ValueError(f"need {f} client_ids, got {len(ids)}")

    needs_data = attack.requires_honest_delta or attack.requires_poisoned_delta
    if needs_data and datasets is None:
        raise ValueError(
            f"attack {attack.name!r} trains locally and requires datasets"
        )
    if datasets is not None and len(datasets) != f:
        raise ValueError(f"need {f} datasets, got {len(datasets)}")

    clients = []
    for i in range(f):
        clients.append(
            ByzantineClient(
                client_id=ids[i],
                attack=attack,
                dataset=None if datasets is None else datasets[i],
                lr=lr,
                local_epochs=local_epochs,
                batch_size=batch_size,
                seed=seed,
            )
        )
    return clients


def create_standard_attack_scenarios(
    n_total_clients: int = 10,
    byzantine_fraction: float = 0.2,
) -> list[AttackScenario]:
    """The attack sweep the paper must report, controls included.

    Order matters for the results table: the two controls come last so a
    reader sees the real attacks first and the F6 reproduction next to them.
    """
    f = max(1, int(round(n_total_clients * byzantine_fraction)))
    return [
        AttackScenario(
            name="none",
            config=AttackConfig(n_byzantine=0),
            description="No Byzantine clients (clean baseline)",
        ),
        AttackScenario(
            name="sign_flip",
            config=AttackConfig(
                n_byzantine=f,
                attack_type="sign_flip",
                params={"scale": 1.0, "norm_match": True},
            ),
            description="Negation of the attacker's own honest gradient",
        ),
        AttackScenario(
            name="ipm",
            config=AttackConfig(
                n_byzantine=f, attack_type="ipm", params={"epsilon": 0.5}
            ),
            description="Inner product manipulation (Xie et al. 2020)",
        ),
        AttackScenario(
            name="alie",
            config=AttackConfig(
                n_byzantine=f, attack_type="alie", params={"z": 1.5}
            ),
            description="A little is enough (Baruch et al. 2019)",
        ),
        AttackScenario(
            name="targeted_safety",
            config=AttackConfig(
                n_byzantine=f,
                attack_type="targeted_safety",
                params={"boost": 4.0},
            ),
            description=(
                "Unsafe-class poisoning, projected into the benign norm "
                "envelope"
            ),
        ),
        AttackScenario(
            name="legacy_gaussian_CONTROL",
            config=AttackConfig(
                n_byzantine=f, attack_type="legacy_gaussian"
            ),
            description=(
                "NEGATIVE CONTROL: the old repo's zero-mean Gaussian 'attack' "
                "(finding F6). Typically improves accuracy."
            ),
        ),
        AttackScenario(
            name="no_attack_CONTROL",
            config=AttackConfig(n_byzantine=f, attack_type="no_attack"),
            description="CONTROL: adversaries behave honestly",
        ),
    ]


def create_attack_scenario(
    n_total_clients: int,
    config: AttackConfig,
    datasets: Sequence[Any] | None = None,
    **kwargs: Any,
) -> list[ByzantineClient]:
    """Backwards-compatible wrapper around :func:`create_byzantine_clients`.

    The attacker ids are the LAST ``config.n_byzantine`` client indices, which
    is the convention the experiment runners already use.
    """
    f = config.n_byzantine
    ids = list(range(n_total_clients - f, n_total_clients))
    return create_byzantine_clients(
        config, datasets=datasets, client_ids=ids, **kwargs
    )
