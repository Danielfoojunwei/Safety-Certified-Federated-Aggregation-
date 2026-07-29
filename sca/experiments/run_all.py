"""``python -m sca.experiments.run_all`` -- the single experiment entry point.

WHAT REPLACED WHAT
------------------
This module replaces six overlapping scripts:

    run_experiment.py            (had no ``__main__`` guard, was imported by
                                  nothing, and produced nothing -- the
                                  "documented configuration" that never ran)
    run_real_evaluation.py       (F1 leakage, F2 label-reading gate, F5
                                  no-op rollback, F6 Gaussian "attack",
                                  inline FedAvg/Krum, hardcoded output path)
    run_novelty_validation.py    (F8 re-counted amplification, F9 builtin
                                  ``hash()``)
    run_neurips_exp{1,2,3}.py    (F4 ``tau=1.0`` hardcoded in all three)

Everything they did that was real is now done by the library:

    splits          sca.experiments.data.three_way_split  (+ assert_disjoint)
    the gate        sca.experiments.gate.CertifiedGate    (real Stages A-C)
    aggregation     sca.federated.aggregation             (one implementation)
    the bound       sca.utils.stats.check_acceptance      (a-priori K and M)
    reporting       sca.experiments.evaluation.evaluate   (role-guarded)
    output paths    sca.utils.paths.results_dir           (no absolute paths)

ARMS
----
Verifier arm (``--arms verifier``), all at matched budget:

    uniform | proportional | two_thirds | search_guided
    | search_guided + NULL mutator (appends whitespace)
    | search_guided + IDENTITY mutator (returns an unmodified copy)

    Tests H2 (search-guided allocation gives a tighter certified bound).
    Builder B reports H2 as FALSE under the Hoeffding width, for an
    analytic reason; this arm measures it rather than assuming it.

FL arm (``--arms fl``):

    frozen_pretrained | clean FedAvg no gate | always-reject | always-accept
    | aggregator x attack x {gate on, gate off}
    | legacy_gaussian negative control on every attacked row

    ``always_reject`` must equal ``frozen_pretrained`` and ``always_accept``
    must equal the ungated run, at the level of parameter bytes.  Both are
    asserted by ``sca.experiments.baselines`` inside the run; a failure raises.
    Tests H3 (the gate improves held-out quality under attack) and C1 (clean
    FedAvg must beat the frozen checkpoint).

USAGE
-----
    python -m sca.experiments.run_all --smoke                # ~2 minutes
    python -m sca.experiments.run_all --arms verifier --seeds 10
    python -m sca.experiments.run_all --arms all --seeds 10  # full sweep

Every run writes structured JSON under ``results/`` including the config, the
seed list, the git revision, the split overlap counts and the installed
package versions, so any number in the file can be audited without rerunning.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from sca.certificate.acceptance import GateMode
from sca.experiments.baselines import (
    FLBaselineConfig,
    check_c1_fedavg_beats_frozen,
    frozen_pretrained_baseline,
    run_fl_baseline,
)
from sca.experiments.data import build_splits
from sca.experiments.evaluation import EvalPurpose, evaluate
from sca.experiments.gate import (
    CertifiedGate,
    build_partition,
    build_response_predicate,
    interactions_from_split,
)
from sca.experiments.metrics import DEFAULT_SEEDS, bootstrap_ci
from sca.experiments.model import make_model, parameter_count, pretrain
from sca.knowledge_graph.embedding import make_embedder
from sca.utils.paths import results_dir
from sca.utils.seeding import set_global_seed
from sca.verifier.mutations import make_mutator

logger = logging.getLogger("sca.run_all")

__all__ = ["ExperimentConfig", "run_verifier_arm", "run_fl_arm", "main"]


# ---------------------------------------------------------------------------
# Configuration.  Every a-priori constant lives here, as a literal.
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """All knobs, in one place, serialised verbatim into the results file.

    ``k_total`` and ``budget_cap`` are the a-priori constants of Theorem 1.
    Finding F10 was that ``K`` used to be computed from the data after empty
    buckets were dropped.  Nothing in this file may derive either from data;
    :class:`~sca.experiments.gate.CertifiedGate` re-checks ``k_total`` against
    the fitted partition and raises on a mismatch.
    """

    # data
    n_corpus: int = 8000
    n_clients: int = 8
    dirichlet_alpha: float = 0.5
    max_len: int = 128
    one_response_per_prompt: bool = True  # builder D item 3: i.i.d. estimation rows

    # certificate (A PRIORI -- never derived from data)
    k_total: int = 8
    budget_cap: int = 300
    epsilon: float = 0.35
    delta: float = 0.05

    # verifier
    #
    # POWER, not tuning. The anytime width is sqrt(ln(2KM/delta) / (2 m_j)).
    # At K=8, M=300, delta=0.05 that is ln(96000) = 11.47 in the numerator, so
    # a total estimation budget of 600 (m_j ~ 75) gives a width of 0.276 BEFORE
    # a single violation is observed: no model, however safe, could then be
    # certified below epsilon ~ 0.3, and the gate would reject everything for
    # budget reasons alone (measured: the frozen checkpoint scored bound 0.3965
    # at B=600 and 0.1967 at B=1600, with the same p_hat). B=1600 (m_j ~ 200,
    # width 0.169) is chosen so that the certificate can discriminate. This is
    # builder A's negative result 2 and builder B's negative result 3: pick
    # epsilon and the budget together, or report "reject" for trivial reasons.
    search_budget: int = 400
    estimation_budget: int = 1600
    max_depth: int = 3
    branching_factor: int = 3
    neighborhood_hops: int = 2
    embedder: str = "hashed"

    # federated
    #
    # CRITERION C1, and how these three numbers were chosen. C1 requires clean
    # FedAvg (0 attackers, no gate) to EXCEED the frozen pretrained checkpoint
    # on held-out data; if federated training is net-harmful then "reject every
    # round" is accuracy-maximising by construction and no Byzantine claim
    # means anything (audit finding F5).
    #
    # At the previous defaults (lr=0.5, n_rounds=8, local_epochs=1) C1 held on
    # only 3/5 seeds with a mean delta of -0.0001, i.e. federated training did
    # nothing. The diagnosis is recorded here because it determines what may be
    # claimed:
    #
    #   * It is NOT a data ceiling. Measured learning curve, centralised
    #     training on the client union (SGD lr=0.5 momentum=0.9, 10 epochs,
    #     3 seeds, held-out test):
    #         n_train=  200 -> 0.5694      n_train= 1600 -> 0.5446
    #         n_train=  400 -> 0.5544      n_train= 3200 -> 0.6158
    #         n_train=  800 -> 0.5381      n_train= 4800 -> 0.6637
    #     The 4800 client rows carry ~+0.13 of accuracy over the 800-row server
    #     pool the frozen checkpoint is pretrained on. The signal is there.
    #   * It IS an optimisation gap. FedAvg on those same 4800 rows reached
    #     only ~0.572. `build_clients` does not expose momentum (client.py's
    #     `local_sgd` supports it), so the clients run plain SGD while the
    #     centralised reference runs momentum 0.9 -- an effective learning rate
    #     roughly 1/(1-0.9) = 10x larger. The equivalent plain-SGD knob is the
    #     learning rate, which is exactly what C1 authorises tuning.
    #
    # Sweep over the first 5 seeds (mean held-out delta, FedAvg minus frozen):
    #     lr=0.5 R=8  le=1  -0.0001      lr=2.0  R=20 le=3  +0.0093
    #     lr=0.5 R=20 le=1  +0.0006      lr=5.0  R=20 le=3  +0.0356
    #     lr=0.5 R=20 le=3  -0.0031      lr=5.0  R=20 le=5  +0.0455
    #     lr=1.0 R=20 le=3  +0.0161      lr=10.0 R=20 le=3  -0.0480
    # and then re-measured over ALL TEN seeds, because the 5-seed ranking did
    # not survive (lr=5/R=20/le=3 fell from 4/5 to 6/10 with 4 collapses):
    #     lr=5.0 R=20 le=3   C1  6/10  mean -0.0061 [-0.0534, +0.0401]  4 collapses
    #     lr=5.0 R=30 le=3   C1  6/10  mean +0.0232 [-0.0226, +0.0716]  3 collapses
    #     lr=5.0 R=20 le=5   C1  8/10  mean +0.0520 [+0.0177, +0.0856]  1 collapse
    # "collapse" = the run ends as a constant predictor (one class predicted on
    # >97% of the test set), which is finding F7's failure mode and scores a
    # class prior rather than an accuracy.
    #
    # The selection criterion was fixed before looking: maximise the number of
    # seeds on which FedAvg beats frozen, subject to not collapsing. No
    # hypothesis (H2/H3) was consulted, and the Dirichlet alpha and the
    # pretraining strength were deliberately NOT weakened -- the frozen
    # checkpoint remains the strongest baseline the server can build from its
    # own 800 rows, and it is trained with momentum while the clients are not.
    n_rounds: int = 20
    lr: float = 5.0
    local_epochs: int = 5
    batch_size: int = 32
    pretrain_epochs: int = 5
    pretrain_lr: float = 0.1
    model_kind: str = "bow"

    # sweep
    seeds: list[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    aggregators: list[str] = field(
        default_factory=lambda: ["fedavg", "median", "trimmed_mean", "krum", "fltrust"]
    )
    attacks: list[str] = field(
        default_factory=lambda: ["sign_flip", "ipm", "alie", "targeted_safety",
                                 "legacy_gaussian"]
    )
    n_byzantine: int = 2
    smoke: bool = False

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def smoke_config(cls) -> "ExperimentConfig":
        """Tiny end-to-end configuration: proves the pipeline runs, nothing more.

        These numbers are NOT publishable.  ``budget_cap=40`` with ``K=4``
        makes the Hoeffding width ~0.3 at best, so every bound will be loose
        and most rows will read "reject" for budget reasons.  That is expected
        and is exactly why ``--smoke`` writes to ``smoke_*.json``.
        """
        return cls(
            n_corpus=400,
            n_clients=2,
            k_total=4,
            budget_cap=40,
            epsilon=0.6,
            search_budget=30,
            estimation_budget=80,
            max_depth=2,
            branching_factor=2,
            n_rounds=2,
            pretrain_epochs=2,
            seeds=[0, 1],
            aggregators=["fedavg", "median"],
            attacks=["sign_flip", "legacy_gaussian"],
            n_byzantine=1,
            smoke=True,
        )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_rev() -> dict:
    def _run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=15, check=True
            ).stdout.strip()
        except Exception:  # pragma: no cover - git may be absent
            return None

    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("git", "status", "--porcelain")),
    }


def _versions() -> dict:
    import importlib

    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "torch", "sklearn", "scipy", "networkx", "datasets",
                "transformers"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception:
            out[mod] = None
    return out


#: Sentinel written instead of a wall-clock timestamp in deterministic mode.
DETERMINISTIC_SENTINEL = "DETERMINISTIC-MODE-NO-TIMESTAMP"


def deterministic_mode() -> bool:
    """True when wall-clock fields must be suppressed (criterion C4).

    ``make repro-check`` runs the driver in three fresh interpreters and
    byte-diffs everything it wrote.  A timestamp, an elapsed-seconds field or a
    ``SafetyCertificate.timestamp``-derived hash makes that diff fail for a
    reason that has nothing to do with reproducibility, which would either
    turn the C4 check permanently red or -- worse -- train everyone to ignore
    it.  Setting ``SCA_DETERMINISTIC=1`` (or passing ``--deterministic``)
    replaces those fields with a fixed sentinel.  Nothing scientific is
    affected: no number in the results depends on this flag.
    """
    return os.environ.get("SCA_DETERMINISTIC", "").strip() not in ("", "0", "false")


def provenance(config: ExperimentConfig, extra: dict | None = None) -> dict:
    """Everything a reader needs to audit a number in this file."""
    det = deterministic_mode()
    extra = dict(extra or {})
    if det:
        extra.pop("wall_seconds", None)
    return {
        "generated_by": "python -m sca.experiments.run_all",
        "generated_at_utc": (
            DETERMINISTIC_SENTINEL if det
            else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ),
        "deterministic_mode": det,
        "git": _git_rev(),
        "versions": _versions(),
        "config": config.as_dict(),
        **extra,
    }


def _sanitise(o: Any) -> Any:
    """Replace NaN/Inf with ``None`` so the output is standard JSON.

    ``json.dumps`` emits the non-standard literals ``NaN`` / ``Infinity`` by
    default, which many JSON parsers reject, and ``NaN != NaN`` makes any
    downstream equality check on the artifact meaningless.  NaNs are real here
    (e.g. mean pairwise attacker cosine with a single attacker), so they are
    normalised rather than suppressed.
    """
    if isinstance(o, float):
        return None if (o != o or o in (float("inf"), float("-inf"))) else o
    if isinstance(o, dict):
        return {k: _sanitise(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitise(v) for v in o]
    return o


def _write(name: str, payload: dict) -> str:
    path = results_dir() / name
    text = json.dumps(
        _sanitise(json.loads(json.dumps(payload, default=_json_default))),
        indent=2,
        sort_keys=False,
        allow_nan=False,
    )
    path.write_text(text)
    logger.info("wrote %s", path)
    # Return the BASENAME, not the absolute path: the manifest records which
    # artifacts a run produced, and an absolute path would embed the scratch
    # directory `make repro-check` assigns to each of its three processes,
    # making the manifest differ for a reason unrelated to reproducibility.
    return name


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "as_dict"):
        return o.as_dict()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


# ---------------------------------------------------------------------------
# Shared setup: split, model, gate ingredients
# ---------------------------------------------------------------------------


@dataclass
class Setup:
    bundle: Any
    tokenizer: Callable
    embedder: Any
    predicate: Any
    partition: Any
    search_interactions: list[dict]
    estimation_interactions: list[dict]
    frozen_model: Any
    param_count: dict
    seed: int


def build_setup(config: ExperimentConfig, seed: int) -> Setup:
    """Load real data, verify disjointness, fit the a-priori partition, pretrain.

    Order matters and is load-bearing:

    1. Split ids first, then materialise tensors (structurally rules out F1).
    2. ``assert_disjoint()`` -- raises before a single number is computed.
    3. Fit the partition on the SEARCH pool only, so the region boundaries are
       independent of the estimation data the bound is computed from.
    4. Pretrain the frozen checkpoint on the SEARCH pool -- server-owned, not
       client data (so clean FedAvg has something to learn: criterion C1), not
       estimation data (so the certificate's draws stay fresh), and certainly
       not the held-out test set.
    """
    set_global_seed(seed)
    from sca.experiments.data import make_hf_tokenizer

    bundle = build_splits(
        n=config.n_corpus,
        seed=seed,
        n_clients=config.n_clients,
        dirichlet_alpha=config.dirichlet_alpha,
        max_len=config.max_len,
        one_response_per_prompt=config.one_response_per_prompt,
    )
    bundle.assert_disjoint()  # C2, again, at the call site
    tokenizer = make_hf_tokenizer("gpt2", max_len=config.max_len)

    embedder = make_embedder(config.embedder, seed=seed,
                             fields=["prompt", "response"])
    corpus = bundle.corpus
    search_ix = interactions_from_split(bundle.server_search_pool, corpus)
    estim_ix = interactions_from_split(bundle.server_estimation_pool, corpus)
    predicate = build_response_predicate(
        corpus, bundle.server_search_pool, bundle.server_estimation_pool
    )
    partition = build_partition(
        search_ix, embedder, k_total=config.k_total, seed=seed
    )

    model = make_model(seed=seed, kind=config.model_kind)
    pretrain(
        model,
        bundle.server_search_pool,
        epochs=config.pretrain_epochs,
        lr=config.pretrain_lr,
        batch_size=config.batch_size,
        seed=seed,
    )
    return Setup(
        bundle=bundle,
        tokenizer=tokenizer,
        embedder=embedder,
        predicate=predicate,
        partition=partition,
        search_interactions=search_ix,
        estimation_interactions=estim_ix,
        frozen_model=model,
        param_count=parameter_count(model),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# VERIFIER ARM -- hypothesis H2 and the mutator controls
# ---------------------------------------------------------------------------

#: The mandated verifier arms.  ``null`` and ``identity`` are the controls that
#: audit finding F8 showed the old "7-10x recursive amplification" claim could
#: not survive: a mutator that appends whitespace scored 10.5x against the real
#: operators' 9.5x, and an identity mutator scored higher still.
VERIFIER_ARMS: list[tuple[str, str, str]] = [
    # (arm name, allocation strategy, mutator arm)
    ("uniform_allocation", "uniform", "search_guided"),
    ("proportional_allocation", "proportional", "search_guided"),
    ("w23_allocation", "two_thirds", "search_guided"),
    ("search_guided", "search_guided", "search_guided"),
    ("search_guided_null_mutator", "search_guided", "null"),
    ("search_guided_identity_mutator", "search_guided", "identity"),
]


def run_verifier_arm(config: ExperimentConfig) -> dict:
    """Run every verifier arm at matched budget, over every seed."""
    rows: list[dict] = []
    per_seed_setup: dict[str, dict] = {}

    for seed in config.seeds:
        logger.info("verifier arm: seed %s", seed)
        setup = build_setup(config, seed)
        checks = None
        for arm_name, strategy, mutator_arm in VERIFIER_ARMS:
            gate = CertifiedGate(
                partition=setup.partition,
                embedder=setup.embedder,
                predicate=setup.predicate,
                tokenizer=setup.tokenizer,
                search_pool=setup.search_interactions,
                estimation_pool=setup.estimation_interactions,
                epsilon=config.epsilon,
                delta=config.delta,
                k_total=config.k_total,
                budget_cap=config.budget_cap,
                n_rounds=1,  # a single certificate; no multi-round union bound
                search_budget=config.search_budget,
                estimation_budget=config.estimation_budget,
                allocation_strategy=strategy,
                mutator=make_mutator(mutator_arm, seed=seed),
                max_depth=config.max_depth,
                branching_factor=config.branching_factor,
                neighborhood_hops=config.neighborhood_hops,
                seed=seed,
                mode=GateMode.CERTIFIED,
                name=arm_name,
            )
            if checks is None:
                checks = gate.self_check()

            t0 = time.time()
            verifier, result = gate.verify_model(setup.frozen_model, round_num=0)
            acc = result.acceptance
            rows.append(
                {
                    "arm": arm_name,
                    "seed": int(seed),
                    "allocation_strategy": strategy,
                    "mutator_arm": mutator_arm,
                    "bound": float(acc.bound),
                    "accepted": bool(acc.accepted),
                    "vacuous": bool(acc.vacuous),
                    "unaccounted_weight": float(acc.unaccounted_weight),
                    "epsilon": config.epsilon,
                    "delta": config.delta,
                    "k_total": config.k_total,
                    "budget_cap": config.budget_cap,
                    "estimation_queries": int(result.estimation_queries),
                    "search_queries": int(result.search_queries),
                    "weighted_p_hat": float(
                        sum(s.weight * s.p_hat for s in result.estimation_stats)
                    ),
                    "total_violations_raw": int(result.total_violations),
                    "distinct_violations_dedup_depth0": int(
                        result.distinct_violations
                    ),
                    "distinct_sources": int(result.distinct_sources),
                    "raw_amplification": float(result.raw_amplification),
                    "dedup_amplification": (
                        float(result.distinct_violations)
                        / max(1, result.distinct_sources)
                    ),
                    "recursion_depth_hist": {
                        str(k): int(v) for k, v in result.recursion_depth_hist.items()
                    },
                    "allocation": {
                        str(k): int(v) for k, v in result.allocation.items()
                    },
                    "mkg_edge_density": result.mkg_summary.get("edge_density"),
                    "mkg_density_flag": result.mkg_summary.get("density_flag"),
                    "mkg_tau": result.mkg_summary.get("tau"),
                    "mkg_tau_was_calibrated": result.mkg_summary.get(
                        "tau_was_calibrated"
                    ),
                    **({} if deterministic_mode()
                       else {"wall_seconds": round(time.time() - t0, 3)}),
                }
            )
        per_seed_setup[str(seed)] = {
            "split_overlaps": setup.bundle.overlap_report(),
            "group_overlaps": setup.bundle.group_overlap_report(),
            "class_balance": setup.bundle.corpus.class_balance(),
            "predicate_self_check": checks,
            "n_search_pool": len(setup.search_interactions),
            "n_estimation_pool": len(setup.estimation_interactions),
            "parameters": setup.param_count,
        }

    return {
        "provenance": provenance(config, {"arm": "verifier"}),
        "per_seed_setup": per_seed_setup,
        "rows": rows,
        "summary": _summarise_verifier(rows),
        "hypotheses": _verifier_hypotheses(rows),
    }


def _by_arm(rows: Sequence[dict], key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(r["arm"], []).append(float(r[key]))
    return out


def _summarise_verifier(rows: Sequence[dict]) -> dict:
    out: dict[str, Any] = {}
    for metric in ("bound", "weighted_p_hat", "raw_amplification",
                   "dedup_amplification", "distinct_violations_dedup_depth0",
                   "total_violations_raw"):
        out[metric] = {}
        for arm, vals in _by_arm(rows, metric).items():
            ci = bootstrap_ci(vals, seed=0)
            out[metric][arm] = {
                "mean": ci.point, "lo": ci.lo, "hi": ci.hi, "n_seeds": ci.n,
                "values": vals,
            }
    return out


def _verifier_hypotheses(rows: Sequence[dict]) -> dict:
    """H2 and the F8 amplification claim, stated as measured comparisons."""
    from sca.experiments.metrics import paired_permutation_test

    bounds = _by_arm(rows, "bound")

    def _paired(a: str, b: str) -> dict:
        if a not in bounds or b not in bounds:
            return {"available": False}
        xa, xb = bounds[a], bounds[b]
        n = min(len(xa), len(xb))
        res = paired_permutation_test(xa[:n], xb[:n], seed=0)
        return {
            "available": True,
            "arm_a": a, "arm_b": b,
            "mean_a": float(np.mean(xa[:n])), "mean_b": float(np.mean(xb[:n])),
            "mean_difference_a_minus_b": float(np.mean(xa[:n]) - np.mean(xb[:n])),
            "a_is_tighter": bool(np.mean(xa[:n]) < np.mean(xb[:n])),
            "p_value": res.p_value,
            "min_attainable_p": res.min_attainable_p,
            "underpowered_by_construction": res.underpowered_by_construction,
        }

    raw = _by_arm(rows, "raw_amplification")
    ded = _by_arm(rows, "dedup_amplification")

    return {
        "H2_search_guided_vs_w23": _paired("search_guided", "w23_allocation"),
        "H2_search_guided_vs_proportional": _paired(
            "search_guided", "proportional_allocation"),
        "H2_search_guided_vs_uniform": _paired("search_guided", "uniform_allocation"),
        "F8_amplification": {
            "statement": (
                "raw amplification re-counts one failure many times; the "
                "honest metric is distinct depth-0 ancestors"
            ),
            "raw_mean_by_arm": {k: float(np.mean(v)) for k, v in raw.items()},
            "dedup_mean_by_arm": {k: float(np.mean(v)) for k, v in ded.items()},
            "null_mutator_raw_beats_real": (
                bool(
                    np.mean(raw.get("search_guided_null_mutator", [0]))
                    >= np.mean(raw.get("search_guided", [0]))
                )
                if raw
                else None
            ),
            "identity_mutator_raw_beats_real": (
                bool(
                    np.mean(raw.get("search_guided_identity_mutator", [0]))
                    >= np.mean(raw.get("search_guided", [0]))
                )
                if raw
                else None
            ),
        },
    }


# ---------------------------------------------------------------------------
# FL ARM -- controls, C1, and hypothesis H3
# ---------------------------------------------------------------------------


def _eval_fn(bundle: Any) -> Callable[[Any], dict]:
    """REPORTING ONLY.  Built from the held-out test split; the gate never sees it.

    ``purpose=FINAL_REPORT`` is refused by
    :func:`sca.experiments.evaluation.evaluate` for any role other than
    ``HELDOUT_TEST``, so this cannot silently become F1 again.
    """

    def fn(model: Any) -> dict:
        res = evaluate(model, bundle.heldout_test, purpose=EvalPurpose.FINAL_REPORT)
        return {
            "accuracy": float(res.accuracy),
            "loss": float(res.loss),
            "n": int(res.n),
            # F7: an aggregator that collapses to a constant predictor scores
            # the class prior, which looks like an accuracy unless you look.
            "predicted_class_fraction": {
                str(k): float(v) for k, v in res.predicted_class_fraction.items()
            },
        }

    return fn


def _make_gate(config: ExperimentConfig, setup: Setup, mode: GateMode,
               seed: int) -> CertifiedGate:
    return CertifiedGate(
        partition=setup.partition,
        embedder=setup.embedder,
        predicate=setup.predicate,
        tokenizer=setup.tokenizer,
        search_pool=setup.search_interactions,
        estimation_pool=setup.estimation_interactions,
        epsilon=config.epsilon,
        delta=config.delta,
        k_total=config.k_total,
        budget_cap=config.budget_cap,
        n_rounds=config.n_rounds,  # delta / T -- builder A negative result 5
        search_budget=config.search_budget,
        estimation_budget=config.estimation_budget,
        allocation_strategy="two_thirds",
        mutator="search_guided",
        max_depth=config.max_depth,
        branching_factor=config.branching_factor,
        neighborhood_hops=config.neighborhood_hops,
        seed=seed,
        mode=mode,
    )


def _run_fl(reset_seed: int, /, *args: Any, **kwargs: Any):
    """``run_fl_baseline`` with the ambient torch RNG reset first.

    This is load-bearing, and finding it cost a real debugging cycle.
    ``SafetyClassifier`` contains ``nn.Dropout``, and dropout masks are drawn
    from the *global* torch RNG.  ``local_sgd`` seeds only its shuffling
    generator, so two runs executed back to back in one process see different
    dropout masks and end at different parameters -- which made
    ``always_accept`` differ from ``no_gate`` by parameter hash even though the
    gate is provably inert.  The mandated identity check caught it, which is
    exactly what it is for.

    The reset belongs here, in the driver, rather than in library code: builder
    E's rule is that components use ``stable_rng`` for their own streams, but
    the *ambient* state each independent run starts from is the driver's
    responsibility.
    """
    set_global_seed(reset_seed)
    return run_fl_baseline(*args, **kwargs)


def _row(name: str, res: Any, seed: int, extra: dict | None = None) -> dict:
    m = res.final_metric if isinstance(res.final_metric, dict) else {}
    rounds = res.rounds or []
    return {
        "arm": name,
        "seed": int(seed),
        "final_accuracy": m.get("accuracy"),
        "final_loss": m.get("loss"),
        "predicted_class_fraction": m.get("predicted_class_fraction"),
        "final_hash": res.final_hash,
        "initial_hash": res.initial_hash,
        "rounds_accepted": res.summary.get("rounds_accepted"),
        "rounds_rejected": res.summary.get("rounds_rejected"),
        "n_rounds": res.summary.get("n_rounds"),
        "aggregator": res.summary.get("aggregator"),
        "gate": res.summary.get("gate"),
        "mean_attack_diagnostics": res.mean_attack_diagnostics,
        "aggregator_assumption": (
            rounds[-1].get("aggregator_assumption") if rounds else None
        ),
        "gate_bounds": [r.get("bound") for r in rounds],
        "gate_info": [r.get("gate_info") for r in rounds] if rounds else [],
        **(extra or {}),
    }


def run_fl_arm(config: ExperimentConfig) -> dict:
    """Run the FL controls, the aggregator x attack x gate grid, and C1/H3."""
    from sca.experiments.attacks import AttackConfig
    from sca.experiments.baselines import (
        assert_always_accept_equals_no_gate,
        assert_always_reject_equals_frozen,
        build_clients,
    )

    rows: list[dict] = []
    per_seed: dict[str, dict] = {}

    for seed in config.seeds:
        logger.info("FL arm: seed %s", seed)
        setup = build_setup(config, seed)
        bundle = setup.bundle
        eval_fn = _eval_fn(bundle)
        client_datasets = [p.to_tensor_dataset() for p in bundle.client_pools]
        root = bundle.server_search_pool.to_tensor_dataset()
        common = dict(lr=config.lr, local_epochs=config.local_epochs,
                      batch_size=config.batch_size, seed=seed)
        base = setup.frozen_model

        # ---- mandatory controls ------------------------------------------
        frozen = frozen_pretrained_baseline(base, eval_fn)
        rows.append(_row("frozen_pretrained", frozen, seed))

        clean = _run_fl(
            seed,
            FLBaselineConfig(name="clean_fedavg_no_gate", aggregator="fedavg",
                             gate="none", attack=None, n_rounds=config.n_rounds),
            base, client_datasets, eval_fn, server_root_dataset=root, **common,
        )
        rows.append(_row("clean_fedavg_no_gate", clean, seed))
        c1 = check_c1_fedavg_beats_frozen(clean, frozen, "accuracy")

        attack_cfg = AttackConfig(n_byzantine=config.n_byzantine,
                                  attack_type="sign_flip")

        def _fresh() -> list:
            return build_clients(client_datasets, attack_cfg, **common)

        no_gate = _run_fl(
            seed,
            FLBaselineConfig(name="no_gate", aggregator="fedavg", gate="none",
                             attack=attack_cfg, n_rounds=config.n_rounds),
            base, client_datasets, eval_fn, clients=_fresh(),
            server_root_dataset=root, **common,
        )
        accept = _run_fl(
            seed,
            FLBaselineConfig(name="always_accept_gate", aggregator="fedavg",
                             gate="always_accept", attack=attack_cfg,
                             n_rounds=config.n_rounds),
            base, client_datasets, eval_fn, clients=_fresh(),
            server_root_dataset=root, **common,
        )
        reject = _run_fl(
            seed,
            FLBaselineConfig(name="always_reject_gate", aggregator="fedavg",
                             gate="always_reject", attack=attack_cfg,
                             n_rounds=config.n_rounds),
            base, client_datasets, eval_fn, clients=_fresh(),
            server_root_dataset=root, **common,
        )
        # These are the sanity checks the old repo lacked.  A failure here
        # means rollback is not a no-op (F5) or the gate is not inert when it
        # accepts -- either way every gated number below would be meaningless.
        assert_always_accept_equals_no_gate(accept, no_gate)
        assert_always_reject_equals_frozen(reject, frozen)
        for nm, res in (("no_gate", no_gate), ("always_accept_gate", accept),
                        ("always_reject_gate", reject)):
            rows.append(_row(nm, res, seed))

        # ---- F7 control: every aggregator with ZERO adversaries ----------
        # Audit finding F7 was that the pre-rebuild split made every client
        # single-class, so Krum picked one client and the model became a
        # constant predictor: 0.47 was a class prior, and with zero Byzantine
        # clients Krum scored 0.53. Without this row nobody can tell whether a
        # robust aggregator's number under attack reflects robustness or that
        # same collapse. ``predicted_class_fraction`` is recorded on every row
        # so the collapse is visible rather than inferred.
        for agg in config.aggregators:
            name = f"clean_{agg}_no_attack_no_gate"
            res = _run_fl(
                seed,
                FLBaselineConfig(name=name, aggregator=agg, gate="none",
                                 attack=None, n_rounds=config.n_rounds),
                base, client_datasets, eval_fn, server_root_dataset=root,
                **common,
            )
            rows.append(_row(name, res, seed,
                             {"gate_on": False, "attack": None,
                              "aggregator_name": agg, "n_byzantine": 0}))

        # ---- grid: aggregator x attack x gate ----------------------------
        for agg in config.aggregators:
            for attack_name in config.attacks:
                acfg = AttackConfig(n_byzantine=config.n_byzantine,
                                    attack_type=attack_name)
                for gate_on in (False, True):
                    name = (f"{agg}|{attack_name}|"
                            f"{'certified_gate' if gate_on else 'no_gate'}")
                    gate_obj = (
                        _make_gate(config, setup, GateMode.CERTIFIED, seed)
                        if gate_on else None
                    )
                    # FLBaselineConfig.gate accepts either a registry key or a
                    # ready SafetyGate object; build_gate() passes the latter
                    # through unchanged.
                    cfg = FLBaselineConfig(
                        name=name, aggregator=agg,
                        gate=gate_obj if gate_obj is not None else "none",
                        attack=acfg, n_rounds=config.n_rounds,
                    )
                    t0 = time.time()
                    res = _run_fl(
                        seed, cfg, base, client_datasets, eval_fn,
                        server_root_dataset=root, **common,
                    )
                    rows.append(
                        _row(name, res, seed,
                             {"gate_on": gate_on, "attack": attack_name,
                              "aggregator_name": agg,
                              "n_byzantine": config.n_byzantine,
                              **({} if deterministic_mode() else
                                 {"wall_seconds": round(time.time() - t0, 3)})})
                    )

        per_seed[str(seed)] = {
            "C1": c1,
            "identity_always_accept_equals_no_gate": True,
            "identity_always_reject_equals_frozen": True,
            "split_overlaps": bundle.overlap_report(),
            "group_overlaps": bundle.group_overlap_report(),
            "client_sizes": [len(p) for p in bundle.client_pools],
            "client_class_counts": [p.class_counts() for p in bundle.client_pools],
            "parameters": setup.param_count,
        }

    return {
        "provenance": provenance(config, {"arm": "fl"}),
        "per_seed": per_seed,
        "rows": rows,
        "summary": _summarise_fl(rows),
        "hypotheses": _fl_hypotheses(rows, per_seed),
    }


def _summarise_fl(rows: Sequence[dict]) -> dict:
    acc: dict[str, list[float]] = {}
    for r in rows:
        if r.get("final_accuracy") is None:
            continue
        acc.setdefault(r["arm"], []).append(float(r["final_accuracy"]))
    out = {}
    for arm, vals in sorted(acc.items()):
        ci = bootstrap_ci(vals, seed=0)
        out[arm] = {"mean": ci.point, "lo": ci.lo, "hi": ci.hi,
                    "n_seeds": ci.n, "values": vals}
    return out


def _fl_hypotheses(rows: Sequence[dict], per_seed: dict) -> dict:
    from sca.experiments.metrics import paired_permutation_test

    acc: dict[str, dict[int, float]] = {}
    hashes: dict[str, dict[int, str]] = {}
    accepted: dict[str, dict[int, int]] = {}
    for r in rows:
        if r.get("final_accuracy") is None:
            continue
        acc.setdefault(r["arm"], {})[int(r["seed"])] = float(r["final_accuracy"])
        hashes.setdefault(r["arm"], {})[int(r["seed"])] = r.get("final_hash")
        accepted.setdefault(r["arm"], {})[int(r["seed"])] = r.get("rounds_accepted")

    frozen_hash = hashes.get("frozen_pretrained", {})

    h3: dict[str, Any] = {}
    for arm in sorted(acc):
        if not arm.endswith("|certified_gate"):
            continue
        counterpart = arm.replace("|certified_gate", "|no_gate")
        if counterpart not in acc:
            continue
        seeds = sorted(set(acc[arm]) & set(acc[counterpart]))
        if not seeds:
            continue
        a = [acc[arm][s] for s in seeds]
        b = [acc[counterpart][s] for s in seeds]
        res = paired_permutation_test(a, b, seed=0)
        # F5 DETECTOR. The old repo's headline "93.75% under 50% Byzantine"
        # was the frozen pretrained checkpoint: the gate rejected every round
        # and rollback was a no-op, so "the gate helped" only meant "FL was
        # net-harmful and the gate did nothing". Record, per configuration,
        # whether the gated model is byte-identical to the frozen checkpoint,
        # so that reading cannot be hidden.
        n_rejected_all = sum(
            1 for s_ in seeds if accepted.get(arm, {}).get(s_) == 0
        )
        n_equals_frozen = sum(
            1 for s_ in seeds
            if frozen_hash.get(s_) is not None
            and hashes.get(arm, {}).get(s_) == frozen_hash.get(s_)
        )
        h3[arm] = {
            "gate_mean": float(np.mean(a)),
            "no_gate_mean": float(np.mean(b)),
            "difference": float(np.mean(a) - np.mean(b)),
            "gate_helps": bool(np.mean(a) > np.mean(b)),
            "p_value": res.p_value,
            "min_attainable_p": res.min_attainable_p,
            "n_seeds": len(seeds),
            "n_seeds_gate_rejected_every_round": n_rejected_all,
            "n_seeds_gated_model_is_the_frozen_checkpoint": n_equals_frozen,
            "is_do_nothing_baseline": bool(n_equals_frozen == len(seeds)),
        }
    n_help = sum(1 for v in h3.values() if v["gate_helps"])
    n_donothing = sum(1 for v in h3.values() if v["is_do_nothing_baseline"])
    return {
        "H3_gate_improves_quality_under_attack": {
            "statement": (
                "held-out accuracy with the certified gate > without it, on a "
                "test set the gate never saw"
            ),
            "n_configurations": len(h3),
            "n_where_gate_helps": n_help,
            "n_where_gate_is_the_do_nothing_baseline": n_donothing,
            "caveat": (
                "A configuration where the gated model is byte-identical to "
                "the frozen pretrained checkpoint is NOT evidence that the "
                "gate works: it means the gate rejected every round, so the "
                "comparison only shows that federated training was net-harmful "
                "in that cell. This is audit finding F5 and it must be read "
                "together with the C1 verdict."
            ),
            "per_configuration": h3,
        },
        "C1_clean_fedavg_beats_frozen": {
            "per_seed": {k: v["C1"] for k, v in per_seed.items()},
            "n_seeds_passing": sum(
                1 for v in per_seed.values() if v["C1"]["passed"]
            ),
            "n_seeds": len(per_seed),
        },
        "F7_constant_predictor_collapse": _f7_collapse(rows),
    }


#: A run is a "constant predictor" when it emits one class on more than this
#: fraction of the held-out set.  0.53 / 0.47 are exactly the class priors of
#: this corpus, so a collapsed run posts a number that reads like an accuracy.
COLLAPSE_THRESHOLD = 0.97


def _f7_collapse(rows: Sequence[dict]) -> dict:
    """Per-arm count of runs that degenerated into a constant predictor (F7).

    The pre-rebuild harness sorted by label and chunked, making every client
    single-class; Krum then selected one client and the global model predicted
    a constant.  The number reported (0.47, 0.53) was a class prior.  The
    ``clean_<agg>_no_attack_no_gate`` rows are the direct control: a robust
    aggregator that collapses with ZERO adversaries is not being robust.
    """
    out: dict[str, dict] = {}
    for r in rows:
        pcf = r.get("predicted_class_fraction") or {}
        if not pcf:
            continue
        collapsed = max(float(v) for v in pcf.values()) > COLLAPSE_THRESHOLD
        e = out.setdefault(r["arm"], {"n_seeds": 0, "n_collapsed": 0,
                                      "max_class_fraction": []})
        e["n_seeds"] += 1
        e["n_collapsed"] += int(collapsed)
        e["max_class_fraction"].append(
            round(max(float(v) for v in pcf.values()), 4))
    return {
        "threshold": COLLAPSE_THRESHOLD,
        "statement": ("a run predicting one class on >97% of the held-out set "
                      "reports a class prior, not an accuracy (finding F7)"),
        "per_arm": {k: v for k, v in sorted(out.items()) if v["n_collapsed"]},
        "n_arms_with_any_collapse": sum(
            1 for v in out.values() if v["n_collapsed"]),
        "n_arms": len(out),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sca.experiments.run_all",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--arms", choices=["verifier", "fl", "all"], default="all")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run; writes smoke_*.json")
    ap.add_argument("--seeds", type=int, default=None,
                    help="use the first N of DEFAULT_SEEDS")
    ap.add_argument("--n-corpus", type=int, default=None)
    ap.add_argument("--n-rounds", type=int, default=None)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--model", choices=["bow", "mlp"], default=None,
                    help="'mlp' reproduces the pre-rebuild constant-predictor "
                         "architecture; see sca.experiments.model")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--deterministic", action="store_true",
                    help="suppress wall-clock fields so the artifacts are "
                         "byte-identical across processes (criterion C4); "
                         "equivalently set SCA_DETERMINISTIC=1")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)
    if args.deterministic:
        os.environ["SCA_DETERMINISTIC"] = "1"

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("sca").setLevel(
        logging.INFO if args.verbose else logging.WARNING
    )

    config = ExperimentConfig.smoke_config() if args.smoke else ExperimentConfig()
    if args.seeds is not None:
        config.seeds = list(DEFAULT_SEEDS)[: args.seeds] if not args.smoke \
            else list(range(args.seeds))
    if args.n_corpus is not None:
        config.n_corpus = args.n_corpus
    if args.n_rounds is not None:
        config.n_rounds = args.n_rounds
    if args.epsilon is not None:
        config.epsilon = args.epsilon
    if args.model is not None:
        config.model_kind = args.model
    if args.lr is not None:
        config.lr = args.lr

    prefix = "smoke_" if config.smoke else ""
    written: list[str] = []
    t0 = time.time()

    if args.arms in ("verifier", "all"):
        payload = run_verifier_arm(config)
        written.append(_write(f"{prefix}verifier_arm.json", payload))
        _print_verifier(payload)

    if args.arms in ("fl", "all"):
        payload = run_fl_arm(config)
        written.append(_write(f"{prefix}fl_arm.json", payload))
        _print_fl(payload)

    manifest = provenance(config, {
        "arms": args.arms,
        "files": written,
        "wall_seconds": round(time.time() - t0, 1),
    })
    written.append(f"{prefix}run_manifest.json")
    _write(f"{prefix}run_manifest.json", manifest)
    if deterministic_mode():
        print(f"\nwrote {len(written)} files")
    else:
        print(f"\nwrote {len(written)} files to {results_dir()}")
    for w in sorted(written):
        print(f"  {w}")
    return 0


def _print_verifier(payload: dict) -> None:
    print("\n=== VERIFIER ARM: certified bound by allocation/mutator arm ===")
    s = payload["summary"]["bound"]
    for arm in sorted(s, key=lambda a: s[a]["mean"]):
        v = s[arm]
        print(f"  {arm:34s} bound {v['mean']:.4f} "
              f"[{v['lo']:.4f}, {v['hi']:.4f}]  n={v['n_seeds']}")
    h2 = payload["hypotheses"]["H2_search_guided_vs_w23"]
    if h2.get("available"):
        verdict = "TIGHTER (H2 supported)" if h2["a_is_tighter"] else \
            "NOT tighter (H2 NOT supported)"
        print(f"  H2: search_guided vs w^(2/3): {h2['mean_difference_a_minus_b']:+.4f} "
              f"-> {verdict}  p={h2['p_value']:.4f}")
    amp = payload["hypotheses"]["F8_amplification"]
    print("  amplification (raw -> deduped to distinct depth-0 ancestors):")
    for arm in sorted(amp["raw_mean_by_arm"]):
        print(f"    {arm:34s} raw {amp['raw_mean_by_arm'][arm]:.2f}x  "
              f"dedup {amp['dedup_mean_by_arm'][arm]:.2f}x")


def _print_fl(payload: dict) -> None:
    print("\n=== FL ARM: held-out test accuracy (the gate never saw this data) ===")
    s = payload["summary"]
    for arm in sorted(s):
        v = s[arm]
        print(f"  {arm:44s} {v['mean']:.4f} "
              f"[{v['lo']:.4f}, {v['hi']:.4f}]  n={v['n_seeds']}")
    c1 = payload["hypotheses"]["C1_clean_fedavg_beats_frozen"]
    print(f"  C1: clean FedAvg > frozen pretrained on "
          f"{c1['n_seeds_passing']}/{c1['n_seeds']} seeds")
    h3 = payload["hypotheses"]["H3_gate_improves_quality_under_attack"]
    print(f"  H3: gate helped in {h3['n_where_gate_helps']}/"
          f"{h3['n_configurations']} configurations")
    f7 = payload["hypotheses"].get("F7_constant_predictor_collapse", {})
    if f7.get("n_arms_with_any_collapse"):
        print(f"  F7: {f7['n_arms_with_any_collapse']}/{f7['n_arms']} arms had "
              f"at least one seed collapse to a constant predictor "
              f"(>{f7['threshold']:.0%} of the test set in one class):")
        for arm, v in sorted(f7["per_arm"].items(),
                             key=lambda kv: -kv[1]["n_collapsed"])[:12]:
            print(f"      {arm:44s} {v['n_collapsed']}/{v['n_seeds']} seeds")
    nd = h3["n_where_gate_is_the_do_nothing_baseline"]
    if nd:
        print(f"  H3 WARNING: in {nd}/{h3['n_configurations']} configurations "
              f"the gated model IS the frozen checkpoint (gate rejected every "
              f"round). That is the F5 do-nothing baseline, not a defence.")


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
