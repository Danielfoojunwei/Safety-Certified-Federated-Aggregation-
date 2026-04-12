#!/usr/bin/env python3
"""Experiment 2: Multi-seed evaluation with error bars and significance tests.

All current results use a single seed (42). This experiment runs 5 seeds and
reports mean +/- std with 95% confidence intervals, plus paired Wilcoxon
significance tests between methods.

Seeds: [42, 123, 456, 789, 1024]
Models: mild_degraded, attacked_severe
Configs: Static (D=0), RMV (D=2, hops=0), Full SCA (D=2, hops=2)
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from sca.experiments.run_real_evaluation import (
    SafetyClassifier,
    load_pku_saferlhf,
    prepare_fl_data,
    pretrain_centralized,
    evaluate_classifier,
)
from sca.experiments.run_novelty_validation import (
    PromptOnlyEmbedder,
    ClassifierModelFn,
    GroundTruthSafetyPredicate,
    create_interactions_from_pku,
)
from sca.knowledge_graph.mkg import ModelKnowledgeGraph
from sca.knowledge_graph.regions import RegionPartition
from sca.verifier.mutations import CompositeMutator
from sca.verifier.rlm_verifier import RLMVerifier
from sca.utils.stats import check_acceptance
from sca.experiments.metrics import (
    compute_confidence_interval,
    format_with_ci,
    MultiSeedResults,
    paired_significance_test,
    DEFAULT_SEEDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = DEFAULT_SEEDS  # [42, 123, 456, 789, 1024]

# (name, max_depth, branching_factor, hops, reserve_frontier, probes_per_focus)
CONFIGS = [
    ("Static (D=0)", 0, 0, 0, 0.0, 1),
    ("RMV (D=2)", 2, 3, 0, 0.35, 1),
    ("Full SCA (D=2,h=2)", 2, 3, 2, 0.35, 3),
]

# Verification hyper-parameters
N_CLUSTERS = 40
TAU = 1.0
TOTAL_BUDGET = 500
EPSILON = 0.45
DELTA = 0.05
N_SEEDS_SAMPLE = 50


# ---------------------------------------------------------------------------
# Core: run a single verification trial with a given seed
# ---------------------------------------------------------------------------

def run_verification_with_seed(
    model: nn.Module,
    tokenizer,
    interactions: list[dict],
    config_name: str,
    max_depth: int,
    branching_factor: int,
    neighborhood_hops: int,
    total_budget: int,
    n_clusters: int,
    tau: float,
    epsilon: float,
    delta: float,
    n_seeds_sample: int,
    reserve_frontier: float,
    ppf: int,
    seed: int,
) -> dict:
    """Run one verification configuration with a specific random seed.

    Returns a dict of per-trial metrics: total_violations, violation_rate,
    frontier_violations, bound, regions_with_violations, etc.
    """
    embedder = PromptOnlyEmbedder(vocab_size=2000, embed_dim=64, seed=seed)
    safety_pred = GroundTruthSafetyPredicate()

    embeddings = embedder.embed_batch(interactions)
    partition = RegionPartition(tau_new=tau)
    partition.initialize_from_embeddings(embeddings, n_clusters=n_clusters)
    mkg = ModelKnowledgeGraph(partition, tau=tau)

    mutator = CompositeMutator(seed=seed)
    verifier = RLMVerifier(
        safety_predicate=safety_pred,
        embedder=embedder,
        mkg=mkg,
        mutator=mutator,
        max_depth=max_depth,
        branching_factor=branching_factor,
        total_budget=total_budget,
        neighborhood_hops=neighborhood_hops,
        seed=seed,
        reserve_frontier_fraction=reserve_frontier,
        probes_per_focus=ppf,
    )

    rng = np.random.RandomState(seed)
    seed_idx = rng.choice(
        len(interactions),
        size=min(n_seeds_sample, len(interactions)),
        replace=False,
    )
    seed_ints = [interactions[i] for i in seed_idx]

    model_fn = ClassifierModelFn(model, tokenizer, max_len=64)

    t0 = time.time()
    accepted, state, region_stats = verifier.verify(
        model_fn, seed_ints, epsilon, delta,
    )
    elapsed = time.time() - t0

    # Normalize region weights
    tw = sum(rs.weight for rs in region_stats)
    if tw > 0:
        for rs in region_stats:
            rs.weight /= tw
    _, bound = check_acceptance(region_stats, epsilon, delta)

    total_violations = sum(1 for t in state.traces if t.violation)
    frontier_viols = sum(
        1 for t in state.traces
        if t.mutation_type == "frontier_probe" and t.violation
    )
    frontier_total = sum(
        1 for t in state.traces if t.mutation_type == "frontier_probe"
    )

    return {
        "seed": seed,
        "config": config_name,
        "total_violations": total_violations,
        "total_queries": state.total_queries,
        "violation_rate": (
            total_violations / state.total_queries
            if state.total_queries > 0
            else 0.0
        ),
        "frontier_violations": frontier_viols,
        "frontier_total": frontier_total,
        "bound": float(bound),
        "accepted": accepted,
        "regions_with_violations": len(state.failure_regions),
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Model creation helpers
# ---------------------------------------------------------------------------

def create_mild_degraded(base_state: dict) -> nn.Module:
    """Create a mildly degraded model (~80% accuracy)."""
    model = SafetyClassifier(
        vocab_size=50257, embed_dim=128, hidden_dim=128, max_len=64,
    )
    model.load_state_dict(copy.deepcopy(base_state))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.add_(torch.randn_like(param) * 0.15)
            elif "fc2" in name:
                param.add_(torch.randn_like(param) * 0.08)
            else:
                param.add_(torch.randn_like(param) * 0.01)
    return model


def create_attacked(base_state: dict) -> nn.Module:
    """Create a severely attacked model (~50% accuracy)."""
    model = SafetyClassifier(
        vocab_size=50257, embed_dim=128, hidden_dim=128, max_len=64,
    )
    model.load_state_dict(copy.deepcopy(base_state))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "classifier" in name or "fc2" in name:
                param.add_(torch.randn_like(param) * 0.3)
            elif "fc1" in name:
                param.add_(torch.randn_like(param) * 0.15)
            else:
                param.add_(torch.randn_like(param) * 0.02)
    return model


# ---------------------------------------------------------------------------
# Aggregation and formatting
# ---------------------------------------------------------------------------

def aggregate_seeds(
    per_seed_results: list[dict],
    metric: str,
) -> MultiSeedResults:
    """Wrap per-seed values for a metric into a MultiSeedResults object."""
    values = [r[metric] for r in per_seed_results]
    seeds = [r["seed"] for r in per_seed_results]
    return MultiSeedResults(metric_name=metric, values=values, seeds=seeds)


def _fmt(values: list[float], decimals: int = 1) -> str:
    """Format mean +/- std [ci_lo, ci_hi] with configurable decimal places."""
    if not values:
        return "N/A"
    mean, ci_lo, ci_hi = compute_confidence_interval(values, 0.95)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return (
        f"{mean:.{decimals}f} +/- {std:.{decimals}f} "
        f"[{ci_lo:.{decimals}f}, {ci_hi:.{decimals}f}]"
    )


def print_model_table(
    model_name: str,
    config_results: dict[str, list[dict]],
) -> None:
    """Print a formatted results table for one model across all configs.

    config_results: {config_name: [per-seed result dicts]}
    """
    config_names = [c[0] for c in CONFIGS]

    print(f"\n  Model: {model_name}")
    header = (
        f"  {'Method':<22} | {'Violations (mean+/-std [CI])':<34} | "
        f"{'ViolRate':<26} | {'Frontier Viols':<26} | "
        f"{'p-value vs prev':<18}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    prev_viols: list[float] | None = None
    for cfg_name in config_names:
        trials = config_results[cfg_name]
        viols = [t["total_violations"] for t in trials]
        vrates = [t["violation_rate"] for t in trials]
        fviols = [t["frontier_violations"] for t in trials]

        viols_str = _fmt(viols, decimals=1)
        vrate_str = _fmt(vrates, decimals=4)
        fviol_str = _fmt(fviols, decimals=1)

        if prev_viols is not None and len(prev_viols) >= 3:
            _, p_val = paired_significance_test(prev_viols, viols)
            sig = "sig" if p_val < 0.05 else "n.s."
            p_str = f"{p_val:.4f} ({sig})"
        else:
            p_str = "-"

        print(
            f"  {cfg_name:<22} | {viols_str:<34} | "
            f"{vrate_str:<26} | {fviol_str:<26} | {p_str:<18}"
        )
        prev_viols = viols

    # Amplification and improvement ratios
    static_viols = [t["total_violations"] for t in config_results[config_names[0]]]
    rmv_viols = [t["total_violations"] for t in config_results[config_names[1]]]
    sca_viols = [t["total_violations"] for t in config_results[config_names[2]]]

    # RMV / Static amplification per seed
    amps = []
    for sv, rv in zip(static_viols, rmv_viols):
        amps.append(rv / sv if sv > 0 else float("inf"))
    amp_mean = float(np.mean(amps))
    amp_std = float(np.std(amps, ddof=1)) if len(amps) > 1 else 0.0

    # SCA / RMV improvement per seed (%)
    imps = []
    for rv, sv in zip(rmv_viols, sca_viols):
        imps.append(((sv - rv) / rv * 100.0) if rv > 0 else 0.0)
    imp_mean = float(np.mean(imps))
    imp_std = float(np.std(imps, ddof=1)) if len(imps) > 1 else 0.0

    print()
    print(f"  RMV/Static amplification: {amp_mean:.1f}x +/- {amp_std:.1f}x")
    print(f"  SCA/RMV improvement:      {imp_mean:.1f}% +/- {imp_std:.1f}%")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    t_global = time.time()
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: Multi-Seed Evaluation with Error Bars")
    logger.info(f"Seeds: {SEEDS}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("\n[1/6] Loading PKU-SafeRLHF data from HuggingFace...")
    pku_data = load_pku_saferlhf(max_samples=2000)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    interactions = create_interactions_from_pku(pku_data, max_samples=500)
    n_unsafe = sum(1 for x in interactions if x["label"] == 1)
    logger.info(
        f"Created {len(interactions)} interactions "
        f"(safe={len(interactions) - n_unsafe}, unsafe={n_unsafe})"
    )

    # ------------------------------------------------------------------
    # 2. Train baseline model
    # ------------------------------------------------------------------
    logger.info("\n[2/6] Training baseline SafetyClassifier...")
    client_datasets, test_dataset = prepare_fl_data(
        tokenizer, pku_data, n_clients=8, max_len=64,
    )
    base_model = SafetyClassifier(
        vocab_size=50257, embed_dim=128, hidden_dim=128, max_len=64,
    )
    pretrain_centralized(base_model, client_datasets, epochs=8, lr=0.001)
    base_eval = evaluate_classifier(base_model, test_dataset)
    logger.info(
        f"Baseline: acc={base_eval['accuracy']:.4f}, "
        f"f1={base_eval['f1_unsafe']:.4f}"
    )
    base_state = copy.deepcopy(base_model.state_dict())

    # ------------------------------------------------------------------
    # 3. Create degraded and attacked models
    # ------------------------------------------------------------------
    logger.info("\n[3/6] Creating degraded and attacked models...")
    mild_model = create_mild_degraded(base_state)
    mild_eval = evaluate_classifier(mild_model, test_dataset)
    logger.info(
        f"Mild-degraded: acc={mild_eval['accuracy']:.4f}, "
        f"f1={mild_eval['f1_unsafe']:.4f}"
    )

    attacked_model = create_attacked(base_state)
    atk_eval = evaluate_classifier(attacked_model, test_dataset)
    logger.info(
        f"Attacked: acc={atk_eval['accuracy']:.4f}, "
        f"f1={atk_eval['f1_unsafe']:.4f}"
    )

    models = {
        "mild_degraded": mild_model,
        "attacked_severe": attacked_model,
    }
    model_evals = {
        "baseline": base_eval,
        "mild_degraded": mild_eval,
        "attacked_severe": atk_eval,
    }

    # ------------------------------------------------------------------
    # 4. Multi-seed verification loop
    # ------------------------------------------------------------------
    logger.info("\n[4/6] Running multi-seed verification...")
    # all_trials[model_name][config_name] = [per-seed dicts]
    all_trials: dict[str, dict[str, list[dict]]] = {}

    for model_name, model in models.items():
        logger.info(f"\n  --- Model: {model_name} ---")
        all_trials[model_name] = {c[0]: [] for c in CONFIGS}

        for seed in SEEDS:
            logger.info(f"    Seed {seed}:")
            for cfg_name, depth, branch, hops, reserve, ppf in CONFIGS:
                logger.info(f"      Config: {cfg_name} ...")
                result = run_verification_with_seed(
                    model=model,
                    tokenizer=tokenizer,
                    interactions=interactions,
                    config_name=cfg_name,
                    max_depth=depth,
                    branching_factor=branch,
                    neighborhood_hops=hops,
                    total_budget=TOTAL_BUDGET,
                    n_clusters=N_CLUSTERS,
                    tau=TAU,
                    epsilon=EPSILON,
                    delta=DELTA,
                    n_seeds_sample=N_SEEDS_SAMPLE,
                    reserve_frontier=reserve,
                    ppf=ppf,
                    seed=seed,
                )
                result["model"] = model_name
                all_trials[model_name][cfg_name].append(result)
                logger.info(
                    f"        violations={result['total_violations']}, "
                    f"rate={result['violation_rate']:.4f}, "
                    f"frontier_viols={result['frontier_violations']}, "
                    f"bound={result['bound']:.4f}"
                )

    # ------------------------------------------------------------------
    # 5. Aggregate across seeds
    # ------------------------------------------------------------------
    logger.info("\n[5/6] Aggregating results and computing statistics...")

    aggregated: dict[str, dict[str, dict]] = {}
    for model_name in all_trials:
        aggregated[model_name] = {}
        for cfg_name in all_trials[model_name]:
            trials = all_trials[model_name][cfg_name]
            agg: dict = {}
            for metric in [
                "total_violations",
                "violation_rate",
                "frontier_violations",
                "bound",
                "regions_with_violations",
            ]:
                msr = aggregate_seeds(trials, metric)
                mean_val, ci_lo, ci_hi = msr.ci_95
                agg[metric] = {
                    "mean": mean_val,
                    "std": msr.std,
                    "ci_95_lower": ci_lo,
                    "ci_95_upper": ci_hi,
                    "per_seed": msr.values,
                }
            aggregated[model_name][cfg_name] = agg

    # ------------------------------------------------------------------
    # 6. Significance tests
    # ------------------------------------------------------------------
    logger.info("\n[6/6] Running paired significance tests (Wilcoxon)...")

    significance: dict[str, dict[str, dict]] = {}
    config_names = [c[0] for c in CONFIGS]

    for model_name in all_trials:
        significance[model_name] = {}
        for i in range(1, len(config_names)):
            prev_cfg = config_names[i - 1]
            curr_cfg = config_names[i]
            prev_viols = [
                t["total_violations"]
                for t in all_trials[model_name][prev_cfg]
            ]
            curr_viols = [
                t["total_violations"]
                for t in all_trials[model_name][curr_cfg]
            ]
            stat, p_val = paired_significance_test(prev_viols, curr_viols)
            pair_key = f"{prev_cfg} vs {curr_cfg}"
            significance[model_name][pair_key] = {
                "statistic": float(stat),
                "p_value": float(p_val),
                "significant_at_005": p_val < 0.05,
            }
            logger.info(
                f"  {model_name}: {pair_key} -> "
                f"stat={stat:.4f}, p={p_val:.4f} "
                f"({'significant' if p_val < 0.05 else 'not significant'})"
            )

    # ------------------------------------------------------------------
    # Print formatted tables
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"MULTI-SEED RESULTS (5 seeds: {SEEDS})")
    print("=" * 90)

    for model_name in all_trials:
        print_model_table(model_name, all_trials[model_name])

        # Print significance details
        print(f"\n  Significance tests (Wilcoxon signed-rank):")
        for pair_key, sig_info in significance[model_name].items():
            sig_label = "sig" if sig_info["significant_at_005"] else "n.s."
            print(
                f"    {pair_key}: "
                f"W={sig_info['statistic']:.2f}, "
                f"p={sig_info['p_value']:.4f} ({sig_label})"
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parent.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "neurips_exp2_multiseed.json"

    output = {
        "experiment": "multi_seed_evaluation",
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "hyperparameters": {
            "n_clusters": N_CLUSTERS,
            "tau": TAU,
            "total_budget": TOTAL_BUDGET,
            "epsilon": EPSILON,
            "delta": DELTA,
            "n_seeds_sample": N_SEEDS_SAMPLE,
        },
        "configs": [
            {
                "name": name,
                "max_depth": d,
                "branching_factor": b,
                "neighborhood_hops": h,
                "reserve_frontier": r,
                "probes_per_focus": p,
            }
            for name, d, b, h, r, p in CONFIGS
        ],
        "model_baselines": {
            k: {
                mk: (float(mv) if isinstance(mv, (int, float, np.floating)) else mv)
                for mk, mv in v.items()
            }
            for k, v in model_evals.items()
        },
        "per_seed_trials": {
            model_name: {
                cfg_name: trials
                for cfg_name, trials in cfg_trials.items()
            }
            for model_name, cfg_trials in all_trials.items()
        },
        "aggregated": aggregated,
        "significance_tests": significance,
        "total_time_seconds": time.time() - t_global,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")

    elapsed = time.time() - t_global
    logger.info(f"Total experiment time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
