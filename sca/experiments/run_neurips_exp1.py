#!/usr/bin/env python3
"""NeurIPS Experiment 1: Fix acceptance gate + ROC curve analysis.

The paper-killer problem: the clean model (93.75% accuracy) is REJECTED
by the acceptance gate (bound=0.86 > epsilon=0.45). Root cause: K=40
regions with 500 queries means ~12 samples/region, giving Hoeffding
widths near 1.0.

This experiment:
1. Sweeps K in {5, 10, 20, 40} to show fewer regions = tighter bounds
2. Compares Hoeffding vs Clopper-Pearson bounds
3. Produces ROC data: for each epsilon, acceptance rate per model quality
4. Demonstrates an operating point where clean models PASS and attacked FAIL
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from the SCA codebase
# ---------------------------------------------------------------------------
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
from sca.utils.stats import (
    RegionStats,
    check_acceptance,
    check_acceptance_cp,
    hoeffding_bound,
)
from sca.verifier.mutations import CompositeMutator
from sca.verifier.rlm_verifier import RLMVerifier


# ---------------------------------------------------------------------------
# Core verification runner
# ---------------------------------------------------------------------------

def run_verification_and_collect_stats(
    model: nn.Module,
    tokenizer,
    interactions: list[dict],
    n_clusters: int,
    total_budget: int = 500,
    delta: float = 0.05,
    seed: int = 42,
) -> list[RegionStats]:
    """Run Full SCA verification and return raw region stats.

    Uses Full SCA config (D=2, B=3, hops=2, reserve=0.35, ppf=3)
    which is the strongest verification configuration.
    """
    embedder = PromptOnlyEmbedder(vocab_size=2000, embed_dim=64, seed=seed)
    safety_pred = GroundTruthSafetyPredicate()
    tau = 1.0

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
        max_depth=2,
        branching_factor=3,
        total_budget=total_budget,
        neighborhood_hops=2,
        seed=seed,
        reserve_frontier_fraction=0.35,
        probes_per_focus=3,
    )

    rng = np.random.RandomState(seed)
    n_seed = min(50, len(interactions))
    seed_idx = rng.choice(len(interactions), size=n_seed, replace=False)
    seed_ints = [interactions[i] for i in seed_idx]

    model_fn = ClassifierModelFn(model, tokenizer, max_len=64)

    accepted, state, region_stats = verifier.verify(
        model_fn, seed_ints, epsilon=0.50, delta=delta,
    )

    # Normalize weights
    tw = sum(rs.weight for rs in region_stats)
    if tw > 0:
        for rs in region_stats:
            rs.weight /= tw

    total_viols = sum(1 for t in state.traces if t.violation)
    total_queries = state.total_queries
    logger.info(
        f"    K={n_clusters}: {total_viols}/{total_queries} violations "
        f"({total_viols/total_queries*100:.1f}%)"
    )

    return region_stats


def compute_bounds_at_epsilon_sweep(
    region_stats: list[RegionStats],
    delta: float,
    epsilon_values: list[float],
) -> dict:
    """Compute acceptance decisions for both Hoeffding and CP bounds."""
    results = {}
    for eps in epsilon_values:
        hoeff_accepted, hoeff_bound = check_acceptance(region_stats, eps, delta)
        cp_accepted, cp_bound = check_acceptance_cp(region_stats, eps, delta)
        results[eps] = {
            "hoeffding_bound": float(hoeff_bound),
            "hoeffding_accepted": hoeff_accepted,
            "cp_bound": float(cp_bound),
            "cp_accepted": cp_accepted,
        }
    return results


# ---------------------------------------------------------------------------
# Model creation helpers
# ---------------------------------------------------------------------------

def create_degraded_model(model: nn.Module) -> nn.Module:
    """Create mildly degraded model (same as original paper)."""
    degraded = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in degraded.named_parameters():
            if "classifier" in name:
                param.add_(torch.randn_like(param) * 0.15)
            elif "fc2" in name:
                param.add_(torch.randn_like(param) * 0.08)
            else:
                param.add_(torch.randn_like(param) * 0.01)
    return degraded


def create_attacked_model(model: nn.Module) -> nn.Module:
    """Create severely attacked model (same as original paper)."""
    attacked = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in attacked.named_parameters():
            if "classifier" in name or "fc2" in name:
                param.add_(torch.randn_like(param) * 0.30)
            elif "fc1" in name:
                param.add_(torch.randn_like(param) * 0.15)
            else:
                param.add_(torch.randn_like(param) * 0.02)
    return attacked


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 80)
    logger.info("NeurIPS EXPERIMENT 1: Acceptance Gate Fix + ROC Curve")
    logger.info("=" * 80)

    # 1. Load data
    logger.info("\n--- Loading PKU-SafeRLHF data ---")
    pku_data = load_pku_saferlhf(max_samples=2000)
    interactions = create_interactions_from_pku(pku_data, max_samples=500)
    logger.info(f"  Created {len(interactions)} interactions")

    # 2. Train baseline model
    logger.info("\n--- Training baseline SafetyClassifier ---")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    client_datasets, test_dataset = prepare_fl_data(
        tokenizer, pku_data, n_clients=8, max_len=64,
    )

    model = SafetyClassifier(vocab_size=50257, embed_dim=128, hidden_dim=128, max_len=64)
    pretrain_centralized(model, client_datasets, epochs=8, lr=0.001)
    clean_eval = evaluate_classifier(model, test_dataset)
    logger.info(f"  Clean model accuracy: {clean_eval['accuracy']:.4f}")

    # 3. Create degraded models
    torch.manual_seed(42)
    degraded = create_degraded_model(model)
    deg_eval = evaluate_classifier(degraded, test_dataset)
    logger.info(f"  Degraded model accuracy: {deg_eval['accuracy']:.4f}")

    attacked = create_attacked_model(model)
    atk_eval = evaluate_classifier(attacked, test_dataset)
    logger.info(f"  Attacked model accuracy: {atk_eval['accuracy']:.4f}")

    models = {
        "clean": (model, clean_eval["accuracy"]),
        "degraded": (degraded, deg_eval["accuracy"]),
        "attacked": (attacked, atk_eval["accuracy"]),
    }

    # 4. Sweep K values and epsilon values
    K_values = [5, 10, 20, 40]
    epsilon_values = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    delta = 0.05
    budget = 500

    all_results = {}

    for K in K_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"  K = {K} regions (budget={budget}, ~{budget//K} samples/region)")
        logger.info(f"{'='*60}")

        all_results[K] = {}

        for model_name, (m, acc) in models.items():
            logger.info(f"\n  Model: {model_name} (accuracy={acc:.4f})")

            # Run verification
            region_stats = run_verification_and_collect_stats(
                m, tokenizer, interactions, n_clusters=K,
                total_budget=budget, delta=delta,
            )

            # Compute both bounds at each epsilon
            sweep = compute_bounds_at_epsilon_sweep(
                region_stats, delta, epsilon_values,
            )

            # Compute summary stats
            k = len(region_stats)
            avg_samples = np.mean([rs.n_samples for rs in region_stats if rs.n_samples > 0])
            total_viols = sum(rs.n_violations for rs in region_stats)
            total_samples = sum(rs.n_samples for rs in region_stats)
            empirical_rate = total_viols / total_samples if total_samples > 0 else 0

            all_results[K][model_name] = {
                "accuracy": acc,
                "empirical_violation_rate": float(empirical_rate),
                "avg_samples_per_region": float(avg_samples),
                "n_active_regions": k,
                "epsilon_sweep": {str(e): v for e, v in sweep.items()},
            }

            # Print bounds for key epsilons
            for eps in [0.10, 0.20, 0.30]:
                s = sweep[eps]
                logger.info(
                    f"    eps={eps:.2f}: Hoeffding={s['hoeffding_bound']:.3f} "
                    f"({'ACCEPT' if s['hoeffding_accepted'] else 'REJECT'}), "
                    f"CP={s['cp_bound']:.3f} "
                    f"({'ACCEPT' if s['cp_accepted'] else 'REJECT'})"
                )

    # 5. Print ROC summary table
    logger.info("\n" + "=" * 90)
    logger.info("ROC SUMMARY: Acceptance decisions across (K, epsilon, model)")
    logger.info("=" * 90)

    for K in K_values:
        logger.info(f"\n  K = {K}:")
        header = f"  {'epsilon':>8}"
        for model_name in ["clean", "degraded", "attacked"]:
            header += f"  | {'Hoeff':>6} {'CP':>6}"
        logger.info(header)
        logger.info(f"  {'-'*70}")

        for eps in epsilon_values:
            row = f"  {eps:>8.2f}"
            for model_name in ["clean", "degraded", "attacked"]:
                s = all_results[K][model_name]["epsilon_sweep"][str(eps)]
                h = "PASS" if s["hoeffding_accepted"] else "FAIL"
                c = "PASS" if s["cp_accepted"] else "FAIL"
                row += f"  | {h:>6} {c:>6}"
            logger.info(row)

    # 6. Find optimal operating point
    logger.info("\n" + "=" * 90)
    logger.info("OPTIMAL OPERATING POINTS (clean=ACCEPT, attacked=REJECT)")
    logger.info("=" * 90)

    for K in K_values:
        for bound_type in ["hoeffding", "cp"]:
            for eps in epsilon_values:
                key = f"{bound_type}_accepted"
                clean_ok = all_results[K]["clean"]["epsilon_sweep"][str(eps)][key]
                attacked_rej = not all_results[K]["attacked"]["epsilon_sweep"][str(eps)][key]
                if clean_ok and attacked_rej:
                    deg_status = "REJECT" if not all_results[K]["degraded"]["epsilon_sweep"][str(eps)][key] else "ACCEPT"
                    logger.info(
                        f"  K={K}, {bound_type}, eps={eps}: "
                        f"clean=ACCEPT, degraded={deg_status}, attacked=REJECT  *** DISCRIMINATIVE ***"
                    )
                    break

    # 7. Bound comparison (Hoeffding vs CP)
    logger.info("\n" + "=" * 90)
    logger.info("BOUND TIGHTNESS: Hoeffding vs Clopper-Pearson")
    logger.info("=" * 90)

    for K in K_values:
        logger.info(f"\n  K = {K}:")
        for model_name in ["clean", "degraded", "attacked"]:
            # Use eps=0.20 as reference point
            s = all_results[K][model_name]["epsilon_sweep"]["0.2"]
            emp = all_results[K][model_name]["empirical_violation_rate"]
            logger.info(
                f"    {model_name:>10}: empirical={emp:.3f}, "
                f"Hoeffding={s['hoeffding_bound']:.3f}, "
                f"CP={s['cp_bound']:.3f}, "
                f"gap(H-CP)={s['hoeffding_bound']-s['cp_bound']:.3f}"
            )

    # 8. Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Convert keys to strings for JSON
    json_results = {}
    for K in K_values:
        json_results[str(K)] = all_results[K]

    output = {
        "experiment": "NeurIPS Experiment 1: Acceptance Gate Fix + ROC",
        "K_values": K_values,
        "epsilon_values": epsilon_values,
        "delta": delta,
        "budget": budget,
        "results": json_results,
    }

    out_path = results_dir / "neurips_exp1_roc.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
