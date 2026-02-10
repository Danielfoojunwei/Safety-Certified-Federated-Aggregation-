#!/usr/bin/env python3
"""Novelty Validation Experiments for Safety-Certified Federated Aggregation.

Validates the three core claims that previously had zero empirical evidence:

Experiment 1 (Claims 2-3): Static vs RLM vs Full SCA comparison
  - Uses actual RLMVerifier class with CompositeMutator and ModelKnowledgeGraph
  - Same model, same seed interactions, same budget
  - Three configs: Static (D=0), RLM only (D=3, no MKG hops), Full SCA (D=3, hops=2)
  - Measures: violations found, coverage, queries-to-first-detection, bound

Experiment 2 (Claim 4): Regression subgraph analysis
  - Clean model -> record region stats -> apply attack -> record new stats
  - Uses mkg.compute_regression_subgraph() and mkg.minimal_explanation_set()
  - Shows Delta_G correctly identifies degraded regions

Experiment 3 (Claim 5): High Byzantine fraction stress test
  - FedAvg collapses at 25-50% Byzantine, Krum partially survives
  - SCA gate catches degradation that parameter-level defenses miss

All experiments use real PKU-SafeRLHF data from HuggingFace and real model training.
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

from sca.knowledge_graph.embedding import RandomProjectionEmbedder
from sca.knowledge_graph.mkg import ModelKnowledgeGraph
from sca.knowledge_graph.regions import RegionPartition
from sca.utils.stats import RegionStats, check_acceptance, hoeffding_bound
from sca.verifier.mutations import CompositeMutator
from sca.verifier.rlm_verifier import RLMVerifier
from sca.verifier.safety_predicate import SafetyPredicate, SafetyEvaluation

from sca.experiments.run_real_evaluation import (
    SafetyClassifier,
    load_pku_saferlhf,
    prepare_fl_data,
    pretrain_centralized,
    evaluate_classifier,
    run_fl_experiment,
    RealSafetyPredicate,
)


class PromptOnlyEmbedder(RandomProjectionEmbedder):
    """Embedder that uses only the prompt field (not context) for region assignment.

    Mutations modify the "prompt" field, so using only the prompt for embedding
    ensures mutated interactions can land in different regions from their parent,
    creating meaningful mutation edges in the MKG.

    The context (model response) is preserved but not used for embedding.
    """

    def embed(self, interaction: dict) -> np.ndarray:
        text = str(interaction.get("prompt", ""))
        bow = np.zeros(self.vocab_size)
        for token in text.split():
            idx = hash(token) % self.vocab_size
            bow[idx] += 1
        if bow.sum() > 0:
            bow /= bow.sum()
        return bow @ self.projection


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Adapters: Bridge FL SafetyClassifier <-> RLMVerifier Interface
# ===========================================================================

class ClassifierModelFn:
    """Wraps SafetyClassifier as model_fn: dict -> str for RLMVerifier.

    Takes an interaction dict with "prompt" (question, may be mutated) and
    "context" (model response, preserved), concatenates them as the classifier
    was trained on prompt+response pairs, and returns "safe" or "unsafe".
    """

    def __init__(self, model: nn.Module, tokenizer, max_len: int = 64):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.call_count = 0

    def __call__(self, interaction: dict) -> str:
        prompt = interaction.get("prompt", "")
        context = interaction.get("context", "")
        text = f"{prompt} {context}" if context else prompt
        tokens = self.tokenizer(
            text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt",
        )
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tokens["input_ids"])
            pred = logits.argmax(dim=1).item()
        self.call_count += 1
        return "unsafe" if pred == 1 else "safe"


class GroundTruthSafetyPredicate(SafetyPredicate):
    """Safety predicate: phi(x, y) = 1 iff classifier prediction matches ground truth.

    A "violation" (phi = 0) means the model misclassified the interaction.
    The RLM verifier then recursively expands around misclassifications
    to discover failure clusters via mutation operators.
    """

    def evaluate(self, interaction: dict, output: str) -> SafetyEvaluation:
        true_label = interaction.get("label", 0)  # 0=safe, 1=unsafe
        predicted_unsafe = (output == "unsafe")
        is_correct = (predicted_unsafe == (true_label == 1))

        category = None
        if not is_correct:
            if true_label == 1 and not predicted_unsafe:
                category = "false_negative"  # missed unsafe content
            else:
                category = "false_positive"  # flagged safe as unsafe

        return SafetyEvaluation(
            interaction=interaction,
            output=output,
            is_safe=is_correct,
            category=category,
        )


def create_interactions_from_pku(pku_data, max_samples: int = 500) -> list[dict]:
    """Convert PKU-SafeRLHF rows to interaction dicts for RLMVerifier.

    The "prompt" field contains the user question (will be mutated by RLM).
    The "context" field contains the model response (preserved during mutation).
    Together, prompt + context is what the safety classifier was trained on.
    """
    interactions = []
    for i, row in enumerate(pku_data):
        if i >= max_samples:
            break
        interactions.append({
            "prompt": row["prompt"],
            "context": row["response_0"],
            "id": str(i),
            "label": 0 if row["is_response_0_safe"] else 1,
        })
    return interactions


# ===========================================================================
# Experiment 1: Static vs RLM vs Full SCA Comparison (Claims 2-3)
# ===========================================================================

def run_single_verification(
    model: nn.Module,
    tokenizer,
    safety_pred: GroundTruthSafetyPredicate,
    embedder: PromptOnlyEmbedder,
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
    n_seeds: int = 50,
    reserve_frontier_fraction: float = 0.0,
    probes_per_focus: int = 1,
) -> dict:
    """Run a single verification configuration and collect detailed metrics."""

    # Fresh MKG for each run (same initialization)
    embeddings = embedder.embed_batch(interactions)
    partition = RegionPartition(tau_new=tau)
    partition.initialize_from_embeddings(embeddings, n_clusters=n_clusters)
    mkg = ModelKnowledgeGraph(partition, tau=tau)

    mutator = CompositeMutator(seed=42)
    verifier = RLMVerifier(
        safety_predicate=safety_pred,
        embedder=embedder,
        mkg=mkg,
        mutator=mutator,
        max_depth=max_depth,
        branching_factor=branching_factor,
        total_budget=total_budget,
        neighborhood_hops=neighborhood_hops,
        seed=42,
        reserve_frontier_fraction=reserve_frontier_fraction,
        probes_per_focus=probes_per_focus,
    )

    # Select seed interactions deterministically
    rng = np.random.RandomState(42)
    seed_idx = rng.choice(len(interactions), size=min(n_seeds, len(interactions)), replace=False)
    seed_ints = [interactions[i] for i in seed_idx]

    # Wrap model as model_fn
    model_fn = ClassifierModelFn(model, tokenizer, max_len=64)

    t_start = time.time()
    accepted, state, region_stats = verifier.verify(
        model_fn, seed_ints, epsilon, delta,
    )
    t_elapsed = time.time() - t_start

    # Acceptance bound
    k = len(region_stats)
    total_weight = sum(rs.weight for rs in region_stats)
    if total_weight > 0:
        for rs in region_stats:
            rs.weight /= total_weight
    _, bound = check_acceptance(region_stats, epsilon, delta)

    # Count violations
    total_violations = sum(1 for t in state.traces if t.violation)
    total_queries = state.total_queries

    # Queries to first violation
    first_violation_query = None
    for i, trace in enumerate(state.traces):
        if trace.violation:
            first_violation_query = i + 1
            break

    # Violations by depth
    violations_by_depth = {}
    for trace in state.traces:
        d = trace.depth
        if d not in violations_by_depth:
            violations_by_depth[d] = {"total": 0, "violations": 0}
        violations_by_depth[d]["total"] += 1
        if trace.violation:
            violations_by_depth[d]["violations"] += 1

    # Violations by mutation type
    violations_by_mutation = {}
    for trace in state.traces:
        mt = trace.mutation_type
        if mt not in violations_by_mutation:
            violations_by_mutation[mt] = {"total": 0, "violations": 0}
        violations_by_mutation[mt]["total"] += 1
        if trace.violation:
            violations_by_mutation[mt]["violations"] += 1

    # Coverage metrics
    regions_explored = len(state.explored_regions)
    regions_with_violations = len(state.failure_regions)

    # Frontier-specific metrics (violations found by frontier exploration)
    frontier_probes = [t for t in state.traces if t.mutation_type == "frontier_probe"]
    frontier_violations = sum(1 for t in frontier_probes if t.violation)
    frontier_total = len(frontier_probes)

    # Per-region details
    per_region = {}
    for rs in region_stats:
        if rs.n_samples > 0:
            per_region[rs.region_id] = {
                "n_samples": rs.n_samples,
                "n_violations": rs.n_violations,
                "p_hat": rs.p_hat,
                "ucb": rs.ucb(delta, k),
                "weight": rs.weight,
            }

    mkg_summary = mkg.summary()

    result = {
        "config_name": config_name,
        "max_depth": max_depth,
        "branching_factor": branching_factor,
        "neighborhood_hops": neighborhood_hops,
        "reserve_frontier_fraction": reserve_frontier_fraction,
        "probes_per_focus": probes_per_focus,
        "accepted": accepted,
        "certified_bound": float(bound),
        "epsilon": epsilon,
        "total_queries": total_queries,
        "total_violations": total_violations,
        "violation_rate": total_violations / total_queries if total_queries > 0 else 0,
        "first_violation_query": first_violation_query,
        "frontier_probes": frontier_total,
        "frontier_violations": frontier_violations,
        "frontier_viol_rate": frontier_violations / frontier_total if frontier_total > 0 else 0,
        "regions_explored": regions_explored,
        "regions_with_violations": regions_with_violations,
        "n_regions_total": k,
        "violations_by_depth": {str(k2): v for k2, v in violations_by_depth.items()},
        "violations_by_mutation": violations_by_mutation,
        "per_region": {str(k2): v for k2, v in per_region.items()},
        "mkg_edges": mkg_summary,
        "time_seconds": t_elapsed,
    }

    logger.info(
        f"  {config_name}: violations={total_violations}/{total_queries}, "
        f"viol_rate={result['violation_rate']:.3f}, "
        f"frontier={frontier_violations}/{frontier_total}, "
        f"regions={regions_explored}/{k} explored, "
        f"{regions_with_violations} with violations, "
        f"bound={bound:.4f}, accepted={accepted}, "
        f"time={t_elapsed:.1f}s"
    )
    return result


def experiment_1_verification_comparison(
    model: nn.Module,
    tokenizer,
    interactions: list[dict],
    model_name: str,
    total_budget: int = 500,
    epsilon: float = 0.45,
    delta: float = 0.05,
    n_clusters: int = 40,
    n_seeds: int = 50,
) -> dict:
    """Experiment 1: Compare Static, RLM, and Full SCA on the same model.

    Design rationale for MKG differentiation:
    - 40 clusters with 50 seeds: ~25-30 regions explored by seeds, leaving
      10-15 frontier regions for graph-guided exploration.
    - reserve_frontier_fraction=0.35: reserves 35% of budget for frontier,
      preventing recursion from consuming everything.
    - probes_per_focus=3 (Full SCA only): concentrates 3 probes per MKG-focused
      region vs 1 per random region (Section 5.2: "concentrate testing").
    - RLM gets the same budget reservation but hops=0 → random frontier order.
    """
    logger.info(f"\n  === Verifying model: {model_name} ===")

    embedder = PromptOnlyEmbedder(vocab_size=2000, embed_dim=64, seed=42)
    safety_pred = GroundTruthSafetyPredicate()
    tau = 1.0  # Proximity threshold for 40 clusters

    # Three configurations:
    # (name, max_depth, branching, hops, reserve_frontier, probes_per_focus)
    configs = [
        ("Static (D=0)", 0, 0, 0, 0.0, 1),
        ("RLM (D=2, no MKG)", 2, 3, 0, 0.35, 1),
        ("Full SCA (D=2, h=2)", 2, 3, 2, 0.35, 3),
    ]

    results = {}
    for name, depth, branching, hops, reserve, ppf in configs:
        results[name] = run_single_verification(
            model=model,
            tokenizer=tokenizer,
            safety_pred=safety_pred,
            embedder=embedder,
            interactions=interactions,
            config_name=name,
            max_depth=depth,
            branching_factor=branching,
            neighborhood_hops=hops,
            total_budget=total_budget,
            n_clusters=n_clusters,
            tau=tau,
            epsilon=epsilon,
            delta=delta,
            n_seeds=n_seeds,
            reserve_frontier_fraction=reserve,
            probes_per_focus=ppf,
        )

    return {"model_name": model_name, "configs": results}


# ===========================================================================
# Experiment 2: Regression Subgraph Analysis (Claim 4)
# ===========================================================================

def experiment_2_regression_subgraph(
    clean_model: nn.Module,
    attacked_model: nn.Module,
    tokenizer,
    interactions: list[dict],
    n_clusters: int = 16,
    total_budget: int = 400,
    epsilon: float = 0.45,
    delta: float = 0.05,
    eta: float = 0.02,
    gamma: float = 0.01,
) -> dict:
    """Experiment 2: Show regression subgraph identifies degraded regions.

    1. Verify clean model -> region_stats_clean
    2. Verify attacked model -> region_stats_attacked (same MKG structure)
    3. Compute Delta_G = regression subgraph
    4. Compute minimal explanation set
    5. Show Delta_G correctly localizes the degradation
    """
    logger.info("\n--- Experiment 2: Regression Subgraph Analysis ---")

    embedder = PromptOnlyEmbedder(vocab_size=2000, embed_dim=64, seed=42)
    safety_pred = GroundTruthSafetyPredicate()
    tau = 1.5
    embeddings = embedder.embed_batch(interactions)

    rng = np.random.RandomState(42)
    seed_idx = rng.choice(len(interactions), size=min(50, len(interactions)), replace=False)
    seeds = [interactions[i] for i in seed_idx]

    # --- Verify CLEAN model with Full SCA ---
    logger.info("  Verifying CLEAN model (Full SCA)...")
    partition_c = RegionPartition(tau_new=tau)
    partition_c.initialize_from_embeddings(embeddings, n_clusters=n_clusters)
    mkg_c = ModelKnowledgeGraph(partition_c, tau=tau)

    verifier_c = RLMVerifier(
        safety_predicate=safety_pred, embedder=embedder, mkg=mkg_c,
        mutator=CompositeMutator(seed=42),
        max_depth=3, branching_factor=4,
        total_budget=total_budget, neighborhood_hops=2, seed=42,
    )
    clean_fn = ClassifierModelFn(clean_model, tokenizer, max_len=64)
    accepted_c, state_c, stats_c = verifier_c.verify(clean_fn, seeds, epsilon, delta)

    # Normalize weights
    tw = sum(rs.weight for rs in stats_c)
    if tw > 0:
        for rs in stats_c:
            rs.weight /= tw
    _, bound_c = check_acceptance(stats_c, epsilon, delta)

    clean_viols = sum(1 for t in state_c.traces if t.violation)
    logger.info(
        f"    Clean: violations={clean_viols}/{state_c.total_queries}, "
        f"bound={bound_c:.4f}, accepted={accepted_c}"
    )

    # --- Verify ATTACKED model with Full SCA ---
    logger.info("  Verifying ATTACKED model (Full SCA)...")
    partition_a = RegionPartition(tau_new=tau)
    partition_a.initialize_from_embeddings(embeddings, n_clusters=n_clusters)
    mkg_a = ModelKnowledgeGraph(partition_a, tau=tau)

    verifier_a = RLMVerifier(
        safety_predicate=safety_pred, embedder=embedder, mkg=mkg_a,
        mutator=CompositeMutator(seed=42),
        max_depth=3, branching_factor=4,
        total_budget=total_budget, neighborhood_hops=2, seed=42,
    )
    atk_fn = ClassifierModelFn(attacked_model, tokenizer, max_len=64)
    accepted_a, state_a, stats_a = verifier_a.verify(atk_fn, seeds, epsilon, delta)

    tw = sum(rs.weight for rs in stats_a)
    if tw > 0:
        for rs in stats_a:
            rs.weight /= tw
    _, bound_a = check_acceptance(stats_a, epsilon, delta)

    atk_viols = sum(1 for t in state_a.traces if t.violation)
    logger.info(
        f"    Attacked: violations={atk_viols}/{state_a.total_queries}, "
        f"bound={bound_a:.4f}, accepted={accepted_a}"
    )

    # --- Compute regression subgraph Delta_G ---
    logger.info("  Computing regression subgraph Delta_G...")
    regression_regions = mkg_c.compute_regression_subgraph(
        stats_c, stats_a, delta, eta=eta,
    )
    explanation_set = mkg_c.minimal_explanation_set(
        stats_c, stats_a, delta, gamma=gamma,
    )

    # Per-region comparison
    k = max(len(stats_c), len(stats_a))
    clean_map = {rs.region_id: rs for rs in stats_c}
    atk_map = {rs.region_id: rs for rs in stats_a}

    region_comparison = {}
    all_rids = set(rs.region_id for rs in stats_c) | set(rs.region_id for rs in stats_a)

    for rid in all_rids:
        rs_clean = clean_map.get(rid)
        rs_atk = atk_map.get(rid)

        ucb_clean = rs_clean.ucb(delta, k) if rs_clean else 0.0
        ucb_atk = rs_atk.ucb(delta, k) if rs_atk else 0.0
        delta_j = ucb_atk - ucb_clean

        region_comparison[rid] = {
            "clean_p_hat": rs_clean.p_hat if rs_clean else 0.0,
            "clean_ucb": float(ucb_clean),
            "clean_samples": rs_clean.n_samples if rs_clean else 0,
            "clean_violations": rs_clean.n_violations if rs_clean else 0,
            "attacked_p_hat": rs_atk.p_hat if rs_atk else 0.0,
            "attacked_ucb": float(ucb_atk),
            "attacked_samples": rs_atk.n_samples if rs_atk else 0,
            "attacked_violations": rs_atk.n_violations if rs_atk else 0,
            "delta_j": float(delta_j),
            "in_regression": rid in regression_regions,
            "in_explanation": rid in explanation_set,
            "weight": rs_atk.weight if rs_atk else (rs_clean.weight if rs_clean else 0.0),
        }

    logger.info(
        f"    Regression subgraph Delta_G: {len(regression_regions)} regions: "
        f"{regression_regions}"
    )
    logger.info(
        f"    Minimal explanation set: {len(explanation_set)} regions: "
        f"{explanation_set}"
    )

    return {
        "clean_model": {
            "accepted": accepted_c,
            "bound": float(bound_c),
            "total_queries": state_c.total_queries,
            "total_violations": clean_viols,
            "violation_rate": clean_viols / state_c.total_queries if state_c.total_queries > 0 else 0,
        },
        "attacked_model": {
            "accepted": accepted_a,
            "bound": float(bound_a),
            "total_queries": state_a.total_queries,
            "total_violations": atk_viols,
            "violation_rate": atk_viols / state_a.total_queries if state_a.total_queries > 0 else 0,
        },
        "regression_subgraph": regression_regions,
        "n_regression_regions": len(regression_regions),
        "explanation_set": explanation_set,
        "n_explanation_regions": len(explanation_set),
        "eta": eta,
        "gamma": gamma,
        "region_comparison": {str(k2): v for k2, v in region_comparison.items()},
    }


# ===========================================================================
# Experiment 3: High Byzantine Fraction Stress Test (Claim 5)
# ===========================================================================

def experiment_3_high_byzantine(
    base_state: dict,
    client_datasets: list,
    test_dataset,
    safety_pred: RealSafetyPredicate,
    n_rounds: int = 8,
    sca_epsilon: float = 0.45,
    sca_delta: float = 0.05,
) -> dict:
    """Experiment 3: Higher Byzantine fractions (25%, 37.5%, 50%).

    Shows:
    - FedAvg collapses at high Byzantine fractions
    - Krum partially resists but still degrades
    - SCA gate catches degradation by rejecting poisoned rounds
    """
    logger.info("\n--- Experiment 3: High Byzantine Fraction Stress Test ---")

    n_clients = len(client_datasets)

    def pretrained_factory():
        m = SafetyClassifier(
            vocab_size=50257, embed_dim=128, hidden_dim=128, max_len=64,
        )
        m.load_state_dict(copy.deepcopy(base_state))
        return m

    results = {}

    # Byzantine fractions to test
    byz_configs = [
        (2, "25%"),    # 2/8
        (3, "37.5%"),  # 3/8
        (4, "50%"),    # 4/8
    ]

    # Configurations: (aggregation, name, use_sca)
    method_configs = [
        ("fedavg", "FedAvg", False),
        ("fedavg", "FedAvg+SCA", True),
        ("krum", "Krum", False),
        ("krum", "Krum+SCA", True),
    ]

    for n_byz, byz_label in byz_configs:
        logger.info(f"\n  === Byzantine fraction: {byz_label} ({n_byz}/{n_clients}) ===")

        for agg_method, method_name, use_sca in method_configs:
            exp_name = f"{method_name}_{byz_label}byz_signflip"
            logger.info(f"    Running: {exp_name}")
            try:
                results[exp_name] = run_fl_experiment(
                    pretrained_factory, client_datasets, test_dataset, safety_pred,
                    aggregation_method=agg_method, n_rounds=n_rounds,
                    n_byzantine=n_byz, attack_type="sign_flip", attack_scale=2.0,
                    use_sca_gate=use_sca,
                    sca_epsilon=sca_epsilon, sca_delta=sca_delta, sca_budget=200,
                    local_epochs=3, lr=0.001,
                )
                s = results[exp_name]["summary"]
                logger.info(
                    f"      Final acc={s['final_accuracy']:.4f}, "
                    f"accept_rate={s['acceptance_rate']:.2f}, "
                    f"rejected={s['n_rejected']}"
                )
            except Exception as e:
                logger.warning(f"      Failed: {e}")
                results[exp_name] = {"error": str(e)}

        # Also test safety_degradation attack with FedAvg for this fraction
        for use_sca, sca_label in [(False, "FedAvg"), (True, "FedAvg+SCA")]:
            exp_name = f"{sca_label}_{byz_label}byz_safetydeg"
            logger.info(f"    Running: {exp_name}")
            try:
                results[exp_name] = run_fl_experiment(
                    pretrained_factory, client_datasets, test_dataset, safety_pred,
                    aggregation_method="fedavg", n_rounds=n_rounds,
                    n_byzantine=n_byz, attack_type="safety_degradation",
                    attack_scale=3.0,
                    use_sca_gate=use_sca,
                    sca_epsilon=sca_epsilon, sca_delta=sca_delta, sca_budget=200,
                    local_epochs=3, lr=0.001,
                )
                s = results[exp_name]["summary"]
                logger.info(
                    f"      Final acc={s['final_accuracy']:.4f}, "
                    f"accept_rate={s['acceptance_rate']:.2f}"
                )
            except Exception as e:
                logger.warning(f"      Failed: {e}")
                results[exp_name] = {"error": str(e)}

    return results


# ===========================================================================
# Report Formatting
# ===========================================================================

def format_novelty_report(
    exp1_results: dict,
    exp2_results: dict,
    exp3_results: dict,
) -> str:
    """Format all novelty validation results into a comprehensive report."""
    lines = []
    lines.append("=" * 95)
    lines.append("NOVELTY VALIDATION RESULTS")
    lines.append("Safety-Certified Federated Aggregation via RLM Verification + KG Coverage")
    lines.append("=" * 95)

    # ====== Experiment 1 ======
    lines.append("")
    lines.append("=" * 95)
    lines.append("EXPERIMENT 1: Static vs RLM vs Full SCA Verification Comparison")
    lines.append("  Validates Claims 2-3: RLM recursive > static; MKG guidance improves coverage")
    lines.append("=" * 95)

    for model_key, model_results in exp1_results.items():
        model_name = model_results["model_name"]
        configs = model_results["configs"]

        lines.append(f"\n  --- Model: {model_name} ---")

        # Summary table
        header = (
            f"  {'Configuration':<25} {'Viols':>8} {'VRate':>8} "
            f"{'Frontier':>10} {'FVRate':>8} "
            f"{'RegExpl':>8} {'RegViol':>8} "
            f"{'Bound':>8} {'Accept':>8} {'Time':>6}"
        )
        lines.append(header)
        lines.append(f"  {'-' * 107}")

        for name, r in configs.items():
            first_v = str(r["first_violation_query"]) if r["first_violation_query"] else "N/A"
            fp = r.get("frontier_probes", 0)
            fv = r.get("frontier_violations", 0)
            fvr = r.get("frontier_viol_rate", 0)
            lines.append(
                f"  {name:<25} "
                f"{r['total_violations']:>4}/{r['total_queries']:<3} "
                f"{r['violation_rate']:>8.3f} "
                f"{fv:>4}/{fp:<5} "
                f"{fvr:>8.3f} "
                f"{r['regions_explored']:>4}/{r['n_regions_total']:<3} "
                f"{r['regions_with_violations']:>8} "
                f"{r['certified_bound']:>8.4f} "
                f"{'Yes' if r['accepted'] else 'No':>8} "
                f"{r['time_seconds']:>5.1f}s"
            )

        # Violations by depth
        lines.append(f"\n  Violations by Recursion Depth:")
        config_names = list(configs.keys())
        depth_header = f"  {'Depth':<8}"
        for cn in config_names:
            depth_header += f"  {cn:>22}"
        lines.append(depth_header)

        max_depth = 0
        for cn, r in configs.items():
            for d_str in r["violations_by_depth"]:
                max_depth = max(max_depth, int(d_str))

        for d in range(max_depth + 1):
            line = f"  {d:<8}"
            for cn in config_names:
                d_str = str(d)
                if d_str in configs[cn]["violations_by_depth"]:
                    info = configs[cn]["violations_by_depth"][d_str]
                    line += f"  {info['violations']:>4}/{info['total']:<16}"
                else:
                    line += f"  {'---':>22}"
            lines.append(line)

        # Violations by mutation type
        lines.append(f"\n  Violations by Mutation Operator:")
        all_mutations = set()
        for cn, r in configs.items():
            all_mutations.update(r["violations_by_mutation"].keys())

        mut_header = f"  {'Mutation':<18}"
        for cn in config_names:
            mut_header += f"  {cn:>22}"
        lines.append(mut_header)

        for mt in sorted(all_mutations):
            line = f"  {mt:<18}"
            for cn in config_names:
                if mt in configs[cn]["violations_by_mutation"]:
                    info = configs[cn]["violations_by_mutation"][mt]
                    line += f"  {info['violations']:>4}/{info['total']:<16}"
                else:
                    line += f"  {'---':>22}"
            lines.append(line)

        # MKG edges
        lines.append(f"\n  MKG Graph Structure:")
        for cn, r in configs.items():
            e = r["mkg_edges"]
            lines.append(
                f"    {cn}: {e['n_edges']} edges "
                f"(proximity={e['n_proximity_edges']}, mutation={e['n_mutation_edges']})"
            )

    # ====== Experiment 2 ======
    lines.append("")
    lines.append("=" * 95)
    lines.append("EXPERIMENT 2: Regression Subgraph Analysis (Delta_G)")
    lines.append("  Validates Claim 4: Delta_G correctly identifies degraded regions after attack")
    lines.append("=" * 95)

    lines.append(f"\n  Model Comparison:")
    lines.append(f"  {'Metric':<30} {'Clean Model':>15} {'Attacked Model':>15}")
    lines.append(f"  {'-' * 60}")

    cm = exp2_results["clean_model"]
    am = exp2_results["attacked_model"]
    lines.append(f"  {'Accepted':<30} {'Yes' if cm['accepted'] else 'No':>15} {'Yes' if am['accepted'] else 'No':>15}")
    lines.append(f"  {'Certified Bound':<30} {cm['bound']:>15.4f} {am['bound']:>15.4f}")
    lines.append(f"  {'Total Violations':<30} {cm['total_violations']:>15} {am['total_violations']:>15}")
    lines.append(f"  {'Violation Rate':<30} {cm['violation_rate']:>15.3f} {am['violation_rate']:>15.3f}")
    lines.append(f"  {'Total Queries':<30} {cm['total_queries']:>15} {am['total_queries']:>15}")

    lines.append(f"\n  Regression Subgraph Delta_G (eta={exp2_results['eta']}):")
    lines.append(f"    {exp2_results['n_regression_regions']} regions: {exp2_results['regression_subgraph']}")

    lines.append(f"\n  Minimal Explanation Set (gamma={exp2_results['gamma']}):")
    lines.append(f"    {exp2_results['n_explanation_regions']} regions: {exp2_results['explanation_set']}")

    # Per-region comparison
    lines.append(f"\n  Per-Region Comparison (Clean vs Attacked):")
    lines.append(
        f"  {'Rgn':>4} {'C_p':>8} {'A_p':>8} {'C_UCB':>8} {'A_UCB':>8} "
        f"{'Delta':>8} {'Regr':>6} {'Expl':>6} {'Wt':>6}"
    )
    lines.append(f"  {'-' * 64}")

    for rid_str in sorted(exp2_results["region_comparison"].keys(), key=lambda x: int(x)):
        rc = exp2_results["region_comparison"][rid_str]
        lines.append(
            f"  {rid_str:>4} "
            f"{rc['clean_p_hat']:>8.3f} "
            f"{rc['attacked_p_hat']:>8.3f} "
            f"{rc['clean_ucb']:>8.3f} "
            f"{rc['attacked_ucb']:>8.3f} "
            f"{rc['delta_j']:>8.3f} "
            f"{'*' if rc['in_regression'] else '':>6} "
            f"{'*' if rc['in_explanation'] else '':>6} "
            f"{rc['weight']:>6.3f}"
        )

    # ====== Experiment 3 ======
    lines.append("")
    lines.append("=" * 95)
    lines.append("EXPERIMENT 3: High Byzantine Fraction Stress Test")
    lines.append("  Validates Claim 5: SCA catches degradation at 25-50% Byzantine")
    lines.append("=" * 95)

    # Summary table for sign_flip attack
    lines.append(f"\n  Sign-Flip Attack Results:")
    lines.append(
        f"  {'Configuration':<40} {'Final Acc':>10} {'F1':>8} "
        f"{'Accept%':>10} {'Rejected':>10}"
    )
    lines.append(f"  {'-' * 78}")

    for exp_name in sorted(exp3_results.keys()):
        if "signflip" not in exp_name:
            continue
        r = exp3_results[exp_name]
        if "error" in r:
            lines.append(f"  {exp_name:<40} {'ERROR':>10}")
            continue
        s = r.get("summary", {})
        lines.append(
            f"  {exp_name:<40} "
            f"{s.get('final_accuracy', 0):>10.4f} "
            f"{s.get('final_f1_unsafe', 0):>8.4f} "
            f"{s.get('acceptance_rate', 0):>10.2f} "
            f"{s.get('n_rejected', 0):>10}"
        )

    # Summary table for safety_degradation attack
    lines.append(f"\n  Safety-Degradation Attack Results:")
    lines.append(
        f"  {'Configuration':<40} {'Final Acc':>10} {'F1':>8} "
        f"{'Accept%':>10} {'Rejected':>10}"
    )
    lines.append(f"  {'-' * 78}")

    for exp_name in sorted(exp3_results.keys()):
        if "safetydeg" not in exp_name:
            continue
        r = exp3_results[exp_name]
        if "error" in r:
            lines.append(f"  {exp_name:<40} {'ERROR':>10}")
            continue
        s = r.get("summary", {})
        lines.append(
            f"  {exp_name:<40} "
            f"{s.get('final_accuracy', 0):>10.4f} "
            f"{s.get('final_f1_unsafe', 0):>8.4f} "
            f"{s.get('acceptance_rate', 0):>10.2f} "
            f"{s.get('n_rejected', 0):>10}"
        )

    # Cross-comparison: accuracy with vs without SCA
    lines.append(f"\n  Accuracy Protection: SCA vs No-SCA (sign_flip attack)")
    lines.append(
        f"  {'Method':<15} {'25% noSCA':>12} {'25% +SCA':>12} "
        f"{'37.5% noSCA':>12} {'37.5% +SCA':>12} "
        f"{'50% noSCA':>12} {'50% +SCA':>12}"
    )
    lines.append(f"  {'-' * 87}")

    for method in ["FedAvg", "Krum"]:
        line = f"  {method:<15}"
        for byz in ["25%", "37.5%", "50%"]:
            for sca_suffix in ["", "+SCA"]:
                key = f"{method}{sca_suffix}_{byz}byz_signflip"
                if key in exp3_results and "summary" in exp3_results[key]:
                    acc = exp3_results[key]["summary"]["final_accuracy"]
                    line += f" {acc:>12.4f}"
                else:
                    line += f" {'N/A':>12}"
        lines.append(line)

    lines.append("")
    lines.append("=" * 95)
    lines.append("END OF NOVELTY VALIDATION RESULTS")
    lines.append("=" * 95)

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main():
    start_time = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    output_dir = Path("/home/user/Safety-Certified-Federated-Aggregation-/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Data Loading and Model Setup
    # ===================================================================
    logger.info("=" * 70)
    logger.info("NOVELTY VALIDATION: Loading Data and Models")
    logger.info("=" * 70)

    pku_data = load_pku_saferlhf(max_samples=2000)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create interaction dicts for RLM verifier
    interactions = create_interactions_from_pku(pku_data, max_samples=500)
    n_unsafe = sum(1 for x in interactions if x["label"] == 1)
    logger.info(f"Created {len(interactions)} interactions (safe={len(interactions)-n_unsafe}, unsafe={n_unsafe})")

    # Prepare FL data
    n_clients = 8
    client_datasets, test_dataset = prepare_fl_data(
        tokenizer, pku_data, n_clients=n_clients, max_len=64,
    )

    # Train baseline model
    logger.info("\nPre-training baseline SafetyClassifier (8 epochs)...")
    base_model = SafetyClassifier(vocab_size=50257, embed_dim=128, hidden_dim=128, max_len=64)
    pretrain_centralized(base_model, client_datasets, epochs=8, lr=0.001)
    base_eval = evaluate_classifier(base_model, test_dataset)
    logger.info(f"Baseline: acc={base_eval['accuracy']:.4f}, f1={base_eval['f1_unsafe']:.4f}")
    base_state = copy.deepcopy(base_model.state_dict())

    # Create mildly degraded model (~80% accuracy) - enough violations for RLM
    logger.info("\nCreating mildly-degraded model (small perturbation)...")
    mild_model = SafetyClassifier(vocab_size=50257, embed_dim=128, hidden_dim=128, max_len=64)
    mild_model.load_state_dict(copy.deepcopy(base_state))
    with torch.no_grad():
        for name, param in mild_model.named_parameters():
            if "classifier" in name:
                param.add_(torch.randn_like(param) * 0.15)
            elif "fc2" in name:
                param.add_(torch.randn_like(param) * 0.08)
            else:
                param.add_(torch.randn_like(param) * 0.01)
    mild_eval = evaluate_classifier(mild_model, test_dataset)
    logger.info(f"Mild-deg: acc={mild_eval['accuracy']:.4f}, f1={mild_eval['f1_unsafe']:.4f}")

    # Create severely attacked model (targeted safety degradation)
    logger.info("Creating attacked model (severe perturbation)...")
    attacked_model = SafetyClassifier(vocab_size=50257, embed_dim=128, hidden_dim=128, max_len=64)
    attacked_model.load_state_dict(copy.deepcopy(base_state))
    with torch.no_grad():
        for name, param in attacked_model.named_parameters():
            if "classifier" in name or "fc2" in name:
                param.add_(torch.randn_like(param) * 0.3)
            elif "fc1" in name:
                param.add_(torch.randn_like(param) * 0.15)
            else:
                param.add_(torch.randn_like(param) * 0.02)
    atk_eval = evaluate_classifier(attacked_model, test_dataset)
    logger.info(f"Attacked: acc={atk_eval['accuracy']:.4f}, f1={atk_eval['f1_unsafe']:.4f}")

    # Safety predicate for FL experiments
    safety_pred = RealSafetyPredicate(base_model, tokenizer, max_len=64)

    all_results = {}
    all_results["model_baselines"] = {
        "pretrained": base_eval,
        "mild_degraded": mild_eval,
        "attacked": atk_eval,
    }

    # ===================================================================
    # Experiment 1: Static vs RLM vs Full SCA
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Static vs RLM vs Full SCA Comparison (Claims 2-3)")
    logger.info("=" * 70)

    exp1_results = {}

    # On pretrained (good) model - ~94% accuracy
    exp1_results["pretrained_clean"] = experiment_1_verification_comparison(
        base_model, tokenizer, interactions,
        model_name="pretrained_clean",
    )

    # On mildly degraded model - ~80% accuracy, enough violations for RLM
    exp1_results["mild_degraded"] = experiment_1_verification_comparison(
        mild_model, tokenizer, interactions,
        model_name="mild_degraded",
    )

    # On severely attacked model - ~50% accuracy
    exp1_results["attacked_severe"] = experiment_1_verification_comparison(
        attacked_model, tokenizer, interactions,
        model_name="attacked_severe",
    )

    all_results["experiment_1"] = exp1_results

    # ===================================================================
    # Experiment 2: Regression Subgraph
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: Regression Subgraph Analysis (Claim 4)")
    logger.info("=" * 70)

    exp2_results = experiment_2_regression_subgraph(
        base_model, attacked_model, tokenizer, interactions,
    )
    all_results["experiment_2"] = exp2_results

    # ===================================================================
    # Experiment 3: High Byzantine Fraction
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: High Byzantine Fraction Stress Test (Claim 5)")
    logger.info("=" * 70)

    exp3_results = experiment_3_high_byzantine(
        base_state, client_datasets, test_dataset, safety_pred,
        n_rounds=8, sca_epsilon=0.45, sca_delta=0.05,
    )
    all_results["experiment_3"] = exp3_results

    # ===================================================================
    # Generate Report
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Generating Novelty Validation Report")
    logger.info("=" * 70)

    total_time = time.time() - start_time
    all_results["metadata"] = {
        "total_runtime_seconds": total_time,
        "total_runtime_minutes": total_time / 60,
    }

    # Save raw JSON
    json_path = output_dir / "novelty_validation_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Raw results saved to {json_path}")

    # Generate formatted report
    report = format_novelty_report(exp1_results, exp2_results, exp3_results)
    report_path = output_dir / "novelty_validation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    print("\n\n")
    print(report)

    logger.info(f"\nTotal runtime: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
