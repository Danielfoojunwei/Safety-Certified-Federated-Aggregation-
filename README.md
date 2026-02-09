# Safety-Certified Federated Aggregation

**Safety-Certified Aggregation (SCA)** via Recursive Language Model Verification and Knowledge-Graph Coverage.

A commit-gated federated learning protocol where each candidate global model must satisfy behavioral safety constraints verified by an adaptive Recursive Language Model (RLM) verifier, backed by cryptographic safety certificates with statistical guarantees.

## Overview

In federated fine-tuning of language models under semi-trusted clients, Byzantine updates can degrade safety alignment while evading vector-based anomaly detection. SCA addresses this by:

1. **RLM Verifier**: An adaptive property tester that recursively probes the model via mutation operators (paraphrase, escalation, template instantiation, tool-pivot), concentrating testing budget on discovered failure regions.

2. **Model Knowledge Graph (MKG)**: A graph over interaction regions in embedding space that tracks coverage and guides adaptive sampling. Edges encode centroid proximity and mutation lineage.

3. **Safety Certificates**: Cryptographic certificates committing to the verifier policy, probed subgraph, and statistically valid upper confidence bounds on per-region violation rates.

4. **Soundness Guarantee**: With probability 1-delta, any accepted global model satisfies a target safety violation bound epsilon under an evaluation distribution (Theorem 1, via Hoeffding + union bound).

## Project Structure

```
sca/
  utils/          Statistical bounds (Hoeffding UCB) and crypto (SHA-256, Merkle trees)
  knowledge_graph/ MKG: interaction regions, embeddings, graph topology
  verifier/       RLM verifier: safety predicates, mutation operators, adaptive testing
  certificate/    Certificate schema, acceptance rule, gate
  federated/      FL clients (benign + Byzantine), aggregation (FedAvg, FedAdam, Median, TrimmedMean, Krum), server
  experiments/    Attack scenarios, benchmarks, evaluation protocol, metrics, experiment runner
tests/            Unit tests (105 tests covering all components)
```

## Key Modules

| Module | Description |
|--------|-------------|
| `sca.utils.stats` | Hoeffding bounds, UCB computation, acceptance rule, optimal sample allocation |
| `sca.utils.crypto` | SHA-256 hashing, Merkle tree for trace commitment |
| `sca.knowledge_graph.mkg` | MKG with proximity/mutation edges, neighborhood queries, regression subgraph detection |
| `sca.knowledge_graph.regions` | Interaction region partition via K-means, dynamic region creation |
| `sca.verifier.rlm_verifier` | Recursive verifier with adaptive expansion and graph-guided frontier exploration |
| `sca.verifier.mutations` | Mutation operators: paraphrase, escalation, template, tool-pivot, composite |
| `sca.verifier.safety_predicate` | Safety predicate interface with keyword and threshold implementations |
| `sca.certificate.certificate` | Certificate schema construction and consistency verification |
| `sca.certificate.acceptance` | Acceptance gate combining verification + certification |
| `sca.federated.aggregation` | FedAvg, FedAdam, coordinate-wise median, trimmed mean, Krum |
| `sca.federated.client` | Benign client (local SGD) and Byzantine clients (sign-flip, noise, targeted, scaling) |
| `sca.federated.server` | FL server with commit-gated SCA protocol |
| `sca.experiments.attacks` | Byzantine attacks: sign-flip, stealthy, safety-degradation, IPM, label-flip, gradient-scaling |
| `sca.experiments.benchmarks` | Safety benchmark suites: SafetyBench, JailbreakBench, TruthfulQA, ToxiGen, CASE-Bench, HHH |
| `sca.experiments.evaluation` | Evaluation protocol, HEM scoring, ablation framework, interpretability analysis |
| `sca.experiments.metrics` | Violation rate, bound tightness, sample efficiency, regression size metrics |
| `sca.experiments.baselines` | 8 baseline configurations from no-gating to full SCA |
| `sca.experiments.run_experiment` | Experiment runner with multi-scenario benchmarking pipeline |

## Installation

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Core Algorithm

At each FL round:

1. **Propose**: Aggregate client updates via aggregation rule A
2. **Verify**: Run RLM verifier on candidate model with budget m
   - Seed evaluation -> recursive expansion on failures -> graph-guided frontier exploration
3. **Certify**: Compute per-region UCBs and check acceptance rule: `sum_j w_j * UCB_j <= epsilon`
4. **Commit/Rollback**: Accept candidate if certified safe; otherwise keep previous model

## Experiment Baselines

1. FedAvg / FedAdam without gating
2. Robust aggregation only (median / trimmed mean / Krum)
3. Static safety suite gating (fixed prompts, no recursion)
4. SCA without MKG (adaptive recursion, no graph coverage)
5. Full SCA (RLM + MKG + certificates)

## Benchmarking & Evaluation

### Safety Benchmarks
- **SafetyBench** (ACL 2024): Safety understanding via 7-category MCQ
- **JailbreakBench** (NeurIPS 2024): Jailbreak robustness + over-refusal
- **CASE-Bench** (ICLR 2025): Context-aware safety evaluation
- **TruthfulQA**: Truthfulness (817 questions, 38 categories)
- **ToxiGen**: Toxicity detection (implicit hate speech)
- **HHH-Alignment**: Helpfulness, honesty, harmlessness (Anthropic RLHF)

### Holistic Evaluation Metrics (HEM)
Weighted aggregation of accuracy, safety, convergence, efficiency, fairness, and privacy scores into a single comparable metric.

### Attack Scenarios
- No attack (clean baseline)
- Sign-flip / noise / scaling attacks
- Stealthy norm-bounded attacks
- Targeted safety-degradation attacks
- Inner product manipulation (IPM)
- Label-flip data poisoning
- Gradient scaling model poisoning

### Ablation Studies
Automated sweeps over recursion depth D, branching factor B, partition granularity K, confidence delta, target epsilon, and verification budget M.

## License

MIT
