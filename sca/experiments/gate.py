"""The certified gate: the adapter that plugs Stages A-C into the FL server.

Audit finding **F2** was that ``run_real_evaluation.py`` contained a *private
reimplementation* of the gate which computed
``is_violation = (pred != true_label)`` on the very set it was scoring, and
which never called the ``safety_predicate`` it was handed.  The certified
quantity was therefore identically the empirical error rate, and no part of the
verifier stack was exercised at all.

This module contains the only gate the experiments use.  It owns no statistics
of its own: it builds ``model_fn`` from the candidate model, runs the real
:class:`~sca.verifier.rlm_verifier.RLMVerifier` three-stage protocol, and hands
``VerificationResult.estimation_stats`` -- and nothing else -- to
:class:`~sca.certificate.acceptance.AcceptanceGate`.

The a-priori quantities (``k_total``, ``budget_cap``, ``epsilon``, ``delta``)
are constructor arguments, fixed before any round runs.  Finding **F10** was
that ``K`` used to be ``len(region_stats)`` *after* empty buckets were dropped,
so the run whose outputs had degenerated received the smallest union-bound
penalty.  Nothing here may derive either quantity from data.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
import torch.nn as nn

from sca.certificate.acceptance import AcceptanceGate, GateMode
from sca.experiments.model import REFUSAL_TEXT, make_model_fn
from sca.federated.server import GateDecision as ServerGateDecision
from sca.knowledge_graph.embedding import InteractionEmbedder
from sca.knowledge_graph.mkg import ModelKnowledgeGraph
from sca.knowledge_graph.regions import RegionPartition
from sca.verifier.mutations import MutationOperator, make_mutator
from sca.verifier.rlm_verifier import RLMVerifier, assert_pools_disjoint
from sca.verifier.safety_predicate import (
    ResponseLookupPredicate,
    assert_output_sensitive,
)

logger = logging.getLogger(__name__)

__all__ = [
    "interactions_from_split",
    "build_response_predicate",
    "build_partition",
    "CertifiedGate",
]


# ---------------------------------------------------------------------------
# Turning a LabeledSplit into the interaction dicts the verifier speaks
# ---------------------------------------------------------------------------


def interactions_from_split(split: Any, corpus: Any) -> list[dict]:
    """Convert a :class:`~sca.experiments.data.LabeledSplit` to interactions.

    ``LabeledSplit.texts`` holds the concatenation ``f"{prompt} {response}"``
    that the model consumes, which cannot be split back apart reliably, so the
    two halves are read from ``corpus`` by row id.  Keeping them separate
    matters: a mutation operator rewrites the *prompt* and leaves the candidate
    response alone, because a mutation is an adversarial rephrasing of the
    request, not a change to the text being judged.

    The ground-truth label is deliberately NOT copied into the interaction.
    Nothing downstream is allowed to read it (finding F2); the label reaches
    the predicate only through the response-keyed annotation table built by
    :func:`build_response_predicate`.
    """
    out: list[dict] = []
    for row in split.ids:
        i = int(row)
        out.append(
            {
                "id": i,
                "prompt": corpus.prompts[i],
                "response": corpus.responses[i],
                "context": "",
            }
        )
    return out


def build_response_predicate(
    corpus: Any,
    *splits: Any,
    refusal_text: str = REFUSAL_TEXT,
) -> ResponseLookupPredicate:
    """Build phi from the human annotations that ship with the corpus.

    The lookup key is the RESPONSE text, so it is structurally impossible to
    obtain a verdict from the prompt.  The refusal string is registered as safe:
    a filter that declines to release anything has not committed a safety
    violation (it has paid a utility cost instead, which the held-out accuracy
    measures separately).

    ``on_missing="raise"`` is kept: if the deployed filter ever emits text
    outside the annotated pool, phi is undefined and the run must stop rather
    than guess.
    """
    labels: dict[str, bool] = {refusal_text: True}
    # Conflicts are counted on the SAME normalised key the predicate uses, so
    # the number reported is the number of genuinely ambiguous verdicts rather
    # than an undercount.
    norm = ResponseLookupPredicate._norm
    seen: dict[str, bool] = {norm(refusal_text): True}
    conflicts = 0
    for split in splits:
        for row in split.ids:
            i = int(row)
            response = corpus.responses[i]
            is_safe = int(corpus.labels[i]) == 0
            key = norm(response)
            prev = seen.get(key)
            if prev is not None and prev != is_safe:
                conflicts += 1
            seen[key] = is_safe
            labels[response] = is_safe
    if conflicts:
        logger.warning(
            "%d responses carry conflicting safe/unsafe annotations under the "
            "predicate's normalisation; the last one wins. The count is "
            "written into the results file rather than hidden.",
            conflicts,
        )
    pred = ResponseLookupPredicate(labels, on_missing="raise")
    pred.n_label_conflicts = conflicts  # type: ignore[attr-defined]
    return pred


def build_partition(
    fit_interactions: Sequence[dict],
    embedder: InteractionEmbedder,
    *,
    k_total: int,
    seed: int,
) -> RegionPartition:
    """Fit the a-priori partition on the SEARCH pool.

    Theorem 1 requires the partition to be declared before the estimation data
    is seen.  Fitting k-means on the estimation pool would make the region
    boundaries -- and hence the weights ``w_j`` -- functions of the data the
    bound is computed from.  Fitting on the Stage-A search pool, which is
    disjoint from the estimation pool by construction, keeps that hypothesis
    literally true.
    """
    vectors = embedder.embed_batch(list(fit_interactions))
    return RegionPartition(k=int(k_total), seed=int(seed)).fit(vectors)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class CertifiedGate:
    """A :class:`sca.federated.server.SafetyGate` backed by the real verifier.

    One instance is reused across the rounds of a single FL run.  Each call to
    :meth:`evaluate` rebuilds ``model_fn`` for the candidate parameters, runs
    Stages A-C from scratch, and returns the server's ``GateDecision`` with the
    full :class:`~sca.certificate.certificate.SafetyCertificate` attached.

    ``mode`` selects the mandated controls.  ``ALWAYS_ACCEPT`` and
    ``ALWAYS_REJECT`` still run the whole protocol and still emit a certificate
    carrying the *true* statistical verdict, so the ablation is visible in the
    artifact rather than hidden.
    """

    def __init__(
        self,
        *,
        partition: RegionPartition,
        embedder: InteractionEmbedder,
        predicate: Any,
        tokenizer: Callable[[Sequence[str]], Any],
        search_pool: Sequence[dict],
        estimation_pool: Sequence[dict],
        epsilon: float,
        delta: float,
        k_total: int,
        budget_cap: int,
        n_rounds: int,
        search_budget: int = 200,
        estimation_budget: int = 400,
        allocation_strategy: str = "two_thirds",
        mutator: MutationOperator | str | None = None,
        max_depth: int = 3,
        branching_factor: int = 3,
        neighborhood_hops: int = 2,
        seed: int = 0,
        mode: GateMode | str = GateMode.CERTIFIED,
        name: str | None = None,
    ) -> None:
        if k_total != partition.k:
            raise ValueError(
                f"k_total={k_total} disagrees with the fitted partition "
                f"(K={partition.k}). K is an a-priori constant (F10)."
            )
        assert_pools_disjoint(search_pool, estimation_pool)

        self.partition = partition
        self.embedder = embedder
        self.predicate = predicate
        self.tokenizer = tokenizer
        self.search_pool = list(search_pool)
        self.estimation_pool = list(estimation_pool)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.k_total = int(k_total)
        self.budget_cap = int(budget_cap)
        self.n_rounds = int(n_rounds)
        self.search_budget = int(search_budget)
        self.estimation_budget = int(estimation_budget)
        self.allocation_strategy = allocation_strategy
        self.max_depth = int(max_depth)
        self.branching_factor = int(branching_factor)
        self.neighborhood_hops = int(neighborhood_hops)
        self.seed = int(seed)
        self.mode = GateMode(mode)
        if isinstance(mutator, str) or mutator is None:
            self.mutator_arm = mutator or "search_guided"
            self.mutator: MutationOperator = make_mutator(self.mutator_arm, seed=seed)
        else:
            self.mutator = mutator
            self.mutator_arm = mutator.name
        self.name = name or f"certified_gate[{self.mode.value}]"

        # delta / T: Theorem 1 covers ONE certificate.  Running the gate for T
        # rounds and reporting each round's bound at the nominal delta would
        # overstate the confidence by a factor of T.
        self.acceptance_gate = AcceptanceGate(
            epsilon=self.epsilon,
            delta=self.delta,
            k_total=self.k_total,
            budget_cap=self.budget_cap,
            mode=self.mode,
            rounds_for_union_bound=max(1, self.n_rounds),
        )

        # Embed the estimation pool once.  The vectors depend only on the pool,
        # never on the candidate model, so caching them cannot leak anything.
        self._estimation_vectors: np.ndarray = self.embedder.embed_batch(
            self.estimation_pool
        )
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    @property
    def effective_delta(self) -> float:
        return self.acceptance_gate.effective_delta

    def _build_verifier(self, round_num: int) -> RLMVerifier:
        # A fresh MKG per round: Stage A adds mutation-lineage edges, so
        # reusing one graph would let round t's density depend on round t-1.
        # tau=None => auto_calibrate_tau, which is the fix for F4 (tau=1.0
        # hardcoded made the MKG the complete graph in every shipped script).
        mkg = ModelKnowledgeGraph(self.partition, tau=None, strict=False)
        return RLMVerifier(
            safety_predicate=self.predicate,
            embedder=self.embedder,
            mkg=mkg,
            mutator=self.mutator,
            max_depth=self.max_depth,
            branching_factor=self.branching_factor,
            search_budget=self.search_budget,
            estimation_budget=self.estimation_budget,
            budget_cap=self.budget_cap,
            neighborhood_hops=self.neighborhood_hops,
            seed=self.seed + 1009 * int(round_num),
            allocation_strategy=self.allocation_strategy,
        )

    def verify_model(self, model: nn.Module, round_num: int = 0):
        """Run Stages A-C against ``model`` and return the VerificationResult."""
        model_fn = make_model_fn(model, self.tokenizer)
        verifier = self._build_verifier(round_num)
        return verifier, verifier.verify(
            model_fn,
            self.search_pool,
            self.estimation_pool,
            epsilon=self.epsilon,
            delta=self.effective_delta,
            estimation_vectors=self._estimation_vectors,
        )

    def evaluate(self, candidate_model: nn.Module, round_num: int) -> ServerGateDecision:
        """The :class:`sca.federated.server.SafetyGate` entry point."""
        verifier, result = self.verify_model(candidate_model, round_num)
        decision = self.acceptance_gate.evaluate(
            result,
            model_params={k: v for k, v in candidate_model.state_dict().items()},
            verifier_descriptor=verifier.get_verifier_descriptor(),
            graph_summary=result.mkg_summary,
            fl_round=int(round_num),
        )
        cert = decision.certificate
        info = {
            "mode": self.mode.value,
            "statistical_accept": bool(decision.statistical_accept),
            "vacuous": bool(decision.acceptance.vacuous),
            "unaccounted_weight": float(decision.acceptance.unaccounted_weight),
            "k_total": self.k_total,
            "budget_cap": self.budget_cap,
            "effective_delta": self.effective_delta,
            "allocation": {str(k): int(v) for k, v in result.allocation.items()},
            "allocation_strategy": result.allocation_strategy,
            "estimation_queries": int(result.estimation_queries),
            "search_queries": int(result.search_queries),
            "total_violations_raw": int(result.total_violations),
            "distinct_violations_dedup_depth0": int(result.distinct_violations),
            "distinct_sources": int(result.distinct_sources),
            "raw_amplification": float(result.raw_amplification),
            "mutator_arm": self.mutator_arm,
            "mkg_edge_density": result.mkg_summary.get("edge_density"),
            "mkg_density_flag": result.mkg_summary.get("density_flag"),
            "mkg_tau": result.mkg_summary.get("tau"),
            # binding_hash, NOT certificate_hash: SafetyCertificate carries a
            # wall-clock `timestamp`, so certificate_hash() is a different
            # value in every process and would break criterion C4. The binding
            # hash covers the raw counts, the partition parameters and the
            # model/verifier/graph digests -- i.e. everything the certificate
            # actually asserts.
            "certificate_binding_hash": cert.binding_hash,
        }
        self.history.append({"round": int(round_num), **info,
                             "bound": float(decision.acceptance.bound)})
        return ServerGateDecision(
            accepted=bool(decision.accepted),
            bound=float(decision.acceptance.bound),
            epsilon=self.epsilon,
            certificate=cert,
            info=info,
        )

    # ------------------------------------------------------------------
    def self_check(self) -> dict:
        """Prove phi is output-sensitive before any number is produced.

        Builder B's instruction C.1: run
        :func:`~sca.verifier.safety_predicate.assert_output_sensitive` at
        harness start-up, because a prompt-only predicate passes every unit
        test while invalidating the entire certificate.
        """
        unsafe = None
        for x in self.estimation_pool:
            ev = self.predicate.evaluate(x, x["response"])
            if not ev.is_safe:
                unsafe = x
                break
        if unsafe is None:
            raise RuntimeError(
                "no annotated-unsafe response in the estimation pool; the "
                "output-sensitivity check cannot be run"
            )
        assert_output_sensitive(
            self.predicate, unsafe, REFUSAL_TEXT, unsafe["response"]
        )
        return {
            "output_sensitive": True,
            "n_annotations": int(getattr(self.predicate, "n_labels", 0)),
            "n_label_conflicts": int(
                getattr(self.predicate, "n_label_conflicts", 0)
            ),
        }
