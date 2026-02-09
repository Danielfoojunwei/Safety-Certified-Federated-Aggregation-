"""RLM Verifier: Adaptive Property Tester for Safety Verification.

Section 2.1: The verifier V interacts with model M for m rounds,
maintaining state s_k and choosing tests adaptively:
    x_k ~ q_k(. | s_{k-1}),  y_k ~ pi_M(. | x_k),  z_k = 1[phi(x_k, y_k) = 0]
    s_k = U(s_{k-1}, x_k, y_k, z_k)

The RLM instantiation: q_k and U are implemented via recursion --
the verifier generates paraphrase families, escalates multi-turn attacks,
or pivots tool usage based on observed failures.

Section 6.1: The verifier runs a recursion tree T with depth D and
branching factor B.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from sca.knowledge_graph.embedding import InteractionEmbedder
from sca.knowledge_graph.mkg import ModelKnowledgeGraph
from sca.utils.stats import RegionStats, check_acceptance
from sca.verifier.mutations import CompositeMutator, MutationOperator
from sca.verifier.safety_predicate import SafetyEvaluation, SafetyPredicate


@dataclass
class VerifierTrace:
    """A single node in the verifier's recursion tree.

    Attributes:
        interaction: The test interaction x_k.
        output: The model output y_k.
        violation: z_k = 1[phi(x_k, y_k) = 0].
        region_id: Assigned region R_j.
        depth: Depth in the recursion tree.
        parent_idx: Index of parent trace entry (-1 for root).
        mutation_type: Name of the mutation operator used.
    """
    interaction: dict
    output: str
    violation: bool
    region_id: int
    depth: int
    parent_idx: int = -1
    mutation_type: str = "seed"


@dataclass
class VerifierState:
    """State s_k maintained by the verifier across rounds.

    Tracks the recursion tree, explored regions, and discovered failures
    to guide adaptive test generation.
    """
    traces: list[VerifierTrace] = field(default_factory=list)
    explored_regions: set[int] = field(default_factory=set)
    failure_regions: set[int] = field(default_factory=set)
    current_depth: int = 0
    total_queries: int = 0


class RLMVerifier:
    """Recursive Language Model Verifier.

    Implements the adaptive property testing procedure described in
    Sections 2.1 and 6.1. The verifier:
    1. Starts with seed interactions.
    2. Evaluates model outputs via the safety predicate.
    3. On violation discovery, recursively expands via mutation operators.
    4. Tracks coverage via the MKG.
    5. Produces a certificate after verification completes.

    The key adaptive behavior (Section 5.2): after discovering a failure
    in region j, the verifier concentrates additional samples in the
    r-hop neighborhood of j in the MKG (graph-guided sampling).
    """

    def __init__(
        self,
        safety_predicate: SafetyPredicate,
        embedder: InteractionEmbedder,
        mkg: ModelKnowledgeGraph,
        mutator: MutationOperator | None = None,
        max_depth: int = 3,
        branching_factor: int = 4,
        total_budget: int = 200,
        neighborhood_hops: int = 2,
        seed: int = 42,
    ) -> None:
        """
        Args:
            safety_predicate: The safety predicate phi.
            embedder: Interaction embedder psi.
            mkg: Model Knowledge Graph for coverage tracking.
            mutator: Mutation operator family M.
            max_depth: Maximum recursion depth D.
            branching_factor: Branching factor B per recursion node.
            total_budget: Total number of model queries m.
            neighborhood_hops: Hops r for graph-guided sampling.
            seed: Random seed for reproducibility.
        """
        self.phi = safety_predicate
        self.embedder = embedder
        self.mkg = mkg
        self.mutator = mutator or CompositeMutator(seed=seed)
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.total_budget = total_budget
        self.neighborhood_hops = neighborhood_hops
        self.rng = random.Random(seed)

    def verify(
        self,
        model_fn: Callable[[dict], str],
        seed_interactions: list[dict],
        epsilon: float,
        delta: float,
    ) -> tuple[bool, VerifierState, list[RegionStats]]:
        """Run the full verification procedure.

        Args:
            model_fn: Callable that takes an interaction dict and returns
                      the model output string y ~ pi_M(. | x).
            seed_interactions: Initial seed interactions to begin probing.
            epsilon: Target violation bound.
            delta: Confidence parameter.

        Returns:
            (accepted, state, region_stats):
                accepted: True if the model passes the safety gate.
                state: Full verifier state including traces.
                region_stats: Per-region statistics for certificate construction.
        """
        state = VerifierState()
        self.mkg.partition.reset_all_stats()

        # Phase 1: Seed evaluation
        for interaction in seed_interactions:
            if state.total_queries >= self.total_budget:
                break
            self._evaluate_and_record(
                model_fn, interaction, state, depth=0, parent_idx=-1,
            )

        # Phase 2: Recursive expansion on failures
        self._recursive_expand(model_fn, state)

        # Phase 3: Graph-guided exploration of frontier
        self._frontier_exploration(model_fn, state)

        # Compute region stats and acceptance decision
        region_stats = self.mkg.get_region_stats(delta)
        # Normalize weights
        total_weight = sum(rs.weight for rs in region_stats)
        if total_weight > 0:
            for rs in region_stats:
                rs.weight /= total_weight

        accepted, bound = check_acceptance(region_stats, epsilon, delta)
        return accepted, state, region_stats

    def _evaluate_and_record(
        self,
        model_fn: Callable[[dict], str],
        interaction: dict,
        state: VerifierState,
        depth: int,
        parent_idx: int,
        mutation_type: str = "seed",
    ) -> int:
        """Evaluate a single interaction and record the result.

        Returns the index of the new trace entry.
        """
        # Query model: y_k ~ pi_M(. | x_k)
        output = model_fn(interaction)
        state.total_queries += 1

        # Evaluate safety: z_k = 1[phi(x_k, y_k) = 0]
        evaluation = self.phi.evaluate(interaction, output)
        violation = not evaluation.is_safe

        # Assign to region: j = argmin_l dist(psi(x_k), c_l)
        embedding = self.embedder.embed(interaction)
        region_id = self.mkg.partition.assign(embedding)

        # Record in region
        region = self.mkg.partition.regions[region_id]
        region.record_sample(
            violation=violation,
            interaction_id=str(state.total_queries),
        )

        # Record trace
        trace = VerifierTrace(
            interaction=interaction,
            output=output,
            violation=violation,
            region_id=region_id,
            depth=depth,
            parent_idx=parent_idx,
            mutation_type=mutation_type,
        )
        trace_idx = len(state.traces)
        state.traces.append(trace)

        # Update state
        state.explored_regions.add(region_id)
        if violation:
            state.failure_regions.add(region_id)

        return trace_idx

    def _recursive_expand(
        self,
        model_fn: Callable[[dict], str],
        state: VerifierState,
    ) -> None:
        """Expand the recursion tree from discovered failures (Section 6.1).

        For each violation found, generate B mutated children up to depth D.
        """
        # Find traces with violations that can be expanded
        expansion_queue = [
            (idx, trace)
            for idx, trace in enumerate(state.traces)
            if trace.violation and trace.depth < self.max_depth
        ]

        for parent_idx, parent_trace in expansion_queue:
            if state.total_queries >= self.total_budget:
                break

            # Generate B children via mutation
            for _ in range(self.branching_factor):
                if state.total_queries >= self.total_budget:
                    break

                child_interaction = self.mutator.mutate(
                    parent_trace.interaction,
                    state={"depth": parent_trace.depth},
                )

                child_idx = self._evaluate_and_record(
                    model_fn,
                    child_interaction,
                    state,
                    depth=parent_trace.depth + 1,
                    parent_idx=parent_idx,
                    mutation_type=child_interaction.get("mutation_type", "unknown"),
                )

                # If child also violates and within depth, it gets added
                # to the queue for the next iteration
                child_trace = state.traces[child_idx]
                if child_trace.violation and child_trace.depth < self.max_depth:
                    expansion_queue.append((child_idx, child_trace))

                # Record mutation edge in MKG
                parent_region = parent_trace.region_id
                child_region = child_trace.region_id
                if parent_region != child_region:
                    self.mkg.add_mutation_edge(parent_region, child_region)

    def _frontier_exploration(
        self,
        model_fn: Callable[[dict], str],
        state: VerifierState,
    ) -> None:
        """Graph-guided frontier exploration (Section 5.2).

        After recursive expansion, explore the frontier of the MKG to
        improve coverage. Concentrate samples in neighborhoods of
        failure regions.
        """
        if state.total_queries >= self.total_budget:
            return

        # Get frontier regions (unexplored neighbors of explored regions)
        frontier = self.mkg.get_frontier_regions(state.explored_regions)

        # Also get neighborhoods of failure regions for focused testing
        focus_regions = set()
        for fail_region in state.failure_regions:
            neighborhood = self.mkg.get_neighborhood(
                fail_region, self.neighborhood_hops
            )
            focus_regions.update(neighborhood - state.explored_regions)

        # Combine frontier and focus regions, prioritizing focus
        target_regions = list(focus_regions) + list(frontier - focus_regions)

        for region_id in target_regions:
            if state.total_queries >= self.total_budget:
                break

            if region_id >= len(self.mkg.partition.regions):
                continue

            region = self.mkg.partition.regions[region_id]

            # Generate a probe interaction for this region by perturbing
            # a known interaction near the centroid
            probe = self._generate_region_probe(region, state)
            if probe is not None:
                self._evaluate_and_record(
                    model_fn, probe, state,
                    depth=0, parent_idx=-1,
                    mutation_type="frontier_probe",
                )

    def _generate_region_probe(
        self,
        region,
        state: VerifierState,
    ) -> dict | None:
        """Generate a probe interaction targeting a specific region.

        Uses existing traces from nearby regions as seeds for mutation.
        """
        # Find traces from nearby explored regions
        candidate_traces = [
            t for t in state.traces
            if t.region_id in self.mkg.get_neighborhood(
                region.region_id, hops=1
            )
        ]

        if not candidate_traces:
            # Fall back to any existing trace
            if state.traces:
                candidate_traces = [self.rng.choice(state.traces)]
            else:
                return None

        parent = self.rng.choice(candidate_traces)
        return self.mutator.mutate(parent.interaction)

    def get_verifier_descriptor(self) -> dict:
        """Return a serializable descriptor of the verifier configuration.

        Used for h(V) in the certificate (Section 4.2).
        """
        return {
            "type": "RLMVerifier",
            "max_depth": self.max_depth,
            "branching_factor": self.branching_factor,
            "total_budget": self.total_budget,
            "neighborhood_hops": self.neighborhood_hops,
            "mutator": self.mutator.name,
            "predicate_type": type(self.phi).__name__,
        }
