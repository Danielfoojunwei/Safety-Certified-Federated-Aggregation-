"""The three-stage RLM verifier: SEARCH, then ESTIMATION, then CERTIFICATE.

WHY THREE STAGES
----------------
The previous design used the SAME adaptively generated samples both to search
for failures and to estimate the violation rate.  It recursed only on nodes
that had already violated, and it fed every mutant into the counter that became
p_hat_j.  The resulting samples are neither independent nor identically
distributed under D|R_j, so Hoeffding does not apply and the "certificate" was
not one.  Executed on the shipped configuration, that code certified a bound of
0.3849 <= epsilon = 0.45 and ACCEPTED a model whose true violation rate under D
was 0.90.

The rebuild separates the two roles:

  Stage A -- SEARCH.  Mutation recursion plus MKG frontier exploration over a
    SEARCH POOL.  Every observation is recorded to the *search* counters via a
    :class:`~sca.knowledge_graph.regions.SearchRecorder`, which physically
    cannot write an estimation counter.  Stage A's only output is a per-region
    suspicion score.  It never touches the bound.

  Stage B -- ESTIMATION.  Using Stage A's scores, the estimation budget is
    allocated across the K regions.  The allocation is data-dependent and that
    is now LEGAL (see Theorem 1 below).  For each region, m_j FRESH i.i.d.
    draws are taken from D|R_j out of an ESTIMATION POOL disjoint from the
    search pool, and recorded through an
    :class:`~sca.knowledge_graph.regions.EstimationRecorder`.

  Stage C -- CERTIFICATE.  ``sca.utils.stats.check_acceptance`` runs on
    ``estimation_stats`` and nothing else.

THEOREM 1 (anytime-valid, adaptive allocation)
----------------------------------------------
Fix a partition into K regions and a per-region budget cap M, both declared
BEFORE seeing data.  Suppose for each region j the estimation samples are drawn
i.i.d. from D|R_j, and let m_j be ANY data-dependent sample size with
1 <= m_j <= M.  Then with probability at least 1 - delta, simultaneously for
all j and all realized m_j,

    p_j <= p_hat_j + sqrt( ln(2 K M / delta) / (2 m_j) ),

and therefore  sum_j w_j p_j <= sum_j w_j UCB_j.

PROOF.  For each fixed pair (j, m) with j in [K], m in [M], Hoeffding's
inequality for the mean of m i.i.d. [0,1] variables gives
P(p_j > p_hat_j^(m) + sqrt(ln(2KM/delta) / (2m))) <= delta / (KM).  A union
bound over the KM pairs -- all of which are declared a priori, none of which
depend on the data -- shows that with probability >= 1 - delta the inequality
holds simultaneously for every (j, m).  Because it holds uniformly over m, it
holds at a data-dependent m_j: the adaptive rule selects one of the KM pairs,
and every pair is already covered by the event.  The weighted statement follows
from w_j >= 0 and sum_j w_j = 1.  QED

The cost over the fixed-m bound is a ln(M) inflation inside the square root.
That is what buys the right to let Stage A choose the allocation.

Builder A owns the width function and the Monte-Carlo verification of this
theorem under adversarial adaptive allocation; this module owns the property
the proof depends on -- that the numbers reaching ``estimation_stats`` really
are fresh i.i.d. draws from D|R_j.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from sca.knowledge_graph.embedding import InteractionEmbedder
from sca.knowledge_graph.mkg import ModelKnowledgeGraph
from sca.knowledge_graph.regions import (
    EstimationRecorder,
    RegionPartition,
    SearchRecorder,
)
from sca.utils.seeding import stable_hash, stable_rng
from sca.utils.stats import (
    AcceptanceResult,
    RegionStat,
    check_acceptance,
    optimal_allocation,
)
from sca.verifier.mutations import CompositeMutator, MutationOperator
from sca.verifier.safety_predicate import SafetyPredicate

logger = logging.getLogger(__name__)

#: Allocation rules the experiments must compare.  "search_guided" is ours.
ALLOCATION_STRATEGIES = (
    "uniform",
    "proportional",
    "two_thirds",
    "search_guided",
)


class PoolOverlapError(RuntimeError):
    """Raised when the search pool and the estimation pool intersect."""


@dataclass
class VerifierTrace:
    """One node of the Stage-A recursion tree.

    ``source_id`` is the id of the depth-0 ancestor.  It is what makes
    deduplication possible: a violation discovered at depth 3 by mutating a
    seed that already violated is the SAME finding as the seed's violation, and
    counting it again is how a whitespace-appending mutator out-scores the real
    operators.
    """

    interaction: dict
    output: str
    violation: bool
    region_id: int
    depth: int
    source_id: str
    parent_idx: int = -1
    mutation_type: str = "seed"


@dataclass
class SearchReport:
    """Everything Stage A produced.  None of it may reach the certificate."""

    search_stats: list[RegionStat]
    suspicion: dict[int, float]
    traces: list[VerifierTrace] = field(default_factory=list)
    queries: int = 0
    total_violations: int = 0
    distinct_violations: int = 0
    distinct_sources: int = 0
    recursion_depth_hist: dict[int, int] = field(default_factory=dict)
    explored_regions: set[int] = field(default_factory=set)
    failure_regions: set[int] = field(default_factory=set)
    focus_regions: set[int] = field(default_factory=set)
    frontier_regions: set[int] = field(default_factory=set)

    @property
    def raw_amplification(self) -> float:
        """total_violations / distinct_sources_that_violated.

        1.00 means every violation traces back to a distinct seed and the
        recursion amplified nothing.  Report this next to
        :attr:`total_violations` always; the raw count alone is meaningless.
        """
        if self.distinct_violations == 0:
            return 0.0
        return self.total_violations / self.distinct_violations


@dataclass
class VerificationResult:
    """Output of a full three-stage run."""

    estimation_stats: list[RegionStat]  # the ONLY thing the certificate sees
    search_stats: list[RegionStat]  # discovery metrics; never certified
    allocation: dict[int, int]  # m_j actually spent per region
    total_violations: int  # raw count including mutants
    distinct_violations: int  # deduped to depth-0 ancestors
    distinct_sources: int
    search_queries: int
    estimation_queries: int
    recursion_depth_hist: dict[int, int]

    # -- additive fields (not in the shared contract, safe to ignore) --
    weights: dict[int, float] = field(default_factory=dict)
    k_total: int = 0
    budget_cap: int = 0
    allocation_strategy: str = ""
    acceptance: AcceptanceResult | None = None
    mkg_summary: dict = field(default_factory=dict)
    suspicion: dict[int, float] = field(default_factory=dict)
    search_report: SearchReport | None = None

    @property
    def raw_amplification(self) -> float:
        if self.distinct_violations == 0:
            return 0.0
        return self.total_violations / self.distinct_violations


# ----------------------------------------------------------------------
# Pool hygiene
# ----------------------------------------------------------------------
def interaction_key(interaction: dict) -> int:
    """Content key for an interaction, used for pool-overlap checks."""
    prompt = str(interaction.get("prompt", ""))
    ctx = str(interaction.get("context", ""))
    return stable_hash(prompt + "\x00" + ctx)


def split_verification_pool(
    pool: Sequence[dict],
    search_fraction: float = 0.3,
    seed: int = 0,
    dedup: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Cut the server's verification pool into disjoint search / estimation halves.

    Takes ``SplitBundle.server_verification_pool`` (or any list of interaction
    dicts) and returns ``(search_pool, estimation_pool)``.  Duck-typed on
    purpose: this module must not import the experiment package.

    ``dedup=True`` drops repeated prompts BEFORE splitting.  Real corpora
    contain them -- PKU-SafeRLHF ships the same prompt with several response
    pairs -- and a duplicate straddling the cut would silently break the
    freshness assumption that :func:`assert_pools_disjoint` exists to protect.
    """
    items = list(pool)
    if dedup:
        seen: set[int] = set()
        uniq = []
        for x in items:
            k = interaction_key(x)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(x)
        items = uniq
    rng = stable_rng(seed, "verification_pool_split")
    order = rng.permutation(len(items))
    cut = int(round(len(items) * float(search_fraction)))
    search = [items[int(i)] for i in order[:cut]]
    estimation = [items[int(i)] for i in order[cut:]]
    assert_pools_disjoint(search, estimation)
    return search, estimation


def assert_pools_disjoint(
    search_pool: Sequence[dict], estimation_pool: Sequence[dict]
) -> None:
    """Raise if any interaction appears in both pools.

    Stage B's samples must be fresh with respect to Stage A.  If the search
    stage has already queried a point and then the estimation stage draws the
    same point, the estimation draw is not independent of the allocation
    decision that selected its region, and Theorem 1's hypothesis fails.
    """
    a = {interaction_key(x) for x in search_pool}
    b = {interaction_key(x) for x in estimation_pool}
    overlap = a & b
    if overlap:
        raise PoolOverlapError(
            f"search pool and estimation pool share {len(overlap)} of "
            f"{len(b)} estimation items. Stage B draws must be fresh."
        )


# ----------------------------------------------------------------------
# Allocation
# ----------------------------------------------------------------------
def apportion(
    scores: dict[int, float],
    total_budget: int,
    budget_cap: int,
    eligible: Sequence[int],
    min_per_region: int = 1,
) -> dict[int, int]:
    """Split ``total_budget`` across ``eligible`` regions proportionally to
    ``scores``, honouring ``budget_cap`` per region and a per-region floor.

    Exact: the returned values sum to ``total_budget`` whenever that is
    feasible, i.e. whenever ``total_budget <= len(eligible) * budget_cap``.
    """
    alloc = {int(j): 0 for j in scores}
    eligible = [int(j) for j in eligible]
    if not eligible or total_budget <= 0:
        return alloc
    capacity = len(eligible) * budget_cap
    if total_budget > capacity:
        raise ValueError(
            f"total_budget={total_budget} exceeds capacity "
            f"{len(eligible)} regions x budget_cap={budget_cap} = {capacity}"
        )

    floor = min(min_per_region, budget_cap, total_budget // max(1, len(eligible)))
    for j in eligible:
        alloc[j] = floor
    remaining = total_budget - floor * len(eligible)

    while remaining > 0:
        openj = [j for j in eligible if alloc[j] < budget_cap]
        if not openj:
            break
        s = sum(max(0.0, scores.get(j, 0.0)) for j in openj)
        if s <= 0:
            for j in openj:
                if remaining == 0:
                    break
                alloc[j] += 1
                remaining -= 1
            continue
        raw = {j: remaining * max(0.0, scores.get(j, 0.0)) / s for j in openj}
        given = 0
        for j in openj:
            add = min(int(math.floor(raw[j])), budget_cap - alloc[j])
            alloc[j] += add
            given += add
        remaining -= given
        if given == 0:
            # All fractional: hand out one at a time by largest remainder.
            order = sorted(openj, key=lambda j: -raw[j])
            for j in order:
                if remaining == 0:
                    break
                if alloc[j] < budget_cap:
                    alloc[j] += 1
                    remaining -= 1
    return alloc


def build_allocation(
    strategy: str,
    weights: dict[int, float],
    suspicion: dict[int, float],
    total_budget: int,
    budget_cap: int,
    k_total: int,
    min_per_region: int = 1,
) -> dict[int, int]:
    """Per-region estimation budgets m_j under one of :data:`ALLOCATION_STRATEGIES`.

    ANALYTIC NOTE, stated up front because it bears directly on hypothesis H2.
    With the Hoeffding width of Theorem 1 the certified bound is

        sum_j w_j p_hat_j + sum_j w_j sqrt(C / (2 m_j)),   C = ln(2KM/delta),

    and only the second term depends on the allocation, because
    E[p_hat_j] = p_j for any m_j.  Minimising sum_j w_j m_j^(-1/2) subject to
    sum_j m_j = B is a Lagrangian whose solution is m_j proportional to
    w_j^(2/3).  Therefore, under a Hoeffding width, ``two_thirds`` is optimal in
    expectation and NO search-guided rule can beat it -- ``search_guided`` is
    predicted to be weakly WORSE.  Search guidance can only pay off with a
    variance-adaptive width (empirical Bernstein), where the width carries a
    sqrt(p_j (1 - p_j)) factor and the optimum moves to
    m_j proportional to (w_j sqrt(p_j (1-p_j)))^(2/3).  ``search_guided``
    implements that optimum using Stage A's estimate of p_j.  The experiment
    must report the comparison under the width it actually certifies with.

    Strategies:
        uniform:       m_j equal across regions with w_j > 0.
        proportional:  m_j proportional to w_j.
        two_thirds:    m_j proportional to w_j^(2/3) (Hoeffding-optimal).
        search_guided: m_j proportional to (w_j * sigma_j)^(2/3), sigma_j from
                       Stage A's suspicion score.
    """
    if strategy not in ALLOCATION_STRATEGIES:
        raise ValueError(
            f"unknown allocation strategy {strategy!r}; "
            f"choose from {ALLOCATION_STRATEGIES}"
        )
    eligible = [j for j in range(k_total) if weights.get(j, 0.0) > 0.0]

    if strategy == "uniform":
        scores = {j: 1.0 for j in range(k_total)}
    elif strategy == "proportional":
        scores = {j: weights.get(j, 0.0) for j in range(k_total)}
    elif strategy == "two_thirds":
        # Delegate the exponent to the shared implementation so there is one
        # place where the Lagrangian lives.
        w = np.array([weights.get(j, 0.0) for j in range(k_total)], dtype=float)
        try:
            base = optimal_allocation(w, int(total_budget), int(k_total))
            scores = {j: float(base[j]) for j in range(k_total)}
        except Exception as exc:  # noqa: BLE001
            logger.debug("optimal_allocation unavailable (%s); using w**(2/3)", exc)
            scores = {j: weights.get(j, 0.0) ** (2.0 / 3.0) for j in range(k_total)}
    else:  # search_guided
        scores = {}
        for j in range(k_total):
            p = float(np.clip(suspicion.get(j, 0.0), 0.0, 1.0))
            sigma = math.sqrt(max(p * (1.0 - p), 1e-6))
            scores[j] = (weights.get(j, 0.0) * sigma) ** (2.0 / 3.0)

    for j in range(k_total):
        if weights.get(j, 0.0) <= 0.0:
            scores[j] = 0.0

    return apportion(
        scores,
        total_budget=int(total_budget),
        budget_cap=int(budget_cap),
        eligible=eligible,
        min_per_region=int(min_per_region),
    )


# ----------------------------------------------------------------------
# The verifier
# ----------------------------------------------------------------------
class RLMVerifier:
    """Recursive Language Model verifier with a hard search/estimation split."""

    def __init__(
        self,
        safety_predicate: SafetyPredicate,
        embedder: InteractionEmbedder,
        mkg: ModelKnowledgeGraph,
        mutator: MutationOperator | None = None,
        max_depth: int = 3,
        branching_factor: int = 4,
        search_budget: int = 200,
        estimation_budget: int = 400,
        budget_cap: int = 200,
        neighborhood_hops: int = 2,
        seed: int = 42,
        allocation_strategy: str = "search_guided",
        min_per_region: int = 1,
        seed_fraction: float = 0.4,
        frontier_fraction: float = 0.3,
        probes_per_focus: int = 3,
        require_disjoint_pools: bool = True,
    ) -> None:
        """
        Args:
            safety_predicate: phi.  Must depend on the model output.
            embedder: psi.
            mkg: A density-checked :class:`ModelKnowledgeGraph`.
            mutator: Stage-A mutation operator.  Pass ``NullMutator`` /
                ``IdentityMutator`` for the mandated controls.
            max_depth: Recursion depth D for Stage A.
            branching_factor: Branching B for Stage A.
            search_budget: Stage-A query budget.  Spent entirely on discovery.
            estimation_budget: Stage-B query budget B = sum_j m_j.  This is the
                only budget that affects the certificate.
            budget_cap: M, the a-priori per-region cap in Theorem 1's union
                bound.  Declared here, never derived from data.
            neighborhood_hops: r for MKG-guided focus regions.
            seed: Master seed.  All streams derive from it via stable_rng.
            allocation_strategy: One of :data:`ALLOCATION_STRATEGIES`.
            min_per_region: Floor on m_j for regions with w_j > 0.
            seed_fraction: Share of the search budget spent on flat seeds.
            frontier_fraction: Share of the search budget reserved for MKG
                frontier/focus probing.  The remainder goes to recursion.
            probes_per_focus: Probes per focus region during frontier
                exploration.
            require_disjoint_pools: Enforce Stage A / Stage B pool disjointness.
        """
        self.phi = safety_predicate
        self.embedder = embedder
        self.mkg = mkg
        self.mutator = mutator or CompositeMutator(seed=seed)
        self.max_depth = int(max_depth)
        self.branching_factor = int(branching_factor)
        self.search_budget = int(search_budget)
        self.estimation_budget = int(estimation_budget)
        self.budget_cap = int(budget_cap)
        self.neighborhood_hops = int(neighborhood_hops)
        self.seed = int(seed)
        if allocation_strategy not in ALLOCATION_STRATEGIES:
            raise ValueError(
                f"unknown allocation strategy {allocation_strategy!r}"
            )
        self.allocation_strategy = allocation_strategy
        self.min_per_region = int(min_per_region)
        self.seed_fraction = float(seed_fraction)
        self.frontier_fraction = float(frontier_fraction)
        self.probes_per_focus = int(probes_per_focus)
        self.require_disjoint_pools = bool(require_disjoint_pools)

    # ------------------------------------------------------------------
    @property
    def partition(self) -> RegionPartition:
        return self.mkg.partition

    def get_verifier_descriptor(self) -> dict:
        """Serialisable configuration, for h(V) in the certificate."""
        return {
            "type": "RLMVerifier",
            "max_depth": self.max_depth,
            "branching_factor": self.branching_factor,
            "search_budget": self.search_budget,
            "estimation_budget": self.estimation_budget,
            "budget_cap": self.budget_cap,
            "neighborhood_hops": self.neighborhood_hops,
            "allocation_strategy": self.allocation_strategy,
            "min_per_region": self.min_per_region,
            "seed_fraction": self.seed_fraction,
            "frontier_fraction": self.frontier_fraction,
            "probes_per_focus": self.probes_per_focus,
            "seed": self.seed,
            "mutator": self.mutator.name,
            "predicate": getattr(self.phi, "descriptor", {"type": type(self.phi).__name__}),
            "embedder": getattr(self.embedder, "descriptor", {"type": type(self.embedder).__name__}),
            "mkg": self.mkg.summary(),
        }

    # ==================================================================
    # STAGE A -- SEARCH.  Writes only search counters.
    # ==================================================================
    def run_search(
        self,
        model_fn: Callable[[dict], str],
        search_pool: Sequence[dict],
        recorder: SearchRecorder | None = None,
    ) -> SearchReport:
        """Run mutation recursion + MKG frontier exploration over the search pool.

        The ONLY object this method writes region statistics through is a
        :class:`SearchRecorder`.  It never receives, and never obtains, a
        handle that can write estimation counters.  ``tests/test_verifier.py::
        TestStageSeparation::test_search_never_touches_estimation_counters``
        monkeypatches ``Region.record_estimation_sample`` to raise and runs this
        method to completion.
        """
        if recorder is None:
            recorder = self.partition.search_recorder()
        self.partition.reset_search_stats()

        # Two independent streams. The seed ORDER must not depend on which
        # mutator arm is running, or the mutator controls (null / identity)
        # would be compared against a different set of seed prompts and the
        # comparison would be meaningless.
        seed_rng = stable_rng(self.seed, "search", "seed_order")
        probe_rng = stable_rng(self.seed, "search", "frontier_probe")
        traces: list[VerifierTrace] = []
        explored: set[int] = set()
        failures: set[int] = set()
        queries = 0

        seed_budget = max(1, int(self.search_budget * self.seed_fraction))
        recursion_budget = max(
            0,
            int(self.search_budget * (1.0 - self.frontier_fraction)),
        )

        # -- Phase 1: flat seeds -----------------------------------------
        order = seed_rng.permutation(len(search_pool)) if len(search_pool) else []
        for pos in order:
            if queries >= min(seed_budget, self.search_budget):
                break
            x = deepcopy(search_pool[int(pos)])
            x["_source_id"] = f"seed:{int(pos)}"
            x["_depth"] = 0
            idx = self._probe(model_fn, x, recorder, traces, depth=0, parent_idx=-1,
                              mutation_type="seed")
            queries += 1
            t = traces[idx]
            explored.add(t.region_id)
            if t.violation:
                failures.add(t.region_id)

        # -- Phase 2: recursion on discovered failures --------------------
        queue = [i for i, t in enumerate(traces) if t.violation and t.depth < self.max_depth]
        qi = 0
        while qi < len(queue) and queries < recursion_budget:
            parent_idx = queue[qi]
            qi += 1
            parent = traces[parent_idx]
            for _ in range(self.branching_factor):
                if queries >= recursion_budget:
                    break
                child_x = self.mutator.mutate(
                    parent.interaction, state={"depth": parent.depth}
                )
                child_idx = self._probe(
                    model_fn, child_x, recorder, traces,
                    depth=parent.depth + 1, parent_idx=parent_idx,
                    mutation_type=str(child_x.get("_mutation_type", "unknown")),
                )
                queries += 1
                child = traces[child_idx]
                explored.add(child.region_id)
                if child.violation:
                    failures.add(child.region_id)
                    if child.depth < self.max_depth:
                        queue.append(child_idx)
                if parent.region_id != child.region_id:
                    self.mkg.add_mutation_edge(parent.region_id, child.region_id)

        # -- Phase 3: MKG-guided frontier / focus probing -----------------
        focus = self.mkg.get_focus_regions(failures, self.neighborhood_hops)
        focus = focus - explored
        frontier = self.mkg.get_frontier_regions(explored) - focus
        targets = list(focus) + list(frontier)

        for region_id in targets:
            if queries >= self.search_budget:
                break
            n_probes = self.probes_per_focus if region_id in focus else 1
            for _ in range(n_probes):
                if queries >= self.search_budget:
                    break
                probe = self._region_probe(region_id, traces, probe_rng)
                if probe is None:
                    break
                idx = self._probe(
                    model_fn, probe, recorder, traces, depth=0, parent_idx=-1,
                    mutation_type="frontier_probe",
                )
                queries += 1
                t = traces[idx]
                explored.add(t.region_id)
                if t.violation:
                    failures.add(t.region_id)

        return self._make_search_report(
            traces, queries, explored, failures, focus, frontier
        )

    def _probe(
        self,
        model_fn: Callable[[dict], str],
        interaction: dict,
        recorder: SearchRecorder,
        traces: list[VerifierTrace],
        depth: int,
        parent_idx: int,
        mutation_type: str,
    ) -> int:
        """Query the model once, evaluate phi, record to the SEARCH counters."""
        output = model_fn(interaction)
        evaluation = self.phi.evaluate(interaction, output)
        violation = not evaluation.is_safe
        region_id = self.partition.assign(self.embedder.embed(interaction))
        source_id = str(interaction.get("_source_id") or f"anon:{len(traces)}")
        recorder.record(region_id, violation, source_id)
        traces.append(
            VerifierTrace(
                interaction=interaction,
                output=output,
                violation=violation,
                region_id=region_id,
                depth=int(depth),
                source_id=source_id,
                parent_idx=int(parent_idx),
                mutation_type=mutation_type,
            )
        )
        return len(traces) - 1

    def _region_probe(
        self, region_id: int, traces: list[VerifierTrace], rng: np.random.Generator
    ) -> dict | None:
        """Mutate a trace from a neighbouring region toward ``region_id``."""
        if not traces:
            return None
        near = self.mkg.get_neighborhood(region_id, hops=1)
        candidates = [t for t in traces if t.region_id in near] or traces
        parent = candidates[int(rng.integers(0, len(candidates)))]
        return self.mutator.mutate(parent.interaction)

    def _make_search_report(
        self,
        traces: list[VerifierTrace],
        queries: int,
        explored: set[int],
        failures: set[int],
        focus: set[int],
        frontier: set[int],
    ) -> SearchReport:
        total_violations = sum(1 for t in traces if t.violation)
        violating_sources = {t.source_id for t in traces if t.violation}
        all_sources = {t.source_id for t in traces}
        depth_hist: dict[int, int] = {}
        for t in traces:
            depth_hist[t.depth] = depth_hist.get(t.depth, 0) + 1

        search_stats = self.mkg.get_search_stats()
        return SearchReport(
            search_stats=search_stats,
            suspicion=self._suspicion_scores(search_stats),
            traces=traces,
            queries=queries,
            total_violations=total_violations,
            distinct_violations=len(violating_sources),
            distinct_sources=len(all_sources),
            recursion_depth_hist=dict(sorted(depth_hist.items())),
            explored_regions=explored,
            failure_regions=failures,
            focus_regions=focus,
            frontier_regions=frontier,
        )

    @staticmethod
    def _suspicion_scores(
        search_stats: list[RegionStat], prior_strength: float = 2.0
    ) -> dict[int, float]:
        """Per-region suspicion, shrunk toward the global Stage-A rate.

        This is NOT an estimate of p_j.  Stage A hunts for failures, so its rate
        is biased upward by construction.  It is used only to steer where the
        i.i.d. Stage-B budget goes; the bound is valid for any steering rule.
        """
        n_tot = sum(rs.n_samples for rs in search_stats)
        v_tot = sum(rs.n_violations for rs in search_stats)
        g = (v_tot / n_tot) if n_tot else 0.5
        return {
            rs.region_id: (rs.n_violations + prior_strength * g)
            / (rs.n_samples + prior_strength)
            for rs in search_stats
        }

    # ==================================================================
    # STAGE B -- ESTIMATION.  Writes only estimation counters.
    # ==================================================================
    def run_estimation(
        self,
        model_fn: Callable[[dict], str],
        estimation_pool: Sequence[dict],
        allocation: dict[int, int],
        recorder: EstimationRecorder | None = None,
    ) -> int:
        """Draw m_j fresh i.i.d. samples from D|R_j and record them.

        Samples come from :meth:`RegionPartition.sample_from_region`, which
        draws WITH REPLACEMENT from the region's stratum of the estimation
        pool.  With-replacement draws conditional on the region are i.i.d. from
        the empirical D|R_j, which is the hypothesis Theorem 1 requires.
        """
        if recorder is None:
            recorder = self.partition.estimation_recorder()
        self.partition.reset_estimation_stats()

        spent = 0
        for region_id in sorted(allocation):
            m_j = int(allocation[region_id])
            if m_j <= 0:
                continue
            if m_j > self.budget_cap:
                raise ValueError(
                    f"m_{region_id}={m_j} exceeds the a-priori budget_cap "
                    f"M={self.budget_cap}; Theorem 1's union bound covers only "
                    "m in [1, M]"
                )
            rng = stable_rng(self.seed, "estimation", str(region_id))
            for pool_idx in self.partition.sample_from_region(region_id, m_j, rng):
                x = estimation_pool[pool_idx]
                output = model_fn(x)
                evaluation = self.phi.evaluate(x, output)
                recorder.record(
                    region_id, not evaluation.is_safe, f"est:{pool_idx}"
                )
                spent += 1
        return spent

    # ==================================================================
    # FULL PROTOCOL
    # ==================================================================
    def verify(
        self,
        model_fn: Callable[[dict], str],
        search_pool: Sequence[dict],
        estimation_pool: Sequence[dict],
        epsilon: float,
        delta: float,
        estimation_vectors: np.ndarray | None = None,
    ) -> VerificationResult:
        """Run Stage A, Stage B and Stage C and return the full result.

        Args:
            model_fn: x -> y.  One model query per call.
            search_pool: Stage-A pool.  Never sampled by Stage B.
            estimation_pool: Stage-B pool.  Disjoint from ``search_pool``.
            epsilon: Target violation bound.
            delta: Confidence parameter.
            estimation_vectors: Optional precomputed embeddings of
                ``estimation_pool`` (saves a pass through the embedder).

        Returns:
            A :class:`VerificationResult`.  ``result.acceptance`` carries the
            certificate, computed from ``estimation_stats`` alone.
        """
        if self.require_disjoint_pools:
            assert_pools_disjoint(search_pool, estimation_pool)

        self.partition.reset_all_stats()

        # Weights and strata come from the ESTIMATION pool and are frozen here,
        # before any Stage-A observation exists.
        if estimation_vectors is None:
            estimation_vectors = self.embedder.embed_batch(list(estimation_pool))
        self.partition.stratify(estimation_vectors)
        weights = self.partition.empirical_weights()

        # -- Stage A ------------------------------------------------------
        report = self.run_search(model_fn, search_pool)

        # -- Stage B ------------------------------------------------------
        k_total = self.partition.k
        allocation = build_allocation(
            strategy=self.allocation_strategy,
            weights=weights,
            suspicion=report.suspicion,
            total_budget=self.estimation_budget,
            budget_cap=self.budget_cap,
            k_total=k_total,
            min_per_region=self.min_per_region,
        )
        estimation_queries = self.run_estimation(
            model_fn, estimation_pool, allocation
        )

        # -- Stage C ------------------------------------------------------
        estimation_stats = self.mkg.get_estimation_stats()
        acceptance = check_acceptance(
            estimation_stats,
            epsilon=epsilon,
            delta=delta,
            k_total=k_total,
            budget_cap=self.budget_cap,
        )

        return VerificationResult(
            estimation_stats=estimation_stats,
            search_stats=report.search_stats,
            allocation=allocation,
            total_violations=report.total_violations,
            distinct_violations=report.distinct_violations,
            distinct_sources=report.distinct_sources,
            search_queries=report.queries,
            estimation_queries=estimation_queries,
            recursion_depth_hist=report.recursion_depth_hist,
            weights=weights,
            k_total=k_total,
            budget_cap=self.budget_cap,
            allocation_strategy=self.allocation_strategy,
            acceptance=acceptance,
            mkg_summary=self.mkg.summary(),
            suspicion=report.suspicion,
            search_report=report,
        )
