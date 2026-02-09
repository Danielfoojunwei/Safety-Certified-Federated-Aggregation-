"""Tests for the RLM Verifier."""

import numpy as np
import pytest

from sca.knowledge_graph.embedding import RandomProjectionEmbedder
from sca.knowledge_graph.mkg import ModelKnowledgeGraph
from sca.knowledge_graph.regions import InteractionRegion, RegionPartition

import numpy as np
from sca.verifier.mutations import (
    CompositeMutator,
    EscalationMutator,
    ParaphraseMutator,
    TemplateMutator,
)
from sca.verifier.rlm_verifier import RLMVerifier
from sca.verifier.safety_predicate import KeywordSafetyPredicate


def make_test_mkg(n_regions=5, dim=64):
    """Create a test MKG."""
    partition = RegionPartition(tau_new=5.0)
    rng = np.random.RandomState(42)
    for j in range(n_regions):
        partition.regions.append(InteractionRegion(
            region_id=j,
            centroid=rng.randn(dim),
            weight=1.0 / n_regions,
        ))
    return ModelKnowledgeGraph(partition, tau=3.0)


class TestMutations:
    def test_paraphrase(self):
        m = ParaphraseMutator(seed=0)
        interaction = {"prompt": "test query"}
        child = m.mutate(interaction)
        assert child["prompt"] != interaction["prompt"]
        assert "mutation_type" in child

    def test_escalation(self):
        m = EscalationMutator(seed=0)
        interaction = {"prompt": "test query"}
        child = m.mutate(interaction)
        assert "turns" in child

    def test_template(self):
        m = TemplateMutator(seed=0)
        interaction = {"prompt": "test query"}
        child = m.mutate(interaction)
        assert child["prompt"] != interaction["prompt"]

    def test_composite(self):
        m = CompositeMutator(seed=0)
        interaction = {"prompt": "test query"}
        child = m.mutate(interaction)
        assert "mutation_operator" in child


class TestRLMVerifier:
    def setup_method(self):
        self.predicate = KeywordSafetyPredicate(
            forbidden_keywords=["unsafe"],
            refusal_phrases=["I cannot"],
        )
        self.embedder = RandomProjectionEmbedder(embed_dim=64, seed=42)
        self.mkg = make_test_mkg(n_regions=5, dim=64)

    def test_safe_model_accepted(self):
        """A model that never produces unsafe output should be accepted.

        With zero violations and sufficient seed coverage, the UCB
        bound should be below epsilon.
        """
        def safe_model(interaction):
            return "This is a safe response."

        # Generate enough seeds to fill the budget, ensuring tight UCBs
        seeds = [{"prompt": f"test prompt {i}"} for i in range(200)]
        embeddings = self.embedder.embed_batch(seeds)

        partition = RegionPartition(tau_new=100.0)
        partition.initialize_from_embeddings(embeddings, n_clusters=2)
        mkg = ModelKnowledgeGraph(partition, tau=100.0)

        verifier = RLMVerifier(
            safety_predicate=self.predicate,
            embedder=self.embedder,
            mkg=mkg,
            total_budget=200,
            max_depth=2,
            branching_factor=2,
        )

        accepted, state, stats = verifier.verify(
            model_fn=safe_model,
            seed_interactions=seeds,
            epsilon=0.2,
            delta=0.05,
        )
        assert accepted
        assert len(state.traces) > 0

    def test_unsafe_model_rejected(self):
        """A model that always produces unsafe output should be rejected."""
        def unsafe_model(interaction):
            return "unsafe response with violation"

        verifier = RLMVerifier(
            safety_predicate=self.predicate,
            embedder=self.embedder,
            mkg=self.mkg,
            total_budget=50,
            max_depth=2,
            branching_factor=2,
        )

        seeds = [{"prompt": f"test prompt {i}"} for i in range(10)]
        accepted, state, stats = verifier.verify(
            model_fn=unsafe_model,
            seed_interactions=seeds,
            epsilon=0.1,
            delta=0.05,
        )
        assert not accepted
        assert any(t.violation for t in state.traces)

    def test_recursive_expansion(self):
        """Verifier should expand recursively on failures."""
        call_count = [0]

        def mixed_model(interaction):
            call_count[0] += 1
            # First few responses are unsafe, rest are safe
            if "escalation" in str(interaction.get("mutation_type", "")):
                return "unsafe output here"
            if call_count[0] <= 3:
                return "unsafe response"
            return "safe response"

        verifier = RLMVerifier(
            safety_predicate=self.predicate,
            embedder=self.embedder,
            mkg=self.mkg,
            total_budget=100,
            max_depth=3,
            branching_factor=3,
        )

        seeds = [{"prompt": f"prompt {i}"} for i in range(5)]
        accepted, state, stats = verifier.verify(
            model_fn=mixed_model,
            seed_interactions=seeds,
            epsilon=0.3,
            delta=0.05,
        )

        # Should have expanded beyond just the seed interactions
        assert len(state.traces) > len(seeds)

    def test_budget_respected(self):
        """Verifier should not exceed the total budget."""
        def model_fn(interaction):
            return "safe response"

        verifier = RLMVerifier(
            safety_predicate=self.predicate,
            embedder=self.embedder,
            mkg=self.mkg,
            total_budget=20,
            max_depth=5,
            branching_factor=10,
        )

        seeds = [{"prompt": f"prompt {i}"} for i in range(50)]
        _, state, _ = verifier.verify(
            model_fn=model_fn,
            seed_interactions=seeds,
            epsilon=0.1,
            delta=0.05,
        )
        assert state.total_queries <= 20

    def test_verifier_descriptor(self):
        verifier = RLMVerifier(
            safety_predicate=self.predicate,
            embedder=self.embedder,
            mkg=self.mkg,
        )
        desc = verifier.get_verifier_descriptor()
        assert desc["type"] == "RLMVerifier"
        assert "max_depth" in desc
        assert "branching_factor" in desc
