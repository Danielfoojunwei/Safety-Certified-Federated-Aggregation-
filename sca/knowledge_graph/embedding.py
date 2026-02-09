"""Embedding functions psi: X -> R^p for mapping interactions to a metric space.

Section 3.1: psi is a prompt+context+trace encoder used to define the
measurable partition X = bigsqcup R_j.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class InteractionEmbedder(ABC):
    """Abstract base class for interaction embedders.

    Maps interaction transcripts x in X to embedding vectors in R^p.
    """

    @abstractmethod
    def embed(self, interaction: dict) -> np.ndarray:
        """Embed a single interaction into R^p.

        Args:
            interaction: A dictionary representing the interaction
                (prompt, context, trace, etc.).

        Returns:
            Embedding vector of shape (p,).
        """

    def embed_batch(self, interactions: list[dict]) -> np.ndarray:
        """Embed a batch of interactions.

        Args:
            interactions: List of interaction dictionaries.

        Returns:
            Embedding matrix of shape (n, p).
        """
        return np.stack([self.embed(x) for x in interactions])


class RandomProjectionEmbedder(InteractionEmbedder):
    """Embedding via random projection of bag-of-words features.

    Suitable for simulation / prototyping. In a real system, this would
    be replaced by a sentence transformer or similar learned encoder.
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64,
                 seed: int = 42) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        rng = np.random.RandomState(seed)
        self.projection = rng.randn(vocab_size, embed_dim) / np.sqrt(embed_dim)

    def embed(self, interaction: dict) -> np.ndarray:
        """Hash-based bag-of-words -> random projection."""
        text = str(interaction.get("prompt", "")) + str(interaction.get("context", ""))
        bow = np.zeros(self.vocab_size)
        for token in text.split():
            idx = hash(token) % self.vocab_size
            bow[idx] += 1
        if bow.sum() > 0:
            bow /= bow.sum()
        return bow @ self.projection


class PrecomputedEmbedder(InteractionEmbedder):
    """Embedder that looks up precomputed embeddings from a dictionary.

    Useful when embeddings have been computed offline (e.g., via a
    sentence transformer) and stored.
    """

    def __init__(self, embeddings: dict[str, np.ndarray]) -> None:
        """
        Args:
            embeddings: Mapping from interaction ID to embedding vector.
        """
        self.embeddings = embeddings

    def embed(self, interaction: dict) -> np.ndarray:
        key = interaction.get("id", str(interaction))
        if key in self.embeddings:
            return self.embeddings[key]
        raise KeyError(f"No precomputed embedding for interaction: {key}")
