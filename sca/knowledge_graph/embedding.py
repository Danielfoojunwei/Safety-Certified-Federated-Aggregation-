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

    Simulation-grade embedder for prototyping. Uses hash-based bag-of-words
    with random projection -- no semantic understanding. Two semantically
    identical sentences with different wording will get different embeddings.

    For production use, see SentenceTransformerEmbedder.
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


class SentenceTransformerEmbedder(InteractionEmbedder):
    """Semantic embedding via sentence-transformers (production-grade).

    Uses a pretrained sentence transformer model to produce semantically
    meaningful embeddings. Region assignments will reflect actual semantic
    similarity, and mutations that change meaning will cross region boundaries.

    Requires: ``pip install sentence-transformers``
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        fields: list[str] | None = None,
    ) -> None:
        """
        Args:
            model_name: HuggingFace sentence-transformer model name.
            fields: Interaction dict fields to concatenate for embedding.
                   Defaults to ["prompt"].
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._available = True
        except ImportError:
            self._available = False
            self._fallback = RandomProjectionEmbedder()
        self.fields = fields or ["prompt"]

    def embed(self, interaction: dict) -> np.ndarray:
        text = " ".join(
            str(interaction.get(f, "")) for f in self.fields
        ).strip()
        if not text:
            text = "empty"

        if self._available:
            return self.model.encode(text, show_progress_bar=False)

        # Fallback to random projection if sentence-transformers unavailable
        return self._fallback.embed(interaction)

    def embed_batch(self, interactions: list[dict]) -> np.ndarray:
        texts = []
        for interaction in interactions:
            text = " ".join(
                str(interaction.get(f, "")) for f in self.fields
            ).strip()
            texts.append(text if text else "empty")

        if self._available:
            return self.model.encode(texts, show_progress_bar=False)

        return np.stack([self._fallback.embed(x) for x in interactions])


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
