"""Embedding functions psi: X -> R^p used to induce the region partition.

REPRODUCIBILITY NOTE (this is the fix for the non-reproducible-results defect).
The previous version of :class:`HashedBagOfWordsEmbedder` (then called
``RandomProjectionEmbedder``) bucketed tokens with the builtin ``hash()``.
CPython randomises string hashing per process unless ``PYTHONHASHSEED`` is set,
which this repository never set, so three processes with ``seed=42`` produced
three different embeddings, three different partitions, and three different
certified bounds.  Every string bucketing in this module now goes through
:func:`sca.utils.seeding.stable_hash`, a blake2b digest that is a pure function
of the string.  ``tests/test_mkg.py::TestEmbeddingReproducibility`` spawns real
subprocesses to check this -- an in-process test cannot catch it, because
within one process ``hash()`` is perfectly consistent.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from sca.utils.seeding import stable_hash

logger = logging.getLogger(__name__)


class InteractionEmbedder(ABC):
    """Maps interaction transcripts x in X to embedding vectors in R^p."""

    @abstractmethod
    def embed(self, interaction: dict) -> np.ndarray:
        """Embed a single interaction into R^p."""

    def embed_batch(self, interactions: list[dict]) -> np.ndarray:
        """Embed a batch.  Returns an (n, p) matrix."""
        if not interactions:
            return np.zeros((0, self.dim))
        return np.stack([self.embed(x) for x in interactions])

    @property
    def dim(self) -> int:  # pragma: no cover - overridden by concrete classes
        raise NotImplementedError

    @property
    def descriptor(self) -> dict:
        """Serialisable identity of this embedder, for the certificate."""
        return {"type": type(self).__name__, "dim": self.dim}


def _interaction_text(interaction: dict, fields: list[str]) -> str:
    parts = []
    for f in fields:
        v = interaction.get(f)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            parts.append(" ".join(str(t) for t in v))
        else:
            parts.append(str(v))
    text = " ".join(parts).strip()
    return text if text else "empty"


class HashedBagOfWordsEmbedder(InteractionEmbedder):
    """Process-independent hashed bag-of-words followed by a random projection.

    Cheap, deterministic, and semantically blind: two paraphrases of the same
    request land far apart.  Use it for unit tests and for ablations where the
    embedder must be free; use :class:`TransformersEmbedder` for anything whose
    number appears in the paper, because a semantically blind partition makes
    "region" mean "surface form" rather than "kind of request".
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 64,
        seed: int = 42,
        fields: list[str] | None = None,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.seed = int(seed)
        self.fields = fields or ["prompt", "context"]
        rng = np.random.default_rng(seed)
        self.projection = rng.standard_normal(
            (self.vocab_size, self.embed_dim)
        ) / np.sqrt(self.embed_dim)

    @property
    def dim(self) -> int:
        return self.embed_dim

    def embed(self, interaction: dict) -> np.ndarray:
        text = _interaction_text(interaction, self.fields)
        bow = np.zeros(self.vocab_size)
        for token in text.split():
            # stable_hash, NOT builtin hash(): see module docstring.
            bow[stable_hash(token) % self.vocab_size] += 1.0
        total = bow.sum()
        if total > 0:
            bow /= total
        return bow @ self.projection

    @property
    def descriptor(self) -> dict:
        return {
            "type": "HashedBagOfWordsEmbedder",
            "dim": self.embed_dim,
            "vocab_size": self.vocab_size,
            "seed": self.seed,
            "fields": list(self.fields),
        }


#: Backwards-compatible alias for the old, hash()-based name.
RandomProjectionEmbedder = HashedBagOfWordsEmbedder


class TransformersEmbedder(InteractionEmbedder):
    """Mean-pooled, L2-normalised sentence embeddings from a HuggingFace model.

    Defaults to ``sentence-transformers/all-MiniLM-L6-v2`` loaded through plain
    ``transformers`` (the ``sentence-transformers`` package is not required).
    Runs on CPU in eval mode with grad disabled, so it is deterministic: the
    same text gives byte-identical vectors across processes.

    This is the embedder that should back any reported partition.  Unlike the
    hashed bag of words it puts "how do I build a bomb" next to "how do I make
    an explosive device", which is the only way region membership can mean
    anything.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        fields: list[str] | None = None,
        max_length: int = 128,
        batch_size: int = 32,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.fields = fields or ["prompt"]
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._dim = int(self.model.config.hidden_size)
        self._cache: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        out = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            out.append(pooled.numpy().astype(np.float64))
        return np.concatenate(out, axis=0)

    def embed(self, interaction: dict) -> np.ndarray:
        return self.embed_batch([interaction])[0]

    def embed_batch(self, interactions: list[dict]) -> np.ndarray:
        if not interactions:
            return np.zeros((0, self.dim))
        texts = [_interaction_text(x, self.fields) for x in interactions]
        missing = [t for t in texts if t not in self._cache]
        if missing:
            uniq = list(dict.fromkeys(missing))
            vecs = self._encode(uniq)
            for t, v in zip(uniq, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])

    @property
    def descriptor(self) -> dict:
        return {
            "type": "TransformersEmbedder",
            "model_name": self.model_name,
            "dim": self._dim,
            "fields": list(self.fields),
            "max_length": self.max_length,
        }


class PrecomputedEmbedder(InteractionEmbedder):
    """Looks up embeddings computed offline, keyed by interaction id.

    Raises on a cache miss rather than silently substituting a default vector:
    a silent fallback would make a partition that looks fitted but is not.
    """

    def __init__(self, embeddings: dict[str, np.ndarray], dim: int | None = None) -> None:
        self.embeddings = embeddings
        if dim is None:
            dim = len(next(iter(embeddings.values()))) if embeddings else 0
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, interaction: dict) -> np.ndarray:
        key = interaction.get("id")
        if key is None:
            key = str(stable_hash(str(sorted(interaction.items()))))
        if key in self.embeddings:
            return np.asarray(self.embeddings[key], dtype=float)
        raise KeyError(f"No precomputed embedding for interaction id={key!r}")


def make_embedder(kind: str = "hashed", **kwargs) -> InteractionEmbedder:
    """Factory used by the experiment drivers.

    ``kind="transformer"`` is the one whose numbers may be reported.
    ``kind="hashed"`` is for tests and free ablations.
    """
    kind = kind.lower()
    if kind in ("hashed", "bow", "random_projection"):
        return HashedBagOfWordsEmbedder(**kwargs)
    if kind in ("transformer", "transformers", "minilm", "st"):
        return TransformersEmbedder(**kwargs)
    raise ValueError(f"unknown embedder kind: {kind!r}")
