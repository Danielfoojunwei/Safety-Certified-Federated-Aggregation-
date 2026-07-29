"""The one model used by every arm of every experiment.

The old harness defined its classifier inline in ``run_real_evaluation.py``,
next to an inline FedAvg, an inline Krum and an inline acceptance gate.  That
is how audit findings F2 (the gate read ground-truth labels), F5 (rollback was
a no-op) and F7 (single-class clients collapsed Krum) all survived: nothing was
shared, so nothing could be tested once.

This module holds the model and nothing else.  It is imported by the FL arm,
the verifier arm and the smoke test, so a change here changes every arm at
once.

Initialisation is deterministic through
:func:`sca.utils.seeding.seed_torch_generator` rather than a global
``torch.manual_seed``.  Builder E's warning applies: a global seed couples
every component's stream, so adding one draw in the data loader would silently
shift the model's initial weights and a 10-seed sweep would stop reproducing
after any edit.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

from sca.utils.seeding import seed_torch_generator

__all__ = [
    "SafetyClassifier",
    "MLPSafetyClassifier",
    "MODEL_REGISTRY",
    "make_model",
    "pretrain",
    "parameter_count",
    "REFUSAL_TEXT",
    "make_model_fn",
]

#: What the safety filter emits when it declines to release a response.
#: It is a fixed string so that the safety predicate can be given a verdict for
#: it up front (a refusal is safe), and so that no released text can ever fall
#: outside the annotated pool.
REFUSAL_TEXT = "[REFUSED] I can't help with that request."


class SafetyClassifier(nn.Module):
    """Bag-of-words logistic classifier over GPT-2 token ids.

    ``logits = mean_{t in x, t != PAD} W[t] + b``, i.e. one weight vector per
    vocabulary entry, mean-pooled over the non-padding tokens.  100,516
    parameters at the default vocabulary.  :func:`parameter_count` is called by
    the runner and the real number is written into the results file, because
    the README used to state a parameter count that did not match the model
    (finding F14).

    WHY THIS AND NOT THE OLD MLP -- a measured decision, not a preference.
    The pre-rebuild harness used ``Embedding -> mean pool -> 2-layer MLP``
    (:class:`MLPSafetyClassifier`, kept below so the measurement is
    reproducible).  Trained centrally on the full 2400-example client union of
    real PKU-SafeRLHF and scored on the held-out test split, that architecture
    never leaves the constant-predictor regime::

        MLP  lr=0.05 mom=0.0  ep10: acc=0.5800  predicted class fractions {0: 0.325, 1: 0.675}
        MLP  lr=0.5  mom=0.9  ep10: acc=0.5300  predicted class fractions {0: 0.000, 1: 1.000}
        MLP  lr=2.0  mom=0.9  ep10: acc=0.5300  predicted class fractions {0: 0.000, 1: 1.000}
        BoW  lr=0.5  mom=0.9  ep10: acc=0.6350  predicted class fractions {0: 0.343, 1: 0.657}
        BoW  lr=5.0  mom=0.9  ep6 : acc=0.7150  predicted class fractions {0: 0.323, 1: 0.677}

    0.53 and 0.47 are exactly the two class priors.  An FL harness whose model
    oscillates between "predict everything unsafe" and "predict everything
    safe" cannot support any Byzantine claim: acceptance criterion C1 would be
    decided by which constant a run happened to land on, which is a variant of
    the same defect F7 describes for Krum.  A zero-initialised bag-of-words
    head puts each token's weight directly in the logit, so the gradient signal
    survives a 2400-example corpus.

    The trade-off is stated rather than hidden: this model is linear in
    bag-of-words features and is not a language model.  No claim about
    "safety classification of LLM outputs" in general is licensed by it.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_classes = int(n_classes)
        self.embedding = nn.Embedding(vocab_size, n_classes, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.embedding(x)                          # (B, L, C)
        mask = (x != 0).float().unsqueeze(-1)          # (B, L, 1)
        pooled = (w * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled + self.bias


class MLPSafetyClassifier(nn.Module):
    """The pre-rebuild architecture, kept so its failure stays reproducible.

    Embedding -> masked mean pool -> 2-layer MLP -> {safe, unsafe}; 6,466,690
    parameters at the default configuration.  Reachable through
    ``make_model(..., kind="mlp")`` and ``run_all --model mlp``.  See
    :class:`SafetyClassifier` for the measurements that disqualified it.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                       # (B, L, E)
        mask = (x != 0).float().unsqueeze(-1)         # (B, L, 1)
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        h = self.dropout(torch.relu(self.norm1(self.fc1(pooled))))
        h = self.dropout(torch.relu(self.norm2(self.fc2(h))))
        return self.classifier(h)


def _init_deterministic(model: nn.Module, seed: int) -> None:
    """Re-initialise every parameter from an explicit torch Generator.

    ``nn.Module`` constructors draw from the *global* torch RNG.  Seeding that
    globally would couple the model's weights to every other component's draw
    order.  We therefore overwrite the parameters from a private stream keyed
    on ``(seed, "model_init")``.

    :class:`SafetyClassifier` is the deliberate exception: its bag-of-words
    weight table starts at exactly zero.  That is what makes it trainable on a
    corpus this size (an unseen token contributes nothing rather than noise),
    and it also makes the initial checkpoint seed-independent, which is a
    property the FL controls rely on.
    """
    if isinstance(model, SafetyClassifier):
        with torch.no_grad():
            model.embedding.weight.zero_()
            model.bias.zero_()
        return

    gen = seed_torch_generator(seed, "model_init")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("weight") and p.dim() == 1:
                # LayerNorm gain -- must start at 1, not at a random value.
                p.fill_(1.0)
            elif p.dim() >= 2:
                fan_in = int(p.shape[1])
                bound = 1.0 / math.sqrt(max(fan_in, 1))
                p.uniform_(-bound, bound, generator=gen)
            else:
                p.zero_()
        emb = getattr(model, "embedding", None)
        if isinstance(emb, nn.Embedding) and emb.padding_idx is not None:
            emb.weight[emb.padding_idx].zero_()


#: ``kind`` -> constructor.  ``"mlp"`` is the pre-rebuild architecture, kept
#: reachable so its constant-predictor failure can be reproduced on demand.
MODEL_REGISTRY: dict[str, Any] = {
    "bow": SafetyClassifier,
    "mlp": MLPSafetyClassifier,
}


def make_model(
    seed: int = 0,
    *,
    kind: str = "bow",
    vocab_size: int = 50257,
    **kwargs: Any,
) -> nn.Module:
    """Build a model with reproducible initial weights.

    Args:
        seed: Master seed; only affects ``kind="mlp"`` (the bag-of-words head
            starts at exactly zero).
        kind: ``"bow"`` (default) or ``"mlp"``.  See :data:`MODEL_REGISTRY`.
        vocab_size: Tokenizer vocabulary.  Must match the tokenizer used to
            build the splits.
    """
    if kind not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model kind {kind!r}; choose from {sorted(MODEL_REGISTRY)}"
        )
    model = MODEL_REGISTRY[kind](vocab_size=vocab_size, **kwargs)
    _init_deterministic(model, seed)
    return model


def parameter_count(model: nn.Module) -> dict[str, int]:
    """Programmatic parameter count (finding F14 -- never hardcode this)."""
    from sca.experiments.metrics import count_parameters

    return count_parameters(model)


def pretrain(
    model: nn.Module,
    split: Any,
    *,
    epochs: int = 3,
    lr: float = 0.05,
    batch_size: int = 32,
    seed: int = 0,
) -> nn.Module:
    """Warm-start ``model`` on ``split``, in place, and return it.

    ``split`` MUST be server-owned data.  The runner passes the Stage-A search
    pool: it is neither client training data (so clean FedAvg still has
    something to learn, which is what acceptance criterion C1 measures) nor the
    Stage-B estimation pool (so the certificate's samples remain fresh with
    respect to everything the model has seen) nor the held-out test set (F1).
    """
    from torch.utils.data import DataLoader

    ds = split.to_tensor_dataset() if hasattr(split, "to_tensor_dataset") else split
    gen = seed_torch_generator(seed, "pretrain")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            opt.zero_grad()
            lossf(model(xb), yb.long()).backward()
            opt.step()
    model.eval()
    return model


def make_model_fn(
    model: nn.Module,
    tokenizer: Callable[[Sequence[str]], torch.Tensor],
    *,
    refusal_text: str = REFUSAL_TEXT,
) -> Callable[[dict], str]:
    """Turn a classifier into the ``x -> y`` map the verifier queries.

    The model is deployed as a *release filter*: it reads the (prompt,
    candidate response) pair and either releases the response verbatim or
    emits ``refusal_text``.  The verifier's safety predicate then judges the
    RELEASED TEXT.

    This is the structural fix for finding F2.  The old gate computed
    ``is_violation = (pred != true_label)``, i.e. the certified quantity was
    identically the empirical error rate on the set being scored.  Here a
    violation is "an unsafe response reached the user", which is a different
    quantity from classification error in both directions: wrongly refusing a
    safe response costs utility but is not a safety violation, and the
    predicate never sees the label of the input.

    The returned callable is memoised on the interaction text, because Stage A
    re-queries mutants of the same prompt many times.  Memoisation is sound
    here only because the model is frozen for the duration of one verification
    run; the gate rebuilds the callable for every candidate model.
    """
    cache: dict[str, str] = {}
    model.eval()

    def model_fn(interaction: dict) -> str:
        prompt = str(interaction.get("prompt", ""))
        response = str(interaction.get("response", ""))
        context = str(interaction.get("context", ""))
        key = context + "\x00" + prompt + "\x00" + response
        hit = cache.get(key)
        if hit is not None:
            return hit
        ids = tokenizer([f"{context} {prompt} {response}".strip()])
        with torch.no_grad():
            pred = int(model(ids).argmax(dim=-1)[0].item())
        out = refusal_text if pred == 1 else response
        cache[key] = out
        return out

    return model_fn
