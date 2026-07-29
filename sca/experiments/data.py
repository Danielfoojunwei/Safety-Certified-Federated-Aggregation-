"""Real data loading and a leakage-proof split protocol.

This module exists to fix audit finding **F1** (100% train/test leakage).  The
previous harness partitioned *all* of the corpus across FL clients and then
drew the "test set" from the same index range, so every reported accuracy was
a training accuracy.

The fix is structural, not cosmetic:

1.  A single immutable :class:`Corpus` holds ``n`` examples with stable integer
    ids ``0 .. n-1``.
2.  :func:`three_way_split` partitions **integer ids first**, into pairwise
    disjoint blocks, before a single tensor is built.
3.  Tensors are materialised *from* those id blocks.  There is no code path
    that can put the same corpus row into two splits, because a row is
    identified by its id and every id is placed exactly once.
4.  :meth:`SplitBundle.assert_disjoint` re-derives the overlaps from the stored
    ids and raises on any non-zero intersection.  Experiments must call it and
    print :meth:`SplitBundle.overlap_report` into their results file so a
    reader can check the zeros themselves.

The split is four-way in practice (the contract calls it a three-way split
because the server verification pool is one logical role):

    client training pools    -- non-IID via a Dirichlet(alpha) label mixture
    server verification pool -- the ONLY data the acceptance gate may see,
                                itself carved into two disjoint halves:
                                  * search pool     (Stage A: mutation /
                                    frontier exploration, never certified)
                                  * estimation pool (Stage B: fresh i.i.d.
                                    draws that feed the bound)
    held-out test set        -- final reporting only; the gate never sees it

Why Dirichlet and not sort-and-chunk: audit finding **F7**.  Sorting by label
and chunking contiguously makes every client single-class, which makes Krum
degenerate into a constant predictor even with zero adversaries.  A
Dirichlet(alpha) label mixture gives genuine, tunable non-IID-ness while
keeping every client multi-class with high probability.

Labels: see :func:`load_pku_saferlhf` for exactly how the binary safe/unsafe
label is derived from PKU-SafeRLHF's ``is_response_0_safe`` /
``is_response_1_safe`` fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Sequence

import numpy as np

from sca.utils.seeding import stable_hash, stable_rng

logger = logging.getLogger(__name__)

PKU_DATASET_ID = "PKU-Alignment/PKU-SafeRLHF"


# ---------------------------------------------------------------------------
# Split roles -- make the split identity part of the type, so that a future
# contributor cannot repeat F2 (the gate reading the set it is scored on).
# ---------------------------------------------------------------------------


class SplitRole(Enum):
    """What a split is allowed to be used for.

    The role travels with the data.  :func:`sca.experiments.evaluation.evaluate`
    refuses combinations of role and purpose that would constitute leakage.
    """

    CLIENT_TRAIN = "client_train"
    #: Stage A of the verification protocol.  Mutants, frontier probes,
    #: anything adaptive.  Never enters a certificate.
    SERVER_SEARCH = "server_search"
    #: Stage B.  Fresh i.i.d. draws per region; the only thing the bound sees.
    SERVER_ESTIMATION = "server_estimation"
    #: Final reporting only.  The gate must never touch this.
    HELDOUT_TEST = "heldout_test"


#: Roles the acceptance gate is permitted to read.
GATE_VISIBLE_ROLES = frozenset({SplitRole.SERVER_SEARCH, SplitRole.SERVER_ESTIMATION})

#: Roles that may back a number printed in the paper as "test accuracy".
REPORTABLE_ROLES = frozenset({SplitRole.HELDOUT_TEST})


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Corpus:
    """An immutable labelled text corpus with stable integer ids.

    Attributes:
        name: Provenance string, e.g. ``"PKU-Alignment/PKU-SafeRLHF"``.
        prompts: One prompt per row.
        responses: One response per row.
        labels: ``0`` = safe, ``1`` = unsafe.  ``numpy`` int64 array.
        label_rule: Human-readable statement of how ``labels`` was derived.
        meta: Free-form provenance (split string, revision, filters applied).
    """

    name: str
    prompts: tuple[str, ...]
    responses: tuple[str, ...]
    labels: np.ndarray
    label_rule: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.prompts)
        if len(self.responses) != n:
            raise ValueError("prompts and responses must have equal length")
        if self.labels.shape != (n,):
            raise ValueError(f"labels must have shape ({n},), got {self.labels.shape}")
        if n and not np.isin(self.labels, (0, 1)).all():
            raise ValueError("labels must be binary 0/1")

    def __len__(self) -> int:
        return len(self.prompts)

    @property
    def ids(self) -> np.ndarray:
        """The canonical id array ``0 .. n-1``."""
        return np.arange(len(self), dtype=np.int64)

    def text(self, i: int) -> str:
        """Concatenated ``prompt + response`` used as the model input."""
        return f"{self.prompts[i]} {self.responses[i]}"

    def texts(self, ids: Sequence[int] | np.ndarray | None = None) -> list[str]:
        if ids is None:
            ids = range(len(self))
        return [self.text(int(i)) for i in ids]

    def class_balance(self) -> dict[str, float | int]:
        """Real, measured class balance.  Never hardcode this in a paper."""
        n = len(self)
        n_unsafe = int(self.labels.sum()) if n else 0
        return {
            "n": n,
            "n_safe": n - n_unsafe,
            "n_unsafe": n_unsafe,
            "frac_unsafe": (n_unsafe / n) if n else 0.0,
        }


# ---------------------------------------------------------------------------
# Loading real data
# ---------------------------------------------------------------------------


def load_pku_saferlhf(
    n: int,
    seed: int,
    *,
    hf_split: str = "train",
    dataset_id: str = PKU_DATASET_ID,
    one_response_per_prompt: bool = False,
) -> Corpus:
    """Load ``n`` real PKU-SafeRLHF examples as a binary safe/unsafe corpus.

    **Label derivation (documented because F12 was about undocumented fiction).**
    Each PKU-SafeRLHF row carries a prompt and *two* responses, each with its
    own boolean human safety annotation (``is_response_0_safe``,
    ``is_response_1_safe``).  We treat each (prompt, response) pair as one
    independent example::

        example (prompt, response_0)  ->  label = 0 if is_response_0_safe else 1
        example (prompt, response_1)  ->  label = 0 if is_response_1_safe else 1

    So one source row yields **two** corpus rows.  Nothing is synthesised and
    no label is inferred from text.  The upstream annotation is used verbatim;
    ``0`` means the human annotator marked the response safe.

    **Independence.**  Two examples derived from one source row share a prompt,
    so they are *not* independent.  Two levels of protection are offered:

    * ``one_response_per_prompt=False`` (default): both siblings are kept,
      adjacent at ids ``2r`` / ``2r+1``, and :func:`three_way_split` splits at
      *source-row* granularity so a prompt never straddles two splits.  This
      removes cross-split leakage but leaves correlated pairs *within* the
      estimation pool, which mildly violates the i.i.d. premise of Theorem 1.
    * ``one_response_per_prompt=True``: exactly one response is drawn per
      source row, so every corpus row has a distinct prompt and the estimation
      draws are genuinely i.i.d.  **Use this whenever the corpus feeds a
      certificate.**  It halves the yield per source row, so ``n`` source rows
      are read for ``n`` corpus rows.

    Args:
        n: Number of corpus rows to return.  With the default sibling layout
            ``ceil(n/2)`` source rows are used; with
            ``one_response_per_prompt`` it is ``n``.
        seed: Seed for the shuffle applied before truncation, so that repeated
            calls with different ``n`` are nested rather than adversarially
            ordered.  Uses :func:`sca.utils.seeding.stable_rng`, not
            ``hash()`` (finding F9).
        hf_split: HuggingFace split name.
        dataset_id: Hub dataset id.
        one_response_per_prompt: See above.

    Returns:
        A :class:`Corpus`.  Its ``meta`` records the exact provenance.

    Raises:
        RuntimeError: If the dataset cannot be loaded.  This function never
            falls back to synthetic data -- that is precisely finding F12.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency is installed
        raise RuntimeError(
            "The `datasets` package is required to load real PKU-SafeRLHF data. "
            "This module deliberately has no synthetic fallback."
        ) from exc

    n_source = n if one_response_per_prompt else (n + 1) // 2
    # Read a margin so that the shuffle has something to shuffle.
    n_read = min(max(n_source * 4, n_source + 256), 100_000)
    slice_spec = f"{hf_split}[:{n_read}]"
    try:
        raw = load_dataset(dataset_id, split=slice_spec)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {dataset_id} split {slice_spec!r}: {exc}. "
            "No synthetic fallback is provided by design."
        ) from exc

    rng = stable_rng(seed, "load_pku_saferlhf", dataset_id, hf_split)
    order = rng.permutation(len(raw))
    if not one_response_per_prompt:
        order = order[:n_source]
    order = np.sort(order)  # keep Hub read order; the sample itself is random

    prompts: list[str] = []
    responses: list[str] = []
    labels: list[int] = []
    pick_rng = stable_rng(seed, "pku_response_choice", dataset_id)
    seen_prompts: set[str] = set()
    n_source_used = 0
    n_dupes_skipped = 0
    for src in order:
        if one_response_per_prompt and len(prompts) >= n:
            break
        row = raw[int(src)]
        if one_response_per_prompt:
            # PKU-SafeRLHF repeats prompts across source rows; drop repeats so
            # that "one row per prompt" is actually true.
            if row["prompt"] in seen_prompts:
                n_dupes_skipped += 1
                continue
            seen_prompts.add(row["prompt"])
        n_source_used += 1
        which = (int(pick_rng.integers(0, 2)),) if one_response_per_prompt else (0, 1)
        for k in which:
            resp = row[f"response_{k}"]
            safe = row[f"is_response_{k}_safe"]
            if resp is None or safe is None:
                continue
            prompts.append(row["prompt"])
            responses.append(resp)
            labels.append(0 if bool(safe) else 1)

    prompts = prompts[:n]
    responses = responses[:n]
    labels = labels[:n]

    corpus = Corpus(
        name=dataset_id,
        prompts=tuple(prompts),
        responses=tuple(responses),
        labels=np.asarray(labels, dtype=np.int64),
        label_rule=(
            "label = 0 if is_response_k_safe else 1, for k in "
            + ("{one k drawn uniformly per source row}"
               if one_response_per_prompt else "{0,1}")
            + "; each (prompt, response_k) pair is one corpus row. Human "
              "annotation used verbatim, nothing inferred from text."
        ),
        meta={
            "dataset_id": dataset_id,
            "hf_split": slice_spec,
            "n_source_rows_read": int(len(raw)),
            "n_source_rows_used": int(n_source_used),
            "n_duplicate_prompts_dropped": int(n_dupes_skipped),
            "seed": int(seed),
            "one_response_per_prompt": bool(one_response_per_prompt),
            **({} if one_response_per_prompt else
               {"sibling_layout": "ids 2r and 2r+1 come from source row r"}),
        },
    )
    logger.info(
        "Loaded %s: %s", dataset_id, corpus.class_balance(),
    )
    return corpus


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

Tokenizer = Callable[[Sequence[str]], "object"]


def make_hf_tokenizer(
    model_name: str = "gpt2", max_len: int = 128
) -> Callable[[Sequence[str]], "object"]:
    """Return a callable ``list[str] -> LongTensor[n, max_len]``.

    Uses the real HuggingFace tokenizer.  Kept out of :func:`three_way_split`
    so that unit tests can inject a cheap deterministic tokenizer instead of
    downloading weights.
    """
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _tokenize(texts: Sequence[str]):
        if not texts:
            return torch.zeros((0, max_len), dtype=torch.long)
        enc = tok(
            list(texts),
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].long()

    _tokenize.model_name = model_name  # type: ignore[attr-defined]
    _tokenize.max_len = max_len  # type: ignore[attr-defined]
    return _tokenize


def make_hashing_tokenizer(vocab_size: int = 50257, max_len: int = 32):
    """A deterministic, dependency-light tokenizer for tests.

    Maps whitespace tokens to ids via :func:`sca.utils.seeding.stable_hash`, so
    it is byte-identical across processes without ``PYTHONHASHSEED`` (F9).
    It is **not** a real tokenizer and must not be used for reported numbers;
    it exists so the split logic can be unit-tested without a Hub download.
    """
    import torch

    def _tokenize(texts: Sequence[str]):
        if not texts:
            return torch.zeros((0, max_len), dtype=torch.long)
        out = torch.zeros((len(texts), max_len), dtype=torch.long)
        for i, t in enumerate(texts):
            toks = t.split()[:max_len]
            for j, w in enumerate(toks):
                out[i, j] = 1 + (stable_hash(w) % (vocab_size - 1))
        return out

    _tokenize.model_name = "hashing"  # type: ignore[attr-defined]
    _tokenize.max_len = max_len  # type: ignore[attr-defined]
    return _tokenize


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabeledSplit:
    """A named, role-tagged block of corpus ids plus its materialised tensors.

    ``ids`` is the ground truth.  ``input_ids`` / ``labels`` are derived from
    it.  Disjointness is always checked on ``ids``, never on tensors, so the
    check cannot be fooled by two splits that happen to hold equal tensors.
    """

    name: str
    role: SplitRole
    ids: np.ndarray
    input_ids: "object"  # torch.Tensor
    labels: "object"  # torch.Tensor
    texts: tuple[str, ...] = ()

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def id_set(self) -> set[int]:
        return {int(i) for i in self.ids}

    def class_counts(self) -> dict[int, int]:
        import torch

        if len(self) == 0:
            return {0: 0, 1: 0}
        lab = self.labels
        return {
            0: int((lab == 0).sum().item()),
            1: int((lab == 1).sum().item()),
        }

    def to_tensor_dataset(self):
        from torch.utils.data import TensorDataset

        return TensorDataset(self.input_ids, self.labels)

    def summary(self) -> dict:
        counts = self.class_counts()
        return {
            "name": self.name,
            "role": self.role.value,
            "n": len(self),
            "n_safe": counts[0],
            "n_unsafe": counts[1],
        }


@dataclass
class SplitBundle:
    """A pairwise-disjoint partition of a :class:`Corpus`.

    Attributes:
        client_pools: One :class:`LabeledSplit` per FL client, non-IID.
        server_verification_pool: The union the gate may see.  Provided for
            callers that want it whole; it is exactly
            ``server_search_pool | server_estimation_pool``.
        server_search_pool: Stage A.  Adaptive / mutated probing lives here.
        server_estimation_pool: Stage B.  Fresh i.i.d. draws; feeds the bound.
        heldout_test: Final reporting only.
        corpus: The corpus the ids index into.
        config: The split configuration, recorded for the results file.
    """

    client_pools: list[LabeledSplit]
    server_verification_pool: LabeledSplit
    server_search_pool: LabeledSplit
    server_estimation_pool: LabeledSplit
    heldout_test: LabeledSplit
    corpus: Corpus
    config: dict = field(default_factory=dict)

    # -- introspection ----------------------------------------------------

    def named_splits(self) -> list[LabeledSplit]:
        """Every *leaf* split.  ``server_verification_pool`` is excluded
        because it is by construction the union of two leaves."""
        return [*self.client_pools, self.server_search_pool,
                self.server_estimation_pool, self.heldout_test]

    def overlap_report(self) -> dict:
        """Pairwise intersection sizes between all leaf splits.

        Every value must be ``0``.  Experiments print this into ``results/``
        so a reader can verify C2 without rerunning anything.
        """
        splits = self.named_splits()
        sets = {s.name: s.id_set() for s in splits}
        pairs: dict[str, int] = {}
        for i, a in enumerate(splits):
            for b in splits[i + 1:]:
                pairs[f"{a.name}|{b.name}"] = len(sets[a.name] & sets[b.name])
        total_ids = sum(len(s) for s in splits)
        union = set().union(*sets.values()) if sets else set()
        return {
            "pairwise_overlaps": pairs,
            "max_pairwise_overlap": max(pairs.values()) if pairs else 0,
            "n_leaf_splits": len(splits),
            "total_ids_across_splits": total_ids,
            "n_distinct_ids": len(union),
            "duplicated_ids": total_ids - len(union),
            "corpus_size": len(self.corpus),
            "verification_pool_is_union_of_search_and_estimation": (
                self.server_verification_pool.id_set()
                == (self.server_search_pool.id_set()
                    | self.server_estimation_pool.id_set())
            ),
        }

    def group_ids(self, split: LabeledSplit) -> set:
        """Group keys covered by ``split``.

        The group key is the **prompt text** when grouping is on.  It is not
        the row index and not ``id // 2``: PKU-SafeRLHF turned out to repeat
        the same prompt across *different* source rows (197 distinct prompts in
        200 consecutive rows), so an index-arithmetic notion of "same prompt"
        silently misses real duplicates.
        """
        if self.config.get("grouped_by_prompt"):
            return {self.corpus.prompts[int(i)] for i in split.ids}
        return split.id_set()

    def group_overlap_report(self) -> dict:
        """Prompt-level overlaps for the pairs where sharing a prompt matters.

        Two client pools may legitimately share a prompt (both are training
        data).  Every other pair may not:

        * client vs held-out test -- that is F1.
        * server pools vs held-out test -- that is F2.
        * search vs estimation -- the Stage-B draws must be independent of the
          Stage-A search that chose the allocation.
        * client vs server -- the gate would be scoring memorised prompts.
        """
        splits = self.named_splits()
        gsets = {s.name: self.group_ids(s) for s in splits}
        out: dict[str, int] = {}
        for i, a in enumerate(splits):
            for b in splits[i + 1:]:
                if a.name.startswith("client_") and b.name.startswith("client_"):
                    continue
                out[f"{a.name}|{b.name}"] = len(gsets[a.name] & gsets[b.name])
        return {
            "grouped_by_prompt": bool(self.config.get("grouped_by_prompt")),
            "pairwise_group_overlaps": out,
            "max_group_overlap": max(out.values()) if out else 0,
        }

    def assert_disjoint(self) -> None:
        """Raise :class:`ValueError` if any corpus id appears in two splits."""
        rep = self.overlap_report()
        bad = {k: v for k, v in rep["pairwise_overlaps"].items() if v}
        if bad:
            raise ValueError(
                f"SPLIT LEAKAGE: non-empty intersections {bad}. "
                "This is exactly audit finding F1."
            )
        if rep["duplicated_ids"] != 0:
            raise ValueError(
                f"SPLIT LEAKAGE: {rep['duplicated_ids']} ids appear in more "
                "than one split."
            )
        if not rep["verification_pool_is_union_of_search_and_estimation"]:
            raise ValueError(
                "server_verification_pool is not the union of the search and "
                "estimation pools."
            )
        grep = self.group_overlap_report()
        gbad = {k: v for k, v in grep["pairwise_group_overlaps"].items() if v}
        if gbad:
            raise ValueError(
                f"PROMPT-LEVEL LEAKAGE: splits share prompts {gbad}. "
                "Two client pools may share a prompt; nothing else may."
            )

    def summary(self) -> dict:
        return {
            "corpus": {
                "name": self.corpus.name,
                "label_rule": self.corpus.label_rule,
                **self.corpus.class_balance(),
                "meta": self.corpus.meta,
            },
            "config": dict(self.config),
            "splits": [s.summary() for s in self.named_splits()],
            "client_label_distribution": [
                s.class_counts() for s in self.client_pools
            ],
            # F7 diagnostic: sort-and-chunk made EVERY client single-class.
            # Report how many survive here rather than claiming none do.
            "n_single_class_clients": sum(
                1 for s in self.client_pools
                if min(s.class_counts().values()) == 0
            ),
            "n_clients": len(self.client_pools),
            "overlap_report": self.overlap_report(),
            "group_overlap_report": self.group_overlap_report(),
        }


# ---------------------------------------------------------------------------
# Dirichlet non-IID partition
# ---------------------------------------------------------------------------


def dirichlet_client_partition(
    labels: np.ndarray,
    ids: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
    min_per_client: int = 2,
    min_classes_per_client: int = 1,
) -> list[np.ndarray]:
    """Partition ``ids`` across ``n_clients`` with a Dirichlet(alpha) label mix.

    For each class ``c`` the class's ids are split across clients in
    proportions drawn from ``Dirichlet(alpha * 1_{n_clients})``.  Small
    ``alpha`` => highly skewed (nearly single-class) clients; large ``alpha``
    => nearly IID.  This replaces the sort-and-chunk scheme that produced
    single-class clients and made Krum collapse (F7).

    Retries the Dirichlet draw until every client holds at least
    ``min_per_client`` examples, so no client is empty.

    ``min_classes_per_client`` additionally rejects draws that leave a client
    single-class.  It defaults to ``1`` (no constraint) because forcing every
    client multi-class changes the distribution being sampled; set it to ``2``
    only if you intend that, and say so in the results.  Note that at
    ``alpha = 0.5`` with 8 clients roughly one client comes out single-class,
    which is still a completely different regime from sort-and-chunk, where
    *every* client is single-class (finding F7).
    """
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    ids = np.asarray(ids, dtype=np.int64)
    labels = np.asarray(labels)
    if ids.shape != labels.shape:
        raise ValueError("ids and labels must have the same shape")
    classes = np.unique(labels)
    lab_of = {int(i): int(l) for i, l in zip(ids.tolist(), labels.tolist())}
    for _attempt in range(200):
        buckets: list[list[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            c_ids = ids[labels == c]
            c_ids = rng.permutation(c_ids)
            props = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(props) * len(c_ids)).astype(int)[:-1]
            for j, chunk in enumerate(np.split(c_ids, cuts)):
                buckets[j].extend(int(x) for x in chunk)
        sizes = [len(b) for b in buckets]
        n_classes_ok = all(
            len({lab_of[i] for i in b}) >= min(min_classes_per_client,
                                               len(classes))
            for b in buckets
        ) if min_classes_per_client > 1 else True
        if min(sizes) >= min_per_client and n_classes_ok:
            return [np.sort(np.asarray(b, dtype=np.int64)) for b in buckets]
    raise RuntimeError(
        f"Could not build a Dirichlet(alpha={alpha}) partition with >= "
        f"{min_per_client} examples and >= {min_classes_per_client} classes "
        f"per client over {n_clients} clients from {len(ids)} examples. "
        "Increase alpha or the pool size."
    )


def _materialise(
    name: str,
    role: SplitRole,
    ids: np.ndarray,
    corpus: Corpus,
    tokenizer,
    keep_texts: bool,
) -> LabeledSplit:
    import torch

    ids = np.asarray(ids, dtype=np.int64)
    texts = corpus.texts(ids)
    input_ids = tokenizer(texts)
    labels = torch.as_tensor(corpus.labels[ids], dtype=torch.long)
    return LabeledSplit(
        name=name,
        role=role,
        ids=ids,
        input_ids=input_ids,
        labels=labels,
        texts=tuple(texts) if keep_texts else (),
    )


def three_way_split(
    corpus: Corpus,
    *,
    n_clients: int = 8,
    dirichlet_alpha: float = 0.5,
    client_frac: float = 0.6,
    server_frac: float = 0.2,
    test_frac: float = 0.2,
    search_frac: float = 0.5,
    seed: int = 0,
    tokenizer=None,
    keep_texts: bool = True,
    group_by_prompt: bool = True,
    min_classes_per_client: int = 1,
) -> SplitBundle:
    """Split ``corpus`` into client / server-verification / held-out-test blocks.

    Indices are partitioned **before** any tensor is built, so overlap is
    structurally impossible rather than merely unlikely.

    Args:
        corpus: The corpus to split.
        n_clients: Number of FL clients.
        dirichlet_alpha: Non-IID-ness of the client label mixture.  Smaller is
            more skewed.  ``alpha -> inf`` approaches IID.
        client_frac, server_frac, test_frac: Fractions of the corpus.  Must sum
            to at most 1; any remainder is discarded (recorded in the config).
        search_frac: Fraction of the server verification pool given to the
            Stage-A search pool; the rest becomes the Stage-B estimation pool.
        seed: Master seed.  Streams are derived with
            :func:`sca.utils.seeding.stable_rng`.
        tokenizer: ``list[str] -> LongTensor``.  Defaults to
            :func:`make_hf_tokenizer` (real GPT-2 tokenizer).
        keep_texts: Store the raw texts on each split (needed by the verifier's
            embedder).
        group_by_prompt: If ``True`` (default), all rows sharing a prompt are
            kept together, so a prompt never straddles two splits.  The group
            key is the prompt **text**, not an index: PKU-SafeRLHF repeats the
            same prompt across different source rows (197 distinct prompts in
            200 consecutive rows), so index arithmetic would miss real
            duplicates.
        min_classes_per_client: Passed to :func:`dirichlet_client_partition`.

    Returns:
        A :class:`SplitBundle`.  :meth:`SplitBundle.assert_disjoint` is called
        before returning, so a leaking bundle can never escape this function.
    """
    fr = client_frac + server_frac + test_frac
    if fr > 1.0 + 1e-9:
        raise ValueError(
            f"client_frac + server_frac + test_frac = {fr} > 1"
        )
    for nm, v in (("client_frac", client_frac), ("server_frac", server_frac),
                  ("test_frac", test_frac)):
        if v <= 0:
            raise ValueError(f"{nm} must be > 0, got {v}")
    if not (0.0 < search_frac < 1.0):
        raise ValueError(f"search_frac must lie in (0, 1), got {search_frac}")

    if tokenizer is None:
        tokenizer = make_hf_tokenizer()

    n = len(corpus)
    rng = stable_rng(seed, "three_way_split", corpus.name, str(n))

    # -- build the atoms that get partitioned -------------------------------
    # An atom is a group of rows that must stay together.  With grouping on,
    # that is "all rows sharing a prompt"; with it off, every row is its own
    # atom.  Everything downstream partitions ATOMS, never rows, so a prompt
    # cannot straddle a split boundary by construction.
    grouped = bool(group_by_prompt)
    if grouped:
        by_prompt: dict[str, list[int]] = {}
        for i in range(n):
            by_prompt.setdefault(corpus.prompts[i], []).append(i)
        # sorted() makes the atom order independent of dict insertion order
        atoms = [np.asarray(by_prompt[k], dtype=np.int64)
                 for k in sorted(by_prompt)]
    else:
        atoms = [np.asarray([i], dtype=np.int64) for i in range(n)]

    n_atoms = len(atoms)
    atom_order = rng.permutation(n_atoms)
    # floor, not round: with fractions summing to 1, rounding up three times
    # can demand one more group than exists.
    n_c = max(int(client_frac * n_atoms), 1)
    n_s = max(int(server_frac * n_atoms), 2)
    n_t = max(int(test_frac * n_atoms), 1)
    if n_c + n_s + n_t > n_atoms:
        raise ValueError(
            f"corpus too small: {n_atoms} groups cannot fill "
            f"{n_c}/{n_s}/{n_t} client/server/test blocks"
        )

    def expand(sel: np.ndarray) -> np.ndarray:
        if len(sel) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.sort(np.concatenate([atoms[int(a)] for a in sel]))

    client_ids = expand(atom_order[:n_c])
    server_atoms = atom_order[n_c:n_c + n_s]
    server_ids = expand(server_atoms)
    test_ids = expand(atom_order[n_c + n_s:n_c + n_s + n_t])
    unused = expand(atom_order[n_c + n_s + n_t:])

    # -- server pool -> disjoint search / estimation halves ----------------
    # Split at ATOM granularity too, so a prompt cannot appear in both Stage A
    # (search) and Stage B (estimation).  If it could, the estimation draws
    # would not be independent of the search that chose the allocation, which
    # is exactly the coupling Theorem 1 forbids.
    srv_rng = stable_rng(seed, "server_pool_split", corpus.name)
    srv_shuffled = srv_rng.permutation(server_atoms)
    n_g = int(round(search_frac * len(srv_shuffled)))
    n_g = min(max(n_g, 1), len(srv_shuffled) - 1)
    search_ids = expand(srv_shuffled[:n_g])
    estim_ids = expand(srv_shuffled[n_g:])

    # -- clients -> Dirichlet non-IID --------------------------------------
    cl_rng = stable_rng(seed, "dirichlet_clients", corpus.name)
    client_id_blocks = dirichlet_client_partition(
        labels=corpus.labels[client_ids],
        ids=client_ids,
        n_clients=n_clients,
        alpha=dirichlet_alpha,
        rng=cl_rng,
        min_classes_per_client=min_classes_per_client,
    )

    client_pools = [
        _materialise(f"client_{j}", SplitRole.CLIENT_TRAIN, blk, corpus,
                     tokenizer, keep_texts)
        for j, blk in enumerate(client_id_blocks)
    ]
    search_pool = _materialise("server_search", SplitRole.SERVER_SEARCH,
                               search_ids, corpus, tokenizer, keep_texts)
    estim_pool = _materialise("server_estimation", SplitRole.SERVER_ESTIMATION,
                              estim_ids, corpus, tokenizer, keep_texts)
    verif_pool = _materialise("server_verification", SplitRole.SERVER_SEARCH,
                              np.sort(server_ids), corpus, tokenizer,
                              keep_texts)
    test_split = _materialise("heldout_test", SplitRole.HELDOUT_TEST,
                              test_ids, corpus, tokenizer, keep_texts)

    bundle = SplitBundle(
        client_pools=client_pools,
        server_verification_pool=verif_pool,
        server_search_pool=search_pool,
        server_estimation_pool=estim_pool,
        heldout_test=test_split,
        corpus=corpus,
        config={
            "n_clients": n_clients,
            "dirichlet_alpha": dirichlet_alpha,
            "client_frac": client_frac,
            "server_frac": server_frac,
            "test_frac": test_frac,
            "search_frac": search_frac,
            "seed": seed,
            "min_classes_per_client": min_classes_per_client,
            "grouped_by_prompt": grouped,
            "n_groups": int(n_atoms),
            "n_discarded": int(len(unused)),
            "tokenizer": getattr(tokenizer, "model_name", "unknown"),
        },
    )
    bundle.assert_disjoint()
    logger.info(
        "three_way_split: clients=%s server_search=%d server_estimation=%d "
        "heldout_test=%d discarded=%d",
        [len(p) for p in client_pools], len(search_pool), len(estim_pool),
        len(test_split), len(unused),
    )
    return bundle


def build_splits(
    n: int = 4000,
    seed: int = 0,
    *,
    n_clients: int = 8,
    dirichlet_alpha: float = 0.5,
    max_len: int = 128,
    tokenizer_name: str = "gpt2",
    one_response_per_prompt: bool = False,
    **split_kwargs,
) -> SplitBundle:
    """Convenience: load real PKU-SafeRLHF and produce a verified split.

    Pass ``one_response_per_prompt=True`` when the estimation pool will feed a
    certificate; see :func:`load_pku_saferlhf` for why.
    """
    corpus = load_pku_saferlhf(
        n=n, seed=seed, one_response_per_prompt=one_response_per_prompt)
    tok = make_hf_tokenizer(tokenizer_name, max_len=max_len)
    return three_way_split(
        corpus,
        n_clients=n_clients,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
        tokenizer=tok,
        **split_kwargs,
    )


def write_split_report(
    bundle: SplitBundle, filename: str = "data_split_report.json"
) -> "object":
    """Write :meth:`SplitBundle.summary` to ``results/`` (criteria C2, C6).

    The reader must be able to see the zeros without rerunning anything, so
    the pairwise overlap table goes into the results file verbatim.
    """
    import json

    from sca.utils.paths import results_dir

    path = results_dir() / filename
    path.write_text(json.dumps(bundle.summary(), indent=2, default=str))
    return path


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """``python -m sca.experiments.data`` -- regenerate the split report."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-clients", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--one-response-per-prompt", action="store_true",
                    help="draw one response per prompt so estimation rows are "
                         "i.i.d. (use this when the pool feeds a certificate)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    bundle = build_splits(n=args.n, seed=args.seed, n_clients=args.n_clients,
                          dirichlet_alpha=args.alpha, max_len=args.max_len,
                          one_response_per_prompt=args.one_response_per_prompt)
    bundle.assert_disjoint()
    path = write_split_report(bundle)
    print(json.dumps(bundle.overlap_report(), indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "PKU_DATASET_ID",
    "SplitRole",
    "GATE_VISIBLE_ROLES",
    "REPORTABLE_ROLES",
    "Corpus",
    "LabeledSplit",
    "SplitBundle",
    "load_pku_saferlhf",
    "three_way_split",
    "build_splits",
    "dirichlet_client_partition",
    "make_hf_tokenizer",
    "make_hashing_tokenizer",
    "write_split_report",
]
