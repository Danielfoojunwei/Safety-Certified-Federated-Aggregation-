"""Process-independent hashing, seeding, and RNG derivation.

This module is the root fix for audit finding **F9**.

The defect
----------
``sca/knowledge_graph/embedding.py`` bucketed tokens with the CPython builtin
``hash()``.  Since PEP 456 / CPython 3.3, string hashing is salted with a
per-process random value drawn at interpreter start unless the environment
variable ``PYTHONHASHSEED`` is set.  Nothing in the repository set it.  The
consequence was that a "seeded" run was reproducible *within* a process and a
different random variable *across* processes: three interpreters with
``seed=42`` produced three different embeddings, three different region
partitions, and three different certified bounds.

Crucially, an in-process test **cannot** observe this.  Within one interpreter
``hash("x")`` is perfectly consistent, so any assertion of the form
``assert f(s) == f(s)`` passes while the bug is live.  The regression test for
this module therefore spawns real subprocesses
(:mod:`tests.test_seeding`).

The fix
-------
:func:`stable_hash` uses ``hashlib.blake2b``, a keyless cryptographic digest
that is a pure function of its input bytes.  It does not consult the
interpreter's hash randomisation salt, so it is identical across processes,
machines, Python versions, and platforms.  Every use of the builtin ``hash()``
on a string anywhere under ``sca/`` must be replaced by it.

Determinism policy for the whole package
----------------------------------------
1. Never call the builtin ``hash()`` on a ``str``, ``bytes``, or any container
   thereof, when the result influences a number that is reported.  Use
   :func:`stable_hash`.
2. Never rely on the ambient global RNG (``random.random()``,
   ``np.random.rand()``, ``torch.rand()`` without a generator) inside library
   code.  Derive an explicit stream with :func:`stable_rng`.
3. Iteration order of ``set`` objects containing strings is *also* affected by
   hash randomisation.  Sort before iterating whenever the order can influence
   output.  ``dict`` preserves insertion order and is safe.

:func:`set_global_seed` exists for scripts and test fixtures that want a
reproducible ambient state.  Library code should still prefer
:func:`stable_rng`, because a global seed is a single stream shared by every
component: adding one extra draw in component A silently shifts every
subsequent draw in components B and C.  :func:`stable_rng` gives each component
its own independent stream keyed by ``(seed, *tags)``, so call sites can be
added and removed without perturbing anybody else's numbers.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.random import Generator

__all__ = [
    "STABLE_HASH_BITS",
    "set_global_seed",
    "stable_hash",
    "stable_bytes",
    "stable_rng",
    "derive_seed",
    "stable_shuffled",
    "seed_torch_generator",
]

#: Width of the integer returned by :func:`stable_hash`.  64 bits keeps the
#: value inside a C ``int64`` / numpy ``uint64`` so it can be handed to
#: ``np.random.SeedSequence`` and used in modular arithmetic without Python
#: bigint promotion, while collisions remain negligible at our scale.
STABLE_HASH_BITS = 64

_DIGEST_BYTES = STABLE_HASH_BITS // 8

#: Domain separator.  Mixed into every digest so that a ``stable_hash`` value
#: can never be confused with a Merkle leaf hash from :mod:`sca.utils.crypto`,
#: even if the two ever hash the same string.
_PERSON = b"sca-stable"


def stable_bytes(*parts: str) -> bytes:
    """Return a process-independent digest of ``parts``.

    Parts are joined with a NUL separator, which cannot occur in ordinary text,
    so ``stable_bytes("ab", "c") != stable_bytes("a", "bc")``.

    Parameters
    ----------
    *parts:
        Strings to digest.  Non-``str`` inputs raise ``TypeError`` rather than
        being silently coerced, because ``str(obj)`` for an object without a
        custom ``__repr__`` embeds its memory address and would reintroduce
        exactly the non-determinism this module exists to remove.
    """
    h = hashlib.blake2b(digest_size=_DIGEST_BYTES, person=_PERSON)
    for i, part in enumerate(parts):
        if not isinstance(part, str):
            raise TypeError(
                f"stable_bytes() argument {i} must be str, got "
                f"{type(part).__name__!r}. Convert it explicitly with a "
                f"deterministic representation (repr of a builtin, or "
                f"format()); str() on an arbitrary object may embed its "
                f"memory address and would break reproducibility."
            )
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.digest()


def stable_hash(s: str) -> int:
    """Process-independent replacement for the builtin ``hash()`` on strings.

    Returns a non-negative integer in ``[0, 2**64)``.  The value is a pure
    function of ``s``: identical in every process, on every machine, under
    every value of ``PYTHONHASHSEED``, and stable across runs and releases of
    this package.

    This is the F9 fix.  Non-negativity matters: callers write
    ``stable_hash(tok) % vocab_size`` to pick a bucket, and the builtin
    ``hash()`` can return a negative value, for which Python's ``%`` yields a
    non-negative result but C-style ``%`` in downstream numpy code does not.

    >>> stable_hash("safety") == stable_hash("safety")
    True
    >>> 0 <= stable_hash("safety") < 2 ** 64
    True
    """
    if not isinstance(s, str):
        raise TypeError(
            f"stable_hash() requires str, got {type(s).__name__!r}"
        )
    return int.from_bytes(stable_bytes(s), "big", signed=False)


def derive_seed(seed: int, *tags: str) -> int:
    """Mix ``seed`` and ``tags`` into a fresh 64-bit seed.

    Used to give each component an independent stream.  Deterministic and
    process-independent.  ``derive_seed(42)`` is *not* ``42`` -- the mixing is
    unconditional so that a component which currently takes no tags can gain
    one later without colliding with the untagged stream of some other
    component.
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(
            f"seed must be an int, got {type(seed).__name__!r}"
        )
    # repr of a Python int is exact and platform independent, unlike float.
    return int.from_bytes(
        stable_bytes("seed:%d" % int(seed), *tags), "big", signed=False
    )


def stable_rng(seed: int, *tags: str) -> "Generator":
    """Return an independent :class:`numpy.random.Generator` for ``(seed, tags)``.

    Two call sites with different ``tags`` get statistically independent
    streams, so adding, removing, or reordering draws in one component does not
    shift the numbers produced by another.  That property is what makes a
    multi-component experiment reproducible in practice; a single global seed
    is reproducible only as long as nobody edits any code path that consumes
    randomness.

    The generator is ``PCG64`` seeded through ``SeedSequence``, which performs
    its own entropy mixing, so nearby ``derive_seed`` values still yield
    well-separated streams.

    Examples
    --------
    >>> a = stable_rng(0, "search").integers(0, 10_000, size=5)
    >>> b = stable_rng(0, "search").integers(0, 10_000, size=5)
    >>> bool((a == b).all())
    True
    >>> c = stable_rng(0, "estimation").integers(0, 10_000, size=5)
    >>> bool((a == c).all())
    False
    """
    return np.random.default_rng(np.random.SeedSequence(derive_seed(seed, *tags)))


def set_global_seed(seed: int) -> None:
    """Seed every ambient RNG this project can reach.

    Seeds :mod:`random`, :mod:`numpy.random`'s legacy global state, and
    :mod:`torch` (CPU and, if present, CUDA).  Torch is imported lazily so that
    this module stays importable in a torch-free environment.

    This deliberately does **not** attempt to set ``PYTHONHASHSEED``.  That
    variable is read by the interpreter at startup and cannot be changed from
    inside a running process; assigning ``os.environ["PYTHONHASHSEED"]`` has no
    effect on the current interpreter and would only create the illusion of a
    fix.  Process independence is achieved by not using the builtin ``hash()``
    at all -- see :func:`stable_hash`.

    Library code should prefer :func:`stable_rng`; use this in scripts and test
    fixtures to pin the ambient state that third-party code (e.g. scikit-learn
    estimators without a ``random_state``) may consult.
    """
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__!r}")
    seed = int(seed)
    if not 0 <= seed < 2 ** 32:
        # numpy's legacy global seeder rejects anything outside this range.
        seed = seed % (2 ** 32)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep in CI
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU-only project
        torch.cuda.manual_seed_all(seed)
    # Deterministic CPU kernels.  Cheap here because the models are tiny.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):  # pragma: no cover - old torch
        pass


def seed_torch_generator(seed: int, *tags: str):
    """Return a ``torch.Generator`` derived from ``(seed, tags)``.

    The torch counterpart of :func:`stable_rng`, for ``DataLoader`` shuffling
    and parameter initialisation.  Raises ``ImportError`` if torch is absent.
    """
    import torch

    g = torch.Generator()
    # torch requires a seed representable as int64.
    g.manual_seed(derive_seed(seed, *tags) % (2 ** 63))
    return g


def stable_shuffled(items: Iterable[Any], seed: int, *tags: str) -> list:
    """Return a deterministically shuffled ``list`` of ``items``.

    Provided because ``sorted(a_set_of_strings)`` followed by an explicit
    shuffle is the only order-safe way to randomise a collection whose natural
    iteration order depends on string hashing.
    """
    out = list(items)
    stable_rng(seed, "stable_shuffled", *tags).shuffle(out)
    return out


def _pythonhashseed_is_set() -> bool:
    """True if the ambient interpreter was launched with ``PYTHONHASHSEED``.

    Only for diagnostics.  Correctness of this package must not depend on it,
    and the CI reproducibility job runs with it explicitly unset to prove that.
    """
    return bool(os.environ.get("PYTHONHASHSEED"))
