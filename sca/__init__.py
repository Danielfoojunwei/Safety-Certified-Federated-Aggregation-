"""Safety-Certified Aggregation (SCA) for federated learning.

A federated aggregation gate that admits a client update only when a
region-stratified, anytime-valid upper bound on the safety-violation rate of
the resulting model clears a pre-declared threshold.

Subpackages
-----------
``sca.utils``
    Seeding and process-independent hashing (:mod:`sca.utils.seeding`),
    repository-relative paths (:mod:`sca.utils.paths`), concentration bounds
    and budget allocation (:mod:`sca.utils.stats`), Merkle commitments
    (:mod:`sca.utils.crypto`).
``sca.knowledge_graph``
    Embeddings, the region partition, and the Model Knowledge Graph.
``sca.verifier``
    The three-stage verifier: search, estimation, certificate.  Search samples
    and estimation samples are recorded to *separate* counters and only the
    estimation counters may reach the bound.
``sca.certificate``
    Certificate schema and the acceptance rule.
``sca.federated``
    Clients, server, and aggregation rules.
``sca.experiments``
    Data splits, attacks, baselines, metrics, and the experiment drivers.

Reproducibility
---------------
Nothing in this package may call the builtin ``hash()`` on a string, and no
library function may draw from an ambient global RNG.  Use
:func:`sca.utils.seeding.stable_hash` and
:func:`sca.utils.seeding.stable_rng`.  Results are byte-identical across
processes without setting ``PYTHONHASHSEED``; ``make repro-check`` and the
``reproducibility`` CI job enforce it.

This module deliberately imports nothing heavy.  ``import sca`` must stay fast
and must not pull in torch, so that a tool inspecting the version does not pay
several seconds of import cost.
"""

from __future__ import annotations

__all__ = ["__version__"]


def _detect_version() -> str:
    """Read the version from installed metadata, falling back to a literal.

    Keeping a hardcoded literal in sync with ``pyproject.toml`` by hand is
    exactly the kind of thing that silently rots -- the pre-rebuild tree
    declared ``0.2.0`` in ``pyproject.toml`` and ``0.1.0`` here.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("sca")
    except Exception:  # pragma: no cover - source tree without install
        return "0.3.0+unknown"


__version__: str = _detect_version()
