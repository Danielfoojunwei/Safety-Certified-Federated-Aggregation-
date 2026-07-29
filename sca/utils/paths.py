"""Repository-relative path resolution.

Both experiment entry points used to write to a hardcoded absolute path::

    output_dir = Path("/home/user/Safety-Certified-Federated-Aggregation-/results")

That path is wrong in three independent ways:

1. It is machine specific.  On any checkout that is not ``/home/user`` the
   scripts either crash or silently create a directory tree outside the repo.
2. The repository name in it is *stale* -- the directory is actually
   ``Safety-Certified-Federated-Aggregation-via-Recursive-Language-Model-Verification-and-Knowledge-Graph``.
   So even on the original machine the results were written next to the repo,
   not into it, which is part of why the committed ``results/`` was never
   refreshed and went stale.
3. It makes CI impossible: a GitHub runner checks out under
   ``/home/runner/work/...``.

Use :func:`results_dir` (or :func:`repo_root`) instead.  Both resolve relative
to this file, so they are correct for an editable install, a source checkout, a
CI runner, and a container, with no environment configuration.

An installed (non-editable) wheel has no writable repository next to it, so
:func:`repo_root` accepts an override through the ``SCA_REPO_ROOT``
environment variable and :func:`results_dir` through ``SCA_RESULTS_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["repo_root", "results_dir", "figures_dir", "cache_dir", "package_root"]


def package_root() -> Path:
    """Absolute path of the ``sca`` package directory."""
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """Absolute path of the repository root.

    Resolution order:

    1. ``$SCA_REPO_ROOT`` if set (for installed wheels / unusual layouts).
    2. The parent of the ``sca`` package directory, which is the repo root for
       a source checkout and for ``pip install -e .``.
    """
    env = os.environ.get("SCA_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return package_root().parent


def results_dir(*subpath: str, create: bool = True) -> Path:
    """Return ``<repo>/results/<subpath>``, creating it by default.

    Honours ``$SCA_RESULTS_DIR`` so a CI smoke run can redirect output to a
    scratch directory without touching the committed results.
    """
    env = os.environ.get("SCA_RESULTS_DIR")
    base = Path(env).expanduser().resolve() if env else repo_root() / "results"
    path = base.joinpath(*subpath)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(*subpath: str, create: bool = True) -> Path:
    """Return ``<repo>/results/figures/<subpath>``."""
    return results_dir("figures", *subpath, create=create)


def cache_dir(*subpath: str, create: bool = True) -> Path:
    """Return a gitignored scratch directory for downloaded datasets etc.

    Honours ``$SCA_CACHE_DIR``.  Never commit anything written here.
    """
    env = os.environ.get("SCA_CACHE_DIR")
    base = Path(env).expanduser().resolve() if env else repo_root() / ".cache"
    path = base.joinpath(*subpath)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
